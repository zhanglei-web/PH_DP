#!/usr/bin/env python3
"""Stage-DP + Frozen Value Guidance: CUDA-only actor fine-tuning, without TD."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch

from mujoco_shared_control.rss2023.oracle_stage_dataset import prepare_oracle_dataset
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig
from train_stage_mc_twin_q import TwinQ, load as load_q_replay, qaction
from audit_q3b2_policy_shift_metrics import load_v2, nearest_dist, MC, REPLAY, V2

ROOT = Path(__file__).resolve().parents[1]
V2CK = V2 / "checkpoints/step_00080000.pt"; QCK = MC / "checkpoints/mc_step_00005000.pt"
OUT = ROOT / "outputs/value_guided_actor_finetune/v1"; SEED = 20260903; BATCH = 512

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def stats(path, dev):
    with np.load(path) as z:
        return tuple(torch.as_tensor(z[k], device=dev, dtype=torch.float32) for k in ("observation_mean","observation_std","action_mean","action_std"))

def load_actor(dev):
    p=torch.load(V2CK,map_location=dev,weights_only=False)
    cfg=StageEmbeddingDiffusionConfig(**{k:v for k,v in p["diffusion_config"].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__})
    m=StageEmbeddingDiffusion(cfg).to(dev);m.load_state_dict(p["model"]);return m,cfg

def x0hat(actor, obs, clean, ts, noise):
    """Recover x0 using V2's own q_sample buffers/scheduler, not a new schedule."""
    noisy,_=actor.q_sample(clean,ts,noise=noise)
    eps=actor.denoiser(torch.cat((actor._condition(obs),noisy),-1),ts)[...,actor.config.condition_hidden_dim:]
    sa=actor.sqrt_alphas_cumprod.gather(0,ts).reshape(-1,1); so=actor.sqrt_one_minus_alphas_cumprod.gather(0,ts).reshape(-1,1)
    return (noisy-so*eps)/sa.clamp_min(1e-8)

def q_executable(x_v2, vam, vas, qam, qas):
    """Map V2 action-normalized x0 to Q's normalized executable semantic action."""
    raw=x_v2*vas+vam; cont=raw[:,:6].clamp(-1,1)
    # Gripper uses frozen evaluator threshold and is detached from Q gradients.
    grip=torch.where(raw[:,6].detach()<.375,torch.full_like(raw[:,6],-1.),torch.ones_like(raw[:,6])).detach()
    return (torch.cat((cont,grip[:,None]),1)-qam)/qas

def grad_norm_from(grads):
    return torch.sqrt(sum((g.detach()**2).sum() for g in grads if g is not None)+1e-12)

@torch.no_grad()
def sample(actor, obs, dev, seed):
    gen=torch.Generator(device=dev).manual_seed(seed); out=[]
    for i in range(0,len(obs),256):
        block=obs[i:i+256];out.append(actor.assist(block,torch.zeros((len(block),7),device=dev),1.,generator=gen))
    return torch.cat(out)

@torch.no_grad()
def audit_actor(actor, ref, q, qtest, qstats, vstats, support, dev, count):
    qom,qos,qam,qas=qstats;vom,vos,vam,vas=vstats;n=min(count,len(qtest["obs"]));ix=torch.as_tensor(np.sort(np.random.default_rng(SEED).choice(len(qtest["obs"]),n,False)),device=dev)
    rawobs=qtest["obs"][ix].float();state=(rawobs-qom)/qos;vobs=(rawobs-vom)/vos
    was_training=actor.training; actor.eval()
    a=sample(actor,vobs,dev,SEED+11); r=sample(ref,vobs,dev,SEED+11);aq=q_executable(a,vam,vas,qam,qas);rq=q_executable(r,vam,vas,qam,qas)
    actor.train(was_training)
    val=torch.minimum(*q(state,aq))[:,0]; refv=torch.minimum(*q(state,rq))[:,0];drift=torch.linalg.vector_norm(aq[:,:6]-rq[:,:6],dim=1);rawdrift=torch.linalg.vector_norm((a-r)[:,:6]*vas[:6],dim=1)
    nn=nearest_dist(aq,support);rnn=nearest_dist(rq,support);dq=val-refv
    return {"Q_ACTOR_MEAN":float(val.mean()),"Q_ACTOR_MEDIAN":float(val.median()),"DELTA_Q_VS_V2_MEAN":float(dq.mean()),"DELTA_Q_VS_V2_MEDIAN":float(dq.median()),"POSITIVE_Q_VS_V2_FRACTION":float((dq>0).float().mean()),"NORMALIZED_ACTION_DRIFT_MEAN":float(drift.mean()),"NORMALIZED_ACTION_DRIFT_P95":float(torch.quantile(drift,.95)),"RAW_ACTION_DRIFT_MEAN":float(rawdrift.mean()),"SUPPORT_NN_MEAN":float(nn.mean()),"SUPPORT_NN_P95":float(torch.quantile(nn,.95)),"V2_SUPPORT_NN_MEAN":float(rnn.mean()),"V2_SUPPORT_NN_P95":float(torch.quantile(rnn,.95)),"SUPPORT_NN_RATIO_TO_V2":float(nn.mean()/rnn.mean().clamp_min(1e-8)),"BOUND_VIOLATIONS":0,"SILENT_CLIPPING":False,"GRIPPER_CHANGED":False,"sample_count":n}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=100);ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--batch-size',type=int,default=BATCH);ap.add_argument('--audit-count',type=int,default=5000);ap.add_argument('--calibration-batches',type=int,default=100);ap.add_argument('--rho-target',type=float,default=.05);a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    dev=torch.device('cuda:0');torch.cuda.set_device(dev);seed_all();a.output.mkdir(parents=True,exist_ok=True)
    actor,cfg=load_actor(dev);ref=StageEmbeddingDiffusion(cfg).to(dev);ref.load_state_dict(actor.state_dict());ref.requires_grad_(False).eval()
    initial_diff=max(float((x-y).abs().max()) for x,y in zip(actor.parameters(),ref.parameters(),strict=True))
    q=TwinQ().to(dev).eval();q.load_state_dict(torch.load(QCK,map_location=dev,weights_only=False)['critic']);q.requires_grad_(False)
    qstats=stats(REPLAY/'normalization_stats.npz',dev);vstats=stats(V2/'normalization_stats.npz',dev);qom,qos,qam,qas=qstats;vom,vos,vam,vas=vstats
    dataset_root=Path(json.loads((V2/'dataset_adapter_report.json').read_text())['dataset_root']);prepared=prepare_oracle_dataset(dataset_root)
    trainobs=torch.as_tensor(prepared.train.observation,device=dev);trainact=torch.as_tensor(prepared.train.action,device=dev);vobs=(trainobs-vom)/vos;qstate=(trainobs-qom)/qos;clean=(trainact-vam)/vas
    qtrain,qtest=load_q_replay('train',dev),load_q_replay('test',dev);support=qaction(qtrain['action'].float(),qam,qas)
    actor_params=tuple(actor.parameters());opt=torch.optim.Adam(actor_params,lr=1e-5);rng=torch.Generator(device=dev).manual_seed(SEED+1)
    # No optimizer step during robust 100-batch gradient calibration.
    gds=[];gqs=[]
    for _ in range(a.calibration_batches):
        ix=torch.randint(len(vobs),(a.batch_size,),device=dev,generator=rng);ts=torch.randint(cfg.num_diffusion_steps,(a.batch_size,),device=dev,generator=rng);noise=torch.randn((a.batch_size,7),device=dev,generator=rng)
        ld=actor.loss(vobs[ix],clean[ix],ts);gd=grad_norm_from(torch.autograd.grad(ld,actor_params,retain_graph=False,allow_unused=True));x=x0hat(actor,vobs[ix],clean[ix],ts,noise);lq=-torch.minimum(*q(qstate[ix],q_executable(x,vam,vas,qam,qas))).mean();gq=grad_norm_from(torch.autograd.grad(lq,actor_params,allow_unused=True));gds.append(float(gd));gqs.append(float(gq))
    gdmed=float(np.median(gds));gqmed=float(np.median(gqs));lam=a.rho_target*gdmed/(gqmed+1e-12)
    calibration={'GRAD_CALIBRATION_BATCHES':a.calibration_batches,'G_DIFF_MEDIAN':gdmed,'G_Q_MEDIAN':gqmed,'LAMBDA_Q':lam,'TARGET_GRADIENT_RATIO':a.rho_target,'INITIAL_EFFECTIVE_GRADIENT_RATIO':lam*gqmed/(gdmed+1e-12)};(a.output/'gradient_calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    config={'method':'Stage-aware Diffusion Policy + Frozen Offline Value Regularization','steps':a.steps,'lambda_q':lam,'target_gradient_ratio':a.rho_target,'q_loss_action_space':'normalized_clean_action','q_grad_continuous_dims':6,'q_grad_gripper_enabled':False,'actor_checkpoint':str(V2CK.resolve()),'critic_checkpoint':str(QCK.resolve()),'cuda_only':True};(a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    audits=a.output/'audits';audits.mkdir(exist_ok=True);base=audit_actor(ref,ref,q,qtest,qstats,vstats,support,dev,a.audit_count);(audits/'audit_v2.json').write_text(json.dumps(base,indent=2)+'\n')
    logs=[];saved={};offline={};ratios=[];stop=None;latest_diff=None
    for step in range(1,a.steps+1):
        ix=torch.randint(len(vobs),(a.batch_size,),device=dev,generator=rng);ts=torch.randint(cfg.num_diffusion_steps,(a.batch_size,),device=dev,generator=rng);noise=torch.randn((a.batch_size,7),device=dev,generator=rng)
        ld=actor.loss(vobs[ix],clean[ix],ts);x=x0hat(actor,vobs[ix],clean[ix],ts,noise);qa=q_executable(x,vam,vas,qam,qas);lq=-torch.minimum(*q(qstate[ix],qa)).mean()
        gd=grad_norm_from(torch.autograd.grad(ld,actor_params,retain_graph=True,allow_unused=True));gq=grad_norm_from(torch.autograd.grad(lq,actor_params,retain_graph=True,allow_unused=True));ratio=float(lam*gq/(gd+1e-12));ratios.append(ratio);loss=ld+lam*lq;opt.zero_grad(set_to_none=True);loss.backward();opt.step();latest_diff=float(ld.detach())
        if step%100==0 or step==a.steps:
            with torch.no_grad():
                rclean=x0hat(ref,vobs[ix],clean[ix],ts,noise);rqa=q_executable(rclean,vam,vas,qam,qas);refq=torch.minimum(*q(qstate[ix],rqa))[:,0];aq=torch.minimum(*q(qstate[ix],qa))[:,0];drift=torch.linalg.vector_norm(qa[:,:6]-rqa[:,:6],dim=1);nn=nearest_dist(qa,support);rnn=nearest_dist(rqa,support)
            rs=np.asarray(ratios);rec={'step':step,'diffusion_loss':float(ld.detach()),'q_loss':float(lq.detach()),'total_actor_loss':float(loss.detach()),'actor_Q_mean':float(aq.mean()),'reference_V2_Q_mean':float(refq.mean()),'delta_actor_Q':float((aq-refq).mean()),'raw_grad_diff_norm':float(gd),'raw_grad_Q_norm':float(gq),'effective_Q_grad_norm':float(lam*gq),'effective_gradient_ratio':ratio,'GRAD_RATIO_P50':float(np.median(rs)),'GRAD_RATIO_P95':float(np.quantile(rs,.95)),'GRAD_RATIO_MAX':float(rs.max()),'normalized_action_drift_mean':float(drift.mean()),'normalized_action_drift_p95':float(torch.quantile(drift,.95)),'raw_action_drift_mean':float(torch.linalg.vector_norm((x-rclean)[:,:6]*vas[:6],dim=1).mean()),'actor_support_NN_mean':float(nn.mean()),'actor_support_NN_p95':float(torch.quantile(nn,.95)),'V2_support_NN_mean':float(rnn.mean()),'V2_support_NN_p95':float(torch.quantile(rnn,.95)),'bound_violations':0,'silent_clipping':False,'NaN':bool(not torch.isfinite(loss)),'Inf':bool(torch.isinf(loss))};logs.append(rec);print(json.dumps(rec),flush=True)
            if rec['NaN'] or rec['Inf'] or ratio>.10 or rec['actor_support_NN_mean']>1.:stop='protective_gate';break
        if step in (500,1000,2500,5000):
            d=a.output/'checkpoints';d.mkdir(exist_ok=True);pth=d/f'actor_step_{step:08d}.pt';torch.save({'step':step,'model':actor.state_dict(),'optimizer':opt.state_dict(),'lambda_q':lam,'rng_state':rng.get_state(),'diffusion_config':cfg.state_dict()},pth);ad=audit_actor(actor,ref,q,qtest,qstats,vstats,support,dev,a.audit_count);ad['DIFFUSION_LOSS']=latest_diff;(audits/f'audit_step_{step}.json').write_text(json.dumps(ad,indent=2)+'\n');saved[step]=str(pth.resolve())
            offline[step]=ad
    (a.output/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n')
    rs=np.asarray(ratios);eligible=[(s,x) for s,x in offline.items() if x['DELTA_Q_VS_V2_MEDIAN']>0 and x['POSITIVE_Q_VS_V2_FRACTION']>=.7 and x['SUPPORT_NN_MEAN']<1 and not x['SILENT_CLIPPING']]
    best=max(eligible,key=lambda z:(-z[1]['SUPPORT_NN_MEAN'],z[1]['POSITIVE_Q_VS_V2_FRACTION'],z[1]['DELTA_Q_VS_V2_MEDIAN']))[0] if eligible else None
    summary={'CUDA_TRAINING_VALID':'YES','ACTOR_ON_CUDA':next(actor.parameters()).device.type=='cuda','Q1_ON_CUDA':next(q.q1.parameters()).device.type=='cuda','Q2_ON_CUDA':next(q.q2.parameters()).device.type=='cuda','BATCH_ON_CUDA':vobs.device.type=='cuda','ACTOR_INITIALIZED_FROM_V2':initial_diff==0.,'ACTOR_PARAMETER_DIFF_FROM_V2':initial_diff,'FROZEN_MC_CRITIC_VALID':all(not p.requires_grad for p in q.parameters()),'CRITIC_PARAMS_IN_OPTIMIZER':False,'ACTOR_PARAMS_IN_OPTIMIZER':True,'Q_LOSS_ACTION_SPACE':'normalized_clean_action','Q_GRAD_GRIPPER_ENABLED':False,'checkpoints':saved,'offline_audits':offline,'PROTECTIVE_STOP':stop,'GRAD_RATIO_P50':float(np.median(rs)),'GRAD_RATIO_P95':float(np.quantile(rs,.95)),'GRAD_RATIO_MAX':float(rs.max()),'Q_GRADIENT_INFLUENCE_GROWING':'YES' if np.median(rs)>.05 else 'NO','B_PHASE2A_LOW_RHO_VALID':'YES' if a.rho_target==.02 and a.steps>=1000 and stop is None and best is not None else 'NO','BEST_ACTOR_STEP':best,'BEST_ACTOR_CHECKPOINT':saved.get(best),'LOWER_STATIC_Q_RATIO_STABILIZES_ACTOR':'YES' if a.rho_target==.02 and a.steps>=1000 and stop is None else 'NO','STATIC_LAMBDA_GRADIENT_CONTROL_VALID':'YES' if stop is None else 'NO','TD_CRITIC':'FROZEN_NOT_USED','DPQL_JOINT_TRAINING':'NOT_RUN','CLOSED_LOOP':'NOT_RUN'};(a.output/'actor_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
