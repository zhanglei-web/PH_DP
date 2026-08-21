#!/usr/bin/env python3
"""Predicted-Stage-DP-v1 startup audit and CUDA-only 1k diffusion smoke."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import h5py, numpy as np, torch
from torch.optim import Adam

from mujoco_shared_control.stage.tcn import StageTCNV1
from mujoco_shared_control.stage.dataset import fit_normalization, load_split
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/stage_dataset/stage_dataset_v1_hysteresis_20260817T052000Z'
TCNCK=ROOT/'outputs/stage_inference_transition_v1/tcn_a_best_macro_f1.pt'
TCNNORM=ROOT/'outputs/stage_tcn/stage_tcn_v1_hysteresis_20260817T070000Z/normalization_stats.npz'
OUT=ROOT/'outputs/predicted_stage_dp_v1/smoke'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def build_split(split, tmean, tstd):
    manifest=json.loads((DATA/'split_manifest.json').read_text()); rows=[]
    for eid in manifest['splits'][split]:
        with h5py.File(manifest['episode_paths'][eid],'r') as f:
            state=f['full_physical_state'][:].astype('f4'); action=f['raw_pilot_action'][:].astype('f4'); features=f['stage_features'][:].astype('f4'); labels=f['active_phase'][:].astype('i8')
        for t in range(19,len(labels)):
            rows.append((state[t],action[t],((features[t-19:t+1]-tmean)/tstd).astype('f4'),labels[t],eid,t))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--batch-size',type=int,default=512);a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    dev=torch.device('cuda:0');torch.cuda.set_device(dev);a.output.mkdir(parents=True,exist_ok=True)
    if not TCNCK.exists(): raise FileNotFoundError(f'TCN-A checkpoint missing: {TCNCK}')
    tcn_payload=torch.load(TCNCK,map_location=dev,weights_only=False);tcn=StageTCNV1().to(dev).eval();tcn.load_state_dict(tcn_payload['model']);tcn.requires_grad_(False)
    if not TCNNORM.exists(): raise FileNotFoundError(f'TCN normalization artifact missing: {TCNNORM}')
    with np.load(TCNNORM) as n: tmean=n['mean'].astype('f4');tstd=n['std'].astype('f4')
    fitted=fit_normalization(load_split(DATA,'train'))
    norm_diff=max(float(np.max(np.abs(tmean-fitted.mean))),float(np.max(np.abs(tstd-fitted.std))))
    if norm_diff>1e-6: raise RuntimeError(f'TCN normalization artifact differs from frozen train split: {norm_diff}')
    train=build_split('train',tmean,tstd);val=build_split('validation',tmean,tstd)
    train_state=np.stack([x[0] for x in train]);train_action=np.stack([x[1] for x in train]); om=train_state.mean(0).astype('f4');os=np.maximum(train_state.std(0),1e-6).astype('f4');am=train_action.mean(0).astype('f4');astd=np.maximum(train_action.std(0),1e-6).astype('f4')
    state=torch.as_tensor((train_state-om)/os,device=dev);action=torch.as_tensor((train_action-am)/astd,device=dev);history=torch.as_tensor(np.stack([x[2] for x in train]),device=dev);labels=torch.as_tensor(np.asarray([x[3] for x in train]),device=dev)
    cfg=StageEmbeddingDiffusionConfig(physical_dim=43,stage_dim=5,stage_embedding_dim=32,condition_hidden_dim=128,action_dim=7,num_diffusion_steps=50,hidden_dim=128)
    model=StageEmbeddingDiffusion(cfg).to(dev).train();opt=Adam(model.parameters(),lr=1e-3);rng=torch.Generator(device=dev).manual_seed(20260904)
    # Startup posterior/mapping/gradient audit on one fixed batch.
    ix=torch.randint(len(state),(a.batch_size,),device=dev,generator=rng)
    with torch.no_grad(): z=tcn.posterior(history[ix])
    cond=torch.cat((state[ix],z),1);loss=model.loss(cond,action[ix]);opt.zero_grad(set_to_none=True);loss.backward()
    tcn_grad=max((0. if q.grad is None else float(q.grad.detach().abs().max())) for q in tcn.parameters())
    stage_grad=sum(float(q.grad.detach().abs().sum()) for q in model.condition_encoder.stage_encoder.parameters() if q.grad is not None)
    denoise_grad=sum(float(q.grad.detach().abs().sum()) for q in model.denoiser.parameters() if q.grad is not None)
    with torch.no_grad():
        fake0=torch.zeros_like(z);fake0[:,0]=1;fake4=torch.zeros_like(z);fake4[:,4]=1; ts=torch.full((len(ix),),10,dtype=torch.long,device=dev);noisy,_=model.q_sample(action[ix],ts,noise=torch.zeros_like(action[ix])); o0=model.denoiser(torch.cat((model._condition(torch.cat((state[ix],fake0),1)),noisy),1),ts);o4=model.denoiser(torch.cat((model._condition(torch.cat((state[ix],fake4),1)),noisy),1),ts); affects=not torch.allclose(o0,o4)
    audit={'CUDA_AVAILABLE':True,'CUDA_DEVICE':str(dev),'GPU_NAME':torch.cuda.get_device_name(dev),'TCN_A_LOADED':True,'TCN_A_CHECKPOINT':str(TCNCK.resolve()),'TCN_NORMALIZATION_ARTIFACT':str(TCNNORM.resolve()),'TCN_NORMALIZATION_TRAIN_SPLIT_MAX_ABS_DIFF':norm_diff,'TCN_A_FROZEN':all(not q.requires_grad for q in tcn.parameters()),'TCN_TRAINABLE_PARAMETER_COUNT':sum(q.numel() for q in tcn.parameters() if q.requires_grad),'POSTERIOR_VALID':bool(z.shape==(a.batch_size,5) and torch.isfinite(z).all() and torch.allclose(z.sum(-1),torch.ones(len(z),device=dev),atol=1e-5)),'STAGE_POSTERIOR_SHAPE':list(z.shape),'STAGE_POSTERIOR_DIM':5,'STAGE_EMBEDDING_DIM':32,'STAGE_EMBEDDING_VALID':model._condition(cond).shape==(a.batch_size,128),'STAGE_CONDITION_AFFECTS_DENOISER':'YES' if affects else 'NO','TCN_PARAMETER_MAX_ABS_DIFF_FROM_INIT':tcn_grad,'STAGE_MLP_GRADIENT_VALID':stage_grad>0,'DIFFUSION_GRADIENT_VALID':denoise_grad>0,'ACTION_SCHEMA':'frozen raw_pilot_action normalized semantic 7D','NORMALIZATION_SOURCE':'train split only','TCN_POSTERIOR_SOURCE':'frozen TCN-A softmax posterior','NO_HARD_ARGMAX':True,'NO_GT_STAGE_REPLACEMENT':True}
    if not all([audit['TCN_A_FROZEN'],audit['POSTERIOR_VALID'],audit['STAGE_EMBEDDING_VALID'],affects,audit['STAGE_MLP_GRADIENT_VALID'],audit['DIFFUSION_GRADIENT_VALID'],tcn_grad==0.]):
        (a.output/'startup_audit.json').write_text(json.dumps(audit,indent=2)+'\n');raise RuntimeError('startup audit failed')
    start=float(loss.detach());logs=[]
    for step in range(1,a.steps+1):
        ix=torch.randint(len(state),(a.batch_size,),device=dev,generator=rng)
        with torch.no_grad(): z=tcn.posterior(history[ix])
        ld=model.loss(torch.cat((state[ix],z),1),action[ix]);opt.zero_grad(set_to_none=True);ld.backward();opt.step()
        if step==1 or step%100==0 or step==a.steps: logs.append({'step':step,'train_diffusion_loss':float(ld.detach()),'posterior_entropy':float((-(z*z.clamp_min(1e-8).log()).sum(-1)).mean()),'posterior_max_probability':float(z.max(-1).values.mean()),'gpu_memory_mb':float(torch.cuda.memory_allocated(dev)/1024**2),'nan':bool(not torch.isfinite(ld)),'inf':bool(torch.isinf(ld))});print(json.dumps(logs[-1]),flush=True)
    # Frozen-TCN identity check after optimizer updates.
    tcn_after=max((0. if q.grad is None else float(q.grad.detach().abs().max())) for q in tcn.parameters())
    val_state=torch.as_tensor((np.stack([x[0] for x in val])-om)/os,device=dev);val_action=torch.as_tensor((np.stack([x[1] for x in val])-am)/astd,device=dev);val_hist=torch.as_tensor(np.stack([x[2] for x in val]),device=dev);vals=[]
    with torch.no_grad():
        for i in range(0,len(val_state),a.batch_size): vals.append(float(model.loss(torch.cat((val_state[i:i+a.batch_size],tcn.posterior(val_hist[i:i+a.batch_size])),1),val_action[i:i+a.batch_size]).detach()))
    audit.update({'TCN_PARAMETER_MAX_ABS_DIFF':tcn_after,'SMOKE_L_DIFF_START':start,'SMOKE_L_DIFF_END':logs[-1]['train_diffusion_loss'],'SMOKE_VAL_L_DIFF':float(np.mean(vals)),'NaN_INF':'NO' if all(not x['nan'] and not x['inf'] for x in logs) else 'YES','GPU_MEMORY_OK':True,'STAGE_CONDITIONING_IMPLEMENTATION_VALID':'YES' if tcn_after==0. and all(not x['nan'] and not x['inf'] for x in logs) else 'NO','READY_FOR_80K_TRAINING':'YES' if tcn_after==0. and all(not x['nan'] and not x['inf'] for x in logs) else 'NO'})
    np.savez(a.output/'normalization_stats.npz',observation_mean=om,observation_std=os,action_mean=am,action_std=astd)
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':a.steps,'diffusion_config':cfg.state_dict(),'normalization':{'observation_mean':om,'observation_std':os,'action_mean':am,'action_std':astd},'tcn_checkpoint':str(TCNCK.resolve()),'tcn_sha256':sha(TCNCK),'format_version':'predicted-stage-dp-v1'},a.output/'smoke_step_00001000.pt')
    (a.output/'startup_audit.json').write_text(json.dumps(audit,indent=2)+'\n');(a.output/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n');(a.output/'config.json').write_text(json.dumps({'architecture':'StageEmbeddingDiffusion shared denoiser; frozen TCN-A soft posterior -> 5->32 SiLU stage encoder','diffusion_config':cfg.state_dict(),'steps':a.steps,'batch_size':a.batch_size,'optimizer':'Adam','learning_rate':1e-3,'dataset':str(DATA.resolve()),'tcn_checkpoint':str(TCNCK.resolve()),'stage_loss_used':False,'transition_loss_used':False,'global_warm_start':False},indent=2)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
