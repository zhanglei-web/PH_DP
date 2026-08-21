#!/usr/bin/env python3
"""Q3: CUDA-only Bellman Twin-Q fine-tuning from the validated MC critic."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from train_stage_mc_twin_q import TwinQ, load, qaction, audit
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig

ROOT=Path(__file__).resolve().parents[1]
MC=ROOT/'outputs/diffusion_ql/stage_mc_twin_q_v2'
REPLAY=MC/'replay'; OUT=ROOT/'outputs/diffusion_ql/stage_bellman_twin_q_v1'
V2=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818'
GAMMA=.995; TAU=.005; BATCH=512; SEED=20260828
CHECKPOINT_STEPS=(250,500,1000,1500,2500,5000)

def load_v2(device):
    p=torch.load(V2/'checkpoints/step_00080000.pt',map_location=device,weights_only=False)
    cfg=StageEmbeddingDiffusionConfig(**{k:v for k,v in p['diffusion_config'].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__})
    model=StageEmbeddingDiffusion(cfg).to(device).eval();model.load_state_dict(p['model'])
    return model

@torch.no_grad()
def next_action(actor, obs, am, astd, generator):
    seed=torch.zeros((len(obs),7),device=obs.device)
    x=actor.assist(obs,seed,gamma=1.,generator=generator)
    return qaction(x*astd+am,am,astd)

@torch.no_grad()
def td_summary(actor,target,data,normal,device,seed):
    om,os,am,astd=normal; obs=data['obs'].float(); nxt=data['next_obs'].float(); rew=data['reward'].float().reshape(-1,1); done=data['done'].float().reshape(-1,1)
    ns=(nxt-om)/os; gen=torch.Generator(device=device).manual_seed(seed); vals=[]
    for i in range(0,len(ns),2048):
        n=ns[i:i+2048]; a=next_action(actor,n,am,astd,gen); vals.append((rew[i:i+2048]+GAMMA*(1-done[i:i+2048])*torch.minimum(*target(n,a))).flatten())
    y=torch.cat(vals);return {'TD_TARGET_MEAN':float(y.mean()),'TD_TARGET_STD':float(y.std()),'TD_TARGET_MIN':float(y.min()),'TD_TARGET_MAX':float(y.max())}

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    device=torch.device('cuda:0');torch.cuda.set_device(device);p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=5000);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    actor=load_v2(device); data=load('train',device); test=load('test',device)
    with np.load(REPLAY/'normalization_stats.npz') as z: normal=tuple(torch.as_tensor(z[k],device=device) for k in ('observation_mean','observation_std','action_mean','action_std'))
    om,os,am,astd=normal; s=(data['obs'].float()-om)/os; act=qaction(data['action'].float(),am,astd); rew=data['reward'].float().reshape(-1,1); done=data['done'].float().reshape(-1,1)
    mc_payload=torch.load(MC/'checkpoints/mc_step_00005000.pt',map_location=device,weights_only=False);q=TwinQ().to(device);q.load_state_dict(mc_payload['critic']);target=TwinQ().to(device);target.load_state_dict(mc_payload['critic']);target.requires_grad_(False);opt=torch.optim.Adam(q.parameters(),lr=3e-4);rng=torch.Generator(device=device).manual_seed(SEED+1)
    cuda_audit={'CUDA_TRAINING_VALID':'YES','CUDA_DEVICE':str(device),'V2_ON_CUDA':str(next(actor.parameters()).device)==str(device),'Q1_ON_CUDA':str(next(q.q1.parameters()).device)==str(device),'Q2_ON_CUDA':str(next(q.q2.parameters()).device)==str(device),'TARGET_Q1_ON_CUDA':str(next(target.q1.parameters()).device)==str(device),'TARGET_Q2_ON_CUDA':str(next(target.q2.parameters()).device)==str(device),'BATCH_ON_CUDA':str(s.device)==str(device)}
    if not all(v for k,v in cuda_audit.items() if k.endswith('_ON_CUDA')) or not cuda_audit['BATCH_ON_CUDA']: raise RuntimeError('CUDA placement audit failed')
    (a.output/'cuda_audit.json').write_text(json.dumps(cuda_audit,indent=2)+'\n')
    logs=[];audits=[];baseline={'step':0,'Q_RETURN_PEARSON':.9562298059,'Q_RETURN_SPEARMAN':.6658250887,'PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY':.9977578475,'MATCHED_PAIR_COVERAGE':.892,'Q1_Q2_DISAGREEMENT_MEAN':.0397039875,'Q_SCALE_EXPLOSION':False,'Q_VALUE_COLLAPSE':False};(a.output/'step0_mc_baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    for step in range(1,a.steps+1):
        ix=torch.randint(len(s),(BATCH,),device=device,generator=rng);sb=s[ix];ab=act[ix];rb=rew[ix];db=done[ix]
        with torch.no_grad():
            na=next_action(actor,((data['next_obs'][ix].float()-om)/os),am,astd,rng);y=rb+GAMMA*(1-db)*torch.minimum(*target(((data['next_obs'][ix].float()-om)/os),na))
        q1,q2=q(sb,ab);l1=F.mse_loss(q1,y);l2=F.mse_loss(q2,y);loss=l1+l2;opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            for src,dst in zip(q.parameters(),target.parameters(),strict=True):dst.lerp_(src,TAU)
        if step%100==0 or step==a.steps:
            logs.append({'step':step,'critic_loss':float(loss),'q1_loss':float(l1),'q2_loss':float(l2),'td_target_mean':float(y.mean()),'td_target_std':float(y.std()),'q_gap':float((q1-q2).abs().mean()),'nan':bool(not torch.isfinite(loss)),'inf':bool(torch.isinf(q1).any() or torch.isinf(q2).any()),'gpu_memory_mb':float(torch.cuda.memory_allocated(device)/1024**2)})
        if step in CHECKPOINT_STEPS:
            payload={'step':step,'critic':q.state_dict(),'critic_target':target.state_dict(),'optimizer_critic':opt.state_dict(),'rng_state':rng.get_state(),'config':{'gamma':GAMMA,'tau':TAU,'batch_size':BATCH,'target_policy':'frozen V2','reward_version':'V1.2'},'normalization':{k:v.cpu().numpy() for k,v in zip(('observation_mean','observation_std','action_mean','action_std'),normal)}};d=a.output/'checkpoints';d.mkdir(exist_ok=True);path=d/f'td_step_{step:08d}.pt';torch.save(payload,path);r=audit(path,test,normal,device,a.output);r.update(td_summary(actor,target,test,normal,device,SEED+step));r['DELTA_PEARSON_FROM_MC']=r['Q_RETURN_PEARSON']-baseline['Q_RETURN_PEARSON'];r['DELTA_SPEARMAN_FROM_MC']=r['Q_RETURN_SPEARMAN']-baseline['Q_RETURN_SPEARMAN'];r['DELTA_RANKING_FROM_MC']=r['PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY']-baseline['PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY'];r['TD_TARGET_STD_RATIO_TO_MC']=r['TD_TARGET_STD']/1.3339414597;r['Q_STD_RATIO_TO_MC']=r['minQ']['std']/1.2360633612;audits.append(r)
            if not (r['Q_RETURN_PEARSON']>=.5 and r['Q_RETURN_SPEARMAN']>=.5 and r['Q_MATCHED_ACTION_RANKING_VALID']=='YES' and r['MATCHED_PAIR_COVERAGE']>=.5 and not r['Q_VALUE_COLLAPSE'] and not r['Q_SCALE_EXPLOSION']): break
    (a.output/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n');valid=bool(audits) and all(x['Q_RETURN_PEARSON']>=.5 and x['Q_RETURN_SPEARMAN']>=.5 and x['Q_MATCHED_ACTION_RANKING_VALID']=='YES' and x['MATCHED_PAIR_COVERAGE']>=.5 and not x['Q_VALUE_COLLAPSE'] and not x['Q_SCALE_EXPLOSION'] for x in audits);best=max(audits,key=lambda x:(x['PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY'],x['Q_RETURN_SPEARMAN'],x['Q_RETURN_PEARSON']));degrading=len(audits)>=2 and all(audits[i]['Q_RETURN_SPEARMAN']<audits[i-1]['Q_RETURN_SPEARMAN'] and audits[i]['PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY']<audits[i-1]['PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY'] for i in range(1,len(audits)));(a.output/'q3_comparison.json').write_text(json.dumps({'step0':baseline,'audits':audits,'Q3_TD_CRITIC_VALID':'YES' if valid else 'NO','TD_CRITIC_DEGRADING':degrading,'BEST_TD_CRITIC_STEP':best['step'] if valid else 'NOT_SELECTED','READY_FOR_Q4_CRITIC_AUDIT':'YES' if valid else 'NO','DPQL_ACTOR':'NOT_RUN','CLOSED_LOOP':'NOT_RUN'},indent=2)+'\n')
if __name__=='__main__':main()
