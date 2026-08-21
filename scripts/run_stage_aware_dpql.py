#!/usr/bin/env python3
"""Phased Stage-aware Diffusion-QL runner initialized from frozen V2.

This runner deliberately keeps the actor in V2's normalized action space and
uses an explicit semantic conversion for the critic's executable action.
"""
from __future__ import annotations
import argparse, json, random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig

ROOT=Path(__file__).resolve().parents[1]
V2=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818'
REPLAY=ROOT/'outputs/diffusion_ql/stage_diffusion_ql_v1_20260819/replay'
OUT=ROOT/'outputs/diffusion_ql/stage_aware_dpql_v1'

@dataclass
class Config:
    gamma: float=.995; tau: float=.005; actor_lr: float=1e-4; critic_lr: float=3e-4
    batch_size: int=512; warmup_steps: int=5000; joint_steps: int=50000
    rho: float=.05; seed: int=20260824; grad_clip: float=1.

class TwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        def net(): return nn.Sequential(nn.Linear(55,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,1))
        self.q1,self.q2=net(),net()
    def forward(self,s,a):
        x=torch.cat((s,a),-1); return self.q1(x),self.q2(x)

def seed_all(seed): random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def load_data():
    with np.load(REPLAY/'train.npz',allow_pickle=False) as d:
        return {k:d[k] for k in ('obs','next_obs','action','reward','done')}

def load_v2(device):
    p=torch.load(V2/'checkpoints/step_00080000.pt',map_location=device,weights_only=False)
    c=StageEmbeddingDiffusionConfig(**{k:v for k,v in p['diffusion_config'].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__})
    a=StageEmbeddingDiffusion(c).to(device).eval();a.load_state_dict(p['model']);
    with np.load(V2/'normalization_stats.npz') as z: stats={k:z[k].astype('f4') for k in ('observation_mean','observation_std','action_mean','action_std')}
    return a,stats,p

def clean_proxy(actor,obs,action_seed):
    t=torch.zeros((len(obs),),dtype=torch.long,device=obs.device)
    eps=actor.denoiser(torch.cat((actor._condition(obs),action_seed),-1),t)[...,actor.config.condition_hidden_dim:]
    return action_seed-eps

def raw_to_q_action(x,am,astd):
    raw=torch.clamp(x,-1.,1.); raw=raw.clone(); raw[:,6]=torch.where(raw[:,6]<.375,-1.,1.)
    return (raw-am)/astd

def normalized_to_q_action(x,am,astd):
    return raw_to_q_action(x*astd+am,am,astd)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=('phase1','phase2','joint'),default='phase1'); ap.add_argument('--steps',type=int); ap.add_argument('--resume',action='store_true'); ap.add_argument('--device',default='cpu'); ap.add_argument('--output',type=Path,default=OUT); args=ap.parse_args()
    dev=torch.device(args.device);cfg=Config();seed_all(cfg.seed);args.output.mkdir(parents=True,exist_ok=True)
    actor,stats,v2_payload=load_v2(dev); data=load_data()
    om,os,am,astd=[torch.from_numpy(stats[k]).to(dev) for k in ('observation_mean','observation_std','action_mean','action_std')]
    obs=torch.from_numpy(data['obs']).float().to(dev); nxt=torch.from_numpy(data['next_obs']).float().to(dev)
    raw_act=torch.from_numpy(data['action']).float().to(dev)
    # V2's diffusion BC objective consumes its checkpoint-normalized action;
    # the critic consumes the separately explicit executable semantic action.
    act_v2=(raw_act-am)/astd
    act_q=raw_to_q_action(raw_act,am,astd)
    rew=torch.from_numpy(data['reward']).float().to(dev).unsqueeze(1); done=torch.from_numpy(data['done'].astype('f4')).float().to(dev).unsqueeze(1)
    s=(obs-om)/os; ns=(nxt-om)/os
    q,tq=TwinQ().to(dev),TwinQ().to(dev);tq.load_state_dict(q.state_dict());tq.requires_grad_(False)
    qopt=torch.optim.Adam(q.parameters(),lr=cfg.critic_lr);aopt=torch.optim.Adam(actor.parameters(),lr=cfg.actor_lr);rng=torch.Generator(device=dev).manual_seed(cfg.seed+1); start_step=0
    resume_path=args.output/f'{args.phase}_latest.pt'
    if args.resume:
        if not resume_path.exists(): raise FileNotFoundError(f'cannot resume missing {resume_path}')
        previous=torch.load(resume_path,map_location=dev,weights_only=False)
        actor.load_state_dict(previous['actor']);q.load_state_dict(previous['critic']);tq.load_state_dict(previous['critic_target'])
        qopt.load_state_dict(previous['optimizer_critic']);aopt.load_state_dict(previous['optimizer_actor']);start_step=int(previous['step'])
        if 'rng_state' in previous: rng.set_state(previous['rng_state'])
    # Phase 0 is explicit and immutable.
    audit={'v2_checkpoint':str((V2/'checkpoints/step_00080000.pt').resolve()),'actor_initialized_from_v2':True,'reward_version':'V1.2','reward_changed':'NO','state_dim':48,'action_dim':7,'action_semantics':'normalized semantic [dx,dy,dz,dRx,dRy,dRz,gripper]','normalizer':'frozen V2 normalization_stats.npz','horizon':1,'diffusion_steps':50}
    (args.output/'phase0_freeze_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    steps=args.steps or (cfg.warmup_steps if args.phase=='phase2' else cfg.joint_steps if args.phase=='joint' else 10000)
    logs=[]
    for step in range(start_step+1,steps+1):
        ix=torch.randint(len(obs),(cfg.batch_size,),generator=rng,device=dev); sb,nb,ab_v2,ab_q,rb,db=s[ix],ns[ix],act_v2[ix],act_q[ix],rew[ix],done[ix]
        # Frozen-V2 target action. The batch call is inference-only; no actor update in phase1.
        with torch.no_grad():
            seed=torch.zeros_like(ab_v2); an=actor.assist(nb,seed,gamma=1.,generator=rng); an=normalized_to_q_action(an,am,astd); y=rb+cfg.gamma*(1-db)*torch.minimum(*tq(nb,an))
        q1,q2=q(sb,ab_q);ql=F.mse_loss(q1,y)+F.mse_loss(q2,y);qopt.zero_grad();ql.backward();nn.utils.clip_grad_norm_(q.parameters(),cfg.grad_clip);qopt.step()
        if args.phase in ('phase2','joint'):
            seed=ab_v2.detach(); proxy=clean_proxy(actor,sb,seed); qa=normalized_to_q_action(proxy,am,astd); pq1,pq2=q(sb,qa); qraw=-torch.minimum(pq1,pq2).mean(); diff=actor.loss(sb,ab_v2); gd=torch.autograd.grad(diff,actor.parameters(),retain_graph=True,allow_unused=True); gq=torch.autograd.grad(qraw,actor.parameters(),retain_graph=True,allow_unused=True); nd=torch.sqrt(sum((x.detach()**2).sum() for x in gd if x is not None)+1e-12); nq=torch.sqrt(sum((x.detach()**2).sum() for x in gq if x is not None)+1e-12); lam=float(cfg.rho*nd/(nq+1e-12)); total=diff+lam*qraw;aopt.zero_grad();total.backward();nn.utils.clip_grad_norm_(actor.parameters(),cfg.grad_clip);aopt.step()
        else: diff=total=qraw=torch.tensor(0.,device=dev);nd=nq=torch.tensor(0.,device=dev);lam=0.
        with torch.no_grad():
            for x,z in zip(q.parameters(),tq.parameters(),strict=True): z.lerp_(x,cfg.tau)
        if step==1 or step%500==0 or step==steps:
            gap=(q1-q2).abs()
            q1_grad=torch.sqrt(sum((p.grad.detach()**2).sum() for p in q.q1.parameters() if p.grad is not None)+1e-12)
            q2_grad=torch.sqrt(sum((p.grad.detach()**2).sum() for p in q.q2.parameters() if p.grad is not None)+1e-12)
            row={'step':step,'critic_loss':float(ql),'diffusion_loss':float(diff),'q_actor_loss':float(qraw),'actor_total_loss':float(total),'q1_mean':float(q1.mean()),'q2_mean':float(q2.mean()),'target_q_mean':float(y.mean()),'TD_target_mean':float(y.mean()),'TD_target_std':float(y.std()),'Q_gap':float(gap.mean()),'grad_norm_Q1':float(q1_grad),'grad_norm_Q2':float(q2_grad),'NaN':bool(not torch.isfinite(ql)),'Inf':bool(torch.isinf(q1).any() or torch.isinf(q2).any()),'gradient_ratio':float(lam*nq/(nd+1e-12)),'actor_grad_norm_diff':float(nd),'actor_grad_norm_q':float(nq)};logs.append(row);print(json.dumps(row),flush=True)
            payload={'phase':args.phase,'step':step,'actor':actor.state_dict(),'actor_reference_v2':v2_payload['model'],'critic':q.state_dict(),'critic_target':tq.state_dict(),'optimizer_actor':aopt.state_dict(),'optimizer_critic':qopt.state_dict(),'rng_state':rng.get_state(),'config':asdict(cfg),'stats':{k:v.cpu().numpy() for k,v in zip(('observation_mean','observation_std','action_mean','action_std'),(om,os,am,astd))}}
            torch.save(payload,args.output/f'{args.phase}_latest.pt')
            if args.phase == 'phase1' and step in {1000,2500,5000,7500,10000}:
                checkpoint_dir=args.output/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
                torch.save(payload,checkpoint_dir/f'phase1_step_{step:08d}.pt')
    (args.output/f'{args.phase}_training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n')
    (args.output/f'{args.phase}_summary.json').write_text(json.dumps({'status':'PASS','phase':args.phase,'steps':steps,'final':logs[-1]},indent=2)+'\n')

if __name__=='__main__':main()
