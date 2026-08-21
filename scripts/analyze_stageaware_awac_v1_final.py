#!/usr/bin/env python3
"""Read-only fixed-sample diagnostics for the completed Stage-aware AWAC run."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from mujoco_shared_control.awac.hybrid import HybridAWACConfig, HybridActor, HybridCritic

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/offline_awac/stageaware_awac_v1_4000/run_20260818T_STAGEAWARE_AWAC_V1_LOCAL20K_FORMAL'
N=100_000; B=1024; SEED=20260814
def main():
 p=torch.load(RUN/'checkpoints/checkpoint_step_20000.pt',map_location='cuda' if torch.cuda.is_available() else 'cpu',weights_only=False); dev=next(iter(p['actor'].values())).device
 cfg=HybridAWACConfig(**p['training_config']); actor=HybridActor(cfg).to(dev); actor.load_state_dict(p['actor']); actor.eval(); q1=HybridCritic(cfg).to(dev);q2=HybridCritic(cfg).to(dev);t1=HybridCritic(cfg).to(dev);t2=HybridCritic(cfg).to(dev)
 for model,key in ((q1,'critic_q1'),(q2,'critic_q2'),(t1,'target_q1'),(t2,'target_q2')): model.load_state_dict(p[key]);model.eval()
 d=np.load(RUN/'train.npz'); idx=np.random.default_rng(SEED).choice(len(d['reward']),size=min(N,len(d['reward'])),replace=False); mean=torch.as_tensor(p['observation_mean'],device=dev,dtype=torch.float32);std=torch.as_tensor(p['observation_std'],device=dev,dtype=torch.float32)
 vals={k:[] for k in ('q1','q2','target','adv','weight','motion_loss','gripper_loss')}
 with torch.inference_mode():
  torch.manual_seed(SEED)
  for ix in np.array_split(idx,max(1,len(idx)//B)):
   o=torch.as_tensor(d['obs'][ix],device=dev,dtype=torch.float32);no=torch.as_tensor(d['next_obs'][ix],device=dev,dtype=torch.float32);ca=torch.as_tensor(d['continuous_action'][ix],device=dev,dtype=torch.float32);ga=torch.as_tensor(d['gripper_action'][ix],device=dev,dtype=torch.float32).unsqueeze(1);rw=torch.as_tensor(d['reward'][ix],device=dev,dtype=torch.float32).unsqueeze(1);done=torch.as_tensor(d['terminated'][ix]|d['truncated'][ix],device=dev,dtype=torch.float32).unsqueeze(1);o=(o-mean)/std;no=(no-mean)/std
   nc,ng,_=actor.sample(no); target=rw+cfg.gamma*(1-done)*torch.minimum(t1(no,nc,ng),t2(no,nc,ng)); acont,agrip,_=actor.sample(o); dataset_q=torch.minimum(q1(o,ca,ga),q2(o,ca,ga)); value=torch.minimum(q1(o,acont,agrip),q2(o,acont,agrip));adv=dataset_q-value;w=torch.exp(torch.clamp(adv/cfg.awac_lambda,max=float(np.log(cfg.max_advantage_weight))));_,clp,glp=actor.dataset_log_prob(o,ca,ga,cfg.beta_gripper)
   for k,x in [('q1',q1(o,ca,ga)),('q2',q2(o,ca,ga)),('target',target),('adv',adv),('weight',w),('motion_loss',-w*clp),('gripper_loss',-w*glp)]: vals[k].append(x.flatten().cpu().numpy())
 vals={k:np.concatenate(v) for k,v in vals.items()};out={'sample_size':len(idx),'sample_seed':SEED,'q1_mean':float(vals['q1'].mean()),'q2_mean':float(vals['q2'].mean()),'target_q_mean':float(vals['target'].mean()),'advantage':{'mean':float(vals['adv'].mean()),'std':float(vals['adv'].std()),'p10':float(np.quantile(vals['adv'],.1)),'p50':float(np.median(vals['adv'])),'p90':float(np.quantile(vals['adv'],.9))},'awac_weight':{'mean':float(vals['weight'].mean()),'p90':float(np.quantile(vals['weight'],.9)),'p99':float(np.quantile(vals['weight'],.99)),'cap_fraction':float(np.mean(vals['weight']>=cfg.max_advantage_weight-1e-6))},'actor_loss_components':{'motion':float(vals['motion_loss'].mean()),'gripper':float(vals['gripper_loss'].mean())},'finite':bool(all(np.isfinite(x).all() for x in vals.values())),'gradient_updates':20000}
 (RUN/'final_fixed_sample_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
