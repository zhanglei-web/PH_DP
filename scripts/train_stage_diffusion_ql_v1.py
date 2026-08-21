#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from mujoco_shared_control.rss2023.model import DiffusionConfig
from mujoco_shared_control.rss2023.stage_diffusion_ql import DiffusionQLConfig, DiffusionQLTrainer, Replay, checkpoint_payload, load_replay_stats, seed_everything

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/diffusion_ql/stage_diffusion_ql_v1_20260819/replay'
OUT=ROOT/'outputs/diffusion_ql/stage_diffusion_ql_v1_20260819'

def main():
 p=argparse.ArgumentParser(); p.add_argument('--eta-q',type=float,default=0.0); p.add_argument('--steps',type=int,default=80000); p.add_argument('--output',type=Path,default=OUT); p.add_argument('--device',default='auto'); a=p.parse_args()
 dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device); seed_everything(20260819)
 mean,std,action_mean,action_std=load_replay_stats(DATA/'train.npz'); cfg=DiffusionQLConfig(eta_q=a.eta_q,steps=a.steps); dcfg=DiffusionConfig(observation_dim=48,action_dim=7,num_diffusion_steps=50,hidden_dim=128)
 a.output.mkdir(parents=True,exist_ok=True); np.savez(a.output/'normalization_stats.npz',observation_mean=mean,observation_std=std,action_mean=action_mean,action_std=action_std)
 (a.output/'config.json').write_text(json.dumps({'format_version':'stage-diffusion-ql-v1','dataset':str(DATA.resolve()),'diffusion':dcfg.state_dict(),'training':cfg.__dict__,'reward_version':'V1.2','gamma':.995,'state_mode':'physical43_active_stage5'},indent=2)+'\n')
 replay=Replay(DATA/'train.npz',dev,action_mean,action_std); trainer=DiffusionQLTrainer(cfg,dcfg,mean,std,dev); log=[]; started=time.monotonic()
 for step in range(1,a.steps+1):
  m=trainer.update(replay)
  if step==1 or step%1000==0 or step==a.steps:
   row={'step':step,**m,'elapsed_seconds':time.monotonic()-started}; log.append(row); print(json.dumps(row),flush=True)
   torch.save(checkpoint_payload(trainer,step,cfg,dcfg,action_mean,action_std),a.output/'latest.pt')
   if step%10000==0:
    (a.output/'checkpoints').mkdir(exist_ok=True); torch.save(checkpoint_payload(trainer,step,cfg,dcfg,action_mean,action_std),a.output/'checkpoints'/f'step_{step:08d}.pt')
 (a.output/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in log)+'\n'); (a.output/'training_report.json').write_text(json.dumps({'status':'PASS','steps':a.steps,'eta_q':a.eta_q,'final':log[-1]},indent=2)+'\n')
if __name__=='__main__': main()
