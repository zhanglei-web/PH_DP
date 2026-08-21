#!/usr/bin/env python3
"""Persistent local 20k Stage-aware Hybrid AWAC formal trainer."""
from __future__ import annotations
import csv,json,os
from pathlib import Path
import numpy as np,torch
from mujoco_shared_control.awac.hybrid import HybridAWACConfig,HybridAWACTrainer,HybridReplay
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ['STAGEAWARE_AWAC_OUT'])
def main():
 p=torch.load(OUT/'actor_step0.pt',map_location='cpu',weights_only=False);cfg=HybridAWACConfig(**p['training_config']);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');r=HybridReplay(OUT/'train.npz',dev);t=HybridAWACTrainer(cfg,np.asarray(p['observation_mean']),np.asarray(p['observation_std']),p['actor'],dev);(OUT/'checkpoints').mkdir(exist_ok=True);torch.save(t.checkpoint({'state_mode':'physical43_active_stage5','reward_version':'V1.2'}),OUT/'checkpoints/checkpoint_step_00000.pt');hist=[]
 for step in range(1,20001):
  x=t.update(r.sample(cfg.batch_size,t.generator));hist.append(x)
  if step in (2500,5000,10000,20000):torch.save(t.checkpoint({'state_mode':'physical43_active_stage5','reward_version':'V1.2'}),OUT/f'checkpoints/checkpoint_step_{step:05d}.pt')
  if step%100==0:
   with (OUT/'training_runtime_status.json').open('w') as f:json.dump({'pid':os.getpid(),'current_step':step,'latest_checkpoint':max([0]+[s for s in (2500,5000,10000,20000) if s<=step])},f)
 with (OUT/'training_history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0]);w.writeheader();w.writerows(hist)
if __name__=='__main__':main()
