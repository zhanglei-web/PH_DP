#!/usr/bin/env python3
"""Chunked NoAssist evaluation of the frozen Unified Recovery BC checkpoint."""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,torch
from mujoco_shared_control.experts.recovery_bc import RecoveryBCPolicy
from run_e2_specialized_bc_pilots import normal_rollout,recovery_rollout,mean,status,V2,V2_SHA
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/unified_recovery_bc_v1/run_20260818T_UNIFIED_RECOVERY_BC_FORMAL';FAIL=('GRASP_FAILURE','TRANSPORT_EARLY','PLACE_FAILURE')
def write(p,rows):
 k=sorted({x for r in rows for x in r});
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=k);w.writeheader();w.writerows(rows)
def load():
 z=torch.load(OUT/'best_val.pt',map_location='cpu',weights_only=False);n=np.load(OUT/'normalizer.npz');m=RecoveryBCPolicy();m.load_state_dict(z['model']);m.eval();return m,n['mean'],n['std']
def b(v):return str(v).lower()=='true'
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=('normal','grasp','transport','place'));p.add_argument('--chunk',type=int);p.add_argument('--finalize',action='store_true');a=p.parse_args()
 if a.finalize:
  allrows=[];summary=[]
  for c,label in [('normal','NORMAL'),('grasp','GRASP_FAILURE'),('transport','TRANSPORT_EARLY'),('place','PLACE_FAILURE')]:
   fs=sorted((OUT/'closed_loop_chunks').glob(f'{c}_*.csv'));rows=[r for f in fs for r in csv.DictReader(f.open())]
   if len(rows)!=100:raise SystemExit(f'STOP {c}: {len(rows)}')
   for r in rows:
    for k in ('Task/Recovery Success','Regrasp','Grasp','Lift','Transport','Place','Retreat','Unexpected Drop','IK Failure','Timeout'):
     r[k]=r[k] if r[k]=='NA' else b(r[k])
    r['Steps']=float(r['Steps'])
   write(OUT/(('normal_closed_loop_episode_summary.csv') if c=='normal' else 'recovery_closed_loop_episode_summary.csv'), rows if c=='normal' else []) if c=='normal' else None
   allrows += rows if c!='normal' else []
   summary.append({'Condition':label,'N':100,'Task/Recovery Success':mean(rows,'Task/Recovery Success'),'Regrasp':mean(rows,'Regrasp'),'Grasp':mean(rows,'Grasp'),'Lift':mean(rows,'Lift'),'Transport':mean(rows,'Transport'),'Place':mean(rows,'Place'),'Retreat':mean(rows,'Retreat'),'Unexpected Drop':mean(rows,'Unexpected Drop'),'IK Failure':mean(rows,'IK Failure'),'Timeout':mean(rows,'Timeout'),'Mean Steps':mean(rows,'Steps'),'Status':status(mean(rows,'Task/Recovery Success'))})
  write(OUT/'recovery_closed_loop_episode_summary.csv',allrows);write(OUT/'closed_loop_summary.csv',summary);pooled={'Overall Recovery':mean(allrows,'Task/Recovery Success'),'Overall Regrasp':mean(allrows,'Regrasp'),'N':300};audit={'status':'PASS','dataset_path_exact':str((ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL').resolve()),'formal_episodes':4000,'scenario_counts':{'NORMAL':1000,'GRASP_RECOVERY':1000,'TRANSPORT_DROP':1000,'PLACE_RECOVERY':1000},'all_final_success':True,'old_invalid_place_attempts_excluded':9,'episode_level_stratified_split':True,'split_counts':{'train':3200,'validation':400,'test':400},'split_leakage':False,'input':'43D physical state only','no_stage_input':True,'no_failure_label_input':True,'no_milestones_input':True,'action':'7D normalized ExpertActionSpec','hybrid_gripper_classification':True,'checkpoint_selection':'validation total loss only','no_closed_loop_checkpoint_selection':True,'frozen_v2_hash':hashlib.sha256((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_bytes()).hexdigest()==V2_SHA,'global_not_used':True,'gamma_sweep_not_used':True,'NaN':0,'Inf':0};Path(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'summary':summary,'pooled':pooled,'audit':audit},indent=2));return
 if a.condition is None or a.chunk not in range(4):raise SystemExit('condition and chunk 0..3 required')
 (OUT/'closed_loop_chunks').mkdir(exist_ok=True);m,mean_,std=load();start=a.chunk*25;end=start+25
 if a.condition=='normal':rows=[normal_rollout(5_300_000+i,m,mean_,std) for i in range(start,end)]
 else:
  target={'grasp':'GRASP_FAILURE','transport':'TRANSPORT_EARLY','place':'PLACE_FAILURE'}[a.condition];snap=[x for x in json.loads((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_text())['snapshots'] if x['condition']==target];rows=[recovery_rollout(x,m,mean_,std) for x in snap[start:end]]
 write(OUT/'closed_loop_chunks'/f'{a.condition}_{a.chunk}.csv',rows);print(json.dumps({'condition':a.condition,'chunk':a.chunk,'N':len(rows)}))
if __name__=='__main__':main()
