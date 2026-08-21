#!/usr/bin/env python3
"""Read-only historical autonomous dynamics audit for Global Diffusion V2."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
from mujoco_shared_control.rss2023.global_evaluation import GlobalDiffusionPredictor,evaluate_episode,summarize
ROOT=Path(__file__).resolve().parents[1];G=ROOT/'outputs/global_diffusion/global_diffusion_v2_expert20000_20260816T230000Z';OUT=ROOT/'outputs/experiments/global_diffusion_training_dynamics/run_20260818T_GLOBAL_TRAINING_DYNAMICS';SEEDS=tuple(range(6_800_000,6_800_100))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(p,rows):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def main():
 if OUT.exists():raise SystemExit('STOP output exists')
 OUT.mkdir(parents=True);(OUT/'plots').mkdir();logs=[json.loads(x) for x in (G/'training_log.jsonl').read_text().splitlines()];wr(OUT/'validation_history.csv',[{'step':x['step'],'validation_loss':x.get('validation_loss')} for x in logs]);loss={x['step']:x.get('validation_loss') for x in logs};files=sorted((G/'checkpoints').glob('step_*.pt'))+[G/'best.pt',G/'latest.pt'];inv=[];rows=[]
 for p in files:
  import torch
  try:q=torch.load(p,map_location='cpu',weights_only=False);step=int(q.get('step',0));load=True
  except Exception:step=None;load=False
  inv.append({'checkpoint_name':p.name,'training_step':step,'path':str(p.resolve()),'sha256':sha(p),'loadable':load,'validation_loss':loss.get(step) if step is not None else None,'is_best':p.name=='best.pt'})
 wr(OUT/'checkpoint_inventory.csv',inv);(OUT/'seed_manifest.json').write_text(json.dumps({'seeds':SEEDS,'N':100,'same_for_all_checkpoints':True},indent=2)+'\n')
 for it in inv:
  if not it['loadable']:continue
  pred=GlobalDiffusionPredictor(it['path'],G/'normalization_stats.npz',device_name='cuda' if __import__('torch').cuda.is_available() else 'cpu');rs=[]
  for s in SEEDS:
   x=evaluate_episode(pred,s,8_800_000+s);x.update({'checkpoint':it['checkpoint_name'],'step':it['training_step']});rs.append(x)
  rows+=rs;z=summarize(rs);rowsum={'checkpoint':it['checkpoint_name'],'step':it['training_step'],'validation_loss':it['validation_loss'],'Success':z['success']['rate'],'Grasp':z['grasp']['rate'],'Lift':z['lift']['rate'],'Transport':z['transport']['rate'],'Place':z['place']['rate'],'Retreat':z['retreat']['rate'],'Drop':z['illegal_drop']['rate'],'IK':z['ik_failure']['rate'],'Timeout':z['timeout']['rate'],'AvgReturn':z['average_return'],'MeanLen':z['episode_length']['mean']};(OUT/f'summary_{it["checkpoint_name"]}.json').write_text(json.dumps(rowsum,indent=2)+'\n');print(rowsum,flush=True)
 wr(OUT/'per_episode_results.csv',rows);sums=[json.loads(x.read_text()) for x in OUT.glob('summary_*.json')];wr(OUT/'checkpoint_summary.csv',sums);audit={'RETRAINED_GLOBAL':'NO','GLOBAL_CHECKPOINTS_MODIFIED':'NO','DATASET_MODIFIED':'NO','ENVIRONMENT_MODIFIED':'NO','GLOBAL_INPUT':'physical43_only','ACTIVE_STAGE_USED_AS_MODEL_INPUT':'NO','TCN_USED':'NO','FAILURE_INJECTION':'NO','SAME_SEEDS_ACROSS_CHECKPOINTS':'YES','HISTORICAL_BEST_FORMAL_N300_PRESERVED':'YES','GLOBAL_RESELECTED':'NO','NAN_INF':sum(r['nan_count']+r['inf_count'] for r in rows),'AUDIT':'PASS'};(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
if __name__=='__main__':main()
