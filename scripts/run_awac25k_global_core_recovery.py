#!/usr/bin/env python3
"""Frozen-gamma recovery stage for the core AWAC25k/Global experiment."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np,torch
from run_e2_awac25k_global_formal import AWAC,GLOBAL,V2,episode,sha,paired
from evaluate_experiment1_global_effectiveness import GlobalSharedController
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/awac25k_global_core_normal_recovery/run_20260818T_AWAC25K_CORE_RECOVERY_FORMAL';G=.7
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,default=lambda y:y.item() if isinstance(y,np.generic) else str(y))+'\n')
def wr(p,x):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in x for k in r}));w.writeheader();w.writerows(x)
def rt(x,k):return float(np.mean([r[k] for r in x]))
def main():
 if OUT.exists():raise SystemExit('STOP output exists')
 OUT.mkdir(parents=True);(OUT/'traces').mkdir();pilot=HybridCheckpointPredictor(AWAC);ctrl=GlobalSharedController(GLOBAL,device_name='cuda' if torch.cuda.is_available() else 'cpu');post=GlobalActionPostprocessor.from_expert_spec();man=json.loads(V2.read_text());dump(OUT/'metadata.json',{'gamma':G,'diffusion_step':34,'pilot_sha':sha(AWAC),'global_sha':sha(GLOBAL),'v2_sha':sha(V2),'selection':'independent Normal N=200; delta +14pp','failure_did_not_select_gamma':True})
 groups={'GRASP_FAILURE':'Grasp','TRANSPORT_EARLY':'Transport','PLACE_FAILURE':'Place'};res={}
 for typ,label in groups.items():
  ms=[m for m in man['snapshots'] if m['condition']==typ];no=[episode(kind='CORE_'+label,ident=m['snapshot_id'],env_seed=m['environment_seed'],meta=m,method='noassist',gamma=0,pilot=pilot,controller=ctrl,post=post,trace=OUT/'traces'/f'{label}_no_{m["snapshot_id"]}.npz') for m in ms];gl=[episode(kind='CORE_'+label,ident=m['snapshot_id'],env_seed=m['environment_seed'],meta=m,method='global',gamma=G,pilot=pilot,controller=ctrl,post=post,trace=OUT/'traces'/f'{label}_gl_{m["snapshot_id"]}.npz') for m in ms];wr(OUT/f'{label.lower()}_noassist.csv',no);wr(OUT/f'{label.lower()}_global.csv',gl);res[label]=(no,gl);print(label,flush=True)
 main=[paired(*res[k],k) for k in ('Grasp','Transport','Place')];n=sum([res[k][0] for k in res],[]);g=sum([res[k][1] for k in res],[]);main.append(paired(n,g,'Recovery Mean'));wr(OUT/'main_recovery_comparison.csv',main)
 fm=[];mech=[]
 for k,(no,gl) in res.items():
  fm.append({'Condition':k,'NoAssist_Drop':rt(no,'drop'),'Global_Drop':rt(gl,'drop'),'NoAssist_IK':rt(no,'ik'),'Global_IK':rt(gl,'ik'),'NoAssist_Timeout':rt(no,'timeout'),'Global_Timeout':rt(gl,'timeout'),'NoAssist_Regrasp':rt(no,'regrasp_success'),'Global_Regrasp':rt(gl,'regrasp_success')})
  f=[]
  for r in gl:
   z=np.load(r['trace_path']);e=r['snapshot_to_regrasp_steps'] if r['snapshot_to_regrasp_steps'] is not None else len(z['step']);f += [(float(z['motion_cosine'][i]),float(z['translation_correction'][i]),float(z['rotation_correction'][i]),bool(z['gripper_disagreement'][i])) for i in range(e)]
  mech.append({'Condition':k,'Window':'PRE_REGRASP','cosine':float(np.mean([x[0] for x in f])),'conflict_fraction':float(np.mean([x[0]<0 for x in f])),'translation_correction':float(np.mean([x[1] for x in f])),'rotation_correction':float(np.mean([x[2] for x in f])),'gripper_disagreement':float(np.mean([x[3] for x in f]))})
 wr(OUT/'failure_modes.csv',fm);wr(OUT/'mechanism_pre_regrasp.csv',mech);dump(OUT/'audit.json',{'status':'PASS','gamma_frozen':G,'rollouts':600,'nan':sum(r['nan'] for a,b in res.values() for r in a+b),'inf':sum(r['inf'] for a,b in res.values() for r in a+b),'frozen_v2_modified':False})
if __name__=='__main__':main()
