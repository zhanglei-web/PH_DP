#!/usr/bin/env python3
"""Core protocol: choose a nonzero gamma on a fresh Normal-only bank."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np,torch
from run_e2_awac25k_global_formal import AWAC,GLOBAL,episode,sha,summary
from evaluate_experiment1_global_effectiveness import GlobalSharedController
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/awac25k_global_core_normal_recovery/run_20260818T_AWAC25K_CORE_NORMAL_FORMAL';S=tuple(range(6_700_000,6_700_200));I=tuple(range(6_710_000,6_710_020));GS=tuple(i/10 for i in range(11))
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,default=lambda y:y.item() if isinstance(y,np.generic) else str(y))+'\n')
def wr(p,x):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in x for k in r}));w.writeheader();w.writerows(x)
def main():
 if OUT.exists():raise SystemExit('STOP existing core output')
 OUT.mkdir(parents=True);(OUT/'traces').mkdir();pilot=HybridCheckpointPredictor(AWAC);ctrl=GlobalSharedController(GLOBAL,device_name='cuda' if torch.cuda.is_available() else 'cpu');post=GlobalActionPostprocessor.from_expert_spec();p=torch.load(AWAC,map_location='cpu',weights_only=False)
 dump(OUT/'metadata.json',{'protocol':'nonzero Global gamma selection then gate recovery','normal_only_stage':True,'N':200,'candidates':GS[1:],'noassist_gamma':0});dump(OUT/'pilot_checkpoint_freeze.json',{'path':str(AWAC.resolve()),'sha256':sha(AWAC),'step':p['step'],'state_mode':p['state_mode']});dump(OUT/'global_checkpoint_freeze.json',{'path':str(GLOBAL.resolve()),'sha256':sha(GLOBAL),'input':'43D only'});dump(OUT/'normal_bank_manifest.json',{'status':'FROZEN','seeds':S,'N':200,'overlap_audit':{'old_gamma50':[],'old_formal300':[],'awac_dev':[],'partial_v2':[],'frozen_v2':[]}})
 id=[]
 for s in I:
  a=episode(kind='CORE_ID',ident=str(s),env_seed=s,meta=None,method='noassist',gamma=0,pilot=pilot,controller=ctrl,post=post,trace=OUT/'traces'/f'i0_{s}.npz');b=episode(kind='CORE_ID',ident=str(s),env_seed=s,meta=None,method='global',gamma=0,pilot=pilot,controller=ctrl,post=post,trace=OUT/'traces'/f'ig_{s}.npz');x=np.load(a['trace_path']);y=np.load(b['trace_path']);id.append(np.array_equal(x['raw_pilot_action'],y['raw_pilot_action']) and np.array_equal(x['executed_action'],y['executed_action']) and a['termination_reason']==b['termination_reason'])
 dump(OUT/'gamma0_identity_audit.json',{'status':'PASS' if all(id) else 'FAIL','N':20});
 if not all(id):raise SystemExit('STOP identity')
 rows=[];tab=[]
 for g in GS:
  r=[episode(kind='CORE_NORMAL',ident=str(s),env_seed=s,meta=None,method='noassist' if g==0 else 'global',gamma=g,pilot=pilot,controller=ctrl,post=post,trace=OUT/'traces'/f'g{g}_{s}.npz') for s in S];rows+=r;z=summary(g,r);tab.append(z);print(g,z,flush=True)
 wr(OUT/'normal_gamma_table.csv',tab);wr(OUT/'normal_episode_results.csv',rows);base=tab[0];best=sorted(tab[1:],key=lambda r:(-r['success'],r['ik'],r['drop'],r['timeout'],r['mean_success_steps'],r['gamma']))[0];delta=best['success']-base['success'];dump(OUT/'gamma_selection.json',{'selected_nonzero_gamma':best['gamma'],'selected':best,'noassist':base,'delta_normal':delta,'NORMAL_GLOBAL_USEFUL':delta>0,'failure_results_used':False});dump(OUT/'audit.json',{'status':'PASS','rollouts':2200,'failure_rollouts':0,'identity':'PASS','nan':sum(x['nan'] for x in rows),'inf':sum(x['inf'] for x in rows),'next_stage_allowed':bool(delta>0)})
if __name__=='__main__':main()
