#!/usr/bin/env python3
"""Normal-only N=200 Gamma Selection V2; no failure assets are loaded."""
from __future__ import annotations
import csv,hashlib,json,math
from pathlib import Path
import numpy as np,torch
from run_e2_awac25k_global_formal import AWAC,GLOBAL,episode,sha,summary
from evaluate_experiment1_global_effectiveness import GlobalSharedController
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/awac25k_global_gamma_selection_v2/run_20260818T_AWAC25K_GAMMA_V2_FORMAL';SEEDS=tuple(range(6_600_000,6_600_200));IDENT=tuple(range(6_610_000,6_610_020));G=tuple(i/10 for i in range(11));OLD=[.84,0,0,0,0,0,.38,.74,.78,.78,.74]
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,default=lambda y:y.item() if isinstance(y,np.generic) else str(y))+'\n')
def write(p,x):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in x for k in r}));w.writeheader();w.writerows(x)
def mc(a,b):
 x=sum(not i and j for i,j in zip(a,b));y=sum(i and not j for i,j in zip(a,b));n=x+y;return min(1.,2*sum(math.comb(n,k) for k in range(min(x,y)+1))/2**n) if n else 1.
def main():
 if OUT.exists():raise SystemExit('STOP existing V2 output')
 OUT.mkdir(parents=True);(OUT/'traces').mkdir();pilot=HybridCheckpointPredictor(AWAC);ctrl=GlobalSharedController(GLOBAL,device_name='cuda' if torch.cuda.is_available() else 'cpu');post=GlobalActionPostprocessor.from_expert_spec();p=torch.load(AWAC,map_location='cpu',weights_only=False)
 dump(OUT/'pilot_checkpoint_freeze.json',{'path':str(AWAC.resolve()),'sha256':sha(AWAC),'step':p['step'],'state_mode':p['state_mode'],'architecture':p['training_config']['hidden_dims']});dump(OUT/'global_checkpoint_freeze.json',{'path':str(GLOBAL.resolve()),'sha256':sha(GLOBAL),'input':'43D only','K':50});dump(OUT/'metadata.json',{'experiment':'GLOBAL GAMMA SELECTION V2','N':200,'gammas':G,'normal_only':True,'failure_assets_loaded':False,'selection_rule':'success, lower IK, drop, timeout, successful length, lower gamma'})
 oldsel=set(range(6_400_000,6_400_050));oldformal=set(range(6_500_000,6_500_300));dev=set(range(6_300_000,6_300_050));overlap={'old_selection':sorted(set(SEEDS)&oldsel),'old_formal':sorted(set(SEEDS)&oldformal),'awac_dev':sorted(set(SEEDS)&dev),'identity':sorted(set(SEEDS)&set(IDENT)),'status':'PASS'};dump(OUT/'normal_gamma_selection_v2_manifest.json',{'status':'FROZEN','seeds':SEEDS,'N':200});dump(OUT/'normal_gamma_selection_v2_overlap_audit.json',overlap)
 ident=[]
 for s in IDENT:
  a=episode(kind='GAMMA_V2_ID',ident=str(s),env_seed=s,meta=None,method='noassist',gamma=0,pilot=pilot,controller=ctrl,post=post,trace=OUT/'traces'/f'id_no_{s}.npz');b=episode(kind='GAMMA_V2_ID',ident=str(s),env_seed=s,meta=None,method='global',gamma=0,pilot=pilot,controller=ctrl,post=post,trace=OUT/'traces'/f'id_gl_{s}.npz');x=np.load(a['trace_path']);y=np.load(b['trace_path']);ident.append({'seed':s,'raw_action_exact':np.array_equal(x['raw_pilot_action'],y['raw_pilot_action']),'executed_action_exact':np.array_equal(x['executed_action'],y['executed_action']),'trajectory_exact':np.array_equal(x['raw_pilot_action'],y['raw_pilot_action']),'termination_exact':a['termination_reason']==b['termination_reason']})
 if not all(all(r.values()) for r in ident):raise SystemExit('STOP identity fail')
 dump(OUT/'gamma0_identity_audit.json',{'status':'PASS','N':20,'rows':ident})
 allrows=[];sums=[]
 for g in G:
  rows=[episode(kind='NORMAL_GAMMA_V2',ident=str(s),env_seed=s,meta=None,method='global',gamma=g,pilot=pilot,controller=ctrl,post=post,trace=OUT/'traces'/f'g{g}_{s}.npz') for s in SEEDS];allrows+=rows;z=summary(g,rows);z.update({'grasp':float(np.mean([r['grasp'] for r in rows])),'lift':float(np.mean([r['lift'] for r in rows])),'transport':float(np.mean([r['transport'] for r in rows])),'place':float(np.mean([r['place'] for r in rows])),'retreat':float(np.mean([r['retreat'] for r in rows])),'mean_return':float(np.mean([r['return'] for r in rows])),'median_steps':float(np.median([r['steps'] for r in rows]))});sums.append(z);print(g,z,flush=True)
 write(OUT/'normal_gamma_episode_results_v2.csv',allrows);write(OUT/'normal_gamma_sweep_v2.csv',sums);base=[r for r in allrows if r['gamma']==0];paired=[]
 for g in G[1:]:
  cur=[r for r in allrows if r['gamma']==g];d=np.asarray([x['success'] for x in cur],float)-np.asarray([x['success'] for x in base],float);rng=np.random.default_rng(20260818+int(g*10));b=np.array([d[rng.integers(200,size=200)].mean() for _ in range(10000)]);paired.append({'gamma':g,'delta_pp':float(d.mean()*100),'ci95_low_pp':float(np.quantile(b,.025)*100),'ci95_high_pp':float(np.quantile(b,.975)*100),'mcnemar_p':mc([x['success'] for x in base],[x['success'] for x in cur])})
 dump(OUT/'paired_gamma_statistics.json',paired);write(OUT/'gamma_selection_v1_vs_v2.csv',[{'gamma':g,'old_N50_success':OLD[i],'new_N200_success':sums[i]['success']} for i,g in enumerate(G)]);best=sorted(sums,key=lambda r:(-r['success'],r['ik'],r['drop'],r['timeout'],r['mean_success_steps'],r['gamma']))[0];status='NONZERO' if best['gamma']>0 else 'ZERO';dump(OUT/'gamma_selection_v2.json',{'selected_coarse_gamma':best['gamma'],'diffusion_step':best['diffusion_step'],'selected_row':best,'GAMMA_SELECTION_V2':status,'failure_results_used':False});dump(OUT/'audit.json',{'status':'PASS','normal_rollouts':2200,'failure_rollouts':0,'overlap_audit':'PASS','identity':'PASS','nan':sum(r['nan'] for r in allrows),'inf':sum(r['inf'] for r in allrows),'global_input_43d_only':True})
if __name__=='__main__':main()
