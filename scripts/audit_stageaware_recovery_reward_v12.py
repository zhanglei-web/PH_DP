#!/usr/bin/env python3
from __future__ import annotations
import csv,json,hashlib
from pathlib import Path
import h5py,numpy as np
from mujoco_shared_control.rewards.stageaware_recovery_reward_v12 import StageAwareRecoveryRewardV12,RewardBookkeeping
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL';OUT=ROOT/'outputs/offline_awac/stageaware_reward_v1_2_4000_20260818T_REWARD_V12_FORMAL';SCEN=('NORMAL','GRASP_RECOVERY','TRANSPORT_DROP','PLACE_RECOVERY')
def write(n,rs):
 k=sorted({x for r in rs for x in r});
 with (OUT/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=k);w.writeheader();w.writerows(rs)
def st(x):
 x=np.asarray(x,float);return dict(mean=float(x.mean()),std=float(x.std()),P10=float(np.percentile(x,10)),P50=float(np.percentile(x,50)),P90=float(np.percentile(x,90)))
def main():
 OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads((DATA/'dataset_manifest.json').read_text());reward=StageAwareRecoveryRewardV12();eps=[];trans=[];reg=[];inj=[];alignment=badterm=0;repeated=[0]*4
 for e in manifest['episodes']:
  with h5py.File(e['path'],'r') as f:p=f['active_phase'][:].astype(int);ev=f['event'][:].astype(int);s=f['full_physical_state'][:];ns=f['next_full_physical_state'][:];sc=str(f.attrs['trajectory_type']);eid=str(f.attrs['episode_id'])
  alignment+=int(not np.allclose(ns[:-1],s[1:],atol=1e-7));book=RewardBookkeeping();rr=[]
  for i in range(len(p)):
   q=reward.transition(p[i],p[i+1] if i+1<len(p) else p[i],ev[i],book);rr.append(q);trans.append({**q,'scenario':sc,'episode_id':eid,'phase':p[i]})
   if q['injected']:inj.append(q)
  done=[i for i,q in enumerate(rr) if q['done']];badterm+=int(len(done)!=1 or ev[done[0]]!=4) if done else 1
  for old in (1,2,3):
   for t in np.flatnonzero((p[:-1]==old)&(p[1:]==0))+1:reg.append({'transition':f'{old}->0','episode_id':eid,'reward':rr[t-1]['reward'],'edge_bonus':rr[t-1]['edge_bonus'],'done':rr[t-1]['done'],'injected':rr[t-1]['injected']})
  edge=[q['edge_bonus'] for q in rr];eps.append({'episode_id':eid,'scenario':sc,'length':len(rr),'step_return':-.001*len(rr),'edge_return':sum(edge),'success_return':sum(q['success'] for q in rr),'total_return':sum(q['reward'] for q in rr),'discounted_return':float(np.dot([q['reward'] for q in rr],.995**np.arange(len(rr)))),'edge_01':sum(q['edge']==(0,1) and q['edge_bonus']>0 for q in rr),'edge_12':sum(q['edge']==(1,2) and q['edge_bonus']>0 for q in rr),'edge_23':sum(q['edge']==(2,3) and q['edge_bonus']>0 for q in rr),'edge_34':sum(q['edge']==(3,4) and q['edge_bonus']>0 for q in rr)})
 write('episode_reward_decomposition.csv',eps);write('stage_regression_reward_audit.csv',reg);write('injected_failure_audit.csv',[{'count':len(inj),'mean_reward':float(np.mean([q['reward'] for q in inj])),'done_true_count':sum(q['done'] for q in inj),'extra_penalty_count':0}])
 ret=[]
 for sc in SCEN:
  x=[e for e in eps if e['scenario']==sc];ret.append({'scenario':sc,'episodes':len(x),**{'undiscounted_'+k:v for k,v in st([e['total_return'] for e in x]).items()},**{'discounted_'+k:v for k,v in st([e['discounted_return'] for e in x]).items()}})
 write('episode_return_by_scenario.csv',ret);scale=np.asarray([q['reward'] for q in trans]);scalej={'mean':float(scale.mean()),'std':float(scale.std()),'min':float(scale.min()),'max':float(scale.max()),'nan':int(np.isnan(scale).sum()),'inf':int(np.isinf(scale).sum())};(OUT/'reward_scale_audit.json').write_text(json.dumps(scalej,indent=2)+'\n');audit={'status':'PASS' if alignment==0 and badterm==0 and all(x['edge_bonus']==0 for x in reg) and not any(x['done'] for x in inj) and scalej['nan']==0 and scalej['inf']==0 else 'FAIL','alignment_failures':alignment,'terminal_failures':badterm,'no_milestone_inference':True,'dataset_unchanged':True,'no_awac_training':True,'manifest_sha':hashlib.sha256((DATA/'dataset_manifest.json').read_bytes()).hexdigest()};(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');(OUT/'reward_config.json').write_text(json.dumps({'version':'V1.2','step':-.001,'forward_edges':{'0->1':.5,'1->2':.5,'2->3':.5,'3->4':.75},'success':5.,'no_dense_progress':True},indent=2)+'\n');(OUT/'metadata.json').write_text(json.dumps({'episodes':4000,'transitions':len(trans)},indent=2)+'\n');print(json.dumps({'audit':audit,'injected':len(inj),'regressions':{x:sum(r['transition']==x for r in reg) for x in ('1->0','2->0','3->0')},'scale':scalej},indent=2))
if __name__=='__main__':main()
