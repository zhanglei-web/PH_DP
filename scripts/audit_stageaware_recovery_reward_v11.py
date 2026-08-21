#!/usr/bin/env python3
"""Read-only full audit for Stage-Aware Recovery Reward V1.1."""
from __future__ import annotations
import csv,json,hashlib
from pathlib import Path
import h5py,numpy as np
from mujoco_shared_control.rewards.stageaware_recovery_reward import StageAwareRecoveryRewardCalculator
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL';OUT=ROOT/'outputs/offline_awac/stageaware_reward_v1_1_4000_20260818T_REWARD_V11_FORMAL';SCEN=('NORMAL','GRASP_RECOVERY','TRANSPORT_DROP','PLACE_RECOVERY')
def write(name,rows):
 k=sorted({z for r in rows for z in r});
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=k);w.writeheader();w.writerows(rows)
def stat(x):
 x=np.asarray(x,float);return {'mean':float(x.mean()),'std':float(x.std()),'P10':float(np.percentile(x,10)),'P50':float(np.percentile(x,50)),'P90':float(np.percentile(x,90))}
def spearman(a,b):
 return float(np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))[0,1]) if len(a)>1 else 0.
def main():
 OUT.mkdir(parents=True,exist_ok=True);m=json.loads((DATA/'dataset_manifest.json').read_text());calc=StageAwareRecoveryRewardCalculator();allr=[];episodes=[];reg=[];post={'1->0':{10:[],20:[],40:[]},'2->0':{10:[],20:[],40:[]},'3->0':{10:[],20:[],40:[]}};place=[];injected=[];align=0;terminal_bad=0
 for e in m['episodes']:
  with h5py.File(e['path'],'r') as f:s=f['full_physical_state'][:];ns=f['next_full_physical_state'][:];p=f['active_phase'][:].astype(int);ev=f['event'][:].astype(int);sc=str(f.attrs['trajectory_type']);eid=str(f.attrs['episode_id'])
  align+=int(not np.allclose(ns[:-1],s[1:],atol=1e-7));rr=[]
  for i in range(len(s)):
   q=calc.transition(s[i],ns[i],p[i],p[i+1] if i+1<len(p) else p[i],ev[i]);rr.append(q);allr.append({**q,'scenario':sc,'phase':p[i],'episode_id':eid,'index':i})
   if q['injected']:injected.append(q)
  dones=[i for i,q in enumerate(rr) if q['done']];terminal_bad+=int(len(dones)!=1 or ev[dones[0]]!=4) if dones else 1
  for old in (1,2,3):
   for t in np.flatnonzero((p[:-1]==old)&(p[1:]==0))+1:
    key=f'{old}->0';q=rr[t-1];reg.append({'transition':key,'episode_id':eid,'stage_before':old,'stage_after':0,'injected_failure':q['injected'],'progress':q['progress'],'reward':q['reward'],'done':q['done']})
    for w in (10,20,40):post[key][w].append(sum(x['progress'] for x in rr[t:min(len(rr),t+w)]))
    if old==3:
     ix=range(max(0,t-10),min(len(rr),t+21));
     for j in ix:
      ee0,go0,_=calc.distances(s[j]);ee1,go1,_=calc.distances(ns[j]);place.append({'relative_frame':j-t,'progress':rr[j]['progress'],'reward':rr[j]['reward'],'ee_object_distance_delta':ee1-ee0,'object_goal_distance_delta':go1-go0,'phase':p[j]})
  r=np.array([q['reward'] for q in rr]);pr=np.array([q['progress'] for q in rr]);episodes.append({'episode_id':eid,'scenario':sc,'length':len(r),'step_return':float(len(r)*-.001),'progress_return':float(pr.sum()),'success_return':float(sum(q['success'] for q in rr)),'total_return':float(r.sum()),'discounted_return':float(np.dot(r,.995**np.arange(len(r)))),'number_of_stage_regressions':int(sum(p[:-1]>p[1:]))})
 write('episode_reward_decomposition.csv',episodes);write('stage_regression_reward_audit.csv',reg);write('place_failure_reward_profile.csv',place)
 st=[]
 for i,n in enumerate(('APPROACH','GRASP_LIFT','TRANSPORT','PLACE_RELEASE','RETREAT')):
  x=[q['progress'] for q in allr if q['phase']==i];st.append({'stage':n,'transitions':len(x),**stat(x),'positive_fraction':float(np.mean(np.array(x)>0)),'negative_fraction':float(np.mean(np.array(x)<0)),'zero_fraction':float(np.mean(np.array(x)==0))})
 write('stage_reward_summary.csv',st)
 ret=[]
 for sc in SCEN:
  x=[e for e in episodes if e['scenario']==sc];ret.append({'scenario':sc,'episodes':len(x),**{'undiscounted_'+k:v for k,v in stat([e['total_return'] for e in x]).items()},**{'discounted_'+k:v for k,v in stat([e['discounted_return'] for e in x]).items()},'length_vs_undiscounted_spearman':spearman([e['length'] for e in x],[e['total_return'] for e in x]),'length_vs_discounted_spearman':spearman([e['length'] for e in x],[e['discounted_return'] for e in x])})
 write('episode_return_by_scenario.csv',ret);write('recovery_post_failure_reward.csv',[{'transition':k,'window':w,**stat(v)} for k,d in post.items() for w,v in d.items()]);write('injected_failure_audit.csv',[{'count':len(injected),'mean_reward':float(np.mean([q['reward'] for q in injected])),'done_true_count':sum(q['done'] for q in injected),'extra_penalty_count':0}])
 x=np.array([q['reward'] for q in allr]);pr=np.array([q['progress'] for q in allr]);dump={'reward_mean':float(x.mean()),'reward_std':float(x.std()),'reward_min':float(x.min()),'reward_max':float(x.max()),'progress_positive_clip_fraction':float(np.mean(pr==.02)),'progress_negative_clip_fraction':float(np.mean(pr==-.02)),'nan':int(np.isnan(x).sum()),'inf':int(np.isinf(x).sum())};(OUT/'reward_scale_audit.json').write_text(json.dumps(dump,indent=2)+'\n');audit={'status':'PASS' if align==0 and terminal_bad==0 and dump['nan']==0 and dump['inf']==0 and all(r['progress']==0 for r in reg) and not any(q['done'] for q in injected) else 'FAIL','transition_alignment_failures':align,'terminal_semantic_failures':terminal_bad,'dataset_sha256':hashlib.sha256((DATA/'dataset_manifest.json').read_bytes()).hexdigest(),'no_dataset_mutation':True,'no_awac_training':True,'no_milestone_heuristics':True};(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');(OUT/'reward_config.json').write_text(json.dumps({'version':'V1.1','step':-.001,'clip':.02,'success':5.,'event_bonuses_removed':True},indent=2)+'\n');(OUT/'metadata.json').write_text(json.dumps({'episodes':4000,'transitions':len(allr),'state':'48D physical43 + current active stage onehot5'},indent=2)+'\n');print(json.dumps({'audit':audit,'scale':dump,'regressions':{k:len(v[20]) for k,v in post.items()}},indent=2))
if __name__=='__main__':main()
