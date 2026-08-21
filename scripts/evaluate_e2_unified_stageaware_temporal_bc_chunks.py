#!/usr/bin/env python3
"""Chunked NoAssist formal evaluator for the causal 20x48D temporal BC."""
from __future__ import annotations
import argparse,csv,json,hashlib,pickle
from collections import deque
from pathlib import Path
import numpy as np,torch,h5py
import build_e2_valid_failure_snapshot_bank as bank
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot
from mujoco_shared_control.experts.temporal_recovery_bc import UnifiedStageAwareTemporalBC
from run_e2_specialized_bc_pilots import mean,status,V2,V2_SHA
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/unified_stageaware_temporal_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_TEMPORAL_BC_FORMAL';STAGE=ROOT/'outputs/experiments/unified_stageaware_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_BC_FORMAL';MEM=ROOT/'outputs/experiments/unified_recovery_bc_v1/run_20260818T_UNIFIED_RECOVERY_BC_FORMAL';MAX=700;IKMAX=5;DT=.05
def write(p,rows):
 k=sorted({z for x in rows for z in x});
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=k);w.writeheader();w.writerows(rows)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load():
 z=torch.load(OUT/'best_val.pt',map_location='cpu',weights_only=False);n=np.load(OUT/'normalizer.npz');m=UnifiedStageAwareTemporalBC();m.load_state_dict(z['model']);m.eval();return m,n['physical_mean'],n['physical_std']
def feature(s,p,mean_,std):return np.r_[((s-mean_)/std),np.eye(5,dtype='f')[p]].astype('f')
def phase(tr,ob,s,tag,step):return int(tr.predict(_expert_observation(tag,0,step,ob,s[:42],None,None))[1])
def rollout(seed,meta=None):
 m,mean_,std=load(); e=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False);spec=ExpertActionSpec();ad=ExpertCommandAdapter(e.ik_controller,spec);tr=RuleBasedRecoveryPilot(); initial,_=e.reset(seed=seed,options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True});ad.reset(initial['ee_pose'],initial['q_obs']);rew=AWACRewardV1Online(bank.state43(e,initial));
 if meta:ob,con=bank.restore(e,ad,tr,rew,pickle.loads(Path(meta['snapshot_path']).read_bytes()));condition=meta['condition'];tag=meta['snapshot_id']
 else:ob=initial;con=0;tr.reset(float(ob['object_pose'][2,3]),seed+17);condition='NORMAL';tag=f'stage_normal_{seed}'
 history=deque(maxlen=20); s=bank.state43(e,ob);p=phase(tr,ob,s,tag,0);history.extend([feature(s,p,mean_,std)]*20);mil=np.zeros(5,bool);regrasp=None;reason='timeout';acts=[];stages=[];attempts=0
 try:
  for step in range(MAX):
   s=bank.state43(e,ob);p=phase(tr,ob,s,tag,step);history.append(feature(s,p,mean_,std));a=m.action(np.asarray(history));res=ad.adapt(spec.denormalize(a));nob,*_=e.step(res.joint_target);ns=bank.state43(e,nob);con=0 if res.accepted else con+1;out=rew.step(s,ns,ik_failure=con>=IKMAX,time_limit=step+1>=MAX);mil|=out.milestones
   if not bool(ob['object_grasped']) and bool(nob['object_grasped']):attempts+=1;regrasp=step if regrasp is None else regrasp
   acts.append(a);stages.append(p);ob=nob
   if out.task_success:reason='task_success';break
   if out.terminated or out.truncated:reason=out.termination_reason;break
  aa=np.asarray(acts);d=np.diff(aa[:,:6],axis=0) if len(aa)>1 else np.zeros((1,6));row={'Condition':condition,'seed':seed,'snapshot_id':meta['snapshot_id'] if meta else 'NA','Task/Recovery Success':bool(reason=='task_success' and (meta is None or regrasp is not None)),'Regrasp':'NA' if meta is None else regrasp is not None,'Grasp':bool(mil[0]),'Lift':bool(mil[1]),'Transport':bool(mil[2]),'Place':bool(mil[3]),'Retreat':bool(mil[4]),'Unexpected Drop':reason=='illegal_drop','IK Failure':reason=='ik_failure_limit','Timeout':reason=='timeout','Steps':step+1,'Snapshot to Regrasp Steps':'NA' if regrasp is None else regrasp,'Snapshot to Success Steps':'NA' if reason!='task_success' else step,'history_initialization':'repeat-current padding' if meta else 'repeat-first observation padding','mean_translation_action_delta':float(np.mean(np.linalg.norm(d[:,:3],axis=1))),'mean_rotation_action_delta':float(np.mean(np.linalg.norm(d[:,3:],axis=1))),'stage_sequence':' '.join(map(str,stages)),'stage_regressions':int(sum(stages[i-1]>0 and stages[i]==0 for i in range(1,len(stages)))),'regrasp_attempts':attempts}
  for i,n in enumerate(('APPROACH','GRASP_LIFT','TRANSPORT','PLACE_RELEASE','RETREAT')):row[f'{n}_dwell']=stages.count(i)
  return row
 finally:e.close()
def b(x):return str(x).lower()=='true'
def trans():
 split=json.loads((OUT/'split_manifest.json').read_text());m,mean_,std=load();rows=[]
 for eid in split['splits']['test']:
  with h5py.File(split['episode_paths'][eid],'r') as f:s=f['full_physical_state'][:].astype('f');p=f['active_phase'][:].astype(int);a=f['raw_pilot_action'][:].astype('f');sc=str(f.attrs['trajectory_type'])
  x=np.c_[((s-mean_)/std),np.eye(5)[p]].astype('f');pad=np.repeat(x[:1],19,0);h=np.asarray([np.r_[pad,x][:][i:i+20] for i in range(len(x))]);motion,logit=m(torch.from_numpy(h));err=motion.detach().numpy()-a[:,:6];g=(logit.detach().numpy()>=0)!=(a[:,6]>0)
  for old in (1,2,3):
   for t in np.flatnonzero((p[:-1]==old)&(p[1:]==0))+1:
    ix=np.arange(max(0,t-10),min(len(p),t+11));rows.append({'Model':'48D Stage+Temporal','transition':f'{old}->0','episode_id':eid,'scenario':sc,'N_frames':len(ix),'Motion MAE':float(np.mean(abs(err[ix]))),'Translation MAE':float(np.mean(abs(err[ix,:3]))),'Rotation MAE':float(np.mean(abs(err[ix,3:]))),'Gripper Error':float(np.mean(g[ix]))})
 for row in csv.DictReader((STAGE/'transition_window_analysis.csv').open()):
  row['Model']='48D Stage-aware Memoryless';rows.append(row)
 write(OUT/'transition_window_analysis.csv',rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--condition',choices=('normal','grasp','transport','place'));ap.add_argument('--chunk',type=int);ap.add_argument('--finalize',action='store_true');a=ap.parse_args();(OUT/'closed_loop_chunks').mkdir(exist_ok=True)
 if a.finalize:
  all=[];summ=[]
  for c,label in [('normal','NORMAL'),('grasp','GRASP_FAILURE'),('transport','TRANSPORT_EARLY'),('place','PLACE_FAILURE')]:
   rs=[r for f in sorted((OUT/'closed_loop_chunks').glob(f'{c}_*.csv')) for r in csv.DictReader(f.open())]
   if len(rs)!=100:raise SystemExit(f'STOP {c} {len(rs)}')
   for r in rs:
    for k in ('Task/Recovery Success','Regrasp','Grasp','Lift','Transport','Place','Retreat','Unexpected Drop','IK Failure','Timeout'):r[k]=r[k] if r[k]=='NA' else b(r[k])
    for k in ('Steps','stage_regressions','regrasp_attempts','APPROACH_dwell','GRASP_LIFT_dwell','TRANSPORT_dwell','PLACE_RELEASE_dwell','RETREAT_dwell'):r[k]=float(r[k])
   if c=='normal':write(OUT/'normal_closed_loop_episode_summary.csv',rs)
   else:all+=rs
   summ.append({'Condition':label,'N':100,**{k:mean(rs,k) for k in ('Task/Recovery Success','Regrasp','Grasp','Lift','Transport','Place','Retreat','Unexpected Drop','IK Failure','Timeout','Steps','mean_translation_action_delta','mean_rotation_action_delta','stage_regressions','regrasp_attempts')},'Status':status(mean(rs,'Task/Recovery Success'))})
  write(OUT/'recovery_closed_loop_episode_summary.csv',all);write(OUT/'closed_loop_summary.csv',summ)
  warm=[]
  for r in all:
   latency=r['Snapshot to Regrasp Steps'];warm.append({'snapshot_id':r['snapshot_id'],'Condition':r['Condition'],'regrasp_within_first_20_steps':latency!='NA' and float(latency)<20,'regrasp_after_20_steps':latency!='NA' and float(latency)>=20,'final_success':r['Task/Recovery Success']})
  write(OUT/'recovery_history_warmup_analysis.csv',warm);old43={r['Condition']:r for r in csv.DictReader((MEM/'closed_loop_summary.csv').open())};old48={r['Condition']:r for r in csv.DictReader((STAGE/'closed_loop_summary.csv').open())};comp=[];fm=[]
  for r in summ:
   k=r['Condition'];comp.append({'Condition':k,'43D Memoryless BC':float(old43[k]['Task/Recovery Success']),'48D Stage-aware BC':float(old48[k]['Task/Recovery Success']),'48D Stage+Temporal BC':r['Task/Recovery Success'],'43D Regrasp':old43[k]['Regrasp'],'48D Stage Regrasp':old48[k]['Regrasp'],'Temporal Regrasp':r['Regrasp']});fm.append({'Condition':k,'43D Drop':float(old43[k]['Unexpected Drop']),'48D Stage Drop':float(old48[k]['Unexpected Drop']),'Temporal Drop':r['Unexpected Drop'],'43D IK':float(old43[k]['IK Failure']),'48D Stage IK':float(old48[k]['IK Failure']),'Temporal IK':r['IK Failure'],'43D Timeout':float(old43[k]['Timeout']),'48D Stage Timeout':float(old48[k]['Timeout']),'Temporal Timeout':r['Timeout']})
  comp.append({'Condition':'OVERALL RECOVERY','43D Memoryless BC':17/300,'48D Stage-aware BC':73/300,'48D Stage+Temporal BC':mean(all,'Task/Recovery Success'),'43D Regrasp':.51,'48D Stage Regrasp':.463333333333,'Temporal Regrasp':mean(all,'Regrasp')});write(OUT/'model_comparison_closed_loop.csv',comp);write(OUT/'failure_mode_comparison.csv',fm);trans();dump(OUT/'metadata.json',{'status':'PASS','frozen_v2_sha':hashlib.sha256((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_bytes()).hexdigest(),'frozen_v2_expected':V2_SHA,'global_used':False,'gamma_sweep_used':False,'recovery_history':'repeat-current padding','normal_seeds':'5200000..5200099','checkpoint':'best validation total loss only'});print(json.dumps({'summary':summ,'overall':mean(all,'Task/Recovery Success')}));return
 if a.condition is None or a.chunk not in range(4):raise SystemExit('condition/chunk')
 start=a.chunk*25
 if a.condition=='normal':rs=[rollout(5200000+i) for i in range(start,start+25)]
 else:
  target={'grasp':'GRASP_FAILURE','transport':'TRANSPORT_EARLY','place':'PLACE_FAILURE'}[a.condition];snap=[x for x in json.loads((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_text())['snapshots'] if x['condition']==target];rs=[rollout(x['environment_seed'],x) for x in snap[start:start+25]]
 write(OUT/'closed_loop_chunks'/f'{a.condition}_{a.chunk}.csv',rs)
if __name__=='__main__':main()
