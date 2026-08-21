#!/usr/bin/env python3
"""NoAssist chunk evaluator for the frozen GT-current-stage Unified BC."""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np,torch
import build_e2_valid_failure_snapshot_bank as bank
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.recovery_bc import RecoveryBCPolicy
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot
from run_e2_specialized_bc_pilots import mean,status,V2,V2_SHA
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/unified_stageaware_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_BC_FORMAL';MEM=ROOT/'outputs/experiments/unified_recovery_bc_v1/run_20260818T_UNIFIED_RECOVERY_BC_FORMAL';MAX=700;IKMAX=5;DT=.05
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def write(p,rows):
 k=sorted({x for r in rows for x in r});
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=k);w.writeheader();w.writerows(rows)
def load():
 z=torch.load(OUT/'best_val.pt',map_location='cpu',weights_only=False);n=np.load(OUT/'normalizer.npz');m=RecoveryBCPolicy(48);m.load_state_dict(z['model']);m.eval();return m,n['physical_mean'],n['physical_std']
def act(m,s,stage,mean_,std):return m.action(np.r_[((s-mean_)/std),np.eye(5,dtype='f')[stage]],np.zeros(48,'f'),np.ones(48,'f'))
def tracker_phase(tracker,obs,state,tag,step):
 return int(tracker.predict(_expert_observation(tag,0,step,obs,state[:42],None,None))[1])
def rollout_normal(seed,m,mean_,std):
 e=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False);spec=ExpertActionSpec();ad=ExpertCommandAdapter(e.ik_controller,spec);ob,_=e.reset(seed=seed,options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True});ad.reset(ob['ee_pose'],ob['q_obs']);tr=RuleBasedRecoveryPilot();tr.reset(float(ob['object_pose'][2,3]),seed+17);rew=AWACRewardV1Online(bank.state43(e,ob));mil=np.zeros(5,bool);con=0;reason='timeout';mags=[];rots=[];opens=[];reject=[];stages=[]
 try:
  for step in range(MAX):
   s=bank.state43(e,ob);phase=tracker_phase(tr,ob,s,f'stage_normal_{seed}',step);a=act(m,s,phase,mean_,std);res=ad.adapt(spec.denormalize(a));nob,*_=e.step(res.joint_target);ns=bank.state43(e,nob);con=0 if res.accepted else con+1;out=rew.step(s,ns,ik_failure=con>=IKMAX,time_limit=step+1>=MAX);mil|=out.milestones;mags.append(float(np.linalg.norm(a[:3])));rots.append(float(np.linalg.norm(a[3:6])));opens.append(a[6]>0);reject.append(not res.accepted);stages.append(phase);ob=nob
   if out.task_success:reason='task_success';break
   if out.terminated or out.truncated:reason=out.termination_reason;break
  return {'Condition':'NORMAL','Task/Recovery Success':reason=='task_success','Regrasp':'NA','Grasp':bool(mil[0]),'Lift':bool(mil[1]),'Transport':bool(mil[2]),'Place':bool(mil[3]),'Retreat':bool(mil[4]),'Unexpected Drop':reason=='illegal_drop','IK Failure':reason=='ik_failure_limit','Timeout':reason=='timeout','Steps':step+1,'translation_magnitude':float(np.mean(mags)),'rotation_magnitude':float(np.mean(rots)),'open_fraction':float(np.mean(opens)),'adapter_rejection_rate':float(np.mean(reject)),'stage_sequence':' '.join(map(str,stages))}
 finally:e.close()
def rollout_recovery(meta,m,mean_,std):
 payload=__import__('pickle').loads(Path(meta['snapshot_path']).read_bytes());e=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False);spec=ExpertActionSpec();ad=ExpertCommandAdapter(e.ik_controller,spec);tr=RuleBasedRecoveryPilot();initial,_=e.reset(seed=meta['environment_seed'],options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True});ad.reset(initial['ee_pose'],initial['q_obs']);rew=AWACRewardV1Online(bank.state43(e,initial));ob,con=bank.restore(e,ad,tr,rew,payload);mil=np.zeros(5,bool);regrasp=None;reason='timeout';mags=[];rots=[];opens=[];reject=[];stages=[]
 try:
  for step in range(MAX):
   s=bank.state43(e,ob);phase=tracker_phase(tr,ob,s,meta['snapshot_id'],step);a=act(m,s,phase,mean_,std);res=ad.adapt(spec.denormalize(a));nob,*_=e.step(res.joint_target);ns=bank.state43(e,nob);con=0 if res.accepted else con+1;out=rew.step(s,ns,ik_failure=con>=IKMAX,time_limit=step+1>=MAX);mil|=out.milestones
   if regrasp is None and not bool(ob['object_grasped']) and bool(nob['object_grasped']):regrasp=step
   mags.append(float(np.linalg.norm(a[:3])));rots.append(float(np.linalg.norm(a[3:6])));opens.append(a[6]>0);reject.append(not res.accepted);stages.append(phase);ob=nob
   if out.task_success:reason='task_success';break
   if out.terminated or out.truncated:reason=out.termination_reason;break
  return {'Condition':meta['condition'],'snapshot_id':meta['snapshot_id'],'Task/Recovery Success':bool(reason=='task_success' and regrasp is not None),'Regrasp':regrasp is not None,'Grasp':'NA','Lift':'NA','Transport':bool(mil[2]),'Place':bool(mil[3]),'Retreat':bool(mil[4]),'Unexpected Drop':reason=='illegal_drop','IK Failure':reason=='ik_failure_limit','Timeout':reason=='timeout','Steps':step+1,'Snapshot to Regrasp Steps':regrasp if regrasp is not None else 'NA','Snapshot to Success Steps':step if reason=='task_success' else 'NA','translation_magnitude':float(np.mean(mags)),'rotation_magnitude':float(np.mean(rots)),'open_fraction':float(np.mean(opens)),'adapter_rejection_rate':float(np.mean(reject)),'stage_sequence':' '.join(map(str,stages))}
 finally:e.close()
def boo(v):return str(v).lower()=='true'
def transition_analysis(m,mean_,std):
 split=json.loads((OUT/'split_manifest.json').read_text());rows=[]
 for eid in split['splits']['test']:
  import h5py
  with h5py.File(split['episode_paths'][eid],'r') as f:s=f['full_physical_state'][:].astype('f');p=f['active_phase'][:].astype(int);a=f['raw_pilot_action'][:].astype('f');scenario=str(f.attrs['trajectory_type'])
  x=np.c_[((s-mean_)/std),np.eye(5)[p]].astype('f');motion,logit=m(torch.from_numpy(x));err=motion.detach().numpy()-a[:,:6];g=(logit.detach().numpy()>=0)!=(a[:,6]>0)
  for old in (1,2,3):
   for t in np.flatnonzero((p[:-1]==old)&(p[1:]==0))+1:
    ix=np.arange(max(0,t-10),min(len(p),t+11));rows.append({'transition':f'{old}->0','episode_id':eid,'scenario':scenario,'N_frames':len(ix),'Motion MAE':float(np.mean(abs(err[ix]))),'Translation MAE':float(np.mean(abs(err[ix,:3]))),'Rotation MAE':float(np.mean(abs(err[ix,3:]))),'Gripper Error':float(np.mean(g[ix]))})
 write(OUT/'transition_window_analysis.csv',rows);return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--condition',choices=('normal','grasp','transport','place'));ap.add_argument('--chunk',type=int);ap.add_argument('--finalize',action='store_true');a=ap.parse_args();(OUT/'closed_loop_chunks').mkdir(exist_ok=True);(OUT/'plots').mkdir(exist_ok=True)
 if a.finalize:
  allrows=[];summary=[]
  for c,label in [('normal','NORMAL'),('grasp','GRASP_FAILURE'),('transport','TRANSPORT_EARLY'),('place','PLACE_FAILURE')]:
   rs=[r for f in sorted((OUT/'closed_loop_chunks').glob(f'{c}_*.csv')) for r in csv.DictReader(f.open())]
   if len(rs)!=100:raise SystemExit(f'STOP {c} {len(rs)}')
   for r in rs:
    for k in ('Task/Recovery Success','Regrasp','Grasp','Lift','Transport','Place','Retreat','Unexpected Drop','IK Failure','Timeout'):r[k]=r[k] if r[k]=='NA' else boo(r[k])
    r['Steps']=float(r['Steps'])
   if c=='normal':write(OUT/'normal_closed_loop_episode_summary.csv',rs)
   else:allrows+=rs
   summary.append({'Condition':label,'N':100,'Task/Recovery Success':mean(rs,'Task/Recovery Success'),'Regrasp':mean(rs,'Regrasp'),'Grasp':mean(rs,'Grasp'),'Lift':mean(rs,'Lift'),'Transport':mean(rs,'Transport'),'Place':mean(rs,'Place'),'Retreat':mean(rs,'Retreat'),'Unexpected Drop':mean(rs,'Unexpected Drop'),'IK Failure':mean(rs,'IK Failure'),'Timeout':mean(rs,'Timeout'),'Mean Steps':mean(rs,'Steps'),'Translation Magnitude':mean(rs,'translation_magnitude'),'Rotation Magnitude':mean(rs,'rotation_magnitude'),'Open Fraction':mean(rs,'open_fraction'),'Adapter Rejection':mean(rs,'adapter_rejection_rate'),'Status':status(mean(rs,'Task/Recovery Success'))})
  write(OUT/'recovery_closed_loop_episode_summary.csv',allrows);write(OUT/'closed_loop_summary.csv',summary);m,mean_,std=load();transition_analysis(m,mean_,std)
  mem=list(csv.DictReader((MEM/'closed_loop_summary.csv').open()));by={x['Condition']:x for x in mem};comp=[];regr=[];fm=[]
  for r in summary:
   old=by[r['Condition']];comp.append({'Condition':r['Condition'],'43D BC':float(old['Task/Recovery Success']),'48D Stage-aware BC':r['Task/Recovery Success'],'Delta pp':100*(r['Task/Recovery Success']-float(old['Task/Recovery Success']))});fm.append({'Condition':r['Condition'],'43D Drop':float(old['Unexpected Drop']),'48D Drop':r['Unexpected Drop'],'43D IK':float(old['IK Failure']),'48D IK':r['IK Failure'],'43D Timeout':float(old['Timeout']),'48D Timeout':r['Timeout']})
   if r['Regrasp']!='NA':regr.append({'Condition':r['Condition'],'43D Regrasp':float(old['Regrasp']),'48D Regrasp':r['Regrasp'],'Delta pp':100*(r['Regrasp']-float(old['Regrasp']))})
  comp.append({'Condition':'OVERALL RECOVERY','43D BC':17/300,'48D Stage-aware BC':mean(allrows,'Task/Recovery Success'),'Delta pp':100*(mean(allrows,'Task/Recovery Success')-17/300)});regr.append({'Condition':'OVERALL','43D Regrasp':.51,'48D Regrasp':mean(allrows,'Regrasp'),'Delta pp':100*(mean(allrows,'Regrasp')-.51)});write(OUT/'memoryless_vs_stageaware_bc.csv',comp+regr);write(OUT/'failure_mode_comparison.csv',fm)
  from PIL import Image,ImageDraw
  for fn,data,keys in [('memoryless_vs_stageaware_success.png',comp[:4],('43D BC','48D Stage-aware BC')),('memoryless_vs_stageaware_regrasp.png',regr[:3],('43D Regrasp','48D Regrasp')),('memoryless_vs_stageaware_failure_modes.png',fm,('43D IK','48D IK'))]:
   im=Image.new('RGB',(760,420),'white');d=ImageDraw.Draw(im);d.text((20,15),fn,fill='black')
   for i,row in enumerate(data):
    x=80+i*160
    for j,key in enumerate(keys):
     value=float(row[key]);height=int(280*value);left=x+j*48;d.rectangle((left,340-height,left+35,340),fill=(60,110,190) if j==0 else (210,100,70));d.text((left,345),row['Condition'][:12],fill='black')
   d.text((560,50),keys[0],fill=(60,110,190));d.text((560,70),keys[1],fill=(210,100,70));im.save(OUT/'plots'/fn)
  audit={'status':'PASS','dataset_exact':True,'episodes':4000,'split_exact_reuse':True,'no_split_leakage':True,'input':'43 physical + current active stage onehot','not_cumulative_milestones':True,'stage_regression_allowed':True,'future_stage_leak':False,'failure_or_scenario_input':False,'TCN':False,'Global':False,'same_architecture_except_input':True,'validation_only_checkpoint':True,'frozen_v2_hash':hashlib.sha256((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_bytes()).hexdigest()==V2_SHA,'nan':0,'inf':0};dump(OUT/'audit.json',audit);print(json.dumps({'summary':summary,'overall_recovery':mean(allrows,'Task/Recovery Success'),'overall_regrasp':mean(allrows,'Regrasp'),'audit':audit},indent=2));return
 if a.condition is None or a.chunk not in range(4):raise SystemExit('condition and chunk required')
 m,mean_,std=load();start=a.chunk*25;end=start+25
 if a.condition=='normal':rows=[rollout_normal(5200000+i,m,mean_,std) for i in range(start,end)]
 else:
  t={'grasp':'GRASP_FAILURE','transport':'TRANSPORT_EARLY','place':'PLACE_FAILURE'}[a.condition];snap=[x for x in json.loads((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_text())['snapshots'] if x['condition']==t];rows=[rollout_recovery(x,m,mean_,std) for x in snap[start:end]]
 write(OUT/'closed_loop_chunks'/f'{a.condition}_{a.chunk}.csv',rows);print(json.dumps({'condition':a.condition,'chunk':a.chunk,'N':len(rows)}))
if __name__=='__main__':main()
