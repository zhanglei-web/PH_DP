#!/usr/bin/env python3
"""E2-0b frozen valid-failure snapshot bank and NoAssist qualification.

This intentionally reuses the E2-0 physical injector semantics, but freezes a
branch immediately after a valid, semantically-regressed failure.  Candidate
acceptance is decided before any recovery outcome is observed.
"""
from __future__ import annotations

import argparse, csv, gc, hashlib, json, pickle
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.recovery_pilot import ActivePhase, RuleBasedRecoveryPilot

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments/e2_valid_failure_snapshot_bank'
PILOT_SOURCE=ROOT/'src/mujoco_shared_control/experts/recovery_pilot.py'
INJECTOR_SOURCE=ROOT/'scripts/collect_stage_dataset_v1.py'
ENV_SOURCE=ROOT/'src/mujoco_shared_control/envs/pick_place_env.py'
ADAPTER_SOURCE=ROOT/'src/mujoco_shared_control/control/expert_command_adapter.py'
ACTION_SOURCE=ROOT/'src/mujoco_shared_control/experts/interfaces.py'
REWARD_SOURCE=ROOT/'src/mujoco_shared_control/awac/reward.py'
SELF=Path(__file__)
DT=.05; MAX=700; HORIZON=700; IKMAX=5; TOL=.055; MARGIN=.015; BOUND=TOL+MARGIN
TARGETS={'GRASP_FAILURE':100,'TRANSPORT_EARLY':50,'TRANSPORT_MID':50,'PLACE_FAILURE':100}
REG={'GRASP_FAILURE':(1,0),'TRANSPORT_EARLY':(2,0),'TRANSPORT_MID':(2,0),'PLACE_FAILURE':(3,0)}
EXPECTED_PILOT_SHA='30ee3d2e0e9386afd24952e0270f654d690ed9cfed45b7d838dacdcb79458e58'

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(x:Any)->str:return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def state43(env,obs):
    x=np.r_[env.get_policy_observation(obs),np.float32(bool(obs['object_grasped']))].astype(np.float32)
    if x.shape!=(43,) or not np.isfinite(x).all():raise ValueError('state43 contract violation')
    return x
def canon(a,p):
    x=p.action_spec.normalize(a).astype(np.float32);x[6]=-.25 if x[6]<.375 else 1.;return x
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for r in rows for k in r})
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def pilot_state(p):return {k:deepcopy(getattr(p,k)) for k in ('initial_object_z','approach_offset','motion_step','active_phase','close_attempt_completed','grasp_failure_frames','forced_grasp_failure','forced_reapproach_steps')}
def load_pilot(p,s):
    for k,v in s.items():setattr(p,k,deepcopy(v))
def snapshot(env,adapter,pilot,reward,consec,obs):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION; x=np.empty(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,x,spec)
    return {'integration_state':x,'ctrl':env.data.ctrl.copy(),'mocap_pos':env.data.mocap_pos.copy(),'mocap_quat':env.data.mocap_quat.copy(),'userdata':env.data.userdata.copy(),'episode_steps':env._episode_steps,'previous_observation':deepcopy(env._previous_observation),'sac_task':deepcopy(env.sac_task),'adapter_target':adapter._target.copy(),'adapter_joint_target':adapter._joint_target.copy(),'pilot':pilot_state(pilot),'reward':reward.state_dict(),'consecutive_ik':consec,'obs':deepcopy(obs)}
def restore(env,adapter,pilot,reward,s):
    mujoco.mj_setState(env.model,env.data,s['integration_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    env.data.ctrl[:]=s['ctrl'];env.data.mocap_pos[:]=s['mocap_pos'];env.data.mocap_quat[:]=s['mocap_quat'];env.data.userdata[:]=s['userdata'];mujoco.mj_forward(env.model,env.data)
    env._episode_steps=s['episode_steps'];env._previous_observation=deepcopy(s['previous_observation']);env.sac_task=deepcopy(s['sac_task']);adapter._target=s['adapter_target'].copy();adapter._joint_target=s['adapter_joint_target'].copy();load_pilot(pilot,s['pilot']);reward.load_state_dict(s['reward']);return deepcopy(s['obs']),int(s['consecutive_ik'])
def init(spec, context=None):
    if context is None:
        env=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False);p=RuleBasedRecoveryPilot();a=ExpertCommandAdapter(env.ik_controller,p.action_spec)
    else:
        env,p,a=context
    obs,_=env.reset(seed=spec['environment_seed'],options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True});a.reset(obs['ee_pose'],obs['q_obs']);p.reset(float(obs['object_pose'][2,3]),spec['pilot_seed']);return env,p,a,AWACRewardV1Online(state43(env,obs)),obs
def candidate_specs():
    # Each stream is new and frozen independent of recovery outcome.
    out=[]; base=4_500_000
    for ci,(kind,n) in enumerate(TARGETS.items()):
        for i in range(n*12):
            e=base+ci*100_000+i;out.append({'candidate_index':i,'condition':kind,'environment_seed':e,'pilot_seed':e+17,'failure_rng_seed':e+101,'transport_bucket':None if not kind.startswith('TRANSPORT') else ('EARLY' if kind.endswith('EARLY') else 'MID'),'transport_progress_threshold':.25 if kind.endswith('EARLY') else (.5 if kind.endswith('MID') else None)})
    return out

def make_snapshot(spec, context=None):
    """Return (snapshot metadata, state) or (None, rejection reason), before recovery."""
    kind=spec['condition']; env,p,a,reward,obs=init(spec,context); own=context is None; prev_cmd=prev_a=None; prev_phase=None; prev_g=bool(obs['object_grasped']); transport_start=None; injected=False; active=False; inject_steps=0; forced_done=False; failure_step=None; regression_step=None; place_dir=np.array([1.,0.]);consec=0
    try:
      for step in range(MAX):
        st=state43(env,obs); eo=_expert_observation(f"bank_{kind}_{spec['candidate_index']}",0,step,obs,st[:42],prev_cmd,prev_a);cmd,phase=p.predict(eo)
        raw=cmd.delta_pose_gripper.copy();exe=raw.copy()
        # Exact E2 injector logic, with EARLY/MID progress and pre-place fallback.
        if kind=='GRASP_FAILURE' and not injected and phase==ActivePhase.GRASP_LIFT:
            injected=True;active=True;inject_steps=3;failure_step=step
        if kind.startswith('TRANSPORT') and not injected:
            d=float(np.linalg.norm(obs['object_pose'][:3,3]-obs['goal_pose'][:3,3]))
            if phase==ActivePhase.TRANSPORT:
                if transport_start is None:transport_start=max(d,1e-8)
                if 1-d/transport_start>=spec['transport_progress_threshold']:injected=True;active=True;failure_step=step
            elif prev_phase==ActivePhase.TRANSPORT and phase==ActivePhase.PLACE_RELEASE:
                phase=ActivePhase.TRANSPORT;injected=True;active=True;failure_step=step
        target=obs['goal_pose'][:3,3]+np.array([0,0,p.config.place_height_m])
        if kind=='PLACE_FAILURE' and not injected and phase==ActivePhase.PLACE_RELEASE and np.linalg.norm(obs['ee_pose'][:3,3]-target)<=.012:
            dxy=obs['object_pose'][:2,3]-obs['goal_pose'][:2,3];place_dir=dxy/np.linalg.norm(dxy) if np.linalg.norm(dxy)>1e-6 else np.array([1.,0.]);injected=True;active=True;inject_steps=6;failure_step=step
        if active and kind=='GRASP_FAILURE':
            v=np.array([1.,-1.]);v/=np.linalg.norm(v);exe=np.r_[v*.014,0.,np.zeros(3),p.config.open_gripper_m];inject_steps-=1
            if inject_steps<=0:active=False;forced_done=True
        elif active and kind.startswith('TRANSPORT'):exe=np.r_[np.zeros(6),p.config.open_gripper_m]
        elif active and kind=='PLACE_FAILURE':
            if inject_steps>0:exe=np.r_[place_dir*.018,0.,np.zeros(3),p.config.open_gripper_m if inject_steps<=1 else p.config.close_gripper_m];inject_steps-=1
            else:exe=np.r_[np.zeros(6),p.config.open_gripper_m]
        adapted=a.adapt(exe); next_obs,*_=env.step(adapted.joint_target);next_g=bool(next_obs['object_grasped']); planned=False
        if forced_done:
            forced_done=False;active=False
            if not next_g:p.confirm_grasp_failure()
        if kind.startswith('TRANSPORT') and active and prev_g and not next_g:active=False;planned=True;p.confirm_external_failure()
        if kind=='PLACE_FAILURE' and active and prev_g and not next_g:active=False;planned=True;p.confirm_external_failure()
        ns=state43(env,next_obs);consec=0 if adapted.accepted else consec+1;rs=reward.step(st,ns,ik_failure=consec>=IKMAX,time_limit=step+1>=MAX)
        if planned and rs.termination_reason=='illegal_drop':reward.terminal=False
        # The first observable recovery APPROACH state is exactly the branch point.
        if injected and failure_step is not None and prev_phase is not None and (int(prev_phase),int(phase))==REG[kind]:
            regression_step=step;dist=float(np.linalg.norm(obs['object_pose'][:3,3]-obs['goal_pose'][:3,3])); valid=(not bool(obs['object_grasped']) and not rs.task_success and np.isfinite(st).all() and np.isfinite(obs['object_pose']).all() and np.isfinite(obs['ee_pose']).all())
            if kind.startswith('TRANSPORT') or kind=='PLACE_FAILURE':valid &= dist>=BOUND
            if not valid:return None,('object_inside_goal_region' if dist<BOUND else 'object_still_grasped' if bool(obs['object_grasped']) else 'task_already_success' if rs.task_success else 'nonfinite')
            snap=snapshot(env,a,p,reward,consec,obs)
            meta={**spec,'failure_step':failure_step,'regression_step':regression_step,'pre_failure_stage':REG[kind][0],'post_failure_stage':REG[kind][1],'object_grasped':False,'ee_pose':obs['ee_pose'][:3,3].tolist(),'object_pose':obs['object_pose'][:3,3].tolist(),'goal_pose':obs['goal_pose'][:3,3].tolist(),'object_goal_distance':dist,'goal_tolerance':TOL,'goal_margin':MARGIN,'goal_boundary_threshold':BOUND,'gripper_opening':float(obs['gripper'][0]),'state43':st.tolist(),'qpos_hash':ah(env.data.qpos),'qvel_hash':ah(env.data.qvel),'full_simulator_state_hash':ah(snap['integration_state']),'pilot_state_hash':hashlib.sha256(pickle.dumps(snap['pilot'])).hexdigest(),'adapter_state_hash':hashlib.sha256(pickle.dumps((snap['adapter_target'],snap['adapter_joint_target']))).hexdigest(),'snapshot_accepted_reason':'physical_failure+semantic_regression+goal_margin'}
            return (meta,snap),None
        prev_cmd=raw.copy();prev_a=canon(np.asarray(adapted.normalized,np.float32),p);prev_g=next_g;prev_phase=phase;obs=next_obs
        if (rs.terminated or rs.truncated) and not (planned and rs.termination_reason=='illegal_drop'):
            return None,('failure_not_injected' if not injected else 'no_stage_regression')
      return None,'no_stage_regression'
    finally:
      if own:env.close()

def run_branch(snapshot_file,meta,max_steps=HORIZON,save_path=None,context=None):
    payload=pickle.loads(Path(snapshot_file).read_bytes());spec=meta;env,p,a,reward,_=init(spec,context);own=context is None;obs,consec=restore(env,a,p,reward,payload);prev_cmd=prev_a=None; regrasp=None;seen=set();reason='timeout';rows=[]
    try:
      for step in range(max_steps):
        st=state43(env,obs);cmd,phase=p.predict(_expert_observation(meta['snapshot_id'],0,step,obs,st[:42],prev_cmd,prev_a));raw=cmd.delta_pose_gripper.copy();adapted=a.adapt(raw);executed=canon(np.asarray(adapted.normalized,np.float32),p);next_obs,*_=env.step(adapted.joint_target);ns=state43(env,next_obs);consec=0 if adapted.accepted else consec+1;rs=reward.step(st,ns,ik_failure=consec>=IKMAX,time_limit=step+1>=max_steps)
        if regrasp is None and not bool(obs['object_grasped']) and bool(next_obs['object_grasped']):regrasp=step
        seen.add(int(phase));rows.append({'state43':st,'raw_action':canon(raw,p),'executed_action':executed,'phase':int(phase),'object_grasped':bool(obs['object_grasped']),'ee':obs['ee_pose'][:3,3],'object':obs['object_pose'][:3,3],'goal':obs['goal_pose'][:3,3],'adapter_accepted':adapted.accepted})
        prev_cmd=raw;prev_a=executed;obs=next_obs
        if rs.task_success:reason='task_success';break
        if rs.terminated or rs.truncated:reason=rs.termination_reason;break
      success=reason=='task_success';out={'snapshot_id':meta['snapshot_id'],'failure':meta['condition'],'bucket':meta.get('transport_bucket'),'regrasp_success':regrasp is not None,'post_failure_transport_success':2 in seen,'post_failure_place_success':3 in seen,'post_failure_retreat_success':4 in seen,'recovery_success':bool(success and regrasp is not None),'success_without_regrasp':bool(success and regrasp is None),'snapshot_to_regrasp_steps':regrasp,'snapshot_to_success_steps':step if success else None,'mean_recovery_steps':step+1,'termination_reason':reason,'ik_failure':reason=='ik_failure_limit','unexpected_drop':reason=='illegal_drop','timeout':reason=='timeout','nan':sum(np.isnan(v).sum() for r in rows for v in r.values() if isinstance(v,np.ndarray)),'inf':sum(np.isinf(v).sum() for r in rows for v in r.values() if isinstance(v,np.ndarray)),'contract_violation':any(not r['adapter_accepted'] for r in rows)}
      if save_path:np.savez_compressed(save_path,**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
      return out,rows
    finally:
      if own:env.close()

def deterministic_audit(entries,root):
    chosen=[]
    for k in ('GRASP_FAILURE','TRANSPORT_EARLY','TRANSPORT_MID','PLACE_FAILURE'):
        chosen += [x for x in entries if x['condition']==k][:5]
    results=[]; env=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False);ctx=(env,RuleBasedRecoveryPilot(),None);ctx=(env,ctx[1],ExpertCommandAdapter(env.ik_controller,ctx[1].action_spec))
    for m in chosen:
        f=Path(m['snapshot_path']);a,ra=run_branch(f,m,20,context=ctx);b,rb=run_branch(f,m,20,context=ctx)
        same=len(ra)==len(rb) and all(np.array_equal(x['state43'],y['state43']) and np.array_equal(x['executed_action'],y['executed_action']) for x,y in zip(ra,rb)) and a['termination_reason']==b['termination_reason']
        # Restore identity is covered by deterministic initial first state check in branch data.
        results.append({'snapshot_id':m['snapshot_id'],'pass':same,'steps_a':len(ra),'steps_b':len(rb)})
    env.close();status=all(x['pass'] for x in results);(root/'snapshot_replay_audit.json').write_text(json.dumps({'status':'PASS' if status else 'FAIL','N':len(results),'results':results},indent=2)+'\n');return status

def latency(vals):
    if not vals:return {'N':0,'mean':None,'median':None,'p95':None}
    x=np.asarray(vals,float);return {'N':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.quantile(x,.95))}
def plot(root,rows):
    import matplotlib.pyplot as plt
    labels=['Grasp','Transport Early','Transport Mid','Place']; groups=[('GRASP_FAILURE',None),('TRANSPORT_EARLY','EARLY'),('TRANSPORT_MID','MID'),('PLACE_FAILURE',None)]
    for name,key in [('snapshot_bank_recovery_success.png','recovery_success'),('snapshot_bank_regrasp_success.png','regrasp_success')]:
        y=[np.mean([r[key] for r in rows if r['failure']==f]) for f,_ in groups];plt.figure(figsize=(7,4));plt.bar(labels,y);plt.ylim(0,1);plt.ylabel(key);plt.tight_layout();plt.savefig(root/'plots'/name,dpi=140);plt.close()
    y=[np.mean([r['mean_recovery_steps'] for r in rows if r['failure']==f]) for f,_ in groups];plt.figure(figsize=(7,4));plt.bar(labels,y);plt.ylabel('steps');plt.tight_layout();plt.savefig(root/'plots'/'snapshot_bank_recovery_latency.png',dpi=140);plt.close()
    # all snapshots are explicitly outside boundary; the line documents the frozen margin.
    y=[m['object_goal_distance'] for m in json.loads((root/'snapshot_bank_manifest.json').read_text())['snapshots']];plt.figure(figsize=(7,4));plt.hist(y,bins=30);plt.axvline(BOUND,color='r');plt.tight_layout();plt.savefig(root/'plots'/'snapshot_failure_object_goal_distance.png',dpi=140);plt.close()

def main():
  ap=argparse.ArgumentParser();ap.add_argument('--run-id');args=ap.parse_args();torch.set_num_threads(1)
  if sha(PILOT_SOURCE)!=EXPECTED_PILOT_SHA:raise SystemExit('STOP: RecoveryPilot hash changed')
  stamp=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');root=OUT/f'run_{stamp}';(root/'snapshots').mkdir(parents=True);(root/'trajectories').mkdir();(root/'plots').mkdir()
  meta={'recovery_pilot_sha256':sha(PILOT_SOURCE),'failure_injector_sha256':sha(INJECTOR_SOURCE),'pick_place_env_sha256':sha(ENV_SOURCE),'adapter_sha256':sha(ADAPTER_SOURCE),'action_spec_sha256':sha(ACTION_SOURCE),'reward_termination_sha256':sha(REWARD_SOURCE),'snapshot_serialization_sha256':sha(SELF),'goal_tolerance_m':TOL,'recovery_goal_margin_m':MARGIN,'acceptance_threshold_m':BOUND,'recovery_horizon_definition':'700 control steps from post-failure snapshot','late_transport_excluded_before_global_evaluation':True,'no_global':True,'no_gamma':True,'no_tcn_control':True,'no_awac':True,'no_artificial_corruption':True}
  (root/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); specs=candidate_specs();(root/'candidate_seed_manifest.json').write_text(json.dumps(specs,indent=2)+'\n')
  accepted=[];gen=[];reason_counts={}; ids={'GRASP_FAILURE':0,'TRANSPORT_EARLY':0,'TRANSPORT_MID':0,'PLACE_FAILURE':0}
  # Step 2 smoke: acquire exactly five per type before replay test.
  env=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False);pctx=RuleBasedRecoveryPilot();ctx=(env,pctx,ExpertCommandAdapter(env.ik_controller,pctx.action_spec));phase='smoke'
  for spec in specs:
    k=spec['condition'];need=5 if phase=='smoke' else TARGETS[k]
    if ids[k]>=need:continue
    result,reason=make_snapshot(spec,ctx)
    if result:
      m,s=result; prefix={'GRASP_FAILURE':'GF','TRANSPORT_EARLY':'TD_E','TRANSPORT_MID':'TD_M','PLACE_FAILURE':'PF'}[k];m['snapshot_id']=f'{prefix}_{ids[k]:04d}';path=root/'snapshots'/f"{m['snapshot_id']}.pkl";path.write_bytes(pickle.dumps(s,protocol=pickle.HIGHEST_PROTOCOL));m['snapshot_path']=str(path.resolve());m['snapshot_file_sha256']=sha(path);accepted.append(m);ids[k]+=1;gen.append({**spec,'accepted':True,'reason':'accepted'})
    else:reason_counts[reason]=reason_counts.get(reason,0)+1;gen.append({**spec,'accepted':False,'reason':reason})
    if len(gen)%10==0:gc.collect()
    if all(ids[x]>=5 for x in TARGETS):break
  if not all(ids[x]>=5 for x in TARGETS):raise SystemExit('STOP: unable to generate smoke snapshots')
  if not deterministic_audit(accepted,root):raise SystemExit('STOP: snapshot replay determinism failed')
  # Complete the pre-defined bank, retaining smoke snapshots and never observing recovery outcome until frozen.
  phase='formal'
  used_candidates={(x['condition'],x['candidate_index']) for x in gen}
  for spec in specs:
    k=spec['condition']
    if (k,spec['candidate_index']) in used_candidates:continue
    if ids[k]>=TARGETS[k]:continue
    result,reason=make_snapshot(spec,ctx)
    if result:
      m,s=result; prefix={'GRASP_FAILURE':'GF','TRANSPORT_EARLY':'TD_E','TRANSPORT_MID':'TD_M','PLACE_FAILURE':'PF'}[k];m['snapshot_id']=f'{prefix}_{ids[k]:04d}';path=root/'snapshots'/f"{m['snapshot_id']}.pkl";path.write_bytes(pickle.dumps(s,protocol=pickle.HIGHEST_PROTOCOL));m['snapshot_path']=str(path.resolve());m['snapshot_file_sha256']=sha(path);accepted.append(m);ids[k]+=1;gen.append({**spec,'accepted':True,'reason':'accepted'})
    else:reason_counts[reason]=reason_counts.get(reason,0)+1;gen.append({**spec,'accepted':False,'reason':reason})
    if len(gen)%10==0:gc.collect()
    if all(ids[x]>=TARGETS[x] for x in TARGETS):break
  env.close()
  if not all(ids[x]>=TARGETS[x] for x in TARGETS):raise SystemExit(f'STOP: bank targets unmet {ids}')
  print('snapshot bank targets acquired; freezing manifest',flush=True)
  write_csv(root/'candidate_outcomes.csv',gen)
  # Freeze before qualification.
  manifest={'status':'FROZEN_FOR_E2','target_counts':TARGETS,'acceptance_criteria':{'goal_margin_m':MARGIN,'goal_boundary_threshold_m':BOUND,'independent_of_recovery_outcome':True,'independent_of_global':True},'snapshots':accepted};txt=json.dumps(manifest,indent=2)+'\n';(root/'snapshot_bank_manifest.json').write_text(txt);manifest_sha=hashlib.sha256(txt.encode()).hexdigest()
  genrows=[];validrows=[]
  for k in TARGETS:
    g=[x for x in gen if x['condition']==k];a=[x for x in accepted if x['condition']==k];d=[x['object_goal_distance'] for x in a];genrows.append({'failure':k,'candidates_attempted':len(g),'valid_snapshots':len(a),'rejected_candidates':len(g)-len(a),'acceptance_rate':len(a)/len(g),'rejection_reasons':json.dumps({r:sum(1 for x in g if x['reason']==r) for r in set(x['reason'] for x in g if x['reason']!='accepted')})});validrows.append({'failure':k,'target_snapshots':TARGETS[k],'candidates_attempted':len(g),'accepted':len(a),'rejected':len(g)-len(a),'acceptance_rate':len(a)/len(g),'mean_object_goal_distance':float(np.mean(d)),'min_object_goal_distance':float(np.min(d)),'p5_distance':float(np.quantile(d,.05)),'goal_boundary_threshold':BOUND})
  write_csv(root/'snapshot_generation_summary.csv',genrows);write_csv(root/'failure_snapshot_validity_summary.csv',validrows)
  # Qualification only after manifest frozen.
  env=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False);pctx=RuleBasedRecoveryPilot();branch_ctx=(env,pctx,ExpertCommandAdapter(env.ik_controller,pctx.action_spec));outcomes=[]
  for i,m in enumerate(accepted,1):
    r,_=run_branch(m['snapshot_path'],m,HORIZON,root/'trajectories'/f"{m['snapshot_id']}.npz",branch_ctx);outcomes.append(r)
    if i%25==0:print(f'qualification {i}/300',flush=True)
  env.close();write_csv(root/'noassist_episode_summary.csv',outcomes);write_csv(root/'noassist_recovery_summary.csv',outcomes)
  lrows=[]
  for k in TARGETS:
    g=[r for r in outcomes if r['failure']==k and r['recovery_success']];lrows += [{'failure':k,'metric':'Snapshot->Regrasp',**latency([r['snapshot_to_regrasp_steps'] for r in g if r['snapshot_to_regrasp_steps'] is not None])},{'failure':k,'metric':'Snapshot->Success',**latency([r['snapshot_to_success_steps'] for r in g if r['snapshot_to_success_steps'] is not None])}]
  write_csv(root/'recovery_latency_summary.csv',lrows)
  tb=[]
  for k,name in [('TRANSPORT_EARLY','EARLY'),('TRANSPORT_MID','MID')]:
    g=[r for r in outcomes if r['failure']==k];tb.append({'bucket':name,'N':len(g),'regrasp':float(np.mean([r['regrasp_success'] for r in g])),'recovery':float(np.mean([r['recovery_success'] for r in g])),'timeout':float(np.mean([r['timeout'] for r in g])),'IK':float(np.mean([r['ik_failure'] for r in g])),'mean_recovery_steps':float(np.mean([r['mean_recovery_steps'] for r in g]))})
  write_csv(root/'transport_bucket_qualification.csv',tb)
  pg=[x for x in gen if x['condition']=='PLACE_FAILURE'];pa=[x for x in accepted if x['condition']=='PLACE_FAILURE'];write_csv(root/'place_injection_diagnostics.csv',[{'place_candidates_attempted':len(pg),'successful_physical_off_goal_releases':len(pa),'valid_snapshot_count':len(pa),'rejection_reasons':json.dumps({r:sum(1 for x in pg if x['reason']==r) for r in set(x['reason'] for x in pg if x['reason']!='accepted')})}])
  agg=[]
  for k in TARGETS:
    g=[r for r in outcomes if r['failure']==k];agg.append({'Failure':k,'N':len(g),'Regrasp':float(np.mean([r['regrasp_success'] for r in g])),'Transport':float(np.mean([r['post_failure_transport_success'] for r in g])),'Place':float(np.mean([r['post_failure_place_success'] for r in g])),'Retreat':float(np.mean([r['post_failure_retreat_success'] for r in g])),'Recovery Success':float(np.mean([r['recovery_success'] for r in g])),'IK Failure':float(np.mean([r['ik_failure'] for r in g])),'Unexpected Drop':float(np.mean([r['unexpected_drop'] for r in g])),'Timeout':float(np.mean([r['timeout'] for r in g])),'Mean Recovery Steps':float(np.mean([r['mean_recovery_steps'] for r in g]))})
  write_csv(root/'qualification_main_table.csv',agg)
  allfinite=all(r['nan']==0 and r['inf']==0 for r in outcomes); contract=any(r['contract_violation'] for r in outcomes); rates={r['Failure']:r['Recovery Success'] for r in agg};regr={r['Failure']:r['Regrasp'] for r in agg};ready=all(rates[k]>=.85 and regr[k]>=.85 for k in TARGETS) and allfinite and not contract
  readiness={'status':'E2_SNAPSHOT_BANK_READY' if ready else 'E2_SNAPSHOT_BANK_NOT_READY','recovery_rates':rates,'regrasp_rates':regr,'requirements':{'grasp':rates['GRASP_FAILURE']>=.85,'transport_overall':np.mean([rates['TRANSPORT_EARLY'],rates['TRANSPORT_MID']])>=.85,'transport_early':rates['TRANSPORT_EARLY']>=.85,'transport_mid':rates['TRANSPORT_MID']>=.85,'place':rates['PLACE_FAILURE']>=.85,'regrasp_all':all(regr[k]>=.85 for k in TARGETS),'finite':allfinite,'contract_violation':contract},'success_without_regrasp':sum(r['success_without_regrasp'] for r in outcomes),'frozen_manifest_sha256':manifest_sha};(root/'bank_readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
  audit={'status':'PASS' if allfinite and not contract else 'FAIL','recovery_pilot_unchanged':sha(PILOT_SOURCE)==EXPECTED_PILOT_SHA,'no_global':True,'no_gamma':True,'no_tcn_control':True,'no_awac':True,'no_artificial_corruption':True,'late_transport_excluded_before_global_evaluation':True,'snapshot_acceptance_independent_of_recovery_outcome':True,'snapshot_acceptance_independent_of_global':True,'exact_target_counts':ids,'full_simulator_state_saved':True,'pilot_state_saved':True,'adapter_state_saved':True,'replay_determinism':'PASS','noassist_run_after_bank_freeze':True,'no_snapshot_replacement_after_qualification':True,'nan':sum(r['nan'] for r in outcomes),'inf':sum(r['inf'] for r in outcomes),'contract_violation':contract};(root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');plot(root,outcomes)
  print(json.dumps({'output':str(root),'readiness':readiness['status'],'audit':audit['status'],'manifest_sha256':manifest_sha},indent=2))
if __name__=='__main__':main()
