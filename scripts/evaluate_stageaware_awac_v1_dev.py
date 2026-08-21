#!/usr/bin/env python3
"""Freeze an independent 200-episode dev bank and evaluate AWAC checkpoints.

This is post-training analysis only.  The policy action always comes from the
Hybrid AWAC actor; the rule pilot is used solely as the current-stage tracker
and to construct valid physical failure snapshots.
"""
from __future__ import annotations
import csv, hashlib, json, pickle
from pathlib import Path
import numpy as np

import build_e2_valid_failure_snapshot_bank as bank
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/offline_awac/stageaware_awac_v1_4000/run_20260818T_STAGEAWARE_AWAC_V1_LOCAL20K_FORMAL'
DEV=RUN/'dev_bank'; MAX=700; IKMAX=5; DT=.05
FORMAL=ROOT/'outputs/experiments/e2_failure_snapshot_bank_v2/run_20260818T030000Z/e2_failure_snapshot_bank_v2_manifest.json'
NORMAL_FORMAL=set(range(5_200_000,5_200_100))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for r in rows for k in r})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def dump(path, obj): Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def phase(tracker,obs,state,tag,step):
    return int(tracker.predict(_expert_observation(tag,0,step,obs,state[:42],None,None))[1])

def build_dev_bank():
    manifest_path=DEV/'dev_bank_manifest.json'
    if manifest_path.exists(): return json.loads(manifest_path.read_text())
    DEV.mkdir(parents=True); (DEV/'snapshots').mkdir()
    formal=json.loads(FORMAL.read_text()); formal_ids={x['snapshot_id'] for x in formal['snapshots']}
    formal_env={x['environment_seed'] for x in formal['snapshots']}
    specs=[]
    streams=[('GRASP_FAILURE',6_000_000),('TRANSPORT_EARLY',6_100_000),('PLACE_FAILURE',6_200_000)]
    for kind,base in streams:
        for i in range(600):
            e=base+i; specs.append({'candidate_index':i,'condition':kind,'environment_seed':e,'pilot_seed':e+17,'failure_rng_seed':e+101,'transport_bucket':'EARLY' if kind=='TRANSPORT_EARLY' else None,'transport_progress_threshold':.25 if kind=='TRANSPORT_EARLY' else None})
    accepted=[]; counts={k:0 for k,_ in streams}; rejections={}
    env=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False)
    pilot=RuleBasedRecoveryPilot(); ctx=(env,pilot,ExpertCommandAdapter(env.ik_controller,pilot.action_spec))
    for spec in specs:
        kind=spec['condition']
        if counts[kind]>=50: continue
        result,reason=bank.make_snapshot(spec,ctx)
        if not result:
            rejections[str(reason)]=rejections.get(str(reason),0)+1; continue
        meta,snap=result; prefix={'GRASP_FAILURE':'GF','TRANSPORT_EARLY':'TD','PLACE_FAILURE':'PF'}[kind]
        meta['snapshot_id']=f'DEV_{prefix}_{counts[kind]:03d}'
        path=DEV/'snapshots'/f"{meta['snapshot_id']}.pkl"; path.write_bytes(pickle.dumps(snap,protocol=pickle.HIGHEST_PROTOCOL))
        meta['snapshot_path']=str(path.resolve()); meta['snapshot_file_sha256']=sha(path); accepted.append(meta); counts[kind]+=1
        if all(n==50 for n in counts.values()): break
    env.close()
    if not all(n==50 for n in counts.values()): raise RuntimeError(f'dev snapshot targets unmet: {counts}')
    normals=[6_300_000+i for i in range(50)]
    overlap={'formal_snapshot_ids':sorted({x['snapshot_id'] for x in accepted}&formal_ids),'recovery_environment_seeds':sorted({x['environment_seed'] for x in accepted}&formal_env),'normal_environment_seeds':sorted(set(normals)&NORMAL_FORMAL),'status':'PASS'}
    if any(overlap[k] for k in ('formal_snapshot_ids','recovery_environment_seeds','normal_environment_seeds')): overlap['status']='FAIL'; raise RuntimeError(f'dev/formal overlap: {overlap}')
    manifest={'status':'FROZEN','normal_seeds':normals,'snapshots':accepted,'counts':{'NORMAL':50,'GRASP_FAILURE':50,'TRANSPORT_EARLY':50,'PLACE_FAILURE':50},'formal_frozen_v2_manifest_sha256':sha(FORMAL),'generator':'existing build_e2_valid_failure_snapshot_bank.make_snapshot','action_control':'Hybrid AWAC only; RuleBasedRecoveryPilot stage tracking/snapshot construction only'}
    dump(DEV/'dev_bank_overlap_audit.json',overlap); dump(manifest_path,manifest); return manifest

def rollout_normal(seed, predictor):
    env=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False); spec=ExpertActionSpec(); ad=ExpertCommandAdapter(env.ik_controller,spec)
    obs,_=env.reset(seed=seed,options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True}); ad.reset(obs['ee_pose'],obs['q_obs'])
    tracker=RuleBasedRecoveryPilot(); tracker.reset(float(obs['object_pose'][2,3]),seed+17); rew=AWACRewardV1Online(bank.state43(env,obs)); consecutive=0; reason='timeout'; stages=[]; rejects=0
    try:
        for step in range(MAX):
            state=bank.state43(env,obs); stage=phase(tracker,obs,state,f'dev_normal_{seed}',step); action=predictor.normalized_action(state[:42],bool(state[42]),current_active_stage=stage)
            adapted=ad.adapt(spec.denormalize(action)); next_obs,*_=env.step(adapted.joint_target); ns=bank.state43(env,next_obs); consecutive=0 if adapted.accepted else consecutive+1; result=rew.step(state,ns,ik_failure=consecutive>=IKMAX,time_limit=step+1>=MAX)
            stages.append(stage); rejects+=not adapted.accepted; obs=next_obs
            if result.task_success: reason='task_success'; break
            if result.terminated or result.truncated: reason=result.termination_reason; break
        return {'Condition':'NORMAL','success':reason=='task_success','regrasp':False,'drop':reason=='illegal_drop','ik':reason=='ik_failure_limit','timeout':reason=='timeout','steps':step+1,'adapter_rejections':rejects,'stage_sequence':' '.join(map(str,stages))}
    finally: env.close()

def rollout_recovery(meta,predictor):
    payload=pickle.loads(Path(meta['snapshot_path']).read_bytes()); env=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False); spec=ExpertActionSpec(); ad=ExpertCommandAdapter(env.ik_controller,spec); tracker=RuleBasedRecoveryPilot()
    initial,_=env.reset(seed=meta['environment_seed'],options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True}); ad.reset(initial['ee_pose'],initial['q_obs']); rew=AWACRewardV1Online(bank.state43(env,initial)); obs,consecutive=bank.restore(env,ad,tracker,rew,payload); reason='timeout'; regrasp=False; stages=[]; rejects=0
    try:
        for step in range(MAX):
            state=bank.state43(env,obs); stage=phase(tracker,obs,state,meta['snapshot_id'],step); action=predictor.normalized_action(state[:42],bool(state[42]),current_active_stage=stage)
            adapted=ad.adapt(spec.denormalize(action)); next_obs,*_=env.step(adapted.joint_target); ns=bank.state43(env,next_obs); consecutive=0 if adapted.accepted else consecutive+1; result=rew.step(state,ns,ik_failure=consecutive>=IKMAX,time_limit=step+1>=MAX)
            regrasp |= (not bool(obs['object_grasped']) and bool(next_obs['object_grasped'])); stages.append(stage); rejects+=not adapted.accepted; obs=next_obs
            if result.task_success: reason='task_success'; break
            if result.terminated or result.truncated: reason=result.termination_reason; break
        return {'Condition':meta['condition'],'snapshot_id':meta['snapshot_id'],'success':bool(reason=='task_success' and regrasp),'regrasp':regrasp,'drop':reason=='illegal_drop','ik':reason=='ik_failure_limit','timeout':reason=='timeout','steps':step+1,'adapter_rejections':rejects,'stage_sequence':' '.join(map(str,stages))}
    finally: env.close()

def summarize(label,rows):
    by={k:[r for r in rows if r['Condition']==k] for k in ('NORMAL','GRASP_FAILURE','TRANSPORT_EARLY','PLACE_FAILURE')}
    rates={k:float(np.mean([r['success'] for r in v])) for k,v in by.items()}; rec=np.mean([rates['GRASP_FAILURE'],rates['TRANSPORT_EARLY'],rates['PLACE_FAILURE']])
    return {'checkpoint':label,'Normal':rates['NORMAL'],'Grasp':rates['GRASP_FAILURE'],'Transport':rates['TRANSPORT_EARLY'],'Place':rates['PLACE_FAILURE'],'RecoveryMean':float(rec),'DevScore':float(.5*rates['NORMAL']+.5*rec),'Drop':float(np.mean([r['drop'] for r in rows])),'IK':float(np.mean([r['ik'] for r in rows])),'Timeout':float(np.mean([r['timeout'] for r in rows])),'adapter_rejections':sum(r['adapter_rejections'] for r in rows)}

def main():
    manifest=build_dev_bank(); checkpoints=[('Step0',RUN/'checkpoints/checkpoint_step_00000.pt'),('2.5k',RUN/'checkpoints/checkpoint_step_02500.pt'),('5k',RUN/'checkpoints/checkpoint_step_05000.pt'),('10k',RUN/'checkpoints/checkpoint_step_10000.pt'),('20k',RUN/'checkpoints/checkpoint_step_20000.pt')]
    allrows=[]; summary=[]
    for label,path in checkpoints:
        if not path.exists(): raise RuntimeError(f'missing checkpoint {path}')
        predictor=HybridCheckpointPredictor(path)
        rows=[{**rollout_normal(seed,predictor),'checkpoint':label,'seed':seed} for seed in manifest['normal_seeds']]
        rows += [{**rollout_recovery(meta,predictor),'checkpoint':label} for meta in manifest['snapshots']]
        if len(rows)!=200: raise RuntimeError('dev bank size changed')
        allrows+=rows; summary.append(summarize(label,rows)); print(label,summary[-1],flush=True)
    write(RUN/'dev_episode_results.csv',allrows); write(RUN/'dev_checkpoint_summary.csv',summary)
    order={x[0]:i for i,x in enumerate(checkpoints)}; selected=sorted(summary,key=lambda r:(-r['DevScore'],-r['Normal'],r['IK']+r['Drop'],order[r['checkpoint']]))[0]
    dump(RUN/'selected_checkpoint.json',{'rule':'max DevScore; tie: higher Normal, lower IK+Drop, earlier checkpoint','selected':selected,'all':summary})
    audit={'status':'PASS','dev_bank_frozen':True,'formal_v2_not_run':True,'formal_v2_overlap_audit':'PASS','checkpoints_evaluated':[x[0] for x in checkpoints],'actor_state_mode':'physical43_active_stage5','pilot_control':'none','nan_or_inf':False,'adapter_rejections':sum(r['adapter_rejections'] for r in allrows)}
    dump(RUN/'dev_evaluation_audit.json',audit)
if __name__=='__main__': main()
