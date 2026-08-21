#!/usr/bin/env python3
"""Collect clean Recovery-Stage-DP-V1 data from frozen Stage-aware AWAC."""
from __future__ import annotations
import argparse, json, shutil
from collections import Counter
from pathlib import Path
import h5py, numpy as np

from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.collection.automatic import CollectionConfig, _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.recovery_pilot import ActivePhase, RuleBasedRecoveryPilot
from mujoco_shared_control.experts.rule_pick_place import RuleExpertConfig

TYPES=("NORMAL_SUCCESS","GRASP_RECOVERY_SUCCESS","TRANSPORT_RECOVERY_SUCCESS","PLACE_RECOVERY_SUCCESS")
INJECT={"GRASP_RECOVERY_SUCCESS":"GRASP_FAILURE","TRANSPORT_RECOVERY_SUCCESS":"DROP","PLACE_RECOVERY_SUCCESS":"PLACE_FAILURE"}
EVENT={"NONE":0,"GRASP_FAILURE":1,"DROP":2,"PLACE_FAILURE":3,"SUCCESS":4}
STAGE_NAMES=("APPROACH","GRASP_LIFT","TRANSPORT","PLACE_RELEASE","RETREAT")

def state43(env,obs): return np.r_[env.get_policy_observation(obs),np.float32(bool(obs['object_grasped']))].astype(np.float32)

def compressed(ph): return [int(ph[0])] + [int(x) for i,x in enumerate(ph[1:],1) if x!=ph[i-1]]

def write_h5(path,rows,meta):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.inprogress.h5'); arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    with h5py.File(tmp,'w') as f:
        for k,v in arrays.items(): f.create_dataset(k,data=v,compression='gzip',compression_opts=1)
        for k,v in meta.items(): f.attrs[k]=v
        f.attrs['schema_version']='recovery_stage_dp_v1'; f.attrs['stage_names_json']=json.dumps(STAGE_NAMES); f.attrs['action_source_names_json']=json.dumps(['EXPERT','FAILURE_INJECTION'])
    tmp.replace(path)

def rollout(kind,seed,predictor,out,attempt):
    cfg=CollectionConfig(max_steps=700); env=PickPlaceEnv(render_mode=None,control_timestep=cfg.control_timestep_s,max_episode_steps=cfg.max_steps,enable_camera=False)
    spec=ExpertActionSpec(); adapter=ExpertCommandAdapter(env.ik_controller,spec); stage_oracle=RuleBasedRecoveryPilot(); rule_cfg=RuleExpertConfig(); failure=INJECT.get(kind)
    try:
        obs,info=env.reset(seed=seed,options={'randomize_arm':True,'arm_joint_noise_scale':1.0,'randomize_object':True,'randomize_goal':True}); adapter.reset(obs['ee_pose'],obs['q_obs']); stage_oracle.reset(float(obs['object_pose'][2,3]),seed+17)
        raw_rows=[]; clean_rows=[]; previous_command=previous_action=None; previous_grasped=bool(obs['object_grasped']); injected=False; active=False; injection_steps=0; failure_step=-1; recovery_start=-1; stable=0; success=False; event=EVENT['NONE']; place_direction=np.array([1.,0.])
        for step in range(cfg.max_steps):
            was_active=active; s=state43(env,obs); expert_obs=_expert_observation(f'recovery_stage_{kind.lower()}_{seed}',0,step,obs,s[:42],previous_command,previous_action); _command,phase=stage_oracle.predict(expert_obs); phase=int(phase)
            if failure=='PLACE_FAILURE' and active:
                phase=int(ActivePhase.PLACE_RELEASE)
            raw=predictor.normalized_action(s[:42],bool(s[42]),current_active_stage=phase).astype(np.float32); executed_physical=spec.denormalize(raw); ev=EVENT['NONE']
            if failure=='GRASP_FAILURE' and not injected and phase==int(ActivePhase.GRASP_LIFT): injected=True; active=True; injection_steps=3; failure_step=step; ev=EVENT['GRASP_FAILURE']
            if failure=='DROP' and not injected and phase==int(ActivePhase.TRANSPORT): injected=True; active=True; failure_step=step
            if failure=='PLACE_FAILURE' and not injected and phase==int(ActivePhase.PLACE_RELEASE):
                d=obs['object_pose'][:2,3]-obs['goal_pose'][:2,3]; place_direction=d/np.linalg.norm(d) if np.linalg.norm(d)>1e-6 else place_direction; injected=True; active=True; injection_steps=6; failure_step=step
            if active and failure=='GRASP_FAILURE':
                d=np.array([1.,-1.]); d/=np.linalg.norm(d); executed_physical=np.r_[d*.014,0.,np.zeros(3),rule_cfg.open_gripper_m]; injection_steps-=1
                if injection_steps<=0: active=False; stage_oracle.confirm_grasp_failure()
            elif active and failure=='DROP': executed_physical=np.r_[np.zeros(6),rule_cfg.open_gripper_m]
            elif active and failure=='PLACE_FAILURE':
                gripper=rule_cfg.open_gripper_m if injection_steps<=1 else rule_cfg.close_gripper_m; executed_physical=np.r_[place_direction*.018,0.,np.zeros(3),gripper]; injection_steps-=1
            adapted=adapter.adapt(executed_physical); executed=np.asarray(adapted.normalized if adapted.accepted else np.r_[np.zeros(6),spec.normalize(np.r_[np.zeros(6),adapted.joint_target[7]])[6]],np.float32)
            next_obs,*_=env.step(adapted.joint_target); next_s=state43(env,next_obs); next_grasped=bool(next_obs['object_grasped'])
            if failure in ('DROP','PLACE_FAILURE') and active and previous_grasped and not next_grasped: ev=EVENT[failure]; active=False; stage_oracle.confirm_external_failure()
            released=bool(not next_grasped and np.linalg.norm(next_obs['object_pose'][:3,3]-next_obs['goal_pose'][:3,3])<.055 and float(next_obs['gripper'][0])>=.055); retreat=bool(released and np.linalg.norm(next_obs['ee_pose'][:3,3]-(next_obs['goal_pose'][:3,3]+np.array([0.,0.,.16])))<=.008); stable=stable+1 if retreat else 0
            if stable>=4: success=True; ev=EVENT['SUCCESS']
            injection_action = bool(was_active or (injected and step==failure_step))
            source='FAILURE_INJECTION' if injection_action else 'EXPERT'
            if kind!='NORMAL_SUCCESS' and recovery_start<0 and injected and not was_active and step>failure_step: recovery_start=step
            row={'full_physical_state':s,'active_phase':np.int8(phase),'stage_onehot':np.eye(5,dtype=np.float32)[phase],'executed_action':executed,'action_source':np.uint8(1 if source=='FAILURE_INJECTION' else 0),'injection_active':injection_action,'event':np.int8(ev),'object_grasped':np.int8(bool(obs['object_grasped'])),'timestep_raw':np.int32(step),'next_full_physical_state':next_s}
            raw_rows.append(row)
            if kind=='NORMAL_SUCCESS' or (recovery_start>=0 and source=='EXPERT'):
                clean=dict(row); clean['timestep_dp']=np.int32(len(clean_rows)); clean_rows.append(clean)
            previous_command=executed_physical.copy(); previous_action=executed.copy(); previous_grasped=next_grasped; obs=next_obs
            if success: break
        valid=bool(success and clean_rows and all(int(r['action_source'])==0 and not r['injection_active'] for r in clean_rows))
        meta={'episode_id':f'recovery_stage_{kind.lower()}_{seed}','episode_type':kind,'seed':seed,'failure_type':failure or 'NONE','failure_step_raw':failure_step,'recovery_start_step_raw':recovery_start,'final_success':success,'valid':valid,'episode_length_raw':len(raw_rows),'episode_length_dp':len(clean_rows),'stage_sequence_raw':compressed([int(r['active_phase']) for r in raw_rows]),'action_target_semantics':'Stage-aware AWAC executable action; injection excluded'}
        write_h5(out/'raw_rollouts'/kind/f"{meta['episode_id']}.h5",raw_rows,meta)
        if valid: write_h5(out/'episodes'/kind/f"{meta['episode_id']}.h5",clean_rows,meta)
        return meta
    finally: env.close()

def build_splits(out,manifest,seed):
    rng=np.random.default_rng(seed); split={'train':[],'validation':[],'test':[]}
    for kind in TYPES:
        ids=[m['episode_id'] for m in manifest if m['episode_type']==kind and m['valid']]; rng.shuffle(ids); n=len(ids); a=int(.8*n); b=int(.9*n); split['train']+=ids[:a]; split['validation']+=ids[a:b]; split['test']+=ids[b:]
    paths={m['episode_id']:str((out/'episodes'/m['episode_type']/(m['episode_id']+'.h5')).resolve()) for m in manifest if m['valid']}
    (out/'split_manifest.json').write_text(json.dumps({'dataset_version':'recovery_stage_dp_v1','split_unit':'episode','shuffle_seed':seed,'splits':split,'episode_paths':paths},indent=2)+'\n')

def existing_manifest(out):
    records=[]
    for path in sorted((out/'episodes').rglob('*.h5')):
        with h5py.File(path,'r') as f:
            def json_value(value):
                if isinstance(value,np.ndarray): return value.tolist()
                if isinstance(value,np.generic): return value.item()
                return value
            records.append({k:json_value(v) for k,v in f.attrs.items()})
    return records

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--normal',type=int,default=1000); p.add_argument('--grasp',type=int,default=300); p.add_argument('--transport',type=int,default=400); p.add_argument('--place',type=int,default=300); p.add_argument('--seed-start',type=int,default=2100000); p.add_argument('--checkpoint',type=Path,default=Path('outputs/offline_awac/stageaware_awac_v1_4000/run_20260818T_STAGEAWARE_AWAC_V1_FORMAL/checkpoints/checkpoint_step_20000.pt')); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    predictor=HybridCheckpointPredictor(a.checkpoint); targets=dict(zip(TYPES,(a.normal,a.grasp,a.transport,a.place))); manifest=existing_manifest(a.output); counts=Counter(m['episode_type'] for m in manifest)
    # Resume uses fresh seeds, never overwriting a previously accepted episode.
    existing_seeds=[int(m['seed']) for m in manifest]
    next_seed=max([a.seed_start, *existing_seeds]) + 1
    attempts=0
    while any(counts[k]<targets[k] for k in TYPES):
        pending=[k for k in TYPES if counts[k]<targets[k]]; kind=pending[attempts%len(pending)]; attempts+=1
        m=rollout(kind,next_seed,predictor,a.output,attempts); next_seed+=1; manifest.append(m); counts[kind]+=int(m['valid'])
        if attempts%10==0: print({'attempts':attempts,'successful':dict(counts)},flush=True)
    (a.output/'episode_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); build_splits(a.output,manifest,20260820); print(json.dumps({'status':'PASS','counts':dict(counts),'attempts':attempts,'dataset':str(a.output.resolve())},indent=2))
if __name__=='__main__': main()
