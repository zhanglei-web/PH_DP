#!/usr/bin/env python3
"""E2-0: unbiased NoAssist audit of the frozen RuleBasedRecoveryPilot."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig, _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.recovery_pilot import ActivePhase, RuleBasedRecoveryPilot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/experiments/e2_recovery_pilot_baseline"
PILOT_SOURCE = PROJECT_ROOT / "src/mujoco_shared_control/experts/recovery_pilot.py"
INJECTOR_SOURCE = PROJECT_ROOT / "scripts/collect_stage_dataset_v1.py"
CONTROL_DT = 0.05
MAX_STEPS = 700
MAX_IK_FAILURES = 5
CONDITIONS = ("NORMAL", "GRASP_FAILURE", "TRANSPORT_DROP", "PLACE_FAILURE")
EVENTS = {"NONE": 0, "GRASP_FAILURE": 1, "DROP": 2, "PLACE_FAILURE": 3, "SUCCESS": 4}
REGRESSIONS = {"GRASP_FAILURE": (1, 0), "TRANSPORT_DROP": (2, 0), "PLACE_FAILURE": (3, 0)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state43(env: PickPlaceEnv, obs: dict[str, Any]) -> np.ndarray:
    value=np.r_[env.get_policy_observation(obs),np.float32(bool(obs["object_grasped"]))].astype(np.float32)
    if value.shape != (43,) or not np.isfinite(value).all(): raise ValueError("state43 contract violation")
    return value


def canonical(action: np.ndarray, pilot: RuleBasedRecoveryPilot) -> np.ndarray:
    value=pilot.action_spec.normalize(action).astype(np.float32)
    value[6]=np.float32(-.25 if value[6] < .375 else 1.0)
    return value


def latency(values: list[int]) -> dict[str, Any]:
    if not values: return {"valid_N":0,"mean":None,"median":None,"p95":None}
    v=np.asarray(values,float); return {"valid_N":len(v),"mean":float(v.mean()),"median":float(np.median(v)),"p95":float(np.quantile(v,.95))}


def compress(phases: list[int]) -> str:
    if not phases:return ""
    chunks=[];current=phases[0];count=0
    for value in phases:
        if value==current:count+=1
        else:chunks.append(f"{current}x{count}");current=value;count=1
    chunks.append(f"{current}x{count}");return " -> ".join(chunks)


def make_seed_manifest() -> dict[str, list[dict[str, Any]]]:
    output={"smoke":[],"formal":[]}
    for kind, count, base in (("smoke",5,4_100_000),("formal",100,4_200_000)):
        for condition_index,condition in enumerate(CONDITIONS):
            for index in range(count):
                seed=base+condition_index*10_000+index
                bucket=(index % 3) if condition=="TRANSPORT_DROP" else None
                output[kind].append({"condition":condition,"episode_index":index,"environment_seed":seed,"pilot_seed":seed+17,"failure_rng_seed":seed+101,"transport_drop_bucket":bucket,"transport_progress_threshold":(0.25,0.50,0.75)[bucket] if bucket is not None else None})
    return output


def run_episode(spec: dict[str, Any], trajectory_path: Path) -> dict[str, Any]:
    """Same injection semantics as collect_stage_dataset_v1, without retries."""
    condition=str(spec["condition"]); env=PickPlaceEnv(render_mode=None,control_timestep=CONTROL_DT,max_episode_steps=MAX_STEPS,enable_camera=False)
    pilot=RuleBasedRecoveryPilot();adapter=ExpertCommandAdapter(env.ik_controller,pilot.action_spec)
    obs,reset=env.reset(seed=int(spec["environment_seed"]),options={"randomize_arm":True,"arm_joint_noise_scale":1.0,"randomize_object":True,"randomize_goal":True})
    adapter.reset(obs["ee_pose"],obs["q_obs"]);pilot.reset(float(obs["object_pose"][2,3]),int(spec["pilot_seed"]))
    reward=AWACRewardV1Online(state43(env,obs)); previous_command=previous_action=None; previous_phase=None; previous_grasped=bool(obs["object_grasped"])
    injected=False;injection_success=False;injection_active=False;injection_steps=0;forced_grasp_complete=False;failure_step=None;regression_step=None;regrasp_step=None;final_success_step=None
    transport_start=None;place_direction=np.array([1.,0.]);consecutive_ik=0;phases=[];events=[];rows=[];post_phase_seen=set();primary="timeout"
    try:
        for step in range(MAX_STEPS):
            state=state43(env,obs)
            expert_obs=_expert_observation(f"e2_{condition.lower()}_{spec['environment_seed']}",0,step,obs,state[:42],previous_command,previous_action)
            command,phase=pilot.predict(expert_obs)
            # Exact E2 re-use of the verified Stage Dataset trigger conditions.
            if condition=="PLACE_FAILURE" and injection_active: phase=ActivePhase.PLACE_RELEASE
            raw_physical=command.delta_pose_gripper.copy(); executed_physical=raw_physical.copy(); event=EVENTS["NONE"]
            if condition=="GRASP_FAILURE" and not injected and phase==ActivePhase.GRASP_LIFT:
                injection_active=True;injection_steps=3;injected=True;failure_step=step;event=EVENTS["GRASP_FAILURE"]
            if condition=="TRANSPORT_DROP" and not injected:
                current=float(np.linalg.norm(obs["object_pose"][:2,3]-obs["goal_pose"][:2,3]))
                if phase==ActivePhase.TRANSPORT:
                    if transport_start is None:transport_start=max(current,1e-8)
                    progress=1-current/transport_start
                    if progress >= float(spec["transport_progress_threshold"]): injection_active=True;injected=True;failure_step=step
                elif previous_phase==ActivePhase.TRANSPORT and phase==ActivePhase.PLACE_RELEASE:
                    phase=ActivePhase.TRANSPORT;injection_active=True;injected=True;failure_step=step
            place_target=obs["goal_pose"][:3,3]+np.array([0.,0.,pilot.config.place_height_m])
            if condition=="PLACE_FAILURE" and not injected and phase==ActivePhase.PLACE_RELEASE and np.linalg.norm(obs["ee_pose"][:3,3]-place_target)<=.012:
                delta=obs["object_pose"][:2,3]-obs["goal_pose"][:2,3]; place_direction=delta/np.linalg.norm(delta) if np.linalg.norm(delta)>1e-6 else np.array([1.,0.])
                injection_active=True;injection_steps=6;injected=True;failure_step=step
            if injection_active and condition=="GRASP_FAILURE":
                direction=np.array([1.,-1.]);direction/=np.linalg.norm(direction);executed_physical=np.r_[direction*.014,0.,np.zeros(3),pilot.config.open_gripper_m];injection_steps-=1
                if injection_steps<=0: injection_active=False;forced_grasp_complete=True
            elif injection_active and condition=="TRANSPORT_DROP": executed_physical=np.r_[np.zeros(6),pilot.config.open_gripper_m]
            elif injection_active and condition=="PLACE_FAILURE":
                if injection_steps>0:
                    gripper=pilot.config.open_gripper_m if injection_steps<=1 else pilot.config.close_gripper_m;executed_physical=np.r_[place_direction*.018,0.,np.zeros(3),gripper];injection_steps-=1
                else: executed_physical=np.r_[np.zeros(6),pilot.config.open_gripper_m]
            raw=canonical(raw_physical,pilot); adapted=adapter.adapt(executed_physical); executed=canonical(np.asarray(adapted.normalized,np.float32),pilot)
            next_obs,_,_,_,_=env.step(adapted.joint_target)
            if forced_grasp_complete:
                # A forced failed close was completed while physical grasp remains false.
                injection_success=not bool(next_obs["object_grasped"]);pilot.confirm_grasp_failure();forced_grasp_complete=False
            next_grasped=bool(next_obs["object_grasped"])
            planned_loss=False
            if condition=="TRANSPORT_DROP" and injection_active and previous_grasped and not next_grasped:
                event=EVENTS["DROP"];injection_active=False;injection_success=True;planned_loss=True;pilot.confirm_external_failure()
            if condition=="PLACE_FAILURE" and injection_active and previous_grasped and not next_grasped:
                event=EVENTS["PLACE_FAILURE"];injection_active=False;injection_success=bool(np.linalg.norm(next_obs["object_pose"][:3,3]-next_obs["goal_pose"][:3,3])>=pilot.config.goal_tolerance_m);planned_loss=True;pilot.confirm_external_failure()
            next_state=state43(env,next_obs);consecutive_ik=0 if adapted.accepted else consecutive_ik+1
            reward_step=reward.step(state,next_state,ik_failure=consecutive_ik>=MAX_IK_FAILURES,time_limit=step+1>=MAX_STEPS)
            # The frozen reward's illegal-drop terminal is intentionally suspended only
            # for the designated injection transition; all unplanned drops still terminate.
            if planned_loss and reward_step.termination_reason=="illegal_drop": reward.terminal=False
            phases.append(int(phase));events.append(event)
            if failure_step is not None and step>=failure_step:
                post_phase_seen.add(int(phase))
                required=REGRESSIONS.get(condition)
                if required and regression_step is None and len(phases)>=2 and phases[-2:]==list(required): regression_step=step
                if injection_success and regrasp_step is None and not previous_grasped and next_grasped: regrasp_step=step
            ee=obs["ee_pose"][:3,3];obj=obs["object_pose"][:3,3];goal=obs["goal_pose"][:3,3]
            rows.append({"state43":state,"raw_pilot_action":raw,"executed_action":executed,"active_stage":int(phase),"object_grasped":bool(obs["object_grasped"]),"ee_position":ee,"object_position":obj,"goal_position":goal,"gripper_opening":float(obs["gripper"][0]),"adapter_accepted":bool(adapted.accepted),"action_clipped":bool(adapted.action_clipped),"fallback_used":bool(adapted.fallback_used),"failure_injected":bool(injected),"failure_event":event,"recovery_relative_step":-1 if failure_step is None else step-failure_step,"reward":reward_step.reward})
            previous_command=raw_physical.copy();previous_action=executed.copy();previous_grasped=next_grasped;previous_phase=phase;obs=next_obs
            if reward_step.task_success:
                final_success_step=step;primary="task_success";break
            if reward_step.terminated or reward_step.truncated:
                primary=reward_step.termination_reason
                if planned_loss and primary=="illegal_drop": continue
                break
        if condition!="NORMAL" and not injection_success: primary="failure_injection_failed"
        array={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
        trajectory_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trajectory_path,**array)
        phase_array=np.asarray(phases,int);required=REGRESSIONS.get(condition); regression=required is None or regression_step is not None
        regrasp=bool(regrasp_step is not None); success=bool(final_success_step is not None)
        recovery=bool(condition!="NORMAL" and injection_success and regression and regrasp and success)
        post={"regrasp":regrasp,"transport":2 in post_phase_seen,"place":3 in post_phase_seen,"retreat":4 in post_phase_seen}
        return {**spec,"episode_id":f"e2_{condition.lower()}_{spec['environment_seed']}","trajectory_path":str(trajectory_path.resolve()),"episode_steps":len(rows),"termination_reason":primary,"final_success":success,"failure_injection_success":injection_success if condition!="NORMAL" else None,"stage_regression_success":regression if condition!="NORMAL" else None,"regrasp_success":regrasp if condition!="NORMAL" else None,"recovery_success":recovery if condition!="NORMAL" else None,"failure_step":failure_step,"regression_step":regression_step,"regrasp_step":regrasp_step,"final_success_step":final_success_step,"post_failure_transport":post['transport'],"post_failure_place":post['place'],"post_failure_retreat":post['retreat'],"grasp":bool(np.any(phase_array==1)),"lift":bool(np.any(phase_array==1) and np.any(array['object_grasped'])),"transport":bool(np.any(phase_array==2)),"place":bool(np.any(phase_array==3)),"retreat":bool(np.any(phase_array==4)),"illegal_drop":primary=="illegal_drop","ik_failure":primary=="ik_failure_limit","timeout":primary=="timeout","stage_sequence":compress(phases),"initial_object_xyz":reset['object_xy'].tolist(),"initial_goal_xyz":reset['goal_xy'].tolist(),"nan_count":int(sum(np.isnan(v).sum() for v in array.values() if v.dtype.kind in 'fc')),"inf_count":int(sum(np.isinf(v).sum() for v in array.values() if v.dtype.kind in 'fc'))}
    finally: env.close()


def aggregate(condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n=len(rows); result={"condition":condition,"N":n,"mean_steps":float(np.mean([r['episode_steps'] for r in rows]))}
    if condition=="NORMAL":
        for key in ('final_success','grasp','lift','transport','place','retreat','illegal_drop','ik_failure','timeout'): result[key]=float(np.mean([bool(r[key]) for r in rows]))
        return result
    inj=[r for r in rows if r['failure_injection_success']]; rec=[r for r in rows if r['recovery_success']]
    result.update({"injection_success":len(inj)/n,"regression_success_end_to_end":float(np.mean([bool(r['stage_regression_success']) for r in rows])),"regression_success_conditional":float(np.mean([bool(r['stage_regression_success']) for r in inj])) if inj else None,"regrasp_success_end_to_end":float(np.mean([bool(r['regrasp_success']) for r in rows])),"regrasp_success_conditional":float(np.mean([bool(r['regrasp_success']) for r in inj])) if inj else None,"recovery_success_end_to_end":len(rec)/n,"recovery_success_conditional":len(rec)/len(inj) if inj else None,"illegal_drop":float(np.mean([r['illegal_drop'] for r in rows])),"ik_failure":float(np.mean([r['ik_failure'] for r in rows])),"timeout":float(np.mean([r['timeout'] for r in rows]))})
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--run-id');args=parser.parse_args();torch.set_num_threads(1)
    stamp=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');root=OUTPUT_ROOT/f'run_{stamp}';root.mkdir(parents=True)
    manifest=make_seed_manifest(); manifest_text=json.dumps(manifest,indent=2)+'\n'
    (root/'seed_manifest.json').write_text(manifest_text)
    (root/'e2_recovery_baseline_seed_manifest.json').write_text(manifest_text)
    meta={"pilot_source":str(PILOT_SOURCE.resolve()),"pilot_source_sha256":sha(PILOT_SOURCE),"failure_injector_source":str(INJECTOR_SOURCE.resolve()),"failure_injector_source_sha256":sha(INJECTOR_SOURCE),"conditions":CONDITIONS,"control_dt":CONTROL_DT,"max_steps":MAX_STEPS,"no_global":True,"no_awac":True,"no_tcn_control":True,"no_retry_until_success":True,"artificial_corruption":False}
    (root/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    smoke=[]
    for spec in manifest['smoke']:
        smoke.append(run_episode(spec,root/'trajectories'/'smoke'/spec['condition']/f"{spec['environment_seed']}.npz"))
    smoke_checks=[]
    for condition in CONDITIONS:
        group=[r for r in smoke if r['condition']==condition]; required=REGRESSIONS.get(condition)
        smoke_checks.append({"condition":condition,"N":len(group),"finite":all(r['nan_count']==0 and r['inf_count']==0 for r in group),"injected":None if required is None else sum(bool(r['failure_injection_success']) for r in group),"regressed":None if required is None else sum(bool(r['stage_regression_success']) for r in group),"normal_completed":None if required else sum(bool(r['final_success']) for r in group)})
    smoke_pass=all(x['finite'] and (x['normal_completed'] is None or x['normal_completed']>0) and (x['injected'] is None or x['injected']>0) and (x['regressed'] is None or x['regressed']>0) for x in smoke_checks)
    (root/'smoke_report.json').write_text(json.dumps({"status":"PASS" if smoke_pass else "FAIL","checks":smoke_checks,"episodes":smoke},indent=2)+'\n')
    if not smoke_pass: raise SystemExit('E2 smoke failed; no formal baseline was run')
    formal=[]
    for index,spec in enumerate(manifest['formal'],1):
        formal.append(run_episode(spec,root/'trajectories'/'formal'/spec['condition']/f"{spec['environment_seed']}.npz"))
        if index%20==0:print(f'E2 formal {index}/400 complete',flush=True)
    write_csv(root/'episode_summary.csv',formal)
    summaries={condition:aggregate(condition,[r for r in formal if r['condition']==condition]) for condition in CONDITIONS}
    for condition,summary in summaries.items():(root/f'{condition.lower()}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    latency_rows=[]
    for condition in CONDITIONS[1:]:
        group=[r for r in formal if r['condition']==condition and r['recovery_success']]
        for label,field in (("Failure->Regression",'regression_step'),("Failure->Regrasp",'regrasp_step'),("Failure->FinalSuccess",'final_success_step')):
            vals=[int(r[field])-int(r['failure_step']) for r in group if r[field] is not None and r['failure_step'] is not None]
            latency_rows.append({"failure":condition,"metric":label,"N":100,"recovered_N":len(group),**latency(vals)})
    write_csv(root/'recovery_latency_summary.csv',latency_rows)
    buckets=[]
    for bucket,name in enumerate(('EARLY','MID','LATE')):
        group=[r for r in formal if r['condition']=='TRANSPORT_DROP' and r['transport_drop_bucket']==bucket]
        buckets.append({"bucket":name,"N":len(group),"injection":float(np.mean([r['failure_injection_success'] for r in group])),"regression":float(np.mean([r['stage_regression_success'] for r in group])),"regrasp":float(np.mean([r['regrasp_success'] for r in group])),"recovery":float(np.mean([r['recovery_success'] for r in group])),"timeout":float(np.mean([r['timeout'] for r in group])),"ik":float(np.mean([r['ik_failure'] for r in group]))})
    write_csv(root/'transport_bucket_summary.csv',buckets)
    regression_rows=[]
    for condition in CONDITIONS[1:]:
        group=[r for r in formal if r['condition']==condition]
        regression_rows.append({"condition":condition,"required_regression":"->".join(map(str,REGRESSIONS[condition])),"N":len(group),"injection_success":float(np.mean([r['failure_injection_success'] for r in group])),"stage_regression_success":float(np.mean([r['stage_regression_success'] for r in group])),"regrasp_success":float(np.mean([r['regrasp_success'] for r in group])),"recovery_success":float(np.mean([r['recovery_success'] for r in group]))})
    write_csv(root/'stage_regression_summary.csv',regression_rows)
    replay=[{key:r.get(key) for key in ('condition','environment_seed','pilot_seed','failure_rng_seed','failure_step','transport_drop_bucket','transport_progress_threshold')}|{"pre_failure_stage":None,"post_failure_stage":None,"object_pose_at_failure":None,"ee_pose_at_failure":None,"goal_pose":None,"gripper_opening":None,"object_grasped":None,"qpos_hash":None,"qvel_hash":None,"state43_hash":None} for r in formal if r['condition']!='NORMAL' and r['failure_injection_success']]
    # Fill deterministic replay state hashes from the saved transition exactly at failure.
    for item in replay:
        source=next(r for r in formal if r['condition']==item['condition'] and r['environment_seed']==item['environment_seed']); z=np.load(source['trajectory_path']);i=int(item['failure_step']);i=min(i,len(z['state43'])-1); item.update({"pre_failure_stage":int(z['active_stage'][i]),"post_failure_stage":int(z['active_stage'][min(i+1,len(z['active_stage'])-1)]),"object_pose_at_failure":z['object_position'][i].tolist(),"ee_pose_at_failure":z['ee_position'][i].tolist(),"goal_pose":z['goal_position'][i].tolist(),"gripper_opening":float(z['gripper_opening'][i]),"object_grasped":bool(z['object_grasped'][i]),"state43_hash":hashlib.sha256(z['state43'][i].tobytes()).hexdigest()})
    write_csv(root/'failure_replay_manifest.csv',replay)
    readiness={"NORMAL_success":summaries['NORMAL']['final_success']>=.90,"failure_requirements":{},"nan_inf":all(r['nan_count']==0 and r['inf_count']==0 for r in formal),"contract_violation":False}
    for condition in CONDITIONS[1:]:
        s=summaries[condition]; readiness['failure_requirements'][condition]={"injection":s['injection_success']>=.95,"regression_conditional":bool(s['regression_success_conditional'] is not None and s['regression_success_conditional']>=.95),"regrasp_conditional":bool(s['regrasp_success_conditional'] is not None and s['regrasp_success_conditional']>=.85),"recovery_end_to_end":s['recovery_success_end_to_end']>=.85}
    readiness['status']='E2_BASELINE_READY' if readiness['NORMAL_success'] and readiness['nan_inf'] and all(all(v.values()) for v in readiness['failure_requirements'].values()) else 'E2_BASELINE_NOT_READY'
    (root/'baseline_readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
    audit={"status":"PASS" if all(r['nan_count']==0 and r['inf_count']==0 for r in formal) else 'FAIL',"scheduled_N":400,"actual_denominator":len(formal),"no_retry_until_success":True,"no_global":True,"no_tcn_control":True,"no_awac":True,"artificial_corruption":False,"planned_failure_distinct_from_illegal_drop":True,"nan":sum(r['nan_count'] for r in formal),"inf":sum(r['inf_count'] for r in formal)}
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({"smoke":"PASS","readiness":readiness['status'],"audit":audit['status'],"output":str(root)},indent=2))


if __name__=='__main__':main()
