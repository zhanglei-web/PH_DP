#!/usr/bin/env python3
"""Paired deterministic BC/SAC rollouts for representative divergence diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mujoco_shared_control.actor_bc.evaluate import ActorPredictor
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from evaluate_sac_actor_initialization import DeterministicSACPredictor


BC=Path("outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt")
SAC=Path("outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/actor_initialized.pt")
OUTPUT=Path("outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/trajectory_divergence.json")
SEEDS=[300005,300007,300012,300029,300035]


def phase(milestones:np.ndarray)->str:
    if milestones[2]:return "P4_PLACE_AND_RETREAT"
    if milestones[1]:return "P3_TRANSPORT"
    if milestones[0]:return "P2_GRASP"
    return "P1_PRE_GRASP"


def run(seed:int,bc:ActorPredictor,sac:DeterministicSACPredictor)->dict:
    config=CollectionConfig();envs=[];adapters=[];observations=[]
    for predictor in (bc,sac):
        env=PickPlaceEnv(render_mode=None,control_timestep=config.control_timestep_s,
                         max_episode_steps=config.max_steps,enable_camera=False)
        obs,_=env.reset(seed=seed,options={"randomize_arm":config.randomize_arm,
            "arm_joint_noise_scale":config.arm_joint_noise_scale,
            "randomize_object":config.randomize_object,"randomize_goal":config.randomize_goal})
        adapter=ExpertCommandAdapter(env.ik_controller,predictor.action_spec);adapter.reset(obs["ee_pose"],obs["q_obs"])
        envs.append(env);adapters.append(adapter);observations.append(obs)
    milestones=np.zeros((2,5),bool);initial_z=[float(o["object_pose"][2,3]) for o in observations]
    first=None
    try:
        for step in range(config.max_steps):
            predictions=[];adapted=[];next_obs=[]
            for index,predictor in enumerate((bc,sac)):
                action,command=predictor.predict(envs[index].get_policy_observation(observations[index]))
                result=adapters[index].adapt(command)
                nxt,*_=envs[index].step(result.joint_target)
                deployed_action=(np.clip(action,-1,1) if index==0 else np.asarray(action))
                predictions.append(deployed_action);adapted.append(result);next_obs.append(nxt)
            ee_delta=float(np.linalg.norm(next_obs[0]["ee_pose"][:3,3]-next_obs[1]["ee_pose"][:3,3]))
            action_delta=np.asarray(predictions[1])-np.asarray(predictions[0])
            grip_class=[bool(a[6]>=.375) for a in predictions]
            significant=bool(np.max(np.abs(action_delta))>.01 or ee_delta>.002 or
                grip_class[0]!=grip_class[1] or adapted[0].fallback_used!=adapted[1].fallback_used or
                adapted[0].action_clipped!=adapted[1].action_clipped)
            if first is None and significant:
                first={"seed":seed,"step":step,"phase":phase(milestones[0]),
                    "criterion":"max|action_delta|>0.01 OR EE delta>0.002m OR gripper/IK/clipping mismatch",
                    "bc_action":np.asarray(predictions[0]).tolist(),"sac_action":np.asarray(predictions[1]).tolist(),
                    "action_delta_sac_minus_bc":action_delta.tolist(),"action_delta_max_abs":float(np.max(np.abs(action_delta))),
                    "ee_position_bc":next_obs[0]["ee_pose"][:3,3].tolist(),"ee_position_sac":next_obs[1]["ee_pose"][:3,3].tolist(),
                    "ee_state_delta_m":ee_delta,
                    "object_grasped_bc":bool(next_obs[0]["object_grasped"]),"object_grasped_sac":bool(next_obs[1]["object_grasped"]),
                    "gripper_class_open_bc":grip_class[0],"gripper_class_open_sac":grip_class[1],
                    "ik_fallback_bc":bool(adapted[0].fallback_used),"ik_fallback_sac":bool(adapted[1].fallback_used),
                    "adapter_clipping_bc":bool(adapted[0].action_clipped),"adapter_clipping_sac":bool(adapted[1].action_clipped)}
            for i,nxt in enumerate(next_obs):
                grasped=bool(nxt["object_grasped"]);obj=nxt["object_pose"][:3,3];goal=nxt["goal_pose"][:3,3];ee=nxt["ee_pose"][:3,3]
                milestones[i,0]|=grasped
                milestones[i,1]|=milestones[i,0] and grasped and obj[2]-initial_z[i]>=.10
                milestones[i,2]|=milestones[i,1] and grasped and np.linalg.norm(obj[:2]-goal[:2])<.055
                milestones[i,3]|=milestones[i,2] and not grasped and np.linalg.norm(obj-goal)<.055
                milestones[i,4]|=milestones[i,3] and np.linalg.norm(ee-(goal+[0,0,.16]))<=.008
            observations=next_obs
            if first is not None:break
        return first or {"seed":seed,"step":None,"phase":None,"criterion":"no significant divergence within horizon"}
    finally:
        for env in envs:env.close()


def main()->None:
    bc=ActorPredictor(BC);sac=DeterministicSACPredictor(SAC)
    rows=[run(seed,bc,sac) for seed in SEEDS]
    report={"definition":"paired environments, identical seed/reset/config; first threshold crossing",
            "representative_group":"BC success -> distilled SAC failure","episodes":rows}
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
