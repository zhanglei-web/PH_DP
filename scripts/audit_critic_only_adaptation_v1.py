#!/usr/bin/env python3
"""Held-out fixed-policy and illegal-drop audits for Critic-only adaptation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

from build_aligned_actor_critic_v1 import (
    MANIFEST, PHASES, AuditState, _heldout_audit_states, _step,
)
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.sac.aligned_initialization import load_aligned_v2
from mujoco_shared_control.sac.critic import TwinSACCritic
from mujoco_shared_control.sac.critic_adaptation import ActorTransitionArrays
from mujoco_shared_control.sac.diagnostics import restore_environment


ALIGNED = Path("outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt")
STEPS = (0,5000,10000,20000)


def correlation(x, y, method: str) -> float | None:
    x=np.asarray(x);y=np.asarray(y)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    value=pearsonr(x,y).statistic if method=="pearson" else spearmanr(x,y).statistic
    return float(value) if np.isfinite(value) else None


def actor_branch(state: AuditState, first_action: np.ndarray, actor, mean, std) -> dict[str, Any]:
    config=CollectionConfig();spec=ExpertActionSpec()
    env=PickPlaceEnv(enable_camera=False,reward_version="sac_reward_v2_candidate",
                     control_timestep=config.control_timestep_s,max_episode_steps=config.max_steps)
    adapter=ExpertCommandAdapter(env.ik_controller,spec)
    try:
        initial,_=env.reset(seed=0);adapter.reset(initial["ee_pose"],initial["q_obs"])
        consecutive=restore_environment(env,adapter,state.snapshot)
        action=np.asarray(first_action,np.float64);rewards=[];reason="none"
        for _ in range(config.max_steps-state.step):
            obs,reward,terminated,truncated,info,consecutive=_step(env,adapter,action,consecutive)
            rewards.append(float(reward))
            if terminated or truncated:
                reason=str(info.get("termination_reason","time_limit" if truncated else "other"));break
            with torch.no_grad():
                normalized=(torch.from_numpy(info["policy_obs"])-mean)/std
                action=actor.deterministic_action(normalized.unsqueeze(0)).squeeze(0).numpy()
        return {"full_return":float(sum(.995**i*r for i,r in enumerate(rewards))),
                "h20_return":float(sum(.995**i*r for i,r in enumerate(rewards[:20]))),
                "length":len(rewards),"reason":reason}
    finally:env.close()


def summarize(rows: list[dict[str,Any]], q_key: str) -> dict[str,Any]:
    groups=[("overall",rows)]+[(p,[r for r in rows if r["phase"]==p]) for p in PHASES]
    groups += [("P4a_PLACE",[r for r in rows if r["p4_substage"]=="P4a_PLACE"]),
               ("P4b_RELEASE_STABILIZE",[r for r in rows if r["p4_substage"]=="P4b_RELEASE_STABILIZE"])]
    result={}
    for name,selected in groups:
        dq=np.asarray([r[q_key] for r in selected]);entry={"samples":len(selected)}
        for horizon in ("full","h20"):
            dg=np.asarray([r[f"delta_g_{horizon}"] for r in selected]);mask=np.abs(dg)>1e-10
            entry[horizon]={"spearman":correlation(dq,dg,"spearman"),
                            "pearson":correlation(dq,dg,"pearson"),
                            "sign_agreement_non_tie":float(np.mean(np.sign(dq[mask])==np.sign(dg[mask]))) if mask.any() else None,
                            "non_tie":int(mask.sum())}
        result[name]=entry
    return result


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path);args=parser.parse_args()
    run=args.run.resolve();actor,_base,_target,payload=load_aligned_v2(ALIGNED)
    actor.eval();actor.requires_grad_(False);mean,std=payload["observation_mean"],payload["observation_std"]
    states,reconstruction=_heldout_audit_states(MANIFEST,per_phase=25,reward_version="sac_reward_v2_candidate")
    branches=[]
    for index,state in enumerate(states):
        normalized=(torch.from_numpy(state.policy_state)-mean)/std
        with torch.no_grad():actor_action=actor.deterministic_action(normalized.unsqueeze(0)).squeeze(0).numpy()
        expert=actor_branch(state,state.recorded_action,actor,mean,std)
        current=actor_branch(state,actor_action,actor,mean,std)
        substage="NONE"
        if state.expert_stage==5:substage="P4a_PLACE"
        elif state.expert_stage==6:substage="P4b_RELEASE_STABILIZE"
        branches.append({"episode_id":state.episode_id,"seed":state.seed,"step":state.step,
                         "phase":state.phase,"p4_substage":substage,
                         "expert_action":state.recorded_action.tolist(),"actor_action":actor_action.tolist(),
                         "delta_g_full":expert["full_return"]-current["full_return"],
                         "delta_g_h20":expert["h20_return"]-current["h20_return"],
                         "expert_branch":expert,"actor_branch":current})
        if (index+1)%20==0:print(f"fixed-policy branches={index+1}/100",flush=True)
    by_step={}
    for step in STEPS:
        checkpoint=torch.load(run/"critic_checkpoints"/f"critic_step_{step:05d}.pt",
                              map_location="cpu",weights_only=False)
        critic=TwinSACCritic();critic.load_state_dict(checkpoint["critic_state_dict"]);critic.eval()
        rows=deepcopy(branches)
        for state,row in zip(states,rows,strict=True):
            x=(torch.from_numpy(state.policy_state)-mean)/std
            ae=torch.tensor(row["expert_action"]);aa=torch.tensor(row["actor_action"])
            with torch.no_grad():
                q1e,q2e=critic(x.unsqueeze(0),ae.unsqueeze(0));q1a,q2a=critic(x.unsqueeze(0),aa.unsqueeze(0))
            row["delta_q"]=float(torch.minimum(q1e,q2e)-torch.minimum(q1a,q2a))
        by_step[str(step)]={"summary":summarize(rows,"delta_q"),"rows":rows}
    audit={"protocol":{"reward_version":"sac_reward_v2_candidate","continuation":"frozen deterministic Actor pi0",
                       "full_return_primary":True,"h20_diagnostic":True,"heldout_expert_seeds":[100900,100999]},
           "reconstruction":reconstruction,"checkpoints":by_step}
    (run/"fixed_policy_action_replacement_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    (run/"phase_audit.json").write_text(json.dumps({step:value["summary"] for step,value in by_step.items()},indent=2)+"\n")

    online=ActorTransitionArrays.load(run/"online_actor_dataset"/"test.npz")
    drop_episodes=np.unique(online.episode_id[online.outcome=="illegal_drop"])
    predecessor={}
    for critic_step in STEPS:
        checkpoint=torch.load(run/"critic_checkpoints"/f"critic_step_{critic_step:05d}.pt",
                              map_location="cpu",weights_only=False)
        critic=TwinSACCritic();critic.load_state_dict(checkpoint["critic_state_dict"]);critic.eval()
        predecessor[str(critic_step)]={}
        for offset in (1,5,10):
            indices=[]
            for episode in drop_episodes:
                episode_indices=np.flatnonzero(online.episode_id==episode)
                if len(episode_indices)>=offset:indices.append(episode_indices[-offset])
            indices=np.asarray(indices,int);obs=torch.from_numpy(online.observation[indices]);action=torch.from_numpy(online.action[indices])
            with torch.no_grad():
                x=(obs-mean)/std;q1,q2=critic(x,action);prediction=((q1+q2)/2).numpy().reshape(-1)
            target=online.mc_return[indices].reshape(-1);error=prediction-target
            predecessor[str(critic_step)][f"drop_minus_{offset}"]={"count":len(indices),
                "prediction_mean":float(prediction.mean()),"return_mean":float(target.mean()),
                "mae":float(np.mean(abs(error))),"rmse":float(np.sqrt(np.mean(error**2))),
                "pearson":correlation(prediction,target,"pearson"),"spearman":correlation(prediction,target,"spearman")}
    (run/"illegal_drop_predecessor_audit.json").write_text(json.dumps(predecessor,indent=2)+"\n")
    print(json.dumps({"fixed_policy":{s:v["summary"]["overall"] for s,v in by_step.items()},
                      "drop_predecessor":predecessor},indent=2))


if __name__=="__main__":main()
