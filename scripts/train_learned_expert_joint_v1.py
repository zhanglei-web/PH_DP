#!/usr/bin/env python3
"""Formal 200k deterministic Learned Expert joint training v1."""

from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,random
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
import torch

from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.sac.agent import SACCore,SACCoreConfig
from mujoco_shared_control.sac.aligned_initialization import load_aligned_v2
from mujoco_shared_control.sac.critic_pretraining import build_arrays_from_semantic_run
from mujoco_shared_control.sac.evaluation import evaluate_sac
from mujoco_shared_control.sac.learned_expert import (
    LearnedExpertConfig,actor_losses,actor_step,calibrate_lambda,critic_step,freeze_log_std,
)
from mujoco_shared_control.sac.replay_buffer import ReplayBatch,SACReplayBuffer


ALIGNED=Path("outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt")
MANIFEST=Path("manifests/rule_expert_v1_formal.json")
SEMANTIC=Path("outputs/sac_reward/sac_reward_v2_candidate_20260814T001000Z")
CHECKPOINTS=(0,25000,50000,100000,150000,200000)
PRIMARY=list(range(300000,300100));SECONDARY=list(range(420000,420100))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_batch(data,indices):
    return ReplayBatch(*(torch.from_numpy(getattr(data,name)[indices]) for name in
        ("observation","action","reward","next_observation","terminated","truncated")))
def merge(a,b):
    return ReplayBatch(*(torch.cat((getattr(a,n),getattr(b,n))) for n in
        ("observation","action","reward","next_observation","terminated","truncated")))
def checksum(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));args=parser.parse_args()
    run=Path("outputs/learned_expert")/f"learned_expert_joint_v1_{args.run_id}";ckdir=run/"checkpoints";ckdir.mkdir(parents=True,exist_ok=False)
    config=LearnedExpertConfig();torch.manual_seed(config.seed);np.random.seed(config.seed);random.seed(config.seed);rng=np.random.default_rng(config.seed)
    actor,critic,target,payload=load_aligned_v2(ALIGNED);freeze_log_std(actor);target.requires_grad_(False)
    initial_actor=deepcopy(actor).eval();initial_actor.requires_grad_(False);initial_logstd=checksum(actor.log_std_head)
    mean,std=payload["observation_mean"],payload["observation_std"]
    expert,audit=build_arrays_from_semantic_run(MANIFEST,SEMANTIC,reward_version="sac_reward_v2_candidate")
    nominal=np.flatnonzero(expert["train"].category=="nominal_success")
    replay=SACReplayBuffer(config.replay_capacity,seed=config.seed)
    actor_optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=config.actor_lr)
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=config.critic_lr)
    spec=ExpertActionSpec(**payload["action_spec"]);env_config=CollectionConfig()
    core=SACCore(payload["actor_source"],SACCoreConfig());core.actor=actor;core.critics=critic;core.target_critics=target
    core.observation_mean=mean;core.observation_std=std
    evaluations={};drifts={};windows=[];outcomes=Counter();phase_counts=Counter();event_counts=Counter();fallback_phase=Counter()
    update_metrics=[];lambda_info=None;gradient_updates=0;episode_count=0;training_seed=900000

    val_states=torch.from_numpy(expert["validation"].observation)
    with torch.no_grad():initial_val=initial_actor.deterministic_action((val_states-mean)/std).numpy()
    def drift():
        with torch.no_grad():current=actor.deterministic_action((val_states-mean)/std).numpy()
        d=current-initial_val
        return {"normalized_mae":float(abs(d).mean()),"xyz_mae":float(abs(d[:,:3]).mean()),
                "rotation_mae":float(abs(d[:,3:6]).mean()),"gripper_mae":float(abs(d[:,6]).mean()),
                "max_abs":float(abs(d).max())}
    def evaluate(step):
        critic_training=critic.training;actor_training=actor.training
        result=evaluate_sac(core,PRIMARY,reward_version="sac_reward_v2_candidate")
        evaluations[str(step)]=result;drifts[str(step)]=drift();critic.train(critic_training);actor.train(actor_training)
        print(f"evaluation step={step} success={result['success']}/100 drift={drifts[str(step)]['normalized_mae']:.6g}",flush=True)
    def checkpoint(step):
        torch.save({"format_version":"learned_expert_joint_v1","global_env_steps":step,"gradient_updates":gradient_updates,
            "episode_count":episode_count,"actor_state_dict":actor.state_dict(),"critic_state_dict":critic.state_dict(),
            "target_critic_state_dict":target.state_dict(),"actor_optimizer_state_dict":actor_optimizer.state_dict(),
            "critic_optimizer_state_dict":critic_optimizer.state_dict(),"replay_state_dict":replay.state_dict(),
            "lambda_info":lambda_info,"config":asdict(config),"observation_mean":mean,"observation_std":std,
            "rng_state":rng.bit_generator.state,"torch_rng_state":torch.get_rng_state(),"reward_version":"sac_reward_v2_candidate"},
            ckdir/f"step_{step:06d}.pt")

    evaluate(0);checkpoint(0)
    env=None;adapter=None;obs=info=None;consecutive=0;episode_events=Counter()
    for env_step in range(1,config.total_env_steps+1):
        if env is None:
            env=PickPlaceEnv(enable_camera=False,reward_version="sac_reward_v2_candidate",control_timestep=env_config.control_timestep_s,max_episode_steps=env_config.max_steps)
            adapter=ExpertCommandAdapter(env.ik_controller,spec)
            obs,info=env.reset(seed=training_seed,options={"randomize_arm":env_config.randomize_arm,"arm_joint_noise_scale":env_config.arm_joint_noise_scale,"randomize_object":env_config.randomize_object,"randomize_goal":env_config.randomize_goal})
            training_seed+=1;adapter.reset(obs["ee_pose"],obs["q_obs"]);consecutive=0;episode_events=Counter()
        state=np.asarray(info["policy_obs"],np.float32)
        with torch.no_grad():action=actor.deterministic_action(((torch.from_numpy(state)-mean)/std).unsqueeze(0)).squeeze(0).numpy()
        if np.linalg.norm(action[:3])>1+1e-6 or np.linalg.norm(action[3:6])>1+1e-6 or abs(action[6])>1+1e-6:raise RuntimeError("action constraint violation")
        adapted=adapter.adapt(spec.denormalize(action));consecutive=0 if adapted.accepted else consecutive+1
        safety=consecutive>=env_config.max_consecutive_ik_failures
        next_obs,reward,terminated,truncated,next_info=env.step(adapted.joint_target,true_failure=safety,failure_reason="ik_failure_limit")
        replay.add(state,action,reward,next_info["policy_obs"],terminated,truncated)
        phase=str(next_info["phase_name"]);phase_counts[phase]+=1
        if adapted.fallback_used:fallback_phase[phase]+=1
        comps=next_info.get("reward_components",{})
        for name,value in comps.items():
            if abs(float(value))>0:event_counts[name]+=1;episode_events[name]+=1

        if len(replay)>=config.replay_seed_transitions:
            # Calibrate once against the untouched aligned Critic/Actor before
            # the first joint optimizer step.
            online_actor=replay.sample(256);bi=rng.choice(nominal,size=256,replace=True);expert_bc=tensor_batch(expert["train"],bi)
            if lambda_info is None:
                qloss,bcloss,_=actor_losses(actor,critic,online_actor.observation,expert_bc.observation,expert_bc.action,mean,std)
                lambda_info=calibrate_lambda(qloss,bcloss,[p for p in actor.parameters() if p.requires_grad])
                print("lambda calibration",lambda_info,flush=True)
            # Critic: exact 128 Expert + 128 Online.
            ei=rng.integers(len(expert["train"]),size=128);eb=tensor_batch(expert["train"],ei);ob=replay.sample(128)
            cm=critic_step(critic,target,actor,critic_optimizer,merge(eb,ob),mean,std,config)
            # Actor Q branch: 256 Online states. BC branch: 256 nominal Expert train states.
            am=actor_step(actor,critic,actor_optimizer,online_actor.observation,expert_bc.observation,expert_bc.action,mean,std,lambda_info["lambda_bc"],config.gradient_clip)
            gradient_updates+=1;row={"env_step":env_step,**cm,**am};update_metrics.append(row)
            if not all(np.isfinite(v) for k,v in row.items() if isinstance(v,(float,int))):raise FloatingPointError("non-finite joint metric")

        if terminated or truncated:
            reason=str(next_info.get("termination_reason","time_limit" if truncated else "other_failure"));outcomes[reason]+=1;episode_count+=1
            env.close();env=None;adapter=None
        else:obs,info=next_obs,next_info
        if env_step%1000==0:
            recent=update_metrics[-1000:];row={"env_step":env_step,"gradient_updates":gradient_updates,"episodes":episode_count,"replay_size":len(replay),
                "outcomes":dict(outcomes),"phase_counts":dict(phase_counts),"fallback_phase":dict(fallback_phase)}
            if recent:
                for key in ("critic_loss","q1_mean","q1_std","q2_mean","q2_std","target_mean","target_std","q_disagreement","actor_loss","actor_q_loss","actor_bc_loss","actor_gradient_norm","critic_gradient_norm"):
                    row[key]=float(np.mean([r[key] for r in recent]))
            windows.append(row)
            with (run/"training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            print(f"step={env_step} episodes={episode_count} replay={len(replay)} q={row.get('q1_mean',0):.3f} loss={row.get('critic_loss',0):.3f}",flush=True)
            # Structural guard only: sustained four-digit Q or non-finite values.
            if abs(row.get("q1_mean",0)) >= 1000 and abs(row.get("q2_mean",0)) >= 1000:
                checkpoint(env_step)
                raise FloatingPointError("protective stop: sustained Q explosion")
        if env_step in CHECKPOINTS[1:]:
            evaluate(env_step);checkpoint(env_step)
            (run/"primary_evaluation.partial.json").write_text(json.dumps(evaluations,indent=2)+"\n")
            completed=[step for step in CHECKPOINTS if str(step) in evaluations]
            if len(completed)>=3 and evaluations[str(completed[-1])]["success"]==0 and evaluations[str(completed[-2])]["success"]==0:
                raise RuntimeError("protective stop: catastrophic success collapse at consecutive checkpoints")
    if env is not None:env.close()
    if checksum(actor.log_std_head)!=initial_logstd:raise RuntimeError("log_std changed")

    def rank(step):
        r=evaluations[str(step)];return (r["success"],-r["termination_reason_counts"].get("illegal_drop",0),-drifts[str(step)]["normalized_mae"])
    best=max(CHECKPOINTS,key=rank)
    initial_secondary=evaluate_sac(SACCore(payload["actor_source"],SACCoreConfig()),SECONDARY,reward_version="sac_reward_v2_candidate")
    best_payload=torch.load(ckdir/f"step_{best:06d}.pt",map_location="cpu",weights_only=False);best_core=SACCore(payload["actor_source"],SACCoreConfig());best_core.actor.load_state_dict(best_payload["actor_state_dict"])
    best_secondary=evaluate_sac(best_core,SECONDARY,reward_version="sac_reward_v2_candidate")
    before={r["seed"]:r["success"] for r in initial_secondary["rows"]};after={r["seed"]:r["success"] for r in best_secondary["rows"]};gained=[s for s in before if not before[s] and after[s]];lost=[s for s in before if before[s] and not after[s]];discord=len(gained)+len(lost)
    paired={"gained":len(gained),"lost":len(lost),"gained_seeds":gained,"lost_seeds":lost,"exact_p":float(binomtest(len(gained),discord,.5).pvalue) if discord else 1.0}
    config_json={**asdict(config),"algorithm":"deterministic Twin-Q + Q policy loss + nominal Expert BC anchor","expert_critic_batch":128,"online_critic_batch":128,"actor_online_batch":256,"bc_nominal_batch":256,"lambda_calibration":lambda_info,"alpha":False,"entropy":False,"log_std_frozen":True,"training_seeds_start":900000,"primary_seeds":[300000,300099],"secondary_seeds":[420000,420099],"final_test_seeds_used":False,"best_step":best}
    outputs={"config.json":config_json,"training_manifest.json":{"env_steps":config.total_env_steps,"gradient_updates":gradient_updates,"episodes":episode_count,"replay_final_size":len(replay)},"actor_initial_source.json":{"aligned":str(ALIGNED.resolve()),"sha256":sha(ALIGNED),"actor_source":payload["actor_source"],"actor_sha256":payload["actor_source_sha256"]},"critic_initial_source.json":{"critic_source":payload["critic_source"],"critic_sha256":payload["critic_source_sha256"]},"reward_source.json":{"version":"sac_reward_v2_candidate","semantic_run":str(SEMANTIC.resolve())},"expert_buffer_manifest.json":audit,"online_replay_statistics.json":{"size":len(replay),"outcomes":dict(outcomes),"phase_counts":dict(phase_counts),"event_counts":dict(event_counts),"fallback_phase":dict(fallback_phase)},"training_metrics.json":{"windows":windows},"primary_evaluation.json":{"best_step":best,"checkpoints":evaluations},"secondary_evaluation.json":{"step0":initial_secondary,"best":best_secondary,"best_step":best,"paired":paired},"actor_drift.json":drifts,"critic_diagnostics.json":{"windows":windows},"failure_statistics.json":{"outcomes":dict(outcomes),"fallback_phase":dict(fallback_phase)}}
    for name,value in outputs.items():(run/name).write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps({"run":str(run.resolve()),"best_step":best,"primary":{s:evaluations[str(s)]["success"] for s in CHECKPOINTS},"secondary":{"initial":initial_secondary["success"],"best":best_secondary["success"],"paired":paired},"lambda":lambda_info},indent=2))


if __name__=="__main__":main()
