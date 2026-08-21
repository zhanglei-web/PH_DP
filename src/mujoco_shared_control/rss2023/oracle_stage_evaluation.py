"""Closed-loop evaluator for Oracle current Active Stage conditioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.milestones import phase_from_milestones
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion
from mujoco_shared_control.rss2023.global_evaluation import summarize


GRIPPER_OPEN_THRESHOLD = 0.375


class OracleStagePredictor:
    def __init__(self, checkpoint_path: str | Path, normalization_path: str | Path, *, device_name: str = "auto") -> None:
        self.device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
        payload = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        config = DiffusionConfig(**payload["diffusion_config"])
        if (config.observation_dim, config.action_dim, config.num_diffusion_steps) != (48, 7, 50):
            raise ValueError("checkpoint is not the frozen 48D/7D/50-step Oracle Diffusion")
        self.model = RSS2023Diffusion(config).to(self.device).eval(); self.model.load_state_dict(payload["model"])
        with np.load(normalization_path, allow_pickle=False) as stats:
            self.observation_mean=np.asarray(stats["observation_mean"],np.float32); self.observation_std=np.asarray(stats["observation_std"],np.float32); self.action_mean=np.asarray(stats["action_mean"],np.float32); self.action_std=np.asarray(stats["action_std"],np.float32)
        if self.observation_mean.shape!=(48,) or self.action_mean.shape!=(7,) or not np.array_equal(self.observation_mean[43:],np.zeros(5,np.float32)) or not np.array_equal(self.observation_std[43:],np.ones(5,np.float32)):
            raise ValueError("Oracle normalization does not preserve raw stage one-hot")
        self.action_spec=ExpertActionSpec(); self.generator=None

    def reset_sampling(self, seed: int) -> None:
        self.generator=torch.Generator(device=self.device).manual_seed(seed)

    @torch.inference_mode()
    def sample(self, observation_43: np.ndarray, active_stage: int) -> np.ndarray:
        if observation_43.shape!=(43,) or not np.isfinite(observation_43).all() or active_stage not in range(5): raise ValueError("invalid Oracle observation/stage")
        stage=np.eye(5,dtype=np.float32)[active_stage]; observation=np.concatenate((observation_43.astype(np.float32),stage))
        normalized=torch.from_numpy((observation-self.observation_mean)/self.observation_std).to(self.device).unsqueeze(0)
        action=self.model.assist(normalized,torch.zeros((1,7),device=self.device),gamma=1.0,generator=self.generator).squeeze(0).cpu().numpy()
        return np.asarray(action*self.action_std+self.action_mean,np.float64)


def evaluate_episode(predictor: OracleStagePredictor, environment_seed: int, sampling_seed: int, reward_config: AWACRewardV1Config=AWACRewardV1Config()) -> dict[str,Any]:
    config=CollectionConfig(); env=PickPlaceEnv(render_mode=None,control_timestep=config.control_timestep_s,max_episode_steps=config.max_steps,enable_camera=False); adapter=ExpertCommandAdapter(env.ik_controller,predictor.action_spec); predictor.reset_sampling(sampling_seed)
    try:
        observation,_=env.reset(seed=environment_seed,options={"randomize_arm":config.randomize_arm,"arm_joint_noise_scale":config.arm_joint_noise_scale,"randomize_object":config.randomize_object,"randomize_goal":config.randomize_goal}); adapter.reset(observation["ee_pose"],observation["q_obs"])
        state43=np.r_[env.get_policy_observation(observation),np.float32(bool(observation["object_grasped"]))].astype(np.float32); reward=AWACRewardV1Online(state43,reward_config); consecutive_ik=0; episode_return=0.; reason="timeout"; clip_steps=clip_values=adapter_clips=nan=inf=0
        for step in range(config.max_steps):
            state43=np.r_[env.get_policy_observation(observation),np.float32(bool(observation["object_grasped"]))].astype(np.float32); active_stage=int(phase_from_milestones(reward.tracker.current)); raw=predictor.sample(state43,active_stage); nan+=int(np.isnan(raw).sum());inf+=int(np.isinf(raw).sum())
            if raw.shape!=(7,) or not np.isfinite(raw).all(): reason="non_finite_diffusion_action";episode_return+=reward_config.failure_penalty;break
            outside=(raw < -1.) | (raw > 1.); clip_values+=int(outside.sum());clip_steps+=int(outside.any());bounded=np.clip(raw,-1.,1.);bounded[6]=-1. if bounded[6]<GRIPPER_OPEN_THRESHOLD else 1.; adapted=adapter.adapt(predictor.action_spec.denormalize(bounded));adapter_clips+=int(adapted.action_clipped);consecutive_ik=0 if adapted.accepted else consecutive_ik+1;next_observation,_,_,_,_=env.step(adapted.joint_target);next_state=np.r_[env.get_policy_observation(next_observation),np.float32(bool(next_observation["object_grasped"]))].astype(np.float32);reward_step=reward.step(state43,next_state,ik_failure=consecutive_ik>=config.max_consecutive_ik_failures,time_limit=step+1>=config.max_steps);episode_return+=reward_step.reward;observation=next_observation
            if reward_step.terminated or reward_step.truncated: reason=reward_step.termination_reason;break
        milestones=reward.tracker.current; success=reason=="task_success"
        return {"environment_seed":environment_seed,"diffusion_sampling_seed":sampling_seed,"success":success,"grasp":bool(milestones[0]),"lift":bool(milestones[1]),"transport":bool(milestones[2]),"place":bool(milestones[3]),"release":bool(milestones[3]),"retreat":bool(milestones[4]),"illegal_drop":reason=="illegal_drop","ik_failure":reason=="ik_failure_limit","timeout":reason=="timeout","termination_reason":reason,"failure_phase":None if success else ("RETREAT" if milestones[3] else "APPROACH"),"episode_return":float(episode_return),"episode_length":step+1,"nan_count":nan,"inf_count":inf,"out_of_bounds_steps":clip_steps,"out_of_bounds_values":clip_values,"policy_clip_steps":clip_steps,"adapter_clip_steps":adapter_clips}
    finally: env.close()


def run_evaluation(checkpoint: Path, normalization: Path, output: Path, *, formal_seeds=range(2_000_000,2_000_100), device="auto") -> dict[str,Any]:
    output.mkdir(parents=True,exist_ok=True); predictor=OracleStagePredictor(checkpoint,normalization,device_name=device); rows=[]
    for i,seed in enumerate(formal_seeds): rows.append(evaluate_episode(predictor,seed,8_000_000+seed)); print(f"formal {i+1}/{len(formal_seeds)}",flush=True)
    report={"policy":"Oracle Stage Diffusion","checkpoint":str(checkpoint.resolve()),"normalization":str(normalization.resolve()),"diffusion_steps":50,"policy_observation":"physical43_active_stage5","policy_uses_phase":True,"policy_uses_milestones":False,"future_stage_leakage":False,"environment_seeds":[formal_seeds.start,formal_seeds.stop-1],"summary":summarize(rows),"rows":rows};(output/"evaluation_report.json").write_text(json.dumps(report,indent=2)+"\n");return report
