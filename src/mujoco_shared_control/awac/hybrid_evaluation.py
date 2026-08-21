"""Inference boundary for Hybrid BC and Hybrid AWAC checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.awac.hybrid import HybridAWACConfig, HybridActor
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.rule_pick_place import RuleExpertConfig
from mujoco_shared_control.awac.stageaware_state import STATE_MODE_ACTIVE_STAGE5, STATE_MODE_MILESTONES5, build_stageaware_state48


class HybridCheckpointPredictor:
    def __init__(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        config = HybridAWACConfig(**payload["training_config"])
        self.model = HybridActor(config)
        state = payload["actor"] if "actor" in payload else payload["actor_state_dict"]
        self.model.load_state_dict(state); self.model.eval()
        self.mean = np.asarray(payload["observation_mean"], np.float32)
        self.std = np.asarray(payload["observation_std"], np.float32)
        self.state_mode = payload.get("state_mode")
        if self.state_mode is None:
            self.state_mode = STATE_MODE_MILESTONES5 if self.mean.shape == (48,) else "physical43"
        self.action_spec = ExpertActionSpec()
        rule = RuleExpertConfig()
        self.normalized_open = float(self.action_spec.normalize(np.r_[np.zeros(6), rule.open_gripper_m])[6])
        self.normalized_close = float(self.action_spec.normalize(np.r_[np.zeros(6), rule.close_gripper_m])[6])

    def normalized_action(self, policy_state: np.ndarray, object_grasped: bool | None = None, task_milestones: np.ndarray | None = None, current_active_stage: int | None = None) -> np.ndarray:
        if object_grasped is None: raise ValueError("Hybrid Actor requires object_grasped")
        physical=np.r_[np.asarray(policy_state,np.float32),np.float32(object_grasped)]
        if self.state_mode==STATE_MODE_ACTIVE_STAGE5:
            if current_active_stage is None:raise ValueError("current-stage predictor requires current_active_stage")
            state=build_stageaware_state48(physical,current_active_stage,self.mean[:43],self.std[:43]);normalized=state
        elif self.state_mode==STATE_MODE_MILESTONES5:
            if task_milestones is None or np.asarray(task_milestones).shape!=(5,):raise ValueError("milestone predictor requires task_milestones[5]")
            state=np.r_[physical,np.asarray(task_milestones,np.float32)];normalized=(state-self.mean)/self.std
        elif self.state_mode=="physical43": normalized=(physical-self.mean)/self.std
        else: raise ValueError(f"unsupported explicit state_mode {self.state_mode}")
        with torch.inference_mode():
            continuous, close, _probability = self.model.deterministic_action(
                torch.from_numpy(normalized).unsqueeze(0)
            )
        gripper = self.normalized_close if bool(close.item()) else self.normalized_open
        return np.r_[continuous.squeeze(0).numpy().astype(np.float64), gripper]
