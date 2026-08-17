"""Inference boundary for Hybrid BC and Hybrid AWAC checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.awac.hybrid import HybridAWACConfig, HybridActor
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.rule_pick_place import RuleExpertConfig


class HybridCheckpointPredictor:
    def __init__(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        config = HybridAWACConfig(**payload["training_config"])
        self.model = HybridActor(config)
        state = payload["actor"] if "actor" in payload else payload["actor_state_dict"]
        self.model.load_state_dict(state); self.model.eval()
        self.mean = np.asarray(payload["observation_mean"], np.float32)
        self.std = np.asarray(payload["observation_std"], np.float32)
        self.action_spec = ExpertActionSpec()
        rule = RuleExpertConfig()
        self.normalized_open = float(self.action_spec.normalize(np.r_[np.zeros(6), rule.open_gripper_m])[6])
        self.normalized_close = float(self.action_spec.normalize(np.r_[np.zeros(6), rule.close_gripper_m])[6])

    def normalized_action(
        self, policy_state: np.ndarray, object_grasped: bool | None = None,
        task_milestones: np.ndarray | None = None,
    ) -> np.ndarray:
        if object_grasped is None:
            raise ValueError("Hybrid Actor requires object_grasped")
        state = np.r_[np.asarray(policy_state, np.float32), np.float32(object_grasped)]
        if self.mean.shape == (48,):
            if task_milestones is None or np.asarray(task_milestones).shape != (5,):
                raise ValueError("48-D Hybrid Actor requires task_milestones[5]")
            state = np.r_[state, np.asarray(task_milestones, np.float32)]
        elif self.mean.shape != (43,):
            raise ValueError(f"unsupported Hybrid observation shape {self.mean.shape}")
        normalized = (state - self.mean) / self.std
        with torch.inference_mode():
            continuous, close, _probability = self.model.deterministic_action(
                torch.from_numpy(normalized).unsqueeze(0)
            )
        gripper = self.normalized_close if bool(close.item()) else self.normalized_open
        return np.r_[continuous.squeeze(0).numpy().astype(np.float64), gripper]
