"""Canonical action projection shared by autonomous and shared-control paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.rule_pick_place import RuleExpertConfig


@dataclass(frozen=True)
class GlobalActionPostprocessor:
    """Project normalized ExpertActionSpec actions to training modes."""

    normalized_close: float
    normalized_open: float

    @classmethod
    def from_expert_spec(
        cls,
        action_spec: ExpertActionSpec = ExpertActionSpec(),
        rule_config: RuleExpertConfig = RuleExpertConfig(),
    ) -> "GlobalActionPostprocessor":
        close = float(action_spec.normalize(np.r_[np.zeros(6), rule_config.close_gripper_m])[6])
        open_ = float(action_spec.normalize(np.r_[np.zeros(6), rule_config.open_gripper_m])[6])
        if not np.isfinite([close, open_]).all() or close >= open_:
            raise ValueError("invalid canonical gripper modes")
        return cls(close, open_)

    @property
    def threshold(self) -> float:
        return 0.5 * (self.normalized_close + self.normalized_open)

    def __call__(self, action_7: np.ndarray) -> np.ndarray:
        action = np.asarray(action_7, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("action must be finite with shape (7,)")
        result = np.clip(action, -1.0, 1.0)
        result[6] = self.normalized_close if result[6] < self.threshold else self.normalized_open
        return result

    def report(self, counts: dict[str, int] | None = None) -> dict[str, object]:
        return {
            "training_close_mode": self.normalized_close,
            "training_open_mode": self.normalized_open,
            "threshold": self.threshold,
            "counts": counts or {},
        }
