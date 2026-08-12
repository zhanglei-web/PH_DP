from __future__ import annotations

from typing import Any

import numpy as np


class PickPlaceTask:
    """Minimal goal evaluation without imposing a phase or recovery state machine."""

    def __init__(self, success_tolerance: float = 0.055) -> None:
        self.success_tolerance = success_tolerance

    def evaluate(self, obs: dict[str, Any]) -> tuple[float, bool, dict[str, Any]]:
        object_position = obs["object_pose"][:3, 3]
        goal_position = obs["goal_pose"][:3, 3]
        distance = float(np.linalg.norm(object_position - goal_position))
        success = distance < self.success_tolerance
        reward = -distance + (1.0 if success else 0.0)
        return reward, success, {
            "success": success,
            "object_goal_distance": distance,
            "object_height": float(object_position[2]),
            "object_grasped": bool(obs["object_grasped"]),
        }

