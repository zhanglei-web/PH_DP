from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CollectionVariant(str, Enum):
    NOMINAL = "nominal"
    PERTURBED = "perturbed"


class EpisodeOutcome(str, Enum):
    SUCCESS = "success"
    RECOVERED = "recovered"
    FAILURE = "failure"


class TerminationReason(str, Enum):
    TASK_SUCCESS = "task_success"
    DELAYED_RECOVERY = "delayed_recovery"
    EXPERT_FAILED = "expert_failed"
    SETTLING_TIMEOUT = "settling_timeout"
    IK_FAILURE_LIMIT = "ik_failure_limit"
    TIME_LIMIT = "time_limit"
    INVALID_STATE = "invalid_state"


@dataclass(frozen=True)
class AutoTransition:
    step_index: int
    observation: dict[str, Any]
    next_observation: dict[str, Any]
    state_26: NDArray[np.float32]
    next_state_26: NDArray[np.float32]
    policy_state_42: NDArray[np.float32]
    next_policy_state_42: NDArray[np.float32]
    simulation_time: float
    next_simulation_time: float
    expert_command: NDArray[np.float64]
    command_after_perturbation: NDArray[np.float64]
    command_after_clipping: NDArray[np.float64]
    normalized_command: NDArray[np.float64]
    cartesian_target: NDArray[np.float64]
    executed_joint_target: NDArray[np.float64]
    mujoco_ctrl: NDArray[np.float64]
    expert_valid: bool
    command_accepted: bool
    action_clipped: bool
    fallback_used: bool
    rejection_reason: str
    reward: float
    terminated: bool
    truncated: bool
    task_success: bool
    termination_reason: str
    expert_stage: int
    next_expert_stage: int
    stage: int
    next_stage: int
    events: int
    perturbation_active: bool
    perturbation_type: str
    perturbation_magnitude: float
    entered_settling: bool
    settling_step: int
    expert_failed_step: int
    milestones: NDArray[np.uint8]
