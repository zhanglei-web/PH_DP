"""Read-only SAC exploration and local counterfactual diagnostic helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import torch

from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.sac.constrained_actor import constrained_transform, project_to_admissible


@dataclass
class EnvironmentSnapshot:
    integration_state: np.ndarray
    episode_steps: int
    previous_observation: dict[str, Any]
    sac_task: Any
    adapter_target: np.ndarray
    adapter_joint_target: np.ndarray
    consecutive_ik: int


def snapshot_environment(
    env: PickPlaceEnv, adapter: ExpertCommandAdapter, consecutive_ik: int
) -> EnvironmentSnapshot:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, state, spec)
    if adapter._target is None or adapter._joint_target is None:  # noqa: SLF001
        raise RuntimeError("adapter must be initialized before snapshot")
    return EnvironmentSnapshot(
        state.copy(), env._episode_steps, deepcopy(env._previous_observation),  # noqa: SLF001
        deepcopy(env.sac_task), adapter._target.copy(),  # noqa: SLF001
        adapter._joint_target.copy(), int(consecutive_ik),  # noqa: SLF001
    )


def restore_environment(
    env: PickPlaceEnv, adapter: ExpertCommandAdapter, snapshot: EnvironmentSnapshot
) -> int:
    mujoco.mj_setState(
        env.model, env.data, snapshot.integration_state,
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    mujoco.mj_forward(env.model, env.data)
    env._episode_steps = snapshot.episode_steps  # noqa: SLF001
    env._previous_observation = deepcopy(snapshot.previous_observation)  # noqa: SLF001
    env.sac_task = deepcopy(snapshot.sac_task)
    adapter._target = snapshot.adapter_target.copy()  # noqa: SLF001
    adapter._joint_target = snapshot.adapter_joint_target.copy()  # noqa: SLF001
    return snapshot.consecutive_ik


@torch.no_grad()
def sample_with_log_std_override(
    actor: torch.nn.Module, normalized_state: torch.Tensor, log_std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample without mutating Actor parameters or buffers."""
    mean, _current_log_std, _std = actor.distribution_stats(normalized_state)
    noise = torch.randn(
        mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
    )
    return constrained_transform(mean + np.exp(log_std) * noise)[0]


def constrained_local_step(
    action: torch.Tensor, direction: torch.Tensor, step_size: float
) -> torch.Tensor:
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if action.shape != direction.shape or action.shape[-1] != 7:
        raise ValueError("action and direction must share shape [...,7]")
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    candidate = action + step_size * direction / norm.clamp_min(1e-12)
    return project_to_admissible(candidate)


def q_action_gradient(core: Any, normalized_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    candidate = action.detach().clone().requires_grad_(True)
    q1, q2 = core.critics(normalized_state, candidate)
    gradient = torch.autograd.grad(torch.minimum(q1, q2).sum(), candidate)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("non-finite Q action gradient")
    return gradient.detach()
