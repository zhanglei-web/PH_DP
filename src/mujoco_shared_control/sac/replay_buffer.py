"""Fixed-capacity in-memory replay for SAC Core v1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mujoco_shared_control.actor_bc.model import ACTOR_ACTION_DIM, POLICY_STATE_DIM


@dataclass(frozen=True)
class ReplayBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class SACReplayBuffer:
    """Stores raw policy_state_42 and the agent-selected constrained policy action."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.observation = np.empty((capacity, POLICY_STATE_DIM), dtype=np.float32)
        self.action = np.empty((capacity, ACTOR_ACTION_DIM), dtype=np.float32)
        self.reward = np.empty((capacity, 1), dtype=np.float32)
        self.next_observation = np.empty((capacity, POLICY_STATE_DIM), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.bool_)
        self.truncated = np.empty((capacity, 1), dtype=np.bool_)
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def position(self) -> int:
        return self._position

    @property
    def allocated_bytes(self) -> int:
        return sum(
            array.nbytes for array in (
                self.observation, self.action, self.reward, self.next_observation,
                self.terminated, self.truncated,
            )
        )

    @staticmethod
    def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite with shape ({size},)")
        return result

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        obs = self._vector(observation, POLICY_STATE_DIM, "observation")
        normalized_action = self._vector(action, ACTOR_ACTION_DIM, "action")
        next_obs = self._vector(next_observation, POLICY_STATE_DIM, "next_observation")
        if np.any(normalized_action < -1.0) or np.any(normalized_action > 1.0):
            raise ValueError("replay policy action must be normalized in [-1,1]")
        if np.linalg.norm(normalized_action[:3]) > 1.0 + 1e-6:
            raise ValueError("replay translation policy action is outside the unit ball")
        if np.linalg.norm(normalized_action[3:6]) > 1.0 + 1e-6:
            raise ValueError("replay rotation policy action is outside the unit ball")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        index = self._position
        self.observation[index] = obs
        self.action[index] = normalized_action
        self.reward[index, 0] = reward
        self.next_observation[index] = next_obs
        self.terminated[index, 0] = bool(terminated)
        self.truncated[index, 0] = bool(truncated)
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError(f"cannot sample {batch_size} transitions from {self._size}")
        indices = self._rng.integers(0, self._size, size=batch_size)
        tensor = lambda value: torch.as_tensor(value[indices], device=device)
        return ReplayBatch(
            tensor(self.observation), tensor(self.action), tensor(self.reward),
            tensor(self.next_observation), tensor(self.terminated), tensor(self.truncated),
        )

    def state_dict(self) -> dict[str, object]:
        """Compact persistent state containing only initialized transitions."""
        return {
            "capacity": self.capacity,
            "position": self._position,
            "size": self._size,
            "rng_state": self._rng.bit_generator.state,
            "observation": self.observation[: self._size].copy(),
            "action": self.action[: self._size].copy(),
            "reward": self.reward[: self._size].copy(),
            "next_observation": self.next_observation[: self._size].copy(),
            "terminated": self.terminated[: self._size].copy(),
            "truncated": self.truncated[: self._size].copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity mismatch")
        size, position = int(state["size"]), int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("invalid replay size/position")
        for name in (
            "observation", "action", "reward", "next_observation", "terminated", "truncated"
        ):
            source = np.asarray(state[name])
            target = getattr(self, name)
            if source.shape != target[:size].shape or source.dtype != target.dtype:
                raise ValueError(f"invalid replay array {name}")
            target[:size] = source
        self._size, self._position = size, position
        self._rng.bit_generator.state = state["rng_state"]  # type: ignore[assignment]
