"""Model-independent per-step experiment trace recording.

The recorder deliberately stores diagnostics such as active stage and milestones
without making them part of a policy input.  E1, E2 and E3 can therefore share
the same trace schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


_VECTOR_FIELDS = {
    "state_43": (43,),
    "clean_pilot_action_7": (7,),
    "raw_pilot_action_7": (7,),
    "assisted_action_7": (7,),
    "postprocessed_action_7": (7,),
    "clipped_assisted_action_7": (7,),
    "executed_action_7": (7,),
    "milestone_t": (5,),
    "ee_position": (3,),
    "object_position": (3,),
    "goal_position": (3,),
}


class EpisodeTraceRecorder:
    """Collect one episode and save a fixed-shape compressed NumPy trace."""

    def __init__(self, episode_id: str) -> None:
        self.episode_id = str(episode_id)
        self._rows: dict[str, list[Any]] = {
            "step_index": [],
            "simulation_time": [],
            **{name: [] for name in _VECTOR_FIELDS},
            "active_stage": [],
            "object_grasped": [],
            "reward": [],
            "adapter_accepted": [],
            "action_clipped": [],
            "fallback_used": [],
            "diffusion_inference_ms": [],
            "ee_object_distance": [],
            "object_goal_distance": [],
            "ee_goal_distance": [],
            "gripper_opening": [],
            "translation_correction_norm_normalized": [],
            "rotation_correction_norm_normalized": [],
            "translation_correction_m": [],
            "rotation_correction_rad": [],
            "motion_cosine_similarity": [],
            "gripper_changed_by_assist": [],
        }

    @property
    def length(self) -> int:
        return len(self._rows["step_index"])

    def append_step(
        self,
        *,
        step_index: int,
        simulation_time: float,
        state_43: np.ndarray,
        clean_pilot_action_7: np.ndarray,
        raw_pilot_action_7: np.ndarray,
        assisted_action_7: np.ndarray,
        postprocessed_action_7: np.ndarray,
        clipped_assisted_action_7: np.ndarray,
        executed_action_7: np.ndarray,
        milestone_t: np.ndarray,
        active_stage: int,
        object_grasped: bool,
        reward: float,
        adapter_accepted: bool,
        action_clipped: bool,
        fallback_used: bool,
        diffusion_inference_ms: float,
        ee_position: np.ndarray,
        object_position: np.ndarray,
        goal_position: np.ndarray,
        ee_object_distance: float,
        object_goal_distance: float,
        ee_goal_distance: float,
        gripper_opening: float,
        translation_correction_norm_normalized: float = 0.0,
        rotation_correction_norm_normalized: float = 0.0,
        translation_correction_m: float = 0.0,
        rotation_correction_rad: float = 0.0,
        motion_cosine_similarity: float = 1.0,
        gripper_changed_by_assist: bool = False,
    ) -> None:
        values: dict[str, Any] = {
            "step_index": int(step_index),
            "simulation_time": float(simulation_time),
            "active_stage": int(active_stage),
            "object_grasped": bool(object_grasped),
            "reward": float(reward),
            "adapter_accepted": bool(adapter_accepted),
            "action_clipped": bool(action_clipped),
            "fallback_used": bool(fallback_used),
            "diffusion_inference_ms": float(diffusion_inference_ms),
            "ee_object_distance": float(ee_object_distance),
            "object_goal_distance": float(object_goal_distance),
            "ee_goal_distance": float(ee_goal_distance),
            "gripper_opening": float(gripper_opening),
            "translation_correction_norm_normalized": float(translation_correction_norm_normalized),
            "rotation_correction_norm_normalized": float(rotation_correction_norm_normalized),
            "translation_correction_m": float(translation_correction_m),
            "rotation_correction_rad": float(rotation_correction_rad),
            "motion_cosine_similarity": float(motion_cosine_similarity),
            "gripper_changed_by_assist": bool(gripper_changed_by_assist),
        }
        for name, shape in _VECTOR_FIELDS.items():
            value = np.asarray(locals()[name], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            values[name] = value.copy()
        finite_scalars = (
            values["simulation_time"], values["reward"],
            values["diffusion_inference_ms"], values["ee_object_distance"],
            values["object_goal_distance"], values["ee_goal_distance"],
            values["gripper_opening"],
        )
        if not np.isfinite(finite_scalars).all():
            raise ValueError("trace scalar fields must be finite")
        for name, value in values.items():
            self._rows[name].append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        if self.length == 0:
            raise ValueError("cannot materialize an empty episode trace")
        arrays: dict[str, np.ndarray] = {}
        for name, values in self._rows.items():
            if name in _VECTOR_FIELDS:
                arrays[name] = np.stack(values).astype(np.float32, copy=False)
            elif name in {"step_index", "active_stage"}:
                arrays[name] = np.asarray(values, dtype=np.int32)
            elif name in {"object_grasped", "adapter_accepted", "action_clipped", "fallback_used", "gripper_changed_by_assist"}:
                arrays[name] = np.asarray(values, dtype=np.bool_)
            else:
                arrays[name] = np.asarray(values, dtype=np.float32)
        lengths = {len(value) for value in arrays.values()}
        if lengths != {self.length}:
            raise RuntimeError(f"trace fields are misaligned: {lengths}")
        if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fc"):
            raise RuntimeError("trace contains NaN/Inf")
        return arrays

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **self.arrays())
        return destination


def load_trace(path: str | Path) -> dict[str, np.ndarray]:
    """Load a trace into owned arrays so the source file can be closed."""

    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    if not arrays or len({len(value) for value in arrays.values()}) != 1:
        raise ValueError("trace fields are missing or misaligned")
    return arrays
