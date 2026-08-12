"""Atomic HDF5 writer for transition-aligned, image-free episodes."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
import uuid

import h5py
import numpy as np

from mujoco_shared_control.collection.types import AutoTransition, EpisodeOutcome
from mujoco_shared_control.data.recording import STATE_26_NAMES, _pose_vector, validate_episode


AUTO_SCHEMA_VERSION = "2.0.0"
UTF8 = h5py.string_dtype("utf-8")


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":"))


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def code_version(project_root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "mujoco_shared_control"],
            cwd=project_root, check=True, capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _contact(obs: dict[str, Any]) -> np.ndarray:
    contact = obs["contact"]
    return np.array([contact["left"][0], contact["right"][0],
                     contact["left_force"][0], contact["right_force"][0],
                     contact["count"][0]], dtype=np.float32)


class AutoEpisodeRecorder:
    """Append one transition at a time to `.inprogress`, then atomically publish."""

    def __init__(self, dataset_root: str | Path, metadata: dict[str, Any]) -> None:
        self.root = Path(dataset_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.episode_id = str(metadata["episode_id"])
        worker = int(metadata["worker_id"])
        index = int(metadata["worker_episode_index"])
        unique = uuid.uuid4().hex[:8]
        self.path = self.root / f"episode_{metadata['run_id']}_w{worker:03d}_{index:06d}_{unique}.inprogress.h5"
        self.file = h5py.File(self.path, "w")
        self.length = 0
        attrs = {
            **metadata,
            "schema_version": AUTO_SCHEMA_VERSION,
            "image_enabled": False,
            "camera_names_json": "[]",
            "state_26_names_json": canonical_json(STATE_26_NAMES),
            "timing_semantics": "obs_t, command_t, reward_t, obs_t_plus_1",
            "transition_alignment": "reward and terminal flags result from executing action at the same row",
        }
        for key, value in attrs.items():
            self.file.attrs[key] = canonical_json(value) if isinstance(value, (dict, list, tuple)) else value
        self._create_datasets()

    def _dataset(self, name: str, tail: tuple[int, ...], dtype: Any) -> None:
        group, leaf = name.rsplit("/", 1)
        self.file.require_group(group).create_dataset(
            leaf, shape=(0, *tail), maxshape=(None, *tail),
            chunks=(256, *tail) if not tail else (1, *tail), dtype=dtype,
        )

    def _create_datasets(self) -> None:
        specs: dict[str, tuple[tuple[int, ...], Any]] = {
            "identity/step_index": ((), np.int64),
            "timestamps/simulation_before": ((), np.float64),
            "timestamps/simulation_after": ((), np.float64),
            "observations/state_26": ((26,), np.float32),
            "observations/policy_state_42": ((42,), np.float32),
            "observations/joint_position": ((7,), np.float64),
            "observations/joint_velocity": ((7,), np.float64),
            "observations/ee_pose_xyz_wxyz": ((7,), np.float64),
            "observations/gripper_opening": ((), np.float64),
            "observations/object_pose_xyz_wxyz": ((7,), np.float64),
            "observations/goal_pose_xyz_wxyz": ((7,), np.float64),
            "observations/object_linear_velocity": ((3,), np.float64),
            "observations/object_angular_velocity": ((3,), np.float64),
            "observations/contact": ((5,), np.float32),
            "observations/object_grasped": ((), np.uint8),
            "next_observations/state_26": ((26,), np.float32),
            "next_observations/policy_state_42": ((42,), np.float32),
            "next_observations/joint_position": ((7,), np.float64),
            "next_observations/joint_velocity": ((7,), np.float64),
            "next_observations/ee_pose_xyz_wxyz": ((7,), np.float64),
            "next_observations/gripper_opening": ((), np.float64),
            "next_observations/object_pose_xyz_wxyz": ((7,), np.float64),
            "next_observations/goal_pose_xyz_wxyz": ((7,), np.float64),
            "next_observations/object_linear_velocity": ((3,), np.float64),
            "next_observations/object_angular_velocity": ((3,), np.float64),
            "next_observations/contact": ((5,), np.float32),
            "next_observations/object_grasped": ((), np.uint8),
            "actions/expert_nominal": ((7,), np.float64),
            "actions/policy_command": ((7,), np.float64),
            "actions/command_after_clipping": ((7,), np.float64),
            "actions/normalized": ((7,), np.float64),
            "actions/cartesian_target": ((4, 4), np.float64),
            "actions/executed_joint_target": ((8,), np.float64),
            "actions/mujoco_ctrl": ((8,), np.float64),
            "actions/expert_valid": ((), np.uint8),
            "actions/status": ((4,), np.uint8),
            "actions/rejection_reason": ((), UTF8),
            "labels/reward": ((), np.float64),
            "labels/terminated": ((), np.uint8),
            "labels/truncated": ((), np.uint8),
            "labels/task_success": ((), np.uint8),
            "labels/termination_reason": ((), UTF8),
            "labels/expert_stage": ((), np.uint8),
            "labels/next_expert_stage": ((), np.uint8),
            "labels/stage": ((), np.uint8),
            "labels/next_stage": ((), np.uint8),
            "labels/events": ((), np.uint32),
            "labels/entered_settling": ((), np.uint8),
            "labels/settling_step": ((), np.int32),
            "labels/expert_failed_step": ((), np.int64),
            "labels/task_milestones": ((5,), np.uint8),
            "perturbations/active": ((), np.uint8),
            "perturbations/type": ((), UTF8),
            "perturbations/magnitude": ((), np.float64),
        }
        for name, (tail, dtype) in specs.items():
            self._dataset(name, tail, dtype)

    @staticmethod
    def _observation_values(prefix: str, obs: dict[str, Any], state: np.ndarray,
                            policy_state: np.ndarray) -> dict[str, Any]:
        return {
            f"{prefix}/state_26": state,
            f"{prefix}/policy_state_42": policy_state,
            f"{prefix}/joint_position": obs["q_obs"],
            f"{prefix}/joint_velocity": obs["dq_obs"],
            f"{prefix}/ee_pose_xyz_wxyz": _pose_vector(obs["ee_pose"]),
            f"{prefix}/gripper_opening": float(obs["gripper"][0]),
            f"{prefix}/object_pose_xyz_wxyz": _pose_vector(obs["object_pose"]),
            f"{prefix}/goal_pose_xyz_wxyz": _pose_vector(obs["goal_pose"]),
            f"{prefix}/object_linear_velocity": obs["object_linear_velocity"],
            f"{prefix}/object_angular_velocity": obs["object_angular_velocity"],
            f"{prefix}/contact": _contact(obs),
            f"{prefix}/object_grasped": int(bool(obs["object_grasped"])),
        }

    def append(self, transition: AutoTransition) -> None:
        t = transition
        values = {
            "identity/step_index": t.step_index,
            "timestamps/simulation_before": t.simulation_time,
            "timestamps/simulation_after": t.next_simulation_time,
            **self._observation_values("observations", t.observation, t.state_26, t.policy_state_42),
            **self._observation_values("next_observations", t.next_observation, t.next_state_26, t.next_policy_state_42),
            "actions/expert_nominal": t.expert_command,
            "actions/policy_command": t.command_after_perturbation,
            "actions/command_after_clipping": t.command_after_clipping,
            "actions/normalized": t.normalized_command,
            "actions/cartesian_target": t.cartesian_target,
            "actions/executed_joint_target": t.executed_joint_target,
            "actions/mujoco_ctrl": t.mujoco_ctrl,
            "actions/expert_valid": int(t.expert_valid),
            "actions/status": np.array([t.command_accepted, t.action_clipped,
                                        t.fallback_used, t.command_accepted], dtype=np.uint8),
            "actions/rejection_reason": t.rejection_reason,
            "labels/reward": t.reward,
            "labels/terminated": int(t.terminated),
            "labels/truncated": int(t.truncated),
            "labels/task_success": int(t.task_success),
            "labels/termination_reason": t.termination_reason,
            "labels/expert_stage": t.expert_stage,
            "labels/next_expert_stage": t.next_expert_stage,
            "labels/stage": t.stage,
            "labels/next_stage": t.next_stage,
            "labels/events": t.events,
            "labels/entered_settling": int(t.entered_settling),
            "labels/settling_step": t.settling_step,
            "labels/expert_failed_step": t.expert_failed_step,
            "labels/task_milestones": t.milestones,
            "perturbations/active": int(t.perturbation_active),
            "perturbations/type": t.perturbation_type,
            "perturbations/magnitude": t.perturbation_magnitude,
        }
        for name, value in values.items():
            dataset = self.file[name]
            dataset.resize(self.length + 1, axis=0)
            dataset[self.length] = value
        self.length += 1
        if self.length % 20 == 0:
            self.file.flush()

    def finalize(self, outcome: EpisodeOutcome, termination_reason: str,
                 final_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.file.attrs["outcome"] = outcome.value
        self.file.attrs["termination_reason"] = termination_reason
        self.file.attrs["written_transitions"] = self.length
        for key, value in (final_metadata or {}).items():
            self.file.attrs[key] = value
        self.file.flush()
        self.file.close()
        report = validate_episode(self.path)
        category = outcome.value if report["valid"] else "invalid"
        destination_dir = self.root / category
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / self.path.name.replace(".inprogress", "")
        self.path.replace(destination)
        report["path"] = str(destination)
        destination.with_suffix(".validation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return report

    def abort(self) -> None:
        if self.file.id.valid:
            self.file.close()
