"""Paper-protocol transition adapter for the local pick-and-place episodes.

The RSS 2023 Block Pushing environment applies a bounded incremental action to
the current end-effector target.  The local recorder stores absolute Cartesian
targets, so this module performs the task-specific coordinate conversion while
preserving the paper's state/action semantics and hiding the goal from the
copilot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from mujoco_shared_control.rss2023.dataset import (
    _canonicalize_pose_sequence,
    _require_dataset,
    discover_episode_files,
)
from mujoco_shared_control.utils.pose import matrix_to_quaternion


PAPER_OBSERVATION_DIM = 23
PAPER_ACTION_DIM = 7
TRANSLATION_STEP_M = 0.025  # 0.5 m/s at the recorded 20 Hz command rate.
ROTATION_STEP_RAD = 0.1  # 2 rad/s at 20 Hz.
GRIPPER_STEP_M = 0.08
ACTION_SCALE = np.array(
    [
        TRANSLATION_STEP_M,
        TRANSLATION_STEP_M,
        TRANSLATION_STEP_M,
        ROTATION_STEP_RAD,
        ROTATION_STEP_RAD,
        ROTATION_STEP_RAD,
        GRIPPER_STEP_M,
    ],
    dtype=np.float64,
)

PAPER_OBSERVATION_NAMES = (
    "object_x", "object_y", "object_z",
    "object_qw", "object_qx", "object_qy", "object_qz",
    "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
    "target_x", "target_y", "target_z",
    "target_qw", "target_qx", "target_qy", "target_qz",
    "gripper_opening", "target_gripper_opening",
)
PAPER_ACTION_NAMES = (
    "delta_x_normalized", "delta_y_normalized", "delta_z_normalized",
    "delta_rotation_x_normalized", "delta_rotation_y_normalized",
    "delta_rotation_z_normalized", "delta_gripper_normalized",
)


def _wxyz_to_xyzw(values: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return array[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(values: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return array[..., [3, 0, 1, 2]]


def command_deltas(
    commands: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return previous targets and normalized incremental actions."""
    current = _canonicalize_pose_sequence(np.asarray(commands, dtype=np.float64))
    if current.ndim != 2 or current.shape[1] != 8:
        raise ValueError("commands must have shape (N, 8)")
    previous = np.concatenate((current[:1], current[:-1]), axis=0)
    translation = current[:, :3] - previous[:, :3]
    previous_rotation = Rotation.from_quat(_wxyz_to_xyzw(previous[:, 3:7]))
    current_rotation = Rotation.from_quat(_wxyz_to_xyzw(current[:, 3:7]))
    rotation = (previous_rotation.inv() * current_rotation).as_rotvec()
    gripper = current[:, 7] - previous[:, 7]
    action = np.concatenate((translation, rotation, gripper[:, None]), axis=1)
    action /= ACTION_SCALE
    # The original environment clips normalized actions to its Box[-1, 1].
    action = np.clip(action, -1.0, 1.0)
    return previous.astype(np.float32), action.astype(np.float32)


def apply_incremental_action(
    target: NDArray[np.floating], action: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Apply a normalized paper-style action to an absolute Cartesian target."""
    previous = np.asarray(target, dtype=np.float64)
    normalized = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    if previous.shape != (8,) or normalized.shape != (PAPER_ACTION_DIM,):
        raise ValueError("target/action must have shapes (8,) and (7,)")
    delta = normalized * ACTION_SCALE
    result = previous.copy()
    result[:3] += delta[:3]
    rotation = Rotation.from_quat(_wxyz_to_xyzw(previous[3:7]))
    result[3:7] = _xyzw_to_wxyz(
        (rotation * Rotation.from_rotvec(delta[3:6])).as_quat()
    )
    if result[3] < 0.0:
        result[3:7] *= -1.0
    result[7] = np.clip(result[7] + delta[6], 0.0, GRIPPER_STEP_M)
    return result


def build_paper_observation(
    raw_observation: dict,
    current_target: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Build the goal-free state analogous to BlockPushExpandObsWrapper."""
    target = np.asarray(current_target, dtype=np.float64)
    if target.shape != (8,):
        raise ValueError("current_target must have shape (8,)")

    def pose7(name: str) -> NDArray[np.float64]:
        pose = np.asarray(raw_observation[name], dtype=np.float64)
        return np.concatenate((pose[:3, 3], matrix_to_quaternion(pose[:3, :3])))

    state = np.concatenate(
        (
            pose7("object_pose"),
            pose7("ee_pose"),
            target[:7],
            np.asarray(raw_observation["gripper"], dtype=np.float64),
            target[7:8],
        )
    ).astype(np.float32)
    if state.shape != (PAPER_OBSERVATION_DIM,) or not np.isfinite(state).all():
        raise ValueError("paper observation must contain 23 finite values")
    return state


@dataclass(frozen=True)
class PaperEpisode:
    path: Path
    observation: NDArray[np.float32]
    action: NDArray[np.float32]
    expert_targets: NDArray[np.float32]
    initial_object_xy: NDArray[np.float64]
    initial_goal_xy: NDArray[np.float64]
    original_frames: int
    retained_frames: int


def load_paper_episode(path: str | Path) -> PaperEpisode:
    episode_path = Path(path).expanduser().resolve()
    with h5py.File(episode_path, "r") as file:
        ee = _canonicalize_pose_sequence(_require_dataset(file, "observations/ee_pose_xyz_wxyz")[:])
        obj = _canonicalize_pose_sequence(_require_dataset(file, "observations/object_pose_xyz_wxyz")[:])
        goal = _canonicalize_pose_sequence(_require_dataset(file, "observations/goal_pose_xyz_wxyz")[:])
        gripper = np.asarray(_require_dataset(file, "observations/gripper_opening")[:], dtype=np.float64)
        commands = _canonicalize_pose_sequence(_require_dataset(file, "actions/user_command")[:])
        valid = np.asarray(_require_dataset(file, "actions/user_command_valid")[:], dtype=bool)
        success = np.asarray(_require_dataset(file, "labels/task_success")[:], dtype=bool)
    lengths = {len(ee), len(obj), len(goal), len(gripper), len(commands), len(valid), len(success)}
    if len(lengths) != 1:
        raise ValueError(f"{episode_path}: required datasets have different lengths")
    original_frames = lengths.pop()
    success_indices = np.flatnonzero(success)
    end = int(success_indices[0]) + 1 if success_indices.size else original_frames
    ee, obj, goal, gripper, commands, valid = (
        value[:end] for value in (ee, obj, goal, gripper, commands, valid)
    )
    previous, action = command_deltas(commands)
    observation = np.concatenate(
        (obj, ee, previous[:, :7], gripper[:, None], previous[:, 7:8]), axis=1
    )
    valid &= np.isfinite(observation).all(axis=1) & np.isfinite(action).all(axis=1)
    observation = observation[valid].astype(np.float32)
    action = action[valid].astype(np.float32)
    targets = commands[valid].astype(np.float32)
    if len(observation) == 0:
        raise ValueError(f"{episode_path}: no valid paper-protocol transitions")
    return PaperEpisode(
        path=episode_path,
        observation=np.ascontiguousarray(observation),
        action=np.ascontiguousarray(action),
        expert_targets=np.ascontiguousarray(targets),
        initial_object_xy=obj[0, :2].copy(),
        initial_goal_xy=goal[0, :2].copy(),
        original_frames=original_frames,
        retained_frames=len(observation),
    )


def paper_split(
    paths: Iterable[Path], *, seed: int = 0, evaluation_episodes: int = 10
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    items = tuple(sorted(Path(path).resolve() for path in paths))
    if len(items) <= evaluation_episodes:
        raise ValueError("dataset needs more episodes than the paper's 10-episode evaluation")
    order = np.random.default_rng(seed).permutation(len(items))
    evaluation = tuple(items[int(index)] for index in order[:evaluation_episodes])
    training = tuple(items[int(index)] for index in order[evaluation_episodes:])
    return training, evaluation


def load_paper_dataset(
    dataset_dir: str | Path, *, split_seed: int = 0
) -> tuple[tuple[PaperEpisode, ...], tuple[PaperEpisode, ...]]:
    training_paths, evaluation_paths = paper_split(
        discover_episode_files(dataset_dir), seed=split_seed
    )
    return (
        tuple(load_paper_episode(path) for path in training_paths),
        tuple(load_paper_episode(path) for path in evaluation_paths),
    )
