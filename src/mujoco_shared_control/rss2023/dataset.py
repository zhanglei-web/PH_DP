"""Load the compact robot state and Cartesian command from episode HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from numpy.typing import NDArray

from mujoco_shared_control.utils.pose import matrix_to_quaternion


OBSERVATION_DIM = 29
ACTION_DIM = 8
POSE_DIM = 7
STD_EPSILON = 1e-6

OBSERVATION_NAMES = (
    *(f"joint_position_{index}" for index in range(1, 8)),
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_qw",
    "ee_qx",
    "ee_qy",
    "ee_qz",
    "gripper_opening",
    "object_x",
    "object_y",
    "object_z",
    "object_qw",
    "object_qx",
    "object_qy",
    "object_qz",
    "goal_x",
    "goal_y",
    "goal_z",
    "goal_qw",
    "goal_qx",
    "goal_qy",
    "goal_qz",
)

ACTION_NAMES = (
    "command_ee_x",
    "command_ee_y",
    "command_ee_z",
    "command_ee_qw",
    "command_ee_qx",
    "command_ee_qy",
    "command_ee_qz",
    "command_gripper_opening",
)


def build_observation_29(raw_observation: dict[str, Any]) -> NDArray[np.float32]:
    """Build q + end-effector/gripper/object/goal poses for training or inference."""

    def pose7(name: str) -> NDArray[np.float64]:
        pose = np.asarray(raw_observation[name], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"{name} must have shape (4, 4)")
        return np.concatenate(
            (pose[:3, 3], matrix_to_quaternion(pose[:3, :3])), axis=0
        )

    observation = np.concatenate(
        (
            np.asarray(raw_observation["q_obs"], dtype=np.float64),
            pose7("ee_pose"),
            np.asarray(raw_observation["gripper"], dtype=np.float64),
            pose7("object_pose"),
            pose7("goal_pose"),
        ),
        axis=0,
    ).astype(np.float32)
    if observation.shape != (OBSERVATION_DIM,) or not np.isfinite(observation).all():
        raise ValueError("29-D observation must contain 29 finite values")
    return observation


@dataclass(frozen=True)
class EpisodeData:
    path: Path
    observation: NDArray[np.float32]
    action: NDArray[np.float32]
    original_frames: int
    retained_frames: int
    first_success_index: int | None


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    @classmethod
    def fit(cls, values: NDArray[np.float32]) -> "FeatureNormalizer":
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("normalizer needs a non-empty 2-D array")
        mean = values.mean(axis=0, dtype=np.float64)
        std = values.std(axis=0, dtype=np.float64)
        std = np.maximum(std, STD_EPSILON)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def normalize(self, values: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.std).astype(
            np.float32, copy=False
        )

    def denormalize(self, values: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        return (np.asarray(values, dtype=np.float32) * self.std + self.mean).astype(
            np.float32, copy=False
        )

    def state_dict(self) -> dict[str, NDArray[np.float32]]:
        return {"mean": self.mean.copy(), "std": self.std.copy()}


@dataclass(frozen=True)
class DataSplit:
    paths: tuple[Path, ...]
    observation: NDArray[np.float32]
    action: NDArray[np.float32]

    def __len__(self) -> int:
        return self.observation.shape[0]


@dataclass(frozen=True)
class PreparedDataset:
    train: DataSplit
    validation: DataSplit
    test: DataSplit
    observation_normalizer: FeatureNormalizer
    action_normalizer: FeatureNormalizer
    episode_summaries: tuple[EpisodeData, ...]

    def manifest(self) -> dict[str, Any]:
        def names(split: DataSplit) -> list[str]:
            return [str(path) for path in split.paths]

        return {
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "observation_names": list(OBSERVATION_NAMES),
            "action_names": list(ACTION_NAMES),
            "train_files": names(self.train),
            "validation_files": names(self.validation),
            "test_files": names(self.test),
            "train_frames": len(self.train),
            "validation_frames": len(self.validation),
            "test_frames": len(self.test),
        }


def discover_episode_files(dataset_dir: str | Path) -> tuple[Path, ...]:
    directory = Path(dataset_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {directory}")
    paths = tuple(
        sorted(
            path
            for path in directory.glob("*.h5")
            if not path.name.endswith(".inprogress.h5")
        )
    )
    if not paths:
        raise FileNotFoundError(f"no completed .h5 episodes found in {directory}")
    return paths


def _require_dataset(file: h5py.File, name: str) -> h5py.Dataset:
    if name not in file:
        raise ValueError(f"{file.filename}: missing required dataset {name!r}")
    return file[name]


def _canonicalize_pose_sequence(poses: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(poses, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] < POSE_DIM:
        raise ValueError(f"pose sequence must have at least 7 columns, got {result.shape}")
    quaternions = result[:, 3:7]
    norms = np.linalg.norm(quaternions, axis=1)
    valid = np.isfinite(norms) & (norms > 1e-8)
    quaternions[valid] /= norms[valid, None]
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        first = int(valid_indices[0])
        if quaternions[first, 0] < 0.0:
            quaternions[first] *= -1.0
        previous = first
        for current in valid_indices[1:]:
            if np.dot(quaternions[previous], quaternions[current]) < 0.0:
                quaternions[current] *= -1.0
            previous = int(current)
    return result


def load_episode(path: str | Path, *, trim_after_success: bool = True) -> EpisodeData:
    episode_path = Path(path).expanduser().resolve()
    with h5py.File(episode_path, "r") as file:
        q_obs = np.asarray(
            _require_dataset(file, "observations/joint_position")[:], dtype=np.float64
        )
        ee_obs = _canonicalize_pose_sequence(
            _require_dataset(file, "observations/ee_pose_xyz_wxyz")[:]
        )
        gripper_obs = np.asarray(
            _require_dataset(file, "observations/gripper_opening")[:],
            dtype=np.float64,
        )[:, None]
        object_obs = _canonicalize_pose_sequence(
            _require_dataset(file, "observations/object_pose_xyz_wxyz")[:]
        )
        goal_obs = _canonicalize_pose_sequence(
            _require_dataset(file, "observations/goal_pose_xyz_wxyz")[:]
        )
        action = _canonicalize_pose_sequence(
            _require_dataset(file, "actions/user_command")[:]
        )
        command_valid = np.asarray(
            _require_dataset(file, "actions/user_command_valid")[:], dtype=bool
        )
        success = np.asarray(
            _require_dataset(file, "labels/task_success")[:], dtype=bool
        )

    lengths = {
        q_obs.shape[0],
        ee_obs.shape[0],
        gripper_obs.shape[0],
        object_obs.shape[0],
        goal_obs.shape[0],
        action.shape[0],
        command_valid.shape[0],
        success.shape[0],
    }
    if len(lengths) != 1:
        raise ValueError(f"{episode_path}: required datasets have different lengths")
    original_frames = lengths.pop()
    if q_obs.shape[1:] != (7,) or action.shape[1:] != (ACTION_DIM,):
        raise ValueError(f"{episode_path}: invalid joint or command shape")

    first_success_index: int | None = None
    success_indices = np.flatnonzero(success)
    end = original_frames
    if success_indices.size:
        first_success_index = int(success_indices[0])
        if trim_after_success:
            end = first_success_index + 1

    observation = np.concatenate(
        (q_obs, ee_obs, gripper_obs, object_obs, goal_obs), axis=1
    )[:end]
    action = action[:end]
    valid = command_valid[:end]
    valid &= np.isfinite(observation).all(axis=1)
    valid &= np.isfinite(action).all(axis=1)
    for start in (7, 15, 22):
        valid &= np.linalg.norm(observation[:, start + 3 : start + 7], axis=1) > 1e-8
    valid &= np.linalg.norm(action[:, 3:7], axis=1) > 1e-8

    observation = observation[valid].astype(np.float32)
    action = action[valid].astype(np.float32)
    if observation.shape[1] != OBSERVATION_DIM:
        raise AssertionError(f"unexpected observation dimension {observation.shape[1]}")
    if observation.shape[0] == 0:
        raise ValueError(f"{episode_path}: no valid training frames remain")
    return EpisodeData(
        path=episode_path,
        observation=np.ascontiguousarray(observation),
        action=np.ascontiguousarray(action),
        original_frames=original_frames,
        retained_frames=observation.shape[0],
        first_success_index=first_success_index,
    )


def split_episode_paths(
    paths: Iterable[Path],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    items = tuple(sorted(Path(path) for path in paths))
    count = len(items)
    if count < 3:
        raise ValueError("at least three episodes are required for train/validation/test")
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("validation and test fractions must be between zero and one")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than one")

    validation_count = max(1, int(round(count * validation_fraction)))
    test_count = max(1, int(round(count * test_fraction)))
    while validation_count + test_count >= count:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("not enough episodes for non-empty splits")

    indices = np.random.default_rng(seed).permutation(count)
    validation_indices = indices[:validation_count]
    test_indices = indices[validation_count : validation_count + test_count]
    train_indices = indices[validation_count + test_count :]

    def select(selected: NDArray[np.integer[Any]]) -> tuple[Path, ...]:
        return tuple(items[int(index)] for index in selected)

    return select(train_indices), select(validation_indices), select(test_indices)


def _combine(episodes: Iterable[EpisodeData], paths: tuple[Path, ...]) -> DataSplit:
    episode_by_path = {episode.path: episode for episode in episodes}
    selected = [episode_by_path[path.resolve()] for path in paths]
    return DataSplit(
        paths=tuple(path.resolve() for path in paths),
        observation=np.ascontiguousarray(
            np.concatenate([episode.observation for episode in selected], axis=0)
        ),
        action=np.ascontiguousarray(
            np.concatenate([episode.action for episode in selected], axis=0)
        ),
    )


def prepare_dataset(
    dataset_dir: str | Path,
    *,
    split_seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    trim_after_success: bool = True,
) -> PreparedDataset:
    paths = discover_episode_files(dataset_dir)
    train_paths, validation_paths, test_paths = split_episode_paths(
        paths,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    episodes = tuple(
        load_episode(path, trim_after_success=trim_after_success) for path in paths
    )
    train = _combine(episodes, train_paths)
    validation = _combine(episodes, validation_paths)
    test = _combine(episodes, test_paths)
    observation_normalizer = FeatureNormalizer.fit(train.observation)
    action_normalizer = FeatureNormalizer.fit(train.action)
    return PreparedDataset(
        train=train,
        validation=validation,
        test=test,
        observation_normalizer=observation_normalizer,
        action_normalizer=action_normalizer,
        episode_summaries=episodes,
    )
