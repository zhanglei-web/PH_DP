from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mujoco_shared_control.rss2023.dataset import (
    ACTION_DIM,
    OBSERVATION_DIM,
    load_episode,
    prepare_dataset,
)
from mujoco_shared_control import PickPlaceEnv
from mujoco_shared_control.rss2023.dataset import build_observation_29


def _write_episode(path: Path, *, offset: float, frames: int = 6) -> None:
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    poses = np.zeros((frames, 7), dtype=np.float64)
    poses[:, :3] = offset
    poses[:, 3:] = quaternion
    poses[2:, 3:] *= -1.0
    commands = np.zeros((frames, 8), dtype=np.float64)
    commands[:, :7] = poses
    commands[:, 7] = np.linspace(0.08, 0.0, frames)
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "observations/joint_position",
            data=np.full((frames, 7), offset, dtype=np.float64),
        )
        file.create_dataset("observations/ee_pose_xyz_wxyz", data=poses)
        file.create_dataset(
            "observations/gripper_opening",
            data=np.linspace(0.08, 0.0, frames),
        )
        file.create_dataset("observations/object_pose_xyz_wxyz", data=poses)
        file.create_dataset("observations/goal_pose_xyz_wxyz", data=poses)
        file.create_dataset("actions/user_command", data=commands)
        file.create_dataset(
            "actions/user_command_valid", data=np.ones(frames, dtype=np.uint8)
        )
        success = np.zeros(frames, dtype=np.uint8)
        success[3:] = 1
        file.create_dataset("labels/task_success", data=success)


def test_episode_builds_29_plus_8_training_pair_and_trims_success(tmp_path) -> None:
    path = tmp_path / "episode_0.h5"
    _write_episode(path, offset=0.1)

    episode = load_episode(path)

    assert episode.original_frames == 6
    assert episode.first_success_index == 3
    assert episode.retained_frames == 4
    assert episode.observation.shape == (4, OBSERVATION_DIM)
    assert episode.action.shape == (4, ACTION_DIM)
    np.testing.assert_allclose(episode.observation[:, 0:7], 0.1)
    np.testing.assert_allclose(episode.observation[:, 10], 1.0)
    np.testing.assert_allclose(episode.action[:, 3], 1.0)


def test_prepare_dataset_splits_by_episode_and_fits_train_statistics(tmp_path) -> None:
    for index in range(10):
        _write_episode(tmp_path / f"episode_{index:02d}.h5", offset=float(index))

    prepared = prepare_dataset(tmp_path, split_seed=7)

    assert len(prepared.train.paths) == 8
    assert len(prepared.validation.paths) == 1
    assert len(prepared.test.paths) == 1
    assert set(prepared.train.paths).isdisjoint(prepared.validation.paths)
    assert set(prepared.train.paths).isdisjoint(prepared.test.paths)
    normalized = prepared.observation_normalizer.normalize(prepared.train.observation)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-5)
    assert np.isfinite(normalized).all()
    manifest = prepared.manifest()
    assert manifest["observation_dim"] == 29
    assert manifest["action_dim"] == 8


def test_invalid_command_frames_are_removed(tmp_path) -> None:
    path = tmp_path / "episode.h5"
    _write_episode(path, offset=0.0)
    with h5py.File(path, "r+") as file:
        file["actions/user_command_valid"][1] = 0

    episode = load_episode(path)

    assert episode.retained_frames == 3


def test_live_observation_builder_has_documented_layout() -> None:
    env = PickPlaceEnv(enable_camera=False)
    try:
        raw, _ = env.reset(
            seed=0,
            options={"randomize_object": False, "randomize_goal": False},
        )
        observation = build_observation_29(raw)
        assert observation.shape == (29,)
        np.testing.assert_allclose(observation[:7], raw["q_obs"])
        np.testing.assert_allclose(observation[7:10], raw["ee_pose"][:3, 3])
        np.testing.assert_allclose(observation[15:18], raw["object_pose"][:3, 3])
        np.testing.assert_allclose(observation[22:25], raw["goal_pose"][:3, 3])
    finally:
        env.close()
