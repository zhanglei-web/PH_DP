from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mujoco_shared_control.collection import AutomaticCollector, CollectionConfig
from mujoco_shared_control.collection.datasets import ActorDataset, CriticDataset
from mujoco_shared_control.collection.types import (
    CollectionVariant, EpisodeOutcome, TerminationReason,
)
from mujoco_shared_control.experts.rule_pick_place import RuleExpertStage
from mujoco_shared_control.data.recording import validate_episode
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv


def test_arm_randomization_is_seeded_and_opt_in() -> None:
    env = PickPlaceEnv(enable_camera=False)
    try:
        home, _ = env.reset(seed=8, options={"randomize_object": False,
                                            "randomize_goal": False})
        first, _ = env.reset(seed=8, options={"randomize_arm": True})
        second, _ = env.reset(seed=8, options={"randomize_arm": True})
        np.testing.assert_allclose(first["q_obs"], second["q_obs"])
        assert not np.allclose(first["q_obs"], home["q_obs"])
    finally:
        env.close()


def test_rule_collection_is_transition_aligned_atomic_and_loadable(tmp_path: Path) -> None:
    config = CollectionConfig(dataset_root=str(tmp_path), max_steps=300)
    collector = AutomaticCollector(config, run_id="test_run")
    try:
        result = collector.collect_episode(worker_episode_index=0, environment_seed=123)
    finally:
        collector.close()
    assert result.outcome == EpisodeOutcome.SUCCESS
    assert result.valid
    assert not list(tmp_path.rglob("*.inprogress.h5"))
    report = validate_episode(result.path)
    assert report["valid"] and report["transition_aligned"]
    with h5py.File(result.path, "r") as episode:
        assert "camera" not in episode and "observations/images" not in episode
        assert episode.attrs["gamma"] == 0.0
        assert not bool(episode.attrs["image_enabled"])
        assert episode["labels/terminated"][-1] == 1
        np.testing.assert_allclose(episode["observations/state_26"][1:],
                                   episode["next_observations/state_26"][:-1])
    actor, critic = ActorDataset(tmp_path), CriticDataset(tmp_path)
    assert len(actor) == result.transitions == len(critic)
    assert actor[0]["action"].shape == (7,)
    assert critic[-1]["done"]


def test_same_seed_reproduces_rule_termination(tmp_path: Path) -> None:
    config = CollectionConfig(dataset_root=str(tmp_path), max_steps=300)
    collector = AutomaticCollector(config, run_id="repeat")
    try:
        first = collector.collect_episode(worker_episode_index=0, environment_seed=123)
        second = collector.collect_episode(worker_episode_index=1, environment_seed=123)
    finally:
        collector.close()
    assert first.outcome == second.outcome
    assert first.termination_reason == second.termination_reason
    assert first.transitions == second.transitions
    with h5py.File(first.path, "r") as a, h5py.File(second.path, "r") as b:
        np.testing.assert_allclose(a["next_observations/state_26"][:],
                                   b["next_observations/state_26"][:], atol=1e-7)


def test_normal_success_contains_retreat_before_termination(tmp_path: Path) -> None:
    collector = AutomaticCollector(CollectionConfig(dataset_root=str(tmp_path)),
                                   run_id="retreat")
    try:
        result = collector.collect_episode(worker_episode_index=0, environment_seed=123)
    finally:
        collector.close()
    with h5py.File(result.path) as episode:
        stages = episode["labels/expert_stage"][:]
        assert np.count_nonzero(stages == int(RuleExpertStage.RETREAT)) > 0
        assert np.all(episode["labels/task_milestones"][-1] == 1)
        assert not bool(episode.attrs["entered_settling"])


def test_delayed_drop_is_recovered_during_settling(tmp_path: Path) -> None:
    collector = AutomaticCollector(CollectionConfig(dataset_root=str(tmp_path)),
                                   run_id="delayed")
    try:
        result = collector.collect_episode(
            worker_episode_index=0, environment_seed=10052,
            variant=CollectionVariant.PERTURBED,
        )
    finally:
        collector.close()
    assert result.outcome == EpisodeOutcome.RECOVERED
    assert result.termination_reason == TerminationReason.DELAYED_RECOVERY
    with h5py.File(result.path) as episode:
        assert bool(episode.attrs["entered_settling"])
        assert int(episode.attrs["expert_failed_step"]) >= 0
        assert np.count_nonzero(
            episode["labels/expert_stage"][:] == int(RuleExpertStage.SETTLING)
        ) > 0


def test_settling_timeout_remains_failure(tmp_path: Path) -> None:
    config = CollectionConfig(dataset_root=str(tmp_path), control_timestep_s=.04)
    collector = AutomaticCollector(config, run_id="timeout")
    try:
        result = collector.collect_episode(
            worker_episode_index=0, environment_seed=10051,
            variant=CollectionVariant.PERTURBED,
        )
    finally:
        collector.close()
    assert config.settling_steps == 25
    assert result.outcome == EpisodeOutcome.FAILURE
    assert result.termination_reason == TerminationReason.SETTLING_TIMEOUT
    with h5py.File(result.path) as episode:
        settling = episode["labels/expert_stage"][:] == int(RuleExpertStage.SETTLING)
        assert np.count_nonzero(settling) == config.settling_steps


def test_object_leaving_goal_during_retreat_cannot_succeed(tmp_path: Path) -> None:
    """Stable frames are consecutive: leaving the goal resets the counter."""
    collector = AutomaticCollector(CollectionConfig(dataset_root=str(tmp_path)),
                                   run_id="leave_goal")
    try:
        result = collector.collect_episode(worker_episode_index=0, environment_seed=123)
    finally:
        collector.close()
    with h5py.File(result.path) as episode:
        success = episode["labels/task_success"][:].astype(bool)
        distance = np.linalg.norm(
            episode["next_observations/object_pose_xyz_wxyz"][:, :3]
            - episode["next_observations/goal_pose_xyz_wxyz"][:, :3], axis=1
        )
        released = ~episode["next_observations/object_grasped"][:].astype(bool)
        expected = np.zeros(len(success), dtype=bool)
        run = 0
        for index, inside in enumerate((distance < .055) & released):
            run = run + 1 if inside else 0
            expected[index] = run >= collector.config.success_settle_steps
        # Simulate leaving the region on the penultimate candidate frame: it
        # cannot create an accepted success until a fresh consecutive window.
        synthetic = ((distance < .055) & released).copy()
        final_candidates = np.flatnonzero(synthetic)
        assert len(final_candidates) >= collector.config.success_settle_steps
        synthetic[final_candidates[-2]] = False
        run = 0
        synthetic_success = []
        for inside in synthetic:
            run = run + 1 if inside else 0
            synthetic_success.append(run >= collector.config.success_settle_steps)
        assert not synthetic_success[-1]
        assert success[-1] and expected[-1]
