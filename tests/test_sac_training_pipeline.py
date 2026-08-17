from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mujoco_shared_control.sac.agent import SACCoreConfig
from mujoco_shared_control.sac.evaluation import evaluate_sac
from mujoco_shared_control.sac.trainer import (
    ACTOR_ARTIFACT,
    EntropyWarmStartConfig,
    InteractionMode,
    MediumHorizonReleaseConfig,
    MeanPolicyImprovementConfig,
    SACTrainer,
    TrainingProtocol,
    is_better_evaluation,
)


def tiny_trainer(
    path: Path, *, critic_starts: int = 3, actor_starts: int = 5
) -> SACTrainer:
    return SACTrainer(
        path,
        TrainingProtocol(
            total_env_steps=7, evaluation_steps=(), checkpoint_frequency=100,
            logging_frequency=2, validation_episodes=1, checkpoint_replay=True,
            critic_learning_starts=critic_starts,
            actor_learning_starts=actor_starts,
            alpha_learning_starts=actor_starts,
        ),
        SACCoreConfig(
            batch_size=2, replay_capacity=16, learning_starts=critic_starts,
        ),
    )


def test_no_update_before_learning_starts_and_update_after(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path / "run")
    first = trainer.train(3)
    assert first["global_env_steps"] == 3
    assert first["gradient_updates"] == 0
    actor_before = copy.deepcopy(trainer.core.actor.state_dict())
    alpha_before = trainer.core.log_alpha.detach().clone()
    target_before = copy.deepcopy(trainer.core.target_critics.state_dict())
    second = trainer.train(5)
    assert second["global_env_steps"] == 5
    assert second["gradient_updates"] == 2
    assert second["actor_updates"] == 0 and second["alpha_updates"] == 0
    assert second["replay_size"] == 5
    for name, value in trainer.core.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[name], rtol=0, atol=0)
    torch.testing.assert_close(trainer.core.log_alpha, alpha_before, rtol=0, atol=0)
    assert any(
        not torch.equal(value, target_before[name])
        for name, value in trainer.core.target_critics.state_dict().items()
    )
    third = trainer.train(7)
    assert third["gradient_updates"] == 4
    assert third["actor_updates"] == 2 and third["alpha_updates"] == 2
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in trainer.core.actor.state_dict().items()
    )
    assert not torch.equal(trainer.core.log_alpha, alpha_before)


def test_checkpoint_resume_restores_all_training_state_and_replay(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path / "source")
    trainer.train(7)
    checkpoint = trainer.save_checkpoint("resume.pt")
    actor = copy.deepcopy(trainer.core.actor.state_dict())
    critics = copy.deepcopy(trainer.core.critics.state_dict())
    alpha = trainer.core.log_alpha.detach().clone()
    restored = tiny_trainer(tmp_path / "restored")
    restored.load_checkpoint(checkpoint)
    assert restored.global_env_steps == 7
    assert restored.gradient_updates == 4
    assert restored.actor_updates == 2 and restored.alpha_updates == 2
    assert restored.training_stage().value == "FULL_SAC"
    assert len(restored.replay) == 7
    assert restored.core.actor_optimizer.state_dict()["state"]
    assert restored.core.critic_optimizer.state_dict()["state"]
    assert restored.core.alpha_optimizer.state_dict()["state"]
    for name, value in restored.core.actor.state_dict().items():
        torch.testing.assert_close(value, actor[name], rtol=0, atol=0)
    for name, value in restored.core.critics.state_dict().items():
        torch.testing.assert_close(value, critics[name], rtol=0, atol=0)
    torch.testing.assert_close(restored.core.log_alpha, alpha, rtol=0, atol=0)
    continued = restored.train(8)
    assert continued["global_env_steps"] == 8
    assert continued["gradient_updates"] == 5
    assert continued["actor_updates"] == 3


def test_deterministic_evaluation_has_no_updates_or_replay_writes(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path / "run", critic_starts=10, actor_starts=20)
    before_actor = copy.deepcopy(trainer.core.actor.state_dict())
    result = evaluate_sac(trainer.core, [410_000])
    assert result["episodes"] == 1
    assert len(trainer.replay) == 0 and trainer.gradient_updates == 0
    for name, value in trainer.core.actor.state_dict().items():
        torch.testing.assert_close(value, before_actor[name], rtol=0, atol=0)


def test_best_selection_uses_success_drop_then_return() -> None:
    base = {"episodes": 20, "success_rate": .2,
            "termination_reason_counts": {"illegal_drop": 4},
            "episode_return": {"mean": 1.0}}
    assert is_better_evaluation(base, None)
    higher_success = copy.deepcopy(base); higher_success["success_rate"] = .25
    assert is_better_evaluation(higher_success, base)
    fewer_drops = copy.deepcopy(base); fewer_drops["termination_reason_counts"]["illegal_drop"] = 3
    assert is_better_evaluation(fewer_drops, base)
    higher_return = copy.deepcopy(base); higher_return["episode_return"]["mean"] = 2.0
    assert is_better_evaluation(higher_return, base)


@pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True)])
def test_episode_terminal_or_truncation_causes_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminated: bool, truncated: bool
) -> None:
    import mujoco_shared_control.sac.trainer as module
    original = module.PickPlaceEnv
    class OneStepEnv(original):
        reset_calls = 0
        def reset(self, *args, **kwargs):
            type(self).reset_calls += 1
            return super().reset(*args, **kwargs)
        def step(self, *args, **kwargs):
            obs, reward, _term, _trunc, info = super().step(*args, **kwargs)
            info["termination_reason"] = "synthetic_end"
            return obs, reward, terminated, truncated, info
    monkeypatch.setattr(module, "PickPlaceEnv", OneStepEnv)
    trainer = tiny_trainer(tmp_path / "reset", critic_starts=10, actor_starts=20)
    result = trainer.train(2)
    assert result["episode_count"] == 2
    assert OneStepEnv.reset_calls == 3


def test_metric_logging_and_frozen_seed_protocol(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path / "run", critic_starts=10, actor_starts=20)
    trainer.train(2)
    rows = [json.loads(line) for line in (tmp_path / "run/training_metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["global_env_steps"] == 2
    assert rows[-1]["gradient_updates"] == 0
    config = json.loads((tmp_path / "run/config.json").read_text())
    assert config["validation_seeds"] == [410000, 410099]
    assert config["final_test_seeds"] == [500000, 500099]
    assert config["reward_version"] == "sac_reward_v1"
    assert Path(config["actor"]["actor_artifact"]) == ACTOR_ARTIFACT.resolve()


def test_stage_boundaries_are_explicit_and_checkpoint_restores_stage(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path / "stage")
    assert trainer.training_stage(3).value == "COLLECT"
    assert trainer.training_stage(4).value == "CRITIC_WARMUP"
    assert trainer.training_stage(5).value == "CRITIC_WARMUP"
    assert trainer.training_stage(6).value == "FULL_SAC"
    trainer.train(5)
    checkpoint = trainer.save_checkpoint("warmup.pt")
    restored = tiny_trainer(tmp_path / "stage_restored")
    restored.load_checkpoint(checkpoint)
    assert restored.training_stage().value == "CRITIC_WARMUP"
    assert restored.actor_checksum() == trainer.actor_checksum()
    torch.testing.assert_close(restored.core.log_alpha, trainer.core.log_alpha, rtol=0, atol=0)


def test_clean_schedule_has_no_critic_warmup(tmp_path: Path) -> None:
    trainer = SACTrainer(
        tmp_path / "clean",
        TrainingProtocol(
            total_env_steps=5, evaluation_steps=(), checkpoint_frequency=100,
            logging_frequency=10, validation_episodes=1,
            critic_learning_starts=3, actor_learning_starts=3,
            alpha_learning_starts=3,
        ),
        SACCoreConfig(batch_size=2, replay_capacity=16, learning_starts=3),
    )
    assert trainer.training_stage(3).value == "COLLECT"
    assert trainer.training_stage(4).value == "FULL_SAC"
    result = trainer.train(5)
    assert result["gradient_updates"] == 2
    assert result["actor_updates"] == 2
    assert result["alpha_updates"] == 2


def test_entropy_warm_start_schedule_reaches_standard_sac_target() -> None:
    schedule = EntropyWarmStartConfig(enabled=True)
    assert schedule.target_at(0) == pytest.approx(-16.7)
    assert schedule.target_at(50_000) == pytest.approx(-16.7)
    assert schedule.target_at(125_000) == pytest.approx(-11.85)
    assert schedule.target_at(200_000) == pytest.approx(-7.0)
    assert schedule.target_at(300_000) == pytest.approx(-7.0)
    assert EntropyWarmStartConfig().target_at(0) == pytest.approx(-7.0)


def test_mean_policy_curriculum_is_fixed_and_validated() -> None:
    config = MeanPolicyImprovementConfig(enabled=True)
    assert config.target_entropy == pytest.approx(-16.7)
    assert config.freeze_log_std and config.freeze_alpha
    assert config.deterministic_episode_fraction == pytest.approx(.5)
    assert config.stable_kl_limit == pytest.approx(1e-4)
    assert config.stable_parameter_limit == pytest.approx(1e-2)
    assert config.kl_limit_after_rollbacks(0) == pytest.approx(1e-4)
    assert config.kl_limit_after_rollbacks(1) == pytest.approx(3e-5)
    assert config.kl_limit_after_rollbacks(10) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        MeanPolicyImprovementConfig(deterministic_episode_fraction=1.1)


def test_mean_policy_episode_mixture_alternates_deterministically(tmp_path: Path) -> None:
    trainer = SACTrainer(
        tmp_path / "mixture",
        TrainingProtocol(
            total_env_steps=1, evaluation_steps=(), checkpoint_frequency=100,
            logging_frequency=100, validation_episodes=1,
            critic_learning_starts=1, actor_learning_starts=1,
            alpha_learning_starts=1,
        ),
        SACCoreConfig(batch_size=1, replay_capacity=4, learning_starts=1),
        mean_policy_improvement_config=MeanPolicyImprovementConfig(
            enabled=True, start_env_step=0,
        ),
    )
    trainer.episode_count = 0
    assert trainer.interaction_mode().value == "DETERMINISTIC_MEAN"
    trainer.episode_count = 1
    assert trainer.interaction_mode().value == "STOCHASTIC_SAC"
    trainer.episode_count = 2
    assert trainer.interaction_mode().value == "DETERMINISTIC_MEAN"


def test_medium_horizon_release_schedule_has_exact_stage_boundaries() -> None:
    schedule = MediumHorizonReleaseConfig(enabled=True)
    assert schedule.values_at(30_000) == {
        "release_stage_index": 0, "release_stage": "R0",
        "kl_limit": 1e-5, "parameter_limit": 1e-4,
        "target_entropy": -16.7,
    }
    assert schedule.values_at(39_999)["release_stage"] == "R0"
    assert schedule.values_at(40_000)["release_stage"] == "R1"
    assert schedule.values_at(60_000)["release_stage"] == "R2"
    assert schedule.values_at(80_000)["release_stage"] == "R3"
    assert schedule.values_at(100_000)["target_entropy"] == pytest.approx(-7.0)
    with pytest.raises(ValueError, match="has not started"):
        schedule.values_at(29_999)


def test_medium_horizon_schedule_resume_state_is_validated(tmp_path: Path) -> None:
    protocol = TrainingProtocol(
        total_env_steps=5, evaluation_steps=(), checkpoint_frequency=100,
        logging_frequency=10, validation_episodes=1,
        critic_learning_starts=3, actor_learning_starts=3,
        alpha_learning_starts=3,
    )
    core = SACCoreConfig(batch_size=2, replay_capacity=16, learning_starts=3)
    release = MediumHorizonReleaseConfig(enabled=True, start_env_step=0,
        boundaries=(2,4,6,8), kl_limits=(1e-5,3e-5,1e-4,3e-4),
        parameter_limits=(1e-4,3e-4,1e-3,3e-3),
        target_entropies=(-16.7,-14.,-10.,-7.))
    trainer = SACTrainer(tmp_path/"source", protocol, core,
                         medium_horizon_release_config=release)
    checkpoint = trainer.save_checkpoint("release.pt")
    restored = SACTrainer(tmp_path/"restored", protocol, core,
                          medium_horizon_release_config=release)
    restored.load_checkpoint(checkpoint)
    assert restored.medium_horizon_release_config == release


def test_deterministic_collection_uses_policy_mean_and_keeps_replay_semantics(
    tmp_path: Path,
) -> None:
    trainer = SACTrainer(
        tmp_path / "deterministic",
        TrainingProtocol(
            total_env_steps=2, evaluation_steps=(), checkpoint_frequency=100,
            logging_frequency=10, validation_episodes=1,
            critic_learning_starts=3, actor_learning_starts=4,
            alpha_learning_starts=4, deterministic_collection_until=2,
        ),
        SACCoreConfig(batch_size=2, replay_capacity=16, learning_starts=3),
    )
    trainer.train(2)
    # Recompute the first seeded rollout action from the frozen initialization.
    from mujoco_shared_control.collection.automatic import CollectionConfig
    from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
    env = PickPlaceEnv(enable_camera=False, reward_version="sac_reward_v1")
    try:
        _obs, info = env.reset(seed=trainer.protocol.training_seed_start)
        expected = trainer.core.select_action(info["policy_obs"], deterministic=True)
    finally:
        env.close()
    np.testing.assert_allclose(trainer.replay.action[0], expected, rtol=0, atol=1e-7)
    assert trainer._replay_policy_mismatch_count == 0


def test_safe_action_support_schedule_switches_before_critic_warmup() -> None:
    protocol = TrainingProtocol(
        critic_learning_starts=10_000,
        actor_learning_starts=20_000,
        alpha_learning_starts=20_000,
        deterministic_collection_until=10_000,
    )
    trainer = SACTrainer.__new__(SACTrainer)
    trainer.protocol = protocol
    trainer.global_env_steps = 0

    assert trainer.training_stage(10_000) == trainer.training_stage(0)
    assert trainer.training_stage(10_001).value == "CRITIC_WARMUP"
    assert trainer.training_stage(20_001).value == "FULL_SAC"
    assert trainer.interaction_mode(0) == InteractionMode.DETERMINISTIC_BC
    assert trainer.interaction_mode(9_999) == InteractionMode.DETERMINISTIC_BC
    assert trainer.interaction_mode(10_000) == InteractionMode.STOCHASTIC_SAC
    assert trainer.interaction_mode(15_000) == InteractionMode.STOCHASTIC_SAC
    assert trainer.interaction_mode(20_000) == InteractionMode.STOCHASTIC_SAC
    with pytest.raises(ValueError, match="Critic updates"):
        TrainingProtocol(
            critic_learning_starts=10_000,
            actor_learning_starts=20_000,
            alpha_learning_starts=20_000,
            deterministic_collection_until=20_000,
        )


def test_warmup_replay_contains_deterministic_then_stochastic_policy_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = SACTrainer(
        tmp_path / "support_warmup",
        TrainingProtocol(
            total_env_steps=5, evaluation_steps=(), checkpoint_frequency=100,
            logging_frequency=10, validation_episodes=1,
            critic_learning_starts=3, actor_learning_starts=5,
            alpha_learning_starts=5, deterministic_collection_until=3,
        ),
        SACCoreConfig(batch_size=2, replay_capacity=16, learning_starts=3),
    )
    deterministic_flags: list[bool] = []
    original_select_action = trainer.core.select_action

    def recording_select_action(policy_state, *, deterministic=False):
        deterministic_flags.append(bool(deterministic))
        return original_select_action(policy_state, deterministic=deterministic)

    monkeypatch.setattr(trainer.core, "select_action", recording_select_action)
    actor_before = copy.deepcopy(trainer.core.actor.state_dict())
    alpha_before = trainer.core.log_alpha.detach().clone()
    result = trainer.train(5)

    assert deterministic_flags == [True, True, True, False, False]
    assert result["gradient_updates"] == 2
    assert result["actor_updates"] == result["alpha_updates"] == 0
    assert trainer._interaction_mode_counts == {
        "DETERMINISTIC_BC": 3, "DETERMINISTIC_MEAN": 0,
        "STOCHASTIC_SAC": 2,
    }
    assert set(trainer._interaction_event_counts) == {
        "DETERMINISTIC_BC", "DETERMINISTIC_MEAN", "STOCHASTIC_SAC",
    }
    assert trainer._replay_policy_mismatch_count == 0
    for name, value in trainer.core.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[name], rtol=0, atol=0)
    torch.testing.assert_close(trainer.core.log_alpha, alpha_before, rtol=0, atol=0)

    checkpoint = trainer.save_checkpoint("support.pt")
    restored = SACTrainer(
        tmp_path / "support_restored", trainer.protocol,
        SACCoreConfig(batch_size=2, replay_capacity=16, learning_starts=3),
    )
    restored.load_checkpoint(checkpoint)
    assert restored._interaction_mode_counts == trainer._interaction_mode_counts
    assert restored._interaction_event_counts == trainer._interaction_event_counts
    assert restored._last_interaction_mode == InteractionMode.STOCHASTIC_SAC
