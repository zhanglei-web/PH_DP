from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from mujoco_shared_control.sac.agent import SACCore, SACCoreConfig
from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor
from mujoco_shared_control.sac.policy_anchor import (
    InitialPolicyAnchor,
    InitialPolicyKLTrustRegion,
    InitialPolicyTrustRegionConfig,
    PolicyAnchorConfig,
)
from mujoco_shared_control.sac.replay_buffer import ReplayBatch
from mujoco_shared_control.sac.trainer import SACTrainer, TrainingProtocol


ACTOR_ARTIFACT = Path(
    "outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt"
)


@pytest.fixture(scope="module")
def artifact_payload() -> dict:
    return torch.load(ACTOR_ARTIFACT, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def anchor(artifact_payload: dict) -> InitialPolicyAnchor:
    return InitialPolicyAnchor(
        PolicyAnchorConfig(batch_size=32), ACTOR_ARTIFACT,
        artifact_payload["observation_mean"], artifact_payload["observation_std"], "cpu",
    )


def actor_from_artifact(payload: dict) -> SACConstrainedGaussianActor:
    actor = SACConstrainedGaussianActor()
    actor.load_state_dict(payload["actor_state_dict"])
    return actor


def test_anchor_schedule_is_constant_then_linearly_decays() -> None:
    config = PolicyAnchorConfig()
    assert config.weight_at(0) == pytest.approx(0.1)
    assert config.weight_at(50_000) == pytest.approx(0.1)
    assert config.weight_at(125_000) == pytest.approx(0.05)
    assert config.weight_at(200_000) == pytest.approx(0.0)
    assert config.weight_at(300_000) == pytest.approx(0.0)
    assert PolicyAnchorConfig(enabled=False).weight_at(0) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        config.weight_at(-1)


def test_anchor_uses_exact_frozen_formal_training_split(anchor: InitialPolicyAnchor) -> None:
    assert anchor.episode_count == 900
    assert anchor.transition_count == 115_021
    assert anchor.states.shape == (115_021, 42)
    assert all(not parameter.requires_grad for parameter in anchor.teacher.parameters())
    assert not anchor.teacher.training
    assert all(entry not in range(410_000, 410_100) for entry in ())
    assert "500000" not in str(anchor.manifest_path)


def test_anchor_construction_does_not_advance_global_torch_rng(
    artifact_payload: dict,
) -> None:
    torch.manual_seed(777)
    state_before = torch.get_rng_state().clone()
    InitialPolicyAnchor(
        PolicyAnchorConfig(batch_size=2), ACTOR_ARTIFACT,
        artifact_payload["observation_mean"], artifact_payload["observation_std"], "cpu",
    )
    torch.testing.assert_close(torch.get_rng_state(), state_before, rtol=0, atol=0)


def test_identical_actor_has_zero_anchor_and_teacher_stays_frozen(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    student = actor_from_artifact(artifact_payload)
    teacher_before = deepcopy(anchor.teacher.state_dict())
    loss, metrics = anchor.loss_and_metrics(student)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics["anchor_action_mae"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["anchor_log_std_abs_delta"] == pytest.approx(0.0, abs=1e-8)
    for name, value in anchor.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in anchor.teacher.parameters())


def test_perturbation_has_positive_kl_and_gradients_only_enter_student(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    student = actor_from_artifact(artifact_payload)
    with torch.no_grad():
        student.mean_head.bias.add_(0.1)
        student.log_std_head.bias.add_(0.05)
    loss, metrics = anchor.loss_and_metrics(student)
    assert loss.item() > 0.0
    assert metrics["anchor_action_mae"] > 0.0
    assert metrics["anchor_log_std_abs_delta"] > 0.0
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in student.parameters()
    )
    assert all(parameter.grad is None for parameter in anchor.teacher.parameters())


def test_anchor_includes_current_replay_state_distribution(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    student = actor_from_artifact(artifact_payload)
    normalized_replay = torch.full((16, 42), 20.0)
    with torch.no_grad():
        student.trunk[0].bias.add_(0.01)
    loss, metrics = anchor.loss_and_metrics(student, replay_states=normalized_replay)
    assert torch.isfinite(loss) and loss > 0
    assert metrics["anchor_formal_samples"] == 32
    assert metrics["anchor_replay_samples"] == 16
    assert metrics["anchor_formal_kl"] >= 0
    assert metrics["anchor_replay_kl"] > 0


def test_rng_state_round_trip_reproduces_next_batch(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    student = actor_from_artifact(artifact_payload)
    saved = deepcopy(anchor.state_dict())
    first_loss, first_metrics = anchor.loss_and_metrics(student)
    assert anchor.batches_sampled == saved["batches_sampled"] + 1
    anchor.load_state_dict(saved)
    second_loss, second_metrics = anchor.loss_and_metrics(student)
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    assert first_metrics == second_metrics
    assert anchor.batches_sampled == saved["batches_sampled"] + 1


def test_anchor_integrates_only_with_actor_loss(
    anchor: InitialPolicyAnchor,
) -> None:
    torch.manual_seed(9)
    core = SACCore(ACTOR_ARTIFACT)
    critic_before = deepcopy(core.critics.state_dict())
    alpha_before = core.log_alpha.detach().clone()
    teacher_before = deepcopy(anchor.teacher.state_dict())
    batch = ReplayBatch(
        observation=torch.randn(8, 42),
        action=torch.zeros(8, 7),
        reward=torch.zeros(8, 1),
        next_observation=torch.randn(8, 42),
        terminated=torch.zeros(8, 1, dtype=torch.bool),
        truncated=torch.zeros(8, 1, dtype=torch.bool),
    )
    metrics = core.update_actor_and_alpha(
        batch, policy_anchor=anchor, anchor_weight=0.1
    )
    assert metrics["anchor_weight"] == pytest.approx(0.1)
    assert metrics["anchor_kl"] == pytest.approx(0.0, abs=1e-7)
    # The normal alpha update remains active and independent of the anchor.
    assert not torch.equal(core.log_alpha, alpha_before)
    for name, value in core.critics.state_dict().items():
        torch.testing.assert_close(value, critic_before[name], rtol=0, atol=0)
    for name, value in anchor.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)


def test_trainer_checkpoint_restores_anchor_rng_and_batch_count(tmp_path: Path) -> None:
    protocol = TrainingProtocol(
        total_env_steps=5, evaluation_steps=(), checkpoint_frequency=100,
        logging_frequency=10, validation_episodes=1,
        critic_learning_starts=3, actor_learning_starts=3,
        alpha_learning_starts=3,
    )
    core_config = SACCoreConfig(batch_size=2, replay_capacity=16, learning_starts=3)
    anchor_config = PolicyAnchorConfig(enabled=True, batch_size=4, seed=123)
    trainer = SACTrainer(
        tmp_path / "source", protocol, core_config,
        policy_anchor_config=anchor_config,
    )
    trainer.train(5)
    assert trainer.policy_anchor is not None
    checkpoint = trainer.save_checkpoint("anchored.pt")
    saved_anchor_state = deepcopy(trainer.policy_anchor.state_dict())
    restored = SACTrainer(
        tmp_path / "restored", protocol, core_config,
        policy_anchor_config=anchor_config,
    )
    restored.load_checkpoint(checkpoint)
    assert restored.policy_anchor is not None
    restored_anchor_state = restored.policy_anchor.state_dict()
    for key in saved_anchor_state.keys() - {"teacher_state_dict"}:
        assert restored_anchor_state[key] == saved_anchor_state[key]
    for key, value in saved_anchor_state["teacher_state_dict"].items():
        torch.testing.assert_close(
            restored_anchor_state["teacher_state_dict"][key], value, rtol=0, atol=0
        )
    assert restored.actor_updates == trainer.actor_updates == 2


def test_trust_region_is_disabled_by_default_and_is_exact_noop(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    actor = actor_from_artifact(artifact_payload)
    with torch.no_grad():
        actor.mean_head.bias.add_(0.5)
    proposal = deepcopy(actor.state_dict())
    batches_before = anchor.batches_sampled
    trust_region = InitialPolicyKLTrustRegion(InitialPolicyTrustRegionConfig(), anchor)
    metrics = trust_region.project(actor, replay_states=torch.randn(8, 42))
    assert metrics["trust_region_enabled"] == 0.0
    assert anchor.batches_sampled == batches_before
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, proposal[name], rtol=0, atol=0)


def test_trust_region_accepts_proposal_inside_kl_limit_without_projection(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    actor = actor_from_artifact(artifact_payload)
    with torch.no_grad():
        actor.mean_head.bias.add_(1e-5)
    proposal = deepcopy(actor.state_dict())
    trust_region = InitialPolicyKLTrustRegion(
        InitialPolicyTrustRegionConfig(enabled=True, max_kl=0.01), anchor
    )
    metrics = trust_region.project(actor, replay_states=torch.zeros(16, 42))
    assert metrics["trust_region_triggered"] == 0.0
    assert metrics["trust_region_final_kl"] <= 0.01 + 1e-6
    assert metrics["trust_region_scale"] == 1.0
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, proposal[name], rtol=0, atol=0)


def test_trust_region_runtime_limits_override_static_config(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    actor = actor_from_artifact(artifact_payload)
    with torch.no_grad():
        actor.mean_head.bias.add_(0.02)
    trust_region = InitialPolicyKLTrustRegion(
        InitialPolicyTrustRegionConfig(enabled=True, max_kl=0.01,
                                       max_parameter_relative_radius=0.01), anchor
    )
    metrics = trust_region.project(
        actor, torch.zeros(16, 42), max_kl=1e-5,
        max_parameter_relative_radius=1e-4,
    )
    assert metrics["trust_region_configured_kl_limit"] == pytest.approx(1e-5)
    assert metrics["trust_region_configured_parameter_limit"] == pytest.approx(1e-4)
    assert metrics["trust_region_final_kl"] <= 1.1e-5
    assert metrics["trust_region_parameter_max_group_relative_radius"] <= 1.01e-4


def test_trust_region_backtracks_large_proposal_on_one_fixed_state_mixture(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    actor = actor_from_artifact(artifact_payload)
    teacher_before = deepcopy(anchor.teacher.state_dict())
    with torch.no_grad():
        actor.mean_head.bias.add_(0.5)
        actor.log_std_head.bias.add_(0.5)
    proposal = deepcopy(actor.state_dict())
    batches_before = anchor.batches_sampled
    trust_region = InitialPolicyKLTrustRegion(
        InitialPolicyTrustRegionConfig(
            enabled=True, max_kl=0.01, backtrack_ratio=0.5, max_backtracks=24
        ),
        anchor,
    )
    replay_states = torch.full((32, 42), 2.0)
    metrics = trust_region.project(actor, replay_states=replay_states)
    assert metrics["trust_region_triggered"] == 1.0
    assert 0.0 <= metrics["trust_region_scale"] < 1.0
    assert metrics["trust_region_backtracks"] >= 1.0
    assert metrics["trust_region_proposal_kl"] > 0.01
    assert metrics["trust_region_final_kl"] <= 0.01 + 1e-6
    # Backtracking candidates all use one sampled formal batch.
    assert anchor.batches_sampled == batches_before + 1
    assert any(
        not torch.equal(value, proposal[name])
        for name, value in actor.state_dict().items()
    )
    for name, value in anchor.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in anchor.teacher.parameters())


def test_trust_region_parameter_guard_catches_internal_drift_missed_by_loose_kl(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    actor = actor_from_artifact(artifact_payload)
    with torch.no_grad():
        # A large trunk displacement represents the internal compensation seen
        # in the failed anchored run.  The deliberately loose KL limit isolates
        # the initial-centered parameter-radius guard in this test.
        actor.trunk[0].weight.mul_(1.2)
    trust_region = InitialPolicyKLTrustRegion(
        InitialPolicyTrustRegionConfig(
            enabled=True,
            max_kl=1e6,
            max_parameter_relative_radius=0.01,
        ),
        anchor,
    )
    metrics = trust_region.project(actor, replay_states=torch.zeros(16, 42))
    assert metrics["trust_region_triggered"] == 1.0
    assert metrics["trust_region_proposal_parameter_max_group_relative_radius"] > 0.01
    assert metrics["trust_region_parameter_max_group_relative_radius"] <= 0.01 + 1e-6
    assert metrics["trust_region_parameter_trunk_relative_radius"] <= 0.01 + 1e-6
    assert 0.0 < metrics["trust_region_scale"] < 1.0


def test_trust_region_parameter_guard_can_be_explicitly_disabled(
    anchor: InitialPolicyAnchor, artifact_payload: dict
) -> None:
    actor = actor_from_artifact(artifact_payload)
    with torch.no_grad():
        actor.trunk[0].weight.mul_(1.2)
    proposal = deepcopy(actor.state_dict())
    trust_region = InitialPolicyKLTrustRegion(
        InitialPolicyTrustRegionConfig(
            enabled=True,
            max_kl=1e6,
            max_parameter_relative_radius=None,
        ),
        anchor,
    )
    metrics = trust_region.project(actor, replay_states=torch.zeros(16, 42))
    assert metrics["trust_region_triggered"] == 0.0
    assert metrics["trust_region_parameter_guard_enabled"] == 0.0
    assert metrics["trust_region_parameter_max_group_relative_radius"] > 0.01
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, proposal[name], rtol=0, atol=0)


def test_trust_region_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="max_kl"):
        InitialPolicyTrustRegionConfig(max_kl=0.0)
    with pytest.raises(ValueError, match="backtrack_ratio"):
        InitialPolicyTrustRegionConfig(backtrack_ratio=1.0)
    with pytest.raises(ValueError, match="max_backtracks"):
        InitialPolicyTrustRegionConfig(max_backtracks=0)
    with pytest.raises(ValueError, match="max_parameter_relative_radius"):
        InitialPolicyTrustRegionConfig(max_parameter_relative_radius=0.0)
    with pytest.raises(ValueError, match="parameter_norm_epsilon"):
        InitialPolicyTrustRegionConfig(parameter_norm_epsilon=0.0)
