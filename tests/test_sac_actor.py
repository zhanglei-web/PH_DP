from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mujoco_shared_control.sac.actor import (
    SACGaussianActor,
    configure_full_mean_path_distillation,
    freeze_for_mean_calibration,
    initialize_from_bc,
)


CHECKPOINT = Path("outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt")


def test_actor_shapes_bounds_and_finite_outputs() -> None:
    actor = SACGaussianActor()
    states = torch.randn(8, 42)
    mean, log_std, std = actor.distribution_stats(states)
    action, log_prob, mean_action = actor.sample_action(states)
    deterministic = actor.deterministic_action(states)
    assert mean.shape == log_std.shape == std.shape == action.shape == (8, 7)
    assert log_prob.shape == (8, 1)
    assert mean_action.shape == deterministic.shape == (8, 7)
    for tensor in (mean, log_std, std, action, log_prob, deterministic):
        assert torch.isfinite(tensor).all()
    assert torch.all(log_std >= -5.0) and torch.all(log_std <= 2.0)
    assert torch.all(action.abs() < 1.0)
    assert torch.all(deterministic.abs() < 1.0)


def test_rsample_propagates_actor_gradient() -> None:
    actor = SACGaussianActor()
    action, log_prob, _ = actor.sample_action(torch.randn(16, 42))
    (action.mean() + log_prob.mean()).backward()
    assert all(parameter.grad is not None for parameter in actor.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in actor.parameters())


def test_tanh_corrected_log_probability_is_stable_for_extreme_values() -> None:
    actor = SACGaussianActor()
    mean = torch.zeros(3, 7)
    normal = torch.distributions.Normal(mean, torch.ones_like(mean))
    pre_squash = torch.tensor([[0.0] * 7, [50.0] * 7, [-50.0] * 7])
    log_prob = actor._squashed_log_prob(normal, pre_squash)
    assert log_prob.shape == (3, 1)
    assert torch.isfinite(log_prob).all()


def test_bc_trunk_and_direct_head_mapping_are_exact() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    actor = SACGaussianActor()
    report = initialize_from_bc(actor, CHECKPOINT, option="direct_head_copy")
    state = actor.state_dict()
    mapping = {
        "network.0.weight": "trunk.0.weight", "network.0.bias": "trunk.0.bias",
        "network.2.weight": "trunk.2.weight", "network.2.bias": "trunk.2.bias",
        "network.4.weight": "trunk.4.weight", "network.4.bias": "trunk.4.bias",
        "network.6.weight": "mean_head.weight", "network.6.bias": "mean_head.bias",
    }
    for source, target in mapping.items():
        torch.testing.assert_close(checkpoint["model_state_dict"][source], state[target])
    assert torch.count_nonzero(state["log_std_head.weight"]) == 0
    torch.testing.assert_close(state["log_std_head.bias"], torch.full((7,), -3.0))
    np.testing.assert_array_equal(report.observation_mean, checkpoint["observation_mean"])


def test_direct_copy_deterministic_action_is_tanh_of_bc_output() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    from mujoco_shared_control.actor_bc.model import ActorBC
    bc = ActorBC(); bc.load_state_dict(checkpoint["model_state_dict"])
    actor = SACGaussianActor(); initialize_from_bc(actor, CHECKPOINT)
    states = torch.randn(5, 42)
    torch.testing.assert_close(actor.deterministic_action(states), torch.tanh(bc(states)))


def test_continuous_gripper_near_one_has_submillimeter_physical_error() -> None:
    normalized = 0.999
    width = 0.5 * (normalized + 1.0) * 0.08
    assert 0.08 - width == pytest.approx(0.00004)


def test_calibration_freezes_everything_except_mean_head() -> None:
    actor = SACGaussianActor()
    freeze_for_mean_calibration(actor)
    trainable = {name for name, parameter in actor.named_parameters() if parameter.requires_grad}
    assert trainable == {"mean_head.weight", "mean_head.bias"}


def test_full_distillation_trains_mean_path_only() -> None:
    actor = SACGaussianActor()
    configure_full_mean_path_distillation(actor)
    trainable = {name for name, parameter in actor.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
        "trunk.4.weight", "trunk.4.bias", "mean_head.weight", "mean_head.bias",
    }
