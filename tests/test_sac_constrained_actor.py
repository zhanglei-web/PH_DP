from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mujoco_shared_control.control.expert_command_adapter import AdaptedCommand
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.sac.constrained_actor import (
    SACConstrainedGaussianActor,
    constrained_transform,
    radial_squash,
)
from mujoco_shared_control.sac.replay_buffer import SACReplayBuffer
from mujoco_shared_control.sac.trainer import _action_semantics_info
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv


@pytest.mark.parametrize("radius", [0.0, 1e-8, 1e-4, 0.7, 10.0, 50.0])
def test_radial_transform_is_finite_with_finite_gradient(radius: float) -> None:
    vector = torch.tensor([[radius, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    action, log_det = radial_squash(vector)
    assert torch.isfinite(action).all() and torch.isfinite(log_det).all()
    assert torch.linalg.vector_norm(action, dim=-1).item() < 1.0
    (action.sum() + log_det.sum()).backward()
    assert torch.isfinite(vector.grad).all()


def test_constrained_actor_shapes_bounds_logprob_and_rsample_gradient() -> None:
    actor = SACConstrainedGaussianActor()
    state = torch.randn(32, 42)
    action, log_prob, deterministic = actor.sample_action(state)
    assert action.shape == deterministic.shape == (32, 7)
    assert log_prob.shape == (32, 1)
    assert torch.isfinite(action).all() and torch.isfinite(log_prob).all()
    assert torch.all(torch.linalg.vector_norm(action[:, :3], dim=-1) < 1.0)
    assert torch.all(torch.linalg.vector_norm(action[:, 3:6], dim=-1) < 1.0)
    assert torch.all(action[:, 6].abs() < 1.0)
    (action.mean() + log_prob.mean()).backward()
    assert all(parameter.grad is not None for parameter in actor.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in actor.parameters())


def test_radial_analytic_logdet_matches_autograd() -> None:
    for vector in (
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1e-5, -2e-5, 3e-5], dtype=torch.float64),
        torch.tensor([0.3, -0.5, 0.7], dtype=torch.float64),
        torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64),
    ):
        analytic = radial_squash(vector.unsqueeze(0))[1].squeeze()
        jacobian = torch.autograd.functional.jacobian(
            lambda value: radial_squash(value.unsqueeze(0))[0].squeeze(0), vector
        )
        numerical = torch.linalg.slogdet(jacobian).logabsdet
        torch.testing.assert_close(analytic, numerical, rtol=1e-8, atol=1e-10)


def test_full_transform_logdet_matches_seven_dimensional_autograd() -> None:
    value = torch.tensor([.2, -.3, .4, -.7, .1, .5, 1.1], dtype=torch.float64)
    action, analytic = constrained_transform(value.unsqueeze(0))
    jacobian = torch.autograd.functional.jacobian(
        lambda item: constrained_transform(item.unsqueeze(0))[0].squeeze(0), value
    )
    numerical = torch.linalg.slogdet(jacobian).logabsdet
    torch.testing.assert_close(analytic.squeeze(), numerical, rtol=1e-8, atol=1e-10)
    assert action.shape == (1, 7)


def test_replay_keeps_attempted_policy_action_when_fallback_deploys_another_action() -> None:
    spec = ExpertActionSpec()
    attempted = np.array([.8, .2, -.1, .1, .2, .3, -.25], np.float32)
    fallback = AdaptedCommand(
        requested=spec.denormalize(attempted),
        clipped=spec.denormalize(attempted),
        normalized=attempted.astype(np.float64),
        cartesian_target=np.eye(4),
        joint_target=np.r_[np.zeros(7), .08],
        ik_result=None,
        accepted=False,
        action_clipped=False,
        fallback_used=True,
        rejection_reason="ik_nonconvergence",
    )
    info = _action_semantics_info(attempted, fallback, spec)
    assert info["fallback_used"] is True
    assert not np.array_equal(info["policy_action"], info["deployed_action"])
    replay = SACReplayBuffer(2)
    replay.add(np.zeros(42), info["policy_action"], -5.0, np.ones(42), True, False)
    np.testing.assert_array_equal(replay.action[0], attempted)
    assert replay.reward[0, 0] == -5.0 and replay.terminated[0, 0]


def test_ik_safety_terminal_penalizes_attempted_policy_action() -> None:
    spec = ExpertActionSpec()
    attempted = np.array([.5, -.2, .1, 0, 0, 0, -.25], np.float32)
    env = PickPlaceEnv(enable_camera=False, reward_version="sac_reward_v1")
    try:
        observation, _ = env.reset(seed=987654)
        fallback_joint_command = np.r_[observation["q_obs"], .08]
        next_observation, reward, terminated, truncated, info = env.step(
            fallback_joint_command, true_failure=True, failure_reason="ik_failure_limit"
        )
        replay = SACReplayBuffer(1)
        replay.add(
            env.get_policy_observation(observation), attempted, reward,
            info["policy_obs"], terminated, truncated,
        )
        np.testing.assert_array_equal(replay.action[0], attempted)
        assert terminated and not truncated
        assert info["reward_components"]["failure_terminal"] == -5.0
        assert reward < 0.0
        assert info["termination_reason"] == "ik_failure_limit"
        assert next_observation["q_obs"].shape == (7,)
    finally:
        env.close()


def test_replay_rejects_componentwise_legal_but_ball_inadmissible_action() -> None:
    replay = SACReplayBuffer(1)
    action = np.array([.8, .8, 0, 0, 0, 0, 0], np.float32)
    with pytest.raises(ValueError, match="translation policy action"):
        replay.add(np.zeros(42), action, 0.0, np.zeros(42), False, False)


def test_save_reload_preserves_deterministic_constrained_output(tmp_path) -> None:
    actor = SACConstrainedGaussianActor()
    state = torch.randn(5, 42)
    before = actor.deterministic_action(state).detach()
    path = tmp_path / "actor.pt"
    torch.save(copy.deepcopy(actor.state_dict()), path)
    loaded = SACConstrainedGaussianActor()
    loaded.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(loaded.deterministic_action(state), before, rtol=0, atol=0)
