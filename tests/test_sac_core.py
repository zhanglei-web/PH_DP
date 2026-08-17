from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mujoco_shared_control.sac.agent import (
    SACCore,
    SACCoreConfig,
    bootstrap_mask,
    polyak_update,
)
from mujoco_shared_control.sac.critic import SACCritic, TwinSACCritic
from mujoco_shared_control.sac.replay_buffer import ReplayBatch, SACReplayBuffer
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec


ACTOR_ARTIFACT = Path(
    "outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/actor_initialized.pt"
)


def synthetic_batch(size: int = 8) -> ReplayBatch:
    return ReplayBatch(
        torch.randn(size, 42), torch.tanh(torch.randn(size, 7)),
        torch.randn(size, 1), torch.randn(size, 42),
        torch.zeros(size, 1, dtype=torch.bool),
        torch.zeros(size, 1, dtype=torch.bool),
    )


def test_critic_shapes_and_twins_are_independent() -> None:
    twins = TwinSACCritic()
    q1, q2 = twins(torch.randn(5, 42), torch.randn(5, 7))
    assert q1.shape == q2.shape == (5, 1)
    q1_parameters = list(twins.q1.parameters())
    q2_parameters = list(twins.q2.parameters())
    assert all(left is not right for left, right in zip(q1_parameters, q2_parameters, strict=True))
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(q1_parameters, q2_parameters, strict=True))
    assert any(not torch.equal(left, right) for left, right in zip(q1_parameters, q2_parameters, strict=True))


def test_replay_push_sample_schema_and_terminal_split() -> None:
    buffer = SACReplayBuffer(8, seed=4)
    policy_action = np.array([-.5, -.3, -.1, 0, .2, .4, .9], dtype=np.float32)
    buffer.add(np.zeros(42), policy_action, 1.5, np.ones(42), False, True)
    batch = buffer.sample(1)
    assert batch.observation.shape == batch.next_observation.shape == (1, 42)
    assert batch.action.shape == (1, 7)
    assert batch.reward.shape == batch.terminated.shape == batch.truncated.shape == (1, 1)
    np.testing.assert_array_equal(batch.action.numpy()[0], policy_action)
    assert not bool(batch.terminated.item()) and bool(batch.truncated.item())


def test_replay_wraparound_and_action_validation() -> None:
    buffer = SACReplayBuffer(3)
    for value in range(5):
        buffer.add(
            np.full(42, value), np.zeros(7), float(value), np.full(42, value + 1),
            False, False,
        )
    assert len(buffer) == 3 and buffer.position == 2
    assert set(buffer.reward[:, 0]) == {2.0, 3.0, 4.0}
    with pytest.raises(ValueError, match="replay policy action"):
        buffer.add(np.zeros(42), np.full(7, 1.01), 0, np.zeros(42), False, False)


def test_replay_capacity_ram_estimate() -> None:
    buffer = SACReplayBuffer(1_000_000)
    assert buffer.allocated_bytes == 370_000_000


def test_bootstrap_mask_ignores_time_limit_truncation() -> None:
    terminated = torch.tensor([[True], [False], [False]])
    truncated = torch.tensor([[False], [True], [False]])
    mask = bootstrap_mask(terminated)
    torch.testing.assert_close(mask, torch.tensor([[0.0], [1.0], [1.0]]))
    assert truncated[1].item()  # Explicitly documents that this row bootstraps.


class FixedActor(nn.Module):
    def __init__(self, log_prob: float = 0.0) -> None:
        super().__init__()
        self.log_prob = float(log_prob)

    def sample_action(self, observation: torch.Tensor):
        size = observation.shape[0]
        return (
            torch.zeros(size, 7),
            torch.full((size, 1), self.log_prob),
            torch.zeros(size, 7),
        )


class FixedTwin(nn.Module):
    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        value = torch.full((observation.shape[0], 1), 2.0)
        return value, value + 1.0


def test_td_target_true_terminal_vs_time_limit() -> None:
    core = object.__new__(SACCore)
    core.config = SACCoreConfig()
    core.device = torch.device("cpu")
    core.actor = FixedActor(log_prob=5.0)
    core.target_critics = FixedTwin()
    core.log_alpha = nn.Parameter(torch.tensor(float(np.log(0.1))))
    core.observation_mean = torch.zeros(42)
    core.observation_std = torch.ones(42)
    batch = ReplayBatch(
        torch.zeros(2, 42), torch.zeros(2, 7), torch.ones(2, 1),
        torch.zeros(2, 42), torch.tensor([[True], [False]]),
        torch.tensor([[False], [True]]),
    )
    target = core.critic_target(batch)
    assert target[0].item() == pytest.approx(1.0)
    assert target[1].item() == pytest.approx(1.0 + 0.995 * (2.0 - 0.1 * 5.0))


def test_critic_target_uses_current_learned_alpha() -> None:
    core = object.__new__(SACCore)
    core.config = SACCoreConfig()
    core.device = torch.device("cpu")
    core.actor = FixedActor(log_prob=5.0)
    core.target_critics = FixedTwin()
    core.log_alpha = nn.Parameter(torch.tensor(float(np.log(0.02))))
    core.observation_mean = torch.zeros(42)
    core.observation_std = torch.ones(42)
    batch = ReplayBatch(
        torch.zeros(1, 42), torch.zeros(1, 7), torch.ones(1, 1),
        torch.zeros(1, 42), torch.zeros(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
    )

    low_alpha_target = core.critic_target(batch)
    assert low_alpha_target.item() == pytest.approx(
        1.0 + 0.995 * (2.0 - 0.02 * 5.0)
    )
    with torch.no_grad():
        core.log_alpha.fill_(float(np.log(0.2)))
    high_alpha_target = core.critic_target(batch)
    assert high_alpha_target.item() == pytest.approx(
        1.0 + 0.995 * (2.0 - 0.2 * 5.0)
    )
    assert high_alpha_target.item() < low_alpha_target.item()


def test_dynamic_entropy_target_does_not_create_a_second_critic_temperature() -> None:
    core = object.__new__(SACCore)
    core.config = SACCoreConfig()
    core.device = torch.device("cpu")
    core.actor = FixedActor(log_prob=5.0)
    core.target_critics = FixedTwin()
    core.log_alpha = nn.Parameter(torch.tensor(float(np.log(0.0025))))
    core.observation_mean = torch.zeros(42)
    core.observation_std = torch.ones(42)
    batch = ReplayBatch(
        torch.zeros(1, 42), torch.zeros(1, 7), torch.zeros(1, 1),
        torch.zeros(1, 42), torch.zeros(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
    )

    target_before = core.critic_target(batch)
    warm_alpha_loss = core.alpha_loss(
        torch.full((1, 1), 5.0), target_entropy=-16.7
    )
    final_alpha_loss = core.alpha_loss(
        torch.full((1, 1), 5.0), target_entropy=-7.0
    )
    target_after = core.critic_target(batch)

    torch.testing.assert_close(target_after, target_before, rtol=0, atol=0)
    assert warm_alpha_loss.item() != pytest.approx(final_alpha_loss.item())


def test_restored_learned_alpha_is_used_by_first_critic_target() -> None:
    learned_alpha = 0.037
    source = SACCore(ACTOR_ARTIFACT)
    with torch.no_grad():
        source.log_alpha.fill_(float(np.log(learned_alpha)))
    saved = source.training_state_dict()

    restored = SACCore(ACTOR_ARTIFACT)
    restored.load_training_state_dict(saved)
    restored.actor = FixedActor(log_prob=5.0)
    restored.target_critics = FixedTwin()
    restored.observation_mean = torch.zeros(42)
    restored.observation_std = torch.ones(42)
    batch = ReplayBatch(
        torch.zeros(1, 42), torch.zeros(1, 7), torch.ones(1, 1),
        torch.zeros(1, 42), torch.zeros(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
    )

    target = restored.critic_target(batch)
    assert restored.alpha.item() == pytest.approx(learned_alpha)
    assert target.item() == pytest.approx(
        1.0 + 0.995 * (2.0 - learned_alpha * 5.0)
    )


def test_polyak_direction() -> None:
    online, target = nn.Linear(1, 1), nn.Linear(1, 1)
    with torch.no_grad():
        for parameter in online.parameters(): parameter.fill_(1.0)
        for parameter in target.parameters(): parameter.zero_()
    polyak_update(online, target, 0.005)
    for parameter in target.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.005))


@pytest.mark.parametrize(("log_prob", "gradient_sign"), [(-10.0, 1), (10.0, -1)])
def test_alpha_loss_is_finite_and_has_expected_gradient_direction(
    log_prob: float, gradient_sign: int
) -> None:
    core = object.__new__(SACCore)
    core.config = SACCoreConfig()
    core.log_alpha = nn.Parameter(torch.tensor(float(np.log(0.1))))
    loss = core.alpha_loss(torch.full((4, 1), log_prob))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(core.log_alpha.grad)
    assert int(torch.sign(core.log_alpha.grad).item()) == gradient_sign
    assert core.config.target_entropy == -7.0


def test_alpha_loss_accepts_dynamic_warm_start_entropy_target() -> None:
    core = object.__new__(SACCore)
    core.config = SACCoreConfig()
    core.log_alpha = nn.Parameter(torch.tensor(float(np.log(0.0025))))
    log_prob = torch.full((4, 1), 16.7)
    warm_loss = core.alpha_loss(log_prob, target_entropy=-16.7)
    warm_loss.backward()
    assert warm_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert core.log_alpha.grad.item() == pytest.approx(0.0, abs=1e-6)


def test_core_loads_frozen_actor_and_target_critics_are_exact_frozen_copies() -> None:
    core = SACCore(ACTOR_ARTIFACT)
    payload = torch.load(ACTOR_ARTIFACT, map_location="cpu", weights_only=False)
    for name, value in core.actor.state_dict().items():
        torch.testing.assert_close(value, payload["actor_state_dict"][name], rtol=0, atol=0)
    for online, target in zip(core.critics.parameters(), core.target_critics.parameters(), strict=True):
        torch.testing.assert_close(online, target, rtol=0, atol=0)
        assert not target.requires_grad
    assert core.alpha.item() == pytest.approx(0.1)
    assert core.observation_mean.shape == core.observation_std.shape == (42,)
    assert len(core.actor_artifact_sha256) == 64


def test_synthetic_sac_update_losses_and_gradients_are_finite() -> None:
    torch.manual_seed(7)
    core = SACCore(ACTOR_ARTIFACT)
    old_actor = [parameter.detach().clone() for parameter in core.actor.parameters()]
    old_critic = [parameter.detach().clone() for parameter in core.critics.parameters()]
    old_alpha = core.log_alpha.detach().clone()
    metrics = core.update(synthetic_batch())
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(old_actor, core.actor.parameters(), strict=True))
    assert any(not torch.equal(old, new) for old, new in zip(old_critic, core.critics.parameters(), strict=True))
    assert not torch.equal(old_alpha, core.log_alpha)
    assert all(parameter.grad is None for parameter in core.target_critics.parameters())


def test_mean_only_actor_update_freezes_log_std_and_alpha() -> None:
    torch.manual_seed(17)
    core = SACCore(ACTOR_ARTIFACT)
    trunk_before = [value.detach().clone() for value in core.actor.trunk.parameters()]
    mean_before = [value.detach().clone() for value in core.actor.mean_head.parameters()]
    std_before = [value.detach().clone() for value in core.actor.log_std_head.parameters()]
    alpha_before = core.log_alpha.detach().clone()
    metrics = core.update_actor_and_alpha(
        synthetic_batch(), target_entropy=-16.7,
        update_alpha=False, freeze_log_std=True,
    )
    assert metrics["alpha_updated"] == 0.0 and metrics["log_std_frozen"] == 1.0
    assert any(not torch.equal(a, b) for a, b in zip(trunk_before, core.actor.trunk.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(mean_before, core.actor.mean_head.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(std_before, core.actor.log_std_head.parameters()))
    torch.testing.assert_close(core.log_alpha, alpha_before, rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in core.actor.log_std_head.parameters())


def test_short_mujoco_replay_and_single_update_smoke() -> None:
    torch.manual_seed(11)
    core = SACCore(ACTOR_ARTIFACT)
    buffer = SACReplayBuffer(8, seed=11)
    env = PickPlaceEnv(
        enable_camera=False, reward_version="sac_reward_v1", control_timestep=0.05
    )
    adapter = ExpertCommandAdapter(env.ik_controller, ExpertActionSpec(**core.action_spec))
    try:
        observation, info = env.reset(
            seed=400_000,
            options={"randomize_arm": True, "randomize_object": True, "randomize_goal": True},
        )
        adapter.reset(observation["ee_pose"], observation["q_obs"])
        policy_state = info["policy_obs"]
        for _ in range(2):
            actor_action = core.select_action(policy_state)
            adapted = adapter.adapt(ExpertActionSpec(**core.action_spec).denormalize(actor_action))
            assert adapted.accepted and not adapted.fallback_used
            next_observation, reward, terminated, truncated, next_info = env.step(adapted.joint_target)
            # The adapter's normalized command is the policy-level action actually
            # deployed through the control pipeline and is what replay stores.
            buffer.add(
                policy_state, adapted.normalized, reward, next_info["policy_obs"],
                terminated, truncated,
            )
            np.testing.assert_array_equal(buffer.action[len(buffer) - 1], adapted.normalized.astype(np.float32))
            observation, policy_state = next_observation, next_info["policy_obs"]
        metrics = core.update(buffer.sample(2))
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        env.close()
