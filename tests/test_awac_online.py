from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.awac.hybrid import HybridAWACConfig, HybridAWACTrainer
from mujoco_shared_control.awac.online import (
    UnifiedHybridReplay, low_noise_behavior_action, restore_hybrid_awac_trainer,
)
from mujoco_shared_control.awac.reward import AWACRewardV1Online


def write_offline(path: Path) -> None:
    np.savez_compressed(
        path, obs=np.zeros((3, 43), np.float32),
        continuous_action=np.zeros((3, 6), np.float32),
        gripper_action=np.asarray([0, 1, 0], np.float32),
        reward=np.zeros(3, np.float32), next_obs=np.ones((3, 43), np.float32),
        terminated=np.asarray([0, 0, 1], bool), truncated=np.zeros(3, bool),
    )


def test_unified_replay_appends_and_samples_without_ratio(tmp_path: Path) -> None:
    path = tmp_path / "offline.npz"; write_offline(path)
    replay = UnifiedHybridReplay(path, capacity=10, device=torch.device("cpu"))
    replay.append(np.zeros(43), np.zeros(6), 1, 1.0, np.ones(43), False, False)
    assert replay.metadata() == {
        "capacity": 10, "size": 4, "offline_transition_count": 3,
        "online_transition_count": 1,
    }
    batch = replay.sample(32, torch.Generator().manual_seed(1))
    assert batch.observation.shape == (32, 43)
    assert 0 <= replay.last_sample_online_count <= 32


def test_restore_preserves_networks_optimizers_and_step(tmp_path: Path) -> None:
    config = HybridAWACConfig(hidden_dims=(8, 8, 8, 8))
    actor = __import__("mujoco_shared_control.awac.hybrid", fromlist=["HybridActor"]).HybridActor(config)
    trainer = HybridAWACTrainer(
        config, np.zeros(43, np.float32), np.ones(43, np.float32),
        actor.state_dict(), torch.device("cpu"),
    )
    trainer.step = 5000
    path = tmp_path / "checkpoint.pt"
    torch.save(trainer.checkpoint({"format_version": "offline_awac_v2_hybrid"}), path)
    restored, _ = restore_hybrid_awac_trainer(path, device=torch.device("cpu"))
    assert restored.step == 5000
    for name, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(value, restored.actor.state_dict()[name])
    assert restored.actor_optimizer.param_groups[0]["weight_decay"] == 0


def test_restore_accepts_continuous_online_checkpoint(tmp_path: Path) -> None:
    config = HybridAWACConfig(hidden_dims=(8, 8, 8, 8))
    actor = __import__("mujoco_shared_control.awac.hybrid", fromlist=["HybridActor"]).HybridActor(config)
    trainer = HybridAWACTrainer(
        config, np.zeros(43, np.float32), np.ones(43, np.float32),
        actor.state_dict(), torch.device("cpu"),
    )
    trainer.step = 6000
    path = tmp_path / "online.pt"
    torch.save(trainer.checkpoint({
        "format_version": "online_awac_v2_hybrid",
        "online_awac_update_step": 1000,
    }), path)
    restored, payload = restore_hybrid_awac_trainer(path, device=torch.device("cpu"))
    assert restored.step == 6000
    assert payload["online_awac_update_step"] == 1000


def test_online_reward_success_and_time_limit_are_one_shot() -> None:
    state = np.zeros(43, np.float32); state[22:25] = [0, 0, .2]; state[29:32] = [0, 0, .2]
    protocol = AWACRewardV1Online(state)
    limited = protocol.step(state, state, time_limit=True)
    assert limited.truncated and not limited.terminated
    assert limited.reward < -4.9
    try:
        protocol.step(state, state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("terminal Reward V1 protocol accepted another step")


def test_low_noise_behavior_is_collection_only_and_gripper_is_deterministic() -> None:
    config = HybridAWACConfig(hidden_dims=(8, 8, 8, 8))
    actor_type = __import__("mujoco_shared_control.awac.hybrid", fromlist=["HybridActor"]).HybridActor
    actor = actor_type(config)
    observation = torch.zeros((32, 43))
    before = {name: value.clone() for name, value in actor.state_dict().items()}
    torch.manual_seed(7)
    continuous, gripper, policy_std, effective_std, probability = low_noise_behavior_action(
        actor, observation, exploration_std_scale=0.25,
    )
    assert continuous.shape == (32, 6)
    assert torch.all(continuous.abs() <= 1.0)
    torch.testing.assert_close(effective_std, policy_std * 0.25)
    torch.testing.assert_close(gripper, (probability >= 0.5).float())
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_low_noise_behavior_rejects_invalid_scale() -> None:
    config = HybridAWACConfig(hidden_dims=(8, 8, 8, 8))
    actor_type = __import__("mujoco_shared_control.awac.hybrid", fromlist=["HybridActor"]).HybridActor
    actor = actor_type(config)
    try:
        low_noise_behavior_action(actor, torch.zeros((1, 43)), exploration_std_scale=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid behavior std scale was accepted")


def test_zero_scale_behavior_is_exact_deterministic_mean() -> None:
    config = HybridAWACConfig(hidden_dims=(8, 8, 8, 8))
    actor_type = __import__("mujoco_shared_control.awac.hybrid", fromlist=["HybridActor"]).HybridActor
    actor = actor_type(config)
    observation = torch.randn((5, 43))
    mean, _log_std, logit = actor.distribution_stats(observation)
    continuous, gripper, policy_std, effective_std, probability = low_noise_behavior_action(
        actor, observation, exploration_std_scale=0.0,
    )
    torch.testing.assert_close(continuous, torch.tanh(mean))
    torch.testing.assert_close(effective_std, torch.zeros_like(policy_std))
    torch.testing.assert_close(probability, torch.sigmoid(logit))
    torch.testing.assert_close(gripper, (probability >= .5).float())
