from __future__ import annotations

import numpy as np
from pathlib import Path
import torch

from mujoco_shared_control.awac.hybrid import (
    HybridAWACConfig, HybridAWACTrainer, HybridActor, HybridBatch, HybridCritic,
)
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor


def small_config() -> HybridAWACConfig:
    return HybridAWACConfig(hidden_dims=(16, 16, 16, 16), batch_size=32)


def test_hybrid_actor_and_critic_shapes() -> None:
    actor = HybridActor(small_config()); critic = HybridCritic(small_config())
    state = torch.randn(8, 43)
    continuous, gripper, log_prob = actor.sample(state)
    assert continuous.shape == (8, 6) and gripper.shape == (8, 1) and log_prob.shape == (8, 1)
    assert torch.all((continuous >= -1) & (continuous <= 1))
    assert torch.all((gripper == 0) | (gripper == 1))
    assert critic(state, continuous, gripper).shape == (8, 1)
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in actor.backbone) == 4
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in critic.backbone) == 4


def test_deterministic_gripper_uses_half_probability_boundary() -> None:
    actor = HybridActor(small_config())
    actor.gripper_logit.weight.data.zero_(); actor.gripper_logit.bias.data.zero_()
    _continuous, close, probability = actor.deterministic_action(torch.zeros(3, 43))
    assert torch.all(probability == 0.5)
    assert torch.all(close == 1)


def test_hybrid_awac_update_is_finite_and_capped() -> None:
    config = small_config(); warm = HybridActor(config).state_dict()
    trainer = HybridAWACTrainer(config, np.zeros(43, np.float32), np.ones(43, np.float32), warm, torch.device("cpu"))
    batch = HybridBatch(
        torch.randn(32, 43), torch.rand(32, 6) * 2 - 1,
        torch.randint(0, 2, (32, 1)).float(), torch.randn(32, 1),
        torch.randn(32, 43), torch.zeros(32, 1),
    )
    metrics = trainer.update(batch)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["awac_weight_max"] <= config.max_advantage_weight + 1e-5


def test_milestone_state_actor_critic_and_trainer_use_48_dimensions() -> None:
    config = HybridAWACConfig(
        observation_dim=48, hidden_dims=(16, 16, 16, 16), batch_size=32,
    )
    actor = HybridActor(config); critic = HybridCritic(config)
    state = torch.randn(8, 48)
    continuous, gripper, _ = actor.sample(state)
    assert critic(state, continuous, gripper).shape == (8, 1)
    trainer = HybridAWACTrainer(
        config, np.zeros(48, np.float32), np.ones(48, np.float32),
        actor.state_dict(), torch.device("cpu"),
    )
    batch = HybridBatch(
        state, continuous.detach(), gripper.detach(), torch.zeros(8, 1),
        torch.randn(8, 48), torch.zeros(8, 1),
    )
    assert np.isfinite(trainer.update(batch)["actor_loss"])


def test_48d_checkpoint_predictor_requires_milestones(tmp_path: Path) -> None:
    config = HybridAWACConfig(observation_dim=48, hidden_dims=(16, 16, 16, 16))
    actor = HybridActor(config)
    path = tmp_path / "v3.pt"
    torch.save({
        "training_config": __import__("dataclasses").asdict(config),
        "actor": actor.state_dict(),
        "observation_mean": torch.zeros(48),
        "observation_std": torch.ones(48),
    }, path)
    predictor = HybridCheckpointPredictor(path)
    action = predictor.normalized_action(np.zeros(42), False, np.zeros(5))
    assert action.shape == (7,)
    try:
        predictor.normalized_action(np.zeros(42), False)
    except ValueError:
        pass
    else:
        raise AssertionError("48-D predictor accepted a missing milestone state")
