from __future__ import annotations

import numpy as np
import torch

from mujoco_shared_control.awac.offline import (
    AWACCritic, AWACGaussianActor, OfflineAWACConfig, OfflineAWACTrainer,
    TransitionBatch,
)


def test_awac_network_shapes_and_bounded_action() -> None:
    config = OfflineAWACConfig(actor_hidden_dims=(16, 16, 16, 16), critic_hidden_dims=(16, 16, 16, 16))
    actor = AWACGaussianActor(config)
    critic = AWACCritic(config)
    observation = torch.randn(8, 42)
    action, log_prob = actor.sample(observation)
    assert action.shape == (8, 7)
    assert log_prob.shape == (8, 1)
    assert torch.all(action >= -1) and torch.all(action <= 1)
    assert critic(observation, action).shape == (8, 1)
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in actor.trunk) == 4
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in critic.trunk) == 4


def test_awac_update_is_finite_and_soft_updates_targets() -> None:
    config = OfflineAWACConfig(actor_hidden_dims=(16, 16, 16, 16), critic_hidden_dims=(16, 16, 16, 16), batch_size=32)
    trainer = OfflineAWACTrainer(config, np.zeros(42, np.float32), np.ones(42, np.float32), device=torch.device("cpu"))
    before = {name: value.clone() for name, value in trainer.target_q1.state_dict().items()}
    batch = TransitionBatch(
        torch.randn(32, 42), torch.rand(32, 7) * 2 - 1, torch.randn(32, 1),
        torch.randn(32, 42), torch.zeros(32, 1),
    )
    metrics = trainer.update(batch)
    assert trainer.step == 1
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["awac_weight_max"] <= config.max_advantage_weight + 1e-5
    assert any(not torch.equal(before[name], value) for name, value in trainer.target_q1.state_dict().items())


def test_dataset_log_probability_is_finite_at_action_boundaries() -> None:
    actor = AWACGaussianActor(OfflineAWACConfig(actor_hidden_dims=(8, 8, 8, 8)))
    observation = torch.zeros(2, 42)
    action = torch.tensor([[-1.0] * 7, [1.0] * 7])
    assert torch.isfinite(actor.log_prob(observation, action)).all()
