from __future__ import annotations

import numpy as np
import torch

from mujoco_shared_control.rss2023.model import DiffusionConfig
from mujoco_shared_control.rss2023.paper_dataset import (
    ACTION_SCALE,
    PAPER_ACTION_DIM,
    PAPER_OBSERVATION_DIM,
    apply_incremental_action,
    command_deltas,
)
from mujoco_shared_control.rss2023.paper_evaluation import Pilot
from mujoco_shared_control.rss2023.paper_model import PaperRSS2023Diffusion


def test_absolute_commands_round_trip_through_paper_increment() -> None:
    commands = np.array(
        [
            [0.50, 0.00, 0.30, 1.0, 0.0, 0.0, 0.0, 0.08],
            [0.51, -0.01, 0.31, 1.0, 0.0, 0.0, 0.0, 0.04],
        ]
    )
    previous, actions = command_deltas(commands)
    np.testing.assert_allclose(actions[0], 0.0)
    expected = (commands[1, [0, 1, 2, 7]] - commands[0, [0, 1, 2, 7]])
    np.testing.assert_allclose(actions[1, [0, 1, 2, 6]], expected / ACTION_SCALE[[0, 1, 2, 6]])
    np.testing.assert_allclose(apply_incremental_action(previous[1], actions[1]), commands[1], atol=1e-7)


def test_paper_model_preserves_action_exactly_at_zero_gamma() -> None:
    config = DiffusionConfig(observation_dim=PAPER_OBSERVATION_DIM, action_dim=PAPER_ACTION_DIM)
    model = PaperRSS2023Diffusion(config)
    observation = torch.randn(3, PAPER_OBSERVATION_DIM)
    action = torch.rand(3, PAPER_ACTION_DIM) * 2.0 - 1.0
    torch.testing.assert_close(model.assist(observation, action, 0.0), action)
    assert torch.isfinite(model.loss(observation, action))


def test_released_pilot_definitions_use_box_uniform_noise_and_lag() -> None:
    expert = np.full(PAPER_ACTION_DIM, 0.25)
    noisy = Pilot("noisy", probability=1.0, seed=3)
    noisy_action = noisy.action(expert)
    assert np.all(noisy_action >= -1.0) and np.all(noisy_action <= 1.0)
    assert not np.allclose(noisy_action, expert)

    laggy = Pilot("laggy", probability=1.0, seed=4)
    initial = laggy.previous.copy()
    np.testing.assert_allclose(laggy.action(expert), initial)
    zero = Pilot("zero", probability=0.6, seed=0)
    np.testing.assert_allclose(zero.action(expert), 0.0)
