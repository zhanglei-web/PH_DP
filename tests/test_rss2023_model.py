from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from mujoco_shared_control.rss2023.model import (  # noqa: E402
    DiffusionConfig,
    RSS2023Diffusion,
)
from mujoco_shared_control.rss2023.inference import RSS2023Predictor  # noqa: E402


def test_diffusion_loss_and_assistance_shapes() -> None:
    config = DiffusionConfig(num_diffusion_steps=5, hidden_dim=16)
    model = RSS2023Diffusion(config)
    observation = torch.randn(4, config.observation_dim)
    action = torch.randn(4, config.action_dim)

    loss = model.loss(observation, action)
    loss.backward()
    assisted = model.assist(observation, action, gamma=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert assisted.shape == action.shape
    assert torch.isfinite(assisted).all()


def test_zero_gamma_returns_the_human_action() -> None:
    config = DiffusionConfig(num_diffusion_steps=5, hidden_dim=16)
    model = RSS2023Diffusion(config)
    observation = torch.randn(config.observation_dim)
    action = torch.randn(config.action_dim)

    assisted = model.assist(observation, action, gamma=0.0)

    torch.testing.assert_close(assisted, action)


def test_predictor_loads_checkpoint_and_returns_safe_command(tmp_path) -> None:
    config = DiffusionConfig(num_diffusion_steps=5, hidden_dim=16)
    model = RSS2023Diffusion(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "diffusion_config": config.state_dict(),
            "model": model.state_dict(),
            "ema": {"decay": 0.9, "shadow": model.state_dict()},
            "observation_normalizer": {
                "mean": torch.zeros(config.observation_dim).numpy(),
                "std": torch.ones(config.observation_dim).numpy(),
            },
            "action_normalizer": {
                "mean": torch.zeros(config.action_dim).numpy(),
                "std": torch.ones(config.action_dim).numpy(),
            },
        },
        checkpoint,
    )
    predictor = RSS2023Predictor.from_checkpoint(checkpoint, device_name="cpu")
    observation = torch.zeros(config.observation_dim).numpy()
    human_action = torch.tensor(
        [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.08]
    ).numpy()

    assisted = predictor.predict(observation, human_action, gamma=0.0)

    assert assisted.shape == (config.action_dim,)
    assert assisted[7] == pytest.approx(0.08)
    assert float((assisted[3:7] ** 2).sum()) == pytest.approx(1.0)

    batch = predictor.predict_batch(
        observation[None, :].repeat(3, axis=0),
        human_action[None, :].repeat(3, axis=0),
        gamma=0.0,
    )
    assert batch.shape == (3, config.action_dim)
