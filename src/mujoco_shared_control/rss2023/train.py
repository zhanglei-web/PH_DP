"""Training entry point for the 29-D observation + 8-D command RSS 2023 model."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.rss2023.dataset import (
    ACTION_DIM,
    OBSERVATION_DIM,
    DataSplit,
    PreparedDataset,
    prepare_dataset,
)
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 30_000
    batch_size: int = 512
    learning_rate: float = 1e-3
    validation_every: int = 500
    checkpoint_every: int = 5_000
    validation_batches: int = 20
    seed: int = 42
    split_seed: int = 42
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    ema_decay: float = 0.9
    num_workers: int = 0


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        for name, value in current.items():
            if self.shadow[name].is_floating_point():
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[name].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}


def _loader(
    split: DataSplit,
    prepared: PreparedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    observation = torch.from_numpy(
        prepared.observation_normalizer.normalize(split.observation)
    )
    action = torch.from_numpy(prepared.action_normalizer.normalize(split.action))
    return DataLoader(
        TensorDataset(observation, action),
        batch_size=min(batch_size, len(split)),
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _infinite(loader: DataLoader[tuple[Tensor, Tensor]]):
    while True:
        yield from loader


@torch.no_grad()
def evaluate_loss(
    model: RSS2023Diffusion,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
    max_batches: int,
) -> float:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device() if device.index is None else device.index
        ]
    model.eval()
    losses: list[float] = []
    # Reuse the same validation timesteps and noise at every evaluation so that
    # best.pt is selected by model quality rather than validation RNG luck.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(12_345)
        for batch_index, (observation, action) in enumerate(loader):
            if batch_index >= max_batches:
                break
            loss = model.loss(observation.to(device), action.to(device))
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def _checkpoint_payload(
    model: RSS2023Diffusion,
    optimizer: Adam,
    ema: ExponentialMovingAverage,
    prepared: PreparedDataset,
    diffusion_config: DiffusionConfig,
    training_config: TrainingConfig,
    *,
    step: int,
    validation_loss: float,
) -> dict[str, Any]:
    return {
        "format_version": "1.0.0",
        "step": step,
        "validation_loss": validation_loss,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "diffusion_config": diffusion_config.state_dict(),
        "training_config": asdict(training_config),
        "observation_normalizer": prepared.observation_normalizer.state_dict(),
        "action_normalizer": prepared.action_normalizer.state_dict(),
        "dataset_manifest": prepared.manifest(),
    }


def _save_checkpoint(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def train(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    training_config: TrainingConfig = TrainingConfig(),
    diffusion_config: DiffusionConfig = DiffusionConfig(),
    device_name: str = "auto",
) -> Path:
    if training_config.steps < 1:
        raise ValueError("steps must be positive")
    if training_config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if training_config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if training_config.validation_every < 1:
        raise ValueError("validation_every must be positive")
    if training_config.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if training_config.validation_batches < 1:
        raise ValueError("validation_batches must be positive")
    if diffusion_config.observation_dim != OBSERVATION_DIM:
        raise ValueError(f"observation_dim must be {OBSERVATION_DIM}")
    if diffusion_config.action_dim != ACTION_DIM:
        raise ValueError(f"action_dim must be {ACTION_DIM}")

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    prepared = prepare_dataset(
        dataset_dir,
        split_seed=training_config.split_seed,
        validation_fraction=training_config.validation_fraction,
        test_fraction=training_config.test_fraction,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = prepared.manifest()
    manifest.update(
        {
            "original_frames": sum(
                episode.original_frames for episode in prepared.episode_summaries
            ),
            "retained_frames": sum(
                episode.retained_frames for episode in prepared.episode_summaries
            ),
        }
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    train_loader = _loader(
        prepared.train,
        prepared,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
    )
    validation_loader = _loader(
        prepared.validation,
        prepared,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    batches = _infinite(train_loader)
    model = RSS2023Diffusion(diffusion_config).to(device)
    optimizer = Adam(model.parameters(), lr=training_config.learning_rate)
    ema = ExponentialMovingAverage(model, training_config.ema_decay)
    best_loss = float("inf")
    last_validation_loss = float("nan")
    recent_losses: list[float] = []
    started = time.monotonic()

    print(
        f"device={device} train_frames={len(prepared.train)} "
        f"validation_frames={len(prepared.validation)} test_frames={len(prepared.test)}"
    )
    model.train()
    for step in range(1, training_config.steps + 1):
        observation, action = next(batches)
        observation = observation.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(observation, action)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema.update(model)
        recent_losses.append(float(loss.item()))

        validate = step == 1 or step % training_config.validation_every == 0
        if validate or step == training_config.steps:
            last_validation_loss = evaluate_loss(
                model,
                validation_loader,
                device=device,
                max_batches=training_config.validation_batches,
            )
            elapsed = time.monotonic() - started
            print(
                f"step={step} train_loss={np.mean(recent_losses):.6f} "
                f"validation_loss={last_validation_loss:.6f} elapsed_s={elapsed:.1f}"
            )
            recent_losses.clear()
            if last_validation_loss < best_loss:
                best_loss = last_validation_loss
                _save_checkpoint(
                    _checkpoint_payload(
                        model,
                        optimizer,
                        ema,
                        prepared,
                        diffusion_config,
                        training_config,
                        step=step,
                        validation_loss=last_validation_loss,
                    ),
                    output / "best.pt",
                )

        if step % training_config.checkpoint_every == 0:
            _save_checkpoint(
                _checkpoint_payload(
                    model,
                    optimizer,
                    ema,
                    prepared,
                    diffusion_config,
                    training_config,
                    step=step,
                    validation_loss=last_validation_loss,
                ),
                output / f"step_{step:08d}.pt",
            )

    final_path = output / "final.pt"
    _save_checkpoint(
        _checkpoint_payload(
            model,
            optimizer,
            ema,
            prepared,
            diffusion_config,
            training_config,
            step=training_config.steps,
            validation_loss=last_validation_loss,
        ),
        final_path,
    )
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the 37-D RSS 2023 diffusion model from HDF5 episodes."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-min", type=float, default=1e-4)
    parser.add_argument("--beta-max", type=float, default=0.26)
    parser.add_argument("--hidden-dim", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    training_config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_every=args.validation_every,
        checkpoint_every=args.checkpoint_every,
        validation_batches=args.validation_batches,
        seed=args.seed,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
    )
    diffusion_config = DiffusionConfig(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        num_diffusion_steps=args.diffusion_steps,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        hidden_dim=args.hidden_dim,
    )
    final_path = train(
        args.dataset_dir,
        args.output_dir,
        training_config=training_config,
        diffusion_config=diffusion_config,
        device_name=args.device,
    )
    print(f"saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
