"""Train the 43D->7D Global Diffusion baseline with the project RSS2023 DDPM."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.rss2023.global_dataset import (
    GLOBAL_ACTION_DIM,
    GLOBAL_OBSERVATION_DIM,
    PreparedGlobalDataset,
    prepare_global_dataset,
)
from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion
from mujoco_shared_control.rss2023.train import ExponentialMovingAverage, TrainingConfig


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(prepared: PreparedGlobalDataset, split_name: str, batch_size: int, shuffle: bool):
    split = getattr(prepared, split_name)
    observations = torch.from_numpy(prepared.observation_normalizer.normalize(split.observation))
    actions = torch.from_numpy(prepared.action_normalizer.normalize(split.action))
    return DataLoader(
        TensorDataset(observations, actions), batch_size=batch_size, shuffle=shuffle,
        drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available(),
    )


def _infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def _validation_loss(model, loader, device, max_batches: int = 20) -> float:
    model.eval()
    losses = []
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(12345)
        for index, (observation, action) in enumerate(loader):
            if index >= max_batches:
                break
            losses.append(float(model.loss(observation.to(device), action.to(device)).item()))
    model.train()
    return float(np.mean(losses))


def _payload(model, optimizer, ema, prepared, diffusion_config, training_config, step, value):
    return {
        "format_version": "global-rss2023-1.0",
        "step": step,
        "validation_loss": value,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "diffusion_config": diffusion_config.state_dict(),
        "training_config": asdict(training_config),
        "observation_normalizer": prepared.observation_normalizer.state_dict(),
        "action_normalizer": prepared.action_normalizer.state_dict(),
        "dataset_manifest": prepared.manifest(),
    }


def _save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sample_global(model, observation, *, generator=None):
    """Use the released full-strength assist sampler as p(a|s)."""
    zero_action = torch.zeros(
        (observation.shape[0], GLOBAL_ACTION_DIM), dtype=observation.dtype,
        device=observation.device,
    )
    return model.assist(observation, zero_action, gamma=1.0, generator=generator)


def _run_smoke(prepared, output: Path, config, device, steps: int, batch_size: int) -> dict[str, Any]:
    _seed(7001)
    model = RSS2023Diffusion(config).to(device).train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    batches = _infinite(_loader(prepared, "train", batch_size, True))
    last_loss = float("nan")
    for _ in range(steps):
        observation, action = next(batches)
        observation, action = observation.to(device), action.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(observation, action)
        if not torch.isfinite(loss):
            raise FloatingPointError("smoke loss is NaN/Inf")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    path = output / "smoke" / "smoke.pt"
    _save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config.state_dict()}, path)
    restored = RSS2023Diffusion(config).to(device)
    restored.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    validation = prepared.observation_normalizer.normalize(prepared.validation.observation[:4])
    sample = _sample_global(
        restored.eval(), torch.from_numpy(validation).to(device),
        generator=torch.Generator(device=device).manual_seed(7002),
    )
    passed = sample.shape == (4, GLOBAL_ACTION_DIM) and bool(torch.isfinite(sample).all())
    report = {
        "status": "PASS" if passed else "FAIL", "steps": steps,
        "last_loss": last_loss, "sample_shape": list(sample.shape),
        "sample_nan": int(torch.isnan(sample).sum()), "sample_inf": int(torch.isinf(sample).sum()),
        "checkpoint_reload": True,
    }
    (output / "smoke").mkdir(parents=True, exist_ok=True)
    (output / "smoke" / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Global Diffusion smoke test failed")
    return report


def train_global(dataset_dir: Path, output: Path, *, device_name: str, smoke_steps: int,
                 training_config: TrainingConfig, diffusion_config: DiffusionConfig) -> Path:
    prepared = prepare_global_dataset(dataset_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = prepared.manifest()
    (output / "dataset_adapter_report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez(
        output / "normalization_stats.npz",
        observation_mean=prepared.observation_normalizer.mean,
        observation_std=prepared.observation_normalizer.std,
        action_mean=prepared.action_normalizer.mean,
        action_std=prepared.action_normalizer.std,
    )
    config_report = {
        "implementation": "mujoco_shared_control.rss2023.model.RSS2023Diffusion",
        "protocol": "conditional vector DDPM / Diffusha RSS2023",
        "horizon": 1,
        "diffusion": diffusion_config.state_dict(),
        "training": asdict(training_config),
        "test_split_used_for_selection": False,
    }
    (output / "training_config.json").write_text(json.dumps(config_report, indent=2) + "\n")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else
                          "cpu" if device_name == "auto" else device_name)
    _run_smoke(prepared, output, diffusion_config, device, smoke_steps, training_config.batch_size)

    _seed(training_config.seed)
    train_loader = _loader(prepared, "train", training_config.batch_size, True)
    validation_loader = _loader(prepared, "validation", training_config.batch_size, False)
    batches = _infinite(train_loader)
    model = RSS2023Diffusion(diffusion_config).to(device).train()
    optimizer = Adam(model.parameters(), lr=training_config.learning_rate)
    ema = ExponentialMovingAverage(model, training_config.ema_decay)
    best = float("inf")
    final_loss = float("nan")
    train_window: list[float] = []
    log_path = output / "training_log.jsonl"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, training_config.steps + 1):
            observation, action = next(batches)
            observation, action = observation.to(device), action.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(observation, action)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"training loss is NaN/Inf at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            train_window.append(float(loss.item()))
            validate = step == 1 or step % training_config.validation_every == 0 or step == training_config.steps
            if validate:
                final_loss = _validation_loss(model, validation_loader, device, training_config.validation_batches)
                record = {"step": step, "train_loss": float(np.mean(train_window)),
                          "validation_loss": final_loss, "elapsed_seconds": time.monotonic() - started}
                log.write(json.dumps(record) + "\n"); log.flush()
                print(json.dumps(record), flush=True)
                train_window.clear()
                payload = _payload(model, optimizer, ema, prepared, diffusion_config,
                                   training_config, step, final_loss)
                _save(payload, output / "latest.pt")
                if final_loss < best:
                    best = final_loss
                    _save(payload, output / "best.pt")
            if step % training_config.checkpoint_every == 0:
                _save(_payload(model, optimizer, ema, prepared, diffusion_config,
                               training_config, step, final_loss), output / "checkpoints" / f"step_{step:08d}.pt")
    summary = {"status": "PASS", "best_validation_loss": best,
               "final_validation_loss": final_loss, "steps": training_config.steps,
               "nan_inf": 0, "device": str(device)}
    (output / "training_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-steps", type=int, default=500)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    args = parser.parse_args()
    training = TrainingConfig(
        steps=args.steps, validation_every=args.validation_every,
        checkpoint_every=args.checkpoint_every,
    )
    diffusion = DiffusionConfig(observation_dim=GLOBAL_OBSERVATION_DIM, action_dim=GLOBAL_ACTION_DIM)
    best = train_global(args.dataset_dir, args.output_dir, device_name=args.device,
                        smoke_steps=args.smoke_steps, training_config=training,
                        diffusion_config=diffusion)
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
