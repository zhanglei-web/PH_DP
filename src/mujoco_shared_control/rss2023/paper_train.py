"""Train the local task with the released RSS 2023 Block Pushing protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.optim import Adam

from mujoco_shared_control.rss2023.model import DiffusionConfig
from mujoco_shared_control.rss2023.paper_dataset import (
    PAPER_ACTION_DIM,
    PAPER_ACTION_NAMES,
    PAPER_OBSERVATION_DIM,
    PAPER_OBSERVATION_NAMES,
    load_paper_dataset,
)
from mujoco_shared_control.rss2023.paper_model import PaperRSS2023Diffusion


def train_paper_model(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    steps: int = 30_000,
    batch_size: int = 4096,
    split_seed: int = 0,
    seed: int = 0,
    device_name: str = "auto",
) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    training, evaluation = load_paper_dataset(dataset_dir, split_seed=split_seed)
    observations = np.concatenate([episode.observation for episode in training])
    actions = np.concatenate([episode.action for episode in training])
    observation_tensor = torch.from_numpy(observations).to(device)
    action_tensor = torch.from_numpy(actions).to(device)
    config = DiffusionConfig(
        observation_dim=PAPER_OBSERVATION_DIM,
        action_dim=PAPER_ACTION_DIM,
        num_diffusion_steps=50,
        beta_schedule="sigmoid",
        beta_min=1e-4,
        beta_max=0.26,
        hidden_dim=128,
    )
    model = PaperRSS2023Diffusion(config).to(device).train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "RSS2023 released Block Pushing training protocol",
        "task_mapping": "3D pose/gripper bounded incremental control",
        "goal_hidden_from_copilot": True,
        "observation_dim": PAPER_OBSERVATION_DIM,
        "action_dim": PAPER_ACTION_DIM,
        "observation_names": list(PAPER_OBSERVATION_NAMES),
        "action_names": list(PAPER_ACTION_NAMES),
        "training_files": [str(episode.path) for episode in training],
        "evaluation_files": [str(episode.path) for episode in evaluation],
        "training_frames": int(len(observations)),
        "training_steps": steps,
        "batch_size": batch_size,
        "split_seed": split_seed,
        "seed": seed,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.monotonic()
    recent: list[float] = []
    for step in range(steps):
        indices = torch.randint(
            len(observation_tensor), (batch_size,), generator=generator, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(observation_tensor[indices], action_tensor[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        recent.append(float(loss.item()))
        if step % 500 == 0 or step == steps - 1:
            print(
                f"step={step} loss={np.mean(recent):.6f} "
                f"elapsed_s={time.monotonic() - started:.1f}", flush=True
            )
            recent.clear()
        if step % 2000 == 0 or step == steps - 1:
            checkpoint = {
                "format_version": "rss2023-paper-1.0",
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "diffusion_config": config.state_dict(),
                "dataset_manifest": manifest,
            }
            destination = output / f"step_{step:08d}.pt"
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(destination)
    final = output / "step_00029999.pt" if steps == 30_000 else output / f"step_{steps - 1:08d}.pt"
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper-protocol RSS2023 model")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    final = train_paper_model(
        args.dataset_dir, args.output_dir, steps=args.steps,
        batch_size=args.batch_size, split_seed=args.split_seed,
        seed=args.seed, device_name=args.device,
    )
    print(f"paper checkpoint: {final}")


if __name__ == "__main__":
    main()
