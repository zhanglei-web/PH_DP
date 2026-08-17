"""Deterministic offline training for the fixed Actor BC v1 baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.actor_bc.model import (
    ACTOR_ACTION_DIM,
    POLICY_STATE_DIM,
    ActorBC,
    parameter_count,
)
from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.experts.interfaces import ExpertActionSpec


RULE_EXPERT_VERSION = "d5ce43ff70af25491c545ec513d56e9f988c4f6b"
SPLIT_SEED = 20260812
ACTION_NORMALIZATION = (
    "xyz / 0.025 m; rotation-vector / 0.10 rad; "
    "gripper = 2 * (width_m / 0.08) - 1; clip [-1, 1]"
)


@dataclass(frozen=True)
class TrainConfig:
    manifest: str = "manifests/rule_expert_v1_formal.json"
    output_root: str = "outputs/actor_bc"
    learning_rate: float = 3e-4
    batch_size: int = 256
    weight_decay: float = 1e-4
    max_epochs: int = 100
    training_seed: int = 20260812
    gradient_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    minimum_lr: float = 1e-6
    early_stopping_patience: int = 12
    normalization_epsilon: float = 1e-6
    num_workers: int = 0


def _seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    deterministic = {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "deterministic_algorithms": True,
        "cuda_available": torch.cuda.is_available(),
    }
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return deterministic


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=project_root, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()

    return {
        "head": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short"),
    }


def _load_arrays(dataset: ManifestActorDataset) -> tuple[np.ndarray, np.ndarray]:
    """Bulk-load only the episodes already authorized by ManifestActorDataset."""
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for entry in dataset.entries:
        with h5py.File(entry.path, "r") as episode:
            states.append(np.asarray(episode["observations/policy_state_42"][:], np.float32))
            actions.append(np.asarray(episode["actions/normalized"][:], np.float32))
    state = np.concatenate(states)
    action = np.concatenate(actions)
    if state.shape != (len(dataset), POLICY_STATE_DIM):
        raise ValueError(f"unexpected policy state shape {state.shape}")
    if action.shape != (len(dataset), ACTOR_ACTION_DIM):
        raise ValueError(f"unexpected normalized action shape {action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("training arrays contain NaN or Inf")
    return state, action


def _normalization(train_state: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_state.astype(np.float64).mean(axis=0)
    raw_std = train_state.astype(np.float64).std(axis=0)
    scale = np.where(raw_std < epsilon, 1.0, raw_std)
    return mean.astype(np.float32), scale.astype(np.float32), raw_std.astype(np.float32)


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    error = prediction - target
    squared = error.square()
    absolute = error.abs()
    predicted_open = prediction[:, 6] >= 0.375
    target_open = target[:, 6] >= 0.375
    xyz_m = torch.linalg.vector_norm(error[:, :3] * 0.025, dim=1)
    return {
        "total_mse": float(squared.mean()),
        "xyz_mse": float(squared[:, :3].mean()),
        "rotation_mse": float(squared[:, 3:6].mean()),
        "gripper_mse": float(squared[:, 6].mean()),
        "total_mae": float(absolute.mean()),
        "per_dimension_mae": absolute.mean(dim=0).tolist(),
        "gripper_accuracy": float((predicted_open == target_open).float().mean()),
        "xyz_error_m": float(xyz_m.mean()),
    }


def _run_epoch(
    model: ActorBC,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for state, target in loader:
        state, target = state.to(device), target.to(device)
        with torch.set_grad_enabled(training):
            prediction = model(state)
            loss = nn.functional.mse_loss(prediction, target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
    return _metrics(torch.cat(predictions), torch.cat(targets))


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint(
    model: ActorBC,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    global_step: int,
    best_validation_mse: float,
    mean: np.ndarray,
    std: np.ndarray,
    config: TrainConfig,
    manifest: dict[str, Any],
    git_state: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": "actor_bc_v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_mse": best_validation_mse,
        "metrics": metrics,
        "policy_state_dim": POLICY_STATE_DIM,
        "policy_state_definition": "policy_state_42_v1",
        "observation_mean": mean,
        "observation_std": std,
        "action_spec": asdict(ExpertActionSpec()),
        "action_normalization_definition": ACTION_NORMALIZATION,
        "manifest_path": str(Path(config.manifest).resolve()),
        "manifest_content_sha": manifest["content_sha256"],
        "split_seed": SPLIT_SEED,
        "training_seed": config.training_seed,
        "rule_expert_frozen_code_version": RULE_EXPERT_VERSION,
        "training_config": asdict(config),
        "git_state": git_state,
        "model_parameter_count": parameter_count(model),
    }


def train(config: TrainConfig, run_id: str | None = None) -> Path:
    project_root = Path.cwd().resolve()
    determinism = _seed_everything(config.training_seed)
    manifest_path = Path(config.manifest).resolve()
    train_dataset = ManifestActorDataset(manifest_path, "train")
    validation_dataset = ManifestActorDataset(manifest_path, "validation")
    if len(train_dataset.entries) != 900 or len(train_dataset) != 115_021:
        raise ValueError("frozen Actor training split does not match 900/115021")
    if len(validation_dataset.entries) != 100 or len(validation_dataset) != 12_817:
        raise ValueError("frozen Actor validation split does not match 100/12817")
    train_ids = {entry.path.name for entry in train_dataset.entries}
    validation_ids = {entry.path.name for entry in validation_dataset.entries}
    if train_ids & validation_ids:
        raise ValueError("episode leakage between Actor train and validation")
    train_state, train_action = _load_arrays(train_dataset)
    validation_state, validation_action = _load_arrays(validation_dataset)
    mean, std, raw_std = _normalization(train_state, config.normalization_epsilon)
    train_state = (train_state - mean) / std
    validation_state = (validation_state - mean) / std
    generator = torch.Generator().manual_seed(config.training_seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_state), torch.from_numpy(train_action)),
        batch_size=config.batch_size, shuffle=True, generator=generator,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_state), torch.from_numpy(validation_action)),
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorBC().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.minimum_lr,
    )
    run_id = run_id or datetime.now(timezone.utc).strftime("actor_bc_v1_%Y%m%dT%H%M%SZ")
    run_dir = Path(config.output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = train_dataset.manifest
    git_state = _git_state(project_root)
    training_record = {
        **asdict(config), "run_id": run_id, "device": str(device),
        "determinism": determinism, "git_state": git_state,
        "model": "42-256-256-256-7 SiLU", "parameter_count": parameter_count(model),
        "train_episodes": len(train_dataset.entries), "train_transitions": len(train_dataset),
        "validation_episodes": len(validation_dataset.entries),
        "validation_transitions": len(validation_dataset),
        "manifest_content_sha": manifest["content_sha256"],
    }
    (run_dir / "training_config.json").write_text(
        json.dumps(training_record, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "normalization.json").write_text(json.dumps({
        "policy_state_definition": "policy_state_42_v1", "epsilon": config.normalization_epsilon,
        "mean": mean.tolist(), "std": std.tolist(), "raw_std": raw_std.tolist(),
        "constant_dimensions": np.flatnonzero(raw_std < config.normalization_epsilon).tolist(),
        "fitted_from": "900 manifest train nominal_success episodes only",
        "action_spec": asdict(ExpertActionSpec()),
        "action_normalization_definition": ACTION_NORMALIZATION,
    }, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "epoch", "global_step", "learning_rate",
        *[f"train_{name}" for name in ("total_mse", "xyz_mse", "rotation_mse", "gripper_mse", "total_mae", "gripper_accuracy", "xyz_error_m")],
        *[f"train_mae_{index}" for index in range(7)],
        *[f"val_{name}" for name in ("total_mse", "xyz_mse", "rotation_mse", "gripper_mse", "total_mae", "gripper_accuracy", "xyz_error_m")],
        *[f"val_mae_{index}" for index in range(7)],
    ]
    best = math.inf
    best_epoch = 0
    stale = 0
    global_step = 0
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, config.max_epochs + 1):
            train_metrics = _run_epoch(model, train_loader, device, optimizer, config.gradient_clip_norm)
            validation_metrics = _run_epoch(model, validation_loader, device, None, config.gradient_clip_norm)
            global_step += len(train_loader)
            scheduler.step(validation_metrics["total_mse"])
            row: dict[str, Any] = {
                "epoch": epoch, "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            for prefix, values in (("train", train_metrics), ("val", validation_metrics)):
                for name in ("total_mse", "xyz_mse", "rotation_mse", "gripper_mse", "total_mae", "gripper_accuracy", "xyz_error_m"):
                    row[f"{prefix}_{name}"] = values[name]
                for index, value in enumerate(values["per_dimension_mae"]):
                    row[f"{prefix}_mae_{index}"] = value
            writer.writerow(row)
            stream.flush()
            improved = validation_metrics["total_mse"] < best
            if improved:
                best = validation_metrics["total_mse"]
                best_epoch = epoch
                stale = 0
                _atomic_torch_save(_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, best, mean, std,
                    config, manifest, git_state, {"train": train_metrics, "validation": validation_metrics},
                ), run_dir / "checkpoint_best.pt")
            else:
                stale += 1
            print(
                f"epoch={epoch:03d} train_mse={train_metrics['total_mse']:.8g} "
                f"val_mse={validation_metrics['total_mse']:.8g} "
                f"val_grip_acc={validation_metrics['gripper_accuracy']:.5f} "
                f"lr={optimizer.param_groups[0]['lr']:.3g} best={best_epoch}", flush=True,
            )
            if stale >= config.early_stopping_patience:
                break
    _atomic_torch_save(_checkpoint(
        model, optimizer, scheduler, epoch, global_step, best, mean, std,
        config, manifest, git_state, {"train": train_metrics, "validation": validation_metrics,
                                      "best_epoch": best_epoch, "early_stopping": epoch < config.max_epochs},
    ), run_dir / "checkpoint_last.pt")
    print(f"run_dir={run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(TrainConfig.manifest))
    parser.add_argument("--output-root", type=Path, default=Path(TrainConfig.output_root))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    train(TrainConfig(manifest=str(args.manifest), output_root=str(args.output_root)), args.run_id)


if __name__ == "__main__":
    main()
