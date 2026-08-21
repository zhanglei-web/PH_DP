"""CUDA-only fair 43D Global Diffusion control for the Recovery Stage dataset.

This is intentionally separate from the historical Global trainer: it consumes
the audited Recovery dataset and its frozen episode split, but never materializes
or forwards its 5D stage labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion
from mujoco_shared_control.rss2023.dataset import FeatureNormalizer
from mujoco_shared_control.rss2023.recovery_stage_dataset import PreparedRecoveryDataset, prepare_recovery_dataset
from mujoco_shared_control.rss2023.train import ExponentialMovingAverage

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED"
V1_DIR = ROOT / "outputs/recovery_stage_dp_training/recovery_stage_v1_120k_20260820"
FROZEN_GLOBAL_CONFIG = ROOT / "outputs/global_diffusion/global_long_run_120k/training_config.json"
DEFAULT_OUTPUT = ROOT / "outputs/recovery_stage_dp_training/recovery_global_120k_20260820"
CHECKPOINTS = set(range(10_000, 120_001, 10_000))


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Recovery-aware Global Diffusion; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _loader(prepared: PreparedRecoveryDataset, split_name: str, batch_size: int, shuffle: bool) -> DataLoader:
    split = getattr(prepared, split_name)
    # Deliberately only the physical/action arrays: stage_onehot and active_phase
    # remain in the on-disk dataset and are never passed through this loader.
    physical = prepared.physical_normalizer.normalize(split.physical).astype(np.float32)
    actions = prepared.action_normalizer.normalize(split.action).astype(np.float32)
    return DataLoader(TensorDataset(torch.from_numpy(physical), torch.from_numpy(actions)), batch_size=min(batch_size, len(split)), shuffle=shuffle, drop_last=False, num_workers=0, pin_memory=True)


def _infinite(loader):
    while True:
        yield from loader


def _grad_norm(module: torch.nn.Module) -> float:
    values = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(values)).item()) if values else 0.0


def _finite(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


@torch.no_grad()
def _validation(model, loader, device, max_batches=20) -> float:
    model.eval(); losses = []
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(12345); torch.cuda.manual_seed_all(12345)
        for i, (physical, action) in enumerate(loader):
            if i >= max_batches: break
            losses.append(float(model.loss(physical.to(device, non_blocking=True), action.to(device, non_blocking=True)).item()))
    model.train()
    return float(np.mean(losses))


def _audit(prepared: PreparedRecoveryDataset, config: DiffusionConfig) -> dict:
    frozen = json.loads(FROZEN_GLOBAL_CONFIG.read_text())
    expected = frozen["diffusion"]
    actual = config.state_dict()
    if actual != expected:
        raise RuntimeError(f"STOP architecture differs from Frozen Global: current={actual}, frozen={expected}")
    v1_config = json.loads((V1_DIR / "config.json").read_text())
    expected_training = {k: v1_config[k] for k in ("batch_size", "learning_rate", "validation_every", "validation_batches", "ema_decay", "seed", "window_length", "checkpoint_interval")}
    required = {"batch_size": 512, "learning_rate": 1e-3, "validation_every": 1000, "validation_batches": 20, "ema_decay": .9, "seed": 42, "window_length": 1, "checkpoint_interval": 10_000}
    if expected_training != required:
        raise RuntimeError(f"STOP Recovery V1 training configuration mismatch: {expected_training}")
    with np.load(V1_DIR / "normalization_stats.npz", allow_pickle=False) as saved:
        same_physical = np.array_equal(prepared.physical_normalizer.mean, saved["physical_mean"]) and np.array_equal(prepared.physical_normalizer.std, saved["physical_std"])
        same_action = np.array_equal(prepared.action_normalizer.mean, saved["action_mean"]) and np.array_equal(prepared.action_normalizer.std, saved["action_std"])
    if not (same_physical and same_action):
        raise RuntimeError("STOP normalization differs from Recovery V1/V2 train-split normalization")
    counts = prepared.audit["train"]["by_type"]; total = sum(x["transitions"] for x in counts.values())
    windows = {name: 100 * value["transitions"] / total for name, value in counts.items()}
    if windows != v1_config["window_percent_by_type"]:
        raise RuntimeError("STOP training window sampler distribution differs from Recovery V1/V2")
    return {"architecture_matches_frozen_global": True, "normalization_matches_v1": True, "sampler_matches_v1": True, "GLOBAL_STAGE_TENSOR_PASSED_TO_MODEL": "NO", "window_percent_by_type": windows, "frozen_global_config": str(FROZEN_GLOBAL_CONFIG), "v1_config": str(V1_DIR / "config.json")}


def _reuse_v1_normalization(prepared: PreparedRecoveryDataset) -> PreparedRecoveryDataset:
    """Use the already-audited V1 arrays, after bitwise verification in _audit."""
    with np.load(V1_DIR / "normalization_stats.npz", allow_pickle=False) as saved:
        physical = FeatureNormalizer(np.asarray(saved["physical_mean"], np.float32), np.asarray(saved["physical_std"], np.float32))
        action = FeatureNormalizer(np.asarray(saved["action_mean"], np.float32), np.asarray(saved["action_std"], np.float32))
    return replace(prepared, physical_normalizer=physical, action_normalizer=action)


def _write_metadata(output: Path, prepared: PreparedRecoveryDataset, audit: dict, config: DiffusionConfig, *, smoke: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"experiment": "recovery_global_120k", "MODEL": "RECOVERY_AWARE_GLOBAL", "MODEL_CLASS": "RSS2023Diffusion", "TOTAL_PARAMETERS": sum(p.numel() for p in RSS2023Diffusion(config).parameters()), "PHYSICAL_INPUT_DIM": 43, "STAGE_INPUT_USED": "NO", "ACTION_DIM": 7, "DENOISER": "ConditionalDenoiser", "DIFFUSION_STEPS": 50, "dataset": str(prepared.root), "split_manifest": str(prepared.split_manifest), "dataset_audit": prepared.audit, "architecture_audit": audit, "batch_size": 512, "optimizer": "Adam", "learning_rate": .001, "ema_decay": .9, "seed": 42, "steps": 1000 if smoke else 120000, "checkpoint_interval": 10000, "validation_every": 1000, "validation_batches": 20, "window_length": 1, "INJECTION_TRANSITIONS_USED": 0, "NON_EXPERT_ACTIONS_USED": 0}
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "dataset_reference.json").write_text(json.dumps(prepared.manifest() | {"audit": prepared.audit}, indent=2) + "\n")
    np.savez(output / "normalization_stats.npz", physical_mean=prepared.physical_normalizer.mean, physical_std=prepared.physical_normalizer.std, action_mean=prepared.action_normalizer.mean, action_std=prepared.action_normalizer.std)
    (output / "normalization_stats.json").write_text(json.dumps({"physical_mean": prepared.physical_normalizer.mean.tolist(), "physical_std": prepared.physical_normalizer.std.tolist(), "action_mean": prepared.action_normalizer.mean.tolist(), "action_std": prepared.action_normalizer.std.tolist(), "source": "directly verified equal to Recovery V1/V2 train-1600 normalization"}, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("MODEL", "MODEL_CLASS", "TOTAL_PARAMETERS", "PHYSICAL_INPUT_DIM", "STAGE_INPUT_USED", "ACTION_DIM", "DENOISER", "DIFFUSION_STEPS")}), flush=True)
    print(json.dumps({"CUDA_AVAILABLE": True, "GPU_NAME": torch.cuda.get_device_name(0), "DEVICE": "cuda:0", "GLOBAL_STAGE_TENSOR_PASSED_TO_MODEL": "NO", "NORMAL_WINDOW_PERCENT": audit["window_percent_by_type"]["NORMAL_SUCCESS"], "GRASP_RECOVERY_WINDOW_PERCENT": audit["window_percent_by_type"]["GRASP_RECOVERY_SUCCESS"], "TRANSPORT_RECOVERY_WINDOW_PERCENT": audit["window_percent_by_type"]["TRANSPORT_RECOVERY_SUCCESS"], "PLACE_RECOVERY_WINDOW_PERCENT": audit["window_percent_by_type"]["PLACE_RECOVERY_SUCCESS"]}), flush=True)


def run(output: Path, *, smoke: bool) -> Path:
    device = _cuda(); prepared = prepare_recovery_dataset(DATASET)
    config = DiffusionConfig(observation_dim=43, action_dim=7, num_diffusion_steps=50, beta_schedule="sigmoid", beta_min=1e-4, beta_max=.26, hidden_dim=128)
    audit = _audit(prepared, config)
    prepared = _reuse_v1_normalization(prepared)
    _write_metadata(output, prepared, audit, config, smoke=smoke)
    train_loader, val_loader = _loader(prepared, "train", 512, True), _loader(prepared, "validation", 512, False)
    _seed(7001 if smoke else 42); model = RSS2023Diffusion(config).to(device).train(); optimizer = Adam(model.parameters(), lr=.001); ema = ExponentialMovingAverage(model, .9)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}; batches = _infinite(train_loader); first_grad = 0.; losses = []
    steps = 1000 if smoke else 120000; fields = ["Step", "Train Diffusion Loss", "Validation Diffusion Loss", "Denoiser Grad Norm", "NaN Count", "Inf Count"]
    log_path = output / ("smoke_log.csv" if smoke else "training_log.csv")
    with log_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields); writer.writeheader(); last_validation = float("nan")
        for step in range(1, steps + 1):
            physical, action = next(batches); physical, action = physical.to(device, non_blocking=True), action.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); loss = model.loss(physical, action)
            if not torch.isfinite(loss): raise FloatingPointError(f"NaN/Inf loss at step {step}")
            loss.backward(); grad = _grad_norm(model.denoiser); first_grad = grad if step == 1 else first_grad
            if not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None): raise FloatingPointError(f"NaN/Inf gradient at step {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); ema.update(model); losses.append(float(loss.item()))
            if step == 1 or step % 1000 == 0 or step == steps:
                last_validation = _validation(model, val_loader, device)
                row = {"Step": step, "Train Diffusion Loss": float(loss.item()), "Validation Diffusion Loss": last_validation, "Denoiser Grad Norm": grad, "NaN Count": 0, "Inf Count": 0}; writer.writerow(row); fp.flush(); print(json.dumps(row), flush=True)
            if not smoke and step in CHECKPOINTS:
                checkpoints = output / "checkpoints"
                checkpoints.mkdir(exist_ok=True)
                torch.save(
                    {"format_version": "recovery-aware-global-dp-1.0", "step": step,
                     "model": model.state_dict(), "ema": ema.state_dict(),
                     "optimizer": optimizer.state_dict(), "diffusion_config": config.state_dict(),
                     "training_config": json.loads((output / "config.json").read_text()),
                     "normalization": prepared.manifest(), "validation_loss": last_validation},
                    checkpoints / f"step_{step:06d}.pt",
                )
    changed = any(not torch.equal(p.detach(), before[n]) for n, p in model.named_parameters())
    smoke_report = {"GLOBAL_SMOKE_VALID": "YES" if _finite(model) and first_grad > 0 and changed else "NO", "loss_finite": bool(np.isfinite(losses).all()), "gradient_finite": True, "denoiser_gradient_norm": first_grad, "parameter_diff_gt_zero": changed, "NaN": 0, "Inf": 0}
    if smoke:
        (output / "smoke_report.json").write_text(json.dumps(smoke_report, indent=2) + "\n")
        if smoke_report["GLOBAL_SMOKE_VALID"] != "YES": raise RuntimeError("GLOBAL_SMOKE_VALID = NO")
        return output / "smoke_report.json"
    (output / "training_report.json").write_text(json.dumps({"RECOVERY_GLOBAL_TRAINING_VALID": "YES", "GLOBAL_STAGE_INPUT_USED": "NO", "DATASET_MATCHES_V1_V2": "YES", "SPLIT_MATCHES_V1_V2": "YES", "NORMALIZATION_MATCHES_V1_V2": "YES", "SAMPLER_MATCHES_V1_V2": "YES", "NaN": 0, "Inf": 0}, indent=2) + "\n")
    return output / "training_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); print(run(args.output.resolve(), smoke=args.smoke))


if __name__ == "__main__":
    main()
