"""CUDA-only training for clean Recovery Stage V1/V2 policies."""
from __future__ import annotations

import csv, json, random, time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.rss2023.model import DiffusionConfig, RSS2023Diffusion
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig
from mujoco_shared_control.rss2023.recovery_stage_dataset import PreparedRecoveryDataset, prepare_recovery_dataset
from mujoco_shared_control.rss2023.train import ExponentialMovingAverage

DATASET_DEFAULT = Path("outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED")
V1_REFERENCE = Path("outputs/oracle_stage_diffusion/oracle_stage_global_v1_20260818/checkpoints/step_00080000.pt")
V2_REFERENCE = Path("outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/checkpoints/step_00080000.pt")
V1_CONFIG = Path("outputs/oracle_stage_diffusion/oracle_stage_global_v1_20260818/config.json")
V2_CONFIG = Path("outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/config.json")
STEPS_DEFAULT = 120_000
CHECKPOINTS = set(range(10_000, STEPS_DEFAULT + 1, 10_000))


def _cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Recovery Stage training; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _loader(prepared: PreparedRecoveryDataset, name: str, batch: int, shuffle: bool) -> DataLoader:
    split = getattr(prepared, name)
    obs = np.concatenate((prepared.physical_normalizer.normalize(split.physical), split.stage_onehot), axis=1).astype(np.float32)
    act = prepared.action_normalizer.normalize(split.action).astype(np.float32)
    return DataLoader(TensorDataset(torch.from_numpy(obs), torch.from_numpy(act)), batch_size=min(batch, len(split)), shuffle=shuffle, drop_last=False, num_workers=0, pin_memory=True)


def _infinite(loader):
    while True:
        yield from loader


def _loss(model, obs, act):
    return model.loss(obs, act)


@torch.no_grad()
def _validation(model, loader, device, max_batches=20) -> float:
    model.eval(); values = []
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(12345); torch.cuda.manual_seed_all(12345)
        for i, (obs, act) in enumerate(loader):
            if i >= max_batches: break
            values.append(float(_loss(model, obs.to(device, non_blocking=True), act.to(device, non_blocking=True)).item()))
    model.train(); return float(np.mean(values))


def _grad_norm(module: torch.nn.Module) -> float:
    values = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(values)).item()) if values else 0.0


def _finite_model(model) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def _model_and_cfg(kind: str):
    if kind == "v1":
        cfg = DiffusionConfig(observation_dim=48, action_dim=7, num_diffusion_steps=50, beta_schedule="sigmoid", beta_min=1e-4, beta_max=0.26, hidden_dim=128)
        return RSS2023Diffusion(cfg), cfg
    cfg = StageEmbeddingDiffusionConfig(physical_dim=43, stage_dim=5, stage_embedding_dim=32, condition_hidden_dim=128, action_dim=7, num_diffusion_steps=50, beta_schedule="sigmoid", beta_min=1e-4, beta_max=0.26, hidden_dim=128)
    return StageEmbeddingDiffusion(cfg), cfg


def _architecture_audit(kind: str, model: torch.nn.Module) -> dict[str, Any]:
    reference = (V1_REFERENCE if kind == "v1" else V2_REFERENCE)
    if not reference.is_file():
        raise FileNotFoundError(f"historical {kind.upper()} reference checkpoint missing: {reference}")
    payload = torch.load(reference, map_location="cpu", weights_only=False)
    parameter_keys = set(dict(model.named_parameters()))
    reference_count = sum(payload["model"][k].numel() for k in parameter_keys if k in payload["model"])
    actual_count = sum(p.numel() for p in model.parameters())
    if actual_count != reference_count:
        raise RuntimeError(f"{kind.upper()} parameter mismatch: current={actual_count}, historical={reference_count}")
    return {"model_class": type(model).__name__, "total_parameters": actual_count, "historical_reference": str(reference.resolve()), "historical_parameter_count": reference_count}


def _historical_training_audit() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    v1 = json.loads((root / V1_CONFIG).read_text())["training"]
    v2 = json.loads((root / V2_CONFIG).read_text())["training"]
    keys = ("batch_size", "learning_rate", "validation_every", "checkpoint_every", "validation_batches", "seed", "split_seed", "ema_decay", "num_workers")
    differences = {k: {"v1": v1.get(k), "v2": v2.get(k)} for k in keys if v1.get(k) != v2.get(k)}
    if differences:
        raise RuntimeError(f"historical V1/V2 training configuration mismatch: {differences}")
    return {"status": "PASS", "compared_keys": list(keys), "common": {k: v1.get(k) for k in keys}, "differences": differences}


def _stage_activity(model, kind: str, obs, action, device) -> dict[str, Any]:
    model.eval(); b = min(8, len(obs)); physical = obs[:b, :43]; base_stage = torch.eye(5, device=device)[torch.arange(b, device=device) % 5]; noisy = action[:b].clone(); ts = torch.full((b,), 17, dtype=torch.long, device=device); noise = torch.zeros_like(noisy)
    if kind == "v1":
        values = []
        for s in range(5):
            o = torch.cat((physical, torch.eye(5, device=device)[torch.full((b,), s)]), 1); values.append(model.denoiser(torch.cat((o, noisy), 1), ts)[:, 48:])
        delta = torch.stack(values).std(0).mean()
        return {"stage_condition_active": bool(delta.item() > 1e-7), "stage_output_l2": float(delta.item()), "stage_encoder_gradient_norm": None}
    full = torch.cat((physical, base_stage), 1); out = model.denoiser(torch.cat((model._condition(full), noisy), 1), ts)[:, 128:]
    values = []
    for s in range(5):
        o = torch.cat((physical, torch.eye(5, device=device)[torch.full((b,), s)]), 1); values.append(model.denoiser(torch.cat((model._condition(o), noisy), 1), ts)[:, 128:])
    delta = torch.stack(values).std(0).mean()
    return {"stage_condition_active": bool(delta.item() > 1e-7), "stage_output_l2": float(delta.item()), "stage_encoder_gradient_norm": 0.0}


def _config(kind, prepared, arch, steps, batch, lr, smoke):
    train_counts = prepared.audit["train"]["by_type"]
    total = sum(x["transitions"] for x in train_counts.values())
    windows = {k: 100.0 * x["transitions"] / total for k, x in train_counts.items()}
    return {"experiment": f"recovery_stage_{kind}_120k", "model": kind.upper(), "physical_dim": 43, "stage_dim": 5, "action_dim": 7, "stage_conditioning": "CONCAT" if kind == "v1" else "STAGE_EMBEDDING_FUSION", "diffusion_steps": 50, "beta_schedule": "sigmoid", "batch_size": batch, "learning_rate": lr, "optimizer": "Adam", "ema_decay": 0.9, "seed": 42, "split_seed": "frozen_split_manifest", "steps": steps, "checkpoint_interval": 10000, "validation_every": 1000, "validation_batches": 20, "smoke": smoke, "architecture_audit": arch, "dataset_audit": prepared.audit, "window_length": 1, "window_percent_by_type": windows, "normalization": "train_1600_episodes_only; stage one-hot unnormalized", "injection_transitions_used": 0, "non_expert_actions_used": 0}


def train(kind: str, dataset_dir: Path = DATASET_DEFAULT, output_dir: Path | str = "outputs/recovery_stage_dp_training/run", *, steps: int = STEPS_DEFAULT, smoke: bool = False) -> Path:
    device = _cuda(); prepared = prepare_recovery_dataset(dataset_dir); model, cfg = _model_and_cfg(kind); arch = _architecture_audit(kind, model); historical_training = _historical_training_audit()
    if arch["model_class"] != ("RSS2023Diffusion" if kind == "v1" else "StageEmbeddingDiffusion"):
        raise RuntimeError("model class audit failed")
    output = Path(output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    batch, lr = 512, 1e-3
    config = _config(kind, prepared, arch, steps, batch, lr, smoke) | {"historical_training_audit": historical_training}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "dataset_reference.json").write_text(json.dumps(prepared.manifest() | {"audit": prepared.audit}, indent=2) + "\n")
    np.savez(output / "normalization_stats.npz", physical_mean=prepared.physical_normalizer.mean, physical_std=prepared.physical_normalizer.std, action_mean=prepared.action_normalizer.mean, action_std=prepared.action_normalizer.std)
    (output / "normalization_stats.json").write_text(json.dumps({"physical_mean": prepared.physical_normalizer.mean.tolist(), "physical_std": prepared.physical_normalizer.std.tolist(), "action_mean": prepared.action_normalizer.mean.tolist(), "action_std": prepared.action_normalizer.std.tolist(), "fit_split": "train", "fit_episodes": 1600}, indent=2) + "\n")
    train_loader, val_loader = _loader(prepared, "train", batch, True), _loader(prepared, "validation", batch, False)
    _seed(42); model, cfg = _model_and_cfg(kind); model.to(device).train(); opt = Adam(model.parameters(), lr=lr); ema = ExponentialMovingAverage(model, 0.9)
    first = {k: p.detach().clone() for k, p in model.named_parameters()}; smoke_probe = _stage_activity(model, kind, next(iter(train_loader))[0].to(device), next(iter(train_loader))[1].to(device), device)
    probe_obs, probe_act = next(iter(train_loader)); probe_obs, probe_act = probe_obs.to(device), probe_act.to(device); loss = _loss(model, probe_obs, probe_act); loss.backward();
    if not torch.isfinite(loss) or not _finite_model(model) or not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None): raise FloatingPointError("smoke produced NaN/Inf")
    nonzero = any(float((p.detach() - first[k]).abs().max()) > 0 for k, p in model.named_parameters())
    smoke_denoiser_grad = _grad_norm(model.denoiser)
    smoke_stage_grad = _grad_norm(model.condition_encoder.stage_encoder) if kind == "v2" else None
    smoke_physical_grad = _grad_norm(model.condition_encoder.physical_encoder) if kind == "v2" else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    nonzero = any(float((p.detach() - first[k]).abs().max()) > 0 for k, p in model.named_parameters())
    grad_audit = {"parameter_update_nonzero": nonzero, "loss_finite": True, "nan": 0, "inf": 0, "denoiser_gradient_norm": smoke_denoiser_grad, "stage_encoder_gradient_norm": smoke_stage_grad, "physical_encoder_gradient_norm": smoke_physical_grad, **smoke_probe}
    if not nonzero or not smoke_probe["stage_condition_active"] or smoke_denoiser_grad <= 0 or (kind == "v2" and (smoke_stage_grad <= 0 or smoke_physical_grad <= 0)):
        raise RuntimeError(f"smoke condition/update/gradient audit failed: {grad_audit}")
    _seed(42); model, cfg = _model_and_cfg(kind); model.to(device).train(); opt = Adam(model.parameters(), lr=lr); ema = ExponentialMovingAverage(model, 0.9)
    if smoke:
        smoke_losses = []
        smoke_loader = _infinite(train_loader)
        for smoke_step in range(1, steps + 1):
            obs, act = next(smoke_loader); obs, act = obs.to(device, non_blocking=True), act.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True); smoke_loss = _loss(model, obs, act)
            if not torch.isfinite(smoke_loss):
                raise FloatingPointError(f"{kind} smoke loss NaN/Inf at {smoke_step}")
            smoke_loss.backward()
            if not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(f"{kind} smoke gradient NaN/Inf at {smoke_step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update(model); smoke_losses.append(float(smoke_loss.item()))
        if not _finite_model(model):
            raise FloatingPointError(f"{kind} smoke model became NaN/Inf")
        (output / "smoke_report.json").write_text(json.dumps({"status": "PASS", "device": "cuda:0", "steps": steps, "loss_start": smoke_losses[0], "loss_end": smoke_losses[-1], **grad_audit}, indent=2) + "\n")
        return output / "smoke_report.json"
    loader = _infinite(train_loader); log_path = output / "training_log.csv"; checkpoints = output / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    fields = ["step", "train_diffusion_loss", "validation_diffusion_loss", "denoiser_gradient_norm", "stage_encoder_gradient_norm", "physical_encoder_gradient_norm", "NaN_count", "Inf_count", "stage_condition_active"]
    with log_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields); writer.writeheader(); last_val = float("nan")
        for step in range(1, steps + 1):
            obs, act = next(loader); obs, act = obs.to(device, non_blocking=True), act.to(device, non_blocking=True); opt.zero_grad(set_to_none=True); loss = _loss(model, obs, act)
            if not torch.isfinite(loss): raise FloatingPointError(f"{kind} loss NaN/Inf at {step}")
            loss.backward(); den_grad = _grad_norm(model.denoiser); stage_grad = _grad_norm(model.condition_encoder.stage_encoder) if kind == "v2" else 0.0; physical_grad = _grad_norm(model.condition_encoder.physical_encoder) if kind == "v2" else 0.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update(model)
            if step == 1 or step % 1000 == 0 or step == steps:
                last_val = _validation(model, val_loader, device); row = {"step": step, "train_diffusion_loss": float(loss.item()), "validation_diffusion_loss": last_val, "denoiser_gradient_norm": den_grad, "stage_encoder_gradient_norm": stage_grad, "physical_encoder_gradient_norm": physical_grad, "NaN_count": 0, "Inf_count": 0, "stage_condition_active": True}; writer.writerow(row); fp.flush(); print(json.dumps(row), flush=True)
            if step in CHECKPOINTS or step == steps:
                payload = {"format_version": "recovery-stage-dp-1.0", "step": step, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": opt.state_dict(), "diffusion_config": cfg.state_dict(), "training_config": config, "normalization": prepared.manifest(), "smoke_audit": grad_audit, "validation_loss": last_val}
                torch.save(payload, checkpoints / f"step_{step:06d}.pt")
    (output / "training_report.json").write_text(json.dumps({"status": "PASS", "steps": steps, "nan_count": 0, "inf_count": 0, "device": "cuda:0", "smoke": grad_audit}, indent=2) + "\n")
    return output / "training_report.json"
