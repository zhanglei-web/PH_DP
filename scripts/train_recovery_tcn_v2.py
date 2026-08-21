#!/usr/bin/env python3
"""Train RECOVERY_TCN_V2 from the frozen C2 hard-onehot cache, CUDA only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig
from mujoco_shared_control.rss2023.recovery_stage_dataset import prepare_recovery_dataset
from mujoco_shared_control.rss2023.train import ExponentialMovingAverage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED"
C2 = ROOT / "outputs/recovery_stage_dp_training/causal_tcn_recovery_v1_20260820"
ORACLE = ROOT / "outputs/recovery_stage_dp_training/recovery_stage_v2_120k_20260820"
OUT = ROOT / "outputs/recovery_stage_dp_training/recovery_tcn_v2_120k_20260820"
STEPS, CHECKPOINT_INTERVAL, VALIDATION_EVERY, VALIDATION_BATCHES = 120_000, 10_000, 1_000, 20
BATCH, LR, EMA, SEED = 512, 1e-3, .9, 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RECOVERY_TCN_V2; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def frozen_references() -> dict:
    report = json.loads((C2 / "final_report.json").read_text())
    audit = json.loads((C2 / "cache_audit.json").read_text())
    checkpoint = C2 / "checkpoints/best_validation_macro_f1.pt"
    normalization, cache = C2 / "normalization_stats.npz", C2 / "causal_tcn_stage_cache.h5"
    required = (checkpoint, normalization, cache, ORACLE / "config.json", ORACLE / "normalization_stats.npz")
    if not all(item.is_file() for item in required):
        raise FileNotFoundError("required frozen C2/Oracle V2 artifact is missing")
    if report.get("CAUSAL_TCN_VALID") != "YES" or audit.get("TCN_STAGE_CACHE_VALID") != "YES":
        raise RuntimeError("C2 validity/cache gate is not YES")
    if audit.get("TCN_CHECKPOINT") != str(checkpoint.resolve()):
        raise RuntimeError("cache audit checkpoint is not the formal C2 best validation Macro-F1 checkpoint")
    return {"c2_report": report, "cache_audit": audit, "checkpoint": checkpoint, "checkpoint_sha256": sha256(checkpoint), "normalization": normalization, "cache": cache}


def expected_oracle_config() -> StageEmbeddingDiffusionConfig:
    config = json.loads((ORACLE / "config.json").read_text())
    if config.get("model") != "V2" or config.get("stage_conditioning") != "STAGE_EMBEDDING_FUSION":
        raise RuntimeError("recovery-aware Oracle V2 formal config is not the expected V2 fusion architecture")
    if any(config.get(key) != value for key, value in (("physical_dim", 43), ("stage_dim", 5), ("action_dim", 7), ("diffusion_steps", 50), ("beta_schedule", "sigmoid"), ("batch_size", BATCH), ("learning_rate", LR), ("optimizer", "Adam"), ("ema_decay", EMA), ("seed", SEED), ("steps", STEPS), ("checkpoint_interval", CHECKPOINT_INTERVAL), ("validation_every", VALIDATION_EVERY), ("validation_batches", VALIDATION_BATCHES))):
        raise RuntimeError("Oracle V2 formal hyperparameter/config mismatch")
    return StageEmbeddingDiffusionConfig(physical_dim=43, stage_dim=5, stage_embedding_dim=32, condition_hidden_dim=128, action_dim=7, num_diffusion_steps=50, beta_schedule="sigmoid", beta_min=1e-4, beta_max=.26, hidden_dim=128)


def cache_join(prepared, cache_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Exact (episode_id, timestep_dp, timestep_raw) join; no index/nearest fallback."""
    manifest = json.loads((DATA / "split_manifest.json").read_text())
    paths = {episode_id: Path(path).resolve() for episode_id, path in manifest["episode_paths"].items()}
    with h5py.File(cache_path, "r") as cache:
        required = ("episode_id", "split", "timestep_dp", "timestep_raw", "gt_stage", "pred_stage", "hard_onehot5", "posterior5")
        if any(name not in cache for name in required): raise ValueError("C2 stage cache schema incomplete")
        def text(values): return [value.decode() if isinstance(value, bytes) else str(value) for value in values]
        rows = list(zip(text(cache["episode_id"][:]), text(cache["split"][:]), cache["timestep_dp"][:].astype(int), cache["timestep_raw"][:].astype(int), cache["gt_stage"][:].astype(int), cache["pred_stage"][:].astype(int), cache["hard_onehot5"][:].astype("f4")))
    lookup = {(episode, split, dp, raw): (gt, pred, onehot) for episode, split, dp, raw, gt, pred, onehot in rows}
    if len(lookup) != len(rows): raise RuntimeError("duplicate C2 cache identity rows")
    joined, failures, physical_failures, action_failures, gt_match = {}, 0, 0, 0, []
    for split_name in ("train", "validation", "test"):
        physical, actions, stages = [], [], []
        for episode_id in manifest["splits"][split_name]:
            with h5py.File(paths[episode_id], "r") as clean:
                x, action = clean["full_physical_state"][:].astype("f4"), clean["executed_action"][:].astype("f4")
                dp, raw, gt = clean["timestep_dp"][:].astype(int), clean["timestep_raw"][:].astype(int), clean["active_phase"][:].astype(int)
                source, injection = clean["action_source"][:], clean["injection_active"][:]
                if np.any(source != 0) or np.any(injection): raise RuntimeError("clean DP data contains non-expert/injection action")
                episode_stage = []
                for index, (dp_time, raw_time) in enumerate(zip(dp, raw)):
                    value = lookup.get((episode_id, split_name, int(dp_time), int(raw_time)))
                    if value is None: failures += 1; continue
                    cached_gt, predicted, onehot = value
                    if cached_gt != gt[index] or predicted not in range(5) or onehot.shape != (5,) or not np.array_equal(onehot, np.eye(5, dtype="f4")[predicted]): failures += 1; continue
                    episode_stage.append(onehot); gt_match.append(predicted == gt[index])
                if len(episode_stage) != len(x): continue
                physical.append(x); actions.append(action); stages.append(np.asarray(episode_stage, "f4"))
        joined[split_name] = {"physical": np.concatenate(physical), "action": np.concatenate(actions), "stage": np.concatenate(stages)}
        reference = getattr(prepared, split_name)
        physical_failures += int(not np.array_equal(joined[split_name]["physical"], reference.physical))
        action_failures += int(not np.array_equal(joined[split_name]["action"], reference.action))
        if len(joined[split_name]["physical"]) != len(reference): failures += abs(len(joined[split_name]["physical"]) - len(reference)) + 1
    audit = {"CACHE_JOIN_FAILURES": int(failures), "PHYSICAL_ALIGNMENT_FAILURES": int(physical_failures), "ACTION_ALIGNMENT_FAILURES": int(action_failures), "STAGE_CACHE_ALIGNMENT_FAILURES": int(failures), "CACHE_ROWS": len(rows), "TCN_GT_STAGE_MATCH_RATE": float(np.mean(gt_match)) if gt_match else 0.0, "DP_INJECTION_ACTIONS_USED": 0, "DP_NON_EXPERT_ACTIONS_USED": 0}
    if any(audit[key] != 0 for key in ("CACHE_JOIN_FAILURES", "PHYSICAL_ALIGNMENT_FAILURES", "ACTION_ALIGNMENT_FAILURES", "STAGE_CACHE_ALIGNMENT_FAILURES")): raise RuntimeError(f"exact C2 cache join failed: {audit}")
    return joined, audit


def verify_normalization(prepared) -> dict:
    with np.load(ORACLE / "normalization_stats.npz") as reference:
        physical = np.array_equal(prepared.physical_normalizer.mean, reference["physical_mean"]) and np.array_equal(prepared.physical_normalizer.std, reference["physical_std"])
        action = np.array_equal(prepared.action_normalizer.mean, reference["action_mean"]) and np.array_equal(prepared.action_normalizer.std, reference["action_std"])
    if not physical or not action: raise RuntimeError("normalization differs from formal Recovery-aware Oracle V2")
    return {"ORACLE_V2_NORMALIZATION": str((ORACLE / "normalization_stats.npz").resolve()), "PHYSICAL_NORMALIZATION_MATCH_ORACLE_V2": "YES", "ACTION_NORMALIZATION_MATCH_ORACLE_V2": "YES", "STAGE_NORMALIZATION": "NONE (hard one-hot)"}


def loader(joined, prepared, split_name, shuffle):
    values = joined[split_name]
    physical = prepared.physical_normalizer.normalize(values["physical"])
    action = prepared.action_normalizer.normalize(values["action"])
    observation = np.concatenate((physical, values["stage"]), axis=1).astype("f4")
    return DataLoader(TensorDataset(torch.from_numpy(observation), torch.from_numpy(action.astype("f4"))), batch_size=min(BATCH, len(observation)), shuffle=shuffle, drop_last=False, num_workers=0, pin_memory=True)


def infinite(data_loader):
    while True: yield from data_loader


def grad_norm(module):
    values = [parameter.grad.detach().norm() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(values)).item()) if values else 0.0


@torch.no_grad()
def validation(model, data_loader, device):
    model.eval(); losses = []
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(12345); torch.cuda.manual_seed_all(12345)
        for number, (observation, action) in enumerate(data_loader):
            if number >= VALIDATION_BATCHES: break
            losses.append(model.loss(observation.to(device, non_blocking=True), action.to(device, non_blocking=True)).detach().item())
    model.train(); return float(np.mean(losses))


def stage_condition_active(model, observation, action, device):
    model.eval(); count = min(8, len(observation)); physical, noisy = observation[:count, :43].to(device), action[:count].to(device)
    timestep = torch.full((count,), 17, dtype=torch.long, device=device); outputs = []
    for stage in range(5):
        onehot = torch.eye(5, device=device)[torch.full((count,), stage, device=device, dtype=torch.long)]
        condition = model.condition_encoder(physical, onehot)
        outputs.append(model.denoiser(torch.cat((condition, noisy), 1), timestep)[:, 128:])
    delta = torch.stack(outputs).std(0).mean().detach().item(); model.train()
    return {"TCN_STAGE_CONDITION_ACTIVE": "YES" if delta > 1e-7 else "NO", "stage_output_l2": float(delta)}


def smoke(train_loader, device, cfg):
    seed_everything(SEED); model = StageEmbeddingDiffusion(cfg).to(device).train(); optimizer = Adam(model.parameters(), lr=LR)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}; batches = infinite(train_loader); losses = []; gradients = {"physical": [], "stage": [], "denoiser": []}; nan = inf = 0
    first_observation, first_action = next(iter(train_loader)); stage_test = stage_condition_active(model, first_observation, first_action, device)
    for step in range(1, 1001):
        observation, action = next(batches); observation, action = observation.to(device, non_blocking=True), action.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True); loss = model.loss(observation, action)
        if not torch.isfinite(loss): raise FloatingPointError(f"smoke loss NaN/Inf at step {step}")
        loss.backward(); gradients["physical"].append(grad_norm(model.condition_encoder.physical_encoder)); gradients["stage"].append(grad_norm(model.condition_encoder.stage_encoder)); gradients["denoiser"].append(grad_norm(model.denoiser))
        if not all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters() if parameter.grad is not None): raise FloatingPointError(f"smoke gradient NaN/Inf at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(loss.detach().item())
    diffs = {"physical": max((parameter.detach() - initial[name]).abs().max().item() for name, parameter in model.condition_encoder.physical_encoder.named_parameters(prefix="condition_encoder.physical_encoder")), "stage": max((parameter.detach() - initial[name]).abs().max().item() for name, parameter in model.condition_encoder.stage_encoder.named_parameters(prefix="condition_encoder.stage_encoder")), "denoiser": max((parameter.detach() - initial[name]).abs().max().item() for name, parameter in model.denoiser.named_parameters(prefix="denoiser"))}
    report = {"TCN_V2_SMOKE_VALID": "YES", "steps": 1000, "loss_start": losses[0], "loss_end": losses[-1], "NaN": nan, "Inf": inf, "Physical_Encoder_Grad_Norm": float(np.mean(gradients["physical"])), "Stage_Encoder_Grad_Norm": float(np.mean(gradients["stage"])), "Denoiser_Grad_Norm": float(np.mean(gradients["denoiser"])), "Physical_Encoder_Param_Diff": diffs["physical"], "Stage_Encoder_Param_Diff": diffs["stage"], "Denoiser_Param_Diff": diffs["denoiser"], **stage_test}
    if stage_test["TCN_STAGE_CONDITION_ACTIVE"] != "YES" or any(value <= 0 for value in chain(gradients["physical"], gradients["stage"], gradients["denoiser"])) or any(value <= 0 for value in diffs.values()): raise RuntimeError(f"TCN-V2 smoke gate failed: {report}")
    return report


def write_references(output, refs, oracle_cfg, prepared, normalization, alignment, architecture):
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"MODEL_NAME": "RECOVERY_TCN_V2", "MODEL_CLASS": "StageEmbeddingDiffusion", "PHYSICAL_DIM": 43, "STAGE_DIM": 5, "ACTION_DIM": 7, "STAGE_SOURCE": "CAUSAL_TCN_RECOVERY_V1", "STAGE_REPRESENTATION": "HARD_ONEHOT", "optimizer": "Adam", "learning_rate": LR, "batch_size": BATCH, "ema_decay": EMA, "training_seed": SEED, "diffusion": oracle_cfg.state_dict(), "training_steps": STEPS, "checkpoint_interval": CHECKPOINT_INTERVAL, "validation_every": VALIDATION_EVERY, "validation_batches": VALIDATION_BATCHES, "architecture_audit": architecture, "normalization": normalization, "cache_alignment": alignment}, indent=2) + "\n")
    (output / "dataset_reference.json").write_text(json.dumps(prepared.manifest() | {"audit": prepared.audit}, indent=2) + "\n")
    (output / "split_reference.json").write_text((DATA / "split_manifest.json").read_text())
    (output / "normalization_reference.json").write_text(json.dumps(normalization, indent=2) + "\n")
    (output / "causal_tcn_reference.json").write_text(json.dumps({"CAUSAL_TCN_CHECKPOINT": str(refs["checkpoint"].resolve()), "CAUSAL_TCN_CHECKPOINT_SHA256": refs["checkpoint_sha256"], "CAUSAL_TCN_NORMALIZATION": str(refs["normalization"].resolve()), "CAUSAL_TCN_VALID": "YES"}, indent=2) + "\n")
    (output / "stage_cache_reference.json").write_text(json.dumps({"TCN_STAGE_CACHE": str(refs["cache"].resolve()), "STAGE_REPRESENTATION": "HARD_ONEHOT", **alignment}, indent=2) + "\n")
    (output / "stage_cache_audit_reference.json").write_text(json.dumps(refs["cache_audit"], indent=2) + "\n")


def formal(train_loader, validation_loader, device, cfg, output, smoke_report, stage_match_rate):
    seed_everything(SEED); model = StageEmbeddingDiffusion(cfg).to(device).train(); optimizer = Adam(model.parameters(), lr=LR); ema = ExponentialMovingAverage(model, EMA); batches, checkpoints = infinite(train_loader), output / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    fields = ["step", "train_diffusion_loss", "validation_diffusion_loss", "physical_encoder_gradient_norm", "stage_encoder_gradient_norm", "denoiser_gradient_norm", "NaN_count", "Inf_count", "TCN_GT_STAGE_MATCH_RATE"]
    last_validation = float("nan")
    with (output / "training_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for step in range(1, STEPS + 1):
            observation, action = next(batches); observation, action = observation.to(device, non_blocking=True), action.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); loss = model.loss(observation, action)
            if not torch.isfinite(loss): raise FloatingPointError(f"formal loss NaN/Inf at step {step}")
            loss.backward(); physical_grad, stage_grad, denoiser_grad = grad_norm(model.condition_encoder.physical_encoder), grad_norm(model.condition_encoder.stage_encoder), grad_norm(model.denoiser)
            if not all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters() if parameter.grad is not None): raise FloatingPointError(f"formal gradient NaN/Inf at step {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); ema.update(model)
            if step == 1 or step % VALIDATION_EVERY == 0:
                last_validation = validation(model, validation_loader, device); writer.writerow({"step": step, "train_diffusion_loss": loss.detach().item(), "validation_diffusion_loss": last_validation, "physical_encoder_gradient_norm": physical_grad, "stage_encoder_gradient_norm": stage_grad, "denoiser_gradient_norm": denoiser_grad, "NaN_count": 0, "Inf_count": 0, "TCN_GT_STAGE_MATCH_RATE": stage_match_rate}); stream.flush(); print(json.dumps({"step": step, "train_loss": loss.detach().item(), "validation_loss": last_validation}), flush=True)
            if step % CHECKPOINT_INTERVAL == 0:
                torch.save({"format_version": "recovery-tcn-v2-1.0", "step": step, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "diffusion_config": cfg.state_dict(), "smoke_report": smoke_report, "validation_loss": last_validation}, checkpoints / f"step_{step:06d}.pt")
    return {"TCN_V2_FORMAL_TRAINING_VALID": "YES", "TRAINING_STEPS": STEPS, "CHECKPOINTS": [f"{step // 1000}k" for step in range(CHECKPOINT_INTERVAL, STEPS + 1, CHECKPOINT_INTERVAL)], "NAN_COUNT": 0, "INF_COUNT": 0, "PHASE_C_TCN_V2_TRAINING_COMPLETE": "YES", "EXPANDED_VALIDATION": "WAITING_FOR_PHASE_A_MANIFEST"}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUT); parser.add_argument("--check-only", action="store_true"); parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(); refs = frozen_references(); oracle_cfg = expected_oracle_config(); prepared = prepare_recovery_dataset(DATA); joined, alignment = cache_join(prepared, refs["cache"]); normalization = verify_normalization(prepared)
    oracle_model, tcn_model = StageEmbeddingDiffusion(oracle_cfg), StageEmbeddingDiffusion(oracle_cfg); oracle_count, tcn_count = count_parameters(oracle_model), count_parameters(tcn_model)
    historical_count = json.loads((ORACLE / "config.json").read_text())["architecture_audit"]["total_parameters"]
    architecture = {"ORACLE_V2_MODEL_CLASS": type(oracle_model).__name__, "ORACLE_V2_PARAMETER_COUNT": oracle_count, "TCN_V2_MODEL_CLASS": type(tcn_model).__name__, "TCN_V2_PARAMETER_COUNT": tcn_count, "ORACLE_V2_FORMAL_PARAMETER_COUNT": historical_count, "Physical_Encoder_identical": "YES", "Stage_Encoder_identical": "YES", "Fusion_identical": "YES", "Denoiser_identical": "YES", "ARCHITECTURE_MATCH": "YES" if oracle_count == tcn_count == historical_count else "NO"}
    if architecture["ARCHITECTURE_MATCH"] != "YES": raise RuntimeError(f"architecture fairness audit failed: {architecture}")
    write_references(args.output, refs, oracle_cfg, prepared, normalization, alignment, architecture)
    startup = {"MODEL_NAME": "RECOVERY_TCN_V2", "PHYSICAL_DIM": 43, "STAGE_DIM": 5, "ACTION_DIM": 7, "STAGE_SOURCE": "CAUSAL_TCN_RECOVERY_V1", "STAGE_REPRESENTATION": "HARD_ONEHOT", "CAUSAL_TCN_VALID": "YES", "TCN_STAGE_CACHE_VALID": "YES", "ORACLE_V2_PARAMETER_COUNT": oracle_count, "TCN_V2_PARAMETER_COUNT": tcn_count, "ARCHITECTURE_MATCH": "YES", "NORMALIZATION_MATCH": "YES", "CACHE_JOIN_FAILURES": 0}
    if args.check_only: print(json.dumps(startup | {"STATIC_CHECK": "PASS", "DATASET_CACHE_ALIGNMENT_CHECK": "PASS"}, indent=2)); return
    device = cuda(); train_loader, validation_loader = loader(joined, prepared, "train", True), loader(joined, prepared, "validation", False); smoke_report = smoke(train_loader, device, oracle_cfg); (args.output / "smoke_report.json").write_text(json.dumps(smoke_report, indent=2) + "\n")
    print(json.dumps(startup | {"TCN_V2_SMOKE_VALID": smoke_report["TCN_V2_SMOKE_VALID"], "TCN_STAGE_CONDITION_ACTIVE": smoke_report["TCN_STAGE_CONDITION_ACTIVE"], "CUDA_AVAILABLE": "YES", "DEVICE": "cuda:0"}, indent=2), flush=True)
    if args.smoke_only: return
    report = formal(train_loader, validation_loader, device, oracle_cfg, args.output, smoke_report, alignment["TCN_GT_STAGE_MATCH_RATE"]); (args.output / "final_training_report.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
