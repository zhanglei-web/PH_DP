#!/usr/bin/env python3
"""Frozen-trunk, mean-head-only BC-to-SAC calibration. No RL updates."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.actor_bc.model import ActorBC
from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.sac.actor import (
    ATANH_EPSILON,
    LOG_STD_INIT,
    SACGaussianActor,
    freeze_for_mean_calibration,
    initialize_from_bc,
)


SEED = 20260812
CHECKPOINT = Path("outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt")
MANIFEST = Path("manifests/rule_expert_v1_formal.json")


def load_states(dataset: ManifestActorDataset) -> np.ndarray:
    values = []
    for entry in dataset.entries:
        with h5py.File(entry.path, "r") as episode:
            values.append(np.asarray(episode["observations/policy_state_42"][:], np.float32))
    return np.concatenate(values)


def equivalence(reference: np.ndarray, action: np.ndarray) -> dict:
    error = action - reference
    absolute = np.abs(error)
    return {
        "normalized_mse": float(np.mean(error ** 2)),
        "normalized_mae": float(absolute.mean()),
        "normalized_max_abs_error": float(absolute.max()),
        "per_dimension_mae": absolute.mean(axis=0).tolist(),
        "xyz_physical_vector_error_m": float(np.linalg.norm(error[:, :3] * .025, axis=1).mean()),
        "rotation_physical_vector_error_rad": float(np.linalg.norm(error[:, 3:6] * .10, axis=1).mean()),
        "gripper_physical_mae_m": float((absolute[:, 6] * .04).mean()),
    }


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["observation_mean"], np.float32)
    std = np.asarray(checkpoint["observation_std"], np.float32)
    train_dataset = ManifestActorDataset(MANIFEST, "train")
    validation_dataset = ManifestActorDataset(MANIFEST, "validation")
    if (len(train_dataset.entries), len(train_dataset)) != (900, 115021):
        raise ValueError("calibration train split changed")
    if (len(validation_dataset.entries), len(validation_dataset)) != (100, 12817):
        raise ValueError("calibration validation split changed")
    train_state = torch.from_numpy((load_states(train_dataset) - mean) / std)
    validation_state = torch.from_numpy((load_states(validation_dataset) - mean) / std)

    bc = ActorBC(); bc.load_state_dict(checkpoint["model_state_dict"]); bc.eval()
    actor = SACGaussianActor()
    init = initialize_from_bc(actor, CHECKPOINT, option="direct_head_copy", log_std_init=LOG_STD_INIT)
    trunk_before = deepcopy(actor.trunk.state_dict())
    log_std_before = deepcopy(actor.log_std_head.state_dict())
    initial_mean_before = deepcopy(actor.mean_head.state_dict())
    freeze_for_mean_calibration(actor)
    with torch.inference_mode():
        train_bc = torch.clamp(bc(train_state), -1.0, 1.0)
        val_bc = torch.clamp(bc(validation_state), -1.0, 1.0)
        train_target = torch.atanh(torch.clamp(train_bc, -1+ATANH_EPSILON, 1-ATANH_EPSILON))
        val_target = torch.atanh(torch.clamp(val_bc, -1+ATANH_EPSILON, 1-ATANH_EPSILON))
        train_feature = actor.trunk(train_state)
        val_feature = actor.trunk(validation_state)

    config = {
        "optimizer": "Adam", "learning_rate": 1e-3, "batch_size": 256,
        "max_epochs": 100, "early_stopping_patience": 10,
        "minimum_improvement": 1e-8, "gradient_clip_norm": 1.0,
        "training_seed": SEED, "atanh_delta": ATANH_EPSILON,
        "trainable_parameters": ["mean_head.weight", "mean_head.bias"],
        "log_std_init": LOG_STD_INIT, "log_std_bounds": [-5.0, 2.0],
    }
    loader = DataLoader(
        TensorDataset(train_feature, train_target), batch_size=config["batch_size"],
        shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(actor.mean_head.parameters(), lr=config["learning_rate"])
    best_loss = float("inf"); stale = 0; best_epoch = 0; best_state = None; history = []
    for epoch in range(1, config["max_epochs"] + 1):
        actor.mean_head.train(); total = 0.0
        for feature, target in loader:
            prediction = actor.mean_head(feature)
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(actor.mean_head.parameters(), config["gradient_clip_norm"])
            optimizer.step(); total += float(loss) * len(feature)
        actor.mean_head.eval()
        with torch.inference_mode():
            validation_loss = float(nn.functional.mse_loss(actor.mean_head(val_feature), val_target))
        train_loss = total / len(train_feature)
        history.append({"epoch":epoch,"train_pre_tanh_mse":train_loss,
                        "validation_pre_tanh_mse":validation_loss})
        improved = validation_loss < best_loss - config["minimum_improvement"]
        if improved:
            best_loss=validation_loss; best_epoch=epoch
            best_state=deepcopy(actor.mean_head.state_dict()); stale=0
        else: stale += 1
        print(f"epoch={epoch:03d} train={train_loss:.8g} val={validation_loss:.8g} best={best_epoch}",flush=True)
        if stale >= config["early_stopping_patience"]: break
    if best_state is None: raise RuntimeError("calibration produced no best state")
    actor.mean_head.load_state_dict(best_state); actor.eval()

    for key, value in actor.trunk.state_dict().items():
        torch.testing.assert_close(value, trunk_before[key], rtol=0, atol=0)
    for key, value in actor.log_std_head.state_dict().items():
        torch.testing.assert_close(value, log_std_before[key], rtol=0, atol=0)
    if all(torch.equal(value, initial_mean_before[key]) for key,value in actor.mean_head.state_dict().items()):
        raise RuntimeError("mean head did not change")
    with torch.inference_mode():
        direct_actor=SACGaussianActor(); initialize_from_bc(direct_actor,CHECKPOINT,option="direct_head_copy")
        direct=direct_actor.deterministic_action(validation_state).numpy()
        calibrated=actor.deterministic_action(validation_state).numpy()
        mu,log_std,std_output=actor.distribution_stats(validation_state)
    reference=val_bc.numpy()
    action_report={
        "samples":len(reference), "reference":"clip(raw BC output,-1,1)",
        "direct_copy":equivalence(reference,direct),
        "calibrated":equivalence(reference,calibrated),
        "pre_tanh_validation_mse":best_loss,
        "mean_stats":{
            "min":float(mu.min()),"max":float(mu.max()),
            "absolute_p95":float(torch.quantile(mu.abs(),.95)),
            "absolute_p99":float(torch.quantile(mu.abs(),.99)),
            "absolute_max":float(mu.abs().max()),
            "gripper_min":float(mu[:,6].min()),"gripper_max":float(mu[:,6].max()),
            "gripper_absolute_p95":float(torch.quantile(mu[:,6].abs(),.95)),
            "gripper_absolute_p99":float(torch.quantile(mu[:,6].abs(),.99)),
        },
        "log_std_stats":{
            "mean":float(log_std.mean()),"std":float(log_std.std(unbiased=False)),
            "min":float(log_std.min()),"max":float(log_std.max()),
            "exp_mean":float(std_output.mean()),
        },
        "finite":bool(torch.isfinite(mu).all() and torch.isfinite(log_std).all()
                      and np.isfinite(calibrated).all()),
    }
    run_id=args.run_id or datetime.now(timezone.utc).strftime("sac_actor_v1_%Y%m%dT%H%M%SZ")
    run=Path("outputs/sac_actor")/run_id; run.mkdir(parents=True,exist_ok=False)
    manifest_sha=hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    payload={
        "format_version":"sac_actor_v1_initialized","actor_state_dict":actor.state_dict(),
        "observation_mean":mean,"observation_std":std,"action_spec":checkpoint["action_spec"],
        "action_normalization_definition":checkpoint["action_normalization_definition"],
        "bc_checkpoint":str(CHECKPOINT.resolve()),"manifest_path":str(MANIFEST.resolve()),
        "manifest_file_sha256":manifest_sha,"manifest_content_sha":checkpoint["manifest_content_sha"],
        "calibration_config":config,"best_epoch":best_epoch,"best_validation_pre_tanh_mse":best_loss,
        "initialization_mapping":asdict(init),"action_equivalence":action_report,
    }
    atomic_save(payload,run/"actor_initialized.pt")
    (run/"initialization_config.json").write_text(json.dumps({**config,"run_id":run_id,
        "bc_checkpoint":str(CHECKPOINT.resolve()),"manifest":str(MANIFEST.resolve()),
        "train_episodes":900,"train_transitions":115021,"validation_episodes":100,
        "validation_transitions":12817,"best_epoch":best_epoch,"epochs_completed":epoch,
        "history":history},indent=2)+"\n")
    (run/"action_equivalence.json").write_text(json.dumps(action_report,indent=2)+"\n")
    print(f"run_dir={run.resolve()}")


if __name__ == "__main__": main()
