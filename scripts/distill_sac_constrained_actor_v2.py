#!/usr/bin/env python3
"""Distill frozen BC deployed commands into the native constrained SAC Actor."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from mujoco_shared_control.sac.constrained_actor import (
    SACConstrainedGaussianActor,
    configure_constrained_distillation,
    constrained_transform,
    initialize_constrained_from_bc,
    project_to_admissible,
    radial_squash,
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
    error, absolute = action - reference, np.abs(action - reference)
    return {
        "samples": len(reference),
        "normalized_mse": float(np.mean(error**2)),
        "normalized_mae": float(absolute.mean()),
        "normalized_max_abs_error": float(absolute.max()),
        "per_dimension_mae": absolute.mean(0).tolist(),
        "xyz_physical_vector_error_m": float(np.linalg.norm(error[:, :3] * .025, axis=1).mean()),
        "rotation_physical_vector_error_rad": float(np.linalg.norm(error[:, 3:6] * .10, axis=1).mean()),
        "gripper_physical_mae_m": float((absolute[:, 6] * .04).mean()),
        "translation_norm_max": float(np.linalg.norm(action[:, :3], axis=1).max()),
        "rotation_norm_max": float(np.linalg.norm(action[:, 3:6], axis=1).max()),
        "translation_constraint_violations": int((np.linalg.norm(action[:, :3], axis=1) >= 1).sum()),
        "rotation_constraint_violations": int((np.linalg.norm(action[:, 3:6], axis=1) >= 1).sum()),
    }


def logprob_validation(actor: SACConstrainedGaussianActor) -> dict:
    rows = []
    for radius in (0.0, 1e-8, 1e-4, .7, 10.0, 50.0):
        vector = torch.tensor([radius, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        action, analytic = radial_squash(vector.unsqueeze(0))
        jacobian = torch.autograd.functional.jacobian(
            lambda value: radial_squash(value.unsqueeze(0))[0].squeeze(0), vector
        )
        numeric = torch.linalg.slogdet(jacobian).logabsdet
        (action.sum() + analytic.sum()).backward()
        rows.append({
            "radius": radius, "analytic_logdet": float(analytic),
            "autograd_logdet": float(numeric), "absolute_error": float(abs(analytic - numeric)),
            "action_finite": bool(torch.isfinite(action).all()),
            "gradient_finite": bool(torch.isfinite(vector.grad).all()),
        })
    state = torch.randn(4096, 42)
    with torch.no_grad():
        sampled, log_prob, _ = actor.sample_action(state)
    return {
        "radial_jacobian": rows,
        "max_logdet_absolute_error": max(row["absolute_error"] for row in rows),
        "sample_action_finite": bool(torch.isfinite(sampled).all()),
        "sample_log_prob_finite": bool(torch.isfinite(log_prob).all()),
        "sample_translation_norm_max": float(torch.linalg.vector_norm(sampled[:, :3], dim=-1).max()),
        "sample_rotation_norm_max": float(torch.linalg.vector_norm(sampled[:, 3:6], dim=-1).max()),
        "sample_gripper_min": float(sampled[:, 6].min()),
        "sample_gripper_max": float(sampled[:, 6].max()),
    }


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["observation_mean"], np.float32)
    std = np.asarray(checkpoint["observation_std"], np.float32)
    train_data = ManifestActorDataset(MANIFEST, "train")
    validation_data = ManifestActorDataset(MANIFEST, "validation")
    if (len(train_data.entries), len(train_data)) != (900, 115021):
        raise ValueError("train split changed")
    if (len(validation_data.entries), len(validation_data)) != (100, 12817):
        raise ValueError("validation split changed")
    train_state = torch.from_numpy((load_states(train_data) - mean) / std)
    validation_state = torch.from_numpy((load_states(validation_data) - mean) / std)

    teacher = ActorBC(); teacher.load_state_dict(checkpoint["model_state_dict"]); teacher.eval()
    teacher.requires_grad_(False)
    student = SACConstrainedGaussianActor()
    initialize_constrained_from_bc(student, CHECKPOINT)
    initial = deepcopy(student.state_dict())
    log_std_before = deepcopy(student.log_std_head.state_dict())
    trainable = configure_constrained_distillation(student)
    names = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
    print("trainable_parameters=" + json.dumps(names), flush=True)
    with torch.inference_mode():
        train_target = project_to_admissible(torch.clamp(teacher(train_state), -1, 1))
        validation_target = project_to_admissible(torch.clamp(teacher(validation_state), -1, 1))
    config = {
        "objective": "MSE(constrained_transform(mu_student), project_adapter(clip(BC(state),-1,1)))",
        "optimizer": "Adam", "learning_rate": 1e-4, "batch_size": 256,
        "max_epochs": args.max_epochs, "early_stopping_patience": 12,
        "minimum_improvement": 1e-9, "gradient_clip_norm": 1.0,
        "training_seed": SEED, "log_std_init": -3.0, "log_std_bounds": [-5.0, 2.0],
        "trainable_parameters": names,
    }
    loader = DataLoader(
        TensorDataset(train_state, train_target), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(trainable, lr=1e-4)
    best, best_epoch, best_state, stale, history = float("inf"), 0, None, 0, []
    for epoch in range(1, args.max_epochs + 1):
        student.train(); total = 0.0
        for state, target in loader:
            loss = nn.functional.mse_loss(student.deterministic_action(state), target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
            total += float(loss.detach()) * len(state)
        student.eval()
        with torch.inference_mode():
            validation_loss = float(nn.functional.mse_loss(
                student.deterministic_action(validation_state), validation_target
            ))
        train_loss = total / len(train_state)
        history.append({"epoch": epoch, "train_action_mse": train_loss, "validation_action_mse": validation_loss})
        if validation_loss < best - 1e-9:
            best, best_epoch, best_state, stale = validation_loss, epoch, deepcopy(student.state_dict()), 0
        else:
            stale += 1
        print(f"epoch={epoch:03d} train={train_loss:.9g} val={validation_loss:.9g} best={best_epoch}", flush=True)
        if stale >= 12:
            break
    if best_state is None:
        raise RuntimeError("distillation produced no checkpoint")
    student.load_state_dict(best_state); student.eval()
    for key, value in student.log_std_head.state_dict().items():
        torch.testing.assert_close(value, log_std_before[key], rtol=0, atol=0)
    changed = {name: not torch.equal(value, initial[name]) for name, value in student.state_dict().items()}
    if any(changed[name] for name in changed if name.startswith("log_std_head.")):
        raise RuntimeError("log_std changed during distillation")
    with torch.inference_mode():
        action = student.deterministic_action(validation_state).numpy()
    report = equivalence(validation_target.numpy(), action)
    report["reference"] = "frozen adapter projection of clipped BC normalized output"
    report["teacher_projection_fraction"] = float(torch.any(
        torch.ne(train_target, torch.clamp(teacher(train_state), -1, 1)), dim=1
    ).float().mean())
    run = Path("outputs/sac_actor") / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    payload = {
        "format_version": "sac_constrained_actor_v2_full_mean_path_distilled",
        "actor_state_dict": student.state_dict(), "observation_mean": mean,
        "observation_std": std, "action_spec": checkpoint["action_spec"],
        "action_normalization_definition": checkpoint["action_normalization_definition"],
        "bc_checkpoint": str(CHECKPOINT.resolve()), "manifest_path": str(MANIFEST.resolve()),
        "manifest_file_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_content_sha": checkpoint["manifest_content_sha"],
        "distillation_config": config, "best_epoch": best_epoch,
        "best_validation_action_mse": best, "action_equivalence": report,
    }
    atomic_save(payload, run / "actor_initialized.pt")
    (run / "distillation_config.json").write_text(json.dumps({
        **config, "run_id": args.run_id, "train_episodes": 900,
        "train_transitions": 115021, "validation_episodes": 100,
        "validation_transitions": 12817, "best_epoch": best_epoch,
        "epochs_completed": epoch,
    }, indent=2) + "\n")
    (run / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (run / "action_equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
    (run / "logprob_validation.json").write_text(json.dumps(logprob_validation(student), indent=2) + "\n")
    loaded = SACConstrainedGaussianActor()
    loaded.load_state_dict(torch.load(run / "actor_initialized.pt", weights_only=False)["actor_state_dict"])
    with torch.inference_mode():
        torch.testing.assert_close(
            loaded.deterministic_action(validation_state[:128]),
            student.deterministic_action(validation_state[:128]), rtol=0, atol=0,
        )
    print(f"run_dir={run.resolve()}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
