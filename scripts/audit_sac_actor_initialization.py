#!/usr/bin/env python3
"""Read-only BC-to-SAC actor initialization audit; performs no fitting."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from mujoco_shared_control.actor_bc.model import ActorBC
from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.sac.actor import SACGaussianActor, initialize_from_bc


SEED = 20260812
CHECKPOINT = Path("outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt")
MANIFEST = Path("manifests/rule_expert_v1_formal.json")
OUTPUT = Path("outputs/sac_actor/sac_actor_v1_initialization_audit.json")
DELTA = 1e-6


def arrays() -> np.ndarray:
    dataset = ManifestActorDataset(MANIFEST, "validation")
    values = []
    for entry in dataset.entries:
        with h5py.File(entry.path, "r") as episode:
            values.append(np.asarray(episode["observations/policy_state_42"][:], np.float32))
    return np.concatenate(values)


def metrics(reference: np.ndarray, action: np.ndarray) -> dict:
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


def main() -> None:
    torch.manual_seed(SEED)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["observation_mean"], np.float32)
    std = np.asarray(checkpoint["observation_std"], np.float32)
    state = (arrays() - mean) / std
    tensor = torch.from_numpy(state)
    bc = ActorBC(); bc.load_state_dict(checkpoint["model_state_dict"]); bc.eval()
    with torch.inference_mode():
        raw_bc = bc(tensor).numpy()
    reference = np.clip(raw_bc, -1.0, 1.0)

    direct = SACGaussianActor(); initialize_from_bc(direct, CHECKPOINT, option="direct_head_copy")
    direct.eval()
    with torch.inference_mode(): direct_action = direct.deterministic_action(tensor).numpy()

    torch.manual_seed(SEED)
    trunk_only = SACGaussianActor(); initialize_from_bc(trunk_only, CHECKPOINT, option="trunk_only")
    trunk_only.eval()
    with torch.inference_mode(): trunk_action = trunk_only.deterministic_action(tensor).numpy()

    target_mu = np.arctanh(np.clip(reference, -1.0 + DELTA, 1.0 - DELTA))
    oracle_action = np.tanh(target_mu)
    result = {
        "format_version": "sac_actor_v1_initialization_audit",
        "checkpoint": str(CHECKPOINT.resolve()),
        "manifest": str(MANIFEST.resolve()),
        "samples": int(len(state)),
        "representative_set": "all 12817 validation transitions",
        "bc_reference": "clip(raw BC normalized action, -1, 1)",
        "options": {
            "A_direct_head_copy": metrics(reference, direct_action),
            "B_trunk_only_fixed_seed": metrics(reference, trunk_action),
            "C_atanh_oracle_target": {
                **metrics(reference, oracle_action),
                "delta": DELTA,
                "target_mu_min": float(target_mu.min()),
                "target_mu_max": float(target_mu.max()),
                "note": "Oracle target only; no calibration optimizer was run.",
            },
        },
        "exploration": {},
    }
    generator = torch.Generator().manual_seed(SEED)
    oracle_mean = torch.from_numpy(target_mu)
    for initial in (-1.0, -2.0, -3.0):
        sigma = float(np.exp(initial))
        noise = torch.randn(oracle_mean.shape, generator=generator) * sigma
        sampled = torch.tanh(oracle_mean + noise).detach().numpy()
        result["exploration"][str(initial)] = {
            "std": sigma,
            "center": "behavior-preserving atanh oracle mean",
            **metrics(oracle_action, sampled),
            "fraction_any_dimension_abs_gt_0_99": float(np.any(np.abs(sampled) > .99, axis=1).mean()),
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
