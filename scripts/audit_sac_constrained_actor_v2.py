#!/usr/bin/env python3
"""Read-only stochastic, Jacobian, and adapter audit for constrained Actor v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from mujoco_shared_control.collection.datasets import ManifestActorDataset
from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor, radial_squash


MANIFEST = Path("manifests/rule_expert_v1_formal.json")


def validation_states(mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    values = []
    dataset = ManifestActorDataset(MANIFEST, "validation")
    for entry in dataset.entries:
        with h5py.File(entry.path, "r") as episode:
            values.append(np.asarray(episode["observations/policy_state_42"][:], np.float32))
    return torch.from_numpy((np.concatenate(values) - mean) / std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    payload = torch.load(run / "actor_initialized.pt", map_location="cpu", weights_only=False)
    actor = SACConstrainedGaussianActor()
    actor.load_state_dict(payload["actor_state_dict"]); actor.eval()
    states = validation_states(
        np.asarray(payload["observation_mean"], np.float32),
        np.asarray(payload["observation_std"], np.float32),
    )
    torch.manual_seed(20260812)
    actions, log_probs = [], []
    with torch.no_grad():
        for _ in range(5):
            action, log_prob, _ = actor.sample_action(states)
            actions.append(action); log_probs.append(log_prob)
    action = torch.cat(actions); log_prob = torch.cat(log_probs)
    jacobian_rows = []
    for radius in (0.0, 1e-8, 1e-4, .7, 3.0, 10.0):
        vector = torch.tensor([radius, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        transformed, analytic = radial_squash(vector.unsqueeze(0))
        jacobian = torch.autograd.functional.jacobian(
            lambda value: radial_squash(value.unsqueeze(0))[0].squeeze(0), vector
        )
        numeric = torch.linalg.slogdet(jacobian).logabsdet
        (transformed.sum() + analytic.sum()).backward()
        jacobian_rows.append({
            "radius": radius, "analytic_logdet": float(analytic.detach()),
            "autograd_logdet": float(numeric.detach()),
            "absolute_error": float(abs(analytic.detach() - numeric.detach())),
            "gradient_finite": bool(torch.isfinite(vector.grad).all()),
        })
    large = torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    large_action, large_logdet = radial_squash(large)
    (large_action.sum() + large_logdet.sum()).backward()
    report = {
        "validation_states": len(states), "samples_per_state": 5,
        "sample_count": len(action), "all_actions_finite": bool(torch.isfinite(action).all()),
        "all_log_probs_finite": bool(torch.isfinite(log_prob).all()),
        "translation_norm_max": float(torch.linalg.vector_norm(action[:, :3], dim=-1).max()),
        "rotation_norm_max": float(torch.linalg.vector_norm(action[:, 3:6], dim=-1).max()),
        "gripper_min": float(action[:, 6].min()), "gripper_max": float(action[:, 6].max()),
        "translation_violations": int((torch.linalg.vector_norm(action[:, :3], dim=-1) >= 1).sum()),
        "rotation_violations": int((torch.linalg.vector_norm(action[:, 3:6], dim=-1) >= 1).sum()),
        "gripper_violations": int((action[:, 6].abs() >= 1).sum()),
        "jacobian_comparison": jacobian_rows,
        "jacobian_max_abs_error": max(row["absolute_error"] for row in jacobian_rows),
        "large_radius_stability": {
            "radius": 50.0, "action_finite": bool(torch.isfinite(large_action).all()),
            "analytic_logdet_finite": bool(torch.isfinite(large_logdet).all()),
            "gradient_finite": bool(torch.isfinite(large.grad).all()),
            "note": "Autograd determinant underflows at r=50; analytic stable formula remains finite.",
        },
    }
    (run / "logprob_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    closed = json.loads((run / "closed_loop_evaluation.json").read_text())
    identity = {
        **closed["summary"]["adapter_identity"],
        "episodes": closed["summary"]["episodes"],
        "fallback_transitions_excluded": closed["summary"]["ik_fallback_count"],
        "adapter_clipping_count_all_episodes": closed["summary"]["action_clipping_count"],
        "policy_transform": "native B3 x B3 x (-1,1)",
    }
    (run / "adapter_identity_audit.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
