#!/usr/bin/env python3
"""Read-only audit of SAC raw, projected, deployed, and replay actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.sac.actor import SACGaussianActor
from mujoco_shared_control.sac.critic import TwinSACCritic


ROOT = Path(__file__).resolve().parents[1]
ACTOR = ROOT / "outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/actor_initialized.pt"
OLD = ROOT / "outputs/sac_training/sac_v1_sanity_20260813T000000Z/checkpoints/latest.pt"
WARM = ROOT / "outputs/sac_training/sac_v1_critic_warmup_sanity_20260813T010000Z/checkpoints/latest.pt"
OUTPUT = ROOT / "outputs/sac_training/sac_action_consistency_audit_20260813T020000Z"


def project(action: torch.Tensor) -> torch.Tensor:
    """Exact normalized-space equivalent of ExpertCommandAdapter radial limits."""
    result = action.clone()
    for start in (0, 3):
        vector = result[:, start:start + 3]
        norm = vector.norm(dim=1, keepdim=True)
        result[:, start:start + 3] = vector / torch.clamp(norm, min=1.0)
    result[:, 6] = result[:, 6].clamp(-1.0, 1.0)
    return result


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)), "max": float(values.max()),
    }


def load_initial() -> tuple[SACGaussianActor, np.ndarray, np.ndarray]:
    payload = torch.load(ACTOR, map_location="cpu", weights_only=False)
    actor = SACGaussianActor(); actor.load_state_dict(payload["actor_state_dict"]); actor.eval()
    return actor, np.asarray(payload["observation_mean"], np.float32), np.asarray(payload["observation_std"], np.float32)


def load_training(path: Path) -> tuple[dict[str, Any], SACGaussianActor, TwinSACCritic]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = SACGaussianActor(); actor.load_state_dict(payload["core"]["actor"]); actor.eval()
    critics = TwinSACCritic(); critics.load_state_dict(payload["core"]["critics"]); critics.eval()
    return payload, actor, critics


@torch.no_grad()
def sample_projection(
    actor: SACGaussianActor, states: np.ndarray, mean: np.ndarray, std: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    normalized = torch.from_numpy((states.astype(np.float32) - mean) / std)
    raw, _log_prob, _ = actor.sample_action(normalized)
    deployed = project(raw)
    xyz = raw[:, :3].norm(dim=1).numpy()
    rotation = raw[:, 3:6].norm(dim=1).numpy()
    difference = (raw - deployed).abs().numpy()
    vector_difference = (raw - deployed).norm(dim=1).numpy()
    xyz_bad, rot_bad = xyz > 1.0, rotation > 1.0
    report = {
        "samples": len(raw), "sample_seed": seed,
        "translation_norm_gt_1": int(xyz_bad.sum()),
        "translation_projection_fraction": float(xyz_bad.mean()),
        "rotation_norm_gt_1": int(rot_bad.sum()),
        "rotation_projection_fraction": float(rot_bad.mean()),
        "gripper_projection_count": int((raw[:, 6].abs() > 1).sum()),
        "gripper_projection_fraction": float((raw[:, 6].abs() > 1).float().mean()),
        "overall_projected": int((xyz_bad | rot_bad).sum()),
        "overall_projection_fraction": float((xyz_bad | rot_bad).mean()),
        "raw_minus_deployed_l2": distribution(vector_difference),
        "per_dimension_absolute_difference_mean": difference.mean(0).tolist(),
        "per_dimension_absolute_difference_p95": np.quantile(difference, .95, axis=0).tolist(),
        "raw_norms": {"translation": distribution(xyz), "rotation": distribution(rotation)},
    }
    return report, normalized, raw, deployed


def replay_support(payload: dict[str, Any]) -> dict[str, Any]:
    replay = payload["replay"]
    action = np.asarray(replay["action"], np.float32)
    xyz, rotation = np.linalg.norm(action[:, :3], axis=1), np.linalg.norm(action[:, 3:6], axis=1)
    tolerance = 2e-6
    return {
        "transitions": len(action),
        "translation_norm_gt_1_plus_tolerance": int((xyz > 1 + tolerance).sum()),
        "rotation_norm_gt_1_plus_tolerance": int((rotation > 1 + tolerance).sum()),
        "gripper_outside_minus1_plus1": int((np.abs(action[:, 6]) > 1).sum()),
        "translation_at_projection_boundary_proxy": int((np.abs(xyz - 1) <= tolerance).sum()),
        "translation_boundary_proxy_fraction": float((np.abs(xyz - 1) <= tolerance).mean()),
        "rotation_at_projection_boundary_proxy": int((np.abs(rotation - 1) <= tolerance).sum()),
        "rotation_boundary_proxy_fraction": float((np.abs(rotation - 1) <= tolerance).mean()),
        "overall_boundary_proxy": int(((np.abs(xyz - 1) <= tolerance) | (np.abs(rotation - 1) <= tolerance)).sum()),
        "overall_boundary_proxy_fraction": float(((np.abs(xyz - 1) <= tolerance) | (np.abs(rotation - 1) <= tolerance)).mean()),
        "note": "Boundary counts are projection signatures/proxies; historical raw actions and clipping flags were not checkpointed.",
    }


@torch.no_grad()
def q_comparison(
    payload: dict[str, Any], actor: SACGaussianActor, critics: TwinSACCritic,
    sample_count: int, seed: int,
) -> dict[str, Any]:
    replay = payload["replay"]
    states = np.asarray(replay["observation"], np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(states), min(sample_count, len(states)), replace=False)
    mean = np.asarray(payload["observation_mean"], np.float32)
    std = np.asarray(payload["observation_std"], np.float32)
    projection, normalized, raw, deployed = sample_projection(actor, states[indices], mean, std, seed)
    q1_raw, q2_raw = critics(normalized, raw)
    q1_projected, q2_projected = critics(normalized, deployed)
    xyz_bad = raw[:, :3].norm(dim=1) > 1
    rot_bad = raw[:, 3:6].norm(dim=1) > 1
    any_bad = xyz_bad | rot_bad
    def delta_report(raw_q: torch.Tensor, projected_q: torch.Tensor, mask: torch.Tensor | None = None):
        delta = (raw_q - projected_q).squeeze(1)
        if mask is not None: delta = delta[mask]
        values = delta.numpy()
        return {"count": len(values), **distribution(values),
                "absolute": distribution(np.abs(values))}
    return {
        "projection": projection,
        "q1_raw_minus_projected": delta_report(q1_raw, q1_projected),
        "q2_raw_minus_projected": delta_report(q2_raw, q2_projected),
        "out_of_ball_subset": {
            "count": int(any_bad.sum()),
            "q1_raw_minus_projected": delta_report(q1_raw, q1_projected, any_bad),
            "q2_raw_minus_projected": delta_report(q2_raw, q2_projected, any_bad),
        },
        "q_values": {
            "q1_raw": distribution(q1_raw.squeeze(1).numpy()),
            "q1_projected": distribution(q1_projected.squeeze(1).numpy()),
            "q2_raw": distribution(q2_raw.squeeze(1).numpy()),
            "q2_projected": distribution(q2_projected.squeeze(1).numpy()),
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    initial, mean, std = load_initial()
    old_payload, old_actor, old_critics = load_training(OLD)
    warm_payload, full_actor, full_critics = load_training(WARM)
    warm_replay = warm_payload["replay"]
    stages = {
        "initial_validation_proxy": sample_projection(
            initial, np.asarray(warm_replay["observation"][:12_817]), mean, std, 9101
        )[0],
        "collect_1_10000": sample_projection(
            initial, np.asarray(warm_replay["observation"][:10_000]), mean, std, 9102
        )[0],
        "critic_warmup_10001_20000": sample_projection(
            initial, np.asarray(warm_replay["observation"][10_000:20_000]), mean, std, 9103
        )[0],
        "full_sac_25000": sample_projection(
            full_actor, np.asarray(warm_replay["observation"]), mean, std, 9104
        )[0],
    }
    action_stats = {
        "method": "one reproducible stochastic Actor sample per recorded state; exact radial adapter projection, excluding IK history-dependent fallback",
        "stages": stages,
        "replay_support": {
            "old_full_sac_20k": replay_support(old_payload),
            "critic_warmup_full_sac_25k": replay_support(warm_payload),
        },
    }
    q_stats = {
        "old_full_sac_20k": q_comparison(old_payload, old_actor, old_critics, 20_000, 9201),
        "critic_warmup_full_sac_25k": q_comparison(warm_payload, full_actor, full_critics, 25_000, 9202),
        "critic_warmup_20k_limitation": "The 20k warmup Critic state was overwritten by later latest.pt; best.pt is the unchanged 10k Actor checkpoint. Exact Q(raw)-Q(projected) at warmup end cannot be reconstructed. Actor-side target support rates are reported exactly on warmup next-state support, and Q drift comes from logged 20k metrics.",
    }
    flow = {
        "actor": {"u": "[B,7] real Gaussian rsample", "raw_action": "tanh(u), each component in (-1,1)", "log_prob": "Gaussian plus elementwise tanh correction for raw_action"},
        "actor_loss": "Q(normalized_state, raw_action); no adapter projection",
        "td_target": "target_Q(normalized_next_state, raw_next_action); no adapter projection",
        "environment": "denormalize raw_action; radial L2 projection of translation and rotation; IK; fallback may replace Cartesian delta with hold",
        "replay": "post-projection deployed normalized action; fallback stored as zero Cartesian delta plus safe normalized gripper",
    }
    (OUTPUT / "action_projection_stats.json").write_text(json.dumps(action_stats, indent=2) + "\n")
    (OUTPUT / "q_raw_vs_projected.json").write_text(json.dumps(q_stats, indent=2) + "\n")
    (OUTPUT / "action_flow.json").write_text(json.dumps(flow, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "action": action_stats, "q": q_stats}, indent=2))


if __name__ == "__main__":
    main()
