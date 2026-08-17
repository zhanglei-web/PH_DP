#!/usr/bin/env python3
"""Deterministic, initialization-only SAC Actor evaluation (no updates)."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.actor_bc.evaluate import MILESTONE_NAMES, evaluate_episode
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.sac.actor import SACGaussianActor, initialize_from_bc
from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor


CHECKPOINT = Path("outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt")


class DeterministicSACPredictor:
    def __init__(self, initialized: Path | None = None) -> None:
        if initialized is None:
            self.model = SACGaussianActor()
            metadata = initialize_from_bc(
                self.model, CHECKPOINT, option="direct_head_copy", log_std_init=-3.0
            )
            self.mean = metadata.observation_mean
            self.std = metadata.observation_std
            self.action_spec = ExpertActionSpec(**metadata.action_spec)
            self.initialization = "BC trunk exact copy + BC head direct copy; tanh(mean)"
            self.checkpoint = CHECKPOINT.resolve()
        else:
            payload = torch.load(initialized, map_location="cpu", weights_only=False)
            if payload.get("format_version") == "sac_constrained_actor_v2_full_mean_path_distilled":
                self.model = SACConstrainedGaussianActor()
            else:
                self.model = SACGaussianActor()
            self.model.load_state_dict(payload["actor_state_dict"])
            self.mean = np.asarray(payload["observation_mean"], np.float32)
            self.std = np.asarray(payload["observation_std"], np.float32)
            self.action_spec = ExpertActionSpec(**payload["action_spec"])
            objective = payload.get("distillation_config", {}).get("objective")
            if payload.get("format_version") == "sac_constrained_actor_v2_full_mean_path_distilled":
                self.initialization = "BC-deployed-target native constrained full mean-path distillation"
            else:
                self.initialization = (
                    "BC-initialized full mean-path action distillation"
                    if objective == "MSE(tanh(mu_student), clip(a_bc_raw,-1,1))"
                    else "frozen-trunk atanh mean-head calibration"
                )
            self.checkpoint = initialized.resolve()
        self.model.eval()

    def predict(self, policy_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = (np.asarray(policy_state, np.float32) - self.mean) / self.std
        with torch.inference_mode():
            action = self.model.deterministic_action(
                torch.from_numpy(normalized).unsqueeze(0)
            ).squeeze(0).numpy().astype(np.float64)
        return action, self.action_spec.denormalize(action)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--initialized",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    predictor = DeterministicSACPredictor(args.initialized)
    rows = []
    for seed in range(300_000, 300_100):
        row = evaluate_episode(predictor, seed)
        rows.append(row)
        print(f"seed={seed} success={int(row['success'])} reason={row['termination_reason']}", flush=True)
    lengths = np.asarray([row["episode_length"] for row in rows])
    report = {
        "format_version": (
            "sac_constrained_actor_v2_initialization_evaluation"
            if isinstance(predictor.model, SACConstrainedGaussianActor)
            else "sac_actor_v1_initialization_evaluation"
        ),
        "initialization": predictor.initialization,
        "gradient_updates": 0,
        "stochastic_sampling": False,
        "checkpoint": str(predictor.checkpoint),
        "seeds": "300000-300099",
        "summary": {
            "episodes": len(rows),
            "success": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "milestone_rates": {name: float(np.mean([row[name] for row in rows])) for name in MILESTONE_NAMES},
            "termination_reason_counts": dict(Counter(row["termination_reason"] for row in rows)),
            "ik_fallback_count": sum(row["ik_fallback_count"] for row in rows),
            "ik_fallback_episodes": sum(row["ik_fallback_count"] > 0 for row in rows),
            "action_clipping_count": sum(row["action_clipping_count"] for row in rows),
            "action_clipping_episodes": sum(row["action_clipping_count"] > 0 for row in rows),
            "adapter_identity": {
                "normal_nonfallback_transitions": sum(row["adapter_identity_transitions"] for row in rows),
                "translation_projection_count": sum(row["adapter_translation_projections"] for row in rows),
                "rotation_projection_count": sum(row["adapter_rotation_projections"] for row in rows),
                "gripper_clip_count": sum(row["adapter_gripper_clips"] for row in rows),
                "difference_mean_l2": (
                    sum(row["adapter_action_difference_sum"] for row in rows)
                    / max(1, sum(row["adapter_identity_transitions"] for row in rows))
                ),
                "difference_max_l2": max(row["adapter_action_difference_max"] for row in rows),
            },
            "drop_count": sum(row["drop_count"] for row in rows),
            "drop_episodes": sum(row["drop_count"] > 0 for row in rows),
            "wrong_gripper_switch_count": sum(row["wrong_gripper_switch_count"] for row in rows),
            "wrong_gripper_switch_episodes": sum(
                row["wrong_gripper_switch_count"] > 0 for row in rows
            ),
            "episode_length": {"mean": float(lengths.mean()), "min": int(lengths.min()), "max": int(lengths.max())},
        },
        "episodes": rows,
    }
    output=args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
