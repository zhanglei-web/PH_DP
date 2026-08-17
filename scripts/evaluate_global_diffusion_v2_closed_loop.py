#!/usr/bin/env python3
"""Evaluate V2 with the exact V1 protocol and write the direct comparison."""

import json
from pathlib import Path

from mujoco_shared_control.rss2023.global_evaluation import run_evaluation


ROOT = Path("outputs/global_diffusion")
V1_REPORT = ROOT / "global_diffusion_final_expert_20260816T180000Z/closed_loop_300/evaluation_report.json"
V2_ROOT = ROOT / "global_diffusion_v2_expert20000_20260816T230000Z"
OUTPUT = V2_ROOT / "closed_loop_300"


def values(report: dict) -> dict[str, float]:
    summary = report["summary"]
    total_steps = sum(row["episode_length"] for row in report["rows"])
    return {
        "success_rate": summary["success"]["rate"],
        "retreat_rate": summary["retreat"]["rate"],
        "timeout_rate": summary["timeout"]["rate"],
        "place_to_success": summary["place_to_success"],
        "average_return": summary["average_return"],
        "episode_length_mean": summary["episode_length"]["mean"],
        "policy_bound_clipping_rate": summary["policy_clip_steps"] / total_steps,
    }


if __name__ == "__main__":
    v2 = run_evaluation(
        V2_ROOT / "best.pt", V2_ROOT / "normalization_stats.npz", OUTPUT,
        formal_seeds=range(2_000_000, 2_000_300),
    )
    v1 = json.loads(V1_REPORT.read_text())
    v1_values, v2_values = values(v1), values(v2)
    comparison = {
        "protocol": "paired configuration/seeds; independently sampled RSS2023 DDPM",
        "environment_seeds": [2_000_000, 2_000_299],
        "diffusion_sampling_seed_rule": "8000000 + environment_seed",
        "v1_dataset_transitions": 254350,
        "v2_dataset_transitions": 2546144,
        "metrics": {
            name: {"v1": v1_values[name], "v2": v2_values[name],
                   "difference_v2_minus_v1": v2_values[name] - v1_values[name]}
            for name in v1_values
        },
    }
    (OUTPUT / "comparison_v1_v2.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2), flush=True)
