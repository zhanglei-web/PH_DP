#!/usr/bin/env python3
"""Finalize integrity and numerical diagnostics for the fixed 5k Hybrid run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.awac.hybrid import HybridAWACConfig, HybridActor, HybridCritic


def stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("run", type=Path); args = parser.parse_args()
    run = args.run.resolve(); final = json.loads((run / "final_report.json").read_text())
    rows = [json.loads(line) for line in (run / "training_metrics.jsonl").read_text().splitlines()]
    required = {
        "actor", "critic_q1", "critic_q2", "target_q1", "target_q2", "actor_optimizer",
        "critic_q1_optimizer", "critic_q2_optimizer", "observation_mean", "observation_std",
        "training_config", "reward_config", "dataset_report_sha256", "dataset_files_sha256",
        "hybrid_bc_checkpoint", "hybrid_bc_checkpoint_sha256",
    }
    checkpoint_integrity = {}
    for step, record in final["hybrid_awac_checkpoints"].items():
        payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
        missing = sorted(required - set(payload))
        checkpoint_integrity[step] = {
            "required_payload_complete": not missing, "missing": missing,
            "observation_mean_shape": list(payload["observation_mean"].shape),
            "observation_std_shape": list(payload["observation_std"].shape),
        }
        if missing or tuple(payload["observation_mean"].shape) != (43,):
            raise RuntimeError(f"invalid Hybrid checkpoint {step}")
    names = [key for key in rows[0] if key != "step"]
    config = HybridAWACConfig(); actor = HybridActor(config); critic = HybridCritic(config)
    diagnostics = {
        "updates": len(rows), "nonfinite_metric_count": int(sum(
            not np.isfinite(value) for row in rows for value in row.values()
        )),
        "q_explosion": bool(max(abs(row["q1_mean"]) for row in rows) > 1e3 or max(abs(row["q2_mean"]) for row in rows) > 1e3),
        "weight_cap": config.max_advantage_weight,
        "observed_weight_max": max(row["awac_weight_max"] for row in rows),
        "metric_global": {name: stats([row[name] for row in rows]) for name in names},
        "checkpoint_integrity": checkpoint_integrity,
        "network": {
            "actor": "43 -> 256 ReLU x4 -> Gaussian6 + Bernoulli1",
            "critic_each": "50 -> 256 ReLU x4 -> scalar",
            "actor_parameters": sum(parameter.numel() for parameter in actor.parameters()),
            "critic_parameters_each": sum(parameter.numel() for parameter in critic.parameters()),
        },
    }
    (run / "diagnostics_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    summary = {
        "dataset": {key: final["dataset"][key] for key in (
            "episode_count", "transition_count", "splits", "episodes_by_category",
            "observation", "continuous_action", "gripper_action",
        )},
        "network": diagnostics["network"], "training_config": final["training_config"],
        "hybrid_bc_validation": final["hybrid_bc_metrics"]["validation"],
        "closed_loop": {
            "legacy_BC": final["comparison"]["BC_baseline"],
            "old_AWAC": final["comparison"]["old_AWAC"],
            "Hybrid_BC": final["hybrid_bc_closed_loop"],
            "Hybrid_AWAC": {
                step: record["closed_loop"] for step, record in final["hybrid_awac_checkpoints"].items()
            },
        },
        "training_windows": {
            step: record["training_window_last_500"] for step, record in final["hybrid_awac_checkpoints"].items()
        },
        "answers": final["answers"], "stop_rule_triggered": final["stop_rule_triggered"],
        "online_awac_started": False,
    }
    (run / "final_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"answers": summary["answers"], "diagnostics": {
        "nonfinite": diagnostics["nonfinite_metric_count"], "q_explosion": diagnostics["q_explosion"],
        "observed_weight_max": diagnostics["observed_weight_max"],
    }}, indent=2))


if __name__ == "__main__": main()
