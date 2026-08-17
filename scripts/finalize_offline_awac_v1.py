#!/usr/bin/env python3
"""Read-only final diagnostics for a completed or protectively stopped AWAC run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.evaluation import AWACCheckpointPredictor
from mujoco_shared_control.awac.offline import AWACCritic, AWACGaussianActor, OfflineAWACConfig


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/awac_dataset/awac_v1_formal_rule"))
    args = parser.parse_args()
    run = args.run.resolve()
    dataset = args.dataset.resolve()
    result = json.loads((run / "results.json").read_text())
    data_report = json.loads((dataset / "report.json").read_text())
    reward_audit = json.loads((dataset / "reward_audit.json").read_text())
    metrics = [json.loads(line) for line in (run / "training_metrics.jsonl").read_text().splitlines()]
    config = OfflineAWACConfig()
    required = {
        "actor", "critic_q1", "critic_q2", "target_q1", "target_q2",
        "actor_optimizer", "critic_q1_optimizer", "critic_q2_optimizer",
        "observation_mean", "observation_std", "training_config", "reward_config",
        "dataset_report_sha256", "dataset_manifest_sha256", "dataset_files_sha256",
    }
    validation: dict[str, Any] = {}
    with np.load(dataset / "validation.npz", allow_pickle=False) as data:
        observation = np.asarray(data["obs"], np.float32)
        target = np.asarray(data["action"], np.float32)
        stage = np.asarray(data["expert_stage"], int)
        for step, record in result["checkpoints"].items():
            path = Path(record["checkpoint"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            missing = required - set(payload)
            if missing:
                raise RuntimeError(f"checkpoint {step} missing fields: {sorted(missing)}")
            predictor = AWACCheckpointPredictor(path)
            predictions = []
            with torch.inference_mode():
                for start in range(0, len(observation), 4096):
                    normalized = (observation[start:start + 4096] - predictor.mean) / predictor.std
                    predictions.append(predictor.model.deterministic_action(torch.from_numpy(normalized)).numpy())
            action = np.concatenate(predictions)
            close = stage == 2
            validation[step] = {
                "checkpoint_sha256": _sha(path),
                "required_payload_complete": True,
                "action_mse": float(np.mean((action - target) ** 2)),
                "action_mae": float(np.mean(np.abs(action - target))),
                "action_min": float(action.min()), "action_max": float(action.max()),
                "close_gripper_stage_prediction_mean": float(action[close, 6].mean()),
                "close_gripper_stage_target_mean": float(target[close, 6].mean()),
            }

    metric_names = [key for key in metrics[0] if key != "step"]
    diagnostics = {
        "updates_completed": len(metrics),
        "metric_global": {name: _stats([row[name] for row in metrics]) for name in metric_names},
        "checkpoint_last_500_windows": {
            step: record["training_window_last_500"] for step, record in result["checkpoints"].items()
        },
        "offline_validation": validation,
        "protective_stop": result["protective_stop"],
        "q_explosion": bool(
            max(abs(row["q1_mean"]) for row in metrics) > 1e3
            or max(abs(row["q2_mean"]) for row in metrics) > 1e3
        ),
        "nonfinite_metrics": int(sum(not np.isfinite(value) for row in metrics for value in row.values())),
        "advantage_weight_cap": config.max_advantage_weight,
        "observed_weight_max": max(row["awac_weight_max"] for row in metrics),
    }
    (run / "diagnostics_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    actor = AWACGaussianActor(config)
    critic = AWACCritic(config)
    checkpoint_table = {
        step: {
            "success": record["closed_loop"]["task_success"],
            "success_rate": record["closed_loop"]["task_success_rate"],
            "grasp_rate": record["closed_loop"]["grasp_success"]["rate"],
            "lift_rate": record["closed_loop"]["lift_success"]["rate"],
            "transport_rate": record["closed_loop"]["transport_success"]["rate"],
            "place_rate": record["closed_loop"]["place_success"]["rate"],
            "illegal_drop_rate": record["closed_loop"]["illegal_drop"]["rate"],
            "ik_failure_rate": record["closed_loop"]["ik_failure"]["rate"],
            "timeout_rate": record["closed_loop"]["timeout"]["rate"],
            "average_episode_return": record["closed_loop"]["average_episode_return"],
        }
        for step, record in result["checkpoints"].items()
    }
    bc = result["bc_baseline"]
    final = {
        "status": result["status"],
        "dataset": {
            "episodes": data_report["episode_count"],
            "episodes_by_split": data_report["episodes_by_split"],
            "episodes_by_category": data_report["episodes_by_category"],
            "transitions": data_report["transition_count"],
            "transitions_by_split": data_report["transitions_by_split"],
            "report_sha256": _sha(dataset / "report.json"),
        },
        "reward_audit": {
            category: reward_audit["categories"][category]["episode_return"]
            for category in ("normal_success", "normal_recovery", "failure")
        },
        "network": {
            "actor": "42 -> 256 ReLU -> 256 ReLU -> 256 ReLU -> 256 ReLU -> mean/log_std -> tanh 7",
            "critic_each": "49 -> 256 ReLU -> 256 ReLU -> 256 ReLU -> 256 ReLU -> 1",
            "actor_parameters": sum(parameter.numel() for parameter in actor.parameters()),
            "critic_parameters_each": sum(parameter.numel() for parameter in critic.parameters()),
        },
        "training_config": json.loads((run / "training_config.json").read_text()),
        "updates_completed": result["updates_completed"],
        "requested_updates": result["requested_updates"],
        "protective_stop": result["protective_stop"],
        "bc_baseline": {
            "success": bc["task_success"], "success_rate": bc["task_success_rate"],
            "average_episode_return": bc["average_episode_return"],
            "grasp_rate": bc["grasp_success"]["rate"], "lift_rate": bc["lift_success"]["rate"],
            "transport_rate": bc["transport_success"]["rate"], "place_rate": bc["place_success"]["rate"],
        },
        "checkpoint_closed_loop": checkpoint_table,
        "selected_checkpoint": {
            "step": result["best_step"],
            "path": str(run / "checkpoint_best.pt"),
            "qualified_for_online": False,
            "note": "forensic best among evaluated AWAC checkpoints; 0% success and not deployment-qualified",
        },
        "online_fine_tuning_condition_met": False,
        "online_awac_started": False,
        "missing_checkpoints_due_to_protective_stop": [step for step in range(7500, 25001, 2500)],
    }
    (run / "final_report.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({
        "status": final["status"], "updates_completed": final["updates_completed"],
        "bc": final["bc_baseline"], "checkpoints": checkpoint_table,
        "online_fine_tuning_condition_met": False,
    }, indent=2))


if __name__ == "__main__":
    main()
