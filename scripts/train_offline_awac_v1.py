#!/usr/bin/env python3
"""Train and closed-loop validate frozen Offline AWAC-v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.evaluation import (
    AWACCheckpointPredictor, BCPredictorAdapter, evaluate_policy,
)
from mujoco_shared_control.awac.offline import (
    OfflineAWACConfig, OfflineAWACTrainer, OfflineReplayBuffer,
    observation_statistics,
)
from mujoco_shared_control.awac.reward import AWACRewardV1Config


CHECKPOINT_STEPS = tuple(range(2_500, 25_001, 2_500))
DATASET = Path("outputs/awac_dataset/awac_v1_formal_rule")
BC_CHECKPOINT = Path("outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt")
VALIDATION_SEEDS = list(range(300_000, 300_100))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    torch.save(value, temporary)
    temporary.replace(path)


def _window(rows: list[dict[str, float]], size: int = 500) -> dict[str, float]:
    chosen = rows[-size:]
    keys = [key for key in chosen[0] if key != "step"]
    return {key: float(np.mean([row[key] for row in chosen])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/awac_training"))
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    args = parser.parse_args()
    run = (args.output_root / f"offline_awac_v1_{args.run_id}").resolve()
    checkpoints = run / "checkpoints"
    evaluations = run / "closed_loop"
    checkpoints.mkdir(parents=True, exist_ok=False)
    evaluations.mkdir()

    dataset = DATASET.resolve()
    report = json.loads((dataset / "report.json").read_text())
    if report["episode_count"] != 1234 or report["episodes_by_category"]["delayed_recovery"] != 0:
        raise RuntimeError("training refused: dataset is not frozen AWAC-v1/1234")
    reward_config = AWACRewardV1Config(**report["reward_config"])
    config = OfflineAWACConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    mean, std = observation_statistics(dataset / "train.npz")
    replay = OfflineReplayBuffer(dataset / "train.npz", device=device)
    trainer = OfflineAWACTrainer(config, mean, std, device=device)
    metadata = {
        "reward_version": "awac_reward_v1",
        "reward_config": asdict(reward_config),
        "dataset_report_sha256": _sha(dataset / "report.json"),
        "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
        "dataset_files_sha256": {name: _sha(dataset / f"{name}.npz") for name in ("train", "validation")},
    }
    config_document = {
        **asdict(config), "optimizer": "Adam", "device": str(device),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "validation_seeds": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[args.evaluation_episodes - 1]],
        "validation_episodes": args.evaluation_episodes,
        "actor_objective": "-E[clip(exp((Q_data-V_pi)/lambda),max=20) * log_pi(a_data|s)]",
        "critic_objective": "twin Bellman MSE; target=min(target_Q1,target_Q2), no entropy term",
        "online_awac": False,
    }
    (run / "training_config.json").write_text(json.dumps(config_document, indent=2) + "\n")
    (run / "dataset_source.json").write_text(json.dumps({**metadata, "report": report}, indent=2) + "\n")
    (run / "normalization.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(), "source": "train.npz only"}, indent=2) + "\n")

    seeds = VALIDATION_SEEDS[:args.evaluation_episodes]
    print("evaluating BC baseline", flush=True)
    bc_evaluation = evaluate_policy(BCPredictorAdapter(BC_CHECKPOINT), seeds, reward_config)
    (evaluations / "bc_baseline.json").write_text(json.dumps(bc_evaluation, indent=2) + "\n")

    history: list[dict[str, float]] = []
    checkpoint_results: dict[str, Any] = {}
    protective_stop: dict[str, Any] | None = None
    consecutive_catastrophic = 0
    metrics_stream = (run / "training_metrics.jsonl").open("w")
    try:
        for step in range(1, config.offline_updates + 1):
            row = trainer.update(replay.sample(config.batch_size, trainer.generator))
            history.append(row)
            metrics_stream.write(json.dumps(row) + "\n")
            if step % 100 == 0:
                metrics_stream.flush()
            if abs(row["q1_mean"]) > 1e3 or abs(row["q2_mean"]) > 1e3 or max(row["critic_loss_q1"], row["critic_loss_q2"]) > 1e6:
                protective_stop = {"step": step, "reason": "Q explosion threshold exceeded", "metrics": row}
                break
            if step in CHECKPOINT_STEPS:
                checkpoint_path = checkpoints / f"step_{step:05d}.pt"
                _atomic_torch_save(trainer.checkpoint_payload(metadata), checkpoint_path)
                print(f"evaluating checkpoint {step}", flush=True)
                evaluation = evaluate_policy(AWACCheckpointPredictor(checkpoint_path), seeds, reward_config)
                (evaluations / f"step_{step:05d}.json").write_text(json.dumps(evaluation, indent=2) + "\n")
                window = _window(history)
                checkpoint_results[str(step)] = {
                    "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha(checkpoint_path),
                    "training_window_last_500": window, "closed_loop": evaluation,
                }
                print(
                    f"step={step} q1={window['q1_mean']:.4f} actor={window['actor_loss']:.4f} "
                    f"weight_p99={window['awac_weight_p99']:.4f} success={evaluation['task_success']}/{len(seeds)}",
                    flush=True,
                )
                catastrophic = evaluation["task_success_rate"] <= max(0.0, bc_evaluation["task_success_rate"] - 0.15)
                consecutive_catastrophic = consecutive_catastrophic + 1 if catastrophic else 0
                if consecutive_catastrophic >= 2:
                    protective_stop = {
                        "step": step,
                        "reason": "closed-loop success at least 15 percentage points below BC for two checkpoints",
                        "bc_success_rate": bc_evaluation["task_success_rate"],
                        "awac_success_rate": evaluation["task_success_rate"],
                    }
                    break
    except FloatingPointError as error:
        protective_stop = {"step": trainer.step, "reason": str(error)}
    finally:
        metrics_stream.close()

    if checkpoint_results:
        best_step = max(
            checkpoint_results,
            key=lambda value: (
                checkpoint_results[value]["closed_loop"]["task_success_rate"],
                checkpoint_results[value]["closed_loop"]["average_episode_return"],
                -int(value),
            ),
        )
        shutil.copy2(checkpoint_results[best_step]["checkpoint"], run / "checkpoint_best.pt")
    else:
        best_step = None
    result = {
        "status": "protective_stop" if protective_stop else "complete",
        "updates_completed": trainer.step,
        "requested_updates": config.offline_updates,
        "protective_stop": protective_stop,
        "bc_baseline": bc_evaluation,
        "checkpoints": checkpoint_results,
        "best_step": None if best_step is None else int(best_step),
        "selection_rule": "maximum deterministic closed-loop validation success; tie average Reward-V1 return; then earlier checkpoint",
        "online_awac_started": False,
    }
    (run / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "run": str(run), "status": result["status"], "updates_completed": trainer.step,
        "bc_success": bc_evaluation["task_success_rate"], "best_step": result["best_step"],
        "checkpoint_success": {step: value["closed_loop"]["task_success_rate"] for step, value in checkpoint_results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
