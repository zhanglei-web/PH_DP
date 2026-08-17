#!/usr/bin/env python3
"""Continue the frozen Offline Hybrid AWAC checkpoint from 5k to 25k."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.awac.evaluation import evaluate_policy
from mujoco_shared_control.awac.hybrid import HybridReplay, actor_metrics
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.online import restore_hybrid_awac_trainer
from mujoco_shared_control.awac.reward import AWACRewardV1Config


SOURCE_RUN = Path("outputs/awac_training/awac_v2_hybrid_20260814T110000Z")
SOURCE_CHECKPOINT = SOURCE_RUN / "checkpoints/hybrid_awac_step_05000.pt"
DATASET = Path("outputs/awac_dataset/awac_v2_hybrid_formal_rule")
CHECKPOINT_STEPS = (7_500, 10_000, 12_500, 15_000, 17_500, 20_000, 22_500, 25_000)
VALIDATION_SEEDS = list(range(300_000, 300_100))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_validation(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "observation": torch.from_numpy(np.asarray(data["obs"], np.float32)),
            "continuous": torch.from_numpy(np.asarray(data["continuous_action"], np.float32)),
            "gripper": torch.from_numpy(np.asarray(data["gripper_action"], np.float32)).unsqueeze(1),
            "stage": torch.from_numpy(np.asarray(data["expert_stage"], np.int64)),
        }


@torch.no_grad()
def validation_metrics(trainer, arrays: dict[str, torch.Tensor]) -> dict[str, Any]:
    actor = trainer.actor
    was_training = actor.training
    actor.eval()
    observation = trainer.normalize(arrays["observation"].to(trainer.device))
    result = actor_metrics(
        actor, observation, arrays["continuous"].to(trainer.device),
        arrays["gripper"].to(trainer.device), arrays["stage"].to(trainer.device),
    )
    actor.train(was_training)
    return result


def metric_stats(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    names = [name for name in rows[0] if name != "step"]
    return {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "std": float(np.std([row[name] for row in rows])),
            "min": float(np.min([row[name] for row in rows])),
            "max": float(np.max([row[name] for row in rows])),
        }
        for name in names
    }


def closed_loop_summary(value: dict[str, Any]) -> dict[str, Any]:
    place = int(value["place_success"]["count"])
    return {
        "success": int(value["task_success"]),
        "grasp": int(value["grasp_success"]["count"]),
        "lift": int(value["lift_success"]["count"]),
        "transport": int(value["transport_success"]["count"]),
        "place": place,
        "release": place,
        "retreat": int(value["retreat_success"]["count"]),
        "illegal_drop": int(value["illegal_drop"]["count"]),
        "ik_failure": int(value["ik_failure"]["count"]),
        "timeout": int(value["timeout"]["count"]),
        "average_return": float(value["average_episode_return"]),
        "place_to_success": float(value["task_success"] / max(place, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/awac_training"))
    args = parser.parse_args()
    run = (args.output_root / f"awac_v2_hybrid_offline25k_{args.run_id}").resolve()
    checkpoints = run / "checkpoints"
    evaluations = run / "closed_loop"
    checkpoints.mkdir(parents=True, exist_ok=False)
    evaluations.mkdir()

    source_checkpoint = SOURCE_CHECKPOINT.resolve()
    dataset = DATASET.resolve()
    report = json.loads((dataset / "report.json").read_text())
    if (
        report["episode_count"] != 1_234
        or report["transition_count"] != 150_406
        or report["splits"]["train"] != {
            **report["splits"]["train"], "episodes": 1_110, "transitions": 135_237,
        }
        or report["splits"]["validation"]["episodes"] != 124
        or report["splits"]["validation"]["transitions"] != 15_169
        or report["episodes_by_category"]["delayed_recovery"] != 0
    ):
        raise RuntimeError("Offline 25k continuation refused: frozen dataset mismatch")

    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source_payload.get("format_version") != "offline_awac_v2_hybrid":
        raise RuntimeError("Offline continuation refuses Online checkpoints")
    if source_payload["step"] != 5_000:
        raise RuntimeError("Offline continuation requires optimizer step 5,000")
    if source_payload["dataset_files_sha256"] != {
        split: sha256(dataset / f"{split}.npz") for split in ("train", "validation")
    }:
        raise RuntimeError("Offline continuation refused: dataset hash mismatch")
    reward_config = AWACRewardV1Config(**source_payload["reward_config"])
    if asdict(reward_config) != asdict(AWACRewardV1Config()):
        raise RuntimeError("Offline continuation refused: Reward V1 mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer, _ = restore_hybrid_awac_trainer(source_checkpoint, device=device)
    if trainer.step != 5_000:
        raise RuntimeError("restored optimizer step is not 5,000")
    if trainer.actor_optimizer.param_groups[0]["weight_decay"] != 0:
        raise RuntimeError("checkpoint continuity requires actor weight_decay=0")
    replay = HybridReplay(dataset / "train.npz", device)
    if len(replay) != 135_237:
        raise RuntimeError("Offline replay must contain exactly 135,237 transitions")
    validation = load_validation(dataset / "validation.npz")

    # The historical checkpoint predates RNG-state persistence. Seed this
    # continuation deterministically and persist every subsequent RNG state.
    random.seed(trainer.config.seed + 3)
    np.random.seed(trainer.config.seed + 3)
    torch.manual_seed(trainer.config.seed + 3)
    trainer.generator.manual_seed(trainer.config.seed + 1)
    for _ in range(5_000):
        torch.randint(
            len(replay), (trainer.config.batch_size,),
            generator=trainer.generator, device=device,
        )

    metadata = {
        "reward_version": source_payload["reward_version"],
        "reward_config": source_payload["reward_config"],
        "dataset_report_sha256": source_payload["dataset_report_sha256"],
        "dataset_files_sha256": source_payload["dataset_files_sha256"],
        "hybrid_bc_checkpoint": source_payload["hybrid_bc_checkpoint"],
        "hybrid_bc_checkpoint_sha256": source_payload["hybrid_bc_checkpoint_sha256"],
        "continuation": {
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256(source_checkpoint),
            "source_step": 5_000,
            "target_step": 25_000,
            "offline_only": True,
            "online_transition_count": 0,
            "replay_transitions": len(replay),
        },
    }

    shutil.copy2(source_checkpoint, checkpoints / "hybrid_awac_step_05000.pt")
    print("evaluating frozen Offline Hybrid AWAC 5000", flush=True)
    closed_loop: dict[str, Any] = {}
    baseline = evaluate_policy(
        HybridCheckpointPredictor(source_checkpoint), VALIDATION_SEEDS, reward_config,
    )
    closed_loop["5000"] = baseline
    (evaluations / "hybrid_awac_step_05000.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print(json.dumps({"step": 5_000, **closed_loop_summary(baseline)}), flush=True)

    results: dict[str, Any] = {
        "5000": {
            "checkpoint": str(source_checkpoint),
            "closed_loop": baseline,
            "closed_loop_summary": closed_loop_summary(baseline),
            "validation_actor_metrics": validation_metrics(trainer, validation),
        }
    }
    all_rows: list[dict[str, float]] = []
    interval_rows: list[dict[str, float]] = []
    metrics_path = run / "training_metrics_05001_25000.jsonl"
    numeric_failure: dict[str, Any] | None = None
    with metrics_path.open("w") as stream:
        for step in range(5_001, 25_001):
            try:
                row = trainer.update(replay.sample(trainer.config.batch_size, trainer.generator))
            except FloatingPointError as error:
                numeric_failure = {"step": step, "reason": str(error)}
                break
            if (
                abs(row["q1_mean"]) > 1_000 or abs(row["q2_mean"]) > 1_000
                or row["critic_loss_q1"] > 1e6 or row["critic_loss_q2"] > 1e6
            ):
                numeric_failure = {"step": step, "reason": "Q/loss numerical explosion", "metrics": row}
                break
            all_rows.append(row)
            interval_rows.append(row)
            stream.write(json.dumps(row) + "\n")
            if step % 100 == 0:
                stream.flush()
            if step in CHECKPOINT_STEPS:
                checkpoint_metadata = {
                    **metadata,
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "trainer_generator": trainer.generator.get_state(),
                    },
                }
                path = checkpoints / f"hybrid_awac_step_{step:05d}.pt"
                atomic_save(trainer.checkpoint(checkpoint_metadata), path)
                print(f"evaluating Offline Hybrid AWAC {step}", flush=True)
                evaluation = evaluate_policy(
                    HybridCheckpointPredictor(path), VALIDATION_SEEDS, reward_config,
                )
                closed_loop[str(step)] = evaluation
                (evaluations / f"hybrid_awac_step_{step:05d}.json").write_text(
                    json.dumps(evaluation, indent=2) + "\n"
                )
                summary = closed_loop_summary(evaluation)
                actor_validation = validation_metrics(trainer, validation)
                results[str(step)] = {
                    "checkpoint": str(path), "checkpoint_sha256": sha256(path),
                    "closed_loop": evaluation, "closed_loop_summary": summary,
                    "validation_actor_metrics": actor_validation,
                    "training_interval": {
                        "start_step": step - len(interval_rows) + 1,
                        "end_step": step,
                        "updates": len(interval_rows),
                        "metrics": metric_stats(interval_rows),
                    },
                }
                interval_rows = []
                print(json.dumps({"step": step, **summary}), flush=True)

    if numeric_failure is not None:
        emergency = checkpoints / f"numeric_stop_step_{trainer.step:05d}.pt"
        atomic_save(trainer.checkpoint({**metadata, "numeric_failure": numeric_failure}), emergency)

    source_final = json.loads((SOURCE_RUN / "final_report.json").read_text())
    table = {
        "Hybrid_BC": closed_loop_summary(source_final["hybrid_bc_closed_loop"]),
        "2500": closed_loop_summary(
            source_final["hybrid_awac_checkpoints"]["2500"]["closed_loop"]
        ),
        **{step: value["closed_loop_summary"] for step, value in results.items()},
    }
    eligible = {step: value for step, value in table.items() if step != "Hybrid_BC"}
    best_step = max(eligible, key=lambda step: (
        eligible[step]["success"], eligible[step]["place_to_success"],
        -eligible[step]["illegal_drop"], -eligible[step]["ik_failure"],
        -eligible[step]["timeout"], eligible[step]["average_return"], -int(step),
    ))
    best_source = (
        source_checkpoint if best_step == "5000"
        else SOURCE_RUN / "checkpoints/hybrid_awac_step_02500.pt" if best_step == "2500"
        else Path(results[best_step]["checkpoint"])
    )
    shutil.copy2(best_source, run / "checkpoint_best.pt")

    diagnostics = {
        "updates_completed": len(all_rows),
        "start_optimizer_step": 5_000,
        "final_optimizer_step": trainer.step,
        "nonfinite_metric_count": int(sum(
            not np.isfinite(value) for row in all_rows for value in row.values()
        )),
        "q_explosion": bool(any(
            abs(row["q1_mean"]) > 1_000 or abs(row["q2_mean"]) > 1_000
            for row in all_rows
        )),
        "observed_weight_max": max(row["awac_weight_max"] for row in all_rows),
        "global_metrics_5k_to_25k": metric_stats(all_rows),
        "numeric_failure": numeric_failure,
    }
    final = {
        "status": "numeric_stop" if numeric_failure else "complete_25k",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "dataset": report,
        "offline_replay": {"transitions": len(replay), "online_transitions": 0, "sampling": "uniform"},
        "training_config": {
            **asdict(trainer.config), "device": str(device), "optimizer": "Adam",
            "actor_weight_decay": trainer.actor_optimizer.param_groups[0]["weight_decay"],
            "continuation_updates": 20_000, "target_optimizer_step": 25_000,
        },
        "reward_version": source_payload["reward_version"],
        "reward_config": source_payload["reward_config"],
        "results": results,
        "comparison_table": table,
        "best_step": int(best_step),
        "best_checkpoint": str((run / "checkpoint_best.pt").resolve()),
        "diagnostics": diagnostics,
        "online_awac_started": False,
    }
    (run / "diagnostics_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (run / "final_report.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({
        "run": str(run), "status": final["status"], "best_step": final["best_step"],
        "closed_loop": table, "diagnostics": {
            "nonfinite": diagnostics["nonfinite_metric_count"],
            "q_explosion": diagnostics["q_explosion"],
            "observed_weight_max": diagnostics["observed_weight_max"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
