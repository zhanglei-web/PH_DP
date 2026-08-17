#!/usr/bin/env python3
"""Continue frozen geometric-state Offline Hybrid AWAC from 10k to 25k."""

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


SOURCE_RUN = Path("outputs/awac_training/awac_v3_geometric_milestone_state_20260814T150000Z")
SOURCE_CHECKPOINT = SOURCE_RUN / "checkpoints/hybrid_awac_step_10000.pt"
DATASET = Path("outputs/awac_dataset/awac_v3_geometric_milestone_state")
CHECKPOINT_STEPS = (12_500, 15_000, 17_500, 20_000, 22_500, 25_000)
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
            "next_observation": torch.from_numpy(np.asarray(data["next_obs"], np.float32)),
            "continuous": torch.from_numpy(np.asarray(data["continuous_action"], np.float32)),
            "gripper": torch.from_numpy(np.asarray(data["gripper_action"], np.float32)).unsqueeze(1),
            "stage": torch.from_numpy(np.asarray(data["expert_stage"], np.int64)),
        }


@torch.no_grad()
def validation_metrics(trainer, arrays: dict[str, torch.Tensor]) -> dict[str, Any]:
    was_training = trainer.actor.training
    trainer.actor.eval()
    observation = trainer.normalize(arrays["observation"].to(trainer.device))
    result = actor_metrics(
        trainer.actor, observation, arrays["continuous"].to(trainer.device),
        arrays["gripper"].to(trainer.device), arrays["stage"].to(trainer.device),
    )
    trainer.actor.train(was_training)
    return result


def values(tensor: torch.Tensor) -> dict[str, float]:
    array = tensor.detach().cpu().numpy().astype(np.float64)
    return {"mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max())}


@torch.no_grad()
def release_retreat_diagnostics(trainer, arrays: dict[str, torch.Tensor], seed: int) -> dict[str, Any]:
    observation = arrays["observation"].to(trainer.device)
    normalized = trainer.normalize(observation)
    continuous = arrays["continuous"].to(trainer.device)
    gripper = arrays["gripper"].to(trainer.device)
    mask = (observation[:, 46] > .5) & (observation[:, 47] <= .5)
    devices = [trainer.device.index or 0] if trainer.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        policy_continuous, policy_gripper, _ = trainer.actor.sample(normalized)
    dataset_q = torch.minimum(
        trainer.q1(normalized, continuous, gripper),
        trainer.q2(normalized, continuous, gripper),
    ).squeeze(1)
    policy_q = torch.minimum(
        trainer.q1(normalized, policy_continuous, policy_gripper),
        trainer.q2(normalized, policy_continuous, policy_gripper),
    ).squeeze(1)
    action, _binary, close_probability = trainer.actor.deterministic_action(normalized)
    return {
        "transitions": int(mask.sum()),
        "dataset_q": values(dataset_q[mask]),
        "advantage": values((dataset_q - policy_q)[mask]),
        "actor_dz_mean": float(action[mask, 2].mean()),
        "actor_dz_positive_ratio": float((action[mask, 2] > 0).float().mean()),
        "gripper_close_probability": values(close_probability[mask].squeeze(1)),
        "actor_continuous_mean": action[mask].mean(0).cpu().numpy().astype(float).tolist(),
    }


def metric_stats(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {name: {
        "mean": float(np.mean([row[name] for row in rows])),
        "std": float(np.std([row[name] for row in rows])),
        "min": float(np.min([row[name] for row in rows])),
        "max": float(np.max([row[name] for row in rows])),
    } for name in rows[0] if name != "step"}


def closed_loop_summary(value: dict[str, Any]) -> dict[str, Any]:
    place = int(value["place_success"]["count"])
    return {
        "success": int(value["task_success"]),
        "grasp": int(value["grasp_success"]["count"]),
        "lift": int(value["lift_success"]["count"]),
        "transport": int(value["transport_success"]["count"]),
        "place": place, "release": place,
        "retreat": int(value["retreat_success"]["count"]),
        "illegal_drop": int(value["illegal_drop"]["count"]),
        "ik_failure": int(value["ik_failure"]["count"]),
        "timeout": int(value["timeout"]["count"]),
        "average_return": float(value["average_episode_return"]),
        "place_to_success": float(value["task_success"] / max(place, 1)),
    }


def optimizer_steps(state: dict[str, Any]) -> set[int]:
    return {int(value["step"]) for value in state["state"].values() if "step" in value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/awac_training"))
    args = parser.parse_args()
    run = (args.output_root / f"awac_v3_geometric_milestone_state_offline25k_{args.run_id}").resolve()
    checkpoint_dir = run / "checkpoints"; evaluation_dir = run / "closed_loop"
    checkpoint_dir.mkdir(parents=True, exist_ok=False); evaluation_dir.mkdir()

    source_checkpoint = SOURCE_CHECKPOINT.resolve(); dataset = DATASET.resolve()
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    report = json.loads((dataset / "report.json").read_text())
    if source_payload.get("format_version") != "offline_awac_v3_geometric_milestone_state":
        raise RuntimeError("continuation refuses non-geometric or Online checkpoint")
    if source_payload["step"] != 10_000 or source_payload["legacy_rule_milestones_used"]:
        raise RuntimeError("continuation requires clean geometric 10k checkpoint")
    for name in ("actor_optimizer", "critic_q1_optimizer", "critic_q2_optimizer"):
        if optimizer_steps(source_payload[name]) != {10_000}:
            raise RuntimeError(f"{name} is not at step 10,000")
    if source_payload["dataset_report_sha256"] != sha256(dataset / "report.json"):
        raise RuntimeError("dataset report hash mismatch")
    if source_payload["dataset_files_sha256"] != {
            split: sha256(dataset / f"{split}.npz") for split in ("train", "validation")}:
        raise RuntimeError("dataset file hash mismatch")
    if (report["episode_count"], report["transition_count"],
            report["splits"]["train"]["transitions"],
            report["splits"]["validation"]["transitions"],
            report["episodes_by_category"]["delayed_recovery"]) != (1234, 150406, 135237, 15169, 0):
        raise RuntimeError("frozen dataset invariants failed")
    reward_config = AWACRewardV1Config(**source_payload["reward_config"])
    if asdict(reward_config) != asdict(AWACRewardV1Config()):
        raise RuntimeError("Reward V1 changed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer, _ = restore_hybrid_awac_trainer(source_checkpoint, device=device)
    if trainer.step != 10_000 or trainer.config.observation_dim != 48:
        raise RuntimeError("trainer did not restore geometric 10k state")
    if any(group["weight_decay"] != 0 for optimizer in (
            trainer.actor_optimizer, trainer.q1_optimizer, trainer.q2_optimizer)
           for group in optimizer.param_groups):
        raise RuntimeError("optimizer continuity mismatch")
    replay = HybridReplay(dataset / "train.npz", device)
    if len(replay) != 135_237:
        raise RuntimeError("offline replay size mismatch")
    validation = load_validation(dataset / "validation.npz")

    # The source checkpoint predates RNG persistence. Reconstruct its uniform replay
    # generator position exactly; seed other stochastic draws deterministically.
    random.seed(trainer.config.seed + 10_000)
    np.random.seed(trainer.config.seed + 10_000)
    torch.manual_seed(trainer.config.seed + 10_000)
    trainer.generator.manual_seed(trainer.config.seed + 1)
    for _ in range(10_000):
        torch.randint(len(replay), (trainer.config.batch_size,),
                      generator=trainer.generator, device=device)

    metadata = {
        "format_version": "offline_awac_v3_geometric_milestone_state",
        "reward_version": source_payload["reward_version"],
        "reward_config": source_payload["reward_config"],
        "dataset_report_sha256": source_payload["dataset_report_sha256"],
        "dataset_files_sha256": source_payload["dataset_files_sha256"],
        "state_definition": source_payload["state_definition"],
        "legacy_rule_milestones_used": False,
        "online_transition_count": 0,
        "hybrid_bc_checkpoint": source_payload["hybrid_bc_checkpoint"],
        "hybrid_bc_checkpoint_sha256": source_payload["hybrid_bc_checkpoint_sha256"],
        "continuation": {
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256(source_checkpoint),
            "source_step": 10_000, "target_step": 25_000,
            "replay_transitions": len(replay), "sampling": "uniform",
            "offline_only": True,
            "source_rng_note": "source 10k checkpoint had no RNG state; replay generator advanced deterministically",
        },
    }

    source_final = json.loads((SOURCE_RUN / "final_report.json").read_text())
    table = dict(source_final["new_48d_curve"])
    results: dict[str, Any] = {
        "10000": source_final["offline_awac"]["10000"],
    }
    all_rows: list[dict[str, float]] = []; interval_rows: list[dict[str, float]] = []
    numeric_failure = None
    metrics_path = run / "training_metrics_10001_25000.jsonl"
    with metrics_path.open("w") as stream:
        for step in range(10_001, 25_001):
            try:
                row = trainer.update(replay.sample(trainer.config.batch_size, trainer.generator))
            except FloatingPointError as error:
                numeric_failure = {"step": step, "reason": str(error)}; break
            if (abs(row["q1_mean"]) > 1_000 or abs(row["q2_mean"]) > 1_000
                    or row["critic_loss_q1"] > 1e6 or row["critic_loss_q2"] > 1e6):
                numeric_failure = {"step": step, "reason": "Q/loss explosion", "metrics": row}; break
            all_rows.append(row); interval_rows.append(row); stream.write(json.dumps(row) + "\n")
            if step % 100 == 0: stream.flush()
            if step in CHECKPOINT_STEPS:
                checkpoint_metadata = {**metadata, "rng_state": {
                    "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "trainer_generator": trainer.generator.get_state(),
                }}
                path = checkpoint_dir / f"hybrid_awac_step_{step:05d}.pt"
                atomic_save(trainer.checkpoint(checkpoint_metadata), path)
                print(f"evaluating Offline Geometric Hybrid AWAC {step}", flush=True)
                evaluation = evaluate_policy(
                    HybridCheckpointPredictor(path), VALIDATION_SEEDS, reward_config)
                (evaluation_dir / f"hybrid_awac_step_{step:05d}.json").write_text(
                    json.dumps(evaluation, indent=2) + "\n")
                summary = closed_loop_summary(evaluation)
                diagnostic = release_retreat_diagnostics(
                    trainer, validation, trainer.config.seed + step)
                result = {
                    "checkpoint": str(path), "checkpoint_sha256": sha256(path),
                    "closed_loop": evaluation, "closed_loop_summary": summary,
                    "validation_actor_metrics": validation_metrics(trainer, validation),
                    "release_done_1_retreat_done_0": diagnostic,
                    "training_interval": {
                        "start_step": step - len(interval_rows) + 1, "end_step": step,
                        "updates": len(interval_rows), "metrics": metric_stats(interval_rows),
                    },
                }
                results[str(step)] = result; table[str(step)] = summary; interval_rows = []
                print(json.dumps({"step": step, **summary,
                                  "release_retreat": diagnostic}), flush=True)

    if numeric_failure:
        atomic_save(trainer.checkpoint({**metadata, "numeric_failure": numeric_failure}),
                    checkpoint_dir / f"numeric_stop_{trainer.step:05d}.pt")

    eligible = {step: value for step, value in table.items() if step != "Hybrid_BC"}
    best_step = max(eligible, key=lambda step: (
        eligible[step]["success"], eligible[step]["place_to_success"],
        eligible[step]["retreat"], -eligible[step]["timeout"],
        -eligible[step]["illegal_drop"], -eligible[step]["ik_failure"],
        eligible[step]["average_return"], -int(step),
    ))
    if int(best_step) <= 10_000:
        best_source = SOURCE_RUN / "checkpoints" / f"hybrid_awac_step_{int(best_step):05d}.pt"
    else:
        best_source = Path(results[best_step]["checkpoint"])
    shutil.copy2(best_source, run / "checkpoint_best.pt")
    diagnostics = {
        "updates_completed": len(all_rows), "start_optimizer_step": 10_000,
        "final_optimizer_step": trainer.step,
        "nonfinite_metric_count": int(sum(
            not np.isfinite(value) for row in all_rows for value in row.values())),
        "q_explosion": bool(any(abs(row["q1_mean"]) > 1_000 or abs(row["q2_mean"]) > 1_000
                                for row in all_rows)),
        "observed_weight_max": max((row["awac_weight_max"] for row in all_rows), default=None),
        "global_metrics_10k_to_25k": metric_stats(all_rows) if all_rows else {},
        "numeric_failure": numeric_failure,
    }
    final = {
        "status": "numeric_stop" if numeric_failure else "complete_25k",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "dataset": report,
        "offline_replay": {"transitions": len(replay), "online_transitions": 0,
                           "sampling": "uniform"},
        "training_config": {**asdict(trainer.config), "device": str(device),
                            "optimizer": "Adam", "actor_weight_decay": 0,
                            "continuation_updates": 15_000, "target_step": 25_000},
        "reward_version": source_payload["reward_version"],
        "reward_config": source_payload["reward_config"],
        "results": results, "comparison_table": table,
        "best_step": int(best_step),
        "best_checkpoint": str((run / "checkpoint_best.pt").resolve()),
        "diagnostics": diagnostics, "online_awac_started": False,
    }
    (run / "diagnostics_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (run / "final_report.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"run": str(run), "status": final["status"],
                      "best_step": final["best_step"], "curve": table,
                      "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
