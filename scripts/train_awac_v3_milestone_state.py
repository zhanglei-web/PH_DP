#!/usr/bin/env python3
"""Train 48-D milestone-state Hybrid BC and Offline AWAC through 10k."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mujoco_shared_control.awac.evaluation import evaluate_policy
from mujoco_shared_control.awac.hybrid import (
    HybridAWACConfig, HybridAWACTrainer, HybridActor, HybridReplay,
    actor_metrics, observation_statistics,
)
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.reward import AWACRewardV1Config


DATASET = Path("outputs/awac_dataset/awac_v3_geometric_milestone_state")
OLD_43D_RUN = Path("outputs/awac_training/awac_v2_hybrid_20260814T110000Z")
OLD_43D_LONG_RUN = Path("outputs/awac_training/awac_v2_hybrid_offline25k_20260814T160000Z")
SEEDS = list(range(300_000, 300_100))
CHECKPOINTS = (2_500, 5_000, 7_500, 10_000)
MILESTONE_NAMES = ("grasp_done", "lift_done", "transport_done", "release_done", "retreat_done")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    torch.save(value, temporary)
    temporary.replace(path)


def load_split(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "observation": torch.from_numpy(np.asarray(data["obs"], np.float32)),
            "next_observation": torch.from_numpy(np.asarray(data["next_obs"], np.float32)),
            "continuous": torch.from_numpy(np.asarray(data["continuous_action"], np.float32)),
            "gripper": torch.from_numpy(np.asarray(data["gripper_action"], np.float32)).unsqueeze(1),
            "stage": torch.from_numpy(np.asarray(data["expert_stage"], np.int64)),
        }


@torch.no_grad()
def validation_loss(actor, arrays, mean, std, config) -> float:
    total = 0.0
    for start in range(0, len(arrays["observation"]), 4096):
        observation = (arrays["observation"][start:start + 4096] - mean) / std
        joint, _continuous, _gripper = actor.dataset_log_prob(
            observation, arrays["continuous"][start:start + 4096],
            arrays["gripper"][start:start + 4096], config.beta_gripper,
        )
        total += float((-joint).sum())
    return total / len(arrays["observation"])


@torch.no_grad()
def split_metrics(actor, arrays, mean, std) -> dict[str, Any]:
    normalized = (arrays["observation"] - mean) / std
    return actor_metrics(actor, normalized, arrays["continuous"], arrays["gripper"], arrays["stage"])


def values(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().cpu().numpy().astype(np.float64)
    return {
        "mean": float(array.mean()), "std": float(array.std()),
        "min": float(array.min()), "max": float(array.max()),
    }


@torch.no_grad()
def milestone_diagnostics(
    trainer: HybridAWACTrainer, arrays: dict[str, torch.Tensor], diagnostic_seed: int,
) -> dict[str, Any]:
    observation = arrays["observation"].to(trainer.device)
    next_observation = arrays["next_observation"].to(trainer.device)
    continuous = arrays["continuous"].to(trainer.device)
    gripper = arrays["gripper"].to(trainer.device)
    normalized = trainer.normalize(observation)
    normalized_next = trainer.normalize(next_observation)
    devices = [trainer.device.index or 0] if trainer.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(diagnostic_seed)
        policy_continuous, policy_gripper, _ = trainer.actor.sample(normalized)
    dataset_q = torch.minimum(
        trainer.q1(normalized, continuous, gripper),
        trainer.q2(normalized, continuous, gripper),
    ).squeeze(1)
    policy_q = torch.minimum(
        trainer.q1(normalized, policy_continuous, policy_gripper),
        trainer.q2(normalized, policy_continuous, policy_gripper),
    ).squeeze(1)
    advantage = dataset_q - policy_q
    actor_continuous, _actor_gripper, close_probability = trainer.actor.deterministic_action(normalized)

    def group(mask: torch.Tensor) -> dict[str, Any]:
        count = int(mask.sum())
        if not count:
            return {"transitions": 0, "available": False}
        action = actor_continuous[mask]
        return {
            "transitions": count, "available": True,
            "dataset_q": values(dataset_q[mask]),
            "advantage": values(advantage[mask]),
            "actor_continuous_mean": action.mean(0).cpu().numpy().astype(float).tolist(),
            "actor_continuous_abs_mean": action.abs().mean(0).cpu().numpy().astype(float).tolist(),
            "actor_dz_mean": float(action[:, 2].mean()),
            "actor_positive_dz_ratio": float((action[:, 2] > 0).float().mean()),
            "gripper_close_probability": values(close_probability[mask].squeeze(1)),
        }

    milestones = observation[:, 43:48] > 0.5
    groups = {
        "release_done_0": group(~milestones[:, 3]),
        "release_done_1": group(milestones[:, 3]),
        "retreat_done_0": group(~milestones[:, 4]),
        "retreat_done_1": group(milestones[:, 4]),
        "release_done_1_retreat_done_0": group(milestones[:, 3] & ~milestones[:, 4]),
    }
    next_retreat = next_observation[:, 47] > 0.5
    if int(next_retreat.sum()):
        terminal_state = normalized_next[next_retreat]
        terminal_action, _terminal_gripper, terminal_close = trainer.actor.deterministic_action(terminal_state)
        terminal_q = torch.minimum(
            trainer.q1(terminal_state, terminal_action, (terminal_close >= .5).float()),
            trainer.q2(terminal_state, terminal_action, (terminal_close >= .5).float()),
        ).squeeze(1)
        groups["next_state_retreat_done_1_terminal_only"] = {
            "transitions": int(next_retreat.sum()),
            "note": "retreat_done=1 occurs only in terminal next_obs; no dataset action/advantage exists",
            "policy_q": values(terminal_q),
            "actor_continuous_mean": terminal_action.mean(0).cpu().numpy().astype(float).tolist(),
            "gripper_close_probability": values(terminal_close.squeeze(1)),
        }
    return groups


def closed_loop_summary(value: dict[str, Any]) -> dict[str, Any]:
    place = int(value["place_success"]["count"])
    return {
        "success": int(value["task_success"]), "grasp": int(value["grasp_success"]["count"]),
        "lift": int(value["lift_success"]["count"]), "transport": int(value["transport_success"]["count"]),
        "place": place, "release": place, "retreat": int(value["retreat_success"]["count"]),
        "illegal_drop": int(value["illegal_drop"]["count"]),
        "ik_failure": int(value["ik_failure"]["count"]), "timeout": int(value["timeout"]["count"]),
        "average_return": float(value["average_episode_return"]),
        "place_to_success": float(value["task_success"] / max(place, 1)),
    }


def metric_stats(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {name: {
        "mean": float(np.mean([row[name] for row in rows])),
        "std": float(np.std([row[name] for row in rows])),
        "min": float(np.min([row[name] for row in rows])),
        "max": float(np.max([row[name] for row in rows])),
    } for name in rows[0] if name != "step"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/awac_training"))
    args = parser.parse_args()
    run = (args.output_root / f"awac_v3_geometric_milestone_state_{args.run_id}").resolve()
    checkpoint_dir = run / "checkpoints"; evaluation_dir = run / "closed_loop"
    checkpoint_dir.mkdir(parents=True, exist_ok=False); evaluation_dir.mkdir()
    dataset = DATASET.resolve(); report = json.loads((dataset / "report.json").read_text())
    if (
        report["episode_count"] != 1_234 or report["transition_count"] != 150_406
        or report["splits"]["train"]["transitions"] != 135_237
        or report["splits"]["validation"]["transitions"] != 15_169
        or report["episodes_by_category"]["delayed_recovery"] != 0
    ):
        raise RuntimeError("AWAC-v3 training refused: dataset is not frozen 1234/150406 corpus")
    config = HybridAWACConfig(observation_dim=48, awac_updates=10_000)
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(dataset / "train.npz"); validation = load_split(dataset / "validation.npz")
    mean_np, std_np = observation_statistics(dataset / "train.npz")
    if mean_np.shape != (48,) or std_np.shape != (48,):
        raise RuntimeError("AWAC-v3 normalization is not 48-D")
    mean = torch.from_numpy(mean_np); std = torch.from_numpy(std_np)
    normalization_report = {
        "mean": mean_np.astype(float).tolist(), "std": std_np.astype(float).tolist(),
        "milestone_mean": mean_np[43:48].astype(float).tolist(),
        "milestone_std": std_np[43:48].astype(float).tolist(),
        "epsilon_floor": 1e-6,
        "epsilon_floored_indices": np.flatnonzero(std_np <= np.float32(1e-6)).astype(int).tolist(),
    }
    (run / "normalization.json").write_text(json.dumps(normalization_report, indent=2) + "\n")

    actor = HybridActor(config).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train["observation"], train["continuous"], train["gripper"]),
        batch_size=config.batch_size, shuffle=True, generator=generator,
    )
    mean_device = mean.to(device); std_device = std.to(device)
    best_loss = float("inf"); best_epoch = 0; stale = 0; history = []
    best_state = deepcopy(actor.state_dict())
    for epoch in range(1, config.bc_max_epochs + 1):
        actor.train(); losses = []; continuous_losses = []; gripper_losses = []
        for observation, continuous, gripper in loader:
            observation = (observation.to(device) - mean_device) / std_device
            continuous = continuous.to(device); gripper = gripper.to(device)
            _joint, continuous_log_prob, gripper_log_prob = actor.dataset_log_prob(
                observation, continuous, gripper, config.beta_gripper,
            )
            continuous_loss = -continuous_log_prob.mean(); gripper_loss = -gripper_log_prob.mean()
            loss = continuous_loss + config.beta_gripper * gripper_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip_norm); optimizer.step()
            losses.append(float(loss.detach())); continuous_losses.append(float(continuous_loss.detach()))
            gripper_losses.append(float(gripper_loss.detach()))
        actor.eval(); value = validation_loss(actor.cpu(), validation, mean, std, config); actor.to(device)
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "train_continuous_nll": float(np.mean(continuous_losses)),
            "train_gripper_bce": float(np.mean(gripper_losses)), "validation_loss": value,
        }
        history.append(row); print(f"BC epoch={epoch} train={row['train_loss']:.5f} val={value:.5f}", flush=True)
        if value < best_loss - 1e-5:
            best_loss = value; best_epoch = epoch; stale = 0; best_state = deepcopy(actor.state_dict())
        else:
            stale += 1
            if stale >= config.bc_early_stopping_patience:
                break
    actor.load_state_dict(best_state); actor.cpu().eval()
    bc_metrics = {
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "train": split_metrics(actor, train, mean, std),
        "validation": split_metrics(actor, validation, mean, std), "history": history,
    }
    metadata = {
        "format_version": "offline_awac_v3_geometric_milestone_state",
        "reward_version": "awac_reward_v1", "reward_config": asdict(AWACRewardV1Config()),
        "dataset_report_sha256": sha(dataset / "report.json"),
        "dataset_files_sha256": {split: sha(dataset / f"{split}.npz") for split in ("train", "validation")},
        "state_definition": "policy_state_42 + object_grasped + geometric MilestoneTracker[5]",
        "legacy_rule_milestones_used": False,
        "online_transition_count": 0,
    }
    bc_payload = {
        **metadata, "format_version": "hybrid_bc_v3_geometric_milestone_state", "actor_state_dict": actor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "observation_mean": mean,
        "observation_std": std, "training_config": asdict(config), "bc_metrics": bc_metrics,
    }
    bc_path = checkpoint_dir / "hybrid_bc_best.pt"; atomic_save(bc_payload, bc_path)
    (run / "hybrid_bc_metrics.json").write_text(json.dumps(bc_metrics, indent=2) + "\n")
    print("evaluating AWAC-v3 Hybrid BC", flush=True)
    bc_closed_loop = evaluate_policy(HybridCheckpointPredictor(bc_path), SEEDS, AWACRewardV1Config())
    (evaluation_dir / "hybrid_bc.json").write_text(json.dumps(bc_closed_loop, indent=2) + "\n")
    bc_summary = closed_loop_summary(bc_closed_loop)
    print(json.dumps({"checkpoint": "Hybrid BC", **bc_summary}), flush=True)

    # The user explicitly requires BC to recover from the known 0/8/5/76 failure
    # before any critic is initialized. This is an interface sanity gate, not tuning.
    bc_sane = bool(
        bc_summary["success"] > 0
        and bc_summary["transport"] > 8
        and bc_summary["place"] > 5
        and bc_summary["illegal_drop"] < 76
    )
    if not bc_sane:
        stopped = {
            "status": "stopped_at_bc_sanity_gate",
            "reason": "geometric 48-D BC did not recover from the known legacy-interface failure",
            "dataset": report, "normalization": normalization_report,
            "training_config": {**asdict(config), "device": str(device), "optimizer": "Adam"},
            "hybrid_bc": {"checkpoint": str(bc_path), "metrics": bc_metrics,
                          "closed_loop": bc_closed_loop, "summary": bc_summary},
            "offline_awac_started": False, "online_awac_started": False,
        }
        (run / "final_report.json").write_text(json.dumps(stopped, indent=2) + "\n")
        print(json.dumps({"run": str(run), **stopped}, indent=2), flush=True)
        return

    replay = HybridReplay(dataset / "train.npz", device)
    trainer = HybridAWACTrainer(config, mean_np, std_np, best_state, device)
    metadata["hybrid_bc_checkpoint"] = str(bc_path); metadata["hybrid_bc_checkpoint_sha256"] = sha(bc_path)
    results: dict[str, Any] = {}; all_rows = []; interval_rows = []; numerical_failure = None
    metrics_path = run / "training_metrics.jsonl"
    with metrics_path.open("w") as stream:
        for step in range(1, config.awac_updates + 1):
            try:
                row = trainer.update(replay.sample(config.batch_size, trainer.generator))
            except FloatingPointError as error:
                numerical_failure = {"step": step, "reason": str(error)}; break
            if abs(row["q1_mean"]) > 1_000 or abs(row["q2_mean"]) > 1_000:
                numerical_failure = {"step": step, "reason": "Q explosion", "metrics": row}; break
            all_rows.append(row); interval_rows.append(row); stream.write(json.dumps(row) + "\n")
            if step % 100 == 0: stream.flush()
            if step in CHECKPOINTS:
                path = checkpoint_dir / f"hybrid_awac_step_{step:05d}.pt"
                atomic_save(trainer.checkpoint(metadata), path)
                print(f"evaluating AWAC-v3 Offline Hybrid AWAC {step}", flush=True)
                closed_loop = evaluate_policy(HybridCheckpointPredictor(path), SEEDS, AWACRewardV1Config())
                (evaluation_dir / f"hybrid_awac_step_{step:05d}.json").write_text(json.dumps(closed_loop, indent=2) + "\n")
                actor_validation = split_metrics(trainer.actor.cpu().eval(), validation, mean, std)
                trainer.actor.to(device).train()
                conditioned = milestone_diagnostics(trainer, validation, config.seed + step)
                summary = closed_loop_summary(closed_loop)
                results[str(step)] = {
                    "checkpoint": str(path), "checkpoint_sha256": sha(path),
                    "closed_loop": closed_loop, "closed_loop_summary": summary,
                    "validation_actor_metrics": actor_validation,
                    "milestone_conditioned_diagnostics": conditioned,
                    "training_interval": {
                        "start_step": step - len(interval_rows) + 1, "end_step": step,
                        "updates": len(interval_rows), "metrics": metric_stats(interval_rows),
                    },
                }
                interval_rows = []
                print(json.dumps({"step": step, **summary}), flush=True)
    if numerical_failure:
        atomic_save(trainer.checkpoint({**metadata, "numerical_failure": numerical_failure}), checkpoint_dir / f"numeric_stop_{trainer.step:05d}.pt")

    old = json.loads((OLD_43D_RUN / "final_report.json").read_text())
    old_curve = {
        "Hybrid_BC": closed_loop_summary(old["hybrid_bc_closed_loop"]),
        "2500": closed_loop_summary(old["hybrid_awac_checkpoints"]["2500"]["closed_loop"]),
        "5000": closed_loop_summary(old["hybrid_awac_checkpoints"]["5000"]["closed_loop"]),
    }
    for step in (7_500, 10_000, 12_500, 15_000, 17_500):
        path = OLD_43D_LONG_RUN / "closed_loop" / f"hybrid_awac_step_{step:05d}.json"
        if path.exists(): old_curve[str(step)] = closed_loop_summary(json.loads(path.read_text()))
    new_curve = {"Hybrid_BC": closed_loop_summary(bc_closed_loop), **{
        step: value["closed_loop_summary"] for step, value in results.items()
    }}
    eligible = results
    best_step = max(eligible, key=lambda step: (
        eligible[step]["closed_loop_summary"]["success"],
        eligible[step]["closed_loop_summary"]["place_to_success"],
        -eligible[step]["closed_loop_summary"]["illegal_drop"],
        -eligible[step]["closed_loop_summary"]["ik_failure"],
        -eligible[step]["closed_loop_summary"]["timeout"],
        eligible[step]["closed_loop_summary"]["average_return"], -int(step),
    )) if eligible else None
    if best_step is not None:
        shutil.copy2(results[best_step]["checkpoint"], run / "checkpoint_best.pt")
    diagnostics = {
        "updates": len(all_rows),
        "nonfinite_metric_count": int(sum(not np.isfinite(v) for row in all_rows for v in row.values())),
        "q_explosion": bool(any(abs(row["q1_mean"]) > 1_000 or abs(row["q2_mean"]) > 1_000 for row in all_rows)),
        "observed_weight_max": max((row["awac_weight_max"] for row in all_rows), default=None),
        "global_metrics": metric_stats(all_rows) if all_rows else {},
        "numerical_failure": numerical_failure,
    }
    final = {
        "status": "numerical_stop" if numerical_failure else "complete_10k",
        "dataset": report, "normalization": normalization_report,
        "training_config": {**asdict(config), "device": str(device), "optimizer": "Adam", "actor_weight_decay": 0},
        "reward_config": asdict(AWACRewardV1Config()),
        "hybrid_bc": {"checkpoint": str(bc_path), "metrics": bc_metrics, "closed_loop": bc_closed_loop},
        "offline_awac": results, "new_48d_curve": new_curve, "old_43d_curve": old_curve,
        "best_step": int(best_step) if best_step else None,
        "best_checkpoint": str((run / "checkpoint_best.pt").resolve()) if best_step else None,
        "diagnostics": diagnostics, "online_awac_started": False,
    }
    (run / "diagnostics_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (run / "comparison_report.json").write_text(json.dumps({"old_43d": old_curve, "new_48d": new_curve}, indent=2) + "\n")
    (run / "final_report.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"run": str(run), "status": final["status"], "best_step": final["best_step"], "new_48d": new_curve}, indent=2))


if __name__ == "__main__":
    main()
