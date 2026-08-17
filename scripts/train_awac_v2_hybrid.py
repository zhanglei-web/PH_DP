#!/usr/bin/env python3
"""Hybrid BC warm start followed by 5k offline AWAC-v2 updates."""

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


DATASET = Path("outputs/awac_dataset/awac_v2_hybrid_formal_rule")
AWAC_V1_REPORT = Path("outputs/awac_dataset/awac_v1_formal_rule/report.json")
SEEDS = list(range(300_000, 300_100))
CHECKPOINTS = (2_500, 5_000)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    torch.save(value, temporary); temporary.replace(path)


def load_split(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "observation": torch.from_numpy(np.asarray(data["obs"], np.float32)),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/awac_training"))
    args = parser.parse_args()
    run = (args.output_root / f"awac_v2_hybrid_{args.run_id}").resolve()
    checkpoint_dir = run / "checkpoints"; evaluation_dir = run / "closed_loop"
    checkpoint_dir.mkdir(parents=True, exist_ok=False); evaluation_dir.mkdir()
    dataset = DATASET.resolve(); dataset_report = json.loads((dataset / "report.json").read_text())
    if dataset_report["episode_count"] != 1234 or dataset_report["transition_count"] != 150406:
        raise RuntimeError("Hybrid training refused: wrong frozen dataset")
    config = HybridAWACConfig(); torch.manual_seed(config.seed); np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(dataset / "train.npz"); validation = load_split(dataset / "validation.npz")
    mean_np, std_np = observation_statistics(dataset / "train.npz")
    mean = torch.from_numpy(mean_np); std = torch.from_numpy(std_np)
    actor = HybridActor(config).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train["observation"], train["continuous"], train["gripper"]),
        batch_size=config.batch_size, shuffle=True, generator=generator,
    )
    mean_device = mean.to(device); std_device = std.to(device)
    best_loss = float("inf"); best_epoch = 0; stale = 0; bc_history = []
    best_state = deepcopy(actor.state_dict())
    for epoch in range(1, config.bc_max_epochs + 1):
        actor.train(); losses = []; continuous_losses = []; gripper_losses = []
        for observation, continuous, gripper in loader:
            observation = (observation.to(device) - mean_device) / std_device
            continuous = continuous.to(device); gripper = gripper.to(device)
            _joint, continuous_log_prob, gripper_log_prob = actor.dataset_log_prob(
                observation, continuous, gripper, config.beta_gripper,
            )
            continuous_loss = -continuous_log_prob.mean()
            gripper_loss = -gripper_log_prob.mean()
            loss = continuous_loss + config.beta_gripper * gripper_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip_norm); optimizer.step()
            losses.append(float(loss.detach())); continuous_losses.append(float(continuous_loss.detach())); gripper_losses.append(float(gripper_loss.detach()))
        actor.eval(); value = validation_loss(actor.cpu(), validation, mean, std, config); actor.to(device)
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "train_continuous_nll": float(np.mean(continuous_losses)),
            "train_gripper_bce": float(np.mean(gripper_losses)), "validation_loss": value,
        }
        bc_history.append(row)
        print(f"BC epoch={epoch} train={row['train_loss']:.5f} val={value:.5f}", flush=True)
        if value < best_loss - 1e-5:
            best_loss = value; best_epoch = epoch; stale = 0
            best_state = deepcopy(actor.state_dict())
        else:
            stale += 1
            if stale >= config.bc_early_stopping_patience:
                break
    actor.load_state_dict(best_state); actor.cpu().eval()
    bc_metrics = {
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "train": split_metrics(actor, train, mean, std),
        "validation": split_metrics(actor, validation, mean, std),
        "history": bc_history,
    }
    metadata = {
        "reward_version": "awac_reward_v1", "reward_config": asdict(AWACRewardV1Config()),
        "dataset_report_sha256": sha(dataset / "report.json"),
        "dataset_files_sha256": {split: sha(dataset / f"{split}.npz") for split in ("train", "validation")},
    }
    bc_payload = {
        "format_version": "hybrid_bc_v1", "actor_state_dict": actor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "observation_mean": mean,
        "observation_std": std, "training_config": asdict(config), "bc_metrics": bc_metrics,
        **metadata,
    }
    bc_path = checkpoint_dir / "hybrid_bc_best.pt"; atomic_save(bc_payload, bc_path)
    (run / "hybrid_bc_metrics.json").write_text(json.dumps(bc_metrics, indent=2) + "\n")
    print("evaluating Hybrid BC", flush=True)
    bc_closed_loop = evaluate_policy(HybridCheckpointPredictor(bc_path), SEEDS, AWACRewardV1Config())
    (evaluation_dir / "hybrid_bc.json").write_text(json.dumps(bc_closed_loop, indent=2) + "\n")

    replay = HybridReplay(dataset / "train.npz", device)
    trainer = HybridAWACTrainer(config, mean_np, std_np, best_state, device)
    metadata["hybrid_bc_checkpoint"] = str(bc_path); metadata["hybrid_bc_checkpoint_sha256"] = sha(bc_path)
    history = []; results = {}
    metrics_file = (run / "training_metrics.jsonl").open("w")
    try:
        for step in range(1, config.awac_updates + 1):
            row = trainer.update(replay.sample(config.batch_size, trainer.generator)); history.append(row)
            metrics_file.write(json.dumps(row) + "\n")
            if step % 100 == 0: metrics_file.flush()
            if step in CHECKPOINTS:
                path = checkpoint_dir / f"hybrid_awac_step_{step:05d}.pt"
                atomic_save(trainer.checkpoint(metadata), path)
                print(f"evaluating Hybrid AWAC {step}", flush=True)
                closed_loop = evaluate_policy(HybridCheckpointPredictor(path), SEEDS, AWACRewardV1Config())
                (evaluation_dir / f"hybrid_awac_step_{step:05d}.json").write_text(json.dumps(closed_loop, indent=2) + "\n")
                trainer.actor.cpu().eval()
                validation_metrics = split_metrics(trainer.actor, validation, mean, std)
                trainer.actor.to(device).train()
                window = {key: float(np.mean([item[key] for item in history[-500:]])) for key in row if key != "step"}
                results[str(step)] = {
                    "checkpoint": str(path), "checkpoint_sha256": sha(path),
                    "validation_actor_metrics": validation_metrics,
                    "training_window_last_500": window, "closed_loop": closed_loop,
                }
                print(f"AWAC step={step} grasp={closed_loop['grasp_success']['count']} success={closed_loop['task_success']}", flush=True)
    finally:
        metrics_file.close()
    old = json.loads(Path("outputs/awac_training/offline_awac_v1_20260814T040000Z/final_report.json").read_text())
    comparison = {
        "BC_baseline": old["bc_baseline"],
        "old_AWAC": old["checkpoint_closed_loop"],
        "Hybrid_BC": bc_closed_loop,
        "Hybrid_AWAC": {step: value["closed_loop"] for step, value in results.items()},
    }
    stopped = bool(
        results["5000"]["closed_loop"]["grasp_success"]["rate"] <= 0.05
        or results["5000"]["closed_loop"]["task_success_rate"] < bc_closed_loop["task_success_rate"]
    )
    final = {
        "status": "stopped_after_required_5k" if stopped else "completed_required_5k",
        "dataset": dataset_report,
        "training_config": {**asdict(config), "device": str(device), "optimizer": "Adam"},
        "hybrid_bc_checkpoint": str(bc_path), "hybrid_bc_metrics": bc_metrics,
        "hybrid_bc_closed_loop": bc_closed_loop, "hybrid_awac_checkpoints": results,
        "comparison": comparison,
        "answers": {
            "hybrid_bc_gripper_learned": bool(
                bc_metrics["validation"]["CLOSE_GRIPPER"]["accuracy"] >= .95
                and bc_metrics["validation"]["OPEN_GRIPPER"]["accuracy"] >= .95
            ),
            "hybrid_awac_restored_grasp": bool(results["5000"]["closed_loop"]["grasp_success"]["rate"] > .05),
            "closed_loop_success_improved_over_old_awac": bool(
                results["5000"]["closed_loop"]["task_success_rate"] > 0.0
            ),
        },
        "online_awac_started": False,
        "stop_rule_triggered": stopped,
    }
    (run / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (run / "final_report.json").write_text(json.dumps(final, indent=2) + "\n")
    best = max(results, key=lambda step: (results[step]["closed_loop"]["task_success_rate"], results[step]["closed_loop"]["grasp_success"]["rate"], -int(step)))
    shutil.copy2(results[best]["checkpoint"], run / "checkpoint_best.pt")
    print(json.dumps({
        "run": str(run), "hybrid_bc": {"metrics": bc_metrics["validation"], "success": bc_closed_loop["task_success"]},
        "hybrid_awac": {step: {"grasp": value["closed_loop"]["grasp_success"]["count"], "success": value["closed_loop"]["task_success"]} for step, value in results.items()},
        "answers": final["answers"], "stop_rule_triggered": stopped,
    }, indent=2))


if __name__ == "__main__":
    main()
