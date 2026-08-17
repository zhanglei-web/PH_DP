#!/usr/bin/env python3
"""Bounded positive-advantage filtered/anchored BC experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from mujoco_shared_control.sac.aligned_initialization import load_aligned_v2
from mujoco_shared_control.sac.critic_pretraining import build_arrays_from_semantic_run
from mujoco_shared_control.sac.offline_adv_bc import (
    AdvantageArrays, OfflineAdvBCConfig, build_advantage_arrays, state_checksum,
    train_step, validate_trainable_actor,
)


ALIGNED = Path("outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt")
MANIFEST = Path("manifests/rule_expert_v1_formal.json")
SEMANTIC = Path("outputs/sac_reward/sac_reward_v2_candidate_20260814T001000Z")
CHECKPOINT_STEPS = (0, 1000, 2000, 5000, 10000)


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, np.float64)
    return {"count": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
            "positive_fraction": float(np.mean(x > 0))}


def advantage_stats(values: AdvantageArrays) -> dict[str, Any]:
    result = {"overall": stats(values.advantage), "phase": {}, "category": {}}
    for phase in ("P1", "P2", "P3", "P4"):
        mask = values.phase == phase
        if mask.any(): result["phase"][phase] = stats(values.advantage[mask])
    for category in ("nominal_success", "normal_recovered"):
        mask = values.category == category
        if mask.any(): result["category"][category] = stats(values.advantage[mask])
    return result


@torch.no_grad()
def metrics(actor, data, initial_actions, mean, std) -> dict[str, Any]:
    predictions = []
    for start in range(0, len(data), 4096):
        state = torch.from_numpy(data.observation[start:start+4096])
        predictions.append(actor.deterministic_action((state-mean)/std).numpy())
    action = np.concatenate(predictions); target = data.action
    error = action-target; drift = action-initial_actions
    return {"mse": float(np.mean(error**2)), "mae": float(np.mean(abs(error))),
            "action_drift": {"normalized_mae": float(np.mean(abs(drift))),
                "normalized_rmse": float(np.sqrt(np.mean(drift**2))),
                "xyz_mae": float(np.mean(abs(drift[:, :3]))),
                "rotation_mae": float(np.mean(abs(drift[:, 3:6]))),
                "gripper_mae": float(np.mean(abs(drift[:, 6]))),
                "max_abs": float(np.max(abs(drift)))}}


def save_actor(path: Path, actor, step: int, payload, config, optimizer) -> None:
    torch.save({"format_version": "offline_adv_weighted_bc_v1",
                "actor_state_dict": actor.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_steps": step, "observation_mean": payload["observation_mean"],
                "observation_std": payload["observation_std"], "action_spec": payload["action_spec"],
                "actor_initial_source": payload["actor_source"],
                "actor_initial_source_sha256": payload["actor_source_sha256"],
                "critic_source": payload["critic_source"], "critic_source_sha256": payload["critic_source_sha256"],
                "reward_version": "sac_reward_v2_candidate", "config": asdict(config)}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args(); run = Path("outputs/actor_improvement") / f"offline_adv_weighted_bc_v1_{args.run_id}"
    checkpoint_dir = run / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=False)
    config = OfflineAdvBCConfig(); torch.manual_seed(config.seed); rng = np.random.default_rng(config.seed)
    actor, critic, target, payload = load_aligned_v2(ALIGNED)
    initial_actor = deepcopy(actor).eval(); initial_actor.requires_grad_(False)
    critic.eval(); critic.requires_grad_(False); target.eval(); target.requires_grad_(False)
    critic_before, target_before = state_checksum(critic), state_checksum(target)
    log_std_before = {key: value.clone() for key, value in actor.log_std_head.state_dict().items()}
    trainable = validate_trainable_actor(actor)
    mean, std = payload["observation_mean"], payload["observation_std"]
    arrays, dataset_audit = build_arrays_from_semantic_run(
        MANIFEST, SEMANTIC, reward_version="sac_reward_v2_candidate")
    eligible, advantages = {}, {}
    for split, data in arrays.items():
        eligible_mask = np.isin(data.category, ["nominal_success", "normal_recovered"])
        eligible[split] = data.subset(eligible_mask)
        advantages[split] = build_advantage_arrays(
            eligible[split], initial_actor, critic, mean, std,
            threshold=config.positive_advantage_threshold)
    train = eligible["train"]; train_adv = advantages["train"]
    anchor_indices = np.flatnonzero(train.category == "nominal_success")
    improve_indices = np.flatnonzero(train_adv.positive)
    if not len(improve_indices): raise RuntimeError("no positive-advantage training samples")

    np.savez_compressed(run / "advantage_dataset_v1.npz", **{
        f"{split}_{name}": getattr(value, name)
        for split, value in advantages.items()
        for name in value.__dataclass_fields__})
    manifest = {"format_version": "advantage_dataset_v1", "threshold": 0.0,
                "reference_actor_fixed": True, "critic_frozen": True,
                "eligible_categories": ["nominal_success", "normal_recovered"],
                "splits": {split: {"transitions": len(value),
                    "positive": int(value.positive.sum()),
                    "episode_ids": sorted(np.unique(value.episode_id).tolist())}
                    for split, value in advantages.items()},
                "source_dataset_audit": dataset_audit}
    (run / "dataset_manifest.json").write_text(json.dumps(dataset_audit, indent=2) + "\n")
    (run / "advantage_dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stats_by_split = {split: advantage_stats(value) for split, value in advantages.items()}
    (run / "advantage_statistics.json").write_text(json.dumps(stats_by_split, indent=2) + "\n")
    (run / "advantage_phase_statistics.json").write_text(json.dumps(
        {split: value["phase"] for split, value in stats_by_split.items()}, indent=2) + "\n")
    config_json = {**asdict(config), "optimizer": "AdamW", "loss": "anchor_MSE + improve_MSE",
                   "checkpoints": list(CHECKPOINT_STEPS), "trainable_parameters": trainable,
                   "primary_seeds": [300000, 300099], "secondary_seeds": [420000, 420099],
                   "sealed_final_test_seeds": [500000, 500099], "online_training": False,
                   "replay_created": False}
    (run / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")
    (run / "actor_initial_source.json").write_text(json.dumps({
        "aligned": str(ALIGNED.resolve()), "aligned_sha256": sha(ALIGNED),
        "actor_source": payload["actor_source"], "actor_sha256": payload["actor_source_sha256"]}, indent=2) + "\n")
    (run / "critic_v2_source.json").write_text(json.dumps({
        "critic_source": payload["critic_source"], "critic_sha256": payload["critic_source_sha256"],
        "checksum_before": critic_before, "target_checksum_before": target_before}, indent=2) + "\n")

    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    initial_actions = {split: value.initial_actor_action for split, value in advantages.items()}
    offline = {"checkpoints": {}}
    save_actor(checkpoint_dir / "actor_step_00000.pt", actor, 0, payload, config, optimizer)
    offline["checkpoints"]["0"] = {
        split: metrics(actor, eligible[split], initial_actions[split], mean, std)
        for split in eligible}
    history = []
    for step in range(1, config.max_optimizer_steps + 1):
        anchor = rng.choice(anchor_indices, config.batch_size, replace=True)
        improve = rng.choice(improve_indices, config.batch_size, replace=True)
        row = train_step(
            actor, optimizer, torch.from_numpy(train.observation[anchor]),
            torch.from_numpy(train.action[anchor]), torch.from_numpy(train.observation[improve]),
            torch.from_numpy(train.action[improve]), mean, std, config.gradient_clip_norm)
        row["step"] = step; history.append(row)
        if step in CHECKPOINT_STEPS:
            save_actor(checkpoint_dir / f"actor_step_{step:05d}.pt", actor, step, payload, config, optimizer)
            offline["checkpoints"][str(step)] = {
                split: metrics(actor, eligible[split], initial_actions[split], mean, std)
                for split in eligible}
            print(f"step={step} total={row['total_loss']:.7g} val={offline['checkpoints'][str(step)]['validation']['mse']:.7g}", flush=True)
    if critic_before != state_checksum(critic) or target_before != state_checksum(target):
        raise RuntimeError("frozen critic changed")
    for key, value in log_std_before.items():
        torch.testing.assert_close(value, actor.log_std_head.state_dict()[key], rtol=0, atol=0)
    offline["training_windows"] = [
        {"end_step": end, **{name: float(np.mean([row[name] for row in history[max(0,end-1000):end]]))
                             for name in ("anchor_loss", "improve_loss", "total_loss")}}
        for end in (1000, 2000, 5000, 10000)]
    offline["critic_checksum_unchanged"] = True; offline["target_checksum_unchanged"] = True
    offline["log_std_unchanged"] = True; offline["alpha_updates"] = 0
    (run / "offline_metrics.json").write_text(json.dumps(offline, indent=2) + "\n")
    (run / "actor_drift_metrics.json").write_text(json.dumps({
        step: values["validation"]["action_drift"] for step, values in offline["checkpoints"].items()}, indent=2) + "\n")
    summary = {"run": str(run.resolve()), "eligible_train_transitions": len(train_adv),
               "positive_train_transitions": int(train_adv.positive.sum()),
               "positive_fraction": float(train_adv.positive.mean()),
               "positive_recovered": int(np.sum(train_adv.positive & (train_adv.category == "normal_recovered"))),
               "advantage_train": stats_by_split["train"], "frozen_checks_passed": True}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
