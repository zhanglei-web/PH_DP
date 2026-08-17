#!/usr/bin/env python3
"""Single-variable mixed Critic retraining under sac_reward_v2_candidate."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor
from mujoco_shared_control.sac.critic_pretraining import (
    CATEGORIES, PHASES, CriticPretrainConfig, build_arrays_from_semantic_run,
    evaluate, train_critic,
)


MANIFEST = Path("manifests/rule_expert_v1_formal.json")
SEMANTIC_RUN = Path("outputs/sac_reward/sac_reward_v2_candidate_20260814T001000Z")
ACTOR = Path("outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt")
V1_RUN = Path("outputs/sac_critic/sac_critic_pretrain_v1_20260813T210000Z")


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(arrays):
    result = {}
    for split, data in arrays.items():
        result[split] = {"transitions": len(data), "phase_counts": dict(Counter(data.phase)),
                         "category_counts": dict(Counter(data.category)),
                         "mc_return": {"mean": float(data.mc_return.mean()),
                                       "std": float(data.mc_return.std()),
                                       "min": float(data.mc_return.min()),
                                       "max": float(data.mc_return.max())}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args(); run = Path("outputs/sac_critic") / f"sac_critic_pretrain_v2_{args.run_id}"
    run.mkdir(parents=True, exist_ok=False)
    # V1 mixed used base seed+1; preserving the actual training seed is required
    # for a strict reward-only comparison.
    config = CriticPretrainConfig(seed=20260814)
    actor_payload = torch.load(ACTOR, map_location="cpu", weights_only=False)
    actor = SACConstrainedGaussianActor(); actor.load_state_dict(actor_payload["actor_state_dict"])
    actor.eval(); actor.requires_grad_(False)
    actor_before = hashlib.sha256(b"".join(v.numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    mean = torch.as_tensor(actor_payload["observation_mean"], dtype=torch.float32)
    std = torch.as_tensor(actor_payload["observation_std"], dtype=torch.float32)
    arrays, audit = build_arrays_from_semantic_run(
        MANIFEST, SEMANTIC_RUN, reward_version="sac_reward_v2_candidate"
    )
    v1_split = json.loads((V1_RUN / "dataset_split.json").read_text())
    for split in ("train", "validation", "test"):
        if audit["episode_ids"][split] != v1_split["episode_ids"][split]:
            raise RuntimeError(f"V1/V2 episode split mismatch: {split}")
    critic, training, _history = train_critic(
        arrays["train"], arrays["validation"], mean, std, config,
        run / "training_history.jsonl",
    )
    validation = evaluate(critic, arrays["validation"], mean, std)
    test = evaluate(critic, arrays["test"], mean, std)
    target = type(critic)(); target.load_state_dict(critic.state_dict()); target.requires_grad_(False)
    public_training = {key: value for key, value in training.items() if key != "optimizer_state_dict"}
    config_json = {"objective": "twin MC-return regression", "reward_version": "sac_reward_v2_candidate",
                   "config": asdict(config), "strict_v1_comparison": True,
                   "v1_actual_mixed_training_seed": 20260814,
                   "manifest": str(MANIFEST.resolve()), "semantic_run": str(SEMANTIC_RUN.resolve()),
                   "actor_reference": str(ACTOR.resolve()), "actor_sha256": sha(ACTOR),
                   "v1_critic_run": str(V1_RUN.resolve())}
    checkpoint = {"format_version": "sac_critic_pretrain_v2_mc",
                  "critic_state_dict": critic.state_dict(),
                  "target_critic_state_dict": target.state_dict(),
                  "optimizer_state_dict": training["optimizer_state_dict"],
                  "epoch": training["best_epoch"], "validation_metrics": validation,
                  "test_metrics": test, "config": asdict(config), "dataset_split": audit,
                  "observation_mean": mean, "observation_std": std,
                  "actor_reference": str(ACTOR.resolve()), "actor_sha256": sha(ACTOR),
                  "reward_version": "sac_reward_v2_candidate"}
    torch.save(checkpoint, run / "critic_pretrained_v2_best.pt")
    (run / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")
    (run / "dataset_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    (run / "return_prediction_test.json").write_text(json.dumps(test, indent=2) + "\n")
    (run / "phase_metrics.json").write_text(json.dumps(test["phase"], indent=2) + "\n")
    (run / "q1_q2_disagreement.json").write_text(json.dumps({
        "mae": test["q1_q2_disagreement_mae"], "q1": test["q1"], "q2": test["q2"]}, indent=2) + "\n")
    (run / "training_summary.json").write_text(json.dumps(public_training, indent=2) + "\n")
    (run / "return_distribution.json").write_text(json.dumps(distribution(arrays), indent=2) + "\n")
    actor_after = hashlib.sha256(b"".join(v.numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    if actor_before != actor_after: raise RuntimeError("frozen Actor changed")
    summary = {"run": str(run.resolve()), "training": public_training,
               "split_episode_ids_equal_v1": True, "actor_frozen": True,
               "online_training": False, "test_overall": test["overall"],
               "test_by_category": {key: test["category"][key] for key in CATEGORIES},
               "test_by_phase": {key: test["phase"][key] for key in PHASES},
               "q1_q2_disagreement_mae": test["q1_q2_disagreement_mae"]}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
