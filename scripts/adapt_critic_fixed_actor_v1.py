#!/usr/bin/env python3
"""Adapt Mixed Critic v2 to a frozen Actor with fixed-policy TD only."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.sac.aligned_initialization import load_aligned_v2
from mujoco_shared_control.sac.critic_adaptation import (
    ActorTransitionArrays, CriticAdaptationConfig, concatenate_batch,
    critic_update, evaluate_actor_data, exact_mixed_indices, module_checksum,
)
from mujoco_shared_control.sac.critic_pretraining import (
    build_arrays_from_semantic_run, evaluate as evaluate_expert,
)


ALIGNED = Path("outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt")
MANIFEST = Path("manifests/rule_expert_v1_formal.json")
SEMANTIC = Path("outputs/sac_reward/sac_reward_v2_candidate_20260814T001000Z")
CHECKPOINTS = (0, 5000, 10000, 20000)


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(path: Path, critic, target, optimizer, step: int, config, payload,
                    actor_checksum: str) -> None:
    torch.save({
        "format_version": "critic_only_online_adaptation_v1",
        "critic_state_dict": critic.state_dict(),
        "target_critic_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "critic_updates": step,
        "config": asdict(config), "observation_mean": payload["observation_mean"],
        "observation_std": payload["observation_std"],
        "actor_source": payload["actor_source"], "actor_checksum": actor_checksum,
        "critic_source": payload["critic_source"],
        "reward_version": "sac_reward_v2_candidate",
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("run", type=Path)
    args = parser.parse_args(); run = args.run.resolve()
    data_dir = run / "online_actor_dataset"; checkpoint_dir = run / "critic_checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)
    config = CriticAdaptationConfig(); torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    actor, critic, target, payload = load_aligned_v2(ALIGNED)
    actor.eval(); actor.requires_grad_(False); actor_checksum = module_checksum(actor)
    log_std_checksum = module_checksum(actor.log_std_head)
    critic.train(); target.eval(); target.requires_grad_(False)
    mean, std = payload["observation_mean"], payload["observation_std"]
    online = {split: ActorTransitionArrays.load(data_dir/f"{split}.npz")
              for split in ("train","validation","test")}
    offline, offline_audit = build_arrays_from_semantic_run(
        MANIFEST, SEMANTIC, reward_version="sac_reward_v2_candidate")
    online_manifest = json.loads((data_dir/"dataset_manifest.json").read_text())
    if online_manifest["seed_range"] != [800000,800999]: raise RuntimeError("unexpected online seeds")
    optimizer = torch.optim.Adam(critic.parameters(),lr=config.learning_rate)
    evaluations = {}; history = []

    def evaluate_checkpoint(step: int) -> None:
        critic.eval()
        evaluations[str(step)] = {
            "online_validation": evaluate_actor_data(critic,online["validation"],mean,std),
            "online_test": evaluate_actor_data(critic,online["test"],mean,std),
            "offline_validation": evaluate_expert(critic,offline["validation"],mean,std),
            "offline_test": evaluate_expert(critic,offline["test"],mean,std),
        }
        save_checkpoint(checkpoint_dir/f"critic_step_{step:05d}.pt",critic,target,optimizer,
                        step,config,payload,actor_checksum)
        critic.train()

    evaluate_checkpoint(0)
    for step in range(1,config.max_updates+1):
        offline_indices, online_indices = exact_mixed_indices(
            len(offline["train"]),len(online["train"]),config,rng)
        batch = concatenate_batch(offline["train"],online["train"],
                                  offline_indices,online_indices)
        row = critic_update(critic,target,actor,optimizer,batch,mean,std,config)
        row["step"] = step; row["offline_batch"] = len(offline_indices)
        row["online_batch"] = len(online_indices); history.append(row)
        if step in CHECKPOINTS:
            evaluate_checkpoint(step)
            metrics=evaluations[str(step)]
            print(f"step={step} online_rho={metrics['online_validation']['overall']['spearman']:.4f} "
                  f"offline_rho={metrics['offline_validation']['overall']['spearman']:.4f}",flush=True)

    if module_checksum(actor) != actor_checksum or module_checksum(actor.log_std_head) != log_std_checksum:
        raise RuntimeError("frozen Actor changed")
    # Selection is lexicographic, not a tuned composite: improve online validation
    # ranking while retaining at least 95% of the initial offline validation rank.
    initial_offline = evaluations["0"]["offline_validation"]["overall"]["spearman"]
    eligible = [step for step in CHECKPOINTS if
                evaluations[str(step)]["offline_validation"]["overall"]["spearman"]
                >= initial_offline - .05]
    best = max(eligible,key=lambda step:(
        evaluations[str(step)]["online_validation"]["overall"]["spearman"],-step))
    config_output = {**asdict(config),"objective":"deterministic fixed-policy TD",
                     "batch":"128 Expert train + 128 Frozen Actor train",
                     "entropy_term":False,"actor_optimizer_steps":0,"alpha_optimizer":False,
                     "checkpoints":list(CHECKPOINTS),"selected_step":best,
                     "selection":"max online validation Spearman subject to offline validation Spearman >= step0-0.05",
                     "final_test_seeds_used":False,"aligned_source":str(ALIGNED.resolve()),
                     "aligned_sha256":sha(ALIGNED)}
    (run/"config.json").write_text(json.dumps(config_output,indent=2)+"\n")
    (run/"actor_source.json").write_text(json.dumps({
        "path":payload["actor_source"],"sha256":payload["actor_source_sha256"],
        "checksum_before":actor_checksum,"checksum_after":module_checksum(actor),
        "log_std_checksum_before":log_std_checksum,
        "log_std_checksum_after":module_checksum(actor.log_std_head),"optimizer_steps":0},indent=2)+"\n")
    (run/"critic_v2_source.json").write_text(json.dumps({
        "path":payload["critic_source"],"sha256":payload["critic_source_sha256"],
        "initial_critic_checksum":module_checksum(load_aligned_v2(ALIGNED)[1])},indent=2)+"\n")
    (run/"reward_v2_source.json").write_text(json.dumps({
        "reward_version":"sac_reward_v2_candidate","semantic_run":str(SEMANTIC.resolve()),
        "semantic_manifest_sha256":sha(SEMANTIC/"dataset_manifest.json")},indent=2)+"\n")
    (run/"online_return_prediction.json").write_text(json.dumps({
        step:{"validation":value["online_validation"],"test":value["online_test"]}
        for step,value in evaluations.items()},indent=2)+"\n")
    (run/"offline_expert_retention.json").write_text(json.dumps({
        step:{"validation":value["offline_validation"],"test":value["offline_test"]}
        for step,value in evaluations.items()},indent=2)+"\n")
    (run/"q1_q2_disagreement.json").write_text(json.dumps({step:{
        "online_validation":value["online_validation"]["q1_q2_disagreement_mae"],
        "online_test":value["online_test"]["q1_q2_disagreement_mae"],
        "offline_validation":value["offline_validation"]["q1_q2_disagreement_mae"],
        "offline_test":value["offline_test"]["q1_q2_disagreement_mae"]}
        for step,value in evaluations.items()},indent=2)+"\n")
    with (run/"training_metrics.jsonl").open("w") as stream:
        for row in history: stream.write(json.dumps(row)+"\n")
    summary={"selected_step":best,"actor_frozen":True,"online_train_transitions":len(online["train"]),
             "offline_train_transitions":len(offline["train"]),"online":{
                 step:value["online_test"]["overall"] for step,value in evaluations.items()},
             "offline":{step:value["offline_test"]["overall"] for step,value in evaluations.items()},
             "offline_split_audit":offline_audit}
    (run/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
