#!/usr/bin/env python3
"""Create machine-readable and Markdown summaries for the bounded SAC v2 run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


RUN = Path("outputs/sac_training/sac_v2_constrained_clean_sanity_20260813T033000Z")


def main() -> None:
    evaluations = [json.loads(line) for line in (RUN / "evaluation_metrics.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (RUN / "training_metrics.jsonl").read_text().splitlines()]
    support = json.loads((RUN / "action_support_stats.json").read_text())
    evaluation_by_step = {row["global_env_steps"]: row for row in evaluations}
    checkpoints = (0, 10_000, 15_000, 20_000, 25_000, 30_000)
    drift = {
        str(step): evaluation_by_step[step]["actor_drift"] for step in checkpoints
    }
    critic = {
        str(row["global_env_steps"]): {
            key: row.get(key) for key in (
                "critic_loss", "q1_mean", "q2_mean", "q1_std", "q2_std",
                "q1_min", "q1_max", "q2_min", "q2_max", "target_mean",
                "target_std", "td_error_mean", "td_error_std", "actor_loss",
                "alpha", "mean_log_prob", "mean_log_std", "min_log_std",
                "max_log_std",
            )
        }
        for row in metrics
    }
    (RUN / "actor_drift.json").write_text(json.dumps(drift, indent=2) + "\n")
    (RUN / "critic_health.json").write_text(json.dumps(critic, indent=2) + "\n")
    episode_rows = list(csv.DictReader((RUN / "episode_metrics.csv").open()))
    reasons: dict[str, int] = {}
    for row in episode_rows:
        reasons[row["termination_reason"]] = reasons.get(row["termination_reason"], 0) + 1
    table = []
    for step in checkpoints:
        row = evaluation_by_step[step]
        table.append(
            f"| {step} | {row['gradient_updates']} | {row['success']}/20 | "
            f"{row['milestone_rates']['grasped']:.0%} | {row['milestone_rates']['lifted']:.0%} | "
            f"{row['milestone_rates']['transported']:.0%} | {row['milestone_rates']['released']:.0%} | "
            f"{row['milestone_rates']['retreated']:.0%} | {row['actor_drift']['normalized_mae']:.9f} |"
        )
    final_health = critic["30000"]
    report = f"""# Clean SAC v2 Sanity Report

Run: `sac_v2_constrained_clean_sanity_20260813T033000Z`

Status: `SAC_V2_ACTION_FIX_INSUFFICIENT`

This bounded run used the native constrained v2 Actor and clean schedule only:
10,000 collection steps followed by 20,000 standard full-SAC updates. It used no
Critic warmup, expert replay, BC regularizer, or frozen-parameter change.

## Fixed validation

Seeds: `410000-410019`, deterministic constrained mean action.

| Env step | Updates | Success | Grasp | Lift | Transport | Release | Retreat | Actor MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Initialization was 6/20. Behavior collapsed to 0/20 after 5,000 Actor updates
and did not recover by 30k. Final normalized deterministic action MAE was
`{evaluation_by_step[30000]['actor_drift']['normalized_mae']}`; final gripper
physical MAE was `{evaluation_by_step[30000]['actor_drift']['gripper_physical_mae_m']} m`.

## Critic and entropy health

The action-semantics repair eliminated the former Q explosion. In the final 1k
window Q1 mean/std/min/max were `{final_health['q1_mean']}` / `{final_health['q1_std']}` /
`{final_health['q1_min']}` / `{final_health['q1_max']}`. Target mean/std were
`{final_health['target_mean']}` / `{final_health['target_std']}` and Critic loss
was `{final_health['critic_loss']}`. All values remained finite. Target variance
rose from `{critic['11000']['target_std']}` at 11k to `{final_health['target_std']}` at 30k,
but did not enter the old hundreds/thousands regime.

Alpha changed from 0.1 to `{evaluation_by_step[30000]['alpha']}`. Evaluation
log-std mean/min/max changed from `-3/-3/-3` to
`{evaluation_by_step[30000]['log_std']['mean']}` /
`{evaluation_by_step[30000]['log_std']['min']}` /
`{evaluation_by_step[30000]['log_std']['max']}`. These are finite but accompany
large policy drift and task collapse, so they are not behaviorally healthy.

## Action semantics and support

```text
translation adapter projections: {support['adapter_translation_projection_count']}
rotation adapter projections:    {support['adapter_rotation_projection_count']}
gripper adapter clips:           {support['adapter_gripper_clip_count']}
Replay-policy mismatches:        {support['replay_policy_mismatch_count']}
normal adapter delta mean:       {support['normal_adapter_difference_l2']['mean']}
normal adapter delta max:        {support['normal_adapter_difference_l2']['max']}
IK fallback transitions:         {support['fallback_count']}
Replay translation norm max:     {support['replay']['translation_norm_max']}
Replay rotation norm max:        {support['replay']['rotation_norm_max']}
Q(policy)-Q(adapter) max:         {support['q_policy_vs_adapter']['q1_max_abs']} / {support['q_policy_vs_adapter']['q2_max_abs']}
```

There were {support['translation_numeric_boundary_count']} translation and
{support['rotation_numeric_boundary_count']} rotation float32 boundary hits at norm exactly
1.0; none exceeded the admissible ball or triggered adapter projection. Replay
stored every attempted policy action exactly, including `{support['fallback_count']}`
fallback transitions, so failure attribution remained correct.

## A/B conclusion

Old v1: componentwise tanh, deployed/projected Replay, action mismatch present,
8/20 at 10k then 0/20 at 20k, Actor MAE 0.6066684, and observed Q explosion.

New v2: radial constrained policy, policy-action Replay, no mismatch, 6/20 at
0/10k then 0/20 from 15k through 30k. Q explosion and out-of-support exploitation
were fixed, and final MAE was lower (`{evaluation_by_step[30000]['actor_drift']['normalized_mae']}`),
but early task-policy collapse remained. Action semantics was a real major bug,
but its repair alone is insufficient for stable online SAC under the frozen setup.

Completed training episodes: {len(episode_rows)}; termination counts: `{reasons}`.
The run stopped at exactly 30,000 environment steps.
"""
    (RUN / "sanity_report.md").write_text(report)


if __name__ == "__main__":
    main()
