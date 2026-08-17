#!/usr/bin/env python3
"""Fixed primary selection and secondary paired confirmation for offline Adv-BC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
import torch

from mujoco_shared_control.sac.agent import SACCore, SACCoreConfig
from mujoco_shared_control.sac.evaluation import evaluate_sac


STEPS = (0, 1000, 2000, 5000, 10000)


def core_for(checkpoint: Path) -> SACCore:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    core = SACCore(Path(payload["actor_initial_source"]), SACCoreConfig())
    core.actor.load_state_dict(payload["actor_state_dict"]); core.actor.eval()
    return core


def paired(initial: dict, candidate: dict) -> dict:
    before = {row["seed"]: row for row in initial["rows"]}
    after = {row["seed"]: row for row in candidate["rows"]}
    gained = [seed for seed in before if not before[seed]["success"] and after[seed]["success"]]
    lost = [seed for seed in before if before[seed]["success"] and not after[seed]["success"]]
    discordant = len(gained) + len(lost)
    return {"initial_failure_to_best_success": len(gained),
            "initial_success_to_best_failure": len(lost), "gained_seeds": gained,
            "lost_seeds": lost, "mcnemar_exact_two_sided_p": (
                float(binomtest(len(gained), discordant, .5).pvalue) if discordant else 1.0)}


def compact(result: dict) -> dict:
    return {key: result[key] for key in (
        "seeds", "episodes", "reward_version", "success", "success_rate",
        "milestone_rates", "termination_reason_counts", "episode_return", "episode_length")}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("run", type=Path)
    args = parser.parse_args(); run = args.run.resolve()
    offline = json.loads((run / "offline_metrics.json").read_text())
    existing_primary = run / "primary_closed_loop_eval.json"
    if existing_primary.exists():
        primary = json.loads(existing_primary.read_text())["checkpoints"]
        print("reusing completed primary evaluation", flush=True)
    else:
        primary = {}
        for step in STEPS:
            core = core_for(run / "checkpoints" / f"actor_step_{step:05d}.pt")
            result = evaluate_sac(core, list(range(300000, 300100)),
                                  reward_version="sac_reward_v2_candidate")
            primary[str(step)] = result
            print(f"primary step={step}: {result['success']}/100", flush=True)
    best_step = max(STEPS, key=lambda step: (
        primary[str(step)]["success"],
        -offline["checkpoints"][str(step)]["validation"]["action_drift"]["normalized_mae"],
    ))
    best_updated_step = max(STEPS[1:], key=lambda step: (
        primary[str(step)]["success"],
        -offline["checkpoints"][str(step)]["validation"]["action_drift"]["normalized_mae"],
    ))
    initial_secondary = evaluate_sac(
        core_for(run / "checkpoints" / "actor_step_00000.pt"), list(range(420000, 420100)),
        reward_version="sac_reward_v2_candidate")
    best_secondary = evaluate_sac(
        core_for(run / "checkpoints" / f"actor_step_{best_updated_step:05d}.pt"),
        list(range(420000, 420100)), reward_version="sac_reward_v2_candidate")
    confirmation = paired(initial_secondary, best_secondary)
    output_primary = {"selection_rule": "max primary success; tie -> minimum initial-action drift",
                      "best_step": best_step, "best_updated_step": best_updated_step,
                      "checkpoints": primary}
    output_secondary = {"initial": initial_secondary, "best_updated": best_secondary,
                        "best_updated_step": best_updated_step, "paired": confirmation}
    (run / "primary_closed_loop_eval.json").write_text(json.dumps(output_primary, indent=2) + "\n")
    (run / "secondary_closed_loop_eval.json").write_text(json.dumps(output_secondary, indent=2) + "\n")
    phase = {"primary": {step: {"success": value["success"],
                                "milestones": value["milestone_rates"]}
                           for step, value in primary.items()},
             "secondary": {"initial": compact(initial_secondary),
                           "best_updated": compact(best_secondary)}}
    (run / "phase_event_metrics.json").write_text(json.dumps(phase, indent=2) + "\n")
    summary = {"best_step": best_step, "best_updated_step": best_updated_step, "primary": {
        "initial_success": primary["0"]["success"], "best_success": primary[str(best_step)]["success"],
        "difference": primary[str(best_step)]["success"] - primary["0"]["success"],
        "best_updated_success": primary[str(best_updated_step)]["success"],
        "best_updated_difference": primary[str(best_updated_step)]["success"] - primary["0"]["success"]},
        "secondary": {"initial_success": initial_secondary["success"],
                      "best_updated_success": best_secondary["success"],
                      "difference": best_secondary["success"] - initial_secondary["success"],
                      "paired": confirmation},
        "best_updated_validation_drift": offline["checkpoints"][str(best_updated_step)]["validation"]["action_drift"]}
    (run / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
