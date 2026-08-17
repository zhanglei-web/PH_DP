#!/usr/bin/env python3
"""Final paired fresh-seed evaluation of frozen Offline and Online AWAC Actors."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
import torch

from mujoco_shared_control.awac.evaluation import MILESTONES, evaluate_policy
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.reward import AWACRewardV1Config


OFFLINE = Path(
    "outputs/awac_training/"
    "awac_v3_geometric_milestone_state_offline25k_20260814T160000Z/"
    "checkpoint_best.pt"
)
ONLINE = Path(
    "outputs/awac_online/"
    "online_awac_v3_geometric_hybrid_20260814T210000Z/"
    "checkpoints/online_step_20000.pt"
)
BLOCKS = {
    "500000-500099": list(range(500_000, 500_100)),
    "600000-600099": list(range(600_000, 600_100)),
    "700000-700099": list(range(700_000, 700_100)),
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_081_400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([row["episode_return"] for row in rows], np.float64)
    lengths = np.asarray([row["episode_length"] for row in rows], np.float64)
    seeds = [int(row["seed"]) for row in rows]
    return {
        "episodes": len(rows),
        "seeds": [min(seeds), max(seeds)],
        "task_success": int(sum(row["task_success"] for row in rows)),
        "task_success_rate": float(np.mean([row["task_success"] for row in rows])),
        **{
            name: {
                "count": int(sum(row[name] for row in rows)),
                "rate": float(np.mean([row[name] for row in rows])),
            }
            for name in MILESTONES
        },
        **{
            name: {
                "count": int(sum(row[name] for row in rows)),
                "rate": float(np.mean([row[name] for row in rows])),
            }
            for name in ("illegal_drop", "ik_failure", "timeout")
        },
        "termination_reason_counts": dict(Counter(row["termination_reason"] for row in rows)),
        "average_episode_return": float(returns.mean()),
        "episode_return": {
            "mean": float(returns.mean()), "std": float(returns.std()),
            "min": float(returns.min()), "max": float(returns.max()),
        },
        "episode_length": {
            "mean": float(lengths.mean()),
            "min": int(lengths.min()), "max": int(lengths.max()),
        },
        "ik_fallback_count": int(sum(row["ik_fallback_count"] for row in rows)),
        "action_clipping_count": int(sum(row["action_clipping_count"] for row in rows)),
        "rows": rows,
    }


def compact(value: dict[str, Any]) -> dict[str, Any]:
    place = int(value["place_success"]["count"])
    return {
        "episodes": int(value["episodes"]),
        "success": int(value["task_success"]),
        "success_rate": float(value["task_success_rate"]),
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
        "place_to_success": int(value["task_success"]) / max(place, 1),
    }


def percentile_ci(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def paired_analysis(
    offline_rows: list[dict[str, Any]], online_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    offline_by_seed = {int(row["seed"]): row for row in offline_rows}
    online_by_seed = {int(row["seed"]): row for row in online_rows}
    if set(offline_by_seed) != set(online_by_seed) or len(offline_by_seed) != 300:
        raise RuntimeError("paired evaluation does not contain the same 300 seeds")
    seeds = np.asarray(sorted(offline_by_seed), np.int64)
    offline = np.asarray([offline_by_seed[int(seed)]["task_success"] for seed in seeds], bool)
    online = np.asarray([online_by_seed[int(seed)]["task_success"] for seed in seeds], bool)
    both_success = int(np.sum(offline & online))
    both_fail = int(np.sum(~offline & ~online))
    gained_mask = ~offline & online
    lost_mask = offline & ~online
    gained = int(np.sum(gained_mask))
    lost = int(np.sum(lost_mask))
    discordant = gained + lost
    p_value = (
        float(binomtest(min(gained, lost), discordant, 0.5, alternative="two-sided").pvalue)
        if discordant else 1.0
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, len(seeds), size=(BOOTSTRAP_SAMPLES, len(seeds)))
    offline_float = offline.astype(np.float64)
    online_float = online.astype(np.float64)
    offline_bootstrap = offline_float[indices].mean(axis=1)
    online_bootstrap = online_float[indices].mean(axis=1)
    difference_bootstrap = (online_float - offline_float)[indices].mean(axis=1)
    return {
        "both_success": both_success,
        "both_fail": both_fail,
        "offline_fail_online_success": gained,
        "offline_success_online_fail": lost,
        "offline_fail_online_success_seeds": seeds[gained_mask].astype(int).tolist(),
        "offline_success_online_fail_seeds": seeds[lost_mask].astype(int).tolist(),
        "discordant_pairs": discordant,
        "mcnemar_exact": {
            "b_offline_success_online_fail": lost,
            "c_offline_fail_online_success": gained,
            "p_value_two_sided": p_value,
            "method": "scipy.stats.binomtest(min(b,c), n=b+c, p=0.5)",
        },
        "success_rates": {
            "offline": float(offline.mean()),
            "online": float(online.mean()),
            "paired_difference_online_minus_offline": float((online_float - offline_float).mean()),
        },
        "paired_bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "paired seed",
            "offline_success_rate_95_ci": percentile_ci(offline_bootstrap),
            "online_success_rate_95_ci": percentile_ci(online_bootstrap),
            "difference_online_minus_offline_95_ci": percentile_ci(difference_bootstrap),
        },
    }


def validate_checkpoint(path: Path, *, online_updates: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["training_config"]["observation_dim"]) != 48:
        raise RuntimeError(f"{path} is not the frozen 48-D policy")
    if np.asarray(payload["observation_mean"]).shape != (48,):
        raise RuntimeError(f"{path} normalization is not 48-D")
    actual_online = int(payload.get("online_awac_update_step", 0))
    if actual_online != online_updates:
        raise RuntimeError(f"{path} online update count {actual_online} != {online_updates}")
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        "format_version": payload.get("format_version"),
        "optimizer_step": int(payload["step"]),
        "offline_pretrain_updates": int(payload.get("offline_pretrain_updates", payload["step"])),
        "online_updates": actual_online,
        "observation_dim": 48,
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "Final Learned Expert fresh-seed evaluation",
        "",
        "Policy | Episodes | Success | Rate | Grasp | Lift | Transport | Place | Release | Retreat | Illegal Drop | IK Failure | Timeout | Avg Return | Place->Success",
    ]
    for name in ("offline_20k", "online_20k"):
        value = report["combined_summary"][name]
        lines.append(
            f"{name} | {value['episodes']} | {value['success']} | {100*value['success_rate']:.2f}% | "
            f"{value['grasp']} | {value['lift']} | {value['transport']} | {value['place']} | "
            f"{value['release']} | {value['retreat']} | {value['illegal_drop']} | "
            f"{value['ik_failure']} | {value['timeout']} | {value['average_return']:.6f} | "
            f"{100*value['place_to_success']:.2f}%"
        )
    lines.extend(["", "Seed block | Offline success | Online success | Difference"])
    for block, value in report["block_comparison"].items():
        lines.append(
            f"{block} | {value['offline_success']} | {value['online_success']} | "
            f"{value['difference_online_minus_offline']:+d}"
        )
    paired = report["paired_analysis"]
    lines.extend([
        "", f"Both success: {paired['both_success']}", f"Both fail: {paired['both_fail']}",
        f"Offline fail -> Online success: {paired['offline_fail_online_success']}",
        f"Offline success -> Online fail: {paired['offline_success_online_fail']}",
        f"McNemar exact p-value: {paired['mcnemar_exact']['p_value_two_sided']:.10g}",
        "Paired Online-Offline success difference 95% CI: "
        f"{paired['paired_bootstrap']['difference_online_minus_offline_95_ci']}",
        "", f"Final selection: {report['final_selection']['policy']}",
        f"Rationale: {report['final_selection']['rationale']}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    output = Path(
        "outputs/awac_evaluation/"
        "learned_expert_final_fresh_seed_20260814T230000Z"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    offline_path = OFFLINE.resolve(); online_path = ONLINE.resolve()
    checkpoints = {
        "offline_20k": validate_checkpoint(offline_path, online_updates=0),
        "online_20k": validate_checkpoint(online_path, online_updates=20_000),
    }
    reward_config = AWACRewardV1Config()
    evaluations: dict[str, dict[str, Any]] = {"offline_20k": {}, "online_20k": {}}
    for policy_name, checkpoint in (("offline_20k", offline_path), ("online_20k", online_path)):
        predictor = HybridCheckpointPredictor(checkpoint)
        for block_name, seeds in BLOCKS.items():
            print(f"evaluating {policy_name} block {block_name}", flush=True)
            evaluation = evaluate_policy(predictor, seeds, reward_config)
            evaluations[policy_name][block_name] = evaluation
            (output / f"{policy_name}_{block_name}.json").write_text(
                json.dumps(evaluation, indent=2) + "\n"
            )
    combined = {
        policy: summarize_rows([
            row for block in BLOCKS for row in evaluations[policy][block]["rows"]
        ])
        for policy in evaluations
    }
    block_comparison = {
        block: {
            "offline_success": int(evaluations["offline_20k"][block]["task_success"]),
            "online_success": int(evaluations["online_20k"][block]["task_success"]),
            "difference_online_minus_offline": int(
                evaluations["online_20k"][block]["task_success"]
                - evaluations["offline_20k"][block]["task_success"]
            ),
        }
        for block in BLOCKS
    }
    paired = paired_analysis(combined["offline_20k"]["rows"], combined["online_20k"]["rows"])
    offline_summary = compact(combined["offline_20k"])
    online_summary = compact(combined["online_20k"])
    difference_ci = paired["paired_bootstrap"]["difference_online_minus_offline_95_ci"]
    positive_blocks = sum(value["difference_online_minus_offline"] > 0 for value in block_comparison.values())
    online_supported = (
        online_summary["success"] > offline_summary["success"]
        and positive_blocks >= 2
        and online_summary["retreat"] >= offline_summary["retreat"]
        and online_summary["timeout"] <= offline_summary["timeout"]
        and (difference_ci[0] > 0 or paired["mcnemar_exact"]["p_value_two_sided"] < 0.05)
    )
    if online_supported:
        selection = {
            "policy": "Online AWAC +20k",
            "checkpoint": str(online_path),
            "rationale": "paired fresh-seed improvement is positive and statistically supported without Retreat/Timeout degradation",
        }
    else:
        selection = {
            "policy": "Offline AWAC 20k",
            "checkpoint": str(offline_path),
            "rationale": "Online did not meet the predeclared stable paired-improvement rule; retain the simpler frozen Offline policy",
        }
    report = {
        "experiment": "final_learned_expert_fresh_seed_evaluation",
        "evaluation_only": True,
        "gradient_updates": 0,
        "replay_appends": 0,
        "deterministic_actor": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_blocks": {name: [seeds[0], seeds[-1]] for name, seeds in BLOCKS.items()},
        "episodes_per_policy": 300,
        "checkpoints": checkpoints,
        "reward_config": reward_config.__dict__,
        "block_summary": {
            policy: {block: compact(value) for block, value in blocks.items()}
            for policy, blocks in evaluations.items()
        },
        "block_comparison": block_comparison,
        "combined_summary": {
            "offline_20k": offline_summary, "online_20k": online_summary,
        },
        "paired_analysis": paired,
        "final_selection": selection,
    }
    (output / "offline_20k_combined.json").write_text(json.dumps(combined["offline_20k"], indent=2) + "\n")
    (output / "online_20k_combined.json").write_text(json.dumps(combined["online_20k"], indent=2) + "\n")
    (output / "final_fresh_seed_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "final_fresh_seed_report.txt").write_text(render_text(report))
    print(json.dumps({
        "output": str(output), "combined": report["combined_summary"],
        "blocks": block_comparison, "paired": paired,
        "selection": selection,
    }, indent=2))


if __name__ == "__main__":
    main()
