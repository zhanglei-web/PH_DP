#!/usr/bin/env python3
"""Pre-registered 1000-pair confirmatory evaluation for the frozen AWAC Actors."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import binomtest
import torch

from evaluate_awac_final_fresh_seeds import compact, sha256, summarize_rows
from mujoco_shared_control.awac.evaluation import evaluate_policy
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
SEED_START = 1_000_000
SEED_STOP = 1_001_000
BLOCKS = {
    f"{start}-{start + 99}": list(range(start, start + 100))
    for start in range(SEED_START, SEED_STOP, 100)
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_081_401


def checkpoint_info(path: Path, expected_step: int, expected_online: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        int(payload["step"]) != expected_step
        or int(payload.get("online_awac_update_step", 0)) != expected_online
        or int(payload["training_config"]["observation_dim"]) != 48
        or np.asarray(payload["observation_mean"]).shape != (48,)
    ):
        raise RuntimeError(f"frozen checkpoint identity mismatch: {path}")
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        "format_version": payload.get("format_version"),
        "optimizer_step": int(payload["step"]),
        "offline_pretrain_updates": 20_000,
        "online_updates": expected_online,
        "observation_dim": 48,
    }


def ci(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def paired_bootstrap(
    offline_rows: list[dict[str, Any]], online_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    offline = {int(row["seed"]): row for row in offline_rows}
    online = {int(row["seed"]): row for row in online_rows}
    seeds = sorted(offline)
    if seeds != sorted(online) or seeds != list(range(SEED_START, SEED_STOP)):
        raise RuntimeError("confirmatory rows are not the pre-registered 1000 paired seeds")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, len(seeds), size=(BOOTSTRAP_SAMPLES, len(seeds)))

    def array(rows: dict[int, dict[str, Any]], field: str) -> np.ndarray:
        return np.asarray([rows[seed][field] for seed in seeds], np.float64)

    def standard_metric(field: str) -> dict[str, Any]:
        a = array(offline, field); b = array(online, field)
        a_boot = a[indices].mean(axis=1); b_boot = b[indices].mean(axis=1)
        return {
            "offline": float(a.mean()), "online": float(b.mean()),
            "difference_online_minus_offline": float((b - a).mean()),
            "offline_95_ci": ci(a_boot), "online_95_ci": ci(b_boot),
            "difference_95_ci": ci(b_boot - a_boot),
        }

    metrics = {
        "success_rate": standard_metric("task_success"),
        "grasp_rate": standard_metric("grasp_success"),
        "lift_rate": standard_metric("lift_success"),
        "transport_rate": standard_metric("transport_success"),
        "release_rate": standard_metric("place_success"),
        "retreat_rate": standard_metric("retreat_success"),
        "illegal_drop_rate": standard_metric("illegal_drop"),
        "ik_failure_rate": standard_metric("ik_failure"),
        "timeout_rate": standard_metric("timeout"),
        "average_return": standard_metric("episode_return"),
        "episode_length": standard_metric("episode_length"),
    }

    def ratio_metric(numerator: str, denominator: str) -> dict[str, Any]:
        an = array(offline, numerator); ad = array(offline, denominator)
        bn = array(online, numerator); bd = array(online, denominator)
        a_boot = an[indices].sum(axis=1) / np.maximum(ad[indices].sum(axis=1), 1.0)
        b_boot = bn[indices].sum(axis=1) / np.maximum(bd[indices].sum(axis=1), 1.0)
        a_value = float(an.sum() / max(ad.sum(), 1.0))
        b_value = float(bn.sum() / max(bd.sum(), 1.0))
        return {
            "offline": a_value, "online": b_value,
            "difference_online_minus_offline": b_value - a_value,
            "offline_95_ci": ci(a_boot), "online_95_ci": ci(b_boot),
            "difference_95_ci": ci(b_boot - a_boot),
        }

    metrics["place_to_success"] = ratio_metric("task_success", "place_success")
    metrics["release_to_retreat"] = ratio_metric("retreat_success", "place_success")

    a_success = array(offline, "task_success").astype(bool)
    b_success = array(online, "task_success").astype(bool)
    both_success = int(np.sum(a_success & b_success))
    both_fail = int(np.sum(~a_success & ~b_success))
    gained = int(np.sum(~a_success & b_success))
    lost = int(np.sum(a_success & ~b_success))
    discordant = gained + lost
    mcnemar_p = (
        float(binomtest(min(gained, lost), discordant, 0.5, alternative="two-sided").pvalue)
        if discordant else 1.0
    )
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling_unit": "paired seed",
        "metrics": metrics,
        "success_pairs": {
            "both_success": both_success, "both_fail": both_fail,
            "offline_fail_online_success": gained,
            "offline_success_online_fail": lost,
            "discordant_pairs": discordant,
            "mcnemar_exact_p_value_two_sided": mcnemar_p,
            "mcnemar_method": "binomtest(min(b,c), n=b+c, p=0.5, two-sided)",
        },
    }


def release_retreat_steps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([
        row["release_to_retreat_steps"] for row in rows
        if row["release_to_retreat_steps"] is not None
    ], np.float64)
    return {
        "episodes": int(len(values)), "mean": float(values.mean()),
        "std": float(values.std()), "min": int(values.min()), "max": int(values.max()),
    }


def write_pairs_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = list(pairs[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(pairs)


def pair_rows(offline_rows: list[dict[str, Any]], online_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offline = {int(row["seed"]): row for row in offline_rows}
    online = {int(row["seed"]): row for row in online_rows}
    fields = (
        "task_success", "grasp_success", "lift_success", "transport_success",
        "place_success", "retreat_success", "illegal_drop", "ik_failure", "timeout",
        "episode_return", "episode_length", "release_step", "retreat_step",
        "release_to_retreat_steps", "termination_reason",
    )
    return [
        {
            "seed": seed,
            **{f"offline_{field}": offline[seed][field] for field in fields},
            **{f"online_{field}": online[seed][field] for field in fields},
        }
        for seed in range(SEED_START, SEED_STOP)
    ]


def main() -> None:
    output = Path(
        "outputs/awac_evaluation/"
        "learned_expert_confirmatory_1000_20260815T000000Z"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    offline_path = OFFLINE.resolve(); online_path = ONLINE.resolve()
    checkpoints = {
        "offline_20k": checkpoint_info(offline_path, 20_000, 0),
        "online_20k": checkpoint_info(online_path, 40_000, 20_000),
    }
    reward = AWACRewardV1Config()
    evaluations: dict[str, dict[str, dict[str, Any]]] = {
        "offline_20k": {}, "online_20k": {},
    }
    for policy, checkpoint in (("offline_20k", offline_path), ("online_20k", online_path)):
        predictor = HybridCheckpointPredictor(checkpoint)
        for block, seeds in BLOCKS.items():
            print(f"evaluating {policy} {block}", flush=True)
            result = evaluate_policy(predictor, seeds, reward)
            evaluations[policy][block] = result
            (output / f"{policy}_{block}.json").write_text(json.dumps(result, indent=2) + "\n")

    combined = {
        policy: summarize_rows([
            row for block in BLOCKS for row in evaluations[policy][block]["rows"]
        ])
        for policy in evaluations
    }
    pairs = pair_rows(combined["offline_20k"]["rows"], combined["online_20k"]["rows"])
    write_pairs_csv(output / "paired_per_seed.csv", pairs)
    (output / "paired_per_seed.json").write_text(json.dumps(pairs, indent=2) + "\n")
    statistics = paired_bootstrap(combined["offline_20k"]["rows"], combined["online_20k"]["rows"])
    block_table = {}
    for block in BLOCKS:
        off = compact(evaluations["offline_20k"][block])
        on = compact(evaluations["online_20k"][block])
        block_table[block] = {
            "offline_success": off["success"], "online_success": on["success"],
            "difference_online_minus_offline": on["success"] - off["success"],
            "offline_timeout": off["timeout"], "online_timeout": on["timeout"],
            "offline_illegal_drop": off["illegal_drop"], "online_illegal_drop": on["illegal_drop"],
        }
    block_directions = {
        "online_better": int(sum(v["difference_online_minus_offline"] > 0 for v in block_table.values())),
        "tie": int(sum(v["difference_online_minus_offline"] == 0 for v in block_table.values())),
        "offline_better": int(sum(v["difference_online_minus_offline"] < 0 for v in block_table.values())),
    }
    primary = statistics["metrics"]["success_rate"]
    pairs_success = statistics["success_pairs"]
    statistically_confirmed = bool(
        primary["online"] > primary["offline"]
        and primary["difference_95_ci"][0] > 0
        and pairs_success["mcnemar_exact_p_value_two_sided"] < 0.05
    )
    offline_summary = compact(combined["offline_20k"]); online_summary = compact(combined["online_20k"])
    engineering_supported = bool(
        online_summary["success"] >= offline_summary["success"]
        and online_summary["retreat"] >= offline_summary["retreat"]
        and online_summary["timeout"] <= offline_summary["timeout"]
        and online_summary["illegal_drop"] <= offline_summary["illegal_drop"]
        and online_summary["ik_failure"] <= offline_summary["ik_failure"]
        and online_summary["average_return"] >= offline_summary["average_return"]
    )
    if statistically_confirmed:
        selection = {
            "policy": "Online AWAC +20k", "checkpoint": str(online_path),
            "basis": "statistically confirmed improvement",
        }
    elif engineering_supported:
        selection = {
            "policy": "Online AWAC +20k", "checkpoint": str(online_path),
            "basis": "performance-based expert selection",
        }
    else:
        selection = {
            "policy": "Offline AWAC 20k", "checkpoint": str(offline_path),
            "basis": "Online did not preserve the pre-registered engineering performance criteria",
        }
    report = {
        "experiment": "final_learned_expert_confirmatory_1000_paired_seeds",
        "evaluation_only": True, "gradient_updates": 0, "replay_appends": 0,
        "deterministic_actor": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_endpoint": "task_success_rate",
        "pre_registered_seeds": [SEED_START, SEED_STOP - 1],
        "episodes_per_policy": 1000,
        "previous_300_evidence_not_pooled": {
            "offline_success_rate": 0.9266666666666666,
            "online_success_rate": 0.9533333333333334,
            "difference_online_minus_offline": 0.02666666666666667,
        },
        "checkpoints": checkpoints, "reward_config": asdict(reward),
        "robustness_test": {
            "performed": False,
            "reason": "no formal pre-existing Easy/Medium/Hard difficulty definition was found",
        },
        "combined_summary": {
            "offline_20k": offline_summary, "online_20k": online_summary,
        },
        "release_to_retreat_steps": {
            "offline_20k": release_retreat_steps(combined["offline_20k"]["rows"]),
            "online_20k": release_retreat_steps(combined["online_20k"]["rows"]),
        },
        "block_table": block_table, "block_directions": block_directions,
        "paired_statistics": statistics,
        "statistically_confirmed": statistically_confirmed,
        "engineering_supported": engineering_supported,
        "final_selection": selection,
    }
    (output / "offline_20k_combined.json").write_text(json.dumps(combined["offline_20k"], indent=2) + "\n")
    (output / "online_20k_combined.json").write_text(json.dumps(combined["online_20k"], indent=2) + "\n")
    (output / "confirmatory_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "confirmatory_report.txt").write_text(
        json.dumps({
            "combined": report["combined_summary"], "blocks": block_table,
            "paired": statistics, "selection": selection,
        }, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str(output), "combined": report["combined_summary"],
        "blocks": block_table, "block_directions": block_directions,
        "paired": statistics, "selection": selection,
    }, indent=2))


if __name__ == "__main__":
    main()
