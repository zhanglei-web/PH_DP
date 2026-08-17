#!/usr/bin/env python3
"""Audit rewards stored in an AWAC transition dataset without changing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_shared_control.data.recording import FrameEvent
from mujoco_shared_control.experts.rule_pick_place import RuleExpertStage


CATEGORY_NAMES = {
    "nominal_success": "normal_success",
    "normal_recovered": "normal_recovery",
    "delayed_recovery": "delayed_recovery",
    "failure": "failure",
}


def _stats(values: np.ndarray, *, percentiles: bool = False) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {"count": int(values.size)}
    if not values.size:
        result.update({key: None for key in ("mean", "std", "min", "max")})
        if percentiles:
            result.update({key: None for key in ("p10", "p50", "p90")})
        return result
    result.update(
        {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    )
    if percentiles:
        p10, p50, p90 = np.percentile(values, [10, 50, 90])
        result.update({"p10": float(p10), "p50": float(p50), "p90": float(p90)})
    return result


def _superiority(left: np.ndarray, right: np.ndarray) -> float | None:
    if not len(left) or not len(right):
        return None
    comparison = left[:, None] - right[None, :]
    return float(np.mean(comparison > 0) + 0.5 * np.mean(comparison == 0))


def _mean_ratio(left: np.ndarray, right: np.ndarray) -> float | None:
    if not len(left) or not len(right) or float(right.mean()) == 0.0:
        return None
    return float(left.mean() / right.mean())


def _mean_difference(left: np.ndarray, right: np.ndarray) -> float | None:
    if not len(left) or not len(right):
        return None
    return float(left.mean() - right.mean())


def _load(paths: list[Path]) -> dict[str, np.ndarray]:
    required = {
        "reward",
        "episode_id",
        "step_index",
        "category",
        "expert_stage",
        "events",
        "task_milestones",
        "task_success",
        "terminated",
        "truncated",
        "termination_reason",
    }
    parts: dict[str, list[np.ndarray]] = {name: [] for name in required}
    episode_splits: dict[str, str] = {}
    for path in paths:
        split = path.stem
        with np.load(path, allow_pickle=False) as dataset:
            missing = required - set(dataset.files)
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            length = len(dataset["reward"])
            if any(len(dataset[name]) != length for name in required):
                raise ValueError(f"{path} contains transition-length mismatch")
            for episode_id in np.unique(dataset["episode_id"]):
                key = str(episode_id)
                if key in episode_splits:
                    raise ValueError(f"episode appears in multiple NPZ files: {key}")
                episode_splits[key] = split
            for name in required:
                parts[name].append(np.asarray(dataset[name]))
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    arrays["split"] = np.asarray(
        [episode_splits[str(episode_id)] for episode_id in arrays["episode_id"]]
    )
    if not np.isfinite(arrays["reward"]).all():
        raise ValueError("frozen AWAC reward contains NaN or Inf")
    return arrays


def _episode_indices(arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
    order = np.lexsort((arrays["step_index"], arrays["episode_id"]))
    episode_ids = arrays["episode_id"][order]
    boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    return [chunk for chunk in np.split(order, boundaries) if len(chunk)]


def _rising_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    previous = np.r_[False, values[:-1]]
    return values & ~previous


def build_audit(paths: list[Path]) -> dict[str, Any]:
    arrays = _load(paths)
    groups = _episode_indices(arrays)
    episode_records: list[dict[str, Any]] = []
    event_values: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in ("before", "at", "after", "after_minus_before")}
        for name in ("grasp", "lift", "transport", "release", "success", "drop_failure")
    }
    event_episode_ids: dict[str, set[str]] = {name: set() for name in event_values}
    event_occurrences = {name: 0 for name in event_values}
    drop_occurrences = 0
    failure_boundary_occurrences = 0
    background_stage_rewards: dict[str, list[float]] = {
        "OPEN_GRIPPER": [], "RETREAT": []
    }

    for indices in groups:
        episode_id = str(arrays["episode_id"][indices[0]])
        source_category = str(arrays["category"][indices[0]])
        if source_category not in CATEGORY_NAMES:
            raise ValueError(f"unknown AWAC category: {source_category}")
        if np.any(arrays["category"][indices] != source_category):
            raise ValueError(f"category changes within episode {episode_id}")
        rewards = arrays["reward"][indices].astype(np.float64)
        steps = arrays["step_index"][indices].astype(np.int64)
        events = arrays["events"][indices].astype(np.uint32)
        milestones = arrays["task_milestones"][indices].astype(bool)
        expert_stages = arrays["expert_stage"][indices].astype(np.uint8)
        terminated = arrays["terminated"][indices].astype(bool)
        truncated = arrays["truncated"][indices].astype(bool)
        termination_reasons = arrays["termination_reason"][indices]
        category = CATEGORY_NAMES[source_category]
        explicit_terminal = terminated | truncated
        episode_records.append(
            {
                "episode_id": episode_id,
                "category": category,
                "split": str(arrays["split"][indices[0]]),
                "return": float(rewards.sum(dtype=np.float64)),
                "length": len(indices),
                "last_retained_reward": float(rewards[-1]),
                "explicit_terminal_count": int(explicit_terminal.sum()),
                "explicit_terminal_rewards": rewards[explicit_terminal],
                "last_task_success": bool(arrays["task_success"][indices[-1]]),
                "last_terminated": bool(terminated[-1]),
                "last_truncated": bool(truncated[-1]),
                "last_termination_reason": str(termination_reasons[-1]),
            }
        )

        milestone_edges = np.column_stack(
            [_rising_edges(milestones[:, column]) for column in range(5)]
        )
        no_event_or_terminal = ~milestone_edges.any(axis=1) & ~explicit_terminal
        background_stage_rewards["OPEN_GRIPPER"].extend(
            rewards[no_event_or_terminal & (expert_stages == int(RuleExpertStage.OPEN_GRIPPER))]
        )
        background_stage_rewards["RETREAT"].extend(
            rewards[no_event_or_terminal & (expert_stages == int(RuleExpertStage.RETREAT))]
        )
        grasp = ((events & int(FrameEvent.GRASP_ACQUIRED)) != 0) | milestone_edges[:, 0]
        lift = milestone_edges[:, 1]
        transport = milestone_edges[:, 2]
        release = milestone_edges[:, 3]
        success = ((events & int(FrameEvent.TASK_SUCCESS)) != 0) | arrays[
            "task_success"
        ][indices].astype(bool)
        grasp_lost = (events & int(FrameEvent.GRASP_LOST)) != 0
        intended_release_stage = np.isin(
            expert_stages,
            [int(RuleExpertStage.OPEN_GRIPPER), int(RuleExpertStage.RETREAT)],
        )
        drop = grasp_lost & ~intended_release_stage
        failure_boundary = np.zeros(len(indices), dtype=bool)
        if category == "failure":
            failure_boundary[-1] = True
        drop_failure = drop | failure_boundary
        drop_occurrences += int(drop.sum())
        failure_boundary_occurrences += int(failure_boundary.sum())

        masks = {
            "grasp": grasp,
            "lift": lift,
            "transport": transport,
            "release": release,
            "success": success,
            "drop_failure": drop_failure,
        }
        for name, mask in masks.items():
            positions = np.flatnonzero(mask)
            event_occurrences[name] += len(positions)
            if len(positions):
                event_episode_ids[name].add(episode_id)
            for position in positions:
                event_values[name]["at"].append(float(rewards[position]))
                before_available = position > 0 and steps[position] - steps[position - 1] == 1
                after_available = (
                    position + 1 < len(indices)
                    and steps[position + 1] - steps[position] == 1
                )
                if before_available:
                    event_values[name]["before"].append(float(rewards[position - 1]))
                if after_available:
                    event_values[name]["after"].append(float(rewards[position + 1]))
                if before_available and after_available:
                    event_values[name]["after_minus_before"].append(
                        float(rewards[position + 1] - rewards[position - 1])
                    )

    category_audit: dict[str, Any] = {}
    returns_by_category: dict[str, np.ndarray] = {}
    for category in CATEGORY_NAMES.values():
        records = [record for record in episode_records if record["category"] == category]
        returns = np.asarray([record["return"] for record in records], np.float64)
        lengths = np.asarray([record["length"] for record in records], np.float64)
        last_rewards = np.asarray(
            [record["last_retained_reward"] for record in records], np.float64
        )
        explicit_rewards = np.concatenate(
            [record["explicit_terminal_rewards"] for record in records]
        ) if records else np.empty(0, np.float64)
        returns_by_category[category] = returns
        category_audit[category] = {
            "episode_count": len(records),
            "transition_count": int(lengths.sum()),
            "episode_return": _stats(returns, percentiles=True),
            "episode_length": _stats(lengths),
            "terminal_step_reward": {
                "definition": "reward of the last retained transition in each episode",
                **_stats(last_rewards),
            },
            "explicit_terminal_rows": {
                "episode_count_with_flag": sum(
                    record["explicit_terminal_count"] > 0 for record in records
                ),
                "transition_count": int(sum(
                    record["explicit_terminal_count"] for record in records
                )),
                "reward": _stats(explicit_rewards),
            },
        }

    stage_audit: dict[str, Any] = {}
    for stage_value in sorted(np.unique(arrays["expert_stage"]).astype(int)):
        mask = arrays["expert_stage"] == stage_value
        try:
            stage_name = RuleExpertStage(stage_value).name
        except ValueError:
            stage_name = "UNKNOWN"
        stage_audit[str(stage_value)] = {
            "name": stage_name,
            "transition_count": int(mask.sum()),
            "reward": _stats(arrays["reward"][mask]),
        }

    events_audit = {
        name: {
            "occurrence_count": event_occurrences[name],
            "episode_count": len(event_episode_ids[name]),
            "reward_before": _stats(np.asarray(values["before"])),
            "reward_at": _stats(np.asarray(values["at"])),
            "reward_after": _stats(np.asarray(values["after"])),
            "reward_after_minus_before": _stats(
                np.asarray(values["after_minus_before"])
            ),
        }
        for name, values in event_values.items()
    }
    events_audit["drop_failure"]["components"] = {
        "illegal_or_early_grasp_lost_occurrences": drop_occurrences,
        "failure_episode_last_retained_boundaries": failure_boundary_occurrences,
    }

    successful_returns = np.concatenate(
        [returns_by_category[name] for name in (
            "normal_success", "normal_recovery", "delayed_recovery"
        )]
    )
    failure_returns = returns_by_category["failure"]
    success_percentiles_for_failures = np.asarray(
        [np.mean(successful_returns < value) for value in failure_returns], np.float64
    )
    transition_rewards = arrays["reward"].astype(np.float64)
    absolute_rewards = np.abs(transition_rewards)
    p99_abs = float(np.percentile(absolute_rewards, 99))
    max_abs = float(absolute_rewards.max())

    success_last = np.asarray(
        [record["last_retained_reward"] for record in episode_records
         if record["category"] != "failure"], np.float64
    )
    failure_last = np.asarray(
        [record["last_retained_reward"] for record in episode_records
         if record["category"] == "failure"], np.float64
    )
    resolved_success_terminal = np.asarray(
        [record["last_retained_reward"] for record in episode_records
         if record["last_terminated"] and record["last_task_success"]], np.float64
    )
    resolved_failure_terminal = np.asarray(
        [record["last_retained_reward"] for record in episode_records
         if record["last_terminated"] and not record["last_task_success"]], np.float64
    )
    censored_terminal = np.asarray(
        [record["last_retained_reward"] for record in episode_records
         if record["last_truncated"]], np.float64
    )
    checks = {
        "success_vs_failure_return": {
            "success_episode_count": len(successful_returns),
            "failure_episode_count": len(failure_returns),
            "success_mean_minus_failure_mean": float(
                successful_returns.mean() - failure_returns.mean()
            ),
            "success_median_minus_failure_median": float(
                np.median(successful_returns) - np.median(failure_returns)
            ),
            "pairwise_probability_success_return_exceeds_failure": _superiority(
                successful_returns, failure_returns
            ),
            "success_p10_exceeds_failure_p90": bool(
                np.percentile(successful_returns, 10)
                > np.percentile(failure_returns, 90)
            ),
        },
        "recovery_return": {
            "normal_recovery_mean_over_normal_success_mean": _mean_ratio(
                returns_by_category["normal_recovery"], returns_by_category["normal_success"]
            ),
            "delayed_recovery_mean_over_normal_success_mean": _mean_ratio(
                returns_by_category["delayed_recovery"], returns_by_category["normal_success"]
            ),
            "normal_recovery_pairwise_probability_exceeds_failure": _superiority(
                returns_by_category["normal_recovery"], failure_returns
            ),
            "delayed_recovery_pairwise_probability_exceeds_failure": _superiority(
                returns_by_category["delayed_recovery"], failure_returns
            ),
            "normal_recovery_mean_minus_failure_mean": _mean_difference(
                returns_by_category["normal_recovery"], failure_returns
            ),
            "delayed_recovery_mean_minus_failure_mean": _mean_difference(
                returns_by_category["delayed_recovery"], failure_returns
            ),
        },
        "high_return_failures": {
            "failure_count_above_success_p10": int(np.sum(
                failure_returns > np.percentile(successful_returns, 10)
            )),
            "failure_count_above_success_p50": int(np.sum(
                failure_returns > np.percentile(successful_returns, 50)
            )),
            "failure_count_above_success_p90": int(np.sum(
                failure_returns > np.percentile(successful_returns, 90)
            )),
            "maximum_failure_return": float(failure_returns.max()),
            "maximum_failure_percentile_among_success_returns": float(
                success_percentiles_for_failures.max()
            ),
            "median_failure_percentile_among_success_returns": float(
                np.median(success_percentiles_for_failures)
            ),
        },
        "reward_scale": {
            "transition_reward": _stats(transition_rewards, percentiles=True),
            "absolute_reward_p99": p99_abs,
            "absolute_reward_p99_9": float(np.percentile(absolute_rewards, 99.9)),
            "absolute_reward_max": max_abs,
            "max_abs_over_p99_abs": max_abs / max(p99_abs, 1e-12),
            "nonfinite_value_count": int((~np.isfinite(transition_rewards)).sum()),
            "scale_explosion_heuristic": bool(
                max_abs > 100.0 or max_abs / max(p99_abs, 1e-12) > 100.0
            ),
            "heuristic_definition": "max |r| > 100 or max |r| / P99(|r|) > 100",
        },
        "terminal_reward_separation": {
            "terminal_definition": "last retained transition per episode",
            "success": _stats(success_last),
            "failure": _stats(failure_last),
            "success_mean_minus_failure_mean": float(
                success_last.mean() - failure_last.mean()
            ),
            "pairwise_probability_success_exceeds_failure": _superiority(
                success_last, failure_last
            ),
            "reward_ranges_overlap": bool(
                failure_last.max() >= success_last.min()
                and success_last.max() >= failure_last.min()
            ),
            "failure_count_above_success_terminal_p10": int(np.sum(
                failure_last > np.percentile(success_last, 10)
            )),
            "resolved_terminal_rows": {
                "success": _stats(resolved_success_terminal),
                "failure": _stats(resolved_failure_terminal),
                "censored_truncation": _stats(censored_terminal),
                "resolved_success_minus_failure_mean": (
                    float(resolved_success_terminal.mean() - resolved_failure_terminal.mean())
                    if len(resolved_success_terminal) and len(resolved_failure_terminal)
                    else None
                ),
                "pairwise_probability_resolved_success_exceeds_failure": _superiority(
                    resolved_success_terminal, resolved_failure_terminal
                ),
            },
            "explicit_terminal_flag_limitation": (
                "Truncated rows are censored boundaries, not resolved success/failure "
                "terminals; inspect resolved_terminal_rows separately."
            ),
        },
        "no_sustained_open_retreat_reward": {
            "definition": "rows in OPEN_GRIPPER/RETREAT excluding milestone rising edges and terminal/truncated boundaries",
            "OPEN_GRIPPER": _stats(np.asarray(background_stage_rewards["OPEN_GRIPPER"])),
            "RETREAT": _stats(np.asarray(background_stage_rewards["RETREAT"])),
            "all_absolute_rewards_below_0_1": bool(
                all(
                    abs(value) < 0.1
                    for values in background_stage_rewards.values()
                    for value in values
                )
            ),
        },
    }

    return {
        "audit_version": "awac_reward_audit_v2",
        "input_files": [str(path.resolve()) for path in paths],
        "definitions": {
            "return": "undiscounted sum of stored reward over retained transitions",
            "std": "population standard deviation (ddof=0)",
            "percentile": "NumPy linear percentile",
            "event_before_after": (
                "immediately adjacent retained transition only when step_index is contiguous"
            ),
            "grasp": "GRASP_ACQUIRED event or grasp milestone rising edge",
            "lift": "lift milestone rising edge",
            "transport": "transport milestone rising edge",
            "release": "release milestone rising edge",
            "success": "TASK_SUCCESS event or task_success flag",
            "drop_failure": (
                "GRASP_LOST outside OPEN_GRIPPER/RETREAT, union the last retained "
                "transition of every failure episode"
            ),
        },
        "total_episode_count": len(episode_records),
        "total_transition_count": len(arrays["reward"]),
        "categories": category_audit,
        "expert_stages": stage_audit,
        "events": events_audit,
        "key_checks": checks,
    }


def _number(value: Any) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def render_text_report(audit: dict[str, Any]) -> str:
    lines = [
        "AWAC Reward Audit",
        "=================",
        "",
        f"Episodes: {audit['total_episode_count']}",
        f"Transitions: {audit['total_transition_count']}",
        "Reward was read from the selected train/validation NPZ files and was not modified by this audit.",
        "No model was trained.",
        "",
        "Category summary",
        "----------------",
        "category | episodes | transitions | return mean/std/min/max | P10/P50/P90 | length mean/std | terminal reward mean/min/max",
    ]
    for name, values in audit["categories"].items():
        returns = values["episode_return"]
        lengths = values["episode_length"]
        terminal = values["terminal_step_reward"]
        lines.append(
            f"{name} | {values['episode_count']} | {values['transition_count']} | "
            f"{_number(returns['mean'])}/{_number(returns['std'])}/"
            f"{_number(returns['min'])}/{_number(returns['max'])} | "
            f"{_number(returns['p10'])}/{_number(returns['p50'])}/"
            f"{_number(returns['p90'])} | "
            f"{_number(lengths['mean'])}/{_number(lengths['std'])} | "
            f"{_number(terminal['mean'])}/{_number(terminal['min'])}/"
            f"{_number(terminal['max'])}"
        )

    lines.extend(["", "Expert-stage summary", "--------------------"])
    for stage, values in audit["expert_stages"].items():
        reward = values["reward"]
        lines.append(
            f"{stage} {values['name']}: n={values['transition_count']}, "
            f"mean={_number(reward['mean'])}, std={_number(reward['std'])}, "
            f"min={_number(reward['min'])}, max={_number(reward['max'])}"
        )

    lines.extend(["", "Event-local rewards", "-------------------"])
    for name, values in audit["events"].items():
        lines.append(
            f"{name}: occurrences={values['occurrence_count']}, "
            f"episodes={values['episode_count']}, "
            f"before={_number(values['reward_before']['mean'])} "
            f"(n={values['reward_before']['count']}), "
            f"at={_number(values['reward_at']['mean'])} "
            f"(n={values['reward_at']['count']}), "
            f"after={_number(values['reward_after']['mean'])} "
            f"(n={values['reward_after']['count']}), "
            f"after-before={_number(values['reward_after_minus_before']['mean'])} "
            f"(n={values['reward_after_minus_before']['count']})"
        )

    checks = audit["key_checks"]
    success = checks["success_vs_failure_return"]
    recovery = checks["recovery_return"]
    high_failure = checks["high_return_failures"]
    scale = checks["reward_scale"]
    terminal = checks["terminal_reward_separation"]
    sustained = checks["no_sustained_open_retreat_reward"]
    lines.extend(
        [
            "",
            "Key checks",
            "----------",
            f"1. Success-vs-failure mean return gap: {_number(success['success_mean_minus_failure_mean'])}.",
            f"   Pairwise P(success return > failure return): {_number(success['pairwise_probability_success_return_exceeds_failure'])}.",
            f"   Success P10 > failure P90: {success['success_p10_exceeds_failure_p90']}.",
            f"2. Normal-recovery/normal-success mean ratio: {_number(recovery['normal_recovery_mean_over_normal_success_mean'])}.",
            f"   Delayed-recovery/normal-success mean ratio: {_number(recovery['delayed_recovery_mean_over_normal_success_mean'])}.",
            f"   Delayed-recovery mean minus failure mean: "
            f"{_number(recovery['delayed_recovery_mean_minus_failure_mean'])}.",
            f"3. Failures above success P10/P50/P90: "
            f"{high_failure['failure_count_above_success_p10']}/"
            f"{high_failure['failure_count_above_success_p50']}/"
            f"{high_failure['failure_count_above_success_p90']}.",
            f"   Maximum failure lies at success percentile: {_number(high_failure['maximum_failure_percentile_among_success_returns'])}.",
            f"4. Transition reward min/max: {_number(scale['transition_reward']['min'])}/"
            f"{_number(scale['transition_reward']['max'])}; max |r| / P99(|r|): "
            f"{_number(scale['max_abs_over_p99_abs'])}.",
            f"   Scale-explosion heuristic: {scale['scale_explosion_heuristic']}.",
            f"5. Last-retained terminal reward success/failure mean gap: "
            f"{_number(terminal['success_mean_minus_failure_mean'])}.",
            f"   Pairwise P(success terminal > failure terminal): "
            f"{_number(terminal['pairwise_probability_success_exceeds_failure'])}.",
            f"   Terminal reward ranges overlap: {terminal['reward_ranges_overlap']}; "
            f"failure rows above success-terminal P10: "
            f"{terminal['failure_count_above_success_terminal_p10']}.",
            "",
            "Resolved terminal rows",
            "----------------------",
            f"Success: n={terminal['resolved_terminal_rows']['success']['count']}, "
            f"mean={_number(terminal['resolved_terminal_rows']['success']['mean'])}.",
            f"Failure: n={terminal['resolved_terminal_rows']['failure']['count']}, "
            f"mean={_number(terminal['resolved_terminal_rows']['failure']['mean'])}.",
            f"Censored truncation: n={terminal['resolved_terminal_rows']['censored_truncation']['count']}, "
            f"mean={_number(terminal['resolved_terminal_rows']['censored_truncation']['mean'])}.",
            f"Resolved success-minus-failure terminal mean gap: "
            f"{_number(terminal['resolved_terminal_rows']['resolved_success_minus_failure_mean'])}.",
            "",
            "Interpretation",
            "--------------",
            f"- Success P10 exceeds failure P90: {success['success_p10_exceeds_failure_p90']}.",
            f"- Normal-recovery/normal-success mean ratio is {_number(recovery['normal_recovery_mean_over_normal_success_mean'])}.",
            f"- Delayed-recovery/failure mean gap is {_number(recovery['delayed_recovery_mean_minus_failure_mean'])}; delayed recovery is censored when its successful tail is absent.",
            f"- Failures above combined-success P10/P50/P90: {high_failure['failure_count_above_success_p10']}/{high_failure['failure_count_above_success_p50']}/{high_failure['failure_count_above_success_p90']}.",
            f"- Reward scale explosion heuristic: {scale['scale_explosion_heuristic']} (range {_number(scale['transition_reward']['min'])} to {_number(scale['transition_reward']['max'])}).",
            f"- Resolved terminal success/failure pairwise separation: {_number(terminal['resolved_terminal_rows']['pairwise_probability_resolved_success_exceeds_failure'])}.",
            f"- OPEN_GRIPPER background reward mean/min/max: {_number(sustained['OPEN_GRIPPER']['mean'])}/{_number(sustained['OPEN_GRIPPER']['min'])}/{_number(sustained['OPEN_GRIPPER']['max'])}.",
            f"- RETREAT background reward mean/min/max: {_number(sustained['RETREAT']['mean'])}/{_number(sustained['RETREAT']['min'])}/{_number(sustained['RETREAT']['max'])}; all |reward| < 0.1: {sustained['all_absolute_rewards_below_0_1']}.",
            "",
            "Terminal caveat",
            "---------------",
            terminal["explicit_terminal_flag_limitation"],
            "The requested terminal-step statistics therefore use each episode's last retained transition.",
            "",
            "Event definitions",
            "-----------------",
        ]
    )
    for name, definition in audit["definitions"].items():
        lines.append(f"{name}: {definition}")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("outputs/awac_dataset/awac_v1_formal_rule"),
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    audit = build_audit(
        [dataset_dir / "train.npz", dataset_dir / "validation.npz"]
    )
    _atomic_write(
        dataset_dir / "reward_audit.json",
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write(dataset_dir / "reward_audit.txt", render_text_report(audit))
    print(dataset_dir / "reward_audit.json")
    print(dataset_dir / "reward_audit.txt")


if __name__ == "__main__":
    main()
