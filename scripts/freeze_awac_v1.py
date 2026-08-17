#!/usr/bin/env python3
"""Freeze the 1234-episode, Reward-V1 AWAC corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from mujoco_shared_control.awac.dataset import convert_formal_rule_dataset
from mujoco_shared_control.awac.reward import AWACRewardV1Config, derive_awac_reward_v1
from mujoco_shared_control.collection.manifest import sha256_file


EXCLUDED = ("delayed_recovery",)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _content_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/rule_expert_v1_formal.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/awac_dataset/awac_v1_formal_rule"))
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    formal = json.loads(manifest_path.read_text())
    selected = [item for item in formal["episodes"] if item["category"] not in EXCLUDED]

    with tempfile.TemporaryDirectory(prefix="awac_v1_freeze_") as temporary_name:
        temporary = Path(temporary_name)
        base_dir = temporary / "base"
        reward_dir = temporary / "reward"
        base_report = convert_formal_rule_dataset(
            manifest_path, base_dir, verify_checksums=True,
            excluded_categories=EXCLUDED,
        )
        reward_report = derive_awac_reward_v1(
            manifest_path, base_dir, reward_dir,
            config=AWACRewardV1Config(), excluded_categories=EXCLUDED,
        )

        # The user explicitly requested replacement of the old delayed-recovery NPZs.
        for split in ("train", "validation"):
            destination = output / f"{split}.npz"
            staged = output / f"{split}.npz.inprogress"
            shutil.copy2(reward_dir / f"{split}.npz", staged)
            if destination.exists():
                destination.unlink()
            staged.replace(destination)

    derived_manifest = {
        "manifest_version": "awac_v1_frozen_manifest_v1",
        "dataset_name": "awac_v1_formal_rule_reward_v1",
        "source_manifest": str(manifest_path),
        "source_manifest_content_sha256": formal["content_sha256"],
        "excluded_categories": list(EXCLUDED),
        "episode_count": len(selected),
        "transition_count_before_action_filter": sum(int(item["transitions"]) for item in selected),
        "episodes": selected,
    }
    derived_manifest["content_sha256"] = _content_hash(derived_manifest)
    _atomic_json(output / "manifest.json", derived_manifest)

    categories: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    network_nonfinite = 0
    action_min, action_max = float("inf"), float("-inf")
    terminal = Counter()
    for split in ("train", "validation"):
        with np.load(output / f"{split}.npz", allow_pickle=False) as data:
            episode_ids = np.unique(data["episode_id"])
            splits[split] = len(episode_ids)
            transitions[split] = len(data["reward"])
            for category in np.unique(data["category"]):
                categories[str(category)] += len(np.unique(data["episode_id"][data["category"] == category]))
            network_nonfinite += sum(int((~np.isfinite(data[name])).sum()) for name in ("obs", "action", "reward", "next_obs"))
            action_min = min(action_min, float(data["action"].min()))
            action_max = max(action_max, float(data["action"].max()))
            terminal["success"] += int(np.sum(data["terminated"] & data["task_success"]))
            terminal["failure"] += int(np.sum(data["terminated"] & ~data["task_success"]))
            terminal["truncated"] += int(data["truncated"].sum())
            if data["obs"].shape[1:] != (42,) or data["next_obs"].shape[1:] != (42,) or data["action"].shape[1:] != (7,):
                raise RuntimeError("frozen AWAC-v1 dimensions are invalid")
    if categories.get("delayed_recovery", 0) != 0 or sum(splits.values()) != 1234:
        raise RuntimeError("frozen dataset still contains delayed recovery or has wrong episode count")

    report = {
        "dataset_version": "awac_v1_frozen_1234_reward_v1",
        "derived_manifest": str((output / "manifest.json").resolve()),
        "derived_manifest_sha256": sha256_file(output / "manifest.json"),
        "source_formal_manifest_unchanged": True,
        "excluded_categories": list(EXCLUDED),
        "episode_count": sum(splits.values()),
        "episodes_by_split": dict(splits),
        "episodes_by_category": {name: int(categories.get(name, 0)) for name in (
            "nominal_success", "normal_recovered", "delayed_recovery", "failure"
        )},
        "transition_count": sum(transitions.values()),
        "transitions_by_split": dict(transitions),
        "terminal_rows": dict(terminal),
        "state": {"source": "observations/policy_state_42", "shape": [42]},
        "action": {"source": "actions/normalized", "shape": [7], "minimum": action_min, "maximum": action_max, "expert_nominal_metadata_only": True},
        "network_nonfinite_value_count": network_nonfinite,
        "reward_version": "awac_reward_v1",
        "reward_config": reward_report["reward_config"],
        "source_conversion": base_report,
        "terminal_reconstruction": reward_report["terminal_reconstruction"],
        "filtered_tail_summary": reward_report["filtered_tail_summary"],
        "files": {
            split: {"path": f"{split}.npz", "sha256": sha256_file(output / f"{split}.npz"), "transitions": transitions[split]}
            for split in ("train", "validation")
        },
        "training_started_by_this_script": False,
    }
    _atomic_json(output / "report.json", report)
    print(json.dumps({key: report[key] for key in (
        "episode_count", "episodes_by_split", "episodes_by_category",
        "transition_count", "transitions_by_split", "terminal_rows"
    )}, indent=2))


if __name__ == "__main__":
    main()
