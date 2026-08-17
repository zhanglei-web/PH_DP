"""Immutable manifests and episode-level splits for formal Rule Expert data."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import h5py

from mujoco_shared_control.data.recording import validate_episode


MANIFEST_VERSION = "1.0.0"
FORMAL_RUN_ID = "formal_rule_v1_20260812T050822Z"
FORMAL_CODE_VERSION = "d5ce43ff70af25491c545ec513d56e9f988c4f6b"
FORMAL_SCHEMA_VERSION = "2.0.0"
FORMAL_CONFIG_VERSION = "rule_collection_v1"
FORMAL_CONFIG_HASH = "9979c6328ad52121804296b512bbf500a94dd732dc37b4c2cfa7c71a897f9160"


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest_content_hash(manifest: dict[str, Any]) -> str:
    content = dict(manifest)
    content.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json(content).encode()).hexdigest()


def _category(variant: str, outcome: str, reason: str) -> str:
    if variant == "nominal" and outcome == "success":
        return "nominal_success"
    if variant == "perturbed" and outcome == "recovered" and reason == "task_success":
        return "normal_recovered"
    if variant == "perturbed" and outcome == "recovered" and reason == "delayed_recovery":
        return "delayed_recovery"
    if variant == "perturbed" and outcome == "failure":
        return "failure"
    raise ValueError(f"unsupported formal episode category: {variant}/{outcome}/{reason}")


def build_formal_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    run_id: str = FORMAL_RUN_ID,
    split_seed: int = 20260812,
    validation_fraction: float = 0.10,
    expected_categories: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate, hash, stratify, and atomically publish one formal-run manifest."""
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    expected = expected_categories or {
        "nominal_success": 1000,
        "normal_recovered": 78,
        "delayed_recovery": 66,
        "failure": 156,
    }
    paths = sorted(root.rglob(f"*{run_id}*.h5"))
    if not paths:
        raise ValueError(f"no HDF5 episodes found for run_id={run_id}")
    entries: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = defaultdict(list)
    episode_ids: set[str] = set()
    for path in paths:
        report = validate_episode(path)
        if not report["valid"]:
            raise ValueError(f"Validator rejected {path}: {report['errors']}")
        with h5py.File(path, "r") as episode:
            attrs = episode.attrs
            metadata = {
                "run_id": _text(attrs["run_id"]),
                "schema_version": _text(attrs["schema_version"]),
                "config_version": _text(attrs["config_version"]),
                "config_hash": _text(attrs["config_hash"]),
                "code_version": _text(attrs["expert_code_version"]),
            }
            required = {
                "run_id": run_id,
                "schema_version": FORMAL_SCHEMA_VERSION,
                "config_version": FORMAL_CONFIG_VERSION,
                "config_hash": FORMAL_CONFIG_HASH,
                "code_version": FORMAL_CODE_VERSION,
            }
            if metadata != required:
                raise ValueError(f"frozen metadata mismatch in {path}: {metadata}")
            episode_id = _text(attrs["episode_id"])
            if episode_id in episode_ids:
                raise ValueError(f"duplicate episode_id: {episode_id}")
            episode_ids.add(episode_id)
            variant = _text(attrs["collection_variant"])
            outcome = _text(attrs["outcome"])
            reason = _text(attrs["termination_reason"])
            category = _category(variant, outcome, reason)
            entry = {
                "episode_id": episode_id,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "transitions": len(episode["labels/reward"]),
                "worker_episode_index": int(attrs["worker_episode_index"]),
                "environment_seed": int(attrs["environment_seed"]),
                "policy_seed": int(attrs["policy_seed"]),
                "perturbation_seed": int(attrs["perturbation_seed"]),
                "variant": variant,
                "outcome": outcome,
                "termination_reason": reason,
                "category": category,
                "split": "",
            }
            groups[category].append(len(entries))
            entries.append(entry)
    counts = Counter(entry["category"] for entry in entries)
    if dict(counts) != expected:
        raise ValueError(f"formal category counts mismatch: {dict(counts)} != {expected}")
    rng = random.Random(split_seed)
    split_counts: dict[str, dict[str, int]] = {}
    for category in sorted(groups):
        indices = groups[category].copy()
        rng.shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation = set(indices[:validation_count])
        for index in indices:
            entries[index]["split"] = "validation" if index in validation else "train"
        split_counts[category] = {
            "train": len(indices) - validation_count,
            "validation": validation_count,
        }
    entries.sort(key=lambda item: item["episode_id"])
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": "rule_expert_v1_formal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": Path(os.path.relpath(root, output.parent)).as_posix(),
        "dataset_root_base": "manifest_parent",
        "run_id": run_id,
        "schema_version": FORMAL_SCHEMA_VERSION,
        "config_version": FORMAL_CONFIG_VERSION,
        "config_hash": FORMAL_CONFIG_HASH,
        "code_version": FORMAL_CODE_VERSION,
        "split": {
            "unit": "episode",
            "strategy": "stratified_exact",
            "seed": split_seed,
            "validation_fraction": validation_fraction,
            "counts": split_counts,
        },
        "category_counts": dict(sorted(counts.items())),
        "episode_count": len(entries),
        "transition_count": sum(item["transitions"] for item in entries),
        "episodes": entries,
    }
    manifest["content_sha256"] = manifest_content_hash(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".inprogress")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unsupported manifest version")
    expected = manifest_content_hash(manifest)
    if manifest.get("content_sha256") != expected:
        raise ValueError("manifest content hash mismatch")
    if len(manifest.get("episodes", [])) != int(manifest.get("episode_count", -1)):
        raise ValueError("manifest episode count mismatch")
    return manifest
