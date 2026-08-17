"""Build an immutable AWAC-v1 transition corpus from the formal Rule manifest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import json

import h5py
import numpy as np

from mujoco_shared_control.collection.manifest import (
    load_manifest,
    sha256_file,
)
from mujoco_shared_control.experts.interfaces import ExpertActionSpec


AWAC_DATASET_VERSION = "awac_transition_dataset_v1"
SPLITS = ("train", "validation")
CATEGORIES = (
    "nominal_success",
    "normal_recovered",
    "delayed_recovery",
    "failure",
)

NETWORK_FIELDS = {
    "obs": (np.float32, (42,)),
    "action": (np.float32, (7,)),
    "reward": (np.float32, ()),
    "next_obs": (np.float32, (42,)),
    "terminated": (np.bool_, ()),
    "truncated": (np.bool_, ()),
}

METADATA_FIELDS = {
    "episode_id": (None, ()),
    "step_index": (np.int64, ()),
    "expert_stage": (np.uint8, ()),
    "stage": (np.uint8, ()),
    "task_success": (np.bool_, ()),
    "termination_reason": (None, ()),
    "events": (np.uint32, ()),
    "task_milestones": (np.uint8, (5,)),
    "perturbation_active": (np.bool_, ()),
    "perturbation_magnitude": (np.float32, ()),
    "expert_valid": (np.bool_, ()),
    "status": (np.uint8, (4,)),
    "rejection_reason": (None, ()),
    "expert_nominal": (np.float32, (7,)),
    "category": (None, ()),
}

REQUIRED_DATASETS = {
    "observations/policy_state_42": (42,),
    "next_observations/policy_state_42": (42,),
    "actions/normalized": (7,),
    "actions/expert_nominal": (7,),
    "actions/command_after_clipping": (7,),
    "actions/executed_joint_target": (8,),
    "actions/mujoco_ctrl": (8,),
    "actions/expert_valid": (),
    "actions/status": (4,),
    "actions/rejection_reason": (),
    "identity/step_index": (),
    "labels/reward": (),
    "labels/terminated": (),
    "labels/truncated": (),
    "labels/task_success": (),
    "labels/termination_reason": (),
    "labels/expert_stage": (),
    "labels/stage": (),
    "labels/events": (),
    "labels/task_milestones": (5,),
    "perturbations/active": (),
    "perturbations/magnitude": (),
}


def _decode_strings(values: np.ndarray) -> np.ndarray:
    decoded = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]
    width = max((len(value) for value in decoded), default=1)
    return np.asarray(decoded, dtype=f"U{width}")


def _validate_manifest_episode(
    episode: h5py.File,
    item: dict[str, Any],
    manifest: dict[str, Any],
) -> int:
    expected_attrs = {
        "episode_id": item["episode_id"],
        "run_id": manifest["run_id"],
        "schema_version": manifest["schema_version"],
        "config_version": manifest["config_version"],
        "config_hash": manifest["config_hash"],
        "expert_code_version": manifest["code_version"],
    }
    actual_attrs = {key: str(episode.attrs.get(key, "")) for key in expected_attrs}
    if actual_attrs != expected_attrs:
        raise ValueError(
            f"manifest metadata mismatch for {item['episode_id']}: {actual_attrs}"
        )
    length = int(item["transitions"])
    for name, tail in REQUIRED_DATASETS.items():
        if name not in episode:
            raise ValueError(f"{item['episode_id']} is missing dataset {name}")
        if episode[name].shape != (length, *tail):
            raise ValueError(
                f"{item['episode_id']} has invalid {name} shape "
                f"{episode[name].shape}; expected {(length, *tail)}"
            )
    return length


def _expected_normalized_action(clipped: np.ndarray) -> np.ndarray:
    spec = ExpertActionSpec()
    expected = clipped.astype(np.float64, copy=True)
    expected[:, :6] /= spec.scale[:6]
    expected[:, 6] = 2.0 * (
        (expected[:, 6] - spec.gripper_min_m)
        / (spec.gripper_max_m - spec.gripper_min_m)
    ) - 1.0
    return np.clip(expected, -1.0, 1.0)


def _empty_accumulator() -> dict[str, list[np.ndarray]]:
    return {name: [] for name in (*NETWORK_FIELDS, *METADATA_FIELDS)}


def _concatenate(
    parts: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, values in parts.items():
        if not values:
            dtype, tail = (NETWORK_FIELDS | METADATA_FIELDS)[name]
            arrays[name] = np.empty((0, *tail), dtype=dtype or "U1")
        else:
            arrays[name] = np.concatenate(values, axis=0)
    return arrays


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _validate_written_npz(path: Path, expected_length: int) -> dict[str, Any]:
    specifications = NETWORK_FIELDS | METADATA_FIELDS
    with np.load(path, allow_pickle=False) as dataset:
        if set(dataset.files) != set(specifications):
            raise RuntimeError(f"written AWAC fields are incomplete in {path}")
        for name, (dtype, tail) in specifications.items():
            values = dataset[name]
            if values.shape != (expected_length, *tail):
                raise RuntimeError(
                    f"written {name} shape mismatch: {values.shape}"
                )
            if dtype is None:
                if values.dtype.kind != "U":
                    raise RuntimeError(f"written {name} is not a Unicode array")
            elif values.dtype != np.dtype(dtype):
                raise RuntimeError(
                    f"written {name} dtype mismatch: {values.dtype} != {np.dtype(dtype)}"
                )
        network_nonfinite = sum(
            int((~np.isfinite(dataset[name])).sum())
            for name in ("obs", "action", "reward", "next_obs")
        )
        action_in_bounds = bool(
            np.all((dataset["action"] >= -1.0) & (dataset["action"] <= 1.0))
        )
    if network_nonfinite or not action_in_bounds:
        raise RuntimeError(f"written AWAC network arrays failed validation in {path}")
    return {
        "fields_complete": True,
        "shapes_and_dtypes_valid": True,
        "network_nonfinite_value_count": network_nonfinite,
        "actions_in_bounds": action_in_bounds,
    }


def convert_formal_rule_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    verify_checksums: bool = True,
    excluded_categories: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Convert only manifest-listed episodes into filtered AWAC transition NPZs.

    The source HDF5 files are opened read-only. Filtering is mutually exclusive in
    this order: invalid expert, fallback, IK rejection, then unconfirmed action.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    if manifest.get("dataset_root_base") != "manifest_parent":
        raise ValueError("manifest dataset_root_base must be 'manifest_parent'")
    source_episodes = manifest["episodes"]
    if set(excluded_categories) - set(CATEGORIES):
        raise ValueError("excluded_categories contains an unsupported category")
    if set(item["split"] for item in source_episodes) - set(SPLITS):
        raise ValueError("formal manifest contains an unsupported split")
    if len({item["episode_id"] for item in source_episodes}) != len(source_episodes):
        raise ValueError("formal manifest contains duplicate episode IDs")
    if sum(int(item["transitions"]) for item in source_episodes) != int(
        manifest["transition_count"]
    ):
        raise ValueError("formal manifest transition count mismatch")
    episodes = [
        item for item in source_episodes
        if item["category"] not in set(excluded_categories)
    ]

    root = (manifest_path.parent / manifest["dataset_root"]).resolve()
    accumulators = {split: _empty_accumulator() for split in SPLITS}
    episode_counts = Counter(item["split"] for item in episodes)
    raw_by_split = Counter()
    kept_by_split = Counter()
    raw_by_category = Counter()
    kept_by_category = Counter()
    exclusion_counts = Counter()
    raw_flag_counts = Counter()
    nonfinite_values = Counter()
    nonfinite_rows = Counter()
    checked_paths: set[Path] = set()

    for item in episodes:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"manifest file is missing or escapes dataset root: {path}")
        if path in checked_paths:
            raise ValueError(f"manifest references a file more than once: {path}")
        checked_paths.add(path)
        if verify_checksums and sha256_file(path) != item["sha256"]:
            raise ValueError(f"HDF5 checksum mismatch: {path}")

        with h5py.File(path, "r") as episode:
            length = _validate_manifest_episode(episode, item, manifest)
            obs = np.asarray(episode["observations/policy_state_42"], np.float32)
            action = np.asarray(episode["actions/normalized"], np.float32)
            reward = np.asarray(episode["labels/reward"], np.float32)
            next_obs = np.asarray(
                episode["next_observations/policy_state_42"], np.float32
            )
            expert_nominal = np.asarray(
                episode["actions/expert_nominal"], np.float32
            )
            clipped = np.asarray(
                episode["actions/command_after_clipping"], np.float64
            )
            executed = np.asarray(
                episode["actions/executed_joint_target"], np.float64
            )
            mujoco_ctrl = np.asarray(episode["actions/mujoco_ctrl"], np.float64)
            expert_valid = np.asarray(episode["actions/expert_valid"], bool)
            status = np.asarray(episode["actions/status"], np.uint8)
            rejection_reason = _decode_strings(
                np.asarray(episode["actions/rejection_reason"])
            )

            finite_fields = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "expert_nominal": expert_nominal,
                "command_after_clipping": clipped,
                "executed_joint_target": executed,
                "mujoco_ctrl": mujoco_ctrl,
                "perturbation_magnitude": np.asarray(
                    episode["perturbations/magnitude"], np.float64
                ),
            }
            finite_by_field: dict[str, np.ndarray] = {}
            for name, values in finite_fields.items():
                finite = np.isfinite(values)
                row_finite = finite if values.ndim == 1 else finite.all(
                    axis=tuple(range(1, values.ndim))
                )
                finite_by_field[name] = row_finite
                nonfinite_values[name] += int((~finite).sum())
                nonfinite_rows[name] += int((~row_finite).sum())

            status_binary = np.all((status == 0) | (status == 1), axis=1)
            status_consistent = status[:, 0] == status[:, 3]
            fallback = status[:, 2].astype(bool)
            rejected = ~status[:, 0].astype(bool) | (rejection_reason != "")
            bounds_valid = np.all((action >= -1.0) & (action <= 1.0), axis=1)
            action_matches = (
                np.isfinite(clipped).all(axis=1)
                & np.isfinite(action).all(axis=1)
                & np.all(
                    np.isclose(
                        action.astype(np.float64),
                        _expected_normalized_action(clipped),
                        rtol=1e-6,
                        atol=1e-6,
                    ),
                    axis=1,
                )
            )
            all_finite = np.logical_and.reduce(list(finite_by_field.values()))
            confirmed = (
                status_binary
                & status_consistent
                & status[:, 0].astype(bool)
                & ~fallback
                & (rejection_reason == "")
                & bounds_valid
                & action_matches
                & all_finite
            )

            invalid_expert_mask = ~expert_valid
            fallback_mask = ~invalid_expert_mask & fallback
            ik_rejected_mask = (
                ~invalid_expert_mask & ~fallback & rejected
            )
            unconfirmed_mask = (
                ~invalid_expert_mask & ~fallback & ~rejected & ~confirmed
            )
            keep = ~(
                invalid_expert_mask
                | fallback_mask
                | ik_rejected_mask
                | unconfirmed_mask
            )

            exclusion_counts["invalid_expert"] += int(invalid_expert_mask.sum())
            exclusion_counts["fallback"] += int(fallback_mask.sum())
            exclusion_counts["ik_rejection"] += int(ik_rejected_mask.sum())
            exclusion_counts["unconfirmed_action_execution"] += int(
                unconfirmed_mask.sum()
            )
            raw_flag_counts["invalid_expert"] += int((~expert_valid).sum())
            raw_flag_counts["fallback"] += int(fallback.sum())
            raw_flag_counts["ik_rejection"] += int(rejected.sum())
            raw_flag_counts["action_clipped"] += int(status[:, 1].sum())

            split = item["split"]
            category = item["category"]
            raw_by_split[split] += length
            kept_by_split[split] += int(keep.sum())
            raw_by_category[category] += length
            kept_by_category[category] += int(keep.sum())

            metadata = {
                "episode_id": np.full(length, item["episode_id"]),
                "step_index": np.asarray(episode["identity/step_index"], np.int64),
                "expert_stage": np.asarray(
                    episode["labels/expert_stage"], np.uint8
                ),
                "stage": np.asarray(episode["labels/stage"], np.uint8),
                "task_success": np.asarray(
                    episode["labels/task_success"], bool
                ),
                "termination_reason": _decode_strings(
                    np.asarray(episode["labels/termination_reason"])
                ),
                "events": np.asarray(episode["labels/events"], np.uint32),
                "task_milestones": np.asarray(
                    episode["labels/task_milestones"], np.uint8
                ),
                "perturbation_active": np.asarray(
                    episode["perturbations/active"], bool
                ),
                "perturbation_magnitude": np.asarray(
                    episode["perturbations/magnitude"], np.float32
                ),
                "expert_valid": expert_valid,
                "status": status,
                "rejection_reason": rejection_reason,
                "expert_nominal": expert_nominal,
                "category": np.full(length, category),
            }
            network = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "terminated": np.asarray(episode["labels/terminated"], bool),
                "truncated": np.asarray(episode["labels/truncated"], bool),
            }
            for name, values in {**network, **metadata}.items():
                accumulators[split][name].append(values[keep])

    arrays_by_split = {
        split: _concatenate(accumulators[split]) for split in SPLITS
    }
    final_count = sum(len(arrays["reward"]) for arrays in arrays_by_split.values())
    raw_count = sum(int(item["transitions"]) for item in episodes)
    if final_count + sum(exclusion_counts.values()) != raw_count:
        raise RuntimeError("exclusive filtering counts do not reconcile")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, dict[str, Any]] = {}
    output_validation: dict[str, dict[str, Any]] = {}
    final_action_min = float("inf")
    final_action_max = float("-inf")
    final_nonfinite = 0
    dimensions_valid = True
    for split, arrays in arrays_by_split.items():
        if arrays["action"].size:
            final_action_min = min(final_action_min, float(arrays["action"].min()))
            final_action_max = max(final_action_max, float(arrays["action"].max()))
        for name in NETWORK_FIELDS:
            values = arrays[name]
            if np.issubdtype(values.dtype, np.floating):
                final_nonfinite += int((~np.isfinite(values)).sum())
        dimensions_valid &= arrays["obs"].shape[1:] == (42,)
        dimensions_valid &= arrays["next_obs"].shape[1:] == (42,)
        dimensions_valid &= arrays["action"].shape[1:] == (7,)
        path = output_dir / f"{split}.npz"
        _atomic_npz(path, arrays)
        output_validation[split] = _validate_written_npz(path, len(arrays["reward"]))
        output_files[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "transitions": len(arrays["reward"]),
        }

    action_in_bounds = bool(final_action_min >= -1.0 and final_action_max <= 1.0)
    report: dict[str, Any] = {
        "dataset_version": AWAC_DATASET_VERSION,
        "source_manifest": str(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "checksums_verified": verify_checksums,
        "source_manifest_episode_count": len(source_episodes),
        "manifest_episode_count": len(episodes),
        "excluded_episode_categories": list(excluded_categories),
        "excluded_episode_count": len(source_episodes) - len(episodes),
        "episodes_by_split": {
            split: int(episode_counts[split]) for split in SPLITS
        },
        "raw_transition_count": raw_count,
        "raw_transitions_by_split": {
            split: int(raw_by_split[split]) for split in SPLITS
        },
        "kept_transition_count": final_count,
        "kept_transitions_by_split": {
            split: int(kept_by_split[split]) for split in SPLITS
        },
        "excluded_transition_count": raw_count - final_count,
        "exclusions_exclusive": {
            name: int(exclusion_counts[name])
            for name in (
                "ik_rejection",
                "fallback",
                "invalid_expert",
                "unconfirmed_action_execution",
            )
        },
        "source_flags_nonexclusive": {
            name: int(raw_flag_counts[name])
            for name in (
                "ik_rejection",
                "fallback",
                "invalid_expert",
                "action_clipped",
            )
        },
        "nonfinite": {
            "source_value_count_by_field": dict(sorted(nonfinite_values.items())),
            "source_row_count_by_field": dict(sorted(nonfinite_rows.items())),
            "final_network_value_count": final_nonfinite,
        },
        "action_validation": {
            "all_within_closed_interval_minus1_plus1": action_in_bounds,
            "minimum": final_action_min if final_count else None,
            "maximum": final_action_max if final_count else None,
            "action_shape": [7],
        },
        "observation_validation": {
            "all_obs_and_next_obs_are_42d": bool(dimensions_valid),
            "obs_shape": [42],
            "next_obs_shape": [42],
        },
        "transitions_by_category": {
            category: {
                "raw": int(raw_by_category[category]),
                "kept": int(kept_by_category[category]),
            }
            for category in CATEGORIES
        },
        "network_fields": list(NETWORK_FIELDS),
        "metadata_fields": list(METADATA_FIELDS),
        "action_source": "actions/normalized",
        "reward_source": "labels/reward (copied unchanged)",
        "filter_order": [
            "invalid_expert",
            "fallback",
            "ik_rejection",
            "unconfirmed_action_execution",
        ],
        "status_layout_schema_2": [
            "command_accepted",
            "action_clipped",
            "fallback_used",
            "command_accepted_duplicate",
        ],
        "files": output_files,
        "output_validation": output_validation,
    }
    _atomic_json(output_dir / "report.json", report)
    return report
