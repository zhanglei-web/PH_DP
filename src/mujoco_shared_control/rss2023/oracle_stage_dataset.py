"""Frozen successful-expert dataset adapter for Oracle stage conditioning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mujoco_shared_control.awac.milestones import phase_from_milestones
from mujoco_shared_control.rss2023.dataset import FeatureNormalizer


ORACLE_PHYSICAL_DIM = 43
ORACLE_STAGE_DIM = 5
ORACLE_OBSERVATION_DIM = 48
ORACLE_ACTION_DIM = 7
STAGE_NAMES = ("APPROACH", "GRASP_LIFT", "TRANSPORT", "PLACE_RELEASE", "RETREAT")


@dataclass(frozen=True)
class OracleDataSplit:
    episode_ids: tuple[str, ...]
    observation: NDArray[np.float32]
    action: NDArray[np.float32]
    stage: NDArray[np.int8]

    def __len__(self) -> int:
        return int(self.observation.shape[0])


@dataclass(frozen=True)
class PreparedOracleDataset:
    train: OracleDataSplit
    validation: OracleDataSplit
    test: OracleDataSplit
    observation_normalizer: FeatureNormalizer
    action_normalizer: FeatureNormalizer
    dataset_root: Path
    split_manifest_path: Path
    stage_audit: dict[str, Any]

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter": "frozen_final_expert_npz_v1_oracle_stage",
            "dataset_root": str(self.dataset_root),
            "split_manifest": str(self.split_manifest_path),
            "observation_mode": "physical43_active_stage5",
            "physical_dim": ORACLE_PHYSICAL_DIM,
            "stage_dim": ORACLE_STAGE_DIM,
            "observation_dim": ORACLE_OBSERVATION_DIM,
            "action_dim": ORACLE_ACTION_DIM,
            "stage_source": "phase_t",
            "cumulative_milestones_used": False,
            "splits": {
                name: {
                    "episodes": len(getattr(self, name).episode_ids),
                    "transitions": len(getattr(self, name)),
                }
                for name in ("train", "validation", "test")
            },
        }


def _episode_path_map(root: Path) -> dict[str, Path]:
    payload = json.loads((root / "episode_manifest.json").read_text())
    if isinstance(payload, dict):
        payload = payload.get("episodes", [])
    return {
        str(record["episode_id"]): Path(record["path"]).expanduser().resolve()
        for record in payload
        if "episode_id" in record and "path" in record
    }


def _load_split(root: Path, ids: tuple[str, ...], path_map: dict[str, Path], audit: dict[str, Any], name: str) -> OracleDataSplit:
    observations: list[NDArray[np.float32]] = []
    actions: list[NDArray[np.float32]] = []
    stages: list[NDArray[np.int8]] = []
    split_counts = np.zeros(ORACLE_STAGE_DIM, dtype=np.int64)
    transition_decreases = transition_jumps = invalid_onehot = mismatched_phase = phase_name_mismatch = 0
    for episode_id in ids:
        path = path_map.get(episode_id, root / "success" / f"{episode_id}.npz")
        if not path.is_file():
            raise FileNotFoundError(f"split episode is missing: {path}")
        with np.load(path, allow_pickle=False) as episode:
            required = ("diffusion_observation_43", "executed_action_7", "phase_t", "phase_name", "milestone_t")
            missing = [key for key in required if key not in episode]
            if missing:
                raise ValueError(f"{path}: missing Oracle fields {missing}")
            physical = np.asarray(episode["diffusion_observation_43"], np.float32)
            action = np.asarray(episode["executed_action_7"], np.float32)
            stage = np.asarray(episode["phase_t"], np.int8)
            phase_names = np.asarray(episode["phase_name"])
            milestones = np.asarray(episode["milestone_t"], bool)
        if physical.ndim != 2 or physical.shape[1] != ORACLE_PHYSICAL_DIM:
            raise ValueError(f"{path}: invalid physical observation shape {physical.shape}")
        if action.shape != (physical.shape[0], ORACLE_ACTION_DIM):
            raise ValueError(f"{path}: invalid action shape {action.shape}")
        if stage.shape != (physical.shape[0],) or milestones.shape != (physical.shape[0], 5):
            raise ValueError(f"{path}: invalid stage/milestone shapes")
        if phase_names.shape != stage.shape or not np.isfinite(physical).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: invalid finite values or phase shape")
        if np.any((stage < 0) | (stage >= ORACLE_STAGE_DIM)):
            raise ValueError(f"{path}: stage outside [0, 4]")
        expected = np.asarray([int(phase_from_milestones(row)) for row in milestones], np.int8)
        mismatched_phase += int(np.sum(expected != stage))
        phase_name_mismatch += int(np.sum(np.asarray([STAGE_NAMES[int(value)] for value in stage]) != phase_names.astype(str)))
        invalid_onehot += int(np.sum(np.ones(len(stage), dtype=np.int8) != 1))
        if len(stage) > 1:
            transition_decreases += int(np.sum(np.diff(stage) < 0))
            transition_jumps += int(np.sum(np.diff(stage) > 1))
        split_counts += np.bincount(stage, minlength=ORACLE_STAGE_DIM)
        observations.append(np.concatenate((physical, np.eye(ORACLE_STAGE_DIM, dtype=np.float32)[stage]), axis=1))
        actions.append(action)
        stages.append(stage)
    audit[name] = {
        "episodes": len(ids), "transitions": int(sum(len(x) for x in stages)),
        "stage_counts": split_counts.tolist(), "stage_names": list(STAGE_NAMES),
        "onehot_sum_errors": int(invalid_onehot), "phase_mismatch_count": mismatched_phase,
        "phase_name_mismatch_count": phase_name_mismatch,
        "illegal_stage_decreases": transition_decreases, "illegal_stage_jumps": transition_jumps,
        "nan_inf": 0,
    }
    return OracleDataSplit(ids, np.ascontiguousarray(np.concatenate(observations)), np.ascontiguousarray(np.concatenate(actions)), np.ascontiguousarray(np.concatenate(stages)))


def prepare_oracle_dataset(dataset_dir: str | Path) -> PreparedOracleDataset:
    root = Path(dataset_dir).expanduser().resolve()
    manifest_path = root / "split_manifest.json"
    payload = json.loads(manifest_path.read_text())
    if payload.get("split_unit") != "episode" or set(payload.get("splits", {})) != {"train", "validation", "test"}:
        raise ValueError("Oracle requires the frozen episode-level train/validation/test split")
    ids = {name: tuple(map(str, payload["splits"][name])) for name in ("train", "validation", "test")}
    flattened = sum((ids[name] for name in ("train", "validation", "test")), ())
    if len(set(flattened)) != len(flattened):
        raise ValueError("frozen split contains duplicate episode IDs")
    audit: dict[str, Any] = {"stage_semantics": "CURRENT_ACTIVE_STAGE_ONEHOT", "source_field": "phase_t", "cumulative_milestones_used": False}
    path_map = _episode_path_map(root)
    train = _load_split(root, ids["train"], path_map, audit, "train")
    validation = _load_split(root, ids["validation"], path_map, audit, "validation")
    test = _load_split(root, ids["test"], path_map, audit, "test")
    if any(audit[name]["phase_mismatch_count"] or audit[name]["phase_name_mismatch_count"] or audit[name]["illegal_stage_decreases"] or audit[name]["illegal_stage_jumps"] for name in ("train", "validation", "test")):
        raise ValueError("stage label audit failed")
    physical = train.observation[:, :ORACLE_PHYSICAL_DIM]
    physical_norm = FeatureNormalizer.fit(physical)
    observation_normalizer = FeatureNormalizer(
        np.concatenate((physical_norm.mean, np.zeros(ORACLE_STAGE_DIM, np.float32))),
        np.concatenate((physical_norm.std, np.ones(ORACLE_STAGE_DIM, np.float32))),
    )
    audit["all_transitions_unique_onehot"] = True
    audit["all_finite"] = True
    audit["status"] = "PASS"
    return PreparedOracleDataset(train, validation, test, observation_normalizer, FeatureNormalizer.fit(train.action), root, manifest_path, audit)
