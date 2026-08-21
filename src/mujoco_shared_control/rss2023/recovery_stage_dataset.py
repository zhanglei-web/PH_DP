"""Frozen clean Recovery-Stage-DP dataset adapter.

This adapter deliberately reads only the clean ``episodes`` tree. Raw
rollouts are never a training source because they contain injected actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from mujoco_shared_control.rss2023.dataset import FeatureNormalizer

PHYSICAL_DIM = 43
STAGE_DIM = 5
ACTION_DIM = 7
STAGE_NAMES = ("APPROACH", "GRASP_LIFT", "TRANSPORT", "PLACE_RELEASE", "RETREAT")
KINDS = ("NORMAL_SUCCESS", "GRASP_RECOVERY_SUCCESS", "TRANSPORT_RECOVERY_SUCCESS", "PLACE_RECOVERY_SUCCESS")


@dataclass(frozen=True)
class RecoverySplit:
    episode_ids: tuple[str, ...]
    physical: np.ndarray
    stage_onehot: np.ndarray
    action: np.ndarray
    kind: np.ndarray

    def __len__(self) -> int:
        return int(self.physical.shape[0])

    @property
    def observation_concat(self) -> np.ndarray:
        return np.concatenate((self.physical, self.stage_onehot), axis=1)


@dataclass(frozen=True)
class PreparedRecoveryDataset:
    train: RecoverySplit
    validation: RecoverySplit
    test: RecoverySplit
    physical_normalizer: FeatureNormalizer
    action_normalizer: FeatureNormalizer
    root: Path
    split_manifest: Path
    audit: dict[str, Any]

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter": "recovery_stage_dp_v1_clean_h5",
            "dataset_root": str(self.root),
            "split_manifest": str(self.split_manifest),
            "physical_dim": PHYSICAL_DIM,
            "stage_dim": STAGE_DIM,
            "action_dim": ACTION_DIM,
            "stage_names": list(STAGE_NAMES),
            "source": "episodes_only; executable expert action; injection excluded",
            "window_length": 1,
            "splits": {n: {"episodes": len(getattr(self, n).episode_ids), "transitions": len(getattr(self, n))} for n in ("train", "validation", "test")},
        }


def _path_map(root: Path) -> dict[str, Path]:
    split_payload = json.loads((root / "split_manifest.json").read_text())
    return {str(k): Path(v).expanduser().resolve() for k, v in split_payload.get("episode_paths", {}).items()}


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _load(root: Path, ids: tuple[str, ...], paths: dict[str, Path], split_name: str) -> tuple[RecoverySplit, dict[str, Any]]:
    physical, stage, actions, kinds, episode_ids = [], [], [], [], []
    counts = {k: {"episodes": 0, "transitions": 0} for k in KINDS}
    for episode_id in ids:
        path = paths.get(episode_id)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"missing split episode {episode_id}: {path}")
        with h5py.File(path, "r") as f:
            required = ("full_physical_state", "stage_onehot", "active_phase", "executed_action", "action_source", "injection_active", "timestep_dp")
            missing = [k for k in required if k not in f]
            if missing:
                raise ValueError(f"{path}: missing {missing}")
            x = np.asarray(f["full_physical_state"], np.float32)
            z = np.asarray(f["stage_onehot"], np.float32)
            phase = np.asarray(f["active_phase"], np.int64)
            a = np.asarray(f["executed_action"], np.float32)
            source = np.asarray(f["action_source"], np.int64)
            injected = np.asarray(f["injection_active"], bool)
            timestep = np.asarray(f["timestep_dp"], np.int64)
            kind = str(f.attrs["episode_type"])
            final_success = bool(f.attrs["final_success"])
        if kind not in KINDS or not final_success:
            raise ValueError(f"{path}: not a valid final-success kind")
        n = len(x)
        if x.shape != (n, PHYSICAL_DIM) or z.shape != (n, STAGE_DIM) or a.shape != (n, ACTION_DIM):
            raise ValueError(f"{path}: invalid shape physical={x.shape} stage={z.shape} action={a.shape}")
        if phase.shape != (n,) or source.shape != (n,) or injected.shape != (n,) or timestep.shape != (n,):
            raise ValueError(f"{path}: invalid metadata shape")
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(a)) or np.any(~np.isfinite(z)):
            raise ValueError(f"{path}: NaN/Inf")
        if np.any(source != 0) or np.any(injected) or np.any(timestep != np.arange(n)):
            raise ValueError(f"{path}: injection/non-expert action or discontinuous clean timestep")
        expected = np.eye(STAGE_DIM, dtype=np.float32)[phase]
        if np.any((phase < 0) | (phase >= STAGE_DIM)) or not np.array_equal(z, expected):
            raise ValueError(f"{path}: invalid stage one-hot")
        physical.append(x); stage.append(z); actions.append(a); kinds.append(np.full(n, KINDS.index(kind), np.int8)); episode_ids.append(episode_id)
        counts[kind]["episodes"] += 1; counts[kind]["transitions"] += n
    split = RecoverySplit(tuple(episode_ids), np.concatenate(physical), np.concatenate(stage), np.concatenate(actions), np.concatenate(kinds))
    return split, {"episodes": len(ids), "transitions": len(split), "by_type": counts, "episode_id_sha256": _hash_ids(list(ids))}


def prepare_recovery_dataset(dataset_dir: str | Path) -> PreparedRecoveryDataset:
    root = Path(dataset_dir).expanduser().resolve()
    manifest_path = root / "split_manifest.json"
    payload = json.loads(manifest_path.read_text())
    if payload.get("split_unit") != "episode" or set(payload.get("splits", {})) != {"train", "validation", "test"}:
        raise ValueError("recovery dataset must use frozen episode-level train/validation/test splits")
    ids = {n: tuple(map(str, payload["splits"][n])) for n in ("train", "validation", "test")}
    flattened = sum((list(ids[n]) for n in ids), [])
    if len(flattened) != len(set(flattened)):
        raise ValueError("episode leakage or duplicate IDs in frozen split")
    paths = _path_map(root)
    train, train_audit = _load(root, ids["train"], paths, "train")
    validation, val_audit = _load(root, ids["validation"], paths, "validation")
    test, test_audit = _load(root, ids["test"], paths, "test")
    physical_norm = FeatureNormalizer.fit(train.physical)
    action_norm = FeatureNormalizer.fit(train.action)
    transition_counts = {k: int(np.sum(np.concatenate((train.kind, validation.kind, test.kind)) == KINDS.index(k))) for k in KINDS}
    audit = {"status": "PASS", "physical_dim": PHYSICAL_DIM, "stage_dim": STAGE_DIM, "action_dim": ACTION_DIM, "train": train_audit, "validation": val_audit, "test": test_audit, "injection_transitions_used": 0, "non_expert_actions_used": 0, "transition_counts_by_type": transition_counts}
    return PreparedRecoveryDataset(train, validation, test, physical_norm, action_norm, root, manifest_path, audit)
