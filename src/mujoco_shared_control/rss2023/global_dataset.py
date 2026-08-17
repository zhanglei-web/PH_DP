"""Read the frozen Learned Expert dataset for the global RSS2023 policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mujoco_shared_control.rss2023.dataset import FeatureNormalizer


GLOBAL_OBSERVATION_DIM = 43
GLOBAL_ACTION_DIM = 7


@dataclass(frozen=True)
class GlobalDataSplit:
    episode_ids: tuple[str, ...]
    observation: NDArray[np.float32]
    action: NDArray[np.float32]

    def __len__(self) -> int:
        return int(self.observation.shape[0])


@dataclass(frozen=True)
class PreparedGlobalDataset:
    train: GlobalDataSplit
    validation: GlobalDataSplit
    test: GlobalDataSplit
    observation_normalizer: FeatureNormalizer
    action_normalizer: FeatureNormalizer
    dataset_root: Path
    split_manifest_path: Path

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter": "frozen_final_expert_npz_v1",
            "dataset_root": str(self.dataset_root),
            "split_manifest": str(self.split_manifest_path),
            "observation_field": "diffusion_observation_43",
            "action_field": "executed_action_7",
            "observation_dim": GLOBAL_OBSERVATION_DIM,
            "action_dim": GLOBAL_ACTION_DIM,
            "splits": {
                "train": {"episodes": len(self.train.episode_ids), "transitions": len(self.train)},
                "validation": {"episodes": len(self.validation.episode_ids), "transitions": len(self.validation)},
                "test": {"episodes": len(self.test.episode_ids), "transitions": len(self.test)},
            },
        }


def _load_split(
    dataset_root: Path, episode_ids: tuple[str, ...], path_by_id: dict[str, Path],
) -> GlobalDataSplit:
    observations: list[NDArray[np.float32]] = []
    actions: list[NDArray[np.float32]] = []
    for episode_id in episode_ids:
        path = path_by_id.get(episode_id, dataset_root / "success" / f"{episode_id}.npz")
        if not path.is_file():
            raise FileNotFoundError(f"split episode is missing: {path}")
        with np.load(path, allow_pickle=False) as episode:
            if "diffusion_observation_43" not in episode or "executed_action_7" not in episode:
                raise ValueError(f"{path}: required Global Diffusion fields are missing")
            observation = np.asarray(episode["diffusion_observation_43"], dtype=np.float32)
            action = np.asarray(episode["executed_action_7"], dtype=np.float32)
        if observation.ndim != 2 or observation.shape[1] != GLOBAL_OBSERVATION_DIM:
            raise ValueError(f"{path}: invalid observation shape {observation.shape}")
        if action.shape != (observation.shape[0], GLOBAL_ACTION_DIM):
            raise ValueError(f"{path}: invalid action shape {action.shape}")
        if not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: NaN/Inf in Global Diffusion fields")
        observations.append(observation)
        actions.append(action)
    return GlobalDataSplit(
        episode_ids=episode_ids,
        observation=np.ascontiguousarray(np.concatenate(observations, axis=0)),
        action=np.ascontiguousarray(np.concatenate(actions, axis=0)),
    )


def prepare_global_dataset(dataset_dir: str | Path) -> PreparedGlobalDataset:
    """Load exactly the episode IDs frozen in ``split_manifest.json``."""

    root = Path(dataset_dir).expanduser().resolve()
    manifest_path = root / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing frozen split manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("split_unit") != "episode":
        raise ValueError("Global Diffusion requires an episode-level frozen split")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("split_manifest.json must contain train/validation/test")
    episode_ids = {name: tuple(str(value) for value in splits[name]) for name in splits}
    flattened = sum((episode_ids[name] for name in ("train", "validation", "test")), ())
    if not flattened or len(set(flattened)) != len(flattened):
        raise ValueError("frozen split must contain unique successful episodes")

    path_by_id: dict[str, Path] = {}
    episode_manifest = root / "episode_manifest.json"
    if episode_manifest.is_file():
        records = json.loads(episode_manifest.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = records.get("episodes", [])
        for record in records:
            if "episode_id" in record and "path" in record:
                path_by_id[str(record["episode_id"])] = Path(record["path"]).expanduser().resolve()
    missing_references = set(flattened) - set(path_by_id)
    missing_local = {
        episode_id for episode_id in missing_references
        if not (root / "success" / f"{episode_id}.npz").is_file()
    }
    if missing_local:
        raise FileNotFoundError(f"manifest paths missing for {len(missing_local)} split episodes")

    train = _load_split(root, episode_ids["train"], path_by_id)
    validation = _load_split(root, episode_ids["validation"], path_by_id)
    test = _load_split(root, episode_ids["test"], path_by_id)
    return PreparedGlobalDataset(
        train=train,
        validation=validation,
        test=test,
        observation_normalizer=FeatureNormalizer.fit(train.observation),
        action_normalizer=FeatureNormalizer.fit(train.action),
        dataset_root=root,
        split_manifest_path=manifest_path,
    )
