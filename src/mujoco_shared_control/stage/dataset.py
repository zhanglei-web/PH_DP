"""Episode-split loader with dynamic length-20 TCN windows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


HISTORY = 20
FEATURE_DIM = 19
BINARY_INDICES = (17,)


@dataclass(frozen=True)
class EpisodeSequence:
    episode_id: str
    trajectory_type: str
    features: np.ndarray
    labels: np.ndarray
    events: np.ndarray


@dataclass(frozen=True)
class StageNormalization:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.mean) / self.std).astype(np.float32)


def load_split(root: str | Path, split: str) -> list[EpisodeSequence]:
    root = Path(root).resolve()
    manifest = json.loads((root / "split_manifest.json").read_text())
    ids = manifest["splits"][split]; paths = manifest["episode_paths"]
    episodes = []
    for episode_id in ids:
        path = Path(paths[episode_id])
        with h5py.File(path, "r") as file:
            features = file["stage_features"][:].astype(np.float32)
            labels = file["active_phase"][:].astype(np.int64)
            events = file["event"][:].astype(np.int64)
            trajectory_type = str(file.attrs["trajectory_type"])
        if features.shape != (len(labels), FEATURE_DIM) or len(labels) != len(events):
            raise ValueError(f"invalid Stage Dataset episode: {path}")
        episodes.append(EpisodeSequence(episode_id, trajectory_type, features, labels, events))
    return episodes


def fit_normalization(episodes: list[EpisodeSequence]) -> StageNormalization:
    values = np.concatenate([episode.features for episode in episodes], axis=0).astype(np.float64)
    mean = values.mean(0); std = np.maximum(values.std(0), 1e-6)
    mean[list(BINARY_INDICES)] = 0.0; std[list(BINARY_INDICES)] = 1.0
    return StageNormalization(mean.astype(np.float32), std.astype(np.float32))


class StageWindowDataset(Dataset):
    def __init__(self, episodes: list[EpisodeSequence], normalization: StageNormalization) -> None:
        windows, labels, episode_indices, time_indices, special = [], [], [], [], []
        for episode_index, episode in enumerate(episodes):
            feature = normalization.apply(episode.features)
            phase_changes = np.flatnonzero(np.r_[False, episode.labels[1:] != episode.labels[:-1]])
            event_steps = np.flatnonzero(episode.events != 0)
            centers = np.unique(np.r_[phase_changes, event_steps])
            for time_index in range(HISTORY - 1, len(episode.labels)):
                windows.append(feature[time_index - HISTORY + 1:time_index + 1])
                labels.append(episode.labels[time_index]); episode_indices.append(episode_index); time_indices.append(time_index)
                special.append(bool(len(centers) and np.min(np.abs(centers - time_index)) <= 10))
        self.windows = np.ascontiguousarray(np.asarray(windows, np.float32))
        self.labels = np.asarray(labels, np.int64)
        self.episode_indices = np.asarray(episode_indices, np.int32)
        self.time_indices = np.asarray(time_indices, np.int32)
        self.special = np.asarray(special, bool)
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.windows[index]), torch.tensor(self.labels[index]), index

    def sampler_weights(self) -> torch.Tensor:
        special_count = int(self.special.sum()); ordinary_count = len(self) - special_count
        if special_count == 0 or ordinary_count == 0:
            raise ValueError("training split lacks ordinary or transition/recovery windows")
        weights = np.where(self.special, 0.30 / special_count, 0.70 / ordinary_count)
        return torch.from_numpy(weights.astype(np.float64))
