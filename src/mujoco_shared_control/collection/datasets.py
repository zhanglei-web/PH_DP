"""Lazy NumPy datasets for behavior-cloning and value-learning consumers."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class EpisodeEntry:
    path: Path
    length: int
    outcome: str
    variant: str


def discover_episodes(roots: str | Path | Iterable[str | Path]) -> list[EpisodeEntry]:
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
    entries: list[EpisodeEntry] = []
    for root in roots:
        for path in sorted(Path(root).expanduser().rglob("*.h5")):
            if ".inprogress." in path.name or path.parent.name == "invalid":
                continue
            with h5py.File(path, "r") as episode:
                if not str(episode.attrs.get("schema_version", "")).startswith("2."):
                    continue
                entries.append(EpisodeEntry(
                    path.resolve(), len(episode["labels/reward"]),
                    str(episode.attrs.get("outcome", "unknown")),
                    str(episode.attrs.get("collection_variant", "unknown")),
                ))
    return entries


class _TransitionDataset:
    def __init__(self, entries: list[EpisodeEntry]) -> None:
        self.entries = entries
        self.ends = np.cumsum([entry.length for entry in entries], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.ends[-1]) if len(self.ends) else 0

    def _location(self, index: int) -> tuple[EpisodeEntry, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.ends, index)
        start = 0 if episode_index == 0 else int(self.ends[episode_index - 1])
        return self.entries[episode_index], index - start


class ActorDataset(_TransitionDataset):
    """By default use only clean, successful Rule Expert demonstrations."""

    def __init__(self, roots: str | Path | Iterable[str | Path], *,
                 include_recovered: bool = False) -> None:
        entries = discover_episodes(roots)
        allowed = {"success", "recovered"} if include_recovered else {"success"}
        super().__init__([entry for entry in entries
                          if entry.outcome in allowed and
                          (include_recovered or entry.variant == "nominal")])

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str | int]:
        entry, row = self._location(index)
        with h5py.File(entry.path, "r") as episode:
            return {
                "state": episode["observations/state_26"][row],
                "policy_state": episode["observations/policy_state_42"][row],
                "action": episode["actions/expert_nominal"][row],
                "normalized_action": episode["actions/normalized"][row],
                "stage": int(episode["labels/stage"][row]),
                "expert_stage": int(episode["labels/expert_stage"][row]),
                "entered_settling": bool(episode["labels/entered_settling"][row])
                if "labels/entered_settling" in episode else False,
                "episode_id": str(episode.attrs["episode_id"]),
                "step_index": row,
            }


class CriticDataset(_TransitionDataset):
    """Use success, recovery and failure transitions with explicit terminal labels."""

    def __init__(self, roots: str | Path | Iterable[str | Path]) -> None:
        super().__init__(discover_episodes(roots))

    def __getitem__(self, index: int) -> dict[str, np.ndarray | float | bool | str | int]:
        entry, row = self._location(index)
        with h5py.File(entry.path, "r") as episode:
            terminated = bool(episode["labels/terminated"][row])
            truncated = bool(episode["labels/truncated"][row])
            return {
                "state": episode["observations/state_26"][row],
                "action": episode["actions/policy_command"][row],
                "reward": float(episode["labels/reward"][row]),
                "next_state": episode["next_observations/state_26"][row],
                "terminated": terminated,
                "truncated": truncated,
                "done": terminated or truncated,
                "outcome": entry.outcome,
                "entered_settling": bool(episode["labels/entered_settling"][row])
                if "labels/entered_settling" in episode else False,
                "expert_failed_step": int(episode["labels/expert_failed_step"][row])
                if "labels/expert_failed_step" in episode else -1,
                "episode_id": str(episode.attrs["episode_id"]),
                "step_index": row,
            }
