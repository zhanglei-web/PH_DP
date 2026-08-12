"""High-speed state-only expert collection."""

from mujoco_shared_control.collection.automatic import (
    AutomaticCollector,
    CollectionConfig,
    EpisodeResult,
)
from mujoco_shared_control.collection.recording import AutoEpisodeRecorder

__all__ = ["AutomaticCollector", "AutoEpisodeRecorder", "CollectionConfig", "EpisodeResult"]
