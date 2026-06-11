"""Compatibility re-export for Ghost Replay helpers."""

from ..training_components.ghost_replay import (
    GhostRecorder,
    GhostReplayer,
    inspect_ghost_fingerprint,
)

__all__ = ["GhostRecorder", "GhostReplayer", "inspect_ghost_fingerprint"]
