"""Training hook snapshot helpers for plugin runtime fast-path checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRAINING_HOOK_EVENTS = [
    "before_forward",
    "after_loss",
    "modify_loss",
    "after_backward",
    "before_optimizer_step",
    "after_optimizer_step",
    "preflight_advisory",
    "dataset_health_advisory",
    "caption_tag_advisory",
    "validation_prompt_advisory",
    "metrics_advisory",
    "report_advisory",
]


def build_training_hooks_snapshot(bus: Any, orchestrator: Any) -> dict[str, Any]:
    events_map = {event: bool(bus.has_handlers(event)) for event in TRAINING_HOOK_EVENTS}
    active_ids = [desc.plugin_id for desc in orchestrator.query_active()]
    return {"events": events_map, "active_plugin_ids": active_ids}


def write_training_hooks_snapshot(path: Path, bus: Any, orchestrator: Any) -> dict[str, Any]:
    snapshot = build_training_hooks_snapshot(bus, orchestrator)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot
