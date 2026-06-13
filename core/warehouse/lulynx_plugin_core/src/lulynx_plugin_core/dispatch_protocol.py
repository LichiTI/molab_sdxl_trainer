"""Dispatch protocol for the lulynx training event system.

Provides training-type normalization, loss-mutation application,
training-snapshot construction, and event-field introspection.

This module is a Warehouse design: it does not import, copy, or
paraphrase any upstream or fork code.  All logic is original.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    return str(value) if value is not None else default


# ---------------------------------------------------------------------------
# Training-type normalization
# ---------------------------------------------------------------------------

class TrainingTypeNormalizer:
    """Resolve ambiguous training-type names to canonical forms.

    Example::

        norm = TrainingTypeNormalizer()
        assert norm.normalize("anima") == "anima-lora"
    """

    _DEFAULT_ALIASES: dict[str, str] = {
        "anima": "anima-lora",
        "anima-sd": "anima-lora",
        "anima-sdxl": "anima-lora",
        "newbie": "newbie-lora",
        "newbie-sd": "newbie-lora",
        "newbie-sdxl": "newbie-lora",
    }

    def __init__(self, extra_aliases: Mapping[str, str] | None = None) -> None:
        self._aliases: dict[str, str] = dict(self._DEFAULT_ALIASES)
        if extra_aliases:
            self._aliases.update(extra_aliases)

    def normalize(self, training_type: str | None) -> str:
        key = str(training_type or "").strip().lower()
        return self._aliases.get(key, key)

    def resolve(self, training_type: str | None, architecture: str | None = None) -> str:
        return self.normalize(training_type)


# ---------------------------------------------------------------------------
# Lightweight descriptors (injected, no external dependency)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteInfo:
    """Injectable route descriptor."""
    route_id: str = ""
    family: str = ""
    label: str = ""
    kind: str = ""
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingSnapshotDescriptor:
    """Declarative inputs for building one training-event payload."""
    architecture: str = ""
    training_type: str = ""
    total_steps: int = 0
    current_step: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Learning-rate extraction
# ---------------------------------------------------------------------------

def _extract_optimizer_lrs(state: Mapping[str, Any]) -> list[float]:
    lrs: list[float] = []
    for key in ("lr", "learning_rate", "base_lr"):
        if key in state:
            lrs.append(_safe_float(state[key]))
    for group in state.get("param_groups", []):
        if isinstance(group, Mapping):
            lrs.append(_safe_float(group.get("lr", 0.0)))
    return lrs


def _extract_scheduler_lrs(state: Mapping[str, Any]) -> list[float]:
    lrs: list[float] = []
    for key in ("last_lr", "learning_rate", "lr"):
        val = state.get(key)
        if isinstance(val, (list, tuple)):
            lrs.extend(_safe_float(v) for v in val)
        elif val is not None:
            lrs.append(_safe_float(val))
    return lrs


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------

def build_training_snapshot(
    descriptor: TrainingSnapshotDescriptor,
    *,
    route: RouteInfo | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
    batch_info: Mapping[str, Any] | None = None,
    loss: float | None = None,
) -> dict[str, Any]:
    """Build a training-event payload from declarative inputs."""
    normalizer = TrainingTypeNormalizer()
    resolved_type = normalizer.normalize(descriptor.training_type)
    opt_lrs = _extract_optimizer_lrs(optimizer_state) if optimizer_state else []
    sched_lrs = _extract_scheduler_lrs(scheduler_state) if scheduler_state else []
    current_lrs = sched_lrs or opt_lrs

    payload: dict[str, Any] = {
        "architecture": descriptor.architecture,
        "training_type": resolved_type,
        "total_steps": descriptor.total_steps,
        "current_step": descriptor.current_step,
        "learning_rates": current_lrs,
    }
    if batch_info:
        payload["batch_size"] = batch_info.get("batch_size")
        payload["grad_accumulation_steps"] = batch_info.get("grad_accumulation_steps")
    if route:
        payload["route_id"] = route.route_id
        payload["route_family"] = route.family
        payload["route_label"] = route.label
    if loss is not None:
        payload["loss"] = loss
    if descriptor.extra:
        payload.update(descriptor.extra)
    return payload


# ---------------------------------------------------------------------------
# Loss mutation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossMutationOutcome:
    """Result of applying a loss mutation."""
    original_loss: float
    scale: float
    bias: float
    final_loss: float
    applied: bool


def apply_loss_mutation(
    loss: float,
    scale: float = 1.0,
    bias: float = 0.0,
) -> LossMutationOutcome:
    """Apply an affine host transform: ``final = loss * scale + bias``."""
    import math
    if math.isnan(scale) or math.isinf(scale):
        return LossMutationOutcome(loss, scale, bias, loss, False)
    if scale == 1.0 and bias == 0.0:
        return LossMutationOutcome(loss, scale, bias, loss, False)
    final = loss * scale + bias
    return LossMutationOutcome(loss, scale, bias, final, True)


# ---------------------------------------------------------------------------
# Event-field map
# ---------------------------------------------------------------------------

_EVENT_FIELD_MAP: dict[str, list[str]] = {
    "before_forward": [
        "architecture", "training_type", "total_steps", "current_step",
        "learning_rates", "batch_size", "grad_accumulation_steps",
    ],
    "after_forward": [
        "architecture", "training_type", "total_steps", "current_step",
        "learning_rates", "loss",
    ],
    "modify_loss": [
        "architecture", "training_type", "current_step", "loss",
    ],
    "after_loss": [
        "architecture", "training_type", "current_step", "loss",
        "learning_rates",
    ],
    "after_backward": [
        "architecture", "training_type", "current_step", "loss",
        "learning_rates", "grad_norm",
    ],
    "before_optimizer_step": [
        "architecture", "training_type", "current_step",
        "learning_rates", "grad_norm",
    ],
    "after_optimizer_step": [
        "architecture", "training_type", "current_step",
        "learning_rates",
    ],
}


def list_training_events() -> list[str]:
    """Return all known training event names."""
    return list(_EVENT_FIELD_MAP.keys())


def describe_training_event(event: str) -> dict[str, Any] | None:
    """Return the field list for a training event, or ``None``."""
    fields = _EVENT_FIELD_MAP.get(event)
    if fields is None:
        return None
    return {"event": event, "fields": list(fields)}

