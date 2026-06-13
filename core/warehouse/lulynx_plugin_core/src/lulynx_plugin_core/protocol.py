"""Training event protocol definitions and payload builders.

Defines the schema for Tier2 training lifecycle events (before_forward,
after_loss, modify_loss, after_backward, before_optimizer_step,
after_optimizer_step) and provides typed payload factory functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lulynx_route_contract import classify_route


PROTOCOL_VERSION = "tier2.training.v1"


@dataclass(frozen=True)
class FieldSpec:
    """Schema entry for a single payload field."""

    name: str
    field_type: str
    required: bool
    description: str


@dataclass(frozen=True)
class EventSpec:
    """Schema entry for a training event type."""

    event: str
    description: str
    fields: tuple[FieldSpec, ...]
    notes: tuple[str, ...] = ()


# Common fields shared by all Tier2 events.
_COMMON_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("protocol_version",  "string",  True, "Protocol schema identifier."),
    FieldSpec("event",             "string",  True, "Event name."),
    FieldSpec("training_type",     "string",  True, "Resolved training type."),
    FieldSpec("route_kind",        "string",  True, "Route kind from contract."),
    FieldSpec("route_label",       "string",  True, "Human-readable route label."),
    FieldSpec("route_capabilities","string[]",True, "Capability tags from contract."),
    FieldSpec("source",            "string",  True, "Emitter identifier."),
    FieldSpec("global_step",       "integer", True, "Zero-based optimizer step index."),
    FieldSpec("gradient_accumulation_steps", "integer", True, "Gradient accumulation factor."),
    FieldSpec("sync_gradients",    "boolean", True, "Whether at gradient sync boundary."),
)

_MICRO_BATCH_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("micro_batch_index", "integer", True, "Current micro-batch index (1-based)."),
    FieldSpec("micro_batch_count", "integer", True, "Total micro-batches in this step."),
    FieldSpec("micro_batch_size",  "integer", True, "Sample count for this micro-batch."),
)

EVENT_SPECS: tuple[EventSpec, ...] = (
    EventSpec("before_forward", "Before model forward.", _COMMON_FIELDS + _MICRO_BATCH_FIELDS),
    EventSpec("after_loss", "After loss computed.", _COMMON_FIELDS + _MICRO_BATCH_FIELDS + (
        FieldSpec("loss", "number", True, "Raw micro-batch loss."),
        FieldSpec("loss_scale", "number", True, "Scale factor for backward."),
        FieldSpec("weighted_loss", "number", True, "Accumulated weighted loss so far."),
    )),
    EventSpec("modify_loss", "Mutation hook for loss transform.", _COMMON_FIELDS + _MICRO_BATCH_FIELDS + (
        FieldSpec("loss", "number", True, "Raw loss before mutation."),
        FieldSpec("loss_scale", "number", True, "Gradient accumulation scale."),
        FieldSpec("mutation", "object", True, "Mutable directive bag."),
    )),
    EventSpec("after_backward", "After backward completes.", _COMMON_FIELDS + _MICRO_BATCH_FIELDS + (
        FieldSpec("loss", "number", True, "Raw loss."),
        FieldSpec("loss_scale", "number", True, "Scale factor."),
        FieldSpec("backward_loss", "number", True, "Effective backward loss."),
        FieldSpec("weighted_loss", "number", True, "Accumulated weighted loss."),
    )),
    EventSpec("before_optimizer_step", "Before optimizer.step().", _COMMON_FIELDS + (
        FieldSpec("current_loss", "number", True, "Step loss."),
        FieldSpec("optimizer_type", "string", True, "Optimizer class name."),
        FieldSpec("scheduler_type", "string", True, "Scheduler class name."),
        FieldSpec("learning_rates", "number[]", True, "Current LRs."),
        FieldSpec("max_grad_norm", "number", True, "Gradient clip threshold."),
    )),
    EventSpec("after_optimizer_step", "After optimizer phase.", _COMMON_FIELDS + (
        FieldSpec("current_loss", "number", True, "Step loss."),
        FieldSpec("optimizer_type", "string", True, "Optimizer class name."),
        FieldSpec("scheduler_type", "string", True, "Scheduler class name."),
        FieldSpec("learning_rates", "number[]", True, "Post-phase LRs."),
        FieldSpec("max_grad_norm", "number", True, "Gradient clip threshold."),
        FieldSpec("optimizer_step_executed", "boolean", True, "Whether optimizer.step() ran."),
        FieldSpec("scheduler_step_executed", "boolean", True, "Whether scheduler.step() ran."),
        FieldSpec("zero_grad_called", "boolean", True, "Whether zero_grad was called."),
    )),
)

_SPEC_INDEX: dict[str, EventSpec] = {s.event: s for s in EVENT_SPECS}


def get_event_spec(event: str) -> EventSpec | None:
    """Look up an event spec by name."""
    return _SPEC_INDEX.get(str(event or "").strip())


def list_event_specs() -> list[dict]:
    """Return all event specs as plain dicts."""
    out: list[dict] = []
    for spec in EVENT_SPECS:
        out.append({
            "event": spec.event,
            "description": spec.description,
            "fields": [
                {"name": f.name, "type": f.field_type, "required": f.required, "description": f.description}
                for f in spec.fields
            ],
            "notes": list(spec.notes),
        })
    return out


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _common_payload(
    *,
    event: str,
    training_type: str | None,
    global_step: Any,
    gradient_accumulation_steps: Any,
    sync_gradients: bool,
    source: str,
) -> dict:
    contract = classify_route(training_type or "")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "event": event,
        "training_type": contract.training_type,
        "route_kind": contract.kind,
        "route_label": contract.label,
        "route_capabilities": list(contract.capability_tags),
        "source": str(source or "").strip() or "unknown",
        "global_step": _to_int(global_step),
        "gradient_accumulation_steps": max(1, _to_int(gradient_accumulation_steps, 1)),
        "sync_gradients": bool(sync_gradients),
    }


def build_before_forward_payload(
    *,
    training_type: str | None,
    global_step: Any,
    micro_batch_index: Any,
    micro_batch_count: Any,
    micro_batch_size: Any,
    gradient_accumulation_steps: Any,
    sync_gradients: bool,
    source: str,
) -> dict:
    p = _common_payload(
        event="before_forward", training_type=training_type,
        global_step=global_step, gradient_accumulation_steps=gradient_accumulation_steps,
        sync_gradients=sync_gradients, source=source,
    )
    p.update({
        "micro_batch_index": max(1, _to_int(micro_batch_index, 1)),
        "micro_batch_count": max(1, _to_int(micro_batch_count, 1)),
        "micro_batch_size": max(1, _to_int(micro_batch_size, 1)),
    })
    return p


def build_after_loss_payload(
    *,
    training_type: str | None,
    global_step: Any,
    micro_batch_index: Any,
    micro_batch_count: Any,
    micro_batch_size: Any,
    loss_value: Any,
    loss_scale: Any,
    weighted_loss: Any,
    gradient_accumulation_steps: Any,
    sync_gradients: bool,
    source: str,
) -> dict:
    p = _common_payload(
        event="after_loss", training_type=training_type,
        global_step=global_step, gradient_accumulation_steps=gradient_accumulation_steps,
        sync_gradients=sync_gradients, source=source,
    )
    p.update({
        "micro_batch_index": max(1, _to_int(micro_batch_index, 1)),
        "micro_batch_count": max(1, _to_int(micro_batch_count, 1)),
        "micro_batch_size": max(1, _to_int(micro_batch_size, 1)),
        "loss": _to_float(loss_value),
        "loss_scale": _to_float(loss_scale, 1.0),
        "weighted_loss": _to_float(weighted_loss),
    })
    return p


def build_modify_loss_payload(
    *,
    training_type: str | None,
    global_step: Any,
    micro_batch_index: Any,
    micro_batch_count: Any,
    micro_batch_size: Any,
    loss_value: Any,
    loss_scale: Any,
    gradient_accumulation_steps: Any,
    sync_gradients: bool,
    source: str,
) -> dict:
    p = _common_payload(
        event="modify_loss", training_type=training_type,
        global_step=global_step, gradient_accumulation_steps=gradient_accumulation_steps,
        sync_gradients=sync_gradients, source=source,
    )
    p.update({
        "micro_batch_index": max(1, _to_int(micro_batch_index, 1)),
        "micro_batch_count": max(1, _to_int(micro_batch_count, 1)),
        "micro_batch_size": max(1, _to_int(micro_batch_size, 1)),
        "loss": _to_float(loss_value),
        "loss_scale": _to_float(loss_scale, 1.0),
        "mutation": {"scale": 1.0, "bias": 0.0, "reason": "", "metadata": {}},
    })
    return p
