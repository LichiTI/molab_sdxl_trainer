"""Behavior-equivalent stage orchestrator shell for Lulynx training.

This module owns only the internal stage envelope. It does not add a training
entrypoint and does not start GPU or DataLoader work by itself; stage handlers
must be supplied by the existing trainer path when the internal gate is enabled.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from .training_pipeline_contract import lulynx_training_stage_ids
from .training_pipeline_execution_readiness import build_lulynx_training_pipeline_execution_readiness


LULYNX_TRAINING_STEP_ORCHESTRATOR_SLICE = "lulynx_training_step_orchestrator_slice_v0"
LulynxStageHandler = Callable[[MutableMapping[str, Any]], Any]


def build_lulynx_training_step_orchestrator_slice(
    *,
    runtime_features: Mapping[str, Any] | None = None,
    execution_readiness: Mapping[str, Any] | None = None,
    internal_gate_enabled: bool = False,
    internal_gate_requested: bool | None = None,
    stage_handlers: Mapping[str, LulynxStageHandler] | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe whether the behavior-equivalent stage shell can execute."""

    readiness = dict(_mapping(execution_readiness) or build_lulynx_training_pipeline_execution_readiness(
        runtime_features=runtime_features
    ))
    readiness_ready = _safe_bool(readiness.get("ready_for_behavior_equivalent_orchestrator_slice"))
    gate_enabled = _safe_bool(internal_gate_enabled)
    gate_requested = gate_enabled if internal_gate_requested is None else _safe_bool(internal_gate_requested)
    ordered_stage_ids = lulynx_training_stage_ids()
    available_handlers = _available_handler_ids(stage_handlers)
    missing_handlers = (
        [stage_id for stage_id in ordered_stage_ids if stage_id not in available_handlers]
        if available_handlers is not None
        else []
    )
    blockers: list[str] = []
    if not readiness_ready:
        readiness_blockers = _string_list(readiness.get("blockers")) or ["not_ready"]
        blockers.extend(f"pipeline_execution_readiness:{item}" for item in readiness_blockers)
    if missing_handlers:
        blockers.append("missing_behavior_equivalent_stage_handlers")
    if not gate_enabled:
        blockers.append("internal_orchestrator_gate_disabled")

    can_execute = readiness_ready and gate_enabled and not missing_handlers
    status = "ready_to_execute_behavior_equivalent_slice" if can_execute else "blocked"
    if readiness_ready and not missing_handlers and not gate_enabled:
        status = "ready_behind_disabled_internal_gate"

    return {
        "schema_version": 1,
        "orchestrator": LULYNX_TRAINING_STEP_ORCHESTRATOR_SLICE,
        "status": status,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "does_not_start_gpu_work_by_itself": True,
        "does_not_start_dataloader_iteration_by_itself": True,
        "behavior_equivalent_only": True,
        "ready_for_internal_gate": readiness_ready and not missing_handlers,
        "internal_gate_requested": gate_requested,
        "internal_gate_enabled": gate_enabled,
        "can_execute_behavior_equivalent_slice": can_execute,
        "ordered_stage_ids": ordered_stage_ids,
        "missing_stage_handlers": missing_handlers,
        "blockers": _dedupe(blockers),
        "readiness_gate": readiness.get("gate", ""),
        "readiness_status": readiness.get("status", ""),
    }


def run_lulynx_training_step_orchestrator_slice(
    *,
    stage_handlers: Mapping[str, LulynxStageHandler],
    runtime_features: Mapping[str, Any] | None = None,
    execution_readiness: Mapping[str, Any] | None = None,
    internal_gate_enabled: bool = False,
    initial_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run supplied behavior-equivalent stage handlers in contract order."""

    slice_report = build_lulynx_training_step_orchestrator_slice(
        runtime_features=runtime_features,
        execution_readiness=execution_readiness,
        internal_gate_enabled=internal_gate_enabled,
        stage_handlers=stage_handlers,
    )
    context: dict[str, Any] = dict(_mapping(initial_context))
    if not slice_report["can_execute_behavior_equivalent_slice"]:
        return _execution_report(
            status="not_executed",
            slice_report=slice_report,
            context=context,
            events=[],
        )

    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for stage_id in slice_report["ordered_stage_ids"]:
            stage_started = time.perf_counter()
            result = stage_handlers[stage_id](context)
            if result is not None:
                stage_results = context.setdefault("stage_results", {})
                if isinstance(stage_results, MutableMapping):
                    stage_results[stage_id] = result
            events.append(
                {
                    "stage_id": stage_id,
                    "elapsed_ms": round((time.perf_counter() - stage_started) * 1000.0, 4),
                }
            )
    except Exception as exc:
        return _execution_report(
            status="failed",
            slice_report=slice_report,
            context=context,
            events=events,
            total_ms=round((time.perf_counter() - started) * 1000.0, 4),
            error=f"{type(exc).__name__}: {exc}",
        )

    return _execution_report(
        status="executed_behavior_equivalent_slice",
        slice_report=slice_report,
        context=context,
        events=events,
        total_ms=round((time.perf_counter() - started) * 1000.0, 4),
    )


def _execution_report(
    *,
    status: str,
    slice_report: Mapping[str, Any],
    context: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    total_ms: float = 0.0,
    error: str = "",
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "orchestrator": LULYNX_TRAINING_STEP_ORCHESTRATOR_SLICE,
        "status": status,
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "slice": dict(slice_report),
        "stage_ids": [str(item.get("stage_id") or "") for item in events],
        "events": [dict(item) for item in events],
        "context": dict(context),
        "total_ms": float(total_ms),
    }
    if error:
        report["error"] = error
    return report


def _available_handler_ids(stage_handlers: Mapping[str, Any] | Sequence[str] | None) -> set[str] | None:
    if stage_handlers is None:
        return None
    if isinstance(stage_handlers, Mapping):
        return {str(key) for key, value in stage_handlers.items() if callable(value)}
    if isinstance(stage_handlers, (str, bytes)):
        return set()
    return {str(item) for item in stage_handlers if item is not None}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


__all__ = [
    "LULYNX_TRAINING_STEP_ORCHESTRATOR_SLICE",
    "LulynxStageHandler",
    "build_lulynx_training_step_orchestrator_slice",
    "run_lulynx_training_step_orchestrator_slice",
]
