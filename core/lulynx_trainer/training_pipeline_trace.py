"""Lightweight stage trace for the Lulynx train-step pipeline.

The trace records stage boundaries and batch contract metadata without changing
the active training computation. It is intentionally diagnostic: useful for
batch2/4/8 stability work, not a release performance claim.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from backend.core.lulynx_trainer.multi_batch_contract import inspect_batch_contract
from backend.core.lulynx_trainer.training_pipeline_contract import lulynx_training_stage_ids


LULYNX_TRAINING_PIPELINE_TRACE_REPORT = "lulynx_training_pipeline_trace_v0"


class LulynxTrainingPipelineTrace:
    """Collect one train-step stage trace with tiny CPU-side overhead."""

    def __init__(self, *, max_events: int = 64) -> None:
        self.max_events = max(int(max_events or 64), 8)
        self._active = False
        self._started = 0.0
        self._last_stage_started = 0.0
        self._last_stage_id = ""
        self._events: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}
        self._summary: dict[str, Any] = {}

    def start(self, *, batch: Mapping[str, Any], expected_physical_batch_size: Any | None = None) -> None:
        now = time.perf_counter()
        self._active = True
        self._started = now
        self._last_stage_started = now
        self._last_stage_id = ""
        self._events = []
        self._metadata = {
            "batch_contract": inspect_batch_contract(
                batch,
                expected_physical_batch_size=expected_physical_batch_size,
            ),
        }
        self._summary = {}

    def mark(self, stage_id: str, **metadata: Any) -> None:
        if not self._active:
            return
        now = time.perf_counter()
        if len(self._events) < self.max_events:
            self._events.append(
                {
                    "stage_id": str(stage_id),
                    "since_previous_ms": round((now - self._last_stage_started) * 1000.0, 4),
                    "after_stage_id": self._last_stage_id,
                    "metadata": {str(key): value for key, value in metadata.items() if value is not None},
                }
            )
        self._last_stage_started = now
        self._last_stage_id = str(stage_id)

    def finish(self, *, status: str = "completed", error: str = "") -> dict[str, Any]:
        if not self._active and self._summary:
            return dict(self._summary)
        now = time.perf_counter()
        stage_ids = [str(item.get("stage_id") or "") for item in self._events]
        expected = lulynx_training_stage_ids()
        missing = [stage for stage in expected if stage not in stage_ids]
        batch_contract = self._metadata.get("batch_contract", {})
        self._summary = {
            "schema_version": 1,
            "report": LULYNX_TRAINING_PIPELINE_TRACE_REPORT,
            "status": str(status or "completed"),
            "release_claim_allowed": False,
            "total_ms": round((now - self._started) * 1000.0, 4) if self._started else 0.0,
            "stage_count": len(stage_ids),
            "stage_ids": stage_ids,
            "last_stage_id": self._last_stage_id,
            "missing_contract_stages": missing,
            "batch_contract": dict(batch_contract) if isinstance(batch_contract, Mapping) else {},
            "events": list(self._events),
        }
        if error:
            self._summary["error"] = str(error)
        self._active = False
        return dict(self._summary)

    def mark_completed_stage(self, stage_id: str, **metadata: Any) -> dict[str, Any]:
        """Append a report-only stage after the train-step trace was finished."""

        if self._active:
            self.mark(stage_id, **metadata)
            return self.finish(status="active_snapshot")
        if not self._summary:
            return {}
        stage = str(stage_id)
        stage_ids = [str(item) for item in self._summary.get("stage_ids", []) if item]
        if stage in stage_ids:
            return dict(self._summary)
        now = time.perf_counter()
        event = {
            "stage_id": stage,
            "since_previous_ms": round((now - self._last_stage_started) * 1000.0, 4)
            if self._last_stage_started
            else 0.0,
            "after_stage_id": str(self._summary.get("last_stage_id") or self._last_stage_id or ""),
            "metadata": {str(key): value for key, value in metadata.items() if value is not None},
        }
        events = list(self._summary.get("events", [])) if isinstance(self._summary.get("events"), Sequence) else []
        if len(events) < self.max_events:
            events.append(event)
        stage_ids.append(stage)
        expected = lulynx_training_stage_ids()
        self._summary["events"] = events
        self._summary["stage_ids"] = stage_ids
        self._summary["stage_count"] = len(stage_ids)
        self._summary["last_stage_id"] = stage
        self._summary["missing_contract_stages"] = [item for item in expected if item not in stage_ids]
        self._summary["total_ms"] = round((now - self._started) * 1000.0, 4) if self._started else 0.0
        self._last_stage_started = now
        self._last_stage_id = stage
        return dict(self._summary)

    def snapshot(self) -> dict[str, Any]:
        if self._summary:
            return dict(self._summary)
        if self._active:
            return self.finish(status="active_snapshot")
        return {}


def compact_lulynx_pipeline_trace(trace: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the stable public subset used in runtime profiles."""

    if not isinstance(trace, Mapping) or not trace:
        return {}
    batch_contract = trace.get("batch_contract") if isinstance(trace.get("batch_contract"), Mapping) else {}
    stage_plans = _compact_lulynx_stage_plans(trace.get("events"))
    execution_strategy = _compact_multi_batch_execution_strategy(trace.get("events"))
    execution_strategy_gate = _compact_multi_batch_execution_strategy_gate(trace.get("events"))
    if not stage_plans and isinstance(trace.get("stage_plans"), Mapping):
        stage_plans = {str(key): dict(value) for key, value in trace["stage_plans"].items() if isinstance(value, Mapping)}
    return {
        "schema_version": 1,
        "report": str(trace.get("report") or LULYNX_TRAINING_PIPELINE_TRACE_REPORT),
        "status": str(trace.get("status") or ""),
        "release_claim_allowed": False,
        "total_ms": float(trace.get("total_ms") or 0.0),
        "stage_count": int(trace.get("stage_count") or 0),
        "stage_ids": [str(item) for item in trace.get("stage_ids", []) if item is not None]
        if isinstance(trace.get("stage_ids"), Sequence) and not isinstance(trace.get("stage_ids"), (str, bytes))
        else [],
        "last_stage_id": str(trace.get("last_stage_id") or ""),
        "missing_contract_stages": [str(item) for item in trace.get("missing_contract_stages", []) if item is not None]
        if isinstance(trace.get("missing_contract_stages"), Sequence)
        and not isinstance(trace.get("missing_contract_stages"), (str, bytes))
        else [],
        "batch_contract": {
            "contract": str(batch_contract.get("contract") or ""),
            "ok": bool(batch_contract.get("ok")),
            "expected_physical_batch_size": int(batch_contract.get("expected_physical_batch_size") or 0),
            "inferred_physical_batch_size": int(batch_contract.get("inferred_physical_batch_size") or 0),
            "real_multi_batch": bool(batch_contract.get("real_multi_batch")),
            "warnings": [str(item) for item in batch_contract.get("warnings", [])]
            if isinstance(batch_contract.get("warnings"), Sequence)
            and not isinstance(batch_contract.get("warnings"), (str, bytes))
            else [],
        },
        "stage_plans": stage_plans,
        "multi_batch_execution_strategy": execution_strategy,
        "multi_batch_execution_strategy_gate": execution_strategy_gate,
        "error": str(trace.get("error") or ""),
    }


def _compact_lulynx_stage_plans(events: Any) -> dict[str, Any]:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return {}
    plans: dict[str, Any] = {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        for key, value in metadata.items():
            key_text = str(key)
            if key_text.endswith("_stage_plan") and isinstance(value, Mapping):
                plans[key_text] = dict(value)
    return plans


def _compact_multi_batch_execution_strategy(events: Any) -> dict[str, Any]:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        value = metadata.get("multi_batch_execution_strategy")
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _compact_multi_batch_execution_strategy_gate(events: Any) -> dict[str, Any]:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return {}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        value = metadata.get("multi_batch_execution_strategy_gate")
        if isinstance(value, Mapping):
            return dict(value)
    return {}


__all__ = [
    "LULYNX_TRAINING_PIPELINE_TRACE_REPORT",
    "LulynxTrainingPipelineTrace",
    "compact_lulynx_pipeline_trace",
]
