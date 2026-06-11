"""Shared runtime report helpers for Lulynx stage handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .training_pipeline_contract import lulynx_training_stage_ids
from .training_step_orchestrator import LULYNX_TRAINING_STEP_ORCHESTRATOR_SLICE


def build_lulynx_stage_orchestrator_runtime(
    *,
    executed_stage_ids: tuple[str, ...],
    status: str,
    handler_source: str,
    stage_plans: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage_ids = lulynx_training_stage_ids()
    known_stage_ids = set(stage_ids)
    executed = [stage_id for stage_id in executed_stage_ids if stage_id in known_stage_ids]
    report = {
        "schema_version": 1,
        "orchestrator": LULYNX_TRAINING_STEP_ORCHESTRATOR_SLICE,
        "status": str(status or ""),
        "release_claim_allowed": False,
        "does_not_add_training_entrypoint": True,
        "behavior_equivalent_only": True,
        "internal_gate_enabled": False,
        "executed_stage_ids": executed,
        "pending_stage_ids": [stage_id for stage_id in stage_ids if stage_id not in executed],
        "handler_source": str(handler_source or ""),
        "stage_plans": dict(stage_plans or {}),
    }
    report.update(dict(extra or {}))
    return report


__all__ = ["build_lulynx_stage_orchestrator_runtime"]
