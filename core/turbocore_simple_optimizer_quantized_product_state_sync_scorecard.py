"""Product optimizer-state sync gate for quantized simple optimizer canaries."""

from __future__ import annotations

from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_simple_optimizer_quantized_e2e_no_regression_scorecard import (
    build_simple_optimizer_quantized_e2e_no_regression_scorecard,
)


TARGET_OPTIMIZERS = (
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
)


def build_simple_optimizer_quantized_product_state_sync_scorecard(
    *,
    e2e_no_regression_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify PyTorch optimizer state_dict compatibility without product dispatch."""

    e2e = dict(e2e_no_regression_report or build_simple_optimizer_quantized_e2e_no_regression_scorecard())
    rows = [_row(optimizer, e2e) for optimizer in TARGET_OPTIMIZERS]
    ready_count = sum(1 for row in rows if row["product_optimizer_state_sync_ready"] is True)
    fallback_count = sum(1 for row in rows if row["fallback_authority_unchanged"] is True)
    request_schema_count = sum(1 for row in rows if row["request_schema_ui_unchanged"] is True)
    ready = ready_count == len(TARGET_OPTIMIZERS)
    blockers = _dedupe(reason for row in rows for reason in _strings(row.get("blocked_reasons")))
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_quantized_product_state_sync_scorecard_v0",
        "gate": "simple_formula_quantized_product_optimizer_state_sync",
        "ok": ready and not blockers,
        "promotion_ready": False,
        "product_optimizer_state_sync_ready": ready and not blockers,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "product_native_dispatch_ready": False,
        "optimizer_family": "simple_formula_quantized",
        "rows": rows,
        "e2e_no_regression_summary": dict(e2e.get("summary") or {}),
        "summary": {
            "target_optimizer_count": len(TARGET_OPTIMIZERS),
            "e2e_no_regression_ready_count": int(
                dict(e2e.get("summary") or {}).get("e2e_no_regression_ready_count", 0) or 0
            ),
            "product_optimizer_state_sync_ready_count": ready_count,
            "fallback_authority_unchanged_count": fallback_count,
            "request_schema_ui_unchanged_count": request_schema_count,
            "optimizer_state_sync_state_tensor_count": sum(
                int(row.get("optimizer_state_sync_state_tensors", 0) or 0) for row in rows
            ),
            "optimizer_state_sync_parameter_tensor_count": sum(
                int(row.get("optimizer_state_sync_parameter_tensors", 0) or 0) for row in rows
            ),
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(blockers + ["product_rollout_review_missing"]),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run default-off quantized rollout review before any product native dispatch"
            if ready
            else "fix simple quantized product optimizer-state sync blockers"
        ),
        "notes": [
            "The sync writes TurboCore-owned quantized state into the existing PyTorch optimizer state_dict namespace.",
            "This makes checkpoint/fallback ownership explicit while keeping request/UI/schema exposure and product dispatch disabled.",
            "Product native ready remains false until rollout review and owner approval.",
        ],
    }


def _row(optimizer: OptimizerType, e2e: Mapping[str, Any]) -> dict[str, Any]:
    source = _source_row(optimizer, e2e)
    e2e_ready = source.get("e2e_no_regression_ready") is True
    sync_ready = (
        source.get("optimizer_state_sync_synced") is True
        and int(source.get("optimizer_state_sync_state_tensors", 0) or 0) >= 2
        and int(source.get("optimizer_state_sync_parameter_tensors", 0) or 0) >= 1
    )
    fallback_unchanged = source.get("fallback_authority_unchanged") is True
    request_schema_ui_unchanged = source.get("request_schema_ui_unchanged") is True
    ready = e2e_ready and sync_ready and fallback_unchanged and request_schema_ui_unchanged
    return {
        "schema_version": 1,
        "optimizer_type": optimizer.value,
        "optimizer_kind": str(source.get("optimizer_kind") or _kind(optimizer)),
        "optimizer_family": "simple_formula_quantized",
        "variant_status": "quantized_product_state_sync_ready" if ready else "quantized_product_state_sync_blocked",
        "e2e_no_regression_ready": e2e_ready,
        "product_optimizer_state_sync_ready": ready,
        "optimizer_state_sync_synced": sync_ready,
        "optimizer_state_sync_state_tensors": int(source.get("optimizer_state_sync_state_tensors", 0) or 0),
        "optimizer_state_sync_parameter_tensors": int(source.get("optimizer_state_sync_parameter_tensors", 0) or 0),
        "fallback_authority_unchanged": fallback_unchanged,
        "request_schema_ui_unchanged": request_schema_ui_unchanged,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_dispatch_ready": False,
        "blocked_reasons": [] if ready else _row_blockers(e2e_ready, sync_ready, fallback_unchanged),
    }


def _source_row(optimizer: OptimizerType, report: Mapping[str, Any]) -> dict[str, Any]:
    for row in report.get("rows", []):
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value:
            return dict(row)
    return {}


def _row_blockers(e2e_ready: bool, sync_ready: bool, fallback_unchanged: bool) -> list[str]:
    blockers: list[str] = []
    if not e2e_ready:
        blockers.append("quantized_e2e_no_regression_missing")
    if not sync_ready:
        blockers.append("quantized_product_optimizer_state_sync_missing")
    if not fallback_unchanged:
        blockers.append("product_fallback_authority_changed")
    return blockers


def _kind(optimizer: OptimizerType) -> str:
    return {
        OptimizerType.LION_8BIT: "lion8bit",
        OptimizerType.PAGED_LION_8BIT: "paged_lion8bit",
        OptimizerType.SGD_NESTEROV_8BIT: "sgd_nesterov8bit",
    }[optimizer]


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


__all__ = ["build_simple_optimizer_quantized_product_state_sync_scorecard"]
