"""E2E no-regression gate for quantized simple optimizer canaries."""

from __future__ import annotations

from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_simple_optimizer_quantized_training_loop_canary_scorecard import (
    build_simple_optimizer_quantized_training_loop_canary_scorecard,
)


TARGET_OPTIMIZERS = (
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
)


def build_simple_optimizer_quantized_e2e_no_regression_scorecard(
    *,
    training_loop_canary_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Review quantized canary e2e evidence without promoting product dispatch."""

    training = dict(training_loop_canary_report or build_simple_optimizer_quantized_training_loop_canary_scorecard())
    rows = [_row(optimizer, training) for optimizer in TARGET_OPTIMIZERS]
    ready_count = sum(1 for row in rows if row["e2e_no_regression_ready"] is True)
    fallback_count = sum(1 for row in rows if row["fallback_authority_unchanged"] is True)
    state_review_count = sum(1 for row in rows if row["product_state_sync_review_ready"] is True)
    ready = ready_count == len(TARGET_OPTIMIZERS)
    blockers = _dedupe(reason for row in rows for reason in _strings(row.get("blocked_reasons")))
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_quantized_e2e_no_regression_scorecard_v0",
        "gate": "simple_formula_quantized_e2e_no_regression",
        "ok": ready and not blockers,
        "promotion_ready": False,
        "e2e_no_regression_ready": ready and not blockers,
        "product_state_sync_review_ready": state_review_count == len(TARGET_OPTIMIZERS),
        "product_optimizer_state_sync_ready": False,
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
        "training_loop_canary_summary": dict(training.get("summary") or {}),
        "summary": {
            "target_optimizer_count": len(TARGET_OPTIMIZERS),
            "training_loop_canary_ready_count": int(
                dict(training.get("summary") or {}).get("training_loop_canary_ready_count", 0) or 0
            ),
            "e2e_no_regression_ready_count": ready_count,
            "fallback_authority_unchanged_count": fallback_count,
            "product_state_sync_review_ready_count": state_review_count,
            "product_optimizer_state_sync_ready_count": 0,
            "request_schema_ui_unchanged_count": sum(
                1 for row in rows if row["request_schema_ui_unchanged"] is True
            ),
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "product_optimizer_state_sync_missing",
                "product_rollout_review_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "review product optimizer-state sync ownership before any quantized product dispatch"
            if ready
            else "fix simple quantized e2e no-regression blockers"
        ),
        "notes": [
            "This gate consumes real TrainingLoop canary evidence and proves the default product route remains untouched.",
            "The canary executor owns temporary uint8 state; product optimizer-state synchronization is reviewed but not claimed ready.",
            "No request, UI, or schema exposure is added by this scorecard.",
        ],
    }


def _row(optimizer: OptimizerType, training: Mapping[str, Any]) -> dict[str, Any]:
    source = _training_row(optimizer, training)
    case = _training_case(optimizer, training)
    source_ready = source.get("training_loop_canary_ready") is True
    default_off = _default_off(source) and _default_off(training)
    fallback_unchanged = default_off and _product_dispatch_disabled(source, training)
    state_review_ready = source_ready and case.get("native_kernel_launched") is True
    optimizer_state_synced = case.get("pytorch_optimizer_state_synced") is True
    request_schema_ui_unchanged = not any(
        source.get(field) is True or training.get(field) is True
        for field in ("request_fields_emitted", "schema_exposure_allowed", "ui_exposure_allowed")
    )
    ready = source_ready and fallback_unchanged and state_review_ready and request_schema_ui_unchanged
    return {
        "schema_version": 1,
        "optimizer_type": optimizer.value,
        "optimizer_kind": str(source.get("optimizer_kind") or _kind(optimizer)),
        "optimizer_family": "simple_formula_quantized",
        "variant_status": "quantized_e2e_no_regression_ready" if ready else "quantized_e2e_no_regression_blocked",
        "training_loop_canary_ready": source_ready,
        "native_kernel_launched": case.get("native_kernel_launched") is True,
        "training_executor_called": case.get("training_executor_called") is True,
        "fallback_authority_unchanged": fallback_unchanged,
        "request_schema_ui_unchanged": request_schema_ui_unchanged,
        "product_state_sync_review_ready": state_review_ready,
        "optimizer_state_sync_synced": optimizer_state_synced,
        "optimizer_state_sync_state_tensors": int(case.get("optimizer_state_sync_state_tensors", 0) or 0),
        "optimizer_state_sync_parameter_tensors": int(case.get("optimizer_state_sync_parameter_tensors", 0) or 0),
        "product_optimizer_state_sync_ready": False,
        "product_optimizer_state_sync_required_before_dispatch": True,
        "e2e_no_regression_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_dispatch_ready": False,
        "state_sync_review": {
            "canary_state_owner": "turbocore_simple_quantized_optimizer_training_executor_v0",
            "product_state_owner": "pytorch_optimizer_state_dict",
            "sync_boundary_reviewed": state_review_ready,
            "sync_implementation_ready": False,
            "required_before_product_dispatch": True,
        },
        "case": dict(case),
        "blocked_reasons": [] if ready else _row_blockers(source_ready, fallback_unchanged, state_review_ready),
    }


def _training_row(optimizer: OptimizerType, report: Mapping[str, Any]) -> dict[str, Any]:
    for row in report.get("rows", []):
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value:
            return dict(row)
    return {}


def _training_case(optimizer: OptimizerType, report: Mapping[str, Any]) -> dict[str, Any]:
    for case in report.get("cases", []):
        if isinstance(case, Mapping) and case.get("optimizer_type") == optimizer.value:
            return dict(case)
    row_case = _training_row(optimizer, report).get("case")
    return dict(row_case) if isinstance(row_case, Mapping) else {}


def _product_dispatch_disabled(*reports: Mapping[str, Any]) -> bool:
    return all(
        report.get("native_dispatch_allowed") is not True
        and report.get("runtime_dispatch_ready") is not True
        and report.get("product_native_dispatch_ready") is not True
        for report in reports
    )


def _default_off(report: Mapping[str, Any]) -> bool:
    return bool(report) and all(
        report.get(field) is not True
        for field in (
            "training_path_enabled",
            "default_behavior_changed",
            "native_dispatch_allowed",
            "runtime_dispatch_ready",
            "product_native_dispatch_ready",
        )
    )


def _row_blockers(source_ready: bool, fallback_unchanged: bool, state_review_ready: bool) -> list[str]:
    blockers: list[str] = []
    if not source_ready:
        blockers.append("quantized_training_loop_canary_missing")
    if not fallback_unchanged:
        blockers.append("product_fallback_authority_changed")
    if not state_review_ready:
        blockers.append("product_state_sync_review_missing")
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


__all__ = ["build_simple_optimizer_quantized_e2e_no_regression_scorecard"]
