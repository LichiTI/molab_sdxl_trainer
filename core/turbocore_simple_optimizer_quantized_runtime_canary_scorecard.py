"""Report-only runtime canary manifests for quantized simple variants."""

from __future__ import annotations

from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_simple_optimizer_quantized_native_scratch_scorecard import (
    build_simple_optimizer_quantized_native_scratch_scorecard,
)


TARGET_OPTIMIZERS = (
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
)
KIND_BY_OPTIMIZER = {
    OptimizerType.LION_8BIT: "lion8bit",
    OptimizerType.PAGED_LION_8BIT: "paged_lion8bit",
    OptimizerType.SGD_NESTEROV_8BIT: "sgd_nesterov8bit",
}
LAUNCH_PLAN_BY_OPTIMIZER = {
    OptimizerType.LION_8BIT: "lion8bit_flat_quantized_launch_plan_v0",
    OptimizerType.PAGED_LION_8BIT: "paged_lion8bit_flat_quantized_launch_plan_v0",
    OptimizerType.SGD_NESTEROV_8BIT: "sgd_nesterov8bit_flat_quantized_launch_plan_v0",
}


def build_simple_optimizer_quantized_runtime_canary_scorecard(
    *,
    native_scratch_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    """Build runtime canary manifests while keeping native dispatch blocked."""

    scratch = dict(native_scratch_report or build_simple_optimizer_quantized_native_scratch_scorecard())
    mode = _normalize_mode(native_training_mode)
    rows = [_route_case(optimizer, scratch, mode) for optimizer in TARGET_OPTIMIZERS]
    manifest_ready_count = sum(1 for row in rows if row["runtime_canary_manifest_ready"] is True)
    manifest_ready = manifest_ready_count == len(TARGET_OPTIMIZERS)
    blockers = _promotion_blockers(scratch, rows, mode, manifest_ready)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_quantized_runtime_canary_scorecard_v0",
        "gate": "simple_formula_quantized_runtime_canary_manifest",
        "ok": bool(scratch.get("ok", False)) and mode in {"off", "observe", "canary", "auto"},
        "promotion_ready": False,
        "runtime_canary_manifest_ready": manifest_ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "native_training_mode": mode,
        "optimizer_family": "simple_formula_quantized",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "product_native_dispatch_ready": False,
        "canary_shadow_route_only": True,
        "route_decisions": rows,
        "manifest_summary": {
            "optimizer_family": "simple_formula_quantized",
            "native_training_mode": mode,
            "target_optimizer_types": [optimizer.value for optimizer in TARGET_OPTIMIZERS],
            "native_scratch_kernel_ready_count": int(
                dict(scratch.get("summary") or {}).get("native_scratch_kernel_ready_count", 0) or 0
            ),
            "runtime_canary_manifest_ready_count": manifest_ready_count,
            "runtime_canary_ready": False,
            "training_path_enabled": False,
        },
        "native_scratch_summary": dict(scratch.get("summary") or {}),
        "summary": {
            "target_optimizer_count": len(TARGET_OPTIMIZERS),
            "runtime_canary_manifest_ready_count": manifest_ready_count,
            "runtime_canary_ready_count": 0,
            "native_route_blocked_count": sum(
                1 for row in rows if row["decision"] == "blocked_before_canary"
            ),
            "would_native_shadow_count": sum(
                1 for row in rows if row["decision"] == "would_native_shadow_but_blocked"
            ),
            "fallback_count": sum(1 for row in rows if row["decision"] == "fallback"),
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe([item for item in blockers if item != "e2e_no_regression_missing"]),
        "recommended_next_step": (
            "add quantized simple variant TrainingLoop canary evidence"
            if manifest_ready
            else "complete quantized simple variant native scratch kernel parity before runtime canary manifest"
        ),
        "notes": [
            "This manifest is request-shaped but never dispatches a native optimizer update.",
            "Native scratch kernels are proven on synthetic buffers only; live training tensor binding remains a later gate.",
            "Runtime dispatch and product exposure remain blocked even when the manifest is ready.",
        ],
    }


def _route_case(optimizer: OptimizerType, scratch: Mapping[str, Any], mode: str) -> dict[str, Any]:
    scratch_ready = _scratch_ready_for(optimizer, scratch)
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif mode == "observe" and scratch_ready:
        decision = "would_native_shadow_but_blocked"
        reason = "training_tensor_binding_and_training_loop_canary_missing"
    elif mode in {"canary", "auto"} and scratch_ready:
        decision = "blocked_before_canary"
        reason = "training_tensor_binding_and_training_loop_canary_missing"
    else:
        decision = "fallback"
        reason = "native_scratch_kernel_not_ready"
    manifest_ready = scratch_ready and decision in {"would_native_shadow_but_blocked", "blocked_before_canary"}
    return {
        "schema_version": 1,
        "feature": "simple_formula_quantized_optimizer",
        "optimizer_type": optimizer.value,
        "optimizer_kind": KIND_BY_OPTIMIZER[optimizer],
        "optimizer_family": "simple_formula_quantized",
        "native_scratch_kernel_ready": scratch_ready,
        "runtime_canary_manifest_ready": manifest_ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "canary_shadow_route_only": True,
        "request_fields": {
            "optimizer_family": "simple_formula_quantized",
            "optimizer_type": optimizer.value,
            "optimizer_kind": KIND_BY_OPTIMIZER[optimizer],
            "native_training_mode": mode,
            "launch_plan": LAUNCH_PLAN_BY_OPTIMIZER[optimizer],
        },
        "missing_before_dispatch": [
            "training_tensor_binding",
            "training_loop_canary",
            "runtime_canary_e2e_no_regression",
            "product_rollout_review",
        ],
    }


def _scratch_ready_for(optimizer: OptimizerType, report: Mapping[str, Any]) -> bool:
    rows = [
        row
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer.value
    ]
    return bool(rows and rows[0].get("native_scratch_kernel_parity_ready") is True)


def _promotion_blockers(
    scratch: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    mode: str,
    manifest_ready: bool,
) -> list[str]:
    blockers: list[str] = []
    if mode == "off":
        blockers.append("simple_quantized_runtime_canary_mode_off")
    if not bool(scratch.get("native_scratch_kernel_parity_ready", False)):
        blockers.append("simple_quantized_native_scratch_kernel_parity_missing")
    for row in rows:
        if not bool(row.get("runtime_canary_manifest_ready", False)):
            blockers.append(f"{row.get('optimizer_type')}_runtime_canary_manifest_missing")
    if manifest_ready:
        blockers.extend(
            [
                "simple_quantized_training_tensor_binding_missing",
                "simple_quantized_training_loop_canary_missing",
                "e2e_no_regression_missing",
                "product_rollout_review_missing",
            ]
        )
    return blockers


def _normalize_mode(value: str) -> str:
    normalized = str(value or "canary").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "canary"


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_simple_optimizer_quantized_runtime_canary_scorecard"]
