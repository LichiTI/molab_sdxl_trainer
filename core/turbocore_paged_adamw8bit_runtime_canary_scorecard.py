"""Report-only runtime canary manifest for PagedAdamW8bit native work."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_paged_adamw8bit_native_scratch_kernel_scorecard import (
    build_paged_adamw8bit_native_scratch_kernel_scorecard,
)
from core.turbocore_paged_adamw8bit_quantized_update_scorecard import (
    build_paged_adamw8bit_quantized_update_scorecard,
)


OPTIMIZER_KIND = "paged_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_paged"
LAUNCH_PLAN = "paged_adamw8bit_flat_quantized_launch_plan_v0"


def build_paged_adamw8bit_runtime_canary_scorecard(
    *,
    quantized_update_report: Mapping[str, Any] | None = None,
    native_kernel_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    """Build a canary manifest without dispatching training updates."""

    update = dict(
        quantized_update_report
        or build_paged_adamw8bit_quantized_update_scorecard(run_live_probe=False)
    )
    kernel = dict(
        native_kernel_report
        or build_paged_adamw8bit_native_scratch_kernel_scorecard(quantized_update_report=update)
    )
    mode = _normalize_mode(native_training_mode)
    route = _route_decision(update, kernel, mode)
    manifest_ready = (
        bool(update.get("quantized_update_contract_ready", False))
        and bool(kernel.get("native_scratch_kernel_parity_ready", False))
        and route["decision"] in {"would_native_shadow_but_blocked", "blocked_before_canary"}
        and not bool(route.get("training_path_enabled", True))
    )
    blockers = _promotion_blockers(update, kernel, mode, manifest_ready)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_runtime_canary_scorecard_v0",
        "gate": "paged_adamw8bit_runtime_canary_manifest",
        "ok": bool(update.get("ok", False)) and bool(kernel.get("ok", False)) and mode in {"off", "observe", "canary", "auto"},
        "promotion_ready": False,
        "runtime_canary_manifest_ready": manifest_ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "native_training_mode": mode,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": True,
        "route_decision": route,
        "manifest_summary": {
            "optimizer_family": OPTIMIZER_FAMILY,
            "optimizer_kind": OPTIMIZER_KIND,
            "native_training_mode": mode,
            "quantized_update_contract_ready": bool(update.get("quantized_update_contract_ready", False)),
            "native_scratch_kernel_parity_ready": bool(kernel.get("native_scratch_kernel_parity_ready", False)),
            "runtime_canary_manifest_ready": manifest_ready,
            "runtime_canary_ready": False,
            "training_path_enabled": False,
        },
        "quantized_update_summary": dict(update.get("summary") or {}),
        "native_kernel_summary": dict(kernel.get("summary") or {}),
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe([item for item in blockers if item != "e2e_no_regression_missing"]),
        "recommended_next_step": (
            "compare native scratch kernel against bnb exact live buffers before tensor binding"
            if manifest_ready
            else "complete PagedAdamW8bit P8E/P8F gates before runtime canary manifest"
        ),
        "notes": [
            "This manifest is request-shaped but never dispatches a native optimizer update.",
            "P8E proves bnb functional oracle parity; P8F proves a Lulynx native qmap-compatible scratch kernel.",
            "The manifest keeps native dispatch blocked until bnb-exact native parity, tensor binding, and checkpoint runtime adapter exist.",
        ],
    }


def _route_decision(update: Mapping[str, Any], kernel: Mapping[str, Any], mode: str) -> dict[str, Any]:
    update_ready = bool(update.get("quantized_update_contract_ready", False))
    kernel_ready = bool(kernel.get("native_scratch_kernel_parity_ready", False))
    ready = update_ready and kernel_ready
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif mode == "observe" and ready:
        decision = "would_native_shadow_but_blocked"
        reason = "bnb_exact_native_parity_and_tensor_binding_missing"
    elif mode in {"canary", "auto"} and ready:
        decision = "blocked_before_canary"
        reason = "bnb_exact_native_parity_and_tensor_binding_missing"
    else:
        decision = "fallback"
        reason = "p8e_or_p8f_gate_not_ready"
    return {
        "schema_version": 1,
        "feature": "paged_adamw8bit_native_optimizer",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": True,
        "request_fields": {
            "optimizer_family": OPTIMIZER_FAMILY,
            "optimizer_kind": OPTIMIZER_KIND,
            "native_training_mode": mode,
            "launch_plan": LAUNCH_PLAN,
        },
        "missing_before_dispatch": [
            "bnb_exact_native_parity",
            "training_tensor_binding",
            "checkpoint_adapter_runtime",
            "runtime_canary_e2e_no_regression",
        ],
    }


def _promotion_blockers(
    update: Mapping[str, Any],
    kernel: Mapping[str, Any],
    mode: str,
    manifest_ready: bool,
) -> list[str]:
    blockers = []
    if mode == "off":
        blockers.append("paged_adamw8bit_runtime_canary_mode_off")
    if not bool(update.get("quantized_update_contract_ready", False)):
        blockers.append("paged_adamw8bit_quantized_update_contract_missing")
    if not bool(kernel.get("native_scratch_kernel_parity_ready", False)):
        blockers.append("paged_adamw8bit_native_scratch_kernel_parity_missing")
    if manifest_ready:
        blockers.extend(
            [
                "paged_adamw8bit_bnb_exact_native_parity_missing",
                "paged_adamw8bit_training_tensor_binding_missing",
                "paged_adamw8bit_checkpoint_adapter_runtime_missing",
                "e2e_no_regression_missing",
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


__all__ = ["build_paged_adamw8bit_runtime_canary_scorecard"]
