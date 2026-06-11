"""Request-shaped runtime canary for V2-P7 simple optimizer kernels."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_simple_optimizer_kernel_parity_scorecard import (
    build_simple_optimizer_kernel_parity_scorecard,
)


TARGET_OPTIMIZER_KINDS = ("lion", "sgd_nesterov")


def build_simple_optimizer_runtime_canary_scorecard(
    *,
    kernel_parity_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
) -> dict[str, Any]:
    """Build a report-only canary route without dispatching training updates."""

    kernel = dict(kernel_parity_report or build_simple_optimizer_kernel_parity_scorecard())
    mode = _normalize_mode(native_training_mode)
    cases = [_route_case(kind, kernel, mode) for kind in TARGET_OPTIMIZER_KINDS]
    hit_count = sum(1 for case in cases if case["decision"] == "native_canary_shadow")
    ready = (
        bool(kernel.get("kernel_parity_stage_ready", False))
        and mode in {"canary", "auto"}
        and hit_count == len(TARGET_OPTIMIZER_KINDS)
        and all(not bool(case.get("training_path_enabled", True)) for case in cases)
    )
    blockers = []
    if mode == "off":
        blockers.append("simple_optimizer_runtime_canary_mode_off")
    if mode == "observe":
        blockers.append("simple_optimizer_runtime_canary_observe_only")
    if not bool(kernel.get("kernel_parity_stage_ready", False)):
        blockers.append("simple_optimizer_kernel_parity_gate_not_ready")
    if hit_count < len(TARGET_OPTIMIZER_KINDS) and mode in {"canary", "auto"}:
        blockers.append("simple_optimizer_runtime_canary_not_all_targets_hit")
    if ready:
        blockers.append("e2e_no_regression_missing")
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_runtime_canary_scorecard_v0",
        "gate": "simple_formula_runtime_canary",
        "ok": bool(kernel.get("ok", False)) and mode in {"off", "observe", "canary", "auto"},
        "promotion_ready": False,
        "runtime_canary_ready": ready,
        "runtime_canary_hit": ready,
        "native_training_mode": mode,
        "optimizer_family": "simple_formula",
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": True,
        "route_decisions": cases,
        "native_route_hit_count": hit_count,
        "would_native_count": sum(1 for case in cases if case["decision"] in {"native_canary_shadow", "would_native_shadow"}),
        "fallback_count": sum(1 for case in cases if case["decision"] == "fallback"),
        "manifest_summary": {
            "optimizer_family": "simple_formula",
            "native_training_mode": mode,
            "target_optimizer_kinds": list(TARGET_OPTIMIZER_KINDS),
            "kernel_parity_stage_ready": bool(kernel.get("kernel_parity_stage_ready", False)),
            "native_route_hit_count": hit_count,
            "training_path_enabled": False,
        },
        "kernel_parity_summary": dict(kernel.get("summary") or {}),
        "promotion_blockers": _dedupe(blockers),
        "blocked_reasons": _dedupe([item for item in blockers if item != "e2e_no_regression_missing"]),
        "recommended_next_step": "run simple formula optimizer e2e no-regression smoke" if ready else "complete simple formula optimizer kernel parity before canary",
    }


def _route_case(optimizer_kind: str, kernel: Mapping[str, Any], mode: str) -> dict[str, Any]:
    gate_name = f"{optimizer_kind}_native_kernel_parity"
    ready = bool(kernel.get(gate_name, False))
    if mode == "observe":
        decision = "would_native_shadow" if ready else "fallback"
    elif mode in {"canary", "auto"} and ready:
        decision = "native_canary_shadow"
    else:
        decision = "fallback"
    return {
        "schema_version": 1,
        "feature": "simple_formula_optimizer",
        "optimizer_kind": optimizer_kind,
        "optimizer_family": "simple_formula",
        "kernel_parity_ready": ready,
        "native_training_mode": mode,
        "decision": decision,
        "reason": "kernel_parity_ready" if ready else "kernel_parity_not_ready",
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": True,
        "request_fields": {
            "optimizer_family": "simple_formula",
            "optimizer_kind": optimizer_kind,
            "native_training_mode": mode,
            "launch_plan": f"{optimizer_kind}_flat_fp32_launch_plan_v0",
        },
    }


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


__all__ = ["build_simple_optimizer_runtime_canary_scorecard"]
