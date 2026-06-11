"""Report-only canary dispatch manifest for KahanAdamW8bit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_kahan_adamw8bit_real_training_matrix_scorecard import (
    build_kahan_adamw8bit_real_training_matrix_scorecard,
)


OPTIMIZER_KIND = "kahan_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_kahan"
MANIFEST_KIND = "kahan_adamw8bit_canary_dispatch_manifest_v0"
LAUNCH_PLAN = "kahan_adamw8bit_training_tensor_binding_launch_plan_v0"


def build_kahan_adamw8bit_canary_dispatch_manifest_scorecard(
    *,
    matrix_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
    run_live_probe: bool = True,
    require_live_matrix: bool = True,
) -> dict[str, Any]:
    """Build a post-matrix dispatch manifest without enabling native dispatch."""

    mode = _normalize_mode(native_training_mode)
    matrix = dict(
        matrix_report
        or build_kahan_adamw8bit_real_training_matrix_scorecard(run_live_probe=run_live_probe)
    )
    route = _route_decision(matrix, mode, require_live_matrix=bool(require_live_matrix))
    validations = _validations(matrix, route, mode, require_live_matrix=bool(require_live_matrix))
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    manifest_ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_kahan_adamw8bit_canary_dispatch_manifest_scorecard_v0",
        "gate": "kahan_adamw8bit_canary_dispatch_manifest",
        "ok": manifest_ready,
        "promotion_ready": False,
        "canary_dispatch_manifest_ready": manifest_ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "canary_dispatch_armed": False,
        "canary_dispatch_review_required": True,
        "manifest_kind": MANIFEST_KIND,
        "launch_plan": LAUNCH_PLAN,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "require_live_matrix": bool(require_live_matrix),
        "route_decision": route,
        "matrix_summary": dict(matrix.get("summary") or {}),
        "validations": validations,
        "summary": {
            "canary_dispatch_manifest_ready": manifest_ready,
            "route_decision": route.get("decision"),
            "route_reason": route.get("reason"),
            "real_training_matrix_passed": bool(matrix.get("real_training_matrix_passed", False)),
            "real_training_matrix_probe_ready": bool(matrix.get("real_training_matrix_probe_ready", False)),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "kahan_adamw8bit_runtime_dispatch_disabled_pending_review",
                "kahan_adamw8bit_runtime_dispatch_adapter_not_connected",
                "kahan_adamw8bit_end_to_end_training_shadow_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add KahanAdamW8bit report-only runtime dispatch adapter shadow"
            if manifest_ready
            else "fix KahanAdamW8bit canary dispatch manifest blockers"
        ),
        "notes": [
            "This manifest is post-matrix evidence only and never dispatches native optimizer updates.",
            "It records the launch plan that a future reviewed canary route would use.",
            "Real dispatch remains blocked until runtime adapter shadow and end-to-end training shadow evidence exist.",
        ],
    }


def _route_decision(
    matrix: Mapping[str, Any],
    mode: str,
    *,
    require_live_matrix: bool,
) -> dict[str, Any]:
    matrix_gate_ready = bool(matrix.get("real_training_matrix_gate_ready", False))
    matrix_probe_ready = bool(matrix.get("real_training_matrix_probe_ready", False))
    matrix_passed = bool(matrix.get("real_training_matrix_passed", False))
    matrix_ready = matrix_gate_ready and (matrix_passed if require_live_matrix else matrix_probe_ready)
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif not matrix_gate_ready:
        decision = "fallback"
        reason = "real_training_matrix_gate_missing"
    elif not matrix_ready:
        decision = "blocked_before_canary_dispatch_manifest"
        reason = "real_training_matrix_live_evidence_missing"
    elif mode == "observe":
        decision = "would_select_native_observe_but_dispatch_disabled"
        reason = "canary_dispatch_manifest_review_required"
    elif mode in {"canary", "auto"}:
        decision = "canary_dispatch_manifest_ready_but_disabled"
        reason = "canary_dispatch_manifest_review_required"
    else:
        decision = "fallback"
        reason = "unknown_native_training_mode"
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "feature": "kahan_adamw8bit_native_optimizer",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "request_supported": matrix_gate_ready,
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "canary_dispatch_armed": False,
        "launch_plan": LAUNCH_PLAN,
        "evidence": {
            "real_training_matrix_gate_ready": matrix_gate_ready,
            "real_training_matrix_probe_ready": matrix_probe_ready,
            "real_training_matrix_passed": matrix_passed,
            "require_live_matrix": bool(require_live_matrix),
        },
        "missing_before_dispatch": [
            "manual_promotion_review",
            "runtime_dispatch_adapter_shadow",
            "end_to_end_training_shadow",
        ],
    }


def _validations(
    matrix: Mapping[str, Any],
    route: Mapping[str, Any],
    mode: str,
    *,
    require_live_matrix: bool,
) -> list[dict[str, Any]]:
    matrix_evidence_ready = bool(matrix.get("real_training_matrix_passed", False))
    if not require_live_matrix:
        matrix_evidence_ready = bool(matrix.get("real_training_matrix_probe_ready", False))
    return [
        _validation(
            "p8y_real_training_matrix_gate_ready",
            bool(matrix.get("real_training_matrix_gate_ready", False)),
            "kahan_adamw8bit_real_training_matrix_gate_missing",
        ),
        _validation(
            "real_training_matrix_evidence_ready",
            matrix_evidence_ready,
            "kahan_adamw8bit_real_training_matrix_live_evidence_missing",
        ),
        _validation(
            "canary_dispatch_manifest_route",
            mode != "off"
            and route.get("decision")
            in {
                "would_select_native_observe_but_dispatch_disabled",
                "canary_dispatch_manifest_ready_but_disabled",
            },
            "kahan_adamw8bit_canary_dispatch_manifest_route_missing",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(route.get("runtime_dispatch_ready", True))
            and not bool(route.get("native_dispatch_allowed", True))
            and not bool(route.get("canary_dispatch_armed", True)),
            "kahan_adamw8bit_canary_dispatch_manifest_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(matrix.get("training_path_enabled", True))
            and not bool(route.get("training_path_enabled", True)),
            "kahan_adamw8bit_canary_dispatch_manifest_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _normalize_mode(value: str) -> str:
    normalized = str(value or "canary").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "canary"


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_kahan_adamw8bit_canary_dispatch_manifest_scorecard"]
