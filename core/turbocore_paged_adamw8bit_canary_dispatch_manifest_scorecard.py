"""Report-only canary dispatch manifest for PagedAdamW8bit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_paged_adamw8bit_dispatch_selector_dry_run_scorecard import (
    build_paged_adamw8bit_dispatch_selector_dry_run_scorecard,
)
from core.turbocore_paged_adamw8bit_real_training_matrix_scorecard import (
    build_paged_adamw8bit_real_training_matrix_scorecard,
)


OPTIMIZER_KIND = "paged_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_paged"
MANIFEST_KIND = "paged_adamw8bit_canary_dispatch_manifest_v0"
LAUNCH_PLAN = "paged_adamw8bit_training_tensor_binding_launch_plan_v0"


def build_paged_adamw8bit_canary_dispatch_manifest_scorecard(
    *,
    selector_report: Mapping[str, Any] | None = None,
    matrix_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
    run_live_probe: bool = True,
    require_live_matrix: bool = True,
) -> dict[str, Any]:
    """Build a post-matrix dispatch manifest without enabling native dispatch."""

    mode = _normalize_mode(native_training_mode)
    selector = dict(
        selector_report
        or build_paged_adamw8bit_dispatch_selector_dry_run_scorecard(native_training_mode=mode)
    )
    matrix = dict(
        matrix_report
        or build_paged_adamw8bit_real_training_matrix_scorecard(
            selector_report=selector,
            run_live_probe=run_live_probe,
        )
    )
    route = _route_decision(
        selector,
        matrix,
        mode,
        require_live_matrix=bool(require_live_matrix),
    )
    validations = _validations(
        selector,
        matrix,
        route,
        mode,
        require_live_matrix=bool(require_live_matrix),
    )
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    manifest_ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_canary_dispatch_manifest_scorecard_v0",
        "gate": "paged_adamw8bit_canary_dispatch_manifest",
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
        "selector_summary": dict(selector.get("summary") or {}),
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
                "paged_adamw8bit_runtime_dispatch_disabled_pending_review",
                "paged_adamw8bit_runtime_dispatch_adapter_not_connected",
                "paged_adamw8bit_end_to_end_training_shadow_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add PagedAdamW8bit report-only runtime dispatch adapter shadow"
            if manifest_ready
            else "fix PagedAdamW8bit canary dispatch manifest blockers"
        ),
        "notes": [
            "This manifest is post-matrix evidence only and never dispatches native optimizer updates.",
            "It records the launch plan that a future reviewed canary route would use.",
            "Real dispatch remains blocked until a runtime adapter shadow and end-to-end training evidence exist.",
        ],
    }


def _route_decision(
    selector: Mapping[str, Any],
    matrix: Mapping[str, Any],
    mode: str,
    *,
    require_live_matrix: bool,
) -> dict[str, Any]:
    selector_ready = bool(selector.get("dispatch_selector_dry_run_ready", False))
    matrix_gate_ready = bool(matrix.get("real_training_matrix_gate_ready", False))
    matrix_probe_ready = bool(matrix.get("real_training_matrix_probe_ready", False))
    matrix_passed = bool(matrix.get("real_training_matrix_passed", False))
    matrix_ready = matrix_gate_ready and (matrix_passed if require_live_matrix else matrix_probe_ready)
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif not selector_ready:
        decision = "fallback"
        reason = "dispatch_selector_dry_run_missing"
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
        "feature": "paged_adamw8bit_native_optimizer",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "request_supported": bool(
            _selector_route(selector).get("request_supported", selector_ready)
        ),
        "training_path_enabled": False,
        "native_dispatch_allowed": False,
        "runtime_dispatch_ready": False,
        "canary_dispatch_armed": False,
        "launch_plan": LAUNCH_PLAN,
        "evidence": {
            "dispatch_selector_dry_run_ready": selector_ready,
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
    selector: Mapping[str, Any],
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
            "p8m_dispatch_selector_dry_run_ready",
            bool(selector.get("dispatch_selector_dry_run_ready", False)),
            "paged_adamw8bit_dispatch_selector_dry_run_missing",
        ),
        _validation(
            "p8n_real_training_matrix_gate_ready",
            bool(matrix.get("real_training_matrix_gate_ready", False)),
            "paged_adamw8bit_real_training_matrix_gate_missing",
        ),
        _validation(
            "real_training_matrix_evidence_ready",
            matrix_evidence_ready,
            "paged_adamw8bit_real_training_matrix_live_evidence_missing",
        ),
        _validation(
            "canary_dispatch_manifest_route",
            mode != "off"
            and route.get("decision")
            in {
                "would_select_native_observe_but_dispatch_disabled",
                "canary_dispatch_manifest_ready_but_disabled",
            },
            "paged_adamw8bit_canary_dispatch_manifest_route_missing",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(route.get("runtime_dispatch_ready", True))
            and not bool(route.get("native_dispatch_allowed", True))
            and not bool(route.get("canary_dispatch_armed", True)),
            "paged_adamw8bit_canary_dispatch_manifest_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(selector.get("training_path_enabled", True))
            and not bool(matrix.get("training_path_enabled", True))
            and not bool(route.get("training_path_enabled", True)),
            "paged_adamw8bit_canary_dispatch_manifest_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _selector_route(selector: Mapping[str, Any]) -> Mapping[str, Any]:
    route = selector.get("route_decision")
    return route if isinstance(route, Mapping) else {}


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


__all__ = ["build_paged_adamw8bit_canary_dispatch_manifest_scorecard"]
