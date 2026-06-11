"""Report-only observe-mode manifest for CUDA graph route candidates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_cuda_graph_route_scorecard import build_cuda_graph_route_scorecard


MANIFEST_KIND = "cuda_graph_observe_manifest_v0"
FEATURE = "cuda_graph_static_shape_route"


def build_cuda_graph_observe_manifest_scorecard(
    *,
    route_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
    run_live_probe: bool = True,
) -> dict[str, Any]:
    """Build a manifest that records CUDA graph candidate routing only."""

    mode = _normalize_mode(native_training_mode)
    route = dict(route_report or build_cuda_graph_route_scorecard(run_live_probe=run_live_probe))
    decision = _route_decision(route, mode)
    manifest = _manifest(route, decision, mode)
    validations = _validations(route, decision, manifest, mode)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_cuda_graph_observe_manifest_scorecard_v0",
        "gate": "p6f_cuda_graph_observe_manifest",
        "ok": ready,
        "promotion_ready": ready,
        "observe_manifest_ready": ready,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "experimental_only": True,
        "manifest_kind": MANIFEST_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "route_decision": decision,
        "manifest": manifest,
        "route_summary": dict(route.get("summary") or {}),
        "validations": validations,
        "summary": {
            "observe_manifest_ready": ready,
            "decision": decision.get("decision"),
            "reason": decision.get("reason"),
            "candidate_recorded": bool(manifest.get("candidate_recorded", False)),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add CUDA graph static-shape training integration review gate"
            if ready
            else "fix CUDA graph observe manifest blockers"
        ),
        "notes": [
            "Observe mode records a CUDA graph candidate route but never dispatches it.",
            "Canary and auto modes remain blocked until an explicit integration review.",
            "The manifest is runtime/request shaped so future integration can reuse the same fields.",
        ],
    }


def _route_decision(route: Mapping[str, Any], mode: str) -> dict[str, Any]:
    route_ready = bool(route.get("promotion_ready", False))
    summary = route.get("summary") if isinstance(route.get("summary"), Mapping) else {}
    static_ready = bool(summary.get("static_contract_ready", False))
    live_ready = bool(summary.get("live_capture_ready", False))
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
        candidate = False
    elif not route_ready or not static_ready or not live_ready:
        decision = "fallback"
        reason = "cuda_graph_route_not_ready"
        candidate = False
    elif mode == "observe":
        decision = "would_select_cuda_graph_observe_but_dispatch_disabled"
        reason = "observe_mode_records_candidate_only"
        candidate = True
    else:
        decision = "blocked_before_cuda_graph_canary"
        reason = "cuda_graph_training_integration_review_required"
        candidate = True
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "request_supported": bool(route_ready and static_ready and live_ready),
        "candidate_recorded": candidate,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "missing_before_dispatch": [
            "manual_integration_review",
            "training_loop_static_shape_probe",
            "rollback_manifest",
        ],
    }


def _manifest(route: Mapping[str, Any], decision: Mapping[str, Any], mode: str) -> dict[str, Any]:
    policy = route.get("policy") if isinstance(route.get("policy"), Mapping) else {}
    return {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "feature": FEATURE,
        "native_training_mode": mode,
        "candidate_recorded": bool(decision.get("candidate_recorded", False)),
        "shape": list(route.get("shape", []) or []),
        "static_contract": dict(route.get("static_contract") or {}),
        "live_probe_summary": dict(route.get("summary") or {}),
        "allowed_initial_modes": list(policy.get("allowed_initial_modes", []) or []),
        "blocked_modes_until_review": list(policy.get("blocked_modes_until_review", []) or []),
        "runtime_incompatibilities": list(policy.get("runtime_incompatibilities", []) or []),
        "audit_fields": [
            "native_training_mode",
            "model_arch",
            "batch_size",
            "resolution",
            "dtype",
            "fixed_token_counts",
            "route_decision",
            "fallback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(
    route: Mapping[str, Any],
    decision: Mapping[str, Any],
    manifest: Mapping[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p6e_cuda_graph_route_ready",
            bool(route.get("promotion_ready", False)),
            "cuda_graph_route_scorecard_missing",
        ),
        _validation(
            "observe_route_decision_ready",
            mode != "off"
            and decision.get("decision")
            in {
                "would_select_cuda_graph_observe_but_dispatch_disabled",
                "blocked_before_cuda_graph_canary",
            },
            "cuda_graph_observe_route_decision_missing",
        ),
        _validation(
            "candidate_manifest_records_contract",
            bool(manifest.get("static_contract")) and bool(manifest.get("runtime_incompatibilities")),
            "cuda_graph_observe_manifest_contract_missing",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(decision.get("runtime_dispatch_ready", True))
            and not bool(manifest.get("native_dispatch_allowed", True))
            and not bool(manifest.get("training_path_enabled", True)),
            "cuda_graph_observe_manifest_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(route.get("training_path_enabled", True))
            and not bool(route.get("default_behavior_changed", True)),
            "cuda_graph_observe_manifest_changed_default_behavior",
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
    normalized = str(value or "observe").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "observe"


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_cuda_graph_observe_manifest_scorecard"]
