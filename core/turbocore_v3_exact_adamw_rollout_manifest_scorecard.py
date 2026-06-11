"""Default-off rollout manifest for V3 exact AdamW native canary."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_v3_exact_adamw_real_canary_scorecard import (
    build_v3_exact_adamw_real_canary_scorecard,
)


OPTIMIZER_KIND = "exact_adamw"
OPTIMIZER_FAMILY = "adamw_exact"
NATIVE_BACKEND = "rust_cuda_adamw_v0"


def build_v3_exact_adamw_rollout_manifest_scorecard(
    *,
    p0_scorecard: Mapping[str, Any] | None = None,
    native_training_mode: str = "canary",
    run_live_training: bool = True,
) -> dict[str, Any]:
    """Build a rollout manifest without changing default training behavior."""

    p0 = dict(
        p0_scorecard
        or build_v3_exact_adamw_real_canary_scorecard(run_live_training=run_live_training)
    )
    mode = _normalize_mode(native_training_mode)
    live = _as_dict(p0.get("live_training_probe"))
    summary = _as_dict(p0.get("summary"))
    p0_ready = (
        bool(p0.get("milestone_completed", False))
        and bool(summary.get("default_off", False))
        and bool(summary.get("explicit_request_allowed", False))
        and bool(summary.get("live_training_native_step", False))
        and bool(live.get("pytorch_optimizer_state_synced", False))
        and live.get("owner_backend") == NATIVE_BACKEND
    )
    route = _route_decision(mode, p0_ready)
    rollback = _rollback_policy(p0_ready)
    manifest_ready = (
        p0_ready
        and bool(rollback.get("fallback_authoritative", False))
        and bool(rollback.get("disable_for_run_on_native_error", False))
        and route["decision"] in {"explicit_canary_ready", "observe_ready_but_default_off"}
        and not bool(route.get("default_training_path_enabled", True))
    )
    blockers = _blockers(p0_ready, mode, route, rollback, manifest_ready)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v3_exact_adamw_rollout_manifest_scorecard_v0",
        "gate": "v3_exact_adamw_rollout_manifest",
        "ok": bool(p0.get("ok", False)) and mode in {"off", "observe", "canary", "auto"},
        "milestone_completed": manifest_ready,
        "rollout_manifest_ready": manifest_ready,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_backend": NATIVE_BACKEND,
        "native_training_mode": mode,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_dispatch_allowed": False,
        "explicit_canary_allowed": route["decision"] == "explicit_canary_ready",
        "auto_rollout_allowed": False,
        "requires_explicit_opt_in": True,
        "p0_summary": {
            "milestone_completed": bool(p0.get("milestone_completed", False)),
            "default_off": bool(summary.get("default_off", False)),
            "explicit_request_allowed": bool(summary.get("explicit_request_allowed", False)),
            "live_training_native_step": bool(summary.get("live_training_native_step", False)),
            "pytorch_fallback_preserved": bool(summary.get("pytorch_fallback_preserved", False)),
            "owner_backend": str(live.get("owner_backend") or ""),
        },
        "route_decision": route,
        "rollback_policy": rollback,
        "manifest_summary": {
            "rollout_manifest_ready": manifest_ready,
            "native_training_mode": mode,
            "default_training_path_enabled": False,
            "explicit_canary_allowed": route["decision"] == "explicit_canary_ready",
            "auto_rollout_allowed": False,
            "native_backend": NATIVE_BACKEND,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run V3-P2 exact AdamW short real-training matrix"
            if manifest_ready
            else "complete V3-P0 exact AdamW real canary before rollout manifest"
        ),
        "notes": [
            "This is a manifest gate, not a dispatcher.",
            "The manifest records the exact opt-in fields required by the runtime/request boundary.",
            "Auto rollout remains blocked until a short real-training matrix and recovery audit pass.",
        ],
    }


def _route_decision(mode: str, p0_ready: bool) -> dict[str, Any]:
    if mode == "off":
        decision = "off"
        reason = "native_training_mode_off"
    elif not p0_ready:
        decision = "blocked"
        reason = "v3_p0_real_canary_missing"
    elif mode == "observe":
        decision = "observe_ready_but_default_off"
        reason = "explicit_canary_manifest_ready"
    elif mode == "canary":
        decision = "explicit_canary_ready"
        reason = "explicit_canary_requires_dev_opt_in"
    else:
        decision = "auto_blocked_until_p2_p5"
        reason = "short_matrix_and_promotion_review_missing"
    return {
        "schema_version": 1,
        "feature": "v3_exact_adamw_native_update",
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_backend": NATIVE_BACKEND,
        "native_training_mode": mode,
        "decision": decision,
        "reason": reason,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_dispatch_allowed": False,
        "explicit_canary_allowed": decision == "explicit_canary_ready",
        "auto_rollout_allowed": False,
        "request_fields": {
            "turbocore_native_update_mode": "native_experimental",
            "turbocore_native_update_dispatch_enabled": True,
            "turbocore_native_update_training_path_enabled": True,
            "turbocore_native_update_require_native_cuda": True,
            "optimizer_kind": OPTIMIZER_KIND,
        },
        "missing_before_auto": [
            "v3_p2_short_real_training_matrix",
            "runtime_recovery_hardening",
            "promotion_review_gate",
        ],
    }


def _rollback_policy(p0_ready: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy": "v3_exact_adamw_canary_rollback_policy_v0",
        "fallback_authoritative": True,
        "fallback_backend": "pytorch_adamw",
        "p0_real_canary_ready": bool(p0_ready),
        "disable_for_run_on_native_error": True,
        "disable_for_run_on_state_sync_failure": True,
        "disable_for_run_on_non_finite": True,
        "rollback_on_resume_mismatch": True,
        "rollback_on_optimizer_kind_mismatch": True,
        "rollback_on_native_backend_mismatch": True,
        "default_training_path_enabled": False,
    }


def _blockers(
    p0_ready: bool,
    mode: str,
    route: Mapping[str, Any],
    rollback: Mapping[str, Any],
    manifest_ready: bool,
) -> list[str]:
    blockers: list[str] = []
    if mode == "off":
        blockers.append("v3_exact_adamw_native_training_mode_off")
    if not p0_ready:
        blockers.append("v3_p0_real_canary_missing")
    if route.get("decision") == "auto_blocked_until_p2_p5":
        blockers.append("v3_exact_adamw_auto_rollout_blocked_until_short_matrix")
    if bool(route.get("default_training_path_enabled", True)):
        blockers.append("v3_exact_adamw_default_training_path_enabled_unexpectedly")
    if not bool(rollback.get("fallback_authoritative", False)):
        blockers.append("v3_exact_adamw_fallback_policy_missing")
    if not manifest_ready and p0_ready and mode in {"observe", "canary"}:
        blockers.append("v3_exact_adamw_rollout_manifest_not_ready")
    return _dedupe(blockers)


def _normalize_mode(value: str) -> str:
    normalized = str(value or "canary").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "canary"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_v3_exact_adamw_rollout_manifest_scorecard"]
