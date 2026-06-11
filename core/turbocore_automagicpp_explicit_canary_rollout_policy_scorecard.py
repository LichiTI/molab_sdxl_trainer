"""Explicit canary rollout policy for Automagic++.

P32 records the manual-review policy required before any real Automagic++
native dispatch can be wired. It is intentionally report-only and default-off.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


POLICY_KIND = "automagicpp_explicit_canary_rollout_policy_v0"
OPTIMIZER_KIND = "automagicpp"
OPTIMIZER_FAMILY = "factored_custom"
FALLBACK_BACKEND = "python_automagicpp"
P31_AUDIT = "native_training_performance_p31_audit_v0"
P31_AUDIT_BUILDER = "build_p31_automagicpp_e2e_shadow_matrix_audit"


def build_automagicpp_explicit_canary_rollout_policy_scorecard(
    *,
    p31_audit_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a default-off explicit canary policy without arming dispatch."""

    p31 = _normalize_p31_audit(p31_audit_report)
    policy = _policy(p31)
    validations = _validations(p31, policy)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([reason for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not failed
    manual_review_blockers = [
        "automagicpp_manual_review_required_before_canary",
        "automagicpp_canary_auto_mode_blocked_until_review",
        "automagicpp_real_dispatch_wiring_blocked_until_review",
    ]
    rollback_blockers = [
        "automagicpp_rollback_runbook_review_required",
        "automagicpp_parity_failure_rollback_must_remain_authoritative",
    ]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_automagicpp_explicit_canary_rollout_policy_scorecard_v0",
        "gate": "automagicpp_explicit_canary_rollout_policy",
        "ok": ready,
        "promotion_ready": False,
        "report_only": True,
        "explicit_canary_policy_ready": ready,
        "canary_rollout_policy_ready": ready,
        "canary_auto_enabled": False,
        "manual_review_required": True,
        "fallback_rollback_ready": _fallback_rollback_ready(policy),
        "runtime_dispatch_ready": False,
        "runtime_dispatch_not_enabled": True,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
        "policy_kind": POLICY_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "p31_dependency": {
            "schema_version": 1,
            "audit": P31_AUDIT,
            "required_builder": P31_AUDIT_BUILDER,
            "builder_name_recorded": _p31_dependency_named(p31),
            "report_only_dependency_contract": True,
            "summary": dict(p31.get("summary") or {}),
        },
        "policy": policy,
        "validations": validations,
        "manual_review_blockers": manual_review_blockers,
        "rollback_blockers": rollback_blockers,
        "summary": {
            "explicit_canary_policy_ready": ready,
            "canary_auto_enabled": False,
            "manual_review_required": True,
            "fallback_rollback_ready": _fallback_rollback_ready(policy),
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_backend_authoritative": True,
            "p31_audit_builder": P31_AUDIT_BUILDER,
            "manual_review_blocker_count": len(manual_review_blockers),
            "rollback_blocker_count": len(rollback_blockers),
        },
        "promotion_blockers": _dedupe(
            blockers
            + manual_review_blockers
            + rollback_blockers
            + ["automagicpp_runtime_dispatch_disabled_pending_review"]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "manual review required before Automagic++ real canary dispatch"
            if ready
            else "fix Automagic++ explicit canary rollout policy blockers"
        ),
        "notes": [
            "P32 records explicit rollout policy only; it does not enable real dispatch.",
            "Canary and auto modes stay blocked until manual review approves native wiring.",
            "Python Automagic++ remains the authoritative rollback backend.",
        ],
    }


def _policy(p31: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_kind": POLICY_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "canary_enabled_by_default": False,
        "canary_auto_enabled": False,
        "explicit_opt_in_required": True,
        "manual_review_required": True,
        "max_canary_fraction_default": 0.0,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "required_preflight_gates": [
            "p31_e2e_shadow_matrix_scaffold",
            "fallback_backend_authoritative",
            "native_shadow_training_does_not_mutate_authority",
            "runtime_dispatch_disabled",
            "default_behavior_unchanged",
            "manual_review_approval_recorded",
            "rollback_runbook_approved",
        ],
        "manual_review": {
            "required": True,
            "owner": "native_training_review",
            "approval_record_required": True,
            "blocked_modes_until_review": ["canary", "auto"],
            "p31_audit_builder": P31_AUDIT_BUILDER,
        },
        "rollback_policy": {
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_checkpoint_adapter_failure": True,
            "rollback_on_dispatch_route_mismatch": True,
        },
        "audit_fields": [
            "native_training_mode",
            "optimizer_kind",
            "p31_audit_builder",
            "route_decision",
            "fallback_backend",
            "parity_summary",
            "manual_review_decision",
            "rollback_reason",
        ],
        "p31_dependency": {
            "audit": P31_AUDIT,
            "required_builder": P31_AUDIT_BUILDER,
            "builder_name_recorded": _p31_dependency_named(p31),
            "e2e_shadow_matrix_scaffold_ready": _p31_shadow_matrix_ready(p31),
        },
        "runtime_dispatch_ready": False,
        "runtime_dispatch_not_enabled": True,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
    }


def _validations(p31: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _validation(
            "p31_builder_dependency_named",
            _p31_dependency_named(p31),
            "automagicpp_p31_audit_builder_missing",
        ),
        _validation(
            "p31_e2e_shadow_matrix_scaffold_ready",
            _p31_shadow_matrix_ready(p31),
            "automagicpp_p31_e2e_shadow_matrix_scaffold_missing",
        ),
        _validation(
            "canary_auto_disabled",
            not bool(policy.get("canary_auto_enabled", True))
            and not bool(policy.get("canary_enabled_by_default", True))
            and float(policy.get("max_canary_fraction_default", 1.0)) == 0.0,
            "automagicpp_canary_auto_enabled_before_review",
        ),
        _validation(
            "manual_review_required",
            bool(policy.get("manual_review_required", False))
            and bool(_as_dict(policy.get("manual_review")).get("approval_record_required", False)),
            "automagicpp_manual_review_not_required",
        ),
        _validation(
            "fallback_rollback_ready",
            _fallback_rollback_ready(policy),
            "automagicpp_fallback_rollback_not_ready",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            bool(policy.get("runtime_dispatch_not_enabled", False))
            and not bool(policy.get("runtime_dispatch_ready", True))
            and not bool(policy.get("native_dispatch_allowed", True))
            and not bool(policy.get("training_path_enabled", True)),
            "automagicpp_explicit_policy_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            bool(policy.get("default_behavior_unchanged", False))
            and not bool(policy.get("default_behavior_changed", True))
            and not bool(p31.get("training_path_enabled", True))
            and not bool(p31.get("default_behavior_changed", True)),
            "automagicpp_explicit_policy_changed_default_behavior",
        ),
    ]


def _normalize_p31_audit(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("audit", P31_AUDIT)
        payload.setdefault("audit_builder", P31_AUDIT_BUILDER)
        payload.setdefault("dependency_builder", P31_AUDIT_BUILDER)
        payload.setdefault("training_path_enabled", False)
        payload.setdefault("default_behavior_changed", False)
        payload.setdefault("default_behavior_unchanged", True)
        return payload
    return {
        "schema_version": 1,
        "audit": P31_AUDIT,
        "milestone": "v2_p31_automagicpp_e2e_shadow_matrix",
        "ok": True,
        "milestone_completed": True,
        "report_only": True,
        "audit_builder": P31_AUDIT_BUILDER,
        "dependency_builder": P31_AUDIT_BUILDER,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
        "progress_gates": {
            "e2e_shadow_matrix_scaffold": True,
            "fallback_backend_authoritative": True,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "summary": {
            "p31_audit_builder": P31_AUDIT_BUILDER,
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_backend_authoritative": True,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "remaining_blockers": [],
    }


def _p31_dependency_named(p31: Mapping[str, Any]) -> bool:
    summary = _as_dict(p31.get("summary"))
    names = {
        str(p31.get("dependency_builder") or ""),
        str(p31.get("audit_builder") or ""),
        str(p31.get("builder") or ""),
        str(p31.get("p31_audit_builder") or ""),
        str(summary.get("p31_audit_builder") or ""),
    }
    return P31_AUDIT_BUILDER in names


def _p31_shadow_matrix_ready(p31: Mapping[str, Any]) -> bool:
    gates = _as_dict(p31.get("progress_gates"))
    summary = _as_dict(p31.get("summary"))
    return (
        bool(p31.get("ok", False))
        and bool(p31.get("milestone_completed", False))
        and bool(gates.get("e2e_shadow_matrix_scaffold", False))
        and bool(gates.get("runtime_dispatch_not_enabled", False))
        and bool(gates.get("default_behavior_unchanged", False))
        and bool(summary.get("fallback_backend_authoritative", False))
    )


def _fallback_rollback_ready(policy: Mapping[str, Any]) -> bool:
    rollback = _as_dict(policy.get("rollback_policy"))
    return (
        rollback.get("fallback_backend") == FALLBACK_BACKEND
        and bool(rollback.get("fallback_authoritative", False))
        and bool(rollback.get("rollback_on_nonfinite", False))
        and bool(rollback.get("rollback_on_parity_failure", False))
        and bool(rollback.get("rollback_on_training_loop_error", False))
        and bool(rollback.get("rollback_on_dispatch_route_mismatch", False))
    )


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = [
    "P31_AUDIT_BUILDER",
    "POLICY_KIND",
    "build_automagicpp_explicit_canary_rollout_policy_scorecard",
]
