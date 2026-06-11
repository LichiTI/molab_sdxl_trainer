"""Default-off canary rollout policy for built-in adaptive-LR optimizers."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_adaptive_lr_e2e_shadow_matrix_scorecard import (
    build_adaptive_lr_e2e_shadow_matrix_scorecard,
)


POLICY_KIND = "adaptive_lr_canary_rollout_policy_v0"
FALLBACK_BACKEND = "python_adaptive_lr_optimizer"
OPTIMIZER_FAMILY = "built_in_adaptive_lr"


def build_adaptive_lr_canary_rollout_policy_scorecard(
    *,
    shadow_matrix_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report-only rollout policy without enabling native dispatch."""

    shadow = _as_dict(shadow_matrix_report or build_adaptive_lr_e2e_shadow_matrix_scorecard())
    rows = [_policy_row(row) for row in shadow.get("rows", []) if isinstance(row, Mapping)]
    policy = _policy(rows)
    validations = _validations(shadow, rows, policy)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adaptive_lr_canary_rollout_policy_scorecard_v0",
        "gate": "adaptive_lr_canary_rollout_policy",
        "ok": ready,
        "promotion_ready": False,
        "canary_rollout_policy_ready": ready,
        "manual_review_required": True,
        "canary_auto_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "policy_kind": POLICY_KIND,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "optimizer_family": OPTIMIZER_FAMILY,
        "policy": policy,
        "rows": rows,
        "shadow_matrix_summary": dict(_as_dict(shadow.get("summary"))),
        "validations": validations,
        "summary": {
            "target_count": len(rows),
            "canary_rollout_policy_ready": ready,
            "canary_rollout_policy_ready_count": sum(
                1 for row in rows if row.get("canary_rollout_policy_ready") is True
            ),
            "manual_review_required": True,
            "canary_auto_enabled": False,
            "canary_enabled_by_default": False,
            "explicit_opt_in_required": True,
            "max_canary_fraction_default": 0.0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
            "fallback_backend_authoritative": True,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adaptive_lr_runtime_dispatch_disabled_pending_review",
                "adaptive_lr_real_dispatch_wiring_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit dispatch integration review before wiring adaptive-LR canary dispatch"
            if ready
            else "fix adaptive-LR canary rollout policy blockers"
        ),
        "notes": [
            "This policy is default-off and does not enable runtime dispatch.",
            "Python adaptive-LR optimizers remain authoritative until explicit review.",
            "Actual product dispatch wiring remains a separate roadmap gate.",
        ],
    }


def _policy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    optimizer_type = str(row.get("optimizer_type") or "")
    ready = row.get("e2e_shadow_matrix_ready") is True
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "family": str(row.get("family") or OPTIMIZER_FAMILY),
        "canary_rollout_policy_ready": ready,
        "e2e_shadow_matrix_ready": ready,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "manual_review_required": True,
        "canary_auto_enabled": False,
        "canary_enabled_by_default": False,
        "explicit_opt_in_required": True,
        "max_canary_fraction_default": 0.0,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
        "next_gate": "adaptive_lr_dispatch_integration_review",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_adaptive_lr_e2e_shadow_matrix_missing"],
    }


def _policy(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ready_names = [
        str(row.get("optimizer_type"))
        for row in rows
        if row.get("canary_rollout_policy_ready") is True
    ]
    return {
        "schema_version": 1,
        "policy_kind": POLICY_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "optimizer_types": ready_names,
        "canary_enabled_by_default": False,
        "canary_auto_enabled": False,
        "explicit_opt_in_required": True,
        "manual_review_required": True,
        "max_canary_fraction_default": 0.0,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "required_preflight_gates": [
            "adaptive_lr_e2e_shadow_matrix",
            "fallback_backend_authoritative",
            "native_shadow_does_not_mutate_authority",
            "runtime_dispatch_disabled",
            "default_behavior_unchanged",
            "manual_review_approval_recorded",
        ],
        "rollback_policy": {
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_state_machine_guard_failure": True,
            "rollback_on_dispatch_route_mismatch": True,
        },
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_native_ready": False,
    }


def _validations(
    shadow: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rollback = _as_dict(policy.get("rollback_policy"))
    return [
        _validation(
            "e2e_shadow_matrix_ready",
            shadow.get("e2e_shadow_matrix_ready") is True
            and all(row.get("e2e_shadow_matrix_ready") is True for row in rows),
            "adaptive_lr_e2e_shadow_matrix_missing",
        ),
        _validation(
            "policy_default_off",
            policy.get("canary_enabled_by_default") is False
            and policy.get("canary_auto_enabled") is False
            and float(policy.get("max_canary_fraction_default", 1.0)) == 0.0,
            "adaptive_lr_policy_not_default_off",
        ),
        _validation(
            "explicit_opt_in_and_manual_review_required",
            policy.get("explicit_opt_in_required") is True and policy.get("manual_review_required") is True,
            "adaptive_lr_policy_missing_manual_review",
        ),
        _validation(
            "fallback_rollback_ready",
            rollback.get("fallback_authoritative") is True
            and rollback.get("rollback_on_nonfinite") is True
            and rollback.get("rollback_on_parity_failure") is True
            and rollback.get("rollback_on_state_machine_guard_failure") is True,
            "adaptive_lr_policy_missing_rollback",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            policy.get("runtime_dispatch_ready") is False
            and policy.get("native_dispatch_allowed") is False
            and policy.get("training_path_enabled") is False
            and shadow.get("runtime_dispatch_ready") is False
            and shadow.get("native_dispatch_allowed") is False
            and shadow.get("training_path_enabled") is False,
            "adaptive_lr_policy_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            policy.get("default_behavior_changed") is False
            and shadow.get("default_behavior_changed") is False,
            "adaptive_lr_policy_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["POLICY_KIND", "build_adaptive_lr_canary_rollout_policy_scorecard"]
