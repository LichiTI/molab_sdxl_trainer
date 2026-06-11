"""Default-off rollout policy for built-in simple schedule-free variants."""

from __future__ import annotations

from typing import Any, Mapping

from core.configs import OptimizerType
from core.turbocore_simple_optimizer_variant_native_canary_scorecard import (
    build_simple_optimizer_variant_native_canary_scorecard,
)


POLICY_KIND = "simple_formula_schedulefree_rollout_policy_v0"
FALLBACK_BACKEND = "existing_schedulefree_optimizer"
TARGET_OPTIMIZERS = (OptimizerType.RADAM_SCHEDULE_FREE.value, OptimizerType.SGD_SCHEDULE_FREE.value)


def build_simple_optimizer_schedulefree_rollout_policy_scorecard(
    *,
    variant_native_canary_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a default-off rollout policy without enabling native dispatch."""

    variant = _as_dict(variant_native_canary_report or build_simple_optimizer_variant_native_canary_scorecard())
    rows = [_schedulefree_row(name, variant) for name in TARGET_OPTIMIZERS]
    policy = _policy(rows)
    validations = _validations(variant, rows, policy)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_schedulefree_rollout_policy_scorecard_v0",
        "gate": "simple_formula_schedulefree_default_off_rollout_policy",
        "ok": ready,
        "promotion_ready": False,
        "canary_rollout_policy_ready": ready,
        "manual_review_required": True,
        "canary_auto_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "product_native_dispatch_ready": False,
        "product_native_ready_count": 0,
        "policy_kind": POLICY_KIND,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "policy": policy,
        "rows": rows,
        "variant_native_canary_summary": _as_dict(variant.get("summary")),
        "validations": validations,
        "summary": {
            "canary_rollout_policy_ready": ready,
            "manual_review_required": True,
            "optimizer_count": len(rows),
            "canary_enabled_by_default": False,
            "explicit_opt_in_required": True,
            "max_canary_fraction_default": 0.0,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "simple_schedulefree_dispatch_integration_review_missing",
                "simple_schedulefree_owner_approval_missing",
                "simple_schedulefree_product_dispatch_not_approved",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "build simple schedule-free dispatch integration review package"
            if ready
            else "fix simple schedule-free rollout policy blockers"
        ),
        "notes": [
            "This policy covers built-in RAdamScheduleFree and SGDScheduleFree variants.",
            "It consumes native canary evidence only and does not enable dispatch.",
            "Request, schema, UI, runtime dispatch, and default training behavior remain unchanged.",
        ],
    }


def _schedulefree_row(optimizer_type: str, variant: Mapping[str, Any]) -> dict[str, Any]:
    source = _variant_row(optimizer_type, variant)
    ready = (
        source.get("variant_status") == "schedule_free_native_canary_ready"
        and source.get("native_canary_ready") is True
        and source.get("runtime_canary_ready") is True
        and source.get("training_path_enabled") is False
        and source.get("native_dispatch_allowed") is False
        and source.get("default_behavior_changed") is False
    )
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "optimizer_family": "simple_formula",
        "variant_kind": "schedule_free_state_machine",
        "rollout_status": "schedule_free_rollout_policy_ready" if ready else "schedule_free_rollout_policy_blocked",
        "native_canary_ready": source.get("native_canary_ready") is True,
        "runtime_canary_ready": source.get("runtime_canary_ready") is True,
        "native_step_count": int(source.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(source.get("native_kernel_launch_count", 0) or 0),
        "product_native_dispatch_ready": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "next_gate": "simple_schedulefree_dispatch_integration_review",
        "blocked_reasons": [] if ready else [f"{optimizer_type}_schedulefree_native_canary_missing"],
    }


def _policy(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_kind": POLICY_KIND,
        "optimizer_family": "simple_formula_schedulefree",
        "optimizer_types": [str(row.get("optimizer_type")) for row in rows],
        "canary_enabled_by_default": False,
        "canary_auto_enabled": False,
        "explicit_opt_in_required": True,
        "manual_review_required": True,
        "max_canary_fraction_default": 0.0,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "required_preflight_gates": [
            "simple_formula_variant_schedule_free_native_canary",
            "fallback_backend_authoritative",
            "native_dispatch_disabled",
            "default_behavior_unchanged",
            "owner_release_hold_recorded",
        ],
        "rollback_policy": {
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_checkpoint_adapter_failure": True,
            "rollback_on_train_eval_mode_mismatch": True,
        },
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def _validations(variant: Mapping[str, Any], rows: list[Mapping[str, Any]], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    rollback = _as_dict(policy.get("rollback_policy"))
    return [
        _validation(
            "variant_native_canary_ready",
            variant.get("variant_schedule_free_native_canary_ready") is True,
            "simple_schedulefree_native_canary_missing",
        ),
        _validation(
            "optimizer_set_complete",
            {str(row.get("optimizer_type")) for row in rows} == set(TARGET_OPTIMIZERS),
            "simple_schedulefree_rollout_optimizer_set_incomplete",
        ),
        _validation(
            "rows_ready",
            all(row.get("rollout_status") == "schedule_free_rollout_policy_ready" for row in rows),
            "simple_schedulefree_rollout_rows_not_ready",
        ),
        _validation(
            "policy_default_off",
            policy.get("canary_enabled_by_default") is False
            and policy.get("canary_auto_enabled") is False
            and float(policy.get("max_canary_fraction_default", 1.0)) == 0.0,
            "simple_schedulefree_rollout_policy_not_default_off",
        ),
        _validation(
            "rollback_manifest_present",
            rollback.get("fallback_authoritative") is True
            and rollback.get("rollback_on_nonfinite") is True
            and rollback.get("rollback_on_train_eval_mode_mismatch") is True,
            "simple_schedulefree_rollout_policy_missing_rollback",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            policy.get("runtime_dispatch_ready") is False
            and policy.get("native_dispatch_allowed") is False
            and policy.get("training_path_enabled") is False
            and variant.get("runtime_dispatch_ready") is False
            and variant.get("native_dispatch_allowed") is False
            and variant.get("training_path_enabled") is False,
            "simple_schedulefree_rollout_policy_enabled_dispatch",
        ),
    ]


def _variant_row(optimizer_type: str, variant: Mapping[str, Any]) -> dict[str, Any]:
    for row in variant.get("rows", []):
        if isinstance(row, Mapping) and row.get("optimizer_type") == optimizer_type:
            return dict(row)
    return {}


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


__all__ = ["POLICY_KIND", "TARGET_OPTIMIZERS", "build_simple_optimizer_schedulefree_rollout_policy_scorecard"]
