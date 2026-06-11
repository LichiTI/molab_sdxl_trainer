"""Default-off rollout policy for quantized simple optimizer canaries."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_simple_optimizer_quantized_product_state_sync_scorecard import (
    build_simple_optimizer_quantized_product_state_sync_scorecard,
)


POLICY_KIND = "simple_formula_quantized_rollout_policy_v0"
FALLBACK_BACKEND = "existing_pytorch_optimizer"


def build_simple_optimizer_quantized_rollout_policy_scorecard(
    *,
    product_state_sync_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build report-only rollout policy without enabling quantized dispatch."""

    state_sync = _as_dict(product_state_sync_report or build_simple_optimizer_quantized_product_state_sync_scorecard())
    policy = _policy(state_sync)
    validations = _validations(state_sync, policy)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_simple_optimizer_quantized_rollout_policy_scorecard_v0",
        "gate": "simple_formula_quantized_rollout_policy",
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
        "policy_kind": POLICY_KIND,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "policy": policy,
        "product_state_sync_summary": dict(_as_dict(state_sync.get("summary"))),
        "validations": validations,
        "summary": {
            "canary_rollout_policy_ready": ready,
            "manual_review_required": True,
            "canary_auto_enabled": False,
            "canary_enabled_by_default": False,
            "explicit_opt_in_required": True,
            "max_canary_fraction_default": 0.0,
            "optimizer_count": len(_optimizer_types(state_sync)),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "fallback_backend_authoritative": True,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "simple_quantized_runtime_dispatch_disabled_pending_review",
                "simple_quantized_real_dispatch_wiring_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit owner review before wiring quantized simple canary dispatch"
            if ready
            else "fix quantized simple rollout policy blockers"
        ),
        "notes": [
            "This policy is default-off and does not enable runtime dispatch.",
            "Existing PyTorch optimizer state remains the fallback authority.",
            "Canary and auto modes stay blocked until manual review approves wiring.",
        ],
    }


def _policy(state_sync: Mapping[str, Any]) -> dict[str, Any]:
    optimizer_types = _optimizer_types(state_sync)
    return {
        "schema_version": 1,
        "policy_kind": POLICY_KIND,
        "optimizer_types": optimizer_types,
        "canary_enabled_by_default": False,
        "canary_auto_enabled": False,
        "explicit_opt_in_required": True,
        "manual_review_required": True,
        "max_canary_fraction_default": 0.0,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "required_preflight_gates": [
            "simple_formula_quantized_training_loop_canary",
            "simple_formula_quantized_e2e_no_regression",
            "simple_formula_quantized_product_optimizer_state_sync",
            "fallback_backend_authoritative",
            "runtime_dispatch_disabled",
            "default_behavior_unchanged",
            "manual_review_approval_recorded",
        ],
        "rollback_policy": {
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_state_sync_failure": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_checkpoint_adapter_failure": True,
            "rollback_on_dispatch_route_mismatch": True,
        },
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def _validations(state_sync: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _validation(
            "product_optimizer_state_sync_ready",
            state_sync.get("product_optimizer_state_sync_ready") is True,
            "simple_quantized_product_optimizer_state_sync_missing",
        ),
        _validation(
            "policy_default_off",
            policy.get("canary_enabled_by_default") is False
            and policy.get("canary_auto_enabled") is False
            and float(policy.get("max_canary_fraction_default", 1.0)) == 0.0,
            "simple_quantized_policy_not_default_off",
        ),
        _validation(
            "explicit_opt_in_and_manual_review_required",
            policy.get("explicit_opt_in_required") is True and policy.get("manual_review_required") is True,
            "simple_quantized_policy_missing_manual_review",
        ),
        _validation(
            "fallback_rollback_ready",
            _as_dict(policy.get("rollback_policy")).get("fallback_authoritative") is True
            and _as_dict(policy.get("rollback_policy")).get("rollback_on_nonfinite") is True
            and _as_dict(policy.get("rollback_policy")).get("rollback_on_state_sync_failure") is True,
            "simple_quantized_policy_missing_rollback",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            policy.get("runtime_dispatch_ready") is False
            and policy.get("native_dispatch_allowed") is False
            and policy.get("training_path_enabled") is False
            and state_sync.get("runtime_dispatch_ready") is False
            and state_sync.get("native_dispatch_allowed") is False
            and state_sync.get("training_path_enabled") is False,
            "simple_quantized_policy_enabled_dispatch",
        ),
        _validation(
            "request_schema_ui_unchanged",
            state_sync.get("request_fields_emitted") is False
            and state_sync.get("schema_exposure_allowed") is False
            and state_sync.get("ui_exposure_allowed") is False,
            "simple_quantized_policy_changed_request_schema_ui",
        ),
    ]


def _optimizer_types(report: Mapping[str, Any]) -> list[str]:
    return [
        str(row.get("optimizer_type"))
        for row in report.get("rows", [])
        if isinstance(row, Mapping) and row.get("product_optimizer_state_sync_ready") is True
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


__all__ = ["FALLBACK_BACKEND", "POLICY_KIND", "build_simple_optimizer_quantized_rollout_policy_scorecard"]
