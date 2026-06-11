"""Default-off rollout policy for AdamW variant native-canary rows."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_adamw_variant_e2e_shadow_matrix_scorecard import (
    build_adamw_variant_e2e_shadow_matrix_scorecard,
)


POLICY_KIND = "adamw_variant_canary_rollout_policy_v0"
FALLBACK_BACKEND = "existing_variant_optimizer"


def build_adamw_variant_canary_rollout_policy_scorecard(
    *,
    shadow_matrix_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build report-only rollout policy with native dispatch still default-off."""

    shadow = _as_dict(shadow_matrix_report or build_adamw_variant_e2e_shadow_matrix_scorecard())
    policy = _policy(shadow)
    validations = _validations(shadow, policy)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw_variant_canary_rollout_policy_scorecard_v0",
        "gate": "adamw_variant_canary_rollout_policy",
        "ok": ready,
        "promotion_ready": False,
        "canary_rollout_policy_ready": ready,
        "manual_review_required": True,
        "canary_auto_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "policy_kind": POLICY_KIND,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "policy": policy,
        "shadow_matrix_summary": dict(_as_dict(shadow.get("summary"))),
        "validations": validations,
        "summary": {
            "canary_rollout_policy_ready": ready,
            "manual_review_required": True,
            "canary_auto_enabled": False,
            "canary_enabled_by_default": False,
            "explicit_opt_in_required": True,
            "max_canary_fraction_default": 0.0,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "fallback_backend_authoritative": True,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "adamw_variant_runtime_dispatch_disabled_pending_review",
                "adamw_variant_real_dispatch_wiring_missing",
                "owner_release_hold_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit owner review before wiring AdamW variant canary dispatch"
            if ready
            else "fix AdamW variant canary rollout policy blockers"
        ),
        "notes": [
            "This policy is default-off and does not enable runtime dispatch.",
            "AdamWScheduleFree is included after its native TrainingLoop canary passes.",
            "Actual dispatch wiring remains a separate decision.",
        ],
    }


def _policy(shadow: Mapping[str, Any]) -> dict[str, Any]:
    ready_names = [
        str(case.get("optimizer_type"))
        for case in shadow.get("matrix_cases", [])
        if isinstance(case, Mapping) and case.get("shadow_matrix_case_ready") is True
    ]
    return {
        "schema_version": 1,
        "policy_kind": POLICY_KIND,
        "optimizer_types": ready_names,
        "canary_enabled_by_default": False,
        "canary_auto_enabled": False,
        "explicit_opt_in_required": True,
        "manual_review_required": True,
        "max_canary_fraction_default": 0.0,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "required_preflight_gates": [
            "adamw_variant_e2e_shadow_matrix",
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
            "rollback_on_checkpoint_adapter_failure": True,
            "rollback_on_dispatch_route_mismatch": True,
        },
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def _validations(shadow: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _validation("e2e_shadow_matrix_ready", shadow.get("e2e_shadow_matrix_ready") is True, "adamw_variant_e2e_shadow_matrix_missing"),
        _validation(
            "policy_default_off",
            policy.get("canary_enabled_by_default") is False
            and policy.get("canary_auto_enabled") is False
            and float(policy.get("max_canary_fraction_default", 1.0)) == 0.0,
            "adamw_variant_policy_not_default_off",
        ),
        _validation(
            "explicit_opt_in_and_manual_review_required",
            policy.get("explicit_opt_in_required") is True and policy.get("manual_review_required") is True,
            "adamw_variant_policy_missing_manual_review",
        ),
        _validation(
            "fallback_rollback_ready",
            _as_dict(policy.get("rollback_policy")).get("fallback_authoritative") is True
            and _as_dict(policy.get("rollback_policy")).get("rollback_on_nonfinite") is True
            and _as_dict(policy.get("rollback_policy")).get("rollback_on_parity_failure") is True,
            "adamw_variant_policy_missing_rollback",
        ),
        _validation(
            "runtime_dispatch_not_enabled",
            policy.get("runtime_dispatch_ready") is False
            and policy.get("native_dispatch_allowed") is False
            and policy.get("training_path_enabled") is False
            and shadow.get("runtime_dispatch_ready") is False
            and shadow.get("native_dispatch_allowed") is False
            and shadow.get("training_path_enabled") is False,
            "adamw_variant_policy_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            policy.get("default_behavior_changed") is False and shadow.get("default_behavior_changed") is False,
            "adamw_variant_policy_changed_default_behavior",
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


__all__ = ["POLICY_KIND", "build_adamw_variant_canary_rollout_policy_scorecard"]
