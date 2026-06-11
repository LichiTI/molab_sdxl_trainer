"""Default-off canary rollout policy for PagedAdamW8bit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_paged_adamw8bit_e2e_shadow_training_matrix_scorecard import (
    build_paged_adamw8bit_e2e_shadow_training_matrix_scorecard,
)


POLICY_KIND = "paged_adamw8bit_canary_rollout_policy_v0"
OPTIMIZER_KIND = "paged_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_paged"


def build_paged_adamw8bit_canary_rollout_policy_scorecard(
    *,
    shadow_matrix_report: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
    require_live_matrix: bool = True,
) -> dict[str, Any]:
    """Build a default-off rollout policy without enabling dispatch."""

    shadow = dict(
        shadow_matrix_report
        or build_paged_adamw8bit_e2e_shadow_training_matrix_scorecard(
            run_live_probe=run_live_probe,
            require_live_matrix=require_live_matrix,
        )
    )
    policy = _policy(shadow)
    validations = _validations(shadow, policy)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_canary_rollout_policy_scorecard_v0",
        "gate": "paged_adamw8bit_canary_rollout_policy",
        "ok": ready,
        "promotion_ready": False,
        "canary_rollout_policy_ready": ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "policy_kind": POLICY_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "policy": policy,
        "shadow_matrix_summary": dict(shadow.get("summary") or {}),
        "validations": validations,
        "summary": {
            "canary_rollout_policy_ready": ready,
            "canary_enabled_by_default": bool(policy.get("canary_enabled_by_default", True)),
            "explicit_opt_in_required": bool(policy.get("explicit_opt_in_required", False)),
            "max_canary_fraction_default": float(policy.get("max_canary_fraction_default", 1.0)),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
        },
        "promotion_blockers": _dedupe(
            blockers
            + [
                "paged_adamw8bit_runtime_dispatch_disabled_pending_review",
                "paged_adamw8bit_real_dispatch_wiring_missing",
            ]
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit review before wiring real PagedAdamW8bit canary dispatch"
            if ready
            else "fix PagedAdamW8bit canary rollout policy blockers"
        ),
        "notes": [
            "This policy is default-off and does not enable runtime dispatch.",
            "It defines the guardrails required before any real canary wiring can be reviewed.",
            "Actual dispatch wiring remains a separate decision.",
        ],
    }


def _policy(shadow: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_kind": POLICY_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "canary_enabled_by_default": False,
        "explicit_opt_in_required": True,
        "max_canary_fraction_default": 0.0,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "required_preflight_gates": [
            "p8q_e2e_shadow_training_matrix",
            "fallback_backend_authoritative",
            "native_shadow_never_updates_original",
            "runtime_dispatch_disabled",
            "default_behavior_unchanged",
        ],
        "numeric_guardrails": {
            "max_param_diff": shadow.get("summary", {}).get("max_param_diff"),
            "max_state_float_diff": shadow.get("summary", {}).get("max_state_float_diff"),
            "max_loss_diff": shadow.get("summary", {}).get("max_loss_diff"),
            "state_uint8_mismatch_count": shadow.get("summary", {}).get("state_uint8_mismatch_count"),
        },
        "rollback_policy": {
            "fallback_backend": "bitsandbytes_paged_adamw8bit",
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_checkpoint_adapter_failure": True,
        },
        "audit_fields": [
            "native_training_mode",
            "optimizer_kind",
            "launch_plan",
            "route_decision",
            "fallback_backend",
            "parity_summary",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(
    shadow: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "p8q_e2e_shadow_training_matrix_ready",
            bool(shadow.get("e2e_shadow_training_matrix_ready", False)),
            "paged_adamw8bit_e2e_shadow_training_matrix_missing",
        ),
        _validation(
            "policy_default_off",
            not bool(policy.get("canary_enabled_by_default", True))
            and float(policy.get("max_canary_fraction_default", 1.0)) == 0.0,
            "paged_adamw8bit_canary_rollout_policy_not_default_off",
        ),
        _validation(
            "explicit_opt_in_required",
            bool(policy.get("explicit_opt_in_required", False)),
            "paged_adamw8bit_canary_rollout_policy_missing_opt_in",
        ),
        _validation(
            "fallback_and_rollback_present",
            bool(policy.get("rollback_policy", {}).get("fallback_authoritative", False))
            and bool(policy.get("rollback_policy", {}).get("rollback_on_nonfinite", False))
            and bool(policy.get("rollback_policy", {}).get("rollback_on_parity_failure", False)),
            "paged_adamw8bit_canary_rollout_policy_missing_rollback",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(policy.get("runtime_dispatch_ready", True))
            and not bool(policy.get("native_dispatch_allowed", True))
            and not bool(policy.get("training_path_enabled", True)),
            "paged_adamw8bit_canary_rollout_policy_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(shadow.get("training_path_enabled", True))
            and not bool(shadow.get("default_behavior_changed", True)),
            "paged_adamw8bit_canary_rollout_policy_changed_default_behavior",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_paged_adamw8bit_canary_rollout_policy_scorecard"]
