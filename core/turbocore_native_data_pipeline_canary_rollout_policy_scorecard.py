"""Default-off canary rollout policy for native TurboCore data pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_native_data_pipeline_e2e_shadow_scorecard import (
    build_native_data_pipeline_e2e_shadow_scorecard,
)


FEATURE = "native_data_pipeline"
POLICY_KIND = "native_data_pipeline_canary_rollout_policy_v0"


def build_native_data_pipeline_canary_rollout_policy_scorecard(
    *,
    shadow_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Build a default-off rollout policy without enabling data dispatch."""

    shadow = dict(
        shadow_report
        or build_native_data_pipeline_e2e_shadow_scorecard(
            native_training_mode=native_training_mode,
        )
    )
    policy = _policy(shadow)
    validations = _validations(shadow, policy)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_native_data_pipeline_canary_rollout_policy_scorecard_v0",
        "gate": "p6l_native_data_pipeline_canary_rollout_policy",
        "ok": ready,
        "promotion_ready": ready,
        "canary_rollout_policy_ready": ready,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "feature": FEATURE,
        "policy_kind": POLICY_KIND,
        "native_training_mode": str(shadow.get("native_training_mode") or native_training_mode),
        "policy": policy,
        "shadow_summary": dict(shadow.get("summary") or {}),
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
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit review before wiring real native data pipeline canary dispatch"
            if ready
            else "fix native data pipeline canary rollout policy blockers"
        ),
        "notes": [
            "This policy is default-off and does not enable runtime dispatch.",
            "It defines guardrails required before real data-path canary wiring can be reviewed.",
            "Actual trainer dispatch wiring remains a separate decision.",
        ],
    }


def _policy(shadow: Mapping[str, Any]) -> dict[str, Any]:
    summary = shadow.get("summary") if isinstance(shadow.get("summary"), Mapping) else {}
    return {
        "schema_version": 1,
        "policy_kind": POLICY_KIND,
        "feature": FEATURE,
        "canary_enabled_by_default": False,
        "explicit_opt_in_required": True,
        "max_canary_fraction_default": 0.0,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "required_preflight_gates": [
            "p6h_native_data_pipeline_observe",
            "p6i_native_data_pipeline_adapter_shadow",
            "p6j_native_data_pipeline_semantic_h2d",
            "p6k_native_data_pipeline_e2e_shadow",
            "fallback_backend_authoritative",
            "runtime_dispatch_disabled",
            "default_behavior_unchanged",
        ],
        "shadow_guardrails": {
            "batch_descriptor_parity_ok": summary.get("batch_descriptor_parity_ok"),
            "loss_parity_ok": summary.get("loss_parity_ok"),
            "native_shadow_updates_original": summary.get("native_shadow_updates_original"),
        },
        "rollback_policy": {
            "fallback_backend": "standardcore_python_data_path",
            "fallback_authoritative": True,
            "rollback_on_descriptor_parity_failure": True,
            "rollback_on_h2d_ownership_failure": True,
            "rollback_on_nonfinite_batch_tensor": True,
            "rollback_on_queue_stall_regression": True,
        },
        "audit_fields": [
            "native_training_mode",
            "dataset_kind",
            "cache_first",
            "batch_size",
            "resolution_bucket",
            "prefetch_depth",
            "chunk_size",
            "route_decision",
            "fallback_backend",
            "parity_summary",
            "h2d_ownership_summary",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(shadow: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    rollback = policy.get("rollback_policy") if isinstance(policy.get("rollback_policy"), Mapping) else {}
    return [
        _validation(
            "p6k_e2e_shadow_ready",
            bool(shadow.get("e2e_shadow_ready", False)),
            "native_data_pipeline_e2e_shadow_missing",
        ),
        _validation(
            "policy_default_off",
            not bool(policy.get("canary_enabled_by_default", True))
            and float(policy.get("max_canary_fraction_default", 1.0)) == 0.0,
            "native_data_pipeline_canary_policy_not_default_off",
        ),
        _validation(
            "explicit_opt_in_required",
            bool(policy.get("explicit_opt_in_required", False)),
            "native_data_pipeline_canary_policy_missing_opt_in",
        ),
        _validation(
            "fallback_and_rollback_present",
            bool(rollback.get("fallback_authoritative", False))
            and bool(rollback.get("rollback_on_descriptor_parity_failure", False))
            and bool(rollback.get("rollback_on_h2d_ownership_failure", False))
            and bool(rollback.get("rollback_on_queue_stall_regression", False)),
            "native_data_pipeline_canary_policy_missing_rollback",
        ),
        _validation(
            "runtime_dispatch_still_disabled",
            not bool(policy.get("runtime_dispatch_ready", True))
            and not bool(policy.get("native_dispatch_allowed", True))
            and not bool(policy.get("training_path_enabled", True)),
            "native_data_pipeline_canary_policy_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(shadow.get("training_path_enabled", True))
            and not bool(shadow.get("default_behavior_changed", True)),
            "native_data_pipeline_canary_policy_changed_default_behavior",
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


__all__ = ["build_native_data_pipeline_canary_rollout_policy_scorecard"]
