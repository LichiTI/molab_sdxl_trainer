"""Review-only gate for PagedAdamW8bit real dispatch integration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_paged_adamw8bit_canary_rollout_policy_scorecard import (
    build_paged_adamw8bit_canary_rollout_policy_scorecard,
)


OPTIMIZER_KIND = "paged_adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized_paged"
REVIEW_KIND = "paged_adamw8bit_dispatch_integration_review_v0"


def build_paged_adamw8bit_dispatch_integration_review_scorecard(
    *,
    rollout_policy_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
    run_live_probe: bool = True,
    require_live_matrix: bool = True,
) -> dict[str, Any]:
    """Build a manual review package without enabling native dispatch."""

    mode = _normalize_mode(native_training_mode)
    policy_report = dict(
        rollout_policy_report
        or build_paged_adamw8bit_canary_rollout_policy_scorecard(
            run_live_probe=run_live_probe,
            require_live_matrix=require_live_matrix,
        )
    )
    review = _review_package(policy_report, mode)
    validations = _validations(policy_report, review)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_paged_adamw8bit_dispatch_integration_review_scorecard_v0",
        "gate": "paged_adamw8bit_dispatch_integration_review",
        "ok": ready,
        "promotion_ready": ready,
        "review_gate_ready": ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "experimental_only": True,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "review_kind": REVIEW_KIND,
        "native_training_mode": mode,
        "review_package": review,
        "policy_summary": dict(policy_report.get("summary") or {}),
        "validations": validations,
        "summary": {
            "review_gate_ready": ready,
            "manual_review_required": True,
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "fallback_backend": review["rollback_policy"]["fallback_backend"],
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit review before wiring PagedAdamW8bit real dispatch"
            if ready
            else "fix PagedAdamW8bit dispatch integration review blockers"
        ),
        "notes": [
            "This gate prepares a real-dispatch integration review only.",
            "bitsandbytes PagedAdamW8bit remains the training update authority.",
            "Canary and auto modes stay blocked until manual review approves wiring.",
        ],
    }


def _review_package(policy_report: Mapping[str, Any], mode: str) -> dict[str, Any]:
    policy = policy_report.get("policy") if isinstance(policy_report.get("policy"), Mapping) else {}
    rollback = policy.get("rollback_policy") if isinstance(policy.get("rollback_policy"), Mapping) else {}
    return {
        "schema_version": 1,
        "review_kind": REVIEW_KIND,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "manual_review_required": True,
        "dispatch_review_outcome": "pending_manual_review",
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "runtime_hook_contract": {
            "optimizer_create_hook": "core.lulynx_trainer.trainer.Trainer._create_optimizer",
            "training_loop_step_hook": "core.lulynx_trainer.training_loop.TrainingLoop.train_epoch",
            "pre_optimizer_shadow_hook": "TurboCoreUpdateShadow.prepare_before_optimizer",
            "post_optimizer_shadow_hook": "TurboCoreUpdateShadow.compare_after_optimizer",
            "checkpoint_state_hook": "TrainingLoop.get_turbocore_update_checkpoint_state",
            "resume_state_hook": "TrainingLoop.load_turbocore_update_checkpoint_state",
        },
        "dispatch_contract": {
            "fallback_update_authority": "bitsandbytes_paged_adamw8bit",
            "native_update_authority": "none_until_review",
            "requires_training_tensor_binding": True,
            "requires_checkpoint_runtime_adapter": True,
            "requires_bnb_exact_oracle_boundary": True,
            "requires_real_training_matrix": True,
            "requires_end_to_end_shadow_matrix": True,
            "requires_rollout_policy": True,
        },
        "numeric_guardrails": {
            "rollback_on_nonfinite": bool(rollback.get("rollback_on_nonfinite", False)),
            "rollback_on_parity_failure": bool(rollback.get("rollback_on_parity_failure", False)),
            "rollback_on_checkpoint_adapter_failure": bool(
                rollback.get("rollback_on_checkpoint_adapter_failure", False)
            ),
            "quantized_state_authority": "bitsandbytes_checkpoint_until_review",
            "state_uint8_parity_required": True,
        },
        "rollback_policy": {
            "fallback_backend": "bitsandbytes_paged_adamw8bit",
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_checkpoint_adapter_failure": True,
            "rollback_on_resume_mismatch": True,
            "rollback_on_bnb_oracle_mismatch": True,
        },
        "audit_fields": [
            "native_training_mode",
            "optimizer_kind",
            "optimizer_family",
            "param_dtype",
            "grad_dtype",
            "state_dtype",
            "launch_plan",
            "route_decision",
            "fallback_backend",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(policy_report: Mapping[str, Any], review: Mapping[str, Any]) -> list[dict[str, Any]]:
    hooks = review.get("runtime_hook_contract") if isinstance(review.get("runtime_hook_contract"), Mapping) else {}
    dispatch = review.get("dispatch_contract") if isinstance(review.get("dispatch_contract"), Mapping) else {}
    numeric = review.get("numeric_guardrails") if isinstance(review.get("numeric_guardrails"), Mapping) else {}
    rollback = review.get("rollback_policy") if isinstance(review.get("rollback_policy"), Mapping) else {}
    return [
        _validation(
            "p8r_rollout_policy_ready",
            bool(policy_report.get("canary_rollout_policy_ready", False)),
            "paged_adamw8bit_canary_rollout_policy_missing",
        ),
        _validation(
            "runtime_hook_contract_present",
            bool(hooks.get("optimizer_create_hook"))
            and bool(hooks.get("training_loop_step_hook"))
            and bool(hooks.get("checkpoint_state_hook"))
            and bool(hooks.get("resume_state_hook")),
            "paged_adamw8bit_dispatch_review_hook_contract_missing",
        ),
        _validation(
            "dispatch_contract_requires_all_prior_evidence",
            bool(dispatch.get("requires_training_tensor_binding", False))
            and bool(dispatch.get("requires_checkpoint_runtime_adapter", False))
            and bool(dispatch.get("requires_bnb_exact_oracle_boundary", False))
            and bool(dispatch.get("requires_end_to_end_shadow_matrix", False)),
            "paged_adamw8bit_dispatch_review_prior_evidence_missing",
        ),
        _validation(
            "numeric_guardrails_present",
            bool(numeric.get("rollback_on_nonfinite", False))
            and bool(numeric.get("rollback_on_parity_failure", False))
            and bool(numeric.get("state_uint8_parity_required", False)),
            "paged_adamw8bit_dispatch_review_numeric_guardrails_missing",
        ),
        _validation(
            "rollback_manifest_present",
            bool(rollback.get("fallback_authoritative", False))
            and bool(rollback.get("rollback_on_resume_mismatch", False))
            and bool(rollback.get("rollback_on_bnb_oracle_mismatch", False)),
            "paged_adamw8bit_dispatch_review_missing_rollback",
        ),
        _validation(
            "manual_review_blocks_canary_auto",
            bool(review.get("manual_review_required", False))
            and review.get("allowed_initial_modes") == ["off", "observe"]
            and review.get("blocked_modes_until_review") == ["canary", "auto"],
            "paged_adamw8bit_dispatch_review_allows_dispatch_before_review",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(review.get("runtime_dispatch_ready", True))
            and not bool(review.get("native_dispatch_allowed", True))
            and not bool(review.get("training_path_enabled", True)),
            "paged_adamw8bit_dispatch_review_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(policy_report.get("training_path_enabled", True))
            and not bool(policy_report.get("default_behavior_changed", True)),
            "paged_adamw8bit_dispatch_review_changed_default_behavior",
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


__all__ = ["build_paged_adamw8bit_dispatch_integration_review_scorecard"]
