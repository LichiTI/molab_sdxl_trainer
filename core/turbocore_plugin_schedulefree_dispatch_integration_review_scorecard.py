"""Review-only gate for selected schedule-free plugin real dispatch integration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.turbocore_plugin_schedulefree_canary_rollout_policy_scorecard import (
    FALLBACK_BACKEND,
    OPTIMIZER_FAMILY,
    build_plugin_schedulefree_canary_rollout_policy_scorecard,
)


REVIEW_KIND = "plugin_schedulefree_dispatch_integration_review_v0"


def build_plugin_schedulefree_dispatch_integration_review_scorecard(
    *,
    rollout_policy_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    """Build a manual review package without enabling native schedule-free dispatch."""

    mode = _normalize_mode(native_training_mode)
    policy_report = dict(
        rollout_policy_report or build_plugin_schedulefree_canary_rollout_policy_scorecard()
    )
    review = _review_package(policy_report, mode)
    validations = _validations(policy_report, review)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []
    )
    ready = not failed
    return {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_schedulefree_dispatch_integration_review_scorecard_v0",
        "gate": "plugin_schedulefree_dispatch_integration_review",
        "ok": ready,
        "promotion_ready": ready,
        "review_gate_ready": ready,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "experimental_only": True,
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
            "requires explicit review before wiring selected schedule-free real dispatch"
            if ready
            else "fix selected schedule-free dispatch integration review blockers"
        ),
        "notes": [
            "This gate prepares a real-dispatch integration review only.",
            "The selected pytorch_optimizer plugin remains the training update authority.",
            "Canary and auto modes stay blocked until manual review approves wiring.",
        ],
    }


def _review_package(policy_report: Mapping[str, Any], mode: str) -> dict[str, Any]:
    policy = policy_report.get("policy") if isinstance(policy_report.get("policy"), Mapping) else {}
    rollback = policy.get("rollback_policy") if isinstance(policy.get("rollback_policy"), Mapping) else {}
    return {
        "schema_version": 1,
        "review_kind": REVIEW_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "native_training_mode": mode,
        "manual_review_required": True,
        "dispatch_review_outcome": "pending_manual_review",
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "runtime_hook_contract": {
            "optimizer_create_hook": "core.lulynx_trainer.trainer.Trainer._create_optimizer",
            "training_loop_step_hook": "core.lulynx_trainer.training_loop.TrainingLoop.train_epoch",
            "optimizer_train_mode_hook": "selected_optimizer.train",
            "optimizer_eval_mode_hook": "selected_optimizer.eval",
            "pre_optimizer_shadow_hook": "TurboCoreUpdateShadow.prepare_before_optimizer",
            "post_optimizer_shadow_hook": "TurboCoreUpdateShadow.compare_after_optimizer",
            "checkpoint_state_hook": "TrainingLoop.get_turbocore_update_checkpoint_state",
            "resume_state_hook": "TrainingLoop.load_turbocore_update_checkpoint_state",
        },
        "dispatch_contract": {
            "fallback_update_authority": FALLBACK_BACKEND,
            "native_update_authority": "none_until_review",
            "requires_selected_plugin_identity": True,
            "requires_native_schedulefree_kernel": True,
            "requires_training_tensor_binding": True,
            "requires_runtime_checkpoint_adapter": True,
            "requires_train_eval_mode_transition": True,
            "requires_end_to_end_shadow_matrix": True,
            "requires_rollout_policy": True,
        },
        "numeric_guardrails": {
            "rollback_on_nonfinite": bool(rollback.get("rollback_on_nonfinite", False)),
            "rollback_on_parity_failure": bool(rollback.get("rollback_on_parity_failure", False)),
            "rollback_on_checkpoint_adapter_failure": bool(
                rollback.get("rollback_on_checkpoint_adapter_failure", False)
            ),
            "rollback_on_train_eval_mode_mismatch": bool(
                rollback.get("rollback_on_train_eval_mode_mismatch", False)
            ),
            "state_machine_authority": "selected_plugin_until_review",
            "max_param_diff_required": 0.0,
            "max_state_tensor_diff_required": 0.0,
        },
        "rollback_policy": {
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_checkpoint_adapter_failure": True,
            "rollback_on_resume_mismatch": True,
            "rollback_on_train_eval_mode_mismatch": True,
            "rollback_on_selected_plugin_mismatch": True,
        },
        "audit_fields": [
            "native_training_mode",
            "selected_optimizer_name",
            "optimizer_family",
            "train_eval_mode",
            "param_dtype",
            "grad_dtype",
            "state_tensor_keys",
            "launch_plan",
            "route_decision",
            "fallback_backend",
            "parity_summary",
            "checkpoint_adapter_summary",
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
            "p18_rollout_policy_ready",
            bool(policy_report.get("canary_rollout_policy_ready", False)),
            "selected_schedulefree_canary_rollout_policy_missing",
        ),
        _validation(
            "runtime_hook_contract_present",
            bool(hooks.get("optimizer_create_hook"))
            and bool(hooks.get("training_loop_step_hook"))
            and bool(hooks.get("optimizer_train_mode_hook"))
            and bool(hooks.get("optimizer_eval_mode_hook"))
            and bool(hooks.get("checkpoint_state_hook"))
            and bool(hooks.get("resume_state_hook")),
            "selected_schedulefree_dispatch_review_hook_contract_missing",
        ),
        _validation(
            "dispatch_contract_requires_all_prior_evidence",
            bool(dispatch.get("requires_selected_plugin_identity", False))
            and bool(dispatch.get("requires_native_schedulefree_kernel", False))
            and bool(dispatch.get("requires_training_tensor_binding", False))
            and bool(dispatch.get("requires_runtime_checkpoint_adapter", False))
            and bool(dispatch.get("requires_train_eval_mode_transition", False))
            and bool(dispatch.get("requires_end_to_end_shadow_matrix", False)),
            "selected_schedulefree_dispatch_review_prior_evidence_missing",
        ),
        _validation(
            "numeric_guardrails_present",
            bool(numeric.get("rollback_on_nonfinite", False))
            and bool(numeric.get("rollback_on_parity_failure", False))
            and bool(numeric.get("rollback_on_checkpoint_adapter_failure", False))
            and bool(numeric.get("rollback_on_train_eval_mode_mismatch", False))
            and numeric.get("state_machine_authority") == "selected_plugin_until_review",
            "selected_schedulefree_dispatch_review_numeric_guardrails_missing",
        ),
        _validation(
            "rollback_manifest_present",
            bool(rollback.get("fallback_authoritative", False))
            and bool(rollback.get("rollback_on_resume_mismatch", False))
            and bool(rollback.get("rollback_on_train_eval_mode_mismatch", False))
            and bool(rollback.get("rollback_on_selected_plugin_mismatch", False)),
            "selected_schedulefree_dispatch_review_missing_rollback",
        ),
        _validation(
            "manual_review_blocks_canary_auto",
            bool(review.get("manual_review_required", False))
            and review.get("allowed_initial_modes") == ["off", "observe"]
            and review.get("blocked_modes_until_review") == ["canary", "auto"],
            "selected_schedulefree_dispatch_review_allows_dispatch_before_review",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(review.get("runtime_dispatch_ready", True))
            and not bool(review.get("native_dispatch_allowed", True))
            and not bool(review.get("training_path_enabled", True)),
            "selected_schedulefree_dispatch_review_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            not bool(policy_report.get("training_path_enabled", True))
            and not bool(policy_report.get("default_behavior_changed", True)),
            "selected_schedulefree_dispatch_review_changed_default_behavior",
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


__all__ = ["build_plugin_schedulefree_dispatch_integration_review_scorecard"]
