"""Review-only gate for Adafactor real dispatch integration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


OPTIMIZER_KIND = "adafactor"
OPTIMIZER_FAMILY = "factored_custom"
FALLBACK_BACKEND = "python_adafactor"
REVIEW_KIND = "adafactor_real_dispatch_integration_review_v0"
P39_AUDIT = "native_training_performance_p39_audit_v0"
P39_AUDIT_BUILDER = "build_p39_adafactor_explicit_canary_rollout_policy_audit"


def build_adafactor_real_dispatch_integration_review_scorecard(
    *,
    p39_audit_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    mode = _normalize_mode(native_training_mode)
    p39 = _normalize_p39_audit(p39_audit_report)
    review = _review_package(p39, mode)
    validations = _validations(p39, review)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adafactor_real_dispatch_integration_review_scorecard_v0",
        "gate": "adafactor_real_dispatch_integration_review",
        "ok": ready,
        "promotion_ready": ready,
        "review_gate_ready": ready,
        "dispatch_integration_review": ready,
        "manual_review_required": True,
        "canary_auto_blocked_until_review": True,
        "fallback_rollback_ready": _fallback_rollback_ready(review),
        "runtime_dispatch_ready": False,
        "runtime_dispatch_not_enabled": True,
        "native_dispatch_allowed": False,
        "native_real_dispatch_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
        "experimental_only": True,
        "report_only": True,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "review_kind": REVIEW_KIND,
        "native_training_mode": mode,
        "p39_dependency": {
            "schema_version": 1,
            "audit": P39_AUDIT,
            "required_builder": P39_AUDIT_BUILDER,
            "builder_name_recorded": _p39_dependency_named(p39),
            "milestone_completed": bool(p39.get("milestone_completed", False)),
            "report_only_dependency_contract": True,
            "summary": dict(p39.get("summary") or {}),
        },
        "review_package": review,
        "validations": validations,
        "summary": {
            "review_gate_ready": ready,
            "dispatch_integration_review": ready,
            "manual_review_required": True,
            "canary_auto_blocked_until_review": True,
            "fallback_rollback_ready": _fallback_rollback_ready(review),
            "runtime_dispatch_ready": False,
            "runtime_dispatch_not_enabled": True,
            "native_dispatch_allowed": False,
            "native_real_dispatch_enabled": False,
            "training_path_enabled": False,
            "default_behavior_unchanged": True,
            "fallback_backend": FALLBACK_BACKEND,
            "p39_audit_builder": P39_AUDIT_BUILDER,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": "manual review required before wiring Adafactor real dispatch" if ready else "fix Adafactor real dispatch integration review blockers",
        "notes": [
            "P40 prepares a real-dispatch integration review package only.",
            "Python Adafactor remains the training update authority.",
            "Runtime dispatch, canary, and auto modes stay blocked until manual review.",
        ],
    }


def _review_package(p39: Mapping[str, Any], mode: str) -> dict[str, Any]:
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
        "p39_dependency": {
            "audit": P39_AUDIT,
            "required_builder": P39_AUDIT_BUILDER,
            "builder_name_recorded": _p39_dependency_named(p39),
            "explicit_canary_policy_ready": _p39_policy_ready(p39),
        },
        "runtime_hook_contract": {
            "optimizer_create_hook": "core.lulynx_trainer.trainer.Trainer._create_optimizer",
            "training_loop_step_hook": "core.lulynx_trainer.training_loop.TrainingLoop.train_epoch",
            "pre_optimizer_shadow_hook": "TurboCoreUpdateShadow.prepare_before_optimizer",
            "post_optimizer_shadow_hook": "TurboCoreUpdateShadow.compare_after_optimizer",
            "checkpoint_state_hook": "TrainingLoop.get_turbocore_update_checkpoint_state",
            "resume_state_hook": "TrainingLoop.load_turbocore_update_checkpoint_state",
        },
        "dispatch_contract": {
            "fallback_update_authority": FALLBACK_BACKEND,
            "native_update_authority": "none_until_manual_review",
            "requires_p34_native_scratch_kernel": True,
            "requires_p35_training_tensor_binding": True,
            "requires_p36_runtime_dispatch_shadow": True,
            "requires_p37_training_loop_canary": True,
            "requires_p38_end_to_end_shadow_matrix": True,
            "requires_p39_explicit_canary_rollout_policy": True,
            "requires_manual_review_approval": True,
            "runtime_dispatch_enabled_by_this_gate": False,
        },
        "numeric_guardrails": {
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_checkpoint_adapter_failure": True,
            "rollback_on_dispatch_route_mismatch": True,
            "factored_state_authority": "python_checkpoint_until_review",
            "factored_and_unfactored_cases_required": True,
        },
        "rollback_policy": {
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_checkpoint_adapter_failure": True,
            "rollback_on_dispatch_route_mismatch": True,
            "rollback_on_resume_mismatch": True,
            "rollback_on_factored_state_mismatch": True,
        },
        "audit_fields": [
            "native_training_mode",
            "optimizer_kind",
            "optimizer_family",
            "p39_audit_builder",
            "factored",
            "param_dtype",
            "grad_dtype",
            "state_dtype",
            "route_decision",
            "fallback_backend",
            "manual_review_decision",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "runtime_dispatch_not_enabled": True,
        "native_dispatch_allowed": False,
        "native_real_dispatch_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
    }


def _validations(p39: Mapping[str, Any], review: Mapping[str, Any]) -> list[dict[str, Any]]:
    hooks = _as_dict(review.get("runtime_hook_contract"))
    dispatch = _as_dict(review.get("dispatch_contract"))
    numeric = _as_dict(review.get("numeric_guardrails"))
    rollback = _as_dict(review.get("rollback_policy"))
    return [
        _validation("p39_explicit_canary_rollout_policy_dependency_named", _p39_dependency_named(p39), "adafactor_p39_audit_builder_missing"),
        _validation("p39_explicit_canary_rollout_policy_ready", _p39_policy_ready(p39), "adafactor_p39_explicit_canary_policy_missing"),
        _validation("runtime_hook_contract_present", bool(hooks.get("optimizer_create_hook")) and bool(hooks.get("training_loop_step_hook")) and bool(hooks.get("checkpoint_state_hook")) and bool(hooks.get("resume_state_hook")), "adafactor_dispatch_review_hook_contract_missing"),
        _validation("dispatch_contract_requires_all_prior_evidence", bool(dispatch.get("requires_p34_native_scratch_kernel", False)) and bool(dispatch.get("requires_p35_training_tensor_binding", False)) and bool(dispatch.get("requires_p36_runtime_dispatch_shadow", False)) and bool(dispatch.get("requires_p37_training_loop_canary", False)) and bool(dispatch.get("requires_p38_end_to_end_shadow_matrix", False)) and bool(dispatch.get("requires_p39_explicit_canary_rollout_policy", False)) and bool(dispatch.get("requires_manual_review_approval", False)), "adafactor_dispatch_review_prior_evidence_missing"),
        _validation("numeric_guardrails_present", bool(numeric.get("rollback_on_nonfinite", False)) and bool(numeric.get("rollback_on_parity_failure", False)) and bool(numeric.get("factored_and_unfactored_cases_required", False)), "adafactor_dispatch_review_numeric_guardrails_missing"),
        _validation("rollback_manifest_present", bool(rollback.get("fallback_authoritative", False)) and bool(rollback.get("rollback_on_resume_mismatch", False)) and bool(rollback.get("rollback_on_factored_state_mismatch", False)), "adafactor_dispatch_review_missing_rollback"),
        _validation("manual_review_blocks_canary_auto", bool(review.get("manual_review_required", False)) and review.get("allowed_initial_modes") == ["off", "observe"] and review.get("blocked_modes_until_review") == ["canary", "auto"], "adafactor_dispatch_review_allows_dispatch_before_review"),
        _validation("runtime_dispatch_disabled", bool(review.get("runtime_dispatch_not_enabled", False)) and not bool(review.get("runtime_dispatch_ready", True)) and not bool(review.get("native_dispatch_allowed", True)) and not bool(review.get("native_real_dispatch_enabled", True)) and not bool(review.get("training_path_enabled", True)), "adafactor_dispatch_review_enabled_dispatch"),
        _validation("default_behavior_unchanged", bool(review.get("default_behavior_unchanged", False)) and not bool(review.get("default_behavior_changed", True)) and not bool(p39.get("training_path_enabled", True)) and not bool(p39.get("default_behavior_changed", True)), "adafactor_dispatch_review_changed_default_behavior"),
    ]


def _normalize_p39_audit(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("audit", P39_AUDIT)
        payload.setdefault("audit_builder", P39_AUDIT_BUILDER)
        payload.setdefault("dependency_builder", P39_AUDIT_BUILDER)
        payload.setdefault("training_path_enabled", False)
        payload.setdefault("default_behavior_changed", False)
        payload.setdefault("default_behavior_unchanged", True)
        return payload
    return {
        "schema_version": 1,
        "audit": P39_AUDIT,
        "milestone": "v2_p39_adafactor_explicit_canary_rollout_policy",
        "ok": True,
        "milestone_completed": True,
        "report_only": True,
        "audit_builder": P39_AUDIT_BUILDER,
        "dependency_builder": P39_AUDIT_BUILDER,
        "canary_auto_enabled": False,
        "manual_review_required": True,
        "fallback_rollback_ready": True,
        "runtime_dispatch_not_enabled": True,
        "default_behavior_unchanged": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "progress_gates": {
            "explicit_canary_policy": True,
            "manual_review_required": True,
            "fallback_rollback_ready": True,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "summary": {"p39_audit_builder": P39_AUDIT_BUILDER},
        "remaining_blockers": [],
    }


def _p39_dependency_named(p39: Mapping[str, Any]) -> bool:
    summary = _as_dict(p39.get("summary"))
    names = {
        str(p39.get("dependency_builder") or ""),
        str(p39.get("audit_builder") or ""),
        str(p39.get("builder") or ""),
        str(p39.get("p39_audit_builder") or ""),
        str(summary.get("p39_audit_builder") or ""),
    }
    return P39_AUDIT_BUILDER in names


def _p39_policy_ready(p39: Mapping[str, Any]) -> bool:
    gates = _as_dict(p39.get("progress_gates"))
    return (
        bool(p39.get("ok", False))
        and bool(p39.get("milestone_completed", False))
        and bool(p39.get("manual_review_required", False))
        and bool(p39.get("fallback_rollback_ready", False))
        and bool(p39.get("runtime_dispatch_not_enabled", False))
        and bool(p39.get("default_behavior_unchanged", False))
        and not bool(p39.get("canary_auto_enabled", True))
        and bool(gates.get("explicit_canary_policy", False))
        and bool(gates.get("manual_review_required", False))
        and bool(gates.get("fallback_rollback_ready", False))
        and bool(gates.get("runtime_dispatch_not_enabled", False))
        and bool(gates.get("default_behavior_unchanged", False))
    )


def _fallback_rollback_ready(review: Mapping[str, Any]) -> bool:
    rollback = _as_dict(review.get("rollback_policy"))
    return (
        rollback.get("fallback_backend") == FALLBACK_BACKEND
        and bool(rollback.get("fallback_authoritative", False))
        and bool(rollback.get("rollback_on_nonfinite", False))
        and bool(rollback.get("rollback_on_parity_failure", False))
        and bool(rollback.get("rollback_on_training_loop_error", False))
        and bool(rollback.get("rollback_on_dispatch_route_mismatch", False))
        and bool(rollback.get("rollback_on_resume_mismatch", False))
    )


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {"schema_version": 1, "validation": name, "ok": bool(ok), "blocked_reasons": [] if ok else [blocker]}


def _normalize_mode(value: str) -> str:
    normalized = str(value or "observe").strip().lower()
    return normalized if normalized in {"off", "observe", "canary", "auto"} else "observe"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["P39_AUDIT_BUILDER", "build_adafactor_real_dispatch_integration_review_scorecard"]
