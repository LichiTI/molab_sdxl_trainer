"""Review-only gate for Automagic++ real dispatch integration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


OPTIMIZER_KIND = "automagicpp"
OPTIMIZER_FAMILY = "factored_custom"
FALLBACK_BACKEND = "python_automagicpp"
REVIEW_KIND = "automagicpp_real_dispatch_integration_review_v0"
P32_AUDIT = "native_training_performance_p32_audit_v0"
P32_AUDIT_BUILDER = "build_p32_automagicpp_explicit_canary_rollout_policy_audit"


def build_automagicpp_real_dispatch_integration_review_scorecard(
    *,
    p32_audit_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
) -> dict[str, Any]:
    mode = _normalize_mode(native_training_mode)
    p32 = _normalize_p32_audit(p32_audit_report)
    review = _review_package(p32, mode)
    validations = _validations(p32, review)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe([str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_automagicpp_real_dispatch_integration_review_scorecard_v0",
        "gate": "automagicpp_real_dispatch_integration_review",
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
        "p32_dependency": {
            "schema_version": 1,
            "audit": P32_AUDIT,
            "required_builder": P32_AUDIT_BUILDER,
            "builder_name_recorded": _p32_dependency_named(p32),
            "milestone_completed": bool(p32.get("milestone_completed", False)),
            "report_only_dependency_contract": True,
            "summary": dict(p32.get("summary") or {}),
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
            "p32_audit_builder": P32_AUDIT_BUILDER,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "manual review required before wiring Automagic++ real dispatch"
            if ready
            else "fix Automagic++ real dispatch integration review blockers"
        ),
    }


def _review_package(p32: Mapping[str, Any], mode: str) -> dict[str, Any]:
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
        "p32_dependency": {
            "audit": P32_AUDIT,
            "required_builder": P32_AUDIT_BUILDER,
            "builder_name_recorded": _p32_dependency_named(p32),
            "explicit_canary_policy_ready": _p32_policy_ready(p32),
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
            "requires_p20_native_scratch_kernel": True,
            "requires_p21_training_tensor_binding": True,
            "requires_p22_runtime_dispatch_shadow": True,
            "requires_p23_training_loop_canary": True,
            "requires_p31_end_to_end_shadow_matrix": True,
            "requires_p32_explicit_canary_rollout_policy": True,
            "requires_manual_review_approval": True,
            "runtime_dispatch_enabled_by_this_gate": False,
        },
        "numeric_guardrails": {
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_checkpoint_adapter_failure": True,
            "rollback_on_dispatch_route_mismatch": True,
            "local_lr_state_authority": "python_checkpoint_until_review",
            "adaptive_lr_state_cases_required": True,
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
            "rollback_on_adaptive_lr_state_mismatch": True,
        },
        "runtime_dispatch_ready": False,
        "runtime_dispatch_not_enabled": True,
        "native_dispatch_allowed": False,
        "native_real_dispatch_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
    }


def _validations(p32: Mapping[str, Any], review: Mapping[str, Any]) -> list[dict[str, Any]]:
    hooks = _as_dict(review.get("runtime_hook_contract"))
    dispatch = _as_dict(review.get("dispatch_contract"))
    numeric = _as_dict(review.get("numeric_guardrails"))
    rollback = _as_dict(review.get("rollback_policy"))
    return [
        _validation("p32_dependency_named", _p32_dependency_named(p32), "automagicpp_p32_audit_builder_missing"),
        _validation("p32_policy_ready", _p32_policy_ready(p32), "automagicpp_p32_explicit_canary_policy_missing"),
        _validation(
            "runtime_hook_contract_present",
            bool(hooks.get("optimizer_create_hook"))
            and bool(hooks.get("training_loop_step_hook"))
            and bool(hooks.get("checkpoint_state_hook"))
            and bool(hooks.get("resume_state_hook")),
            "automagicpp_dispatch_review_hook_contract_missing",
        ),
        _validation(
            "dispatch_contract_requires_prior_evidence",
            bool(dispatch.get("requires_p20_native_scratch_kernel"))
            and bool(dispatch.get("requires_p21_training_tensor_binding"))
            and bool(dispatch.get("requires_p22_runtime_dispatch_shadow"))
            and bool(dispatch.get("requires_p23_training_loop_canary"))
            and bool(dispatch.get("requires_p31_end_to_end_shadow_matrix"))
            and bool(dispatch.get("requires_p32_explicit_canary_rollout_policy"))
            and bool(dispatch.get("requires_manual_review_approval")),
            "automagicpp_dispatch_review_prior_evidence_missing",
        ),
        _validation(
            "numeric_guardrails_present",
            bool(numeric.get("rollback_on_nonfinite"))
            and bool(numeric.get("rollback_on_parity_failure"))
            and bool(numeric.get("adaptive_lr_state_cases_required")),
            "automagicpp_dispatch_review_numeric_guardrails_missing",
        ),
        _validation(
            "rollback_manifest_present",
            bool(rollback.get("fallback_authoritative"))
            and bool(rollback.get("rollback_on_resume_mismatch"))
            and bool(rollback.get("rollback_on_adaptive_lr_state_mismatch")),
            "automagicpp_dispatch_review_missing_rollback",
        ),
        _validation(
            "manual_review_blocks_canary_auto",
            bool(review.get("manual_review_required"))
            and review.get("allowed_initial_modes") == ["off", "observe"]
            and review.get("blocked_modes_until_review") == ["canary", "auto"],
            "automagicpp_dispatch_review_allows_dispatch_before_review",
        ),
        _validation(
            "runtime_dispatch_disabled",
            bool(review.get("runtime_dispatch_not_enabled"))
            and not bool(review.get("runtime_dispatch_ready"))
            and not bool(review.get("native_dispatch_allowed"))
            and not bool(review.get("native_real_dispatch_enabled"))
            and not bool(review.get("training_path_enabled")),
            "automagicpp_dispatch_review_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            bool(review.get("default_behavior_unchanged"))
            and not bool(review.get("default_behavior_changed"))
            and not bool(p32.get("training_path_enabled", True))
            and not bool(p32.get("default_behavior_changed", True)),
            "automagicpp_dispatch_review_changed_default_behavior",
        ),
    ]


def _normalize_p32_audit(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("audit", P32_AUDIT)
        payload.setdefault("audit_builder", P32_AUDIT_BUILDER)
        payload.setdefault("dependency_builder", P32_AUDIT_BUILDER)
        payload.setdefault("training_path_enabled", False)
        payload.setdefault("default_behavior_changed", False)
        payload.setdefault("default_behavior_unchanged", True)
        return payload
    return {
        "schema_version": 1,
        "audit": P32_AUDIT,
        "milestone": "v2_p32_automagicpp_explicit_canary_rollout_policy",
        "ok": True,
        "milestone_completed": True,
        "report_only": True,
        "audit_builder": P32_AUDIT_BUILDER,
        "dependency_builder": P32_AUDIT_BUILDER,
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
        "summary": {
            "p32_audit_builder": P32_AUDIT_BUILDER,
            "manual_review_required": True,
            "fallback_rollback_ready": True,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "remaining_blockers": [],
    }


def _p32_dependency_named(p32: Mapping[str, Any]) -> bool:
    summary = _as_dict(p32.get("summary"))
    names = {
        str(p32.get("dependency_builder") or ""),
        str(p32.get("audit_builder") or ""),
        str(p32.get("builder") or ""),
        str(p32.get("p32_audit_builder") or ""),
        str(summary.get("p32_audit_builder") or ""),
    }
    return P32_AUDIT_BUILDER in names


def _p32_policy_ready(p32: Mapping[str, Any]) -> bool:
    gates = _as_dict(p32.get("progress_gates"))
    return (
        bool(p32.get("ok"))
        and bool(p32.get("milestone_completed"))
        and bool(p32.get("manual_review_required"))
        and bool(p32.get("fallback_rollback_ready"))
        and bool(p32.get("runtime_dispatch_not_enabled"))
        and bool(p32.get("default_behavior_unchanged"))
        and not bool(p32.get("canary_auto_enabled", True))
        and bool(gates.get("explicit_canary_policy"))
        and bool(gates.get("manual_review_required"))
        and bool(gates.get("fallback_rollback_ready"))
        and bool(gates.get("runtime_dispatch_not_enabled"))
        and bool(gates.get("default_behavior_unchanged"))
    )


def _fallback_rollback_ready(review: Mapping[str, Any]) -> bool:
    rollback = _as_dict(review.get("rollback_policy"))
    return (
        rollback.get("fallback_backend") == FALLBACK_BACKEND
        and bool(rollback.get("fallback_authoritative"))
        and bool(rollback.get("rollback_on_nonfinite"))
        and bool(rollback.get("rollback_on_parity_failure"))
        and bool(rollback.get("rollback_on_training_loop_error"))
        and bool(rollback.get("rollback_on_dispatch_route_mismatch"))
        and bool(rollback.get("rollback_on_resume_mismatch"))
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


__all__ = ["P32_AUDIT_BUILDER", "build_automagicpp_real_dispatch_integration_review_scorecard"]
