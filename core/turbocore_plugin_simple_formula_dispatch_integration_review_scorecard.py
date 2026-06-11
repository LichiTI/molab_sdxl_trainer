"""Review-only gate for selected plugin simple-formula dispatch integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.turbocore_plugin_simple_formula_canary_rollout_policy_scorecard import (
    FALLBACK_BACKEND,
    OPTIMIZER_FAMILY,
    build_plugin_simple_formula_canary_rollout_policy_scorecard,
)


REVIEW_KIND = "plugin_simple_formula_dispatch_integration_review_v0"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_plugin_simple_formula_dispatch_integration_review_scorecard(
    *,
    rollout_policy_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Build a manual review package without enabling native dispatch."""

    mode = _normalize_mode(native_training_mode)
    policy_report = _as_dict(rollout_policy_report or build_plugin_simple_formula_canary_rollout_policy_scorecard())
    review = _review_package(policy_report, mode)
    validations = _validations(policy_report, review)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not failed
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_plugin_simple_formula_dispatch_integration_review_scorecard_v0",
        "gate": "plugin_simple_formula_dispatch_integration_review",
        "ok": ready,
        "promotion_ready": ready,
        "review_gate_ready": ready,
        "dispatch_integration_review": ready,
        "manual_review_required": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "experimental_only": True,
        "optimizer_family": OPTIMIZER_FAMILY,
        "review_kind": REVIEW_KIND,
        "native_training_mode": mode,
        "review_package": review,
        "policy_summary": dict(_as_dict(policy_report.get("summary"))),
        "validations": validations,
        "summary": {
            "optimizer_count": len(review["selected_optimizer_names"]),
            "review_gate_ready": ready,
            "dispatch_review_gate_ready": ready,
            "manual_review_required": True,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "product_native_ready_count": 0,
            "fallback_backend": review["rollback_policy"]["fallback_backend"],
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit owner/release approval before wiring selected plugin simple-formula dispatch"
            if ready
            else "fix selected plugin simple-formula dispatch integration review blockers"
        ),
        "notes": [
            "This gate prepares a real-dispatch integration review only.",
            "The selected pytorch_optimizer plugin remains the training update authority.",
            "Canary and auto modes stay blocked until manual review approves wiring.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _review_package(policy_report: Mapping[str, Any], mode: str) -> dict[str, Any]:
    policy = _as_dict(policy_report.get("policy"))
    rollback = _as_dict(policy.get("rollback_policy"))
    selected = [str(item) for item in policy.get("selected_optimizer_names", []) if str(item)]
    return {
        "schema_version": 1,
        "review_kind": REVIEW_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "selected_optimizer_names": selected,
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
            "fallback_update_authority": FALLBACK_BACKEND,
            "native_update_authority": "none_until_review",
            "requires_selected_plugin_identity": True,
            "requires_native_simple_formula_kernel": True,
            "requires_training_tensor_binding": True,
            "requires_end_to_end_shadow_matrix": True,
            "requires_rollout_policy": True,
            "requires_owner_release_approval": True,
        },
        "numeric_guardrails": {
            "rollback_on_nonfinite": bool(rollback.get("rollback_on_nonfinite", False)),
            "rollback_on_parity_failure": bool(rollback.get("rollback_on_parity_failure", False)),
            "state_authority": "selected_plugin_until_review",
            "max_param_diff_required": 0.0,
            "max_state_tensor_diff_required": 0.0,
        },
        "rollback_policy": {
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_resume_mismatch": True,
            "rollback_on_selected_plugin_mismatch": True,
        },
        "audit_fields": [
            "native_training_mode",
            "selected_optimizer_name",
            "optimizer_family",
            "param_dtype",
            "grad_dtype",
            "state_tensor_keys",
            "launch_plan",
            "route_decision",
            "fallback_backend",
            "parity_summary",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
    }


def _validations(policy_report: Mapping[str, Any], review: Mapping[str, Any]) -> list[dict[str, Any]]:
    hooks = _as_dict(review.get("runtime_hook_contract"))
    dispatch = _as_dict(review.get("dispatch_contract"))
    numeric = _as_dict(review.get("numeric_guardrails"))
    rollback = _as_dict(review.get("rollback_policy"))
    return [
        _validation(
            "rollout_policy_ready",
            policy_report.get("canary_rollout_policy_ready") is True,
            "plugin_simple_formula_canary_rollout_policy_missing",
        ),
        _validation(
            "runtime_hook_contract_present",
            bool(hooks.get("optimizer_create_hook"))
            and bool(hooks.get("training_loop_step_hook"))
            and bool(hooks.get("checkpoint_state_hook"))
            and bool(hooks.get("resume_state_hook")),
            "plugin_simple_formula_dispatch_review_hook_contract_missing",
        ),
        _validation(
            "dispatch_contract_requires_all_prior_evidence",
            dispatch.get("requires_selected_plugin_identity") is True
            and dispatch.get("requires_native_simple_formula_kernel") is True
            and dispatch.get("requires_training_tensor_binding") is True
            and dispatch.get("requires_end_to_end_shadow_matrix") is True
            and dispatch.get("requires_rollout_policy") is True
            and dispatch.get("requires_owner_release_approval") is True,
            "plugin_simple_formula_dispatch_review_prior_evidence_missing",
        ),
        _validation(
            "numeric_guardrails_present",
            numeric.get("rollback_on_nonfinite") is True
            and numeric.get("rollback_on_parity_failure") is True
            and numeric.get("state_authority") == "selected_plugin_until_review",
            "plugin_simple_formula_dispatch_review_numeric_guardrails_missing",
        ),
        _validation(
            "rollback_manifest_present",
            rollback.get("fallback_authoritative") is True
            and rollback.get("rollback_on_resume_mismatch") is True
            and rollback.get("rollback_on_selected_plugin_mismatch") is True,
            "plugin_simple_formula_dispatch_review_missing_rollback",
        ),
        _validation(
            "manual_review_blocks_canary_auto",
            review.get("manual_review_required") is True
            and review.get("allowed_initial_modes") == ["off", "observe"]
            and review.get("blocked_modes_until_review") == ["canary", "auto"],
            "plugin_simple_formula_dispatch_review_allows_dispatch_before_review",
        ),
        _validation(
            "runtime_dispatch_disabled",
            review.get("runtime_dispatch_ready") is False
            and review.get("native_dispatch_allowed") is False
            and review.get("training_path_enabled") is False,
            "plugin_simple_formula_dispatch_review_enabled_dispatch",
        ),
        _validation(
            "product_boundaries_not_exposed",
            review.get("request_fields_emitted") is False
            and review.get("schema_exposure_allowed") is False
            and review.get("ui_exposure_allowed") is False,
            "plugin_simple_formula_dispatch_review_exposed_product_boundary",
        ),
        _validation(
            "default_behavior_unchanged",
            policy_report.get("training_path_enabled") is False
            and policy_report.get("default_behavior_changed") is False,
            "plugin_simple_formula_dispatch_review_changed_default_behavior",
        ),
    ]


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


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_simple_formula_dispatch_integration_review_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["REVIEW_KIND", "build_plugin_simple_formula_dispatch_integration_review_scorecard"]
