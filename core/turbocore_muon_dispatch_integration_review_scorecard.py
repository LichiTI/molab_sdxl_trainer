"""Manual dispatch review package for built-in Muon canary dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_muon_canary_rollout_policy_scorecard import (
    FALLBACK_BACKEND,
    build_muon_canary_rollout_policy_scorecard,
)


REVIEW_KIND = "muon_model_shape_aware_dispatch_integration_review_v0"
TARGET_OPTIMIZER = "Muon"
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_muon_dispatch_integration_review_scorecard(
    *,
    rollout_policy_report: Mapping[str, Any] | None = None,
    native_training_mode: str = "observe",
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Build a default-off Muon dispatch review package without enabling dispatch."""

    mode = _normalize_mode(native_training_mode)
    policy_report = _as_dict(rollout_policy_report or build_muon_canary_rollout_policy_scorecard())
    review = _review_package(policy_report, mode)
    validations = _validations(policy_report, review)
    failed = [item for item in validations if item.get("ok") is not True]
    blockers = _dedupe(reason for item in failed for reason in item.get("blocked_reasons", []) or [])
    ready = not blockers
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_muon_dispatch_integration_review_scorecard_v0",
        "gate": "muon_model_shape_aware_dispatch_integration_review",
        "ok": ready,
        "promotion_ready": ready,
        "review_gate_ready": ready,
        "dispatch_integration_review": ready,
        "manual_review_required": True,
        "canary_auto_blocked_until_review": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "experimental_only": True,
        "report_only": True,
        "product_native_dispatch_ready": False,
        "product_native_ready": False,
        "review_kind": REVIEW_KIND,
        "native_training_mode": mode,
        "fallback_backend": FALLBACK_BACKEND,
        "fallback_backend_authoritative": True,
        "review_package": review,
        "policy_summary": _as_dict(policy_report.get("summary")),
        "validations": validations,
        "summary": {
            "review_gate_ready": ready,
            "dispatch_integration_review": ready,
            "manual_review_required": True,
            "optimizer_count": len(_optimizer_types(policy_report)),
            "runtime_dispatch_ready": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "fallback_backend": FALLBACK_BACKEND,
            "product_native_ready_count": 0,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "requires explicit owner approval before wiring Muon canary dispatch"
            if ready
            else "fix Muon dispatch integration review blockers"
        ),
        "notes": [
            "This gate prepares a real-dispatch integration review only.",
            "Python Muon remains the training update authority.",
            "Canary and auto modes stay blocked until manual review approves wiring.",
        ],
    }
    if write_artifact:
        _write_artifact(report)
    return report


def _review_package(policy_report: Mapping[str, Any], mode: str) -> dict[str, Any]:
    optimizer_types = _optimizer_types(policy_report)
    return {
        "schema_version": 1,
        "review_kind": REVIEW_KIND,
        "optimizer_family": "built_in_muon_model_shape_aware",
        "optimizer_types": optimizer_types,
        "native_training_mode": mode,
        "manual_review_required": True,
        "dispatch_review_outcome": "pending_manual_review",
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "runtime_hook_contract": {
            "optimizer_create_hook": "core.lulynx_trainer.trainer.Trainer._create_optimizer",
            "training_loop_step_hook": "core.lulynx_trainer.training_loop.TrainingLoop.train_epoch",
            "muon_training_executor": "core.turbocore_muon_training_executor",
            "checkpoint_state_hook": "TrainingLoop.get_turbocore_update_checkpoint_state",
            "resume_state_hook": "TrainingLoop.load_turbocore_update_checkpoint_state",
        },
        "dispatch_contract": {
            "fallback_update_authority": FALLBACK_BACKEND,
            "native_update_authority": "none_until_review",
            "requires_muon_training_loop_canary": True,
            "requires_e2e_shadow_matrix": True,
            "requires_rollout_policy": True,
            "requires_manual_review_approval": True,
            "runtime_dispatch_enabled_by_this_gate": False,
            "request_schema_ui_enabled_by_this_gate": False,
        },
        "numeric_guardrails": {
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_shape_partition_mismatch": True,
            "rollback_on_dispatch_route_mismatch": True,
            "shape_authority": "fallback_optimizer_until_review",
        },
        "rollback_policy": {
            "fallback_backend": FALLBACK_BACKEND,
            "fallback_authoritative": True,
            "rollback_on_nonfinite": True,
            "rollback_on_parity_failure": True,
            "rollback_on_training_loop_error": True,
            "rollback_on_resume_mismatch": True,
            "rollback_on_shape_partition_mismatch": True,
            "rollback_on_dispatch_route_mismatch": True,
        },
        "audit_fields": [
            "native_training_mode",
            "optimizer_type",
            "optimizer_family",
            "param_shape",
            "param_dtype",
            "grad_dtype",
            "route_decision",
            "fallback_backend",
            "manual_review_decision",
            "rollback_reason",
        ],
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def _validations(policy_report: Mapping[str, Any], review: Mapping[str, Any]) -> list[dict[str, Any]]:
    hooks = _as_dict(review.get("runtime_hook_contract"))
    dispatch = _as_dict(review.get("dispatch_contract"))
    numeric = _as_dict(review.get("numeric_guardrails"))
    rollback = _as_dict(review.get("rollback_policy"))
    optimizer_types = set(_optimizer_types(policy_report))
    return [
        _validation(
            "rollout_policy_ready",
            policy_report.get("canary_rollout_policy_ready") is True,
            "muon_canary_rollout_policy_missing",
        ),
        _validation(
            "optimizer_set_complete",
            optimizer_types == {TARGET_OPTIMIZER},
            "muon_dispatch_review_optimizer_set_incomplete",
        ),
        _validation(
            "runtime_hook_contract_present",
            bool(hooks.get("optimizer_create_hook"))
            and bool(hooks.get("training_loop_step_hook"))
            and bool(hooks.get("muon_training_executor"))
            and bool(hooks.get("checkpoint_state_hook"))
            and bool(hooks.get("resume_state_hook")),
            "muon_dispatch_review_hook_contract_missing",
        ),
        _validation(
            "dispatch_contract_requires_prior_evidence",
            bool(dispatch.get("requires_muon_training_loop_canary"))
            and bool(dispatch.get("requires_e2e_shadow_matrix"))
            and bool(dispatch.get("requires_rollout_policy"))
            and bool(dispatch.get("requires_manual_review_approval")),
            "muon_dispatch_review_prior_evidence_missing",
        ),
        _validation(
            "numeric_guardrails_present",
            bool(numeric.get("rollback_on_nonfinite"))
            and bool(numeric.get("rollback_on_parity_failure"))
            and bool(numeric.get("rollback_on_shape_partition_mismatch"))
            and bool(numeric.get("rollback_on_dispatch_route_mismatch")),
            "muon_dispatch_review_numeric_guardrails_missing",
        ),
        _validation(
            "rollback_manifest_present",
            bool(rollback.get("fallback_authoritative"))
            and bool(rollback.get("rollback_on_resume_mismatch"))
            and bool(rollback.get("rollback_on_shape_partition_mismatch"))
            and bool(rollback.get("rollback_on_dispatch_route_mismatch")),
            "muon_dispatch_review_missing_rollback",
        ),
        _validation(
            "manual_review_blocks_canary_auto",
            review.get("manual_review_required") is True
            and review.get("allowed_initial_modes") == ["off", "observe"]
            and review.get("blocked_modes_until_review") == ["canary", "auto"],
            "muon_dispatch_review_allows_dispatch_before_review",
        ),
        _validation(
            "runtime_dispatch_disabled",
            policy_report.get("runtime_dispatch_ready") is False
            and policy_report.get("native_dispatch_allowed") is False
            and policy_report.get("training_path_enabled") is False
            and review.get("runtime_dispatch_ready") is False
            and review.get("native_dispatch_allowed") is False
            and review.get("training_path_enabled") is False,
            "muon_dispatch_review_enabled_dispatch",
        ),
        _validation(
            "default_behavior_unchanged",
            policy_report.get("default_behavior_changed") is False
            and review.get("default_behavior_changed") is False,
            "muon_dispatch_review_changed_default_behavior",
        ),
    ]


def _optimizer_types(report: Mapping[str, Any]) -> list[str]:
    policy = _as_dict(report.get("policy"))
    values = policy.get("optimizer_types")
    if not isinstance(values, list):
        return []
    return sorted(str(value) for value in values if str(value))


def _normalize_mode(value: str) -> str:
    text = str(value or "observe").strip().lower()
    return text if text in {"off", "observe"} else "observe"


def _write_artifact(report: Mapping[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_muon_dispatch_integration_review_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


__all__ = ["REVIEW_KIND", "TARGET_OPTIMIZER", "build_muon_dispatch_integration_review_scorecard"]
