"""Smoke checks for V5-P44 runtime activation contract boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
CORE_ROOT = BACKEND_ROOT / "core"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT), str(CORE_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_runtime_activation_contract_boundary import (  # noqa: E402
    build_v5_runtime_activation_contract_boundary,
)
from lulynx_trainer.turbocore_v5_runtime_wiring_contract_boundary_smoke import (  # noqa: E402
    _gate as _p43_gate,
    _p42_ready,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "auto_launch_allowed",
    "runs_dispatched",
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "ui_exposure_allowed",
    "product_ui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "request_adapter_registered",
    "runtime_adapter_registered",
    "runtime_wiring_allowed",
    "runtime_activation_allowed",
    "runtime_activation_enabled",
    "runtime_adapter_enabled",
    "native_runtime_enabled",
    "native_dispatch_enabled",
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "runtime_resolver_patch_allowed",
    "execution_resolver_patch_allowed",
    "training_manager_patch_allowed",
    "rollout_authorization_allowed",
)
READY_DECISION = "runtime_activation_contract_boundary_recorded_default_off"
BLOCKED_DECISION = "runtime_activation_contract_boundary_blocked_default_off"
HOLD_DECISION = "runtime_activation_contract_boundary_hold_for_signed_review_default_off"
REJECTED_DECISION = "runtime_activation_contract_boundary_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p43_ready = _p43_ready()
    ready = _gate(p43_ready)
    assert ready["ok"] is True, ready
    assert ready["runtime_activation_contract_boundary_ready"] is True, ready
    assert ready["runtime_activation_contract_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p43_ready, review=None)
    _assert_hold(missing_review, "review", "missing")

    rejected_review = _gate(p43_ready, review=_activation_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p43_missing = _gate(None)
    _assert_blocked(p43_missing, "p43", "missing")

    p43_not_ready = _gate({**p43_ready, "ok": False, "runtime_wiring_contract_boundary_ready": False})
    _assert_blocked(p43_not_ready, "p43", "not_ready")

    p43_decision_mismatch = _gate({**p43_ready, "decision": "wrong"})
    _assert_blocked(p43_decision_mismatch, "p43", "not_ready")

    p43_runtime_adapter = _gate({**p43_ready, "runtime_adapter_registered": True})
    _assert_blocked(p43_runtime_adapter, "runtime_adapter_registered")

    p43_runtime_wiring = _gate({**p43_ready, "runtime_wiring_allowed": True})
    _assert_blocked(p43_runtime_wiring, "runtime_wiring_allowed")

    p43_post_fields = _gate({**p43_ready, "post_p43_request_fields": {"bad": True}})
    _assert_blocked(p43_post_fields, "post_p43_request_fields")

    missing_source = _gate(p43_ready, evidence=_without(_activation_evidence(), "source"))
    _assert_blocked(missing_source, "source_missing")

    missing_digest = _gate(p43_ready, evidence=_without_many(_activation_evidence(), "sha256", "artifact_digest"))
    _assert_blocked(missing_digest, "digest_missing")

    not_report_only = _gate(p43_ready, evidence={**_activation_evidence(), "report_only": False})
    _assert_blocked(not_report_only, "report_only")

    not_contract_only = _gate(p43_ready, evidence={**_activation_evidence(), "contract_only": False})
    _assert_blocked(not_contract_only, "contract_only")

    missing_later_operator = _gate(
        p43_ready,
        evidence={**_activation_evidence(), "requires_later_operator_activation_contract": False},
    )
    _assert_blocked(missing_later_operator, "later_operator_activation_contract")

    missing_section = _gate(p43_ready, evidence={**_activation_evidence(), "available_sections": ["rollback_policy"]})
    _assert_blocked(missing_section, "section_missing")

    missing_plan_inventory = _gate(p43_ready, evidence=_without(_activation_evidence(), "runtime_activation_plan_inventory"))
    _assert_blocked(missing_plan_inventory, "runtime_activation_plan_inventory")

    missing_preconditions = _gate(p43_ready, evidence=_without(_activation_evidence(), "activation_precondition_inventory"))
    _assert_blocked(missing_preconditions, "activation_precondition_inventory")

    runtime_activation = _plan_claim("runtime_activation_enabled")
    _assert_blocked(runtime_activation, "runtime_activation_enabled")

    adapter_enabled = _plan_claim("runtime_adapter_enabled")
    _assert_blocked(adapter_enabled, "runtime_adapter_enabled")

    native_runtime = _plan_claim("native_runtime_enabled")
    _assert_blocked(native_runtime, "native_runtime_enabled")

    native_dispatch = _plan_claim("native_dispatch_enabled")
    _assert_blocked(native_dispatch, "native_dispatch_enabled")

    request_fields = _plan_claim("request_fields_emitted")
    _assert_blocked(request_fields, "request_fields_emitted")

    training_launch = _plan_claim("training_launch_allowed")
    _assert_blocked(training_launch, "training_launch_allowed")

    precondition_active = _precondition_claim("precondition_active")
    _assert_blocked(precondition_active, "precondition_active")

    precondition_registered = _precondition_claim("precondition_registered")
    _assert_blocked(precondition_registered, "precondition_registered")

    review_missing_reviewer = _gate(p43_ready, review={**_activation_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")

    review_missing_reviewed_at = _gate(p43_ready, review={**_activation_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")

    review_scope_mismatch = _gate(p43_ready, review={**_activation_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")

    review_missing_ack = _gate(p43_ready, review={**_activation_review(), "acknowledge_no_runtime_activation": False})
    _assert_blocked(review_missing_ack, "ack_missing", "runtime_activation")

    review_approve_activation = _unsafe_review("approve_runtime_activation_allowed")
    _assert_blocked(review_approve_activation, "approve_runtime_activation_allowed")

    review_approve_adapter = _unsafe_review("approve_runtime_adapter_enabled")
    _assert_blocked(review_approve_adapter, "approve_runtime_adapter_enabled")

    review_approve_dispatch = _unsafe_review("approve_native_dispatch_enabled")
    _assert_blocked(review_approve_dispatch, "approve_native_dispatch_enabled")

    review_approve_fields = _unsafe_review("approve_request_fields_emitted")
    _assert_blocked(review_approve_fields, "approve_request_fields_emitted")

    review_approve_launch = _unsafe_review("approve_training_launch_allowed")
    _assert_blocked(review_approve_launch, "approve_training_launch_allowed")

    failure_history = _gate(
        p43_ready,
        failure_history=[{"reason": "runtime_activation_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    rollback_history = _gate(p43_ready, rollback_history=[{"kind": "runtime_activation_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")

    closed_failure = _gate(
        p43_ready,
        failure_history=[{"reason": "closed_activation_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p44_runtime_activation_contract_boundary_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p43_missing": _summary(p43_missing),
        "p43_not_ready": _summary(p43_not_ready),
        "p43_decision_mismatch": _summary(p43_decision_mismatch),
        "p43_runtime_adapter": _summary(p43_runtime_adapter),
        "p43_runtime_wiring": _summary(p43_runtime_wiring),
        "p43_post_fields": _summary(p43_post_fields),
        "missing_source": _summary(missing_source),
        "missing_digest": _summary(missing_digest),
        "missing_plan_inventory": _summary(missing_plan_inventory),
        "missing_preconditions": _summary(missing_preconditions),
        "runtime_activation": _summary(runtime_activation),
        "adapter_enabled": _summary(adapter_enabled),
        "native_dispatch": _summary(native_dispatch),
        "review_approve_activation": _summary(review_approve_activation),
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p43: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _activation_review() if review is ... else review
    return build_v5_runtime_activation_contract_boundary(
        p43_runtime_wiring_contract_boundary=p43,
        runtime_activation_contract_evidence=_activation_evidence() if evidence is None else evidence,
        runtime_activation_contract_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p43_ready() -> dict[str, Any]:
    return _p43_gate(_p42_ready())


def _activation_evidence() -> dict[str, Any]:
    sections = [
        "p43_runtime_wiring_boundary_reference",
        "runtime_activation_plan_inventory",
        "activation_precondition_inventory",
        "no_runtime_activation_boundary",
        "no_runtime_adapter_enabled_boundary",
        "no_request_fields_boundary",
        "no_training_launch_boundary",
        "fallback_policy",
        "rollback_policy",
        "observability_policy",
    ]
    return {
        "evidence_id": "runtime_activation_contract_boundary_v0",
        "evidence_version": "v0",
        "ok": True,
        "runtime_activation_contract_boundary_ready": True,
        "report_only": True,
        "boundary_only": True,
        "contract_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_operator_activation_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "runtime_activation_plan_inventory": [
            {"plan_id": "native_update_activation", **_safe_row_flags()},
            {"plan_id": "lora_fused_activation", **_safe_row_flags()},
        ],
        "activation_precondition_inventory": [
            {"check_id": "owner_opt_in", "precondition_active": False},
            {"check_id": "rollback_ready", "precondition_active": False},
        ],
        "sha256": "sha256:p44:runtime-activation:ready",
        "artifact_digest": "sha256:p44:runtime-activation:ready",
        "source": "temp/turbocore_v5_p44_runtime_activation.json",
        **_safe_row_flags(),
    }


def _activation_review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "runtime_activation_contract_boundary",
        "approve_runtime_activation_contract_boundary": approve,
    }
    for field in (
        "training_launch_allowed",
        "auto_launch_allowed",
        "runs_dispatched",
        "default_training_path_enabled",
        "training_path_enabled",
        "default_rollout_allowed",
        "auto_rollout_allowed",
        "ui_exposure_allowed",
        "product_ui_exposure_allowed",
        "request_adapter_mapping_allowed",
        "request_fields_emitted",
        "request_adapter_registered",
        "runtime_adapter_registered",
        "runtime_wiring_allowed",
        "runtime_activation_allowed",
        "runtime_activation_enabled",
        "runtime_adapter_enabled",
        "native_runtime_enabled",
        "native_dispatch_enabled",
        "generation_request_patch_allowed",
        "config_adapter_patch_allowed",
        "runtime_resolver_patch_allowed",
        "execution_resolver_patch_allowed",
        "training_manager_patch_allowed",
        "rollout_authorization_allowed",
    ):
        review[f"approve_{field}"] = False
    for field in (
        "acknowledge_p43_runtime_wiring_boundary_recorded",
        "acknowledge_default_off_boundary",
        "acknowledge_no_training_launch",
        "acknowledge_no_ui_exposure",
        "acknowledge_no_runtime_activation",
        "acknowledge_no_runtime_adapter_enabled",
        "acknowledge_no_request_fields_emitted",
        "acknowledge_no_request_config_runtime_patch",
        "acknowledge_no_default_or_auto_rollout",
        "acknowledge_activation_evidence_replayable",
        "acknowledge_later_operator_activation_contract_required",
        "acknowledge_manual_review_only",
    ):
        review[field] = True
    return review


def _safe_row_flags() -> dict[str, Any]:
    return {
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "ui_exposure_allowed": False,
        "product_ui_exposure_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "runtime_adapter_registered": False,
        "runtime_wiring_allowed": False,
        "runtime_activation_allowed": False,
        "runtime_activation_enabled": False,
        "runtime_adapter_enabled": False,
        "native_runtime_enabled": False,
        "native_dispatch_enabled": False,
        "generation_request_patch_allowed": False,
        "config_adapter_patch_allowed": False,
        "runtime_resolver_patch_allowed": False,
        "execution_resolver_patch_allowed": False,
        "training_manager_patch_allowed": False,
        "rollout_authorization_allowed": False,
        "default_behavior_changed": False,
        "blocked_reasons": [],
        "promotion_blockers": [],
    }


def _plan_claim(field: str) -> dict[str, Any]:
    row = {"plan_id": "native", **_safe_row_flags(), field: True}
    return _gate(_p43_ready(), evidence={**_activation_evidence(), "runtime_activation_plan_inventory": [row]})


def _precondition_claim(field: str) -> dict[str, Any]:
    row = {"check_id": "owner_opt_in", field: True}
    return _gate(_p43_ready(), evidence={**_activation_evidence(), "activation_precondition_inventory": [row]})


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p43_ready(), review={**_activation_review(), field: True})


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    copied = dict(value)
    copied.pop(key, None)
    return copied


def _without_many(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    copied = dict(value)
    for key in keys:
        copied.pop(key, None)
    return copied


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report
    assert report["post_p44_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["runtime_activation_contract_boundary_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["runtime_activation_contract_boundary_ready"] is False, report
    assert report["decision"] == HOLD_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_reason_fragments(report: dict[str, Any], *fragments: str) -> None:
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "boundary_ready": bool(report.get("runtime_activation_contract_boundary_ready", False)),
        "review_signed": bool(report.get("runtime_activation_contract_review_signed", False)),
        "rollback_required": bool(report.get("rollback_required", False)),
        "blocked_reasons": _blocked_reasons(report),
    }


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
