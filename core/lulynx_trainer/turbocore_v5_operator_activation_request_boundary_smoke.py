"""Smoke checks for V5-P45 operator activation request boundary."""

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

from core.turbocore_v5_operator_activation_request_boundary import (  # noqa: E402
    build_v5_operator_activation_request_boundary,
)
from lulynx_trainer.turbocore_v5_runtime_activation_contract_boundary_smoke import (  # noqa: E402
    _gate as _p44_gate,
    _p43_ready,
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
    "operator_activation_request_allowed",
    "operator_activation_request_submitted",
    "activation_request_submitted",
    "operator_activation_requested",
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
READY_DECISION = "operator_activation_request_boundary_recorded_default_off"
BLOCKED_DECISION = "operator_activation_request_boundary_blocked_default_off"
HOLD_DECISION = "operator_activation_request_boundary_hold_for_signed_review_default_off"
REJECTED_DECISION = "operator_activation_request_boundary_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p44_ready = _p44_ready()
    ready = _gate(p44_ready)
    assert ready["ok"] is True, ready
    assert ready["operator_activation_request_boundary_ready"] is True, ready
    assert ready["operator_activation_request_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p44_ready, review=None)
    _assert_hold(missing_review, "review", "missing")

    rejected_review = _gate(p44_ready, review=_operator_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    _assert_default_off(rejected_review)

    p44_missing = _gate(None)
    _assert_blocked(p44_missing, "p44", "missing")

    p44_not_ready = _gate({**p44_ready, "ok": False, "runtime_activation_contract_boundary_ready": False})
    _assert_blocked(p44_not_ready, "p44", "not_ready")

    p44_decision_mismatch = _gate({**p44_ready, "decision": "wrong"})
    _assert_blocked(p44_decision_mismatch, "p44", "not_ready")

    p44_runtime_activation = _gate({**p44_ready, "runtime_activation_enabled": True})
    _assert_blocked(p44_runtime_activation, "runtime_activation_enabled")

    p44_native_dispatch = _gate({**p44_ready, "native_dispatch_enabled": True})
    _assert_blocked(p44_native_dispatch, "native_dispatch_enabled")

    p44_post_fields = _gate({**p44_ready, "post_p44_request_fields": {"bad": True}})
    _assert_blocked(p44_post_fields, "post_p44_request_fields")

    missing_source = _gate(p44_ready, evidence=_without(_operator_evidence(), "source"))
    _assert_blocked(missing_source, "source_missing")

    missing_digest = _gate(p44_ready, evidence=_without_many(_operator_evidence(), "sha256", "artifact_digest"))
    _assert_blocked(missing_digest, "digest_missing")

    not_report_only = _gate(p44_ready, evidence={**_operator_evidence(), "report_only": False})
    _assert_blocked(not_report_only, "report_only")

    not_request_only = _gate(p44_ready, evidence={**_operator_evidence(), "request_only": False})
    _assert_blocked(not_request_only, "request_only")

    missing_later_execution = _gate(
        p44_ready,
        evidence={**_operator_evidence(), "requires_later_operator_activation_execution_contract": False},
    )
    _assert_blocked(missing_later_execution, "later_operator_activation_execution_contract")

    missing_section = _gate(p44_ready, evidence={**_operator_evidence(), "available_sections": ["rollback_policy"]})
    _assert_blocked(missing_section, "section_missing")

    missing_request_inventory = _gate(p44_ready, evidence=_without(_operator_evidence(), "operator_activation_request_inventory"))
    _assert_blocked(missing_request_inventory, "operator_activation_request_inventory")

    missing_scope_inventory = _gate(p44_ready, evidence=_without(_operator_evidence(), "activation_scope_inventory"))
    _assert_blocked(missing_scope_inventory, "activation_scope_inventory")

    request_submitted = _request_claim("operator_activation_request_submitted")
    _assert_blocked(request_submitted, "operator_activation_request_submitted")

    activation_submitted = _request_claim("activation_request_submitted")
    _assert_blocked(activation_submitted, "activation_request_submitted")

    activation_requested = _request_claim("operator_activation_requested")
    _assert_blocked(activation_requested, "operator_activation_requested")

    scope_enabled = _scope_claim("activation_scope_enabled")
    _assert_blocked(scope_enabled, "activation_scope_enabled")

    runtime_enabled = _scope_claim("runtime_activation_enabled")
    _assert_blocked(runtime_enabled, "runtime_activation_enabled")

    adapter_enabled = _scope_claim("runtime_adapter_enabled")
    _assert_blocked(adapter_enabled, "runtime_adapter_enabled")

    native_dispatch = _scope_claim("native_dispatch_enabled")
    _assert_blocked(native_dispatch, "native_dispatch_enabled")

    review_missing_reviewer = _gate(p44_ready, review={**_operator_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")

    review_missing_reviewed_at = _gate(p44_ready, review={**_operator_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")

    review_scope_mismatch = _gate(p44_ready, review={**_operator_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")

    review_missing_ack = _gate(
        p44_ready,
        review={**_operator_review(), "acknowledge_no_operator_activation_request_submitted": False},
    )
    _assert_blocked(review_missing_ack, "ack_missing", "operator_activation")

    review_approve_submit = _unsafe_review("approve_operator_activation_request_submitted")
    _assert_blocked(review_approve_submit, "approve_operator_activation_request_submitted")

    review_approve_activation = _unsafe_review("approve_runtime_activation_enabled")
    _assert_blocked(review_approve_activation, "approve_runtime_activation_enabled")

    review_approve_dispatch = _unsafe_review("approve_native_dispatch_enabled")
    _assert_blocked(review_approve_dispatch, "approve_native_dispatch_enabled")

    review_approve_launch = _unsafe_review("approve_training_launch_allowed")
    _assert_blocked(review_approve_launch, "approve_training_launch_allowed")

    failure_history = _gate(
        p44_ready,
        failure_history=[{"reason": "operator_request_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    rollback_history = _gate(p44_ready, rollback_history=[{"kind": "operator_request_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")

    closed_failure = _gate(
        p44_ready,
        failure_history=[{"reason": "closed_operator_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p45_operator_activation_request_boundary_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p44_missing": _summary(p44_missing),
        "p44_not_ready": _summary(p44_not_ready),
        "p44_decision_mismatch": _summary(p44_decision_mismatch),
        "p44_runtime_activation": _summary(p44_runtime_activation),
        "p44_native_dispatch": _summary(p44_native_dispatch),
        "p44_post_fields": _summary(p44_post_fields),
        "missing_source": _summary(missing_source),
        "missing_digest": _summary(missing_digest),
        "missing_request_inventory": _summary(missing_request_inventory),
        "missing_scope_inventory": _summary(missing_scope_inventory),
        "request_submitted": _summary(request_submitted),
        "activation_requested": _summary(activation_requested),
        "runtime_enabled": _summary(runtime_enabled),
        "native_dispatch": _summary(native_dispatch),
        "review_approve_submit": _summary(review_approve_submit),
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p44: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _operator_review() if review is ... else review
    return build_v5_operator_activation_request_boundary(
        p44_runtime_activation_contract_boundary=p44,
        operator_activation_request_evidence=_operator_evidence() if evidence is None else evidence,
        operator_activation_request_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p44_ready() -> dict[str, Any]:
    return _p44_gate(_p43_ready())


def _operator_evidence() -> dict[str, Any]:
    sections = [
        "p44_runtime_activation_boundary_reference",
        "operator_activation_request_inventory",
        "operator_identity_boundary",
        "activation_scope_boundary",
        "no_operator_request_submission_boundary",
        "no_runtime_activation_boundary",
        "no_runtime_adapter_enabled_boundary",
        "no_request_fields_boundary",
        "no_training_launch_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    return {
        "evidence_id": "operator_activation_request_boundary_v0",
        "evidence_version": "v0",
        "ok": True,
        "operator_activation_request_boundary_ready": True,
        "report_only": True,
        "boundary_only": True,
        "request_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_operator_activation_execution_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "operator_activation_request_inventory": [
            {"request_id": "native_update_operator_request", **_safe_row_flags()},
            {"request_id": "lora_fused_operator_request", **_safe_row_flags()},
        ],
        "activation_scope_inventory": [
            {"scope_id": "native_update_activation_scope", **_safe_row_flags()},
            {"scope_id": "lora_fused_activation_scope", **_safe_row_flags()},
        ],
        "sha256": "sha256:p45:operator-request:ready",
        "artifact_digest": "sha256:p45:operator-request:ready",
        "source": "temp/turbocore_v5_p45_operator_activation_request.json",
        **_safe_row_flags(),
    }


def _operator_review(approve: bool = True) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "operator_activation_request_boundary",
        "approve_operator_activation_request_boundary": approve,
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
        "operator_activation_request_allowed",
        "operator_activation_request_submitted",
        "activation_request_submitted",
        "operator_activation_requested",
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
        "acknowledge_p44_runtime_activation_boundary_recorded",
        "acknowledge_default_off_boundary",
        "acknowledge_no_training_launch",
        "acknowledge_no_ui_exposure",
        "acknowledge_no_operator_activation_request_submitted",
        "acknowledge_no_runtime_activation",
        "acknowledge_no_runtime_adapter_enabled",
        "acknowledge_no_request_fields_emitted",
        "acknowledge_no_request_config_runtime_patch",
        "acknowledge_no_default_or_auto_rollout",
        "acknowledge_operator_request_evidence_replayable",
        "acknowledge_later_operator_activation_execution_contract_required",
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
        "operator_activation_request_allowed": False,
        "operator_activation_request_submitted": False,
        "activation_request_submitted": False,
        "operator_activation_requested": False,
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


def _request_claim(field: str) -> dict[str, Any]:
    row = {"request_id": "native", **_safe_row_flags(), field: True}
    return _gate(_p44_ready(), evidence={**_operator_evidence(), "operator_activation_request_inventory": [row]})


def _scope_claim(field: str) -> dict[str, Any]:
    row = {"scope_id": "native", **_safe_row_flags(), field: True}
    return _gate(_p44_ready(), evidence={**_operator_evidence(), "activation_scope_inventory": [row]})


def _unsafe_review(field: str) -> dict[str, Any]:
    return _gate(_p44_ready(), review={**_operator_review(), field: True})


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
    assert report["post_p45_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["operator_activation_request_boundary_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["operator_activation_request_boundary_ready"] is False, report
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
        "boundary_ready": bool(report.get("operator_activation_request_boundary_ready", False)),
        "review_signed": bool(report.get("operator_activation_request_review_signed", False)),
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
