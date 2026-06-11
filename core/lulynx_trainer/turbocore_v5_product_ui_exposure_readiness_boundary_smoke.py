"""Smoke checks for V5-P40 product/UI exposure readiness boundary."""

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

from core.turbocore_v5_controlled_rollout_policy_evidence_gate import (  # noqa: E402
    build_v5_controlled_rollout_policy_evidence_gate,
)
from core.turbocore_v5_product_ui_exposure_readiness_boundary import (  # noqa: E402
    build_v5_product_ui_exposure_readiness_boundary,
)
from lulynx_trainer.turbocore_v5_controlled_rollout_policy_evidence_gate_smoke import (  # noqa: E402
    _p38_ready,
    _policy_evidence,
    _policy_review,
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
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "ui_entry_enabled",
    "ready_for_ui",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "rollout_authorization_allowed",
)
READY_DECISION = "product_ui_exposure_readiness_evidence_recorded_default_off"
BLOCKED_DECISION = "product_ui_exposure_readiness_blocked_default_off"
HOLD_DECISION = "product_ui_exposure_hold_for_signed_review_default_off"
REJECTED_DECISION = "product_ui_exposure_readiness_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p39_ready = _p39_ready()
    ready = _gate(p39_ready)
    assert ready["ok"] is True, ready
    assert ready["product_ui_exposure_readiness_boundary_ready"] is True, ready
    assert ready["product_ui_exposure_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p39_ready, review=None)
    _assert_hold(missing_review, "review", "missing")

    rejected_review = _gate(p39_ready, review=_ui_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rejected_for_default_off_hold"] is True, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    assert rejected_review["product_ui_exposure_readiness_boundary_ready"] is False, rejected_review
    _assert_default_off(rejected_review)

    p39_missing = _gate(None)
    _assert_blocked(p39_missing, "p39", "missing")

    p39_not_ready = _gate({**p39_ready, "ok": False, "controlled_rollout_policy_evidence_ready": False})
    _assert_blocked(p39_not_ready, "p39", "not_ready")

    p39_decision_mismatch = _gate({**p39_ready, "decision": "wrong"})
    _assert_blocked(p39_decision_mismatch, "p39", "not_ready")

    p39_unsafe_ui = _gate({**p39_ready, "ui_exposure_allowed": True})
    _assert_blocked(p39_unsafe_ui, "ui_exposure_allowed")

    p39_post_fields = _gate({**p39_ready, "post_p39_request_fields": {"bad": True}})
    _assert_blocked(p39_post_fields, "post_p39_request_fields")

    missing_source = _gate(p39_ready, evidence=_without(_ui_evidence(), "source"))
    _assert_blocked(missing_source, "source_missing")

    missing_digest = _gate(p39_ready, evidence=_without_many(_ui_evidence(), "sha256", "artifact_digest"))
    _assert_blocked(missing_digest, "digest_missing")

    not_report_only = _gate(p39_ready, evidence={**_ui_evidence(), "report_only": False})
    _assert_blocked(not_report_only, "report_only")

    not_boundary_only = _gate(p39_ready, evidence={**_ui_evidence(), "boundary_only": False})
    _assert_blocked(not_boundary_only, "boundary_only")

    missing_later_contract = _gate(
        p39_ready,
        evidence={**_ui_evidence(), "requires_later_integration_contract": False},
    )
    _assert_blocked(missing_later_contract, "later_integration_contract")

    missing_section = _gate(p39_ready, evidence={**_ui_evidence(), "available_sections": ["surface_inventory"]})
    _assert_blocked(missing_section, "section_missing")

    missing_inventory = _gate(p39_ready, evidence=_without(_ui_evidence(), "product_surface_inventory"))
    _assert_blocked(missing_inventory, "surface_inventory")

    exposed_surface = _gate(
        p39_ready,
        evidence={**_ui_evidence(), "product_surface_inventory": [{"surface_id": "launcher", "exposure_enabled": True}]},
    )
    _assert_blocked(exposed_surface, "surface_exposure")

    route_registration = _gate(p39_ready, evidence={**_ui_evidence(), "ui_route_registration": {"bad": True}})
    _assert_blocked(route_registration, "ui_route_registration")

    request_fields = _gate(p39_ready, evidence={**_ui_evidence(), "request_adapter_fields": {"bad": True}})
    _assert_blocked(request_fields, "request_adapter_fields")

    review_missing_reviewer = _gate(p39_ready, review={**_ui_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")

    review_missing_reviewed_at = _gate(p39_ready, review={**_ui_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")

    review_scope_mismatch = _gate(p39_ready, review={**_ui_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")

    review_missing_ack = _gate(p39_ready, review={**_ui_review(), "acknowledge_no_ui_entry": False})
    _assert_blocked(review_missing_ack, "ack_missing", "no_ui_entry")

    review_approve_ui = _gate(p39_ready, review={**_ui_review(), "approve_product_ui_exposure_allowed": True})
    _assert_blocked(review_approve_ui, "approve_product_ui_exposure_allowed")

    review_approve_launcher = _gate(p39_ready, review={**_ui_review(), "approve_launcher_exposure_allowed": True})
    _assert_blocked(review_approve_launcher, "approve_launcher_exposure_allowed")

    review_approve_adapter = _gate(p39_ready, review={**_ui_review(), "approve_request_adapter_mapping_allowed": True})
    _assert_blocked(review_approve_adapter, "approve_request_adapter_mapping_allowed")

    review_approve_launch = _gate(p39_ready, review={**_ui_review(), "approve_training_launch_allowed": True})
    _assert_blocked(review_approve_launch, "approve_training_launch_allowed")

    failure_history = _gate(
        p39_ready,
        failure_history=[{"reason": "ui_readiness_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    rollback_history = _gate(p39_ready, rollback_history=[{"kind": "ui_boundary_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")

    closed_failure = _gate(
        p39_ready,
        failure_history=[{"reason": "closed_ui_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p40_product_ui_exposure_readiness_boundary_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p39_missing": _summary(p39_missing),
        "p39_not_ready": _summary(p39_not_ready),
        "p39_decision_mismatch": _summary(p39_decision_mismatch),
        "p39_unsafe_ui": _summary(p39_unsafe_ui),
        "p39_post_fields": _summary(p39_post_fields),
        "missing_source": _summary(missing_source),
        "missing_digest": _summary(missing_digest),
        "not_report_only": _summary(not_report_only),
        "not_boundary_only": _summary(not_boundary_only),
        "missing_later_contract": _summary(missing_later_contract),
        "missing_section": _summary(missing_section),
        "missing_inventory": _summary(missing_inventory),
        "exposed_surface": _summary(exposed_surface),
        "route_registration": _summary(route_registration),
        "request_fields": _summary(request_fields),
        "review_missing_reviewer": _summary(review_missing_reviewer),
        "review_missing_reviewed_at": _summary(review_missing_reviewed_at),
        "review_scope_mismatch": _summary(review_scope_mismatch),
        "review_missing_ack": _summary(review_missing_ack),
        "review_approve_ui": _summary(review_approve_ui),
        "review_approve_launcher": _summary(review_approve_launcher),
        "review_approve_adapter": _summary(review_approve_adapter),
        "review_approve_launch": _summary(review_approve_launch),
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p39: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _ui_review() if review is ... else review
    return build_v5_product_ui_exposure_readiness_boundary(
        p39_controlled_rollout_policy_gate=p39,
        product_ui_exposure_readiness_evidence=_ui_evidence() if evidence is None else evidence,
        product_ui_exposure_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p39_ready() -> dict[str, Any]:
    return build_v5_controlled_rollout_policy_evidence_gate(
        p38_packaging_observability_gate=_p38_ready(),
        controlled_rollout_policy_evidence=_policy_evidence(),
        controlled_rollout_policy_review=_policy_review(),
    )


def _ui_evidence() -> dict[str, Any]:
    sections = [
        "surface_inventory",
        "no_ui_exposure_boundary",
        "no_request_adapter_boundary",
        "operator_opt_in_policy",
        "rollback_policy",
        "observability_policy",
    ]
    return {
        "evidence_id": "product_ui_exposure_boundary_v0",
        "evidence_version": "v0",
        "ok": True,
        "product_ui_exposure_readiness_evidence_ready": True,
        "report_only": True,
        "boundary_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_integration_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "product_surface_inventory": [
            {"surface_id": "launcher", "exposure_enabled": False, "entry_registered": False},
            {"surface_id": "webui", "exposure_enabled": False, "entry_registered": False},
            {"surface_id": "backend_api", "exposure_enabled": False, "entry_registered": False},
        ],
        "sha256": "sha256:p40:ui-readiness:ready",
        "artifact_digest": "sha256:p40:ui-readiness:ready",
        "source": "temp/turbocore_v5_p40_ui_readiness.json",
        **_safe_row_flags(),
    }


def _ui_review(approve: bool = True) -> dict[str, Any]:
    return {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "product_ui_exposure_readiness_boundary",
        "approve_product_ui_exposure_readiness_evidence": approve,
        "approve_training_launch_allowed": False,
        "approve_auto_launch_allowed": False,
        "approve_runs_dispatched": False,
        "approve_default_training_path_enabled": False,
        "approve_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
        "approve_ui_exposure_allowed": False,
        "approve_product_ui_exposure_allowed": False,
        "approve_launcher_exposure_allowed": False,
        "approve_webui_exposure_allowed": False,
        "approve_ui_entry_enabled": False,
        "approve_ready_for_ui": False,
        "approve_request_adapter_mapping_allowed": False,
        "approve_request_fields_emitted": False,
        "approve_rollout_authorization_allowed": False,
        "acknowledge_p39_policy_evidence_recorded": True,
        "acknowledge_default_off_boundary": True,
        "acknowledge_no_training_launch": True,
        "acknowledge_no_ui_exposure": True,
        "acknowledge_no_ui_entry": True,
        "acknowledge_no_request_adapter_mapping": True,
        "acknowledge_no_default_or_auto_rollout": True,
        "acknowledge_readiness_evidence_replayable": True,
        "acknowledge_later_integration_contract_required": True,
        "acknowledge_manual_review_only": True,
    }


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
        "launcher_exposure_allowed": False,
        "webui_exposure_allowed": False,
        "ui_entry_enabled": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "rollout_authorization_allowed": False,
        "default_behavior_changed": False,
        "blocked_reasons": [],
        "promotion_blockers": [],
    }


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
    assert report["post_p40_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["product_ui_exposure_readiness_boundary_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["product_ui_exposure_readiness_boundary_ready"] is False, report
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
        "boundary_ready": bool(report.get("product_ui_exposure_readiness_boundary_ready", False)),
        "review_signed": bool(report.get("product_ui_exposure_review_signed", False)),
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
