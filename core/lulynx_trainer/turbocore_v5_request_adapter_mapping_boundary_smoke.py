"""Smoke checks for V5-P41 request-adapter mapping boundary."""

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

from core.turbocore_v5_product_ui_exposure_readiness_boundary import (  # noqa: E402
    build_v5_product_ui_exposure_readiness_boundary,
)
from core.turbocore_v5_request_adapter_mapping_boundary import (  # noqa: E402
    build_v5_request_adapter_mapping_boundary,
)
from lulynx_trainer.turbocore_v5_product_ui_exposure_readiness_boundary_smoke import (  # noqa: E402
    _p39_ready,
    _ui_evidence,
    _ui_review,
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
    "generation_request_patch_allowed",
    "config_adapter_patch_allowed",
    "rollout_authorization_allowed",
)
READY_DECISION = "request_adapter_mapping_boundary_evidence_recorded_default_off"
BLOCKED_DECISION = "request_adapter_mapping_boundary_blocked_default_off"
HOLD_DECISION = "request_adapter_mapping_hold_for_signed_review_default_off"
REJECTED_DECISION = "request_adapter_mapping_boundary_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p40_ready = _p40_ready()
    ready = _gate(p40_ready)
    assert ready["ok"] is True, ready
    assert ready["request_adapter_mapping_boundary_ready"] is True, ready
    assert ready["request_adapter_mapping_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    _assert_default_off(ready)

    missing_review = _gate(p40_ready, review=None)
    _assert_hold(missing_review, "review", "missing")

    rejected_review = _gate(p40_ready, review=_mapping_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    assert rejected_review["request_adapter_mapping_boundary_ready"] is False, rejected_review
    _assert_default_off(rejected_review)

    p40_missing = _gate(None)
    _assert_blocked(p40_missing, "p40", "missing")

    p40_not_ready = _gate({**p40_ready, "ok": False, "product_ui_exposure_readiness_boundary_ready": False})
    _assert_blocked(p40_not_ready, "p40", "not_ready")

    p40_decision_mismatch = _gate({**p40_ready, "decision": "wrong"})
    _assert_blocked(p40_decision_mismatch, "p40", "not_ready")

    p40_request_adapter = _gate({**p40_ready, "request_adapter_mapping_allowed": True})
    _assert_blocked(p40_request_adapter, "request_adapter_mapping_allowed")

    p40_post_fields = _gate({**p40_ready, "post_p40_request_fields": {"bad": True}})
    _assert_blocked(p40_post_fields, "post_p40_request_fields")

    missing_source = _gate(p40_ready, evidence=_without(_mapping_evidence(), "source"))
    _assert_blocked(missing_source, "source_missing")

    missing_digest = _gate(p40_ready, evidence=_without_many(_mapping_evidence(), "sha256", "artifact_digest"))
    _assert_blocked(missing_digest, "digest_missing")

    not_report_only = _gate(p40_ready, evidence={**_mapping_evidence(), "report_only": False})
    _assert_blocked(not_report_only, "report_only")

    missing_later_contract = _gate(
        p40_ready,
        evidence={**_mapping_evidence(), "requires_later_adapter_contract": False},
    )
    _assert_blocked(missing_later_contract, "later_adapter_contract")

    missing_section = _gate(p40_ready, evidence={**_mapping_evidence(), "available_sections": ["schema_inventory"]})
    _assert_blocked(missing_section, "section_missing")

    missing_inventory = _gate(p40_ready, evidence=_without(_mapping_evidence(), "request_adapter_mapping_inventory"))
    _assert_blocked(missing_inventory, "mapping_inventory")

    mapping_enabled = _gate(
        p40_ready,
        evidence={**_mapping_evidence(), "request_adapter_mapping_inventory": [{"mapping_id": "native", "mapping_enabled": True}]},
    )
    _assert_blocked(mapping_enabled, "mapping_not_disabled")

    adapter_registered = _gate(
        p40_ready,
        evidence={
            **_mapping_evidence(),
            "request_adapter_mapping_inventory": [
                {"mapping_id": "native", "mapping_enabled": False, "adapter_registered": True}
            ],
        },
    )
    _assert_blocked(adapter_registered, "adapter_registered")

    request_fields = _gate(p40_ready, evidence={**_mapping_evidence(), "request_adapter_fields": {"bad": True}})
    _assert_blocked(request_fields, "request_adapter_fields")

    generation_patch = _gate(p40_ready, evidence={**_mapping_evidence(), "generation_request_patch": {"bad": True}})
    _assert_blocked(generation_patch, "generation_request_patch")

    config_patch = _gate(p40_ready, evidence={**_mapping_evidence(), "config_adapter_patch": {"bad": True}})
    _assert_blocked(config_patch, "config_adapter_patch")

    review_missing_reviewer = _gate(p40_ready, review={**_mapping_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")

    review_missing_reviewed_at = _gate(p40_ready, review={**_mapping_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")

    review_scope_mismatch = _gate(p40_ready, review={**_mapping_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")

    review_missing_ack = _gate(p40_ready, review={**_mapping_review(), "acknowledge_no_request_fields_emitted": False})
    _assert_blocked(review_missing_ack, "ack_missing", "request_fields")

    review_approve_adapter = _gate(
        p40_ready,
        review={**_mapping_review(), "approve_request_adapter_mapping_allowed": True},
    )
    _assert_blocked(review_approve_adapter, "approve_request_adapter_mapping_allowed")

    review_approve_fields = _gate(p40_ready, review={**_mapping_review(), "approve_request_fields_emitted": True})
    _assert_blocked(review_approve_fields, "approve_request_fields_emitted")

    review_approve_generation = _gate(
        p40_ready,
        review={**_mapping_review(), "approve_generation_request_patch_allowed": True},
    )
    _assert_blocked(review_approve_generation, "approve_generation_request_patch_allowed")

    review_approve_launch = _gate(p40_ready, review={**_mapping_review(), "approve_training_launch_allowed": True})
    _assert_blocked(review_approve_launch, "approve_training_launch_allowed")

    failure_history = _gate(
        p40_ready,
        failure_history=[{"reason": "mapping_schema_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    rollback_history = _gate(p40_ready, rollback_history=[{"kind": "mapping_rollback", "rollback_required": True}])
    _assert_blocked(rollback_history, "rollback_history")

    closed_failure = _gate(
        p40_ready,
        failure_history=[{"reason": "closed_mapping_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p41_request_adapter_mapping_boundary_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p40_missing": _summary(p40_missing),
        "p40_not_ready": _summary(p40_not_ready),
        "p40_decision_mismatch": _summary(p40_decision_mismatch),
        "p40_request_adapter": _summary(p40_request_adapter),
        "p40_post_fields": _summary(p40_post_fields),
        "missing_source": _summary(missing_source),
        "missing_digest": _summary(missing_digest),
        "not_report_only": _summary(not_report_only),
        "missing_later_contract": _summary(missing_later_contract),
        "missing_section": _summary(missing_section),
        "missing_inventory": _summary(missing_inventory),
        "mapping_enabled": _summary(mapping_enabled),
        "adapter_registered": _summary(adapter_registered),
        "request_fields": _summary(request_fields),
        "generation_patch": _summary(generation_patch),
        "config_patch": _summary(config_patch),
        "review_missing_reviewer": _summary(review_missing_reviewer),
        "review_missing_reviewed_at": _summary(review_missing_reviewed_at),
        "review_scope_mismatch": _summary(review_scope_mismatch),
        "review_missing_ack": _summary(review_missing_ack),
        "review_approve_adapter": _summary(review_approve_adapter),
        "review_approve_fields": _summary(review_approve_fields),
        "review_approve_generation": _summary(review_approve_generation),
        "review_approve_launch": _summary(review_approve_launch),
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p40: dict[str, Any] | None,
    *,
    evidence: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _mapping_review() if review is ... else review
    return build_v5_request_adapter_mapping_boundary(
        p40_product_ui_exposure_boundary=p40,
        request_adapter_mapping_evidence=_mapping_evidence() if evidence is None else evidence,
        request_adapter_mapping_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p40_ready() -> dict[str, Any]:
    return build_v5_product_ui_exposure_readiness_boundary(
        p39_controlled_rollout_policy_gate=_p39_ready(),
        product_ui_exposure_readiness_evidence=_ui_evidence(),
        product_ui_exposure_review=_ui_review(),
    )


def _mapping_evidence() -> dict[str, Any]:
    sections = [
        "schema_inventory",
        "generation_request_boundary",
        "config_adapter_boundary",
        "no_request_fields_boundary",
        "rollback_policy",
        "observability_policy",
    ]
    return {
        "evidence_id": "request_adapter_mapping_boundary_v0",
        "evidence_version": "v0",
        "ok": True,
        "request_adapter_mapping_boundary_evidence_ready": True,
        "report_only": True,
        "boundary_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_later_adapter_contract": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": sections,
        "available_sections": sections,
        "request_adapter_mapping_inventory": [
            {"mapping_id": "native_update", "mapping_enabled": False, "adapter_registered": False},
            {"mapping_id": "lora_fused", "mapping_enabled": False, "adapter_registered": False},
        ],
        "sha256": "sha256:p41:mapping:ready",
        "artifact_digest": "sha256:p41:mapping:ready",
        "source": "temp/turbocore_v5_p41_mapping.json",
        **_safe_row_flags(),
    }


def _mapping_review(approve: bool = True) -> dict[str, Any]:
    return {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "request_adapter_mapping_boundary",
        "approve_request_adapter_mapping_boundary_evidence": approve,
        "approve_training_launch_allowed": False,
        "approve_auto_launch_allowed": False,
        "approve_runs_dispatched": False,
        "approve_default_training_path_enabled": False,
        "approve_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
        "approve_ui_exposure_allowed": False,
        "approve_product_ui_exposure_allowed": False,
        "approve_request_adapter_mapping_allowed": False,
        "approve_request_fields_emitted": False,
        "approve_request_adapter_registered": False,
        "approve_generation_request_patch_allowed": False,
        "approve_config_adapter_patch_allowed": False,
        "approve_rollout_authorization_allowed": False,
        "acknowledge_p40_product_ui_boundary_recorded": True,
        "acknowledge_default_off_boundary": True,
        "acknowledge_no_training_launch": True,
        "acknowledge_no_ui_exposure": True,
        "acknowledge_no_request_adapter_mapping": True,
        "acknowledge_no_request_fields_emitted": True,
        "acknowledge_no_generation_request_patch": True,
        "acknowledge_no_default_or_auto_rollout": True,
        "acknowledge_mapping_evidence_replayable": True,
        "acknowledge_later_adapter_contract_required": True,
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
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "generation_request_patch_allowed": False,
        "config_adapter_patch_allowed": False,
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
    assert report["post_p41_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["request_adapter_mapping_boundary_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["request_adapter_mapping_boundary_ready"] is False, report
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
        "boundary_ready": bool(report.get("request_adapter_mapping_boundary_ready", False)),
        "review_signed": bool(report.get("request_adapter_mapping_review_signed", False)),
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
