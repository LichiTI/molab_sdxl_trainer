"""Smoke checks for V5-P39 controlled rollout policy evidence gate."""

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
from core.turbocore_v5_packaging_observability_evidence_gate import (  # noqa: E402
    build_v5_packaging_observability_evidence_gate,
)
from lulynx_trainer.turbocore_v5_packaging_observability_evidence_gate_smoke import (  # noqa: E402
    _packages,
    _p37_ready,
    _report,
    _telemetry_rows,
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
    "ui_entry_enabled",
    "ready_for_ui",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "rollout_authorization_allowed",
)
READY_DECISION = "controlled_rollout_policy_evidence_recorded_default_off"
BLOCKED_DECISION = "controlled_rollout_policy_evidence_blocked_default_off"
HOLD_DECISION = "controlled_rollout_policy_hold_for_signed_review_default_off"
REJECTED_DECISION = "controlled_rollout_policy_rejected_default_off"


def run_smoke() -> dict[str, Any]:
    p38_ready = _p38_ready()
    ready = _gate(p38_ready)
    assert ready["ok"] is True, ready
    assert ready["controlled_rollout_policy_evidence_ready"] is True, ready
    assert ready["controlled_rollout_policy_review_signed"] is True, ready
    assert ready["decision"] == READY_DECISION, ready
    assert ready["post_p39_request_fields"] == {}, ready
    _assert_default_off(ready)

    missing_review = _gate(p38_ready, review=None)
    _assert_hold(missing_review, "review", "missing")

    rejected_review = _gate(p38_ready, review=_policy_review(approve=False))
    assert rejected_review["ok"] is True, rejected_review
    assert rejected_review["decision"] == REJECTED_DECISION, rejected_review
    assert rejected_review["rejected_for_default_off_hold"] is True, rejected_review
    assert rejected_review["rollback_required"] is True, rejected_review
    assert rejected_review["controlled_rollout_policy_evidence_ready"] is False, rejected_review
    _assert_default_off(rejected_review)

    p38_missing = _gate(None)
    _assert_blocked(p38_missing, "p38", "missing")

    p38_not_ready = _gate({**p38_ready, "ok": False, "packaging_observability_ready": False})
    _assert_blocked(p38_not_ready, "p38", "not_ready")

    p38_decision_mismatch = _gate({**p38_ready, "decision": "wrong"})
    _assert_blocked(p38_decision_mismatch, "p38", "not_ready")

    p38_unsafe_ui = _gate({**p38_ready, "ui_exposure_allowed": True})
    _assert_blocked(p38_unsafe_ui, "ui_exposure_allowed")

    p38_unsafe_post_fields = _gate({**p38_ready, "post_p38_request_fields": {"bad": True}})
    _assert_blocked(p38_unsafe_post_fields, "post_p38_request_fields")

    policy_missing_source = _gate(p38_ready, policy=_without(_policy_evidence(), "source"))
    _assert_blocked(policy_missing_source, "policy_source_missing")

    policy_missing_digest = _gate(p38_ready, policy=_without_many(_policy_evidence(), "sha256", "policy_digest"))
    _assert_blocked(policy_missing_digest, "policy_digest_missing")

    policy_not_manual_only = _gate(p38_ready, policy={**_policy_evidence(), "manual_only": False})
    _assert_blocked(policy_not_manual_only, "manual_only")

    policy_not_report_only = _gate(p38_ready, policy={**_policy_evidence(), "report_only": False})
    _assert_blocked(policy_not_report_only, "report_only")

    policy_missing_owner_approval = _gate(
        p38_ready,
        policy={**_policy_evidence(), "requires_explicit_owner_approval": False},
    )
    _assert_blocked(policy_missing_owner_approval, "owner_approval")

    policy_missing_operator_opt_in = _gate(
        p38_ready,
        policy={**_policy_evidence(), "requires_explicit_operator_opt_in": False},
    )
    _assert_blocked(policy_missing_operator_opt_in, "operator_opt_in")

    policy_missing_rollback = _gate(p38_ready, policy=_without(_policy_evidence(), "rollback_policy"))
    _assert_blocked(policy_missing_rollback, "rollback_policy")

    policy_missing_monitoring = _gate(p38_ready, policy=_without(_policy_evidence(), "monitoring_policy"))
    _assert_blocked(policy_missing_monitoring, "monitoring_policy")

    policy_unsafe_default_rollout = _gate(p38_ready, policy={**_policy_evidence(), "default_rollout_allowed": True})
    _assert_blocked(policy_unsafe_default_rollout, "default_rollout_allowed")

    policy_unsafe_authorization = _gate(
        p38_ready,
        policy={**_policy_evidence(), "rollout_authorization_allowed": True},
    )
    _assert_blocked(policy_unsafe_authorization, "rollout_authorization_allowed")

    policy_unsafe_request_fields = _gate(
        p38_ready,
        policy={**_policy_evidence(), "request_adapter_fields": {"bad": True}},
    )
    _assert_blocked(policy_unsafe_request_fields, "request_adapter_fields")

    review_missing_reviewer = _gate(p38_ready, review={**_policy_review(), "reviewer": ""})
    _assert_blocked(review_missing_reviewer, "reviewer")

    review_missing_reviewed_at = _gate(p38_ready, review={**_policy_review(), "reviewed_at": ""})
    _assert_blocked(review_missing_reviewed_at, "reviewed_at")

    review_scope_mismatch = _gate(p38_ready, review={**_policy_review(), "requested_scope": "wrong"})
    _assert_blocked(review_scope_mismatch, "scope")

    review_missing_ack = _gate(
        p38_ready,
        review={**_policy_review(), "acknowledge_policy_evidence_replayable": False},
    )
    _assert_blocked(review_missing_ack, "ack_missing", "policy_evidence_replayable")

    review_unsafe_approve_ui = _gate(p38_ready, review={**_policy_review(), "approve_ui_exposure_allowed": True})
    _assert_blocked(review_unsafe_approve_ui, "approve_ui_exposure_allowed")

    review_unsafe_approve_request_adapter = _gate(
        p38_ready,
        review={**_policy_review(), "approve_request_adapter_mapping_allowed": True},
    )
    _assert_blocked(review_unsafe_approve_request_adapter, "approve_request_adapter_mapping_allowed")

    review_unsafe_approve_launch = _gate(
        p38_ready,
        review={**_policy_review(), "approve_training_launch_allowed": True},
    )
    _assert_blocked(review_unsafe_approve_launch, "approve_training_launch_allowed")

    review_unsafe_approve_rollout = _gate(
        p38_ready,
        review={**_policy_review(), "approve_default_rollout_allowed": True},
    )
    _assert_blocked(review_unsafe_approve_rollout, "approve_default_rollout_allowed")

    failure_history = _gate(
        p38_ready,
        failure_history=[{"reason": "policy_observability_gap", "open": True, "severity": "high"}],
    )
    _assert_blocked(failure_history, "failure_history")

    rollback_history = _gate(
        p38_ready,
        rollback_history=[{"kind": "policy_rollback", "rollback_required": True}],
    )
    _assert_blocked(rollback_history, "rollback_history")

    closed_failure = _gate(
        p38_ready,
        failure_history=[{"reason": "closed_policy_warning", "status": "closed", "severity": "high"}],
    )
    assert closed_failure["ok"] is True, closed_failure
    assert closed_failure["decision"] == READY_DECISION, closed_failure
    _assert_default_off(closed_failure)

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p39_controlled_rollout_policy_evidence_gate_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_review": _summary(missing_review),
        "rejected_review": _summary(rejected_review),
        "p38_missing": _summary(p38_missing),
        "p38_not_ready": _summary(p38_not_ready),
        "p38_decision_mismatch": _summary(p38_decision_mismatch),
        "p38_unsafe_ui": _summary(p38_unsafe_ui),
        "p38_unsafe_post_fields": _summary(p38_unsafe_post_fields),
        "policy_missing_source": _summary(policy_missing_source),
        "policy_missing_digest": _summary(policy_missing_digest),
        "policy_not_manual_only": _summary(policy_not_manual_only),
        "policy_not_report_only": _summary(policy_not_report_only),
        "policy_missing_owner_approval": _summary(policy_missing_owner_approval),
        "policy_missing_operator_opt_in": _summary(policy_missing_operator_opt_in),
        "policy_missing_rollback": _summary(policy_missing_rollback),
        "policy_missing_monitoring": _summary(policy_missing_monitoring),
        "policy_unsafe_default_rollout": _summary(policy_unsafe_default_rollout),
        "policy_unsafe_authorization": _summary(policy_unsafe_authorization),
        "policy_unsafe_request_fields": _summary(policy_unsafe_request_fields),
        "review_missing_reviewer": _summary(review_missing_reviewer),
        "review_missing_reviewed_at": _summary(review_missing_reviewed_at),
        "review_scope_mismatch": _summary(review_scope_mismatch),
        "review_missing_ack": _summary(review_missing_ack),
        "review_unsafe_approve_ui": _summary(review_unsafe_approve_ui),
        "review_unsafe_approve_request_adapter": _summary(review_unsafe_approve_request_adapter),
        "review_unsafe_approve_launch": _summary(review_unsafe_approve_launch),
        "review_unsafe_approve_rollout": _summary(review_unsafe_approve_rollout),
        "failure_history": _summary(failure_history),
        "rollback_history": _summary(rollback_history),
        "closed_failure": _summary(closed_failure),
    }


def _gate(
    p38: dict[str, Any] | None,
    *,
    policy: dict[str, Any] | None = None,
    review: dict[str, Any] | None = ...,
    failure_history: list[Any] | None = None,
    rollback_history: list[Any] | None = None,
) -> dict[str, Any]:
    actual_review = _policy_review() if review is ... else review
    return build_v5_controlled_rollout_policy_evidence_gate(
        p38_packaging_observability_gate=p38,
        controlled_rollout_policy_evidence=_policy_evidence() if policy is None else policy,
        controlled_rollout_policy_review=actual_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p38_ready() -> dict[str, Any]:
    return build_v5_packaging_observability_evidence_gate(
        p37_coverage_gate=_p37_ready(),
        packaging_evidence=_packages(),
        observability_evidence=_telemetry_rows(),
        report_evidence=[_report()],
    )


def _policy_evidence() -> dict[str, Any]:
    return {
        "policy_id": "controlled_rollout_policy_internal_v0",
        "policy_version": "v0",
        "ok": True,
        "controlled_rollout_policy_evidence_ready": True,
        "report_only": True,
        "manual_only": True,
        "internal_only": True,
        "requires_explicit_owner_approval": True,
        "requires_explicit_operator_opt_in": True,
        "default_off": True,
        "request_adapter_off": True,
        "required_sections": [
            "scope",
            "default_off_boundary",
            "operator_opt_in",
            "rollback_policy",
            "monitoring_policy",
        ],
        "available_sections": [
            "scope",
            "default_off_boundary",
            "operator_opt_in",
            "rollback_policy",
            "monitoring_policy",
        ],
        "rollback_policy_ready": True,
        "rollback_policy": {"ready": True, "manual_only": True},
        "monitoring_policy_ready": True,
        "monitoring_policy": {"ready": True, "report_only": True},
        "sha256": "sha256:p39:policy:ready",
        "policy_digest": "sha256:p39:policy:ready",
        "source": "temp/turbocore_v5_p39_policy.json",
        **_safe_row_flags(),
    }


def _policy_review(approve: bool = True) -> dict[str, Any]:
    return {
        "reviewer": "owner",
        "reviewed_at": "2026-06-02T00:00:00Z",
        "requested_scope": "controlled_rollout_policy_evidence_gate",
        "approve_controlled_rollout_policy_evidence": approve,
        "approve_training_launch_allowed": False,
        "approve_auto_launch_allowed": False,
        "approve_runs_dispatched": False,
        "approve_default_training_path_enabled": False,
        "approve_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
        "approve_ui_exposure_allowed": False,
        "approve_request_adapter_mapping_allowed": False,
        "approve_request_fields_emitted": False,
        "approve_rollout_authorization_allowed": False,
        "acknowledge_p38_packaging_observability_ready": True,
        "acknowledge_default_off_boundary": True,
        "acknowledge_no_training_launch": True,
        "acknowledge_no_ui_exposure": True,
        "acknowledge_no_request_adapter_mapping": True,
        "acknowledge_no_default_or_auto_rollout": True,
        "acknowledge_policy_evidence_replayable": True,
        "acknowledge_rollback_policy_ready": True,
        "acknowledge_observability_policy_ready": True,
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
    assert report["post_p39_request_fields"] == {}, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["controlled_rollout_policy_evidence_ready"] is False, report
    assert report["decision"] == BLOCKED_DECISION, report
    _assert_default_off(report)
    _assert_reason_fragments(report, *fragments)


def _assert_hold(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["controlled_rollout_policy_evidence_ready"] is False, report
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
        "controlled_rollout_policy_evidence_ready": bool(
            report.get("controlled_rollout_policy_evidence_ready", False)
        ),
        "review_signed": bool(report.get("controlled_rollout_policy_review_signed", False)),
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
