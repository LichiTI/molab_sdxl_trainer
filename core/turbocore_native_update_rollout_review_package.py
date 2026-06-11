"""Native-update rollout review package for TurboCore optimizer dispatch.

This package is deliberately narrower than the V5 owner-review package: it
records native-update evidence and default-off invariants, but it never emits
request fields or authorizes product/UI exposure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_controlled_rollout_policy_evidence_gate_utils import (  # noqa: E402
    as_dict as _as_dict,
    dedupe as _dedupe,
    event_list as _event_list,
    history_summary as _history_summary,
    string_list as _string_list,
)


NATIVE_UPDATE_REVIEW_SCOPE = "native_update_rollout_review_package"
HOLD_DECISION = "native_update_rollout_review_hold_for_owner_review_default_off"
READY_DECISION = "native_update_rollout_review_recorded_default_off"
BLOCKED_DECISION = "native_update_rollout_review_blocked_default_off"
REJECTED_DECISION = "native_update_rollout_review_rejected_default_off"
REQUIRED_REVIEW_ACKS = (
    "acknowledge_representative_performance_ready",
    "acknowledge_training_loop_dispatch_smoke_ready",
    "acknowledge_native_kernel_launched_in_explicit_smoke",
    "acknowledge_default_off_boundary",
    "acknowledge_no_product_training_dispatch",
    "acknowledge_no_ui_exposure",
    "acknowledge_no_request_adapter_or_schema_exposure",
    "acknowledge_later_integration_contract_required",
)
UNSAFE_TRUE_FIELDS = (
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
    "schema_exposure_allowed",
    "rollout_authorization_allowed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_approval_request_fields",
    "post_native_update_request_fields",
    "request_adapter",
    "request_adapter_fields",
    "request_schema_fields",
    "launch_request",
    "training_request",
    "ui_route_registration",
    "launcher_menu_entry",
    "webui_tab_entry",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = (
    "approve_product_training_dispatch",
    "approve_training_launch_allowed",
    "approve_default_training_path_enabled",
    "approve_training_path_enabled",
    "approve_default_rollout_allowed",
    "approve_auto_rollout_allowed",
    "approve_ui_exposure_allowed",
    "approve_ready_for_ui",
    "approve_request_adapter_mapping_allowed",
    "approve_request_fields_emitted",
    "approve_schema_exposure_allowed",
)


def build_native_update_rollout_review_package(
    *,
    readiness_report: Mapping[str, Any] | None = None,
    performance_matrix: Mapping[str, Any] | None = None,
    owner_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build a default-off review package for native-update rollout evidence."""

    readiness = _readiness_summary(_as_dict(readiness_report))
    performance = _performance_summary(_as_dict(performance_matrix))
    review = _review_summary(_as_dict(owner_review))
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    evidence_blockers = _evidence_blockers(
        readiness=readiness,
        performance=performance,
        failure_events=failure_events,
        rollback_events=rollback_events,
    )
    review_blockers = _review_blockers(review)
    if evidence_blockers:
        decision = BLOCKED_DECISION
    elif not review.get("present"):
        decision = HOLD_DECISION
    elif review_blockers:
        decision = BLOCKED_DECISION
    elif review.get("approve_native_update_rollout_review_package") is True:
        decision = READY_DECISION
    else:
        decision = REJECTED_DECISION

    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == HOLD_DECISION:
        blockers.append("native_update_rollout_owner_review_not_signed")
    blockers = _dedupe(blockers)
    evidence_ready = not evidence_blockers
    review_recorded = decision in {READY_DECISION, REJECTED_DECISION}
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_rollout_review_package_v0",
        "gate": "native_update_rollout_review_package",
        "ok": evidence_ready and decision != BLOCKED_DECISION,
        "evidence_package_ready": evidence_ready,
        "ready_for_owner_review": evidence_ready,
        "owner_review_action_required": decision == HOLD_DECISION,
        "owner_review_recorded": review_recorded,
        "native_update_rollout_review_recorded": decision == READY_DECISION,
        "native_update_rollout_review_package_ready": decision == READY_DECISION,
        "manual_review_required": True,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "default_behavior_changed": False,
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
        "schema_exposure_allowed": False,
        "rollout_authorization_allowed": False,
        "post_native_update_request_fields": {},
        "readiness_summary": readiness,
        "performance_matrix_summary": performance,
        "owner_review": review,
        "owner_review_template": _owner_review_template(readiness, performance),
        "progress_gates": {
            "readiness_report_present": bool(readiness.get("present")),
            "readiness_report_ready": bool(readiness.get("ready")),
            "performance_matrix_present": bool(performance.get("present")),
            "representative_performance_ready": bool(performance.get("performance_gate_ready")),
            "training_loop_dispatch_smoke_ready": bool(readiness.get("training_loop_dispatch_smoke_ok")),
            "native_kernel_launched": bool(readiness.get("native_kernel_launched")),
            "default_off_boundary_confirmed": bool(readiness.get("default_off")),
            "request_ui_schema_exposure_blocked": bool(readiness.get("request_ui_schema_exposure_blocked")),
            "signed_owner_review_present": bool(review.get("present")),
        },
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, evidence_ready),
        "recommended_next_step": _recommended_next_step(decision, evidence_ready),
        "notes": [
            "This package records native-update review evidence only.",
            "It does not emit request fields, expose UI/schema, launch training, or change defaults.",
            "A later explicit integration contract is required before any product training dispatch exposure.",
        ],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload.setdefault("_source_path", str(source))
        payload.setdefault("_source_digest", _digest_payload(payload))
        return payload
    return {}


def _readiness_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    sections = _as_dict(report.get("sections"))
    smoke = _as_dict(sections.get("native_update_training_loop_dispatch_smoke"))
    scorecard = _as_dict(sections.get("native_update_promotion_scorecard"))
    perf_gate = _as_dict(scorecard.get("performance_gate"))
    blockers = _dedupe(
        _string_list(summary.get("native_update_performance_blockers"))
        + _unsafe_claims(report, "readiness_report")
        + _unsafe_claims(summary, "readiness_summary")
        + _unsafe_claims(smoke, "training_loop_dispatch_smoke")
    )
    return {
        "present": bool(report),
        "source": str(report.get("_source_path") or ""),
        "digest": str(report.get("_source_digest") or _digest_payload(report) if report else ""),
        "ok": summary.get("ok") is True,
        "ready": bool(
            report
            and summary.get("ok") is True
            and summary.get("native_update_representative_performance_ready") is True
            and not _string_list(summary.get("native_update_performance_blockers"))
            and summary.get("native_update_training_executor_available") is True
            and summary.get("native_update_training_loop_dispatch_smoke_ok") is True
            and summary.get("native_update_native_kernel_launched") is True
            and summary.get("native_training_path_locked") is True
            and summary.get("ready_for_ui") is False
            and smoke.get("training_path_enabled") is False
            and smoke.get("default_behavior_changed") is False
            and not blockers
        ),
        "native_update_representative_performance_ready": summary.get(
            "native_update_representative_performance_ready"
        )
        is True,
        "native_update_performance_blockers": _string_list(summary.get("native_update_performance_blockers")),
        "training_executor_available": summary.get("native_update_training_executor_available") is True,
        "training_loop_dispatch_smoke_ok": summary.get("native_update_training_loop_dispatch_smoke_ok") is True,
        "native_kernel_launched": summary.get("native_update_native_kernel_launched") is True,
        "native_training_path_locked": summary.get("native_training_path_locked") is True,
        "ready_for_ui": summary.get("ready_for_ui"),
        "default_off": bool(
            summary.get("native_training_path_locked") is True
            and summary.get("ready_for_ui") is False
            and smoke.get("training_path_enabled") is False
            and smoke.get("default_behavior_changed") is False
        ),
        "request_ui_schema_exposure_blocked": not bool(
            summary.get("ready_for_ui") is True
            or report.get("request_adapter_mapping_allowed") is True
            or report.get("request_fields_emitted") is True
            or report.get("schema_exposure_allowed") is True
        ),
        "representative_speedup": perf_gate.get("representative_end_to_end_speedup"),
        "remaining_native_update_promotion_blockers": _string_list(summary.get("native_update_promotion_blockers")),
        "recommended_next_step": str(summary.get("recommended_next_step") or ""),
        "blocked_reasons": blockers,
    }


def _performance_summary(matrix: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(matrix.get("summary"))
    gate_summary = _as_dict(summary.get("native_update_performance_gate"))
    report = _as_dict(matrix.get("native_update_performance_report"))
    gate = _as_dict(report.get("performance_gate"))
    evidence = _as_dict(gate.get("evidence"))
    training = _as_dict(evidence.get("training_matrix"))
    optimizer = _as_dict(evidence.get("optimizer_microbenchmark"))
    ctx_free = _as_dict(summary.get("native_dispatch_ctx_sync_free_comparison"))
    blocked = _dedupe(
        _string_list(gate_summary.get("blocked_reasons"))
        + _string_list(gate.get("blocked_reasons"))
        + _unsafe_claims(matrix, "performance_matrix")
        + _unsafe_claims(report, "native_update_performance_report")
    )
    performance_gate_ready = bool(gate_summary.get("ready") is True or gate.get("representative_performance_gate_ready") is True)
    report_only = bool(report.get("training_dispatch") is False and report.get("runtime_dispatch_allowed") is False)
    end_to_end_speedup = _float_or_none(training.get("end_to_end_speedup"))
    required_speedup = _float_or_none(gate.get("required_end_to_end_speedup")) or 1.03
    semantic_blockers = _performance_semantic_blockers(
        matrix=matrix,
        summary=summary,
        performance_gate_ready=performance_gate_ready,
        report_only=report_only,
        optimizer=optimizer,
        training=training,
        end_to_end_speedup=end_to_end_speedup,
        required_speedup=required_speedup,
    )
    blocked = _dedupe(blocked + semantic_blockers)
    return {
        "present": bool(matrix),
        "source": str(matrix.get("_source_path") or matrix.get("matrix_summary_path") or ""),
        "digest": str(matrix.get("_source_digest") or _digest_payload(matrix) if matrix else ""),
        "ok": bool(matrix.get("run") is True and summary.get("all_success") is True),
        "ready": bool(
            matrix
            and matrix.get("run") is True
            and summary.get("all_success") is True
            and performance_gate_ready
            and report_only
            and optimizer.get("present") is True
            and training.get("representative_steps", 0)
            and end_to_end_speedup is not None
            and end_to_end_speedup >= required_speedup
            and not blocked
        ),
        "case_count": len(matrix.get("cases", []) or []),
        "executed_count": int(summary.get("executed_count", 0) or 0),
        "performance_gate_ready": performance_gate_ready,
        "blocked_reasons": blocked,
        "report_only_runtime_dispatch_off": report_only,
        "representative_native_case": str(training.get("native_case") or ""),
        "representative_steps": int(training.get("representative_steps", 0) or 0),
        "representative_end_to_end_speedup": end_to_end_speedup,
        "required_end_to_end_speedup": required_speedup,
        "optimizer_evidence_present": optimizer.get("present") is True,
        "optimizer_evidence_quality": str(optimizer.get("evidence_quality") or ""),
        "optimizer_best_speedup_vs_baseline": optimizer.get("best_speedup_vs_baseline"),
        "ctx_sync_free_case": str(ctx_free.get("ctx_sync_free_case") or ""),
        "ctx_sync_free_speedup_vs_baseline": ctx_free.get("ctx_sync_free_speedup_vs_baseline"),
        "ctx_sync_free_speedup_vs_context_sync_native": ctx_free.get("ctx_sync_free_speedup_vs_context_sync_native"),
        "ctx_sync_free_representative_candidate_ready": ctx_free.get("representative_candidate_ready") is True,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_native_update_rollout_review_package": review.get(
            "approve_native_update_rollout_review_package"
        )
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _performance_semantic_blockers(
    *,
    matrix: Mapping[str, Any],
    summary: Mapping[str, Any],
    performance_gate_ready: bool,
    report_only: bool,
    optimizer: Mapping[str, Any],
    training: Mapping[str, Any],
    end_to_end_speedup: float | None,
    required_speedup: float,
) -> list[str]:
    blocked: list[str] = []
    if matrix.get("run") is not True:
        blocked.append("native_update_rollout_performance_matrix_not_run")
    if summary.get("all_success") is not True:
        blocked.append("native_update_rollout_performance_matrix_not_all_success")
    if not performance_gate_ready:
        blocked.append("native_update_rollout_representative_performance_gate_not_ready")
    if not report_only:
        blocked.append("native_update_rollout_performance_report_training_dispatch_or_runtime_dispatch_on")
    if optimizer.get("present") is not True:
        blocked.append("native_update_rollout_optimizer_microbenchmark_missing")
    if not int(training.get("representative_steps", 0) or 0):
        blocked.append("native_update_rollout_representative_training_steps_missing")
    if end_to_end_speedup is None:
        blocked.append("native_update_rollout_end_to_end_speedup_missing")
    elif end_to_end_speedup < required_speedup:
        blocked.append("native_update_rollout_end_to_end_speedup_below_threshold")
    return _dedupe(blocked)


def _evidence_blockers(
    *,
    readiness: Mapping[str, Any],
    performance: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not readiness.get("present"):
        blocked.append("native_update_rollout_readiness_report_missing")
    elif not readiness.get("ready"):
        blocked.append("native_update_rollout_readiness_report_not_ready")
        blocked.extend(_string_list(readiness.get("native_update_performance_blockers")))
        blocked.extend(_string_list(readiness.get("blocked_reasons")))
    if not performance.get("present"):
        blocked.append("native_update_rollout_performance_matrix_missing")
    elif not performance.get("ready"):
        blocked.append("native_update_rollout_performance_matrix_not_ready")
        blocked.extend(_string_list(performance.get("blocked_reasons")))
    for event in failure_events:
        blocked.append(f"native_update_rollout_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"native_update_rollout_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("native_update_rollout_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("native_update_rollout_reviewed_at_missing")
    if review.get("requested_scope") != NATIVE_UPDATE_REVIEW_SCOPE:
        blocked.append("native_update_rollout_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"native_update_rollout_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"native_update_rollout_review_ack_missing:{field}")
    return _dedupe(blocked)


def _owner_review_template(readiness: Mapping[str, Any], performance: Mapping[str, Any]) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": NATIVE_UPDATE_REVIEW_SCOPE,
        "approve_native_update_rollout_review_package": False,
    }
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_evidence"] = {
        "readiness_digest": readiness.get("digest"),
        "performance_digest": performance.get("digest"),
        "representative_end_to_end_speedup": performance.get("representative_end_to_end_speedup"),
        "ctx_sync_free_case": performance.get("ctx_sync_free_case"),
    }
    return template


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"native_update_rollout_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"native_update_rollout_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _allowed_next_actions(decision: str, evidence_ready: bool) -> list[str]:
    if decision == READY_DECISION:
        return [
            "archive_native_update_review_package",
            "prepare_later_default_off_integration_contract",
        ]
    if evidence_ready:
        return ["collect_signed_owner_review_without_request_ui_schema_exposure"]
    return ["refresh_readiness_and_performance_evidence"]


def _recommended_next_step(decision: str, evidence_ready: bool) -> str:
    if decision == READY_DECISION:
        return "archive native-update review evidence and prepare the next explicit default-off integration contract"
    if evidence_ready:
        return "record owner review while keeping native-update product dispatch, request fields, and UI exposure disabled"
    return "refresh native-update readiness and representative performance evidence before owner review"


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {k: v for k, v in value.items() if not str(k).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build native-update rollout review evidence package")
    parser.add_argument("--readiness-report", required=True)
    parser.add_argument("--performance-matrix", required=True)
    parser.add_argument("--owner-review")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    package = build_native_update_rollout_review_package(
        readiness_report=load_json(args.readiness_report),
        performance_matrix=load_json(args.performance_matrix),
        owner_review=load_json(args.owner_review) if args.owner_review else None,
    )
    text = json.dumps(package, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if package.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
