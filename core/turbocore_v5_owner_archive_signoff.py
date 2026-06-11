"""Owner archive signoff contract for TurboCore V5-P33.

This report-only contract consumes a P32 owner package replay and an optional
owner archive review. It records whether the evidence package can be archived
for the next manual stage, while keeping training launch, default rollout, and
request-adapter mapping disabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_owner_review_evidence_package import load_json


P32_READY_DECISION = "p31_collector_replay_owner_package_ready_default_off"
P31_READY_DECISION = "longer_replicate_manual_run_audit_ready_default_off"
P26_READY_DECISION = "longer_replicate_failure_history_review_ready"
P27_APPROVED_DECISION = "signed_next_stage_review_recorded_default_off"
P29_READY_DECISION = "owner_next_stage_package_ready_default_off"
P33_APPROVED_DECISION = "p32_final_owner_archive_signoff_ready_default_off"
P33_HOLD_DECISION = "p32_final_owner_archive_hold_for_signed_signoff_default_off"
P33_REJECTED_DECISION = "p32_final_owner_archive_signoff_rejected_default_off"
P33_BLOCKED_DECISION = "p32_final_owner_archive_signoff_blocked_default_off"
P33_SCOPE = "p32_final_owner_archive_signoff"


def build_v5_owner_archive_signoff(
    *,
    p32_owner_package: Mapping[str, Any] | None = None,
    owner_archive_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a P33 owner archive signoff without launching training."""

    package = _as_dict(p32_owner_package)
    review = _as_dict(owner_archive_review)
    package_summary = _p32_summary(package)
    progress = _progress_gates(package_summary, review)
    decision = _decision(progress, review)
    blockers = _blocked_reasons(progress, decision, package_summary)
    decision_ready = bool(review) and not blockers
    approved = decision_ready and decision == P33_APPROVED_DECISION
    rejected = decision_ready and decision == P33_REJECTED_DECISION
    waiting_for_review = (
        package_summary["ready"]
        and not bool(review)
        and "v5_p33_signed_owner_archive_review_missing" in blockers
    )
    return {
        "schema_version": 1,
        "package": "turbocore_v5_p32_final_owner_archive_signoff_v0",
        "gate": "v5_p32_final_owner_archive_signoff",
        "ok": decision_ready,
        "decision_record_ready": decision_ready,
        "owner_archive_signoff_recorded": decision_ready,
        "owner_archive_signoff_signed": decision_ready,
        "owner_archive_package_ready": approved,
        "archive_package_ready": approved,
        "approved_for_owner_archive": approved,
        "approved_for_next_manual_stage_archive": approved,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "ready_for_signed_owner_archive_review": waiting_for_review,
        "manual_review_required": True,
        "decision": decision,
        "package_decision": decision,
        "gate_decision": decision,
        "default_behavior_changed": False,
        "training_launch_allowed": False,
        "auto_launch_allowed": False,
        "runs_dispatched": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_archive_request_fields": {},
        "p32_owner_package_summary": package_summary,
        "owner_archive_review": _review_summary(review),
        "owner_archive_review_template": _review_template(package_summary),
        "progress_gates": progress,
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "recommended_next_step": _recommended_next_step(approved, rejected, waiting_for_review, blockers),
        "notes": [
            "P33 records owner archive signoff only; it does not launch training.",
            "Archive approval does not enable default rollout or request-adapter mapping.",
            "Rejected signoff keeps the evidence chain default-off for repair or rollback.",
        ],
    }


def _p32_summary(package: Mapping[str, Any]) -> dict[str, Any]:
    p31 = _as_dict(package.get("p31_manual_run_audit_summary"))
    p29 = _as_dict(package.get("p29_owner_next_stage_package"))
    p27 = _as_dict(package.get("p27_decision"))
    p26 = _as_dict(package.get("p26_gate"))
    p28 = _as_dict(package.get("p28_collector_bundle"))
    p29_decision = str(p29.get("decision") or p29.get("gate_decision") or p29.get("package_decision") or "")
    p27_decision = str(p27.get("decision") or p27.get("gate_decision") or p27.get("next_stage_review_decision") or "")
    p26_decision = str(p26.get("decision") or p26.get("gate_decision") or p26.get("rollout_review_decision") or "")
    return {
        "present": bool(package),
        "source_path": str(package.get("_source_path") or package.get("source_path") or ""),
        "ok": bool(package.get("ok", False)),
        "p31_collector_replay_ready": bool(package.get("p31_collector_replay_ready", False)),
        "owner_next_stage_package_ready": bool(package.get("owner_next_stage_package_ready", False)),
        "decision": str(package.get("decision") or package.get("gate_decision") or package.get("package_decision") or ""),
        "manual_review_required": bool(package.get("manual_review_required", False)),
        "training_launch_allowed": bool(package.get("training_launch_allowed", True)),
        "auto_launch_allowed": bool(package.get("auto_launch_allowed", True)),
        "runs_dispatched": bool(package.get("runs_dispatched", True)),
        "default_off": _default_off_confirmed(package),
        "request_adapter_off": _request_adapter_off(package),
        "post_fields_empty": not bool(_as_dict(package.get("post_replay_request_fields"))),
        "p31_summary_present": bool(p31),
        "p31_manual_run_audit_ready": bool(p31.get("manual_run_audit_ready", False)),
        "p31_collector_evidence_ready": bool(p31.get("collector_evidence_ready", False)),
        "p31_decision": str(p31.get("decision") or ""),
        "p29_package_present": bool(p29),
        "p29_package_ready": bool(p29.get("package_ready", False)),
        "p29_ready_for_owner_archive": bool(p29.get("ready_for_owner_archive", False)),
        "p29_decision": p29_decision,
        "p29_default_off": _default_off_confirmed(p29),
        "p29_request_adapter_off": _request_adapter_off(p29),
        "p29_post_fields_empty": not bool(_as_dict(p29.get("post_package_request_fields"))),
        "p27_approved": bool(p27.get("approved_for_next_contract_stage", False)),
        "p27_signed": bool(p27.get("signed_next_stage_review_signed", False))
        and bool(p27.get("signed_next_stage_review_recorded", False)),
        "p27_decision": p27_decision,
        "p27_rejected": bool(p27.get("rejected_for_default_off_hold", False)),
        "p27_rollback_required": bool(p27.get("rollback_required", False)),
        "p26_ready": bool(p26.get("longer_replicate_failure_history_gate_ready", False)),
        "p26_decision": p26_decision,
        "p28_ready": bool(p28.get("longer_replicate_evidence_ready", False)),
        "run_count": int(p28.get("run_count", 0) or 0),
        "ready_run_count": int(p28.get("ready_run_count", 0) or 0),
        "min_speedup": _aggregate_value(p28, "min_speedup"),
        "speedup_spread_ratio": _aggregate_value(p28, "speedup_spread_ratio"),
        "blocked_reasons": _string_list(package.get("blocked_reasons")),
        "ready": _p32_ready(package, p29),
    }


def _p32_ready(package: Mapping[str, Any], p29: Mapping[str, Any]) -> bool:
    return bool(
        package
        and package.get("ok") is True
        and package.get("p31_collector_replay_ready") is True
        and package.get("owner_next_stage_package_ready") is True
        and str(package.get("decision") or package.get("gate_decision") or package.get("package_decision") or "")
        == P32_READY_DECISION
        and package.get("manual_review_required") is True
        and package.get("training_launch_allowed") is False
        and package.get("auto_launch_allowed") is False
        and package.get("runs_dispatched") is False
        and _default_off_confirmed(package)
        and _request_adapter_off(package)
        and not _as_dict(package.get("post_replay_request_fields"))
        and _as_dict(package.get("p31_manual_run_audit_summary")).get("manual_run_audit_ready") is True
        and _as_dict(package.get("p31_manual_run_audit_summary")).get("collector_evidence_ready") is True
        and str(_as_dict(package.get("p31_manual_run_audit_summary")).get("decision") or "") == P31_READY_DECISION
        and _as_dict(package.get("p26_gate")).get("longer_replicate_failure_history_gate_ready") is True
        and str(
            _as_dict(package.get("p26_gate")).get("decision")
            or _as_dict(package.get("p26_gate")).get("gate_decision")
            or _as_dict(package.get("p26_gate")).get("rollout_review_decision")
            or ""
        )
        == P26_READY_DECISION
        and _as_dict(package.get("p27_decision")).get("signed_next_stage_review_signed") is True
        and _as_dict(package.get("p27_decision")).get("signed_next_stage_review_recorded") is True
        and _as_dict(package.get("p27_decision")).get("approved_for_next_contract_stage") is True
        and not bool(_as_dict(package.get("p27_decision")).get("rejected_for_default_off_hold", False))
        and not bool(_as_dict(package.get("p27_decision")).get("rollback_required", False))
        and str(
            _as_dict(package.get("p27_decision")).get("decision")
            or _as_dict(package.get("p27_decision")).get("gate_decision")
            or _as_dict(package.get("p27_decision")).get("next_stage_review_decision")
            or ""
        )
        == P27_APPROVED_DECISION
        and p29.get("ok") is True
        and p29.get("package_ready") is True
        and p29.get("ready_for_owner_archive") is True
        and str(p29.get("decision") or p29.get("gate_decision") or p29.get("package_decision") or "") == P29_READY_DECISION
        and _default_off_confirmed(p29)
        and _request_adapter_off(p29)
        and not _as_dict(p29.get("post_package_request_fields"))
    )


def _progress_gates(package_summary: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p32_package_present": bool(package_summary.get("present", False)),
        "p32_package_ready": bool(package_summary.get("ready", False)),
        "p32_no_blockers": not _string_list(package_summary.get("blocked_reasons")),
        "p32_default_off": bool(package_summary.get("default_off", False)),
        "p32_request_adapter_off": bool(package_summary.get("request_adapter_off", False)),
        "p32_post_fields_empty": bool(package_summary.get("post_fields_empty", False)),
        "p32_training_launch_disabled": not bool(package_summary.get("training_launch_allowed", True)),
        "p32_auto_launch_disabled": not bool(package_summary.get("auto_launch_allowed", True)),
        "p32_runs_not_dispatched": not bool(package_summary.get("runs_dispatched", True)),
        "p29_package_ready": bool(package_summary.get("p29_package_ready", False))
        and bool(package_summary.get("p29_ready_for_owner_archive", False))
        and str(package_summary.get("p29_decision") or "") == P29_READY_DECISION,
        "p29_default_off": bool(package_summary.get("p29_default_off", False)),
        "p29_request_adapter_off": bool(package_summary.get("p29_request_adapter_off", False)),
        "p29_post_fields_empty": bool(package_summary.get("p29_post_fields_empty", False)),
        "p31_ready": bool(package_summary.get("p31_manual_run_audit_ready", False))
        and bool(package_summary.get("p31_collector_evidence_ready", False))
        and str(package_summary.get("p31_decision") or "") == P31_READY_DECISION,
        "p27_approved": bool(package_summary.get("p27_approved", False))
        and bool(package_summary.get("p27_signed", False))
        and str(package_summary.get("p27_decision") or "") == P27_APPROVED_DECISION
        and not bool(package_summary.get("p27_rejected", False))
        and not bool(package_summary.get("p27_rollback_required", False)),
        "p26_ready": bool(package_summary.get("p26_ready", False))
        and str(package_summary.get("p26_decision") or "") == P26_READY_DECISION,
        "p28_ready": bool(package_summary.get("p28_ready", False)),
        "signed_owner_archive_review_present": bool(review),
        "requested_scope_valid": str(review.get("requested_scope") or "") == P33_SCOPE,
        "review_defaults_off": _review_defaults_off(review),
        "review_launch_not_requested": not bool(review.get("approve_training_launch_allowed", False))
        and not bool(review.get("approve_auto_launch_allowed", False))
        and not bool(review.get("approve_runs_dispatched", False)),
        "review_request_adapter_not_requested": not bool(review.get("approve_request_adapter_mapping_allowed", False))
        and not bool(review.get("approve_request_fields_emitted", False)),
        "p32_owner_package_acknowledged": bool(review.get("acknowledge_p32_owner_package_ready", False)),
        "p31_replay_acknowledged": bool(review.get("acknowledge_p31_collector_replay_ready", False)),
        "p29_archive_acknowledged": bool(review.get("acknowledge_p29_owner_package_ready", False)),
        "p27_signed_acknowledged": bool(review.get("acknowledge_signed_p27_review", False)),
        "p26_gate_acknowledged": bool(review.get("acknowledge_p26_gate_ready", False)),
        "p28_evidence_acknowledged": bool(review.get("acknowledge_p28_collector_bundle_ready", False)),
        "default_rollout_acknowledged_off": bool(review.get("acknowledge_default_rollout_disabled", False)),
        "ui_request_adapter_acknowledged_off": bool(
            review.get("acknowledge_no_ui_or_request_adapter", False)
        ),
        "request_adapter_acknowledged_off": bool(review.get("acknowledge_no_request_adapter_mapping", False)),
        "training_launch_acknowledged_off": bool(review.get("acknowledge_no_training_launch", False)),
        "default_auto_rollout_acknowledged_off": bool(
            review.get("acknowledge_default_and_auto_rollout_off", False)
        ),
        "manual_review_only_acknowledged": bool(review.get("acknowledge_manual_review_only", False)),
        "full_chain_acknowledged": bool(review.get("acknowledge_p31_p28_p26_p27_p29_chain_archived", False)),
    }


def _decision(progress: Mapping[str, bool], review: Mapping[str, Any]) -> str:
    if not bool(progress.get("signed_owner_archive_review_present", False)):
        return P33_HOLD_DECISION
    required = (
        "p32_package_ready",
        "p32_no_blockers",
        "p32_default_off",
        "p32_request_adapter_off",
        "p32_post_fields_empty",
        "p32_training_launch_disabled",
        "p32_auto_launch_disabled",
        "p32_runs_not_dispatched",
        "p29_package_ready",
        "p29_default_off",
        "p29_request_adapter_off",
        "p29_post_fields_empty",
        "p31_ready",
        "p27_approved",
        "p26_ready",
        "p28_ready",
        "requested_scope_valid",
        "review_defaults_off",
        "review_launch_not_requested",
        "review_request_adapter_not_requested",
        "p32_owner_package_acknowledged",
        "p31_replay_acknowledged",
        "p29_archive_acknowledged",
        "p27_signed_acknowledged",
        "p26_gate_acknowledged",
        "p28_evidence_acknowledged",
        "default_rollout_acknowledged_off",
        "ui_request_adapter_acknowledged_off",
        "request_adapter_acknowledged_off",
        "training_launch_acknowledged_off",
        "default_auto_rollout_acknowledged_off",
        "manual_review_only_acknowledged",
        "full_chain_acknowledged",
    )
    if any(not bool(progress.get(name, False)) for name in required):
        return P33_BLOCKED_DECISION
    if bool(review.get("approve_final_owner_archive", review.get("approve_owner_archive_package", False))):
        return P33_APPROVED_DECISION
    return P33_REJECTED_DECISION


def _blocked_reasons(
    progress: Mapping[str, bool],
    decision: str,
    package_summary: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if not bool(progress.get("p32_package_present", False)):
        blocked.append("v5_p33_p32_owner_package_missing")
    if not bool(progress.get("p32_package_ready", False)):
        blocked.append("v5_p33_p32_owner_package_not_ready")
        blocked.extend(_string_list(package_summary.get("blocked_reasons")))
    if not bool(progress.get("p32_no_blockers", False)):
        blocked.append("v5_p33_p32_blockers_present")
        blocked.extend(_string_list(package_summary.get("blocked_reasons")))
    if not bool(progress.get("p32_default_off", False)):
        blocked.append("v5_p33_p32_default_off_violation")
    if not bool(progress.get("p32_request_adapter_off", False)):
        blocked.append("v5_p33_p32_request_adapter_violation")
    if not bool(progress.get("p32_post_fields_empty", False)):
        blocked.append("v5_p33_p32_post_fields_present")
    if not bool(progress.get("p32_training_launch_disabled", False)):
        blocked.append("v5_p33_p32_training_launch_allowed_violation")
    if not bool(progress.get("p32_auto_launch_disabled", False)):
        blocked.append("v5_p33_p32_auto_launch_allowed_violation")
    if not bool(progress.get("p32_runs_not_dispatched", False)):
        blocked.append("v5_p33_p32_runs_dispatched_violation")
    if not bool(progress.get("p29_package_ready", False)):
        blocked.append("v5_p33_p29_owner_package_not_ready")
    if not bool(progress.get("p29_default_off", False)):
        blocked.append("v5_p33_p29_default_off_violation")
    if not bool(progress.get("p29_request_adapter_off", False)):
        blocked.append("v5_p33_p29_request_adapter_violation")
    if not bool(progress.get("p29_post_fields_empty", False)):
        blocked.append("v5_p33_p29_post_fields_present")
    if not bool(progress.get("p31_ready", False)):
        blocked.append("v5_p33_p31_manual_run_audit_not_ready")
    if not bool(progress.get("p27_approved", False)):
        blocked.append("v5_p33_p27_signed_review_not_approved")
    if not bool(progress.get("p26_ready", False)):
        blocked.append("v5_p33_p26_gate_not_ready")
    if not bool(progress.get("p28_ready", False)):
        blocked.append("v5_p33_p28_collector_bundle_not_ready")
    if decision == P33_HOLD_DECISION:
        blocked.append("v5_p33_signed_owner_archive_review_missing")
    if decision == P33_BLOCKED_DECISION:
        _append_review_blockers(blocked, progress)
    return _dedupe(blocked)


def _append_review_blockers(blocked: list[str], progress: Mapping[str, bool]) -> None:
    checks = {
        "requested_scope_valid": "v5_p33_requested_scope_invalid",
        "review_defaults_off": "v5_p33_default_off_confirmation_missing",
        "review_launch_not_requested": "v5_p33_review_training_launch_requested",
        "review_request_adapter_not_requested": "v5_p33_review_request_adapter_requested",
        "p32_owner_package_acknowledged": "v5_p33_p32_owner_package_ack_missing",
        "p31_replay_acknowledged": "v5_p33_p31_replay_ack_missing",
        "p29_archive_acknowledged": "v5_p33_p29_archive_ack_missing",
        "p27_signed_acknowledged": "v5_p33_p27_signed_review_ack_missing",
        "p26_gate_acknowledged": "v5_p33_p26_gate_ack_missing",
        "p28_evidence_acknowledged": "v5_p33_p28_collector_ack_missing",
        "default_rollout_acknowledged_off": "v5_p33_default_rollout_disabled_ack_missing",
        "ui_request_adapter_acknowledged_off": "v5_p33_no_ui_or_request_adapter_ack_missing",
        "request_adapter_acknowledged_off": "v5_p33_request_adapter_ack_missing",
        "training_launch_acknowledged_off": "v5_p33_training_launch_ack_missing",
        "default_auto_rollout_acknowledged_off": "v5_p33_default_auto_rollout_ack_missing",
        "manual_review_only_acknowledged": "v5_p33_manual_review_only_ack_missing",
        "full_chain_acknowledged": "v5_p33_full_chain_archive_ack_missing",
    }
    for key, reason in checks.items():
        if not bool(progress.get(key, False)):
            blocked.append(reason)


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_final_owner_archive": bool(
            review.get("approve_final_owner_archive", review.get("approve_owner_archive_package", False))
        ),
        "approve_owner_archive_package": bool(review.get("approve_owner_archive_package", False)),
        "approve_training_launch_allowed": bool(review.get("approve_training_launch_allowed", False)),
        "approve_auto_launch_allowed": bool(review.get("approve_auto_launch_allowed", False)),
        "approve_runs_dispatched": bool(review.get("approve_runs_dispatched", False)),
        "approve_default_training_path_enabled": bool(review.get("approve_default_training_path_enabled", False)),
        "approve_default_rollout_allowed": bool(review.get("approve_default_rollout_allowed", False)),
        "approve_auto_rollout_allowed": bool(review.get("approve_auto_rollout_allowed", False)),
        "approve_request_adapter_mapping_allowed": bool(review.get("approve_request_adapter_mapping_allowed", False)),
        "approve_request_fields_emitted": bool(review.get("approve_request_fields_emitted", False)),
        "acknowledge_p32_owner_package_ready": bool(review.get("acknowledge_p32_owner_package_ready", False)),
        "acknowledge_p31_collector_replay_ready": bool(
            review.get("acknowledge_p31_collector_replay_ready", False)
        ),
        "acknowledge_p29_owner_package_ready": bool(review.get("acknowledge_p29_owner_package_ready", False)),
        "acknowledge_signed_p27_review": bool(review.get("acknowledge_signed_p27_review", False)),
        "acknowledge_p26_gate_ready": bool(review.get("acknowledge_p26_gate_ready", False)),
        "acknowledge_p28_collector_bundle_ready": bool(
            review.get("acknowledge_p28_collector_bundle_ready", False)
        ),
        "acknowledge_default_rollout_disabled": bool(review.get("acknowledge_default_rollout_disabled", False)),
        "acknowledge_no_ui_or_request_adapter": bool(review.get("acknowledge_no_ui_or_request_adapter", False)),
        "acknowledge_no_request_adapter_mapping": bool(review.get("acknowledge_no_request_adapter_mapping", False)),
        "acknowledge_no_training_launch": bool(review.get("acknowledge_no_training_launch", False)),
        "acknowledge_default_and_auto_rollout_off": bool(
            review.get("acknowledge_default_and_auto_rollout_off", False)
        ),
        "acknowledge_manual_review_only": bool(review.get("acknowledge_manual_review_only", False)),
        "acknowledge_p31_p28_p26_p27_p29_chain_archived": bool(
            review.get("acknowledge_p31_p28_p26_p27_p29_chain_archived", False)
        ),
    }


def _review_template(package_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": P33_SCOPE,
        "approve_final_owner_archive": False,
        "approve_owner_archive_package": False,
        "approve_training_launch_allowed": False,
        "approve_auto_launch_allowed": False,
        "approve_runs_dispatched": False,
        "approve_default_training_path_enabled": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
        "approve_request_adapter_mapping_allowed": False,
        "approve_request_fields_emitted": False,
        "acknowledge_p32_owner_package_ready": False,
        "acknowledge_p31_collector_replay_ready": False,
        "acknowledge_p29_owner_package_ready": False,
        "acknowledge_signed_p27_review": False,
        "acknowledge_p26_gate_ready": False,
        "acknowledge_p28_collector_bundle_ready": False,
        "acknowledge_default_rollout_disabled": False,
        "acknowledge_no_ui_or_request_adapter": False,
        "acknowledge_no_request_adapter_mapping": False,
        "acknowledge_no_training_launch": False,
        "acknowledge_default_and_auto_rollout_off": False,
        "acknowledge_manual_review_only": False,
        "acknowledge_p31_p28_p26_p27_p29_chain_archived": False,
        "acknowledged_run_count": package_summary.get("run_count"),
        "acknowledged_ready_run_count": package_summary.get("ready_run_count"),
        "acknowledged_min_speedup": package_summary.get("min_speedup"),
        "acknowledged_speedup_spread_ratio": package_summary.get("speedup_spread_ratio"),
    }


def _recommended_next_step(
    approved: bool,
    rejected: bool,
    waiting_for_review: bool,
    blockers: list[str],
) -> str:
    if approved:
        return "archive the P33 signoff record; any next run remains explicit and default-off"
    if rejected:
        return "record the P33 rejection and keep the native path default-off for repair"
    if waiting_for_review:
        return "collect a signed owner archive review for the P32 package"
    if any(item.startswith("v5_p33_p32") for item in blockers):
        return "repair the P32 owner package before archive signoff"
    return "hold P33 until archive review acknowledgements and default-off evidence are complete"


def _default_off_confirmed(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("default_training_path_enabled") is False
        and value.get("training_path_enabled") is False
        and value.get("default_rollout_allowed") is False
        and value.get("auto_rollout_allowed") is False
    )


def _request_adapter_off(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("request_adapter_mapping_allowed") is False
        and value.get("request_fields_emitted") is False
    )


def _review_defaults_off(review: Mapping[str, Any]) -> bool:
    return bool(
        review.get("approve_default_training_path_enabled") is False
        and review.get("approve_default_rollout_allowed") is False
        and review.get("approve_auto_rollout_allowed") is False
    )


def _aggregate_value(bundle: Mapping[str, Any], key: str) -> Any:
    aggregate = _as_dict(bundle.get("aggregate"))
    if key in aggregate:
        return aggregate.get(key)
    return bundle.get(key)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P33 owner archive signoff contract.")
    parser.add_argument("--p32-owner-package", default="", help="P32 P31 collector replay owner package JSON.")
    parser.add_argument("--owner-archive-review", default="", help="Optional signed P33 owner archive review JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_owner_archive_signoff(
        p32_owner_package=load_json(args.p32_owner_package) if args.p32_owner_package else None,
        owner_archive_review=load_json(args.owner_archive_review) if args.owner_archive_review else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_owner_archive_signoff"]
