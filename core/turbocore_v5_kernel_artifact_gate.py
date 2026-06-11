"""Kernel artifact gate for TurboCore V5-P53.

P53 records future kernel artifact gate evidence after P52. It does not
register or enable artifacts, execute native dispatch, launch kernels, run
parity, execute training steps, emit request fields, expose UI, or launch
training.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT, REPO_ROOT = Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_controlled_rollout_policy_evidence_gate_utils import (
    as_dict as _as_dict,
    dedupe as _dedupe,
    default_off_confirmed as _default_off_confirmed,
    digest as _digest,
    event_list as _event_list,
    history_clear as _history_clear,
    history_summary as _history_summary,
    request_adapter_off as _request_adapter_off,
    source as _source,
    string_list as _string_list,
)
from core.turbocore_v5_owner_review_evidence_package import load_json


P52_READY_DECISION = "native_execution_dry_run_parity_boundary_recorded_default_off"
P53_READY_DECISION = "kernel_artifact_gate_recorded_default_off"
P53_BLOCKED_DECISION = "kernel_artifact_gate_blocked_default_off"
P53_HOLD_DECISION = "kernel_artifact_gate_hold_for_signed_review_default_off"
P53_REJECTED_DECISION = "kernel_artifact_gate_rejected_default_off"
P53_SCOPE = "kernel_artifact_gate"
DEFAULT_REQUIRED_SECTIONS = (
    "p52_native_execution_dry_run_parity_boundary_reference", "kernel_artifact_inventory",
    "kernel_artifact_digest_inventory", "artifact_abi_boundary", "artifact_loader_boundary",
    "request_adapter_boundary", "no_kernel_artifact_registration_boundary", "no_native_dispatch_boundary",
    "no_kernel_launch_boundary", "no_parity_execution_boundary", "no_training_step_boundary",
    "no_request_fields_boundary", "no_training_launch_boundary", "rollback_policy", "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p52_native_execution_dry_run_parity_boundary_recorded", "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch", "acknowledge_no_ui_exposure",
    "acknowledge_no_kernel_artifact_registration", "acknowledge_no_kernel_artifact_enabled",
    "acknowledge_no_native_dispatch_executed", "acknowledge_no_kernel_launch_executed",
    "acknowledge_no_parity_executed", "acknowledge_no_training_step_executed",
    "acknowledge_no_request_adapter_enabled", "acknowledge_no_request_fields_emitted",
    "acknowledge_no_default_or_auto_rollout", "acknowledge_kernel_artifact_gate_evidence_replayable",
    "acknowledge_later_result_ingestion_or_native_dry_run_execution_contract_required",
    "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = (
    "training_launch_allowed", "auto_launch_allowed", "runs_dispatched", "default_training_path_enabled",
    "training_path_enabled", "default_rollout_allowed", "auto_rollout_allowed", "ui_exposure_allowed",
    "product_ui_exposure_allowed", "launcher_exposure_allowed", "webui_exposure_allowed",
    "request_adapter_mapping_allowed", "request_fields_emitted", "request_adapter_registered",
    "request_adapter_enabled", "runtime_adapter_registered", "runtime_adapter_enabled",
    "runtime_execution_allowed", "runtime_execution_executed", "runtime_dispatch_allowed",
    "runtime_dispatch_enabled", "runtime_dispatch_executed", "native_runtime_enabled",
    "native_execution_allowed", "native_execution_enabled", "native_execution_executed",
    "native_execution_dry_run_allowed", "native_execution_dry_run_enabled", "native_execution_dry_run_executed",
    "native_dispatch_allowed", "native_dispatch_enabled", "native_dispatch_started", "native_dispatch_executed",
    "kernel_artifact_gate_allowed", "kernel_artifact_gate_enabled", "kernel_artifact_gate_executed",
    "kernel_artifact_registered", "kernel_artifact_enabled", "kernel_implementation_loaded",
    "kernel_binary_loaded", "artifact_loader_enabled", "artifact_loaded", "artifact_abi_loaded",
    "kernel_launch_allowed", "kernel_launch_enabled", "kernel_launch_executed",
    "parity_check_allowed", "parity_check_enabled", "parity_check_executed", "parity_result_recorded",
    "native_dry_run_result_ingested", "artifact_replay_executed", "training_step_allowed",
    "training_step_enabled", "training_step_executed", "generation_request_patch_allowed",
    "config_adapter_patch_allowed", "runtime_resolver_patch_allowed", "execution_resolver_patch_allowed",
    "training_manager_patch_allowed", "rollout_authorization_allowed", "default_behavior_changed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_p52_request_fields", "post_p53_request_fields", "post_kernel_artifact_gate_fields",
    "native_execution_request", "native_execution_payload", "native_dry_run_request", "native_dry_run_payload",
    "native_dispatch_request", "native_dispatch_payload", "kernel_artifact_registration",
    "kernel_artifact_payload", "kernel_artifact_loader_payload", "artifact_loader_payload",
    "kernel_launch_request", "kernel_launch_payload", "parity_check_request", "parity_check_payload",
    "parity_result_payload", "training_step_request", "training_step_payload", "request_adapter_fields",
    "request_fields", "api_route_registration", "backend_route_registration", "ui_route_registration",
    "launcher_menu_entry", "webui_tab_entry", "launch_request", "training_request",
    "generation_request_patch", "config_adapter_patch", "runtime_resolver_patch", "execution_resolver_patch",
    "entry_train_patch", "training_manager_patch", "auto_launch_plan",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(
    f"approve_{field}" for field in UNSAFE_TRUE_FIELDS if field != "default_behavior_changed"
)
P52_REQUIRED_FALSE_FIELDS = (
    "default_behavior_changed", "training_launch_allowed", "auto_launch_allowed", "runs_dispatched",
    "ui_exposure_allowed", "request_fields_emitted", "request_adapter_enabled", "runtime_adapter_enabled",
    "native_runtime_enabled", "native_execution_dry_run_executed", "native_dispatch_executed",
    "kernel_launch_executed", "parity_check_executed", "parity_result_recorded",
    "kernel_artifact_registered", "kernel_artifact_enabled", "training_step_executed",
)


def build_v5_kernel_artifact_gate(
    *,
    p52_native_execution_dry_run_parity_boundary: Mapping[str, Any] | None = None,
    kernel_artifact_gate_evidence: Mapping[str, Any] | None = None,
    kernel_artifact_gate_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record kernel artifact gate evidence without registering artifacts."""

    p52 = _as_dict(p52_native_execution_dry_run_parity_boundary)
    evidence = _as_dict(kernel_artifact_gate_evidence)
    review = _as_dict(kernel_artifact_gate_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p52_summary = _p52_summary(p52)
    evidence_summary = _evidence_summary(evidence)
    review_summary = _review_summary(review)
    evidence_blockers = _evidence_blockers(p52_summary, evidence_summary, failure_events, rollback_events)
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P53_HOLD_DECISION:
        blockers.append("v5_p53_signed_kernel_artifact_gate_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P53_READY_DECISION
    rejected = decision == P53_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_kernel_artifact_gate_v0",
        "gate": "v5_kernel_artifact_gate",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "kernel_artifact_gate_review_recorded": decision_record_ready,
        "kernel_artifact_gate_review_signed": decision_record_ready,
        "kernel_artifact_gate_ready": ready,
        "kernel_artifact_gate_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_p53_request_fields": {},
        "p52_native_execution_dry_run_parity_boundary_summary": p52_summary,
        "kernel_artifact_gate_evidence_summary": evidence_summary,
        "kernel_artifact_gate_review": review_summary,
        "kernel_artifact_gate_review_template": _review_template(),
        "progress_gates": _progress_gates(p52_summary, evidence_summary, review_summary),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P53 records kernel artifact gate evidence only.",
            "P53 does not register artifacts, enable artifacts, execute native dispatch, launch kernels, run parity, emit request fields, or launch training.",
            "A later native dry-run result ingestion or artifact replay verifier contract is still required before runtime behavior can become active.",
        ],
    }


def _p52_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "decision_record_ready": report.get("decision_record_ready") is True,
        "native_execution_dry_run_parity_boundary_ready": report.get(
            "native_execution_dry_run_parity_boundary_ready"
        )
        is True,
        "native_dry_run_parity_evidence_recorded": report.get("native_dry_run_parity_evidence_recorded") is True,
        "native_dry_run_parity_review_signed": report.get("native_dry_run_parity_review_signed") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        **{field: report.get(field) for field in P52_REQUIRED_FALSE_FIELDS},
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p52_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p52"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _evidence_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(evidence.get("required_sections")) or list(DEFAULT_REQUIRED_SECTIONS)
    missing_sections = [item for item in required_sections if item not in _section_set(evidence)]
    artifact_rows = _rows(evidence, "kernel_artifact_inventory", "artifact_inventory")
    digest_rows = _rows(evidence, "kernel_artifact_digest_inventory", "artifact_digest_inventory")
    abi_rows = _rows(evidence, "artifact_abi_boundary", "artifact_abi_inventory")
    loader_rows = _rows(evidence, "artifact_loader_boundary", "artifact_loader_inventory")
    blockers = _evidence_blocker_list(evidence, missing_sections, artifact_rows, digest_rows, abi_rows, loader_rows)
    return {
        "present": bool(evidence),
        "evidence_id": str(evidence.get("evidence_id") or evidence.get("id") or ""),
        "evidence_version": str(evidence.get("evidence_version") or evidence.get("version") or ""),
        "ok": evidence.get("ok") is True,
        "ready": not blockers,
        "kernel_artifact_gate_ready": evidence.get("kernel_artifact_gate_ready") is True,
        "report_only": evidence.get("report_only") is True,
        "boundary_only": evidence.get("boundary_only") is True,
        "contract_only": evidence.get("contract_only") is True,
        "artifact_gate_only": evidence.get("artifact_gate_only") is True,
        "records_evidence_only": evidence.get("records_evidence_only") is True,
        "manual_only": evidence.get("manual_only") is True,
        "internal_only": evidence.get("internal_only") is True,
        "requires_later_result_ingestion_or_native_dry_run_execution_contract": evidence.get(
            "requires_later_result_ingestion_or_native_dry_run_execution_contract"
        )
        is True,
        "requires_explicit_owner_approval": evidence.get("requires_explicit_owner_approval") is True,
        "requires_explicit_operator_opt_in": evidence.get("requires_explicit_operator_opt_in") is True,
        "default_off": evidence.get("default_off") is True and _default_off_confirmed(evidence),
        "request_adapter_off": evidence.get("request_adapter_off") is True and _request_adapter_off(evidence),
        "digest": _digest(evidence),
        "source": _source(evidence),
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "blocked_reasons": _string_list(evidence.get("blocked_reasons")),
        "promotion_blockers": _string_list(evidence.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(evidence, "kernel_artifact_gate"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_kernel_artifact_gate": review.get("approve_kernel_artifact_gate") is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(p52: Mapping[str, Any], evidence: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "p52_native_execution_dry_run_parity_boundary_present": bool(p52.get("present", False)),
        "p52_native_execution_dry_run_parity_boundary_ready": _p52_ready(p52),
        "kernel_artifact_gate_evidence_present": bool(evidence.get("present", False)),
        "kernel_artifact_gate_evidence_ready": bool(evidence.get("ready", False)),
        "signed_kernel_artifact_gate_review_present": bool(review.get("present", False)),
        "reviewer_present": bool(review.get("reviewer")),
        "reviewed_at_present": bool(review.get("reviewed_at")),
        "requested_scope_valid": review.get("requested_scope") == P53_SCOPE,
        "required_acknowledgements_present": all(bool(review.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P53_BLOCKED_DECISION
    if not review:
        return P53_HOLD_DECISION
    if review_blockers:
        return P53_BLOCKED_DECISION
    if review.get("approve_kernel_artifact_gate") is True:
        return P53_READY_DECISION
    return P53_REJECTED_DECISION


def _evidence_blockers(
    p52_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p52_summary.get("present", False)):
        blocked.append("v5_p53_p52_native_execution_dry_run_parity_boundary_missing")
    elif not _p52_ready(p52_summary):
        blocked.append("v5_p53_p52_native_execution_dry_run_parity_boundary_not_ready")
        blocked.extend(_string_list(p52_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p52_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p52_summary.get("unsafe_claims")))
    if not bool(evidence_summary.get("present", False)):
        blocked.append("v5_p53_kernel_artifact_gate_evidence_missing")
    elif not bool(evidence_summary.get("ready", False)):
        blocked.extend(_string_list(evidence_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p53_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p53_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p53_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p53_reviewed_at_missing")
    if review.get("requested_scope") != P53_SCOPE:
        blocked.append("v5_p53_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p53_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p53_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p52_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("decision_record_ready")
        and summary.get("native_execution_dry_run_parity_boundary_ready")
        and summary.get("native_dry_run_parity_evidence_recorded")
        and summary.get("native_dry_run_parity_review_signed")
        and summary.get("decision") == P52_READY_DECISION
        and summary.get("manual_review_required")
        and all(summary.get(field) is False for field in P52_REQUIRED_FALSE_FIELDS)
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
        and not _string_list(summary.get("blocked_reasons"))
        and not _string_list(summary.get("promotion_blockers"))
        and not _string_list(summary.get("unsafe_claims"))
    )


def _evidence_blocker_list(
    evidence: Mapping[str, Any],
    missing_sections: list[str],
    artifact_rows: list[dict[str, Any]],
    digest_rows: list[dict[str, Any]],
    abi_rows: list[dict[str, Any]],
    loader_rows: list[dict[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append("v5_p53_kernel_artifact_gate_not_ok")
    if evidence.get("kernel_artifact_gate_ready") is not True:
        blocked.append("v5_p53_kernel_artifact_gate_evidence_not_ready")
    for field in (
        "report_only", "boundary_only", "contract_only", "artifact_gate_only", "records_evidence_only",
        "manual_only", "internal_only", "requires_later_result_ingestion_or_native_dry_run_execution_contract",
        "requires_explicit_owner_approval", "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"v5_p53_kernel_artifact_gate_{field}_missing")
    if evidence.get("default_off") is not True or not _default_off_confirmed(evidence):
        blocked.append("v5_p53_kernel_artifact_gate_default_off_violation")
    if evidence.get("request_adapter_off") is not True or not _request_adapter_off(evidence):
        blocked.append("v5_p53_kernel_artifact_gate_request_adapter_violation")
    if not _digest(evidence):
        blocked.append("v5_p53_kernel_artifact_gate_digest_missing")
    if not _source(evidence):
        blocked.append("v5_p53_kernel_artifact_gate_source_missing")
    for section in missing_sections:
        blocked.append(f"v5_p53_kernel_artifact_gate_section_missing:{section}")
    blocked.extend(_artifact_inventory_blockers(artifact_rows))
    blocked.extend(_digest_inventory_blockers(digest_rows))
    blocked.extend(_boundary_inventory_blockers(abi_rows, "artifact_abi"))
    blocked.extend(_boundary_inventory_blockers(loader_rows, "artifact_loader"))
    blocked.extend(_unsafe_claims(evidence, "kernel_artifact_gate"))
    blocked.extend(_string_list(evidence.get("blocked_reasons")))
    blocked.extend(_string_list(evidence.get("promotion_blockers")))
    return _dedupe(blocked)


def _artifact_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["v5_p53_kernel_artifact_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("artifact_id") or row.get("kernel_id") or row.get("id") or f"artifact_{index}")
        for field in (
            "kernel_artifact_registered", "kernel_artifact_enabled", "kernel_implementation_loaded",
            "kernel_binary_loaded", "artifact_loaded", "kernel_launch_enabled", "kernel_launch_executed",
            "native_dispatch_executed", "parity_check_executed", "training_step_executed",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p53_kernel_artifact_claim:{item_id}:{field}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _digest_inventory_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["v5_p53_kernel_artifact_digest_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("artifact_id") or row.get("digest_id") or row.get("id") or f"digest_{index}")
        if not _digest(row):
            blocked.append(f"v5_p53_kernel_artifact_digest_missing:{item_id}")
        if not _source(row):
            blocked.append(f"v5_p53_kernel_artifact_digest_source_missing:{item_id}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _boundary_inventory_blockers(rows: list[Mapping[str, Any]], owner: str) -> list[str]:
    if not rows:
        return [f"v5_p53_{owner}_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        item_id = str(row.get("boundary_id") or row.get("id") or f"{owner}_{index}")
        for field in (
            "artifact_loader_enabled", "artifact_loaded", "artifact_abi_loaded", "kernel_binary_loaded",
            "kernel_implementation_loaded", "kernel_artifact_registered", "kernel_artifact_enabled",
            "native_dispatch_executed", "kernel_launch_executed", "parity_check_executed",
        ):
            if row.get(field) is True:
                blocked.append(f"v5_p53_{owner}_claim:{item_id}:{field}")
        blocked.extend(_unsafe_claims(row, item_id))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p53_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p53_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _rows(payload: Mapping[str, Any], field: str, fallback: str) -> list[dict[str, Any]]:
    value = payload.get(field, payload.get(fallback))
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("available_sections")))
    sections.update(_string_list(value.get("sections")))
    if isinstance(value.get("section_status"), Mapping):
        for section, ready in _as_dict(value.get("section_status")).items():
            if ready:
                sections.add(str(section))
    return {str(item).strip() for item in sections if str(item).strip()}


def _review_template() -> dict[str, Any]:
    template = {"reviewer": "", "reviewed_at": "", "requested_scope": P53_SCOPE, "approve_kernel_artifact_gate": False}
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P53_READY_DECISION:
        return ["archive_p53_kernel_artifact_gate"]
    if decision == P53_REJECTED_DECISION:
        return ["record_p53_default_off_rejection_or_repair_kernel_artifact_gate_evidence"]
    if decision == P53_HOLD_DECISION:
        return ["collect_signed_kernel_artifact_gate_review"]
    if any("p52" in item for item in blockers):
        return ["repair_p52_native_execution_dry_run_parity_boundary"]
    if any("artifact" in item or "digest" in item or "loader" in item or "abi" in item for item in blockers):
        return ["repair_kernel_artifact_gate_evidence"]
    return ["clear_failure_or_rollback_history_before_p53_gate"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P53_READY_DECISION:
        return "archive P53 gate; native dry-run result ingestion or artifact replay verifier still requires a later explicit contract"
    if decision == P53_REJECTED_DECISION:
        return "record the signed rejection and keep kernel artifact gate default-off for repair"
    if decision == P53_HOLD_DECISION:
        return "collect a signed kernel artifact gate review over P52 evidence"
    if any("p52" in item for item in blockers):
        return "repair the P52 native execution dry-run/parity boundary before P53"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable kernel artifact source and digest evidence"
    return "hold P53 until evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P53 kernel artifact gate.")
    parser.add_argument("--p52-native-execution-dry-run-parity-boundary", default="", help="P52 boundary JSON.")
    parser.add_argument("--kernel-artifact-gate-evidence", default="", help="P53 evidence JSON.")
    parser.add_argument("--kernel-artifact-gate-review", default="", help="Signed P53 review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    p52 = (
        load_json(args.p52_native_execution_dry_run_parity_boundary)
        if args.p52_native_execution_dry_run_parity_boundary
        else None
    )
    report = build_v5_kernel_artifact_gate(
        p52_native_execution_dry_run_parity_boundary=p52,
        kernel_artifact_gate_evidence=load_json(args.kernel_artifact_gate_evidence)
        if args.kernel_artifact_gate_evidence
        else None,
        kernel_artifact_gate_review=load_json(args.kernel_artifact_gate_review)
        if args.kernel_artifact_gate_review
        else None,
        failure_history=load_json(args.failure_history) if args.failure_history else None,
        rollback_history=load_json(args.rollback_history) if args.rollback_history else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
