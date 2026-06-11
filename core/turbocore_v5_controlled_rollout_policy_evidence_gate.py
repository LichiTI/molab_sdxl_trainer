"""Controlled rollout policy evidence gate for TurboCore V5-P39.

P39 records a signed internal/manual/default-off policy review over the P38
packaging and observability gate. It is not a release gate: it never launches
training, emits request-adapter fields, exposes UI, or enables default rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_owner_review_evidence_package import load_json
from core.turbocore_v5_controlled_rollout_policy_evidence_gate_utils import (
    as_dict as _as_dict,
    dedupe as _dedupe,
    default_off_confirmed as _default_off_confirmed,
    digest as _digest,
    event_list as _event_list,
    history_clear as _history_clear,
    history_summary as _history_summary,
    int_value as _int,
    request_adapter_off as _request_adapter_off,
    source as _source,
    string_list as _string_list,
)


P38_READY_DECISION = "packaging_observability_evidence_gate_ready_default_off"
P39_READY_DECISION = "controlled_rollout_policy_evidence_recorded_default_off"
P39_BLOCKED_DECISION = "controlled_rollout_policy_evidence_blocked_default_off"
P39_HOLD_DECISION = "controlled_rollout_policy_hold_for_signed_review_default_off"
P39_REJECTED_DECISION = "controlled_rollout_policy_rejected_default_off"
P39_SCOPE = "controlled_rollout_policy_evidence_gate"
DEFAULT_REQUIRED_POLICY_SECTIONS = (
    "scope",
    "default_off_boundary",
    "operator_opt_in",
    "rollback_policy",
    "monitoring_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_p38_packaging_observability_ready",
    "acknowledge_default_off_boundary",
    "acknowledge_no_training_launch",
    "acknowledge_no_ui_exposure",
    "acknowledge_no_request_adapter_mapping",
    "acknowledge_no_default_or_auto_rollout",
    "acknowledge_policy_evidence_replayable",
    "acknowledge_rollback_policy_ready",
    "acknowledge_observability_policy_ready",
    "acknowledge_manual_review_only",
)
UNSAFE_TRUE_FIELDS = (
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
    "broader_rollout_claim_ready",
    "default_behavior_changed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_p38_request_fields",
    "post_p39_request_fields",
    "post_policy_request_fields",
    "request_adapter",
    "request_adapter_fields",
    "launch_request",
    "training_request",
    "auto_launch_plan",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = (
    "approve_training_launch_allowed",
    "approve_auto_launch_allowed",
    "approve_runs_dispatched",
    "approve_default_training_path_enabled",
    "approve_training_path_enabled",
    "approve_default_rollout_allowed",
    "approve_auto_rollout_allowed",
    "approve_ui_exposure_allowed",
    "approve_request_adapter_mapping_allowed",
    "approve_request_fields_emitted",
    "approve_rollout_authorization_allowed",
)


def build_v5_controlled_rollout_policy_evidence_gate(
    *,
    p38_packaging_observability_gate: Mapping[str, Any] | None = None,
    controlled_rollout_policy_evidence: Mapping[str, Any] | None = None,
    controlled_rollout_policy_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record controlled-rollout policy evidence without enabling rollout."""

    p38 = _as_dict(p38_packaging_observability_gate)
    policy = _as_dict(controlled_rollout_policy_evidence)
    review = _as_dict(controlled_rollout_policy_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    p38_summary = _p38_summary(p38)
    policy_summary = _policy_summary(policy)
    review_summary = _review_summary(review)
    progress = _progress_gates(p38_summary, policy_summary, review_summary)
    evidence_blockers = _evidence_blockers(
        p38_summary=p38_summary,
        policy_summary=policy_summary,
        failure_events=failure_events,
        rollback_events=rollback_events,
    )
    review_blockers = _review_blockers(review_summary)
    decision = _decision(evidence_blockers, review, review_blockers)
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == P39_HOLD_DECISION:
        blockers.append("v5_p39_signed_controlled_rollout_policy_review_missing")
    blockers = _dedupe(blockers)
    ready = decision == P39_READY_DECISION
    rejected = decision == P39_REJECTED_DECISION
    decision_record_ready = ready or rejected
    return {
        "schema_version": 1,
        "package": "turbocore_v5_controlled_rollout_policy_evidence_gate_v0",
        "gate": "v5_controlled_rollout_policy_evidence",
        "ok": decision_record_ready,
        "decision_record_ready": decision_record_ready,
        "controlled_rollout_policy_review_recorded": decision_record_ready,
        "controlled_rollout_policy_review_signed": decision_record_ready,
        "controlled_rollout_policy_evidence_ready": ready,
        "controlled_rollout_policy_evidence_recorded": ready,
        "recorded_for_next_contract_stage": ready,
        "rejected_for_default_off_hold": rejected,
        "rollback_required": rejected,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        "manual_review_required": True,
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
        "ui_entry_enabled": False,
        "ready_for_ui": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "rollout_authorization_allowed": False,
        "post_p39_request_fields": {},
        "p38_packaging_observability_summary": p38_summary,
        "controlled_rollout_policy_evidence_summary": policy_summary,
        "controlled_rollout_policy_review": review_summary,
        "controlled_rollout_policy_review_template": _review_template(policy_summary),
        "progress_gates": progress,
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, blockers),
        "recommended_next_step": _recommended_next_step(decision, blockers),
        "notes": [
            "P39 records signed controlled-rollout policy evidence only.",
            "P39 does not authorize product UI, request adapters, auto launch, default rollout, or training dispatch.",
            "Any later exposure boundary still needs another explicit default-off evidence contract.",
        ],
    }


def _p38_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _as_dict(report.get("artifact_digest_summary"))
    telemetry = _as_dict(report.get("telemetry_channel_summary"))
    surface = _as_dict(report.get("report_surface_summary"))
    decision = str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "packaging_observability_ready": report.get("packaging_observability_ready") is True,
        "evidence_gate_ready": report.get("evidence_gate_ready") is True,
        "decision": decision,
        "manual_review_required": report.get("manual_review_required") is True,
        "default_behavior_changed": report.get("default_behavior_changed"),
        "training_launch_allowed": report.get("training_launch_allowed"),
        "auto_launch_allowed": report.get("auto_launch_allowed"),
        "runs_dispatched": report.get("runs_dispatched"),
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "post_fields_empty": not bool(_as_dict(report.get("post_p38_request_fields"))),
        "package_counts_ready": _counts_ready(artifact, "ready_package_count", "required_package_count"),
        "telemetry_counts_ready": _counts_ready(telemetry, "ready_channel_count", "required_channel_count"),
        "report_counts_ready": _counts_ready(surface, "ready_report_count", "required_report_count"),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "promotion_blockers": _string_list(report.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(report, "p38"),
        "failure_history_clear": _history_clear(report, "failure_history_summary"),
        "rollback_history_clear": _history_clear(report, "rollback_history_summary"),
    }


def _policy_summary(policy: Mapping[str, Any]) -> dict[str, Any]:
    required_sections = _string_list(policy.get("required_sections")) or list(DEFAULT_REQUIRED_POLICY_SECTIONS)
    available_sections = _section_set(policy)
    missing_sections = [item for item in required_sections if item not in available_sections]
    rollback_policy = _as_dict(policy.get("rollback_policy"))
    monitoring_policy = _as_dict(policy.get("monitoring_policy"))
    blockers = _policy_blockers(
        policy,
        required_sections=required_sections,
        missing_sections=missing_sections,
        rollback_policy=rollback_policy,
        monitoring_policy=monitoring_policy,
    )
    return {
        "present": bool(policy),
        "policy_id": str(policy.get("policy_id") or policy.get("id") or ""),
        "policy_version": str(policy.get("policy_version") or policy.get("version") or ""),
        "ok": policy.get("ok") is True,
        "ready": not blockers,
        "controlled_rollout_policy_evidence_ready": policy.get("controlled_rollout_policy_evidence_ready") is True,
        "report_only": policy.get("report_only") is True,
        "manual_only": policy.get("manual_only") is True,
        "internal_only": policy.get("internal_only") is True,
        "requires_explicit_owner_approval": policy.get("requires_explicit_owner_approval") is True,
        "requires_explicit_operator_opt_in": policy.get("requires_explicit_operator_opt_in") is True,
        "default_off": policy.get("default_off") is True and _default_off_confirmed(policy),
        "request_adapter_off": policy.get("request_adapter_off") is True and _request_adapter_off(policy),
        "digest": _digest(policy),
        "source": _source(policy),
        "required_sections": required_sections,
        "available_sections": sorted(available_sections),
        "missing_sections": missing_sections,
        "rollback_policy_ready": _component_ready(policy, "rollback_policy", rollback_policy),
        "monitoring_policy_ready": _component_ready(policy, "monitoring_policy", monitoring_policy),
        "blocked_reasons": _string_list(policy.get("blocked_reasons")),
        "promotion_blockers": _string_list(policy.get("promotion_blockers")),
        "unsafe_claims": _unsafe_claims(policy, "policy"),
        "blockers": blockers,
    }


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_controlled_rollout_policy_evidence": review.get("approve_controlled_rollout_policy_evidence")
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _progress_gates(
    p38_summary: Mapping[str, Any],
    policy_summary: Mapping[str, Any],
    review_summary: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "p38_gate_present": bool(p38_summary.get("present", False)),
        "p38_gate_ready": _p38_ready(p38_summary),
        "controlled_rollout_policy_evidence_present": bool(policy_summary.get("present", False)),
        "controlled_rollout_policy_evidence_ready": bool(policy_summary.get("ready", False)),
        "signed_controlled_rollout_policy_review_present": bool(review_summary.get("present", False)),
        "reviewer_present": bool(review_summary.get("reviewer")),
        "reviewed_at_present": bool(review_summary.get("reviewed_at")),
        "requested_scope_valid": review_summary.get("requested_scope") == P39_SCOPE,
        "review_no_launch_requested": not bool(review_summary.get("approve_training_launch_allowed", False))
        and not bool(review_summary.get("approve_auto_launch_allowed", False))
        and not bool(review_summary.get("approve_runs_dispatched", False)),
        "review_no_default_or_auto_rollout_requested": not bool(
            review_summary.get("approve_default_rollout_allowed", False)
        )
        and not bool(review_summary.get("approve_auto_rollout_allowed", False)),
        "review_no_ui_requested": not bool(review_summary.get("approve_ui_exposure_allowed", False)),
        "review_no_request_adapter_requested": not bool(
            review_summary.get("approve_request_adapter_mapping_allowed", False)
        )
        and not bool(review_summary.get("approve_request_fields_emitted", False)),
        "required_acknowledgements_present": all(bool(review_summary.get(field, False)) for field in REQUIRED_REVIEW_ACKS),
    }


def _decision(evidence_blockers: list[str], review: Mapping[str, Any], review_blockers: list[str]) -> str:
    if evidence_blockers:
        return P39_BLOCKED_DECISION
    if not review:
        return P39_HOLD_DECISION
    if review_blockers:
        return P39_BLOCKED_DECISION
    if review.get("approve_controlled_rollout_policy_evidence") is True:
        return P39_READY_DECISION
    return P39_REJECTED_DECISION


def _evidence_blockers(
    *,
    p38_summary: Mapping[str, Any],
    policy_summary: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p38_summary.get("present", False)):
        blocked.append("v5_p39_p38_packaging_observability_gate_missing")
    elif not _p38_ready(p38_summary):
        blocked.append("v5_p39_p38_packaging_observability_gate_not_ready")
        blocked.extend(_string_list(p38_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p38_summary.get("promotion_blockers")))
        blocked.extend(_string_list(p38_summary.get("unsafe_claims")))
    if not bool(policy_summary.get("present", False)):
        blocked.append("v5_p39_controlled_rollout_policy_evidence_missing")
    elif not bool(policy_summary.get("ready", False)):
        blocked.extend(_string_list(policy_summary.get("blockers")))
    for event in failure_events:
        blocked.append(f"v5_p39_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p39_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("v5_p39_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("v5_p39_reviewed_at_missing")
    if review.get("requested_scope") != P39_SCOPE:
        blocked.append("v5_p39_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"v5_p39_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"v5_p39_review_ack_missing:{field}")
    return _dedupe(blocked)


def _p38_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("present")
        and summary.get("ok")
        and summary.get("packaging_observability_ready")
        and summary.get("evidence_gate_ready")
        and summary.get("decision") == P38_READY_DECISION
        and summary.get("manual_review_required")
        and summary.get("default_behavior_changed") is False
        and summary.get("training_launch_allowed") is False
        and summary.get("auto_launch_allowed") is False
        and summary.get("runs_dispatched") is False
        and summary.get("default_off")
        and summary.get("request_adapter_off")
        and summary.get("post_fields_empty")
        and summary.get("package_counts_ready")
        and summary.get("telemetry_counts_ready")
        and summary.get("report_counts_ready")
        and summary.get("failure_history_clear")
        and summary.get("rollback_history_clear")
        and not _string_list(summary.get("blocked_reasons"))
        and not _string_list(summary.get("promotion_blockers"))
        and not _string_list(summary.get("unsafe_claims"))
    )


def _policy_blockers(
    policy: Mapping[str, Any],
    *,
    required_sections: list[str],
    missing_sections: list[str],
    rollback_policy: Mapping[str, Any],
    monitoring_policy: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if policy.get("ok") is not True:
        blocked.append("v5_p39_policy_not_ok")
    if policy.get("controlled_rollout_policy_evidence_ready") is not True:
        blocked.append("v5_p39_policy_evidence_not_ready")
    if policy.get("report_only") is not True:
        blocked.append("v5_p39_policy_report_only_missing")
    if policy.get("manual_only") is not True:
        blocked.append("v5_p39_policy_manual_only_missing")
    if policy.get("internal_only") is not True:
        blocked.append("v5_p39_policy_internal_only_missing")
    if policy.get("requires_explicit_owner_approval") is not True:
        blocked.append("v5_p39_policy_explicit_owner_approval_missing")
    if policy.get("requires_explicit_operator_opt_in") is not True:
        blocked.append("v5_p39_policy_operator_opt_in_missing")
    if policy.get("default_off") is not True or not _default_off_confirmed(policy):
        blocked.append("v5_p39_policy_default_off_violation")
    if policy.get("request_adapter_off") is not True or not _request_adapter_off(policy):
        blocked.append("v5_p39_policy_request_adapter_violation")
    if not _digest(policy):
        blocked.append("v5_p39_policy_digest_missing")
    if not _source(policy):
        blocked.append("v5_p39_policy_source_missing")
    if not required_sections:
        blocked.append("v5_p39_policy_required_sections_empty")
    for section in missing_sections:
        blocked.append(f"v5_p39_policy_section_missing:{section}")
    if not rollback_policy:
        blocked.append("v5_p39_policy_rollback_policy_missing")
    elif not _component_ready(policy, "rollback_policy", rollback_policy):
        blocked.append("v5_p39_policy_rollback_policy_not_ready")
    if not monitoring_policy:
        blocked.append("v5_p39_policy_monitoring_policy_missing")
    elif not _component_ready(policy, "monitoring_policy", monitoring_policy):
        blocked.append("v5_p39_policy_monitoring_policy_not_ready")
    blocked.extend(_unsafe_claims(policy, "policy"))
    blocked.extend(_string_list(policy.get("blocked_reasons")))
    blocked.extend(_string_list(policy.get("promotion_blockers")))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"v5_p39_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"v5_p39_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _counts_ready(summary: Mapping[str, Any], ready_key: str, required_key: str) -> bool:
    required = _int(summary.get(required_key))
    ready = _int(summary.get(ready_key))
    return required > 0 and ready >= required


def _component_ready(parent: Mapping[str, Any], name: str, component: Mapping[str, Any]) -> bool:
    return bool(
        component
        and (
            parent.get(f"{name}_ready") is True
            or component.get("ready") is True
            or component.get("ok") is True
        )
    )


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("available_sections")))
    sections.update(_string_list(value.get("sections")))
    if isinstance(value.get("section_status"), Mapping):
        for section, ready in _as_dict(value.get("section_status")).items():
            if ready:
                sections.add(str(section))
    return {str(item).strip() for item in sections if str(item).strip()}


def _review_template(policy_summary: Mapping[str, Any]) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": P39_SCOPE,
        "approve_controlled_rollout_policy_evidence": False,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        template[field] = False
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_policy_id"] = policy_summary.get("policy_id")
    template["acknowledged_policy_version"] = policy_summary.get("policy_version")
    template["acknowledged_policy_digest"] = policy_summary.get("digest")
    return template


def _allowed_next_actions(decision: str, blockers: list[str]) -> list[str]:
    if decision == P39_READY_DECISION:
        return ["archive_p39_controlled_rollout_policy_evidence"]
    if decision == P39_REJECTED_DECISION:
        return ["record_p39_default_off_rejection_or_repair_policy"]
    if decision == P39_HOLD_DECISION:
        return ["collect_signed_controlled_rollout_policy_review"]
    if any("p38" in item for item in blockers):
        return ["repair_p38_packaging_observability_gate"]
    if any("policy" in item for item in blockers):
        return ["repair_controlled_rollout_policy_evidence"]
    return ["clear_failure_or_rollback_history_before_policy_record"]


def _recommended_next_step(decision: str, blockers: list[str]) -> str:
    if decision == P39_READY_DECISION:
        return "archive the signed P39 policy evidence; any later exposure still needs a separate default-off boundary"
    if decision == P39_REJECTED_DECISION:
        return "record the signed rejection and keep the native path default-off for repair"
    if decision == P39_HOLD_DECISION:
        return "collect a signed controlled-rollout policy review over P38 evidence"
    if any("p38" in item for item in blockers):
        return "repair the P38 packaging/observability evidence gate before P39"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable policy source and digest evidence"
    return "hold P39 until policy evidence, review acknowledgements, and histories are clear"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P39 controlled-rollout policy evidence gate.")
    parser.add_argument("--p38-packaging-observability-gate", default="", help="P38 evidence gate JSON.")
    parser.add_argument("--controlled-rollout-policy-evidence", default="", help="Controlled rollout policy JSON.")
    parser.add_argument("--controlled-rollout-policy-review", default="", help="Signed P39 policy review JSON.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_controlled_rollout_policy_evidence_gate(
        p38_packaging_observability_gate=load_json(args.p38_packaging_observability_gate)
        if args.p38_packaging_observability_gate
        else None,
        controlled_rollout_policy_evidence=load_json(args.controlled_rollout_policy_evidence)
        if args.controlled_rollout_policy_evidence
        else None,
        controlled_rollout_policy_review=load_json(args.controlled_rollout_policy_review)
        if args.controlled_rollout_policy_review
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


__all__ = ["build_v5_controlled_rollout_policy_evidence_gate"]
