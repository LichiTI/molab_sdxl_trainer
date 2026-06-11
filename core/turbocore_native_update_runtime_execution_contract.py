"""Default-off runtime execution contract for TurboCore native updates."""

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


SCOPE = "native_update_runtime_execution_contract"
READY_DECISION = "native_update_runtime_execution_contract_recorded_default_off"
HOLD_DECISION = "native_update_runtime_execution_contract_hold_for_review_default_off"
BLOCKED_DECISION = "native_update_runtime_execution_contract_blocked_default_off"
REJECTED_DECISION = "native_update_runtime_execution_contract_rejected_default_off"
REQUIRED_SECTIONS = (
    "activation_contract_reference",
    "runtime_execution_plan_inventory",
    "runtime_execution_precondition_inventory",
    "runtime_isolation_boundary",
    "runtime_state_boundary",
    "no_runtime_execution_boundary",
    "no_operator_execution_boundary",
    "no_manual_execution_boundary",
    "no_runtime_state_refresh_boundary",
    "no_runtime_dispatch_boundary",
    "no_native_dispatch_boundary",
    "no_kernel_launch_boundary",
    "no_training_step_boundary",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_activation_contract_ready",
    "acknowledge_runtime_execution_plan_default_off",
    "acknowledge_runtime_execution_preconditions_default_off",
    "acknowledge_no_runtime_execution",
    "acknowledge_no_operator_or_manual_execution",
    "acknowledge_no_runtime_state_refresh",
    "acknowledge_no_native_dispatch_or_kernel_launch",
    "acknowledge_no_training_step",
    "acknowledge_no_request_ui_schema_exposure",
    "acknowledge_later_runtime_dispatch_contract_required",
)
UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "training_launch_allowed",
    "runs_dispatched",
    "training_path_enabled",
    "training_dispatch",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_started",
    "native_dispatch_executed",
    "runtime_execution_allowed",
    "runtime_execution_enabled",
    "runtime_execution_started",
    "runtime_execution_executed",
    "operator_execution_allowed",
    "operator_execution_enabled",
    "operator_execution_executed",
    "manual_execution_allowed",
    "manual_execution_executed",
    "runtime_state_refresh_allowed",
    "runtime_state_refreshed",
    "runtime_dispatch_allowed",
    "runtime_dispatch_enabled",
    "runtime_dispatch_executed",
    "artifact_loaded",
    "execution_replay_executed",
    "parity_executed",
    "kernel_launch_allowed",
    "kernel_launch_executed",
    "training_step_executed",
    "ready_for_ui",
    "ui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "rollout_authorization_allowed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_runtime_execution_request_fields",
    "post_activation_request_fields",
    "runtime_execution_request",
    "runtime_execution_payload",
    "operator_execution_payload",
    "manual_execution_payload",
    "runtime_dispatch_request",
    "native_dispatch_request",
    "kernel_launch_request",
    "training_step_request",
    "request_adapter_fields",
    "request_schema_fields",
    "ui_route_registration",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_native_update_runtime_execution_contract(
    *,
    activation_contract: Mapping[str, Any] | None = None,
    runtime_execution_evidence: Mapping[str, Any] | None = None,
    runtime_execution_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record runtime execution evidence without executing runtime work."""

    activation = _activation_summary(_as_dict(activation_contract))
    evidence = _runtime_execution_summary(_as_dict(runtime_execution_evidence))
    review = _review_summary(_as_dict(runtime_execution_review))
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    evidence_blockers = _evidence_blockers(
        activation=activation,
        execution=evidence,
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
    elif review.get("approve_native_update_runtime_execution_contract") is True:
        decision = READY_DECISION
    else:
        decision = REJECTED_DECISION
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == HOLD_DECISION:
        blockers.append("native_update_runtime_execution_review_missing")
    blockers = _dedupe(blockers)
    evidence_ready = not evidence_blockers
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_runtime_execution_contract_v0",
        "gate": "native_update_runtime_execution_contract",
        "ok": evidence_ready and decision != BLOCKED_DECISION,
        "evidence_ready": evidence_ready,
        "ready_for_runtime_execution_review": evidence_ready,
        "runtime_execution_contract_recorded": decision == READY_DECISION,
        "native_update_runtime_execution_contract_ready": decision == READY_DECISION,
        "manual_review_required": True,
        "runtime_execution_review_action_required": decision == HOLD_DECISION,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_runtime_execution_request_fields": {},
        "activation_contract_summary": activation,
        "runtime_execution_evidence_summary": evidence,
        "runtime_execution_review": review,
        "runtime_execution_review_template": _review_template(activation, evidence),
        "progress_gates": {
            "activation_contract_ready": bool(activation.get("ready")),
            "runtime_execution_evidence_ready": bool(evidence.get("ready")),
            "runtime_execution_plan_default_off_count": evidence.get("runtime_execution_plan_default_off_count", 0),
            "runtime_execution_precondition_default_off_count": evidence.get(
                "runtime_execution_precondition_default_off_count", 0
            ),
            "request_ui_schema_exposure_blocked": bool(
                activation.get("request_ui_schema_exposure_blocked")
                and evidence.get("request_ui_schema_exposure_blocked")
            ),
            "signed_runtime_execution_review_present": bool(review.get("present")),
        },
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, evidence_ready),
        "recommended_next_step": _recommended_next_step(decision, evidence_ready),
        "notes": [
            "This contract records runtime execution evidence only.",
            "It does not execute runtime/operator/manual work, refresh runtime state, dispatch native work, launch kernels, run parity, emit request fields, expose UI/schema, or launch training.",
            "A later runtime dispatch contract is required before runtime behavior can become active.",
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


def _activation_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    unsafe = _unsafe_claims(contract, "activation_contract")
    return {
        "present": bool(contract),
        "source": str(contract.get("_source_path") or ""),
        "digest": str(contract.get("_source_digest") or _digest_payload(contract) if contract else ""),
        "ok": contract.get("ok") is True,
        "evidence_ready": contract.get("evidence_ready") is True,
        "ready_for_activation_review": contract.get("ready_for_activation_review") is True,
        "activation_contract_recorded": contract.get("activation_contract_recorded") is True,
        "ready": bool(
            contract
            and contract.get("ok") is True
            and contract.get("evidence_ready") is True
            and contract.get("ready_for_activation_review") is True
            and contract.get("post_activation_request_fields", {}) == {}
            and not unsafe
        ),
        "request_ui_schema_exposure_blocked": bool(
            contract.get("request_adapter_mapping_allowed") is False
            and contract.get("request_fields_emitted") is False
            and contract.get("schema_exposure_allowed") is False
            and contract.get("ready_for_ui") is False
        ),
        "decision": str(contract.get("decision") or ""),
        "blocked_reasons": unsafe,
    }


def _runtime_execution_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = _section_set(evidence)
    missing = [section for section in REQUIRED_SECTIONS if section not in sections]
    plans = _rows(evidence, "runtime_execution_plan_inventory")
    preconditions = _rows(evidence, "runtime_execution_precondition_inventory")
    unsafe = _unsafe_claims(evidence, "runtime_execution_evidence")
    blockers = _dedupe(
        _flag_blockers(evidence)
        + [f"native_update_runtime_execution_section_missing:{section}" for section in missing]
        + _plan_blockers(plans)
        + _precondition_blockers(preconditions)
        + unsafe
        + _string_list(evidence.get("blocked_reasons"))
        + _string_list(evidence.get("promotion_blockers"))
    )
    return {
        "present": bool(evidence),
        "source": str(evidence.get("source") or evidence.get("_source_path") or ""),
        "digest": str(evidence.get("sha256") or evidence.get("artifact_digest") or _digest_payload(evidence) if evidence else ""),
        "ok": evidence.get("ok") is True,
        "ready": bool(evidence and evidence.get("ok") is True and evidence.get("runtime_execution_contract_ready") is True and not blockers),
        "runtime_execution_contract_ready": evidence.get("runtime_execution_contract_ready") is True,
        "runtime_execution_plan_count": len(plans),
        "runtime_execution_plan_default_off_count": sum(1 for row in plans if _plan_default_off(row)),
        "runtime_execution_precondition_count": len(preconditions),
        "runtime_execution_precondition_default_off_count": sum(
            1 for row in preconditions if _precondition_default_off(row)
        ),
        "request_ui_schema_exposure_blocked": not bool(
            evidence.get("request_adapter_mapping_allowed") is True
            or evidence.get("request_fields_emitted") is True
            or evidence.get("schema_exposure_allowed") is True
            or evidence.get("ready_for_ui") is True
        ),
        "blocked_reasons": blockers,
    }


def _flag_blockers(evidence: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    for field in (
        "report_only",
        "contract_only",
        "runtime_execution_contract_only",
        "records_evidence_only",
        "manual_only",
        "internal_only",
        "requires_later_runtime_dispatch_contract",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"native_update_runtime_execution_{field}_missing")
    if evidence.get("default_off") is not True:
        blocked.append("native_update_runtime_execution_default_off_violation")
    if evidence.get("runtime_execution_contract_ready") is not True:
        blocked.append("native_update_runtime_execution_evidence_not_ready")
    if not evidence.get("source") and not evidence.get("_source_path"):
        blocked.append("native_update_runtime_execution_source_missing")
    return blocked


def _plan_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["native_update_runtime_execution_plan_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        plan_id = str(row.get("plan_id") or row.get("id") or f"plan_{index}")
        if not _plan_default_off(row):
            blocked.append(f"native_update_runtime_execution_plan_not_default_off:{plan_id}")
        blocked.extend(_unsafe_claims(row, plan_id))
    return _dedupe(blocked)


def _precondition_blockers(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return ["native_update_runtime_execution_precondition_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        check_id = str(row.get("check_id") or row.get("id") or f"check_{index}")
        if not _precondition_default_off(row):
            blocked.append(f"native_update_runtime_execution_precondition_not_default_off:{check_id}")
        blocked.extend(_unsafe_claims(row, check_id))
    return _dedupe(blocked)


def _plan_default_off(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("runtime_execution_allowed") is False
        and row.get("runtime_execution_enabled") is False
        and row.get("runtime_execution_executed") is False
        and row.get("operator_execution_executed") is False
        and row.get("manual_execution_executed") is False
        and row.get("runtime_state_refreshed") is False
        and row.get("runtime_dispatch_executed") is False
        and row.get("native_dispatch_executed") is False
        and row.get("kernel_launch_executed") is False
        and row.get("training_step_executed") is False
        and row.get("request_fields_emitted") is False
    )


def _precondition_default_off(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("precondition_registered") is False
        and row.get("precondition_active") is False
        and row.get("runtime_execution_check_enabled") is False
    )


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_native_update_runtime_execution_contract": review.get(
            "approve_native_update_runtime_execution_contract"
        )
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    return summary


def _evidence_blockers(
    *,
    activation: Mapping[str, Any],
    execution: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not activation.get("present"):
        blocked.append("native_update_runtime_execution_activation_contract_missing")
    elif not activation.get("ready"):
        blocked.append("native_update_runtime_execution_activation_contract_not_ready")
        blocked.extend(_string_list(activation.get("blocked_reasons")))
    if not execution.get("present"):
        blocked.append("native_update_runtime_execution_evidence_missing")
    elif not execution.get("ready"):
        blocked.append("native_update_runtime_execution_evidence_not_ready")
        blocked.extend(_string_list(execution.get("blocked_reasons")))
    for event in failure_events:
        blocked.append(f"native_update_runtime_execution_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"native_update_runtime_execution_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("native_update_runtime_execution_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("native_update_runtime_execution_reviewed_at_missing")
    if review.get("requested_scope") != SCOPE:
        blocked.append("native_update_runtime_execution_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"native_update_runtime_execution_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"native_update_runtime_execution_review_ack_missing:{field}")
    return _dedupe(blocked)


def _review_template(activation: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": SCOPE,
        "approve_native_update_runtime_execution_contract": False,
    }
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_evidence"] = {
        "activation_digest": activation.get("digest"),
        "runtime_execution_digest": execution.get("digest"),
        "runtime_execution_plan_default_off_count": execution.get("runtime_execution_plan_default_off_count"),
    }
    return template


def _rows(payload: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    value = payload.get(field)
    if isinstance(value, Mapping):
        return [_as_dict(row) for row in value.values()]
    if isinstance(value, (list, tuple)):
        return [_as_dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("sections")))
    sections.update(_string_list(value.get("available_sections")))
    if isinstance(value.get("section_status"), Mapping):
        for section, ready in _as_dict(value.get("section_status")).items():
            if ready:
                sections.add(str(section))
    return {item for item in sections if item}


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"native_update_runtime_execution_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"native_update_runtime_execution_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _allowed_next_actions(decision: str, evidence_ready: bool) -> list[str]:
    if decision == READY_DECISION:
        return ["archive_native_update_runtime_execution_contract", "prepare_later_runtime_dispatch_contract"]
    if evidence_ready:
        return ["collect_signed_native_update_runtime_execution_review"]
    return ["repair_activation_or_runtime_execution_evidence"]


def _recommended_next_step(decision: str, evidence_ready: bool) -> str:
    if decision == READY_DECISION:
        return "archive runtime execution contract and prepare the later runtime dispatch contract"
    if evidence_ready:
        return "record runtime execution review while keeping runtime/native dispatch and request/UI/schema exposure disabled"
    return "repair activation or runtime execution evidence before runtime execution review"


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {k: v for k, v in value.items() if not str(k).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build native-update default-off runtime execution contract.")
    parser.add_argument("--activation-contract", required=True)
    parser.add_argument("--runtime-execution-evidence", required=True)
    parser.add_argument("--runtime-execution-review")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    payload = build_native_update_runtime_execution_contract(
        activation_contract=load_json(args.activation_contract),
        runtime_execution_evidence=load_json(args.runtime_execution_evidence),
        runtime_execution_review=load_json(args.runtime_execution_review) if args.runtime_execution_review else None,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
