"""Default-off product exposure decision gate for TurboCore native updates."""

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

from core.turbocore_v5_controlled_rollout_policy_evidence_gate_utils import (  # noqa: E402
    as_dict as _as_dict,
    dedupe as _dedupe,
    event_list as _event_list,
    history_summary as _history_summary,
    string_list as _string_list,
)
from core.turbocore_optimizer_v2_approval_preflight_guard import (  # noqa: E402
    approval_preflight_phase_ready as _approval_preflight_phase_ready,
    approval_preflight_record_binding as _approval_preflight_record_binding,
    approval_preflight_record_blockers as _approval_preflight_record_blockers,
    approval_preflight_signed_digest_match as _approval_preflight_signed_digest_match,
    digest_payload as _digest_payload,
)


SCOPE = "native_update_product_exposure_decision"
READY_DECISION = "native_update_product_exposure_decision_recorded_default_off"
HOLD_DECISION = "native_update_product_exposure_decision_hold_for_owner_review_default_off"
BLOCKED_DECISION = "native_update_product_exposure_decision_blocked_default_off"
REJECTED_DECISION = "native_update_product_exposure_decision_rejected_default_off"
REQUIRED_SECTIONS = (
    "training_launch_contract_reference",
    "owner_exposure_decision_boundary",
    "request_adapter_boundary",
    "request_schema_boundary",
    "backend_router_boundary",
    "launcher_ui_boundary",
    "webui_boundary",
    "release_gate_boundary",
    "no_training_launch_boundary",
    "no_request_submission_boundary",
    "no_request_ui_schema_boundary",
    "rollback_policy",
    "observability_policy",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_training_launch_contract_ready",
    "acknowledge_product_exposure_default_off",
    "acknowledge_no_request_adapter_or_schema_change",
    "acknowledge_no_backend_router_registration",
    "acknowledge_no_launcher_or_webui_exposure",
    "acknowledge_no_training_launch_or_request_submission",
    "acknowledge_release_requires_separate_owner_decision",
)
UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "product_exposure_allowed",
    "product_exposure_enabled",
    "product_exposure_approved",
    "release_gate_open",
    "training_launch_allowed",
    "training_launch_enabled",
    "training_launch_executed",
    "training_path_enabled",
    "training_dispatch",
    "request_submission_allowed",
    "request_submitted",
    "request_payload_materialized",
    "job_created",
    "queue_enqueued",
    "run_record_written",
    "ready_for_ui",
    "ui_exposure_allowed",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "backend_router_registered",
    "rollout_authorization_allowed",
)
UNSAFE_NON_EMPTY_FIELDS = (
    "post_product_exposure_request_fields",
    "post_training_launch_request_fields",
    "product_exposure_request",
    "product_exposure_payload",
    "training_launch_request",
    "training_request",
    "request_payload",
    "request_adapter_fields",
    "request_schema_fields",
    "backend_router_registration",
    "ui_route_registration",
    "launcher_menu_entry",
    "webui_tab_entry",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_native_update_product_exposure_decision(
    *,
    training_launch_contract: Mapping[str, Any] | None = None,
    product_exposure_evidence: Mapping[str, Any] | None = None,
    product_exposure_review: Mapping[str, Any] | None = None,
    approval_preflight: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Record product-exposure evidence without exposing product surfaces."""

    launch = _training_launch_summary(_as_dict(training_launch_contract))
    exposure = _product_exposure_summary(_as_dict(product_exposure_evidence))
    raw_review = _as_dict(product_exposure_review)
    review = _review_summary(raw_review)
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    evidence_blockers = _evidence_blockers(
        launch=launch,
        exposure=exposure,
        failure_events=failure_events,
        rollback_events=rollback_events,
    )
    review_blockers = _review_blockers(review) + (
        _approval_preflight_blockers(
            _as_dict(approval_preflight),
            phase="phase1",
            signature_id="product_exposure_review",
            signed_payload=raw_review,
        )
        if review.get("present")
        else []
    )
    if evidence_blockers:
        decision = BLOCKED_DECISION
    elif not review.get("present"):
        decision = HOLD_DECISION
    elif review_blockers:
        decision = BLOCKED_DECISION
    elif review.get("approve_native_update_product_exposure_decision") is True:
        decision = READY_DECISION
    else:
        decision = REJECTED_DECISION
    blockers = _dedupe(evidence_blockers + review_blockers)
    if decision == HOLD_DECISION:
        blockers.append("native_update_product_exposure_owner_review_missing")
    blockers = _dedupe(blockers)
    evidence_ready = not evidence_blockers
    preflight_binding = _approval_preflight_record_binding(
        _as_dict(approval_preflight),
        "product_exposure_review",
        raw_review,
    )
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_product_exposure_decision_v0",
        "gate": "native_update_product_exposure_decision",
        "ok": evidence_ready and decision != BLOCKED_DECISION,
        "evidence_ready": evidence_ready,
        "ready_for_product_exposure_review": evidence_ready,
        "product_exposure_decision_recorded": decision == READY_DECISION,
        "native_update_product_exposure_decision_ready": decision == READY_DECISION,
        "manual_review_required": True,
        "product_exposure_review_action_required": decision == HOLD_DECISION,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_product_exposure_request_fields": {},
        "training_launch_contract_summary": launch,
        "product_exposure_evidence_summary": exposure,
        "product_exposure_review": review,
        "approval_preflight_present": bool(approval_preflight),
        "approval_preflight_phase1_ready": _approval_preflight_phase_ready(
            _as_dict(approval_preflight),
            phase="phase1",
        ),
        "approval_preflight_signed_product_exposure_review_digest_match": _approval_preflight_signed_digest_match(
            _as_dict(approval_preflight),
            "product_exposure_review",
            raw_review,
        ),
        **preflight_binding,
        "product_exposure_review_template": _review_template(launch, exposure),
        "progress_gates": {
            "training_launch_contract_ready": bool(launch.get("ready")),
            "product_exposure_evidence_ready": bool(exposure.get("ready")),
            "approval_execution_preflight_phase1_ready": _approval_preflight_phase_ready(
                _as_dict(approval_preflight),
                phase="phase1",
            ),
            "approval_execution_preflight_signed_review_digest_match": _approval_preflight_signed_digest_match(
                _as_dict(approval_preflight),
                "product_exposure_review",
                raw_review,
            ),
            "owner_decision_boundary_default_off_count": exposure.get("owner_boundary_default_off_count", 0),
            "request_adapter_boundary_default_off_count": exposure.get("request_boundary_default_off_count", 0),
            "schema_boundary_default_off_count": exposure.get("schema_boundary_default_off_count", 0),
            "ui_boundary_default_off_count": exposure.get("ui_boundary_default_off_count", 0),
            "signed_product_exposure_review_present": bool(review.get("present")),
        },
        "summary": {
            "product_exposure_decision_recorded_count": 1 if decision == READY_DECISION else 0,
            "approval_preflight_phase1_ready_count": 1
            if _approval_preflight_phase_ready(_as_dict(approval_preflight), phase="phase1")
            else 0,
            "approval_preflight_signed_review_digest_match_count": 1
            if _approval_preflight_signed_digest_match(
                _as_dict(approval_preflight),
                "product_exposure_review",
                raw_review,
            )
            else 0,
            "approval_preflight_digest_present_count": 1
            if preflight_binding["approval_preflight_digest"]
            else 0,
            "approval_preflight_signed_payload_digest_present_count": 1
            if preflight_binding["approval_preflight_signed_payload_digest"]
            else 0,
            "approval_preflight_signed_bundle_entry_digest_present_count": 1
            if preflight_binding["approval_preflight_signed_bundle_entry_digest"]
            else 0,
            "approval_preflight_binding_ready_count": 1
            if preflight_binding["approval_preflight_binding_ready"]
            else 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
        },
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, evidence_ready),
        "recommended_next_step": _recommended_next_step(decision, evidence_ready),
        "notes": [
            "This decision gate records product-exposure evidence only.",
            "It does not emit request fields, register backend routes, expose launcher/webui controls, submit jobs, or launch training.",
            "A separate owner release decision is required before any product exposure can be implemented.",
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


def _training_launch_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    unsafe = _unsafe_claims(contract, "training_launch_contract")
    return {
        "present": bool(contract),
        "source": str(contract.get("_source_path") or ""),
        "digest": str(contract.get("_source_digest") or _digest_payload(contract) if contract else ""),
        "ok": contract.get("ok") is True,
        "evidence_ready": contract.get("evidence_ready") is True,
        "ready_for_training_launch_review": contract.get("ready_for_training_launch_review") is True,
        "training_launch_contract_recorded": contract.get("training_launch_contract_recorded") is True,
        "ready": bool(
            contract
            and contract.get("ok") is True
            and contract.get("evidence_ready") is True
            and contract.get("ready_for_training_launch_review") is True
            and contract.get("post_training_launch_request_fields", {}) == {}
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


def _product_exposure_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    sections = _section_set(evidence)
    missing = [section for section in REQUIRED_SECTIONS if section not in sections]
    owners = _rows(evidence, "owner_exposure_decision_boundary")
    requests = _rows(evidence, "request_adapter_boundary")
    schemas = _rows(evidence, "request_schema_boundary")
    routers = _rows(evidence, "backend_router_boundary")
    launcher = _rows(evidence, "launcher_ui_boundary")
    webui = _rows(evidence, "webui_boundary")
    releases = _rows(evidence, "release_gate_boundary")
    rollback = _rows(evidence, "rollback_policy")
    observability = _rows(evidence, "observability_policy")
    unsafe = _unsafe_claims(evidence, "product_exposure_evidence")
    blockers = _dedupe(
        _flag_blockers(evidence)
        + [f"native_update_product_exposure_section_missing:{section}" for section in missing]
        + _row_blockers(owners, "owner_decision_boundary", require_default_off=True)
        + _row_blockers(requests, "request_adapter_boundary", require_default_off=True)
        + _row_blockers(schemas, "request_schema_boundary", require_default_off=True)
        + _row_blockers(routers, "backend_router_boundary", require_default_off=True)
        + _row_blockers(launcher, "launcher_ui_boundary", require_default_off=True)
        + _row_blockers(webui, "webui_boundary", require_default_off=True)
        + _row_blockers(releases, "release_gate_boundary", require_default_off=True)
        + _row_blockers(rollback, "rollback_policy", require_default_off=False)
        + _row_blockers(observability, "observability_policy", require_default_off=False)
        + unsafe
        + _string_list(evidence.get("blocked_reasons"))
        + _string_list(evidence.get("promotion_blockers"))
    )
    return {
        "present": bool(evidence),
        "source": str(evidence.get("source") or evidence.get("_source_path") or ""),
        "digest": str(evidence.get("sha256") or evidence.get("artifact_digest") or _digest_payload(evidence) if evidence else ""),
        "ok": evidence.get("ok") is True,
        "ready": bool(evidence and evidence.get("ok") is True and evidence.get("product_exposure_decision_ready") is True and not blockers),
        "product_exposure_decision_ready": evidence.get("product_exposure_decision_ready") is True,
        "owner_boundary_default_off_count": sum(1 for row in owners if _row_default_off(row)),
        "request_boundary_default_off_count": sum(1 for row in requests if _row_default_off(row)),
        "schema_boundary_default_off_count": sum(1 for row in schemas if _row_default_off(row)),
        "router_boundary_default_off_count": sum(1 for row in routers if _row_default_off(row)),
        "ui_boundary_default_off_count": sum(
            1 for row in [*launcher, *webui, *releases] if _row_default_off(row)
        ),
        "blocked_reasons": blockers,
    }


def _flag_blockers(evidence: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    for field in (
        "report_only",
        "contract_only",
        "product_exposure_decision_only",
        "records_evidence_only",
        "manual_only",
        "internal_only",
        "requires_explicit_owner_approval",
        "requires_explicit_operator_opt_in",
    ):
        if evidence.get(field) is not True:
            blocked.append(f"native_update_product_exposure_{field}_missing")
    if evidence.get("default_off") is not True:
        blocked.append("native_update_product_exposure_default_off_violation")
    if evidence.get("product_exposure_decision_ready") is not True:
        blocked.append("native_update_product_exposure_evidence_not_ready")
    if not evidence.get("source") and not evidence.get("_source_path"):
        blocked.append("native_update_product_exposure_source_missing")
    return blocked


def _row_blockers(rows: list[Mapping[str, Any]], kind: str, *, require_default_off: bool) -> list[str]:
    if not rows:
        return [f"native_update_product_exposure_{kind}_inventory_missing"]
    blocked: list[str] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id") or row.get("plan_id") or row.get("check_id") or f"{kind}_{index}")
        if row.get("ready") is not True:
            blocked.append(f"native_update_product_exposure_{kind}_not_ready:{row_id}")
        if not row.get("source"):
            blocked.append(f"native_update_product_exposure_{kind}_source_missing:{row_id}")
        if require_default_off and not _row_default_off(row):
            blocked.append(f"native_update_product_exposure_{kind}_not_default_off:{row_id}")
        blocked.extend(_unsafe_claims(row, row_id))
    return _dedupe(blocked)


def _row_default_off(row: Mapping[str, Any]) -> bool:
    return all(row.get(field) is False for field in UNSAFE_TRUE_FIELDS if field in row)


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_native_update_product_exposure_decision": review.get(
            "approve_native_update_product_exposure_decision"
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
    launch: Mapping[str, Any],
    exposure: Mapping[str, Any],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not launch.get("present"):
        blocked.append("native_update_product_exposure_training_launch_contract_missing")
    elif not launch.get("ready"):
        blocked.append("native_update_product_exposure_training_launch_contract_not_ready")
        blocked.extend(_string_list(launch.get("blocked_reasons")))
    if not exposure.get("present"):
        blocked.append("native_update_product_exposure_evidence_missing")
    elif not exposure.get("ready"):
        blocked.append("native_update_product_exposure_evidence_not_ready")
        blocked.extend(_string_list(exposure.get("blocked_reasons")))
    for event in failure_events:
        blocked.append(f"native_update_product_exposure_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"native_update_product_exposure_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _review_blockers(review: Mapping[str, Any]) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("native_update_product_exposure_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("native_update_product_exposure_reviewed_at_missing")
    if review.get("requested_scope") != SCOPE:
        blocked.append("native_update_product_exposure_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"native_update_product_exposure_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"native_update_product_exposure_review_ack_missing:{field}")
    return _dedupe(blocked)


def _approval_preflight_blockers(
    preflight: Mapping[str, Any],
    *,
    phase: str,
    signature_id: str,
    signed_payload: Mapping[str, Any],
) -> list[str]:
    return _approval_preflight_record_blockers(
        preflight,
        phase=phase,
        signature_id=signature_id,
        signed_payload=signed_payload,
        unsafe_blockers=lambda value: _unsafe_claims(value, "approval_execution_preflight"),
    )


def _review_template(launch: Mapping[str, Any], exposure: Mapping[str, Any]) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": SCOPE,
        "approve_native_update_product_exposure_decision": False,
    }
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_evidence"] = {
        "training_launch_contract_digest": launch.get("digest"),
        "product_exposure_digest": exposure.get("digest"),
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
            blocked.append(f"native_update_product_exposure_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"native_update_product_exposure_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _allowed_next_actions(decision: str, evidence_ready: bool) -> list[str]:
    if decision == READY_DECISION:
        return ["archive_native_update_product_exposure_decision", "await_owner_release_direction"]
    if evidence_ready:
        return ["collect_signed_native_update_product_exposure_review"]
    return ["repair_training_launch_or_product_exposure_evidence"]


def _recommended_next_step(decision: str, evidence_ready: bool) -> str:
    if decision == READY_DECISION:
        return "archive product exposure decision and wait for a separate owner release direction"
    if evidence_ready:
        return "record product exposure review while keeping request/UI/schema exposure and product launch disabled"
    return "repair training-launch or product-exposure evidence before owner review"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build native-update default-off product exposure decision.")
    parser.add_argument("--training-launch-contract", required=True)
    parser.add_argument("--product-exposure-evidence", required=True)
    parser.add_argument("--product-exposure-review")
    parser.add_argument("--approval-preflight")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    payload = build_native_update_product_exposure_decision(
        training_launch_contract=load_json(args.training_launch_contract),
        product_exposure_evidence=load_json(args.product_exposure_evidence),
        product_exposure_review=load_json(args.product_exposure_review) if args.product_exposure_review else None,
        approval_preflight=load_json(args.approval_preflight) if args.approval_preflight else None,
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
