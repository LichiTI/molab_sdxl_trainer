"""Release review package for TurboCore native-update default-off gates."""

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


SCOPE = "native_update_release_review_package"
READY_DECISION = "native_update_release_review_recorded_default_off"
HOLD_DECISION = "native_update_release_review_hold_for_owner_review_default_off"
BLOCKED_DECISION = "native_update_release_review_blocked_default_off"
REJECTED_DECISION = "native_update_release_review_rejected_default_off"
EXPECTED_GATES = (
    "turbocore_phase1_success_review",
    "native_update_rollout_review_package",
    "native_update_training_dispatch_integration_contract",
    "native_update_activation_contract",
    "native_update_runtime_execution_contract",
    "native_update_runtime_dispatch_contract",
    "native_update_native_dispatch_execution_contract",
    "native_update_kernel_launch_execution_contract",
    "native_update_parity_execution_contract",
    "native_update_training_step_execution_contract",
    "native_update_training_launch_contract",
    "native_update_product_exposure_decision",
)
SUPPLEMENTAL_GATES = (
    "optimizer_family_coverage",
    "native_update_optimizer_multitensor_release_hold",
)
REQUIRED_REVIEW_ACKS = (
    "acknowledge_all_expected_gates_present",
    "acknowledge_all_gates_default_off",
    "acknowledge_no_request_ui_schema_exposure",
    "acknowledge_no_training_launch_or_native_execution",
    "acknowledge_product_exposure_requires_separate_owner_direction",
)
UNSAFE_TRUE_FIELDS = (
    "default_behavior_changed",
    "product_exposure_allowed",
    "release_gate_open",
    "training_launch_allowed",
    "training_launch_enabled",
    "training_launch_executed",
    "training_path_enabled",
    "training_dispatch",
    "training_activation_allowed",
    "runtime_dispatch_allowed",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_executed",
    "kernel_launch_executed",
    "parity_executed",
    "training_step_executed",
    "request_submitted",
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
    "post_native_update_request_fields",
    "post_integration_request_fields",
    "post_activation_request_fields",
    "post_runtime_execution_request_fields",
    "post_runtime_dispatch_request_fields",
    "post_native_dispatch_execution_request_fields",
    "post_kernel_launch_execution_request_fields",
    "post_parity_execution_request_fields",
    "post_training_step_execution_request_fields",
    "post_training_launch_request_fields",
    "post_product_exposure_request_fields",
    "post_phase1_request_fields",
    "request_adapter_fields",
    "request_schema_fields",
    "ui_route_registration",
    "backend_router_registration",
)
UNSAFE_REVIEW_APPROVAL_FIELDS = tuple(f"approve_{field}" for field in UNSAFE_TRUE_FIELDS)


def build_native_update_release_review_package(
    *,
    gate_artifacts: Mapping[str, Any] | Sequence[Any] | None = None,
    release_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Aggregate native-update default-off gates for owner release review."""

    artifacts = _artifact_map(gate_artifacts)
    summaries = {gate: _artifact_summary(gate, artifacts.get(gate, {})) for gate in EXPECTED_GATES}
    supplemental = {gate: _artifact_summary(gate, artifacts.get(gate, {})) for gate in SUPPLEMENTAL_GATES}
    review = _review_summary(_as_dict(release_review))
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    evidence_blockers = _evidence_blockers(summaries, failure_events, rollback_events)
    supplemental_blockers = _supplemental_blockers(supplemental)
    review_blockers = _review_blockers(review, summaries, supplemental)
    if evidence_blockers or supplemental_blockers:
        decision = BLOCKED_DECISION
    elif not review.get("present"):
        decision = HOLD_DECISION
    elif review_blockers:
        decision = BLOCKED_DECISION
    elif review.get("approve_native_update_release_review_package") is True:
        decision = READY_DECISION
    else:
        decision = REJECTED_DECISION
    blockers = _dedupe(evidence_blockers + supplemental_blockers + review_blockers)
    if decision == HOLD_DECISION:
        blockers.append("native_update_release_owner_review_missing")
    blockers = _dedupe(blockers)
    evidence_ready = not evidence_blockers
    review_template = _review_template(summaries, supplemental)
    report = {
        "schema_version": 1,
        "package": "turbocore_native_update_release_review_package_v0",
        "gate": SCOPE,
        "ok": evidence_ready and decision != BLOCKED_DECISION,
        "evidence_ready": evidence_ready,
        "ready_for_review": evidence_ready,
        "ready_for_owner_release_review": evidence_ready,
        "release_review_recorded": decision == READY_DECISION,
        "native_update_release_review_ready": decision == READY_DECISION,
        "manual_review_required": True,
        "release_review_action_required": decision == HOLD_DECISION,
        "decision": decision,
        "gate_decision": decision,
        "package_decision": decision,
        **{field: False for field in UNSAFE_TRUE_FIELDS},
        "post_release_request_fields": {},
        "expected_gate_count": len(EXPECTED_GATES),
        "present_gate_count": sum(1 for item in summaries.values() if item["present"]),
        "default_off_gate_count": sum(1 for item in summaries.values() if item["default_off"]),
        "supplemental_gate_count": len(SUPPLEMENTAL_GATES),
        "present_supplemental_gate_count": sum(1 for item in supplemental.values() if item["present"]),
        "default_off_supplemental_gate_count": sum(
            1 for item in supplemental.values() if item["default_off"]
        ),
        "gate_summaries": summaries,
        "supplemental_gate_summaries": supplemental,
        "release_review": review,
        "release_review_template": review_template,
        "owner_release_review_handoff": _owner_release_review_handoff(
            decision,
            evidence_ready,
            review_template,
            blockers,
        ),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(decision, evidence_ready),
        "recommended_next_step": _recommended_next_step(decision, evidence_ready),
        "notes": [
            "This package aggregates release-review evidence only.",
            "It does not open product exposure, emit request fields, register routes, launch kernels, or launch training.",
            "Optimizer family coverage is supplemental for first release; unsafe supplemental evidence still blocks release review.",
            "A separate owner release direction is required before request/UI/schema exposure can be implemented.",
        ],
    }
    report["default_off"] = _top_level_default_off(report)
    return report


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload_digest = _digest_payload(payload)
        payload.setdefault("_source_path", str(source))
        payload.setdefault("_source_name", source.name)
        payload.setdefault("_payload_digest", payload_digest)
        payload.setdefault("_source_digest", _digest_payload(payload))
        return payload
    return {}


def load_gate_artifacts(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    artifacts: dict[str, dict[str, Any]] = {}
    for pattern in (
        "native_update*.json",
        "turbocore_phase1_success_review.json",
        "turbocore_optimizer_family_coverage_scorecard.json",
        "turbocore_optimizer_coverage_scorecard.json",
        "native_update_optimizer_multitensor_release_hold.json",
    ):
        for child in source.glob(pattern):
            payload = load_json(child)
            gate = str(payload.get("gate") or "")
            if gate in EXPECTED_GATES or gate in SUPPLEMENTAL_GATES:
                existing = artifacts.get(gate)
                if existing:
                    payload = _merge_loader_sources(existing, payload)
                else:
                    payload["_loader_sources"] = [_loader_source(payload)]
                    payload["_loader_source_count"] = 1
                    payload["_loader_payload_digest_match"] = True
                artifacts[gate] = payload
    return artifacts


def _merge_loader_sources(existing: Mapping[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    sources = _loader_sources(existing) + [_loader_source(payload)]
    digests = {source.get("payload_digest") for source in sources if source.get("payload_digest")}
    payload["_loader_sources"] = sources
    payload["_loader_source_count"] = len(sources)
    payload["_loader_payload_digest_match"] = len(digests) <= 1
    return payload


def _loader_source(payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source": str(payload.get("_source_path") or payload.get("source") or ""),
        "source_name": str(payload.get("_source_name") or ""),
        "payload_digest": str(payload.get("_payload_digest") or payload.get("_source_digest") or ""),
    }


def _loader_sources(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    value = payload.get("_loader_sources")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sources = []
        for item in value:
            source = _as_dict(item)
            sources.append(
                {
                    "source": str(source.get("source") or ""),
                    "source_name": str(source.get("source_name") or ""),
                    "payload_digest": str(source.get("payload_digest") or ""),
                }
            )
        return sources
    return [_loader_source(payload)] if payload else []


def _artifact_map(value: Mapping[str, Any] | Sequence[Any] | None) -> dict[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        result: dict[str, dict[str, Any]] = {}
        for key, item in value.items():
            report = _as_dict(item)
            gate = str(report.get("gate") or key)
            if gate:
                result[gate] = report
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = {}
        for item in value:
            report = _as_dict(item)
            gate = str(report.get("gate") or "")
            if gate:
                result[gate] = report
        return result
    return {}


def _artifact_summary(expected_gate: str, report: Mapping[str, Any]) -> dict[str, Any]:
    unsafe = _unsafe_claims(report, expected_gate)
    gate = str(report.get("gate") or "")
    request_fields_empty = all(not bool(report.get(field)) for field in UNSAFE_NON_EMPTY_FIELDS)
    default_off = bool(report and not unsafe and request_fields_empty and all(report.get(field) is not True for field in UNSAFE_TRUE_FIELDS))
    priority_next_gates = _priority_next_gates(report.get("priority_groups"))
    summary = {
        "present": bool(report),
        "gate": gate,
        "expected_gate": expected_gate,
        "source": str(report.get("_source_path") or report.get("source") or ""),
        "source_name": str(report.get("_source_name") or ""),
        "source_count": int(report.get("_loader_source_count", 1) or 0) if report else 0,
        "source_names": [source["source_name"] for source in _loader_sources(report) if source["source_name"]],
        "payload_digest": str(report.get("_payload_digest") or ""),
        "source_payload_digest_match": report.get("_loader_payload_digest_match") is not False,
        "digest": str(report.get("_source_digest") or report.get("artifact_digest") or _digest_payload(report) if report else ""),
        "ok": report.get("ok") is True,
        "evidence_ready": report.get("evidence_ready") is True or report.get("evidence_package_ready") is True,
        "ready_for_review": _ready_for_review(report),
        "decision": str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or ""),
        "default_off": default_off,
        "request_fields_empty": request_fields_empty,
        "unsafe_claims": unsafe,
        "blocked_reasons": _string_list(report.get("blocked_reasons")) + _string_list(report.get("promotion_blockers")),
        "recommended_next_step": str(report.get("recommended_next_step") or ""),
        "priority_group_count": len(priority_next_gates),
        "priority_next_gates": priority_next_gates,
    }
    if expected_gate == "optimizer_family_coverage":
        summary["optimizer_family_counts"] = _optimizer_family_counts(report)
    return summary


def _optimizer_family_counts(report: Mapping[str, Any]) -> dict[str, int]:
    source = _as_dict(report.get("summary"))
    fields = (
        "total_optimizer_types",
        "missing_classification_count",
        "native_ready_count",
        "exact_adamw_stream_event_chain_ownership_abi_ready_count",
        "exact_adamw_stream_lifetime_ownership_bound_evidence_count",
        "exact_adamw_stream_event_chain_verified_count",
        "exact_adamw_stream_event_chain_product_native_ready_count",
        "simple_formula_request_schema_ui_non_exposure_ready_count",
        "simple_formula_request_schema_ui_product_native_ready_count",
        "adamw_variant_owner_release_hold_ready_count",
        "adamw_variant_owner_release_hold_product_native_ready_count",
        "adamw_variant_request_schema_ui_non_exposure_ready_count",
        "adamw_variant_request_schema_ui_product_native_ready_count",
        "adaptive_lr_request_schema_ui_non_exposure_ready_count",
        "adaptive_lr_dispatch_integration_review_product_native_ready_count",
        "factored_custom_request_schema_ui_non_exposure_ready_count",
        "factored_custom_product_native_ready_count",
        "muon_model_shape_aware_dispatch_integration_review_ready_count",
        "muon_model_shape_aware_native_scratch_kernel_ready_count",
        "muon_model_shape_aware_native_scratch_kernel_executed_count",
        "muon_model_shape_aware_native_scratch_product_native_ready_count",
        "muon_model_shape_aware_training_tensor_binding_ready_count",
        "muon_model_shape_aware_training_tensor_binding_parity_ready_count",
        "muon_model_shape_aware_training_tensor_binding_kernel_executed_count",
        "muon_model_shape_aware_training_tensor_binding_product_native_ready_count",
        "muon_model_shape_aware_training_loop_ready_count",
        "muon_model_shape_aware_training_loop_native_step_count",
        "muon_model_shape_aware_training_loop_native_kernel_launch_count",
        "muon_model_shape_aware_training_loop_product_native_ready_count",
        "muon_model_shape_aware_e2e_shadow_matrix_ready_count",
        "muon_model_shape_aware_e2e_shadow_matrix_case_count",
        "muon_model_shape_aware_e2e_shadow_matrix_report_only_case_count",
        "muon_model_shape_aware_e2e_shadow_matrix_product_native_ready_count",
        "muon_model_shape_aware_canary_rollout_policy_ready_count",
        "muon_model_shape_aware_canary_rollout_policy_runtime_dispatch_ready_count",
        "muon_model_shape_aware_canary_rollout_policy_native_dispatch_allowed_count",
        "muon_model_shape_aware_canary_rollout_policy_training_path_enabled_count",
        "muon_model_shape_aware_canary_rollout_policy_product_native_ready_count",
        "muon_model_shape_aware_dispatch_review_gate_ready_count",
        "muon_model_shape_aware_dispatch_review_product_native_ready_count",
        "muon_model_shape_aware_owner_release_hold_ready_count",
        "muon_model_shape_aware_owner_release_hold_product_native_ready_count",
        "muon_model_shape_aware_request_schema_ui_non_exposure_ready_count",
        "muon_model_shape_aware_request_schema_ui_forbidden_token_hit_count",
        "muon_model_shape_aware_product_native_ready_count",
        "plugin_optimizer_count",
        "plugin_selected_optimizer_gate_ready_count",
        "plugin_selected_optimizer_gate_pending_count",
        "plugin_selected_family_owner_release_hold_family_count",
        "plugin_selected_family_owner_release_hold_optimizer_count",
        "plugin_selected_family_owner_release_hold_product_native_ready_count",
        "plugin_selected_family_request_schema_ui_family_count",
        "plugin_selected_family_request_schema_ui_optimizer_count",
        "plugin_selected_family_request_schema_ui_forbidden_token_hit_count",
        "plugin_selected_family_request_schema_ui_product_native_ready_count",
        "plugin_selected_simple_formula_e2e_shadow_case_count",
        "plugin_selected_simple_formula_canary_rollout_policy_ready_count",
        "plugin_selected_simple_formula_dispatch_review_ready_count",
        "plugin_selected_native_ready_count",
    )
    counts = {field: int(source.get(field, 0) or 0) for field in fields}
    for field, value in source.items():
        field_name = str(field)
        if not (field_name.endswith("_count") or field_name.endswith("_ready")):
            continue
        if isinstance(value, bool):
            counts[field_name] = int(value)
        elif isinstance(value, int):
            counts[field_name] = int(value)
    return counts


def _priority_next_gates(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    gates: list[str] = []
    for item in value:
        report = _as_dict(item)
        gate = str(report.get("next_gate") or "")
        if gate:
            gates.append(gate)
    return gates


def _ready_for_review(report: Mapping[str, Any]) -> bool:
    for key, value in report.items():
        if str(key).startswith("ready_for_") and str(key).endswith("_review") and value is True:
            return True
    return report.get("ready_for_owner_review") is True or report.get("ready_for_owner_release_review") is True


def _top_level_default_off(report: Mapping[str, Any]) -> bool:
    unsafe_non_empty_fields = ("post_release_request_fields", *UNSAFE_NON_EMPTY_FIELDS)
    return all(report.get(field) is not True for field in UNSAFE_TRUE_FIELDS) and all(
        not _as_dict(report.get(field)) for field in unsafe_non_empty_fields
    )


def _evidence_blockers(
    summaries: Mapping[str, Mapping[str, Any]],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    for gate, summary in summaries.items():
        if not summary.get("present"):
            blocked.append(f"native_update_release_gate_missing:{gate}")
            continue
        if summary.get("gate") != gate:
            blocked.append(f"native_update_release_gate_mismatch:{gate}")
        if not summary.get("ok"):
            blocked.append(f"native_update_release_gate_not_ok:{gate}")
        if not summary.get("evidence_ready"):
            blocked.append(f"native_update_release_gate_evidence_not_ready:{gate}")
        if not summary.get("ready_for_review"):
            blocked.append(f"native_update_release_gate_not_ready_for_review:{gate}")
        if not summary.get("default_off"):
            blocked.append(f"native_update_release_gate_not_default_off:{gate}")
        if summary.get("source_payload_digest_match") is False:
            blocked.append(f"native_update_release_gate_source_payload_digest_mismatch:{gate}")
        blocked.extend(_string_list(summary.get("unsafe_claims")))
    for event in failure_events:
        blocked.append(f"native_update_release_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"native_update_release_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _supplemental_blockers(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    for gate, summary in summaries.items():
        if not summary.get("present"):
            continue
        if summary.get("gate") != gate:
            blocked.append(f"native_update_release_supplemental_gate_mismatch:{gate}")
        if not summary.get("ok"):
            blocked.append(f"native_update_release_supplemental_gate_not_ok:{gate}")
        if not summary.get("evidence_ready"):
            blocked.append(f"native_update_release_supplemental_gate_evidence_not_ready:{gate}")
        if not summary.get("ready_for_review"):
            blocked.append(f"native_update_release_supplemental_gate_not_ready_for_review:{gate}")
        if not summary.get("default_off"):
            blocked.append(f"native_update_release_supplemental_gate_not_default_off:{gate}")
        if summary.get("source_payload_digest_match") is False:
            blocked.append(f"native_update_release_supplemental_gate_source_payload_digest_mismatch:{gate}")
        blocked.extend(_string_list(summary.get("unsafe_claims")))
        if gate == "optimizer_family_coverage":
            blocked.extend(_optimizer_family_coverage_blockers(summary))
    return _dedupe(blocked)


def _optimizer_family_coverage_blockers(summary: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    next_step = str(summary.get("recommended_next_step") or "")
    if not next_step:
        blocked.append("native_update_release_optimizer_family_coverage_next_step_missing")
    elif not _optimizer_owner_release_gate(next_step):
        blocked.append("native_update_release_optimizer_family_coverage_next_step_not_owner_release_hold")
    priority_next_gates = _string_list(summary.get("priority_next_gates"))
    if not priority_next_gates:
        blocked.append("native_update_release_optimizer_family_coverage_priority_gates_missing")
    for gate in priority_next_gates:
        if not _optimizer_owner_release_gate(gate):
            blocked.append(
                "native_update_release_optimizer_family_coverage_priority_gate_not_owner_release_hold"
            )
    return blocked


def _optimizer_owner_release_gate(value: object) -> bool:
    text = str(value).lower()
    return (
        "await explicit owner" in text
        or "until explicit owner" in text
        or "record explicit owner" in text
        or "explicit owner/release approval" in text
    )


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "present": bool(review),
        "reviewer": str(review.get("reviewer") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "requested_scope": str(review.get("requested_scope") or ""),
        "approve_native_update_release_review_package": review.get(
            "approve_native_update_release_review_package"
        )
        is True,
    }
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        summary[field] = bool(review.get(field, False))
    for field in REQUIRED_REVIEW_ACKS:
        summary[field] = review.get(field) is True
    acknowledged_gates = _as_dict(review.get("acknowledged_gates"))
    summary["acknowledged_gates"] = {str(gate): _as_dict(value) for gate, value in acknowledged_gates.items()}
    acknowledged_supplemental = _as_dict(review.get("acknowledged_supplemental_gates"))
    summary["acknowledged_supplemental_gates"] = {
        str(gate): _as_dict(value) for gate, value in acknowledged_supplemental.items()
    }
    return summary


def _review_blockers(
    review: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    supplemental: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if not review.get("present"):
        return []
    blocked: list[str] = []
    if not review.get("reviewer"):
        blocked.append("native_update_release_reviewer_missing")
    if not review.get("reviewed_at"):
        blocked.append("native_update_release_reviewed_at_missing")
    if review.get("requested_scope") != SCOPE:
        blocked.append("native_update_release_requested_scope_invalid")
    for field in UNSAFE_REVIEW_APPROVAL_FIELDS:
        if review.get(field):
            blocked.append(f"native_update_release_unsafe_review_approval:{field}")
    for field in REQUIRED_REVIEW_ACKS:
        if not review.get(field):
            blocked.append(f"native_update_release_review_ack_missing:{field}")
    if review.get("approve_native_update_release_review_package") is True:
        blocked.extend(_review_expected_gate_ack_blockers(review, summaries))
        blocked.extend(_review_supplemental_ack_blockers(review, supplemental))
    return _dedupe(blocked)


def _review_expected_gate_ack_blockers(
    review: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    acknowledgements = _as_dict(review.get("acknowledged_gates"))
    for gate, summary in summaries.items():
        acknowledgement = _as_dict(acknowledgements.get(gate))
        if not acknowledgement:
            blocked.append(f"native_update_release_review_gate_ack_missing:{gate}")
            continue
        if str(acknowledgement.get("digest") or "") != str(summary.get("digest") or ""):
            blocked.append(f"native_update_release_review_gate_ack_digest_mismatch:{gate}")
        if str(acknowledgement.get("decision") or "") != str(summary.get("decision") or ""):
            blocked.append(f"native_update_release_review_gate_ack_decision_mismatch:{gate}")
        for field in ("evidence_ready", "ready_for_review", "default_off"):
            if acknowledgement.get(field) is not summary.get(field):
                blocked.append(f"native_update_release_review_gate_ack_{field}_mismatch:{gate}")
    return blocked


def _review_supplemental_ack_blockers(
    review: Mapping[str, Any],
    supplemental: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blocked: list[str] = []
    acknowledgements = _as_dict(review.get("acknowledged_supplemental_gates"))
    for gate, summary in supplemental.items():
        if not summary.get("present"):
            continue
        acknowledgement = _as_dict(acknowledgements.get(gate))
        if not acknowledgement:
            blocked.append(f"native_update_release_review_supplemental_ack_missing:{gate}")
            continue
        if str(acknowledgement.get("digest") or "") != str(summary.get("digest") or ""):
            blocked.append(f"native_update_release_review_supplemental_ack_digest_mismatch:{gate}")
        if str(acknowledgement.get("recommended_next_step") or "") != str(
            summary.get("recommended_next_step") or ""
        ):
            blocked.append(f"native_update_release_review_supplemental_ack_next_step_mismatch:{gate}")
        if _string_list(acknowledgement.get("priority_next_gates")) != _string_list(
            summary.get("priority_next_gates")
        ):
            blocked.append(f"native_update_release_review_supplemental_ack_priority_gates_mismatch:{gate}")
        if gate == "optimizer_family_coverage" and _as_dict(
            acknowledgement.get("optimizer_family_counts")
        ) != _as_dict(summary.get("optimizer_family_counts")):
            blocked.append(f"native_update_release_review_supplemental_ack_optimizer_family_counts_mismatch:{gate}")
        if gate == "optimizer_family_coverage":
            if int(acknowledgement.get("source_count", 0) or 0) != int(summary.get("source_count", 0) or 0):
                blocked.append(f"native_update_release_review_supplemental_ack_source_count_mismatch:{gate}")
            if _string_list(acknowledgement.get("source_names")) != _string_list(summary.get("source_names")):
                blocked.append(f"native_update_release_review_supplemental_ack_source_names_mismatch:{gate}")
            if acknowledgement.get("source_payload_digest_match") is not summary.get("source_payload_digest_match"):
                blocked.append(
                    f"native_update_release_review_supplemental_ack_source_payload_digest_match_mismatch:{gate}"
                )
        for field in ("evidence_ready", "ready_for_review", "default_off"):
            if acknowledgement.get(field) is not summary.get(field):
                blocked.append(f"native_update_release_review_supplemental_ack_{field}_mismatch:{gate}")
    return blocked


def _review_template(
    summaries: Mapping[str, Mapping[str, Any]],
    supplemental: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    template = {
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": SCOPE,
        "approve_native_update_release_review_package": False,
    }
    for field in REQUIRED_REVIEW_ACKS:
        template[field] = False
    template["acknowledged_gates"] = {
        gate: {
            "digest": summary.get("digest"),
            "decision": summary.get("decision"),
            "evidence_ready": summary.get("evidence_ready"),
            "ready_for_review": summary.get("ready_for_review"),
            "default_off": summary.get("default_off"),
        }
        for gate, summary in summaries.items()
    }
    template["acknowledged_supplemental_gates"] = {
        gate: {
            "digest": summary.get("digest"),
            "decision": summary.get("decision"),
            "evidence_ready": summary.get("evidence_ready"),
            "ready_for_review": summary.get("ready_for_review"),
            "default_off": summary.get("default_off"),
            "recommended_next_step": summary.get("recommended_next_step"),
            "priority_next_gates": summary.get("priority_next_gates"),
            **(
                {
                    "optimizer_family_counts": summary.get("optimizer_family_counts"),
                    "source_count": summary.get("source_count"),
                    "source_names": summary.get("source_names"),
                    "source_payload_digest_match": summary.get("source_payload_digest_match"),
                }
                if gate == "optimizer_family_coverage"
                else {}
            ),
        }
        for gate, summary in supplemental.items()
        if summary.get("present")
    }
    return template


def _owner_release_review_handoff(
    decision: str,
    evidence_ready: bool,
    review_template: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "handoff": "native_update_release_owner_review_handoff_v0",
        "ready_for_owner_release_review": evidence_ready,
        "decision": decision,
        "action_required": "collect_signed_native_update_release_review"
        if decision == HOLD_DECISION and evidence_ready
        else _recommended_next_step(decision, evidence_ready),
        "blocked_reasons": list(blockers),
        "required_review_fields": [
            "reviewer",
            "reviewed_at",
            "requested_scope",
            "approve_native_update_release_review_package",
            *REQUIRED_REVIEW_ACKS,
        ],
        "required_requested_scope": SCOPE,
        "required_supplemental_acknowledgements": sorted(
            _as_dict(review_template.get("acknowledged_supplemental_gates")).keys()
        ),
        "supplemental_acknowledgement_counts": {
            gate: _as_dict(ack.get("optimizer_family_counts"))
            for gate, ack in _as_dict(review_template.get("acknowledged_supplemental_gates")).items()
            if gate == "optimizer_family_coverage" and _as_dict(ack.get("optimizer_family_counts"))
        },
        "supplemental_acknowledgement_sources": {
            gate: {
                "source_count": int(ack.get("source_count", 0) or 0),
                "source_names": _string_list(ack.get("source_names")),
                "source_payload_digest_match": ack.get("source_payload_digest_match") is True,
            }
            for gate, ack in _as_dict(review_template.get("acknowledged_supplemental_gates")).items()
            if gate == "optimizer_family_coverage"
        },
        "required_gate_acknowledgements": sorted(_as_dict(review_template.get("acknowledged_gates")).keys()),
        "release_review_template_digest": _digest_payload(_as_dict(review_template)),
        "must_remain_false": list(UNSAFE_TRUE_FIELDS),
        "must_remain_empty": ["post_release_request_fields", *UNSAFE_NON_EMPTY_FIELDS],
        "notes": [
            "This handoff is report-only and does not record owner approval.",
            "Signing the package still keeps request/UI/schema/backend routes and training launch disabled.",
            "A separate owner release direction is required before product exposure work.",
        ],
    }


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    for field in UNSAFE_TRUE_FIELDS:
        if value.get(field) is True:
            blocked.append(f"native_update_release_unsafe_claim:{owner}:{field}")
    for field in UNSAFE_NON_EMPTY_FIELDS:
        if bool(value.get(field)):
            blocked.append(f"native_update_release_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _allowed_next_actions(decision: str, evidence_ready: bool) -> list[str]:
    if decision == READY_DECISION:
        return ["archive_native_update_release_review_package", "await_owner_release_direction"]
    if evidence_ready:
        return ["collect_signed_native_update_release_review"]
    return ["repair_missing_or_unsafe_native_update_release_evidence"]


def _recommended_next_step(decision: str, evidence_ready: bool) -> str:
    if decision == READY_DECISION:
        return "archive release review package and wait for a separate owner release direction"
    if evidence_ready:
        return "record owner release review while keeping product exposure disabled"
    return "repair missing or unsafe native-update release evidence before review"


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {k: v for k, v in value.items() if not str(k).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build native-update release review package.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--release-review")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    payload = build_native_update_release_review_package(
        gate_artifacts=load_gate_artifacts(args.artifact_dir),
        release_review=load_json(args.release_review) if args.release_review else None,
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
