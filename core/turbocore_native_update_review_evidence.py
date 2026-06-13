"""Product/release review evidence helpers for native-update promotion."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def compact_product_exposure_decision(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "gate": str(report.get("gate", "") or ""),
        "decision": str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or ""),
        "ok": bool(report.get("ok", False)),
        "evidence_ready": bool(report.get("evidence_ready", False)),
        "ready_for_product_exposure_review": bool(report.get("ready_for_product_exposure_review", False)),
        "product_exposure_decision_recorded": bool(report.get("product_exposure_decision_recorded", False)),
        "manual_review_required": bool(report.get("manual_review_required", False)),
        "product_exposure_allowed": bool(report.get("product_exposure_allowed", False)),
        "training_launch_allowed": bool(report.get("training_launch_allowed", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ready_for_ui": bool(report.get("ready_for_ui", False)),
        "post_product_exposure_request_fields": _as_dict(report.get("post_product_exposure_request_fields")),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def compact_release_review_package(report: Mapping[str, Any]) -> dict[str, Any]:
    supplemental_summaries = _as_dict(report.get("supplemental_gate_summaries"))
    optimizer_handoff_counts = _optimizer_family_handoff_counts(report)
    optimizer_handoff_sources = _optimizer_family_handoff_sources(report)
    template_digest = _release_review_template_digest(report)
    handoff_template_digest = _handoff_release_review_template_digest(report)
    owner_record = _as_dict(report.get("owner_release_review_record"))
    compact_supplemental = {
        gate: _compact_supplemental_gate(gate, _as_dict(summary))
        for gate, summary in supplemental_summaries.items()
    }
    return {
        "present": bool(report),
        "ok": bool(report.get("ok", False)),
        "evidence_ready": bool(report.get("evidence_ready", False)),
        "ready_for_review": bool(report.get("ready_for_review", False)),
        "ready_for_owner_release_review": bool(report.get("ready_for_owner_release_review", False)),
        "release_review_recorded": bool(report.get("release_review_recorded", False)),
        "default_off": bool(report.get("default_off", False)),
        "expected_gate_count": int(report.get("expected_gate_count", 0) or 0),
        "present_gate_count": int(report.get("present_gate_count", 0) or 0),
        "default_off_gate_count": int(report.get("default_off_gate_count", 0) or 0),
        "supplemental_gate_count": int(report.get("supplemental_gate_count", 0) or 0),
        "present_supplemental_gate_count": int(report.get("present_supplemental_gate_count", 0) or 0),
        "default_off_supplemental_gate_count": int(report.get("default_off_supplemental_gate_count", 0) or 0),
        "supplemental_gate_summaries": compact_supplemental,
        "optimizer_family_coverage": _compact_optimizer_family_coverage(
            _as_dict(supplemental_summaries.get("optimizer_family_coverage")),
            optimizer_handoff_counts=optimizer_handoff_counts,
            optimizer_handoff_sources=optimizer_handoff_sources,
        ),
        "native_update_optimizer_multitensor_release_hold": compact_supplemental.get(
            "native_update_optimizer_multitensor_release_hold",
            _compact_supplemental_gate("native_update_optimizer_multitensor_release_hold", {}),
        ),
        "release_review_template_digest": template_digest,
        "handoff_release_review_template_digest": handoff_template_digest,
        "handoff_release_review_template_digest_match": bool(
            template_digest and handoff_template_digest == template_digest
        ),
        "owner_release_review_record": _compact_owner_release_review_record(owner_record),
        "post_release_request_fields": _as_dict(report.get("post_release_request_fields")),
        "decision": str(report.get("decision", "") or ""),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def compact_stable_first_release_scope(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "gate": str(report.get("gate", "") or ""),
        "ok": bool(report.get("ok", False)),
        "stable_first_release_scope": str(report.get("stable_first_release_scope") or ""),
        "stable_first_release_blocked_by_turbocore_optimizer": bool(
            report.get("stable_first_release_blocked_by_turbocore_optimizer", False)
        ),
        "turbocore_optimizer_default_off_release_scope_ready": bool(
            report.get("turbocore_optimizer_default_off_release_scope_ready", False)
        ),
        "release_claim_allowed": bool(report.get("release_claim_allowed", False)),
        "native_training_claim_allowed": bool(report.get("native_training_claim_allowed", False)),
        "product_exposure_allowed": bool(report.get("product_exposure_allowed", False)),
        "runtime_dispatch_allowed": bool(report.get("runtime_dispatch_allowed", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "summary": {
            "stable_first_release_turbocore_optimizer_blocker_count": int(
                summary.get("stable_first_release_turbocore_optimizer_blocker_count", 0) or 0
            ),
            "turbocore_optimizer_default_off_release_scope_ready_count": int(
                summary.get("turbocore_optimizer_default_off_release_scope_ready_count", 0) or 0
            ),
            "owner_release_approval_recorded_count": int(
                summary.get("owner_release_approval_recorded_count", 0) or 0
            ),
            "owner_release_direction_recorded_count": int(
                summary.get("owner_release_direction_recorded_count", 0) or 0
            ),
            "owner_release_direction_approval_recorded_count": int(
                summary.get("owner_release_direction_approval_recorded_count", 0) or 0
            ),
            "product_exposure_decision_recorded_count": int(
                summary.get("product_exposure_decision_recorded_count", 0) or 0
            ),
            "product_training_route_binding_ready_count": int(
                summary.get("product_training_route_binding_ready_count", 0) or 0
            ),
            "run_local_adapter_staged_count": int(summary.get("run_local_adapter_staged_count", 0) or 0),
            "runtime_config_patch_applied_count": int(
                summary.get("runtime_config_patch_applied_count", 0) or 0
            ),
            "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        },
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def compact_owner_release_direction_record(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "gate": str(report.get("gate", "") or ""),
        "ok": bool(report.get("ok", False)),
        "owner_direction_packet_ready": report.get("owner_direction_packet_ready") is True,
        "signed_direction_present": report.get("signed_direction_present") is True,
        "signed_direction_valid": report.get("signed_direction_valid") is True,
        "owner_release_direction_recorded": report.get("owner_release_direction_recorded") is True,
        "owner_release_approval_recorded": report.get("owner_release_approval_recorded") is True,
        "signed_owner_release_direction_digest_match": (
            report.get("signed_owner_release_direction_digest_match") is True
        ),
        "decision": str(report.get("decision") or ""),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "summary": _as_dict(report.get("summary")),
        "unsafe_claims": _owner_direction_record_unsafe_claims(report),
    }


def owner_release_direction_record_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return ["native_update_owner_release_direction_record_missing"]
    blocked = _strings(report.get("blocked_reasons"))
    if report.get("ok") is not True:
        blocked.append("native_update_owner_release_direction_record_not_ok")
    for field in (
        "owner_direction_packet_ready",
        "signed_direction_present",
        "signed_direction_valid",
        "owner_release_direction_recorded",
        "owner_release_approval_recorded",
        "signed_owner_release_direction_digest_match",
    ):
        if report.get(field) is not True:
            blocked.append(f"native_update_owner_release_direction_record_{field}_failed")
    if str(report.get("decision") or "") != "native_update_owner_release_direction_recorded_default_off":
        blocked.append("native_update_owner_release_direction_record_decision_not_recorded_default_off")
    for claim in _owner_direction_record_unsafe_claims(report):
        blocked.append(f"native_update_owner_release_direction_record_unsafe:{claim}")
    return _dedupe(blocked)


def stable_first_release_scope_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    compact = compact_stable_first_release_scope(report)
    summary = _as_dict(compact.get("summary"))
    blocked = _strings(report.get("blocked_reasons"))
    if compact.get("ok") is not True:
        blocked.append("turbocore_optimizer_stable_first_release_scope_not_ok")
    if compact.get("turbocore_optimizer_default_off_release_scope_ready") is not True:
        blocked.append("turbocore_optimizer_default_off_release_scope_not_ready")
    if compact.get("stable_first_release_blocked_by_turbocore_optimizer") is True:
        blocked.append("stable_first_release_blocked_by_turbocore_optimizer")
    if int(summary.get("stable_first_release_turbocore_optimizer_blocker_count", 0) or 0) != 0:
        blocked.append("stable_first_release_turbocore_optimizer_blockers_present")
    if int(summary.get("turbocore_optimizer_default_off_release_scope_ready_count", 0) or 0) != 1:
        blocked.append("turbocore_optimizer_default_off_release_scope_ready_count_missing")
    for field in (
        "native_training_claim_allowed",
        "product_exposure_allowed",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "training_path_enabled",
    ):
        if compact.get(field) is True:
            blocked.append(f"stable_first_release_scope_unsafe:{field}")
    for field in (
        "owner_release_approval_recorded_count",
        "product_exposure_decision_recorded_count",
        "product_training_route_binding_ready_count",
        "run_local_adapter_staged_count",
        "runtime_config_patch_applied_count",
        "training_path_enabled_count",
    ):
        if int(summary.get(field, 0) or 0) != 0:
            blocked.append(f"stable_first_release_scope_unsafe:{field}")
    return _dedupe(blocked)


def product_exposure_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return ["native_update_product_exposure_decision_missing"]
    blocked = _strings(report.get("blocked_reasons")) + _strings(report.get("promotion_blockers"))
    if report.get("ok") is not True:
        blocked.append("native_update_product_exposure_decision_not_ok")
    if report.get("evidence_ready") is not True:
        blocked.append("native_update_product_exposure_evidence_not_ready")
    if report.get("ready_for_product_exposure_review") is not True:
        blocked.append("native_update_product_exposure_not_ready_for_review")
    if report.get("product_exposure_decision_recorded") is not True:
        blocked.append("native_update_product_exposure_decision_not_recorded")
    if report.get("post_product_exposure_request_fields", {}) != {}:
        blocked.append("native_update_product_exposure_request_fields_present")
    for field in (
        "product_exposure_allowed",
        "training_launch_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ready_for_ui",
        "backend_router_registered",
    ):
        if report.get(field) is True:
            blocked.append(f"native_update_product_exposure_unsafe:{field}")
    return _dedupe(blocked)


def release_review_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return ["native_update_release_review_package_missing"]
    blocked = _strings(report.get("blocked_reasons")) + _strings(report.get("promotion_blockers"))
    if report.get("ok") is not True:
        blocked.append("native_update_release_review_package_not_ok")
    if report.get("evidence_ready") is not True:
        blocked.append("native_update_release_review_evidence_not_ready")
    if report.get("ready_for_review") is not True:
        blocked.append("native_update_release_review_not_ready_for_review")
    if report.get("ready_for_owner_release_review") is not True:
        blocked.append("native_update_release_review_not_ready_for_owner_review")
    if report.get("release_review_recorded") is not True:
        blocked.append("native_update_release_review_not_recorded")
    if report.get("default_off") is not True:
        blocked.append("native_update_release_review_not_default_off")
    expected = int(report.get("expected_gate_count", 0) or 0)
    present = int(report.get("present_gate_count", 0) or 0)
    default_off = int(report.get("default_off_gate_count", 0) or 0)
    if expected <= 0 or present != expected:
        blocked.append("native_update_release_review_expected_gates_not_present")
    if expected <= 0 or default_off != expected:
        blocked.append("native_update_release_review_gates_not_default_off")
    supplemental_expected = int(report.get("supplemental_gate_count", 0) or 0)
    supplemental_present = int(report.get("present_supplemental_gate_count", 0) or 0)
    supplemental_default_off = int(report.get("default_off_supplemental_gate_count", 0) or 0)
    if supplemental_expected <= 0:
        blocked.append("native_update_release_review_supplemental_gates_missing")
    elif supplemental_present != supplemental_expected:
        blocked.append("native_update_release_review_supplemental_gates_not_present")
    if supplemental_expected <= 0 or supplemental_default_off != supplemental_expected:
        blocked.append("native_update_release_review_supplemental_gates_not_default_off")
    supplemental_summaries = _as_dict(report.get("supplemental_gate_summaries"))
    for gate, summary in supplemental_summaries.items():
        if gate == "optimizer_family_coverage":
            continue
        blocked.extend(_supplemental_gate_blockers(str(gate), _as_dict(summary)))
    blocked.extend(_optimizer_family_coverage_blockers(report))
    if _handoff_release_review_template_digest(report) != _release_review_template_digest(report):
        blocked.append("native_update_release_review_handoff_template_digest_mismatch")
    blocked.extend(_owner_release_review_record_blockers(_as_dict(report.get("owner_release_review_record"))))
    if report.get("post_release_request_fields", {}) != {}:
        blocked.append("native_update_release_review_request_fields_present")
    for field in (
        "release_gate_open",
        "training_launch_allowed",
        "runtime_dispatch_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ready_for_ui",
        "backend_router_registered",
    ):
        if report.get(field) is True:
            blocked.append(f"native_update_release_review_unsafe:{field}")
    return _dedupe(blocked)


def _compact_owner_release_review_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(record),
        "owner_packet_ready": record.get("owner_packet_ready") is True,
        "signed_review_present": record.get("signed_review_present") is True,
        "signed_review_valid": record.get("signed_review_valid") is True,
        "approval_recorded": record.get("approval_recorded") is True,
        "release_review_recorded": record.get("release_review_recorded") is True,
        "signed_review_digest_match": record.get("signed_review_digest_match") is True,
        "release_package_decision": str(record.get("release_package_decision") or ""),
        "blocked_reasons": _strings(record.get("blocked_reasons")),
        "unsafe_claims": _owner_record_unsafe_claims(record),
    }


def _owner_release_review_record_blockers(record: Mapping[str, Any]) -> list[str]:
    if not record:
        return []
    blocked: list[str] = []
    for field in (
        "owner_packet_ready",
        "signed_review_present",
        "signed_review_valid",
        "approval_recorded",
        "release_review_recorded",
        "signed_review_digest_match",
    ):
        if record.get(field) is not True:
            blocked.append(f"native_update_release_review_owner_record_{field}_failed")
    if str(record.get("release_package_decision") or "") != "native_update_release_review_recorded_default_off":
        blocked.append("native_update_release_review_owner_record_package_decision_not_recorded_default_off")
    for claim in _owner_record_unsafe_claims(record):
        blocked.append(f"native_update_release_review_owner_record_unsafe:{claim}")
    blocked.extend(
        f"native_update_release_review_owner_record:{reason}"
        for reason in _strings(record.get("blocked_reasons"))
    )
    return blocked


def _compact_optimizer_family_coverage(
    summary: Mapping[str, Any],
    *,
    optimizer_handoff_counts: Mapping[str, Any] | None = None,
    optimizer_handoff_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    optimizer_counts = _int_dict(summary.get("optimizer_family_counts"))
    handoff_counts = _int_dict(optimizer_handoff_counts)
    source_summary = _optimizer_family_source_summary(summary)
    handoff_sources = _optimizer_family_source_summary(_as_dict(optimizer_handoff_sources))
    return {
        "present": bool(summary.get("present", False)),
        "ok": bool(summary.get("ok", False)),
        "evidence_ready": bool(summary.get("evidence_ready", False)),
        "ready_for_review": bool(summary.get("ready_for_review", False)),
        "default_off": bool(summary.get("default_off", False)),
        "source_count": source_summary["source_count"],
        "source_names": source_summary["source_names"],
        "source_payload_digest_match": source_summary["source_payload_digest_match"],
        "handoff_sources": handoff_sources,
        "handoff_sources_match": bool(source_summary and handoff_sources == source_summary),
        "optimizer_family_counts": optimizer_counts,
        "handoff_counts": handoff_counts,
        "handoff_counts_match": bool(optimizer_counts and handoff_counts == optimizer_counts),
        "recommended_next_step": str(summary.get("recommended_next_step") or ""),
        "priority_group_count": int(summary.get("priority_group_count", 0) or 0),
        "priority_next_gates": _strings(summary.get("priority_next_gates")),
        "unsafe_claims": _strings(summary.get("unsafe_claims")),
        "blocked_reasons": _strings(summary.get("blocked_reasons")),
    }


def _compact_supplemental_gate(gate: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(summary.get("present", False)),
        "gate": str(summary.get("gate") or gate),
        "ok": bool(summary.get("ok", False)),
        "evidence_ready": bool(summary.get("evidence_ready", False)),
        "ready_for_review": bool(summary.get("ready_for_review", False)),
        "default_off": bool(summary.get("default_off", False)),
        "recommended_next_step": str(summary.get("recommended_next_step") or ""),
        "priority_group_count": int(summary.get("priority_group_count", 0) or 0),
        "priority_next_gates": _strings(summary.get("priority_next_gates")),
        "unsafe_claims": _strings(summary.get("unsafe_claims")),
        "blocked_reasons": _strings(summary.get("blocked_reasons")),
    }


def _supplemental_gate_blockers(gate: str, summary: Mapping[str, Any]) -> list[str]:
    if not summary or summary.get("present") is not True:
        return [f"native_update_release_review_supplemental_gate_missing:{gate}"]
    blocked: list[str] = []
    for field in ("ok", "evidence_ready", "ready_for_review", "default_off"):
        if summary.get(field) is not True:
            blocked.append(f"native_update_release_review_supplemental_gate_{field}_failed:{gate}")
    for claim in _strings(summary.get("unsafe_claims")):
        blocked.append(f"native_update_release_review_supplemental_gate_unsafe:{gate}:{claim}")
    return blocked


def _optimizer_family_coverage_blockers(report: Mapping[str, Any]) -> list[str]:
    summaries = _as_dict(report.get("supplemental_gate_summaries"))
    summary = _as_dict(summaries.get("optimizer_family_coverage"))
    if not summary or summary.get("present") is not True:
        return ["native_update_release_review_optimizer_family_coverage_missing"]
    blocked = [
        f"native_update_release_review_optimizer_family_coverage:{reason}"
        for reason in _strings(summary.get("blocked_reasons"))
    ]
    if summary.get("ok") is not True:
        blocked.append("native_update_release_review_optimizer_family_coverage_not_ok")
    if summary.get("evidence_ready") is not True:
        blocked.append("native_update_release_review_optimizer_family_coverage_not_evidence_ready")
    if summary.get("ready_for_review") is not True:
        blocked.append("native_update_release_review_optimizer_family_coverage_not_ready_for_review")
    if summary.get("default_off") is not True:
        blocked.append("native_update_release_review_optimizer_family_coverage_not_default_off")
    if summary.get("source_payload_digest_match") is False:
        blocked.append("native_update_release_review_optimizer_family_coverage_source_payload_digest_mismatch")
    for claim in _strings(summary.get("unsafe_claims")):
        blocked.append(f"native_update_release_review_optimizer_family_coverage_unsafe:{claim}")
    next_step = str(summary.get("recommended_next_step") or "")
    if not next_step:
        blocked.append("native_update_release_review_optimizer_family_coverage_next_step_missing")
    elif not _optimizer_owner_release_gate(next_step):
        blocked.append("native_update_release_review_optimizer_family_coverage_next_step_not_owner_release_hold")
    priority_next_gates = _strings(summary.get("priority_next_gates"))
    if not priority_next_gates:
        blocked.append("native_update_release_review_optimizer_family_coverage_priority_gates_missing")
    for gate in priority_next_gates:
        if not _optimizer_owner_release_gate(gate):
            blocked.append(
                "native_update_release_review_optimizer_family_coverage_priority_gate_not_owner_release_hold"
            )
    handoff_counts = _optimizer_family_handoff_counts(report)
    optimizer_counts = _int_dict(summary.get("optimizer_family_counts"))
    if not handoff_counts or handoff_counts != optimizer_counts:
        blocked.append("native_update_release_review_optimizer_family_handoff_counts_mismatch")
    handoff_sources = _optimizer_family_handoff_sources(report)
    source_summary = _optimizer_family_source_summary(summary)
    if handoff_sources != source_summary:
        blocked.append("native_update_release_review_optimizer_family_handoff_sources_mismatch")
    return blocked


def _optimizer_owner_release_gate(value: object) -> bool:
    text = str(value).lower()
    return (
        "await explicit owner" in text
        or "until explicit owner" in text
        or "record explicit owner" in text
        or "explicit owner/release approval" in text
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_dict(value: Any) -> dict[str, int]:
    return {str(key): int(item or 0) for key, item in _as_dict(value).items()}


def _optimizer_family_handoff_counts(report: Mapping[str, Any]) -> dict[str, int]:
    handoff = _as_dict(report.get("owner_release_review_handoff"))
    counts = _as_dict(handoff.get("supplemental_acknowledgement_counts"))
    return _int_dict(counts.get("optimizer_family_coverage"))


def _optimizer_family_source_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_count": int(summary.get("source_count", 0) or 0),
        "source_names": _strings(summary.get("source_names")),
        "source_payload_digest_match": summary.get("source_payload_digest_match") is not False,
    }


def _optimizer_family_handoff_sources(report: Mapping[str, Any]) -> dict[str, Any]:
    handoff = _as_dict(report.get("owner_release_review_handoff"))
    sources = _as_dict(handoff.get("supplemental_acknowledgement_sources"))
    return _optimizer_family_source_summary(_as_dict(sources.get("optimizer_family_coverage")))


def _handoff_release_review_template_digest(report: Mapping[str, Any]) -> str:
    handoff = _as_dict(report.get("owner_release_review_handoff"))
    return str(handoff.get("release_review_template_digest") or "")


def _release_review_template_digest(report: Mapping[str, Any]) -> str:
    return _digest_payload(_as_dict(report.get("release_review_template")))


def _owner_record_unsafe_claims(record: Mapping[str, Any]) -> list[str]:
    unsafe = []
    for key in (
        "product_exposure_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "training_path_enabled",
        "training_launch_executed",
    ):
        if record.get(key) is True:
            unsafe.append(key)
    return unsafe


def _owner_direction_record_unsafe_claims(record: Mapping[str, Any]) -> list[str]:
    unsafe = []
    for key in (
        "product_exposure_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "backend_router_registered",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "training_path_enabled",
        "training_launch_executed",
    ):
        if record.get(key) is True:
            unsafe.append(key)
    if _as_dict(record.get("post_owner_release_request_fields")):
        unsafe.append("post_owner_release_request_fields")
    return unsafe


def _digest_payload(value: Mapping[str, Any]) -> str:
    if not value:
        return ""
    payload = {str(key): item for key, item in value.items() if not str(key).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = [
    "compact_owner_release_direction_record",
    "compact_product_exposure_decision",
    "compact_release_review_package",
    "compact_stable_first_release_scope",
    "owner_release_direction_record_blockers",
    "product_exposure_blockers",
    "release_review_blockers",
    "stable_first_release_scope_blockers",
]
