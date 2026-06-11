"""Packaging / observability evidence gate for TurboCore V5-P38.

P38 consumes the already-built P37 broader real-route coverage gate and
packaging, telemetry, and report evidence payloads. It is a report-only gate:
it never launches training, emits request-adapter fields, exposes UI, or enables
default/auto rollout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_packaging_observability_evidence_gate_utils import (
    allowed_next_actions as _allowed_next_actions,
    as_dict as _as_dict,
    dedupe as _dedupe,
    default_off_confirmed as _default_off_confirmed,
    digest as _digest,
    event_list as _event_list,
    history_clear as _history_clear,
    history_summary as _history_summary,
    int_value as _int,
    list_value as _list,
    missing_row as _missing_row,
    recommended_next_step as _recommended_next_step,
    request_adapter_off as _request_adapter_off,
    required as _required,
    run_cli as _run_cli,
    source as _source,
    source_replayable as _source_replayable,
    string_list as _string_list,
)


P37_READY_DECISION = "broader_real_route_coverage_ready_default_off"
P38_READY_DECISION = "packaging_observability_evidence_gate_ready_default_off"
P38_BLOCKED_DECISION = "packaging_observability_evidence_gate_blocked_default_off"
DEFAULT_REQUIRED_PACKAGES = (
    "archive_ledger",
    "offline_runtime_pack",
    "evidence_bundle",
)
DEFAULT_REQUIRED_TELEMETRY_CHANNELS = (
    "state",
    "logs",
    "events",
    "reports",
)
DEFAULT_REQUIRED_REPORT_IDS = ("training_report",)
DEFAULT_REQUIRED_REPORT_SECTIONS = (
    "metrics",
    "diagnostics",
    "recovery",
    "reproducibility",
)
MIN_ARTIFACT_COUNT = 1
MIN_TELEMETRY_SAMPLE_RUNS = 1


def build_v5_packaging_observability_evidence_gate(
    *,
    p37_coverage_gate: Mapping[str, Any] | None = None,
    packaging_evidence: Sequence[Mapping[str, Any]] | None = None,
    observability_evidence: Sequence[Mapping[str, Any]] | None = None,
    report_evidence: Sequence[Mapping[str, Any]] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
    required_packages: Sequence[str] | None = None,
    required_telemetry_channels: Sequence[str] | None = None,
    required_report_ids: Sequence[str] | None = None,
    required_report_sections: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Gate packaging and observability evidence without enabling rollout."""

    p37 = _as_dict(p37_coverage_gate)
    package_inputs = _indexed_inputs(packaging_evidence or [], _package_id)
    telemetry_inputs = _indexed_inputs(observability_evidence or [], _channel_id)
    report_inputs = _indexed_inputs(report_evidence or [], _report_id)
    required_package_ids = _required(required_packages, DEFAULT_REQUIRED_PACKAGES)
    required_channels = _required(required_telemetry_channels, DEFAULT_REQUIRED_TELEMETRY_CHANNELS)
    required_reports = _required(required_report_ids, DEFAULT_REQUIRED_REPORT_IDS)
    required_sections = _required(required_report_sections, DEFAULT_REQUIRED_REPORT_SECTIONS)

    p37_summary = _p37_summary(p37)
    packaging_rows = [_package_row(name, package_inputs["items"].get(name)) for name in required_package_ids]
    telemetry_rows = [_telemetry_row(name, telemetry_inputs["items"].get(name)) for name in required_channels]
    report_rows = [_report_row(name, report_inputs["items"].get(name), required_sections) for name in required_reports]
    unexpected_packages = sorted(name for name in package_inputs["items"] if name not in set(required_package_ids))
    unexpected_channels = sorted(name for name in telemetry_inputs["items"] if name not in set(required_channels))
    unexpected_reports = sorted(name for name in report_inputs["items"] if name not in set(required_reports))
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    blockers = _blockers(
        p37_summary=p37_summary,
        packaging_rows=packaging_rows,
        telemetry_rows=telemetry_rows,
        report_rows=report_rows,
        unexpected_packages=unexpected_packages,
        unexpected_channels=unexpected_channels,
        unexpected_reports=unexpected_reports,
        package_duplicates=package_inputs["duplicates"],
        telemetry_duplicates=telemetry_inputs["duplicates"],
        report_duplicates=report_inputs["duplicates"],
        malformed_packages=package_inputs["malformed"],
        malformed_channels=telemetry_inputs["malformed"],
        malformed_reports=report_inputs["malformed"],
        required_package_ids=required_package_ids,
        required_channels=required_channels,
        required_reports=required_reports,
        required_sections=required_sections,
        failure_events=failure_events,
        rollback_events=rollback_events,
    )
    ready = not blockers
    decision = P38_READY_DECISION if ready else P38_BLOCKED_DECISION
    return {
        "schema_version": 1,
        "package": "turbocore_v5_packaging_observability_evidence_gate_v0",
        "gate": "v5_packaging_observability_evidence",
        "ok": ready,
        "packaging_observability_ready": ready,
        "evidence_gate_ready": ready,
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
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_p38_request_fields": {},
        "p37_coverage_gate_summary": p37_summary,
        "required_package_ids": required_package_ids,
        "required_telemetry_channels": required_channels,
        "required_report_ids": required_reports,
        "required_report_sections": required_sections,
        "packaging_matrix": packaging_rows,
        "observability_matrix": telemetry_rows,
        "report_matrix": report_rows,
        "unexpected_package_ids": unexpected_packages,
        "unexpected_telemetry_channels": unexpected_channels,
        "unexpected_report_ids": unexpected_reports,
        "duplicate_package_ids": package_inputs["duplicates"],
        "duplicate_telemetry_channels": telemetry_inputs["duplicates"],
        "duplicate_report_ids": report_inputs["duplicates"],
        "malformed_packaging_evidence": package_inputs["malformed"],
        "malformed_observability_evidence": telemetry_inputs["malformed"],
        "malformed_report_evidence": report_inputs["malformed"],
        "artifact_digest_summary": _artifact_digest_summary(packaging_rows),
        "telemetry_channel_summary": _telemetry_summary(telemetry_rows),
        "report_surface_summary": _report_summary(report_rows, required_sections),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "allowed_next_actions": _allowed_next_actions(ready, blockers),
        "recommended_next_step": _recommended_next_step(ready, blockers),
        "notes": [
            "P38 consumes already-produced packaging and observability evidence only.",
            "Fallback telemetry can satisfy observability when it is explicit and replayable.",
            "Even ready packaging/observability evidence does not authorize rollout, UI, request mapping, or launch.",
        ],
    }


def _p37_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(report),
        "ok": bool(report.get("ok", False)),
        "broader_real_route_coverage_ready": bool(report.get("broader_real_route_coverage_ready", False)),
        "coverage_gate_ready": bool(report.get("coverage_gate_ready", report.get("route_coverage_gate_ready", False))),
        "decision": str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or ""),
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "manual_review_required": bool(report.get("manual_review_required", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", True)),
        "post_fields_empty": not bool(_as_dict(report.get("post_p37_request_fields"))),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "unsafe_claims": _unsafe_claims(report, "p37"),
        "ready": _p37_ready(report),
    }


def _p37_ready(report: Mapping[str, Any]) -> bool:
    return bool(
        report
        and report.get("ok") is True
        and report.get("broader_real_route_coverage_ready") is True
        and bool(report.get("coverage_gate_ready", report.get("route_coverage_gate_ready", False)))
        and str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
        == P37_READY_DECISION
        and report.get("manual_review_required") is True
        and report.get("default_behavior_changed") is False
        and _default_off_confirmed(report)
        and _request_adapter_off(report)
        and not _as_dict(report.get("post_p37_request_fields"))
        and not _string_list(report.get("blocked_reasons"))
        and not _string_list(report.get("promotion_blockers"))
        and _history_clear(report, "failure_history_summary")
        and _history_clear(report, "rollback_history_summary")
        and not _unsafe_claims(report, "p37")
    )


def _package_row(package_id: str, evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    item = _as_dict(evidence)
    if not item:
        return _missing_row(package_id, "package", f"v5_p38_package_missing:{package_id}")
    blockers = _package_blockers(package_id, item)
    return {
        "package_id": package_id,
        "kind": str(item.get("kind") or item.get("package_kind") or ""),
        "present": True,
        "ready": not blockers,
        "ok": item.get("ok") is True,
        "complete": item.get("complete") is True,
        "portable": item.get("portable") is True,
        "artifact_count": _int(item.get("artifact_count", len(_list(item.get("artifacts"))))),
        "digest": _digest(item),
        "source": _source(item),
        "default_off": item.get("default_off") is True and _default_off_confirmed(item),
        "request_adapter_off": item.get("request_adapter_off") is True and _request_adapter_off(item),
        "blockers": blockers,
    }


def _package_blockers(package_id: str, item: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if item.get("ok") is not True:
        blocked.append(f"v5_p38_package_not_ok:{package_id}")
    if item.get("packaging_evidence_ready") is not True:
        blocked.append(f"v5_p38_package_evidence_not_ready:{package_id}")
    if item.get("complete") is not True:
        blocked.append(f"v5_p38_package_incomplete:{package_id}")
    if item.get("portable") is not True:
        blocked.append(f"v5_p38_package_not_portable:{package_id}")
    if _int(item.get("artifact_count", len(_list(item.get("artifacts"))))) < MIN_ARTIFACT_COUNT:
        blocked.append(f"v5_p38_package_artifact_count_insufficient:{package_id}")
    if not _digest(item):
        blocked.append(f"v5_p38_package_digest_missing:{package_id}")
    if not _source_replayable(item):
        blocked.append(f"v5_p38_package_source_missing:{package_id}")
    blocked.extend(_row_boundary_blockers(package_id, item))
    for reason in _string_list(item.get("promotion_blockers")):
        blocked.append(f"{package_id}:{reason}")
    for reason in _string_list(item.get("blocked_reasons")):
        blocked.append(f"{package_id}:{reason}")
    return _dedupe(blocked)


def _telemetry_row(channel_id: str, evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    item = _as_dict(evidence)
    if not item:
        return _missing_row(channel_id, "telemetry", f"v5_p38_telemetry_channel_missing:{channel_id}")
    available = item.get("available") is True
    fallback_ready = item.get("fallback_available") is True and item.get("fallback_ready", True) is not False
    blockers = _telemetry_blockers(channel_id, item, available, fallback_ready)
    return {
        "channel_id": channel_id,
        "provider": str(item.get("provider") or ""),
        "present": True,
        "ready": not blockers,
        "ok": item.get("ok") is True,
        "available": available,
        "fallback_available": item.get("fallback_available") is True,
        "fallback_ready": fallback_ready,
        "degraded": bool(not available and fallback_ready),
        "sample_run_count": _int(item.get("sample_run_count", item.get("run_count", 0))),
        "digest": _digest(item),
        "source": _source(item),
        "default_off": item.get("default_off") is True and _default_off_confirmed(item),
        "request_adapter_off": item.get("request_adapter_off") is True and _request_adapter_off(item),
        "blockers": blockers,
    }


def _telemetry_blockers(
    channel_id: str,
    item: Mapping[str, Any],
    available: bool,
    fallback_ready: bool,
) -> list[str]:
    blocked: list[str] = []
    if item.get("ok") is not True:
        blocked.append(f"v5_p38_telemetry_not_ok:{channel_id}")
    if item.get("observability_evidence_ready") is not True:
        blocked.append(f"v5_p38_telemetry_evidence_not_ready:{channel_id}")
    if item.get("report_only") is not True:
        blocked.append(f"v5_p38_telemetry_report_only_missing:{channel_id}")
    if not available and not fallback_ready:
        blocked.append(f"v5_p38_telemetry_channel_unavailable:{channel_id}")
    if _int(item.get("sample_run_count", item.get("run_count", 0))) < MIN_TELEMETRY_SAMPLE_RUNS:
        blocked.append(f"v5_p38_telemetry_sample_run_count_insufficient:{channel_id}")
    if not _digest(item):
        blocked.append(f"v5_p38_telemetry_digest_missing:{channel_id}")
    if not _source_replayable(item):
        blocked.append(f"v5_p38_telemetry_source_missing:{channel_id}")
    blocked.extend(_row_boundary_blockers(channel_id, item))
    for reason in _string_list(item.get("promotion_blockers")):
        blocked.append(f"{channel_id}:{reason}")
    for reason in _string_list(item.get("blocked_reasons")):
        blocked.append(f"{channel_id}:{reason}")
    return _dedupe(blocked)


def _report_row(report_id: str, evidence: Mapping[str, Any] | None, required_sections: list[str]) -> dict[str, Any]:
    item = _as_dict(evidence)
    if not item:
        return _missing_row(report_id, "report", f"v5_p38_report_missing:{report_id}")
    available_sections = _section_set(item)
    missing_sections = _dedupe(_string_list(item.get("missing_sections")))
    for section in required_sections:
        if section not in available_sections and section not in missing_sections:
            missing_sections.append(section)
    blockers = _report_blockers(report_id, item, required_sections, missing_sections)
    return {
        "report_id": report_id,
        "present": True,
        "ready": not blockers,
        "ok": item.get("ok") is True,
        "schema_version": _int(item.get("schema_version", 0)),
        "required_sections": required_sections,
        "available_sections": sorted(available_sections),
        "missing_sections": missing_sections,
        "digest": _digest(item),
        "source": _source(item),
        "default_off": item.get("default_off") is True and _default_off_confirmed(item),
        "request_adapter_off": item.get("request_adapter_off") is True and _request_adapter_off(item),
        "blockers": blockers,
    }


def _report_blockers(
    report_id: str,
    item: Mapping[str, Any],
    required_sections: list[str],
    missing_sections: list[str],
) -> list[str]:
    blocked: list[str] = []
    if item.get("ok") is not True:
        blocked.append(f"v5_p38_report_not_ok:{report_id}")
    if item.get("report_evidence_ready") is not True:
        blocked.append(f"v5_p38_report_evidence_not_ready:{report_id}")
    if _int(item.get("schema_version", 0)) < 1:
        blocked.append(f"v5_p38_report_schema_missing:{report_id}")
    for digest_name in ("p37_digest", "packaging_digest", "observability_digest"):
        if not str(item.get(digest_name) or "").strip():
            blocked.append(f"v5_p38_report_cross_digest_missing:{report_id}:{digest_name}")
    if not required_sections:
        blocked.append("v5_p38_required_report_sections_empty")
    for section in missing_sections:
        blocked.append(f"v5_p38_report_section_missing:{report_id}:{section}")
    if not _digest(item):
        blocked.append(f"v5_p38_report_digest_missing:{report_id}")
    if not _source_replayable(item):
        blocked.append(f"v5_p38_report_source_missing:{report_id}")
    blocked.extend(_row_boundary_blockers(report_id, item))
    for reason in _string_list(item.get("promotion_blockers")):
        blocked.append(f"{report_id}:{reason}")
    for reason in _string_list(item.get("blocked_reasons")):
        blocked.append(f"{report_id}:{reason}")
    return _dedupe(blocked)


def _blockers(
    *,
    p37_summary: Mapping[str, Any],
    packaging_rows: list[Mapping[str, Any]],
    telemetry_rows: list[Mapping[str, Any]],
    report_rows: list[Mapping[str, Any]],
    unexpected_packages: list[str],
    unexpected_channels: list[str],
    unexpected_reports: list[str],
    package_duplicates: list[str],
    telemetry_duplicates: list[str],
    report_duplicates: list[str],
    malformed_packages: list[str],
    malformed_channels: list[str],
    malformed_reports: list[str],
    required_package_ids: list[str],
    required_channels: list[str],
    required_reports: list[str],
    required_sections: list[str],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p37_summary.get("present", False)):
        blocked.append("v5_p38_p37_coverage_gate_missing")
    elif not bool(p37_summary.get("ready", False)):
        blocked.append("v5_p38_p37_coverage_gate_not_ready")
        blocked.extend(_string_list(p37_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p37_summary.get("unsafe_claims")))
    if not required_package_ids:
        blocked.append("v5_p38_required_packages_empty")
    if not required_channels:
        blocked.append("v5_p38_required_telemetry_channels_empty")
    if not required_reports:
        blocked.append("v5_p38_required_reports_empty")
    if not required_sections:
        blocked.append("v5_p38_required_report_sections_empty")
    for row in packaging_rows + telemetry_rows + report_rows:
        blocked.extend(_string_list(row.get("blockers")))
    for item in unexpected_packages:
        blocked.append(f"v5_p38_unexpected_package_evidence:{item}")
    for item in unexpected_channels:
        blocked.append(f"v5_p38_unexpected_telemetry_evidence:{item}")
    for item in unexpected_reports:
        blocked.append(f"v5_p38_unexpected_report_evidence:{item}")
    for item in package_duplicates:
        blocked.append(f"v5_p38_duplicate_package_evidence:{item}")
    for item in telemetry_duplicates:
        blocked.append(f"v5_p38_duplicate_telemetry_evidence:{item}")
    for item in report_duplicates:
        blocked.append(f"v5_p38_duplicate_report_evidence:{item}")
    for item in malformed_packages:
        blocked.append(f"v5_p38_package_id_missing:{item}")
    for item in malformed_channels:
        blocked.append(f"v5_p38_telemetry_channel_id_missing:{item}")
    for item in malformed_reports:
        blocked.append(f"v5_p38_report_id_missing:{item}")
    for event in failure_events:
        blocked.append(f"v5_p38_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p38_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _indexed_inputs(values: Sequence[Mapping[str, Any]], id_getter: Any) -> dict[str, Any]:
    items: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    malformed: list[str] = []
    for index, value in enumerate(values):
        item = _as_dict(value)
        item_id = id_getter(item)
        if not item_id:
            malformed.append(f"index_{index}")
            continue
        if item_id in items:
            duplicates.append(item_id)
        items[item_id] = item
    return {"items": items, "duplicates": _dedupe(duplicates), "malformed": _dedupe(malformed)}


def _row_boundary_blockers(owner: str, item: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if item.get("default_off") is not True or not _default_off_confirmed(item):
        blocked.append(f"v5_p38_default_off_violation:{owner}")
    if item.get("request_adapter_off") is not True or not _request_adapter_off(item):
        blocked.append(f"v5_p38_request_adapter_violation:{owner}")
    blocked.extend(_unsafe_claims(item, owner))
    return _dedupe(blocked)


def _unsafe_claims(value: Mapping[str, Any], owner: str) -> list[str]:
    blocked: list[str] = []
    true_fields = (
        "training_launch_allowed",
        "auto_launch_allowed",
        "runs_dispatched",
        "default_training_path_enabled",
        "training_path_enabled",
        "default_rollout_allowed",
        "auto_rollout_allowed",
        "ui_exposure_allowed",
        "request_adapter_mapping_allowed",
        "request_fields_emitted",
        "rollout_authorization_allowed",
        "broader_rollout_claim_ready",
        "product_ui_exposure_allowed",
        "ui_entry_enabled",
        "ready_for_ui",
        "default_behavior_changed",
        "rollback_required",
    )
    non_empty_fields = (
        "post_p37_request_fields",
        "post_p38_request_fields",
        "post_packaging_request_fields",
        "post_observability_request_fields",
        "post_report_request_fields",
        "request_adapter",
        "request_adapter_fields",
        "launch_request",
        "training_request",
        "auto_launch_plan",
    )
    for field in true_fields:
        if value.get(field) is True:
            blocked.append(f"v5_p38_unsafe_claim:{owner}:{field}")
    for field in non_empty_fields:
        if bool(value.get(field)):
            blocked.append(f"v5_p38_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _package_id(value: Mapping[str, Any]) -> str:
    return str(value.get("package_id") or value.get("package") or value.get("id") or value.get("name") or "").strip()


def _channel_id(value: Mapping[str, Any]) -> str:
    return str(value.get("channel_id") or value.get("channel") or value.get("id") or value.get("name") or "").strip()


def _report_id(value: Mapping[str, Any]) -> str:
    return str(value.get("report_id") or value.get("report") or value.get("id") or value.get("name") or "").strip()


def _section_set(value: Mapping[str, Any]) -> set[str]:
    sections = set(_string_list(value.get("available_sections")))
    sections.update(_string_list(value.get("sections")))
    required = _string_list(value.get("required_sections"))
    missing = set(_string_list(value.get("missing_sections")))
    for section in required:
        if section not in missing:
            sections.add(section)
    if isinstance(value.get("section_status"), Mapping):
        for section, ready in _as_dict(value.get("section_status")).items():
            if ready:
                sections.add(str(section))
    return {str(item).strip() for item in sections if str(item).strip()}


def _artifact_digest_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "ready_package_count": sum(1 for row in rows if row.get("ready")),
        "required_package_count": len(rows),
        "digest_count": sum(1 for row in rows if row.get("digest")),
        "portable_count": sum(1 for row in rows if row.get("portable")),
        "artifact_count": sum(_int(row.get("artifact_count")) for row in rows),
    }


def _telemetry_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "ready_channel_count": sum(1 for row in rows if row.get("ready")),
        "required_channel_count": len(rows),
        "native_available_count": sum(1 for row in rows if row.get("available")),
        "fallback_ready_count": sum(1 for row in rows if row.get("fallback_ready")),
        "degraded_channel_count": sum(1 for row in rows if row.get("degraded")),
    }


def _report_summary(rows: list[Mapping[str, Any]], required_sections: list[str]) -> dict[str, Any]:
    covered: set[str] = set()
    missing: list[str] = []
    for row in rows:
        covered.update(_string_list(row.get("available_sections")))
        missing.extend(_string_list(row.get("missing_sections")))
    return {
        "ready_report_count": sum(1 for row in rows if row.get("ready")),
        "required_report_count": len(rows),
        "required_sections": required_sections,
        "covered_sections": sorted(covered),
        "missing_sections": _dedupe(missing),
    }


def main(argv: list[str] | None = None) -> None:
    _run_cli(build_v5_packaging_observability_evidence_gate, argv)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_packaging_observability_evidence_gate"]
