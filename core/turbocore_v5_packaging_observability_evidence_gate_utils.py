"""Shared helpers for the V5-P38 packaging / observability evidence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.turbocore_v5_owner_review_evidence_package import load_json


def allowed_next_actions(ready: bool, blockers: list[str]) -> list[str]:
    if ready:
        return ["archive_p38_packaging_observability_evidence"]
    if any("p37" in item for item in blockers):
        return ["repair_p37_route_coverage_gate_before_packaging_observability"]
    if any("telemetry" in item for item in blockers):
        return ["repair_telemetry_channel_evidence"]
    if any("report" in item for item in blockers):
        return ["repair_training_report_evidence"]
    return ["repair_packaging_evidence_or_clear_history"]


def recommended_next_step(ready: bool, blockers: list[str]) -> str:
    if ready:
        return "archive P38 evidence; any rollout authorization still needs a later explicit default-off policy"
    if any("p37" in item for item in blockers):
        return "repair or provide the P37 broader route coverage gate before packaging/observability"
    if any("digest_missing" in item or "source_missing" in item for item in blockers):
        return "collect replayable source and digest evidence for packaging, telemetry, and reports"
    if any("telemetry" in item for item in blockers):
        return "repair missing telemetry channels or provide explicit fallback telemetry evidence"
    if any("report" in item for item in blockers):
        return "repair required training report sections before claiming observability readiness"
    return "hold P38 until packaging evidence and failure/rollback histories are clear"


def required(values: Sequence[str] | None, default: Sequence[str]) -> list[str]:
    source = default if values is None else values
    return dedupe([str(item).strip() for item in source if str(item).strip()])


def event_list(values: Sequence[Any] | None) -> list[str]:
    out: list[str] = []
    for index, value in enumerate(values or []):
        if isinstance(value, Mapping):
            if not history_event_active(value):
                continue
            text = str(
                value.get("reason")
                or value.get("event")
                or value.get("kind")
                or value.get("id")
                or f"event_{index}"
            )
        else:
            text = str(value or "")
        if text:
            out.append(text)
    return dedupe(out)


def history_event_active(value: Mapping[str, Any]) -> bool:
    status = str(value.get("status") or "").strip().lower()
    severity = str(value.get("severity") or "").strip().lower()
    if status in {"closed", "resolved", "cleared", "clear", "ignored", "dismissed"}:
        return False
    return bool(
        value.get("open") is True
        or value.get("active") is True
        or value.get("cooldown") is True
        or value.get("cooldown_active") is True
        or value.get("rollback_required") is True
        or status in {"open", "active", "cooldown", "pending", "blocked"}
        or severity in {"high", "critical", "blocker", "fatal"}
    )


def history_clear(report: Mapping[str, Any], field: str) -> bool:
    summary = as_dict(report.get(field))
    if not summary:
        return True
    return bool(summary.get("clear", False) and not string_list(summary.get("events")))


def history_summary(events: list[str]) -> dict[str, Any]:
    return {"clear": not events, "count": len(events), "events": events}


def missing_row(item_id: str, kind: str, blocker: str) -> dict[str, Any]:
    return {
        f"{kind}_id": item_id,
        "present": False,
        "ready": False,
        "blockers": [blocker],
    }


def digest(value: Mapping[str, Any]) -> str:
    return str(
        value.get("sha256")
        or value.get("ledger_digest")
        or value.get("source_report_digest")
        or value.get("report_digest")
        or value.get("artifact_digest")
        or ""
    ).strip()


def source(value: Mapping[str, Any]) -> str:
    return str(value.get("source") or value.get("path") or "").strip()


def source_replayable(value: Mapping[str, Any]) -> bool:
    return bool(source(value) or as_dict(value.get("artifact")) or list_value(value.get("artifacts")))


def default_off_confirmed(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("default_training_path_enabled") is False
        and value.get("training_path_enabled") is False
        and value.get("default_rollout_allowed") is False
        and value.get("auto_rollout_allowed") is False
    )


def request_adapter_off(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("request_adapter_mapping_allowed") is False
        and value.get("request_fields_emitted") is False
    )


def as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def run_cli(builder: Callable[..., dict[str, Any]], argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = builder(
        p37_coverage_gate=load_json(args.p37_coverage_gate) if args.p37_coverage_gate else None,
        packaging_evidence=[load_json(path) for path in args.packaging_evidence],
        observability_evidence=[load_json(path) for path in args.observability_evidence],
        report_evidence=[load_json(path) for path in args.report_evidence],
        failure_history=load_json(args.failure_history) if args.failure_history else None,
        rollback_history=load_json(args.rollback_history) if args.rollback_history else None,
        required_packages=args.required_package or None,
        required_telemetry_channels=args.required_telemetry_channel or None,
        required_report_ids=args.required_report or None,
        required_report_sections=args.required_report_section or None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P38 packaging/observability evidence gate.")
    parser.add_argument("--p37-coverage-gate", default="", help="P37 route coverage gate JSON.")
    parser.add_argument("--packaging-evidence", action="append", default=[], help="Packaging evidence JSON. Repeatable.")
    parser.add_argument("--observability-evidence", action="append", default=[], help="Telemetry evidence JSON. Repeatable.")
    parser.add_argument("--report-evidence", action="append", default=[], help="Report evidence JSON. Repeatable.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--required-package", action="append", default=[], help="Required package id. Repeatable.")
    parser.add_argument("--required-telemetry-channel", action="append", default=[], help="Required telemetry channel id.")
    parser.add_argument("--required-report", action="append", default=[], help="Required report id. Repeatable.")
    parser.add_argument("--required-report-section", action="append", default=[], help="Required report section. Repeatable.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser
