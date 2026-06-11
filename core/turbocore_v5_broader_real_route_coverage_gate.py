"""Broader real-route coverage gate for TurboCore V5-P37.

P37 turns the P36 "broader route coverage missing" classification blocker into
a replayable evidence gate. It only ingests route evidence that already exists;
it never launches training, emits request-adapter fields, exposes UI, or enables
default/auto rollout.
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


P36_READY_DECISION = "post_archive_rollout_scope_classified_default_off"
P37_READY_DECISION = "broader_real_route_coverage_ready_default_off"
P37_BLOCKED_DECISION = "broader_real_route_coverage_blocked_default_off"
DEFAULT_REQUIRED_ROUTES = (
    "native_update_dispatch",
    "lora_fused_dispatch",
    "native_data_prefetch",
)
MIN_ROUTE_STEPS = 1
MIN_ROUTE_SAMPLE_COUNT = 1
PROMOTION_QUALITIES = {
    "real_route_manual_replay",
    "real_route_manual_matrix",
    "promotion_benchmark",
}


def build_v5_broader_real_route_coverage_gate(
    *,
    p36_scope_classification: Mapping[str, Any] | None = None,
    route_evidence: Sequence[Mapping[str, Any]] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
    required_routes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Gate broader route coverage without enabling broader rollout."""

    p36 = _as_dict(p36_scope_classification)
    route_inputs = _route_inputs(route_evidence or [])
    routes = route_inputs["routes"]
    required = _required_routes(required_routes)
    p36_summary = _p36_summary(p36)
    route_rows = [_route_row(name, routes.get(name)) for name in required]
    unexpected = sorted(name for name in routes if name not in set(required))
    missing = [str(row["route_id"]) for row in route_rows if not row.get("present")]
    covered = [str(row["route_id"]) for row in route_rows if row.get("ready")]
    failure_events = _event_list(failure_history)
    rollback_events = _event_list(rollback_history)
    blockers = _blockers(
        p36_summary,
        route_rows,
        unexpected,
        route_inputs["duplicate_routes"],
        route_inputs["malformed_routes"],
        failure_events,
        rollback_events,
    )
    ready = not blockers
    decision = P37_READY_DECISION if ready else P37_BLOCKED_DECISION
    return {
        "schema_version": 1,
        "package": "turbocore_v5_broader_real_route_coverage_gate_v0",
        "gate": "v5_broader_real_route_coverage",
        "ok": ready,
        "broader_real_route_coverage_ready": ready,
        "coverage_gate_ready": ready,
        "route_coverage_gate_ready": ready,
        "route_coverage_decision": decision,
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
        "post_p37_request_fields": {},
        "p36_scope_classification_summary": p36_summary,
        "required_routes": required,
        "required_route_ids": required,
        "covered_route_ids": covered,
        "missing_route_ids": missing,
        "coverage_matrix": route_rows,
        "route_evidence": route_rows,
        "unexpected_routes": unexpected,
        "duplicate_route_ids": route_inputs["duplicate_routes"],
        "malformed_route_evidence": route_inputs["malformed_routes"],
        "evidence_quality_summary": _quality_summary(route_rows),
        "failure_history_summary": _history_summary(failure_events),
        "rollback_history_summary": _history_summary(rollback_events),
        "blocked_reasons": blockers,
        "promotion_blockers": blockers,
        "recommended_next_step": _recommended_next_step(ready, blockers),
        "notes": [
            "P37 ingests already-produced route evidence only; it does not run broader tests.",
            "Smoke or short evidence is reported but cannot satisfy broader route coverage.",
            "Even ready coverage does not authorize UI exposure, request mapping, or rollout.",
        ],
    }


def _p36_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    scope = _as_dict(report.get("scope_classification"))
    p35_summary = _as_dict(report.get("p35_archive_replay_summary"))
    return {
        "present": bool(report),
        "ok": bool(report.get("ok", False)),
        "scope_classification_ready": bool(report.get("scope_classification_ready", False)),
        "decision": str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or ""),
        "default_off": _default_off_confirmed(report),
        "request_adapter_off": _request_adapter_off(report),
        "default_behavior_changed": bool(report.get("default_behavior_changed", True)),
        "manual_review_required": bool(report.get("manual_review_required", False)),
        "training_launch_allowed": bool(report.get("training_launch_allowed", True)),
        "auto_launch_allowed": bool(report.get("auto_launch_allowed", True)),
        "runs_dispatched": bool(report.get("runs_dispatched", True)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "post_fields_empty": not bool(_as_dict(report.get("post_scope_request_fields"))),
        "p35_archive_replay_ready": bool(p35_summary.get("ready", False)),
        "archive_replay_evidence_state": _scope_state(scope, "archive_replay_evidence"),
        "broader_real_route_coverage_state": _scope_state(scope, "broader_real_route_coverage"),
        "packaging_observability_state": _scope_state(scope, "packaging_observability"),
        "controlled_rollout_policy_state": _scope_state(scope, "controlled_rollout_policy"),
        "controlled_rollout_policy_recorded": bool(report.get("controlled_rollout_policy_recorded", False)),
        "broader_route_evidence_claim_ready": bool(report.get("broader_route_evidence_claim_ready", False)),
        "broader_rollout_claim_ready": bool(report.get("broader_rollout_claim_ready", True)),
        "rollout_authorization_allowed": bool(report.get("rollout_authorization_allowed", True)),
        "unsafe_claims": _unsafe_claims(report, "p36"),
        "blocked_reasons": _string_list(report.get("blocked_reasons")),
        "ready": _p36_ready(report),
    }


def _p36_ready(report: Mapping[str, Any]) -> bool:
    scope = _as_dict(report.get("scope_classification"))
    p35_summary = _as_dict(report.get("p35_archive_replay_summary"))
    return bool(
        report
        and report.get("ok") is True
        and report.get("scope_classification_ready") is True
        and str(report.get("decision") or report.get("gate_decision") or report.get("package_decision") or "")
        == P36_READY_DECISION
        and report.get("default_behavior_changed") is False
        and report.get("manual_review_required") is True
        and report.get("training_launch_allowed") is False
        and report.get("auto_launch_allowed") is False
        and report.get("runs_dispatched") is False
        and not bool(report.get("ui_exposure_allowed", False))
        and _default_off_confirmed(report)
        and _request_adapter_off(report)
        and not _as_dict(report.get("post_scope_request_fields"))
        and not _string_list(report.get("blocked_reasons"))
        and not _string_list(report.get("classification_blockers"))
        and not _unsafe_claims(report, "p36")
        and p35_summary.get("ready") is True
        and _scope_state(scope, "archive_replay_evidence") == "ready"
        and _scope_state(scope, "broader_real_route_coverage") == "ready"
        and _scope_state(scope, "packaging_observability") == "ready"
        and _scope_state(scope, "controlled_rollout_policy") == "recorded"
        and report.get("controlled_rollout_policy_recorded") is True
        and report.get("broader_route_evidence_claim_ready") is True
        and report.get("broader_rollout_claim_ready") is False
        and report.get("rollout_authorization_allowed") is False
    )


def _route_row(route_id: str, evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    item = _as_dict(evidence)
    if not item:
        return {
            "route_id": route_id,
            "present": False,
            "ready": False,
            "evidence_quality": "missing",
            "steps": 0,
            "sample_count": 0,
            "blockers": [f"v5_p37_route_missing:{route_id}"],
        }
    quality = _quality(item)
    blockers = _route_blockers(route_id, item, quality)
    return {
        "route_id": route_id,
        "present": True,
        "ready": not blockers,
        "evidence_quality": quality,
        "source_report_digest": _digest(item),
        "source_replayable": _source_replayable(item),
        "manual_only": item.get("manual_only") is True,
        "real_route": item.get("real_route") is True,
        "default_off": item.get("default_off") is True and _default_off_confirmed(item),
        "request_adapter_off": item.get("request_adapter_off") is True and _request_adapter_off(item),
        "steps": int(item.get("representative_steps", item.get("steps", 0)) or 0),
        "sample_count": int(item.get("sample_count", item.get("run_count", 0)) or 0),
        "source": str(item.get("source") or item.get("path") or ""),
        "decision": str(item.get("decision") or item.get("gate_decision") or ""),
        "blockers": blockers,
    }


def _route_blockers(route_id: str, evidence: Mapping[str, Any], quality: str) -> list[str]:
    blocked: list[str] = []
    if evidence.get("ok") is not True:
        blocked.append(f"v5_p37_route_not_ok:{route_id}")
    if evidence.get("coverage_ready") is not True or evidence.get("promotion_ready") is not True:
        blocked.append(f"v5_p37_route_not_ready:{route_id}")
    if quality not in PROMOTION_QUALITIES:
        blocked.append(f"v5_p37_route_quality_insufficient:{route_id}:{quality}")
    if not _digest(evidence):
        blocked.append(f"v5_p37_route_digest_missing:{route_id}")
    if not _source_replayable(evidence):
        blocked.append(f"v5_p37_route_source_missing:{route_id}")
    if evidence.get("manual_only") is not True:
        blocked.append(f"v5_p37_route_manual_only_missing:{route_id}")
    if evidence.get("real_route") is not True:
        blocked.append(f"v5_p37_route_real_route_missing:{route_id}")
    if int(evidence.get("representative_steps", evidence.get("steps", 0)) or 0) < MIN_ROUTE_STEPS:
        blocked.append(f"v5_p37_route_steps_insufficient:{route_id}")
    if int(evidence.get("sample_count", evidence.get("run_count", 0)) or 0) < MIN_ROUTE_SAMPLE_COUNT:
        blocked.append(f"v5_p37_route_sample_count_insufficient:{route_id}")
    if evidence.get("default_off") is not True:
        blocked.append(f"v5_p37_route_default_off_ack_missing:{route_id}")
    if evidence.get("request_adapter_off") is not True:
        blocked.append(f"v5_p37_route_request_adapter_ack_missing:{route_id}")
    if evidence.get("training_launch_allowed") is True or evidence.get("auto_launch_allowed") is True:
        blocked.append(f"v5_p37_route_launch_requested:{route_id}")
    if evidence.get("runs_dispatched") is True:
        blocked.append(f"v5_p37_route_runs_dispatched:{route_id}")
    if evidence.get("ui_exposure_allowed") is True:
        blocked.append(f"v5_p37_route_ui_exposure_requested:{route_id}")
    if not _default_off_confirmed(evidence):
        blocked.append(f"v5_p37_route_default_off_violation:{route_id}")
    if not _request_adapter_off(evidence):
        blocked.append(f"v5_p37_route_request_adapter_violation:{route_id}")
    blocked.extend(_unsafe_claims(evidence, route_id))
    for reason in _string_list(evidence.get("blocked_reasons")):
        blocked.append(f"{route_id}:{reason}")
    return _dedupe(blocked)


def _quality(evidence: Mapping[str, Any]) -> str:
    raw = evidence.get("evidence_quality", evidence.get("quality", evidence.get("validation_level", "")))
    return str(raw or "unknown").strip().lower().replace("-", "_").replace(" ", "_")


def _digest(evidence: Mapping[str, Any]) -> str:
    return str(
        evidence.get("source_report_digest")
        or evidence.get("report_digest")
        or evidence.get("artifact_digest")
        or evidence.get("sha256")
        or evidence.get("ledger_digest")
        or ""
    ).strip()


def _source_replayable(evidence: Mapping[str, Any]) -> bool:
    return bool(
        str(evidence.get("source") or evidence.get("path") or "").strip()
        or _as_dict(evidence.get("artifact"))
        or evidence.get("artifact_digests")
    )


def _scope_state(scope: Mapping[str, Any], name: str) -> str:
    item = _as_dict(scope.get(name))
    return str(item.get("state") or "").strip()


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
        "default_behavior_changed",
    )
    non_empty_fields = (
        "post_scope_request_fields",
        "post_p37_request_fields",
        "post_route_request_fields",
        "request_adapter",
        "request_adapter_fields",
        "launch_request",
        "training_request",
    )
    for field in true_fields:
        if value.get(field) is True:
            blocked.append(f"v5_p37_unsafe_claim:{owner}:{field}")
    for field in non_empty_fields:
        if bool(value.get(field)):
            blocked.append(f"v5_p37_unsafe_claim:{owner}:{field}")
    return _dedupe(blocked)


def _blockers(
    p36_summary: Mapping[str, Any],
    route_rows: list[Mapping[str, Any]],
    unexpected_routes: list[str],
    duplicate_routes: list[str],
    malformed_routes: list[str],
    failure_events: list[str],
    rollback_events: list[str],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p36_summary.get("present", False)):
        blocked.append("v5_p37_p36_scope_classification_missing")
    elif not bool(p36_summary.get("ready", False)):
        blocked.append("v5_p37_p36_scope_classification_not_ready")
        blocked.extend(_string_list(p36_summary.get("blocked_reasons")))
        blocked.extend(_string_list(p36_summary.get("unsafe_claims")))
    if not route_rows:
        blocked.append("v5_p37_required_routes_empty")
    for row in route_rows:
        blocked.extend(_string_list(row.get("blockers")))
    for route_id in unexpected_routes:
        blocked.append(f"v5_p37_unexpected_route_evidence:{route_id}")
    for route_id in duplicate_routes:
        blocked.append(f"v5_p37_duplicate_route_evidence:{route_id}")
    for route_id in malformed_routes:
        blocked.append(f"v5_p37_route_id_missing:{route_id}")
    for event in failure_events:
        blocked.append(f"v5_p37_failure_history_not_clear:{event}")
    for event in rollback_events:
        blocked.append(f"v5_p37_rollback_history_not_clear:{event}")
    return _dedupe(blocked)


def _route_inputs(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mapped: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    malformed: list[str] = []
    for index, value in enumerate(values):
        item = _as_dict(value)
        route_id = _route_id(item)
        if not route_id:
            malformed.append(f"index_{index}")
            continue
        if route_id in mapped:
            duplicates.append(route_id)
        mapped[route_id] = item
    return {
        "routes": mapped,
        "duplicate_routes": _dedupe(duplicates),
        "malformed_routes": _dedupe(malformed),
    }


def _route_id(value: Mapping[str, Any]) -> str:
    return str(value.get("route_id") or value.get("route") or value.get("name") or "").strip()


def _required_routes(values: Sequence[str] | None) -> list[str]:
    source = DEFAULT_REQUIRED_ROUTES if values is None else values
    routes = [str(item).strip() for item in source if str(item).strip()]
    return _dedupe(routes)


def _event_list(values: Sequence[Any] | None) -> list[str]:
    out: list[str] = []
    for index, value in enumerate(values or []):
        if isinstance(value, Mapping):
            if not _history_event_active(value):
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
    return _dedupe(out)


def _history_event_active(value: Mapping[str, Any]) -> bool:
    status = str(value.get("status") or "").strip().lower()
    severity = str(value.get("severity") or "").strip().lower()
    if status in {"closed", "resolved", "cleared", "clear", "ignored", "dismissed"}:
        return False
    return bool(
        value.get("open") is True
        or value.get("active") is True
        or value.get("cooldown") is True
        or value.get("cooldown_active") is True
        or status in {"open", "active", "cooldown", "pending", "blocked"}
        or severity in {"high", "critical", "blocker", "fatal"}
    )


def _quality_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    qualities: dict[str, int] = {}
    for row in rows:
        quality = str(row.get("evidence_quality") or "unknown")
        qualities[quality] = qualities.get(quality, 0) + 1
    return {
        "qualities": qualities,
        "ready_route_count": sum(1 for row in rows if row.get("ready")),
        "required_route_count": len(rows),
        "digest_count": sum(1 for row in rows if row.get("source_report_digest")),
        "manual_route_count": sum(1 for row in rows if row.get("manual_only")),
        "real_route_count": sum(1 for row in rows if row.get("real_route")),
    }


def _history_summary(events: list[str]) -> dict[str, Any]:
    return {"clear": not events, "count": len(events), "events": events}


def _recommended_next_step(ready: bool, blockers: list[str]) -> str:
    if ready:
        return "feed P37 coverage into P36 follow-up; rollout authorization still needs later policy"
    if any("p36" in item for item in blockers):
        return "repair P36 scope classification before broader route coverage gate"
    if any("quality_insufficient" in item for item in blockers):
        return "collect representative/manual route evidence instead of smoke-only coverage"
    if any("route_missing" in item for item in blockers):
        return "collect missing real-route evidence for the required TurboCore routes"
    return "hold broader coverage until route evidence and histories are clear"


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


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 P37 broader real-route coverage gate.")
    parser.add_argument("--p36-scope-classification", default="", help="P36 scope classification JSON.")
    parser.add_argument("--route-evidence", action="append", default=[], help="Route evidence JSON. Repeatable.")
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON list.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON list.")
    parser.add_argument("--required-route", action="append", default=[], help="Required route id. Repeatable.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_broader_real_route_coverage_gate(
        p36_scope_classification=load_json(args.p36_scope_classification) if args.p36_scope_classification else None,
        route_evidence=[load_json(path) for path in args.route_evidence],
        failure_history=load_json(args.failure_history) if args.failure_history else None,
        rollback_history=load_json(args.rollback_history) if args.rollback_history else None,
        required_routes=args.required_route or None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_broader_real_route_coverage_gate"]
