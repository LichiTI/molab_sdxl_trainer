"""TurboCore V5-P26 longer replicate and failure-history review gate.

This module is contract-only. It joins the P25 owner rollout review decision,
longer replicate evidence, and failure/rollback history while keeping all
default rollout and request-adapter behavior disabled.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

CORE_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = CORE_ROOT.parent
for import_root in (str(CORE_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

try:
    from core.turbocore_v5_failure_history_review import summarize_v5_failure_history
except ModuleNotFoundError:  # pragma: no cover - direct script execution from backend/core
    from turbocore_v5_failure_history_review import summarize_v5_failure_history

MIN_LONGER_REPLICATE_RUNS = 5
MIN_END_TO_END_SPEEDUP = 1.03
MAX_SPEEDUP_SPREAD_RATIO = 0.30

APPROVED_P25_DECISION = "owner_rollout_review_recorded_default_off"


def build_v5_longer_replicate_failure_history_gate(
    *,
    owner_rollout_review_decision: Mapping[str, Any] | None = None,
    longer_replicate_evidence: Any | None = None,
    longer_replicate_evidence_paths: Iterable[str | Path] | None = None,
    failure_history: Any | None = None,
    rollback_history: Any | None = None,
    min_longer_replicate_runs: int = MIN_LONGER_REPLICATE_RUNS,
    min_end_to_end_speedup: float = MIN_END_TO_END_SPEEDUP,
    max_speedup_spread_ratio: float = MAX_SPEEDUP_SPREAD_RATIO,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Build the P26 default-off review gate without running training."""

    min_runs = max(int(min_longer_replicate_runs), 1)
    min_speedup = float(min_end_to_end_speedup)
    max_spread = float(max_speedup_spread_ratio)
    p25_summary = _p25_summary(_as_dict(owner_rollout_review_decision))
    replicate_summary = _replicate_summary(
        _load_replicate_inputs(longer_replicate_evidence, longer_replicate_evidence_paths),
        min_runs=min_runs,
        min_end_to_end_speedup=min_speedup,
        max_speedup_spread_ratio=max_spread,
    )
    failure_summary = summarize_v5_failure_history(failure_history, kind="failure", now_ts=now_ts)
    rollback_summary = summarize_v5_failure_history(rollback_history, kind="rollback", now_ts=now_ts)
    blocked = _blockers(p25_summary, replicate_summary, failure_summary, rollback_summary)
    ready = not blocked
    decision = "longer_replicate_failure_history_review_ready" if ready else "hold_default_off"
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_longer_replicate_failure_history_gate_v0",
        "gate": "v5_longer_replicate_failure_history_gate",
        "ok": ready,
        "longer_replicate_failure_history_gate_ready": ready,
        "manual_next_stage_review_allowed": ready,
        "decision": decision,
        "gate_decision": decision,
        "rollout_review_decision": decision,
        "manual_review_required": True,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_gate_request_fields": {},
        "min_longer_replicate_runs": min_runs,
        "min_end_to_end_speedup": min_speedup,
        "max_speedup_spread_ratio": max_spread,
        "p25_decision_summary": p25_summary,
        "longer_replicate_summary": replicate_summary,
        "failure_history_summary": failure_summary,
        "rollback_history_summary": rollback_summary,
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(blocked),
        "notes": [
            "This gate only reviews existing JSON evidence; it does not run training.",
            "P26 remains default-off even when longer replicate evidence is ready.",
            "No request-adapter mapping or request fields are emitted by this gate.",
        ],
    }


def _p25_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    rollout_decision = str(decision.get("rollout_decision") or "")
    return {
        "present": bool(decision),
        "source_path": str(decision.get("_source_path") or decision.get("source_path") or ""),
        "ok": bool(decision.get("ok", False)),
        "decision_record_ready": bool(decision.get("decision_record_ready", False)),
        "owner_rollout_review_recorded": bool(decision.get("owner_rollout_review_recorded", False)),
        "owner_rollout_review_signed": bool(decision.get("owner_rollout_review_signed", False)),
        "rollout_decision": rollout_decision,
        "approved_for_next_stage": bool(decision.get("approved_for_next_stage", False)),
        "rejected_for_default_off_hold": bool(decision.get("rejected_for_default_off_hold", False)),
        "rollback_required": bool(decision.get("rollback_required", False)),
        "approved_default_off_decision": rollout_decision == APPROVED_P25_DECISION,
        "default_off_confirmed": _default_off_confirmed(decision),
        "request_adapter_off_confirmed": (
            decision.get("request_adapter_mapping_allowed") is False
            and decision.get("request_fields_emitted") is False
        ),
        "blocked_reasons": _string_list(decision.get("blocked_reasons")),
    }


def _replicate_summary(
    loaded: list[dict[str, Any]],
    *,
    min_runs: int,
    min_end_to_end_speedup: float,
    max_speedup_spread_ratio: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    declared_counts: list[int] = []
    blockers: list[str] = []
    for item in loaded:
        source = str(item["source"])
        payload = item["payload"]
        if item["load_error"]:
            blockers.append(str(item["load_error"]))
        records.extend(_extract_speedup_records(payload, source))
        declared = _declared_run_count(payload)
        if declared:
            declared_counts.append(declared)
        payload_blockers = _replicate_payload_blockers(payload)
        if payload_blockers:
            blockers.append("v5_p26_replicate_evidence_blocked")
            blockers.extend(payload_blockers)

    speedups = [float(item["end_to_end_speedup"]) for item in records]
    run_count = max([len(speedups), *declared_counts], default=0)
    aggregate = _speedup_aggregate(speedups)
    if not loaded:
        blockers.append("v5_p26_longer_replicate_evidence_missing")
    if run_count < min_runs:
        blockers.append("v5_p26_replicate_runs_too_few")
    if len(speedups) < min_runs:
        blockers.append("v5_p26_replicate_speedup_samples_too_few")
    if aggregate["min_speedup"] is not None and aggregate["min_speedup"] < min_end_to_end_speedup:
        blockers.append("v5_p26_min_speedup_below_threshold")
    if (
        aggregate["speedup_spread_ratio"] is not None
        and aggregate["speedup_spread_ratio"] > max_speedup_spread_ratio
    ):
        blockers.append("v5_p26_speedup_spread_too_high")
    return {
        "present": bool(loaded),
        "source_count": len(loaded),
        "run_count": run_count,
        "speedup_sample_count": len(speedups),
        "speedup_samples": [round(item, 4) for item in speedups],
        "sources": [str(item["source"]) for item in loaded],
        "records": records[-10:],
        **aggregate,
        "blocked_reasons": _dedupe(blockers),
        "ready": not _dedupe(blockers),
    }


def _blockers(
    p25: Mapping[str, Any],
    replicate: Mapping[str, Any],
    failure: Mapping[str, Any],
    rollback: Mapping[str, Any],
) -> list[str]:
    blocked: list[str] = []
    if not bool(p25.get("present", False)):
        blocked.append("v5_p26_p25_owner_rollout_review_decision_missing")
    if not bool(p25.get("ok", False)):
        blocked.append("v5_p26_p25_decision_not_ready")
        blocked.extend(_string_list(p25.get("blocked_reasons")))
    if not bool(p25.get("decision_record_ready", False)):
        blocked.append("v5_p26_p25_decision_record_not_ready")
    if not bool(p25.get("owner_rollout_review_signed", False)):
        blocked.append("v5_p26_p25_owner_rollout_review_not_signed")
    if not bool(p25.get("owner_rollout_review_recorded", False)):
        blocked.append("v5_p26_p25_owner_rollout_review_not_recorded")
    if not bool(p25.get("approved_default_off_decision", False)):
        blocked.append("v5_p26_p25_decision_not_approved_default_off")
    if not bool(p25.get("approved_for_next_stage", False)):
        blocked.append("v5_p26_p25_not_approved_for_next_stage")
    if bool(p25.get("rejected_for_default_off_hold", False)):
        blocked.append("v5_p26_p25_rejected_for_default_off_hold")
    if bool(p25.get("rollback_required", False)):
        blocked.append("v5_p26_p25_rollback_required")
    if not bool(p25.get("default_off_confirmed", False)):
        blocked.append("v5_p26_p25_default_off_violation")
    if not bool(p25.get("request_adapter_off_confirmed", False)):
        blocked.append("v5_p26_p25_request_adapter_violation")
    blocked.extend(_string_list(replicate.get("blocked_reasons")))
    blocked.extend(_string_list(failure.get("blocked_reasons")))
    blocked.extend(_string_list(rollback.get("blocked_reasons")))
    return _dedupe(blocked)


def _recommended_next_step(blocked: list[str]) -> str:
    blockers = set(blocked)
    if not blocked:
        return "continue manual review with longer replicate evidence; default and auto remain off"
    if any(item.startswith("v5_p26_p25_") for item in blockers):
        return "wait for an approved P25 signed owner rollout review decision"
    if "v5_p26_replicate_runs_too_few" in blockers:
        return "collect more longer replicate samples before P26 review"
    if "v5_p26_min_speedup_below_threshold" in blockers:
        return "hold P26 and investigate the slower replicate sample"
    if "v5_p26_speedup_spread_too_high" in blockers:
        return "rerun longer replicates under quieter conditions before review"
    if any("history_" in item for item in blockers):
        return "resolve failure or rollback history before P26 review"
    return "hold P26 until all default-off review evidence passes"


def _load_replicate_inputs(value: Any, paths: Iterable[str | Path] | None) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for index, payload in enumerate(_payload_items(value)):
        loaded.append({"source": f"provided_payload:{index}", "payload": payload, "load_error": ""})
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            loaded.append(
                {
                    "source": str(path),
                    "payload": {},
                    "load_error": "v5_p26_longer_replicate_evidence_path_missing",
                }
            )
            continue
        try:
            loaded.append(
                {
                    "source": str(path),
                    "payload": json.loads(path.read_text(encoding="utf-8")),
                    "load_error": "",
                }
            )
        except Exception as exc:
            loaded.append(
                {
                    "source": str(path),
                    "payload": {},
                    "load_error": f"v5_p26_longer_replicate_evidence_path_error:{type(exc).__name__}",
                }
            )
    return loaded


def _payload_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return list(value)
    return [value]


def _extract_speedup_records(value: Any, source: str) -> list[dict[str, Any]]:
    if isinstance(value, (int, float, str)):
        speedup = _float_or_none(value)
        return [_speedup_record(source, "speedup", speedup)] if speedup is not None else []
    if isinstance(value, list):
        records: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            records.extend(_extract_speedup_records(item, f"{source}#{index}"))
        return records
    if not isinstance(value, Mapping):
        return []

    samples = _sample_list(value.get("speedup_samples")) or _sample_list(_as_dict(value.get("aggregate")).get("speedup_samples"))
    if samples:
        return [_speedup_record(source, f"speedup_samples[{index}]", sample) for index, sample in enumerate(samples)]

    direct = _first_speedup(value)
    if direct is not None:
        return [_speedup_record(source, "direct", direct)]

    training = _as_dict(_as_dict(_as_dict(value.get("native_update_performance_report")).get("performance_gate")).get("evidence")).get(
        "training_matrix"
    )
    speedup = _float_or_none(_as_dict(training).get("end_to_end_speedup"))
    if speedup is not None:
        return [_speedup_record(source, "native_update_performance_report", speedup)]

    records: list[dict[str, Any]] = []
    primary_run_keys = (
        "runs",
        "samples",
        "replicates",
        "replicate_runs",
        "longer_replicate_runs",
        "longer_replicates",
    )
    for key in (
        *primary_run_keys,
        "evidence",
        "items",
        "records",
    ):
        nested = value.get(key)
        if isinstance(nested, (list, tuple)):
            for index, item in enumerate(nested):
                records.extend(_extract_speedup_records(item, f"{source}.{key}[{index}]"))
            if records and key in primary_run_keys:
                return records
    return records


def _speedup_record(source: str, field: str, speedup: float) -> dict[str, Any]:
    return {
        "source": source,
        "field": field,
        "end_to_end_speedup": round(float(speedup), 4),
    }


def _first_speedup(value: Mapping[str, Any]) -> float | None:
    for key in (
        "end_to_end_speedup",
        "representative_end_to_end_speedup",
        "native_end_to_end_speedup",
        "speedup",
        "speedup_vs_baseline",
    ):
        speedup = _float_or_none(value.get(key))
        if speedup is not None:
            return speedup
    for key in ("performance", "performance_summary", "run_result_summary"):
        nested = _as_dict(value.get(key))
        if not nested:
            continue
        speedup = _first_speedup(nested)
        if speedup is not None:
            return speedup
    return None


def _sample_list(value: Any) -> list[float]:
    samples: list[float] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            speedup = _float_or_none(item)
            if speedup is not None:
                samples.append(speedup)
    return samples


def _declared_run_count(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    for key in ("run_count", "ready_run_count", "replicate_run_count", "sample_count"):
        try:
            count = int(value.get(key, 0) or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            return count
    return 0


def _replicate_payload_blockers(value: Any) -> list[str]:
    blocked: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, Mapping):
            return
        blocked.extend(_string_list(item.get("blocked_reasons")))
        blocked.extend(_string_list(item.get("promotion_blockers")))
        if item.get("ok") is False:
            blocked.append("replicate_payload_ok_false")
        if item.get("success") is False:
            blocked.append("replicate_payload_success_false")
        if item.get("stability_gate_ready") is False:
            blocked.append("replicate_payload_stability_gate_not_ready")
        if item.get("longer_replicate_evidence_ready") is False:
            blocked.append("longer_replicate_evidence_not_ready")
        rollback_events = _string_list(item.get("rollback_events"))
        failure_events = _string_list(item.get("failure_events"))
        if rollback_events:
            blocked.append("replicate_payload_rollback_events_present")
            blocked.extend(f"rollback:{event}" for event in rollback_events)
        if failure_events:
            blocked.append("replicate_payload_failure_events_present")
            blocked.extend(f"failure:{event}" for event in failure_events)
        for key in (
            "runs",
            "samples",
            "replicates",
            "replicate_runs",
            "longer_replicate_runs",
            "longer_replicates",
            "items",
            "records",
        ):
            nested = item.get(key)
            if isinstance(nested, list):
                visit(nested)

    visit(value)
    return _dedupe(blocked)


def _speedup_aggregate(speedups: list[float]) -> dict[str, Any]:
    if not speedups:
        return {
            "min_speedup": None,
            "mean_speedup": None,
            "median_speedup": None,
            "speedup_spread_ratio": None,
        }
    median = statistics.median(speedups)
    spread = (max(speedups) - min(speedups)) / median if median > 0.0 else None
    return {
        "min_speedup": round(min(speedups), 4),
        "mean_speedup": round(statistics.fmean(speedups), 4),
        "median_speedup": round(median, 4),
        "speedup_spread_ratio": round(float(spread), 4) if spread is not None else None,
    }


def _default_off_confirmed(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("default_training_path_enabled") is False
        and value.get("training_path_enabled") is False
        and value.get("default_rollout_allowed") is False
        and value.get("auto_rollout_allowed") is False
    )


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0.0 else None


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


def _load_json_any(path: str | Path) -> Any:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload.setdefault("_source_path", str(source))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TurboCore V5-P26 default-off review gate.")
    parser.add_argument("--owner-rollout-review-decision", default="", help="P25 review decision JSON.")
    parser.add_argument(
        "--longer-replicate-evidence",
        action="append",
        default=[],
        help="Longer replicate evidence JSON. Repeat for multiple files.",
    )
    parser.add_argument("--failure-history", default="", help="Optional failure history JSON.")
    parser.add_argument("--rollback-history", default="", help="Optional rollback history JSON.")
    parser.add_argument("--min-longer-replicate-runs", type=int, default=MIN_LONGER_REPLICATE_RUNS)
    parser.add_argument("--min-end-to-end-speedup", type=float, default=MIN_END_TO_END_SPEEDUP)
    parser.add_argument("--max-speedup-spread-ratio", type=float, default=MAX_SPEEDUP_SPREAD_RATIO)
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_longer_replicate_failure_history_gate(
        owner_rollout_review_decision=(
            _load_json_any(args.owner_rollout_review_decision)
            if args.owner_rollout_review_decision
            else None
        ),
        longer_replicate_evidence_paths=args.longer_replicate_evidence,
        failure_history=_load_json_any(args.failure_history) if args.failure_history else None,
        rollback_history=_load_json_any(args.rollback_history) if args.rollback_history else None,
        min_longer_replicate_runs=args.min_longer_replicate_runs,
        min_end_to_end_speedup=args.min_end_to_end_speedup,
        max_speedup_spread_ratio=args.max_speedup_spread_ratio,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_longer_replicate_failure_history_gate"]
