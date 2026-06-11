"""Collect P26-compatible longer replicate evidence for TurboCore V5-P28.

This module is contract-only. It reads existing matrix/run summaries and
normalizes them into a longer-replicate evidence bundle; it never launches
training and never enables default rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


MIN_LONGER_REPLICATE_RUNS = 5
MIN_REPRESENTATIVE_STEPS = 100
MIN_END_TO_END_SPEEDUP = 1.03
MAX_SPEEDUP_SPREAD_RATIO = 0.30


def build_v5_longer_replicate_evidence_bundle(
    *,
    run_payloads: Iterable[Mapping[str, Any]] | None = None,
    run_summary_paths: Iterable[str | Path] | None = None,
    min_runs: int = MIN_LONGER_REPLICATE_RUNS,
    min_representative_steps: int = MIN_REPRESENTATIVE_STEPS,
    min_end_to_end_speedup: float = MIN_END_TO_END_SPEEDUP,
    max_speedup_spread_ratio: float = MAX_SPEEDUP_SPREAD_RATIO,
) -> dict[str, Any]:
    loaded = _load_inputs(run_payloads, run_summary_paths)
    samples = [
        _sample_summary(
            payload=item["payload"],
            source=str(item["source"]),
            load_error=str(item["load_error"]),
            min_steps=max(int(min_representative_steps), 1),
            min_speedup=float(min_end_to_end_speedup),
        )
        for item in loaded
    ]
    aggregate = _aggregate(
        samples=samples,
        min_runs=max(int(min_runs), 1),
        min_speedup=float(min_end_to_end_speedup),
        max_spread=float(max_speedup_spread_ratio),
    )
    blocked = _dedupe(
        [reason for sample in samples for reason in sample["blocked_reasons"]]
        + list(aggregate["blocked_reasons"])
    )
    ready = bool(samples) and not blocked
    return {
        "schema_version": 1,
        "evidence": "turbocore_v5_longer_replicate_evidence_bundle_v0",
        "scorecard": "turbocore_v5_longer_replicate_evidence_collector_v0",
        "gate": "v5_longer_replicate_evidence_collector",
        "ok": ready,
        "longer_replicate_evidence_ready": ready,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "post_gate_request_fields": {},
        "min_longer_replicate_runs": max(int(min_runs), 1),
        "min_representative_steps": max(int(min_representative_steps), 1),
        "min_end_to_end_speedup": float(min_end_to_end_speedup),
        "max_speedup_spread_ratio": float(max_speedup_spread_ratio),
        "run_count": len(samples),
        "ready_run_count": len([sample for sample in samples if not sample["blocked_reasons"]]),
        "speedup_samples": aggregate["speedup_samples"],
        "aggregate": aggregate,
        "samples": samples,
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(blocked),
        "notes": [
            "This collector only normalizes existing run evidence; it does not run training.",
            "The output is P26-compatible longer replicate evidence.",
            "Default rollout and request-adapter fields remain disabled.",
        ],
    }


def _load_inputs(
    run_payloads: Iterable[Mapping[str, Any]] | None,
    run_summary_paths: Iterable[str | Path] | None,
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for index, payload in enumerate(run_payloads or []):
        loaded.append({"source": f"provided_payload:{index}", "payload": dict(payload), "load_error": ""})
    for raw_path in run_summary_paths or []:
        path = Path(raw_path)
        if not path.exists():
            loaded.append({"source": str(path), "payload": {}, "load_error": "v5_p28_run_summary_path_missing"})
            continue
        try:
            loaded.append({"source": str(path), "payload": json.loads(path.read_text(encoding="utf-8")), "load_error": ""})
        except Exception as exc:
            loaded.append(
                {
                    "source": str(path),
                    "payload": {},
                    "load_error": f"v5_p28_run_summary_path_error:{type(exc).__name__}",
                }
            )
    return loaded


def _sample_summary(
    *,
    payload: Mapping[str, Any],
    source: str,
    load_error: str,
    min_steps: int,
    min_speedup: float,
) -> dict[str, Any]:
    fields = _extract_fields(payload)
    rollback_events = _string_list(payload.get("rollback_events")) + _string_list(fields.get("rollback_events"))
    blocked = _sample_blockers(
        payload=payload,
        load_error=load_error,
        fields=fields,
        rollback_events=rollback_events,
        min_steps=min_steps,
        min_speedup=min_speedup,
    )
    return {
        "schema_version": 1,
        "source": source,
        "run_id": str(payload.get("run_id") or payload.get("matrix_summary_path") or source),
        "success": bool(fields.get("success", False)),
        "native_case": str(fields.get("native_case") or ""),
        "steps_completed": int(fields.get("steps_completed", 0) or 0),
        "representative_steps": int(fields.get("representative_steps", 0) or 0),
        "representative_end_to_end_speedup": fields.get("representative_end_to_end_speedup"),
        "speedup_vs_baseline": fields.get("representative_end_to_end_speedup"),
        "baseline_mean_step_ms": fields.get("baseline_mean_step_ms"),
        "native_mean_step_ms": fields.get("native_mean_step_ms"),
        "native_dispatch_executed": bool(fields.get("native_dispatch_executed", False)),
        "rollback_events": _dedupe(rollback_events),
        "default_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "request_fields_emitted": False,
        "blocked_reasons": blocked,
    }


def _extract_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    report = _as_dict(payload.get("native_update_performance_report"))
    gate = _as_dict(report.get("performance_gate"))
    evidence = _as_dict(gate.get("evidence"))
    training = _as_dict(evidence.get("training_matrix"))
    native_case = str(training.get("native_case") or payload.get("native_case") or "")
    case_summary = _matrix_case_summary(payload, native_case)
    summary = _as_dict(payload.get("summary"))
    performance = _as_dict(payload.get("performance"))
    speedup = _first_float(
        training.get("end_to_end_speedup"),
        payload.get("representative_end_to_end_speedup"),
        payload.get("end_to_end_speedup"),
        payload.get("speedup_vs_baseline"),
        performance.get("representative_end_to_end_speedup"),
        performance.get("speedup_vs_baseline"),
    )
    representative_steps = _first_int(
        training.get("representative_steps"),
        payload.get("representative_steps"),
        payload.get("steps_completed"),
        summary.get("steps_completed"),
    )
    success = _success(payload, summary)
    return {
        "success": success,
        "native_case": native_case,
        "steps_completed": _first_int(payload.get("steps_completed"), summary.get("steps_completed"), representative_steps),
        "representative_steps": representative_steps,
        "representative_end_to_end_speedup": speedup,
        "baseline_mean_step_ms": _first_float(training.get("baseline_mean_step_ms"), summary.get("baseline_mean_step_ms")),
        "native_mean_step_ms": _first_float(training.get("native_mean_step_ms"), summary.get("native_mean_step_ms")),
        "native_dispatch_executed": bool(
            training.get("native_dispatch_executed", False)
            or payload.get("native_dispatch_executed", False)
            or summary.get("native_dispatch_executed", False)
            or case_summary.get("native_dispatch_executed", False)
        ),
        "rollback_events": _string_list(summary.get("rollback_events")) + _matrix_rollback_events(case_summary),
    }


def _sample_blockers(
    *,
    payload: Mapping[str, Any],
    load_error: str,
    fields: Mapping[str, Any],
    rollback_events: list[str],
    min_steps: int,
    min_speedup: float,
) -> list[str]:
    blocked: list[str] = []
    if load_error:
        blocked.append(load_error)
    if not payload:
        blocked.append("v5_p28_run_payload_missing")
    if not bool(fields.get("success", False)):
        blocked.append("v5_p28_run_not_successful")
    speedup = _float_or_none(fields.get("representative_end_to_end_speedup"))
    if speedup is None:
        blocked.append("v5_p28_speedup_missing")
    elif speedup < min_speedup:
        blocked.append("v5_p28_speedup_below_threshold")
    if int(fields.get("representative_steps", 0) or 0) < min_steps:
        blocked.append("v5_p28_representative_steps_too_low")
    if not bool(fields.get("native_dispatch_executed", False)):
        blocked.append("v5_p28_native_dispatch_not_executed")
    blocked.extend(_string_list(payload.get("blocked_reasons")))
    blocked.extend(_string_list(payload.get("promotion_blockers")))
    if rollback_events:
        blocked.append("v5_p28_rollback_events_present")
        blocked.extend(f"rollback:{event}" for event in rollback_events)
    if bool(payload.get("default_behavior_changed", False)):
        blocked.append("v5_p28_default_behavior_changed")
    if bool(payload.get("default_training_path_enabled", False)) or bool(payload.get("training_path_enabled", False)):
        blocked.append("v5_p28_default_training_path_violation")
    if bool(payload.get("default_rollout_allowed", False)) or bool(payload.get("auto_rollout_allowed", False)):
        blocked.append("v5_p28_default_rollout_violation")
    if bool(payload.get("request_adapter_mapping_allowed", False)) or bool(payload.get("request_fields_emitted", False)):
        blocked.append("v5_p28_request_adapter_violation")
    if bool(_as_dict(payload.get("post_gate_request_fields"))):
        blocked.append("v5_p28_post_gate_request_fields_present")
    return _dedupe(blocked)


def _aggregate(
    *,
    samples: list[Mapping[str, Any]],
    min_runs: int,
    min_speedup: float,
    max_spread: float,
) -> dict[str, Any]:
    speedups = [
        float(sample["representative_end_to_end_speedup"])
        for sample in samples
        if _float_or_none(sample.get("representative_end_to_end_speedup")) is not None
    ]
    blocked: list[str] = []
    ready_runs = [sample for sample in samples if not sample["blocked_reasons"]]
    if len(samples) < min_runs:
        blocked.append("v5_p28_replicate_runs_too_few")
    if len(ready_runs) < len(samples):
        blocked.append("v5_p28_replicate_run_blocked")
    if len(speedups) < min_runs:
        blocked.append("v5_p28_speedup_samples_too_few")
    min_observed = min(speedups) if speedups else None
    mean = statistics.fmean(speedups) if speedups else None
    median = statistics.median(speedups) if speedups else None
    spread = (max(speedups) - min(speedups)) / median if speedups and median and median > 0.0 else None
    if min_observed is not None and min_observed < min_speedup:
        blocked.append("v5_p28_min_speedup_below_threshold")
    if spread is not None and spread > max_spread:
        blocked.append("v5_p28_speedup_spread_too_high")
    return {
        "ready": not blocked,
        "ready_run_count": len(ready_runs),
        "speedup_samples": [round(float(item), 4) for item in speedups],
        "min_speedup": round(float(min_observed), 4) if min_observed is not None else None,
        "mean_speedup": round(float(mean), 4) if mean is not None else None,
        "median_speedup": round(float(median), 4) if median is not None else None,
        "speedup_spread_ratio": round(float(spread), 4) if spread is not None else None,
        "blocked_reasons": _dedupe(blocked),
    }


def _recommended_next_step(blocked: list[str]) -> str:
    if not blocked:
        return "feed this longer replicate evidence bundle into the P26 gate"
    if "v5_p28_replicate_runs_too_few" in blocked:
        return "collect more longer replicate run summaries before P26 review"
    if "v5_p28_replicate_run_blocked" in blocked:
        return "resolve blocked longer replicate samples before P26 review"
    if "v5_p28_speedup_spread_too_high" in blocked:
        return "rerun longer replicates under quieter conditions"
    return "hold longer replicate evidence until all collector checks pass"


def _matrix_case_summary(payload: Mapping[str, Any], native_case: str) -> dict[str, Any]:
    for item in payload.get("cases", []) if isinstance(payload.get("cases"), list) else []:
        entry = _as_dict(item)
        meta = _as_dict(entry.get("case"))
        if str(meta.get("name") or "") == native_case:
            return _as_dict(entry.get("summary"))
    return {}


def _matrix_rollback_events(summary: Mapping[str, Any]) -> list[str]:
    events: list[str] = []
    if summary.get("native_dispatch_training_executor_last_error"):
        events.append("native_error")
    if summary.get("native_dispatch_disabled_for_run"):
        events.append("native_dispatch_disabled_for_run")
    return events


def _success(payload: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    if "success" in payload:
        return bool(payload.get("success"))
    if "all_success" in summary:
        return bool(summary.get("all_success"))
    if "run" in payload and "summary" in payload:
        return bool(payload.get("run")) and bool(summary.get("all_success", False))
    return bool(payload)


def _first_float(*values: Any) -> float | None:
    for value in values:
        out = _float_or_none(value)
        if out is not None:
            return out
    return None


def _first_int(*values: Any) -> int:
    for value in values:
        try:
            out = int(value)
        except (TypeError, ValueError):
            continue
        if out > 0:
            return out
    return 0


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
    parser = argparse.ArgumentParser(description="Collect V5 longer replicate evidence from existing run summaries.")
    parser.add_argument("--run-summary", action="append", default=[], help="Run or matrix summary JSON path.")
    parser.add_argument("--min-runs", type=int, default=MIN_LONGER_REPLICATE_RUNS)
    parser.add_argument("--min-representative-steps", type=int, default=MIN_REPRESENTATIVE_STEPS)
    parser.add_argument("--min-end-to-end-speedup", type=float, default=MIN_END_TO_END_SPEEDUP)
    parser.add_argument("--max-speedup-spread-ratio", type=float, default=MAX_SPEEDUP_SPREAD_RATIO)
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_longer_replicate_evidence_bundle(
        run_summary_paths=args.run_summary,
        min_runs=args.min_runs,
        min_representative_steps=args.min_representative_steps,
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


__all__ = ["build_v5_longer_replicate_evidence_bundle"]
