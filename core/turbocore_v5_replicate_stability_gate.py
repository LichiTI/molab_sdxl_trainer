"""V5 replicate stability gate for TurboCore native AdamW promotion samples."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.lulynx_trainer.turbocore_update_benchmark_matrix import _build_matrix_performance_report


MIN_REPLICATE_RUNS = 3
MIN_END_TO_END_SPEEDUP = 1.03
MAX_SPEEDUP_SPREAD_RATIO = 0.30


def build_v5_replicate_stability_gate(
    *,
    matrix_payloads: Iterable[Mapping[str, Any]] | None = None,
    matrix_summary_paths: Iterable[str | Path] | None = None,
    min_runs: int = MIN_REPLICATE_RUNS,
    min_end_to_end_speedup: float = MIN_END_TO_END_SPEEDUP,
    max_speedup_spread_ratio: float = MAX_SPEEDUP_SPREAD_RATIO,
) -> dict[str, Any]:
    """Review repeated short matrix samples without enabling rollout."""

    loaded = _load_inputs(matrix_payloads, matrix_summary_paths)
    run_summaries = [
        _run_summary(
            payload=item["payload"],
            source=item["source"],
            load_error=item["load_error"],
            min_end_to_end_speedup=float(min_end_to_end_speedup),
        )
        for item in loaded
    ]
    speedups = [
        float(item["end_to_end_speedup"])
        for item in run_summaries
        if item.get("end_to_end_speedup") is not None
    ]
    ready_runs = [item for item in run_summaries if not item["blocked_reasons"]]
    aggregate = _aggregate_stability(
        speedups=speedups,
        ready_runs=len(ready_runs),
        total_runs=len(run_summaries),
        min_runs=max(int(min_runs), 1),
        min_end_to_end_speedup=float(min_end_to_end_speedup),
        max_speedup_spread_ratio=float(max_speedup_spread_ratio),
    )
    blocked = _dedupe(
        [
            reason
            for item in run_summaries
            for reason in item["blocked_reasons"]
        ]
        + list(aggregate["blocked_reasons"])
    )
    ready = not blocked
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_replicate_stability_gate_v0",
        "gate": "v5_replicate_stability_gate",
        "ok": ready,
        "stability_gate_ready": ready,
        "manual_wider_canary_allowed": ready,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "min_replicate_runs": max(int(min_runs), 1),
        "min_end_to_end_speedup": float(min_end_to_end_speedup),
        "max_speedup_spread_ratio": float(max_speedup_spread_ratio),
        "run_count": len(run_summaries),
        "ready_run_count": len(ready_runs),
        "aggregate": aggregate,
        "runs": run_summaries,
        "blocked_reasons": blocked,
        "promotion_blockers": blocked,
        "recommended_next_step": _recommended_next_step(ready, aggregate, run_summaries),
        "notes": [
            "This gate aggregates existing matrix summaries; it does not run training.",
            "A passing stability gate only permits manual wider canary review.",
            "Default and auto rollout remain disabled even when this gate passes.",
        ],
    }


def _load_inputs(
    matrix_payloads: Iterable[Mapping[str, Any]] | None,
    matrix_summary_paths: Iterable[str | Path] | None,
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for index, payload in enumerate(matrix_payloads or []):
        loaded.append({"source": f"provided_payload:{index}", "payload": dict(payload), "load_error": ""})
    for raw_path in matrix_summary_paths or []:
        path = Path(raw_path)
        if not path.exists():
            loaded.append({"source": str(path), "payload": {}, "load_error": "matrix_summary_path_missing"})
            continue
        try:
            loaded.append({"source": str(path), "payload": json.loads(path.read_text(encoding="utf-8")), "load_error": ""})
        except Exception as exc:
            loaded.append(
                {
                    "source": str(path),
                    "payload": {},
                    "load_error": f"matrix_summary_path_error:{type(exc).__name__}",
                }
            )
    return loaded


def _run_summary(
    *,
    payload: Mapping[str, Any],
    source: str,
    load_error: str,
    min_end_to_end_speedup: float,
) -> dict[str, Any]:
    report = _performance_report(payload) if payload else {}
    gate = _as_dict(report.get("performance_gate"))
    evidence = _as_dict(gate.get("evidence"))
    training = _as_dict(evidence.get("training_matrix"))
    native_summary = _native_case_summary(payload, str(training.get("native_case", "") or ""))
    speedup = _float_or_none(training.get("end_to_end_speedup"))
    blocked = _run_blockers(
        payload=payload,
        load_error=load_error,
        gate=gate,
        training=training,
        native_summary=native_summary,
        speedup=speedup,
        min_end_to_end_speedup=min_end_to_end_speedup,
    )
    return {
        "source": source,
        "matrix": str(payload.get("matrix", "") or "") if payload else "",
        "executed_count": int(_as_dict(payload.get("summary")).get("executed_count", 0) or 0) if payload else 0,
        "all_success": _as_dict(payload.get("summary")).get("all_success") if payload else None,
        "native_case": str(training.get("native_case", "") or ""),
        "baseline_mean_step_ms": _float_or_none(training.get("baseline_mean_step_ms")),
        "native_mean_step_ms": _float_or_none(training.get("native_mean_step_ms")),
        "end_to_end_speedup": speedup,
        "representative_steps": int(training.get("representative_steps", 0) or 0),
        "native_dispatch_executed": bool(training.get("native_dispatch_executed", False)),
        "performance_gate_ready": bool(gate.get("representative_performance_gate_ready", False)),
        "probe_cache_retained": bool(native_summary.get("native_dispatch_probe_cache_retained", False)),
        "timing_summary_present": _timing_present(native_summary),
        "runtime_synchronization": str(native_summary.get("native_dispatch_owner_native_runtime_synchronization", "") or ""),
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "blocked_reasons": blocked,
    }


def _run_blockers(
    *,
    payload: Mapping[str, Any],
    load_error: str,
    gate: Mapping[str, Any],
    training: Mapping[str, Any],
    native_summary: Mapping[str, Any],
    speedup: float | None,
    min_end_to_end_speedup: float,
) -> list[str]:
    blocked: list[str] = []
    summary = _as_dict(payload.get("summary"))
    if load_error:
        blocked.append(load_error)
    if not payload:
        blocked.append("v5_p3_matrix_summary_missing")
    elif payload.get("matrix") != "turbocore_update_benchmark_matrix_v0":
        blocked.append("v5_p3_matrix_schema_invalid")
    if payload and payload.get("run") is not True:
        blocked.append("v5_p3_matrix_not_executed")
    if payload and int(summary.get("executed_count", 0) or 0) < 2:
        blocked.append("v5_p3_matrix_cases_missing")
    if payload and summary.get("all_success") is not True:
        blocked.append("v5_p3_matrix_cases_not_all_success")
    if not bool(gate.get("representative_performance_gate_ready", False)):
        blocked.append("v5_p3_single_run_performance_gate_not_ready")
        blocked.extend(str(item) for item in list(gate.get("blocked_reasons", []) or []))
    if int(training.get("representative_steps", 0) or 0) < 20:
        blocked.append("v5_p3_representative_steps_too_low")
    if not bool(training.get("native_dispatch_executed", False)):
        blocked.append("v5_p3_native_dispatch_not_executed")
    if speedup is None:
        blocked.append("v5_p3_end_to_end_speedup_missing")
    elif speedup < min_end_to_end_speedup:
        blocked.append("v5_p3_end_to_end_speedup_below_threshold")
    if not bool(native_summary.get("native_dispatch_probe_cache_retained", False)):
        blocked.append("v5_p3_probe_cache_retention_missing")
    if not _timing_present(native_summary):
        blocked.append("v5_p3_native_timing_summary_missing")
    return _dedupe(blocked)


def _aggregate_stability(
    *,
    speedups: list[float],
    ready_runs: int,
    total_runs: int,
    min_runs: int,
    min_end_to_end_speedup: float,
    max_speedup_spread_ratio: float,
) -> dict[str, Any]:
    blocked: list[str] = []
    if total_runs < min_runs:
        blocked.append("v5_p3_replicate_runs_too_few")
    if ready_runs < total_runs:
        blocked.append("v5_p3_replicate_run_blocked")
    if len(speedups) < min_runs:
        blocked.append("v5_p3_replicate_speedup_samples_too_few")
    min_speedup = min(speedups) if speedups else None
    mean_speedup = statistics.fmean(speedups) if speedups else None
    median_speedup = statistics.median(speedups) if speedups else None
    spread_ratio = None
    if speedups and median_speedup and median_speedup > 0.0:
        spread_ratio = (max(speedups) - min(speedups)) / median_speedup
    if min_speedup is not None and min_speedup < min_end_to_end_speedup:
        blocked.append("v5_p3_min_speedup_below_threshold")
    if spread_ratio is not None and spread_ratio > max_speedup_spread_ratio:
        blocked.append("v5_p3_speedup_spread_too_high")
    return {
        "ready": not blocked,
        "speedup_samples": [round(float(item), 4) for item in speedups],
        "min_speedup": round(float(min_speedup), 4) if min_speedup is not None else None,
        "mean_speedup": round(float(mean_speedup), 4) if mean_speedup is not None else None,
        "median_speedup": round(float(median_speedup), 4) if median_speedup is not None else None,
        "speedup_spread_ratio": round(float(spread_ratio), 4) if spread_ratio is not None else None,
        "blocked_reasons": _dedupe(blocked),
    }


def _performance_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    report = _as_dict(payload.get("native_update_performance_report"))
    if report:
        return report
    return _build_matrix_performance_report(dict(payload))


def _native_case_summary(payload: Mapping[str, Any], native_case: str) -> dict[str, Any]:
    for item in payload.get("cases", []) if isinstance(payload.get("cases"), list) else []:
        case = _as_dict(item)
        meta = _as_dict(case.get("case"))
        if str(meta.get("name", "") or "") == native_case:
            return _as_dict(case.get("summary"))
    return {}


def _timing_present(native_summary: Mapping[str, Any]) -> bool:
    return bool(
        native_summary.get("native_dispatch_training_executor_timing_present", False)
        and native_summary.get("native_dispatch_update_report_present", False)
        and native_summary.get("native_dispatch_owner_native_report_present", False)
        and native_summary.get("native_dispatch_owner_native_runtime_synchronization")
    )


def _recommended_next_step(ready: bool, aggregate: Mapping[str, Any], runs: list[Mapping[str, Any]]) -> str:
    if ready:
        return "manual wider canary review is ready; default and auto remain off"
    blockers = set(str(item) for item in aggregate.get("blocked_reasons", []) or [])
    if "v5_p3_replicate_runs_too_few" in blockers:
        return "collect more repeated V5 promotion matrix summaries before wider canary"
    if "v5_p3_speedup_spread_too_high" in blockers:
        return "repeat benchmark under quieter conditions or reduce runtime overhead variance"
    if any("timing_summary_missing" in str(reason) for run in runs for reason in run.get("blocked_reasons", []) or []):
        return "rerun with V5-P2 timing-enabled matrix before stability review"
    return "hold manual wider canary until all V5 replicate gates pass"


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_v5_replicate_stability_gate"]
