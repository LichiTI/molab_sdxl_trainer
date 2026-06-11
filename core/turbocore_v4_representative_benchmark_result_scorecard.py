"""V4 representative benchmark result ingestion for exact AdamW canary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.lulynx_trainer.turbocore_update_benchmark_matrix import _build_matrix_performance_report


BENCHMARK_CASES = ("baseline_phase", "native_update_dispatch_perf")
NATIVE_BENCHMARK_CASES = ("native_update_dispatch_promotion_perf", "native_update_dispatch_perf")
MIN_REPRESENTATIVE_STEPS = 20


def build_v4_representative_benchmark_result_scorecard(
    *,
    matrix_payload: Mapping[str, Any] | None = None,
    matrix_summary_path: str | Path | None = None,
    p0_audit: Mapping[str, Any] | None = None,
    require_full_promotion_gate: bool = False,
) -> dict[str, Any]:
    """Evaluate an existing benchmark matrix result without enabling rollout."""

    payload, source, load_error = _load_payload(matrix_payload, matrix_summary_path)
    p0_ready = bool(p0_audit.get("milestone_completed", False)) if isinstance(p0_audit, Mapping) else True
    summary = _as_dict(payload.get("summary")) if payload else {}
    report = _performance_report(payload) if payload else {}
    gate = _as_dict(report.get("performance_gate"))
    evidence = _as_dict(gate.get("evidence"))
    training_matrix = _as_dict(evidence.get("training_matrix"))
    optimizer = _as_dict(evidence.get("optimizer_microbenchmark"))
    native_case = str(training_matrix.get("native_case", "") or "")
    baseline = _case_summary(payload, "baseline_phase") if payload else {}
    native = _case_summary(payload, native_case) if payload and native_case else {}
    speedup = _float_or_none(training_matrix.get("end_to_end_speedup"))
    steps = int(training_matrix.get("representative_steps", 0) or _min_steps(baseline, native))
    progress_gates = {
        "p0_manifest_complete": p0_ready,
        "result_input_present": bool(payload),
        "matrix_schema_ok": bool(payload and payload.get("matrix") == "turbocore_update_benchmark_matrix_v0"),
        "matrix_executed": bool(payload and payload.get("run") is True and int(summary.get("executed_count", 0) or 0) >= 2),
        "baseline_case_executed": bool(baseline.get("success", False)),
        "native_perf_case_executed": bool(native.get("success", False)),
        "all_cases_success": summary.get("all_success") is True,
        "representative_steps_met": steps >= MIN_REPRESENTATIVE_STEPS,
        "native_dispatch_executed": bool(training_matrix.get("native_dispatch_executed", False))
        and native_case in NATIVE_BENCHMARK_CASES,
        "end_to_end_speedup_measured": speedup is not None and speedup > 0.0,
        "performance_gate_evaluated": bool(gate),
        "default_behavior_unchanged": True,
    }
    result_ready = all(progress_gates.values())
    promotion_ready = result_ready and bool(gate.get("representative_performance_gate_ready", False))
    performance_blockers = [str(item) for item in list(gate.get("blocked_reasons", []) or []) if str(item or "")]
    ok = promotion_ready if require_full_promotion_gate else result_ready
    blockers = _build_blockers(progress_gates, gate, load_error, require_full_promotion_gate)
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v4_representative_benchmark_result_scorecard_v0",
        "gate": "v4_representative_benchmark_result_ingestion",
        "ok": ok,
        "milestone_completed": ok,
        "benchmark_result_ready": result_ready,
        "promotion_performance_gate_ready": promotion_ready,
        "real_benchmark_input_present": bool(progress_gates["result_input_present"]),
        "real_benchmark_executed": bool(progress_gates["matrix_executed"]),
        "real_benchmark_contract_ready": result_ready,
        "real_benchmark_performance_gate_ready": promotion_ready,
        "real_benchmark_performance_blockers": performance_blockers,
        "real_benchmark_status": _benchmark_status(
            input_present=bool(progress_gates["result_input_present"]),
            executed=bool(progress_gates["matrix_executed"]),
            contract_ready=result_ready,
            performance_ready=promotion_ready,
        ),
        "require_full_promotion_gate": bool(require_full_promotion_gate),
        "source": source,
        "matrix_summary_path": str(matrix_summary_path or ""),
        "benchmark_cases": ["baseline_phase", native_case or NATIVE_BENCHMARK_CASES[-1]],
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "result_summary": {
            "executed_count": int(summary.get("executed_count", 0) or 0),
            "all_success": summary.get("all_success"),
            "baseline_mean_step_ms": _float_or_none(training_matrix.get("baseline_mean_step_ms")),
            "native_mean_step_ms": _float_or_none(training_matrix.get("native_mean_step_ms")),
            "end_to_end_speedup": speedup,
            "representative_steps": steps,
            "native_case": native_case or None,
            "optimizer_evidence_present": bool(optimizer.get("present", False)),
            "optimizer_evidence_quality": optimizer.get("evidence_quality"),
            "performance_gate_blocked_reasons": performance_blockers,
        },
        "progress_gates": progress_gates,
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": _recommended_next_step(ok, result_ready, promotion_ready),
        "notes": [
            "This scorecard ingests an existing benchmark matrix result; it does not run training.",
            "Synthetic fixture validation is allowed for parser smoke tests, but it is not real promotion evidence.",
            "Default and auto rollout stay disabled regardless of the benchmark result.",
        ],
    }


def _load_payload(
    matrix_payload: Mapping[str, Any] | None,
    matrix_summary_path: str | Path | None,
) -> tuple[dict[str, Any], str, str]:
    if isinstance(matrix_payload, Mapping):
        return dict(matrix_payload), "provided_payload", ""
    if matrix_summary_path:
        path = Path(matrix_summary_path)
        if not path.exists():
            return {}, "matrix_summary_path_missing", f"matrix_summary_path_missing:{path}"
        try:
            return json.loads(path.read_text(encoding="utf-8")), "matrix_summary_path", ""
        except Exception as exc:
            return {}, "matrix_summary_path_error", f"matrix_summary_path_error:{type(exc).__name__}"
    return {}, "missing", "v4_p1_result_input_missing"


def _performance_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    report = _as_dict(payload.get("native_update_performance_report"))
    if report:
        return report
    return _build_matrix_performance_report(dict(payload))


def _case_summary(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    for item in payload.get("cases", []) if isinstance(payload.get("cases"), list) else []:
        case = _as_dict(item)
        meta = _as_dict(case.get("case"))
        if str(meta.get("name", "") or "") == name:
            return _as_dict(case.get("summary"))
    return {}


def _min_steps(*summaries: Mapping[str, Any]) -> int:
    steps = [int(_as_dict(summary).get("steps_completed", 0) or 0) for summary in summaries]
    ready = [step for step in steps if step > 0]
    return min(ready) if ready else 0


def _build_blockers(
    progress_gates: Mapping[str, bool],
    performance_gate: Mapping[str, Any],
    load_error: str,
    require_full_promotion_gate: bool,
) -> list[str]:
    blockers: list[str] = []
    if load_error:
        blockers.append(load_error)
    blockers.extend(f"v4_p1_{name}_missing" for name, ready in progress_gates.items() if not ready)
    if require_full_promotion_gate and not bool(performance_gate.get("representative_performance_gate_ready", False)):
        blockers.extend(str(item) for item in performance_gate.get("blocked_reasons", []) or [])
    return _dedupe(blockers)


def _recommended_next_step(ok: bool, result_ready: bool, promotion_ready: bool) -> str:
    if ok and promotion_ready:
        return "V4-P1 benchmark result ingestion is promotion-grade; proceed to checkpoint/resume validation"
    if ok:
        return "V4-P1 benchmark result ingestion is ready; collect optimizer promotion evidence before rollout review"
    if result_ready:
        return "benchmark result parsed, but promotion performance gate is still blocked"
    return "provide a real matrix_summary.json from V4 representative benchmark run"


def _benchmark_status(
    *,
    input_present: bool,
    executed: bool,
    contract_ready: bool,
    performance_ready: bool,
) -> str:
    if performance_ready:
        return "promotion_ready"
    if contract_ready:
        return "performance_gate_blocked"
    if executed:
        return "contract_blocked"
    if input_present:
        return "input_present_not_executed"
    return "missing"


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_v4_representative_benchmark_result_scorecard"]
