"""Representative performance gate for future TurboCore native updates.

This is a report-only contract.  It can say that benchmark evidence is strong
enough for the next validation stage, but it never enables optimizer dispatch.
"""

from __future__ import annotations

from typing import Any, Mapping


MIN_REPRESENTATIVE_STEPS = 20
MIN_END_TO_END_SPEEDUP = 1.03
MIN_KERNEL_SPEEDUP = 1.10
PROMOTION_KERNEL_SPEEDUP = 1.20
NATIVE_UPDATE_CASES = (
    "native_update_dispatch_promotion_perf",
    "native_update_dispatch_perf",
    "native_dispatch_perf",
    "native_dispatch_clean_perf",
    "native_dispatch",
    "native_update_dispatch",
    "turbocore_native_update",
    "owner_native_dispatch",
)


def build_native_update_performance_gate(
    *,
    readiness_report: Mapping[str, Any] | None = None,
    shadow_report: Mapping[str, Any] | None = None,
    performance_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether native update performance evidence is representative."""

    readiness = _as_dict(readiness_report)
    shadow = _as_dict(shadow_report)
    perf = _as_dict(performance_report)
    optimizer_gate = _optimizer_gate_status(_find_optimizer_gate(readiness, shadow, perf))
    owner_kernel = _owner_kernel_status(shadow)
    training_matrix = _training_matrix_status(_find_training_matrix(readiness, shadow, perf))
    blockers = _dedupe(
        optimizer_gate["blocked_reasons"]
        + owner_kernel["blocked_reasons"]
        + training_matrix["blocked_reasons"]
    )
    ready = bool(not blockers)
    return {
        "schema_version": 1,
        "gate": "turbocore_native_update_performance_gate_v0",
        "policy_defined": True,
        "performance_test_ready": ready,
        "representative_performance_gate_ready": ready,
        "promotion_gate_ok": ready,
        "training_dispatch": False,
        "training_path_enabled": False,
        "runtime_dispatch_allowed": False,
        "required_training_steps": MIN_REPRESENTATIVE_STEPS,
        "required_end_to_end_speedup": MIN_END_TO_END_SPEEDUP,
        "required_kernel_speedup": MIN_KERNEL_SPEEDUP,
        "promotion_kernel_speedup": PROMOTION_KERNEL_SPEEDUP,
        "evidence": {
            "optimizer_microbenchmark": optimizer_gate,
            "owner_native_kernel": owner_kernel,
            "training_matrix": training_matrix,
        },
        "blocked_reasons": blockers,
    }


def _optimizer_gate_status(gate: Mapping[str, Any]) -> dict[str, Any]:
    blocked: list[str] = []
    best = _as_dict(gate.get("best_candidate") or gate.get("best_measured_candidate"))
    speedup = _float_or_none(best.get("speedup_vs_baseline"))
    quality = str(gate.get("evidence_quality", "") or "")
    if not gate:
        blocked.append("optimizer_microbenchmark_missing")
    elif not bool(gate.get("ok", False)):
        blocked.append("optimizer_microbenchmark_gate_not_ok")
    if gate and not bool(gate.get("promotion_gate_ok", False)):
        blocked.append("optimizer_microbenchmark_promotion_gate_not_ok")
    if gate and quality != "promotion_benchmark":
        blocked.append("optimizer_microbenchmark_not_promotion_grade")
    if speedup is None:
        if gate:
            blocked.append("optimizer_microbenchmark_speedup_missing")
    elif speedup < PROMOTION_KERNEL_SPEEDUP:
        blocked.append("optimizer_microbenchmark_speedup_below_promotion")
    return {
        "ok": not blocked,
        "present": bool(gate),
        "source_gate": str(gate.get("gate", "") or gate.get("benchmark", "") or ""),
        "status": str(gate.get("status", "") or ""),
        "evidence_quality": quality or None,
        "best_candidate_optimizer": str(best.get("optimizer", "") or "") if best else None,
        "best_speedup_vs_baseline": speedup,
        "promotion_gate_ok": bool(gate.get("promotion_gate_ok", False)) if gate else False,
        "runtime_dispatch_allowed": bool(gate.get("runtime_dispatch_allowed", False)) if gate else False,
        "blocked_reasons": _dedupe(blocked),
    }


def _owner_kernel_status(shadow: Mapping[str, Any]) -> dict[str, Any]:
    owner = _as_dict(shadow.get("owner_native_launch_probe"))
    blocked: list[str] = []
    if not owner:
        blocked.append("owner_backed_native_kernel_evidence_missing")
    elif bool(owner.get("skipped", False)):
        blocked.append(str(owner.get("reason", "owner_native_launch_probe_skipped") or "owner_native_launch_probe_skipped"))
    elif not bool(owner.get("ok", False)):
        blocked.append("owner_backed_native_kernel_probe_failed")
    if owner and not bool(owner.get("kernel_executed", False)):
        blocked.append("owner_backed_native_kernel_not_executed")
    if owner and not bool(owner.get("parity_ok", False)):
        blocked.append("owner_backed_native_kernel_parity_failed")
    if owner and bool(owner.get("persistent_owner_mutated", False)):
        blocked.append("owner_backed_probe_mutated_persistent_owner")
    return {
        "ok": not blocked,
        "present": bool(owner),
        "attempted": bool(owner.get("attempted", False)) if owner else False,
        "kernel_executed": bool(owner.get("kernel_executed", False)) if owner else False,
        "parity_ok": bool(owner.get("parity_ok", False)) if owner else False,
        "owner_numel": _int_or_none(owner.get("owner_numel")) if owner else None,
        "elapsed_ms": _float_or_none(owner.get("elapsed_ms")) if owner else None,
        "diagnostic_only": True,
        "blocked_reasons": _dedupe(blocked),
    }


def _training_matrix_status(matrix: Mapping[str, Any]) -> dict[str, Any]:
    blocked: list[str] = []
    cases = matrix.get("cases") if isinstance(matrix.get("cases"), list) else []
    by_name = {_case_name(case): _case_summary(case) for case in cases if _case_name(case)}
    summary = _as_dict(matrix.get("summary"))
    mean_by_case = _as_dict(summary.get("mean_step_ms_by_case"))
    baseline_ms = _case_mean_ms("baseline_phase", by_name, mean_by_case)
    native_name = _find_native_case_name(by_name, mean_by_case)
    native_ms = _case_mean_ms(native_name, by_name, mean_by_case) if native_name else None
    baseline_steps = _case_steps("baseline_phase", by_name)
    native_steps = _case_steps(native_name, by_name) if native_name else 0
    native_dispatch_executed = _case_native_dispatch_executed(native_name, by_name) if native_name else False
    min_steps = min(step for step in (baseline_steps, native_steps) if step > 0) if baseline_steps and native_steps else 0
    speedup = baseline_ms / native_ms if baseline_ms and native_ms else None
    if not matrix:
        blocked.append("representative_training_matrix_missing")
    elif not bool(matrix.get("run", False)) and int(summary.get("executed_count", 0) or 0) <= 0:
        blocked.append("representative_training_matrix_not_executed")
    if matrix and summary.get("all_success") is False:
        blocked.append("representative_training_matrix_failed")
    if matrix and baseline_ms is None:
        blocked.append("baseline_training_case_missing")
    if matrix and not native_name:
        blocked.append("native_dispatch_benchmark_case_missing")
    if matrix and native_name and not native_dispatch_executed:
        blocked.append("native_dispatch_not_executed_in_benchmark_case")
    if matrix and min_steps and min_steps < MIN_REPRESENTATIVE_STEPS:
        blocked.append("representative_training_steps_too_low")
    elif matrix and not min_steps:
        blocked.append("representative_training_steps_missing")
    if speedup is None:
        if matrix and native_name:
            blocked.append("end_to_end_speedup_missing")
    elif speedup < MIN_END_TO_END_SPEEDUP:
        blocked.append("end_to_end_speedup_below_threshold")
    return {
        "ok": not blocked,
        "present": bool(matrix),
        "matrix": str(matrix.get("matrix", "") or "") if matrix else "",
        "executed_count": int(summary.get("executed_count", 0) or 0),
        "all_success": summary.get("all_success") if summary else None,
        "baseline_case": "baseline_phase" if baseline_ms is not None else None,
        "native_case": native_name,
        "native_dispatch_executed": native_dispatch_executed if native_name else None,
        "baseline_mean_step_ms": baseline_ms,
        "native_mean_step_ms": native_ms,
        "end_to_end_speedup": round(speedup, 4) if speedup is not None else None,
        "representative_steps": int(min_steps or 0),
        "required_steps": MIN_REPRESENTATIVE_STEPS,
        "blocked_reasons": _dedupe(blocked),
    }


def _find_optimizer_gate(*reports: Mapping[str, Any]) -> dict[str, Any]:
    for report in reports:
        for key in ("native_update_optimizer_performance_gate", "optimizer_performance_gate", "performance_gate"):
            value = _as_dict(report.get(key))
            if value:
                return value
        if report.get("gate") == "turbocore_optimizer_performance_gate":
            return dict(report)
    return {}


def _find_training_matrix(*reports: Mapping[str, Any]) -> dict[str, Any]:
    for report in reports:
        for key in ("native_update_benchmark_matrix", "update_benchmark_matrix", "benchmark_matrix"):
            value = _as_dict(report.get(key))
            if value:
                return value
        if report.get("matrix") == "turbocore_update_benchmark_matrix_v0":
            return dict(report)
    return {}


def _case_name(case: Mapping[str, Any]) -> str:
    meta = _as_dict(case.get("case"))
    return str(meta.get("name", "") or case.get("name", "") or "")


def _case_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(case.get("summary"))


def _find_native_case_name(by_name: Mapping[str, Any], mean_by_case: Mapping[str, Any]) -> str:
    names = set(by_name) | set(str(name) for name in mean_by_case)
    for name in NATIVE_UPDATE_CASES:
        if name in names:
            return name
    return ""


def _case_mean_ms(name: str, by_name: Mapping[str, Any], mean_by_case: Mapping[str, Any]) -> float | None:
    if not name:
        return None
    summary = _as_dict(by_name.get(name))
    return _float_or_none(summary.get("steady_mean_step_ms") or summary.get("mean_step_ms") or mean_by_case.get(name))


def _case_steps(name: str, by_name: Mapping[str, Any]) -> int:
    if not name:
        return 0
    return int(_as_dict(by_name.get(name)).get("steps_completed", 0) or 0)


def _case_native_dispatch_executed(name: str, by_name: Mapping[str, Any]) -> bool:
    if not name:
        return False
    return bool(_as_dict(by_name.get(name)).get("native_dispatch_executed", False))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def _int_or_none(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_native_update_performance_gate"]
