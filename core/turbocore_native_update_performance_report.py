"""Build representative-performance evidence from profile benchmark summaries."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_native_update_performance import build_native_update_performance_gate


NATIVE_DISPATCH_CASE_NAMES = {
    "native_dispatch",
    "native_update_dispatch",
    "turbocore_native_update",
    "owner_native_dispatch",
}


def build_native_update_profile_performance_report(profile_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Convert native_runtime_profile_benchmark output into performance-gate evidence."""

    summary = _as_dict(profile_summary)
    matrix = build_native_update_benchmark_matrix_from_profile_summary(summary)
    primary_run = _primary_run(summary)
    shadow = _last_item(primary_run.get("update_shadow_reports"))
    readiness = _as_dict(primary_run.get("native_update_readiness"))
    performance_gate = build_native_update_performance_gate(
        readiness_report=readiness,
        shadow_report=shadow,
        performance_report={"benchmark_matrix": matrix},
    )
    return {
        "schema_version": 1,
        "report": "turbocore_native_update_profile_performance_report_v0",
        "source": "native_runtime_profile_benchmark_summary",
        "training_dispatch": False,
        "training_path_enabled": False,
        "runtime_dispatch_allowed": False,
        "family": str(_as_dict(summary.get("benchmark")).get("family", "") or ""),
        "profiles": sorted(str(name) for name in _as_dict(summary.get("runs"))),
        "benchmark_matrix": matrix,
        "performance_gate": performance_gate,
        "blocked_reasons": list(performance_gate.get("blocked_reasons", [])),
    }


def build_native_update_benchmark_matrix_from_profile_summary(profile_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a matrix-shaped report that the native update performance gate can consume."""

    summary = _as_dict(profile_summary)
    benchmark = _as_dict(summary.get("benchmark"))
    runs = _as_dict(summary.get("runs"))
    cases = [_case_from_run(name, run) for name, run in runs.items() if isinstance(run, Mapping)]
    case_summaries = [_as_dict(case.get("summary")) for case in cases]
    return {
        "schema_version": 1,
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "source": "native_runtime_profile_benchmark_summary",
        "run": bool(cases),
        "family": str(benchmark.get("family", "") or ""),
        "profiles": [str(name) for name in runs],
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "executed_count": len(cases),
            "all_success": all(bool(item.get("success", False)) for item in case_summaries) if cases else None,
            "mean_step_ms_by_case": {
                str(case.get("case", {}).get("name")): round(float(case.get("summary", {}).get("mean_step_ms", 0.0) or 0.0), 4)
                for case in cases
            },
            "steady_mean_step_ms_by_case": {
                str(case.get("case", {}).get("name")): round(float(case.get("summary", {}).get("steady_mean_step_ms", 0.0) or 0.0), 4)
                for case in cases
            },
            "native_dispatch_case_present": any(
                str(case.get("case", {}).get("name")) in NATIVE_DISPATCH_CASE_NAMES for case in cases
            ),
            "native_dispatch_executed": any(
                bool(case.get("summary", {}).get("native_dispatch_executed", False)) for case in cases
            ),
        },
    }


def _case_from_run(name: str, run: Mapping[str, Any]) -> dict[str, Any]:
    case_name = _case_name_for_run(str(name), run)
    return {
        "case": {
            "name": case_name,
            "label": _case_label(case_name, str(name)),
            "profile": str(run.get("profile", name) or name),
            "source_run": str(name),
        },
        "summary": {
            "success": bool(run.get("success", False)),
            "steps_completed": int(run.get("steps_completed", 0) or 0),
            "mean_step_ms": _float(run.get("mean_step_ms")),
            "steady_mean_step_ms": _float(run.get("steady_mean_step_ms")),
            "peak_vram_mb": _float(run.get("peak_vram_mb")),
            "optimizer_type": str(run.get("optimizer_type", "") or ""),
            "native_dispatch_requested": _dispatch_requested(run),
            "native_dispatch_executed": _dispatch_executed(run),
            "gate_blocked_reasons": _last_gate_blockers(run),
        },
    }


def _case_name_for_run(name: str, run: Mapping[str, Any]) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in NATIVE_DISPATCH_CASE_NAMES or _dispatch_executed(run):
        return "native_update_dispatch"
    if normalized in {"standard", "baseline", "baseline_phase"}:
        return "baseline_phase"
    return f"{normalized}_profile" if normalized else "unnamed_profile"


def _case_label(case_name: str, source_run: str) -> str:
    if case_name == "baseline_phase":
        return "PyTorch baseline phase profile"
    if case_name == "native_update_dispatch":
        return "TurboCore native update dispatch"
    return f"{source_run} runtime profile"


def _primary_run(summary: Mapping[str, Any]) -> dict[str, Any]:
    runs = _as_dict(summary.get("runs"))
    for name in ("native_update_dispatch", "standard"):
        run = _as_dict(runs.get(name))
        if run:
            return run
    for value in runs.values():
        run = _as_dict(value)
        if run:
            return run
    return {}


def _dispatch_requested(run: Mapping[str, Any]) -> bool:
    reports = run.get("native_update_dispatch_runtime_reports")
    if not isinstance(reports, list):
        return False
    return any(bool(_as_dict(item).get("requested", False)) for item in reports)


def _dispatch_executed(run: Mapping[str, Any]) -> bool:
    reports = run.get("native_update_dispatch_runtime_reports")
    if not isinstance(reports, list):
        return False
    return any(bool(_as_dict(item).get("native_step_executed", False)) for item in reports)


def _last_gate_blockers(run: Mapping[str, Any]) -> list[str]:
    gate = _last_item(run.get("native_update_gate_reports"))
    value = gate.get("blocked_reasons")
    return [str(item) for item in value] if isinstance(value, list) else []


def _last_item(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value:
        return _as_dict(value[-1])
    return {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "build_native_update_benchmark_matrix_from_profile_summary",
    "build_native_update_profile_performance_report",
]
