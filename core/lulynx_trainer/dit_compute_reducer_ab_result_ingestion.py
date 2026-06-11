"""Report-only result ingestion for DiT compute reducer A/B evidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_RESULT_FIELDS = (
    "reducer_id",
    "baseline_step_time_ms",
    "candidate_step_time_ms",
    "baseline_peak_vram_mb",
    "candidate_peak_vram_mb",
    "quality_drift",
    "loss_delta",
)


def build_dit_compute_reducer_ab_result_ingestion(
    *,
    evidence_package: Mapping[str, Any],
    result_summaries: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    package = dict(evidence_package)
    results = [dict(item) for item in result_summaries if isinstance(item, Mapping)]
    selected = tuple(str(item) for item in package.get("selected_reducers", ()) if str(item).strip())
    limits = _thresholds(package, thresholds)
    blockers: list[str] = []

    if package.get("scorecard") != "dit_compute_reducer_ab_evidence_package_v0":
        blockers.append("unexpected_ab_evidence_package")
    if not bool(package.get("evidence_package_ready", package.get("ok", False))):
        blockers.append("ab_evidence_package_not_ready")
    if _unsafe_flags(package):
        blockers.append("unsafe_evidence_package_flag")
    if not selected:
        blockers.append("selected_reducers_missing")
    if not results:
        blockers.append("result_summaries_missing")

    by_reducer = {str(item.get("reducer_id") or ""): item for item in results if str(item.get("reducer_id") or "")}
    rows = [_result_row(reducer, by_reducer.get(reducer), limits) for reducer in selected]
    blockers.extend(f"{row['reducer_id']}:{reason}" for row in rows for reason in row["blocked_reasons"])
    if any(_unsafe_flags(item) for item in results):
        blockers.append("unsafe_result_summary_flag")

    ready = not blockers
    passed = [row["reducer_id"] for row in rows if row["ok"]]
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_ab_result_ingestion_v0",
        "ok": ready,
        "ab_result_ingestion_ready": ready,
        "selected_reducers": list(selected),
        "passed_reducers": passed,
        "result_rows": rows,
        "thresholds": limits,
        "ab_execution_allowed": False,
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "trainer_wiring_allowed": False,
        "trainer_wiring_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare default-off quality review for passed compute reducers"
            if ready
            else "collect passing compute-reducer A/B result summaries before review"
        ),
    }


def _result_row(reducer_id: str, result: Mapping[str, Any] | None, thresholds: Mapping[str, float]) -> dict[str, Any]:
    if result is None:
        return {
            "reducer_id": reducer_id,
            "ok": False,
            "step_time_improvement": 0.0,
            "vram_delta_fraction": 0.0,
            "quality_drift": None,
            "loss_delta": None,
            "blocked_reasons": ["result_missing"],
        }
    blockers: list[str] = []
    for name in REQUIRED_RESULT_FIELDS:
        if name not in result:
            blockers.append(f"result_field_missing:{name}")
    baseline_step = _positive_float(result.get("baseline_step_time_ms"))
    candidate_step = _positive_float(result.get("candidate_step_time_ms"))
    baseline_vram = _positive_float(result.get("baseline_peak_vram_mb"))
    candidate_vram = _positive_float(result.get("candidate_peak_vram_mb"))
    quality_drift = _float_or_none(result.get("quality_drift"))
    loss_delta = _float_or_none(result.get("loss_delta"))
    step_improvement = 0.0 if baseline_step <= 0 else (baseline_step - candidate_step) / baseline_step
    vram_delta = 0.0 if baseline_vram <= 0 else (candidate_vram - baseline_vram) / baseline_vram
    if baseline_step <= 0 or candidate_step <= 0:
        blockers.append("step_time_invalid")
    elif step_improvement < thresholds["min_step_time_improvement"]:
        blockers.append("step_time_improvement_below_threshold")
    if baseline_vram <= 0 or candidate_vram <= 0:
        blockers.append("peak_vram_invalid")
    elif vram_delta > thresholds["max_vram_regression"]:
        blockers.append("vram_regression_above_threshold")
    if quality_drift is None:
        blockers.append("quality_drift_missing")
    elif quality_drift > thresholds["max_quality_drift"]:
        blockers.append("quality_drift_above_threshold")
    if loss_delta is None:
        blockers.append("loss_delta_missing")
    elif loss_delta > thresholds["max_loss_delta"]:
        blockers.append("loss_delta_above_threshold")
    if _unsafe_flags(result):
        blockers.append("unsafe_result_flag")
    return {
        "reducer_id": reducer_id,
        "ok": not blockers,
        "step_time_improvement": float(step_improvement),
        "vram_delta_fraction": float(vram_delta),
        "quality_drift": quality_drift,
        "loss_delta": loss_delta,
        "blocked_reasons": blockers,
    }


def _thresholds(package: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, float]:
    package_thresholds = dict(package.get("thresholds") or {})
    if override:
        package_thresholds.update(dict(override))
    return {
        "min_step_time_improvement": _float_or_default(package_thresholds.get("min_step_time_improvement"), 0.05),
        "max_quality_drift": _float_or_default(package_thresholds.get("max_quality_drift"), 0.01),
        "max_loss_delta": _float_or_default(package_thresholds.get("max_loss_delta"), 0.01),
        "max_vram_regression": _float_or_default(package_thresholds.get("max_vram_regression"), 0.05),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "default_rollout_allowed",
        "auto_rollout_allowed",
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "ab_execution_allowed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


def _positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0.0 else 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = ["build_dit_compute_reducer_ab_result_ingestion"]
