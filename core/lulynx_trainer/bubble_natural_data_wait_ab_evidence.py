"""A/B evidence for non-injected natural data_wait probes."""

from __future__ import annotations

import math
from typing import Any, Mapping


NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT = "bubble_natural_data_wait_ab_evidence_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _optional_round(value: Any, digits: int = 6) -> float | None:
    if value is None or value == "":
        return None
    return _round(value, digits)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(report.get("metrics"))


def _analysis(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(report.get("analysis"))


def _axes(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_analysis(report).get("matrix_axes"))


def _cache_probe_only_reason(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    for report in (before, after):
        axes = _axes(report)
        cache_state = str(axes.get("cache_state") or "").strip().lower()
        native_cache_mode = str(axes.get("native_cache_mode") or "").strip().lower()
        source_fixture = str(axes.get("source_fixture") or "").strip()
        if cache_state == "missing_at_start":
            return "cache_miss_fixture_probe_only"
        if native_cache_mode in {"online_cache", "rebuild_cache"}:
            return f"{native_cache_mode}_probe_only"
        if source_fixture == "real_material_canary_v0":
            if cache_state and cache_state != "warm_cache":
                return "real_material_family_cache_missing_probe_only"
            if axes.get("cache_has_family_cache") is False:
                return "real_material_family_cache_missing_probe_only"
    return ""


def _comparison(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_metrics = _metrics(before)
    after_metrics = _metrics(after)
    before_sps = _safe_float(before_metrics.get("steady_samples_per_second"))
    after_sps = _safe_float(after_metrics.get("steady_samples_per_second"))
    gain_ratio = (after_sps / before_sps - 1.0) if before_sps > 0.0 and after_sps > 0.0 else 0.0
    before_wait = _safe_float(before_metrics.get("data_wait_share"))
    after_wait = _safe_float(after_metrics.get("data_wait_share"))
    wait_reduction = before_wait - after_wait
    wait_reduction_ratio = wait_reduction / before_wait if before_wait > 0.0 else 0.0
    before_loss = _finite_float_or_none(before_metrics.get("final_loss"))
    after_loss = _finite_float_or_none(after_metrics.get("final_loss"))
    loss_delta = (after_loss - before_loss) if before_loss is not None and after_loss is not None else None
    loss_regression_ratio = (
        loss_delta / max(abs(before_loss), 1e-12)
        if loss_delta is not None and before_loss is not None
        else None
    )
    return {
        "steady_samples_per_second_before": _round(before_sps, 6),
        "steady_samples_per_second_after": _round(after_sps, 6),
        "steady_samples_per_second_gain_ratio": _round(gain_ratio),
        "steady_samples_per_second_gain_pct": _round(gain_ratio * 100.0, 4),
        "data_wait_share_before": _round(before_wait),
        "data_wait_share_after": _round(after_wait),
        "data_wait_share_delta": _round(after_wait - before_wait),
        "data_wait_reduction_ratio": _round(wait_reduction_ratio),
        "data_wait_reduction_pct": _round(wait_reduction_ratio * 100.0, 4),
        "h2d_transfer_share_delta": _round(
            _safe_float(after_metrics.get("h2d_transfer_share")) - _safe_float(before_metrics.get("h2d_transfer_share"))
        ),
        "optimizer_share_delta": _round(
            _safe_float(after_metrics.get("optimizer_share")) - _safe_float(before_metrics.get("optimizer_share"))
        ),
        "peak_vram_mb_delta": _round(
            _safe_float(after_metrics.get("peak_vram_mb")) - _safe_float(before_metrics.get("peak_vram_mb")),
            4,
        ),
        "final_loss_before": _optional_round(before_loss, 6),
        "final_loss_after": _optional_round(after_loss, 6),
        "final_loss_delta": _optional_round(loss_delta, 6),
        "loss_regression_ratio": _optional_round(loss_regression_ratio, 6),
    }


def _loss_stability(comparison: Mapping[str, Any], *, max_loss_regression_ratio: float) -> dict[str, Any]:
    before_loss = _finite_float_or_none(comparison.get("final_loss_before"))
    after_loss = _finite_float_or_none(comparison.get("final_loss_after"))
    loss_delta = _finite_float_or_none(comparison.get("final_loss_delta"))
    regression_ratio = _finite_float_or_none(comparison.get("loss_regression_ratio"))
    if before_loss is None or after_loss is None or loss_delta is None or regression_ratio is None:
        status = "missing"
    elif regression_ratio > float(max_loss_regression_ratio):
        status = "loss_regressed"
    else:
        status = "stable"
    return {
        "schema": "bubble_loss_stability_v0",
        "source": "natural_data_wait_ab",
        "status": status,
        "final_loss_before": _optional_round(before_loss, 6),
        "final_loss_after": _optional_round(after_loss, 6),
        "final_loss_delta": _optional_round(loss_delta, 6),
        "loss_regression_ratio": _optional_round(regression_ratio, 6),
        "max_loss_regression_ratio": _round(max_loss_regression_ratio),
    }


def _action(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_axes = _axes(before)
    after_axes = _axes(after)
    return {
        "domain": "data_supply",
        "action_kind": "next_run_dataloader_workers_prefetch",
        "status": "observed",
        "before": {
            "dataloader_workers": before_axes.get("dataloader_workers"),
            "dataloader_prefetch_factor": before_axes.get("dataloader_prefetch_factor"),
            "pin_memory": before_axes.get("pin_memory"),
        },
        "after": {
            "dataloader_workers": after_axes.get("dataloader_workers"),
            "dataloader_prefetch_factor": after_axes.get("dataloader_prefetch_factor"),
            "pin_memory": after_axes.get("pin_memory"),
        },
    }


def _decision(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    loss_stability: Mapping[str, Any],
    data_wait_threshold: float,
    min_throughput_gain: float,
    rollback_max_regression_ratio: float,
) -> dict[str, Any]:
    blockers = [
        *(_string_list(before.get("benchmark_injection_blockers"))),
        *(_string_list(after.get("benchmark_injection_blockers"))),
    ]
    cache_probe_only = _cache_probe_only_reason(before, after)
    before_wait = _safe_float(comparison.get("data_wait_share_before"))
    after_wait = _safe_float(comparison.get("data_wait_share_after"))
    gain = _safe_float(comparison.get("steady_samples_per_second_gain_ratio"))
    reasons: list[str] = []
    if blockers:
        return {
            "status": "blocked_benchmark_injection",
            "recommended_action": "do_not_publish",
            "reasons": sorted(set(blockers)),
        }
    if cache_probe_only:
        return {
            "status": "probe_only",
            "recommended_action": "do_not_publish",
            "reasons": [cache_probe_only],
        }
    if before_wait < data_wait_threshold:
        return {
            "status": "insufficient_baseline_data_wait",
            "recommended_action": "collect_more_evidence",
            "reasons": ["before_data_wait_below_threshold"],
        }
    loss_status = str(loss_stability.get("status") or "")
    if loss_status == "missing":
        return {
            "status": "needs_review",
            "recommended_action": "collect_more_evidence",
            "reasons": ["loss_stability_missing"],
        }
    if loss_status == "loss_regressed":
        return {
            "status": "needs_review",
            "recommended_action": "review_or_rollback",
            "reasons": ["loss_regressed"],
        }
    if gain <= -abs(rollback_max_regression_ratio):
        return {
            "status": "rollback_recommended",
            "recommended_action": "rollback",
            "reasons": ["throughput_regressed"],
        }
    if gain >= min_throughput_gain and after_wait < data_wait_threshold and after_wait < before_wait:
        reasons.extend(["throughput_gain_met", "steady_data_wait_reduced_below_threshold"])
        return {"status": "keep_recommended", "recommended_action": "keep", "reasons": reasons}
    return {
        "status": "needs_review",
        "recommended_action": "review",
        "reasons": ["throughput_or_data_wait_delta_below_threshold"],
    }


def build_bubble_natural_data_wait_ab_evidence_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    case_id: str = "",
    data_wait_threshold: float = 0.08,
    min_throughput_gain: float = 0.03,
    rollback_max_regression_ratio: float = 0.02,
    max_loss_regression_ratio: float = 0.05,
) -> dict[str, Any]:
    """Compare two non-injected natural data_wait evidence reports."""

    before_report = _mapping(before)
    after_report = _mapping(after)
    comparison = _comparison(before_report, after_report)
    loss_stability = _loss_stability(
        comparison,
        max_loss_regression_ratio=max(float(max_loss_regression_ratio or 0.0), 0.0),
    )
    decision = _decision(
        before_report,
        after_report,
        comparison,
        loss_stability=loss_stability,
        data_wait_threshold=max(float(data_wait_threshold or 0.0), 0.0),
        min_throughput_gain=max(float(min_throughput_gain or 0.0), 0.0),
        rollback_max_regression_ratio=max(float(rollback_max_regression_ratio or 0.0), 0.0),
    )
    eligible = decision["status"] == "keep_recommended"
    return {
        "schema_version": 1,
        "report": NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT,
        "status": decision["status"],
        "case_id": str(case_id or after_report.get("case_id") or before_report.get("case_id") or "natural_data_wait_ab"),
        "family": str(after_report.get("family") or before_report.get("family") or ""),
        "thresholds": {
            "data_wait_threshold": _round(data_wait_threshold),
            "min_throughput_gain": _round(min_throughput_gain),
            "rollback_max_regression_ratio": _round(rollback_max_regression_ratio),
            "max_loss_regression_ratio": _round(max_loss_regression_ratio),
        },
        "action": _action(before_report, after_report),
        "before": {
            "case_id": before_report.get("case_id"),
            "status": before_report.get("status"),
            "metrics": dict(_metrics(before_report)),
            "matrix_axes": dict(_axes(before_report)),
        },
        "after": {
            "case_id": after_report.get("case_id"),
            "status": after_report.get("status"),
            "metrics": dict(_metrics(after_report)),
            "matrix_axes": dict(_axes(after_report)),
        },
        "comparison": comparison,
        "loss_stability": loss_stability,
        "decision": decision,
        "benchmark_injection_blockers": sorted(
            set(_string_list(before_report.get("benchmark_injection_blockers")) + _string_list(after_report.get("benchmark_injection_blockers")))
        ),
        "release_claim": {
            "eligible": bool(eligible),
            "scope": "case_specific_natural_data_wait_ab" if eligible else "not_eligible",
            "reason": "requires non-injected baseline data_wait, reduced steady data_wait, positive throughput, and stable loss",
        },
    }


__all__ = [
    "NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT",
    "build_bubble_natural_data_wait_ab_evidence_report",
]
