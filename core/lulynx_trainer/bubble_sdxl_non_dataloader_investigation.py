"""Aggregate SDXL real-material evidence after DataLoader source-axis exhaustion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any


SDXL_NON_DATALOADER_INVESTIGATION_REPORT = "bubble_sdxl_non_dataloader_investigation_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _metric(side: Mapping[str, Any], name: str) -> float:
    return _safe_float(_mapping(side.get("metrics")).get(name))


def _axis(side: Mapping[str, Any], name: str, default: int = 0) -> int:
    return _safe_int(_mapping(side.get("matrix_axes")).get(name), default)


def _case_offset(evidence: Mapping[str, Any], path: Path | None = None) -> int:
    before = _mapping(evidence.get("before"))
    axes = _mapping(before.get("matrix_axes"))
    offset = _safe_int(axes.get("sample_offset"), -1)
    if offset >= 0:
        return offset
    text = " ".join(str(item or "") for item in (path, path.parent.name if path else "", evidence.get("case_id")))
    match = re.search(r"offset(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _source_sha(evidence: Mapping[str, Any]) -> str:
    before = _mapping(evidence.get("before"))
    return str(_mapping(before.get("matrix_axes")).get("source_manifest_sha1") or "")


def _case_id(evidence: Mapping[str, Any], path: Path | None) -> str:
    offset = _case_offset(evidence, path)
    if offset >= 0:
        return f"sdxl_offset{offset}"
    if path is not None:
        return path.parent.name
    return str(evidence.get("case_id") or "sdxl_real_material_ab")


def _dominant_counts(cases: Sequence[Mapping[str, Any]], side: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = str(_mapping(case.get(side)).get("dominant_bottleneck") or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: Sequence[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def _case_summary(evidence: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    before = _mapping(evidence.get("before"))
    after = _mapping(evidence.get("after"))
    comparison = _mapping(evidence.get("comparison"))
    loss = _mapping(evidence.get("loss_stability"))
    before_metrics = _mapping(before.get("metrics"))
    after_metrics = _mapping(after.get("metrics"))
    before_data_wait = _metric(before, "data_wait_share")
    after_data_wait = _metric(after, "data_wait_share")
    throughput_gain = _safe_float(comparison.get("steady_samples_per_second_gain_pct"))
    baseline_compute_bound = str(before_metrics.get("dominant_bottleneck") or "") == "compute_bound"
    after_data_wait_worse = after_data_wait > before_data_wait
    release_eligible = bool(_mapping(evidence.get("release_claim")).get("eligible"))
    return {
        "case_id": _case_id(evidence, path),
        "path": str(path or ""),
        "status": str(evidence.get("status") or ""),
        "decision_status": str(_mapping(evidence.get("decision")).get("status") or ""),
        "decision_reasons": _string_list(_mapping(evidence.get("decision")).get("reasons")),
        "release_claim_eligible": release_eligible,
        "source_manifest_sha1": _source_sha(evidence),
        "sample_offset": _case_offset(evidence, path),
        "resolution": _axis(before, "resolution"),
        "samples": _axis(before, "samples"),
        "steps": _axis(before, "steps"),
        "before": {
            "dataloader_workers": _axis(before, "dataloader_workers"),
            "dataloader_prefetch_factor": _axis(before, "dataloader_prefetch_factor"),
            "dominant_bottleneck": str(before_metrics.get("dominant_bottleneck") or "unknown"),
            "data_wait_share": before_data_wait,
            "steady_samples_per_second": _metric(before, "steady_samples_per_second"),
            "h2d_transfer_share": _metric(before, "h2d_transfer_share"),
            "optimizer_share": _metric(before, "optimizer_share"),
            "host_gap_share": _metric(before, "host_gap_share"),
            "final_loss": _metric(before, "final_loss"),
        },
        "after": {
            "dataloader_workers": _axis(after, "dataloader_workers"),
            "dataloader_prefetch_factor": _axis(after, "dataloader_prefetch_factor"),
            "dominant_bottleneck": str(after_metrics.get("dominant_bottleneck") or "unknown"),
            "data_wait_share": after_data_wait,
            "steady_samples_per_second": _metric(after, "steady_samples_per_second"),
            "h2d_transfer_share": _metric(after, "h2d_transfer_share"),
            "optimizer_share": _metric(after, "optimizer_share"),
            "host_gap_share": _metric(after, "host_gap_share"),
            "final_loss": _metric(after, "final_loss"),
        },
        "comparison": {
            "data_wait_share_delta": _safe_float(comparison.get("data_wait_share_delta")),
            "steady_samples_per_second_gain_pct": throughput_gain,
            "loss_regression_ratio": _safe_float(comparison.get("loss_regression_ratio")),
            "optimizer_share_delta": _safe_float(comparison.get("optimizer_share_delta")),
            "h2d_transfer_share_delta": _safe_float(comparison.get("h2d_transfer_share_delta")),
        },
        "loss_stability_status": str(loss.get("status") or "unknown"),
        "signals": {
            "baseline_compute_bound": baseline_compute_bound,
            "baseline_data_wait_below_threshold": before_data_wait < 0.08,
            "after_data_wait_worse": after_data_wait_worse,
            "throughput_positive": throughput_gain > 0.0,
            "throughput_gate_passed": throughput_gain >= 3.0,
            "loss_stable": str(loss.get("status") or "") == "stable",
        },
    }


def _next_actions(summary: Mapping[str, Any]) -> list[str]:
    actions = [
        "do_not_repeat_workers_prefetch_on_current_sdxl_source_axis",
        "keep_sdxl_6_lulu_workers_prefetch_as_negative_data_supply_evidence",
    ]
    if int(summary.get("baseline_compute_bound_count") or 0) >= int(summary.get("case_count") or 0) // 2:
        actions.append("prioritize_compute_or_workload_shape_review")
    if int(summary.get("after_data_wait_worse_count") or 0):
        actions.append("treat_workers_prefetch_as_data_wait_regression_on_this_axis")
    if int(summary.get("large_throughput_without_data_wait_gain_count") or 0):
        actions.append("require_phase_gate_before_using_throughput_deltas")
    actions.extend(
        [
            "scan_or_prepare_new_sdxl_source_axis_before_more_data_supply_canaries",
            "keep_release_claim_blocked_until_natural_data_wait_and_loss_gates_pass",
        ]
    )
    return actions


def build_sdxl_non_dataloader_investigation_report(
    evidences: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Build an aggregate SDXL investigation report from existing A/B evidence."""

    path_list = list(paths or [])
    cases = [
        _case_summary(evidence, path=path_list[index] if index < len(path_list) else None)
        for index, evidence in enumerate(evidences)
        if _mapping(evidence).get("family") in {"sdxl", None} or str(_mapping(evidence).get("family") or "") == "sdxl"
    ]
    cases.sort(key=lambda item: (int(item.get("sample_offset") or 999999), str(item.get("case_id") or "")))
    baseline_compute_bound = [case for case in cases if _mapping(case.get("signals")).get("baseline_compute_bound")]
    baseline_low_data_wait = [
        case for case in cases if _mapping(case.get("signals")).get("baseline_data_wait_below_threshold")
    ]
    after_worse = [case for case in cases if _mapping(case.get("signals")).get("after_data_wait_worse")]
    eligible = [case for case in cases if bool(case.get("release_claim_eligible"))]
    large_misleading = [
        case
        for case in cases
        if _safe_float(_mapping(case.get("comparison")).get("steady_samples_per_second_gain_pct")) >= 25.0
        and _mapping(case.get("signals")).get("baseline_data_wait_below_threshold")
    ]
    summary = {
        "case_count": len(cases),
        "release_eligible_count": len(eligible),
        "baseline_compute_bound_count": len(baseline_compute_bound),
        "baseline_data_wait_below_threshold_count": len(baseline_low_data_wait),
        "after_data_wait_worse_count": len(after_worse),
        "large_throughput_without_data_wait_gain_count": len(large_misleading),
        "mean_before_data_wait_share": _mean([_safe_float(_mapping(case.get("before")).get("data_wait_share")) for case in cases]),
        "mean_after_data_wait_share": _mean([_safe_float(_mapping(case.get("after")).get("data_wait_share")) for case in cases]),
        "mean_throughput_gain_pct": _mean(
            [_safe_float(_mapping(case.get("comparison")).get("steady_samples_per_second_gain_pct")) for case in cases]
        ),
        "before_dominant_bottleneck_counts": _dominant_counts(cases, "before"),
        "after_dominant_bottleneck_counts": _dominant_counts(cases, "after"),
    }
    status = "no_sdxl_ab_evidence" if not cases else "sdxl_dataloader_axis_negative_evidence"
    if eligible:
        status = "sdxl_release_candidate_present"
    summary["next_actions"] = _next_actions(summary) if cases else ["collect_sdxl_real_material_ab_evidence"]
    return {
        "schema_version": 1,
        "report": SDXL_NON_DATALOADER_INVESTIGATION_REPORT,
        "status": status,
        "family": "sdxl",
        "release_claim_allowed": bool(eligible),
        "summary": summary,
        "cases": cases,
        "notes": [
            "This report is an investigation summary, not a release claim.",
            "Positive throughput deltas are not sufficient when the baseline data_wait gate fails.",
            "Use this after the source-axis scout marks the current SDXL axis exhausted.",
        ],
    }


__all__ = [
    "SDXL_NON_DATALOADER_INVESTIGATION_REPORT",
    "build_sdxl_non_dataloader_investigation_report",
]
