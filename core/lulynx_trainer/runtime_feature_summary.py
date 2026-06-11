"""Compact runtime feature evidence for benchmark summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return default


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _copy_number_fields(source: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, int | float]:
    copied: dict[str, int | float] = {}
    for field in fields:
        value = source.get(field)
        if isinstance(value, bool) or value is None:
            continue
        number = _safe_float(value)
        copied[field] = int(number) if float(number).is_integer() else _round(number)
    return copied


def summarize_residency_profile(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    source = _mapping(profile)
    if not source:
        return {}

    prefetch = _mapping(source.get("prefetch"))
    pcie_delta = _mapping(source.get("pcie_delta_cache"))
    pcie_cache = _mapping(source.get("pcie_cache_v0"))
    mode = str(source.get("mode", "") or "")
    prefetch_enabled = bool(source.get("prefetch_enabled", prefetch.get("enabled", False)))
    active_linear_count = _safe_int(source.get("active_linear_count"))
    planned_linear_count = _safe_int(source.get("planned_linear_count"))
    prefetch_submitted = _safe_int(prefetch.get("submitted"))
    prefetch_consumed = _safe_int(prefetch.get("consumed"))
    prefetch_missed = _safe_int(prefetch.get("missed"))

    summary: dict[str, Any] = {
        "mode": mode,
        "strategy": str(source.get("strategy", "") or ""),
        "active_linear_count": active_linear_count,
        "managed_linear_count": _safe_int(source.get("managed_linear_count")),
        "planned_linear_count": planned_linear_count,
        "planned_cpu_parameter_mb": _round(source.get("planned_cpu_parameter_mb"), 3),
        "cpu_parameter_mb": _round(source.get("cpu_parameter_mb"), 3),
        "transfer_h2d_mb": _round(source.get("transfer_h2d_mb"), 3),
        "prefetch_enabled": prefetch_enabled,
        "prefetch_depth": _safe_int(source.get("prefetch_depth")),
        "proof_flags": {
            "streaming_offload_active": mode == "streaming_offload" and active_linear_count > 0,
            "prefetch_active": prefetch_enabled and prefetch_submitted > 0,
            "prefetch_all_consumed": prefetch_submitted > 0 and prefetch_consumed >= prefetch_submitted,
            "prefetch_has_misses": prefetch_missed > 0,
            "pcie_cache_active": bool(pcie_cache.get("enabled", False)),
        },
    }

    if prefetch:
        summary["prefetch"] = {
            "enabled": bool(prefetch.get("enabled", prefetch_enabled)),
            "reason": str(prefetch.get("reason", "") or ""),
            "submitted": prefetch_submitted,
            "consumed": prefetch_consumed,
            "missed": prefetch_missed,
            "skipped": _safe_int(prefetch.get("skipped")),
            "errors": _safe_int(prefetch.get("errors")),
            "before_block_calls": _safe_int(prefetch.get("before_block_calls")),
            "prefetch_calls": _safe_int(prefetch.get("prefetch_calls")),
            "planned_block_count": _safe_int(prefetch.get("planned_block_count")),
            "planned_linear_count": _safe_int(prefetch.get("planned_linear_count")),
        }

    if pcie_delta:
        summary["pcie_delta_cache"] = {
            "enabled": bool(pcie_delta.get("enabled", False)),
            "reason": str(pcie_delta.get("reason", "") or ""),
            "next_action": str(pcie_delta.get("next_action", "") or ""),
            **_copy_number_fields(
                pcie_delta,
                (
                    "candidate_count",
                    "high_value_count",
                    "medium_value_count",
                    "low_value_count",
                    "total_transfer_mb",
                    "estimated_cache_mb",
                    "prefetch_submitted_total",
                    "prefetch_consumed_total",
                    "prefetch_missed_total",
                ),
            ),
        }

    if pcie_cache:
        summary["pcie_cache_v0"] = {
            "enabled": bool(pcie_cache.get("enabled", False)),
            "mode": str(pcie_cache.get("mode", "") or ""),
            "reason": str(pcie_cache.get("reason", "") or ""),
            **_copy_number_fields(
                pcie_cache,
                (
                    "budget_mb",
                    "cache_mb",
                    "selected_count",
                    "skipped_count",
                    "hit_count",
                    "miss_count",
                    "error_count",
                ),
            ),
        }

    return summary


def summarize_training_loop_runtime(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    source = _mapping(profile)
    if not source:
        return {}

    step_phase = _mapping(source.get("step_phase_profile"))
    bubble = _mapping(step_phase.get("gpu_bubble_profile"))
    last_step_phase = _mapping(step_phase.get("last"))
    optimizer_breakdown = _mapping(last_step_phase.get("optimizer_update_breakdown"))
    evidence = _mapping(bubble.get("evidence"))
    transfer = _mapping(evidence.get("transfer"))
    data_transfer = _mapping(source.get("data_transfer_profile"))
    last_transfer = _mapping(data_transfer.get("last"))

    summary: dict[str, Any] = {}
    if bubble:
        summary["step_phase_profile"] = {
            "enabled": bool(step_phase.get("enabled", False)),
            "scope": "manifest_cumulative_or_last",
            "dominant_bottleneck": str(bubble.get("dominant_bottleneck", "unknown") or "unknown"),
            "bubble_ratio_estimate": _round(bubble.get("bubble_ratio_estimate")),
            "data_wait_share": _round(evidence.get("data_wait_share")),
            "h2d_transfer_share": _round(evidence.get("h2d_transfer_share")),
            "optimizer_share": _round(evidence.get("optimizer_share")),
            "host_gap_share": _round(evidence.get("host_gap_share")),
        }
        if transfer:
            summary["step_phase_profile"]["transfer"] = {
                "step_share": _round(transfer.get("step_share")),
                "transfer_seconds": _round(transfer.get("transfer_seconds")),
                "mib": _round(transfer.get("mib"), 4),
                "ops": _safe_int(transfer.get("ops")),
                "recommendation": str(transfer.get("recommendation", "") or ""),
            }
        if optimizer_breakdown:
            summary["step_phase_profile"]["optimizer_update_breakdown"] = {
                "profile": str(optimizer_breakdown.get("profile") or ""),
                "has_outer_phase": bool(optimizer_breakdown.get("has_outer_phase")),
                "has_subphase_profile": bool(
                    optimizer_breakdown.get("has_subphase_profile")
                ),
                "optimizer_update_total_ms": _round(
                    optimizer_breakdown.get("optimizer_update_total_ms"),
                    4,
                ),
                "optimizer_step_ms": _round(
                    optimizer_breakdown.get("optimizer_step_ms"),
                    4,
                ),
                "scheduler_step_ms": _round(
                    optimizer_breakdown.get("scheduler_step_ms"),
                    4,
                ),
                "zero_grad_ms": _round(
                    optimizer_breakdown.get("zero_grad_ms"),
                    4,
                ),
                "subphase_accounted_ms": _round(
                    optimizer_breakdown.get("subphase_accounted_ms"),
                    4,
                ),
                "unaccounted_optimizer_update_ms": _round(
                    optimizer_breakdown.get("unaccounted_optimizer_update_ms"),
                    4,
                ),
                "accounted_exceeds_total_ms": _round(
                    optimizer_breakdown.get("accounted_exceeds_total_ms"),
                    4,
                ),
                "unaccounted_share_of_update": _round(
                    optimizer_breakdown.get("unaccounted_share_of_update"),
                ),
                "unaccounted_share_of_step": _round(
                    optimizer_breakdown.get("unaccounted_share_of_step"),
                ),
            }

    if data_transfer:
        summary["data_transfer_profile"] = {
            "enabled": bool(data_transfer.get("enabled", False)),
            "scope": "manifest_last_window",
            "mode": str(data_transfer.get("mode", "") or ""),
            "window": _safe_int(data_transfer.get("window")),
        }
        if last_transfer:
            summary["data_transfer_profile"]["last"] = {
                "step_share": _round(last_transfer.get("step_share")),
                "transfer_seconds": _round(last_transfer.get("transfer_seconds")),
                "mib": _round(last_transfer.get("mib"), 4),
                "ops": _safe_int(last_transfer.get("ops")),
                "recommendation": str(last_transfer.get("recommendation", "") or ""),
            }

    return summary


def build_runtime_feature_summary(manifest_extra: Mapping[str, Any] | None) -> dict[str, Any]:
    extra = _mapping(manifest_extra)
    summary: dict[str, Any] = {}

    for key in ("anima_block_residency", "newbie_block_residency"):
        residency = summarize_residency_profile(_mapping(extra.get(key)))
        if residency:
            summary[key] = residency

    loop = summarize_training_loop_runtime(_mapping(extra.get("training_loop_runtime")))
    if loop:
        summary["training_loop_runtime"] = loop

    experiments = summarize_training_loop_runtime(_mapping(extra.get("anima_full_finetune_experiments")))
    if experiments:
        summary["anima_full_finetune_experiments"] = experiments

    return summary


def load_runtime_feature_summary_from_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return build_runtime_feature_summary(_mapping(payload.get("extra")))


__all__ = [
    "build_runtime_feature_summary",
    "load_runtime_feature_summary_from_manifest",
    "summarize_residency_profile",
    "summarize_training_loop_runtime",
]
