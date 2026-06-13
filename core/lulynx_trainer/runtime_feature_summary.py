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


def _list(value: Any) -> list[Any]:
    return [] if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) else list(value)


def summarize_newbie_backward_op_profile(profile: Mapping[str, Any] | None, *, top_k: int = 12) -> dict[str, Any]:
    source = _mapping(profile)
    if not source:
        return {}

    latest = _mapping(source.get("latest"))
    top_ops = [_mapping(item) for item in _list(latest.get("top_ops")) if _mapping(item)]
    summary: dict[str, Any] = {
        "report": str(source.get("report") or ""),
        "enabled": bool(source.get("enabled", False)),
        "sample_count": _safe_int(source.get("sample_count")),
        "max_samples": _safe_int(source.get("max_samples")),
    }
    if latest:
        top_shape_groups = [
            _mapping(item) for item in _list(latest.get("top_shape_groups")) if _mapping(item)
        ]
        top_matmul_shape_groups = [
            _mapping(item) for item in _list(latest.get("top_matmul_shape_groups")) if _mapping(item)
        ]
        summary["latest"] = {
            "report": str(latest.get("report") or ""),
            "status": str(latest.get("status") or ""),
            "step": _safe_int(latest.get("step")),
            "sort_key": str(latest.get("sort_key") or ""),
            "top_k": _safe_int(latest.get("top_k")),
            "event_count": _safe_int(latest.get("event_count")),
            "record_shapes": bool(latest.get("record_shapes", False)),
            "shape_group_count": _safe_int(latest.get("shape_group_count")),
            "cuda_activity_available": bool(latest.get("cuda_activity_available", False)),
            "activities": [str(item) for item in _list(latest.get("activities"))],
            "top_ops": [
                {
                    "key": str(row.get("key") or ""),
                    "count": _safe_int(row.get("count")),
                    "self_cuda_ms": _round(row.get("self_cuda_ms"), 4),
                    "cuda_ms": _round(row.get("cuda_ms"), 4),
                    "self_cpu_ms": _round(row.get("self_cpu_ms"), 4),
                    "cpu_ms": _round(row.get("cpu_ms"), 4),
                }
                for row in top_ops[: max(int(top_k), 1)]
            ],
        }
        if top_shape_groups:
            summary["latest"]["top_shape_groups"] = [
                {
                    "key": str(row.get("key") or ""),
                    "input_shapes": str(row.get("input_shapes") or ""),
                    "count": _safe_int(row.get("count")),
                    "self_cuda_ms": _round(row.get("self_cuda_ms"), 4),
                    "cuda_ms": _round(row.get("cuda_ms"), 4),
                    "self_cpu_ms": _round(row.get("self_cpu_ms"), 4),
                    "cpu_ms": _round(row.get("cpu_ms"), 4),
                }
                for row in top_shape_groups[: max(int(top_k), 1)]
            ]
        if top_matmul_shape_groups:
            summary["latest"]["top_matmul_shape_groups"] = [
                {
                    "key": str(row.get("key") or ""),
                    "input_shapes": str(row.get("input_shapes") or ""),
                    "count": _safe_int(row.get("count")),
                    "self_cuda_ms": _round(row.get("self_cuda_ms"), 4),
                    "cuda_ms": _round(row.get("cuda_ms"), 4),
                    "self_cpu_ms": _round(row.get("self_cpu_ms"), 4),
                    "cpu_ms": _round(row.get("cpu_ms"), 4),
                }
                for row in top_matmul_shape_groups[: max(int(top_k), 1)]
            ]
    return summary


def summarize_newbie_module_timing_profile(profile: Mapping[str, Any] | None, *, top_k: int = 12) -> dict[str, Any]:
    source = _mapping(profile)
    if not source:
        return {}

    latest = _mapping(source.get("latest"))
    top_groups = [_mapping(item) for item in _list(latest.get("top_groups")) if _mapping(item)]
    summary: dict[str, Any] = {
        "report": str(source.get("report") or ""),
        "enabled": bool(source.get("enabled", False)),
        "sample_count": _safe_int(source.get("sample_count")),
        "max_samples": _safe_int(source.get("max_samples")),
    }
    if latest:
        summary["latest"] = {
            "report": str(latest.get("report") or ""),
            "status": str(latest.get("status") or ""),
            "step": _safe_int(latest.get("step")),
            "top_k": _safe_int(latest.get("top_k")),
            "tracked_module_count": _safe_int(latest.get("tracked_module_count")),
            "group_count": _safe_int(latest.get("group_count")),
            "cuda_activity_available": bool(latest.get("cuda_activity_available", False)),
            "runtime_default_change": bool(latest.get("runtime_default_change", False)),
            "probe_only": bool(latest.get("probe_only", False)),
            "top_groups": [
                {
                    "group": str(row.get("group") or ""),
                    "module_count": _safe_int(row.get("module_count")),
                    "forward_count": _safe_int(row.get("forward_count")),
                    "backward_count": _safe_int(row.get("backward_count")),
                    "forward_cuda_ms": _round(row.get("forward_cuda_ms"), 4),
                    "backward_cuda_ms": _round(row.get("backward_cuda_ms"), 4),
                    "forward_cpu_ms": _round(row.get("forward_cpu_ms"), 4),
                    "backward_cpu_ms": _round(row.get("backward_cpu_ms"), 4),
                    "module_name_examples": [str(item) for item in _list(row.get("module_name_examples"))[:5]],
                }
                for row in top_groups[: max(int(top_k), 1)]
            ],
        }
    return summary


def summarize_triton_ops_runtime(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    source = _mapping(profile)
    if not source:
        return {}
    return {
        "report": str(source.get("report") or ""),
        "enabled": bool(source.get("enabled", False)),
        "requested": bool(source.get("requested", False)),
        "available": bool(source.get("available", False)),
        "status": str(source.get("status") or ""),
        "reason": str(source.get("reason") or ""),
        "dtype": str(source.get("dtype") or ""),
        "gpu": str(source.get("gpu") or ""),
        "patched_lora_layers": _safe_int(source.get("patched_lora_layers")),
        "patched_qkv_blocks": _safe_int(source.get("patched_qkv_blocks")),
        "patched_adaln_blocks": _safe_int(source.get("patched_adaln_blocks")),
        "inject_lora": bool(source.get("inject_lora", False)),
        "inject_qkv": bool(source.get("inject_qkv", False)),
        "inject_adaln": bool(source.get("inject_adaln", False)),
        "fp32_backward": bool(source.get("fp32_backward", False)),
    }


def summarize_adapter_runtime(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    source = _mapping(profile)
    if not source:
        return {}
    return {
        "source": str(source.get("source") or ""),
        "enabled": bool(source.get("enabled", False)),
        "model_arch": str(source.get("model_arch") or ""),
        "adapter_method": str(source.get("adapter_method") or ""),
        "rank": _safe_int(source.get("rank")),
        "injected_layer_count": _safe_int(source.get("injected_layer_count")),
        "newbie_target_scope": str(source.get("newbie_target_scope") or ""),
        "newbie_target_module_count": _safe_int(source.get("newbie_target_module_count")),
        "total_adapter_parameter_count": _safe_int(source.get("total_adapter_parameter_count")),
        "trainable_adapter_parameter_count": _safe_int(source.get("trainable_adapter_parameter_count")),
        "prefix_counts": {
            str(key): _safe_int(value)
            for key, value in _mapping(source.get("prefix_counts")).items()
        },
        "sample_layers": [str(item) for item in _list(source.get("sample_layers"))[:8]],
    }


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
    backward_op_profile = summarize_newbie_backward_op_profile(
        _mapping(source.get("newbie_backward_op_profile"))
    )
    module_timing_profile = summarize_newbie_module_timing_profile(
        _mapping(source.get("newbie_module_timing_profile"))
    )

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

    if backward_op_profile:
        summary["newbie_backward_op_profile"] = backward_op_profile
    if module_timing_profile:
        summary["newbie_module_timing_profile"] = module_timing_profile

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

    triton_ops = summarize_triton_ops_runtime(_mapping(extra.get("triton_ops_runtime")))
    if triton_ops:
        summary["triton_ops_runtime"] = triton_ops

    adapter = summarize_adapter_runtime(_mapping(extra.get("adapter_runtime")))
    if adapter:
        summary["adapter_runtime"] = adapter

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
    "summarize_adapter_runtime",
    "summarize_newbie_backward_op_profile",
    "summarize_newbie_module_timing_profile",
    "summarize_residency_profile",
    "summarize_training_loop_runtime",
    "summarize_triton_ops_runtime",
]
