"""Normalize runtime evidence for bubble-aware controller decisions."""

from __future__ import annotations

import os
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


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _mapping_list(value: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)][: max(int(limit), 0)]


def _is_auto(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "auto"}


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _sum_keyword_share(phase_share: Mapping[str, Any], keywords: tuple[str, ...]) -> float:
    total = 0.0
    for label, share in phase_share.items():
        lowered = str(label).lower()
        if any(keyword in lowered for keyword in keywords):
            total += _safe_float(share)
    return min(max(total, 0.0), 1.0)


def _phase_breakdown(
    phase_share: Mapping[str, Any],
    phase_mean: Mapping[str, Any],
    keywords: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    share: dict[str, float] = {}
    mean_ms: dict[str, float] = {}
    for label, value in phase_share.items():
        lowered = str(label).lower()
        if any(keyword in lowered for keyword in keywords):
            share[str(label)] = _round(value)
    for label, value in phase_mean.items():
        lowered = str(label).lower()
        if any(keyword in lowered for keyword in keywords):
            mean_ms[str(label)] = _round(value, 4)
    return share, mean_ms


def _cfg(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _first_mapping(*values: Any) -> Mapping[str, Any]:
    for value in values:
        mapped = _mapping(value)
        if mapped:
            return mapped
    return {}


def _step_phase_source(features: Mapping[str, Any]) -> Mapping[str, Any]:
    loop = _mapping(features.get("training_loop_runtime"))
    experiments = _mapping(features.get("anima_full_finetune_experiments"))
    summary = _mapping(features.get("runtime_feature_summary"))
    summary_loop = _mapping(summary.get("training_loop_runtime"))
    summary_experiments = _mapping(summary.get("anima_full_finetune_experiments"))
    return _first_mapping(
        _mapping(loop.get("step_phase_profile")),
        _mapping(experiments.get("step_phase_profile")),
        _mapping(summary_loop.get("step_phase_profile")),
        _mapping(summary_experiments.get("step_phase_profile")),
    )


def _extract_step_phase(features: Mapping[str, Any]) -> dict[str, Any]:
    source = _step_phase_source(features)
    bubble = _mapping(source.get("gpu_bubble_profile")) or source
    evidence = _mapping(bubble.get("evidence"))
    transfer = _mapping(evidence.get("transfer"))
    phase_share = _mapping(bubble.get("phase_share"))
    phase_mean = _mapping(bubble.get("phase_mean_ms"))
    data_transfer = _mapping(_mapping(features.get("training_loop_runtime")).get("data_transfer_profile"))
    last_transfer = _mapping(data_transfer.get("last"))
    closed_loop_window = _mapping(features.get("bubble_closed_loop_window"))
    host_keywords = ("log", "callback", "checkpoint", "save", "validation", "safeguard")

    h2d_share = _safe_float(evidence.get("h2d_transfer_share"), _safe_float(source.get("h2d_transfer_share")))
    if h2d_share <= 0.0:
        h2d_share = _safe_float(transfer.get("step_share"), _safe_float(last_transfer.get("step_share")))
    logging_checkpoint_share = _safe_float(
        evidence.get("logging_checkpoint_share"),
        _sum_keyword_share(phase_share, host_keywords),
    )
    top_phases = evidence.get("top_phases", bubble.get("top_phases", []))
    if not isinstance(top_phases, list):
        top_phases = []
    host_phase_share, host_phase_mean_ms = _phase_breakdown(phase_share, phase_mean, host_keywords)

    return {
        "available": bool(source),
        "dominant_bottleneck": str(bubble.get("dominant_bottleneck", source.get("dominant_bottleneck", "unknown")) or "unknown"),
        "bubble_ratio_estimate": _round(bubble.get("bubble_ratio_estimate", source.get("bubble_ratio_estimate"))),
        "data_wait_share": _round(evidence.get("data_wait_share", source.get("data_wait_share"))),
        "h2d_transfer_share": _round(h2d_share),
        "optimizer_share": _round(evidence.get("optimizer_share", source.get("optimizer_share"))),
        "host_gap_share": _round(evidence.get("host_gap_share", source.get("host_gap_share"))),
        "logging_checkpoint_share": _round(logging_checkpoint_share),
        "train_step_share": _round(evidence.get("train_step_share", phase_share.get("train_step_total"))),
        "mean_step_ms": _round(
            closed_loop_window.get("mean_step_ms", bubble.get("mean_step_ms", source.get("mean_step_ms"))),
            4,
        ),
        "steady_samples_per_second": _round(
            closed_loop_window.get(
                "steady_samples_per_second",
                bubble.get("steady_samples_per_second", source.get("steady_samples_per_second")),
            ),
            6,
        ),
        "throughput_estimated": _safe_bool(closed_loop_window.get("throughput_estimated"), False),
        "final_loss": _round(closed_loop_window.get("final_loss"), 6),
        "window_step_count": _safe_int(closed_loop_window.get("step_count")),
        "train_step_ms": _round(phase_mean.get("train_step_total"), 4),
        "host_phase_share": host_phase_share,
        "host_phase_mean_ms": host_phase_mean_ms,
        "top_phases": [dict(item) for item in top_phases if isinstance(item, Mapping)][:8],
        "transfer_mib": _round(transfer.get("mib", last_transfer.get("mib")), 4),
        "transfer_ops": _safe_int(transfer.get("ops", last_transfer.get("ops"))),
    }


def _extract_gpu(features: Mapping[str, Any]) -> dict[str, Any]:
    windows = _mapping(features.get("gpu_telemetry_windows"))
    active = _mapping(windows.get("active_window_gpu20"))
    full = _mapping(features.get("gpu_telemetry"))
    classification = _mapping(features.get("classification"))
    source = _first_mapping(active, full, classification)
    return {
        "available": bool(source),
        "scope": str(source.get("scope", "full_run") or "full_run"),
        "active_gpu_util_pct_mean": _round(
            source.get("gpu_util_pct_mean", source.get("active_gpu_util_pct_mean")),
            4,
        ),
        "active_gpu_saturated_sample_ratio": _round(
            source.get("gpu_saturated_sample_ratio", source.get("active_gpu_saturated_sample_ratio")),
        ),
        "active_gpu_idle_sample_ratio": _round(
            source.get("gpu_idle_sample_ratio", source.get("active_gpu_idle_sample_ratio")),
        ),
        "memory_used_mb_max": _round(source.get("memory_used_mb_max"), 3),
        "memory_total_mb": _round(source.get("memory_total_mb"), 3),
        "pcie_rx_mib_s_mean": _round(source.get("pcie_rx_mib_s_mean"), 4),
        "pcie_tx_mib_s_mean": _round(source.get("pcie_tx_mib_s_mean"), 4),
    }


def _residency_source(features: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    summary = _mapping(features.get("runtime_feature_summary"))
    candidates = (
        ("anima_block_residency", features.get("anima_block_residency")),
        ("newbie_block_residency", features.get("newbie_block_residency")),
        ("anima_block_residency", summary.get("anima_block_residency")),
        ("newbie_block_residency", summary.get("newbie_block_residency")),
    )
    for key, value in candidates:
        mapped = _mapping(value)
        if mapped:
            return key, mapped
    return "", {}


def _extract_runtime(config: Any, features: Mapping[str, Any]) -> dict[str, Any]:
    residency_source, residency = _residency_source(features)
    dataloader_rebuild_readiness = dict(_mapping(features.get("dataloader_rebuild_readiness")))
    prefetch = _mapping(residency.get("prefetch"))
    proof = _mapping(residency.get("proof_flags"))
    mode = str(residency.get("mode", "") or "")
    offload_active = bool(proof.get("streaming_offload_active", False)) or (
        mode == "streaming_offload" and _safe_int(residency.get("active_linear_count")) > 0
    )
    auto_policy = _safe_bool(_cfg(config, "cached_dataloader_auto_policy", True), True)
    raw_workers = _cfg(config, "cached_dataloader_workers", "auto")
    if auto_policy and _is_auto(raw_workers):
        cpu_count = max(int(os.cpu_count() or 1), 1)
        workers = 2 if os.name == "nt" and cpu_count >= 8 else 1 if os.name == "nt" else max(1, min(4, cpu_count // 2))
    else:
        workers = _safe_int(raw_workers, _safe_int(_cfg(config, "dataloader_num_workers", 0)))
    workers = max(workers, 0)

    raw_prefetch = _cfg(config, "cached_dataloader_prefetch_factor", "auto")
    if workers <= 0:
        prefetch_factor = 0
    elif auto_policy and _is_auto(raw_prefetch):
        prefetch_factor = 4 if workers > 2 else 2
    else:
        prefetch_factor = max(_safe_int(raw_prefetch, _safe_int(_cfg(config, "prefetch_factor", 2), 2)), 1)

    raw_pin = _cfg(config, "cached_dataloader_pin_memory", "auto")
    pin_memory = _safe_bool(_cfg(config, "pin_memory", True), True) if _is_auto(raw_pin) else _safe_bool(raw_pin, True)

    prefetch_key = ""
    prefetch_depth_key = ""
    if residency_source == "anima_block_residency":
        prefetch_key = "anima_block_prefetch"
        prefetch_depth_key = "anima_block_prefetch_depth"
    elif residency_source == "newbie_block_residency":
        prefetch_key = "newbie_block_prefetch"
        prefetch_depth_key = "newbie_block_prefetch_depth"

    return {
        "cached_dataloader_auto_policy": auto_policy,
        "cached_dataloader_workers_raw": raw_workers,
        "cached_dataloader_prefetch_factor_raw": raw_prefetch,
        "workers": workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "train_batch_size": _safe_int(_cfg(config, "train_batch_size", 1), 1),
        "gradient_accumulation_steps": _safe_int(_cfg(config, "gradient_accumulation_steps", 1), 1),
        "data_transfer_non_blocking": _safe_bool(_cfg(config, "data_transfer_non_blocking", True), True),
        "data_transfer_profile_mode": str(_cfg(config, "data_transfer_profile_mode", "event") or "event"),
        "step_phase_profile_enabled": _safe_bool(_cfg(config, "step_phase_profile_enabled", False), False),
        "benchmark_data_wait_stall_ms": _round(_cfg(config, "bubble_controller_benchmark_data_wait_stall_ms", 0.0), 4),
        "benchmark_data_wait_direct_action": _safe_bool(
            _cfg(config, "bubble_controller_benchmark_data_wait_direct_action", False),
            False,
        ),
        "log_with": str(_cfg(config, "log_with", "") or ""),
        "tensorboard_flush_interval_steps": _safe_int(_cfg(config, "tensorboard_flush_interval_steps", 10), 10),
        "adaptive_step_logging_enabled": _safe_bool(_cfg(config, "adaptive_step_logging_enabled", True), True),
        "adaptive_step_logging_max_interval": _safe_int(_cfg(config, "adaptive_step_logging_max_interval", 64), 64),
        "layer_monitor_enabled": _safe_bool(_cfg(config, "layer_monitor_enabled", True), True),
        "layer_monitor_interval": _safe_int(_cfg(config, "layer_monitor_interval", 3), 3),
        "save_every_n_steps": _safe_int(_cfg(config, "save_every_n_steps", 0), 0),
        "save_every_n_epochs": _safe_int(_cfg(config, "save_every_n_epochs", 1), 1),
        "eval_every_n_steps": _safe_int(_cfg(config, "eval_every_n_steps", 0), 0),
        "validation_every_n_epochs": _safe_int(_cfg(config, "validation_every_n_epochs", 1), 1),
        "sample_every": _safe_int(_cfg(config, "sample_every", 0), 0),
        "sample_every_n_epochs": _safe_int(_cfg(config, "sample_every_n_epochs", 0), 0),
        "optimizer_backend": str(_cfg(config, "optimizer_backend", "auto") or "auto"),
        "optimizer_args": str(_cfg(config, "optimizer_args", "") or ""),
        "offload_active": offload_active,
        "residency_source": residency_source,
        "residency_mode": mode,
        "prefetch_key": prefetch_key,
        "prefetch_depth_key": prefetch_depth_key,
        "prefetch_enabled": _safe_bool(residency.get("prefetch_enabled", prefetch.get("enabled")), False),
        "prefetch_depth": _safe_int(residency.get("prefetch_depth", prefetch.get("depth")), 0),
        "prefetch_submitted": _safe_int(prefetch.get("submitted")),
        "prefetch_consumed": _safe_int(prefetch.get("consumed")),
        "prefetch_missed": _safe_int(prefetch.get("missed")),
        "residency_transfer_h2d_mb": _round(residency.get("transfer_h2d_mb"), 3),
        "dataloader_rebuild_readiness": dataloader_rebuild_readiness,
    }


def build_bubble_runtime_snapshot(
    config: Any = None,
    runtime_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    features = _mapping(runtime_features)
    gpu = _extract_gpu(features)
    memory_ratio = 0.0
    if gpu.get("memory_total_mb"):
        memory_ratio = _safe_float(gpu.get("memory_used_mb_max")) / max(_safe_float(gpu.get("memory_total_mb")), 1.0)

    return {
        "schema_version": 1,
        "snapshot": "bubble_runtime_snapshot_v0",
        "config": {
            "family": str(_cfg(config, "model_arch", _cfg(config, "model_type", "")) or ""),
            "controller_enabled": _safe_bool(_cfg(config, "bubble_controller_enabled", False), False),
            "controller_mode": str(_cfg(config, "bubble_controller_mode", "report_only") or "report_only"),
            "target_active_gpu_util": _round(_cfg(config, "bubble_controller_target_active_gpu_util", 0.90)),
            "target_saturated_ratio": _round(_cfg(config, "bubble_controller_target_saturated_ratio", 0.50)),
            "min_throughput_gain": _round(_cfg(config, "bubble_controller_min_throughput_gain", 0.03)),
            "warmup_steps": _safe_int(_cfg(config, "bubble_controller_warmup_steps", 8), 8),
            "tune_interval_steps": _safe_int(_cfg(config, "bubble_controller_tune_interval_steps", 32), 32),
            "max_actions_per_run": _safe_int(_cfg(config, "bubble_controller_max_actions_per_run", 3), 3),
            "max_vram_ratio": _round(_cfg(config, "bubble_controller_max_vram_ratio", 0.92)),
            "allow_worker_tuning": _safe_bool(_cfg(config, "bubble_controller_allow_worker_tuning", True), True),
            "allow_batch_growth": _safe_bool(_cfg(config, "bubble_controller_allow_batch_growth", True), True),
            "allow_transfer_prefetch": _safe_bool(_cfg(config, "bubble_controller_allow_transfer_prefetch", True), True),
            "allow_optimizer_swap": _safe_bool(_cfg(config, "bubble_controller_allow_optimizer_swap", False), False),
            "allow_checkpoint_async": _safe_bool(_cfg(config, "bubble_controller_allow_checkpoint_async", True), True),
            "allow_dataloader_rebuild_current_run": _safe_bool(
                _cfg(config, "bubble_controller_allow_dataloader_rebuild_current_run", False),
                False,
            ),
            "cross_run_cooldown_runs": _safe_int(_cfg(config, "bubble_closed_loop_cross_run_cooldown_runs", 1), 1),
            "cross_run_action_history": _mapping_list(_cfg(config, "bubble_closed_loop_action_history", [])),
        },
        "step_phase": _extract_step_phase(features),
        "gpu": gpu,
        "runtime": _extract_runtime(config, features),
        "safety": {
            "memory_ratio": _round(memory_ratio),
            "max_vram_ratio": _round(_cfg(config, "bubble_controller_max_vram_ratio", 0.92)),
            "vram_safe": memory_ratio <= _safe_float(_cfg(config, "bubble_controller_max_vram_ratio", 0.92), 0.92)
            if memory_ratio > 0.0 else True,
        },
    }


__all__ = ["build_bubble_runtime_snapshot"]
