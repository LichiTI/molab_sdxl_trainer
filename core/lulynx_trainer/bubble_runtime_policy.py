"""Conservative report-only policy for bubble-aware runtime tuning."""

from __future__ import annotations

import sys
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


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _action(kind: str, reason: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "reason": reason,
        "apply_mode": "report_only",
    }
    payload.update(fields)
    return payload


def _recent_data_supply_failure(config: Mapping[str, Any]) -> dict[str, Any]:
    history = config.get("cross_run_action_history")
    items = history if isinstance(history, list) else []
    for raw in reversed(items[-8:]):
        item = _mapping(raw)
        action_kind = str(item.get("action_kind") or item.get("kind") or "")
        status = str(item.get("status") or "")
        domain = str(item.get("domain") or "")
        if action_kind in {"set_dataloader_workers", "set_dataloader_prefetch_factor"} or domain == "data_supply":
            if status in {"rolled_back", "rollback_observed", "rollback_recommended", "failed"}:
                return {
                    "action_kind": action_kind,
                    "status": status,
                    "reason": str(item.get("reason") or item.get("decision") or "recent data-supply rollback"),
                }
    return {}


def _data_tuning_profile(
    runtime: Mapping[str, Any],
    step: Mapping[str, Any],
    gpu: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    data_wait = _safe_float(step.get("data_wait_share"))
    active_gpu = _safe_float(gpu.get("active_gpu_util_pct_mean"))
    workers = _safe_int(runtime.get("workers"))
    prefetch = max(_safe_int(runtime.get("prefetch_factor"), 2), 1)
    window_steps = _safe_int(step.get("window_step_count"))
    tune_interval = max(_safe_int(config.get("tune_interval_steps"), 32), 1)
    recent_failure = _recent_data_supply_failure(config)
    reasons: list[str] = []
    if data_wait >= 0.20:
        level = "aggressive_probe"
        reasons.append("data_wait_high")
    elif data_wait >= 0.08:
        level = "standard_probe"
        reasons.append("data_wait_over_threshold")
    else:
        level = "observe"
        reasons.append("data_wait_below_threshold")
    if active_gpu > 0.0 and active_gpu < 55.0:
        reasons.append("gpu_underfed")
    if 0 < window_steps < tune_interval:
        reasons.append("short_evidence_window")
        if level == "aggressive_probe":
            level = "standard_probe"
    if recent_failure:
        reasons.append("recent_data_supply_failure")
        level = "hold"
    return {
        "profile": level,
        "reason_codes": reasons,
        "data_wait_share": _round(data_wait),
        "active_gpu_util_pct_mean": _round(active_gpu, 4),
        "workers": workers,
        "prefetch_factor": prefetch,
        "window_step_count": window_steps,
        "tune_interval_steps": tune_interval,
        "recent_failure": recent_failure,
    }


def _data_action(
    runtime: Mapping[str, Any],
    safety: Mapping[str, Any],
    *,
    step: Mapping[str, Any],
    gpu: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    workers = _safe_int(runtime.get("workers"))
    prefetch = max(_safe_int(runtime.get("prefetch_factor"), 2), 1)
    is_windows = sys.platform.startswith("win")
    profile = _data_tuning_profile(runtime, step, gpu, config)
    reason_prefix = (
        "data_wait remains near threshold after sync-profiler probe; "
        if _safe_bool(runtime.get("sync_profiler_data_wait_retry"), False)
        else ""
    )
    if profile["profile"] == "hold":
        return _action(
            "recommend_cache_first",
            "recent data-supply tuning rolled back; keep worker policy stable and collect a stronger cache/data axis",
            current_workers=workers,
            current_prefetch_factor=prefetch,
            tuning_profile=profile,
            vram_safe=_safe_bool(safety.get("vram_safe"), True),
        )
    if workers <= 0:
        return _action(
            "set_dataloader_workers",
            f"{reason_prefix}workers2 is the first conservative candidate",
            current=workers,
            recommended=2,
            tuning_profile=profile,
        )
    if workers == 2 and not is_windows and prefetch < 4:
        return _action(
            "set_dataloader_prefetch_factor",
            "data_wait still visible after workers2; non-Windows can test prefetch4",
            current=prefetch,
            recommended=4,
            tuning_profile=profile,
        )
    if workers >= 2 and is_windows:
        return _action(
            "keep_dataloader_workers",
            "Windows workers4 showed negative evidence; keep workers2 unless local A/B proves otherwise",
            current=workers,
            current_prefetch_factor=prefetch,
            tuning_profile=profile,
        )
    return _action(
        "recommend_cache_first",
        "data_wait persists after worker tuning; prefer cache-first or dataset preprocessing",
        current_workers=workers,
        current_prefetch_factor=prefetch,
        tuning_profile=profile,
        vram_safe=_safe_bool(safety.get("vram_safe"), True),
    )


def _transfer_action(runtime: Mapping[str, Any], step: Mapping[str, Any]) -> dict[str, Any]:
    offload_active = _safe_bool(runtime.get("offload_active"))
    prefetch_enabled = _safe_bool(runtime.get("prefetch_enabled"))
    missed = _safe_int(runtime.get("prefetch_missed"))
    depth = max(_safe_int(runtime.get("prefetch_depth"), 1), 1)
    if offload_active and not prefetch_enabled:
        return _action(
            "enable_block_prefetch",
            "streaming/offload is active and transfer evidence is visible; test prefetch depth1",
            current=False,
            recommended=True,
            depth=1,
        )
    if prefetch_enabled and missed > 0:
        return _action(
            "increase_block_prefetch_depth",
            "prefetch miss is visible; test one deeper prefetch window",
            current=depth,
            recommended=min(depth + 1, 2),
            missed=missed,
        )
    if not _safe_bool(runtime.get("pin_memory"), True):
        return _action("enable_pin_memory", "H2D transfer is visible and pin_memory is disabled", recommended=True)
    if not _safe_bool(runtime.get("data_transfer_non_blocking"), True):
        return _action(
            "enable_non_blocking_transfer",
            "H2D transfer is visible and non_blocking transfer is disabled",
            recommended=True,
        )
    return _action(
        "profile_transfer_path",
        "transfer_bound remains after basic guards; collect profiler/Nsight evidence before deeper changes",
        h2d_transfer_share=_round(step.get("h2d_transfer_share")),
    )


def _optimizer_action(runtime: Mapping[str, Any]) -> dict[str, Any]:
    optimizer_args = str(runtime.get("optimizer_args", "") or "").lower()
    optimizer_backend = str(runtime.get("optimizer_backend", "auto") or "auto").strip().lower()
    if optimizer_backend not in {"torch_fused", "lulynx_fused"} and "fused=true" not in optimizer_args:
        return _action(
            "enable_fused_adamw",
            "optimizer_update_share >= 15%; test torch fused AdamW before native optimizer promotion",
            current_backend=runtime.get("optimizer_backend", "auto"),
            current_args=runtime.get("optimizer_args", ""),
            recommended_backend="torch_fused",
            recommended="fused=True",
        )
    return _action(
        "profile_native_optimizer",
        "optimizer is still hot after fused AdamW; use native optimizer gates before promotion",
    )


def _workload_action(runtime: Mapping[str, Any], safety: Mapping[str, Any]) -> dict[str, Any]:
    batch = max(_safe_int(runtime.get("train_batch_size"), 1), 1)
    if _safe_bool(safety.get("vram_safe"), True):
        return _action(
            "increase_train_batch_size",
            "GPU active util is low while data/transfer/optimizer are not primary; increase effective work first",
            current=batch,
            recommended=batch * 2,
        )
    return _action(
        "explain_workload_underfilled",
        "GPU util is low because workload is light, but VRAM headroom is not safe for automatic growth",
        current_batch=batch,
    )


def _top_phase_labels(step: Mapping[str, Any]) -> list[str]:
    phases = step.get("top_phases")
    if not isinstance(phases, list):
        return []
    labels: list[str] = []
    for item in phases:
        if isinstance(item, Mapping):
            labels.append(str(item.get("label") or "").lower())
    return labels


def _host_action(runtime: Mapping[str, Any], step: Mapping[str, Any]) -> dict[str, Any]:
    transfer_profile_mode = str(runtime.get("data_transfer_profile_mode", "event") or "event").strip().lower()
    if _safe_bool(runtime.get("step_phase_profile_enabled"), False) or transfer_profile_mode == "sync":
        return _action(
            "disable_sync_profiler_mode",
            "sync profiler mode is enabled; keep it for probes, not normal throughput windows",
            step_phase_profile_enabled=_safe_bool(runtime.get("step_phase_profile_enabled"), False),
            data_transfer_profile_mode=transfer_profile_mode,
        )

    labels = _top_phase_labels(step)
    if any("checkpoint" in label or "save" in label for label in labels):
        return _action(
            "increase_checkpoint_interval",
            "checkpoint/save phase is visible in the hot path; increase interval before async runtime work",
            save_every_n_steps=_safe_int(runtime.get("save_every_n_steps")),
            save_every_n_epochs=_safe_int(runtime.get("save_every_n_epochs"), 1),
        )
    if any("validation" in label or "sample" in label for label in labels):
        return _action(
            "move_validation_after_training_window",
            "validation/sample work is visible in the hot path; move it to an epoch/end window",
            eval_every_n_steps=_safe_int(runtime.get("eval_every_n_steps")),
            sample_every=_safe_int(runtime.get("sample_every")),
        )
    if any("log" in label or "callback" in label or "safeguard" in label for label in labels) or _safe_float(
        step.get("logging_checkpoint_share")
    ) >= 0.05:
        return _action(
            "increase_logging_interval",
            "logging/callback/SafeGuard work is visible in the hot path; reduce its cadence",
            logging_checkpoint_share=_round(step.get("logging_checkpoint_share")),
        )
    if 0 < _safe_int(runtime.get("save_every_n_steps")) <= 100:
        return _action(
            "increase_checkpoint_interval",
            "step checkpoint interval is short for a throughput-sensitive run",
            save_every_n_steps=_safe_int(runtime.get("save_every_n_steps")),
        )
    if _safe_int(runtime.get("eval_every_n_steps")) > 0:
        return _action(
            "move_validation_after_training_window",
            "step validation is configured and can create GPU idle boundaries",
            eval_every_n_steps=_safe_int(runtime.get("eval_every_n_steps")),
        )
    return _action(
        "reduce_hot_path_sync",
        "host_gap_share >= 12%; inspect logging, callbacks, profiler sync and checkpoint boundary work",
        host_gap_share=_round(step.get("host_gap_share")),
    )


def _sync_profiler_enabled(runtime: Mapping[str, Any]) -> bool:
    transfer_profile_mode = str(runtime.get("data_transfer_profile_mode", "event") or "event").strip().lower()
    return _safe_bool(runtime.get("step_phase_profile_enabled"), False) or transfer_profile_mode == "sync"


def _benchmark_data_wait_probe_enabled(runtime: Mapping[str, Any]) -> bool:
    return _safe_float(runtime.get("benchmark_data_wait_stall_ms")) > 0.0 and _safe_bool(
        runtime.get("benchmark_data_wait_direct_action"),
        False,
    )


def _sync_profiler_data_wait_retry(runtime: Mapping[str, Any], data_wait: float) -> bool:
    return _safe_bool(runtime.get("sync_profiler_data_wait_retry"), False) and data_wait >= 0.075


def classify_bubble(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    config = _mapping(snapshot.get("config"))
    step = _mapping(snapshot.get("step_phase"))
    gpu = _mapping(snapshot.get("gpu"))
    runtime = _mapping(snapshot.get("runtime"))
    safety = _mapping(snapshot.get("safety"))

    data_wait = _safe_float(step.get("data_wait_share"))
    h2d = _safe_float(step.get("h2d_transfer_share"))
    optimizer = _safe_float(step.get("optimizer_share"))
    host_gap = _safe_float(step.get("host_gap_share"))
    logging_checkpoint = _safe_float(step.get("logging_checkpoint_share"))
    train_step = _safe_float(step.get("train_step_share"))
    active_gpu = _safe_float(gpu.get("active_gpu_util_pct_mean"))
    saturated = _safe_float(gpu.get("active_gpu_saturated_sample_ratio"))
    dominant = str(step.get("dominant_bottleneck", "unknown") or "unknown")
    controller_mode = str(config.get("controller_mode", "report_only") or "report_only").strip().lower()
    data_wait_retry = _sync_profiler_data_wait_retry(runtime, data_wait)

    if (
        controller_mode in {"advisor_patch", "auto_apply"}
        and _sync_profiler_enabled(runtime)
        and not (_benchmark_data_wait_probe_enabled(runtime) and data_wait >= 0.08)
        and not data_wait_retry
    ):
        kind = "host_scheduling_bound"
        confidence = 0.82
        action = _host_action(runtime, step)
    elif h2d >= 0.05:
        kind = "transfer_bound"
        confidence = 0.88
        action = _transfer_action(runtime, step)
    elif data_wait >= 0.08 or data_wait_retry:
        kind = "data_bound"
        confidence = 0.84
        action = _data_action(runtime, safety, step=step, gpu=gpu, config=config)
    elif optimizer >= 0.15:
        kind = "optimizer_bound"
        confidence = 0.80
        action = _optimizer_action(runtime)
    elif logging_checkpoint >= 0.05 or host_gap >= 0.12:
        kind = "host_scheduling_bound"
        confidence = 0.76
        action = _host_action(runtime, step)
    elif active_gpu >= 85.0 or saturated >= 0.60:
        kind = "gpu_saturated"
        confidence = 0.90
        action = _action(
            "no_action_monitor_throughput",
            "active GPU window is already saturated; optimize throughput and VRAM, not utilization",
        )
    elif active_gpu > 0.0 and active_gpu < 60.0 and h2d < 0.01 and data_wait < 0.08 and optimizer < 0.15:
        kind = "workload_underfilled"
        confidence = 0.72 if train_step >= 0.50 or dominant == "compute_bound" else 0.62
        action = _workload_action(runtime, safety)
    elif dominant == "compute_bound":
        kind = "compute_bound"
        confidence = 0.66
        action = _action(
            "profile_compute_kernel",
            "step phase is compute-bound; compare attention, compile and static-shape options",
        )
    else:
        kind = "unknown"
        confidence = 0.35
        action = _action(
            "collect_more_evidence",
            "enable step_phase_profile and GPU telemetry window sampling for a short probe",
        )

    return {
        "schema_version": 1,
        "diagnosis": "bubble_diagnosis_v0",
        "kind": kind,
        "confidence": round(confidence, 4),
        "dominant_bottleneck": dominant,
        "recommended_action": action,
        "evidence": {
            "data_wait_share": _round(data_wait),
            "h2d_transfer_share": _round(h2d),
            "optimizer_share": _round(optimizer),
            "host_gap_share": _round(host_gap),
            "logging_checkpoint_share": _round(logging_checkpoint),
            "train_step_share": _round(train_step),
            "active_gpu_util_pct_mean": _round(active_gpu, 4),
            "active_gpu_saturated_sample_ratio": _round(saturated),
            "vram_safe": _safe_bool(safety.get("vram_safe"), True),
        },
    }


__all__ = ["classify_bubble"]
