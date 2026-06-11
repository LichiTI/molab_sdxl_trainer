"""Offline replay for bubble-aware runtime controller evidence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from .bubble_runtime_controller import build_bubble_controller_report
from .bubble_runtime_patch_apply import apply_bubble_advisor_patch_to_request


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


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _family_from_report(report: Mapping[str, Any]) -> str:
    benchmark = _mapping(report.get("benchmark"))
    family = str(
        benchmark.get("family")
        or benchmark.get("model_family")
        or benchmark.get("model")
        or benchmark.get("name")
        or ""
    ).strip().lower()
    text = " ".join(str(value).lower() for value in benchmark.values())
    if family:
        return family
    for candidate in ("anima", "newbie", "sdxl", "sd15", "flux"):
        if candidate in text:
            return candidate
    return "sd15"


def _config_for_replay(report: Mapping[str, Any], *, mode: str) -> SimpleNamespace:
    family = _family_from_report(report)
    return SimpleNamespace(
        model_arch=family,
        model_type=family,
        training_type="lora",
        cached_dataloader_workers=0,
        cached_dataloader_prefetch_factor=2,
        cached_dataloader_pin_memory=True,
        data_transfer_non_blocking=True,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        optimizer_backend="auto",
        optimizer_args="",
        step_phase_profile_enabled=False,
        data_transfer_profile_mode="event",
        tensorboard_flush_interval_steps=10,
        adaptive_step_logging_enabled=True,
        layer_monitor_enabled=True,
        layer_monitor_interval=3,
        save_every_n_steps=0,
        save_every_n_epochs=1,
        eval_every_n_steps=0,
        sample_every=0,
        bubble_controller_enabled=True,
        bubble_controller_mode=mode,
        bubble_controller_allow_worker_tuning=True,
        bubble_controller_allow_batch_growth=True,
        bubble_controller_allow_transfer_prefetch=True,
        bubble_controller_allow_optimizer_swap=True,
        bubble_controller_allow_checkpoint_async=True,
        bubble_controller_max_vram_ratio=0.92,
        bubble_controller_min_throughput_gain=0.03,
        bubble_controller_warmup_steps=8,
        bubble_controller_tune_interval_steps=32,
        bubble_controller_max_actions_per_run=3,
    )


def _run_summaries_from_benchmark_payload(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    runs = report.get("runs")
    if not isinstance(runs, Mapping):
        return []
    summaries: list[dict[str, Any]] = []
    for label, run in runs.items():
        if not isinstance(run, Mapping):
            continue
        bubble = _mapping(run.get("steady_bubble_profile"))
        evidence = _mapping(bubble.get("evidence"))
        summary = {
            "label": str(label),
            "success": bool(run.get("success", False)),
            "steps_completed": _safe_int(run.get("steps_completed")),
            "steady_mean_step_ms": _round(run.get("steady_mean_step_ms"), 4),
            "steady_samples_per_second": _round(run.get("steady_samples_per_second"), 6),
            "peak_vram_mb": _round(run.get("peak_vram_mb"), 4),
            "final_loss": _round(run.get("final_loss"), 6),
            "dominant_bottleneck": str(bubble.get("dominant_bottleneck", "unknown") or "unknown"),
            "bubble_ratio_estimate": _round(bubble.get("bubble_ratio_estimate")),
            "data_wait_share": _round(evidence.get("data_wait_share")),
            "h2d_transfer_share": _round(evidence.get("h2d_transfer_share")),
            "optimizer_share": _round(evidence.get("optimizer_share")),
            "host_gap_share": _round(evidence.get("host_gap_share")),
            "logging_checkpoint_share": _round(evidence.get("logging_checkpoint_share")),
        }
        top_phases = evidence.get("top_phases", bubble.get("top_phases"))
        if isinstance(top_phases, list):
            summary["top_phases"] = [dict(item) for item in top_phases if isinstance(item, Mapping)]
        runtime_features = run.get("runtime_feature_summary")
        if isinstance(runtime_features, Mapping):
            summary["runtime_feature_summary"] = dict(runtime_features)
        summaries.append(summary)
    return summaries


def _run_summaries(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = report.get("run_summaries")
    if isinstance(summaries, list):
        return [dict(item) for item in summaries if isinstance(item, Mapping)]
    return _run_summaries_from_benchmark_payload(report)


def _active_gpu_window(report: Mapping[str, Any]) -> dict[str, Any]:
    windows = _mapping(report.get("gpu_telemetry_windows"))
    active = _mapping(windows.get("active_window_gpu20"))
    if active:
        return dict(active)
    classification = _mapping(report.get("classification"))
    telemetry = _mapping(report.get("gpu_telemetry"))
    return {
        "scope": str(classification.get("active_gpu_scope", "replay_inferred") or "replay_inferred"),
        "gpu_util_pct_mean": _safe_float(
            classification.get("active_gpu_util_pct_mean"),
            _safe_float(telemetry.get("gpu_util_pct_mean"), 0.0),
        ),
        "gpu_saturated_sample_ratio": _safe_float(classification.get("active_gpu_saturated_sample_ratio")),
        "memory_used_mb_max": _safe_float(telemetry.get("memory_used_mb_max"), _safe_float(telemetry.get("memory_used_mb"))),
        "memory_total_mb": _safe_float(telemetry.get("memory_total_mb"), 16000.0),
    }


def _runtime_features_for_run(report: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    evidence = {
        "data_wait_share": _round(run.get("data_wait_share")),
        "h2d_transfer_share": _round(run.get("h2d_transfer_share")),
        "optimizer_share": _round(run.get("optimizer_share")),
        "host_gap_share": _round(run.get("host_gap_share")),
        "logging_checkpoint_share": _round(run.get("logging_checkpoint_share")),
        "train_step_share": max(
            0.0,
            _round(
                1.0
                - _safe_float(run.get("data_wait_share"))
                - _safe_float(run.get("h2d_transfer_share"))
                - _safe_float(run.get("optimizer_share"))
                - _safe_float(run.get("host_gap_share"))
                - _safe_float(run.get("logging_checkpoint_share"))
            ),
        ),
    }
    top_phases = run.get("top_phases", [])
    if not isinstance(top_phases, list):
        top_phases = []
    phase_share: dict[str, Any] = {"train_step_total": evidence["train_step_share"]}
    phase_mean_ms: dict[str, Any] = {"train_step_total": _round(run.get("steady_mean_step_ms"), 4)}
    if evidence["logging_checkpoint_share"] > 0.0:
        phase_share["logging_checkpoint"] = evidence["logging_checkpoint_share"]
        phase_mean_ms["logging_checkpoint"] = _round(
            _safe_float(run.get("steady_mean_step_ms")) * evidence["logging_checkpoint_share"],
            4,
        )
    features: dict[str, Any] = {
        "training_loop_runtime": {
            "step_phase_profile": {
                "gpu_bubble_profile": {
                    "profile": "step_phase_bubble_profile_v0",
                    "dominant_bottleneck": str(run.get("dominant_bottleneck", "unknown") or "unknown"),
                    "bubble_ratio_estimate": _round(run.get("bubble_ratio_estimate")),
                    "mean_step_ms": _round(run.get("steady_mean_step_ms"), 4),
                    "evidence": evidence,
                    "phase_share": phase_share,
                    "phase_mean_ms": phase_mean_ms,
                    "top_phases": [dict(item) for item in top_phases if isinstance(item, Mapping)][:8],
                }
            },
            "data_transfer_profile": {
                "last": {
                    "step_share": evidence["h2d_transfer_share"],
                    "mib": 0.0,
                    "ops": 0,
                }
            },
        },
        "gpu_telemetry_windows": {
            "active_window_gpu20": _active_gpu_window(report),
        },
    }
    runtime_feature_summary = _mapping(run.get("runtime_feature_summary"))
    for key in ("anima_block_residency", "newbie_block_residency"):
        value = _mapping(runtime_feature_summary.get(key))
        if value:
            features[key] = dict(value)
    return features


def _expected_kind(report: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    classification = str(_mapping(report.get("classification")).get("status") or "").strip().lower()
    dominant = str(run.get("dominant_bottleneck", "unknown") or "unknown").strip().lower()
    active_gpu = _safe_float(_active_gpu_window(report).get("gpu_util_pct_mean"))
    data_wait = _safe_float(run.get("data_wait_share"))
    h2d = _safe_float(run.get("h2d_transfer_share"))
    optimizer = _safe_float(run.get("optimizer_share"))
    host_gap = _safe_float(run.get("host_gap_share"))
    logging_checkpoint = _safe_float(run.get("logging_checkpoint_share"))
    if classification == "gpu_saturated":
        return "gpu_saturated"
    if h2d >= 0.05 or classification == "transfer_bound":
        return "transfer_bound"
    if data_wait >= 0.08 or classification == "data_bound":
        return "data_bound"
    if optimizer >= 0.15:
        return "optimizer_bound"
    if host_gap >= 0.12 or logging_checkpoint >= 0.05:
        return "host_scheduling_bound"
    if (
        active_gpu > 0.0
        and active_gpu < 60.0
        and data_wait < 0.08
        and h2d < 0.01
        and optimizer < 0.15
        and host_gap < 0.12
        and logging_checkpoint < 0.05
    ):
        return "workload_underfilled"
    if dominant in {"data_bound", "transfer_bound", "optimizer_bound", "host_scheduling_bound", "compute_bound"}:
        return dominant
    return classification or dominant or "unknown"


def _base_request_from_config(config: SimpleNamespace) -> dict[str, Any]:
    keys = (
        "cached_dataloader_workers",
        "cached_dataloader_prefetch_factor",
        "cached_dataloader_pin_memory",
        "data_transfer_non_blocking",
        "train_batch_size",
        "gradient_accumulation_steps",
        "optimizer_backend",
        "step_phase_profile_enabled",
        "data_transfer_profile_mode",
        "tensorboard_flush_interval_steps",
        "layer_monitor_interval",
        "save_every_n_steps",
        "save_every_n_epochs",
        "eval_every_n_steps",
    )
    return {key: getattr(config, key) for key in keys if hasattr(config, key)}


def build_bubble_replay_report(
    reports: Iterable[Mapping[str, Any]],
    *,
    mode: str = "advisor_patch",
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for report_index, report in enumerate(reports):
        mapped_report = _mapping(report)
        config = _config_for_replay(mapped_report, mode=mode)
        for run_index, run in enumerate(_run_summaries(mapped_report)):
            if not bool(run.get("success", True)):
                continue
            controller_report = build_bubble_controller_report(
                config,
                runtime_features=_runtime_features_for_run(mapped_report, run),
            )
            expected = _expected_kind(mapped_report, run)
            actual = str(_mapping(controller_report.get("diagnosis")).get("kind") or "unknown")
            patch_result = apply_bubble_advisor_patch_to_request(
                _base_request_from_config(config),
                controller_report,
            )
            cases.append(
                {
                    "schema_version": 1,
                    "case": "bubble_replay_case_v0",
                    "source_report_index": report_index,
                    "source_run_index": run_index,
                    "run_label": str(run.get("label", f"run_{run_index}") or f"run_{run_index}"),
                    "expected_kind": expected,
                    "actual_kind": actual,
                    "matched": expected in {"", "mixed", "unknown"} or actual == expected,
                    "controller_status": str(controller_report.get("status") or ""),
                    "action_kind": str(
                        _mapping(_mapping(controller_report.get("diagnosis")).get("recommended_action")).get("kind")
                        or ""
                    ),
                    "action_plan_status": str(_mapping(controller_report.get("action_plan")).get("status") or ""),
                    "config_overlay": dict(_mapping(_mapping(controller_report.get("action_plan")).get("config_overlay"))),
                    "patch_apply_status": str(patch_result.get("status") or ""),
                    "patchable": patch_result.get("status") in {"applied", "no_change"},
                    "applied_overlay": dict(_mapping(patch_result.get("applied_overlay"))),
                    "blocked_reasons": list(patch_result.get("blocked_reasons") or []),
                }
            )
    total = len(cases)
    matched = sum(1 for case in cases if bool(case.get("matched")))
    patchable = sum(1 for case in cases if case.get("patch_apply_status") in {"applied", "no_change"})
    return {
        "schema_version": 1,
        "report": "bubble_runtime_replay_report_v0",
        "mode": mode,
        "case_count": total,
        "matched_count": matched,
        "match_rate": _round(matched / max(total, 1), 6),
        "patchable_count": patchable,
        "patchable_rate": _round(patchable / max(total, 1), 6),
        "cases": cases,
        "status": "ok" if total > 0 and matched == total else "needs_review",
    }


__all__ = ["build_bubble_replay_report"]
