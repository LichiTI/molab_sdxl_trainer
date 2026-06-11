"""Before/after evidence for semi-automatic bubble advisor patches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _object_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        try:
            data = value.model_dump(mode="python")
        except TypeError:
            data = value.model_dump()
        return data if isinstance(data, Mapping) else {}
    if hasattr(value, "dict"):
        data = value.dict()
        return data if isinstance(data, Mapping) else {}
    try:
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    except TypeError:
        return {}


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


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _first_float(source: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in source:
            value = _safe_float(source.get(key))
            if value != 0.0:
                return value
    return 0.0


def _first_mapping(*values: Any) -> Mapping[str, Any]:
    for value in values:
        mapped = _mapping(value)
        if mapped:
            return mapped
    return {}


def _successful_runs(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    runs = report.get("run_summaries", [])
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        return []
    rows = [item for item in runs if isinstance(item, Mapping)]
    successful = [item for item in rows if _safe_bool(item.get("success"), False)]
    return successful or rows


def _best_run_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = _successful_runs(report)
    if not rows:
        return {}
    return max(rows, key=lambda item: _safe_float(item.get("steady_samples_per_second")))


def _step_phase_from_features(features: Mapping[str, Any]) -> Mapping[str, Any]:
    loop = _mapping(features.get("training_loop_runtime"))
    summary = _mapping(features.get("runtime_feature_summary"))
    summary_loop = _mapping(summary.get("training_loop_runtime"))
    experiments = _mapping(features.get("anima_full_finetune_experiments"))
    summary_experiments = _mapping(summary.get("anima_full_finetune_experiments"))
    source = _first_mapping(
        _mapping(loop.get("step_phase_profile")),
        _mapping(summary_loop.get("step_phase_profile")),
        _mapping(experiments.get("step_phase_profile")),
        _mapping(summary_experiments.get("step_phase_profile")),
    )
    last = _mapping(source.get("last"))
    return _first_mapping(_mapping(source.get("gpu_bubble_profile")), _mapping(last.get("gpu_bubble_profile")), source)


def _extract_phase_metrics(features: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    bubble = _step_phase_from_features(features)
    evidence = _mapping(bubble.get("evidence"))
    return {
        "dominant_bottleneck": str(
            fallback.get("dominant_bottleneck")
            or bubble.get("dominant_bottleneck")
            or "unknown"
        ),
        "bubble_ratio_estimate": _round(fallback.get("bubble_ratio_estimate", bubble.get("bubble_ratio_estimate"))),
        "data_wait_share": _round(fallback.get("data_wait_share", evidence.get("data_wait_share"))),
        "h2d_transfer_share": _round(fallback.get("h2d_transfer_share", evidence.get("h2d_transfer_share"))),
        "optimizer_share": _round(fallback.get("optimizer_share", evidence.get("optimizer_share"))),
        "host_gap_share": _round(fallback.get("host_gap_share", evidence.get("host_gap_share"))),
        "mean_step_ms": _round(fallback.get("steady_mean_step_ms", bubble.get("mean_step_ms")), 4),
    }


def _extract_gpu_metrics(report: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    windows = _mapping(report.get("gpu_telemetry_windows"))
    active = _mapping(windows.get("active_window_gpu20"))
    full = _mapping(report.get("gpu_telemetry"))
    classification = _mapping(report.get("classification"))
    source = _first_mapping(active, full, classification)
    memory_used = _first_float(source, "memory_used_mb_max")
    if memory_used <= 0.0:
        memory_used = _safe_float(fallback.get("peak_vram_mb"))
    memory_total = _safe_float(source.get("memory_total_mb"))
    memory_ratio = memory_used / max(memory_total, 1.0) if memory_total > 0.0 else 0.0
    return {
        "active_gpu_util_pct_mean": _round(
            source.get("gpu_util_pct_mean", source.get("active_gpu_util_pct_mean")),
            4,
        ),
        "active_gpu_saturated_sample_ratio": _round(
            source.get("gpu_saturated_sample_ratio", source.get("active_gpu_saturated_sample_ratio")),
        ),
        "peak_vram_mb": _round(memory_used, 4),
        "memory_total_mb": _round(memory_total, 4),
        "memory_ratio": _round(memory_ratio),
    }


def _extract_ledger(report: Mapping[str, Any]) -> dict[str, Any]:
    extra = _mapping(report.get("extra"))
    config = _mapping(report.get("config"))
    return dict(
        _first_mapping(
            report.get("bubble_advisor_action_ledger"),
            extra.get("bubble_advisor_action_ledger"),
            config.get("bubble_advisor_action_ledger"),
        )
    )


def _experiment_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _best_run_summary(report)
    benchmark = _mapping(report.get("benchmark"))
    features = _mapping(summary.get("runtime_feature_summary"))
    phase = _extract_phase_metrics(features, summary)
    gpu = _extract_gpu_metrics(report, summary)
    throughput = _first_float(summary, "steady_samples_per_second", "samples_per_second")
    return {
        "source_kind": "gpu_bubble_experiment_report",
        "case_id": _experiment_case_id(benchmark),
        "family": str(benchmark.get("family") or benchmark.get("model_family") or ""),
        "status": "success" if _safe_bool(summary.get("success"), False) else str(report.get("status") or "unknown"),
        "steps_completed": _safe_int(summary.get("steps_completed")),
        "metrics": {
            "steady_samples_per_second": _round(throughput, 6),
            "throughput_estimated": False,
            "final_loss": _round(summary.get("final_loss"), 6),
            **phase,
            **gpu,
        },
        "config": dict(benchmark),
        "action_ledger": _extract_ledger(report),
    }


def _experiment_case_id(benchmark: Mapping[str, Any]) -> str:
    explicit = str(benchmark.get("case") or benchmark.get("case_id") or benchmark.get("name") or "").strip()
    if explicit:
        return explicit
    family = str(benchmark.get("family") or benchmark.get("model_family") or "unknown").strip() or "unknown"
    parts = [family]
    for key, prefix in (
        ("resolution", "res"),
        ("steps", "steps"),
        ("train_batch_size", "batch"),
        ("dataloader_workers", "workers"),
        ("dataloader_prefetch_factor", "prefetch"),
    ):
        value = benchmark.get(key)
        if value not in {None, ""}:
            parts.append(f"{prefix}{value}")
    if _safe_bool(benchmark.get("anima_block_prefetch"), False):
        parts.append(f"block_prefetch{benchmark.get('anima_block_prefetch_depth', '')}")
    elif _safe_bool(benchmark.get("newbie_block_prefetch"), False):
        parts.append(f"block_prefetch{benchmark.get('newbie_block_prefetch_depth', '')}")
    return "_".join(str(part).strip().replace(" ", "_") for part in parts if str(part).strip())


def _manifest_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    config = _object_mapping(report.get("config"))
    extra = _mapping(report.get("extra"))
    controller = _mapping(extra.get("bubble_controller"))
    snapshot = _mapping(controller.get("snapshot"))
    snapshot_step = _mapping(snapshot.get("step_phase"))
    snapshot_gpu = _mapping(snapshot.get("gpu"))
    phase = _extract_phase_metrics(extra, snapshot_step)
    gpu = {
        "active_gpu_util_pct_mean": _round(snapshot_gpu.get("active_gpu_util_pct_mean"), 4),
        "active_gpu_saturated_sample_ratio": _round(snapshot_gpu.get("active_gpu_saturated_sample_ratio")),
        "peak_vram_mb": _round(snapshot_gpu.get("memory_used_mb_max"), 4),
        "memory_total_mb": _round(snapshot_gpu.get("memory_total_mb"), 4),
        "memory_ratio": _round(_mapping(snapshot.get("safety")).get("memory_ratio")),
    }
    throughput = _first_float(extra, "steady_samples_per_second", "samples_per_second")
    throughput_estimated = False
    if throughput <= 0.0 and _safe_float(phase.get("mean_step_ms")) > 0.0:
        batch = max(_safe_float(config.get("train_batch_size"), 1.0), 1.0)
        throughput = batch / max(_safe_float(phase.get("mean_step_ms")) / 1000.0, 1e-9)
        throughput_estimated = True
    return {
        "source_kind": "run_manifest",
        "case_id": str(config.get("output_name") or report.get("checkpoint_path") or "run_manifest"),
        "family": str(config.get("model_arch") or config.get("model_type") or ""),
        "status": str(report.get("status") or "unknown"),
        "steps_completed": _safe_int(report.get("global_step")),
        "metrics": {
            "steady_samples_per_second": _round(throughput, 6),
            "throughput_estimated": throughput_estimated,
            "final_loss": _round(extra.get("final_loss"), 6),
            **phase,
            **gpu,
        },
        "config": dict(config),
        "action_ledger": _extract_ledger(report),
    }


def _source_report_before_evidence(source: Mapping[str, Any], ledger: Mapping[str, Any], after_config: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _mapping(source.get("snapshot"))
    step = _mapping(snapshot.get("step_phase"))
    gpu = _mapping(snapshot.get("gpu"))
    runtime = _mapping(snapshot.get("runtime"))
    safety = _mapping(snapshot.get("safety"))
    metrics = _mapping(source.get("metrics"))
    batch = _safe_float(
        runtime.get("train_batch_size"),
        _safe_float(after_config.get("train_batch_size"), 1.0),
    )
    mean_step_ms = _safe_float(step.get("mean_step_ms"))
    throughput = _first_float(metrics, "steady_samples_per_second", "samples_per_second")
    throughput_estimated = _safe_bool(metrics.get("throughput_estimated"), False)
    if throughput <= 0.0 and batch > 0.0 and mean_step_ms > 0.0:
        throughput = batch / max(mean_step_ms / 1000.0, 1e-9)
        throughput_estimated = True
    family = str(
        runtime.get("family")
        or after_config.get("model_arch")
        or after_config.get("model_type")
        or source.get("family")
        or ""
    )
    action_id = str(ledger.get("action_id") or "")
    return {
        "source_kind": "bubble_action_ledger_source_report",
        "case_id": f"{action_id}:before" if action_id else "advisor_source_report_before",
        "family": family,
        "status": str(source.get("status") or "source_report"),
        "steps_completed": _safe_int(source.get("steps_completed")),
        "steady_samples_per_second": _round(throughput, 6),
        "throughput_estimated": throughput_estimated,
        "final_loss": _round(metrics.get("final_loss"), 6),
        "dominant_bottleneck": str(step.get("dominant_bottleneck") or source.get("diagnosis_kind") or "unknown"),
        "bubble_ratio_estimate": _round(step.get("bubble_ratio_estimate")),
        "data_wait_share": _round(step.get("data_wait_share")),
        "h2d_transfer_share": _round(step.get("h2d_transfer_share")),
        "optimizer_share": _round(step.get("optimizer_share")),
        "host_gap_share": _round(step.get("host_gap_share")),
        "mean_step_ms": _round(mean_step_ms, 4),
        "active_gpu_util_pct_mean": _round(gpu.get("active_gpu_util_pct_mean"), 4),
        "active_gpu_saturated_sample_ratio": _round(gpu.get("active_gpu_saturated_sample_ratio")),
        "peak_vram_mb": _round(gpu.get("memory_used_mb_max"), 4),
        "memory_total_mb": _round(gpu.get("memory_total_mb"), 4),
        "memory_ratio": _round(safety.get("memory_ratio")),
        "config": {
            "train_batch_size": _safe_int(batch, 1),
            "gradient_accumulation_steps": _safe_int(runtime.get("gradient_accumulation_steps"), 1),
            "cached_dataloader_workers": _safe_int(runtime.get("workers")),
            "cached_dataloader_prefetch_factor": _safe_int(runtime.get("prefetch_factor")),
            "pin_memory": runtime.get("pin_memory"),
            "optimizer_backend": runtime.get("optimizer_backend"),
        },
        "action_ledger": {},
    }


def normalize_bubble_ab_run_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one before/after run evidence document."""

    data = _mapping(report)
    if data.get("report") == "gpu_bubble_experiment_report_v0":
        return _experiment_metrics(data)
    if int(data.get("manifest_version", 0) or 0) > 0 or "extra" in data:
        return _manifest_metrics(data)
    return {
        "source_kind": str(data.get("source_kind") or "unknown"),
        "case_id": str(data.get("case_id") or data.get("source_kind") or "unknown"),
        "family": str(data.get("family") or ""),
        "status": str(data.get("status") or "unknown"),
        "steps_completed": _safe_int(data.get("steps_completed")),
        "metrics": {
            "steady_samples_per_second": _round(data.get("steady_samples_per_second"), 6),
            "throughput_estimated": _safe_bool(data.get("throughput_estimated"), False),
            "final_loss": _round(data.get("final_loss"), 6),
            "dominant_bottleneck": str(data.get("dominant_bottleneck") or "unknown"),
            "bubble_ratio_estimate": _round(data.get("bubble_ratio_estimate")),
            "data_wait_share": _round(data.get("data_wait_share")),
            "h2d_transfer_share": _round(data.get("h2d_transfer_share")),
            "optimizer_share": _round(data.get("optimizer_share")),
            "host_gap_share": _round(data.get("host_gap_share")),
            "mean_step_ms": _round(data.get("mean_step_ms"), 4),
            "active_gpu_util_pct_mean": _round(data.get("active_gpu_util_pct_mean"), 4),
            "active_gpu_saturated_sample_ratio": _round(data.get("active_gpu_saturated_sample_ratio")),
            "peak_vram_mb": _round(data.get("peak_vram_mb"), 4),
            "memory_total_mb": _round(data.get("memory_total_mb"), 4),
            "memory_ratio": _round(data.get("memory_ratio")),
        },
        "config": dict(_mapping(data.get("config"))),
        "action_ledger": _extract_ledger(data),
    }


def _before_from_after_manifest(after_manifest: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(after_manifest)
    if not data:
        return {}
    ledger = _extract_ledger(data)
    source = _mapping(ledger.get("source_report"))
    if not source:
        return {}
    return _source_report_before_evidence(source, ledger, _object_mapping(data.get("config")))


def _metric(run: Mapping[str, Any], name: str) -> float:
    return _safe_float(_mapping(run.get("metrics")).get(name))


def _delta(after: Mapping[str, Any], before: Mapping[str, Any], name: str) -> float:
    return _metric(after, name) - _metric(before, name)


def _comparison(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_sps = _metric(before, "steady_samples_per_second")
    after_sps = _metric(after, "steady_samples_per_second")
    gain_ratio = (after_sps / before_sps - 1.0) if before_sps > 0.0 and after_sps > 0.0 else 0.0
    return {
        "steady_samples_per_second_before": _round(before_sps, 6),
        "steady_samples_per_second_after": _round(after_sps, 6),
        "steady_samples_per_second_delta": _round(after_sps - before_sps, 6),
        "steady_samples_per_second_gain_ratio": _round(gain_ratio),
        "steady_samples_per_second_gain_pct": _round(gain_ratio * 100.0, 4),
        "active_gpu_util_pct_delta": _round(_delta(after, before, "active_gpu_util_pct_mean"), 4),
        "data_wait_share_delta": _round(_delta(after, before, "data_wait_share")),
        "h2d_transfer_share_delta": _round(_delta(after, before, "h2d_transfer_share")),
        "optimizer_share_delta": _round(_delta(after, before, "optimizer_share")),
        "host_gap_share_delta": _round(_delta(after, before, "host_gap_share")),
        "peak_vram_mb_delta": _round(_delta(after, before, "peak_vram_mb"), 4),
        "memory_ratio_delta": _round(_delta(after, before, "memory_ratio")),
        "final_loss_delta": _round(_delta(after, before, "final_loss"), 6),
    }


def _decision(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    min_throughput_gain: float,
    rollback_max_regression_ratio: float,
    max_loss_regression_ratio: float,
    max_vram_ratio: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    before_sps = _metric(before, "steady_samples_per_second")
    after_sps = _metric(after, "steady_samples_per_second")
    gain = _safe_float(comparison.get("steady_samples_per_second_gain_ratio"))
    after_loss = _metric(after, "final_loss")
    before_loss = _metric(before, "final_loss")
    loss_ratio = (after_loss / before_loss - 1.0) if before_loss > 0.0 and after_loss > 0.0 else 0.0
    after_memory_ratio = _metric(after, "memory_ratio")
    ledger = _mapping(after.get("action_ledger"))
    rollback_restore = dict(_mapping(_mapping(ledger.get("rollback")).get("restore")))

    if before_sps <= 0.0 or after_sps <= 0.0:
        reasons.append("missing_throughput_evidence")
        action = "collect_more_evidence"
        status = "insufficient_evidence"
    elif gain <= -abs(rollback_max_regression_ratio):
        reasons.append("throughput_regressed")
        action = "rollback"
        status = "rollback_recommended"
    elif max_vram_ratio > 0.0 and after_memory_ratio > max_vram_ratio:
        reasons.append("vram_ratio_exceeded")
        action = "rollback"
        status = "rollback_recommended"
    elif loss_ratio > max_loss_regression_ratio:
        reasons.append("loss_regressed")
        action = "review_or_rollback"
        status = "needs_review"
    elif gain >= min_throughput_gain:
        reasons.append("throughput_gain_met")
        action = "keep"
        status = "keep_recommended"
    else:
        reasons.append("throughput_gain_below_threshold")
        action = "review"
        status = "needs_review"

    if _safe_bool(_mapping(before.get("metrics")).get("throughput_estimated")):
        reasons.append("before_throughput_estimated")
    if _safe_bool(_mapping(after.get("metrics")).get("throughput_estimated")):
        reasons.append("after_throughput_estimated")

    return {
        "status": status,
        "recommended_action": action,
        "reasons": reasons,
        "rollback_overlay": rollback_restore if action in {"rollback", "review_or_rollback"} else {},
        "loss_regression_ratio": _round(loss_ratio),
    }


def build_bubble_advisor_ab_evidence_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    min_throughput_gain: float = 0.03,
    rollback_max_regression_ratio: float = 0.02,
    max_loss_regression_ratio: float = 0.05,
    max_vram_ratio: float = 0.92,
) -> dict[str, Any]:
    """Build a release/audit friendly before-after report for one advisor patch."""

    before_run = normalize_bubble_ab_run_evidence(before)
    after_run = normalize_bubble_ab_run_evidence(after)
    comparison = _comparison(before_run, after_run)
    decision = _decision(
        before_run,
        after_run,
        comparison,
        min_throughput_gain=max(float(min_throughput_gain or 0.0), 0.0),
        rollback_max_regression_ratio=max(float(rollback_max_regression_ratio or 0.0), 0.0),
        max_loss_regression_ratio=max(float(max_loss_regression_ratio or 0.0), 0.0),
        max_vram_ratio=max(float(max_vram_ratio or 0.0), 0.0),
    )
    action = dict(_mapping(after_run.get("action_ledger")))
    if not action:
        action = {"status": "missing", "reason": "after-run evidence has no bubble_advisor_action_ledger"}
    return {
        "schema_version": 1,
        "report": "bubble_advisor_ab_evidence_v0",
        "status": decision["status"],
        "thresholds": {
            "min_throughput_gain": _round(min_throughput_gain),
            "rollback_max_regression_ratio": _round(rollback_max_regression_ratio),
            "max_loss_regression_ratio": _round(max_loss_regression_ratio),
            "max_vram_ratio": _round(max_vram_ratio),
        },
        "action": action,
        "before": before_run,
        "after": after_run,
        "comparison": comparison,
        "decision": decision,
    }


def build_bubble_advisor_ab_evidence_from_run_manifest(
    after_manifest: Mapping[str, Any],
    before: Mapping[str, Any] | None = None,
    *,
    min_throughput_gain: float = 0.03,
    rollback_max_regression_ratio: float = 0.02,
    max_loss_regression_ratio: float = 0.05,
    max_vram_ratio: float = 0.92,
) -> dict[str, Any]:
    """Build A/B evidence from a patched run manifest and its source ledger."""

    inferred_before = _mapping(before) or _before_from_after_manifest(after_manifest)
    report = build_bubble_advisor_ab_evidence_report(
        inferred_before,
        after_manifest,
        min_throughput_gain=min_throughput_gain,
        rollback_max_regression_ratio=rollback_max_regression_ratio,
        max_loss_regression_ratio=max_loss_regression_ratio,
        max_vram_ratio=max_vram_ratio,
    )
    report["auto_pair"] = {
        "source": "explicit_before" if before else "action_ledger.source_report",
        "baseline_found": bool(inferred_before),
        "after_source_kind": normalize_bubble_ab_run_evidence(after_manifest).get("source_kind"),
    }
    if not inferred_before:
        report["decision"]["reasons"].append("missing_action_ledger_source_report")
        report["status"] = "insufficient_evidence"
        report["decision"]["status"] = "insufficient_evidence"
        report["decision"]["recommended_action"] = "collect_more_evidence"
    return report


__all__ = [
    "build_bubble_advisor_ab_evidence_from_run_manifest",
    "build_bubble_advisor_ab_evidence_report",
    "normalize_bubble_ab_run_evidence",
]
