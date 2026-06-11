"""Evaluate SDXL non-DataLoader probe command results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SDXL_NON_DATALOADER_PROBE_RESULTS_REPORT = "bubble_sdxl_non_dataloader_probe_results_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"


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


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _load_json(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, Mapping) else {}


def _run_from_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    runs = _mapping(summary.get("runs"))
    standard = _mapping(runs.get("standard"))
    if standard:
        return standard
    candidates = [item for item in runs.values() if isinstance(item, Mapping)]
    successful = [item for item in candidates if _safe_bool(item.get("success"), False)]
    rows = successful or candidates
    if not rows:
        return {}
    return max(rows, key=lambda item: _safe_float(_mapping(item).get("steady_samples_per_second")))


def _phase_metrics(run: Mapping[str, Any]) -> dict[str, Any]:
    bubble = _mapping(run.get("steady_bubble_profile"))
    evidence = _mapping(bubble.get("evidence"))
    return {
        "phase_profile_available": bool(bubble),
        "dominant_bottleneck": str(bubble.get("dominant_bottleneck") or "unknown"),
        "bubble_ratio_estimate": _round(bubble.get("bubble_ratio_estimate")),
        "data_wait_share": _round(evidence.get("data_wait_share")),
        "h2d_transfer_share": _round(evidence.get("h2d_transfer_share")),
        "optimizer_share": _round(evidence.get("optimizer_share")),
        "host_gap_share": _round(evidence.get("host_gap_share")),
    }


def _runtime_gpu_metrics(run: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(run.get("runtime_feature_summary"))
    windows = _mapping(runtime.get("gpu_telemetry_windows"))
    active20 = _mapping(windows.get("active_window_gpu20"))
    active50 = _mapping(windows.get("active_window_gpu50"))
    gpu = _mapping(runtime.get("gpu"))
    telemetry = _mapping(runtime.get("gpu_telemetry"))
    return {
        "active_gpu_util_pct_mean": _round(
            active20.get(
                "gpu_util_pct_mean",
                gpu.get(
                    "active_gpu_util_pct_mean",
                    telemetry.get("active_gpu_util_pct_mean", run.get("active_gpu_util_pct_mean")),
                ),
            ),
            4,
        ),
        "active_gpu50_util_pct_mean": _round(active50.get("gpu_util_pct_mean"), 4),
        "active_gpu_saturated_sample_ratio": _round(
            active20.get(
                "gpu_saturated_sample_ratio",
                gpu.get("active_gpu_saturated_sample_ratio", telemetry.get("active_gpu_saturated_sample_ratio")),
            )
        ),
        "active_gpu_idle_sample_ratio": _round(
            active20.get(
                "gpu_idle_sample_ratio",
                gpu.get("active_gpu_idle_sample_ratio", telemetry.get("active_gpu_idle_sample_ratio")),
            )
        ),
        "gpu_telemetry_available": bool(active20 or active50 or gpu or telemetry),
    }


def _normalize_summary(summary: Mapping[str, Any], *, command: Mapping[str, Any]) -> dict[str, Any]:
    run = _run_from_summary(summary)
    benchmark = _mapping(summary.get("benchmark"))
    return {
        "source_kind": "native_runtime_profile_benchmark_summary",
        "summary_path": str(command.get("expected_summary") or ""),
        "command_id": str(command.get("id") or ""),
        "role": str(command.get("role") or ""),
        "axis": str(command.get("axis") or ""),
        "status": "success" if _safe_bool(run.get("success"), False) else "failed_or_unknown",
        "success": _safe_bool(run.get("success"), False),
        "steps_completed": _safe_int(run.get("steps_completed")),
        "metrics": {
            "steady_samples_per_second": _round(run.get("steady_samples_per_second")),
            "samples_per_second": _round(run.get("samples_per_second")),
            "steady_mean_step_ms": _round(run.get("steady_mean_step_ms"), 4),
            "peak_vram_mb": _round(run.get("peak_vram_mb"), 4),
            "training_peak_vram_mb": _round(run.get("training_peak_vram_mb"), 4),
            "final_loss": _round(run.get("final_loss"), 6),
            **_phase_metrics(run),
            **_runtime_gpu_metrics(run),
        },
        "config": {
            "family": str(benchmark.get("family") or "sdxl"),
            "resolution": _safe_int(benchmark.get("resolution")),
            "train_batch_size": _safe_int(benchmark.get("train_batch_size")),
            "dataloader_workers": _safe_int(benchmark.get("dataloader_workers")),
            "dataloader_prefetch_factor": _safe_int(benchmark.get("dataloader_prefetch_factor")),
            "fused_adamw": _safe_bool(benchmark.get("fused_adamw"), False),
            "optimizer_args": str(run.get("optimizer_args") or ""),
            "requested_attention_backend": str(benchmark.get("attention_backend") or ""),
            "actual_attention_backend": str(run.get("attention_backend") or ""),
            "sdpa_backend_policy": str(benchmark.get("sdpa_backend_policy") or ""),
        },
    }


def _metric(row: Mapping[str, Any], name: str) -> float:
    return _safe_float(_mapping(row.get("metrics")).get(name))


def _comparison(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    before_sps = _metric(baseline, "steady_samples_per_second")
    after_sps = _metric(candidate, "steady_samples_per_second")
    gain_ratio = (after_sps / before_sps - 1.0) if before_sps > 0.0 and after_sps > 0.0 else 0.0
    before_vram = _metric(baseline, "peak_vram_mb")
    after_vram = _metric(candidate, "peak_vram_mb")
    before_loss = _metric(baseline, "final_loss")
    after_loss = _metric(candidate, "final_loss")
    phase_delta = {
        "data_wait_share_delta": _round(_metric(candidate, "data_wait_share") - _metric(baseline, "data_wait_share")),
        "h2d_transfer_share_delta": _round(_metric(candidate, "h2d_transfer_share") - _metric(baseline, "h2d_transfer_share")),
        "optimizer_share_delta": _round(_metric(candidate, "optimizer_share") - _metric(baseline, "optimizer_share")),
        "host_gap_share_delta": _round(_metric(candidate, "host_gap_share") - _metric(baseline, "host_gap_share")),
    }
    return {
        "steady_samples_per_second_before": _round(before_sps),
        "steady_samples_per_second_after": _round(after_sps),
        "steady_samples_per_second_delta": _round(after_sps - before_sps),
        "steady_samples_per_second_gain_ratio": _round(gain_ratio),
        "steady_samples_per_second_gain_pct": _round(gain_ratio * 100.0, 4),
        "peak_vram_mb_delta": _round(after_vram - before_vram, 4),
        "peak_vram_growth_ratio": _round(after_vram / before_vram if before_vram > 0.0 and after_vram > 0.0 else 0.0),
        "final_loss_delta": _round(after_loss - before_loss, 6),
        "loss_regression_ratio": _round(after_loss / before_loss - 1.0 if before_loss > 0.0 and after_loss > 0.0 else 0.0),
        "active_gpu_util_pct_before": _round(_metric(baseline, "active_gpu_util_pct_mean"), 4),
        "active_gpu_util_pct_after": _round(_metric(candidate, "active_gpu_util_pct_mean"), 4),
        "active_gpu_util_pct_delta": _round(
            _metric(candidate, "active_gpu_util_pct_mean") - _metric(baseline, "active_gpu_util_pct_mean"),
            4,
        ),
        "active_gpu_saturated_sample_ratio_delta": _round(
            _metric(candidate, "active_gpu_saturated_sample_ratio")
            - _metric(baseline, "active_gpu_saturated_sample_ratio")
        ),
        **phase_delta,
        "phase_delta_summary": phase_delta,
    }


def _decision(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    min_throughput_gain: float,
    rollback_max_regression_ratio: float,
    max_loss_regression_ratio: float,
    max_peak_vram_growth_ratio: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    before_sps = _metric(baseline, "steady_samples_per_second")
    after_sps = _metric(candidate, "steady_samples_per_second")
    gain = _safe_float(comparison.get("steady_samples_per_second_gain_ratio"))
    loss_regression = _safe_float(comparison.get("loss_regression_ratio"))
    vram_growth = _safe_float(comparison.get("peak_vram_growth_ratio"))
    baseline_config = _mapping(baseline.get("config"))
    candidate_config = _mapping(candidate.get("config"))
    candidate_axis = str(candidate.get("axis") or "")
    phase_available = bool(_mapping(baseline.get("metrics")).get("phase_profile_available")) and bool(
        _mapping(candidate.get("metrics")).get("phase_profile_available")
    )

    if not _safe_bool(baseline.get("success"), False) or not _safe_bool(candidate.get("success"), False):
        reasons.append("summary_run_failed_or_missing_success")
        return {"status": "insufficient_evidence", "recommended_action": "collect_more_evidence", "reasons": reasons}
    if before_sps <= 0.0 or after_sps <= 0.0:
        reasons.append("missing_throughput_evidence")
        return {"status": "insufficient_evidence", "recommended_action": "collect_more_evidence", "reasons": reasons}
    if candidate_axis == "attention_backend":
        baseline_requested = str(baseline_config.get("requested_attention_backend") or "")
        baseline_actual = str(baseline_config.get("actual_attention_backend") or "")
        candidate_requested = str(candidate_config.get("requested_attention_backend") or "")
        candidate_actual = str(candidate_config.get("actual_attention_backend") or "")
        if baseline_requested and baseline_requested != "auto" and baseline_actual != baseline_requested:
            reasons.append("attention_backend_baseline_not_applied_or_fell_back")
            return {"status": "needs_review", "recommended_action": "manual_review_backend_fallback", "reasons": reasons}
        if not candidate_actual or candidate_actual == baseline_actual or candidate_actual != candidate_requested:
            reasons.append("attention_backend_candidate_not_applied_or_fell_back")
            return {"status": "needs_review", "recommended_action": "manual_review_backend_fallback", "reasons": reasons}
    if candidate_axis == "sdpa_backend_policy":
        baseline_actual = str(baseline_config.get("actual_attention_backend") or "")
        candidate_actual = str(candidate_config.get("actual_attention_backend") or "")
        baseline_policy = str(baseline_config.get("sdpa_backend_policy") or "")
        candidate_policy = str(candidate_config.get("sdpa_backend_policy") or "")
        if baseline_actual != "sdpa" or candidate_actual != "sdpa":
            reasons.append("sdpa_policy_probe_backend_not_sdpa_or_fell_back")
            return {"status": "needs_review", "recommended_action": "manual_review_backend_fallback", "reasons": reasons}
        if not candidate_policy or candidate_policy == baseline_policy:
            reasons.append("sdpa_backend_policy_candidate_not_applied")
            return {"status": "needs_review", "recommended_action": "manual_review_backend_fallback", "reasons": reasons}
    if gain <= -abs(float(rollback_max_regression_ratio)):
        reasons.append("throughput_regressed")
        return {"status": "rollback_recommended", "recommended_action": "do_not_promote_candidate", "reasons": reasons}
    if vram_growth > max(float(max_peak_vram_growth_ratio), 0.0) > 0.0:
        reasons.append("peak_vram_growth_exceeded")
        return {"status": "rollback_recommended", "recommended_action": "do_not_promote_candidate", "reasons": reasons}
    if loss_regression > float(max_loss_regression_ratio):
        reasons.append("loss_regressed")
        return {"status": "needs_review", "recommended_action": "manual_review", "reasons": reasons}
    if not phase_available:
        reasons.append("missing_phase_profile_boundary")
        return {"status": "needs_review", "recommended_action": "manual_review", "reasons": reasons}
    if gain >= float(min_throughput_gain):
        reasons.append("throughput_gain_met")
        return {"status": "keep_candidate_review", "recommended_action": "review_for_promotion", "reasons": reasons}
    reasons.append("throughput_gain_below_threshold")
    return {"status": "needs_review", "recommended_action": "manual_review", "reasons": reasons}


def _summary_path(command: Mapping[str, Any]) -> Path:
    return Path(str(command.get("expected_summary") or ""))


def _commands_by_role(group: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    by_role: dict[str, Mapping[str, Any]] = {}
    for raw in group.get("commands", []):
        item = _mapping(raw)
        role = str(item.get("role") or "")
        if role:
            by_role[role] = item
    return by_role


def _evaluate_group(
    group: Mapping[str, Any],
    *,
    min_throughput_gain: float,
    rollback_max_regression_ratio: float,
    max_loss_regression_ratio: float,
    max_peak_vram_growth_ratio: float,
) -> dict[str, Any]:
    by_role = _commands_by_role(group)
    baseline_cmd = _mapping(by_role.get("baseline"))
    candidate_cmd = _mapping(by_role.get("candidate"))
    missing = [
        str(command.get("expected_summary") or "")
        for command in (baseline_cmd, candidate_cmd)
        if not _summary_path(command).is_file()
    ]
    base = {
        "id": str(group.get("id") or ""),
        "family": "sdxl",
        "track": str(group.get("track") or ""),
        "category": str(group.get("category") or "manual_gpu_probe_pair"),
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "source_group_status": str(group.get("status") or ""),
        "required_gates": list(group.get("required_gates") or []),
    }
    if missing:
        return {
            **base,
            "status": "pending_manual_gpu_commands",
            "missing_summary_count": len(missing),
            "missing_summaries": missing,
            "decision": {
                "status": "pending_manual_gpu_commands",
                "recommended_action": "run_or_reuse_manual_gpu_commands",
                "reasons": ["expected_summary_missing"],
            },
        }

    baseline = _normalize_summary(_load_json(_summary_path(baseline_cmd)), command=baseline_cmd)
    candidate = _normalize_summary(_load_json(_summary_path(candidate_cmd)), command=candidate_cmd)
    comparison = _comparison(baseline, candidate)
    decision = _decision(
        baseline,
        candidate,
        comparison,
        min_throughput_gain=min_throughput_gain,
        rollback_max_regression_ratio=rollback_max_regression_ratio,
        max_loss_regression_ratio=max_loss_regression_ratio,
        max_peak_vram_growth_ratio=max_peak_vram_growth_ratio,
    )
    return {
        **base,
        "status": decision["status"],
        "missing_summary_count": 0,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
        "decision": decision,
    }


def _top_status(groups: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(item.get("status") or "") for item in groups}
    if not groups:
        return "no_probe_groups"
    if "pending_manual_gpu_commands" in statuses:
        return "pending_manual_gpu_commands"
    if "rollback_recommended" in statuses:
        return "rollback_recommended"
    if "keep_candidate_review" in statuses:
        return "keep_candidate_review"
    if "needs_review" in statuses:
        return "needs_review"
    if "insufficient_evidence" in statuses:
        return "insufficient_evidence"
    return "probe_results_ready"


def build_sdxl_non_dataloader_probe_results_report(
    command_plan: Mapping[str, Any],
    *,
    min_throughput_gain: float = 0.03,
    rollback_max_regression_ratio: float = 0.02,
    max_loss_regression_ratio: float = 0.05,
    max_peak_vram_growth_ratio: float = 1.20,
) -> dict[str, Any]:
    """Evaluate supported SDXL probe command groups from their summary JSON files."""

    groups = [
        _evaluate_group(
            _mapping(group),
            min_throughput_gain=max(float(min_throughput_gain), 0.0),
            rollback_max_regression_ratio=max(float(rollback_max_regression_ratio), 0.0),
            max_loss_regression_ratio=max(float(max_loss_regression_ratio), 0.0),
            max_peak_vram_growth_ratio=max(float(max_peak_vram_growth_ratio), 0.0),
        )
        for group in command_plan.get("groups", [])
        if _mapping(group)
    ]
    statuses = sorted({str(group.get("status") or "") for group in groups})
    return {
        "schema_version": 1,
        "report": SDXL_NON_DATALOADER_PROBE_RESULTS_REPORT,
        "roadmap": ROADMAP,
        "status": _top_status(groups),
        "family": "sdxl",
        "source_command_plan_report": str(command_plan.get("report") or ""),
        "source_command_plan_status": str(command_plan.get("status") or ""),
        "not_release_evidence": True,
        "publishable": False,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "thresholds": {
            "min_throughput_gain": _round(min_throughput_gain),
            "rollback_max_regression_ratio": _round(rollback_max_regression_ratio),
            "max_loss_regression_ratio": _round(max_loss_regression_ratio),
            "max_peak_vram_growth_ratio": _round(max_peak_vram_growth_ratio),
        },
        "summary": {
            "group_count": len(groups),
            "pending_group_count": sum(1 for item in groups if item.get("status") == "pending_manual_gpu_commands"),
            "keep_candidate_review_count": sum(1 for item in groups if item.get("status") == "keep_candidate_review"),
            "rollback_recommended_count": sum(1 for item in groups if item.get("status") == "rollback_recommended"),
            "needs_review_count": sum(1 for item in groups if item.get("status") == "needs_review"),
            "status_counts": {status: sum(1 for item in groups if item.get("status") == status) for status in statuses},
        },
        "groups": groups,
        "notes": [
            "This report only evaluates existing summary JSON files; it does not run GPU work.",
            "A keep_candidate_review status is not a release claim and still requires case-specific wording gates.",
            "Missing phase profile keeps otherwise positive results in review instead of automatic promotion.",
        ],
    }


__all__ = [
    "SDXL_NON_DATALOADER_PROBE_RESULTS_REPORT",
    "build_sdxl_non_dataloader_probe_results_report",
]
