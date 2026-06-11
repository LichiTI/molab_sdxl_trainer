"""Release-claim evidence builder for bubble-aware runtime reports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Iterable, Mapping

from .bubble_natural_release_guard import (
    DEFAULT_DATA_WAIT_THRESHOLD,
    natural_ab_release_reasons,
    natural_data_wait_release_reasons,
)
from .bubble_runtime_release_provenance import build_release_evidence_provenance


ROADMAP = "gpu_bubble_elimination_roadmap.md"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


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


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _pct(value: float) -> float:
    return round(float(value or 0.0), 2)


def _max_float(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    return max((_safe_float(row.get(key)) for row in rows), default=0.0)


def _first_float(source: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in source:
            return _safe_float(source.get(key))
    return 0.0


def _loss_delta(report: Mapping[str, Any]) -> float | None:
    comparison = _mapping(report.get("comparison"))
    for key in ("loss_delta", "final_loss_delta", "relative_loss_delta"):
        if key in comparison:
            return _safe_float(comparison.get(key))
    return None


def _search_text(*parts: Any) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        else:
            try:
                text_parts.append(json.dumps(part, sort_keys=True))
            except Exception:
                text_parts.append(str(part))
    return " ".join(text_parts).lower()


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _iter_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _iter_mappings(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _iter_mappings(nested)


def _nested_values(source: Any, keys: set[str]) -> list[Any]:
    values: list[Any] = []
    for item in _iter_mappings(source):
        for key, value in item.items():
            if _normalized_text(key) in keys:
                values.append(value)
    return values


def _label_text(source: Any) -> str:
    label_keys = {"case", "case_id", "name", "description", "label", "title"}
    return _search_text(_nested_values(source, label_keys))


def _has_true_probe_flag(source: Any) -> bool:
    for item in _iter_mappings(source):
        for key, value in item.items():
            if _normalized_text(key).endswith("_probe") and _safe_bool(value, False):
                return True
    return False


def _release_probe_only_reasons(item: Mapping[str, Any]) -> list[str]:
    """Return reasons this evidence must stay out of performance claims."""
    benchmark = _mapping(item.get("benchmark"))
    probe_context = {
        "benchmark": benchmark,
        "release_probe_context": _mapping(item.get("release_probe_context")),
    }
    reasons: list[str] = []
    for key in ("native_cache_mode", "anima_cache_mode"):
        for value in _nested_values(probe_context, {key}):
            mode = _normalized_text(value)
            if mode in {"online_cache", "rebuild_cache"}:
                reasons.append(f"{key}_{mode}_probe_only")

    if any(_normalized_text(value) == "missing_at_start" for value in _nested_values(probe_context, {"cache_state"})):
        reasons.append("cache_state_missing_at_start_probe_only")

    text = _search_text(item.get("case_id"), item.get("family"), _label_text(probe_context))
    probe_markers = (
        "cache_miss",
        "cache-miss",
        "missing_at_start",
        "guard_probe",
        "prepare_guard_probe",
        "_probe",
    )
    if any(marker in text for marker in probe_markers):
        reasons.append("benchmark_probe_only")
    if _has_true_probe_flag(probe_context):
        reasons.append("benchmark_probe_only")

    return sorted(set(reasons))


def _annotate_release_probe_only(item: dict[str, Any]) -> dict[str, Any]:
    reasons = _release_probe_only_reasons(item)
    if reasons:
        item["release_probe_only"] = True
        item["release_probe_only_reasons"] = reasons
    return item


def _release_claim_perf_eligible(item: Mapping[str, Any]) -> bool:
    return not bool(item.get("release_probe_only")) and not _release_probe_only_reasons(item)


def _normalize_gpu_bubble_report(report: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = _mapping(report.get("benchmark"))
    classification = _mapping(report.get("classification"))
    comparison = _mapping(report.get("comparison"))
    telemetry = _mapping(report.get("gpu_telemetry"))
    windows = _mapping(report.get("gpu_telemetry_windows"))
    active_window = _mapping(windows.get("active_window_gpu20"))
    runs = [dict(run) for run in report.get("run_summaries", []) if isinstance(run, Mapping)]
    active_gpu = _safe_float(
        classification.get("active_gpu_util_pct_mean"),
        _safe_float(active_window.get("gpu_util_pct_mean")),
    )
    memory_total = _safe_float(telemetry.get("memory_total_mb"))
    memory_used = _safe_float(telemetry.get("memory_used_mb_max"))
    memory_ratio = memory_used / max(memory_total, 1.0) if memory_total > 0.0 else 0.0
    speedup_pct = _first_float(
        comparison,
        "steady_samples_per_second_gain_pct",
        "steady_step_speedup_pct",
        "samples_per_second_gain_pct",
        "speedup_pct",
    )
    family = str(
        benchmark.get("family")
        or benchmark.get("model_family")
        or benchmark.get("model")
        or benchmark.get("name")
        or ""
    )
    return _annotate_release_probe_only({
        "kind": "gpu_bubble_experiment",
        "report": "gpu_bubble_experiment_report_v0",
        "family": family,
        "case_id": str(benchmark.get("case") or benchmark.get("case_id") or benchmark.get("name") or family or "unknown"),
        "benchmark": dict(benchmark),
        "status": str(classification.get("status") or "unknown"),
        "success_run_count": sum(1 for run in runs if _safe_bool(run.get("success"), False)),
        "steady_samples_per_second": _round(_max_float(runs, "steady_samples_per_second"), 6),
        "speedup_pct": _round(speedup_pct, 4),
        "active_gpu_util_pct_mean": _round(active_gpu, 4),
        "active_gpu_saturated_sample_ratio": _round(classification.get("active_gpu_saturated_sample_ratio")),
        "max_data_wait_share": _round(max(_safe_float(classification.get("max_data_wait_share")), _max_float(runs, "data_wait_share"))),
        "max_h2d_transfer_share": _round(max(_safe_float(classification.get("max_h2d_transfer_share")), _max_float(runs, "h2d_transfer_share"))),
        "max_optimizer_share": _round(_max_float(runs, "optimizer_share")),
        "max_host_gap_share": _round(_max_float(runs, "host_gap_share")),
        "peak_vram_mb": _round(max(_max_float(runs, "peak_vram_mb"), memory_used), 4),
        "memory_ratio": _round(memory_ratio),
        "loss_delta": _loss_delta(report),
        "search_text": _search_text(benchmark, classification, [run.get("label") for run in runs]),
    })


def _normalize_controller_report(report: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _mapping(report.get("snapshot"))
    config = _mapping(snapshot.get("config"))
    gpu = _mapping(snapshot.get("gpu"))
    safety = _mapping(snapshot.get("safety"))
    diagnosis = _mapping(report.get("diagnosis"))
    evidence = _mapping(diagnosis.get("evidence"))
    action_plan = _mapping(report.get("action_plan"))
    return {
        "kind": "bubble_controller",
        "report": "bubble_aware_runtime_controller_v0",
        "family": str(config.get("family") or ""),
        "case_id": str(config.get("family") or diagnosis.get("kind") or "controller_report"),
        "benchmark": {},
        "status": str(diagnosis.get("kind") or report.get("status") or "unknown"),
        "success_run_count": 0,
        "steady_samples_per_second": 0.0,
        "speedup_pct": 0.0,
        "active_gpu_util_pct_mean": _round(evidence.get("active_gpu_util_pct_mean", gpu.get("active_gpu_util_pct_mean")), 4),
        "active_gpu_saturated_sample_ratio": _round(evidence.get("active_gpu_saturated_sample_ratio")),
        "max_data_wait_share": _round(evidence.get("data_wait_share")),
        "max_h2d_transfer_share": _round(evidence.get("h2d_transfer_share")),
        "max_optimizer_share": _round(evidence.get("optimizer_share")),
        "max_host_gap_share": _round(evidence.get("host_gap_share")),
        "peak_vram_mb": _round(gpu.get("memory_used_mb_max"), 4),
        "memory_ratio": _round(safety.get("memory_ratio")),
        "loss_delta": None,
        "action_domain": str(action_plan.get("domain") or ""),
        "action_status": str(action_plan.get("status") or ""),
        "has_advisor_patch": bool(action_plan.get("mutations")),
        "search_text": _search_text(config, diagnosis, action_plan),
    }


def _normalize_ab_evidence_report(report: Mapping[str, Any]) -> dict[str, Any]:
    matrix_case = _mapping(report.get("matrix_case"))
    before = _mapping(report.get("before"))
    after = _mapping(report.get("after"))
    after_metrics = _mapping(after.get("metrics"))
    comparison = _mapping(report.get("comparison"))
    action = _mapping(report.get("action"))
    release_probe_context = {
        "matrix_case": dict(matrix_case),
        "before_config": dict(_mapping(before.get("config"))),
        "after_config": dict(_mapping(after.get("config"))),
        "before_cache_state": before.get("cache_state"),
        "after_cache_state": after.get("cache_state"),
        "before_source_fixture": dict(_mapping(before.get("source_fixture"))),
        "after_source_fixture": dict(_mapping(after.get("source_fixture"))),
    }
    return _annotate_release_probe_only({
        "kind": "bubble_ab_evidence",
        "report": "bubble_advisor_ab_evidence_v0",
        "family": str(matrix_case.get("family") or after.get("family") or before.get("family") or ""),
        "case_id": str(matrix_case.get("case_id") or after.get("case_id") or before.get("case_id") or "bubble_ab_evidence"),
        "benchmark": dict(matrix_case),
        "release_probe_context": release_probe_context,
        "status": str(report.get("status") or "unknown"),
        "success_run_count": 2 if report.get("status") in {"keep_recommended", "needs_review"} else 1,
        "steady_samples_per_second": _round(after_metrics.get("steady_samples_per_second"), 6),
        "speedup_pct": _round(comparison.get("steady_samples_per_second_gain_pct"), 4),
        "active_gpu_util_pct_mean": _round(after_metrics.get("active_gpu_util_pct_mean"), 4),
        "active_gpu_saturated_sample_ratio": _round(after_metrics.get("active_gpu_saturated_sample_ratio")),
        "max_data_wait_share": _round(after_metrics.get("data_wait_share")),
        "max_h2d_transfer_share": _round(after_metrics.get("h2d_transfer_share")),
        "max_optimizer_share": _round(after_metrics.get("optimizer_share")),
        "max_host_gap_share": _round(after_metrics.get("host_gap_share")),
        "peak_vram_mb": _round(after_metrics.get("peak_vram_mb"), 4),
        "memory_ratio": _round(after_metrics.get("memory_ratio")),
        "loss_delta": _safe_float(comparison.get("final_loss_delta")) if "final_loss_delta" in comparison else None,
        "action_domain": str(action.get("domain") or ""),
        "action_status": str(action.get("status") or ""),
        "has_advisor_patch": bool(action.get("applied_overlay")),
        "search_text": _search_text(matrix_case, before, after, comparison, action),
    })


def _normalize_closed_loop_evidence_report(report: Mapping[str, Any]) -> dict[str, Any]:
    latest = _mapping(report.get("latest_action"))
    comparison = _mapping(report.get("comparison"))
    decision = _mapping(report.get("decision"))
    safety = _mapping(report.get("safety"))
    runtime_adapter = _mapping(report.get("runtime_adapter"))
    rollback_adapter = _mapping(latest.get("rollback_adapter"))
    adapter_id = str(runtime_adapter.get("adapter_id") or rollback_adapter.get("adapter_id") or "")
    return {
        "kind": "bubble_closed_loop_evidence",
        "report": "bubble_runtime_closed_loop_evidence_v0",
        "family": str(report.get("family") or ""),
        "case_id": str(report.get("case_id") or "bubble_closed_loop"),
        "benchmark": {},
        "status": str(report.get("status") or decision.get("status") or "unknown"),
        "success_run_count": 1 if _safe_float(report.get("action_count")) > 0 else 0,
        "steady_samples_per_second": _round(comparison.get("steady_samples_per_second_after"), 6),
        "speedup_pct": _round(comparison.get("steady_samples_per_second_gain_pct"), 4),
        "active_gpu_util_pct_mean": _round(comparison.get("active_gpu_util_pct_after"), 4),
        "active_gpu_saturated_sample_ratio": 0.0,
        "max_data_wait_share": 0.0,
        "max_h2d_transfer_share": 0.0,
        "max_optimizer_share": 0.0,
        "max_host_gap_share": _round(comparison.get("host_gap_share_after")),
        "peak_vram_mb": 0.0,
        "memory_ratio": 0.0,
        "loss_delta": None,
        "action_domain": str(latest.get("domain") or ""),
        "action_status": str(latest.get("status") or ""),
        "has_advisor_patch": False,
        "action_count": _safe_float(report.get("action_count")),
        "kept_count": _safe_float(report.get("kept_count")),
        "rolled_back_count": _safe_float(report.get("rolled_back_count")),
        "rollback_failed_count": _safe_float(report.get("rollback_failed_count")),
        "duplicate_action_blocked": _safe_bool(safety.get("duplicate_action_blocked"), False),
        "cross_run_action_blocked": _safe_bool(safety.get("cross_run_action_blocked"), False),
        "current_run_adapter_blocked": _safe_bool(safety.get("current_run_adapter_blocked"), False),
        "next_request_only_adapter": _safe_bool(
            safety.get("next_request_only_adapter"),
            _safe_bool(runtime_adapter.get("next_request_only"), False),
        ),
        "runtime_adapter_id": adapter_id,
        "latest_action_adapter_id": str(rollback_adapter.get("adapter_id") or ""),
        "required_evidence": _string_list(safety.get("required_evidence") or runtime_adapter.get("required_evidence")),
        "search_text": _search_text(report),
    }


def _normalize_natural_data_wait_evidence_report(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping(report.get("metrics"))
    release_claim = _mapping(report.get("release_claim"))
    decision = _mapping(report.get("decision"))
    loss_stability = _mapping(report.get("loss_stability"))
    action_chain = report.get("action_chain")
    actions = action_chain if isinstance(action_chain, Sequence) and not isinstance(action_chain, (str, bytes)) else []
    latest_action = _mapping(actions[-1]) if actions else {}
    loss_status = str(loss_stability.get("status") or "")
    loss_observed = loss_status in {"observed", "stable"}
    release_gate_reasons = natural_data_wait_release_reasons(report)
    return {
        "kind": "bubble_natural_data_wait_evidence",
        "report": "bubble_natural_data_wait_evidence_v0",
        "family": str(report.get("family") or ""),
        "case_id": str(report.get("case_id") or "bubble_natural_data_wait"),
        "benchmark": {},
        "status": str(report.get("status") or "unknown"),
        "success_run_count": 1 if not release_gate_reasons else 0,
        "steady_samples_per_second": _round(metrics.get("steady_samples_per_second"), 6),
        "speedup_pct": _round(latest_action.get("steady_samples_per_second_gain_pct"), 4),
        "active_gpu_util_pct_mean": 0.0,
        "active_gpu_saturated_sample_ratio": 0.0,
        "max_data_wait_share": _round(metrics.get("data_wait_share")),
        "max_h2d_transfer_share": _round(metrics.get("h2d_transfer_share")),
        "max_optimizer_share": _round(metrics.get("optimizer_share")),
        "max_host_gap_share": _round(metrics.get("host_gap_share")),
        "peak_vram_mb": _round(metrics.get("peak_vram_mb"), 4),
        "memory_ratio": 0.0,
        "loss_delta": None,
        "final_loss": loss_stability.get("final_loss", metrics.get("final_loss")),
        "loss_stability_status": loss_status,
        "action_domain": "data_supply" if _safe_bool(decision.get("dataloader_rebuild_observed"), False) else "",
        "action_status": str(latest_action.get("status") or ""),
        "has_advisor_patch": False,
        "action_count": _safe_float(report.get("action_count")),
        "natural_release_eligible": not release_gate_reasons,
        "natural_release_gate_reasons": release_gate_reasons,
        "natural_release_scope": str(release_claim.get("scope") or ""),
        "dataloader_rebuild_observed": _safe_bool(decision.get("dataloader_rebuild_observed"), False),
        "benchmark_injection_blockers": _string_list(report.get("benchmark_injection_blockers")),
        "diagnostic_only": _safe_bool(report.get("diagnostic_only"), False),
        "search_text": _search_text(report),
    }


def _normalize_natural_data_wait_ab_evidence_report(report: Mapping[str, Any]) -> dict[str, Any]:
    before = _mapping(report.get("before"))
    after = _mapping(report.get("after"))
    before_metrics = _mapping(before.get("metrics"))
    after_metrics = _mapping(after.get("metrics"))
    comparison = _mapping(report.get("comparison"))
    action = _mapping(report.get("action"))
    release_claim = _mapping(report.get("release_claim"))
    loss_stability = _mapping(report.get("loss_stability"))
    before_action = _mapping(action.get("before"))
    after_action = _mapping(action.get("after"))
    loss_status = str(loss_stability.get("status") or "")
    release_gate_reasons = natural_ab_release_reasons(report)
    return {
        "kind": "bubble_natural_data_wait_ab_evidence",
        "report": "bubble_natural_data_wait_ab_evidence_v0",
        "family": str(report.get("family") or ""),
        "case_id": str(report.get("case_id") or "bubble_natural_data_wait_ab"),
        "benchmark": {},
        "status": str(report.get("status") or "unknown"),
        "success_run_count": 2 if not release_gate_reasons else 1,
        "steady_samples_per_second": _round(after_metrics.get("steady_samples_per_second"), 6),
        "speedup_pct": _round(comparison.get("steady_samples_per_second_gain_pct"), 4),
        "active_gpu_util_pct_mean": _round(after_metrics.get("active_gpu_util_pct_mean"), 4),
        "active_gpu_saturated_sample_ratio": 0.0,
        "max_data_wait_share": _round(after_metrics.get("data_wait_share")),
        "max_h2d_transfer_share": _round(after_metrics.get("h2d_transfer_share")),
        "max_optimizer_share": _round(after_metrics.get("optimizer_share")),
        "max_host_gap_share": _round(after_metrics.get("host_gap_share")),
        "peak_vram_mb": _round(after_metrics.get("peak_vram_mb"), 4),
        "memory_ratio": _round(after_metrics.get("memory_ratio")),
        "loss_delta": _safe_float(comparison.get("final_loss_delta")) if "final_loss_delta" in comparison else None,
        "final_loss_before": comparison.get("final_loss_before"),
        "final_loss_after": comparison.get("final_loss_after"),
        "loss_regression_ratio": comparison.get("loss_regression_ratio"),
        "loss_stability_status": loss_status,
        "action_domain": str(action.get("domain") or "data_supply"),
        "action_status": str(action.get("status") or ""),
        "action_kind": str(action.get("action_kind") or ""),
        "has_advisor_patch": False,
        "natural_ab_release_eligible": not release_gate_reasons,
        "natural_release_gate_reasons": release_gate_reasons,
        "natural_release_scope": str(release_claim.get("scope") or ""),
        "data_wait_share_before": _round(comparison.get("data_wait_share_before")),
        "data_wait_share_after": _round(comparison.get("data_wait_share_after")),
        "data_wait_share_delta": _round(comparison.get("data_wait_share_delta")),
        "data_wait_reduction_pct": _round(comparison.get("data_wait_reduction_pct"), 4),
        "before_dataloader_workers": before_action.get("dataloader_workers"),
        "before_dataloader_prefetch_factor": before_action.get("dataloader_prefetch_factor"),
        "before_pin_memory": before_action.get("pin_memory"),
        "after_dataloader_workers": after_action.get("dataloader_workers"),
        "after_dataloader_prefetch_factor": after_action.get("dataloader_prefetch_factor"),
        "after_pin_memory": after_action.get("pin_memory"),
        "benchmark_injection_blockers": _string_list(report.get("benchmark_injection_blockers")),
        "diagnostic_only": _safe_bool(report.get("diagnostic_only"), False),
        "search_text": _search_text(report),
    }


def _normalize_reports(reports: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in reports:
        report = _mapping(raw)
        if not report:
            continue
        if report.get("report") == "gpu_bubble_experiment_report_v0":
            normalized.append(_normalize_gpu_bubble_report(report))
        elif report.get("report") == "bubble_advisor_ab_evidence_v0":
            normalized.append(_normalize_ab_evidence_report(report))
        elif report.get("report") == "bubble_runtime_closed_loop_evidence_v0":
            normalized.append(_normalize_closed_loop_evidence_report(report))
        elif report.get("report") == "bubble_natural_data_wait_evidence_v0":
            normalized.append(_normalize_natural_data_wait_evidence_report(report))
        elif report.get("report") == "bubble_natural_data_wait_ab_evidence_v0":
            normalized.append(_normalize_natural_data_wait_ab_evidence_report(report))
        elif report.get("controller") == "bubble_aware_runtime_controller_v0":
            normalized.append(_normalize_controller_report(report))
        elif isinstance(report.get("reports"), Sequence) and not isinstance(report.get("reports"), (str, bytes)):
            normalized.extend(_normalize_reports(item for item in report.get("reports", []) if isinstance(item, Mapping)))
    return normalized


def _case_coverage(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    release_items = [item for item in items if _release_claim_perf_eligible(item)]

    def has(predicate) -> bool:
        return any(predicate(item, str(item.get("search_text") or "")) for item in release_items)

    cases = [
        (
            "sd15_lora_512",
            "SD15 LoRA 512",
            lambda item, text: "sd15" in text and "512" in text,
        ),
        (
            "sdxl_lora_1024",
            "SDXL LoRA 1024",
            lambda item, text: "sdxl" in text and "1024" in text,
        ),
        (
            "anima_cache_first",
            "Anima tiny/cache-first",
            lambda item, text: "anima" in text and ("cache" in text or "tiny" in text),
        ),
        (
            "anima_saturation_boundary",
            "Anima saturation boundary",
            lambda item, text: "anima" in text
            and ("saturation" in text or str(item.get("status")) == "gpu_saturated" or _safe_float(item.get("active_gpu_util_pct_mean")) >= 85.0),
        ),
        (
            "newbie_dit_cache_first",
            "Newbie/DiT cache-first",
            lambda item, text: ("newbie" in text or "dit" in text) and "cache" in text,
        ),
    ]
    return [
        {"case_id": case_id, "label": label, "covered": has(predicate)}
        for case_id, label, predicate in cases
    ]


def _claim(claim_id: str, text: str, *, evidence: Sequence[Any], status: str = "supported") -> dict[str, Any]:
    return {
        "id": claim_id,
        "status": status,
        "claim": text,
        "evidence": list(evidence),
    }


def _publishable_claims(items: Sequence[Mapping[str, Any]], *, min_throughput_gain_pct: float) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    benchmark_items = [item for item in items if item.get("kind") == "gpu_bubble_experiment"]
    ab_items = [item for item in items if item.get("kind") == "bubble_ab_evidence"]
    perf_items = [item for item in [*benchmark_items, *ab_items] if _release_claim_perf_eligible(item)]
    controller_items = [item for item in items if item.get("kind") == "bubble_controller"]
    speedups = [item for item in perf_items if _safe_float(item.get("speedup_pct")) >= min_throughput_gain_pct]
    if speedups:
        best = max(speedups, key=lambda item: _safe_float(item.get("speedup_pct")))
        claims.append(
            _claim(
                "steady_throughput_gain_observed",
                f"Observed up to +{_pct(_safe_float(best.get('speedup_pct')))}% steady samples/s on covered benchmark evidence.",
                evidence=[{"case_id": item.get("case_id"), "speedup_pct": item.get("speedup_pct")} for item in speedups],
            )
        )
    high_util = [
        item for item in perf_items
        if _safe_float(item.get("active_gpu_util_pct_mean")) >= 85.0
        or _safe_float(item.get("active_gpu_saturated_sample_ratio")) >= 0.60
    ]
    if high_util:
        claims.append(
            _claim(
                "active_window_high_util_observed",
                "At least one benchmark reached high active-window GPU utilization; keep wording case-specific.",
                evidence=[
                    {"case_id": item.get("case_id"), "active_gpu_util_pct_mean": item.get("active_gpu_util_pct_mean")}
                    for item in high_util
                ],
            )
        )
    domains = sorted(
        {
            str(item.get("action_domain"))
            for item in [*controller_items, *ab_items]
            if item.get("has_advisor_patch") and str(item.get("action_domain") or "")
        }
    )
    if domains:
        claims.append(
            _claim(
                "advisor_patch_domains_supported",
                "Advisor mode can produce next-request patches for observed bubble domains.",
                evidence=[{"domains": domains}],
                status="supported_by_controller_report",
            )
        )
    underfilled = [item for item in items if str(item.get("status")) == "workload_underfilled"]
    if underfilled:
        claims.append(
            _claim(
                "workload_underfilled_explained",
                "The advisor can explain low GPU utilization caused by tiny workloads instead of treating it as an optimizer failure.",
                evidence=[{"case_id": item.get("case_id"), "active_gpu_util_pct_mean": item.get("active_gpu_util_pct_mean")} for item in underfilled],
                status="supported_by_controller_report",
            )
        )
    closed_loop_items = [item for item in items if item.get("kind") == "bubble_closed_loop_evidence"]
    safe_closed_loop = [
        item
        for item in closed_loop_items
        if _safe_float(item.get("action_count")) > 0 and _safe_float(item.get("rollback_failed_count")) <= 0.0
    ]
    if safe_closed_loop:
        claims.append(
            _claim(
                "online_closed_loop_safety_observed",
                "Online auto-apply evidence exists for low-risk host-scheduling actions with cooldown and rollback accounting.",
                evidence=[
                    {
                        "case_id": item.get("case_id"),
                        "status": item.get("status"),
                        "action_count": item.get("action_count"),
                        "rolled_back_count": item.get("rolled_back_count"),
                    }
                    for item in safe_closed_loop
                ],
                status="supported_by_closed_loop_evidence",
            )
        )
    natural_data_wait = [
        item
        for item in items
        if item.get("kind") == "bubble_natural_data_wait_evidence"
        and _safe_bool(item.get("natural_release_eligible"), False)
    ]
    if natural_data_wait:
        claims.append(
            _claim(
                "natural_data_wait_dataloader_rebuild_observed",
                "A non-injected data_wait case observed DataLoader rebuild through the closed-loop evidence path; keep wording case-specific.",
                evidence=[
                    {"case_id": item.get("case_id"), "data_wait_share": item.get("max_data_wait_share"), "speedup_pct": item.get("speedup_pct")}
                    for item in natural_data_wait
                ],
                status="supported_by_natural_data_wait_evidence",
            )
        )
    natural_data_wait_ab = [
        item
        for item in items
        if item.get("kind") == "bubble_natural_data_wait_ab_evidence"
        and _safe_bool(item.get("natural_ab_release_eligible"), False)
        and _safe_float(item.get("data_wait_share_before")) >= DEFAULT_DATA_WAIT_THRESHOLD
        and _safe_float(item.get("data_wait_share_after")) < DEFAULT_DATA_WAIT_THRESHOLD
        and _safe_float(item.get("data_wait_share_after")) < _safe_float(item.get("data_wait_share_before"))
        and _safe_float(item.get("speedup_pct")) >= max(float(min_throughput_gain_pct or 0.0), 0.0)
        and not _string_list(item.get("benchmark_injection_blockers"))
        and str(item.get("status") or "") == "keep_recommended"
        and str(item.get("loss_stability_status") or "") == "stable"
    ]
    if natural_data_wait_ab:
        claims.append(
            _claim(
                "natural_data_wait_next_run_workers_prefetch_gain_observed",
                "A non-injected natural data_wait A/B probe observed next-run DataLoader worker/prefetch tuning reducing steady data_wait with higher throughput; keep wording case-specific.",
                evidence=[
                    {key: item.get(key) for key in (
                        "case_id", "speedup_pct", "data_wait_share_before", "data_wait_share_after",
                        "before_dataloader_workers", "after_dataloader_workers", "after_dataloader_prefetch_factor",
                    )}
                    for item in natural_data_wait_ab
                ],
                status="supported_by_natural_data_wait_ab_evidence",
            )
        )
    return claims


def _evidence_gaps(items: Sequence[Mapping[str, Any]], coverage: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    release_items = [item for item in items if _release_claim_perf_eligible(item)]
    benchmark_count = sum(1 for item in release_items if item.get("kind") == "gpu_bubble_experiment")
    if benchmark_count <= 0:
        gaps.append({"id": "benchmark_reports_missing", "reason": "no gpu_bubble_experiment_report_v0 evidence was provided"})
    for case in coverage:
        if not case.get("covered"):
            gaps.append({"id": "benchmark_case_missing", "case_id": case.get("case_id"), "label": case.get("label")})
    if not any(item.get("loss_delta") is not None for item in release_items):
        gaps.append({"id": "loss_delta_missing", "reason": "loss stability evidence is required for quality-sensitive release claims"})
    if not any(_safe_float(item.get("memory_ratio")) > 0.0 or _safe_float(item.get("peak_vram_mb")) > 0.0 for item in release_items):
        gaps.append({"id": "vram_evidence_missing", "reason": "peak VRAM or memory ratio evidence is required"})
    if not any(_safe_float(item.get("speedup_pct")) > 0.0 for item in release_items):
        gaps.append({"id": "throughput_gain_missing", "reason": "no positive steady throughput gain was found"})
    return gaps


def _release_gate_fields(readiness: str) -> dict[str, Any]:
    claim_allowed = readiness == "ready_with_case_specific_wording"
    return {
        "publishable": claim_allowed,
        "release_claim_allowed": claim_allowed,
        "not_release_evidence": not claim_allowed,
        "safe_to_auto_start": False,
        "claim_publication_scope": (
            "case_specific_benchmark_claims" if claim_allowed else "non_release_benchmark_claims"
        ),
    }


def build_release_claim_report(
    reports: Iterable[Mapping[str, Any]],
    *,
    min_throughput_gain_pct: float = 3.0,
) -> dict[str, Any]:
    items = _normalize_reports(reports)
    for item in items:
        item["release_perf_eligible"] = _release_claim_perf_eligible(item)
    coverage = _case_coverage(items)
    claims = _publishable_claims(items, min_throughput_gain_pct=max(float(min_throughput_gain_pct or 0.0), 0.0))
    gaps = _evidence_gaps(items, coverage)
    annotated_items, provenance = build_release_evidence_provenance(
        items,
        min_throughput_gain_pct=max(float(min_throughput_gain_pct or 0.0), 0.0),
    )
    missing_matrix = any(gap.get("id") == "benchmark_case_missing" for gap in gaps)
    has_perf_claim = any(claim.get("id") == "steady_throughput_gain_observed" for claim in claims)
    readiness = "ready_with_case_specific_wording" if not missing_matrix and has_perf_claim and not gaps else "blocked_pending_evidence"
    return {
        "schema_version": 1,
        "report": "bubble_runtime_release_claims_v0",
        "roadmap": ROADMAP,
        "release_readiness": readiness,
        **_release_gate_fields(readiness),
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "min_throughput_gain_pct": _round(min_throughput_gain_pct, 4),
        "evidence_count": len(items),
        "benchmark_count": sum(
            1
            for item in items
            if item.get("kind") == "gpu_bubble_experiment" and _release_claim_perf_eligible(item)
        ),
        "ab_evidence_count": sum(1 for item in items if item.get("kind") == "bubble_ab_evidence"),
        "controller_report_count": sum(1 for item in items if item.get("kind") == "bubble_controller"),
        "closed_loop_evidence_count": sum(1 for item in items if item.get("kind") == "bubble_closed_loop_evidence"),
        "natural_data_wait_evidence_count": sum(1 for item in items if item.get("kind") == "bubble_natural_data_wait_evidence"),
        "natural_data_wait_ab_evidence_count": sum(1 for item in items if item.get("kind") == "bubble_natural_data_wait_ab_evidence"),
        "coverage": coverage,
        "publishable_claims": claims,
        "supported_benchmark_claims": claims,
        "evidence_provenance": provenance,
        "blocked_claims": [
            {
                "id": "universal_99pct_gpu_util",
                "claim": "All training jobs reach 99% GPU utilization.",
                "status": "blocked_always",
                "reason": "utilization depends on workload size, validation/checkpoint boundaries, CPU/IO, and throughput tradeoffs",
            },
            {
                "id": "low_gpu_util_means_bad_optimization",
                "claim": "Low GPU utilization always means the trainer is poorly optimized.",
                "status": "blocked_always",
                "reason": "tiny workloads can be throughput-healthy while underfilling the GPU",
            },
            {
                "id": "default_online_auto_apply",
                "claim": "The controller mutates running training jobs by default.",
                "status": "blocked_by_default_policy",
                "reason": "online mutation is opt-in and limited to low-risk host-scheduling actions with cooldown and rollback evidence",
            },
        ],
        "evidence_gaps": gaps,
        "normalized_evidence": [
            {key: value for key, value in item.items() if key != "search_text"}
            for item in annotated_items
        ],
    }


__all__ = ["ROADMAP", "build_release_claim_report"]
