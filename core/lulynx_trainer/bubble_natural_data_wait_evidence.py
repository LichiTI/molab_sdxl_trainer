"""Evidence classifier for non-injected data_wait probes."""

from __future__ import annotations

import math
from typing import Any, Mapping


NATURAL_DATA_WAIT_EVIDENCE_REPORT = "bubble_natural_data_wait_evidence_v0"
DATALOADER_REBUILD_ADAPTER_ID = "dataloader_rebuild_runtime_contract_v0"


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


def _first_run(summary: Mapping[str, Any], preferred_label: str) -> Mapping[str, Any]:
    runs = _mapping(summary.get("runs"))
    preferred = _mapping(runs.get(preferred_label))
    if preferred:
        return preferred
    for run in runs.values():
        mapped = _mapping(run)
        if mapped:
            return mapped
    return {}


def _loss_stability(run: Mapping[str, Any]) -> dict[str, Any]:
    raw_losses = run.get("losses")
    losses = raw_losses if isinstance(raw_losses, list) else []
    finite_losses = [value for value in (_finite_float_or_none(item) for item in losses) if value is not None]
    non_finite_count = max(len(losses) - len(finite_losses), 0)
    final_loss = _finite_float_or_none(run.get("final_loss"))
    status = "observed" if final_loss is not None and non_finite_count == 0 else "missing"
    if non_finite_count > 0:
        status = "non_finite"
    return {
        "schema": "bubble_loss_stability_v0",
        "source": "natural_data_wait",
        "status": status,
        "final_loss": _optional_round(final_loss, 6),
        "loss_count": len(losses),
        "non_finite_loss_count": non_finite_count,
    }


def _bubble_profile(run: Mapping[str, Any]) -> Mapping[str, Any]:
    profile = _mapping(run.get("steady_bubble_profile"))
    if profile:
        return profile
    runtime = _mapping(run.get("runtime_feature_summary"))
    loop = _mapping(runtime.get("training_loop_runtime"))
    step_phase = _mapping(loop.get("step_phase_profile"))
    profile = _mapping(step_phase.get("gpu_bubble_profile"))
    if profile:
        return profile
    evidence = {
        "data_wait_share": step_phase.get("data_wait_share"),
        "h2d_transfer_share": step_phase.get("h2d_transfer_share"),
        "optimizer_share": step_phase.get("optimizer_share"),
        "host_gap_share": step_phase.get("host_gap_share"),
    }
    return {
        "dominant_bottleneck": step_phase.get("dominant_bottleneck", ""),
        "bubble_ratio_estimate": step_phase.get("bubble_ratio_estimate", 0.0),
        "evidence": evidence,
    }


def _benchmark_injection(summary: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = _mapping(summary.get("benchmark"))
    controlled_data_wait_share = _safe_float(benchmark.get("bubble_controller_controlled_data_wait_share"))
    controlled_rollback_ratio = _safe_float(benchmark.get("bubble_controller_controlled_rollback_slowdown_ratio"), 1.0)
    stall_ms = _safe_float(benchmark.get("bubble_controller_benchmark_data_wait_stall_ms"))
    controlled_data_wait = run.get("bubble_controlled_data_wait_observations")
    controlled_rollback = run.get("bubble_controlled_rollback_observations")
    benchmark_stall = _mapping(run.get("bubble_benchmark_data_wait_stall"))
    return {
        "controlled_data_wait": controlled_data_wait_share > 0.0
        or (isinstance(controlled_data_wait, list) and bool(controlled_data_wait)),
        "controlled_rollback": controlled_rollback_ratio > 1.0
        or (isinstance(controlled_rollback, list) and bool(controlled_rollback)),
        "benchmark_data_wait_stall": stall_ms > 0.0 or bool(benchmark_stall),
        "controlled_data_wait_share": round(controlled_data_wait_share, 6),
        "controlled_rollback_slowdown_ratio": round(controlled_rollback_ratio, 6),
        "benchmark_data_wait_stall_ms": round(stall_ms, 4),
    }


def _closed_loop_state(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    extra = _mapping(_mapping(manifest).get("extra"))
    state = _mapping(extra.get("bubble_closed_loop_state"))
    if state:
        return state
    controller = _mapping(extra.get("bubble_controller"))
    return _mapping(controller.get("closed_loop"))


def _action_chain(manifest: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    state = _closed_loop_state(manifest)
    raw_history = state.get("action_history")
    if not isinstance(raw_history, list):
        executor = _mapping(state.get("executor"))
        raw_history = executor.get("action_history")
    history = raw_history if isinstance(raw_history, list) else []
    actions: list[dict[str, Any]] = []
    for item in history:
        action = _mapping(item)
        if not action:
            continue
        evaluation = _mapping(action.get("evaluation"))
        before_metrics = _mapping(action.get("before_metrics"))
        eval_before = _mapping(evaluation.get("before"))
        eval_after = _mapping(evaluation.get("after"))
        profiler_handoff = _mapping(action.get("profiler_handoff"))
        rebuild = _mapping(action.get("dataloader_rebuild"))
        plan = _mapping(rebuild.get("runtime_rebuild_plan"))
        next_descriptor = _mapping(plan.get("next_descriptor"))
        rollback_descriptor = _mapping(plan.get("rollback_descriptor"))
        actions.append(
            {
                "status": str(action.get("status") or ""),
                "action_kind": str(action.get("action_kind") or ""),
                "adapter_id": str(action.get("adapter_id") or _mapping(action.get("runtime_apply")).get("adapter_id") or ""),
                "apply_boundary": str(action.get("apply_boundary") or _mapping(action.get("runtime_apply")).get("apply_boundary") or ""),
                "applied_step": _safe_int(action.get("applied_step"), 0),
                "closed_step": _safe_int(action.get("closed_step"), 0),
                "steady_samples_per_second_gain_pct": _round(
                    evaluation.get("steady_samples_per_second_gain_pct"),
                    4,
                ),
                "before_data_wait_share": _optional_round(before_metrics.get("data_wait_share")),
                "eval_before_data_wait_share": _optional_round(eval_before.get("data_wait_share")),
                "eval_after_data_wait_share": _optional_round(eval_after.get("data_wait_share")),
                "profiler_handoff_kind": str(profiler_handoff.get("kind") or ""),
                "profiler_handoff_data_wait_share": _optional_round(profiler_handoff.get("data_wait_share")),
                "next_workers": next_descriptor.get("num_workers"),
                "next_prefetch_factor": next_descriptor.get("prefetch_factor"),
                "next_pin_memory": next_descriptor.get("pin_memory"),
                "next_persistent_workers": next_descriptor.get("persistent_workers"),
                "rollback_workers": rollback_descriptor.get("num_workers"),
                "rollback_prefetch_factor": rollback_descriptor.get("prefetch_factor"),
                "rollback_pin_memory": rollback_descriptor.get("pin_memory"),
            }
        )
    return actions


def _has_dataloader_rebuild_observation(actions: list[Mapping[str, Any]]) -> bool:
    for action in actions:
        if str(action.get("action_kind") or "") != "set_dataloader_workers":
            continue
        if str(action.get("adapter_id") or "") != DATALOADER_REBUILD_ADAPTER_ID:
            continue
        if str(action.get("status") or "") in {"kept", "rolled_back", "rollback_failed", "needs_more_evidence"}:
            return True
    return False


def _dataloader_rebuild_actions(actions: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        action
        for action in actions
        if str(action.get("action_kind") or "") == "set_dataloader_workers"
        and str(action.get("adapter_id") or "") == DATALOADER_REBUILD_ADAPTER_ID
    ]


def _matrix_axes(benchmark: Mapping[str, Any], source_fixture: Mapping[str, Any] | None) -> dict[str, Any]:
    fixture = _mapping(source_fixture)
    return {
        "family": str(benchmark.get("family") or ""),
        "native_cache_mode": str(benchmark.get("native_cache_mode") or ""),
        "resolution": _safe_int(benchmark.get("resolution"), 0),
        "train_batch_size": _safe_int(benchmark.get("train_batch_size"), 0),
        "steps": _safe_int(benchmark.get("steps"), 0),
        "samples": _safe_int(benchmark.get("samples"), 0),
        "dataloader_workers": _safe_int(benchmark.get("dataloader_workers"), 0),
        "dataloader_prefetch_factor": _safe_int(benchmark.get("dataloader_prefetch_factor"), 0),
        "pin_memory": bool(benchmark.get("pin_memory", True)),
        "source_fixture": str(fixture.get("fixture") or ""),
        "fixture_width": _safe_int(fixture.get("width"), 0),
        "fixture_height": _safe_int(fixture.get("height"), 0),
        "fixture_samples": _safe_int(fixture.get("samples"), 0),
        "fixture_variants": _string_list(fixture.get("variants")),
        "cache_state": str(fixture.get("cache_state") or ""),
        "cache_present_before": fixture.get("cache_present_before"),
        "cache_has_family_cache": fixture.get("cache_has_family_cache"),
        "cache_inventory": dict(_mapping(fixture.get("cache_inventory"))),
        "source_image_count": _safe_int(fixture.get("source_image_count"), 0),
        "source_file_count": _safe_int(fixture.get("source_file_count"), 0),
        "source_manifest_sha1": str(fixture.get("source_manifest_sha1") or ""),
        "material_source_label": str(fixture.get("label") or ""),
    }


def _cache_probe_only_reason(benchmark: Mapping[str, Any], source_fixture: Mapping[str, Any] | None) -> str:
    fixture = _mapping(source_fixture)
    cache_state = str(fixture.get("cache_state") or "").strip().lower()
    native_cache_mode = str(benchmark.get("native_cache_mode") or "").strip().lower()
    if cache_state == "missing_at_start":
        return "cache_miss_fixture_probe_only"
    if native_cache_mode in {"online_cache", "rebuild_cache"}:
        return f"{native_cache_mode}_probe_only"
    return ""


def _rollback_analysis(
    *,
    benchmark: Mapping[str, Any],
    source_fixture: Mapping[str, Any] | None,
    actions: list[Mapping[str, Any]],
    data_wait_share: float,
    dominant: str,
) -> dict[str, Any]:
    axes = _matrix_axes(benchmark, source_fixture)
    rebuild_actions = _dataloader_rebuild_actions(actions)
    latest = rebuild_actions[-1] if rebuild_actions else {}
    status = str(latest.get("status") or "")
    gain_pct = _safe_float(latest.get("steady_samples_per_second_gain_pct"))
    transient_data_wait = max(
        (
            _safe_float(action.get("profiler_handoff_data_wait_share"), -1.0)
            for action in actions
        ),
        default=-1.0,
    )
    transient_data_wait = max(
        transient_data_wait,
        max((_safe_float(action.get("before_data_wait_share"), -1.0) for action in actions), default=-1.0),
    )
    steady_data_wait = _safe_float(data_wait_share)
    reasons: list[str] = []
    hypotheses: list[dict[str, Any]] = []
    next_probes: list[dict[str, Any]] = []
    if status == "rolled_back" or gain_pct < 0.0:
        reasons.append("dataloader_rebuild_rolled_back_or_regressed")
        if axes["steps"] and axes["steps"] <= 24:
            hypotheses.append(
                {
                    "id": "worker_startup_overhead_dominates_short_window",
                    "reason": "short training windows can measure worker spawn/warmup more than steady decode overlap",
                }
            )
            next_probes.append(
                {
                    "id": "longer_window",
                    "change": "increase steps to 32-48 and compare post-warmup windows",
                }
            )
        if axes["samples"] and axes["samples"] <= max(_safe_int(latest.get("next_workers"), 2) * 2, 4):
            hypotheses.append(
                {
                    "id": "tiny_dataset_reuse_limits_worker_pipeline",
                    "reason": "few source samples can make multiprocessing overhead larger than the data_wait it hides",
                }
            )
            next_probes.append(
                {
                    "id": "more_source_samples",
                    "change": "use 8-16 heavy raw samples to reduce repeated tiny-dataset effects",
                }
            )
        if axes["resolution"] and axes["resolution"] <= 256 and axes["fixture_width"] >= 2048:
            hypotheses.append(
                {
                    "id": "large_decode_small_train_resolution_mismatch",
                    "reason": "4096px PNG decode plus downscale to a tiny training resolution can stress CPU/IPC without adding GPU work",
                }
            )
            next_probes.append(
                {
                    "id": "resolution_sweep",
                    "change": "run 512/1024 variants to see whether more GPU work amortizes decode overhead",
                }
            )
        if _safe_int(latest.get("next_prefetch_factor"), 0) <= 2:
            hypotheses.append(
                {
                    "id": "prefetch_depth_may_be_too_shallow_or_late",
                    "reason": "workers2 with prefetch2 may not hide high-entropy PNG decode latency after an epoch-boundary rebuild",
                }
            )
            next_probes.append(
                {
                    "id": "prefetch_axis",
                    "change": "compare workers2/prefetch4 as a next-run probe, especially off Windows",
                }
            )
        if axes["pin_memory"]:
            next_probes.append(
                {
                    "id": "pin_memory_axis",
                    "change": "compare the same source fixture with --no-pin-memory to separate data_wait from H2D staging effects",
                }
            )
    elif status == "kept":
        reasons.append("dataloader_rebuild_kept")
    elif transient_data_wait >= 0.08 and steady_data_wait < 0.08:
        reasons.append("transient_data_wait_resolved_in_steady_window")
        hypotheses.append(
            {
                "id": "workers_prefetch_hide_steady_decode_wait",
                "reason": "early handoff saw data_wait, but steady summary fell below threshold after workers/prefetch warmed up",
            }
        )
        next_probes.append(
            {
                "id": "compare_workers0_baseline",
                "change": "run the matching workers0 fixture and compare steady data_wait_share against this prewarmed policy",
            }
        )
        if axes["steps"] and axes["steps"] <= 16:
            next_probes.append(
                {
                    "id": "longer_window",
                    "change": "repeat with 32+ steps to separate startup spikes from steady throughput",
                }
            )
    elif dominant == "data_bound" and data_wait_share >= 0.08:
        reasons.append("natural_data_wait_seen_without_rebuild_decision")
        hypotheses.append(
            {
                "id": "data_wait_visible_after_host_profiler_action_but_no_rebuild",
                "reason": "steady summary is data_bound, but no DataLoader rebuild action was observed in this run",
            }
        )
        next_probes.append(
            {
                "id": "prewarmed_workers_prefetch_axis",
                "change": "run a matching workers2/prefetch4 next-run probe and compare steady data_wait_share",
            }
        )
        next_probes.append(
            {
                "id": "longer_or_more_frequent_controller_window",
                "change": "increase steps or shorten tune interval so post-profiler data_wait can arm a second action",
            }
        )
    else:
        reasons.append("no_dataloader_rebuild_regression_to_explain")

    return {
        "schema_version": 1,
        "analysis": "bubble_natural_data_wait_analysis_v0",
        "matrix_axes": axes,
        "transient_data_wait_share": _round(transient_data_wait if transient_data_wait > 0.0 else 0.0),
        "steady_data_wait_share": _round(steady_data_wait),
        "transient_data_wait_resolved": bool(transient_data_wait >= 0.08 and steady_data_wait < 0.08),
        "latest_dataloader_rebuild_status": status,
        "latest_dataloader_rebuild_gain_pct": _round(gain_pct, 4),
        "reasons": reasons[:8],
        "rollback_hypotheses": hypotheses[:8],
        "next_probe_suggestions": next_probes[:8],
    }


def build_bubble_natural_data_wait_evidence_report(
    summary: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    source_fixture: Mapping[str, Any] | None = None,
    case_id: str = "",
    preferred_label: str = "standard",
    data_wait_threshold: float = 0.08,
) -> dict[str, Any]:
    """Classify a benchmark summary as natural data_wait evidence or a negative probe.

    The report intentionally blocks benchmark-only stall and controlled overlays,
    so release claims cannot accidentally cite injected data_wait evidence.
    """

    benchmark = _mapping(summary.get("benchmark"))
    run = _first_run(summary, preferred_label)
    bubble = _bubble_profile(run)
    evidence = _mapping(bubble.get("evidence"))
    injections = _benchmark_injection(summary, run)
    cache_probe_only_reason = _cache_probe_only_reason(benchmark, source_fixture)
    injection_blockers = [
        key
        for key in ("controlled_data_wait", "controlled_rollback", "benchmark_data_wait_stall")
        if injections.get(key)
    ]
    actions = _action_chain(manifest)
    loss_stability = _loss_stability(run)
    data_wait_share = _safe_float(evidence.get("data_wait_share"))
    dominant = str(bubble.get("dominant_bottleneck") or "").strip()
    natural_candidate = not injection_blockers and dominant == "data_bound" and data_wait_share >= float(data_wait_threshold)
    dataloader_observed = natural_candidate and _has_dataloader_rebuild_observation(actions)
    if injection_blockers:
        status = "blocked_benchmark_injection"
        reasons = injection_blockers
    elif dataloader_observed:
        status = "natural_dataloader_rebuild_observed"
        reasons = ["natural_data_wait_and_dataloader_rebuild_observed"]
    elif natural_candidate:
        status = "natural_data_wait_candidate"
        reasons = ["natural_data_wait_detected_without_rebuild_observation"]
    else:
        status = "no_natural_data_wait"
        reasons = ["data_wait_below_threshold_or_not_dominant"]

    return {
        "schema_version": 1,
        "report": NATURAL_DATA_WAIT_EVIDENCE_REPORT,
        "status": status,
        "case_id": str(case_id or benchmark.get("family") or "bubble_natural_data_wait"),
        "family": str(benchmark.get("family") or run.get("family") or ""),
        "profile_label": str(run.get("label") or preferred_label),
        "steps_completed": _safe_int(run.get("steps_completed"), 0),
        "natural_data_wait_threshold": round(float(data_wait_threshold), 6),
        "metrics": {
            "dominant_bottleneck": dominant,
            "bubble_ratio_estimate": _round(bubble.get("bubble_ratio_estimate")),
            "data_wait_share": _round(data_wait_share),
            "h2d_transfer_share": _round(evidence.get("h2d_transfer_share")),
            "optimizer_share": _round(evidence.get("optimizer_share")),
            "host_gap_share": _round(evidence.get("host_gap_share")),
            "steady_samples_per_second": _round(run.get("steady_samples_per_second")),
            "steady_mean_step_ms": _round(run.get("steady_mean_step_ms"), 4),
            "peak_vram_mb": _round(run.get("peak_vram_mb"), 3),
            "final_loss": loss_stability.get("final_loss"),
        },
        "loss_stability": loss_stability,
        "benchmark_injection": injections,
        "benchmark_injection_blockers": injection_blockers,
        "action_count": len(actions),
        "action_chain": actions[-10:],
        "analysis": _rollback_analysis(
            benchmark=benchmark,
            source_fixture=source_fixture,
            actions=actions,
            data_wait_share=data_wait_share,
            dominant=dominant,
        ),
        "release_claim": {
            "eligible": bool(dataloader_observed and not cache_probe_only_reason),
            "scope": "case_specific_natural_data_wait" if dataloader_observed and not cache_probe_only_reason else "not_eligible",
            "reason": cache_probe_only_reason
            or "requires non-injected data_wait plus DataLoader rebuild keep/rollback evidence",
        },
        "decision": {
            "reasons": [*reasons, *([cache_probe_only_reason] if cache_probe_only_reason else [])],
            "natural_candidate": bool(natural_candidate),
            "dataloader_rebuild_observed": bool(dataloader_observed),
            "cache_probe_only": bool(cache_probe_only_reason),
        },
    }


__all__ = [
    "NATURAL_DATA_WAIT_EVIDENCE_REPORT",
    "build_bubble_natural_data_wait_evidence_report",
]
