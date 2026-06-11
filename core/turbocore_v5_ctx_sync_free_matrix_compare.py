"""Ctx-sync-free canary comparison helpers for TurboCore V5 matrix output."""

from __future__ import annotations

from typing import Any


CTX_SYNC_FREE_POLICY = "borrowed_stream_event_chain_no_ctx_sync"
CONTEXT_SYNC_POLICY = "cuCtxSynchronize_after_native_step"


def summarize_ctx_sync_free_matrix(executed_entries: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = build_ctx_sync_free_comparison(executed_entries)
    return {
        "native_dispatch_ctx_sync_free_comparison": comparison,
        "native_dispatch_ctx_sync_free_speedup_vs_baseline": comparison.get("ctx_sync_free_speedup_vs_baseline"),
        "native_dispatch_ctx_sync_free_speedup_vs_context_sync_native": comparison.get(
            "ctx_sync_free_speedup_vs_context_sync_native"
        ),
        "native_dispatch_context_sync_speedup_vs_baseline": comparison.get("context_sync_speedup_vs_baseline"),
        "native_dispatch_ctx_sync_free_representative_candidate_ready": False,
        "native_dispatch_ctx_sync_free_promotion_priority_unchanged": True,
    }


def build_ctx_sync_free_comparison(executed_entries: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = _entry_by_name(executed_entries, "baseline_phase")
    context_sync = _native_sync_entry(executed_entries, CONTEXT_SYNC_POLICY)
    ctx_sync_free = _native_sync_entry(executed_entries, CTX_SYNC_FREE_POLICY)

    blocked: list[str] = []
    if not baseline:
        blocked.append("baseline_phase_missing")
    if not context_sync:
        blocked.append("context_sync_native_case_missing")
    if not ctx_sync_free:
        blocked.append("ctx_sync_free_canary_case_missing")

    baseline_ms = _step_ms(baseline)
    context_sync_ms = _step_ms(context_sync)
    ctx_sync_free_ms = _step_ms(ctx_sync_free)
    if baseline and baseline_ms <= 0:
        blocked.append("baseline_step_ms_missing")
    if context_sync and context_sync_ms <= 0:
        blocked.append("context_sync_native_step_ms_missing")
    if ctx_sync_free and ctx_sync_free_ms <= 0:
        blocked.append("ctx_sync_free_canary_step_ms_missing")

    ready = not blocked
    speedup_vs_baseline = _speedup(baseline_ms, ctx_sync_free_ms) if ready else None
    speedup_vs_context = _speedup(context_sync_ms, ctx_sync_free_ms) if ready else None
    context_speedup = _speedup(baseline_ms, context_sync_ms) if ready else None
    recommendation = _recommendation(speedup_vs_context) if ready else "collect full three-case matrix evidence"

    return {
        "ready": ready,
        "metric": "steady_mean_step_ms_preferred",
        "baseline_case": _case_name(baseline),
        "context_sync_case": _case_name(context_sync),
        "ctx_sync_free_case": _case_name(ctx_sync_free),
        "baseline_step_ms": _round_or_none(baseline_ms),
        "context_sync_step_ms": _round_or_none(context_sync_ms),
        "ctx_sync_free_step_ms": _round_or_none(ctx_sync_free_ms),
        "ctx_sync_free_speedup_vs_baseline": speedup_vs_baseline,
        "ctx_sync_free_speedup_vs_context_sync_native": speedup_vs_context,
        "context_sync_speedup_vs_baseline": context_speedup,
        "ctx_sync_free_canary_executed": bool(ctx_sync_free),
        "representative_candidate_ready": False,
        "representative_candidate_blocked_reasons": [
            "ctx_sync_free_canary_is_not_representative_promotion_case",
            "manual_review_required_for_representative_case_change",
        ],
        "promotion_priority_unchanged": True,
        "recommended_next_step": recommendation,
        "blocked_reasons": blocked,
    }


def _recommendation(speedup_vs_context: float | None) -> str:
    if speedup_vs_context is not None and speedup_vs_context >= 1.03:
        return "collect replicate evidence before any manual representative-case review"
    return "keep ctx-sync-free as canary and optimize borrowed-stream overhead"


def _entry_by_name(entries: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for entry in entries:
        if _case_name(entry) == name:
            return entry
    return {}


def _native_sync_entry(entries: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    preferred = "native_update_dispatch_promotion_perf" if policy == CONTEXT_SYNC_POLICY else ""
    fallback: dict[str, Any] = {}
    for entry in entries:
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
        if not bool(summary.get("native_dispatch_executed", False)):
            continue
        if str(summary.get("native_dispatch_owner_native_runtime_synchronization", "") or "") != policy:
            continue
        if preferred and _case_name(entry) == preferred:
            return entry
        if not fallback:
            fallback = entry
    return fallback


def _case_name(entry: dict[str, Any]) -> str:
    case = entry.get("case") if isinstance(entry.get("case"), dict) else {}
    return str(case.get("name", "") or "")


def _step_ms(entry: dict[str, Any]) -> float:
    summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
    steady = float(summary.get("steady_mean_step_ms", 0.0) or 0.0)
    if steady > 0:
        return steady
    return float(summary.get("mean_step_ms", 0.0) or 0.0)


def _speedup(baseline_ms: float, candidate_ms: float) -> float | None:
    if baseline_ms <= 0 or candidate_ms <= 0:
        return None
    return round(baseline_ms / candidate_ms, 4)


def _round_or_none(value: float) -> float | None:
    return round(value, 4) if value > 0 else None
