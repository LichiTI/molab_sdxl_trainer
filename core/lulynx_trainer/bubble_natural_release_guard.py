"""Fail-closed gates for natural data-wait release evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_DATA_WAIT_THRESHOLD = 0.08
DEFAULT_MIN_THROUGHPUT_GAIN_PCT = 3.0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _release_claim_reasons(report: Mapping[str, Any], expected_scope: str) -> list[str]:
    release_claim = _mapping(report.get("release_claim"))
    scope = str(release_claim.get("scope") or "")
    reasons: list[str] = []
    if not _safe_bool(release_claim.get("eligible"), False):
        reasons.append("release_claim_not_eligible")
    if not scope.startswith(expected_scope):
        reasons.append("case_specific_scope_missing")
    return reasons


def _benchmark_blocker_reasons(report: Mapping[str, Any]) -> list[str]:
    blockers = _string_list(report.get("benchmark_injection_blockers"))
    return ["benchmark_injection_blockers_present"] if blockers else []


def _diagnostic_reasons(report: Mapping[str, Any]) -> list[str]:
    return ["diagnostic_only_evidence"] if _safe_bool(report.get("diagnostic_only"), False) else []


def _loss_observed_status(report: Mapping[str, Any]) -> str:
    return str(_mapping(report.get("loss_stability")).get("status") or "")


def _has_epoch_boundary_dataloader_rebuild(report: Mapping[str, Any]) -> bool:
    decision = _mapping(report.get("decision"))
    if _safe_bool(decision.get("dataloader_rebuild_observed"), False):
        return True
    actions = report.get("action_chain")
    chain = actions if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)) else []
    for item in chain:
        action = _mapping(item)
        if str(action.get("action_kind") or "") != "set_dataloader_workers":
            continue
        if str(action.get("adapter_id") or "") != "dataloader_rebuild_runtime_contract_v0":
            continue
        if str(action.get("apply_boundary") or "") in {"epoch_start", "epoch_boundary"}:
            return True
    return False


def natural_data_wait_release_reasons(
    report: Mapping[str, Any],
    *,
    data_wait_threshold: float = DEFAULT_DATA_WAIT_THRESHOLD,
) -> list[str]:
    """Return blockers for a single natural data-wait closed-loop report."""

    reasons = [
        *_release_claim_reasons(report, "case_specific_natural_data_wait"),
        *_benchmark_blocker_reasons(report),
        *_diagnostic_reasons(report),
    ]
    if str(report.get("status") or "") != "natural_dataloader_rebuild_observed":
        reasons.append("natural_data_wait_status_not_observed")
    if not _has_epoch_boundary_dataloader_rebuild(report):
        reasons.append("dataloader_rebuild_epoch_boundary_action_missing")

    loss_status = _loss_observed_status(report)
    if loss_status not in {"observed", "stable"}:
        reasons.append(f"loss_stability_{loss_status or 'missing'}")

    metrics = _mapping(report.get("metrics"))
    if _safe_float(metrics.get("data_wait_share")) < max(float(data_wait_threshold or 0.0), 0.0):
        reasons.append("natural_data_wait_below_threshold")
    return sorted(set(reasons))


def natural_data_wait_release_eligible(
    report: Mapping[str, Any],
    *,
    data_wait_threshold: float = DEFAULT_DATA_WAIT_THRESHOLD,
) -> bool:
    return not natural_data_wait_release_reasons(report, data_wait_threshold=data_wait_threshold)


def natural_ab_release_reasons(
    report: Mapping[str, Any],
    *,
    data_wait_threshold: float = DEFAULT_DATA_WAIT_THRESHOLD,
    min_throughput_gain_pct: float = DEFAULT_MIN_THROUGHPUT_GAIN_PCT,
) -> list[str]:
    """Return blockers for natural data-wait before/after evidence."""

    reasons = [
        *_release_claim_reasons(report, "case_specific_natural_data_wait_ab"),
        *_benchmark_blocker_reasons(report),
        *_diagnostic_reasons(report),
    ]
    if str(report.get("status") or "") != "keep_recommended":
        reasons.append("natural_ab_status_not_keep_recommended")

    loss_status = _loss_observed_status(report)
    if loss_status != "stable":
        reasons.append(f"loss_stability_{loss_status or 'missing'}")

    action = _mapping(report.get("action"))
    if str(action.get("action_kind") or "") != "next_run_dataloader_workers_prefetch":
        reasons.append("natural_ab_action_kind_missing")
    after_action = _mapping(action.get("after"))
    if _safe_float(after_action.get("dataloader_workers"), -1.0) < 0.0:
        reasons.append("natural_ab_after_workers_missing")

    comparison = _mapping(report.get("comparison"))
    before_metrics = _mapping(_mapping(report.get("before")).get("metrics"))
    after_metrics = _mapping(_mapping(report.get("after")).get("metrics"))
    before_wait = _safe_float(comparison.get("data_wait_share_before"), _safe_float(before_metrics.get("data_wait_share")))
    after_wait = _safe_float(comparison.get("data_wait_share_after"), _safe_float(after_metrics.get("data_wait_share")))
    gain_pct = _safe_float(comparison.get("steady_samples_per_second_gain_pct"))
    threshold = max(float(data_wait_threshold or 0.0), 0.0)
    min_gain = max(float(min_throughput_gain_pct or 0.0), 0.0)

    if before_wait < threshold:
        reasons.append("before_data_wait_below_threshold")
    if after_wait >= threshold:
        reasons.append("after_data_wait_not_below_threshold")
    if after_wait >= before_wait:
        reasons.append("data_wait_not_reduced")
    if gain_pct < min_gain:
        reasons.append("throughput_gain_below_threshold")
    return sorted(set(reasons))


def natural_ab_release_eligible(
    report: Mapping[str, Any],
    *,
    data_wait_threshold: float = DEFAULT_DATA_WAIT_THRESHOLD,
    min_throughput_gain_pct: float = DEFAULT_MIN_THROUGHPUT_GAIN_PCT,
) -> bool:
    return not natural_ab_release_reasons(
        report,
        data_wait_threshold=data_wait_threshold,
        min_throughput_gain_pct=min_throughput_gain_pct,
    )


__all__ = [
    "DEFAULT_DATA_WAIT_THRESHOLD",
    "DEFAULT_MIN_THROUGHPUT_GAIN_PCT",
    "natural_ab_release_eligible",
    "natural_ab_release_reasons",
    "natural_data_wait_release_eligible",
    "natural_data_wait_release_reasons",
]
