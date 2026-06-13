"""Loss-curve A/B evidence for Newbie BlockSkip follow-up runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_newbie_blockskip_loss_curve_ab_evidence_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
FAMILY = "newbie"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _losses(summary: Mapping[str, Any], label: str = "standard") -> list[float]:
    runs = _mapping(summary.get("runs"))
    run = _mapping(runs.get(label))
    if not run:
        for item in runs.values():
            run = _mapping(item)
            if run:
                break
    raw = run.get("losses")
    if not isinstance(raw, list):
        return []
    return [_safe_float(item, float("nan")) for item in raw]


def _finite_losses(values: Sequence[float]) -> list[float]:
    return [float(item) for item in values if math.isfinite(float(item))]


def _mean(values: Sequence[float]) -> float:
    finite = _finite_losses(values)
    return sum(finite) / len(finite) if finite else 0.0


def _tail(values: Sequence[float], fraction: float = 0.25) -> list[float]:
    if not values:
        return []
    count = max(1, int(math.ceil(len(values) * max(min(fraction, 1.0), 0.0))))
    return list(values[-count:])


def _non_finite_count(values: Sequence[float]) -> int:
    return sum(1 for item in values if not math.isfinite(float(item)))


def _pair_row(
    *,
    pair_id: str,
    baseline_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    max_tail_mean_delta: float,
) -> dict[str, Any]:
    baseline = _losses(baseline_summary)
    candidate = _losses(candidate_summary)
    aligned = min(len(baseline), len(candidate))
    baseline_aligned = baseline[:aligned]
    candidate_aligned = candidate[:aligned]
    baseline_tail = _tail(baseline_aligned)
    candidate_tail = _tail(candidate_aligned)
    baseline_tail_mean = _mean(baseline_tail)
    candidate_tail_mean = _mean(candidate_tail)
    tail_delta = candidate_tail_mean - baseline_tail_mean
    final_delta = (_safe_float(candidate_aligned[-1]) - _safe_float(baseline_aligned[-1])) if aligned else 0.0
    mean_delta = _mean(candidate_aligned) - _mean(baseline_aligned) if aligned else 0.0
    blockers: list[str] = []
    if aligned <= 0:
        blockers.append("loss_curve_missing")
    if len(baseline) != len(candidate):
        blockers.append("loss_curve_length_mismatch")
    if _non_finite_count(baseline) or _non_finite_count(candidate):
        blockers.append("non_finite_loss_observed")
    if abs(tail_delta) > float(max_tail_mean_delta):
        blockers.append("tail_mean_loss_delta_above_threshold")
    return {
        "pair_id": pair_id,
        "loss_count_baseline": len(baseline),
        "loss_count_candidate": len(candidate),
        "aligned_loss_count": aligned,
        "baseline_initial_loss": _round(baseline_aligned[0] if aligned else 0.0),
        "candidate_initial_loss": _round(candidate_aligned[0] if aligned else 0.0),
        "baseline_final_loss": _round(baseline_aligned[-1] if aligned else 0.0),
        "candidate_final_loss": _round(candidate_aligned[-1] if aligned else 0.0),
        "final_loss_delta": _round(final_delta),
        "baseline_tail_mean_loss": _round(baseline_tail_mean),
        "candidate_tail_mean_loss": _round(candidate_tail_mean),
        "tail_mean_loss_delta": _round(tail_delta),
        "mean_loss_delta": _round(mean_delta),
        "non_finite_loss_count_baseline": _non_finite_count(baseline),
        "non_finite_loss_count_candidate": _non_finite_count(candidate),
        "ok": not blockers,
        "blocked_reasons": blockers,
    }


def _digest(rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> str:
    payload = json.dumps({"rows": list(rows), "thresholds": dict(thresholds)}, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_newbie_blockskip_loss_curve_ab_evidence(
    pairs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    candidate: str = "dit_compute_reducer:blockskip_skip25",
    max_tail_mean_delta: float = 0.25,
) -> dict[str, Any]:
    """Build fail-closed loss-curve evidence from paired summary JSON payloads."""

    rows = [
        _pair_row(
            pair_id=str(pair_id),
            baseline_summary=baseline,
            candidate_summary=candidate,
            max_tail_mean_delta=float(max_tail_mean_delta),
        )
        for pair_id, baseline, candidate in pairs
    ]
    blockers: list[str] = []
    if not rows:
        blockers.append("loss_curve_pair_rows_missing")
    blockers.extend(
        f"{row['pair_id']}:{reason}"
        for row in rows
        for reason in row.get("blocked_reasons", [])
    )
    thresholds = {"max_tail_mean_delta": float(max_tail_mean_delta)}
    ready = bool(rows) and not blockers
    max_abs_tail_delta = max((abs(_safe_float(row.get("tail_mean_loss_delta"))) for row in rows), default=0.0)
    max_abs_final_delta = max((abs(_safe_float(row.get("final_loss_delta"))) for row in rows), default=0.0)
    artifact_digest = _digest(rows, thresholds)
    return {
        "report": REPORT,
        "schema_version": 1,
        "roadmap": ROADMAP,
        "family": FAMILY,
        "candidate": str(candidate or "dit_compute_reducer:blockskip_skip25"),
        "review_type": "loss_curve_ab",
        "status": "loss_curve_ab_ready_nonrelease" if ready else "loss_curve_ab_blocked",
        "review_ready": ready,
        "summary": {
            "pair_count": len(rows),
            "ready_pair_count": sum(1 for row in rows if row.get("ok")),
            "max_abs_tail_mean_loss_delta": _round(max_abs_tail_delta),
            "max_abs_final_loss_delta": _round(max_abs_final_delta),
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "thresholds": thresholds,
        "rows": rows,
        "loss_curve_delta": _round(max_abs_tail_delta),
        "max_loss_curve_delta": float(max_tail_mean_delta),
        "quality_drift": _round(max_abs_tail_delta),
        "max_quality_drift": float(max_tail_mean_delta),
        "shape_stable": False,
        "disabled_parity_ok": False,
        "checkpoint_semantics_ok": False,
        "residual_reuse_parity_ok": False,
        "reviewer": "json_loss_curve_ab_builder",
        "artifact_digest": artifact_digest,
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
    }


__all__ = [
    "FAMILY",
    "REPORT",
    "ROADMAP",
    "build_newbie_blockskip_loss_curve_ab_evidence",
]
