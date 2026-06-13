"""JSON-only BlockSkip quality/stability review for Newbie GPU-bubble evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_newbie_blockskip_quality_stability_review_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
DEFAULT_MIN_SEED_PAIRS = 2
DEFAULT_MIN_THROUGHPUT_GAIN_PCT = 3.0
DEFAULT_MAX_ABS_LOSS_DELTA = 0.25


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _case_id(payload: Mapping[str, Any]) -> str:
    return str(payload.get("case_id") or "")


def _seed_from_case_id(case_id: str, default: int = 1337) -> int:
    marker = "seed"
    lowered = case_id.lower()
    if marker not in lowered:
        return default
    suffix = lowered.split(marker, 1)[1]
    digits = []
    for char in suffix:
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    return int("".join(digits)) if digits else default


def _metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("metrics"))


def _loss(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("loss_stability"))


def _row(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _metrics(payload)
    loss = _loss(payload)
    case_id = _case_id(payload)
    return {
        "case_id": case_id,
        "seed": _seed_from_case_id(case_id),
        "status": str(payload.get("status") or ""),
        "dominant_bottleneck": str(metrics.get("dominant_bottleneck") or ""),
        "data_wait_share": _round(metrics.get("data_wait_share")),
        "steady_mean_step_ms": _round(metrics.get("steady_mean_step_ms"), 4),
        "steady_samples_per_second": _round(metrics.get("steady_samples_per_second"), 6),
        "final_loss": _round(loss.get("final_loss"), 6),
        "non_finite_loss_count": _safe_int(loss.get("non_finite_loss_count")),
        "steps_completed": _safe_int(payload.get("steps_completed")),
        "release_claim_eligible": bool(_mapping(payload.get("release_claim")).get("eligible", False)),
    }


def build_newbie_blockskip_quality_stability_review(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    candidate: str = "dit_compute_reducer:blockskip_skip25",
    min_seed_pairs: int = DEFAULT_MIN_SEED_PAIRS,
    min_throughput_gain_pct: float = DEFAULT_MIN_THROUGHPUT_GAIN_PCT,
    max_abs_loss_delta: float = DEFAULT_MAX_ABS_LOSS_DELTA,
) -> dict[str, Any]:
    """Build a fail-closed review from paired baseline and BlockSkip evidence."""

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    signs: set[int] = set()
    for baseline_payload, candidate_payload in pairs:
        baseline = _row(baseline_payload)
        candidate_row = _row(candidate_payload)
        seed = candidate_row["seed"] or baseline["seed"]
        baseline_step = _safe_float(baseline["steady_mean_step_ms"])
        candidate_step = _safe_float(candidate_row["steady_mean_step_ms"])
        baseline_sps = _safe_float(baseline["steady_samples_per_second"])
        candidate_sps = _safe_float(candidate_row["steady_samples_per_second"])
        loss_delta = _safe_float(candidate_row["final_loss"]) - _safe_float(baseline["final_loss"])
        if loss_delta > 0:
            signs.add(1)
        elif loss_delta < 0:
            signs.add(-1)
        step_reduction_pct = (
            ((baseline_step - candidate_step) / baseline_step) * 100.0 if baseline_step > 0.0 else 0.0
        )
        throughput_gain_pct = (
            ((candidate_sps - baseline_sps) / baseline_sps) * 100.0 if baseline_sps > 0.0 else 0.0
        )
        pair_blockers: list[str] = []
        if baseline["status"] != "no_natural_data_wait" or candidate_row["status"] != "no_natural_data_wait":
            pair_blockers.append("unexpected_natural_data_wait_status")
        if baseline["dominant_bottleneck"] != "compute_bound" or candidate_row["dominant_bottleneck"] != "compute_bound":
            pair_blockers.append("pair_not_compute_bound")
        if throughput_gain_pct < float(min_throughput_gain_pct):
            pair_blockers.append("throughput_gain_below_threshold")
        if baseline["non_finite_loss_count"] or candidate_row["non_finite_loss_count"]:
            pair_blockers.append("non_finite_loss_observed")
        if abs(loss_delta) > float(max_abs_loss_delta):
            pair_blockers.append("loss_delta_outside_review_band")
        if baseline["release_claim_eligible"] or candidate_row["release_claim_eligible"]:
            pair_blockers.append("unexpected_release_claim_eligible_pair")
        rows.append(
            {
                "seed": seed,
                "baseline": baseline,
                "candidate": candidate_row,
                "step_reduction_pct": _round(step_reduction_pct, 2),
                "throughput_gain_pct": _round(throughput_gain_pct, 2),
                "loss_delta": _round(loss_delta, 6),
                "abs_loss_delta": _round(abs(loss_delta), 6),
                "pair_blockers": pair_blockers,
                "throughput_candidate": throughput_gain_pct >= float(min_throughput_gain_pct),
                "loss_quality_ok": not any(
                    blocker in pair_blockers
                    for blocker in ("non_finite_loss_observed", "loss_delta_outside_review_band")
                ),
            }
        )
    completed_pair_count = len(rows)
    throughput_ready = completed_pair_count >= int(min_seed_pairs) and all(
        bool(row.get("throughput_candidate")) for row in rows
    )
    quality_ready = completed_pair_count >= int(min_seed_pairs) and all(
        bool(row.get("loss_quality_ok")) for row in rows
    )
    if completed_pair_count < int(min_seed_pairs):
        blockers.append("repeat_seed_pair_count_below_threshold")
    if not throughput_ready:
        blockers.append("throughput_repeat_gate_not_ready")
    if not quality_ready:
        blockers.append("loss_quality_gate_not_ready")
    if len(signs) > 1:
        blockers.append("loss_delta_direction_inconsistent")
    pair_blocker_ids = sorted({str(blocker) for row in rows for blocker in row.get("pair_blockers", [])})
    blockers.extend(blocker for blocker in pair_blocker_ids if blocker not in blockers)
    status = (
        "blockskip_repeat_throughput_candidate_quality_blocked"
        if throughput_ready and blockers
        else "blockskip_repeat_quality_review_ready"
        if not blockers
        else "blockskip_repeat_evidence_incomplete"
    )
    return {
        "report": REPORT,
        "schema_version": 1,
        "roadmap": ROADMAP,
        "family": "newbie",
        "candidate": str(candidate or "dit_compute_reducer:blockskip_skip25"),
        "status": status,
        "completed_seed_pair_count": completed_pair_count,
        "required_seed_pair_count": int(min_seed_pairs),
        "thresholds": {
            "min_throughput_gain_pct": float(min_throughput_gain_pct),
            "max_abs_loss_delta": float(max_abs_loss_delta),
        },
        "summary": {
            "throughput_repeat_ready": throughput_ready,
            "loss_quality_ready": quality_ready,
            "min_throughput_gain_pct_observed": _round(
                min([row["throughput_gain_pct"] for row in rows], default=0.0),
                2,
            ),
            "max_abs_loss_delta_observed": _round(
                max([row["abs_loss_delta"] for row in rows], default=0.0),
                6,
            ),
            "loss_delta_signs": sorted(signs),
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "rows": rows,
        "next_actions": [
            {
                "id": "run_blockskip_quality_drift_ab_or_render_review",
                "kind": "quality_gate_followup",
                "reason": "dual-seed throughput is positive but loss deltas are outside the review band or directionally inconsistent",
                "requires_gpu_heavy_run": True,
                "release_claim_allowed": False,
                "release_claim_allowed_after_success": False,
                "safe_to_auto_start": False,
            },
            {
                "id": "review_newbie_natural_load_gate_semantics",
                "kind": "gate_semantics_review",
                "reason": "BlockSkip is a compute-path candidate and does not resolve the Newbie natural-load canary gate",
                "requires_gpu_heavy_run": False,
                "release_claim_allowed": False,
                "release_claim_allowed_after_success": False,
                "safe_to_auto_start": False,
            },
        ],
        "release_claim": {
            "eligible": False,
            "reason": "BlockSkip repeat evidence is non-release until quality drift and Newbie natural-load gate semantics are resolved",
            "scope": "not_eligible",
        },
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
    }


__all__ = [
    "DEFAULT_MAX_ABS_LOSS_DELTA",
    "DEFAULT_MIN_SEED_PAIRS",
    "DEFAULT_MIN_THROUGHPUT_GAIN_PCT",
    "REPORT",
    "ROADMAP",
    "build_newbie_blockskip_quality_stability_review",
]
