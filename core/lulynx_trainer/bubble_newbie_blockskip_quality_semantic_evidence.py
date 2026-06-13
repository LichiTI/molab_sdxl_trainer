"""Semantic quality evidence for Newbie BlockSkip follow-up reviews."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


REPORT = "bubble_newbie_blockskip_quality_semantic_evidence_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
FAMILY = "newbie"
CANDIDATE = "dit_compute_reducer:blockskip_skip25"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _benchmark(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(summary.get("benchmark"))


def _pair_row(
    *,
    pair_id: str,
    baseline_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _benchmark(baseline_summary)
    candidate = _benchmark(candidate_summary)
    baseline_strategy = str(baseline.get("dit_compute_reducer_strategy") or "none")
    candidate_strategy = str(candidate.get("dit_compute_reducer_strategy") or "none")
    baseline_checkpoint_policy = str(baseline.get("checkpoint_policy") or "")
    candidate_checkpoint_policy = str(candidate.get("checkpoint_policy") or "")
    baseline_newbie_checkpointing = bool(baseline.get("newbie_block_checkpointing"))
    candidate_newbie_checkpointing = bool(candidate.get("newbie_block_checkpointing"))
    blockers: list[str] = []
    if baseline_strategy != "none":
        blockers.append("baseline_reducer_not_disabled")
    if candidate_strategy != "blockskip":
        blockers.append("candidate_reducer_not_blockskip")
    if baseline_checkpoint_policy != "off" or candidate_checkpoint_policy != "off":
        blockers.append("checkpoint_policy_not_off_for_pair")
    if baseline_newbie_checkpointing or candidate_newbie_checkpointing:
        blockers.append("newbie_block_checkpointing_enabled_for_pair")
    return {
        "pair_id": str(pair_id),
        "baseline_strategy": baseline_strategy,
        "candidate_strategy": candidate_strategy,
        "baseline_checkpoint_policy": baseline_checkpoint_policy,
        "candidate_checkpoint_policy": candidate_checkpoint_policy,
        "baseline_newbie_block_checkpointing": baseline_newbie_checkpointing,
        "candidate_newbie_block_checkpointing": candidate_newbie_checkpointing,
        "checkpoint_semantics_ok": not blockers,
        "blocked_reasons": blockers,
    }


def _newbie_cached_token_row(cached_token_ab: Mapping[str, Any]) -> Mapping[str, Any]:
    for row in _sequence(cached_token_ab.get("family_reports")):
        payload = _mapping(row)
        if str(payload.get("family") or "").strip().lower() == "newbie":
            return payload
    return {}


def build_newbie_blockskip_quality_semantic_evidence(
    *,
    loss_curve_evidence: Mapping[str, Any],
    cached_token_ab: Mapping[str, Any],
    pairs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build a fail-closed semantic evidence packet for the quality drift review."""

    loss_curve = _mapping(loss_curve_evidence)
    cached = _mapping(cached_token_ab)
    newbie_row = _newbie_cached_token_row(cached)
    pair_rows = [
        _pair_row(pair_id=str(pair_id), baseline_summary=baseline, candidate_summary=candidate)
        for pair_id, baseline, candidate in pairs
    ]
    cached_blockers = [str(item) for item in _sequence(cached.get("blocked_reasons")) if str(item)]
    row_blockers = [
        f"{row['pair_id']}:{reason}"
        for row in pair_rows
        for reason in _sequence(row.get("blocked_reasons"))
    ]

    plan = _mapping(newbie_row.get("plan"))
    policy = _mapping(plan.get("policy"))
    shape_stable = bool(newbie_row.get("ok")) and "shape_stable_missing" not in _sequence(
        newbie_row.get("blocked_reasons")
    )
    disabled_parity_ok = bool(newbie_row.get("disabled_parity_ok"))
    checkpoint_semantics_ok = bool(pair_rows) and all(bool(row.get("checkpoint_semantics_ok")) for row in pair_rows)
    residual_reuse_parity_ok = (
        bool(newbie_row.get("ok"))
        and bool(plan.get("enabled"))
        and _safe_int(plan.get("skipped_blocks")) > 0
        and bool(policy.get("reuse_residual"))
    )

    blockers: list[str] = []
    if not bool(loss_curve.get("review_ready")):
        blockers.append("loss_curve_evidence_not_ready")
    if bool(loss_curve.get("release_claim_allowed")) or bool(loss_curve.get("safe_to_auto_start")):
        blockers.append("unsafe_loss_curve_evidence_flag")
    if not bool(cached.get("ok")):
        blockers.append("cached_token_ab_not_ready")
    if not newbie_row:
        blockers.append("newbie_cached_token_row_missing")
    if not shape_stable:
        blockers.append("shape_stability_evidence_missing")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_evidence_missing")
    if not checkpoint_semantics_ok:
        blockers.append("checkpoint_semantics_evidence_missing")
    if not residual_reuse_parity_ok:
        blockers.append("residual_reuse_parity_evidence_missing")
    blockers.extend(cached_blockers)
    blockers.extend(row_blockers)

    semantic_ready = not blockers
    digest_payload = {
        "loss_curve_digest": loss_curve.get("artifact_digest") or loss_curve.get("result_digest"),
        "cached_token_ab": {
            "ok": bool(cached.get("ok")),
            "family_count": _safe_int(cached.get("family_count")),
            "estimated_block_compute_reduction": _round(cached.get("estimated_block_compute_reduction")),
        },
        "pair_rows": pair_rows,
        "semantic_flags": {
            "shape_stable": shape_stable,
            "disabled_parity_ok": disabled_parity_ok,
            "checkpoint_semantics_ok": checkpoint_semantics_ok,
            "residual_reuse_parity_ok": residual_reuse_parity_ok,
        },
    }
    return {
        "report": REPORT,
        "schema_version": 1,
        "roadmap": ROADMAP,
        "family": FAMILY,
        "candidate": CANDIDATE,
        "review_type": "loss_curve_ab",
        "status": "blockskip_quality_semantics_ready_nonrelease" if semantic_ready else "blockskip_quality_semantics_blocked",
        "review_ready": semantic_ready,
        "summary": {
            "semantic_ready": semantic_ready,
            "loss_curve_ready": bool(loss_curve.get("review_ready")),
            "cached_token_ab_ready": bool(cached.get("cached_token_ab_ready") or cached.get("ok")),
            "pair_count": len(pair_rows),
            "checkpoint_semantics_pair_count": sum(1 for row in pair_rows if row.get("checkpoint_semantics_ok")),
            "shape_stable": shape_stable,
            "disabled_parity_ok": disabled_parity_ok,
            "checkpoint_semantics_ok": checkpoint_semantics_ok,
            "residual_reuse_parity_ok": residual_reuse_parity_ok,
            "loss_curve_delta": _round(loss_curve.get("loss_curve_delta")),
            "max_loss_curve_delta": _round(loss_curve.get("max_loss_curve_delta"), 6),
            "max_abs_final_loss_delta": _round(_mapping(loss_curve.get("summary")).get("max_abs_final_loss_delta")),
            "blocker_count": len(blockers),
            "blockers": blockers,
        },
        "loss_curve_delta": _round(loss_curve.get("loss_curve_delta")),
        "max_loss_curve_delta": _safe_float(loss_curve.get("max_loss_curve_delta"), 0.25),
        "quality_drift": _round(loss_curve.get("quality_drift", loss_curve.get("loss_curve_delta"))),
        "max_quality_drift": _safe_float(loss_curve.get("max_quality_drift", loss_curve.get("max_loss_curve_delta")), 0.25),
        "shape_stable": shape_stable,
        "disabled_parity_ok": disabled_parity_ok,
        "checkpoint_semantics_ok": checkpoint_semantics_ok,
        "residual_reuse_parity_ok": residual_reuse_parity_ok,
        "reviewer": "json_blockskip_semantic_evidence_builder",
        "artifact_digest": _digest(digest_payload),
        "semantic_evidence": digest_payload,
        "pair_rows": pair_rows,
        "cached_token_ab_summary": {
            "scorecard": str(cached.get("scorecard") or ""),
            "ok": bool(cached.get("ok")),
            "cached_token_ab_ready": bool(cached.get("cached_token_ab_ready")),
            "family_count": _safe_int(cached.get("family_count")),
            "estimated_block_compute_reduction": _round(cached.get("estimated_block_compute_reduction")),
            "newbie_plan_enabled": bool(plan.get("enabled")),
            "newbie_skipped_blocks": _safe_int(plan.get("skipped_blocks")),
            "newbie_reuse_residual": bool(policy.get("reuse_residual")),
            "newbie_disabled_parity_ok": disabled_parity_ok,
        },
        "not_release_evidence": True,
        "release_claim_allowed": False,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "does_not_run_gpu_heavy": True,
        "cpu_replay_only": True,
        "default_behavior_changed": False,
        "runtime_activation_enabled": False,
        "trainer_wiring_allowed": False,
    }


__all__ = [
    "CANDIDATE",
    "FAMILY",
    "REPORT",
    "ROADMAP",
    "build_newbie_blockskip_quality_semantic_evidence",
]
