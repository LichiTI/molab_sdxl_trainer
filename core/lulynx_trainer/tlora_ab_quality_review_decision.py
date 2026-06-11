"""Quality-review decision gate for reviewed T-LoRA A/B results."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


APPROVED_QUALITY_DECISIONS = {"approved", "approve", "quality_passed"}


def build_tlora_ab_quality_review_decision(
    *,
    result_ingestion: Mapping[str, Any],
    quality_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ingestion = dict(result_ingestion)
    review = dict(quality_review or {})
    result_gate = dict(ingestion.get("result_gate") or {})
    scorecards = [dict(item) for item in result_gate.get("scorecards", []) if isinstance(item, Mapping)]
    decision = str(review.get("decision") or "").strip().lower()
    blockers: list[str] = []

    if ingestion.get("scorecard") != "tlora_ab_dispatch_result_ingestion_v0":
        blockers.append("unexpected_result_ingestion")
    if not bool(ingestion.get("result_ingestion_ready", ingestion.get("ok", False))):
        blockers.append("result_ingestion_not_ready")
    if _unsafe_flags(ingestion, result_gate, review):
        blockers.append("unsafe_child_flag")
    if not scorecards:
        blockers.append("scorecards_missing")
    failed = [str(item.get("case_id") or idx) for idx, item in enumerate(scorecards) if not bool(item.get("ok", False))]
    blockers.extend(f"case_quality_failed:{case_id}" for case_id in failed)
    min_case_count = max(int(review.get("min_case_count", 1) or 1), 1)
    if int(ingestion.get("case_count") or 0) < min_case_count:
        blockers.append("case_count_below_review_minimum")
    if decision not in APPROVED_QUALITY_DECISIONS:
        blockers.append("quality_decision_not_approved")
    if not str(review.get("reviewer") or "").strip():
        blockers.append("reviewer_missing")
    if not str(review.get("result_digest") or review.get("artifact_digest") or "").strip():
        blockers.append("result_digest_missing")
    if not bool(review.get("acknowledge_default_off", False)):
        blockers.append("default_off_acknowledgement_missing")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_ab_quality_review_decision_v0",
        "ok": ready,
        "quality_review_ready": ready,
        "promotion_review_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "case_count": int(ingestion.get("case_count") or 0),
        "scorecard_count": len(scorecards),
        "review": _summary(
            review,
            ("decision", "reviewer", "result_digest", "artifact_digest", "acknowledge_default_off", "min_case_count"),
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare default-off rollout proposal from reviewed T-LoRA A/B evidence"
            if ready
            else "complete signed quality review before any T-LoRA rollout proposal"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    return any(
        bool(payload.get("training_path_enabled", False))
        or bool(payload.get("default_behavior_changed", False))
        or bool(payload.get("promotion_ready", False))
        for payload in payloads
    )


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


__all__ = ["build_tlora_ab_quality_review_decision"]
