"""Quality-review decision for DiT compute reducer A/B result ingestion."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


APPROVED_QUALITY_DECISIONS = {"approved", "approve", "quality_passed"}


def build_dit_compute_reducer_quality_review_decision(
    *,
    result_ingestion: Mapping[str, Any],
    quality_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ingestion = dict(result_ingestion)
    review = dict(quality_review or {})
    rows = [dict(item) for item in ingestion.get("result_rows", ()) if isinstance(item, Mapping)]
    passed = tuple(str(item) for item in ingestion.get("passed_reducers", ()) if str(item).strip())
    decision = str(review.get("decision") or "").strip().lower()
    min_passed = max(int(review.get("min_passed_reducers", 1) or 1), 1)
    blockers: list[str] = []

    if ingestion.get("scorecard") != "dit_compute_reducer_ab_result_ingestion_v0":
        blockers.append("unexpected_ab_result_ingestion")
    if not bool(ingestion.get("ab_result_ingestion_ready", ingestion.get("ok", False))):
        blockers.append("ab_result_ingestion_not_ready")
    if _unsafe_flags(ingestion, review):
        blockers.append("unsafe_child_flag")
    if not rows:
        blockers.append("result_rows_missing")
    failed = [str(row.get("reducer_id") or idx) for idx, row in enumerate(rows) if not bool(row.get("ok", False))]
    blockers.extend(f"reducer_quality_failed:{reducer_id}" for reducer_id in failed)
    if len(passed) < min_passed:
        blockers.append("passed_reducer_count_below_review_minimum")
    if decision not in APPROVED_QUALITY_DECISIONS:
        blockers.append("quality_decision_not_approved")
    if not str(review.get("reviewer") or "").strip():
        blockers.append("reviewer_missing")
    if not str(review.get("result_digest") or review.get("artifact_digest") or "").strip():
        blockers.append("result_digest_missing")
    if not bool(review.get("acknowledge_default_off", False)):
        blockers.append("default_off_acknowledgement_missing")
    if not bool(review.get("acknowledge_no_trainer_wiring", False)):
        blockers.append("trainer_wiring_acknowledgement_missing")
    if review.get("default_enable_allowed") is not False:
        blockers.append("default_enable_allowed_must_be_false")
    if review.get("trainer_wiring_allowed") is not False:
        blockers.append("trainer_wiring_allowed_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_quality_review_decision_v0",
        "ok": ready,
        "quality_review_ready": ready,
        "promotion_review_ready": ready,
        "passed_reducers": list(passed),
        "passed_reducer_count": len(passed),
        "result_row_count": len(rows),
        "review": _summary(
            review,
            (
                "decision",
                "reviewer",
                "result_digest",
                "artifact_digest",
                "acknowledge_default_off",
                "acknowledge_no_trainer_wiring",
                "min_passed_reducers",
            ),
        ),
        "trainer_wiring_allowed": False,
        "trainer_wiring_executed": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "ab_execution_allowed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare default-off compute reducer rollout proposal"
            if ready
            else "complete signed compute reducer quality review before rollout proposal"
        ),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "default_rollout_allowed",
        "auto_rollout_allowed",
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "ab_execution_allowed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


__all__ = ["build_dit_compute_reducer_quality_review_decision"]
