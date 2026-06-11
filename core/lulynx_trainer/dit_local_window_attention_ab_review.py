"""Default-off A/B evidence and review chain for local-window DiT attention."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_EVIDENCE_FIELDS = (
    "experiment_id",
    "baseline_run_id",
    "candidate_run_id",
    "evidence_scope",
    "owner",
    "reviewer",
    "rollback_plan",
)


def build_local_window_attention_ab_evidence_package(
    evidence: Mapping[str, Any],
    *,
    threshold_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(evidence)
    thresholds = _thresholds(threshold_policy)
    blockers: list[str] = []
    for name in REQUIRED_EVIDENCE_FIELDS:
        if not str(payload.get(name) or "").strip():
            blockers.append(f"evidence_field_missing:{name}")
    if not bool(payload.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(payload.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(payload.get("acknowledge_default_off", False)):
        blockers.append("default_off_ack_missing")
    if _unsafe(payload):
        blockers.append("unsafe_execution_claim")
    ok = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_ab_evidence_package_v0",
        "ok": ok,
        "evidence_package_ready": ok,
        "experiment_id": str(payload.get("experiment_id") or ""),
        "evidence_scope": str(payload.get("evidence_scope") or ""),
        "baseline_run_id": str(payload.get("baseline_run_id") or ""),
        "candidate_run_id": str(payload.get("candidate_run_id") or ""),
        "owner": str(payload.get("owner") or ""),
        "reviewer": str(payload.get("reviewer") or ""),
        "threshold_policy": thresholds,
        "ab_execution_allowed": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "default_behavior_changed": False,
        "training_path_enabled": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "ingest manually collected local-window attention A/B results"
            if ok
            else "complete local-window attention A/B evidence package prerequisites"
        ),
    }


def ingest_local_window_attention_ab_results(
    evidence_package: Mapping[str, Any],
    result_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    package = dict(evidence_package)
    results = [dict(item) for item in result_summaries]
    thresholds = _thresholds(package.get("threshold_policy"))
    blockers: list[str] = []
    if package.get("scorecard") != "dit_local_window_attention_ab_evidence_package_v0":
        blockers.append("unexpected_evidence_package")
    if not bool(package.get("evidence_package_ready", package.get("ok", False))):
        blockers.append("evidence_package_not_ready")
    if not results:
        blockers.append("result_summaries_missing")

    rows = [_result_row(item, thresholds) for item in results]
    for row in rows:
        blockers.extend(f"{row['case_id']}:{reason}" for reason in row["blocked_reasons"])
    if any(_unsafe(row) for row in rows):
        blockers.append("unsafe_result_claim")
    ok = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_ab_result_ingestion_v0",
        "ok": ok,
        "result_ingestion_ready": ok,
        "experiment_id": str(package.get("experiment_id") or ""),
        "case_count": len(rows),
        "rows": rows,
        "threshold_policy": thresholds,
        "ab_execution_allowed": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "default_behavior_changed": False,
        "training_path_enabled": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare signed local-window attention quality review"
            if ok
            else "collect passing local-window attention A/B result summaries before review"
        ),
    }


def build_local_window_attention_quality_review_decision(
    ingestion: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    ingested = dict(ingestion)
    payload = dict(review)
    blockers: list[str] = []
    if ingested.get("scorecard") != "dit_local_window_attention_ab_result_ingestion_v0":
        blockers.append("unexpected_result_ingestion")
    if not bool(ingested.get("result_ingestion_ready", ingested.get("ok", False))):
        blockers.append("result_ingestion_not_ready")
    for name in ("review_id", "reviewer", "decision", "result_digest"):
        if not str(payload.get(name) or "").strip():
            blockers.append(f"review_field_missing:{name}")
    if str(payload.get("decision") or "").strip().lower() != "approve_default_off":
        blockers.append("review_decision_not_default_off_approval")
    if not bool(payload.get("acknowledge_no_default_enable", False)):
        blockers.append("default_enable_ack_missing")
    if _unsafe(payload) or _unsafe(ingested):
        blockers.append("unsafe_review_claim")
    ok = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_quality_review_decision_v0",
        "ok": ok,
        "promotion_review_ready": ok,
        "review_id": str(payload.get("review_id") or ""),
        "reviewer": str(payload.get("reviewer") or ""),
        "decision": str(payload.get("decision") or ""),
        "result_digest": str(payload.get("result_digest") or ""),
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "default_enable_allowed": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "training_path_enabled": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare default-off local-window attention rollout proposal"
            if ok
            else "complete signed local-window attention quality review before rollout proposal"
        ),
    }


def build_local_window_attention_default_off_rollout_proposal(
    quality_review: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    review = dict(quality_review)
    payload = dict(proposal)
    blockers: list[str] = []
    if review.get("scorecard") != "dit_local_window_attention_quality_review_decision_v0":
        blockers.append("unexpected_quality_review")
    if not bool(review.get("promotion_review_ready", review.get("ok", False))):
        blockers.append("quality_review_not_ready")
    for name in ("proposal_id", "owner", "reviewer", "rollout_scope", "rollback_plan", "quality_monitoring_plan"):
        if not str(payload.get(name) or "").strip():
            blockers.append(f"proposal_field_missing:{name}")
    if payload.get("default_enable_allowed") is not False:
        blockers.append("default_enable_allowed_must_be_false")
    if payload.get("auto_rollout_allowed") is not False:
        blockers.append("auto_rollout_allowed_must_be_false")
    if not bool(payload.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if _unsafe(review, payload):
        blockers.append("unsafe_rollout_claim")
    ok = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_default_off_rollout_proposal_v0",
        "ok": ok,
        "rollout_proposal_ready": ok,
        "proposal_id": str(payload.get("proposal_id") or ""),
        "rollout_scope": str(payload.get("rollout_scope") or ""),
        "owner": str(payload.get("owner") or ""),
        "reviewer": str(payload.get("reviewer") or ""),
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "training_path_enabled": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hold local-window attention rollout default-off until explicit runtime activation review"
            if ok
            else "complete default-off local-window attention rollout proposal fields before activation review"
        ),
    }


def build_local_window_attention_runtime_activation_review(
    rollout_proposal: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    proposal = dict(rollout_proposal)
    payload = dict(review)
    blockers: list[str] = []
    if proposal.get("scorecard") != "dit_local_window_attention_default_off_rollout_proposal_v0":
        blockers.append("unexpected_rollout_proposal")
    if not bool(proposal.get("rollout_proposal_ready", proposal.get("ok", False))):
        blockers.append("rollout_proposal_not_ready")
    for name in ("activation_review_id", "reviewer", "timestamp", "requested_scope", "proposal_digest"):
        if not str(payload.get(name) or "").strip():
            blockers.append(f"activation_review_field_missing:{name}")
    if payload.get("runtime_activation_enabled") is not False:
        blockers.append("runtime_activation_must_remain_false")
    if payload.get("request_fields_emitted") is not False:
        blockers.append("request_fields_must_remain_false")
    if payload.get("training_launch_allowed") is not False:
        blockers.append("training_launch_must_remain_false")
    if not bool(payload.get("acknowledge_default_off", False)):
        blockers.append("default_off_ack_missing")
    if _unsafe(proposal, payload):
        blockers.append("unsafe_activation_claim")
    ok = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_runtime_activation_review_v0",
        "ok": ok,
        "runtime_activation_review_ready": ok,
        "activation_review_id": str(payload.get("activation_review_id") or ""),
        "reviewer": str(payload.get("reviewer") or ""),
        "requested_scope": str(payload.get("requested_scope") or ""),
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "training_path_enabled": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare separate request-field emission contract while keeping local-window attention default-off"
            if ok
            else "complete signed local-window attention runtime activation review before request wiring"
        ),
    }


def _result_row(item: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    row = dict(item)
    case_id = str(row.get("case_id") or "case")
    step_time = float(row.get("step_time_improvement", 0.0) or 0.0)
    compute_fraction = _fraction(row.get("attention_compute_fraction", 1.0))
    vram_regression = float(row.get("vram_regression", 0.0) or 0.0)
    quality_drift = float(row.get("quality_drift", 1.0) or 1.0)
    loss_delta = float(row.get("loss_delta", 1.0) or 1.0)
    blockers: list[str] = []
    if step_time < thresholds["min_step_time_improvement"]:
        blockers.append("step_time_improvement_below_threshold")
    if compute_fraction > thresholds["max_attention_compute_fraction"]:
        blockers.append("attention_compute_fraction_above_threshold")
    if vram_regression > thresholds["max_vram_regression"]:
        blockers.append("vram_regression_above_threshold")
    if quality_drift > thresholds["max_quality_drift"]:
        blockers.append("quality_drift_above_threshold")
    if loss_delta > thresholds["max_loss_delta"]:
        blockers.append("loss_delta_above_threshold")
    for flag in ("shape_stable", "disabled_parity_ok", "masked_attention_parity_ok"):
        if not bool(row.get(flag, False)):
            blockers.append(f"{flag}_missing")
    if _unsafe(row):
        blockers.append("unsafe_result_flag")
    return {
        "case_id": case_id,
        "ok": not blockers,
        "step_time_improvement": step_time,
        "attention_compute_fraction": compute_fraction,
        "vram_regression": vram_regression,
        "quality_drift": quality_drift,
        "loss_delta": loss_delta,
        "blocked_reasons": blockers,
    }


def _thresholds(policy: Mapping[str, Any] | None) -> dict[str, float]:
    payload = dict(policy or {})
    return {
        "min_step_time_improvement": float(payload.get("min_step_time_improvement", 0.05) or 0.05),
        "max_attention_compute_fraction": _fraction(payload.get("max_attention_compute_fraction", 0.70)),
        "max_vram_regression": float(payload.get("max_vram_regression", 0.03) or 0.03),
        "max_quality_drift": float(payload.get("max_quality_drift", 0.02) or 0.02),
        "max_loss_delta": float(payload.get("max_loss_delta", 0.02) or 0.02),
    }


def _fraction(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 1.0


def _unsafe(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "trainer_wiring_allowed",
        "runtime_activation_enabled",
        "request_fields_emitted",
        "request_adapter_registered",
        "training_launch_allowed",
        "runs_dispatched",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "auto_rollout_allowed",
        "ab_execution_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "build_local_window_attention_ab_evidence_package",
    "build_local_window_attention_default_off_rollout_proposal",
    "build_local_window_attention_quality_review_decision",
    "build_local_window_attention_runtime_activation_review",
    "ingest_local_window_attention_ab_results",
]
