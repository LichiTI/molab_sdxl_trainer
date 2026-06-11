"""Default-off A/B evidence and review boundaries for DiffCR token compression."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_POLICY_FIELDS = (
    "owner",
    "review_id",
    "evidence_scope",
    "baseline_case_ref",
    "candidate_case_ref",
    "rollback_plan",
)
REQUIRED_OUTPUTS = (
    "baseline_metrics",
    "candidate_metrics",
    "quality_report",
    "loss_report",
    "shape_report",
    "compression_plan_report",
)
APPROVED_QUALITY_DECISIONS = {"approved", "approve", "quality_passed"}
APPROVED_REVIEW_DECISIONS = {"approved", "approve", "signed"}
EXPECTED_RUNTIME_SCOPE = "diffcr_runtime_activation_review"


def build_diffcr_ab_evidence_package(
    *,
    probe_scorecard: Mapping[str, Any],
    evidence_policy: Mapping[str, Any],
) -> dict[str, Any]:
    scorecard = dict(probe_scorecard)
    policy = dict(evidence_policy)
    outputs = _items(policy.get("required_outputs") or policy.get("evidence_outputs"))
    metrics = _items(policy.get("required_metrics") or policy.get("evidence_metrics"))
    thresholds = dict(policy.get("thresholds") or policy.get("acceptance_thresholds") or {})
    plan = dict(scorecard.get("plan") or {})
    blockers: list[str] = []

    if scorecard.get("scorecard") != "diffcr_token_compression_probe_v0":
        blockers.append("unexpected_probe_scorecard")
    if not bool(scorecard.get("probe_ready", scorecard.get("ok", False))):
        blockers.append("probe_not_ready")
    if _unsafe_flags(scorecard, policy):
        blockers.append("unsafe_child_flag")
    if not bool(plan.get("enabled", False)):
        blockers.append("compression_plan_not_enabled")
    for name in REQUIRED_POLICY_FIELDS:
        if not str(policy.get(name) or "").strip():
            blockers.append(f"evidence_policy_field_missing:{name}")
    for name in REQUIRED_OUTPUTS:
        if name not in outputs:
            blockers.append(f"evidence_output_missing:{name}")
    for name in (
        "step_time_ms",
        "peak_vram_mb",
        "attention_fraction",
        "quality_drift",
        "loss_delta",
        "shape_stable",
        "disabled_parity",
        "expand_parity",
    ):
        if name not in metrics:
            blockers.append(f"metric_missing:{name}")
    if not bool(policy.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(policy.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(policy.get("acknowledge_no_ab_execution", False)):
        blockers.append("ab_execution_ack_missing")
    if not bool(policy.get("requires_later_ab_result_ingestion", False)):
        blockers.append("later_ab_result_ingestion_missing")
    for key in ("ab_execution_allowed", "ab_dispatch_allowed", "trainer_wiring_allowed"):
        if policy.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "diffcr_ab_evidence_package_v0",
        "ok": ready,
        "evidence_package_ready": ready,
        "evidence_scope": str(policy.get("evidence_scope") or ""),
        "owner": str(policy.get("owner") or ""),
        "review_id": str(policy.get("review_id") or ""),
        "compression_plan": _summary(
            plan,
            (
                "enabled",
                "token_count",
                "compressed_count",
                "ratio",
                "estimated_attention_fraction",
                "layer_index",
                "timestep_index",
                "score_mode",
            ),
        ),
        "required_outputs": list(outputs),
        "required_metrics": list(metrics),
        "thresholds": thresholds,
        "baseline_case_ref": str(policy.get("baseline_case_ref") or ""),
        "candidate_case_ref": str(policy.get("candidate_case_ref") or ""),
        **_safe_flags(),
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "trainer_wiring_executed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "collect manual DiffCR high-resolution A/B results and ingest them through a separate result gate"
            if ready
            else "complete DiffCR A/B evidence package prerequisites"
        ),
    }


def build_diffcr_ab_result_ingestion(
    *,
    evidence_package: Mapping[str, Any],
    result_summaries: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    package = dict(evidence_package)
    results = [dict(item) for item in result_summaries if isinstance(item, Mapping)]
    limits = _thresholds(package, thresholds)
    blockers: list[str] = []

    if package.get("scorecard") != "diffcr_ab_evidence_package_v0":
        blockers.append("unexpected_ab_evidence_package")
    if not bool(package.get("evidence_package_ready", package.get("ok", False))):
        blockers.append("ab_evidence_package_not_ready")
    if _unsafe_flags(package):
        blockers.append("unsafe_evidence_package_flag")
    if not results:
        blockers.append("result_summaries_missing")

    rows = [_result_row(idx, result, limits) for idx, result in enumerate(results)]
    blockers.extend(f"{row['case_id']}:{reason}" for row in rows for reason in row["blocked_reasons"])
    if any(_unsafe_flags(item) for item in results):
        blockers.append("unsafe_result_summary_flag")

    ready = not blockers
    passed = [row["case_id"] for row in rows if row["ok"]]
    return {
        "schema_version": 1,
        "scorecard": "diffcr_ab_result_ingestion_v0",
        "ok": ready,
        "ab_result_ingestion_ready": ready,
        "passed_cases": passed,
        "passed_case_count": len(passed),
        "result_rows": rows,
        "thresholds": limits,
        **_safe_flags(),
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "trainer_wiring_executed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare default-off DiffCR quality review"
            if ready
            else "collect passing DiffCR A/B result summaries before review"
        ),
    }


def build_diffcr_quality_review_decision(
    *,
    result_ingestion: Mapping[str, Any],
    quality_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ingestion = dict(result_ingestion)
    review = dict(quality_review or {})
    rows = [dict(item) for item in ingestion.get("result_rows", ()) if isinstance(item, Mapping)]
    passed = tuple(str(item) for item in ingestion.get("passed_cases", ()) if str(item).strip())
    decision = str(review.get("decision") or "").strip().lower()
    min_passed = max(int(review.get("min_passed_cases", 1) or 1), 1)
    blockers: list[str] = []

    if ingestion.get("scorecard") != "diffcr_ab_result_ingestion_v0":
        blockers.append("unexpected_ab_result_ingestion")
    if not bool(ingestion.get("ab_result_ingestion_ready", ingestion.get("ok", False))):
        blockers.append("ab_result_ingestion_not_ready")
    if _unsafe_flags(ingestion, review):
        blockers.append("unsafe_child_flag")
    if not rows:
        blockers.append("result_rows_missing")
    blockers.extend(
        f"case_quality_failed:{row.get('case_id') or idx}"
        for idx, row in enumerate(rows)
        if not bool(row.get("ok", False))
    )
    if len(passed) < min_passed:
        blockers.append("passed_case_count_below_review_minimum")
    if decision not in APPROVED_QUALITY_DECISIONS:
        blockers.append("quality_decision_not_approved")
    if not str(review.get("reviewer") or "").strip():
        blockers.append("reviewer_missing")
    if not str(review.get("result_digest") or review.get("artifact_digest") or "").strip():
        blockers.append("result_digest_missing")
    for key in ("acknowledge_default_off", "acknowledge_no_trainer_wiring", "acknowledge_shape_and_expand_parity"):
        if not bool(review.get(key, False)):
            blockers.append(f"{key}_missing")
    for key in ("default_enable_allowed", "trainer_wiring_allowed"):
        if review.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "diffcr_quality_review_decision_v0",
        "ok": ready,
        "quality_review_ready": ready,
        "promotion_review_ready": ready,
        "passed_cases": list(passed),
        "passed_case_count": len(passed),
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
                "acknowledge_shape_and_expand_parity",
                "min_passed_cases",
            ),
        ),
        **_safe_flags(),
        "trainer_wiring_executed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare default-off DiffCR rollout proposal"
            if ready
            else "complete signed DiffCR quality review before rollout proposal"
        ),
    }


def build_diffcr_default_off_rollout_proposal(
    *,
    quality_decision: Mapping[str, Any],
    rollout_proposal: Mapping[str, Any],
) -> dict[str, Any]:
    decision = dict(quality_decision)
    proposal = dict(rollout_proposal)
    passed = tuple(str(item) for item in decision.get("passed_cases", ()) if str(item).strip())
    blockers: list[str] = []

    if decision.get("scorecard") != "diffcr_quality_review_decision_v0":
        blockers.append("unexpected_quality_decision")
    if not bool(decision.get("quality_review_ready", decision.get("ok", False))):
        blockers.append("quality_review_not_ready")
    if not bool(decision.get("promotion_review_ready", False)):
        blockers.append("promotion_review_not_ready")
    if _unsafe_flags(decision, proposal):
        blockers.append("unsafe_child_flag")
    if not passed:
        blockers.append("passed_cases_missing")
    for name in (
        "proposal_id",
        "owner",
        "reviewer",
        "compression_scope",
        "rollback_plan",
        "quality_monitoring_plan",
        "shape_monitoring_plan",
        "canary_scope",
        "activation_boundary",
    ):
        if not str(proposal.get(name) or "").strip():
            blockers.append(f"proposal_field_missing:{name}")
    for key in ("default_enable_allowed", "auto_rollout_allowed", "trainer_wiring_allowed"):
        if proposal.get(key) is not False:
            blockers.append(f"{key}_boundary_missing")
    if not bool(proposal.get("acknowledge_default_off", False)):
        blockers.append("default_off_acknowledgement_missing")
    if not bool(proposal.get("requires_later_runtime_activation_review", False)):
        blockers.append("later_runtime_activation_review_missing")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "diffcr_default_off_rollout_proposal_v0",
        "ok": ready,
        "rollout_proposal_ready": ready,
        "activation_boundary_recorded": ready,
        "passed_cases": list(passed),
        "passed_case_count": len(passed),
        "proposal": _summary(
            proposal,
            (
                "proposal_id",
                "owner",
                "reviewer",
                "compression_scope",
                "rollback_plan",
                "quality_monitoring_plan",
                "shape_monitoring_plan",
                "canary_scope",
                "activation_boundary",
                "acknowledge_default_off",
            ),
        ),
        **_safe_flags(),
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "quality_review_ready": bool(decision.get("quality_review_ready", decision.get("ok", False))),
        "promotion_review_ready": bool(decision.get("promotion_review_ready", False)),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "hold DiffCR rollout proposal default-off until explicit runtime activation review"
            if ready
            else "complete default-off DiffCR rollout proposal fields before activation review"
        ),
    }


def build_diffcr_runtime_activation_review(
    *,
    rollout_proposal: Mapping[str, Any],
    activation_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal = dict(rollout_proposal)
    review = dict(activation_review or {})
    blockers: list[str] = []

    if proposal.get("scorecard") != "diffcr_default_off_rollout_proposal_v0":
        blockers.append("unexpected_rollout_proposal")
    if not bool(proposal.get("rollout_proposal_ready", proposal.get("ok", False))):
        blockers.append("rollout_proposal_not_ready")
    if not bool(proposal.get("activation_boundary_recorded", False)):
        blockers.append("activation_boundary_not_recorded")
    if _unsafe_flags(proposal, review):
        blockers.append("unsafe_child_flag")
    if str(review.get("decision") or "").strip().lower() not in APPROVED_REVIEW_DECISIONS:
        blockers.append("activation_review_not_approved")
    for name in ("signed_review_id", "reviewer", "reviewed_at"):
        if not str(review.get(name) or "").strip():
            blockers.append(f"{name}_missing")
    if str(review.get("requested_scope") or "").strip() != EXPECTED_RUNTIME_SCOPE:
        blockers.append("requested_scope_mismatch")
    if not str(review.get("proposal_digest") or review.get("artifact_digest") or "").strip():
        blockers.append("proposal_digest_missing")
    for key in (
        "acknowledge_default_off",
        "acknowledge_no_runtime_activation",
        "acknowledge_no_request_fields_emitted",
        "acknowledge_no_trainer_wiring",
        "acknowledge_no_training_launch",
        "acknowledge_manual_activation_required",
    ):
        if not bool(review.get(key, False)):
            blockers.append(f"{key}_missing")
    for key in (
        "approve_runtime_activation_allowed",
        "approve_runtime_activation_enabled",
        "approve_request_fields_emitted",
        "approve_trainer_wiring_allowed",
        "approve_training_launch_allowed",
        "approve_default_rollout_allowed",
        "approve_auto_rollout_allowed",
    ):
        if review.get(key) is not False:
            blockers.append(f"{key}_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "diffcr_runtime_activation_review_v0",
        "ok": ready,
        "runtime_activation_review_ready": ready,
        "activation_review_signed": ready,
        "runtime_activation_allowed": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        **_safe_flags(),
        "trainer_wiring_executed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "proposal_ready": bool(proposal.get("rollout_proposal_ready", proposal.get("ok", False))),
        "activation_boundary_recorded": bool(proposal.get("activation_boundary_recorded", False)),
        "passed_case_count": int(proposal.get("passed_case_count") or 0),
        "passed_cases": list(proposal.get("passed_cases", ()) or ()),
        "review": _summary(
            review,
            (
                "signed_review_id",
                "decision",
                "reviewer",
                "reviewed_at",
                "requested_scope",
                "proposal_digest",
                "artifact_digest",
            ),
        ),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "prepare separate request-field emission contract while keeping DiffCR default-off"
            if ready
            else "complete signed DiffCR runtime activation review before request wiring"
        ),
    }


def _result_row(index: int, result: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    case_id = str(result.get("case_id") or f"case-{index}")
    blockers: list[str] = []
    for name in (
        "baseline_step_time_ms",
        "candidate_step_time_ms",
        "baseline_peak_vram_mb",
        "candidate_peak_vram_mb",
        "candidate_attention_fraction",
        "quality_drift",
        "loss_delta",
        "shape_stable",
        "disabled_parity_ok",
        "expand_parity_ok",
    ):
        if name not in result:
            blockers.append(f"result_field_missing:{name}")
    baseline_step = _positive_float(result.get("baseline_step_time_ms"))
    candidate_step = _positive_float(result.get("candidate_step_time_ms"))
    baseline_vram = _positive_float(result.get("baseline_peak_vram_mb"))
    candidate_vram = _positive_float(result.get("candidate_peak_vram_mb"))
    attention_fraction = _float_or_none(result.get("candidate_attention_fraction"))
    quality_drift = _float_or_none(result.get("quality_drift"))
    loss_delta = _float_or_none(result.get("loss_delta"))
    step_improvement = 0.0 if baseline_step <= 0 else (baseline_step - candidate_step) / baseline_step
    vram_delta = 0.0 if baseline_vram <= 0 else (candidate_vram - baseline_vram) / baseline_vram
    if baseline_step <= 0 or candidate_step <= 0:
        blockers.append("step_time_invalid")
    elif step_improvement < thresholds["min_step_time_improvement"]:
        blockers.append("step_time_improvement_below_threshold")
    if baseline_vram <= 0 or candidate_vram <= 0:
        blockers.append("peak_vram_invalid")
    elif vram_delta > thresholds["max_vram_regression"]:
        blockers.append("vram_regression_above_threshold")
    if attention_fraction is None:
        blockers.append("attention_fraction_missing")
    elif attention_fraction > thresholds["max_attention_fraction"]:
        blockers.append("attention_fraction_above_threshold")
    if quality_drift is None:
        blockers.append("quality_drift_missing")
    elif quality_drift > thresholds["max_quality_drift"]:
        blockers.append("quality_drift_above_threshold")
    if loss_delta is None:
        blockers.append("loss_delta_missing")
    elif loss_delta > thresholds["max_loss_delta"]:
        blockers.append("loss_delta_above_threshold")
    for key in ("shape_stable", "disabled_parity_ok", "expand_parity_ok"):
        if result.get(key) is not True:
            blockers.append(f"{key}_missing")
    if _unsafe_flags(result):
        blockers.append("unsafe_result_flag")
    return {
        "case_id": case_id,
        "ok": not blockers,
        "step_time_improvement": float(step_improvement),
        "vram_delta_fraction": float(vram_delta),
        "candidate_attention_fraction": attention_fraction,
        "quality_drift": quality_drift,
        "loss_delta": loss_delta,
        "shape_stable": bool(result.get("shape_stable", False)),
        "disabled_parity_ok": bool(result.get("disabled_parity_ok", False)),
        "expand_parity_ok": bool(result.get("expand_parity_ok", False)),
        "blocked_reasons": blockers,
    }


def _thresholds(package: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, float]:
    values = dict(package.get("thresholds") or {})
    if override:
        values.update(dict(override))
    return {
        "min_step_time_improvement": _float_or_default(values.get("min_step_time_improvement"), 0.05),
        "max_attention_fraction": _float_or_default(values.get("max_attention_fraction"), 0.64),
        "max_vram_regression": _float_or_default(values.get("max_vram_regression"), 0.05),
        "max_quality_drift": _float_or_default(values.get("max_quality_drift"), 0.01),
        "max_loss_delta": _float_or_default(values.get("max_loss_delta"), 0.01),
    }


def _safe_flags() -> dict[str, bool]:
    return {
        "ab_execution_allowed": False,
        "ab_dispatch_allowed": False,
        "trainer_wiring_allowed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
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
        "ab_execution_started",
        "ab_execution_completed",
        "runtime_activation_allowed",
        "runtime_activation_enabled",
        "request_fields_emitted",
        "request_adapter_registered",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
        "approve_runtime_activation_allowed",
        "approve_runtime_activation_enabled",
        "approve_request_fields_emitted",
        "approve_trainer_wiring_allowed",
        "approve_training_launch_allowed",
        "approve_default_rollout_allowed",
        "approve_auto_rollout_allowed",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


def _items(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0.0 else 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


__all__ = [
    "build_diffcr_ab_evidence_package",
    "build_diffcr_ab_result_ingestion",
    "build_diffcr_default_off_rollout_proposal",
    "build_diffcr_quality_review_decision",
    "build_diffcr_runtime_activation_review",
]
