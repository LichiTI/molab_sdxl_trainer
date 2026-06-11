"""Shared report-only A/B result bundle contract for DiT frontier routes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


_COMMON_CASE_FIELDS = (
    "case_id",
    "baseline_step_time_ms",
    "candidate_step_time_ms",
    "baseline_peak_vram_mb",
    "candidate_peak_vram_mb",
)

FEATURE_RESULT_FIELDS: dict[str, tuple[str, ...]] = {
    "sra2_haste": (
        *_COMMON_CASE_FIELDS,
        "baseline_alignment_loss",
        "candidate_alignment_loss",
        "quality_drift",
        "holdout_loss_delta",
    ),
    "cdm_qta_lora": (
        *_COMMON_CASE_FIELDS,
        "baseline_energy_per_step_j",
        "candidate_energy_per_step_j",
        "quality_drift",
        "loss_delta",
        "optimizer_state_parity",
    ),
    "diffcr": (
        *_COMMON_CASE_FIELDS,
        "candidate_attention_fraction",
        "quality_drift",
        "loss_delta",
        "shape_stable",
        "disabled_parity_ok",
        "expand_parity_ok",
    ),
    "dit_blockskip": (
        *_COMMON_CASE_FIELDS,
        "candidate_block_compute_fraction",
        "quality_drift",
        "loss_delta",
        "shape_stable",
        "disabled_parity_ok",
        "checkpoint_semantics_ok",
        "residual_reuse_parity_ok",
    ),
    "dit_local_window_attention": (
        "case_id",
        "step_time_improvement",
        "attention_compute_fraction",
        "vram_regression",
        "quality_drift",
        "loss_delta",
        "shape_stable",
        "disabled_parity_ok",
        "masked_attention_parity_ok",
    ),
    "dit_compute_reducer": (
        "reducer_id",
        "baseline_step_time_ms",
        "candidate_step_time_ms",
        "baseline_peak_vram_mb",
        "candidate_peak_vram_mb",
        "quality_drift",
        "loss_delta",
    ),
    "tlora_ab": (
        "case_id",
        "baseline",
        "tlora",
    ),
}

FEATURE_SCORECARDS = {
    "sra2_haste": "sra2_haste_ab_evidence_package_v0",
    "cdm_qta_lora": "cdm_qta_lora_ab_evidence_package_v0",
    "diffcr": "diffcr_ab_evidence_package_v0",
    "dit_blockskip": "dit_blockskip_ab_evidence_package_v0",
    "dit_local_window_attention": "dit_local_window_attention_ab_evidence_package_v0",
    "dit_compute_reducer": "dit_compute_reducer_ab_evidence_package_v0",
    "tlora_ab": "tlora_ab_dispatch_manifest_v0",
}


def build_dit_frontier_ab_result_bundle_plan(
    *,
    feature_id: str,
    evidence_package: Mapping[str, Any] | None = None,
    artifact_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    feature = _feature(feature_id)
    package = dict(evidence_package or {})
    policy = dict(artifact_policy or {})
    fields = FEATURE_RESULT_FIELDS.get(feature, ())
    blockers: list[str] = []

    if not fields:
        blockers.append("unsupported_feature_id")
    expected_scorecard = FEATURE_SCORECARDS.get(feature)
    if expected_scorecard and package:
        actual_scorecard = str(package.get("scorecard") or package.get("manifest") or "")
        if actual_scorecard != expected_scorecard:
            blockers.append("unexpected_evidence_package")
    if feature == "tlora_ab" and bool(package.get("execution_performed", False)):
        blockers.append("tlora_dispatch_manifest_claims_execution")
    if package and not _package_ready(package, feature):
        blockers.append("evidence_package_not_ready")
    if _unsafe_flags(package, policy):
        blockers.append("unsafe_bundle_plan_input_flag")
    if policy.get("report_only") is False:
        blockers.append("report_only_must_not_be_false")

    ready = bool(fields) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_ab_result_bundle_plan_v0",
        "ok": ready,
        "bundle_plan_ready": ready,
        "feature_id": feature,
        "expected_input_scorecard": expected_scorecard or "",
        "required_result_fields": list(fields),
        "result_primary_key": _primary_key(feature),
        "artifact_policy": _summary(policy, ("report_only", "manual_only", "artifact_root", "owner", "review_id")),
        "expected_case_refs": _expected_case_refs(feature, package),
        "result_summary_template": {field: None for field in fields},
        **_safe_flags(),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "ab_dispatch_executed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "trainer_wiring_executed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "collect observed frontier A/B result payloads and audit them"
            if ready
            else "complete frontier A/B bundle plan prerequisites"
        ),
    }


def build_dit_frontier_ab_result_bundle_audit(
    *,
    bundle_plan: Mapping[str, Any],
    observed_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = dict(bundle_plan)
    feature = _feature(plan.get("feature_id"))
    required = tuple(str(item) for item in plan.get("required_result_fields", ()) if str(item).strip())
    observed = [dict(item) for item in observed_results if isinstance(item, Mapping)]
    blockers: list[str] = []

    if plan.get("scorecard") != "dit_frontier_ab_result_bundle_plan_v0":
        blockers.append("unexpected_bundle_plan")
    if not bool(plan.get("bundle_plan_ready", plan.get("ok", False))):
        blockers.append("bundle_plan_not_ready")
    if _unsafe_flags(plan):
        blockers.append("unsafe_bundle_plan_flag")
    if not observed:
        blockers.append("observed_results_missing")
    if not required:
        blockers.append("required_result_fields_missing")

    rows = [_audit_row(feature, index, result, required) for index, result in enumerate(observed)]
    blockers.extend(f"{row['result_id']}:{reason}" for row in rows for reason in row["blocked_reasons"])
    if any(_unsafe_flags(item) for item in observed):
        blockers.append("unsafe_observed_result_flag")

    ready = bool(rows) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_ab_result_bundle_audit_v0",
        "ok": ready,
        "bundle_audit_ready": ready,
        "feature_id": feature,
        "result_count": len(observed),
        "audit_rows": rows,
        **_safe_flags(),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "ab_dispatch_executed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "trainer_wiring_executed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "convert audited frontier A/B payloads into route-specific result summaries"
            if ready
            else "collect complete safe observed result payloads before summary conversion"
        ),
    }


def build_dit_frontier_ab_result_summaries(
    *,
    bundle_audit: Mapping[str, Any],
    observed_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    audit = dict(bundle_audit)
    feature = _feature(audit.get("feature_id"))
    observed = [dict(item) for item in observed_results if isinstance(item, Mapping)]
    blockers: list[str] = []

    if audit.get("scorecard") != "dit_frontier_ab_result_bundle_audit_v0":
        blockers.append("unexpected_bundle_audit")
    if not bool(audit.get("bundle_audit_ready", audit.get("ok", False))):
        blockers.append("bundle_audit_not_ready")
    if _unsafe_flags(audit) or any(_unsafe_flags(item) for item in observed):
        blockers.append("unsafe_result_summary_input_flag")
    if not observed:
        blockers.append("observed_results_missing")

    summaries = [_summary_row(feature, item) for item in observed]
    ready = bool(summaries) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_ab_result_summaries_v0",
        "ok": ready,
        "result_summaries_ready": ready,
        "feature_id": feature,
        "result_summary_count": len(summaries),
        "result_summaries": summaries if ready else [],
        **_safe_flags(),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "ab_dispatch_executed": False,
        "training_launch_executed": False,
        "run_dispatch_executed": False,
        "trainer_wiring_executed": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "feed result_summaries into the feature-specific A/B ingestion gate"
            if ready
            else "complete bundle audit before feature-specific ingestion"
        ),
    }


def _audit_row(feature: str, index: int, result: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
    result_id = _result_id(feature, index, result)
    blockers = [f"result_field_missing:{name}" for name in required if name not in result]
    if _unsafe_flags(result):
        blockers.append("unsafe_result_flag")
    return {
        "result_id": result_id,
        "ok": not blockers,
        "missing_fields": [name for name in required if name not in result],
        "blocked_reasons": blockers,
    }


def _summary_row(feature: str, result: Mapping[str, Any]) -> dict[str, Any]:
    row = {field: result.get(field) for field in FEATURE_RESULT_FIELDS.get(feature, ()) if field in result}
    if feature != "tlora_ab":
        row.setdefault("case_id", result.get("case_id"))
    return row


def _feature(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _primary_key(feature: str) -> str:
    return "reducer_id" if feature == "dit_compute_reducer" else "case_id"


def _result_id(feature: str, index: int, result: Mapping[str, Any]) -> str:
    key = _primary_key(feature)
    return str(result.get(key) or f"{key}-{index}")


def _package_ready(package: Mapping[str, Any], feature: str) -> bool:
    if feature == "tlora_ab":
        return bool(package.get("dispatch_manifest_ready", package.get("ok", False)))
    return bool(package.get("evidence_package_ready", package.get("ok", False)))


def _expected_case_refs(feature: str, package: Mapping[str, Any]) -> list[str]:
    if not package:
        return []
    if feature == "dit_compute_reducer":
        return [str(item) for item in package.get("selected_reducers", ()) if str(item).strip()]
    if feature == "tlora_ab":
        return [str(item.get("case_id")) for item in package.get("payloads", ()) if isinstance(item, Mapping)]
    refs = (package.get("baseline_case_ref"), package.get("candidate_case_ref"))
    return [str(item) for item in refs if str(item or "").strip()]


def _summary(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


def _safe_flags() -> dict[str, bool]:
    return {
        "ab_execution_allowed": False,
        "ab_dispatch_allowed": False,
        "trainer_wiring_allowed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
        "runtime_activation_enabled": False,
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
        "runtime_activation_enabled",
        "request_fields_emitted",
        "request_adapter_registered",
        "trainer_wiring_allowed",
        "trainer_wiring_executed",
        "ab_execution_allowed",
        "ab_execution_started",
        "ab_execution_completed",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "FEATURE_RESULT_FIELDS",
    "FEATURE_SCORECARDS",
    "build_dit_frontier_ab_result_bundle_audit",
    "build_dit_frontier_ab_result_bundle_plan",
    "build_dit_frontier_ab_result_summaries",
]
