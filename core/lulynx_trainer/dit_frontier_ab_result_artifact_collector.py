"""Report-only trainer A/B result artifact collector for DiT frontier routes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


COMMON_ARM_FIELDS = ("case_id", "arm", "step_time_ms", "peak_vram_mb")
FEATURE_ARMS = {
    "tlora_ab": ("baseline", "tlora"),
}


def build_dit_frontier_ab_result_artifact_plan(
    *,
    feature_id: str,
    cases: Sequence[Mapping[str, Any]],
    artifact_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    feature = _feature(feature_id)
    policy = dict(artifact_policy or {})
    normalized_cases = [_case(item, index) for index, item in enumerate(cases) if isinstance(item, Mapping)]
    blockers: list[str] = []

    if not _supported_feature(feature):
        blockers.append("unsupported_feature_id")
    if not normalized_cases:
        blockers.append("artifact_cases_missing")
    if _duplicates([case["case_id"] for case in normalized_cases]):
        blockers.append("duplicate_case_id")
    if _unsafe_flags(policy):
        blockers.append("unsafe_artifact_policy_flag")
    if policy.get("report_only") is False:
        blockers.append("report_only_must_not_be_false")

    artifact_root = str(policy.get("artifact_root") or f"temp/dit_frontier_ab/{feature}")
    plans = [_case_plan(feature, case, artifact_root) for case in normalized_cases]
    ready = bool(plans) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_ab_result_artifact_plan_v0",
        "ok": ready,
        "artifact_plan_ready": ready,
        "feature_id": feature,
        "case_count": len(plans),
        "artifact_count": sum(len(plan["artifacts"]) for plan in plans),
        "case_artifact_plans": plans if ready else [],
        "artifact_policy": _summary(policy, ("report_only", "manual_only", "artifact_root", "owner", "review_id")),
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
            "collect external trainer A/B result artifacts and audit them"
            if ready
            else "complete frontier A/B artifact plan prerequisites"
        ),
    }


def build_dit_frontier_ab_result_artifact_audit(
    *,
    artifact_plan: Mapping[str, Any],
    observed_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = dict(artifact_plan)
    feature = _feature(plan.get("feature_id"))
    observed = [dict(item) for item in observed_artifacts if isinstance(item, Mapping)]
    expected = {
        (str(item.get("case_id") or ""), str(item.get("arm") or "")): dict(item)
        for case in plan.get("case_artifact_plans", ())
        if isinstance(case, Mapping)
        for item in case.get("artifacts", ())
        if isinstance(item, Mapping)
    }
    observed_by_key = {
        (str(item.get("case_id") or ""), str(item.get("arm") or "")): item
        for item in observed
        if str(item.get("case_id") or "") and str(item.get("arm") or "")
    }
    blockers: list[str] = []

    if plan.get("scorecard") != "dit_frontier_ab_result_artifact_plan_v0":
        blockers.append("unexpected_artifact_plan")
    if not bool(plan.get("artifact_plan_ready", plan.get("ok", False))):
        blockers.append("artifact_plan_not_ready")
    if _unsafe_flags(plan):
        blockers.append("unsafe_artifact_plan_flag")
    if not expected:
        blockers.append("expected_artifacts_missing")

    rows = [_audit_row(key, expected_item, observed_by_key.get(key)) for key, expected_item in expected.items()]
    blockers.extend(f"{row['case_id']}:{row['arm']}:{reason}" for row in rows for reason in row["blocked_reasons"])
    for case_id, arm in sorted(set(observed_by_key) - set(expected)):
        blockers.append(f"unexpected_artifact:{case_id}:{arm}")
    if any(_unsafe_flags(item) for item in observed):
        blockers.append("unsafe_observed_artifact_flag")

    ready = bool(rows) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_ab_result_artifact_audit_v0",
        "ok": ready,
        "artifact_audit_ready": ready,
        "feature_id": feature,
        "case_count": int(plan.get("case_count") or 0),
        "expected_artifact_count": len(expected),
        "observed_artifact_count": len(observed),
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
            "combine audited arm artifacts into frontier A/B result summaries"
            if ready
            else "collect complete safe arm artifacts before summary assembly"
        ),
    }


def build_dit_frontier_ab_result_summary_from_artifacts(
    *,
    artifact_audit: Mapping[str, Any],
    observed_artifacts: Sequence[Mapping[str, Any]],
    quality_measurements: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = dict(artifact_audit)
    feature = _feature(audit.get("feature_id"))
    observed_by_case_arm = {
        (str(item.get("case_id") or ""), str(item.get("arm") or "")): dict(item)
        for item in observed_artifacts
        if isinstance(item, Mapping) and str(item.get("case_id") or "") and str(item.get("arm") or "")
    }
    quality_by_case = _quality_by_case(quality_measurements)
    case_ids = sorted(
        {
            str(row.get("case_id") or "")
            for row in audit.get("audit_rows", ())
            if isinstance(row, Mapping) and str(row.get("case_id") or "")
        }
    )
    blockers: list[str] = []

    if audit.get("scorecard") != "dit_frontier_ab_result_artifact_audit_v0":
        blockers.append("unexpected_artifact_audit")
    if not bool(audit.get("artifact_audit_ready", audit.get("ok", False))):
        blockers.append("artifact_audit_not_ready")
    if _unsafe_flags(audit) or any(_unsafe_flags(item) for item in observed_by_case_arm.values()):
        blockers.append("unsafe_summary_input_flag")
    if not case_ids:
        blockers.append("audited_cases_missing")

    summaries = []
    for case_id in case_ids:
        row, row_blockers = _summary_row(feature, case_id, observed_by_case_arm, quality_by_case.get(case_id, {}))
        summaries.append(row)
        blockers.extend(f"{case_id}:{reason}" for reason in row_blockers)

    ready = bool(summaries) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_frontier_ab_result_summary_from_artifacts_v0",
        "ok": ready,
        "artifact_summary_ready": ready,
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
            "feed artifact-derived summaries into the shared frontier result bundle"
            if ready
            else "complete artifact audit and quality measurements before bundle conversion"
        ),
    }


def _case_plan(feature: str, case: Mapping[str, Any], artifact_root: str) -> dict[str, Any]:
    arms = FEATURE_ARMS.get(feature, ("baseline", "candidate"))
    return {
        "case_id": case["case_id"],
        "family": str(case.get("family") or "anima"),
        "feature_id": feature,
        "artifacts": [
            {
                "case_id": case["case_id"],
                "arm": arm,
                "family": str(case.get("family") or "anima"),
                "expected_result_path": str(
                    case.get(f"{arm}_result_path") or f"{artifact_root}/{case['case_id']}/{arm}_result.json"
                ),
                "required_fields": list(_arm_required_fields(feature, arm)),
                "template": {field: None for field in _arm_required_fields(feature, arm)},
            }
            for arm in arms
        ],
    }


def _arm_required_fields(feature: str, arm: str) -> tuple[str, ...]:
    fields = list(COMMON_ARM_FIELDS)
    if feature == "sra2_haste":
        fields.append("alignment_loss")
    elif feature == "cdm_qta_lora":
        fields.append("energy_per_step_j")
    elif feature == "diffcr" and arm == "candidate":
        fields.extend(("attention_fraction", "shape_stable", "disabled_parity_ok", "expand_parity_ok"))
    elif feature == "dit_blockskip" and arm == "candidate":
        fields.extend(("block_compute_fraction", "shape_stable", "disabled_parity_ok"))
        fields.extend(("checkpoint_semantics_ok", "residual_reuse_parity_ok"))
    elif feature == "dit_local_window_attention" and arm == "candidate":
        fields.extend(("attention_compute_fraction", "shape_stable", "disabled_parity_ok"))
        fields.append("masked_attention_parity_ok")
    elif feature == "dit_compute_reducer" and arm == "candidate":
        fields.append("reducer_id")
    elif feature == "tlora_ab":
        fields.extend(("train_loss", "holdout_loss", "prompt_score", "steps"))
    return tuple(fields)


def _audit_row(
    key: tuple[str, str],
    expected: Mapping[str, Any],
    observed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    case_id, arm = key
    required = tuple(str(item) for item in expected.get("required_fields", ()) if str(item).strip())
    if observed is None:
        return {
            "case_id": case_id,
            "arm": arm,
            "ok": False,
            "missing_fields": list(required),
            "blocked_reasons": ["artifact_missing"],
        }
    missing = [field for field in required if field not in observed]
    blockers = [f"artifact_field_missing:{field}" for field in missing]
    if _unsafe_flags(observed):
        blockers.append("unsafe_artifact_flag")
    return {
        "case_id": case_id,
        "arm": arm,
        "ok": not blockers,
        "missing_fields": missing,
        "blocked_reasons": blockers,
    }


def _summary_row(
    feature: str,
    case_id: str,
    artifacts: Mapping[tuple[str, str], Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    baseline = artifacts.get((case_id, "baseline"), {})
    candidate = artifacts.get((case_id, "tlora" if feature == "tlora_ab" else "candidate"), {})
    blockers: list[str] = []
    if not baseline:
        blockers.append("baseline_artifact_missing")
    if not candidate:
        blockers.append("candidate_artifact_missing")
    if feature == "tlora_ab":
        row, tlora_blockers = _tlora_summary(case_id, baseline, candidate)
        return row, [*blockers, *tlora_blockers]

    row: dict[str, Any] = {
        "case_id": case_id,
        "baseline_step_time_ms": _float(baseline.get("step_time_ms")),
        "candidate_step_time_ms": _float(candidate.get("step_time_ms")),
        "baseline_peak_vram_mb": _float(baseline.get("peak_vram_mb")),
        "candidate_peak_vram_mb": _float(candidate.get("peak_vram_mb")),
    }
    if feature == "sra2_haste":
        row["baseline_alignment_loss"] = _float(baseline.get("alignment_loss"))
        row["candidate_alignment_loss"] = _float(candidate.get("alignment_loss"))
        row["quality_drift"] = _required_float(quality, "quality_drift", blockers)
        row["holdout_loss_delta"] = _required_float(quality, "holdout_loss_delta", blockers)
    elif feature == "cdm_qta_lora":
        row["baseline_energy_per_step_j"] = _float(baseline.get("energy_per_step_j"))
        row["candidate_energy_per_step_j"] = _float(candidate.get("energy_per_step_j"))
        row["quality_drift"] = _required_float(quality, "quality_drift", blockers)
        row["loss_delta"] = _required_float(quality, "loss_delta", blockers)
        row["optimizer_state_parity"] = bool(quality.get("optimizer_state_parity", candidate.get("optimizer_state_parity")))
    elif feature == "diffcr":
        row["candidate_attention_fraction"] = _float(candidate.get("attention_fraction"))
        _quality_loss_and_flags(row, quality, candidate, blockers, ("shape_stable", "disabled_parity_ok", "expand_parity_ok"))
    elif feature == "dit_blockskip":
        row["candidate_block_compute_fraction"] = _float(candidate.get("block_compute_fraction"))
        _quality_loss_and_flags(
            row,
            quality,
            candidate,
            blockers,
            ("shape_stable", "disabled_parity_ok", "checkpoint_semantics_ok", "residual_reuse_parity_ok"),
        )
    elif feature == "dit_local_window_attention":
        row = {
            "case_id": case_id,
            "step_time_improvement": _improvement(baseline.get("step_time_ms"), candidate.get("step_time_ms")),
            "attention_compute_fraction": _float(candidate.get("attention_compute_fraction")),
            "vram_regression": _regression(baseline.get("peak_vram_mb"), candidate.get("peak_vram_mb")),
            "quality_drift": _required_float(quality, "quality_drift", blockers),
            "loss_delta": _required_float(quality, "loss_delta", blockers),
            "shape_stable": bool(quality.get("shape_stable", candidate.get("shape_stable"))),
            "disabled_parity_ok": bool(quality.get("disabled_parity_ok", candidate.get("disabled_parity_ok"))),
            "masked_attention_parity_ok": bool(
                quality.get("masked_attention_parity_ok", candidate.get("masked_attention_parity_ok"))
            ),
        }
    elif feature == "dit_compute_reducer":
        row["reducer_id"] = str(candidate.get("reducer_id") or quality.get("reducer_id") or case_id)
        row["quality_drift"] = _required_float(quality, "quality_drift", blockers)
        row["loss_delta"] = _required_float(quality, "loss_delta", blockers)
        row.pop("case_id", None)
    else:
        blockers.append("unsupported_feature_id")
    return row, blockers


def _tlora_summary(case_id: str, baseline: Mapping[str, Any], tlora: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    return {
        "case_id": case_id,
        "baseline": _adapter_metrics("baseline", baseline),
        "tlora": _adapter_metrics("tlora", tlora),
    }, []


def _adapter_metrics(method: str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": str(artifact.get("method") or method),
        "train_loss": _float(artifact.get("train_loss")),
        "holdout_loss": _float(artifact.get("holdout_loss")),
        "prompt_score": _float(artifact.get("prompt_score")),
        "steps": int(artifact.get("steps") or 0),
        "image_count": int(artifact.get("image_count") or 1),
    }


def _quality_loss_and_flags(
    row: dict[str, Any],
    quality: Mapping[str, Any],
    candidate: Mapping[str, Any],
    blockers: list[str],
    flags: Sequence[str],
) -> None:
    row["quality_drift"] = _required_float(quality, "quality_drift", blockers)
    row["loss_delta"] = _required_float(quality, "loss_delta", blockers)
    for flag in flags:
        row[flag] = bool(quality.get(flag, candidate.get(flag)))


def _quality_by_case(value: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        if "case_id" in value:
            return {str(value.get("case_id") or "case"): dict(value)}
        return {str(key): dict(item) for key, item in value.items() if isinstance(item, Mapping)}
    return {str(item.get("case_id") or ""): dict(item) for item in value if isinstance(item, Mapping)}


def _case(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    case_id = str(item.get("case_id") or item.get("reducer_id") or f"case-{index}")
    return {**dict(item), "case_id": case_id}


def _supported_feature(feature: str) -> bool:
    return feature in {
        "sra2_haste",
        "cdm_qta_lora",
        "diffcr",
        "dit_blockskip",
        "dit_local_window_attention",
        "dit_compute_reducer",
        "tlora_ab",
    }


def _required_float(payload: Mapping[str, Any], key: str, blockers: list[str]) -> float:
    if key not in payload:
        blockers.append(f"quality_measurement_missing:{key}")
    return _float(payload.get(key))


def _improvement(baseline: Any, candidate: Any) -> float:
    base = _float(baseline)
    return 0.0 if base <= 0.0 else (base - _float(candidate)) / base


def _regression(baseline: Any, candidate: Any) -> float:
    base = _float(baseline)
    return 0.0 if base <= 0.0 else (_float(candidate) - base) / base


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _feature(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return sorted(dupes)


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
        "request_payload_materialized",
        "request_payload_submitted",
        "execution_job_created",
        "training_job_created",
        "job_record_written",
        "queue_enqueued",
        "training_launch_allowed",
        "training_launch_executed",
        "training_runtime_started",
        "training_process_started",
        "operator_training_launch_allowed",
        "operator_training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "COMMON_ARM_FIELDS",
    "build_dit_frontier_ab_result_artifact_audit",
    "build_dit_frontier_ab_result_artifact_plan",
    "build_dit_frontier_ab_result_summary_from_artifacts",
]
