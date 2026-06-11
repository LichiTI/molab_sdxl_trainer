"""Result artifact plan for DiT compute reducer trainer A/B handoff.

This module defines the expected files and JSON schemas for a later operator-run
trainer A/B. It is intentionally report-only: no jobs are submitted and no
training/runtime path is enabled.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .dit_compute_reducer_trainer_ab_manifest import REQUIRED_RESULT_FIELDS


ARM_RESULT_FIELDS = (
    "case_id",
    "arm",
    "reducer_id",
    "family",
    "step_time_ms",
    "peak_vram_mb",
    "steady_state_steps",
    "cache_first",
    "native_dit",
    "loss",
)


def build_dit_compute_reducer_trainer_ab_result_artifact_plan(
    *,
    dispatch_manifest: Mapping[str, Any],
    artifact_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = dict(dispatch_manifest)
    policy = dict(artifact_policy or {})
    payloads = [dict(item) for item in manifest.get("payloads", ()) if isinstance(item, Mapping)]
    blockers: list[str] = []
    if manifest.get("scorecard") != "dit_compute_reducer_trainer_ab_dispatch_manifest_v0":
        blockers.append("unexpected_dispatch_manifest")
    if not bool(manifest.get("dispatch_manifest_ready", manifest.get("ok", False))):
        blockers.append("dispatch_manifest_not_ready")
    if bool(manifest.get("execution_performed", False)):
        blockers.append("dispatch_manifest_claims_execution")
    if _unsafe_flags(manifest, policy):
        blockers.append("unsafe_artifact_plan_input_flag")
    if not payloads:
        blockers.append("dispatch_payloads_missing")

    cases = _case_rows(payloads)
    for case_id, row in cases.items():
        arms = set(row["arms"])
        if arms != {"baseline", "candidate"}:
            blockers.append(f"{case_id}:baseline_candidate_pair_missing")
        paths = [item["expected_result_path"] for item in row["artifacts"]]
        if len(set(paths)) != len(paths):
            blockers.append(f"{case_id}:duplicate_artifact_path")

    ready = bool(cases) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_trainer_ab_result_artifact_plan_v0",
        "ok": ready,
        "artifact_plan_ready": ready,
        "result_templates_emitted": ready,
        "case_count": len(cases),
        "artifact_count": sum(len(row["artifacts"]) for row in cases.values()),
        "required_result_fields": list(REQUIRED_RESULT_FIELDS),
        "required_arm_result_fields": list(ARM_RESULT_FIELDS),
        "artifact_policy": _policy_summary(policy),
        "case_artifact_plans": list(cases.values()) if ready else [],
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run trainer A/B externally and audit produced result artifacts"
            if ready
            else "complete dispatch manifest and artifact prerequisites"
        ),
    }


def build_dit_compute_reducer_trainer_ab_result_artifact_audit(
    *,
    artifact_plan: Mapping[str, Any],
    observed_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan = dict(artifact_plan)
    observed = [dict(item) for item in observed_artifacts if isinstance(item, Mapping)]
    expected = {
        (str(item.get("case_id") or ""), str(item.get("arm") or "")): item
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
    if plan.get("scorecard") != "dit_compute_reducer_trainer_ab_result_artifact_plan_v0":
        blockers.append("unexpected_artifact_plan")
    if not bool(plan.get("artifact_plan_ready", plan.get("ok", False))):
        blockers.append("artifact_plan_not_ready")
    if _unsafe_flags(plan):
        blockers.append("unsafe_artifact_plan_flag")
    if not expected:
        blockers.append("expected_artifacts_missing")

    rows = [_audit_row(key, expected_item, observed_by_key.get(key)) for key, expected_item in expected.items()]
    blockers.extend(f"{row['case_id']}:{row['arm']}:{reason}" for row in rows for reason in row["blocked_reasons"])
    for key in sorted(set(observed_by_key) - set(expected)):
        blockers.append(f"unexpected_artifact:{key[0]}:{key[1]}")
    if any(_unsafe_flags(item) for item in observed):
        blockers.append("unsafe_observed_artifact_flag")

    ready = bool(rows) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_trainer_ab_result_artifact_audit_v0",
        "ok": ready,
        "artifact_audit_ready": ready,
        "case_count": int(plan.get("case_count") or 0),
        "expected_artifact_count": len(expected),
        "observed_artifact_count": len(observed),
        "audit_rows": rows,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "combine passing arm artifacts into trainer A/B case summaries"
            if ready
            else "collect complete arm artifacts before case-summary ingestion"
        ),
    }


def build_dit_compute_reducer_trainer_ab_case_summary_from_artifacts(
    *,
    artifact_audit: Mapping[str, Any],
    observed_artifacts: Sequence[Mapping[str, Any]],
    quality_measurements: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = dict(artifact_audit)
    observed = [dict(item) for item in observed_artifacts if isinstance(item, Mapping)]
    observed_by_case_arm = {
        (str(item.get("case_id") or ""), str(item.get("arm") or "")): item
        for item in observed
        if str(item.get("case_id") or "") and str(item.get("arm") or "")
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
    if audit.get("scorecard") != "dit_compute_reducer_trainer_ab_result_artifact_audit_v0":
        blockers.append("unexpected_artifact_audit")
    if not bool(audit.get("artifact_audit_ready", audit.get("ok", False))):
        blockers.append("artifact_audit_not_ready")
    if _unsafe_flags(audit) or any(_unsafe_flags(item) for item in observed):
        blockers.append("unsafe_case_summary_input_flag")
    if not case_ids:
        blockers.append("audited_cases_missing")

    case_results = []
    for case_id in case_ids:
        row, row_blockers = _case_summary_row(case_id, observed_by_case_arm, quality_by_case.get(case_id, {}))
        case_results.append(row)
        blockers.extend(f"{case_id}:{reason}" for reason in row_blockers)

    ready = bool(case_results) and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_trainer_ab_case_summary_from_artifacts_v0",
        "ok": ready,
        "case_summary_ready": ready,
        "case_result_count": len(case_results),
        "case_results": case_results if ready else [],
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "trainer_wiring_allowed": False,
        "runtime_activation_enabled": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "feed case_results into trainer A/B dispatch result ingestion"
            if ready
            else "complete artifact audit and quality measurements before case-summary ingestion"
        ),
    }


def _case_rows(payloads: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        case_id = str(payload.get("case_id") or "")
        arm = str(payload.get("arm") or "")
        request = dict(payload.get("request") or {})
        if not case_id or arm not in {"baseline", "candidate"}:
            continue
        row = rows.setdefault(
            case_id,
            {
                "case_id": case_id,
                "reducer_id": str(payload.get("reducer_id") or ""),
                "family": str(payload.get("family") or ""),
                "arms": [],
                "summary_result_template": _summary_template(payload),
                "artifacts": [],
            },
        )
        row["arms"].append(arm)
        row["artifacts"].append(
            {
                "case_id": case_id,
                "arm": arm,
                "reducer_id": str(payload.get("reducer_id") or ""),
                "family": str(payload.get("family") or ""),
                "expected_result_path": str(payload.get("expected_result_path") or ""),
                "required_fields": list(ARM_RESULT_FIELDS),
                "template": {
                    "case_id": case_id,
                    "arm": arm,
                    "reducer_id": str(payload.get("reducer_id") or ""),
                    "family": str(payload.get("family") or ""),
                    "cache_first": bool(request.get("cache_first", False)),
                    "native_dit": bool(request.get("native_dit_required", False)),
                    "steady_state_steps": None,
                    "step_time_ms": None,
                    "peak_vram_mb": None,
                    "loss": None,
                },
            }
        )
    return rows


def _summary_template(payload: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(payload.get("request") or {})
    return {
        "case_id": str(payload.get("case_id") or ""),
        "reducer_id": str(payload.get("reducer_id") or ""),
        "family": str(payload.get("family") or ""),
        "baseline_step_time_ms": None,
        "candidate_step_time_ms": None,
        "baseline_peak_vram_mb": None,
        "candidate_peak_vram_mb": None,
        "quality_drift": None,
        "loss_delta": None,
        "steady_state_steps": None,
        "cache_first": bool(request.get("cache_first", False)),
        "native_dit": bool(request.get("native_dit_required", False)),
    }


def _audit_row(key: tuple[str, str], expected: Mapping[str, Any], observed: Mapping[str, Any] | None) -> dict[str, Any]:
    case_id, arm = key
    blockers: list[str] = []
    if observed is None:
        blockers.append("artifact_missing")
        observed_path = ""
    else:
        observed_path = str(observed.get("path") or observed.get("result_path") or "")
        if observed_path != str(expected.get("expected_result_path") or ""):
            blockers.append("artifact_path_mismatch")
        for field in ARM_RESULT_FIELDS:
            if field not in observed:
                blockers.append(f"field_missing:{field}")
        if observed.get("cache_first") is not True:
            blockers.append("cache_first_missing")
        if observed.get("native_dit") is not True:
            blockers.append("native_dit_missing")
    return {
        "case_id": case_id,
        "arm": arm,
        "ok": not blockers,
        "expected_result_path": str(expected.get("expected_result_path") or ""),
        "observed_result_path": observed_path,
        "blocked_reasons": blockers,
    }


def _case_summary_row(
    case_id: str,
    observed_by_case_arm: Mapping[tuple[str, str], Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    baseline = dict(observed_by_case_arm.get((case_id, "baseline")) or {})
    candidate = dict(observed_by_case_arm.get((case_id, "candidate")) or {})
    blockers: list[str] = []
    if not baseline:
        blockers.append("baseline_artifact_missing")
    if not candidate:
        blockers.append("candidate_artifact_missing")
    reducer_id = str(candidate.get("reducer_id") or baseline.get("reducer_id") or "")
    family = str(candidate.get("family") or baseline.get("family") or "")
    quality_drift = _float_or_none(quality.get("quality_drift"))
    loss_delta = _float_or_none(quality.get("loss_delta"))
    if loss_delta is None and baseline and candidate:
        baseline_loss = _float_or_none(baseline.get("loss"))
        candidate_loss = _float_or_none(candidate.get("loss"))
        if baseline_loss is not None and candidate_loss is not None:
            loss_delta = candidate_loss - baseline_loss
    if quality_drift is None:
        blockers.append("quality_drift_missing")
    if loss_delta is None:
        blockers.append("loss_delta_missing")
    row = {
        "case_id": case_id,
        "reducer_id": reducer_id,
        "family": family,
        "baseline_step_time_ms": _positive_float(baseline.get("step_time_ms")),
        "candidate_step_time_ms": _positive_float(candidate.get("step_time_ms")),
        "baseline_peak_vram_mb": _positive_float(baseline.get("peak_vram_mb")),
        "candidate_peak_vram_mb": _positive_float(candidate.get("peak_vram_mb")),
        "quality_drift": quality_drift,
        "loss_delta": loss_delta,
        "steady_state_steps": min(
            int(_positive_float(baseline.get("steady_state_steps"))),
            int(_positive_float(candidate.get("steady_state_steps"))),
        ),
        "cache_first": baseline.get("cache_first") is True and candidate.get("cache_first") is True,
        "native_dit": baseline.get("native_dit") is True and candidate.get("native_dit") is True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "trainer_wiring_allowed": False,
        "ab_dispatch_allowed": False,
        "training_launch_allowed": False,
        "runs_dispatched": False,
    }
    for field in REQUIRED_RESULT_FIELDS:
        if field not in row:
            blockers.append(f"result_field_missing:{field}")
    return row, blockers


def _quality_by_case(value: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        if "case_id" in value:
            return {str(value.get("case_id") or ""): dict(value)}
        return {str(case_id): dict(payload) for case_id, payload in value.items() if isinstance(payload, Mapping)}
    return {
        str(item.get("case_id") or ""): dict(item)
        for item in value
        if isinstance(item, Mapping) and str(item.get("case_id") or "")
    }


def _policy_summary(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": str(policy.get("format") or "json"),
        "write_mode": str(policy.get("write_mode") or "operator_collected"),
        "summary_builder": str(policy.get("summary_builder") or "manual_case_summary"),
    }


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
        "default_enable_allowed",
        "auto_rollout_allowed",
        "trainer_wiring_allowed",
        "request_fields_emitted",
        "request_adapter_registered",
        "runtime_activation_enabled",
        "ab_dispatch_allowed",
        "ab_dispatch_executed",
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


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


__all__ = [
    "build_dit_compute_reducer_trainer_ab_case_summary_from_artifacts",
    "build_dit_compute_reducer_trainer_ab_result_artifact_audit",
    "build_dit_compute_reducer_trainer_ab_result_artifact_plan",
]
