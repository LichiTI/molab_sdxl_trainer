"""Report-only A/B evidence package for DiT compute reducer candidates."""

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
REQUIRED_OUTPUTS = ("baseline_metrics", "candidate_metrics", "quality_report", "loss_report")


def build_dit_compute_reducer_ab_evidence_package(
    *,
    candidate_plan: Mapping[str, Any],
    evidence_policy: Mapping[str, Any],
) -> dict[str, Any]:
    plan = dict(candidate_plan)
    policy = dict(evidence_policy)
    selected = tuple(str(item) for item in plan.get("selected_reducers", ()) if str(item).strip())
    outputs = _items(policy.get("required_outputs") or policy.get("evidence_outputs"))
    metrics = _items(policy.get("required_metrics") or policy.get("evidence_metrics"))
    thresholds = dict(policy.get("thresholds") or policy.get("acceptance_thresholds") or {})
    blockers: list[str] = []

    if plan.get("scorecard") != "dit_compute_reducer_candidate_plan_v0":
        blockers.append("unexpected_compute_reducer_candidate_plan")
    if not bool(plan.get("candidate_plan_ready", plan.get("ok", False))):
        blockers.append("candidate_plan_not_ready")
    if _unsafe_flags(plan, policy):
        blockers.append("unsafe_child_flag")
    for name in REQUIRED_POLICY_FIELDS:
        if not str(policy.get(name) or "").strip():
            blockers.append(f"evidence_policy_field_missing:{name}")
    for name in REQUIRED_OUTPUTS:
        if name not in outputs:
            blockers.append(f"evidence_output_missing:{name}")
    if "step_time_ms" not in metrics:
        blockers.append("metric_missing:step_time_ms")
    if "peak_vram_mb" not in metrics:
        blockers.append("metric_missing:peak_vram_mb")
    if "quality_drift" not in metrics:
        blockers.append("metric_missing:quality_drift")
    if "loss_delta" not in metrics:
        blockers.append("metric_missing:loss_delta")
    if not selected:
        blockers.append("selected_reducers_missing")
    if not bool(policy.get("report_only", False)):
        blockers.append("report_only_missing")
    if not bool(policy.get("manual_only", False)):
        blockers.append("manual_only_missing")
    if not bool(policy.get("acknowledge_no_ab_execution", False)):
        blockers.append("ab_execution_ack_missing")
    if not bool(policy.get("requires_later_ab_result_ingestion", False)):
        blockers.append("later_ab_result_ingestion_missing")
    if policy.get("ab_execution_allowed") is not False:
        blockers.append("ab_execution_allowed_must_be_false")
    if policy.get("ab_dispatch_allowed") is not False:
        blockers.append("ab_dispatch_allowed_must_be_false")
    if policy.get("trainer_wiring_allowed") is not False:
        blockers.append("trainer_wiring_allowed_must_be_false")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_ab_evidence_package_v0",
        "ok": ready,
        "evidence_package_ready": ready,
        "evidence_scope": str(policy.get("evidence_scope") or ""),
        "owner": str(policy.get("owner") or ""),
        "review_id": str(policy.get("review_id") or ""),
        "selected_reducers": list(selected),
        "required_outputs": list(outputs),
        "required_metrics": list(metrics),
        "thresholds": thresholds,
        "baseline_case_ref": str(policy.get("baseline_case_ref") or ""),
        "candidate_case_ref": str(policy.get("candidate_case_ref") or ""),
        "ab_execution_allowed": False,
        "ab_execution_started": False,
        "ab_execution_completed": False,
        "ab_dispatch_allowed": False,
        "ab_dispatch_executed": False,
        "trainer_wiring_allowed": False,
        "trainer_wiring_executed": False,
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
            "collect manual A/B results and ingest them through a separate result-ingestion gate"
            if ready
            else "complete compute-reducer A/B evidence package prerequisites"
        ),
    }


def _items(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


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
        "training_launch_allowed",
        "training_launch_executed",
        "run_dispatch_executed",
        "runs_dispatched",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = ["build_dit_compute_reducer_ab_evidence_package"]
