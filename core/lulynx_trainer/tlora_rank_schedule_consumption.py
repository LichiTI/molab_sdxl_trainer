"""Default-off rank-schedule consumption contract for T-LoRA."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SUPPORTED_TLORA_SCHEDULES = {"constant", "linear", "geometric"}


def build_tlora_rank_schedule_consumption_plan(
    *,
    request_patch_plan: Mapping[str, Any],
    module_capability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(request_patch_plan)
    capability = dict(module_capability or {})
    patches = [dict(item) for item in plan.get("patches", ()) if isinstance(item, Mapping)]
    tlora_patches = [row for row in patches if str(row.get("arm") or "") == "tlora"]
    supported_schedules = {
        str(item).strip().lower()
        for item in capability.get("supported_schedules", SUPPORTED_TLORA_SCHEDULES)
        if str(item).strip()
    }
    max_rank = max(int(capability.get("max_rank", 0) or 0), 0)
    blockers: list[str] = []

    if plan.get("plan") != "tlora_ab_request_patch_plan_v0":
        blockers.append("unexpected_request_patch_plan")
    if not bool(plan.get("request_fields_emitted", plan.get("ok", False))):
        blockers.append("request_fields_not_emitted")
    if not bool(plan.get("dry_run_only", False)):
        blockers.append("dry_run_boundary_missing")
    if _unsafe_flags(plan, capability):
        blockers.append("unsafe_child_flag")
    if not tlora_patches:
        blockers.append("tlora_patches_missing")
    if not bool(capability.get("set_global_step_available", False)):
        blockers.append("set_global_step_hook_missing")
    if not str(capability.get("total_steps_source") or "").strip():
        blockers.append("total_steps_source_missing")
    if not bool(capability.get("rank_mask_buffer_available", False)):
        blockers.append("rank_mask_buffer_missing")

    rows = [_schedule_row(row, supported_schedules, max_rank) for row in tlora_patches]
    blockers.extend(f"{row['case_id']}:{reason}" for row in rows for reason in row["blocked_reasons"])
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_rank_schedule_consumption_plan_v0",
        "ok": ready,
        "rank_schedule_consumption_plan_ready": ready,
        "real_trainer_consumption_allowed": False,
        "case_count": len(rows),
        "schedule_rows": rows,
        "step_hook_plan": {
            "hook": "set_global_step",
            "total_steps_source": str(capability.get("total_steps_source") or ""),
            "call_frequency": "every_train_step",
            "required_before_forward": True,
        },
        "request_fields_emitted": False,
        "request_adapter_registered": False,
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
            "audit observed T-LoRA rank schedule updates in representative trainer evidence"
            if ready
            else "complete T-LoRA rank schedule consumption prerequisites"
        ),
    }


def build_tlora_rank_schedule_consumption_audit(
    *,
    consumption_plan: Mapping[str, Any],
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(consumption_plan)
    payload = dict(observed or {})
    rows = [dict(item) for item in plan.get("schedule_rows", ()) if isinstance(item, Mapping)]
    observed_rows = {
        str(item.get("case_id") or ""): dict(item)
        for item in payload.get("observed_schedule_rows", ())
        if isinstance(item, Mapping)
    }
    blockers: list[str] = []

    if plan.get("scorecard") != "tlora_rank_schedule_consumption_plan_v0":
        blockers.append("unexpected_consumption_plan")
    if not bool(plan.get("rank_schedule_consumption_plan_ready", plan.get("ok", False))):
        blockers.append("consumption_plan_not_ready")
    if _unsafe_flags(plan, payload):
        blockers.append("unsafe_child_flag")
    if not bool(payload.get("set_global_step_called_before_forward", False)):
        blockers.append("set_global_step_not_called_before_forward")
    if not bool(payload.get("rank_mask_updated", False)):
        blockers.append("rank_mask_not_updated")

    for row in rows:
        case_id = str(row.get("case_id") or "")
        observed_row = observed_rows.get(case_id)
        if observed_row is None:
            blockers.append(f"{case_id}:observed_schedule_missing")
            continue
        if str(observed_row.get("schedule") or "") != str(row.get("schedule") or ""):
            blockers.append(f"{case_id}:schedule_mismatch")
        expected = list(row.get("expected_rank_trace") or [])
        actual = list(observed_row.get("observed_rank_trace") or [])
        if expected != actual:
            blockers.append(f"{case_id}:rank_trace_mismatch")

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "tlora_rank_schedule_consumption_audit_v0",
        "ok": ready,
        "rank_schedule_consumption_audit_ready": ready,
        "case_count": len(rows),
        "observed_case_count": len(observed_rows),
        "request_fields_emitted": False,
        "request_adapter_registered": False,
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
            "feed T-LoRA schedule-consumption evidence into trainer A/B review"
            if ready
            else "fix observed T-LoRA rank schedule consumption evidence"
        ),
    }


def _schedule_row(row: Mapping[str, Any], supported_schedules: set[str], max_rank: int) -> dict[str, Any]:
    patch = dict(row.get("request_patch") or {})
    case_id = str(row.get("case_id") or patch.get("tlora_ab_case_id") or "")
    schedule = str(patch.get("tlora_rank_schedule") or "").strip().lower()
    min_rank = max(int(patch.get("tlora_min_rank") or 0), 0)
    total_steps = max(int(patch.get("max_train_steps") or 0), 0)
    blockers: list[str] = []
    if not case_id:
        blockers.append("case_id_missing")
    if schedule not in SUPPORTED_TLORA_SCHEDULES:
        blockers.append("unsupported_tlora_schedule")
    if supported_schedules and schedule not in supported_schedules:
        blockers.append("schedule_not_supported_by_module")
    if min_rank <= 0:
        blockers.append("tlora_min_rank_invalid")
    if max_rank > 0 and min_rank > max_rank:
        blockers.append("tlora_min_rank_exceeds_max_rank")
    if total_steps <= 0:
        blockers.append("max_train_steps_missing")
    effective_max_rank = max(max_rank, min_rank)
    trace_steps = _trace_steps(total_steps)
    return {
        "case_id": case_id,
        "family": str(row.get("family") or patch.get("model_family") or ""),
        "schedule": schedule,
        "min_rank": int(min_rank),
        "max_rank": int(effective_max_rank),
        "total_steps": int(total_steps),
        "trace_steps": trace_steps,
        "expected_rank_trace": [
            _expected_rank(step, schedule=schedule, min_rank=min_rank, max_rank=effective_max_rank, total_steps=total_steps)
            for step in trace_steps
        ],
        "blocked_reasons": blockers,
    }


def _expected_rank(step: int, *, schedule: str, min_rank: int, max_rank: int, total_steps: int) -> int:
    if schedule == "constant" or max_rank <= min_rank:
        return int(min_rank)
    progress = min(max(float(step) / max(float(total_steps), 1.0), 0.0), 1.0)
    if schedule == "geometric":
        ratio = max_rank / max(float(min_rank), 1.0)
        rank = min_rank * (ratio**progress)
    else:
        rank = min_rank + (max_rank - min_rank) * progress
    return int(max(min_rank, min(round(rank), max_rank)))


def _trace_steps(total_steps: int) -> list[int]:
    total = max(int(total_steps), 1)
    return sorted({0, total // 2, total})


def _unsafe_flags(*payloads: Mapping[str, Any]) -> bool:
    unsafe_keys = (
        "request_adapter_registered",
        "training_launch_allowed",
        "training_launch_executed",
        "runs_dispatched",
        "trainer_wiring_allowed",
        "runtime_activation_enabled",
        "training_path_enabled",
        "default_behavior_changed",
        "promotion_ready",
    )
    return any(bool(payload.get(key, False)) for payload in payloads for key in unsafe_keys)


__all__ = [
    "build_tlora_rank_schedule_consumption_audit",
    "build_tlora_rank_schedule_consumption_plan",
]
