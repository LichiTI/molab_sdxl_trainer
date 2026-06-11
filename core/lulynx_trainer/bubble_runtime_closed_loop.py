"""Closed-loop state report for bubble-aware runtime tuning."""

from __future__ import annotations

from typing import Any, Mapping

from .bubble_runtime_closed_loop_executor import build_closed_loop_executor_state


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _normalized_mode(mode: Any) -> str:
    value = str(mode or "report_only").strip().lower().replace("-", "_")
    if value in {"advisor", "advisory", "advisor_patch"}:
        return "advisor_patch"
    if value in {"auto", "auto_apply"}:
        return "auto_apply"
    return "report_only"


def _state(name: str, status: str, reason: str, **fields: Any) -> dict[str, Any]:
    payload = {"name": name, "status": status, "reason": reason}
    payload.update(fields)
    return payload


def build_closed_loop_state(
    *,
    snapshot: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    action_plan: Mapping[str, Any],
    mode: Any,
    prior_state: Mapping[str, Any] | None = None,
    current_step: int | None = None,
) -> dict[str, Any]:
    config = _mapping(snapshot.get("config"))
    resolved_mode = _normalized_mode(mode)
    enabled = _safe_bool(config.get("controller_enabled"), False)
    action_kind = str(_mapping(diagnosis.get("recommended_action")).get("kind") or "")
    action_status = str(action_plan.get("status") or "")
    has_patch = bool(action_plan.get("mutations"))
    diagnosis_kind = str(diagnosis.get("kind") or "unknown")
    rollback = _mapping(action_plan.get("rollback"))

    policy = {
        "warmup_steps": max(_safe_int(config.get("warmup_steps"), 8), 0),
        "tune_interval_steps": max(_safe_int(config.get("tune_interval_steps"), 32), 1),
        "max_actions_per_run": max(_safe_int(config.get("max_actions_per_run"), 3), 0),
        "min_throughput_gain": max(_safe_float(config.get("min_throughput_gain"), 0.03), 0.0),
        "cooldown_steps": max(_safe_int(config.get("tune_interval_steps"), 32), 1),
        "cross_run_cooldown_runs": max(_safe_int(config.get("cross_run_cooldown_runs"), 1), 0),
        "rollback_max_regression_ratio": _safe_float(rollback.get("max_regression_ratio"), 0.0),
        "compare_window": str(rollback.get("compare_window") or "post_warmup_steady_window"),
    }
    executor = build_closed_loop_executor_state(
        snapshot=snapshot,
        diagnosis=diagnosis,
        action_plan=action_plan,
        mode=mode,
        prior_state=prior_state,
        current_step=current_step,
    )

    if not enabled:
        status = "disabled"
        apply_status = "blocked"
        apply_reason = "bubble_controller_enabled is false"
    elif resolved_mode == "report_only":
        status = "observe_only"
        apply_status = "blocked"
        apply_reason = "controller mode is report_only"
    elif resolved_mode == "advisor_patch":
        status = "advisor_patch_only"
        apply_status = "blocked"
        apply_reason = "advisor_patch only emits a next-request config overlay"
    elif executor.get("status") == "ready_to_apply":
        status = "auto_apply_ready"
        apply_status = "ready"
        apply_reason = str(executor.get("reason") or "low-risk runtime action is ready")
    elif executor.get("status") == "dataloader_rebuild_epoch_boundary_ready":
        status = "dataloader_rebuild_epoch_boundary_ready"
        apply_status = "ready"
        apply_reason = str(executor.get("reason") or "DataLoader rebuild is ready for the next epoch boundary")
    elif executor.get("status") == "cooldown":
        status = "auto_apply_cooldown"
        apply_status = "complete"
        apply_reason = str(executor.get("reason") or "applied action is cooling down")
    elif executor.get("status") in {"keep_recommended", "keep_observed"}:
        status = str(executor.get("status"))
        apply_status = "complete"
        apply_reason = str(executor.get("reason") or "action evaluation completed")
    elif executor.get("status") == "rollback_recommended":
        status = "rollback_recommended"
        apply_status = "rollback_ready"
        apply_reason = str(executor.get("reason") or "rollback is recommended")
    elif executor.get("status") == "dataloader_rebuild_rollback_epoch_boundary_ready":
        status = "dataloader_rebuild_rollback_epoch_boundary_ready"
        apply_status = "rollback_ready"
        apply_reason = str(executor.get("reason") or "DataLoader rollback is ready for the next epoch boundary")
    elif executor.get("status") == "needs_more_evidence":
        status = "needs_more_evidence"
        apply_status = "blocked"
        apply_reason = str(executor.get("reason") or "requires more evidence")
    elif not has_patch:
        status = "auto_apply_no_patch"
        apply_status = "blocked"
        apply_reason = f"action plan status is {action_status or 'empty'}"
    else:
        status = str(executor.get("status") or "auto_apply_blocked_pending_runtime_executor")
        apply_status = "blocked"
        apply_reason = str(executor.get("reason") or "P7 runtime executor is not armed")

    state_machine = [
        _state("observe", "complete", "runtime snapshot was built"),
        _state(
            "diagnose",
            "complete" if diagnosis_kind not in {"unknown", ""} else "needs_more_evidence",
            f"diagnosis={diagnosis_kind}",
            diagnosis_kind=diagnosis_kind,
        ),
        _state(
            "propose",
            "complete" if action_kind else "needs_more_evidence",
            f"action={action_kind or 'none'}, plan_status={action_status or 'none'}",
            action_kind=action_kind,
            action_plan_status=action_status,
        ),
        _state("apply_one_action", apply_status, apply_reason),
        _state(
            "cooldown",
            "active" if status == "auto_apply_cooldown" else "complete" if status in {"keep_recommended", "keep_observed", "rollback_recommended", "dataloader_rebuild_rollback_epoch_boundary_ready"} else "pending",
            "requires a successful apply_one_action transition",
        ),
        _state(
            "evaluate",
            "complete" if executor.get("evaluation") else "pending",
            "requires before/after throughput windows",
            evaluation=executor.get("evaluation", {}),
        ),
        _state(
            "keep_or_rollback",
            "rollback_ready" if status in {"rollback_recommended", "dataloader_rebuild_rollback_epoch_boundary_ready"} else "complete" if status in {"keep_recommended", "keep_observed"} else "pending",
            "requires evaluation against rollback policy",
        ),
    ]

    return {
        "schema_version": 1,
        "controller": "bubble_closed_loop_state_v0",
        "phase": "P7_online_closed_loop_gate",
        "mode": resolved_mode,
        "enabled": enabled,
        "status": status,
        "safe_to_auto_apply": bool(executor.get("safe_to_auto_apply", False)),
        "can_apply_during_current_run": bool(executor.get("can_apply_during_current_run", False)),
        "can_apply_to_next_request": bool(action_plan.get("can_apply_to_next_request")),
        "candidate_action": action_kind,
        "policy": policy,
        "state_machine": state_machine,
        "executor": executor,
        "action_history": list(executor.get("action_history", [])),
        "notes": [
            "P7 runtime executor only arms low-risk current-run mutations with rollback adapters.",
            "Transfer non_blocking can use transfer_runtime_overlay_v0; pin_memory and block prefetch remain next-request boundaries.",
            "DataLoader rebuild is guarded by an explicit current-run gate and only applies at epoch boundaries.",
        ],
    }


__all__ = ["build_closed_loop_state"]
