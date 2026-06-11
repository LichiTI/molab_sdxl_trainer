"""Evidence reports for P7 bubble closed-loop runtime actions."""

from __future__ import annotations

from typing import Any, Mapping


CLOSED_LOOP_EVIDENCE_REPORT = "bubble_runtime_closed_loop_evidence_v0"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str, digits: int = 6) -> float:
    return round(_safe_float(after.get(key)) - _safe_float(before.get(key)), digits)


def _copy_present(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value is None or value == "":
            continue
        result[key] = value
    return result


def _metrics(source: Mapping[str, Any]) -> dict[str, Any]:
    return _copy_present(
        source,
        (
            "steady_samples_per_second",
            "throughput_estimated",
            "mean_step_ms",
            "active_gpu_util_pct_mean",
            "active_gpu_saturated_sample_ratio",
            "memory_ratio",
            "data_wait_share",
            "h2d_transfer_share",
            "optimizer_share",
            "host_gap_share",
            "logging_checkpoint_share",
            "final_loss",
            "window_step_count",
        ),
    )


def _action(source: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _mapping(source.get("evaluation"))
    before = _mapping(evaluation.get("before"))
    after = _mapping(evaluation.get("after"))
    action = {
        **_copy_present(
            source,
            (
                "schema_version",
                "ledger",
                "action_id",
                "adapter_id",
                "apply_boundary",
                "status",
                "domain",
                "action_kind",
                "applied_step",
                "cooldown_until_step",
                "closed_step",
                "diagnosis_kind",
            ),
        ),
        "applied_overlay": dict(_mapping(source.get("applied_overlay"))),
        "rollback_restore": dict(_mapping(source.get("rollback_restore"))),
        "rollback_adapter": dict(_mapping(source.get("rollback_adapter"))),
        "rollback_applied_overlay": dict(_mapping(source.get("rollback_applied_overlay"))),
        "dataloader_rebuild": dict(_mapping(source.get("dataloader_rebuild"))),
        "profiler_handoff": dict(_mapping(source.get("profiler_handoff"))),
        "before_metrics": _metrics(_mapping(source.get("before_metrics"))),
        "evaluation": {
            **_copy_present(
                evaluation,
                (
                    "steady_samples_per_second_gain_ratio",
                    "steady_samples_per_second_gain_pct",
                    "evaluated_step",
                ),
            ),
            "before": _metrics(before),
            "after": _metrics(after),
        },
    }
    return action


def _state_from_source(source: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if source.get("manifest_version") or source.get("extra"):
        extra = _mapping(source.get("extra"))
        return (
            _mapping(extra.get("bubble_closed_loop_state")),
            _mapping(extra.get("bubble_controller")),
            _mapping(source.get("config")),
        )
    if source.get("controller") == "bubble_aware_runtime_controller_v0":
        return _mapping(_mapping(source.get("closed_loop")).get("executor")), source, _mapping(_mapping(source.get("snapshot")).get("config"))
    if source.get("report") == "bubble_runtime_closed_loop_state_v0":
        return source, {}, {}
    return source, {}, {}


def _actions_from_state(state: Mapping[str, Any], controller: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_history = state.get("action_history")
    if not isinstance(raw_history, list):
        raw_history = _mapping(_mapping(controller.get("closed_loop")).get("executor")).get("action_history")
    history = raw_history if isinstance(raw_history, list) else []
    actions = [_action(_mapping(item)) for item in history if isinstance(item, Mapping)]
    actions = [item for item in actions if item.get("action_id") or item.get("status")]
    active = _mapping(state.get("active_action"))
    if active and not any(str(item.get("action_id") or "") == str(active.get("action_id") or "") for item in actions):
        actions.append(_action(active))
    return actions


def _executor_from_controller(controller: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(controller.get("closed_loop")).get("executor"))


def _runtime_adapter_from_controller(controller: Mapping[str, Any]) -> dict[str, Any]:
    adapter = _mapping(_executor_from_controller(controller).get("runtime_adapter"))
    return dict(adapter) if adapter else {}


def _decision(actions: list[Mapping[str, Any]], controller: Mapping[str, Any]) -> dict[str, Any]:
    statuses = [str(item.get("status") or "") for item in actions]
    closed_loop = _mapping(controller.get("closed_loop"))
    executor = _mapping(closed_loop.get("executor"))
    runtime_adapter = _mapping(executor.get("runtime_adapter"))
    blocked = [str(item) for item in executor.get("blocked_reasons", [])] if isinstance(executor.get("blocked_reasons"), list) else []
    reasons: list[str] = []
    if "rollback_failed" in statuses:
        status = "needs_review"
        reasons.append("rollback_failed")
    elif "needs_more_evidence" in statuses:
        status = "needs_review"
        reasons.append("needs_more_evidence")
    elif "rolled_back" in statuses:
        status = "rollback_observed"
        reasons.append("rollback_path_exercised")
    elif "kept" in statuses:
        status = "keep_observed"
        reasons.append("closed_loop_action_kept")
    elif any(item in {"applied", "cooldown"} for item in statuses):
        status = "cooldown"
        reasons.append("action_pending_evaluation")
    elif "action_already_attempted" in blocked or str(executor.get("status") or "") == "action_already_attempted":
        status = "duplicate_blocked"
        reasons.append("duplicate_action_blocked")
    elif "cross_run_action_cooldown" in blocked or str(executor.get("status") or "") == "cross_run_action_cooldown":
        status = "cross_run_cooldown_blocked"
        reasons.append("cross_run_action_cooldown")
    elif (
        "missing_current_run_rollback_adapter" in blocked
        or "transfer_prefetch_next_request_only" in blocked
        or bool(runtime_adapter.get("next_request_only"))
    ):
        status = "next_request_adapter_blocked"
        reasons.append("next_request_adapter_boundary")
    else:
        status = "no_action"
        reasons.append("no_closed_loop_action")
    reasons.extend(item for item in blocked if item not in reasons)
    return {
        "status": status,
        "reasons": reasons[:8],
        "executor_status": str(executor.get("status") or closed_loop.get("status") or ""),
        "executor_reason": str(executor.get("reason") or ""),
    }


def build_bubble_closed_loop_evidence_report(
    source: Mapping[str, Any],
    *,
    case_id: str = "",
    family: str = "",
) -> dict[str, Any]:
    """Build compact evidence from a run manifest, controller report, or closed-loop state."""

    state, controller, config = _state_from_source(_mapping(source))
    actions = _actions_from_state(state, controller)
    decision = _decision(actions, controller)
    runtime_adapter = _runtime_adapter_from_controller(controller)
    latest = actions[-1] if actions else {}
    latest_eval = _mapping(latest.get("evaluation"))
    after = _mapping(latest_eval.get("after"))
    before = _mapping(latest_eval.get("before"))
    return {
        "schema_version": 1,
        "report": CLOSED_LOOP_EVIDENCE_REPORT,
        "status": decision["status"],
        "case_id": str(case_id or config.get("output_name") or config.get("config_name") or "bubble_closed_loop"),
        "family": str(family or config.get("model_arch") or config.get("model_type") or ""),
        "mode": str(_mapping(controller.get("closed_loop")).get("mode") or _mapping(controller.get("closed_loop")).get("status") or ""),
        "action_count": len(actions),
        "kept_count": sum(1 for item in actions if str(item.get("status") or "") == "kept"),
        "rolled_back_count": sum(1 for item in actions if str(item.get("status") or "") == "rolled_back"),
        "rollback_failed_count": sum(1 for item in actions if str(item.get("status") or "") == "rollback_failed"),
        "needs_more_evidence_count": sum(1 for item in actions if str(item.get("status") or "") == "needs_more_evidence"),
        "latest_action": dict(latest),
        "actions": actions[-10:],
        "runtime_adapter": runtime_adapter,
        "comparison": {
            "steady_samples_per_second_before": _round(before.get("steady_samples_per_second"), 6),
            "steady_samples_per_second_after": _round(after.get("steady_samples_per_second"), 6),
            "steady_samples_per_second_gain_pct": _round(latest_eval.get("steady_samples_per_second_gain_pct"), 4),
            "active_gpu_util_pct_before": _round(before.get("active_gpu_util_pct_mean"), 4),
            "active_gpu_util_pct_after": _round(after.get("active_gpu_util_pct_mean"), 4),
            "active_gpu_util_pct_delta": _metric_delta(before, after, "active_gpu_util_pct_mean", 4),
            "host_gap_share_before": _round(before.get("host_gap_share")),
            "host_gap_share_after": _round(after.get("host_gap_share")),
            "host_gap_share_delta": _metric_delta(before, after, "host_gap_share"),
            "data_wait_share_before": _round(before.get("data_wait_share")),
            "data_wait_share_after": _round(after.get("data_wait_share")),
            "data_wait_share_delta": _metric_delta(before, after, "data_wait_share"),
            "h2d_transfer_share_before": _round(before.get("h2d_transfer_share")),
            "h2d_transfer_share_after": _round(after.get("h2d_transfer_share")),
            "h2d_transfer_share_delta": _metric_delta(before, after, "h2d_transfer_share"),
            "optimizer_share_before": _round(before.get("optimizer_share")),
            "optimizer_share_after": _round(after.get("optimizer_share")),
            "optimizer_share_delta": _metric_delta(before, after, "optimizer_share"),
            "logging_checkpoint_share_before": _round(before.get("logging_checkpoint_share")),
            "logging_checkpoint_share_after": _round(after.get("logging_checkpoint_share")),
            "logging_checkpoint_share_delta": _metric_delta(before, after, "logging_checkpoint_share"),
            "final_loss_before": _round(before.get("final_loss"), 6),
            "final_loss_after": _round(after.get("final_loss"), 6),
        },
        "decision": decision,
        "safety": {
            "current_run_allowlist_only": True,
            "high_risk_actions_remain_next_request": True,
            "default_auto_apply": False,
            "duplicate_action_blocked": "duplicate_action_blocked" in decision.get("reasons", []),
            "cross_run_action_blocked": "cross_run_action_cooldown" in decision.get("reasons", []),
            "current_run_adapter_blocked": "missing_current_run_rollback_adapter" in decision.get("reasons", []),
            "next_request_only_adapter": bool(runtime_adapter.get("next_request_only")),
            "required_evidence": list(runtime_adapter.get("required_evidence") or []),
        },
    }


__all__ = [
    "CLOSED_LOOP_EVIDENCE_REPORT",
    "build_bubble_closed_loop_evidence_report",
]
