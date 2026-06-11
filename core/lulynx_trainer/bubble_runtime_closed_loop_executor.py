"""Runtime executor planning for bubble-aware closed-loop tuning."""

from __future__ import annotations

from typing import Any, Mapping

from .bubble_runtime_rollback_adapters import (
    DATALOADER_REBUILD_RUNTIME_CONTRACT_ID,
    LOW_RISK_RUNTIME_ACTIONS,
    LOW_RISK_RUNTIME_PATHS,
    build_runtime_apply_candidate,
    build_runtime_rollback_plan,
    finalize_runtime_apply_candidate,
    runtime_action_support,
    stable_runtime_action_id,
)

CROSS_RUN_COOLDOWN_STATUSES = {
    "apply_failed",
    "needs_more_evidence",
    "rollback_failed",
    "rolled_back",
}


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


def _round(value: Any, digits: int = 6) -> float:
    return round(_safe_float(value), digits)


def _normalized_mode(mode: Any) -> str:
    value = str(mode or "report_only").strip().lower().replace("-", "_")
    if value in {"advisor", "advisory", "advisor_patch"}:
        return "advisor_patch"
    if value in {"auto", "auto_apply"}:
        return "auto_apply"
    return "report_only"


def _policy(snapshot: Mapping[str, Any], action_plan: Mapping[str, Any]) -> dict[str, Any]:
    config = _mapping(snapshot.get("config"))
    rollback = _mapping(action_plan.get("rollback"))
    return {
        "warmup_steps": max(_safe_int(config.get("warmup_steps"), 8), 0),
        "tune_interval_steps": max(_safe_int(config.get("tune_interval_steps"), 32), 1),
        "max_actions_per_run": max(_safe_int(config.get("max_actions_per_run"), 3), 0),
        "min_throughput_gain": max(_safe_float(config.get("min_throughput_gain"), 0.03), 0.0),
        "cooldown_steps": max(_safe_int(config.get("tune_interval_steps"), 32), 1),
        "cross_run_cooldown_runs": max(_safe_int(config.get("cross_run_cooldown_runs"), 1), 0),
        "rollback_max_regression_ratio": max(_safe_float(rollback.get("max_regression_ratio"), 0.02), 0.0),
        "compare_window": str(rollback.get("compare_window") or "runtime_closed_loop_window"),
    }


def _metrics(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    step = _mapping(snapshot.get("step_phase"))
    gpu = _mapping(snapshot.get("gpu"))
    runtime = _mapping(snapshot.get("runtime"))
    safety = _mapping(snapshot.get("safety"))
    throughput = _safe_float(step.get("steady_samples_per_second"), 0.0)
    throughput_estimated = _safe_bool(step.get("throughput_estimated"), False)
    mean_step_ms = _safe_float(step.get("mean_step_ms"), 0.0)
    if throughput <= 0.0 and mean_step_ms > 0.0:
        batch = max(_safe_float(runtime.get("train_batch_size"), 1.0), 1.0)
        throughput = batch / max(mean_step_ms / 1000.0, 1e-9)
        throughput_estimated = True
    return {
        "steady_samples_per_second": _round(throughput),
        "throughput_estimated": throughput_estimated,
        "mean_step_ms": _round(mean_step_ms, 4),
        "active_gpu_util_pct_mean": _round(gpu.get("active_gpu_util_pct_mean"), 4),
        "active_gpu_saturated_sample_ratio": _round(gpu.get("active_gpu_saturated_sample_ratio")),
        "memory_ratio": _round(safety.get("memory_ratio")),
        "data_wait_share": _round(step.get("data_wait_share")),
        "h2d_transfer_share": _round(step.get("h2d_transfer_share")),
        "optimizer_share": _round(step.get("optimizer_share")),
        "host_gap_share": _round(step.get("host_gap_share")),
        "logging_checkpoint_share": _round(step.get("logging_checkpoint_share")),
        "final_loss": _round(step.get("final_loss")),
        "window_step_count": _safe_int(step.get("window_step_count")),
    }


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str, digits: int = 6) -> float:
    return _round(_safe_float(after.get(key)) - _safe_float(before.get(key)), digits)


def _evidence_quality(before: Mapping[str, Any], after: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    min_steps = max(_safe_int(policy.get("tune_interval_steps"), 1), 1)
    before_count = _safe_int(before.get("window_step_count"))
    after_count = _safe_int(after.get("window_step_count"))
    blocked: list[str] = []
    warnings: list[str] = []
    if "window_step_count" in before and before_count > 0 and before_count < min_steps:
        blocked.append("before_window_too_short")
    if "window_step_count" in after and after_count > 0 and after_count < min_steps:
        blocked.append("after_window_too_short")
    if _safe_bool(before.get("throughput_estimated"), False):
        warnings.append("before_throughput_estimated")
    if _safe_bool(after.get("throughput_estimated"), False):
        warnings.append("after_throughput_estimated")
    return {
        "min_window_step_count": min_steps,
        "before_window_step_count": before_count,
        "after_window_step_count": after_count,
        "throughput_estimated": bool(warnings),
        "warnings": warnings,
        "blocked_reasons": blocked,
    }


def _history(prior_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = prior_state.get("action_history")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _active_action(prior_state: Mapping[str, Any]) -> dict[str, Any]:
    active = _mapping(prior_state.get("active_action"))
    if active.get("status") in {"applied", "cooldown"}:
        return dict(active)
    return {}


def _empty_executor(
    *,
    status: str,
    reason: str,
    mode: str,
    action_plan: Mapping[str, Any],
    policy: Mapping[str, Any],
    prior_state: Mapping[str, Any],
    blocked_reasons: list[str] | None = None,
    runtime_adapter: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "executor": "bubble_runtime_closed_loop_executor_v0",
        "status": status,
        "reason": reason,
        "mode": mode,
        "safe_to_auto_apply": False,
        "can_apply_during_current_run": False,
        "candidate_action": str(action_plan.get("action_kind") or ""),
        "domain": str(action_plan.get("domain") or ""),
        "policy": dict(policy),
        "runtime_apply": {},
        "active_action": _active_action(prior_state),
        "evaluation": {},
        "rollback": {},
        "runtime_adapter": dict(_mapping(runtime_adapter)),
        "action_history": _history(prior_state),
        "blocked_reasons": list(blocked_reasons or []),
    }


def _count_closed_actions(history: list[Mapping[str, Any]]) -> int:
    closed = {"applied", "apply_failed", "kept", "rolled_back", "rollback_failed", "needs_more_evidence"}
    return sum(1 for item in history if str(item.get("status") or "") in closed)


def _find_attempted_action(history: list[Mapping[str, Any]], action_id: str) -> Mapping[str, Any]:
    if not action_id:
        return {}
    closed = {"applied", "apply_failed", "kept", "rolled_back", "rollback_failed", "needs_more_evidence"}
    for item in reversed(history):
        if str(item.get("action_id") or "") == action_id and str(item.get("status") or "") in closed:
            return item
    return {}


def _cross_run_history(snapshot: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = _mapping(snapshot.get("config"))
    raw = config.get("cross_run_action_history")
    if not isinstance(raw, list):
        return []
    limit = max(_safe_int(policy.get("cross_run_cooldown_runs"), 1), 0)
    history = [dict(item) for item in raw if isinstance(item, Mapping)]
    return history[-limit:] if limit > 0 else []


def _action_record(source: Mapping[str, Any]) -> Mapping[str, Any]:
    latest = _mapping(source.get("latest_action"))
    if latest:
        return latest
    actions = source.get("actions")
    if isinstance(actions, list):
        for item in reversed(actions):
            if isinstance(item, Mapping):
                return item
    return source


def _find_cross_run_cooldown(
    history: list[Mapping[str, Any]],
    *,
    action_id: str,
    action_kind: str,
    domain: str,
) -> Mapping[str, Any]:
    for item in reversed(history):
        record = _action_record(item)
        status = str(record.get("status") or item.get("status") or "")
        if status not in CROSS_RUN_COOLDOWN_STATUSES:
            continue
        record_action_id = str(record.get("action_id") or item.get("action_id") or "")
        record_kind = str(record.get("action_kind") or item.get("action_kind") or "")
        record_domain = str(record.get("domain") or item.get("domain") or "")
        id_matches = bool(action_id and record_action_id == action_id)
        kind_matches = bool(action_kind and record_kind == action_kind and (not domain or not record_domain or record_domain == domain))
        if id_matches or kind_matches:
            return item
    return {}


def _dataloader_config_overlay(action_plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay: dict[str, Any] = {}
    restore: dict[str, Any] = {}
    mutations = action_plan.get("mutations", [])
    if not isinstance(mutations, list):
        return overlay, restore
    for item in mutations:
        mutation = dict(item) if isinstance(item, Mapping) else {}
        if mutation.get("op") != "set":
            continue
        path = str(mutation.get("path") or "")
        if not path:
            continue
        overlay[path] = mutation.get("recommended")
        restore[path] = mutation.get("current")
    return overlay, restore


def _evaluate_active_action(
    *,
    snapshot: Mapping[str, Any],
    action_plan: Mapping[str, Any],
    policy: Mapping[str, Any],
    prior_state: Mapping[str, Any],
    current_step: int,
    mode: str,
) -> dict[str, Any]:
    active = _active_action(prior_state)
    history = _history(prior_state)
    cooldown_until = _safe_int(active.get("cooldown_until_step"), _safe_int(active.get("applied_step")))
    if current_step < cooldown_until:
        return {
            **_empty_executor(
                status="cooldown",
                reason=f"waiting until step {cooldown_until} before evaluating the applied action",
                mode=mode,
                action_plan=action_plan,
                policy=policy,
                prior_state=prior_state,
            ),
            "active_action": active,
        }

    before = _mapping(active.get("before_metrics"))
    after = _metrics(snapshot)
    before_sps = _safe_float(before.get("steady_samples_per_second"))
    after_sps = _safe_float(after.get("steady_samples_per_second"))
    gain_ratio = (after_sps / before_sps - 1.0) if before_sps > 0.0 and after_sps > 0.0 else 0.0
    quality = _evidence_quality(before, after, policy)
    evaluation = {
        "before": dict(before),
        "after": after,
        "evidence_quality": quality,
        "steady_samples_per_second_delta": _round(after_sps - before_sps),
        "steady_samples_per_second_gain_ratio": _round(gain_ratio),
        "steady_samples_per_second_gain_pct": _round(gain_ratio * 100.0, 4),
        "active_gpu_util_pct_delta": _metric_delta(before, after, "active_gpu_util_pct_mean", 4),
        "data_wait_share_delta": _metric_delta(before, after, "data_wait_share"),
        "h2d_transfer_share_delta": _metric_delta(before, after, "h2d_transfer_share"),
        "optimizer_share_delta": _metric_delta(before, after, "optimizer_share"),
        "host_gap_share_delta": _metric_delta(before, after, "host_gap_share"),
        "logging_checkpoint_share_delta": _metric_delta(before, after, "logging_checkpoint_share"),
        "final_loss_delta": _metric_delta(before, after, "final_loss", 6),
        "evaluated_step": current_step,
    }
    rollback_limit = _safe_float(policy.get("rollback_max_regression_ratio"), 0.02)
    min_gain = _safe_float(policy.get("min_throughput_gain"), 0.03)
    active_dataloader_rebuild = dict(_mapping(active.get("dataloader_rebuild")))
    if active_dataloader_rebuild:
        if before_sps <= 0.0 or after_sps <= 0.0:
            status = "needs_more_evidence"
            reason = "before/after throughput evidence is missing"
            blocked = ["missing_throughput_evidence"]
            safe_to_apply = False
        elif quality["blocked_reasons"]:
            status = "needs_more_evidence"
            reason = "before/after throughput windows are too short for closed-loop evaluation"
            blocked = list(quality["blocked_reasons"])
            safe_to_apply = False
        elif gain_ratio < -rollback_limit:
            status = "dataloader_rebuild_rollback_epoch_boundary_ready"
            reason = "throughput regressed beyond the rollback threshold; DataLoader rollback is ready for the next epoch boundary"
            blocked = []
            safe_to_apply = True
        elif gain_ratio >= min_gain:
            status = "keep_recommended"
            reason = "throughput gain met the configured threshold"
            blocked = []
            safe_to_apply = False
        else:
            status = "keep_observed"
            reason = "no rollback trigger; keep the DataLoader rebuild and continue observing"
            blocked = []
            safe_to_apply = False
        return {
            "schema_version": 1,
            "executor": "bubble_runtime_closed_loop_executor_v0",
            "status": status,
            "reason": reason,
            "mode": mode,
            "safe_to_auto_apply": safe_to_apply,
            "can_apply_during_current_run": safe_to_apply,
            "apply_boundary": "epoch_start",
            "candidate_action": str(active.get("action_kind") or action_plan.get("action_kind") or ""),
            "domain": str(active.get("domain") or action_plan.get("domain") or ""),
            "policy": dict(policy),
            "runtime_apply": {},
            "active_action": active,
            "evaluation": evaluation,
            "rollback": {
                "adapter_id": DATALOADER_REBUILD_RUNTIME_CONTRACT_ID,
                "restore": dict(_mapping(active.get("rollback_restore"))),
                "mutations": [],
                "current_run_reversible": True,
                "action_id": str(active.get("action_id") or ""),
                "dataloader_rebuild": active_dataloader_rebuild,
            },
            "action_history": history,
            "blocked_reasons": blocked,
        }
    rollback_plan = build_runtime_rollback_plan(_mapping(active.get("rollback_restore")))
    rollback_overlay = dict(_mapping(rollback_plan.get("restore")))
    rollback_mutations = list(rollback_plan.get("mutations") or [])

    if before_sps <= 0.0 or after_sps <= 0.0:
        status = "needs_more_evidence"
        reason = "before/after throughput evidence is missing"
        blocked = ["missing_throughput_evidence"]
    elif quality["blocked_reasons"]:
        status = "needs_more_evidence"
        reason = "before/after throughput windows are too short for closed-loop evaluation"
        blocked = list(quality["blocked_reasons"])
    elif gain_ratio < -rollback_limit:
        status = "rollback_recommended"
        reason = "throughput regressed beyond the rollback threshold"
        blocked = []
    elif gain_ratio >= min_gain:
        status = "keep_recommended"
        reason = "throughput gain met the configured threshold"
        blocked = []
    else:
        status = "keep_observed"
        reason = "no rollback trigger; keep the low-risk action and continue observing"
        blocked = []

    return {
        "schema_version": 1,
        "executor": "bubble_runtime_closed_loop_executor_v0",
        "status": status,
        "reason": reason,
        "mode": mode,
        "safe_to_auto_apply": status == "rollback_recommended" and bool(rollback_mutations),
        "can_apply_during_current_run": status == "rollback_recommended" and bool(rollback_mutations),
        "candidate_action": str(active.get("action_kind") or action_plan.get("action_kind") or ""),
        "domain": str(active.get("domain") or action_plan.get("domain") or ""),
        "policy": dict(policy),
        "runtime_apply": {},
        "active_action": active,
        "evaluation": evaluation,
        "rollback": {
            "adapter_id": str(rollback_plan.get("adapter_id") or ""),
            "restore": rollback_overlay,
            "mutations": rollback_mutations,
            "skipped_restore_mutations": list(rollback_plan.get("skipped_restore_mutations") or []),
            "current_run_reversible": bool(rollback_plan.get("current_run_reversible")),
            "action_id": str(active.get("action_id") or ""),
        },
        "action_history": history,
        "blocked_reasons": blocked,
    }


def build_closed_loop_executor_state(
    *,
    snapshot: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    action_plan: Mapping[str, Any],
    mode: Any,
    prior_state: Mapping[str, Any] | None = None,
    current_step: int | None = None,
) -> dict[str, Any]:
    resolved_mode = _normalized_mode(mode)
    prior = _mapping(prior_state)
    policy = _policy(snapshot, action_plan)
    step = _safe_int(current_step, 0)
    enabled = _safe_bool(_mapping(snapshot.get("config")).get("controller_enabled"), False)

    if _active_action(prior):
        return _evaluate_active_action(
            snapshot=snapshot,
            action_plan=action_plan,
            policy=policy,
            prior_state=prior,
            current_step=step,
            mode=resolved_mode,
        )
    if not enabled:
        return _empty_executor(
            status="disabled",
            reason="bubble_controller_enabled is false",
            mode=resolved_mode,
            action_plan=action_plan,
            policy=policy,
            prior_state=prior,
            blocked_reasons=["controller_disabled"],
        )
    if resolved_mode != "auto_apply":
        return _empty_executor(
            status="observe_only" if resolved_mode == "report_only" else "advisor_patch_only",
            reason=f"controller mode is {resolved_mode}",
            mode=resolved_mode,
            action_plan=action_plan,
            policy=policy,
            prior_state=prior,
            blocked_reasons=["mode_not_auto_apply"],
        )
    if step < _safe_int(policy.get("warmup_steps"), 0):
        return _empty_executor(
            status="warmup",
            reason=f"waiting for warmup_steps={policy.get('warmup_steps')}",
            mode=resolved_mode,
            action_plan=action_plan,
            policy=policy,
            prior_state=prior,
            blocked_reasons=["warmup"],
        )

    history = _history(prior)
    if _count_closed_actions(history) >= _safe_int(policy.get("max_actions_per_run"), 0):
        return _empty_executor(
            status="max_actions_reached",
            reason="max_actions_per_run has been reached",
            mode=resolved_mode,
            action_plan=action_plan,
            policy=policy,
            prior_state=prior,
            blocked_reasons=["max_actions_per_run"],
        )

    support = runtime_action_support(action_plan, runtime_context=_mapping(snapshot.get("runtime")))
    action_kind = str(support.get("action_kind") or action_plan.get("action_kind") or "")
    domain = str(support.get("domain") or action_plan.get("domain") or "")
    if support.get("adapter_id") == DATALOADER_REBUILD_RUNTIME_CONTRACT_ID and support.get("current_run_rebuild_ready"):
        config = _mapping(snapshot.get("config"))
        if not _safe_bool(config.get("allow_dataloader_rebuild_current_run"), False):
            blocked = list(support.get("blocked_reasons") or [])
            blocked.insert(0, "dataloader_rebuild_current_run_gate_disabled")
            return _empty_executor(
                status="blocked_dataloader_rebuild_current_run_gate_disabled",
                reason="DataLoader current-run rebuild requires an explicit experimental gate",
                mode=resolved_mode,
                action_plan=action_plan,
                policy=policy,
                prior_state=prior,
                blocked_reasons=blocked,
                runtime_adapter=support,
            )
        rebuild_plan = dict(_mapping(support.get("runtime_rebuild_plan")))
        before = _metrics(snapshot)
        if _safe_float(before.get("steady_samples_per_second")) <= 0.0:
            return _empty_executor(
                status="blocked_missing_baseline",
                reason="DataLoader current-run rebuild needs a baseline throughput window",
                mode=resolved_mode,
                action_plan=action_plan,
                policy=policy,
                prior_state=prior,
                blocked_reasons=["missing_baseline_throughput"],
                runtime_adapter=support,
            )
        if not list(rebuild_plan.get("applied_descriptor_mutations") or []):
            return _empty_executor(
                status="blocked_no_dataloader_rebuild_mutations",
                reason="DataLoader rebuild plan has no runtime descriptor mutations",
                mode=resolved_mode,
                action_plan=action_plan,
                policy=policy,
                prior_state=prior,
                blocked_reasons=["no_dataloader_rebuild_descriptor_mutations"],
                runtime_adapter=support,
            )
        action_id = stable_runtime_action_id(action_plan, list(action_plan.get("mutations") or []), adapter_id=DATALOADER_REBUILD_RUNTIME_CONTRACT_ID)
        previous_attempt = _find_attempted_action(history, action_id)
        if previous_attempt:
            return _empty_executor(
                status="action_already_attempted",
                reason="this exact runtime action was already attempted in the current run",
                mode=resolved_mode,
                action_plan=action_plan,
                policy=policy,
                prior_state=prior,
                blocked_reasons=["action_already_attempted"],
                runtime_adapter=support,
            )
        cross_run_cooldown = _find_cross_run_cooldown(
            _cross_run_history(snapshot, policy),
            action_id=action_id,
            action_kind=action_kind,
            domain=domain,
        )
        if cross_run_cooldown:
            return {
                **_empty_executor(
                    status="cross_run_action_cooldown",
                    reason="this action recently rolled back or lacked evidence in a previous run",
                    mode=resolved_mode,
                    action_plan=action_plan,
                    policy=policy,
                    prior_state=prior,
                    blocked_reasons=["cross_run_action_cooldown"],
                    runtime_adapter=support,
                ),
                "cross_run_cooldown": {
                    "action_id": action_id,
                    "action_kind": action_kind,
                    "domain": domain,
                    "matched_status": str(_action_record(cross_run_cooldown).get("status") or cross_run_cooldown.get("status") or ""),
                    "matched_action_id": str(
                        _action_record(cross_run_cooldown).get("action_id") or cross_run_cooldown.get("action_id") or ""
                    ),
                    "cooldown_runs": _safe_int(policy.get("cross_run_cooldown_runs"), 1),
                },
            }
        config_overlay, config_restore = _dataloader_config_overlay(action_plan)
        dataloader_rebuild = {
            "adapter_id": DATALOADER_REBUILD_RUNTIME_CONTRACT_ID,
            "action_id": action_id,
            "runtime_rebuild_plan": rebuild_plan,
        }
        runtime_apply = {
            "action_id": action_id,
            "status": "pending_epoch_boundary_apply",
            "step": step,
            "cooldown_until_step": step + _safe_int(policy.get("cooldown_steps"), 1),
            "adapter_id": DATALOADER_REBUILD_RUNTIME_CONTRACT_ID,
            "apply_boundary": "epoch_start",
            "mutations": [],
            "skipped_mutations": list(rebuild_plan.get("skipped_mutations") or []),
            "applied_overlay": config_overlay,
            "rollback_restore": config_restore,
            "before_metrics": before,
            "diagnosis_kind": str(diagnosis.get("kind") or ""),
            "rollback_adapter": {"adapter_id": DATALOADER_REBUILD_RUNTIME_CONTRACT_ID},
            "dataloader_rebuild": dataloader_rebuild,
        }
        return {
            "schema_version": 1,
            "executor": "bubble_runtime_closed_loop_executor_v0",
            "status": "dataloader_rebuild_epoch_boundary_ready",
            "reason": "DataLoader rebuild plan is ready for epoch-boundary current-run apply",
            "mode": resolved_mode,
            "safe_to_auto_apply": True,
            "can_apply_during_current_run": True,
            "apply_boundary": "epoch_start",
            "candidate_action": action_kind,
            "domain": domain,
            "policy": dict(policy),
            "runtime_apply": runtime_apply,
            "dataloader_rebuild": dataloader_rebuild,
            "active_action": {},
            "evaluation": {},
            "rollback": {},
            "runtime_adapter": dict(support),
            "action_history": history,
            "blocked_reasons": [],
        }
    if not support.get("supported"):
        return _empty_executor(
            status="blocked_action_not_runtime_safe",
            reason="only low-risk host scheduling actions can run in the current training run",
            mode=resolved_mode,
            action_plan=action_plan,
            policy=policy,
            prior_state=prior,
            blocked_reasons=list(support.get("blocked_reasons") or ["action_not_low_risk"]),
            runtime_adapter=support,
        )

    apply_candidate = build_runtime_apply_candidate(action_plan)
    mutations = list(apply_candidate.get("mutations") or [])
    if not mutations:
        return _empty_executor(
            status="blocked_no_runtime_mutations",
            reason="action plan has no low-risk current-run mutations",
            mode=resolved_mode,
            action_plan=action_plan,
            policy=policy,
            prior_state=prior,
            blocked_reasons=["no_low_risk_runtime_mutations"],
        )

    before = _metrics(snapshot)
    if _safe_float(before.get("steady_samples_per_second")) <= 0.0:
        return _empty_executor(
            status="blocked_missing_baseline",
            reason="current-run auto apply needs a baseline throughput window",
            mode=resolved_mode,
            action_plan=action_plan,
            policy=policy,
            prior_state=prior,
            blocked_reasons=["missing_baseline_throughput"],
        )

    action_id = str(apply_candidate.get("action_id") or "")
    previous_attempt = _find_attempted_action(history, action_id)
    if previous_attempt:
        return _empty_executor(
            status="action_already_attempted",
            reason="this exact runtime action was already attempted in the current run",
            mode=resolved_mode,
            action_plan=action_plan,
            policy=policy,
            prior_state=prior,
            blocked_reasons=["action_already_attempted"],
        )
    cross_run_cooldown = _find_cross_run_cooldown(
        _cross_run_history(snapshot, policy),
        action_id=action_id,
        action_kind=action_kind,
        domain=domain,
    )
    if cross_run_cooldown:
        return {
            **_empty_executor(
                status="cross_run_action_cooldown",
                reason="this action recently rolled back or lacked evidence in a previous run",
                mode=resolved_mode,
                action_plan=action_plan,
                policy=policy,
                prior_state=prior,
                blocked_reasons=["cross_run_action_cooldown"],
            ),
            "cross_run_cooldown": {
                "action_id": action_id,
                "action_kind": action_kind,
                "domain": domain,
                "matched_status": str(_action_record(cross_run_cooldown).get("status") or cross_run_cooldown.get("status") or ""),
                "matched_action_id": str(
                    _action_record(cross_run_cooldown).get("action_id") or cross_run_cooldown.get("action_id") or ""
                ),
                "cooldown_runs": _safe_int(policy.get("cross_run_cooldown_runs"), 1),
            },
        }
    runtime_apply = finalize_runtime_apply_candidate(
        apply_candidate,
        step=step,
        cooldown_steps=_safe_int(policy.get("cooldown_steps"), 1),
        before_metrics=before,
        diagnosis=diagnosis,
    )
    return {
        "schema_version": 1,
        "executor": "bubble_runtime_closed_loop_executor_v0",
        "status": "ready_to_apply",
        "reason": "low-risk host scheduling action is ready for one-step current-run apply",
        "mode": resolved_mode,
        "safe_to_auto_apply": True,
        "can_apply_during_current_run": True,
        "candidate_action": action_kind,
        "domain": domain,
        "policy": dict(policy),
        "runtime_apply": runtime_apply,
        "active_action": {},
        "evaluation": {},
        "rollback": {},
        "action_history": history,
        "blocked_reasons": [],
    }


def mark_closed_loop_action_applied(
    executor_state: Mapping[str, Any],
    *,
    current_step: int,
    applied_overlay: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_apply = _mapping(executor_state.get("runtime_apply"))
    action_id = str(runtime_apply.get("action_id") or "")
    history = [dict(item) for item in executor_state.get("action_history", []) if isinstance(item, Mapping)]
    active = {
        "schema_version": 1,
        "ledger": "bubble_runtime_closed_loop_action_v0",
        "action_id": action_id,
        "status": "applied",
        "applied_step": int(current_step),
        "cooldown_until_step": _safe_int(runtime_apply.get("cooldown_until_step"), int(current_step)),
        "adapter_id": str(runtime_apply.get("adapter_id") or ""),
        "apply_boundary": str(runtime_apply.get("apply_boundary") or "step"),
        "domain": str(executor_state.get("domain") or ""),
        "action_kind": str(executor_state.get("candidate_action") or ""),
        "applied_overlay": dict(applied_overlay),
        "rollback_restore": dict(_mapping(runtime_apply.get("rollback_restore"))),
        "rollback_adapter": dict(_mapping(runtime_apply.get("rollback_adapter"))),
        "dataloader_rebuild": dict(_mapping(runtime_apply.get("dataloader_rebuild"))),
        "profiler_handoff": dict(_mapping(runtime_apply.get("profiler_handoff"))),
        "before_metrics": dict(_mapping(runtime_apply.get("before_metrics"))),
        "skipped_mutations": list(runtime_apply.get("skipped_mutations") or []),
    }
    history.append(dict(active))
    return {
        "schema_version": 1,
        "executor": "bubble_runtime_closed_loop_executor_state_v0",
        "status": "cooldown",
        "active_action": active,
        "action_history": history,
    }


def mark_closed_loop_action_closed(
    executor_state: Mapping[str, Any],
    *,
    status: str,
    current_step: int,
    applied_overlay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active = dict(_mapping(executor_state.get("active_action")))
    action_id = str(active.get("action_id") or _mapping(executor_state.get("rollback")).get("action_id") or "")
    history = [dict(item) for item in executor_state.get("action_history", []) if isinstance(item, Mapping)]
    for item in reversed(history):
        if action_id and str(item.get("action_id") or "") == action_id:
            item["status"] = status
            item["closed_step"] = int(current_step)
            item["evaluation"] = dict(_mapping(executor_state.get("evaluation")))
            if applied_overlay is not None:
                item["rollback_applied_overlay"] = dict(applied_overlay)
            break
    else:
        history.append(
            {
                "schema_version": 1,
                "ledger": "bubble_runtime_closed_loop_action_v0",
                "action_id": action_id,
                "status": status,
                "closed_step": int(current_step),
                "evaluation": dict(_mapping(executor_state.get("evaluation"))),
                "rollback_applied_overlay": dict(applied_overlay or {}),
            }
        )
    return {
        "schema_version": 1,
        "executor": "bubble_runtime_closed_loop_executor_state_v0",
        "status": status,
        "active_action": {},
        "action_history": history,
    }


__all__ = [
    "LOW_RISK_RUNTIME_ACTIONS",
    "LOW_RISK_RUNTIME_PATHS",
    "CROSS_RUN_COOLDOWN_STATUSES",
    "build_closed_loop_executor_state",
    "mark_closed_loop_action_applied",
    "mark_closed_loop_action_closed",
]
