"""Host scheduling advisor patches for bubble-aware runtime actions."""

from __future__ import annotations

from typing import Any, Mapping


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


def _mutation(path: str, current: Any, recommended: Any, reason: str) -> dict[str, Any]:
    return {
        "op": "set",
        "path": path,
        "current": current,
        "recommended": recommended,
        "reason": reason,
    }


def _append_mutation(
    mutations: list[dict[str, Any]],
    *,
    path: str,
    current: Any,
    recommended: Any,
    reason: str,
) -> None:
    if current == recommended:
        return
    mutations.append(_mutation(path, current, recommended, reason))


def _config_overlay(mutations: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {str(item["path"]): item.get("recommended") for item in mutations if item.get("op") == "set" and item.get("path")}


def _empty_plan(
    *,
    mode: str,
    status: str,
    action_kind: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "plan": "bubble_runtime_action_plan_v0",
        "phase": "P1_report_only",
        "mode": mode,
        "domain": "host_scheduling",
        "status": status,
        "action_kind": action_kind,
        "apply_mode": "report_only",
        "can_apply_to_next_request": False,
        "can_apply_during_current_run": False,
        "mutations": [],
        "config_overlay": {},
        "effective_policy": {},
        "rollback": {},
        "notes": [reason],
    }


def _append_profiler_mutations(mutations: list[dict[str, Any]], runtime: Mapping[str, Any], reason: str) -> None:
    if _safe_bool(runtime.get("step_phase_profile_enabled"), False):
        _append_mutation(
            mutations,
            path="step_phase_profile_enabled",
            current=True,
            recommended=False,
            reason=reason,
        )
    transfer_mode = str(runtime.get("data_transfer_profile_mode", "event") or "event").strip().lower()
    if transfer_mode == "sync":
        _append_mutation(
            mutations,
            path="data_transfer_profile_mode",
            current=transfer_mode,
            recommended="event",
            reason="replace sync transfer profiling with CUDA-event profiling outside benchmark probes",
        )


def _append_logging_mutations(mutations: list[dict[str, Any]], runtime: Mapping[str, Any]) -> None:
    if not _safe_bool(runtime.get("adaptive_step_logging_enabled"), True):
        _append_mutation(
            mutations,
            path="adaptive_step_logging_enabled",
            current=False,
            recommended=True,
            reason="let the trainer automatically back off hot step logging",
        )
    flush_interval = max(_safe_int(runtime.get("tensorboard_flush_interval_steps"), 10), 1)
    _append_mutation(
        mutations,
        path="tensorboard_flush_interval_steps",
        current=flush_interval,
        recommended=max(flush_interval, 100),
        reason="reduce TensorBoard flush frequency on the hot path",
    )
    if _safe_bool(runtime.get("layer_monitor_enabled"), True):
        layer_interval = max(_safe_int(runtime.get("layer_monitor_interval"), 3), 1)
        _append_mutation(
            mutations,
            path="layer_monitor_interval",
            current=layer_interval,
            recommended=max(layer_interval * 2, 8),
            reason="sample layer monitor less often during throughput-sensitive runs",
        )


def _append_checkpoint_mutations(mutations: list[dict[str, Any]], runtime: Mapping[str, Any]) -> None:
    step_interval = max(_safe_int(runtime.get("save_every_n_steps"), 0), 0)
    epoch_interval = max(_safe_int(runtime.get("save_every_n_epochs"), 1), 0)
    if step_interval > 0:
        _append_mutation(
            mutations,
            path="save_every_n_steps",
            current=step_interval,
            recommended=max(step_interval * 2, 250),
            reason="make step checkpoint saves less frequent when they appear in the hot path",
        )
    elif epoch_interval > 0:
        _append_mutation(
            mutations,
            path="save_every_n_epochs",
            current=epoch_interval,
            recommended=max(epoch_interval + 1, 2),
            reason="make epoch checkpoint saves less frequent for long throughput probes",
        )


def _append_validation_mutations(mutations: list[dict[str, Any]], runtime: Mapping[str, Any]) -> None:
    step_eval = max(_safe_int(runtime.get("eval_every_n_steps"), 0), 0)
    if step_eval > 0:
        _append_mutation(
            mutations,
            path="eval_every_n_steps",
            current=step_eval,
            recommended=0,
            reason="move validation out of the step hot path and keep epoch/end-window validation",
        )


def build_host_scheduling_plan(
    *,
    snapshot: Mapping[str, Any],
    action: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    config = _mapping(snapshot.get("config"))
    runtime = _mapping(snapshot.get("runtime"))
    step = _mapping(snapshot.get("step_phase"))
    action_kind = str(action.get("kind") or "")
    controller_enabled = _safe_bool(config.get("controller_enabled"), False)
    allow_checkpoint_async = _safe_bool(config.get("allow_checkpoint_async"), True)

    if mode == "report_only":
        return _empty_plan(
            mode=mode,
            status="report_only",
            action_kind=action_kind,
            reason="controller mode is report_only; host scheduling patch is not materialized",
        )
    if not controller_enabled:
        return _empty_plan(
            mode=mode,
            status="disabled",
            action_kind=action_kind,
            reason="bubble_controller_enabled is false; emitting diagnosis only",
        )
    if action_kind in {"increase_checkpoint_interval", "enable_async_checkpoint_save"} and not allow_checkpoint_async:
        return _empty_plan(
            mode=mode,
            status="blocked_by_config",
            action_kind=action_kind,
            reason="bubble_controller_allow_checkpoint_async is false",
        )
    if action_kind == "enable_async_checkpoint_save":
        return _empty_plan(
            mode=mode,
            status="manual_followup",
            action_kind=action_kind,
            reason="async checkpoint queue needs runtime integration before a config patch can be emitted",
        )

    mutations: list[dict[str, Any]] = []
    if action_kind == "disable_sync_profiler_mode":
        _append_profiler_mutations(
            mutations,
            runtime,
            str(action.get("reason") or "disable synchronized profiler modes outside benchmark probes"),
        )
    elif action_kind == "increase_logging_interval":
        _append_logging_mutations(mutations, runtime)
    elif action_kind == "increase_checkpoint_interval":
        _append_checkpoint_mutations(mutations, runtime)
    elif action_kind == "move_validation_after_training_window":
        _append_validation_mutations(mutations, runtime)
    elif action_kind == "reduce_hot_path_sync":
        _append_profiler_mutations(
            mutations,
            runtime,
            "remove profiler synchronization from the normal training hot path",
        )
        _append_logging_mutations(mutations, runtime)
        _append_validation_mutations(mutations, runtime)
        if allow_checkpoint_async:
            _append_checkpoint_mutations(mutations, runtime)
    else:
        return _empty_plan(
            mode=mode,
            status="unsupported_phase",
            action_kind=action_kind,
            reason="this host scheduling action is not implemented",
        )

    if not mutations:
        return _empty_plan(
            mode=mode,
            status="no_patch",
            action_kind=action_kind,
            reason="recommended host scheduling settings already match the current snapshot",
        )

    status = "advisor_patch_ready" if mode == "advisor_patch" else "auto_apply_blocked_pending_p7"
    overlay = _config_overlay(mutations)
    effective_policy = {
        "step_phase_profile_enabled": overlay.get(
            "step_phase_profile_enabled",
            _safe_bool(runtime.get("step_phase_profile_enabled"), False),
        ),
        "data_transfer_profile_mode": overlay.get(
            "data_transfer_profile_mode",
            str(runtime.get("data_transfer_profile_mode", "event") or "event"),
        ),
        "tensorboard_flush_interval_steps": overlay.get(
            "tensorboard_flush_interval_steps",
            _safe_int(runtime.get("tensorboard_flush_interval_steps"), 10),
        ),
        "layer_monitor_interval": overlay.get(
            "layer_monitor_interval",
            _safe_int(runtime.get("layer_monitor_interval"), 3),
        ),
        "save_every_n_steps": overlay.get(
            "save_every_n_steps",
            _safe_int(runtime.get("save_every_n_steps"), 0),
        ),
        "save_every_n_epochs": overlay.get(
            "save_every_n_epochs",
            _safe_int(runtime.get("save_every_n_epochs"), 1),
        ),
        "eval_every_n_steps": overlay.get(
            "eval_every_n_steps",
            _safe_int(runtime.get("eval_every_n_steps"), 0),
        ),
    }
    return {
        "schema_version": 1,
        "plan": "bubble_runtime_action_plan_v0",
        "phase": "P6_host_scheduling_advisor_patch",
        "mode": mode,
        "domain": "host_scheduling",
        "status": status,
        "action_kind": action_kind,
        "apply_mode": "advisor_patch",
        "can_apply_to_next_request": mode == "advisor_patch",
        "can_apply_during_current_run": False,
        "mutations": mutations,
        "config_overlay": overlay,
        "effective_policy": effective_policy,
        "hot_path_evidence": {
            "host_gap_share": step.get("host_gap_share"),
            "logging_checkpoint_share": step.get("logging_checkpoint_share"),
            "host_phase_share": step.get("host_phase_share", {}),
            "top_phases": step.get("top_phases", []),
        },
        "rollback": {
            "metric": "steady_samples_per_second",
            "secondary_metric": "host_gap_share",
            "max_regression_ratio": 0.03,
            "compare_window": "post_warmup_steady_window",
            "restore": {item["path"]: item.get("current") for item in mutations},
        },
        "safety": {
            "current_run_hot_mutation": False,
            "checkpoint_async_runtime_implemented": False,
            "checkpoint_mutations_require_gate": True,
            "allow_checkpoint_async": allow_checkpoint_async,
        },
        "notes": [
            "P6 only materializes host/logging/checkpoint advisor patches for the next request.",
            "Async checkpoint save is intentionally not enabled until the runtime queue/flush path exists.",
            "The action plan itself does not mutate the current run; P7 auto_apply may consume an allowlisted host-scheduling subset.",
        ],
    }


__all__ = ["build_host_scheduling_plan"]
