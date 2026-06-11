"""Bubble-aware runtime controller reports and safe action plans."""

from __future__ import annotations

from typing import Any, Mapping

from .bubble_runtime_actions import build_bubble_action_plan
from .bubble_runtime_closed_loop import build_closed_loop_state
from .bubble_runtime_policy import classify_bubble
from .bubble_runtime_snapshot import build_bubble_runtime_snapshot


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _cfg(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


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


def _latest_profiler_handoff(closed_loop_state: Mapping[str, Any] | None) -> Mapping[str, Any]:
    state = _mapping(closed_loop_state)
    history = state.get("action_history")
    if not isinstance(history, list):
        return {}
    for item in reversed(history):
        action = _mapping(item)
        if str(action.get("status") or "") != "kept":
            continue
        handoff = _mapping(action.get("profiler_handoff"))
        if str(handoff.get("kind") or "") == "data_wait_after_sync_profiler_disable_v0":
            return handoff
    return {}


def _latest_sync_profiler_action(closed_loop_state: Mapping[str, Any] | None) -> Mapping[str, Any]:
    state = _mapping(closed_loop_state)
    history = state.get("action_history")
    if not isinstance(history, list):
        return {}
    for item in reversed(history):
        action = _mapping(item)
        if str(action.get("action_kind") or "") == "disable_sync_profiler_mode":
            return action
    return {}


def _apply_sync_profiler_retry_after_rollback(
    snapshot: Mapping[str, Any],
    closed_loop_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    action = _latest_sync_profiler_action(closed_loop_state)
    if str(action.get("status") or "") != "rolled_back":
        return dict(snapshot)
    snap = dict(snapshot)
    step_phase = _mapping(snap.get("step_phase"))
    data_wait = _safe_float(step_phase.get("data_wait_share"))
    if data_wait < 0.075:
        return snap
    runtime = dict(_mapping(snap.get("runtime")))
    runtime["sync_profiler_data_wait_retry"] = True
    runtime["sync_profiler_data_wait_retry_source_status"] = "rolled_back"
    snap["runtime"] = runtime
    return snap


def _apply_profiler_handoff(
    snapshot: Mapping[str, Any],
    closed_loop_state: Mapping[str, Any] | None,
    *,
    current_step: int | None,
) -> dict[str, Any]:
    handoff = _latest_profiler_handoff(closed_loop_state)
    data_wait = _safe_float(handoff.get("data_wait_share"))
    if data_wait < 0.08:
        return dict(snapshot)
    snap = dict(snapshot)
    runtime = dict(_mapping(snap.get("runtime")))
    if _safe_bool(runtime.get("step_phase_profile_enabled"), False):
        return snap
    if str(runtime.get("data_transfer_profile_mode") or "event").strip().lower() == "sync":
        return snap
    observed_step = _safe_int(handoff.get("observed_step"), -1)
    step = _safe_int(current_step, observed_step)
    max_age = max(_safe_int(_mapping(snap.get("config")).get("tune_interval_steps"), 32) * 4, 32)
    if observed_step >= 0 and step >= observed_step and step - observed_step > max_age:
        return snap

    step_phase = dict(_mapping(snap.get("step_phase")))
    if _safe_float(step_phase.get("data_wait_share")) < data_wait:
        step_phase.update(
            {
                "available": True,
                "dominant_bottleneck": "data_bound",
                "data_wait_share": round(data_wait, 6),
                "h2d_transfer_share": round(_safe_float(handoff.get("h2d_transfer_share")), 6),
                "optimizer_share": round(_safe_float(handoff.get("optimizer_share")), 6),
                "host_gap_share": round(_safe_float(handoff.get("host_gap_share")), 6),
                "logging_checkpoint_share": round(_safe_float(handoff.get("logging_checkpoint_share")), 6),
                "profiler_handoff": dict(handoff),
            }
        )
        snap["step_phase"] = step_phase
    return snap


def build_bubble_controller_report(
    config: Any = None,
    runtime_features: Mapping[str, Any] | None = None,
    closed_loop_state: Mapping[str, Any] | None = None,
    current_step: int | None = None,
) -> dict[str, Any]:
    snapshot = build_bubble_runtime_snapshot(config=config, runtime_features=runtime_features)
    snapshot = _apply_profiler_handoff(
        snapshot,
        closed_loop_state,
        current_step=current_step,
    )
    snapshot = _apply_sync_profiler_retry_after_rollback(snapshot, closed_loop_state)
    diagnosis = classify_bubble(snapshot)
    mode = str(_cfg(config, "bubble_controller_mode", "report_only") or "report_only")
    enabled = _safe_bool(_cfg(config, "bubble_controller_enabled", False), False)
    action_plan = build_bubble_action_plan(snapshot, diagnosis, mode=mode)
    closed_loop = build_closed_loop_state(
        snapshot=snapshot,
        diagnosis=diagnosis,
        action_plan=action_plan,
        mode=mode,
        prior_state=closed_loop_state,
        current_step=current_step,
    )
    if action_plan.get("apply_mode") != "report_only":
        diagnosis = dict(diagnosis)
        action = dict(diagnosis.get("recommended_action") or {})
        action["apply_mode"] = action_plan["apply_mode"]
        diagnosis["recommended_action"] = action
    phase = "P1_report_only"
    if action_plan.get("status") in {"advisor_patch_ready", "auto_apply_blocked_pending_p7"}:
        phase = str(action_plan.get("phase") or phase)
    if action_plan.get("mode") == "report_only":
        status = "report_only"
    elif not enabled:
        status = "disabled"
    else:
        status = str(action_plan.get("status") or "advisory_only")
    return {
        "schema_version": 1,
        "controller": "bubble_aware_runtime_controller_v0",
        "phase": phase,
        "enabled": enabled,
        "mode": mode,
        "status": status,
        "safe_to_auto_apply": bool(closed_loop.get("safe_to_auto_apply", False)),
        "snapshot": snapshot,
        "diagnosis": diagnosis,
        "action_plan": action_plan,
        "closed_loop": closed_loop,
        "action_history": list(closed_loop.get("action_history", [])),
        "notes": [
            "P1-P6 do not mutate the current trainer config.",
            "P7 auto-apply is limited to low-risk host-scheduling mutations with cooldown and rollback evidence.",
        ],
    }


def observe_bubble_runtime(config: Any = None, runtime_features: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return build_bubble_controller_report(config=config, runtime_features=runtime_features)


__all__ = ["build_bubble_controller_report", "observe_bubble_runtime"]
