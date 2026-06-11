"""Plugin hook execution helpers for Lulynx staged train steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxLossPluginHookStageExecution:
    loss: Any
    raw_loss: Any
    raw_loss_value: float
    loss_tracker_value: float | None
    loss_scale: float
    emit_after_backward_event: Callable[..., Any] | None
    after_loss_event_emitted: bool
    modify_loss_event_applied: bool
    plugin_mutation_applied: bool
    mutation_report: Any
    orchestrator_runtime: dict[str, Any]


def run_lulynx_loss_plugin_hook_stage_handler(
    *,
    hook_context: Mapping[str, Any],
    loss: Any,
    loss_tracker_value: float | None,
    accumulation_steps: int,
    loss_scalars: Any,
    loss_tracker: Any,
    mutation_from_report: Callable[[dict[str, Any] | None], tuple[float, float]],
    hook_api: Mapping[str, Any] | None = None,
) -> LulynxLossPluginHookStageExecution:
    """Run loss-stage plugin hooks while preserving the existing semantics."""

    hooks = dict(hook_api) if isinstance(hook_api, Mapping) else _load_training_hook_api()
    emit_after_loss_event = hooks.get("emit_after_loss_event")
    apply_modify_loss_event = hooks.get("apply_modify_loss_event")
    emit_after_backward_event = hooks.get("emit_after_backward_event")
    raw_loss = loss
    raw_loss_value = loss_scalars.get(raw_loss)
    loss_scale = 1.0 / float(max(int(accumulation_steps or 1), 1))
    after_loss_event_emitted = False
    modify_loss_event_applied = False
    plugin_mutation_applied = False
    mutation_report: Any = None

    if callable(emit_after_loss_event):
        emit_after_loss_event(
            **dict(hook_context),
            loss_value=raw_loss_value,
            loss_scale=loss_scale,
            weighted_loss=raw_loss_value,
        )
        after_loss_event_emitted = True

    if callable(apply_modify_loss_event):
        _, mutation_report = apply_modify_loss_event(
            **dict(hook_context),
            loss_value=raw_loss_value,
            loss_scale=loss_scale,
        )
        modify_loss_event_applied = True
        scale, bias = mutation_from_report(mutation_report if isinstance(mutation_report, dict) else None)
        if scale != 1.0 or bias != 0.0:
            loss = loss * loss.new_tensor(scale)
            if bias != 0.0:
                loss = loss + loss.new_tensor(bias)
            raw_loss = loss
            raw_loss_value = loss_scalars.get(raw_loss)
            plugin_mutation_applied = True
            if loss_tracker:
                loss_tracker.record(
                    "plugin_mutation",
                    loss_tracker_value,
                    raw_loss_value,
                    scale=scale,
                    bias=bias,
                )
                loss_tracker_value = raw_loss_value

    return LulynxLossPluginHookStageExecution(
        loss=loss,
        raw_loss=raw_loss,
        raw_loss_value=raw_loss_value,
        loss_tracker_value=loss_tracker_value,
        loss_scale=loss_scale,
        emit_after_backward_event=emit_after_backward_event if callable(emit_after_backward_event) else None,
        after_loss_event_emitted=after_loss_event_emitted,
        modify_loss_event_applied=modify_loss_event_applied,
        plugin_mutation_applied=plugin_mutation_applied,
        mutation_report=mutation_report,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "host_to_device", "conditioning", "noise_timestep", "forward", "loss"),
            status="loss_plugin_hook_stage_handler_executed",
            handler_source="existing_training_loop_loss_plugin_hook_path",
            extra={
                "after_loss_event_emitted": after_loss_event_emitted,
                "modify_loss_event_applied": modify_loss_event_applied,
                "plugin_mutation_applied": plugin_mutation_applied,
            },
        ),
    )


def _load_training_hook_api() -> dict[str, Any]:
    try:
        from core.services.training_hooks import (
            apply_modify_loss_event,
            emit_after_backward_event,
            emit_after_loss_event,
        )
    except Exception:  # pragma: no cover - optional launcher/plugin surface
        return {}
    return {
        "apply_modify_loss_event": apply_modify_loss_event,
        "emit_after_backward_event": emit_after_backward_event,
        "emit_after_loss_event": emit_after_loss_event,
    }


__all__ = [
    "LulynxLossPluginHookStageExecution",
    "run_lulynx_loss_plugin_hook_stage_handler",
]
