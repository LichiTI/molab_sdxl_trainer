"""TurboCore shadow execution handlers for Lulynx optimizer stages."""

from __future__ import annotations

import time
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxTurboCoreShadowPrepareStageExecution:
    shadow_report: dict[str, Any]
    shadow_prepare_executed: bool
    orchestrator_runtime: dict[str, Any]


@dataclass(frozen=True)
class LulynxTurboCoreShadowCompareStageExecution:
    shadow_report: dict[str, Any]
    shadow_compare_executed: bool
    orchestrator_runtime: dict[str, Any]


def run_lulynx_turbocore_shadow_prepare_stage_handler(
    *,
    shadow: Any,
    trainable_params: Sequence[Any],
    optimizer: Any,
    max_grad_norm: float,
    step: int,
    native_update_loop_timing: MutableMapping[str, Any],
    logger: Any,
) -> LulynxTurboCoreShadowPrepareStageExecution:
    """Run TurboCore shadow prepare before optimizer.step with old fallback semantics."""

    if not bool(getattr(shadow, "enabled", False)):
        return LulynxTurboCoreShadowPrepareStageExecution(
            shadow_report={},
            shadow_prepare_executed=False,
            orchestrator_runtime=_runtime(
                "turbocore_shadow_prepare_stage_handler_noop",
                shadow_prepare_executed=False,
                shadow_report_present=False,
            ),
        )
    started = time.perf_counter()
    try:
        report = shadow.prepare_before_optimizer(
            list(trainable_params),
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            step=step,
        )
        shadow_report = dict(report or {})
    except Exception as exc:
        shadow_report = {
            "schema_version": 1,
            "mode": _shadow_mode(shadow),
            "stage": "before_optimizer",
            "error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
        }
        _debug(logger, "TurboCore update shadow prepare skipped: %s", exc)
    native_update_loop_timing["shadow_prepare_ms"] = _elapsed_ms(started)
    return LulynxTurboCoreShadowPrepareStageExecution(
        shadow_report=shadow_report,
        shadow_prepare_executed=True,
        orchestrator_runtime=_runtime(
            "turbocore_shadow_prepare_stage_handler_executed",
            shadow_prepare_executed=True,
            shadow_report_present=bool(shadow_report),
        ),
    )


def run_lulynx_turbocore_shadow_compare_stage_handler(
    *,
    shadow: Any,
    shadow_report: Any,
    step: int,
    native_update_loop_timing: MutableMapping[str, Any],
    logger: Any,
) -> LulynxTurboCoreShadowCompareStageExecution:
    """Run TurboCore shadow compare after optimizer.step with old fallback semantics."""

    report = dict(shadow_report) if isinstance(shadow_report, dict) else {}
    if not bool(getattr(shadow, "enabled", False)) or not report:
        return LulynxTurboCoreShadowCompareStageExecution(
            shadow_report=report,
            shadow_compare_executed=False,
            orchestrator_runtime=_runtime(
                "turbocore_shadow_compare_stage_handler_noop",
                shadow_compare_executed=False,
                shadow_report_present=bool(report),
            ),
        )
    started = time.perf_counter()
    try:
        report["after_optimizer"] = shadow.compare_after_optimizer(step=step)
    except Exception as exc:
        report["after_optimizer"] = {
            "schema_version": 1,
            "mode": _shadow_mode(shadow),
            "stage": "after_optimizer",
            "error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
        }
        _debug(logger, "TurboCore update shadow compare skipped: %s", exc)
    native_update_loop_timing["shadow_compare_ms"] = _elapsed_ms(started)
    return LulynxTurboCoreShadowCompareStageExecution(
        shadow_report=report,
        shadow_compare_executed=True,
        orchestrator_runtime=_runtime(
            "turbocore_shadow_compare_stage_handler_executed",
            shadow_compare_executed=True,
            shadow_report_present=bool(report),
        ),
    )


def _runtime(status: str, **extra: Any) -> dict[str, Any]:
    return build_lulynx_stage_orchestrator_runtime(
        executed_stage_ids=(
            "batch_contract",
            "host_to_device",
            "conditioning",
            "noise_timestep",
            "forward",
            "loss",
            "backward",
            "optimizer_step",
        ),
        status=status,
        handler_source="existing_training_loop_turbocore_shadow_path",
        extra=extra,
    )


def _shadow_mode(shadow: Any) -> str:
    return str(getattr(getattr(shadow, "config", None), "mode", "off") or "off")


def _debug(logger: Any, message: str, *args: Any) -> None:
    debug = getattr(logger, "debug", None)
    if callable(debug):
        debug(message, *args)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 4)


__all__ = [
    "LulynxTurboCoreShadowCompareStageExecution",
    "LulynxTurboCoreShadowPrepareStageExecution",
    "run_lulynx_turbocore_shadow_compare_stage_handler",
    "run_lulynx_turbocore_shadow_prepare_stage_handler",
]
