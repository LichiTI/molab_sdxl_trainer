"""Layer monitor handler for Lulynx optimizer stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxLayerMonitorStageExecution:
    layer_monitor_info: dict[str, Any] | None
    collection_attempted: bool
    collection_error: str
    orchestrator_runtime: dict[str, Any]


def run_lulynx_layer_monitor_stage_handler(
    *,
    enabled: bool,
    global_step: int,
    interval: int,
    lora_injector: Any,
    optimizer: Any,
    max_layers: int,
    sparsity_epsilon: float,
    mode: str,
    sample_size: int,
    logger: Any,
) -> LulynxLayerMonitorStageExecution:
    """Collect optional layer monitor stats while preserving nonfatal fallback."""

    info = None
    attempted = False
    error = ""
    if bool(enabled):
        try:
            from .layer_monitor import collect_lora_layer_stats, should_collect_layer_monitor

            if should_collect_layer_monitor(
                enabled=True,
                step=int(global_step) + 1,
                interval=interval,
            ):
                attempted = True
                snapshot = collect_lora_layer_stats(
                    lora_injector,
                    optimizer,
                    max_layers=max_layers,
                    sparsity_epsilon=sparsity_epsilon,
                    mode=mode,
                    sample_size=sample_size,
                )
                info = {
                    "layers": snapshot.layers,
                    "elapsed_seconds": snapshot.elapsed_seconds,
                    "sampled_layers": snapshot.sampled_layers,
                    "total_layers": snapshot.total_layers,
                    "interval": interval,
                    "mode": snapshot.mode,
                    "sample_size": snapshot.sample_size,
                }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            debug = getattr(logger, "debug", None)
            if callable(debug):
                debug("layer monitor collection skipped: %s", exc)
    return LulynxLayerMonitorStageExecution(
        layer_monitor_info=info,
        collection_attempted=attempted,
        collection_error=error,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "forward", "loss", "backward", "optimizer_step"),
            status="layer_monitor_stage_handler_executed",
            handler_source="existing_training_loop_layer_monitor_path",
            extra={
                "enabled": bool(enabled),
                "collection_attempted": attempted,
                "has_layer_monitor_info": info is not None,
                "collection_error": error,
            },
        ),
    )


__all__ = [
    "LulynxLayerMonitorStageExecution",
    "run_lulynx_layer_monitor_stage_handler",
]
