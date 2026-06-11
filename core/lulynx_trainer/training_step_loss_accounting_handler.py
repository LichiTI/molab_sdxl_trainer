"""Loss accounting handler for Lulynx train epochs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxLossAccountingStageExecution:
    loss: float
    total_loss: float
    num_steps: int
    pending_loss_total: Any
    pending_loss_count: int
    pending_filenames: list[str]
    last_loss: float | None
    last_loss_updated: bool
    orchestrator_runtime: dict[str, Any]


def run_lulynx_loss_accounting_stage_handler(
    *,
    loss_result: Any,
    fast_accumulation: bool,
    is_accumulation_boundary: bool,
    batch_filenames: Any,
    pending_loss_total: Any,
    pending_loss_count: int,
    pending_filenames: Sequence[str] | None,
    total_loss: float,
    num_steps: int,
    last_loss: Any,
) -> LulynxLossAccountingStageExecution:
    """Update epoch loss accounting without changing train-step math."""

    filenames = [str(item) for item in (pending_filenames or [])]
    next_pending_total = pending_loss_total
    next_pending_count = int(pending_loss_count or 0)
    next_total_loss = float(total_loss or 0.0)
    next_num_steps = int(num_steps or 0)
    next_last_loss = None
    last_loss_updated = False

    if bool(fast_accumulation) and isinstance(loss_result, torch.Tensor):
        detached_loss = loss_result.detach().float()
        next_pending_total = (
            detached_loss if next_pending_total is None else next_pending_total + detached_loss
        )
        next_pending_count += 1
        filenames.extend(_normalize_filenames(batch_filenames))
        if bool(is_accumulation_boundary):
            if next_pending_total is not None and next_pending_count > 0:
                loss = float((next_pending_total / next_pending_count).item())
                next_total_loss += loss * next_pending_count
                next_num_steps += next_pending_count
            else:
                loss = _float_last_loss(last_loss)
            next_pending_total = None
            next_pending_count = 0
        else:
            loss = _float_last_loss(last_loss)
    else:
        loss = float(loss_result)
        next_last_loss = loss
        last_loss_updated = True
        next_total_loss += loss
        next_num_steps += 1

    return LulynxLossAccountingStageExecution(
        loss=loss,
        total_loss=next_total_loss,
        num_steps=next_num_steps,
        pending_loss_total=next_pending_total,
        pending_loss_count=next_pending_count,
        pending_filenames=filenames,
        last_loss=next_last_loss,
        last_loss_updated=last_loss_updated,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "forward", "loss"),
            status="loss_accounting_stage_handler_executed",
            handler_source="existing_training_loop_loss_accounting_path",
            extra={
                "fast_accumulation": bool(fast_accumulation),
                "is_accumulation_boundary": bool(is_accumulation_boundary),
                "pending_loss_count": next_pending_count,
                "num_steps": next_num_steps,
            },
        ),
    )


def _normalize_filenames(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(name) for name in value]
    if value:
        return [str(value)]
    return []


def _float_last_loss(value: Any) -> float:
    return value if isinstance(value, float) else 0.0


__all__ = [
    "LulynxLossAccountingStageExecution",
    "run_lulynx_loss_accounting_stage_handler",
]
