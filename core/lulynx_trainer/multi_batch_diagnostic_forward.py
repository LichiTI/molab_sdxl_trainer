"""Diagnostic microbatch forward helpers for Lulynx multi-batch isolation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


LULYNX_MULTI_BATCH_DIAGNOSTIC_FORWARD = "lulynx_multi_batch_diagnostic_forward_v0"


@dataclass(frozen=True)
class LulynxDiagnosticForwardResult:
    output: Any
    report: dict[str, Any]


def run_lulynx_diagnostic_microbatch_forward(
    *,
    unet: Any,
    unet_kwargs: Mapping[str, Any],
    microbatch_size: int = 1,
) -> LulynxDiagnosticForwardResult:
    """Run UNet forward in diagnostic microbatches and concatenate samples."""

    kwargs = dict(unet_kwargs or {})
    batch_size = _leading_dim(kwargs.get("sample"))
    step = max(int(microbatch_size or 1), 1)
    if batch_size <= 0:
        raise ValueError("diagnostic_microbatch_requires_sample_batch_dim")
    outputs: list[Any] = []
    for start in range(0, batch_size, step):
        end = min(start + step, batch_size)
        sliced = {key: _slice_value(value, start=start, end=end, batch_size=batch_size) for key, value in kwargs.items()}
        result = unet(**sliced)
        outputs.append(result.sample if hasattr(result, "sample") else result)
    output = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
    return LulynxDiagnosticForwardResult(
        output=output,
        report={
            "schema_version": 1,
            "report": LULYNX_MULTI_BATCH_DIAGNOSTIC_FORWARD,
            "release_claim_allowed": False,
            "diagnostic_only": True,
            "batch_size": batch_size,
            "microbatch_size": step,
            "microbatch_count": len(outputs),
            "output_batch_size": _leading_dim(output),
        },
    )


def _slice_value(value: Any, *, start: int, end: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and int(value.shape[0]) == batch_size:
            return value[start:end]
        return value
    if isinstance(value, Mapping):
        return {
            key: _slice_value(item, start=start, end=end, batch_size=batch_size)
            for key, item in value.items()
        }
    return value


def _leading_dim(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if not shape:
        return 0
    try:
        return max(int(shape[0]), 0)
    except (TypeError, ValueError, IndexError):
        return 0


__all__ = [
    "LULYNX_MULTI_BATCH_DIAGNOSTIC_FORWARD",
    "LulynxDiagnosticForwardResult",
    "run_lulynx_diagnostic_microbatch_forward",
]
