"""Batch-stage planning for the Lulynx train-step pipeline.

This module only describes the incoming batch. It does not move tensors and it
does not call model code. The goal is to keep the earliest train-step decisions
auditable before native batch 2/4/8 promotion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


LULYNX_TRAINING_STEP_BATCH_STAGE_PLAN = "lulynx_training_step_batch_stage_plan_v0"

_CACHED_NATIVE_ARCHES = {"sdxl", "sd15", "anima", "newbie"}


@dataclass(frozen=True)
class LulynxTrainingStepBatchStagePlan:
    model_arch: str
    expected_physical_batch_size: int
    cached_native: bool
    host_to_device_route: str
    missing_required_fields: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_required_fields

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_TRAINING_STEP_BATCH_STAGE_PLAN,
            "ok": self.ok,
            "model_arch": self.model_arch,
            "expected_physical_batch_size": self.expected_physical_batch_size,
            "cached_native": self.cached_native,
            "host_to_device_route": self.host_to_device_route,
            "missing_required_fields": list(self.missing_required_fields),
        }


def infer_lulynx_batch_leading_dim(batch: Mapping[str, Any]) -> int:
    if not isinstance(batch, Mapping):
        return 0
    for key in ("latents", "images"):
        value = batch.get(key)
        shape = getattr(value, "shape", None)
        if shape:
            try:
                return max(int(shape[0]), 0)
            except (TypeError, ValueError, IndexError):
                return 0
    return 0


def build_lulynx_training_step_batch_stage_plan(
    *,
    batch: Mapping[str, Any],
    model_arch: str,
) -> LulynxTrainingStepBatchStagePlan:
    arch = str(model_arch or "").strip().lower()
    has_cached_payload = "latents" in batch and "encoder_hidden_states" in batch
    cached_native = arch in _CACHED_NATIVE_ARCHES and has_cached_payload
    required = ("latents", "encoder_hidden_states", "captions") if cached_native else ("images", "captions")
    missing = tuple(name for name in required if name not in batch)
    return LulynxTrainingStepBatchStagePlan(
        model_arch=arch,
        expected_physical_batch_size=infer_lulynx_batch_leading_dim(batch),
        cached_native=cached_native,
        host_to_device_route="cached_native" if cached_native else "live_image",
        missing_required_fields=missing,
    )


__all__ = [
    "LULYNX_TRAINING_STEP_BATCH_STAGE_PLAN",
    "LulynxTrainingStepBatchStagePlan",
    "build_lulynx_training_step_batch_stage_plan",
    "infer_lulynx_batch_leading_dim",
]
