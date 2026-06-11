"""Host-to-device transfer-stage planning for the Lulynx train-step pipeline.

The planner only describes the transfer route that the current train step is
about to use. It does not move tensors, pin memory, or synchronize CUDA.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


LULYNX_TRAINING_STEP_TRANSFER_STAGE_PLAN = "lulynx_training_step_transfer_stage_plan_v0"


@dataclass(frozen=True)
class LulynxTrainingStepTransferStagePlan:
    route: str
    model_arch: str
    cached_native: bool
    target_dtype: str
    has_images: bool
    has_latents: bool
    has_encoder_hidden_states: bool
    has_attention_mask: bool
    compile_caution_reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.route)

    @property
    def compile_static_graph_risk(self) -> bool:
        return bool(self.compile_caution_reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_TRAINING_STEP_TRANSFER_STAGE_PLAN,
            "ok": self.ok,
            "route": self.route,
            "model_arch": self.model_arch,
            "cached_native": self.cached_native,
            "target_dtype": self.target_dtype,
            "has_images": self.has_images,
            "has_latents": self.has_latents,
            "has_encoder_hidden_states": self.has_encoder_hidden_states,
            "has_attention_mask": self.has_attention_mask,
            "compile_static_graph_risk": self.compile_static_graph_risk,
            "compile_caution_reasons": list(self.compile_caution_reasons),
        }


def build_lulynx_training_step_transfer_stage_plan(
    *,
    batch: Mapping[str, Any],
    route: str,
    model_arch: str,
    cached_native: bool,
    target_dtype: Any,
) -> LulynxTrainingStepTransferStagePlan:
    source = batch if isinstance(batch, Mapping) else {}
    cautions: list[str] = []
    if not cached_native and "images" not in source:
        cautions.append("live_image_route_missing_images")
    if cached_native and "latents" not in source:
        cautions.append("cached_native_route_missing_latents")
    if cached_native and "encoder_hidden_states" not in source:
        cautions.append("cached_native_route_missing_encoder_hidden_states")
    if source.get("attention_mask") is not None:
        cautions.append("attention_mask_transfer_surface_present")
    return LulynxTrainingStepTransferStagePlan(
        route=str(route or ""),
        model_arch=str(model_arch or "").strip().lower(),
        cached_native=bool(cached_native),
        target_dtype=str(target_dtype or ""),
        has_images="images" in source,
        has_latents="latents" in source,
        has_encoder_hidden_states="encoder_hidden_states" in source,
        has_attention_mask=source.get("attention_mask") is not None,
        compile_caution_reasons=tuple(_dedupe(cautions)),
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = [
    "LULYNX_TRAINING_STEP_TRANSFER_STAGE_PLAN",
    "LulynxTrainingStepTransferStagePlan",
    "build_lulynx_training_step_transfer_stage_plan",
]
