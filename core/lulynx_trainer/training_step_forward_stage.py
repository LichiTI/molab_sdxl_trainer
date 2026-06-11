"""Forward-stage planning for the Lulynx train-step pipeline.

The planner describes the UNet input surface and execution route. It does not
call model code, allocate tensors, or decide whether a route should be used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


LULYNX_TRAINING_STEP_FORWARD_STAGE_PLAN = "lulynx_training_step_forward_stage_plan_v0"


@dataclass(frozen=True)
class LulynxTrainingStepForwardStagePlan:
    model_arch: str
    execution_route: str
    batch_size: int
    has_added_cond_kwargs: bool
    added_cond_keys: tuple[str, ...]
    has_padding_mask: bool
    has_qwen3_hidden_states: bool
    has_qwen3_attention_mask: bool
    has_control_residual_route: bool
    has_ip_adapter_route: bool
    compile_caution_reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.batch_size > 0

    @property
    def compile_static_graph_risk(self) -> bool:
        return bool(self.compile_caution_reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_TRAINING_STEP_FORWARD_STAGE_PLAN,
            "ok": self.ok,
            "model_arch": self.model_arch,
            "execution_route": self.execution_route,
            "batch_size": self.batch_size,
            "has_added_cond_kwargs": self.has_added_cond_kwargs,
            "added_cond_keys": list(self.added_cond_keys),
            "has_padding_mask": self.has_padding_mask,
            "has_qwen3_hidden_states": self.has_qwen3_hidden_states,
            "has_qwen3_attention_mask": self.has_qwen3_attention_mask,
            "has_control_residual_route": self.has_control_residual_route,
            "has_ip_adapter_route": self.has_ip_adapter_route,
            "compile_static_graph_risk": self.compile_static_graph_risk,
            "compile_caution_reasons": list(self.compile_caution_reasons),
        }


def build_lulynx_training_step_forward_stage_plan(
    *,
    unet_kwargs: Mapping[str, Any],
    model_arch: str,
    cudagraph_active: bool = False,
    cudagraph_requested: bool = False,
    cpu_offload_checkpointing: bool = False,
    offloaded_checkpoint_context_available: bool = False,
    control_residual_applied: bool = False,
    ip_adapter_applied: bool = False,
) -> LulynxTrainingStepForwardStagePlan:
    kwargs = unet_kwargs if isinstance(unet_kwargs, Mapping) else {}
    arch = str(model_arch or "").strip().lower()
    route = _resolve_execution_route(
        cudagraph_active=bool(cudagraph_active),
        cudagraph_requested=bool(cudagraph_requested),
        cpu_offload_checkpointing=bool(cpu_offload_checkpointing),
        offloaded_checkpoint_context_available=bool(offloaded_checkpoint_context_available),
    )
    added_cond = kwargs.get("added_cond_kwargs")
    added_cond_keys = tuple(sorted(str(key) for key in added_cond.keys())) if isinstance(added_cond, Mapping) else ()
    cautions: list[str] = []
    if route in {"cudagraph_replay_or_eager", "cudagraph_capture_candidate"}:
        cautions.append("cudagraph_requires_static_unet_input_surface")
    if route in {"offloaded_checkpoint_forward", "cpu_offload_checkpoint_forward"}:
        cautions.append("checkpoint_forward_route_changes_forward_execution")
    if bool(control_residual_applied):
        cautions.append("control_residual_modifies_noisy_latents_before_forward")
    if bool(ip_adapter_applied):
        cautions.append("ip_adapter_merges_image_tokens_into_text_conditioning")
    if "sample" not in kwargs:
        cautions.append("sample_not_observable")
    if "encoder_hidden_states" not in kwargs:
        cautions.append("encoder_hidden_states_not_observable")
    if "timestep" not in kwargs:
        cautions.append("timestep_not_observable")
    return LulynxTrainingStepForwardStagePlan(
        model_arch=arch,
        execution_route=route,
        batch_size=_leading_dim(kwargs.get("sample")),
        has_added_cond_kwargs=isinstance(added_cond, Mapping),
        added_cond_keys=added_cond_keys,
        has_padding_mask=hasattr(kwargs.get("padding_mask"), "shape"),
        has_qwen3_hidden_states=hasattr(kwargs.get("qwen3_hidden_states"), "shape"),
        has_qwen3_attention_mask=hasattr(kwargs.get("qwen3_attention_mask"), "shape"),
        has_control_residual_route=bool(control_residual_applied),
        has_ip_adapter_route=bool(ip_adapter_applied),
        compile_caution_reasons=tuple(_dedupe(cautions)),
    )


def _resolve_execution_route(
    *,
    cudagraph_active: bool,
    cudagraph_requested: bool,
    cpu_offload_checkpointing: bool,
    offloaded_checkpoint_context_available: bool,
) -> str:
    if cudagraph_active and cudagraph_requested:
        return "cudagraph_replay_or_eager"
    if cpu_offload_checkpointing:
        return "offloaded_checkpoint_forward" if offloaded_checkpoint_context_available else "cpu_offload_checkpoint_forward"
    if cudagraph_requested:
        return "cudagraph_capture_candidate"
    return "eager_unet_forward"


def _leading_dim(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if not shape:
        return 0
    try:
        return max(int(shape[0]), 0)
    except (TypeError, ValueError, IndexError):
        return 0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = [
    "LULYNX_TRAINING_STEP_FORWARD_STAGE_PLAN",
    "LulynxTrainingStepForwardStagePlan",
    "build_lulynx_training_step_forward_stage_plan",
]
