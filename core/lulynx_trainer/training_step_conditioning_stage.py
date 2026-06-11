"""Conditioning-stage planning for the Lulynx train-step pipeline.

The planner summarizes prompt-conditioning shape and stochastic dropout routes.
It never mutates prompt tensors and never calls encoder modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


LULYNX_TRAINING_STEP_CONDITIONING_STAGE_PLAN = "lulynx_training_step_conditioning_stage_plan_v0"


@dataclass(frozen=True)
class LulynxTrainingStepConditioningStagePlan:
    model_arch: str
    conditioning_route: str
    batch_size: int
    has_encoder_hidden_states: bool
    has_pooled_prompt_embeds: bool
    has_attention_mask: bool
    has_qwen3_hidden_states: bool
    has_qwen3_attention_mask: bool
    has_t5_hidden_states: bool
    dropout_features: tuple[str, ...]
    compile_caution_reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.has_encoder_hidden_states and self.batch_size > 0

    @property
    def compile_static_graph_risk(self) -> bool:
        return bool(self.compile_caution_reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_TRAINING_STEP_CONDITIONING_STAGE_PLAN,
            "ok": self.ok,
            "model_arch": self.model_arch,
            "conditioning_route": self.conditioning_route,
            "batch_size": self.batch_size,
            "has_encoder_hidden_states": self.has_encoder_hidden_states,
            "has_pooled_prompt_embeds": self.has_pooled_prompt_embeds,
            "has_attention_mask": self.has_attention_mask,
            "has_qwen3_hidden_states": self.has_qwen3_hidden_states,
            "has_qwen3_attention_mask": self.has_qwen3_attention_mask,
            "has_t5_hidden_states": self.has_t5_hidden_states,
            "dropout_features": list(self.dropout_features),
            "compile_static_graph_risk": self.compile_static_graph_risk,
            "compile_caution_reasons": list(self.compile_caution_reasons),
        }


def build_lulynx_training_step_conditioning_stage_plan(
    *,
    prompt_embeds: Mapping[str, Any] | None,
    model_arch: str,
    cached_native: bool,
    do_backward: bool,
    qwen3_encoder_available: bool = False,
    te_dropout: float = 0.0,
    clip_l_dropout_rate: float = 0.0,
    clip_g_dropout_rate: float = 0.0,
    t5_dropout_rate: float = 0.0,
) -> LulynxTrainingStepConditioningStagePlan:
    embeds = prompt_embeds if isinstance(prompt_embeds, Mapping) else {}
    arch = str(model_arch or "").strip().lower()
    encoder_hidden_states = embeds.get("encoder_hidden_states")
    pooled_prompt_embeds = embeds.get("pooled_prompt_embeds")
    attention_mask = embeds.get("attention_mask")
    qwen3_hidden_states = embeds.get("qwen3_hidden_states")
    qwen3_attention_mask = embeds.get("qwen3_attention_mask")
    t5_hidden_states = embeds.get("t5_hidden_states")
    route = _resolve_conditioning_route(
        model_arch=arch,
        cached_native=bool(cached_native),
        has_qwen3=hasattr(qwen3_hidden_states, "shape"),
        qwen3_encoder_available=bool(qwen3_encoder_available),
    )
    dropout_features = _resolve_dropout_features(
        do_backward=bool(do_backward),
        te_dropout=te_dropout,
        clip_l_dropout_rate=clip_l_dropout_rate,
        clip_g_dropout_rate=clip_g_dropout_rate,
        t5_dropout_rate=t5_dropout_rate,
    )
    cautions: list[str] = []
    if dropout_features:
        cautions.append("conditioning_dropout_is_sampled_per_step")
    if not hasattr(encoder_hidden_states, "shape"):
        cautions.append("encoder_hidden_states_not_observable")
    if arch == "anima" and bool(qwen3_encoder_available) and route == "live_text_encoder":
        cautions.append("qwen3_secondary_conditioning_missing_after_live_encode")
    return LulynxTrainingStepConditioningStagePlan(
        model_arch=arch,
        conditioning_route=route,
        batch_size=_leading_dim(encoder_hidden_states),
        has_encoder_hidden_states=hasattr(encoder_hidden_states, "shape"),
        has_pooled_prompt_embeds=hasattr(pooled_prompt_embeds, "shape"),
        has_attention_mask=hasattr(attention_mask, "shape"),
        has_qwen3_hidden_states=hasattr(qwen3_hidden_states, "shape"),
        has_qwen3_attention_mask=hasattr(qwen3_attention_mask, "shape"),
        has_t5_hidden_states=hasattr(t5_hidden_states, "shape"),
        dropout_features=dropout_features,
        compile_caution_reasons=tuple(cautions),
    )


def _resolve_conditioning_route(
    *,
    model_arch: str,
    cached_native: bool,
    has_qwen3: bool,
    qwen3_encoder_available: bool,
) -> str:
    base = "cached_prompt" if cached_native else "live_text_encoder"
    if model_arch == "anima" and (has_qwen3 or qwen3_encoder_available):
        return f"{base}_with_qwen3"
    return base


def _resolve_dropout_features(
    *,
    do_backward: bool,
    te_dropout: float,
    clip_l_dropout_rate: float,
    clip_g_dropout_rate: float,
    t5_dropout_rate: float,
) -> tuple[str, ...]:
    if not do_backward:
        return ()
    features: list[str] = []
    if _as_float(te_dropout) > 0.0:
        features.append("text_encoder_dropout")
    if _as_float(clip_l_dropout_rate) > 0.0:
        features.append("clip_l_dropout")
    if _as_float(clip_g_dropout_rate) > 0.0:
        features.append("clip_g_dropout")
    if _as_float(t5_dropout_rate) > 0.0:
        features.append("t5_dropout")
    return tuple(features)


def _leading_dim(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if not shape:
        return 0
    try:
        return max(int(shape[0]), 0)
    except (TypeError, ValueError, IndexError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "LULYNX_TRAINING_STEP_CONDITIONING_STAGE_PLAN",
    "LulynxTrainingStepConditioningStagePlan",
    "build_lulynx_training_step_conditioning_stage_plan",
]
