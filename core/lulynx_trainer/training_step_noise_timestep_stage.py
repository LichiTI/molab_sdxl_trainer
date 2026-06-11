"""Noise/timestep-stage planning for the Lulynx train-step pipeline.

The planner records which stochastic branches a train step is about to use. It
does not sample noise, touch tensors, call schedulers, or change RNG order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LULYNX_TRAINING_STEP_NOISE_TIMESTEP_STAGE_PLAN = "lulynx_training_step_noise_timestep_stage_plan_v0"

_FLOW_ARCHES = {"anima", "newbie"}
_SDXL_FLOW_ARCHES = {"sdxl", "sd15"}


@dataclass(frozen=True)
class LulynxTrainingStepNoiseTimestepStagePlan:
    model_arch: str
    route: str
    batch_size: int
    latent_rank: int
    uses_flow_matching: bool
    uses_sdxl_flow: bool
    uses_ddpm_scheduler: bool
    timestep_sampling: str
    randomization_features: tuple[str, ...]
    compile_caution_reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.batch_size > 0 and self.latent_rank > 0

    @property
    def compile_static_graph_risk(self) -> bool:
        return bool(self.compile_caution_reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_TRAINING_STEP_NOISE_TIMESTEP_STAGE_PLAN,
            "ok": self.ok,
            "model_arch": self.model_arch,
            "route": self.route,
            "batch_size": self.batch_size,
            "latent_rank": self.latent_rank,
            "uses_flow_matching": self.uses_flow_matching,
            "uses_sdxl_flow": self.uses_sdxl_flow,
            "uses_ddpm_scheduler": self.uses_ddpm_scheduler,
            "timestep_sampling": self.timestep_sampling,
            "randomization_features": list(self.randomization_features),
            "compile_static_graph_risk": self.compile_static_graph_risk,
            "compile_caution_reasons": list(self.compile_caution_reasons),
        }


def build_lulynx_training_step_noise_timestep_stage_plan(
    *,
    model_arch: str,
    batch_size: int,
    latent_rank: int,
    flow_model: str = "",
    optimal_noise_enabled: bool = False,
    optimal_noise_candidates: int = 0,
    multires_noise_iterations: int = 0,
    spectral_noise_blend: float = 0.0,
    noise_offset: float = 0.0,
    adaptive_noise_scale: float = 0.0,
    noise_offset_random_strength: bool = False,
    perlin_noise_offset_enabled: bool = False,
    perlin_noise_offset_strength: float = 0.0,
    flow_use_ot: bool = False,
    immiscible_enabled: bool = False,
    immiscible_metric: str = "l2",
    ddpm_timestep_sampling: str = "",
    anima_timestep_sampling: str = "sigma",
    sdxl_timestep_sampling: str = "uniform",
    ip_noise_gamma: float = 0.0,
    ip_noise_gamma_random_strength: bool = False,
) -> LulynxTrainingStepNoiseTimestepStagePlan:
    arch = str(model_arch or "").strip().lower()
    size = max(_as_int(batch_size), 0)
    rank = max(_as_int(latent_rank), 0)
    uses_flow_matching = arch in _FLOW_ARCHES
    uses_sdxl_flow = arch in _SDXL_FLOW_ARCHES and bool(str(flow_model or "").strip())
    route = _resolve_route(arch=arch, uses_flow_matching=uses_flow_matching, uses_sdxl_flow=uses_sdxl_flow)
    features = ["gaussian_noise"]
    cautions: list[str] = []

    if bool(optimal_noise_enabled) and _as_int(optimal_noise_candidates) > 1:
        features[0] = "optimal_noise_candidate_selection"
        cautions.append("candidate_noise_selection_uses_host_loss_probe")
    if _as_int(multires_noise_iterations) > 0:
        features.append("multires_pyramid_noise")
        cautions.append("multires_noise_changes_noise_construction")
    if _as_float(spectral_noise_blend) > 0.0:
        features.append("spectral_noise_blend")
        cautions.append("spectral_noise_uses_frequency_transform")
    if _as_float(noise_offset) > 0.0:
        features.append("noise_offset")
        if _as_float(adaptive_noise_scale) > 0.0:
            features.append("adaptive_noise_scale")
            cautions.append("adaptive_noise_scale_depends_on_latent_statistics")
        if bool(noise_offset_random_strength):
            features.append("noise_offset_random_strength")
            cautions.append("noise_offset_strength_is_sampled_per_step")
    if bool(perlin_noise_offset_enabled) and _as_float(perlin_noise_offset_strength) > 0.0:
        features.append("perlin_noise_offset")
        cautions.append("perlin_noise_offset_adds_dynamic_noise_field")
    if bool(flow_use_ot) and route in {"anima_flow", "transport_flow", "sdxl_flow"}:
        features.append("flow_minibatch_ot")
        cautions.append("flow_ot_reorders_noise_within_batch")
    if bool(immiscible_enabled):
        metric = str(immiscible_metric or "l2").lower()
        features.append(f"immiscible_diffusion_{metric}")
        cautions.append("immiscible_reorders_noise_within_batch")

    timestep_sampling = _resolve_timestep_sampling(
        route=route,
        ddpm_timestep_sampling=ddpm_timestep_sampling,
        anima_timestep_sampling=anima_timestep_sampling,
        sdxl_timestep_sampling=sdxl_timestep_sampling,
    )
    if route == "ddpm" and timestep_sampling == "logit_normal":
        features.append("ddpm_logit_normal_timestep")
    if route == "transport_flow" and timestep_sampling == "logit_normal":
        features.append("flow_logit_normal_timestep")
    if route == "ddpm" and _as_float(ip_noise_gamma) > 0.0:
        features.append("ip_noise_gamma")
        if bool(ip_noise_gamma_random_strength):
            features.append("ip_noise_gamma_random_strength")
            cautions.append("ip_noise_gamma_strength_is_sampled_per_step")
    if size <= 0:
        cautions.append("batch_size_not_observable")
    if rank <= 0:
        cautions.append("latent_rank_not_observable")

    return LulynxTrainingStepNoiseTimestepStagePlan(
        model_arch=arch,
        route=route,
        batch_size=size,
        latent_rank=rank,
        uses_flow_matching=uses_flow_matching,
        uses_sdxl_flow=uses_sdxl_flow,
        uses_ddpm_scheduler=route == "ddpm",
        timestep_sampling=timestep_sampling,
        randomization_features=tuple(features),
        compile_caution_reasons=tuple(_dedupe(cautions)),
    )


def _resolve_route(*, arch: str, uses_flow_matching: bool, uses_sdxl_flow: bool) -> str:
    if arch == "anima":
        return "anima_flow"
    if uses_flow_matching:
        return "transport_flow"
    if uses_sdxl_flow:
        return "sdxl_flow"
    return "ddpm"


def _resolve_timestep_sampling(
    *,
    route: str,
    ddpm_timestep_sampling: str,
    anima_timestep_sampling: str,
    sdxl_timestep_sampling: str,
) -> str:
    if route == "anima_flow":
        return str(anima_timestep_sampling or "sigma").strip().lower()
    if route == "sdxl_flow":
        return str(sdxl_timestep_sampling or "uniform").strip().lower()
    return str(ddpm_timestep_sampling or "uniform").strip().lower()


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = [
    "LULYNX_TRAINING_STEP_NOISE_TIMESTEP_STAGE_PLAN",
    "LulynxTrainingStepNoiseTimestepStagePlan",
    "build_lulynx_training_step_noise_timestep_stage_plan",
]
