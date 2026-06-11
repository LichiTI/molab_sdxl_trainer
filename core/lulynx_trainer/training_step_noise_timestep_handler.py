"""Noise/timestep handler for Lulynx train steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .training_pipeline_trace import LulynxTrainingPipelineTrace
from .training_step_noise_timestep_stage import (
    LulynxTrainingStepNoiseTimestepStagePlan,
    build_lulynx_training_step_noise_timestep_stage_plan,
)
from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxNoiseTimestepStageExecution:
    noise_timestep_stage_plan: LulynxTrainingStepNoiseTimestepStagePlan
    noise: Any
    noisy_latents: Any
    target: Any
    timesteps: Any
    batch_size: int
    uses_flow_matching: bool
    uses_sdxl_flow: bool
    sdxl_flow_sigmas: Any
    sdxl_flow_weighting: str
    orchestrator_runtime: dict[str, Any]


def run_lulynx_noise_timestep_stage_handler(
    *,
    latents: Any,
    model_arch: str,
    trace: LulynxTrainingPipelineTrace,
    device: Any,
    flow_model: str,
    noise_scheduler: Any,
    v_parameterization: bool,
    optimal_noise_enabled: bool,
    optimal_noise_candidates: int,
    multires_noise_iterations: int,
    multires_noise_discount: float,
    spectral_noise_blend: float,
    spectral_noise_sigma: float,
    noise_offset: float,
    adaptive_noise_scale: float,
    noise_offset_random_strength: bool,
    perlin_noise_offset_enabled: bool,
    perlin_noise_offset_strength: float,
    perlin_noise_offset_scale: float,
    flow_use_ot: bool,
    immiscible_enabled: bool = False,
    immiscible_metric: str = "l2",
    ddpm_timestep_sampling: str,
    anima_timestep_sampling: str,
    anima_sigmoid_scale: float,
    anima_discrete_flow_shift: float,
    anima_weighting_scheme: str,
    anima_model_prediction_type: str,
    sdxl_timestep_sampling: str,
    sdxl_sigmoid_scale: float,
    sdxl_flow_shift: float,
    sdxl_flow_weighting_scheme: str,
    sdxl_model_prediction_type: str,
    flow_logit_mean: float,
    flow_logit_std: float,
    ip_noise_gamma: float,
    ip_noise_gamma_random_strength: bool,
    sample_strength: Callable[..., Any],
    velocity_target: Callable[[Any, Any, Any], Any],
    log_debug: Callable[[str], Any] | None = None,
) -> LulynxNoiseTimestepStageExecution:
    if log_debug is not None:
        log_debug("Sampling Noise...")
    if optimal_noise_enabled and int(optimal_noise_candidates or 0) > 1:
        from .optimal_noise import select_optimal_noise

        noise = select_optimal_noise(
            latents,
            loss_fn=lambda n: torch.nn.functional.mse_loss(n, latents).item(),
            n_candidates=int(optimal_noise_candidates),
        )
    else:
        noise = torch.randn_like(latents)
    batch_size = int(latents.shape[0])
    arch = str(model_arch or "").strip().lower()
    uses_flow_matching = arch in {"anima", "newbie"}
    uses_sdxl_flow = arch in {"sdxl", "sd15"} and bool(flow_model)
    plan = build_lulynx_training_step_noise_timestep_stage_plan(
        model_arch=arch,
        batch_size=batch_size,
        latent_rank=latents.dim(),
        flow_model=flow_model,
        optimal_noise_enabled=optimal_noise_enabled,
        optimal_noise_candidates=optimal_noise_candidates,
        multires_noise_iterations=multires_noise_iterations,
        spectral_noise_blend=spectral_noise_blend,
        noise_offset=noise_offset,
        adaptive_noise_scale=adaptive_noise_scale,
        noise_offset_random_strength=noise_offset_random_strength,
        perlin_noise_offset_enabled=perlin_noise_offset_enabled,
        perlin_noise_offset_strength=perlin_noise_offset_strength,
        flow_use_ot=flow_use_ot,
        immiscible_enabled=immiscible_enabled,
        immiscible_metric=immiscible_metric,
        ddpm_timestep_sampling=ddpm_timestep_sampling,
        anima_timestep_sampling=anima_timestep_sampling,
        sdxl_timestep_sampling=sdxl_timestep_sampling,
        ip_noise_gamma=ip_noise_gamma,
        ip_noise_gamma_random_strength=ip_noise_gamma_random_strength,
    )
    trace.mark("noise_timestep", batch_size=batch_size, latent_dims=int(latents.dim()), noise_timestep_stage_plan=plan.as_dict())
    noise = _apply_noise_transforms(
        noise=noise,
        latents=latents,
        multires_noise_iterations=multires_noise_iterations,
        multires_noise_discount=multires_noise_discount,
        spectral_noise_blend=spectral_noise_blend,
        spectral_noise_sigma=spectral_noise_sigma,
        noise_offset=noise_offset,
        adaptive_noise_scale=adaptive_noise_scale,
        noise_offset_random_strength=noise_offset_random_strength,
        perlin_noise_offset_enabled=perlin_noise_offset_enabled,
        perlin_noise_offset_strength=perlin_noise_offset_strength,
        perlin_noise_offset_scale=perlin_noise_offset_scale,
        sample_strength=sample_strength,
    )
    noisy_latents, target, timesteps, sdxl_sigmas, sdxl_weighting = _build_noisy_inputs(
        arch=arch,
        uses_flow_matching=uses_flow_matching,
        uses_sdxl_flow=uses_sdxl_flow,
        latents=latents,
        noise=noise,
        batch_size=batch_size,
        device=device,
        noise_scheduler=noise_scheduler,
        v_parameterization=v_parameterization,
        flow_use_ot=flow_use_ot,
        immiscible_enabled=immiscible_enabled,
        immiscible_metric=immiscible_metric,
        ddpm_timestep_sampling=ddpm_timestep_sampling,
        anima_timestep_sampling=anima_timestep_sampling,
        anima_sigmoid_scale=anima_sigmoid_scale,
        anima_discrete_flow_shift=anima_discrete_flow_shift,
        anima_weighting_scheme=anima_weighting_scheme,
        anima_model_prediction_type=anima_model_prediction_type,
        sdxl_timestep_sampling=sdxl_timestep_sampling,
        sdxl_sigmoid_scale=sdxl_sigmoid_scale,
        sdxl_flow_shift=sdxl_flow_shift,
        sdxl_flow_weighting_scheme=sdxl_flow_weighting_scheme,
        sdxl_model_prediction_type=sdxl_model_prediction_type,
        flow_logit_mean=flow_logit_mean,
        flow_logit_std=flow_logit_std,
        ip_noise_gamma=ip_noise_gamma,
        ip_noise_gamma_random_strength=ip_noise_gamma_random_strength,
        sample_strength=sample_strength,
        velocity_target=velocity_target,
    )
    return LulynxNoiseTimestepStageExecution(
        noise_timestep_stage_plan=plan,
        noise=noise,
        noisy_latents=noisy_latents,
        target=target,
        timesteps=timesteps,
        batch_size=batch_size,
        uses_flow_matching=uses_flow_matching,
        uses_sdxl_flow=uses_sdxl_flow,
        sdxl_flow_sigmas=sdxl_sigmas,
        sdxl_flow_weighting=sdxl_weighting,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "host_to_device", "conditioning", "noise_timestep"),
            status="noise_timestep_stage_handler_executed",
            handler_source="existing_training_loop_noise_timestep_path",
            stage_plans={"noise_timestep_stage_plan": plan.as_dict()},
        ),
    )


def _apply_noise_transforms(
    *,
    noise: Any,
    latents: Any,
    multires_noise_iterations: int,
    multires_noise_discount: float,
    spectral_noise_blend: float,
    spectral_noise_sigma: float,
    noise_offset: float,
    adaptive_noise_scale: float,
    noise_offset_random_strength: bool,
    perlin_noise_offset_enabled: bool,
    perlin_noise_offset_strength: float,
    perlin_noise_offset_scale: float,
    sample_strength: Callable[..., Any],
) -> Any:
    if int(multires_noise_iterations or 0) > 0:
        from ..training_components.noise_utils import pyramid_noise_like

        noise = pyramid_noise_like(noise, int(multires_noise_iterations), multires_noise_discount)
    if float(spectral_noise_blend or 0.0) > 0.0:
        from .spectral_noise_blend import blend_spectral_noise

        noise = blend_spectral_noise(noise, spectral_noise_blend, spectral_noise_sigma)
    if float(noise_offset or 0.0) > 0.0:
        offset = noise_offset
        if float(adaptive_noise_scale or 0.0) > 0.0:
            from ..training_components.noise_utils import apply_adaptive_noise_scale

            offset = apply_adaptive_noise_scale(offset, latents, adaptive_noise_scale)
        strength = sample_strength(offset, noise_offset_random_strength, (latents.shape[0], 1, 1, 1), latents)
        noise += strength * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device, dtype=latents.dtype)
    if bool(perlin_noise_offset_enabled) and float(perlin_noise_offset_strength or 0.0) > 0.0:
        from .perlin_noise import apply_perlin_noise_offset

        noise = apply_perlin_noise_offset(noise, strength=perlin_noise_offset_strength, scale=perlin_noise_offset_scale)
    return noise


def _maybe_assign_noise(latents: Any, noise: Any, **kwargs: Any) -> Any:
    """Optionally reorder noise to data within the minibatch.

    Unified hook for both families: flow routes keep their existing cosine OT
    (``flow_use_ot``, unchanged), while ``immiscible_diffusion_enabled`` turns on
    Immiscible assignment (L2 by default) across DDPM *and* flow.  When nothing
    is enabled, returns ``noise`` untouched (bit-identical to legacy).
    """
    immiscible = bool(kwargs.get("immiscible_enabled", False))
    metric = str(kwargs.get("immiscible_metric", "l2") or "l2").lower()
    flow_use_ot = bool(kwargs.get("flow_use_ot", False))
    if immiscible and metric == "l2":
        from .immiscible_diffusion import minibatch_immiscible_l2

        return minibatch_immiscible_l2(latents, noise)
    if flow_use_ot or immiscible:  # cosine path: legacy flow OT, or metric=cosine
        from .cosine_ot import minibatch_ot_cosine

        return minibatch_ot_cosine(latents, noise)
    return noise


def _noise_assignment_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "immiscible_enabled": kwargs.get("immiscible_enabled", False),
        "immiscible_metric": kwargs.get("immiscible_metric", "l2"),
        "flow_use_ot": kwargs.get("flow_use_ot", False),
    }


def _build_noisy_inputs(**kwargs: Any) -> tuple[Any, Any, Any, Any, str]:
    arch = kwargs["arch"]
    if arch == "anima":
        return (*_build_anima_noisy_inputs(**kwargs), None, "none")
    if kwargs["uses_flow_matching"]:
        return (*_build_transport_noisy_inputs(**kwargs), None, "none")
    if kwargs["uses_sdxl_flow"]:
        noisy, target, timesteps, sigmas, weighting = _build_sdxl_noisy_inputs(**kwargs)
        return noisy, target, timesteps, sigmas, weighting
    return (*_build_ddpm_noisy_inputs(**kwargs), None, "none")


def _build_anima_noisy_inputs(**kwargs: Any) -> tuple[Any, Any, Any]:
    from .anima_flow import AnimaFlowConfig, build_anima_flow_inputs, sample_anima_sigmas

    latents = kwargs["latents"]
    noise = kwargs["noise"]
    cfg = AnimaFlowConfig(
        timestep_sampling=kwargs["anima_timestep_sampling"],
        sigmoid_scale=kwargs["anima_sigmoid_scale"],
        discrete_flow_shift=kwargs["anima_discrete_flow_shift"],
        weighting_scheme=kwargs["anima_weighting_scheme"],
        logit_mean=kwargs["flow_logit_mean"],
        logit_std=kwargs["flow_logit_std"],
    )
    sigmas = sample_anima_sigmas(kwargs["batch_size"], device=kwargs["device"], dtype=latents.dtype, config=cfg)
    noise = _maybe_assign_noise(latents, noise, **_noise_assignment_options(kwargs))
    return build_anima_flow_inputs(
        latents,
        noise,
        sigmas,
        num_train_timesteps=1000,
        model_prediction_type=kwargs["anima_model_prediction_type"] or "velocity",
    )


def _build_transport_noisy_inputs(**kwargs: Any) -> tuple[Any, Any, Any]:
    latents = kwargs["latents"]
    noise = kwargs["noise"]
    batch_size = kwargs["batch_size"]
    if kwargs["ddpm_timestep_sampling"] == "logit_normal":
        normals = torch.randn((batch_size,), device=kwargs["device"], dtype=latents.dtype)
        flow_t = torch.sigmoid(kwargs["flow_logit_mean"] + kwargs["flow_logit_std"] * normals)
    else:
        flow_t = torch.rand((batch_size,), device=kwargs["device"], dtype=latents.dtype)
    view_t = flow_t.view(batch_size, *([1] * (latents.dim() - 1)))
    noise = _maybe_assign_noise(latents, noise, **_noise_assignment_options(kwargs))
    noisy = (torch.ones_like(view_t) - view_t) * latents + view_t * noise
    return noisy, noise - latents, (flow_t * latents.new_tensor(1000.0)).to(device=kwargs["device"], dtype=latents.dtype)


def _build_sdxl_noisy_inputs(**kwargs: Any) -> tuple[Any, Any, Any, Any, str]:
    from .sdxl_flow import SDXLFlowConfig, build_sdxl_flow_inputs, sample_sdxl_flow_sigmas

    latents = kwargs["latents"]
    noise = kwargs["noise"]
    cfg = SDXLFlowConfig(
        timestep_sampling=kwargs["sdxl_timestep_sampling"] or "uniform",
        sigmoid_scale=kwargs["sdxl_sigmoid_scale"],
        discrete_flow_shift=kwargs["sdxl_flow_shift"],
        weighting_scheme=kwargs["sdxl_flow_weighting_scheme"] or "none",
        model_prediction_type=kwargs["sdxl_model_prediction_type"] or "epsilon",
        logit_mean=kwargs["flow_logit_mean"],
        logit_std=kwargs["flow_logit_std"],
    )
    sigmas = sample_sdxl_flow_sigmas(kwargs["batch_size"], device=kwargs["device"], dtype=latents.dtype, config=cfg)
    noise = _maybe_assign_noise(latents, noise, **_noise_assignment_options(kwargs))
    noisy, target, timesteps = build_sdxl_flow_inputs(
        latents,
        noise,
        sigmas,
        num_train_timesteps=1000,
        model_prediction_type=cfg.model_prediction_type,
    )
    return noisy, target, timesteps, sigmas, cfg.weighting_scheme


def _build_ddpm_noisy_inputs(**kwargs: Any) -> tuple[Any, Any, Any]:
    latents = kwargs["latents"]
    noise = kwargs["noise"]
    scheduler = kwargs["noise_scheduler"]
    batch_size = kwargs["batch_size"]
    device = kwargs["device"]

    # Immiscible assignment on the raw noise (before ip-noise / add_noise). This
    # is the new DDPM/standard-diffusion coverage — flow routes handle their own.
    noise = _maybe_assign_noise(latents, noise, **_noise_assignment_options(kwargs))

    # Timestep sampling
    if kwargs["ddpm_timestep_sampling"] == "logit_normal":
        max_t = int(scheduler.config.num_train_timesteps)
        normals = torch.randn((batch_size,), device=device)
        probs = torch.sigmoid(kwargs["flow_logit_mean"] + kwargs["flow_logit_std"] * normals)
        timesteps = (probs * max_t).clamp(0, max_t - 1).long()
    elif kwargs["ddpm_timestep_sampling"] in ("low_snr_bias", "snr_weighted"):
        # FasterDiT SNR-aware timestep sampling
        from .faster_dit_snr import sample_timesteps_with_snr_bias
        timesteps = sample_timesteps_with_snr_bias(
            batch_size=batch_size,
            num_train_timesteps=scheduler.config.num_train_timesteps,
            alphas_cumprod=scheduler.alphas_cumprod,
            device=device,
            strategy=kwargs["ddpm_timestep_sampling"],
            bias_strength=float(kwargs.get("faster_dit_snr_bias_strength", 1.5)),
        )
    else:
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

    if float(kwargs["ip_noise_gamma"] or 0.0) > 0.0:
        from ..training_components.noise_utils import apply_ip_noise

        gamma: float | torch.Tensor = kwargs["ip_noise_gamma"]
        if kwargs["ip_noise_gamma_random_strength"]:
            gamma = kwargs["sample_strength"](kwargs["ip_noise_gamma"], True, (batch_size,), latents)
        noise = apply_ip_noise(noise, gamma, timesteps, scheduler.alphas_cumprod)
    noisy = scheduler.add_noise(latents, noise, timesteps)
    target = kwargs["velocity_target"](latents, noise, timesteps) if kwargs["v_parameterization"] else noise
    return noisy, target, timesteps


__all__ = ["LulynxNoiseTimestepStageExecution", "run_lulynx_noise_timestep_stage_handler"]
