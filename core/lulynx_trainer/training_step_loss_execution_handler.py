"""Loss execution handler for Lulynx train steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .training_step_orchestrator_runtime import build_lulynx_stage_orchestrator_runtime


@dataclass(frozen=True)
class LulynxLossExecutionStageExecution:
    loss: Any
    loss_tracker_value: float | None
    orchestrator_runtime: dict[str, Any]


def run_lulynx_loss_execution_stage_handler(
    *,
    owner: Any,
    batch: dict[str, Any],
    prompt_embeds: dict[str, Any],
    noise_pred: Any,
    target: Any,
    timesteps: Any,
    padding_mask: Any,
    noisy_latents: Any,
    uses_flow_matching: bool,
    uses_sdxl_flow: bool,
    do_backward: bool,
    loss_scalars: Any,
    logger: Any,
) -> LulynxLossExecutionStageExecution:
    """Execute the current core and auxiliary loss path unchanged."""

    loss_pred, loss_target = owner._loss_operands(noise_pred, target)
    loss = _compute_core_loss(
        owner=owner,
        batch=batch,
        loss_pred=loss_pred,
        loss_target=loss_target,
        timesteps=timesteps,
        padding_mask=padding_mask,
        uses_flow_matching=uses_flow_matching,
        uses_sdxl_flow=uses_sdxl_flow,
    )
    tracker = owner._loss_tracker
    tracker_value = None
    if tracker:
        tracker_value = loss_scalars.get(loss)
        tracker.record("core", tracker_value, tracker_value)

    loss, tracker_value = _apply_frequency_losses(
        owner=owner,
        batch=batch,
        loss=loss,
        loss_pred=loss_pred,
        loss_target=loss_target,
        tracker_value=tracker_value,
        loss_scalars=loss_scalars,
    )
    loss, tracker_value = _apply_prior_loss(
        owner=owner,
        batch=batch,
        loss=loss,
        uses_flow_matching=uses_flow_matching,
        do_backward=do_backward,
        tracker_value=tracker_value,
        loss_scalars=loss_scalars,
        logger=logger,
    )
    loss, tracker_value = _apply_auxiliary_losses(
        owner=owner,
        batch=batch,
        prompt_embeds=prompt_embeds,
        noise_pred=noise_pred,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        loss=loss,
        do_backward=do_backward,
        tracker_value=tracker_value,
        loss_scalars=loss_scalars,
    )
    return LulynxLossExecutionStageExecution(
        loss=loss,
        loss_tracker_value=tracker_value,
        orchestrator_runtime=build_lulynx_stage_orchestrator_runtime(
            executed_stage_ids=("batch_contract", "host_to_device", "conditioning", "noise_timestep", "forward", "loss"),
            status="loss_execution_stage_handler_executed",
            handler_source="existing_training_loop_loss_execution_path",
        ),
    )


def _compute_core_loss(
    *,
    owner: Any,
    batch: dict[str, Any],
    loss_pred: Any,
    loss_target: Any,
    timesteps: Any,
    padding_mask: Any,
    uses_flow_matching: bool,
    uses_sdxl_flow: bool,
) -> Any:
    if owner._model_arch == "anima":
        from .anima_flow import compute_anima_loss_weighting

        loss = owner._compute_diffusion_loss(loss_pred, loss_target, reduction="none", timesteps=timesteps)
        if padding_mask is not None:
            valid = (~padding_mask).to(device=loss.device, dtype=loss.dtype)
            while valid.dim() < loss.dim():
                valid = valid.unsqueeze(1)
            valid = valid.expand_as(loss)
            loss = (loss * valid).sum(dim=list(range(1, loss.dim()))) / valid.sum(
                dim=list(range(1, valid.dim()))
            ).clamp_min(1.0)
        else:
            loss = loss.mean(dim=list(range(1, loss.dim())))
        weighting = compute_anima_loss_weighting(
            timesteps.float() / 1000.0,
            owner.anima_weighting_scheme,
            mode_scale=owner.anima_mode_scale,
        ).to(device=loss.device, dtype=loss.dtype)
        return owner._weighted_mean_loss(loss * weighting, batch)

    if uses_sdxl_flow and owner._sdxl_flow_sigmas is not None:
        from .sdxl_flow import compute_sdxl_loss_weighting

        loss = owner._compute_diffusion_loss(loss_pred, loss_target, reduction="none", timesteps=timesteps)
        loss = owner._loss_to_per_sample(loss, batch)
        if owner._sdxl_flow_weighting and owner._sdxl_flow_weighting != "none":
            weighting = compute_sdxl_loss_weighting(
                owner._sdxl_flow_sigmas,
                scheme=owner._sdxl_flow_weighting,
                logit_mean=owner.flow_logit_mean,
                logit_std=owner.flow_logit_std,
            ).to(device=loss.device, dtype=loss.dtype)
            loss = loss * weighting
        owner._sdxl_flow_sigmas = None
        return owner._weighted_mean_loss(loss, batch)

    if (owner.snr_gamma or owner.debiased_estimation) and not uses_flow_matching:
        loss = owner._compute_diffusion_loss(loss_pred, loss_target, reduction="none", timesteps=timesteps)
        if owner.v_parameterization:
            loss = owner._scale_v_prediction_loss(loss, timesteps)
        loss = owner._loss_to_per_sample(loss, batch)
        if owner.debiased_estimation:
            alphas_cumprod = owner.noise_scheduler.alphas_cumprod.to(owner.device)
            alpha_t = alphas_cumprod[timesteps]
            loss = loss * (1.0 / (1.0 - alpha_t + 1e-8))
        if owner.adaptive_loss_weighter is not None:
            snr = owner._compute_snr(timesteps)
            loss = loss * owner.adaptive_loss_weighter(snr, v_parameterization=owner.v_parameterization)
        elif owner.faster_dit_snr_weighter is not None:
            snr = owner._compute_snr(timesteps)
            loss = loss * owner.faster_dit_snr_weighter(snr, v_parameterization=owner.v_parameterization)
        elif owner.snr_gamma:
            snr = owner._compute_snr(timesteps)
            divisor = snr + 1.0 if owner.v_parameterization else snr
            loss = loss * (torch.clamp(snr, max=owner.snr_gamma) / divisor)
        return owner._weighted_mean_loss(loss, batch)

    loss = owner._compute_diffusion_loss(loss_pred, loss_target, reduction="none", timesteps=timesteps)
    if owner.v_parameterization and not uses_flow_matching:
        loss = owner._scale_v_prediction_loss(loss, timesteps)
    return owner._weighted_mean_loss(owner._loss_to_per_sample(loss, batch), batch)


def _apply_frequency_losses(
    *,
    owner: Any,
    batch: dict[str, Any],
    loss: Any,
    loss_pred: Any,
    loss_target: Any,
    tracker_value: float | None,
    loss_scalars: Any,
) -> tuple[Any, float | None]:
    tracker = owner._loss_tracker
    if owner.wavelet_loss_enabled:
        from .wavelet_loss import wavelet_loss

        wavelet = wavelet_loss(
            loss_pred,
            loss_target,
            levels=owner.wavelet_loss_levels,
            high_freq_weight=owner.wavelet_loss_high_freq_weight,
            approx_weight=owner.wavelet_loss_approx_weight,
            base_loss=owner.wavelet_loss_base_loss,
            reduction="none",
        )
        loss = loss + owner._weighted_mean_loss(owner._loss_to_per_sample(wavelet, batch), batch)
        tracker_value = _record_loss_delta(tracker, loss_scalars, "wavelet", tracker_value, loss)
    if owner.pattern_loss_enabled:
        from .pattern_loss import pattern_loss

        pattern = pattern_loss(
            loss_pred,
            loss_target,
            levels=owner.pattern_loss_levels,
            ll_type=owner.pattern_loss_ll_type,
            ll_weight=owner.pattern_loss_ll_weight,
            high_type=owner.pattern_loss_high_type,
            high_weight=owner.pattern_loss_high_weight,
            high_huber_c=owner.pattern_loss_high_huber_c,
            reduction="none",
        )
        loss = loss + owner._weighted_mean_loss(owner._loss_to_per_sample(pattern, batch), batch)
        tracker_value = _record_loss_delta(tracker, loss_scalars, "pattern", tracker_value, loss)
    return loss, tracker_value


def _apply_prior_loss(
    *,
    owner: Any,
    batch: dict[str, Any],
    loss: Any,
    uses_flow_matching: bool,
    do_backward: bool,
    tracker_value: float | None,
    loss_scalars: Any,
    logger: Any,
) -> tuple[Any, float | None]:
    if not (do_backward and owner.prior_loss_weight > 0 and owner.reg_dataloader is not None):
        return loss, tracker_value
    try:
        if owner._reg_iter is None:
            owner._reg_iter = iter(owner.reg_dataloader)
        try:
            reg_batch = next(owner._reg_iter)
        except StopIteration:
            owner._reg_iter = iter(owner.reg_dataloader)
            reg_batch = next(owner._reg_iter)
        reg_images = reg_batch["images"].to(owner.device, dtype=owner.dtype)
        reg_latents = owner._encode_latents_with_vae(reg_images)
        reg_noise = torch.randn_like(reg_latents)
        if uses_flow_matching:
            reg_t = torch.rand((reg_latents.shape[0],), device=owner.device, dtype=reg_latents.dtype)
            reg_view_t = reg_t.view(reg_latents.shape[0], *([1] * (reg_latents.dim() - 1)))
            reg_noisy = (1.0 - reg_view_t) * reg_latents + reg_view_t * reg_noise
            reg_timesteps = (reg_t * 1000.0).to(device=owner.device, dtype=reg_latents.dtype)
            reg_target = reg_noise - reg_latents
        else:
            reg_timesteps = torch.randint(
                0,
                owner.noise_scheduler.config.num_train_timesteps,
                (reg_latents.shape[0],),
                device=owner.device,
            ).long()
            reg_noisy = owner.noise_scheduler.add_noise(reg_latents, reg_noise, reg_timesteps)
            reg_target = owner._velocity_target(reg_latents, reg_noise, reg_timesteps) if owner.v_parameterization else reg_noise
        reg_prompt_embeds = owner._encode_prompt(reg_batch.get("captions", [""] * reg_latents.shape[0]))
        with torch.autocast(device_type="cuda", dtype=owner.dtype):
            reg_pred = owner.unet(
                sample=reg_noisy,
                timestep=reg_timesteps,
                encoder_hidden_states=reg_prompt_embeds["encoder_hidden_states"],
            ).sample
        reg_loss_pred, reg_loss_target = owner._loss_operands(reg_pred, reg_target)
        reg_loss = owner._compute_diffusion_loss(reg_loss_pred, reg_loss_target, timesteps=reg_timesteps)
        loss = loss + owner.prior_loss_weight * reg_loss
        tracker_value = _record_loss_delta(owner._loss_tracker, loss_scalars, "prior", tracker_value, loss, scale=owner.prior_loss_weight)
    except Exception as exc:
        logger.warning(f"[PriorPreservation] Failed: {exc}")
    return loss, tracker_value


def _apply_auxiliary_losses(
    *,
    owner: Any,
    batch: dict[str, Any],
    prompt_embeds: dict[str, Any],
    noise_pred: Any,
    noisy_latents: Any,
    timesteps: Any,
    loss: Any,
    do_backward: bool,
    tracker_value: float | None,
    loss_scalars: Any,
) -> tuple[Any, float | None]:
    tracker = owner._loss_tracker
    if do_backward and owner.lulynx_wrapper:
        extra_loss = owner.lulynx_wrapper.compute_loss(
            model=owner.unet,
            current_features=owner.lulynx_wrapper._current_features,
            step=owner.global_step,
            timestep=timesteps[0].item() if timesteps.dim() > 0 else timesteps.item(),
            network=owner.lora_injector,
        )
        loss = loss + extra_loss.to(loss.device)
        tracker_value = _record_loss_delta(tracker, loss_scalars, "lulynx", tracker_value, loss)
    repa_loss = owner._compute_repa_loss(batch, prompt_embeds)
    if repa_loss is not None:
        loss = loss + repa_loss.to(loss.device)
        tracker_value = _record_loss_delta(tracker, loss_scalars, "repa", tracker_value, loss)
    sra2_haste_loss = owner._compute_sra2_haste_loss(batch, prompt_embeds)
    if sra2_haste_loss is not None:
        loss = loss + sra2_haste_loss.to(loss.device)
        tracker_value = _record_loss_delta(tracker, loss_scalars, "sra2_haste", tracker_value, loss)
    if owner.dop is not None and owner.dop.should_compute(owner.global_step):
        added_cond = batch.get("added_cond_kwargs") if isinstance(batch, dict) else None
        dop_loss = owner.dop.compute_loss(
            current_output=noise_pred,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=prompt_embeds.get("encoder_hidden_states", prompt_embeds)
            if isinstance(prompt_embeds, dict)
            else prompt_embeds,
            added_cond_kwargs=added_cond,
        )
        loss = loss + dop_loss.to(loss.device)
        tracker_value = _record_loss_delta(tracker, loss_scalars, "dop", tracker_value, loss)
    if do_backward and owner.b_tier_runtime is not None:
        b_tier_loss, b_tier_state = owner.b_tier_runtime.compute_loss(
            step=owner.global_step,
            timesteps=timesteps,
            loss_device=loss.device,
        )
        owner._b_tier_last_state = dict(b_tier_state or {})
        if b_tier_loss is not None:
            loss = loss + b_tier_loss.to(loss.device)
            tracker_value = _record_loss_delta(tracker, loss_scalars, "b_tier", tracker_value, loss)
    return loss, tracker_value


def _record_loss_delta(
    tracker: Any,
    loss_scalars: Any,
    name: str,
    old_value: float | None,
    loss: Any,
    *,
    scale: float | None = None,
) -> float | None:
    if not tracker:
        return old_value
    new_value = loss_scalars.get(loss)
    if scale is None:
        tracker.record(name, old_value, new_value)
    else:
        tracker.record(name, old_value, new_value, scale=scale)
    return new_value


__all__ = ["LulynxLossExecutionStageExecution", "run_lulynx_loss_execution_stage_handler"]
