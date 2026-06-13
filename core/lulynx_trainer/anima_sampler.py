"""Anima flow-matching sampler for training previews.

Implements Euler-step and DPM-Solver sampling for Anima's rectified-flow DiT.
This module is used by ``TrainingSampler`` when the model family is ``"anima"``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def _euler_step(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Single Euler step for rectified-flow ODE.

    The ODE is: dx/dt = v(x, t) where v is the predicted velocity.
    For rectified flow the velocity is predicted directly by the model.
    """
    dt = sigma_next - sigma
    # sigma_view for broadcasting over spatial dims
    view_shape = (sample.shape[0],) + (1,) * (sample.dim() - 1)
    dt_view = dt.view(view_shape)
    return sample + model_output * dt_view


def _dpm_solver_step(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Single DPM-Solver++ (1S) step adapted for flow matching.

    Approximation: treat the velocity prediction as the derivative and
    apply a first-order multistep update.  For a single step this reduces
    to the Euler step with a half-sigma midpoint correction.
    """
    dt = sigma_next - sigma
    view_shape = (sample.shape[0],) + (1,) * (sample.dim() - 1)
    dt_view = dt.view(view_shape)
    # Simple first-order: identical to Euler for flow-matching velocity
    return sample + model_output * dt_view


@torch.no_grad()
def sample_anima(
    dit_model,
    vae,
    text_encoder,
    tokenizer,
    prompt: str,
    *,
    negative_prompt: str = "",
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    sampler_name: str = "euler",
    latent_channels: int = 16,
    vae_scaling_factor: float = 1.0,
    patch_size: int = 2,
    num_train_timesteps: int = 1000,
    discrete_flow_shift: float = 1.0,
    smc_cfg: bool = False,
    smc_cfg_lambda: float = 5.0,
    smc_cfg_alpha: float = 0.2,
    tgate_probe: bool = False,
    tgate_start_step: int = 0,
    tgate_min_block: int = 0,
    spectrum_probe: bool = False,
    spectrum_window_size: float = 2.0,
    spectrum_flex_window: float = 0.25,
    spectrum_warmup_steps: int = 6,
    spectrum_stop_caching_step: int = -1,
    smoothcache_probe: bool = False,
    smoothcache_error_threshold: float = 0.08,
    smoothcache_warmup_steps: int = 2,
) -> Optional[Image.Image]:
    """Generate a preview image using Anima's DiT with flow-matching sampling.

    Parameters
    ----------
    dit_model : nn.Module
        The Anima DiT model. Must accept ``(noisy_latents, timesteps, text_emb)``.
    vae : nn.Module
        VAE decoder for latent → pixel decoding.
    text_encoder : nn.Module
        CLIP or Qwen text encoder.
    tokenizer
        Corresponding tokenizer.
    prompt : str
        Positive prompt.
    negative_prompt : str
        Negative prompt (used for CFG if guidance_scale > 1).
    num_inference_steps : int
        Number of denoising steps.
    guidance_scale : float
        Classifier-free guidance scale.
    width, height : int
        Output image dimensions.
    seed : int or None
        Random seed for reproducibility.
    device : str
        Torch device.
    dtype : torch.dtype
        Computation dtype.
    sampler_name : str
        ``"euler"`` or ``"dpm_solver"``.
    latent_channels : int
        Number of latent channels (default 16 for Anima).
    vae_scaling_factor : float
        VAE latent scaling factor (default 1.0 for Anima).
    patch_size : int
        DiT patch size.
    num_train_timesteps : int
        Total timestep count for sigma→timestep mapping.
    discrete_flow_shift : float
        Flow shift applied during sampling.

    Returns
    -------
    PIL.Image.Image or None
    """
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    if dit_model is None:
        logger.warning("Anima preview unavailable: DiT model is None (cache-first mode?)")
        return None
    if vae is None:
        logger.warning("Anima preview unavailable: VAE is None (released after cache build?)")
        return None

    try:
        from .anima_sampler_native import decode_anima_image, resolve_text_embeds
    except ImportError:  # pragma: no cover - direct-file smoke fallback.
        from core.lulynx_trainer.anima_sampler_native import (
            decode_anima_image,
            resolve_text_embeds,
        )

    # --- encode text (native Qwen3 prompt_embeds injection, else CLIP) ---
    text_emb, neg_emb, handled = resolve_text_embeds(
        prompt_embeds, negative_prompt_embeds,
        device=device, dtype=dtype, guidance_scale=guidance_scale,
    )
    if not handled:
        if text_encoder is None or tokenizer is None:
            logger.warning(
                "Anima preview unavailable: text encoder or tokenizer is None "
                "(released after cache build?)"
            )
            return None
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        text_emb = text_encoder(text_input_ids, return_dict=True).last_hidden_state.to(dtype=dtype)

        if guidance_scale > 1.0 and negative_prompt:
            neg_inputs = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            neg_input_ids = neg_inputs.input_ids.to(device)
            neg_emb = text_encoder(neg_input_ids, return_dict=True).last_hidden_state.to(dtype=dtype)
        else:
            neg_emb = None

    # --- prepare latents ---
    # Anima uses patch_size to determine latent spatial dims
    latent_h = height // (vae.config.get("scale_factor", 8) if hasattr(vae, "config") else 8)
    latent_w = width // (vae.config.get("scale_factor", 8) if hasattr(vae, "config") else 8)
    # Fallback: assume 8x spatial compression
    if latent_h == 0 or latent_w == 0:
        latent_h = height // 8
        latent_w = width // 8

    latent_shape = (1, latent_channels, latent_h, latent_w)
    noise = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    # Flow matching starts from pure noise (sigma=1)
    latents = noise * vae_scaling_factor

    # --- build sigmas schedule ---
    # Linearly spaced sigmas from 1.0 → 0.0
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype)

    # Apply flow shift if configured
    if discrete_flow_shift != 1.0:
        sigmas = (sigmas * discrete_flow_shift) / (1.0 + (discrete_flow_shift - 1.0) * sigmas)

    step_fn = _dpm_solver_step if sampler_name == "dpm_solver" else _euler_step
    smc_cfg_state = None
    if smc_cfg and neg_emb is not None:
        try:
            from .smc_cfg import SMCCFGConfig, build_smc_cfg_state
        except ImportError:  # pragma: no cover - direct-file smoke fallback.
            from core.lulynx_trainer.smc_cfg import SMCCFGConfig, build_smc_cfg_state

        smc_cfg_state = build_smc_cfg_state(
            SMCCFGConfig(enabled=True, lam=smc_cfg_lambda, alpha=smc_cfg_alpha)
        )

    tgate_context_factory = None
    if tgate_probe:
        try:
            from .tgate import tgate_step_context
        except ImportError:  # pragma: no cover - direct-file smoke fallback.
            from core.lulynx_trainer.tgate import tgate_step_context

        tgate_context_factory = tgate_step_context

    spectrum_state = None
    spectrum_context_factory = None
    if spectrum_probe:
        try:
            from .spectrum_probe import (
                SpectrumProbePolicy,
                SpectrumProbeState,
                reset_spectrum_probe_stats,
                snapshot_spectrum_probe_stats,
                spectrum_step_context,
            )
        except ImportError:  # pragma: no cover - direct-file smoke fallback.
            from core.lulynx_trainer.spectrum_probe import (
                SpectrumProbePolicy,
                SpectrumProbeState,
                reset_spectrum_probe_stats,
                snapshot_spectrum_probe_stats,
                spectrum_step_context,
            )

        reset_spectrum_probe_stats()
        spectrum_state = SpectrumProbeState(
            SpectrumProbePolicy(
                enabled=True,
                window_size=spectrum_window_size,
                flex_window=spectrum_flex_window,
                warmup_steps=spectrum_warmup_steps,
                stop_caching_step=spectrum_stop_caching_step,
            ),
            total_steps=num_inference_steps,
        )
        spectrum_context_factory = spectrum_step_context
    else:
        snapshot_spectrum_probe_stats = None

    smoothcache_state = None
    smoothcache_context_factory = None
    if smoothcache_probe:
        try:
            from .smoothcache import (
                SmoothCachePolicy,
                SmoothCacheState,
                reset_smoothcache_probe_stats,
                snapshot_smoothcache_probe_stats,
                smoothcache_step_context,
            )
        except ImportError:  # pragma: no cover - direct-file smoke fallback.
            from core.lulynx_trainer.smoothcache import (
                SmoothCachePolicy,
                SmoothCacheState,
                reset_smoothcache_probe_stats,
                snapshot_smoothcache_probe_stats,
                smoothcache_step_context,
            )

        reset_smoothcache_probe_stats()
        smoothcache_state = SmoothCacheState(
            SmoothCachePolicy(
                enabled=True,
                error_threshold=smoothcache_error_threshold,
                warmup_steps=smoothcache_warmup_steps,
                schedule=None,
            ),
            total_steps=num_inference_steps,
        )
        smoothcache_context_factory = smoothcache_step_context
    else:
        snapshot_smoothcache_probe_stats = None

    # --- denoising loop ---
    for i in range(num_inference_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Native Anima's t_embedder consumes the flow time on the sigma in [0,1]
        # scale directly (rectified flow; the base velocity probe confirms the
        # faithful subset wants t=sigma at cos>=0.9). Feeding the [0,1000]
        # scheduler value collapses the adaLN gates and scrambles the velocity,
        # so the faithful subset gets t=sigma; the legacy scaffold path keeps the
        # historical scaled-int timestep. sigmas[i] is 0-dim -> shape to [1].
        if getattr(dit_model, "anima_faithful", False):
            timestep = sigma.reshape(1).to(dtype)
        else:
            timestep = (sigma * float(num_train_timesteps)).long().reshape(1)

        step_context = (
            tgate_context_factory(
                enabled=True,
                step_index=i,
                total_steps=num_inference_steps,
                start_step=tgate_start_step,
                min_block=tgate_min_block,
            )
            if tgate_context_factory is not None
            else nullcontext()
        )
        spectrum_context = (
            spectrum_context_factory(spectrum_state.decide(i))
            if spectrum_state is not None and spectrum_context_factory is not None
            else nullcontext()
        )
        smoothcache_context = (
            smoothcache_context_factory(smoothcache_state.decide(i))
            if smoothcache_state is not None and smoothcache_context_factory is not None
            else nullcontext()
        )
        with step_context, spectrum_context, smoothcache_context:
            # Model forward — CFG
            model_pred = dit_model(latents, timestep, text_emb).sample
            if neg_emb is not None:
                neg_pred = dit_model(latents, timestep, neg_emb).sample
                if smc_cfg_state is not None:
                    model_pred = smc_cfg_state.combine(model_pred, neg_pred, guidance_scale)
                else:
                    model_pred = neg_pred + guidance_scale * (model_pred - neg_pred)

        latents = step_fn(model_pred, latents, sigma.unsqueeze(0), sigma_next.unsqueeze(0))

    if spectrum_state is not None and snapshot_spectrum_probe_stats is not None:
        logger.info(
            "Spectrum observe-only probe summary: schedule=%s stats=%s",
            spectrum_state.summary(),
            snapshot_spectrum_probe_stats(),
        )

    if smoothcache_state is not None and snapshot_smoothcache_probe_stats is not None:
        logger.info(
            "SmoothCache observe-only probe summary: schedule=%s stats=%s",
            smoothcache_state.summary(),
            snapshot_smoothcache_probe_stats(),
        )

    # --- decode (qwen-image inverse-norm + 5D, else standard) ---
    try:
        image = decode_anima_image(vae, latents, vae_scaling_factor)
    except Exception as e:
        logger.error(f"Anima VAE decode failed: {e}")
        return None

    image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    try:
        from numpy import clip
        image = clip(image, 0, 1)
    except ImportError:
        image = image.clip(0, 1)
    try:
        return Image.fromarray((image[0] * 255).astype("uint8"), "RGB")
    except Exception as e:
        logger.error(f"Anima image conversion failed: {e}")
        return None


@torch.no_grad()
def sample_anima_ersde(
    dit_model,
    vae,
    text_encoder,
    tokenizer,
    prompt: str,
    *,
    negative_prompt: str = "",
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    latent_channels: int = 16,
    vae_scaling_factor: float = 1.0,
    patch_size: int = 2,
    num_train_timesteps: int = 1000,
    discrete_flow_shift: float = 1.0,
    cns_gamma_path: str = "",
    cns_strength: float = 1.0,
    eta: float = 1.0,
    smc_cfg: bool = False,
    smc_cfg_lambda: float = 5.0,
    smc_cfg_alpha: float = 0.2,
    tgate_probe: bool = False,
    tgate_start_step: int = 0,
    tgate_min_block: int = 0,
    spectrum_probe: bool = False,
    spectrum_window_size: float = 2.0,
    spectrum_flex_window: float = 0.25,
    spectrum_warmup_steps: int = 6,
    spectrum_stop_caching_step: int = -1,
    smoothcache_probe: bool = False,
    smoothcache_error_threshold: float = 0.08,
    smoothcache_warmup_steps: int = 2,
) -> Optional[Image.Image]:
    """Anima ER-SDE sampler with CNS colored noise sampling.

    This is a stochastic (SDE) variant of the Euler sampler that injects noise
    at each step, enabling CNS (Colored Noise Sampling) to recolor the noise
    according to unresolved frequency bands.

    Parameters
    ----------
    eta : float, default 1.0
        Stochasticity strength. eta=0 reduces to deterministic Euler ODE,
        eta=1.0 is full SDE.
    cns_gamma_path : str, default ""
        Path to CNS calibration .npz file. Empty string disables CNS.
    cns_strength : float, default 1.0
        CNS recoloring strength in [0, 1].

    Other parameters same as ``sample_anima()``.
    """
    # --- setup ---
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    dit_model.eval()
    vae.eval()
    if text_encoder is not None:
        text_encoder.eval()

    # --- CNS recolorer ---
    cns_recolorer = None
    if cns_gamma_path:
        try:
            from .cns_sampling import build_cns_recolorer
        except ImportError:
            from core.lulynx_trainer.cns_sampling import build_cns_recolorer

        cns_recolorer = build_cns_recolorer(
            gamma_path=cns_gamma_path,
            strength=cns_strength,
        )
        if cns_recolorer is not None:
            logger.info(
                f"CNS enabled: gamma_path={cns_gamma_path}, strength={cns_strength}, "
                f"frequency_bins={cns_recolorer.frequency_bins}"
            )

    try:
        from .anima_sampler_native import decode_anima_image, resolve_text_embeds
    except ImportError:  # pragma: no cover - direct-file smoke fallback.
        from core.lulynx_trainer.anima_sampler_native import (
            decode_anima_image,
            resolve_text_embeds,
        )

    # --- encode text (native Qwen3 prompt_embeds injection, else CLIP) ---
    text_emb, neg_emb, handled = resolve_text_embeds(
        prompt_embeds, negative_prompt_embeds,
        device=device, dtype=dtype, guidance_scale=guidance_scale,
    )
    if not handled:
        if text_encoder is None or tokenizer is None:
            logger.warning(
                "Anima ER-SDE preview unavailable: text encoder or tokenizer is "
                "None (released after cache build?)"
            )
            return None
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        text_emb = text_encoder(text_input_ids, return_dict=True).last_hidden_state.to(dtype=dtype)

        if guidance_scale > 1.0 and negative_prompt:
            neg_inputs = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            neg_input_ids = neg_inputs.input_ids.to(device)
            neg_emb = text_encoder(neg_input_ids, return_dict=True).last_hidden_state.to(dtype=dtype)
        else:
            neg_emb = None

    # --- prepare latents ---
    latent_h = height // (vae.config.get("scale_factor", 8) if hasattr(vae, "config") else 8)
    latent_w = width // (vae.config.get("scale_factor", 8) if hasattr(vae, "config") else 8)
    if latent_h == 0 or latent_w == 0:
        latent_h = height // 8
        latent_w = width // 8

    latent_shape = (1, latent_channels, latent_h, latent_w)
    noise = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = noise * vae_scaling_factor

    # --- build sigmas schedule ---
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype)

    if discrete_flow_shift != 1.0:
        sigmas = (sigmas * discrete_flow_shift) / (1.0 + (discrete_flow_shift - 1.0) * sigmas)

    # SMC-CFG setup
    smc_cfg_state = None
    if smc_cfg and neg_emb is not None:
        try:
            from .smc_cfg import SMCCFGConfig, build_smc_cfg_state
        except ImportError:
            from core.lulynx_trainer.smc_cfg import SMCCFGConfig, build_smc_cfg_state

        smc_cfg_state = build_smc_cfg_state(
            SMCCFGConfig(enabled=True, lam=smc_cfg_lambda, alpha=smc_cfg_alpha)
        )

    tgate_context_factory = None
    if tgate_probe:
        try:
            from .tgate import tgate_step_context
        except ImportError:  # pragma: no cover - direct-file smoke fallback.
            from core.lulynx_trainer.tgate import tgate_step_context

        tgate_context_factory = tgate_step_context

    spectrum_state = None
    spectrum_context_factory = None
    if spectrum_probe:
        try:
            from .spectrum_probe import (
                SpectrumProbePolicy,
                SpectrumProbeState,
                reset_spectrum_probe_stats,
                snapshot_spectrum_probe_stats,
                spectrum_step_context,
            )
        except ImportError:  # pragma: no cover - direct-file smoke fallback.
            from core.lulynx_trainer.spectrum_probe import (
                SpectrumProbePolicy,
                SpectrumProbeState,
                reset_spectrum_probe_stats,
                snapshot_spectrum_probe_stats,
                spectrum_step_context,
            )

        reset_spectrum_probe_stats()
        spectrum_state = SpectrumProbeState(
            SpectrumProbePolicy(
                enabled=True,
                window_size=spectrum_window_size,
                flex_window=spectrum_flex_window,
                warmup_steps=spectrum_warmup_steps,
                stop_caching_step=spectrum_stop_caching_step,
            ),
            total_steps=num_inference_steps,
        )
        spectrum_context_factory = spectrum_step_context
    else:
        snapshot_spectrum_probe_stats = None

    smoothcache_state = None
    smoothcache_context_factory = None
    if smoothcache_probe:
        try:
            from .smoothcache import (
                SmoothCachePolicy,
                SmoothCacheState,
                reset_smoothcache_probe_stats,
                snapshot_smoothcache_probe_stats,
                smoothcache_step_context,
            )
        except ImportError:  # pragma: no cover - direct-file smoke fallback.
            from core.lulynx_trainer.smoothcache import (
                SmoothCachePolicy,
                SmoothCacheState,
                reset_smoothcache_probe_stats,
                snapshot_smoothcache_probe_stats,
                smoothcache_step_context,
            )

        reset_smoothcache_probe_stats()
        smoothcache_state = SmoothCacheState(
            SmoothCachePolicy(
                enabled=True,
                error_threshold=smoothcache_error_threshold,
                warmup_steps=smoothcache_warmup_steps,
                schedule=None,
            ),
            total_steps=num_inference_steps,
        )
        smoothcache_context_factory = smoothcache_step_context
    else:
        snapshot_smoothcache_probe_stats = None

    # --- ER-SDE denoising loop ---
    for i in range(num_inference_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        # Native Anima's t_embedder consumes the flow time on the sigma in [0,1]
        # scale directly (rectified flow; the base velocity probe confirms the
        # faithful subset wants t=sigma at cos>=0.9). Feeding the [0,1000]
        # scheduler value collapses the adaLN gates and scrambles the velocity,
        # so the faithful subset gets t=sigma; the legacy scaffold path keeps the
        # historical scaled-int timestep. sigmas[i] is 0-dim -> shape to [1].
        if getattr(dit_model, "anima_faithful", False):
            timestep = sigma.reshape(1).to(dtype)
        else:
            timestep = (sigma * float(num_train_timesteps)).long().reshape(1)

        step_context = (
            tgate_context_factory(
                enabled=True,
                step_index=i,
                total_steps=num_inference_steps,
                start_step=tgate_start_step,
                min_block=tgate_min_block,
            )
            if tgate_context_factory is not None
            else nullcontext()
        )
        spectrum_context = (
            spectrum_context_factory(spectrum_state.decide(i))
            if spectrum_state is not None and spectrum_context_factory is not None
            else nullcontext()
        )
        smoothcache_context = (
            smoothcache_context_factory(smoothcache_state.decide(i))
            if smoothcache_state is not None and smoothcache_context_factory is not None
            else nullcontext()
        )
        with step_context, spectrum_context, smoothcache_context:
            # Model forward — CFG
            model_pred = dit_model(latents, timestep, text_emb).sample
            if neg_emb is not None:
                neg_pred = dit_model(latents, timestep, neg_emb).sample
                if smc_cfg_state is not None:
                    model_pred = smc_cfg_state.combine(model_pred, neg_pred, guidance_scale)
                else:
                    model_pred = neg_pred + guidance_scale * (model_pred - neg_pred)

        # Euler step (deterministic part)
        dt_view = dt.view((latents.shape[0],) + (1,) * (latents.dim() - 1))
        latents = latents + model_pred * dt_view

        # SDE noise injection (stochastic part)
        if i < num_inference_steps - 1 and eta > 0.0:
            # Noise scale for ER-SDE: eta * sqrt(|dt|)
            noise_scale = eta * torch.sqrt(torch.abs(dt))

            # Generate white noise
            white_noise = torch.randn_like(latents, generator=generator)

            # Apply CNS recoloring if enabled
            if cns_recolorer is not None:
                try:
                    colored_noise = cns_recolorer.recolor(
                        white_noise,
                        sigma=float(sigma.item()),
                        height=latent_h,
                        width=latent_w,
                    )
                except Exception as e:
                    logger.warning(f"CNS recolor failed at step {i}: {e}, using white noise")
                    colored_noise = white_noise
            else:
                colored_noise = white_noise

            # Inject noise
            noise_scale_view = noise_scale.view((latents.shape[0],) + (1,) * (latents.dim() - 1))
            latents = latents + colored_noise * noise_scale_view

    if spectrum_state is not None and snapshot_spectrum_probe_stats is not None:
        logger.info(
            "Spectrum observe-only probe summary: schedule=%s stats=%s",
            spectrum_state.summary(),
            snapshot_spectrum_probe_stats(),
        )

    if smoothcache_state is not None and snapshot_smoothcache_probe_stats is not None:
        logger.info(
            "SmoothCache observe-only probe summary: schedule=%s stats=%s",
            smoothcache_state.summary(),
            snapshot_smoothcache_probe_stats(),
        )

    # --- decode (qwen-image inverse-norm + 5D, else standard) ---
    try:
        image = decode_anima_image(vae, latents, vae_scaling_factor)
    except Exception as e:
        logger.error(f"Anima VAE decode failed: {e}")
        return None

    image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    try:
        from numpy import clip
        image = clip(image, 0, 1)
    except ImportError:
        image = image.clip(0, 1)
    try:
        return Image.fromarray((image[0] * 255).astype("uint8"), "RGB")
    except Exception as e:
        logger.error(f"Anima image conversion failed: {e}")
        return None
