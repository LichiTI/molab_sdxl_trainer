"""Newbie DiT sampler for training previews.

Implements flow-matching sampling for Newbie models.  Newbie uses a DiT
architecture with dual text encoders (CLIP + T5/Jina) and CLIP pooled
features, with 16 latent channels.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _euler_step(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Single Euler step for flow-matching ODE."""
    dt = sigma_next - sigma
    view_shape = (sample.shape[0],) + (1,) * (sample.dim() - 1)
    dt_view = dt.view(view_shape)
    return sample + model_output * dt_view


@torch.no_grad()
def sample_newbie(
    dit_model,
    vae,
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    prompt: str,
    *,
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    sampler_name: str = "euler",
    latent_channels: int = 16,
    vae_scaling_factor: float = 0.3611,
    latent_scale_factor: int = 0,
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
    """Generate a preview image using Newbie's DiT with flow-matching sampling.

    Parameters
    ----------
    dit_model : nn.Module
        The Newbie DiT model. Must accept
        ``(noisy_latents, timesteps, text_emb, pooled_emb)``.
    vae : nn.Module
        VAE decoder.
    text_encoder_1 : nn.Module
        CLIP text encoder (with pooled output).
    text_encoder_2 : nn.Module
        T5 / Jina text encoder (sequence output only).
    tokenizer_1, tokenizer_2
        Corresponding tokenizers.
    prompt : str
        Positive prompt.
    negative_prompt : str
        Negative prompt for CFG.
    num_inference_steps : int
        Denoising step count.
    guidance_scale : float
        CFG scale.
    width, height : int
        Output dimensions.
    seed : int or None
        Random seed.
    device : str
        Torch device.
    dtype : torch.dtype
        Computation dtype.
    sampler_name : str
        Sampling strategy (``"euler"`` or ``"dpm_solver"``).
    latent_channels : int
        Latent channel count (default 16 for Newbie).
    vae_scaling_factor : float
        VAE scaling factor (default 0.3611 for Newbie).
    latent_scale_factor : int
        Optional latent downscale override.  Defaults to VAE config/default and
        is used by static TensorRT routes to keep sampler latents aligned with
        the validated engine shape.
    num_train_timesteps : int
        Timestep count for sigma→t mapping.
    discrete_flow_shift : float
        Flow shift parameter.

    Returns
    -------
    PIL.Image.Image or None
    """
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    if dit_model is None:
        logger.warning("Newbie preview unavailable: DiT model is None (cache-first mode?)")
        return None
    if vae is None:
        logger.warning("Newbie preview unavailable: VAE is None (released after cache build?)")
        return None
    if text_encoder_1 is None or tokenizer_1 is None:
        logger.warning(
            "Newbie preview unavailable: text encoder 1 or tokenizer 1 is None "
            "(released after cache build?)"
        )
        return None
    if text_encoder_2 is None or tokenizer_2 is None:
        logger.warning(
            "Newbie preview unavailable: text encoder 2 or tokenizer 2 is None "
            "(released after cache build?)"
        )
        return None

    # --- encode text ---
    # Encoder 1: CLIP (with pooled features)
    text_inputs_1 = tokenizer_1(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    )
    te1_out = text_encoder_1(text_inputs_1.input_ids.to(device), return_dict=True)
    text_emb_1 = te1_out.last_hidden_state.to(dtype=dtype)
    pooled_emb = te1_out.text_embeds.to(dtype=dtype) if hasattr(te1_out, "text_embeds") else text_emb_1[:, 0]

    # Encoder 2: T5 / Jina (sequence output only)
    text_inputs_2 = tokenizer_2(
        prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
    )
    te2_out = text_encoder_2(text_inputs_2.input_ids.to(device), return_dict=True)
    text_emb_2 = te2_out.last_hidden_state.to(dtype=dtype)

    # Concatenate text embeddings
    text_emb = torch.cat([text_emb_1, text_emb_2], dim=1)

    # Negative embeddings for CFG
    if guidance_scale > 1.0 and negative_prompt:
        neg_inputs_1 = tokenizer_1(
            negative_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
        )
        neg_te1 = text_encoder_1(neg_inputs_1.input_ids.to(device), return_dict=True)
        neg_emb_1 = neg_te1.last_hidden_state.to(dtype=dtype)
        neg_pooled = neg_te1.text_embeds.to(dtype=dtype) if hasattr(neg_te1, "text_embeds") else neg_emb_1[:, 0]

        neg_inputs_2 = tokenizer_2(
            negative_prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
        )
        neg_te2 = text_encoder_2(neg_inputs_2.input_ids.to(device), return_dict=True)
        neg_emb_2 = neg_te2.last_hidden_state.to(dtype=dtype)

        neg_emb = torch.cat([neg_emb_1, neg_emb_2], dim=1)
    else:
        neg_emb = None
        neg_pooled = None

    # --- prepare latents ---
    scale_factor = int(latent_scale_factor or 0)
    if scale_factor <= 0:
        scale_factor = 8
    if int(latent_scale_factor or 0) <= 0 and hasattr(vae, "config") and hasattr(vae.config, "scale_factor"):
        scale_factor = vae.config.scale_factor
    latent_h = height // scale_factor
    latent_w = width // scale_factor
    latent_shape = (1, latent_channels, latent_h, latent_w)
    noise = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = noise * vae_scaling_factor

    # --- build sigmas schedule ---
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype)
    if discrete_flow_shift != 1.0:
        sigmas = (sigmas * discrete_flow_shift) / (1.0 + (discrete_flow_shift - 1.0) * sigmas)

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
        timestep = (sigma * float(num_train_timesteps)).long()

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
            # Model forward with pooled features
            model_pred = dit_model(
                latents, timestep, text_emb, pooled_emb=pooled_emb,
            )
            # Handle both .sample and direct tensor returns
            if hasattr(model_pred, "sample"):
                model_pred = model_pred.sample

            # CFG
            if neg_emb is not None:
                neg_pred = dit_model(
                    latents, timestep, neg_emb, pooled_emb=neg_pooled,
                )
                if hasattr(neg_pred, "sample"):
                    neg_pred = neg_pred.sample
                if smc_cfg_state is not None:
                    model_pred = smc_cfg_state.combine(model_pred, neg_pred, guidance_scale)
                else:
                    model_pred = neg_pred + guidance_scale * (model_pred - neg_pred)

        latents = _euler_step(model_pred, latents, sigma.unsqueeze(0), sigma_next.unsqueeze(0))

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

    # --- decode ---
    latents = latents / vae_scaling_factor
    try:
        image = vae.decode(latents.to(vae.dtype)).sample
    except Exception as e:
        logger.error(f"Newbie VAE decode failed: {e}")
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
        logger.error(f"Newbie image conversion failed: {e}")
        return None


@torch.no_grad()
def sample_newbie_ersde(
    dit_model,
    vae,
    text_encoder,
    tokenizer,
    prompt: str,
    *,
    negative_prompt: str = "",
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
) -> Optional[Image.Image]:
    """Newbie ER-SDE sampler with CNS colored noise sampling.

    This is a stochastic (SDE) variant for Newbie models (e.g., Hunyuan DiT).
    Parameters same as ``sample_anima_ersde()``.
    """
    # --- setup ---
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    dit_model.eval()
    vae.eval()
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

    # --- encode text ---
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    # Newbie needs both hidden states and pooled embeddings
    encoder_output = text_encoder(text_input_ids, return_dict=True)
    text_emb = encoder_output.last_hidden_state.to(dtype=dtype)
    pooled_emb = encoder_output.pooler_output.to(dtype=dtype) if hasattr(encoder_output, "pooler_output") else None

    if guidance_scale > 1.0 and negative_prompt:
        neg_inputs = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        neg_input_ids = neg_inputs.input_ids.to(device)
        neg_output = text_encoder(neg_input_ids, return_dict=True)
        neg_emb = neg_output.last_hidden_state.to(dtype=dtype)
        neg_pooled = neg_output.pooler_output.to(dtype=dtype) if hasattr(neg_output, "pooler_output") else None
    else:
        neg_emb = None
        neg_pooled = None

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

    # --- ER-SDE denoising loop ---
    for i in range(num_inference_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        timestep = (sigma * float(num_train_timesteps)).long()

        # Model forward with pooled features
        model_pred = dit_model(latents, timestep, text_emb, pooled_emb=pooled_emb)
        if hasattr(model_pred, "sample"):
            model_pred = model_pred.sample

        # CFG
        if neg_emb is not None:
            neg_pred = dit_model(latents, timestep, neg_emb, pooled_emb=neg_pooled)
            if hasattr(neg_pred, "sample"):
                neg_pred = neg_pred.sample
            if smc_cfg_state is not None:
                model_pred = smc_cfg_state.combine(model_pred, neg_pred, guidance_scale)
            else:
                model_pred = neg_pred + guidance_scale * (model_pred - neg_pred)

        # Euler step (deterministic part)
        dt_view = dt.view((latents.shape[0],) + (1,) * (latents.dim() - 1))
        latents = latents + model_pred * dt_view

        # SDE noise injection (stochastic part)
        if i < num_inference_steps - 1 and eta > 0.0:
            noise_scale = eta * torch.sqrt(torch.abs(dt))

            white_noise = torch.randn_like(latents, generator=generator)

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

            noise_scale_view = noise_scale.view((latents.shape[0],) + (1,) * (latents.dim() - 1))
            latents = latents + colored_noise * noise_scale_view

    # --- decode ---
    latents = latents / vae_scaling_factor
    try:
        image = vae.decode(latents.to(vae.dtype)).sample
    except Exception as e:
        logger.error(f"Newbie VAE decode failed: {e}")
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
        logger.error(f"Newbie image conversion failed: {e}")
        return None

