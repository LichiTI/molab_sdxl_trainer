"""
Noise Utilities — Warehouse implementations of advanced noise generation.

All functions are based on public mathematical formulas and PyTorch APIs.
No AGPL-licensed code was referenced.

Functions:
  - pyramid_noise_like: Multi-resolution (pyramid) noise
  - apply_adaptive_noise_scale: Scale noise offset by latent magnitude
  - apply_ip_noise: IP-Noise gamma injection
  - rescale_zero_terminal_snr: Rescale scheduler for zero terminal SNR
"""

import math
import torch
import logging

logger = logging.getLogger(__name__)


def pyramid_noise_like(
    noise: torch.Tensor,
    iterations: int = 6,
    discount: float = 0.4,
) -> torch.Tensor:
    """Generate multi-resolution (pyramid) noise.

    Adds progressively downsampled and scaled noise layers on top of
    the input noise.  This gives the model exposure to noise at multiple
    spatial frequencies, improving detail preservation.

    Args:
        noise: Base noise tensor (B, C, H, W).
        iterations: Number of pyramid levels (including the base).
        discount: Scaling factor for each successive downsampled layer.

    Returns:
        Modified noise tensor with same shape as input.
    """
    if iterations <= 1:
        return noise

    b, c, h, w = noise.shape
    cumulative = noise.clone()

    for i in range(1, iterations):
        # Downsample by factor 2^i
        scale = 2 ** i
        nh, nw = h // scale, w // scale
        if nh < 1 or nw < 1:
            break

        # Generate noise at this resolution
        low_res = torch.randn(b, c, nh, nw, device=noise.device, dtype=noise.dtype)

        # Upsample back to original size using nearest-neighbor
        upsampled = torch.nn.functional.interpolate(
            low_res, size=(h, w), mode="nearest",
        )

        # Scale and accumulate
        cumulative += upsampled * (discount ** i)

    return cumulative


def apply_adaptive_noise_scale(
    noise_offset: float,
    latents: torch.Tensor,
    adaptive_scale: float,
) -> float:
    """Scale noise offset by the absolute mean of latents.

    When latents have large magnitudes, a fixed noise offset may be too
    small relative to the signal.  This adjusts the offset proportionally.

    Args:
        noise_offset: Base noise offset value.
        latents: Latent tensor from VAE encoding.
        adaptive_scale: Multiplier for the adaptive component.

    Returns:
        Adjusted noise offset value.
    """
    if adaptive_scale <= 0:
        return noise_offset
    mean_abs = latents.abs().mean().item()
    return noise_offset * mean_abs * adaptive_scale


def apply_ip_noise(
    noise: torch.Tensor,
    gamma: float | torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Apply IP-Noise (Image-Propagation noise) based on timesteps.

    IP-Noise adds noise proportional to the signal strength at each
    timestep, controlled by gamma.  At high noise levels (low alpha),
    less IP noise is added; at low noise levels (high alpha), more is added.

    Args:
        noise: Base noise tensor.
        gamma: IP-Noise strength multiplier.
        timesteps: Timestep indices (B,).
        alphas_cumprod: Cumulative alpha product schedule.

    Returns:
        Modified noise tensor.
    """
    if isinstance(gamma, torch.Tensor):
        gamma_tensor = gamma.to(device=noise.device, dtype=noise.dtype)
        if torch.all(gamma_tensor <= 0):
            return noise
    elif gamma <= 0:
        return noise
    else:
        gamma_tensor = torch.tensor(float(gamma), device=noise.device, dtype=noise.dtype)

    alpha_t = alphas_cumprod.to(noise.device)[timesteps]
    # Shape: (B,) -> (B, 1, 1, 1) for broadcasting
    while alpha_t.dim() < noise.dim():
        alpha_t = alpha_t.unsqueeze(-1)

    # IP noise is stronger when alpha is high (less diffusion noise)
    while gamma_tensor.dim() < noise.dim():
        gamma_tensor = gamma_tensor.unsqueeze(-1)

    ip_noise = gamma_tensor * alpha_t * torch.randn_like(noise)
    return noise + ip_noise


def rescale_zero_terminal_snr(scheduler) -> None:
    """Rescale the noise scheduler's alphas_cumprod for zero terminal SNR.

    This modifies the scheduler in-place so that alphas_cumprod[-1] == 0,
    which ensures the signal-to-noise ratio at the final timestep is zero.
    This prevents the model from having to predict clean images at extreme
    noise levels where it's essentially impossible.

    Based on the "Zero Terminal SNR" technique (arXiv:2305.08891).

    Args:
        scheduler: A diffusers noise scheduler with alphas_cumprod attribute.
    """
    if not hasattr(scheduler, "alphas_cumprod"):
        logger.warning("[ZeroTerminalSNR] Scheduler has no alphas_cumprod, skipping")
        return

    alphas_cumprod = scheduler.alphas_cumprod
    device = alphas_cumprod.device
    dtype = alphas_cumprod.dtype

    # Convert to float for computation
    alphas = alphas_cumprod.float()

    # Compute sqrt(alpha) and sqrt(1-alpha) for SNR calculation
    sqrt_alphas = alphas.sqrt()
    sqrt_one_minus_alphas = (1.0 - alphas).sqrt()

    # Current SNR in dB
    snr = (sqrt_alphas / sqrt_one_minus_alphas) ** 2
    snr_db = 10.0 * torch.log10(snr + 1e-10)

    # Target: linearly interpolate SNR from current first to -20 dB at the end
    # This ensures terminal SNR approaches zero
    terminal_snr_db = -20.0
    target_snr_db = torch.linspace(
        snr_db[0].item(), terminal_snr_db, len(snr_db), device=device,
    )
    target_snr = 10.0 ** (target_snr_db / 10.0)

    # Convert target SNR back to alphas_cumprod
    # SNR = alpha / (1 - alpha) => alpha = SNR / (1 + SNR)
    new_alphas = target_snr / (1.0 + target_snr)

    # Ensure monotonicity (alphas should decrease)
    for i in range(1, len(new_alphas)):
        if new_alphas[i] > new_alphas[i - 1]:
            new_alphas[i] = new_alphas[i - 1]

    # Clamp final value to exactly zero
    new_alphas[-1] = 0.0

    # Write back to scheduler
    scheduler.alphas_cumprod = new_alphas.to(dtype=dtype, device=device)

    logger.info(
        f"[ZeroTerminalSNR] Rescaled alphas_cumprod: "
        f"first={new_alphas[0]:.6f}, last={new_alphas[-1]:.6f}, "
        f"SNR range=[{target_snr_db[0]:.1f}, {target_snr_db[-1]:.1f}] dB"
    )

