# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Default-off opt-in runtime seam for the P4 DP-DMD / Turbo and SPD reserves.

The P4 primitives (``dp_dmd_turbo`` distillation losses, ``spd_inference``
multi-resolution planning) were proven in isolation.  This module is the *real
integration* step the roadmap tracked as "separate route decisions": seams that
actually drive those primitives end-to-end on a real (tiny) fixture when a caller
opts in, while staying a no-op / parity when off.

Two opt-in surfaces, both default-off:

* **SPD inference** -- :func:`run_spd_multiresolution_denoise` wraps a caller's
  per-step ``denoise_fn`` and, when enabled, runs it across the SPD
  multi-resolution schedule (downscaled early levels, upscaled to base).  When
  disabled it runs every step at base resolution, which is bitwise-identical to
  the caller's own plain loop -- and an SPD config whose scales are all ``1.0`` is
  bitwise-identical to the disabled path.
* **DP-DMD distillation** -- :func:`compose_dp_dmd_turbo_loss` composes the
  DP-DMD/Turbo distillation losses onto a base training loss.  When disabled it
  returns the base loss unchanged (no added terms, parity); when enabled it adds
  the real consistency / diversity / fake-critic terms and gradients flow.

Honesty red-lines: installing either seam is caller/opt-in driven and is not
auto-consumed by the production sampler or training loop.  The readiness report
keeps ``runtime_activation_enabled`` / ``request_fields_emitted`` /
``trainer_wiring_allowed`` / ``training_path_enabled`` / ``promotion_ready`` all
``False``.  Clean-room Lulynx module.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch

try:  # package import
    from .dp_dmd_turbo import DpDmdTurboConfig, compute_dp_dmd_turbo_loss
    from .spd_inference import (
        SpdInferenceConfig,
        build_spd_inference_plan,
        resize_latents_for_spd,
    )
except ImportError:  # pragma: no cover - direct-file smoke fallback
    from core.lulynx_trainer.dp_dmd_turbo import DpDmdTurboConfig, compute_dp_dmd_turbo_loss
    from core.lulynx_trainer.spd_inference import (
        SpdInferenceConfig,
        build_spd_inference_plan,
        resize_latents_for_spd,
    )


# A per-step denoise callable: ``fn(latents, *, level, scale, step_index) -> latents``.
SpdDenoiseFn = Callable[..., torch.Tensor]


def run_spd_multiresolution_denoise(
    denoise_fn: SpdDenoiseFn,
    latents: torch.Tensor,
    *,
    config: SpdInferenceConfig | Mapping[str, Any] | None = None,
    enabled: bool = False,
) -> torch.Tensor:
    """Drive ``denoise_fn`` either plainly (off) or across the SPD schedule (on).

    ``denoise_fn`` is called as ``denoise_fn(latents, level=, scale=, step_index=)``
    and must return the updated latents.  With ``enabled=False`` every step runs at
    the base resolution (parity with a plain loop).  With ``enabled=True`` early
    levels run at their downscaled resolution and are upscaled toward the base; the
    returned latents are always at the base resolution.
    """
    if latents.dim() != 4:
        raise ValueError("latents must be [batch, channels, height, width]")
    plan = build_spd_inference_plan(tuple(latents.shape), config)
    base_hw = (int(latents.shape[-2]), int(latents.shape[-1]))
    resize_mode = str(plan["resize_mode"])

    if not enabled:
        cur = latents
        for step_index in range(int(plan["total_steps"])):
            cur = denoise_fn(cur, level=0, scale=1.0, step_index=step_index)
        return cur

    cur = latents
    global_step = 0
    for level in plan["levels"]:
        target_hw = (int(level["latent_shape"][-2]), int(level["latent_shape"][-1]))
        if (int(cur.shape[-2]), int(cur.shape[-1])) != target_hw:
            cur = resize_latents_for_spd(cur, target_hw, mode=resize_mode)
        for _ in range(int(level["steps"])):
            cur = denoise_fn(cur, level=int(level["level"]), scale=float(level["scale"]), step_index=global_step)
            global_step += 1
    if (int(cur.shape[-2]), int(cur.shape[-1])) != base_hw:
        cur = resize_latents_for_spd(cur, base_hw, mode=resize_mode)
    return cur


def compose_dp_dmd_turbo_loss(
    student_pred: torch.Tensor,
    teacher_cond_pred: torch.Tensor,
    *,
    config: DpDmdTurboConfig | Mapping[str, Any] | None = None,
    enabled: bool = False,
    base_loss: Optional[torch.Tensor] = None,
    teacher_uncond_pred: Optional[torch.Tensor] = None,
    student_latents: Optional[torch.Tensor] = None,
    anchor_latents: Optional[torch.Tensor] = None,
    negative_pred: Optional[torch.Tensor] = None,
) -> dict:
    """Compose DP-DMD/Turbo distillation losses onto a base loss (default-off).

    With ``enabled=False`` the returned ``total`` is exactly ``base_loss`` (or a
    zero scalar when no base loss is given) and no distillation terms are added —
    the base training objective is unchanged (parity).  With ``enabled=True`` the
    real DP-DMD terms are computed and added to the base loss; gradients flow.
    """
    zeros = student_pred.new_zeros(())
    base = base_loss if base_loss is not None else zeros
    if not enabled:
        return {
            "total": base,
            "consistency": zeros,
            "diversity_anchor": zeros,
            "fake_critic": zeros,
            "dp_dmd_applied": False,
        }
    losses = compute_dp_dmd_turbo_loss(
        student_pred,
        teacher_cond_pred,
        config=config,
        teacher_uncond_pred=teacher_uncond_pred,
        student_latents=student_latents,
        anchor_latents=anchor_latents,
        negative_pred=negative_pred,
    )
    losses["total"] = losses["total"] + base
    losses["dp_dmd_applied"] = True
    return losses


def dp_dmd_spd_reserve_readiness() -> dict:
    """Read-only readiness report for the P4 DP-DMD/Turbo + SPD reserves."""
    return {
        "family": "anima",
        "scope": "dp_dmd_spd_reserve_only",
        "surfaces": {
            "spd_inference": {"wired": True, "kind": "multiresolution_denoise_wrapper"},
            "dp_dmd_turbo": {"wired": True, "kind": "distillation_loss_composition"},
        },
        "wired": True,
        # Honesty red-lines: opt-in only, not auto-consumed by sampler/trainer.
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "trainer_wiring_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
    }


__all__ = [
    "run_spd_multiresolution_denoise",
    "compose_dp_dmd_turbo_loss",
    "dp_dmd_spd_reserve_readiness",
]
