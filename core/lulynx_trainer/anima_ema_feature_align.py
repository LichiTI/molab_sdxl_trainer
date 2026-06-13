# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""EMA feature self-distillation alignment (default-off reserve).

╔════════════════════════════════════════════════════════════════════════╗
║ STATUS: TECHNICAL RESERVE                                              ║
║  Runtime-only, default-off. Intentionally NOT exposed in the webui.    ║
║  No positive-benefit evidence for LoRA fine-tuning — JLT's headline    ║
║  result does not use it (it only appears in an async-teacher side      ║
║  script with no ablation), and the cost is certain (≈2x forward + loses║
║  cudagraph/offload). Kept here, parity-safe and smoke-tested, so it can║
║  be activated for a real anima-LoRA A/B; promote to UI only if that A/B║
║  shows a gain.                                                         ║
╚════════════════════════════════════════════════════════════════════════╝

A cleanroom adaptation of JLT's ``ema_feat_align`` (arXiv 2605.27102) to the
Anima LoRA training path. The student (current weights, normal timestep) is
pulled toward an EMA-of-LoRA *teacher* evaluated at a *smaller* timestep
(cleaner input): block features at selected layers are aligned by a cosine
loss, ``mean_pairs(1 - cos(student_feat, teacher_feat))``.

Design choices specific to LoRA fine-tuning:
  * The teacher uses an exponential moving average of the *trainable* (LoRA)
    parameters kept in fp32 — bf16 cannot represent ``1 - decay`` (e.g. 1e-4)
    so a bf16 EMA would freeze/drift (same reasoning as JLT ``update_ema``).
  * The teacher forward runs by temporarily swapping the live LoRA weights for
    the EMA shadow under ``torch.no_grad`` (swap, not ``functional_call``,
    because the Anima unet is a custom wrapper). A ``try/finally`` guarantees
    the live weights are restored even on error — the student is never
    polluted.
  * Clean latents ``x`` and noise ``e`` are recovered analytically from the
    student's ``(noisy_latents, target, sigma)`` so no extra tensors need to be
    threaded down from the forward stage.

Everything here only runs when ``anima_ema_feat_align_enabled`` is set; the
default path never imports the capture seam's active context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

try:  # package import
    from .anima_flow import AnimaFlowConfig, build_anima_flow_inputs, sample_anima_sigmas
    from .anima_feature_capture import feature_capture_scope, parse_layer_list
except ImportError:  # pragma: no cover - direct-file smoke fallback
    from core.lulynx_trainer.anima_flow import (
        AnimaFlowConfig,
        build_anima_flow_inputs,
        sample_anima_sigmas,
    )
    from core.lulynx_trainer.anima_feature_capture import feature_capture_scope, parse_layer_list


class EmaLoraShadow:
    """fp32 exponential moving average of the trainable (LoRA) parameters."""

    def __init__(self) -> None:
        self._shadow: Dict[str, torch.Tensor] = {}
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _named_trainable(self, named_params: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
        return [(name, p) for name, p in named_params if p.requires_grad]

    @torch.no_grad()
    def update(self, named_params: List[Tuple[str, torch.Tensor]], decay: float) -> None:
        """Update (or lazily initialize) the EMA shadow from current params."""
        decay = float(decay)
        for name, param in self._named_trainable(named_params):
            src = param.detach().float()
            if name not in self._shadow:
                self._shadow[name] = src.clone()
            else:
                self._shadow[name].mul_(decay).add_(src, alpha=1.0 - decay)
        self._initialized = True

    def state(self) -> Dict[str, torch.Tensor]:
        return self._shadow


def _recover_latent_and_noise(
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    sigma_view: torch.Tensor,
    pred_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recover ``(clean_latents, noise)`` from the student's mixture.

    ``noisy = (1 - sigma) * x + sigma * e`` with target depending on pred_type.
    The velocity path needs no division (most stable at sigma extremes).
    """
    pred_type = (pred_type or "velocity").lower()
    if pred_type == "velocity":  # target = e - x
        x = noisy_latents - sigma_view * target
        e = x + target
    elif pred_type == "sample":  # target = x
        x = target
        e = (noisy_latents - (1.0 - sigma_view) * x) / sigma_view.clamp_min(1e-6)
    elif pred_type in {"noise", "epsilon"}:  # target = e
        e = target
        x = (noisy_latents - sigma_view * e) / (1.0 - sigma_view).clamp_min(1e-6)
    else:
        raise ValueError(f"Unsupported model_prediction_type={pred_type!r}")
    return x, e


def _build_anima_teacher_kwargs(
    owner: Any,
    *,
    sample: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: Dict[str, Any],
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    """Mirror the main/prior anima forward conditioning for the teacher pass."""
    kwargs: Dict[str, Any] = {
        "sample": sample,
        "timestep": timesteps,
        "encoder_hidden_states": prompt_embeds.get("encoder_hidden_states", prompt_embeds)
        if isinstance(prompt_embeds, dict)
        else prompt_embeds,
    }
    if isinstance(prompt_embeds, dict):
        qwen3_hs = prompt_embeds.get("qwen3_hidden_states")
        if qwen3_hs is not None:
            kwargs["qwen3_hidden_states"] = qwen3_hs
            qwen3_mask = prompt_embeds.get("qwen3_attention_mask")
            if qwen3_mask is not None:
                kwargs["qwen3_attention_mask"] = qwen3_mask
    anima_llm_adapter = getattr(owner.unet, "anima_llm_adapter", None)
    if bool(getattr(owner, "anima_faithful_forward", False)) and anima_llm_adapter is not None:
        try:
            from .anima_faithful_train_context import resolve_anima_faithful_context
        except ImportError:  # pragma: no cover
            from core.lulynx_trainer.anima_faithful_train_context import resolve_anima_faithful_context
        kwargs["encoder_hidden_states"] = resolve_anima_faithful_context(
            prompt_embeds, batch, anima_llm_adapter, owner.device, owner.dtype
        )
        kwargs.pop("qwen3_hidden_states", None)
        kwargs.pop("qwen3_attention_mask", None)
    return kwargs


def _swap_in_shadow(
    named_params: List[Tuple[str, torch.Tensor]],
    shadow: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Swap live param data for EMA shadow; return saved live data for restore."""
    saved: Dict[str, torch.Tensor] = {}
    for name, param in named_params:
        if not param.requires_grad or name not in shadow:
            continue
        saved[name] = param.data
        param.data = shadow[name].to(device=param.device, dtype=param.dtype)
    return saved


def _restore_live(named_params: List[Tuple[str, torch.Tensor]], saved: Dict[str, torch.Tensor]) -> None:
    name_to_param = {name: p for name, p in named_params}
    for name, data in saved.items():
        name_to_param[name].data = data


def compute_ema_feat_align_loss(
    *,
    owner: Any,
    student_features: Dict[int, torch.Tensor],
    prompt_embeds: Dict[str, Any],
    batch: Dict[str, Any],
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Return the EMA feature-alignment loss for the current step.

    Returns a zero scalar (no graph contribution) when the teacher shadow is
    not yet initialized or when no student features were captured.
    """
    teacher_layers = parse_layer_list(getattr(owner, "anima_ema_feat_align_teacher_layers", ""))
    student_layers = parse_layer_list(getattr(owner, "anima_ema_feat_align_student_layers", ""))
    if len(teacher_layers) != len(student_layers):
        raise ValueError(
            "anima_ema_feat_align_teacher_layers and ..._student_layers must have equal length."
        )
    shadow_obj: EmaLoraShadow = owner._ema_lora_shadow
    if not teacher_layers or not student_features or not shadow_obj.initialized:
        return noisy_latents.new_zeros(())

    # Recover clean latents / noise from the student mixture (self-contained).
    sigma_scale = 1.0 if bool(getattr(owner, "anima_faithful_forward", False)) else 1000.0
    sigmas = timesteps.float() / sigma_scale
    view_shape = (sigmas.shape[0],) + (1,) * (noisy_latents.dim() - 1)
    sigma_view = sigmas.to(device=noisy_latents.device, dtype=noisy_latents.dtype).view(view_shape)
    pred_type = getattr(owner, "anima_model_prediction_type", "velocity") or "velocity"
    clean, noise = _recover_latent_and_noise(noisy_latents, target, sigma_view, pred_type)

    # Teacher timestep: a freshly sampled sigma, floored elementwise by the
    # student's sigma so the teacher always sees a cleaner (smaller-t) input.
    cfg = AnimaFlowConfig(
        timestep_sampling=getattr(owner, "anima_timestep_sampling", "sigma") or "sigma",
        sigmoid_scale=getattr(owner, "anima_sigmoid_scale", 1.0),
        discrete_flow_shift=getattr(owner, "anima_discrete_flow_shift", 1.0),
        logit_mean=getattr(owner, "flow_logit_mean", 0.0),
        logit_std=getattr(owner, "flow_logit_std", 1.0),
    )
    fresh = sample_anima_sigmas(
        clean.shape[0], device=clean.device, dtype=clean.dtype, config=cfg
    )
    teacher_sigma = torch.minimum(fresh, sigmas.to(device=clean.device, dtype=clean.dtype))
    num_train = 1 if bool(getattr(owner, "anima_faithful_forward", False)) else 1000
    teacher_noisy, _t_target, teacher_timesteps = build_anima_flow_inputs(
        clean, noise, teacher_sigma, num_train_timesteps=num_train, model_prediction_type=pred_type
    )

    teacher_kwargs = _build_anima_teacher_kwargs(
        owner, sample=teacher_noisy, timesteps=teacher_timesteps, prompt_embeds=prompt_embeds, batch=batch
    )

    named_params = list(owner.unet.named_parameters())
    saved = _swap_in_shadow(named_params, shadow_obj.state())
    try:
        with torch.no_grad(), feature_capture_scope(teacher_layers) as cap:
            with torch.autocast(device_type="cuda", dtype=owner.dtype) if torch.cuda.is_available() else _nullctx():
                owner.unet(**teacher_kwargs)
            teacher_features = dict(cap.features)
    finally:
        _restore_live(named_params, saved)

    feat_loss = noisy_latents.new_zeros(())
    pairs = 0
    for t_layer, s_layer in zip(teacher_layers, student_layers):
        if t_layer not in teacher_features or s_layer not in student_features:
            continue
        s_feat = student_features[s_layer].float()
        t_feat = teacher_features[t_layer].float().to(s_feat.device)
        cosine = torch.nn.functional.cosine_similarity(s_feat, t_feat, dim=-1, eps=1e-8)
        feat_loss = feat_loss + (1.0 - cosine).mean()
        pairs += 1
    if pairs == 0:
        return noisy_latents.new_zeros(())
    return feat_loss / pairs


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
