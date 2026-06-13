from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch


def _first_attr(obj: Any, *names: str) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class AnimaCacheEncodeBundle:
    vae_encode_fn: Callable[[torch.Tensor], torch.Tensor]
    text_encode_fn: Callable[[str], Dict[str, torch.Tensor]]
    summary: str
    primary_text_source: str


def build_anima_cache_encode_bundle(
    *,
    model: Any,
    device: str,
    dtype: torch.dtype,
    config: Any,
) -> AnimaCacheEncodeBundle:
    """Create cache-builder encode callables from the currently loaded Anima model.

    The preferred primary text-conditioning source is ``text_encoder_1`` +
    ``tokenizer_1``. When a single-file native Anima checkpoint is loaded, those
    components may be absent while Qwen3 is available. In that case we promote
    Qwen3 hidden states to the cache schema's required ``prompt_embeds`` field so
    cache-first training can still proceed without forcing a diffusers directory.
    """

    vae = getattr(model, "vae", None)
    if vae is None:
        raise RuntimeError("Anima cache generation requires a loaded VAE.")

    text_encoder = getattr(model, "text_encoder_1", None)
    tokenizer = getattr(model, "tokenizer_1", None)
    qwen3_encoder = _first_attr(model, "anima_qwen3_encoder", "qwen3_encoder")
    qwen3_tokenizer = _first_attr(model, "anima_qwen3_tokenizer", "qwen3_tokenizer")
    t5_tokenizer = _first_attr(model, "anima_t5_tokenizer", "t5_tokenizer")

    has_clip_primary = text_encoder is not None and tokenizer is not None
    has_qwen3_primary = qwen3_encoder is not None and qwen3_tokenizer is not None
    if not has_clip_primary and not has_qwen3_primary:
        raise RuntimeError(
            "Anima cache generation requires either text_encoder_1 + tokenizer_1, "
            "or anima_qwen3_encoder + anima_qwen3_tokenizer."
        )

    primary_text_source = "clip" if has_clip_primary else "qwen3"

    def _vae_encode(image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vae.to(device=device, dtype=torch.float32)
            vae_input = image.to(device=device, dtype=torch.float32)
            if _is_qwen_image_vae(vae) and vae_input.dim() == 4:
                vae_input = vae_input.unsqueeze(2)
            latents = vae.encode(vae_input).latent_dist.sample()
            if _is_qwen_image_vae(vae):
                latents = _normalize_qwen_image_latents(vae, latents)
                if latents.dim() == 5 and latents.shape[2] == 1:
                    latents = latents.squeeze(2)
            else:
                scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0)
                latents = latents * scaling_factor
            return latents.to(dtype)

    def _encode_qwen3(caption: str) -> tuple[torch.Tensor, torch.Tensor]:
        qwen3_max_len = int(getattr(config, "anima_qwen3_max_token_length", 512) or 512)
        qwen3_tokens = qwen3_tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=qwen3_max_len,
            return_tensors="pt",
        )
        qwen3_input_ids = qwen3_tokens["input_ids"].to(device)
        qwen3_attn = qwen3_tokens["attention_mask"].to(device)
        qwen3_encoder.to(device=device, dtype=dtype)
        try:
            qwen3_outputs = qwen3_encoder(
                input_ids=qwen3_input_ids,
                attention_mask=qwen3_attn,
                output_hidden_states=True,
                use_cache=False,
            )
        except TypeError:
            qwen3_outputs = qwen3_encoder(input_ids=qwen3_input_ids, attention_mask=qwen3_attn)
        qwen3_hidden = getattr(qwen3_outputs, "last_hidden_state", None)
        if qwen3_hidden is None:
            hidden_states = getattr(qwen3_outputs, "hidden_states", None)
            if hidden_states:
                qwen3_hidden = hidden_states[-1]
        if qwen3_hidden is None:
            first_output = qwen3_outputs[0]
            hidden_size = int(getattr(getattr(qwen3_encoder, "config", None), "hidden_size", 0) or 0)
            if hidden_size > 0 and int(first_output.shape[-1]) == hidden_size:
                qwen3_hidden = first_output
        if qwen3_hidden is None:
            raise RuntimeError("Qwen3 encoder did not return hidden states for Anima cache generation.")
        return qwen3_hidden[0].to(dtype), qwen3_attn[0].to(torch.bool)

    def _text_encode(caption: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            result: Dict[str, torch.Tensor] = {}
            qwen3_primary = False

            if has_clip_primary:
                tokens = tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=getattr(tokenizer, "model_max_length", 512),
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(device)
                attn = tokens["attention_mask"].to(device)
                outputs = text_encoder(input_ids=input_ids, attention_mask=attn)
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    hidden = outputs[0]
                result["prompt_embeds"] = hidden[0].to(dtype)
                result["attn_mask"] = attn[0].to(torch.bool)
            else:
                qwen3_hidden, qwen3_attn = _encode_qwen3(caption)
                result["prompt_embeds"] = qwen3_hidden
                result["attn_mask"] = qwen3_attn
                qwen3_primary = True

            if has_qwen3_primary and not qwen3_primary:
                qwen3_hidden, qwen3_attn = _encode_qwen3(caption)
                result["qwen3_hidden_states"] = qwen3_hidden
                result["qwen3_attention_mask"] = qwen3_attn

            if t5_tokenizer is not None:
                t5_max_len = int(getattr(config, "anima_t5_max_token_length", 512) or 512)
                t5_tokens = t5_tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=t5_max_len,
                    return_tensors="pt",
                )
                result["t5_input_ids"] = t5_tokens["input_ids"][0].long()
                result["t5_attn_mask"] = t5_tokens["attention_mask"][0].to(torch.bool)

            return result

    summary = (
        f"primary={primary_text_source}, "
        f"qwen3={'yes' if has_qwen3_primary else 'no'}, "
        f"t5={'yes' if t5_tokenizer is not None else 'no'}"
    )
    return AnimaCacheEncodeBundle(
        vae_encode_fn=_vae_encode,
        text_encode_fn=_text_encode,
        summary=summary,
        primary_text_source=primary_text_source,
    )


def _is_qwen_image_vae(vae: Any) -> bool:
    config = getattr(vae, "config", None)
    return hasattr(config, "latents_mean") and hasattr(config, "latents_std") and hasattr(config, "z_dim")


def _normalize_qwen_image_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    config = getattr(vae, "config", None)
    z_dim = int(getattr(config, "z_dim", latents.shape[1]) or latents.shape[1])
    mean = torch.tensor(getattr(config, "latents_mean"), device=latents.device, dtype=latents.dtype)
    std = torch.tensor(getattr(config, "latents_std"), device=latents.device, dtype=latents.dtype)
    view_shape = (1, z_dim) + (1,) * (latents.dim() - 2)
    mean = mean.view(view_shape)
    inv_std = (1.0 / std).view(view_shape)
    return (latents - mean) * inv_std


def _denormalize_qwen_image_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_normalize_qwen_image_latents` (``latents*std+mean``).

    Used at decode time: the DiT operates in the normalised latent space, so a
    sampled latent must be de-normalised before the qwen-image VAE decoder.
    """
    config = getattr(vae, "config", None)
    z_dim = int(getattr(config, "z_dim", latents.shape[1]) or latents.shape[1])
    mean = torch.tensor(getattr(config, "latents_mean"), device=latents.device, dtype=latents.dtype)
    std = torch.tensor(getattr(config, "latents_std"), device=latents.device, dtype=latents.dtype)
    view_shape = (1, z_dim) + (1,) * (latents.dim() - 2)
    mean = mean.view(view_shape)
    std = std.view(view_shape)
    return latents * std + mean
