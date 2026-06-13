"""FLUX-specific preflight messages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from .model_optimization_capabilities import get_model_optimization_capability


def _flag(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _str(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _str(value).lower()


_FLUX_NETWORK_ALIASES = {
    "networks.lora_flux": "networks.lora",
}
_FLUX_UNSUPPORTED_NETWORKS = {
    "networks.tlora_flux",
    "networks.oft_flux",
    "networks.oft",
    "lycoris.kohya",
    "lycoris.locon",
}


def normalize_flux_network_module(value: Any) -> str:
    normalized = _lower(value)
    if normalized in {"", "lora"}:
        return "networks.lora"
    return _FLUX_NETWORK_ALIASES.get(normalized, normalized)


def is_flux_network_module_supported(value: Any) -> bool:
    raw = _lower(value)
    return raw not in _FLUX_UNSUPPORTED_NETWORKS and normalize_flux_network_module(raw) == "networks.lora"


@dataclass
class FluxPreflightMessages:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_flux_preflight_messages(config: Mapping[str, Any], training_type: str) -> FluxPreflightMessages:
    messages = FluxPreflightMessages()
    route = _lower(training_type or config.get("schema_id") or config.get("model_type"))
    model_type = _lower(config.get("model_type"))
    if not route.startswith("flux") and model_type != "flux":
        return messages

    capability = get_model_optimization_capability("flux")
    if capability is not None:
        messages.notes.extend(capability.notes)

    if route in {"flux-finetune", "flux-controlnet"} or route.endswith("-finetune") or route.endswith("-controlnet"):
        messages.errors.append("FLUX full finetune/ControlNet is not wired to the native trainer yet.")
        return messages
    messages.notes.append("FLUX LoRA uses the native preview trainer; full finetune and ControlNet remain gated.")

    raw_network = _lower(config.get("network_module"))
    network_module = normalize_flux_network_module(raw_network)
    if not is_flux_network_module_supported(raw_network):
        messages.errors.append(
            "FLUX LoRA preview currently supports network_module=networks.lora only. "
            f"Requested: {raw_network or network_module}."
        )

    base_model = (
        _str(config.get("pretrained_model_name_or_path"))
        or _str(config.get("base_model_path"))
        or _str(config.get("pretrained_model"))
        or _str(config.get("diffusers_path"))
    )
    transformer_path = _str(config.get("transformer_path")) or _str(config.get("flux_transformer_path"))
    if not base_model:
        messages.errors.append("FLUX LoRA requires a Diffusers base model path or single-file checkpoint.")
    for label, raw in (
        ("FLUX base model", base_model),
        ("FLUX transformer", transformer_path),
        ("FLUX AE", _str(config.get("ae_path")) or _str(config.get("ae"))),
        ("FLUX T5XXL", _str(config.get("t5xxl_path")) or _str(config.get("t5xxl"))),
        ("FLUX CLIP-L", _str(config.get("clip_l_path")) or _str(config.get("clip_l"))),
    ):
        if raw and not os.path.exists(raw):
            messages.errors.append(label + " path does not exist: " + raw)

    cache_text = _flag(config.get("cache_text_encoder_outputs")) or _flag(config.get("cache_text_encoder_outputs_to_disk"))
    cache_latents = _flag(config.get("cache_latents")) or _flag(config.get("cache_latents_to_disk"))
    if cache_text or cache_latents:
        messages.notes.append("FLUX cache-first route requested; keep text encoders frozen when cached text outputs are used.")
    else:
        messages.notes.append("For FLUX, prefer cache_latents + cache_text_encoder_outputs before trying heavier offload strategies.")

    compression_requested = (
        _flag(config.get("weight_compression_enabled"))
        or _flag(config.get("fp8_base"))
        or bool(_lower(config.get("weight_compression_preset")))
    )
    if compression_requested:
        messages.notes.append("FLUX transformer weight compression is the planned first native optimization track.")
    else:
        messages.notes.append("When the FLUX route is enabled, backbone weight compression should be preferred over channels_last.")

    if _flag(config.get("train_text_encoder")) or _flag(config.get("train_t5xxl")):
        messages.warnings.append("FLUX LoRA preview keeps CLIP/T5 frozen; text encoder training requests will be ignored.")
    if _flag(config.get("opt_channels_last")):
        messages.warnings.append("FLUX is transformer-heavy; channels_last is expected to have limited benefit.")
    if _flag(config.get("module_offload_enabled")):
        messages.warnings.append("FLUX module_offload is not part of the preview trainer yet; use smaller batch/rank first.")
    return messages


__all__ = [
    "FluxPreflightMessages",
    "build_flux_preflight_messages",
    "is_flux_network_module_supported",
    "normalize_flux_network_module",
]
