"""Runtime profile for SD15/SDXL cache-first training decisions."""

from __future__ import annotations

from typing import Any, Mapping


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _field(config: Any, name: str, default: Any = "") -> Any:
    try:
        return getattr(config, name, default)
    except Exception:
        return default


def _low_vram_enabled(profile: Mapping[str, Any] | None, config: Any) -> bool:
    if isinstance(profile, Mapping) and _boolish(profile.get("enabled", False)):
        return True
    return _boolish(_field(config, "sdxl_low_vram_optimization", False))


def build_diffusers_cache_runtime_profile(
    config: Any,
    *,
    model_arch: str,
    cache_first: bool,
    cache_root: str = "",
    component_cpu_residency: bool = False,
    text_cache_forced_unet_only: bool = False,
    text_cache_disabled_reason: str = "",
    blockers: list[str] | tuple[str, ...] | None = None,
    low_vram_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build JSON-friendly runtime evidence for Diffusers U-Net cache routes."""

    arch = str(model_arch or _field(config, "model_arch", "") or _field(config, "model_type", "")).strip().lower()
    requested_cache_latents = _boolish(_field(config, "cache_latents", False))
    requested_text_cache = _boolish(_field(config, "cache_text_encoder_outputs", False))
    train_text_encoder = _boolish(_field(config, "train_text_encoder", False))
    train_unet = _boolish(_field(config, "train_unet", True))
    text_cache_active = requested_text_cache and not train_text_encoder
    return {
        "enabled": arch in {"sdxl", "sd15"},
        "source": "diffusers_unet_runtime",
        "model_arch": arch,
        "cache_first": bool(cache_first),
        "cache_root": str(cache_root or ""),
        "cache_latents": requested_cache_latents,
        "cache_text_encoder_outputs": requested_text_cache,
        "text_cache_active": bool(text_cache_active),
        "text_cache_forced_unet_only": bool(text_cache_forced_unet_only),
        "text_cache_disabled_reason": str(text_cache_disabled_reason or ""),
        "train_unet": train_unet,
        "train_text_encoder": train_text_encoder,
        "network_train_unet_only": _boolish(_field(config, "network_train_unet_only", False)),
        "network_train_text_encoder_only": _boolish(_field(config, "network_train_text_encoder_only", False)),
        "component_cpu_residency": bool(component_cpu_residency),
        "te_vae_offload_strategy": str(_field(config, "te_vae_offload_strategy", "phase") or "phase"),
        "sdxl_low_vram_optimization": _boolish(_field(config, "sdxl_low_vram_optimization", False)),
        "low_vram_profile": str(_field(config, "low_vram_profile", "off") or "off"),
        "low_vram_profile_enabled": _low_vram_enabled(low_vram_profile, config),
        "blockers": [str(item) for item in (blockers or [])],
        "vae_slicing": _boolish(_field(config, "vae_slicing", False)),
        "attention_slicing": _boolish(_field(config, "attention_slicing", False)),
        "gradient_checkpointing": _boolish(_field(config, "gradient_checkpointing", False)),
        "cache_latents_to_disk": _boolish(_field(config, "cache_latents_to_disk", False)),
        "model_to_condition_enabled": _boolish(_field(config, "model_to_condition_enabled", True)),
    }


__all__ = ["build_diffusers_cache_runtime_profile"]
