"""Memory/runtime profile helpers shared by trainer manifest and loop state."""

from __future__ import annotations

from typing import Any, Dict, Mapping


_PROFILE_ATTRS: tuple[tuple[str, str], ...] = (
    ("auto_vram_enhancement", "_auto_vram_enhancement_profile"),
    ("low_vram_guardrail", "_low_vram_guardrail_profile"),
    ("sdxl_lora_low_vram_profile", "_sdxl_lora_low_vram_profile"),
    ("anima_block_residency", "_anima_block_residency_profile"),
    ("anima_block_checkpointing", "_anima_block_checkpoint_profile"),
    ("newbie_block_residency", "_newbie_block_residency_profile"),
    ("newbie_block_checkpointing", "_newbie_block_checkpoint_profile"),
    ("native_weight_residency", "_native_weight_residency_profile"),
)


def _dict_profile(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping) and value:
        return dict(value)
    return {}


def build_memory_runtime_profiles(source: Any) -> Dict[str, Dict[str, Any]]:
    """Collect low-VRAM/residency profiles already realized by the trainer."""

    profiles: Dict[str, Dict[str, Any]] = {}
    for state_key, attr_name in _PROFILE_ATTRS:
        profile = _dict_profile(getattr(source, attr_name, None))
        if profile:
            profiles[state_key] = profile

    native_status = _dict_profile(getattr(source, "_native_unet_status", None))
    if native_status:
        profiles["native_unet"] = native_status
        weight_residency = _dict_profile(native_status.get("weight_residency"))
        profiles.setdefault("native_weight_residency", weight_residency)
        if not profiles["native_weight_residency"]:
            profiles.pop("native_weight_residency", None)

    return profiles


def attach_memory_runtime_profiles_to_state(state: Dict[str, Any], source: Any) -> Dict[str, Dict[str, Any]]:
    profiles = build_memory_runtime_profiles(source)
    if isinstance(state, dict):
        for state_key, profile in profiles.items():
            state[state_key] = dict(profile)
    return profiles


__all__ = [
    "attach_memory_runtime_profiles_to_state",
    "build_memory_runtime_profiles",
]
