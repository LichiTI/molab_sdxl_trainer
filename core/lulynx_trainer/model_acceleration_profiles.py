"""Acceleration profile constants and aliases."""

from __future__ import annotations

from typing import Any


PROFILE_OFF = "off"
PROFILE_SAFE = "safe"
PROFILE_BALANCED = "balanced"
PROFILE_AGGRESSIVE = "aggressive"
PROFILE_LOW_VRAM = "low_vram"

ACCELERATION_PROFILES = {PROFILE_OFF, PROFILE_SAFE, PROFILE_BALANCED, PROFILE_AGGRESSIVE, PROFILE_LOW_VRAM}
_PROFILE_ALIASES = {
    **dict.fromkeys(("", "none", "default", "disabled", "disable", "false", "0"), PROFILE_OFF),
    **dict.fromkeys(("conservative", "stable", "safe_profile"), PROFILE_SAFE),
    **dict.fromkeys(("auto", "balance", "normal", "speed", "fast", "faster"), PROFILE_BALANCED),
    **dict.fromkeys(("turbo", "max", "maximum", "fastest", "compile", "compile_cache"), PROFILE_AGGRESSIVE),
    **dict.fromkeys(("lowvram", "low_memory", "memory", "vram", "8g", "8gb", "12g", "12gb"), PROFILE_LOW_VRAM),
}


def normalize_acceleration_profile(value: Any) -> str:
    raw = str(value or PROFILE_OFF).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _PROFILE_ALIASES.get(raw, raw)
    return normalized if normalized in ACCELERATION_PROFILES else PROFILE_OFF


__all__ = [
    "ACCELERATION_PROFILES",
    "PROFILE_AGGRESSIVE",
    "PROFILE_BALANCED",
    "PROFILE_LOW_VRAM",
    "PROFILE_OFF",
    "PROFILE_SAFE",
    "normalize_acceleration_profile",
]
