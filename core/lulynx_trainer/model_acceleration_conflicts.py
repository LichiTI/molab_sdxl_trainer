"""Conflict detectors for acceleration policy recommendations."""

from __future__ import annotations

from typing import Any, Mapping


ACTIVE_COMPILE_RUNTIMES = {"compile", "compile_cache", "cudagraph", "compile_cudagraph"}
WEIGHT_COMPRESSION_PRESET_ALIASES = {
    "safe": "stable_backbone_int8",
    "stable": "stable_backbone_int8",
    "backbone_int8": "stable_backbone_int8",
    "int8": "stable_backbone_int8",
    "aggressive": "aggressive_backbone_uint4",
    "backbone_uint4": "aggressive_backbone_uint4",
    "uint4": "aggressive_backbone_uint4",
    "te_int8": "text_encoder_int8",
    "text_int8": "text_encoder_int8",
    "text_encoder": "text_encoder_int8",
    "all_int8": "both_int8",
    "both": "both_int8",
    "float8": "experimental_float8",
    "fp8_experimental": "experimental_float8",
}
WEIGHT_COMPRESSION_PRESETS = {
    "stable_backbone_int8",
    "aggressive_backbone_uint4",
    "text_encoder_int8",
    "both_int8",
    "experimental_float8",
}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def active_compile_intent(config: Mapping[str, Any]) -> bool:
    return (
        _norm(config.get("compile_runtime")) in ACTIVE_COMPILE_RUNTIMES
        or _flag(config.get("torch_compile"))
        or _flag(config.get("compile_cache_enabled"))
    )


def weight_compression_requested(config: Mapping[str, Any]) -> bool:
    preset = WEIGHT_COMPRESSION_PRESET_ALIASES.get(_norm(config.get("weight_compression_preset")), _norm(config.get("weight_compression_preset")))
    return _flag(config.get("weight_compression_enabled")) or _flag(config.get("fp8_base")) or preset in WEIGHT_COMPRESSION_PRESETS


__all__ = ["active_compile_intent", "weight_compression_requested"]
