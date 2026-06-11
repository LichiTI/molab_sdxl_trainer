"""Advisor helper for SDXL/LoRA low-VRAM profile recommendations."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from typing import Any, Dict, Optional

try:
    from .sdxl_lora_low_vram_profile import (
        PROFILE_EXPERIMENTAL,
        PROFILE_LOW_12G,
        PROFILE_OFF,
        PROFILE_STANDARD_16G,
        PROFILE_VERY_LOW_8G,
        normalize_low_vram_profile,
    )
except Exception:
    try:
        from core.lulynx_trainer.sdxl_lora_low_vram_profile import (
            PROFILE_EXPERIMENTAL,
            PROFILE_LOW_12G,
            PROFILE_OFF,
            PROFILE_STANDARD_16G,
            PROFILE_VERY_LOW_8G,
            normalize_low_vram_profile,
        )
    except Exception:
        module_path = Path(__file__).resolve().with_name("sdxl_lora_low_vram_profile.py")
        spec = importlib.util.spec_from_file_location("_lulynx_sdxl_low_vram_profile_for_advisor", module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        PROFILE_EXPERIMENTAL = module.PROFILE_EXPERIMENTAL
        PROFILE_LOW_12G = module.PROFILE_LOW_12G
        PROFILE_OFF = module.PROFILE_OFF
        PROFILE_STANDARD_16G = module.PROFILE_STANDARD_16G
        PROFILE_VERY_LOW_8G = module.PROFILE_VERY_LOW_8G
        normalize_low_vram_profile = module.normalize_low_vram_profile


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _training_route(config: Any) -> str:
    for key in ("training_type", "schema_id", "schema", "task_type"):
        raw_value = _get(config, key, "")
        value = str(getattr(raw_value, "value", raw_value) or "").strip().lower().replace("-", "_")
        if value:
            return value
    return "lora"


def _is_sdxl_lora_route(config: Any, family: str) -> bool:
    if family not in {"sdxl", "sd15", "sd1.5"}:
        return False
    route = _training_route(config)
    if route in {"controlnet", "ip_adapter", "ipadapter", "lllite"}:
        return False
    if route in {"", "lora", "sdxl_lora", "sd_lora"}:
        return True
    return "lora" in route and "controlnet" not in route and "adapter" not in route


def _profile_rank(profile: str) -> int:
    return {
        PROFILE_OFF: 0,
        PROFILE_STANDARD_16G: 1,
        PROFILE_LOW_12G: 2,
        PROFILE_VERY_LOW_8G: 3,
        PROFILE_EXPERIMENTAL: 99,
    }.get(normalize_low_vram_profile(profile), 0)


def recommend_sdxl_lora_low_vram_profile(
    config: Any,
    *,
    family: str,
    safety: str,
    width: int,
    height: int,
    batch: int,
    hardware_profile: Optional[Dict[str, Any]] = None,
    available_gb: Optional[float] = None,
) -> Dict[str, Any]:
    current = normalize_low_vram_profile(_get(config, "low_vram_profile", PROFILE_OFF))
    result: Dict[str, Any] = {
        "available": False,
        "current": current,
        "target": PROFILE_OFF,
        "should_patch": False,
        "reason": "",
        "notes": [],
    }
    if not _is_sdxl_lora_route(config, family):
        result["reason"] = "not_sdxl_lora_route"
        return result
    result["available"] = True
    if current == PROFILE_EXPERIMENTAL:
        result["reason"] = "experimental_profile_preserved"
        result["notes"].append("experimental low_vram_profile is treated as an explicit research choice.")
        return result

    hardware_profile = hardware_profile or {}
    device_tier = str(hardware_profile.get("device_tier") or "").strip().lower()
    constrained = bool(hardware_profile.get("constrained_vram_target"))
    low_target = bool(hardware_profile.get("low_vram_target"))
    megapixels = (width * height) / (1024 * 1024)
    high_pressure_shape = megapixels >= 1.0 and batch >= 2
    available = _as_float(available_gb, 0.0) if available_gb is not None else 0.0

    target = PROFILE_OFF
    reason = ""
    if safety == "danger" or device_tier in {"very_low", "low"} or (available > 0.0 and available <= 8.5):
        target = PROFILE_VERY_LOW_8G if safety in {"danger", "tight"} else PROFILE_LOW_12G
        reason = "low_vram_or_oom_risk"
    elif safety == "tight" or device_tier == "constrained" or (available > 0.0 and available <= 12.5):
        target = PROFILE_LOW_12G
        reason = "constrained_vram"
    elif safety == "watch" and (constrained or low_target or high_pressure_shape or device_tier == "standard"):
        target = PROFILE_STANDARD_16G
        reason = "watch_zone_conservative_profile"

    if target == PROFILE_OFF:
        result["reason"] = "no_low_vram_profile_needed"
        return result

    result["target"] = target
    result["reason"] = reason
    if _profile_rank(current) < _profile_rank(target):
        result["should_patch"] = True
    else:
        result["notes"].append(f"current low_vram_profile={current} is already at least as protective as {target}.")
    return result


__all__ = ["recommend_sdxl_lora_low_vram_profile"]
