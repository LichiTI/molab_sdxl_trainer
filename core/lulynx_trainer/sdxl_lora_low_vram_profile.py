"""Runtime low-VRAM profile resolver for SDXL/LoRA style training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PROFILE_OFF = "off"
PROFILE_STANDARD_16G = "standard_16g"
PROFILE_LOW_12G = "low_12g"
PROFILE_VERY_LOW_8G = "very_low_8g"
PROFILE_EXPERIMENTAL = "experimental"

LOW_VRAM_PROFILES = {
    PROFILE_OFF,
    PROFILE_STANDARD_16G,
    PROFILE_LOW_12G,
    PROFILE_VERY_LOW_8G,
    PROFILE_EXPERIMENTAL,
}


@dataclass
class LowVramProfileDecision:
    requested: str
    effective: str
    enabled: bool
    changes: dict[str, Any] = field(default_factory=dict)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "enabled": self.enabled,
            "changes": dict(self.changes),
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def normalize_low_vram_profile(value: Any) -> str:
    raw = str(value or PROFILE_OFF).strip().lower().replace("-", "_")
    aliases = {
        "": PROFILE_OFF,
        "none": PROFILE_OFF,
        "disabled": PROFILE_OFF,
        "false": PROFILE_OFF,
        "standard": PROFILE_STANDARD_16G,
        "balanced": PROFILE_STANDARD_16G,
        "16g": PROFILE_STANDARD_16G,
        "16gb": PROFILE_STANDARD_16G,
        "standard16g": PROFILE_STANDARD_16G,
        "standard_16gb": PROFILE_STANDARD_16G,
        "low": PROFILE_LOW_12G,
        "12g": PROFILE_LOW_12G,
        "12gb": PROFILE_LOW_12G,
        "low12g": PROFILE_LOW_12G,
        "low_12gb": PROFILE_LOW_12G,
        "very_low": PROFILE_VERY_LOW_8G,
        "8g": PROFILE_VERY_LOW_8G,
        "8gb": PROFILE_VERY_LOW_8G,
        "verylow8g": PROFILE_VERY_LOW_8G,
        "very_low_8gb": PROFILE_VERY_LOW_8G,
        "exp": PROFILE_EXPERIMENTAL,
        "research": PROFILE_EXPERIMENTAL,
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in LOW_VRAM_PROFILES else PROFILE_OFF


def _get(config: Any, key: str, default: Any = None) -> Any:
    return getattr(config, key, default)


def _set(config: Any, key: str, value: Any, decision: LowVramProfileDecision) -> None:
    before = _get(config, key, None)
    if before == value:
        return
    setattr(config, key, value)
    decision.changes[key] = {"before": before, "after": value}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _manual_swap_requested(config: Any) -> bool:
    return (
        str(_get(config, "swap_granularity", "off") or "off").strip().lower().replace("-", "_") != "off"
        or _as_int(_get(config, "swap_count", 0), 0) > 0
        or _as_float(_get(config, "swap_ratio", 0.0), 0.0) > 0.0
        or _as_int(_get(config, "blocks_to_swap", 0), 0) > 0
    )


def _can_cache_text_encoder(config: Any) -> bool:
    if bool(_get(config, "train_text_encoder", False)):
        return False
    if bool(_get(config, "network_train_text_encoder_only", False)):
        return False
    if bool(_get(config, "shuffle_caption", False)) and not bool(_get(config, "shuffle_caption_tags_only", False)):
        return False
    if _as_float(_get(config, "caption_dropout_rate", 0.0), 0.0) > 0.0:
        return False
    if _as_float(_get(config, "tag_dropout_rate", 0.0), 0.0) > 0.0:
        return False
    if str(_get(config, "caption_tag_dropout_targets", "") or "").strip():
        return False
    return True


def _set_cache_text_encoder(config: Any, decision: LowVramProfileDecision) -> None:
    if _can_cache_text_encoder(config):
        _set(config, "cache_text_encoder_outputs", True, decision)
        return
    decision.skipped.append(
        {
            "key": "cache_text_encoder_outputs",
            "reason": "caption/text-encoder settings need live text encoding",
        }
    )


def _set_staged_resolution_defaults(config: Any, decision: LowVramProfileDecision) -> None:
    if bool(_get(config, "enable_mixed_resolution_training", False)):
        return
    _set(config, "enable_mixed_resolution_training", True, decision)
    final_res = max(_as_int(_get(config, "resolution", 1024), 1024), 1)
    if final_res >= 768 and _as_int(_get(config, "staged_resolution_ratio_768", 0), 0) <= 0:
        _set(config, "staged_resolution_ratio_768", 35, decision)
    if final_res >= 1024 and _as_int(_get(config, "staged_resolution_ratio_1024", 0), 0) <= 0:
        _set(config, "staged_resolution_ratio_1024", 65, decision)
    elif final_res < 1024 and _as_int(_get(config, "staged_resolution_ratio_512", 0), 0) <= 0:
        _set(config, "staged_resolution_ratio_512", 100, decision)


def apply_sdxl_lora_low_vram_profile(config: Any, *, model_arch: str = "") -> LowVramProfileDecision:
    requested = normalize_low_vram_profile(_get(config, "low_vram_profile", PROFILE_OFF))
    legacy_enabled = bool(_get(config, "sdxl_low_vram_optimization", False))
    if requested == PROFILE_OFF and legacy_enabled:
        requested = PROFILE_VERY_LOW_8G

    decision = LowVramProfileDecision(
        requested=requested,
        effective=PROFILE_OFF,
        enabled=False,
    )
    if requested == PROFILE_OFF:
        decision.notes.append("low_vram_profile is off")
        return decision

    arch = str(model_arch or _get(config, "model_type", "") or "").strip().lower()
    if arch and arch not in {"sdxl", "sd15"}:
        decision.skipped.append({"key": "low_vram_profile", "reason": f"unsupported model_arch={arch}"})
        return decision

    training_type = str(_get(config, "training_type", "lora") or "lora").strip().lower().replace("-", "_")
    if training_type in {"controlnet", "ip_adapter", "lllite"}:
        decision.skipped.append({"key": "low_vram_profile", "reason": f"pipeline route keeps its own memory policy: {training_type}"})
        return decision

    decision.effective = requested
    decision.enabled = True
    _set(config, "sdxl_low_vram_optimization", True, decision)
    _set(config, "cache_latents", True, decision)
    _set(config, "gradient_checkpointing", True, decision)
    _set(config, "vae_slicing", True, decision)
    _set(config, "attention_slicing", True, decision)
    _set(config, "pytorch_cuda_expandable_segments", True, decision)
    _set(config, "model_to_condition_enabled", True, decision)
    _set_cache_text_encoder(config, decision)

    if requested == PROFILE_STANDARD_16G:
        if str(_get(config, "te_vae_offload_strategy", "phase") or "phase").strip().lower() in {"", "resident"}:
            _set(config, "te_vae_offload_strategy", "phase", decision)
        _set(config, "cuda_cache_release_strategy", "phase_boundary", decision)
        decision.notes.append("16G profile keeps block/module offload disabled unless explicitly selected.")
        return decision

    if requested in {PROFILE_LOW_12G, PROFILE_VERY_LOW_8G, PROFILE_EXPERIMENTAL}:
        _set(config, "te_vae_offload_strategy", "aggressive", decision)
        _set(config, "cache_latents_to_disk", True, decision)
        _set(config, "cuda_cache_release_strategy", "after_optimizer", decision)
        _set_staged_resolution_defaults(config, decision)

    if requested in {PROFILE_LOW_12G, PROFILE_EXPERIMENTAL}:
        if not _manual_swap_requested(config):
            _set(config, "swap_granularity", "merged_block", decision)
            _set(config, "swap_ratio", 0.25, decision)
            _set(config, "block_merge_size", 2, decision)
            _set(config, "block_swap_strategy", "auto", decision)
        else:
            decision.skipped.append({"key": "swap_granularity", "reason": "manual swap settings preserved"})

    if requested == PROFILE_VERY_LOW_8G:
        _set(config, "te_vae_offload_strategy", "aggressive", decision)
        _set(config, "cuda_cache_release_strategy", "aggressive", decision)
        _set(config, "checkpoint_policy", "offloaded", decision)
        _set(config, "cpu_offload_checkpointing_mode", "pinned_async", decision)
        if not _manual_swap_requested(config):
            _set(config, "swap_granularity", "merged_block", decision)
            _set(config, "swap_ratio", 0.4, decision)
            _set(config, "block_merge_size", 2, decision)
            _set(config, "block_swap_strategy", "auto", decision)
        else:
            decision.skipped.append({"key": "swap_granularity", "reason": "manual swap settings preserved"})
        batch_size = max(_as_int(_get(config, "train_batch_size", 1), 1), 1)
        if batch_size > 1:
            _set(config, "train_batch_size", 1, decision)
            decision.warnings.append("very_low_8g forces train_batch_size=1; use gradient_accumulation_steps for effective batch.")

    if requested == PROFILE_EXPERIMENTAL:
        if bool(_get(config, "gradient_checkpointing", False)):
            decision.skipped.append(
                {
                    "key": "module_offload_enabled",
                    "reason": "kept disabled because module_offload conflicts with gradient_checkpointing in TrainingLoop v1",
                }
            )
        if bool(_get(config, "weight_compression_enabled", False)):
            decision.warnings.append("weight compression remains experimental; verify a short run before long training.")

    return decision


__all__ = [
    "LOW_VRAM_PROFILES",
    "LowVramProfileDecision",
    "apply_sdxl_lora_low_vram_profile",
    "normalize_low_vram_profile",
]
