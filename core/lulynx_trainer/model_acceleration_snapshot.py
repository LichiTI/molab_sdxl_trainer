"""Small acceleration snapshots for preflight and runtime telemetry."""

from __future__ import annotations

from typing import Any, Mapping

from .model_acceleration_conflicts import active_compile_intent, weight_compression_requested
from .model_acceleration_benchmark_matrix import build_model_acceleration_benchmark_matrix
from .model_acceleration_data import stable_caption_conditioning
from .model_acceleration_matrix import acceleration_matrix_summary_for


def _field(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _boolish(value: Any, *, default: bool = False) -> bool:
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


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_model_family_from_config(config: Any, *, schema_id: str = "", training_type: str = "") -> str:
    text = " ".join(
        str(value or "")
        for value in (
            schema_id,
            training_type,
            _field(config, "schema_id", ""),
            _field(config, "training_type", ""),
            _field(config, "model_type", ""),
            _field(config, "model_arch", ""),
            _field(config, "route", ""),
        )
    ).lower().replace("\\", "/")
    compact = text.replace("_", "-")
    if "newbie" in compact:
        return "newbie"
    if "anima" in compact or "qwen-image" in compact:
        return "anima"
    if "flux" in compact:
        return "flux"
    if "sdxl" in compact or "stable-diffusion-xl" in compact or "xl-lora" in compact:
        return "sdxl"
    if "sd15" in compact or "sd-1" in compact or "sd1.5" in compact or compact.startswith("sd-lora"):
        return "sd15"
    return "unknown"


def _config_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    keys = (
        "shuffle_caption",
        "shuffle_caption_tags_only",
        "caption_dropout_rate",
        "tag_dropout_rate",
        "caption_tag_dropout_targets",
        "train_text_encoder",
        "network_train_text_encoder_only",
    )
    return {key: _field(config, key) for key in keys}


def build_model_acceleration_cache_safety(
    config: Any,
    *,
    family: str = "",
    profile: str = "",
) -> dict[str, Any]:
    resolved_family = family or normalize_model_family_from_config(config)
    resolved_profile = str(profile or _field(config, "acceleration_profile", _field(config, "speed_profile", "")) or "").strip().lower()
    cache_mode = _norm(_field(config, "native_cache_mode", _field(config, "anima_cache_mode", "")))
    cache_first = cache_mode in {"cache_first", "force_cache_only"} or _boolish(_field(config, "anima_cached_training")) or _boolish(_field(config, "use_cache"))
    text_static = stable_caption_conditioning(_config_mapping(config), default_shuffle=resolved_family == "flux")
    train_text_encoder = _boolish(_field(config, "train_text_encoder")) or _boolish(_field(config, "network_train_text_encoder_only"))
    text_requested = _boolish(_field(config, "cache_text_encoder_outputs"))
    text_active = text_requested and text_static and not train_text_encoder
    compile_active = active_compile_intent(_config_mapping_with_compile(config))
    blockers: list[str] = []
    if text_requested and not text_static:
        blockers.append("dynamic_caption_conditioning")
    if text_requested and train_text_encoder:
        blockers.append("train_text_encoder")
    if compile_active and _boolish(_field(config, "compile_require_cache_first", True), default=True):
        if resolved_family in {"anima", "newbie"} and not cache_first:
            blockers.append("compile_requires_cache_first")
    if compile_active and weight_compression_requested(_config_mapping_with_compile(config)):
        blockers.append("compile_weight_compression_conflict")
    return {
        "family": resolved_family,
        "profile": resolved_profile,
        "cache_first": bool(cache_first),
        "cache_mode": cache_mode or "auto",
        "latent_cache_requested": _boolish(_field(config, "cache_latents")),
        "latent_cache_to_disk": _boolish(_field(config, "cache_latents_to_disk")),
        "text_cache_requested": text_requested,
        "text_cache_active": bool(text_active),
        "text_cache_to_disk": _boolish(_field(config, "cache_text_encoder_outputs_to_disk")),
        "text_conditioning_static": bool(text_static),
        "train_text_encoder": bool(train_text_encoder),
        "compile_active": bool(compile_active),
        "compile_shape_strategy": str(_field(config, "compile_shape_strategy", "auto") or "auto"),
        "compile_target_strategy": str(_field(config, "compile_target_strategy", "auto") or "auto"),
        "compile_static_shape_drop_last": _boolish(_field(config, "compile_static_shape_drop_last", True), default=True),
        "compile_require_cache_first": _boolish(_field(config, "compile_require_cache_first", True), default=True),
        "weight_compression_requested": bool(weight_compression_requested(_config_mapping_with_compile(config))),
        "blockers": blockers,
        "fingerprint_fields": {
            "shuffle_caption": _boolish(_field(config, "shuffle_caption")),
            "shuffle_caption_tags_only": _boolish(_field(config, "shuffle_caption_tags_only")),
            "caption_dropout_rate": _safe_float(_field(config, "caption_dropout_rate", 0.0), 0.0),
            "tag_dropout_rate": _safe_float(_field(config, "tag_dropout_rate", 0.0), 0.0),
            "max_token_length": _safe_int(_field(config, "max_token_length", 0), 0),
            "resolution": str(_field(config, "resolution", "") or ""),
            "train_batch_size": _safe_int(_field(config, "train_batch_size", _field(config, "batch_size", 1)), 1),
        },
    }


def _config_mapping_with_compile(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    keys = (
        "compile_runtime",
        "torch_compile",
        "compile_cache_enabled",
        "weight_compression_enabled",
        "fp8_base",
        "weight_compression_preset",
    )
    return {key: _field(config, key) for key in keys}


def build_model_acceleration_runtime_snapshot(
    config: Any,
    *,
    family: str = "",
    profile: str = "",
    runtime_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_family = family or normalize_model_family_from_config(config)
    cache_safety = build_model_acceleration_cache_safety(config, family=resolved_family, profile=profile)
    return {
        "family": resolved_family,
        "profile": str(profile or _field(config, "acceleration_profile", _field(config, "speed_profile", "")) or ""),
        "matrix_summary": acceleration_matrix_summary_for(resolved_family),
        "cache_safety": cache_safety,
        "benchmark_matrix": build_model_acceleration_benchmark_matrix(
            config,
            family=resolved_family,
            profile=profile,
            runtime_features=runtime_features,
            cache_safety=cache_safety,
            resolve_policy=False,
        ),
    }


__all__ = [
    "build_model_acceleration_cache_safety",
    "build_model_acceleration_runtime_snapshot",
    "normalize_model_family_from_config",
]
