"""Model-aware acceleration policy resolver.

The resolver translates an opt-in speed profile into conservative preflight
recommendations.  It is deliberately stdlib-only so routers, launchers, and
tests can import it without touching torch or model runtimes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .model_acceleration_adapters import lora_fa_policy_patch
from .model_acceleration_conflicts import active_compile_intent, weight_compression_requested
from .model_acceleration_data import apply_data_backend_policy, can_cache_text_encoder_outputs
from .model_acceleration_decision import AccelerationPolicyDecision
from .model_acceleration_flux import flux_low_vram_patch
from .model_acceleration_matrix import (
    adapter_init_patch_for,
    advanced_optimizer_strategy_for,
    attention_backend_for,
    cache_patch_for,
    checkpoint_patch_for,
    compile_policy_for,
    lora_recompute_mode_for,
    low_bit_patch_for,
    low_bit_preset_for,
    optimizer_backend_for,
    runtime_profile_for,
    should_cache_text_encoder_outputs,
    should_cache_text_encoder_outputs_to_disk,
)
from .model_acceleration_native_dit import native_dit_low_vram_patch
from .model_acceleration_patch import apply_typed_patch_map
from .model_acceleration_profiles import (
    PROFILE_AGGRESSIVE,
    PROFILE_BALANCED,
    PROFILE_LOW_VRAM,
    PROFILE_OFF,
    normalize_acceleration_profile,
)

_MISSING = object()

def _str(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return _str(value).lower().replace("-", "_").replace(" ", "_")


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


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_from_config(config: Mapping[str, Any]) -> str:
    if "acceleration_profile" in config:
        return normalize_acceleration_profile(config.get("acceleration_profile"))
    if "speed_profile" in config:
        return normalize_acceleration_profile(config.get("speed_profile"))
    return PROFILE_OFF


def _model_family(config: Mapping[str, Any], *, schema_id: str = "", training_type: str = "") -> str:
    text = " ".join(
        _str(value)
        for value in (
            schema_id,
            training_type,
            config.get("schema_id"),
            config.get("training_type"),
            config.get("model_type"),
            config.get("model_arch"),
            config.get("route"),
        )
        if _str(value)
    ).lower().replace("\\", "/")
    compact = text.replace("_", "-")
    if "newbie" in compact:
        return "newbie"
    if "anima" in compact or "qwen-image" in compact or "qwen_image" in text:
        return "anima"
    if "flux" in compact:
        return "flux"
    if "sdxl" in compact or "stable-diffusion-xl" in compact or "xl-lora" in compact:
        return "sdxl"
    if "sd15" in compact or "sd-1" in compact or "sd1.5" in compact or compact.startswith("sd-lora"):
        return "sd15"
    return "unknown"


def _is_lora_route(config: Mapping[str, Any], schema_id: str, training_type: str) -> bool:
    text = " ".join(
        _str(value)
        for value in (schema_id, training_type, config.get("schema_id"), config.get("training_type"))
    ).lower().replace("_", "-")
    return "lora" in text or not text


def _decision_is_lora_route(decision: AccelerationPolicyDecision) -> bool:
    text = f"{decision.schema_id} {decision.training_type}".lower().replace("_", "-")
    return "lora" in text or not text.strip()


def _same_value(current: Any, desired: Any) -> bool:
    if isinstance(desired, bool):
        return _flag(current, default=not desired) is desired
    return _norm(current) == _norm(desired)


def _patch_text(
    decision: AccelerationPolicyDecision,
    config: Mapping[str, Any],
    *,
    track: str,
    key: str,
    value: str,
    open_values: set[str] | None = None,
    message: str = "",
) -> None:
    current = config.get(key, _MISSING)
    if current is not _MISSING and _same_value(current, value):
        return
    allowed = open_values or {"", "auto", "default"}
    if current is _MISSING or current is None or _norm(current) in allowed:
        decision.add_patch(track, key, value, message)
        return
    decision.add_skip(track, key, current, "explicit value")


def _patch_bool(
    decision: AccelerationPolicyDecision,
    config: Mapping[str, Any],
    *,
    track: str,
    key: str,
    value: bool,
    allow_false_override: bool = True,
    message: str = "",
) -> None:
    current = config.get(key, _MISSING)
    if current is not _MISSING and _same_value(current, value):
        return
    if current is _MISSING or current is None or _norm(current) in {"", "auto", "default"}:
        decision.add_patch(track, key, value, message)
        return
    if allow_false_override and _flag(current, default=value) is not value:
        decision.add_patch(track, key, value, message)
        return
    decision.add_skip(track, key, current, "explicit value")


def _optimizer_family_allows_backend(config: Mapping[str, Any]) -> bool:
    raw = _norm(config.get("optimizer_type") or config.get("optimizer") or config.get("optimizer_name"))
    if not raw:
        return True
    return "adamw" in raw or raw in {"adam", "torch_adamw", "adamw8bit", "adamw_8bit"}


def _requested_vram_gb(config: Mapping[str, Any]) -> float | None:
    for key in ("gpu_vram_gb", "available_vram_gb", "vram_gb", "target_vram_gb"):
        value = _float_or_none(config.get(key))
        if value is not None and value > 0:
            return value
    return None


def _decision_recommends_compile(decision: AccelerationPolicyDecision) -> bool:
    return active_compile_intent(decision.recommended_config_patch)


def _patch_int(
    decision: AccelerationPolicyDecision,
    config: Mapping[str, Any],
    *,
    track: str,
    key: str,
    value: int,
) -> None:
    current = config.get(key, _MISSING)
    if current is _MISSING or current is None or _norm(current) in {"", "auto", "default"}:
        decision.add_patch(track, key, value)
    elif int(_float_or_none(current) or value) != value:
        decision.add_skip(track, key, current, "explicit value")


def _patch_value(
    decision: AccelerationPolicyDecision,
    config: Mapping[str, Any],
    *,
    track: str,
    key: str,
    value: Any,
    open_values: set[str] | None = None,
    allow_false_override: bool = True,
) -> None:
    if isinstance(value, bool):
        _patch_bool(decision, config, track=track, key=key, value=value, allow_false_override=allow_false_override)
    elif isinstance(value, int):
        _patch_int(decision, config, track=track, key=key, value=value)
    else:
        _patch_text(decision, config, track=track, key=key, value=str(value), open_values=open_values)


def _allow_acceleration_opt_in(config: Mapping[str, Any], *keys: str) -> bool:
    return any(_flag(config.get(key)) for key in keys)


def _apply_optimizer_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    profile = decision.effective_profile
    if profile == PROFILE_OFF or not _optimizer_family_allows_backend(config):
        if profile != PROFILE_OFF:
            decision.notes.append("Acceleration policy kept optimizer backend unchanged for non-AdamW optimizer.")
        return
    backend = optimizer_backend_for(decision.model_family, profile)
    if not backend:
        return
    _patch_text(
        decision,
        config,
        track="optimizer",
        key="optimizer_backend",
        value=backend,
        open_values={"", "auto", "default"},
        message="Use the faster torch AdamW backend when available; trainer fallback remains in force.",
    )


def _apply_cache_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    profile = decision.effective_profile
    family = decision.model_family
    if profile == PROFILE_OFF:
        return

    cache_spec = cache_patch_for(family, profile)
    for key, value in cache_spec.patch.items():
        track = "data" if key == "cached_collate_mode" else "cache"
        open_values = {"", "auto", "default", "standard"} if key == "native_cache_mode" else None
        _patch_value(decision, config, track=track, key=key, value=value, open_values=open_values)
    decision.notes.extend(cache_spec.notes)

    if should_cache_text_encoder_outputs(family, profile):
        if can_cache_text_encoder_outputs(config, family):
            _patch_bool(decision, config, track="cache", key="cache_text_encoder_outputs", value=True)
            if should_cache_text_encoder_outputs_to_disk(family, profile):
                _patch_bool(decision, config, track="cache", key="cache_text_encoder_outputs_to_disk", value=True)
        else:
            decision.add_skip("cache", "cache_text_encoder_outputs", config.get("cache_text_encoder_outputs"), "text conditioning is not static")


def _apply_runtime_profile_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    profile = decision.effective_profile
    family = decision.model_family
    if profile == PROFILE_AGGRESSIVE and not _decision_is_lora_route(decision):
        decision.notes.append("Aggressive native runtime profile is skipped for non-LoRA routes.")
        return
    runtime_profile = runtime_profile_for(family, profile)
    if runtime_profile:
        _patch_text(
            decision,
            config,
            track="runtime_profile",
            key="native_runtime_profile",
            value=runtime_profile,
            open_values={"", "auto", "default", "standard"},
        )


def _apply_attention_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    profile = decision.effective_profile
    family = decision.model_family
    if profile == PROFILE_OFF:
        return
    backend = attention_backend_for(family, profile)
    if not backend:
        return
    message = (
        "Prefer FlashAttention 2 on native DiT routes when the runtime supports it."
        if backend == "flash2"
        else "Use the model-family default attention backend with runtime fallback."
    )
    _patch_text(
        decision,
        config,
        track="attention",
        key="attention_backend",
        value=backend,
        open_values={"", "auto", "default"},
        message=message,
    )


def _apply_compile_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    profile = decision.effective_profile
    if profile != PROFILE_AGGRESSIVE:
        if profile == PROFILE_LOW_VRAM:
            decision.notes.append("Low-VRAM acceleration keeps compile disabled unless explicitly selected.")
        return
    if weight_compression_requested(config):
        decision.notes.append("Aggressive compile was skipped because frozen-weight compression is already requested.")
        decision.add_skip("compile", "compile_runtime", config.get("compile_runtime", "off"), "weight_compression requested")
        return
    if not _decision_is_lora_route(decision):
        decision.notes.append("Aggressive compile recommendations are skipped for non-LoRA routes.")
        return

    compile_spec = compile_policy_for(decision.model_family, profile)
    if compile_spec is None:
        decision.notes.append("Aggressive compile is skipped for this model family until route-specific guards exist.")
        return

    _patch_text(
        decision,
        config,
        track="compile",
        key="compile_runtime",
        value=compile_spec.runtime,
        open_values={"", "auto", "default", "off"},
    )
    _patch_text(decision, config, track="compile", key="compile_shape_strategy", value=compile_spec.shape_strategy)
    _patch_text(decision, config, track="compile", key="compile_target_strategy", value=compile_spec.target_strategy)
    for key, value in compile_spec.extra_patch.items():
        if isinstance(value, bool):
            _patch_bool(
                decision,
                config,
                track="compile",
                key=key,
                value=value,
                allow_false_override=False,
            )
        elif isinstance(value, int):
            _patch_int(decision, config, track="compile", key=key, value=value)
        else:
            _patch_text(decision, config, track="compile", key=key, value=str(value))
    decision.notes.extend(compile_spec.notes)


def _apply_checkpoint_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    profile = decision.effective_profile
    checkpoint_spec = checkpoint_patch_for(decision.model_family, profile)
    for key, value in checkpoint_spec.patch.items():
        open_values = {"", "auto", "default", "off"} if key == "checkpoint_policy" else None
        _patch_value(decision, config, track="checkpoint", key=key, value=value, open_values=open_values)
    decision.notes.extend(checkpoint_spec.notes)


def _apply_low_vram_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    if decision.effective_profile != PROFILE_LOW_VRAM:
        return
    family = decision.model_family
    if family in {"sdxl", "sd15"}:
        requested_vram = _requested_vram_gb(config)
        low_profile = "very_low_8g" if requested_vram is not None and requested_vram <= 9.0 else "low_12g"
        _patch_text(
            decision,
            config,
            track="low_vram",
            key="low_vram_profile",
            value=low_profile,
            open_values={"", "auto", "default", "off"},
        )
        _patch_bool(decision, config, track="low_vram", key="sdxl_low_vram_optimization", value=True)
    elif family == "flux":
        recommendation = flux_low_vram_patch()
        apply_typed_patch_map(
            patch=recommendation.patch,
            patch_bool=lambda key, value: _patch_bool(decision, config, track="low_vram", key=key, value=value),
            patch_int=lambda key, value: _patch_int(decision, config, track="low_vram", key=key, value=value),
            patch_text=lambda key, value: _patch_text(decision, config, track="low_vram", key=key, value=value, open_values={"", "auto", "default", "phase"}),
        )
        decision.notes.extend(recommendation.notes)
    elif family in {"anima", "newbie"}:
        recommendation = native_dit_low_vram_patch(family)
        apply_typed_patch_map(
            patch=recommendation.patch,
            patch_bool=lambda key, value: _patch_bool(decision, config, track="low_vram", key=key, value=value),
            patch_int=lambda key, value: _patch_int(decision, config, track="low_vram", key=key, value=value),
            patch_text=lambda key, value: _patch_text(
                decision,
                config,
                track="low_vram",
                key=key,
                value=value,
                open_values={"", "auto", "default", "standard", "resident", "off", "observe"},
            ),
        )
        decision.notes.extend(recommendation.notes)


def _apply_lora_memory_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    profile = decision.effective_profile
    family = decision.model_family
    if profile == PROFILE_OFF or family not in {"sdxl", "sd15", "anima", "newbie", "flux"}:
        return
    if not _decision_is_lora_route(decision):
        return
    mode = lora_recompute_mode_for(family, profile)
    if mode:
        _patch_text(
            decision,
            config,
            track="lora_recompute",
            key="lora_activation_recompute_mode",
            value=mode,
            open_values={"", "auto", "default", "off"} if mode == "on" else {"", "auto", "default"},
        )

    if profile == PROFILE_AGGRESSIVE and _flag(config.get("acceleration_allow_lora_fa")):
        adapter_patch = lora_fa_policy_patch(family, config)
        if not adapter_patch.supported:
            decision.notes.append(adapter_patch.reason)
            decision.add_skip("adapter", "lora_fa_enabled", config.get("lora_fa_enabled", False), "route unsupported")
            return
        for key, value in adapter_patch.patch.items():
            if isinstance(value, bool):
                _patch_bool(decision, config, track="adapter", key=key, value=value)
            else:
                _patch_text(
                    decision,
                    config,
                    track="adapter",
                    key=key,
                    value=str(value),
                    open_values={"", "auto", "default", "lora", "networks.lora"},
                )
        decision.notes.extend(adapter_patch.notes)
    elif profile in {PROFILE_BALANCED, PROFILE_AGGRESSIVE, PROFILE_LOW_VRAM}:
        decision.notes.append("LoRA-FA is available as a manual adapter choice; it is not auto-applied because it changes adapter semantics.")


def _apply_convergence_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    profile = decision.effective_profile
    if profile == PROFILE_OFF or not _decision_is_lora_route(decision):
        return
    if _allow_acceleration_opt_in(
        config,
        "acceleration_allow_lora_plus",
        "acceleration_allow_advanced_optimizer",
        "acceleration_allow_convergence",
    ):
        strategy = advanced_optimizer_strategy_for(decision.model_family, profile)
        if strategy:
            _patch_text(
                decision,
                config,
                track="advanced_optimizer",
                key="advanced_optimizer_strategy",
                value=strategy,
                open_values={"", "auto", "default"},
                message="Use the model-family convergence optimizer strategy only after explicit opt-in.",
            )
    elif profile in {PROFILE_BALANCED, PROFILE_AGGRESSIVE, PROFILE_LOW_VRAM}:
        decision.notes.append("LoRA+/RS-LoRA/GaLore-style convergence strategies stay manual unless explicitly allowed.")

    if _allow_acceleration_opt_in(
        config,
        "acceleration_allow_adapter_init",
        "acceleration_allow_pissa",
        "acceleration_allow_convergence",
    ):
        init_spec = adapter_init_patch_for(decision.model_family, profile)
        for key, value in init_spec.patch.items():
            _patch_value(
                decision,
                config,
                track="adapter_init",
                key=key,
                value=value,
                allow_false_override=False,
            )
        decision.notes.extend(init_spec.notes)
    elif profile in {PROFILE_BALANCED, PROFILE_AGGRESSIVE, PROFILE_LOW_VRAM}:
        decision.notes.append("PiSSA/OLoRA/LoftQ adapter init stays manual unless explicitly allowed.")


def _apply_weight_compression_policy(decision: AccelerationPolicyDecision, config: Mapping[str, Any]) -> None:
    profile = decision.effective_profile
    if profile == PROFILE_OFF:
        return
    allow = _flag(config.get("acceleration_allow_low_bit")) or _flag(config.get("acceleration_allow_weight_compression"))
    if not allow:
        decision.notes.append("Low-bit frozen-weight compression is left as a manual/explicit option for quality safety.")
        return
    if profile in {PROFILE_AGGRESSIVE, PROFILE_LOW_VRAM}:
        if active_compile_intent(config) or _decision_recommends_compile(decision):
            decision.notes.append("Low-bit frozen-weight compression was skipped because torch.compile is active or recommended.")
            decision.add_skip(
                "weight_compression",
                "weight_compression_enabled",
                config.get("weight_compression_enabled", False),
                "compile active",
            )
            return
        low_bit_spec = low_bit_patch_for(decision.model_family, profile)
        decision.notes.extend(low_bit_spec.notes)
        preset = str(low_bit_spec.patch.get("weight_compression_preset") or low_bit_preset_for(decision.model_family))
        if not preset or preset == "off":
            decision.add_skip(
                "weight_compression",
                "weight_compression_preset",
                config.get("weight_compression_preset", "off"),
                "route has no low-bit preset",
            )
            return
        _patch_bool(decision, config, track="weight_compression", key="weight_compression_enabled", value=True)
        for key, value in low_bit_spec.patch.items():
            _patch_value(
                decision,
                config,
                track="weight_compression",
                key=key,
                value=value,
                open_values={"", "auto", "default", "off", "none"},
            )


def resolve_model_acceleration_policy(
    config: Mapping[str, Any],
    *,
    schema_id: str = "",
    training_type: str = "",
    data_dir: str | Path | None = None,
) -> AccelerationPolicyDecision:
    """Resolve an opt-in speed profile into route-safe config recommendations."""

    profile = _profile_from_config(config)
    family = _model_family(config, schema_id=schema_id, training_type=training_type)
    sid = schema_id or _str(config.get("schema_id"))
    tt = training_type or _str(config.get("training_type"))
    decision = AccelerationPolicyDecision(profile, profile, family, sid, tt)

    if profile == PROFILE_OFF:
        return decision

    decision.notes.append(f"Acceleration profile {profile} resolved for {family}.")
    if not _is_lora_route(config, sid, tt) and profile in {PROFILE_AGGRESSIVE, PROFILE_LOW_VRAM}:
        decision.warnings.append("Acceleration policy is tuned first for LoRA/native DiT routes; non-LoRA routes keep conservative recommendations.")

    _apply_cache_policy(decision, config)
    _apply_optimizer_policy(decision, config)
    _apply_runtime_profile_policy(decision, config)
    _apply_attention_policy(decision, config)
    _apply_checkpoint_policy(decision, config)
    _apply_low_vram_policy(decision, config)
    _apply_compile_policy(decision, config)
    _apply_lora_memory_policy(decision, config)
    _apply_convergence_policy(decision, config)
    _apply_weight_compression_policy(decision, config)
    apply_data_backend_policy(
        decision,
        config,
        data_dir=data_dir,
        patch_text=lambda key, value, message: _patch_text(decision, config, track="data_backend", key=key, value=value, open_values={"", "auto", "default", "caption", "raw"}, message=message),
    )
    return decision
