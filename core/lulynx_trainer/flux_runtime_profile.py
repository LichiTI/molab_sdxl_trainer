"""Flux LoRA runtime profile helpers.

The preview Flux trainer has its own lightweight loop, so it cannot reuse the
main TrainingLoop profile directly.  This module keeps the observable contract
small and route-specific while still exposing the same evidence shape to state,
events, and run manifests.
"""

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


def _field(source: Any, name: str, default: Any = "") -> Any:
    try:
        return getattr(source, name, default)
    except Exception:
        return default


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _device_type(device: Any) -> str:
    return str(getattr(device, "type", device) or "")


def _dtype_name(dtype: Any) -> str:
    text = str(dtype or "")
    return text.replace("torch.", "")


def _active(name: str, profile: Mapping[str, Any]) -> bool:
    if not profile:
        return False
    if name == "compile_runtime":
        return _boolish(profile.get("applied", False)) or _int(profile.get("compiled_targets", 0)) > 0
    if name == "attention_runtime":
        return _boolish(profile.get("applied", False)) or _int(profile.get("patched_processors", 0)) > 0
    if name == "adapter_runtime":
        return _boolish(profile.get("enabled", False)) or _int(profile.get("injected_layer_count", 0)) > 0
    if name == "optimizer_backend":
        return bool(str(profile.get("resolved", "") or profile.get("optimizer_type", "") or ""))
    if name == "cache_runtime":
        return _boolish(profile.get("enabled", False))
    if name == "component_offload":
        resolved = str(profile.get("resolved", "") or "").strip().lower()
        return _boolish(profile.get("enabled", False)) or resolved not in {"", "resident", "none", "off", "disabled"}
    if name in {"transformer_offload", "gradient_checkpointing", "lora_activation_recompute"}:
        return _boolish(profile.get("enabled", False))
    return _boolish(profile.get("active", profile.get("enabled", False)))


def _collect_active(profiles: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [name for name, profile in profiles.items() if _active(name, profile)]


def _collect_warnings(profiles: Mapping[str, Mapping[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for name, profile in profiles.items():
        reason = str(profile.get("fallback_reason", "") or profile.get("skip_reason", "") or "")
        if reason:
            warnings.append(f"{name}: {reason}")
        for item in profile.get("warnings", []) or []:
            warnings.append(f"{name}: {item}")
    return warnings[-12:]


def build_flux_runtime_profile(
    *,
    config: Any,
    training_loop: Any = None,
    device: Any = None,
    weight_dtype: Any = None,
    memory_state: Mapping[str, Any] | None = None,
    component_offload_profile: Mapping[str, Any] | None = None,
    gradient_checkpointing_profile: Mapping[str, Any] | None = None,
    attention_backend_profile: Mapping[str, Any] | None = None,
    compile_runtime_profile: Mapping[str, Any] | None = None,
    adapter_runtime_profile: Mapping[str, Any] | None = None,
    optimizer_backend_profile: Mapping[str, Any] | None = None,
    data_backend_profile: Mapping[str, Any] | None = None,
    cache_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = _dict(memory_state or _field(training_loop, "memory_optimization_state", {}))
    component = _dict(component_offload_profile or state.get("component_offload", {}))
    transformer = _dict(state.get("transformer_offload", {}))
    checkpointing = _dict(gradient_checkpointing_profile or state.get("gradient_checkpointing", {}))
    attention = _dict(attention_backend_profile or state.get("attention_backend", {}))
    compile_runtime = _dict(compile_runtime_profile or state.get("compile_runtime", {}))
    adapter = _dict(adapter_runtime_profile or state.get("adapter_runtime", {}))
    optimizer = _dict(optimizer_backend_profile or state.get("optimizer_backend", {}))
    data_backend = _dict(data_backend_profile or state.get("data_backend", {}))
    cache_runtime = _dict(cache_profile or {})
    recompute = _dict(state.get("lora_activation_recompute", {}))

    profiles: dict[str, Mapping[str, Any]] = {
        "component_offload": component,
        "transformer_offload": transformer,
        "gradient_checkpointing": checkpointing,
        "attention_runtime": attention,
        "compile_runtime": compile_runtime,
        "adapter_runtime": adapter,
        "optimizer_backend": optimizer,
        "data_backend": data_backend,
        "cache_runtime": cache_runtime,
        "lora_activation_recompute": recompute,
    }
    active = _collect_active(profiles)

    profile: dict[str, Any] = {
        "profile": "flux_lora_runtime_profile_v0",
        "enabled": True,
        "source": "flux_lora_preview",
        "model_arch": "flux",
        "training_type": str(_field(config, "training_type", "lora") or "lora"),
        "schema_id": str(_field(config, "schema_id", "flux-lora") or "flux-lora"),
        "device": _device_type(device or _field(config, "device", "")),
        "weight_dtype": _dtype_name(weight_dtype),
        "global_step": _int(_field(training_loop, "global_step", 0)),
        "total_steps": _int(_field(training_loop, "total_steps", 0)),
        "component_offload_strategy": str(
            state.get("component_offload_strategy", component.get("resolved", "resident")) or "resident"
        ),
        "mode": str(state.get("mode", "none") or "none"),
        "active_accelerations": active,
        "component_offload": component,
        "transformer_offload": transformer,
        "gradient_checkpointing": checkpointing,
        "attention_runtime": attention,
        "compile_runtime": compile_runtime,
        "adapter_runtime": adapter,
        "optimizer_backend": optimizer,
        "data_backend": data_backend,
        "cache_runtime": cache_runtime,
        "lora_activation_recompute": recompute,
        "warnings": _collect_warnings(profiles),
    }
    return profile


def attach_flux_runtime_profile_to_state(state: dict[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(state, dict) and profile:
        state["flux_runtime"] = dict(profile)
    return dict(profile or {})


def refresh_flux_runtime_profile(trainer: Any) -> dict[str, Any]:
    cache = _field(trainer, "_cache", None)
    training_loop = _field(trainer, "training_loop", None)
    memory_state = _field(training_loop, "memory_optimization_state", {})
    profile = build_flux_runtime_profile(
        config=_field(trainer, "config", None),
        training_loop=training_loop,
        device=_field(trainer, "device", None),
        weight_dtype=_field(trainer, "weight_dtype", None),
        memory_state=memory_state,
        component_offload_profile=_field(trainer, "_component_offload_profile", {}),
        gradient_checkpointing_profile=_field(trainer, "_gradient_checkpointing_profile", {}),
        attention_backend_profile=_field(trainer, "_attention_backend_profile", {}),
        compile_runtime_profile=_field(trainer, "_compile_runtime_profile", {}),
        adapter_runtime_profile=_field(trainer, "_adapter_runtime_profile", {}),
        optimizer_backend_profile=_field(trainer, "_optimizer_backend_profile", {}),
        data_backend_profile=_field(trainer, "_data_backend_profile", {}),
        cache_profile=dict(cache.profile) if cache is not None else {},
    )
    try:
        trainer._flux_runtime_profile = profile
    except Exception:
        pass
    if isinstance(memory_state, dict):
        attach_flux_runtime_profile_to_state(memory_state, profile)
    return dict(profile)


def build_flux_trainer_runtime_features(trainer: Any) -> dict[str, Any]:
    flux_runtime = refresh_flux_runtime_profile(trainer)
    training_loop = _field(trainer, "training_loop", None)
    features: dict[str, Any] = {
        "flux_runtime": flux_runtime,
        "memory_optimization": dict(_field(training_loop, "memory_optimization_state", {}) or {}),
    }
    for key, attr in (
        ("attention_runtime", "_attention_backend_profile"),
        ("compile_runtime", "_compile_runtime_profile"),
        ("adapter_runtime", "_adapter_runtime_profile"),
        ("optimizer_backend", "_optimizer_backend_profile"),
        ("data_backend", "_data_backend_profile"),
    ):
        profile = _dict(_field(trainer, attr, {}))
        if profile:
            features[key] = profile
    cache = _field(trainer, "_cache", None)
    if cache is not None:
        features["cache_runtime"] = dict(cache.profile)
    return features


__all__ = [
    "attach_flux_runtime_profile_to_state",
    "build_flux_runtime_profile",
    "build_flux_trainer_runtime_features",
    "refresh_flux_runtime_profile",
]
