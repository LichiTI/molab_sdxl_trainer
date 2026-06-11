"""Runtime-facing attention backend/profile summary helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


def _profile_value(profile: Any, key: str, default: Any = None) -> Any:
    if isinstance(profile, Mapping):
        return profile.get(key, default)
    return getattr(profile, key, default)


def _list_tail(value: Any, limit: int = 8) -> list[str]:
    if not value:
        return []
    try:
        items = list(value)
    except TypeError:
        items = [value]
    if limit > 0:
        items = items[-limit:]
    return [str(item) for item in items]


def attention_profile_to_dict(profile: Any) -> dict[str, Any]:
    if profile is None:
        return {
            "enabled": False,
            "active": False,
        }
    if is_dataclass(profile):
        data = asdict(profile)
    elif isinstance(profile, Mapping):
        data = dict(profile)
    else:
        data = {
            "enabled": bool(getattr(profile, "enabled", False)),
            "window_size": int(getattr(profile, "window_size", 0) or 0),
            "backend": str(getattr(profile, "backend", "auto") or "auto"),
            "torch_fallback_max_tokens": int(getattr(profile, "torch_fallback_max_tokens", 2048) or 2048),
            "launcher_attention_backend": str(getattr(profile, "launcher_attention_backend", "auto") or "auto"),
            "flex_runtime_active": bool(getattr(profile, "flex_runtime_active", False)),
        }
    enabled = bool(data.get("enabled", False))
    window_size = int(data.get("window_size", 0) or 0)
    data["enabled"] = enabled
    data["window_size"] = window_size
    data["active"] = enabled and window_size > 0
    return data


def build_attention_runtime_profile(
    *,
    config: Any,
    runtime_plan: Any = None,
    model_arch: str = "",
    route: str = "",
    profile: Any = None,
    patched: int = 0,
    patch_target: str = "",
    applied: bool | None = None,
    skip_reason: str = "",
    error: str = "",
    source: str = "trainer_runtime",
) -> dict[str, Any]:
    """Build a compact attention runtime profile for manifests and loop state."""

    requested_backend = str(getattr(runtime_plan, "requested_attention_backend", "") or "")
    resolved_backend = str(getattr(runtime_plan, "attention_backend", "") or "")
    if not requested_backend:
        requested_backend = str(getattr(config, "attention_backend", "auto") or "auto")
    if not resolved_backend:
        resolved_backend = requested_backend

    attention_profile = attention_profile_to_dict(profile)
    patched_count = int(patched or 0)
    if applied is None:
        applied = patched_count > 0

    result: dict[str, Any] = {
        "source": source,
        "model_arch": str(model_arch or getattr(config, "model_arch", "") or ""),
        "route": str(route or model_arch or getattr(config, "model_arch", "") or ""),
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "sdpa_backend_policy": str(getattr(runtime_plan, "sdpa_backend_policy", "") or ""),
        "attention_split_chunks": int(getattr(runtime_plan, "attention_split_chunks", 0) or 0),
        "attention_early_deletion": bool(getattr(runtime_plan, "attention_early_deletion", False)),
        "amd_sdpa_slice_trigger_gb": float(getattr(runtime_plan, "amd_sdpa_slice_trigger_gb", 0.0) or 0.0),
        "amd_sdpa_slice_target_gb": float(getattr(runtime_plan, "amd_sdpa_slice_target_gb", 0.0) or 0.0),
        "attention_profile": attention_profile,
        "profile_active": bool(attention_profile.get("active", False)),
        "patch_target": str(patch_target or ""),
        "patched_module_count": patched_count,
        "applied": bool(applied),
        "warnings": _list_tail(getattr(runtime_plan, "warnings", [])),
        "reasons": _list_tail(getattr(runtime_plan, "reasons", [])),
    }
    if skip_reason:
        result["skip_reason"] = str(skip_reason)
    if error:
        result["error"] = str(error)
        result["applied"] = False
    return result


__all__ = [
    "attention_profile_to_dict",
    "build_attention_runtime_profile",
]
