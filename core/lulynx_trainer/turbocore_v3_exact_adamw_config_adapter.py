"""Config/request adapter for the V3 exact AdamW native canary."""

from __future__ import annotations

from typing import Any, MutableMapping


EXACT_ADAMW_OPTIMIZERS = {"adamw"}
ALLOWED_BACKENDS = {"auto", "torch_adamw", "foreach_adamw", "torch_fused"}


def apply_v3_exact_adamw_canary_config_adapter(config: MutableMapping[str, Any]) -> dict[str, Any]:
    """Resolve the explicit V3 exact AdamW canary request into existing fields.

    This helper intentionally only reacts to the new high-level canary request
    fields.  It does not create a new training entry and does not silently
    promote legacy low-level flags.
    """

    requested = _requested(config)
    optimizer = _optimizer_key(config.get("optimizer_type"))
    backend = _backend_key(config.get("optimizer_backend", "auto"))
    exact_adamw = optimizer in EXACT_ADAMW_OPTIMIZERS
    backend_allowed = backend in ALLOWED_BACKENDS
    blockers: list[str] = []
    if requested and not exact_adamw:
        blockers.append("v3_exact_adamw_canary_requires_optimizer_type_adamw")
    if requested and not backend_allowed:
        blockers.append("v3_exact_adamw_canary_backend_not_allowed")
    allowed = bool(requested and exact_adamw and backend_allowed)
    if requested:
        if allowed:
            config["turbocore_native_update_mode"] = "native_experimental"
            config["turbocore_native_update_dispatch_enabled"] = True
            config["turbocore_native_update_training_path_enabled"] = True
            config["turbocore_native_update_require_native_cuda"] = True
        else:
            config["turbocore_native_update_mode"] = "off"
            config["turbocore_native_update_dispatch_enabled"] = False
            config["turbocore_native_update_training_path_enabled"] = False
            config["turbocore_native_update_require_native_cuda"] = False
    return {
        "schema_version": 1,
        "adapter": "v3_exact_adamw_canary_config_adapter_v0",
        "requested": requested,
        "allowed": allowed,
        "default_off": not requested,
        "optimizer_type": str(config.get("optimizer_type") or ""),
        "optimizer_key": optimizer,
        "optimizer_exact_adamw": exact_adamw,
        "optimizer_backend": backend,
        "backend_allowed": backend_allowed,
        "resolved_fields": {
            "turbocore_native_update_mode": str(config.get("turbocore_native_update_mode", "off") or "off"),
            "turbocore_native_update_dispatch_enabled": bool(
                config.get("turbocore_native_update_dispatch_enabled", False)
            ),
            "turbocore_native_update_training_path_enabled": bool(
                config.get("turbocore_native_update_training_path_enabled", False)
            ),
            "turbocore_native_update_require_native_cuda": bool(
                config.get("turbocore_native_update_require_native_cuda", False)
            ),
        },
        "blocked_reasons": _dedupe(blockers),
    }


def _requested(config: MutableMapping[str, Any]) -> bool:
    return bool(
        _boolish(config.get("turbocore_exact_adamw_canary"), False)
        or _boolish(config.get("turbocore_native_update_canary"), False)
        or _optimizer_key(config.get("turbocore_native_update_canary_optimizer")) == "exact_adamw"
    )


def _optimizer_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.replace(" ", "").replace("-", "_")


def _backend_key(value: Any) -> str:
    return str(value or "auto").strip().lower().replace("-", "_").replace(" ", "")


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["apply_v3_exact_adamw_canary_config_adapter"]
