"""LoRA activation recompute mode resolver shared by native routes."""

from __future__ import annotations

from typing import Any


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def resolve_lora_activation_recompute(config: Any, *, auto_default: bool = False) -> bool:
    mode = str(getattr(config, "lora_activation_recompute_mode", "auto") or "auto").strip().lower()
    if mode in {"on", "true", "1", "yes", "enabled"}:
        return True
    if mode in {"off", "false", "0", "no", "disabled"}:
        return False
    return _truthy(getattr(config, "lora_activation_recompute", False)) or bool(auto_default)


__all__ = ["resolve_lora_activation_recompute"]
