"""TrainingLoop kwargs for default-off TurboCore native-update route binding."""

from __future__ import annotations

from typing import Any


ROUTE_BINDING_KWARGS = (
    "turbocore_native_update_mode",
    "turbocore_native_update_dispatch_enabled",
    "turbocore_native_update_training_path_enabled",
    "turbocore_native_update_require_native_cuda",
)


def build_turbocore_native_update_training_loop_kwargs(config: Any) -> dict[str, Any]:
    return {
        "turbocore_native_update_mode": str(
            getattr(config, "turbocore_native_update_mode", "off") or "off"
        ),
        "turbocore_native_update_dispatch_enabled": bool(
            getattr(config, "turbocore_native_update_dispatch_enabled", False)
        ),
        "turbocore_native_update_training_path_enabled": bool(
            getattr(config, "turbocore_native_update_training_path_enabled", False)
        ),
        "turbocore_native_update_require_native_cuda": bool(
            getattr(config, "turbocore_native_update_require_native_cuda", False)
        ),
    }


__all__ = ["ROUTE_BINDING_KWARGS", "build_turbocore_native_update_training_loop_kwargs"]
