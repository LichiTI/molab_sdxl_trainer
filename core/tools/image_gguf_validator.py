"""Validation helpers for Lulynx image GGUF containers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from core.tools.image_gguf_shape_loader import GGUF_ARCH, load_image_gguf_shape_contract
except ImportError:
    from backend.core.tools.image_gguf_shape_loader import GGUF_ARCH, load_image_gguf_shape_contract


def validate_image_gguf_container(path: str | Path, *, sidecar_path: str | Path | None = None) -> dict[str, Any]:
    """Validate an image GGUF container through the Python reference shape loader."""
    return load_image_gguf_shape_contract(path, sidecar_path=sidecar_path)


__all__ = ["GGUF_ARCH", "validate_image_gguf_container"]
