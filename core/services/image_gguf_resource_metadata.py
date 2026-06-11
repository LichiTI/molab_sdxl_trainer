"""Lightweight metadata helpers for Lulynx image GGUF resource discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMAGE_GGUF_ARCH = "lulynx_image"


def read_image_gguf_sidecar(path: Path) -> tuple[dict[str, Any], str]:
    if path.suffix.lower() != ".gguf":
        return {}, ""
    candidate = path.with_suffix(path.suffix + ".manifest.json")
    try:
        if not candidate.is_file() or candidate.stat().st_size > 1024 * 1024:
            return {}, ""
        data = json.loads(candidate.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}, ""
    if not isinstance(data, dict) or not is_image_gguf_sidecar(data):
        return {}, ""
    return normalize_image_gguf_sidecar(data), "image_gguf_sidecar"


def is_image_gguf_sidecar(data: dict[str, Any]) -> bool:
    arch = str(data.get("gguf_arch") or data.get("general.architecture") or "").strip().lower()
    component = str(data.get("component") or data.get("lulynx.image_gguf.component") or "").strip()
    family = str(data.get("family") or data.get("lulynx.image_gguf.family") or "").strip()
    compatibility = str(data.get("compatibility") or data.get("lulynx.image_gguf.compatibility") or "").strip().lower()
    return arch == IMAGE_GGUF_ARCH or bool(component and family and compatibility.startswith("container_"))


def normalize_image_gguf_sidecar(data: dict[str, Any]) -> dict[str, Any]:
    component = str(data.get("component") or data.get("lulynx.image_gguf.component") or "").strip().lower()
    family = str(data.get("family") or data.get("lulynx.image_gguf.family") or "").strip().lower()
    compatibility = str(data.get("compatibility") or data.get("lulynx.image_gguf.compatibility") or "").strip().lower()
    schema = str(data.get("schema_version") or data.get("lulynx.image_gguf.schema") or "1")
    payload: dict[str, Any] = {
        "model_type": "image-gguf",
        "artifact_kind": "image_gguf",
        "model_family": family,
        "architecture": IMAGE_GGUF_ARCH,
        "format": "gguf",
        "gguf_arch": IMAGE_GGUF_ARCH,
        "lulynx.artifact_kind": "image_gguf",
        "lulynx.model_family": family,
        "lulynx.image_gguf.schema": schema,
        "lulynx.image_gguf.component": component,
        "lulynx.image_gguf.family": family,
        "lulynx.image_gguf.compatibility": compatibility,
    }
    for key in (
        "tensor_count", "gguf_file_type", "output_size_bytes", "converted_tensors", "skipped_tensors",
        "dtype_counts", "rank_counts",
    ):
        if key in data:
            payload[key] = data[key]
    if data.get("source_paths"):
        payload["lulynx.image_gguf.source_count"] = str(len(data.get("source_paths") or []))
    return {key: value for key, value in payload.items() if value is not None and value != ""}


__all__ = [
    "IMAGE_GGUF_ARCH",
    "is_image_gguf_sidecar",
    "normalize_image_gguf_sidecar",
    "read_image_gguf_sidecar",
]
