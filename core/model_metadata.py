"""Clean-room model metadata read/write helpers."""

from __future__ import annotations

import importlib
import json
import os
import struct
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def _native_metadata_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_METADATA", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _native_metadata_read_enabled() -> bool:
    return str(os.environ.get("LULYNX_ENABLE_NATIVE_METADATA_READ", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def _load_native_metadata_api() -> Any:
    artifact_dir = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if artifact_dir and artifact_dir not in sys.path and Path(artifact_dir).is_dir():
        sys.path.insert(0, artifact_dir)
    try:
        return importlib.import_module("lulynx_native")
    except Exception:
        return None


def _native_metadata_api() -> Any:
    if _native_metadata_disabled():
        return None
    native = _load_native_metadata_api()
    if not hasattr(native, "read_safetensors_metadata"):
        return None
    if not hasattr(native, "patch_safetensors_metadata"):
        return None
    return native


def read_model_metadata(path: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if target.suffix.lower() == ".safetensors":
        native = _native_metadata_api() if _native_metadata_read_enabled() else None
        if native is not None:
            return native.read_safetensors_metadata(str(target))
        return _read_safetensors_metadata(target)
    if target.suffix.lower() in {".ckpt", ".pt", ".pth"}:
        return {"format": target.suffix.lower().lstrip("."), "metadata": {}, "editable": False}
    return {"format": target.suffix.lower().lstrip(".") or "unknown", "metadata": {}, "editable": False}


def patch_safetensors_metadata(input_path: str, output_path: str, updates: dict[str, Any], remove_existing: bool = False) -> dict[str, Any]:
    src = Path(input_path)
    dst = Path(output_path)
    if src.suffix.lower() != ".safetensors":
        raise ValueError("Only safetensors metadata patching is supported")
    normalized_updates = {str(k): str(v) for k, v in updates.items()}
    native = _native_metadata_api()
    if native is not None and dst.suffix.lower() == ".safetensors":
        return native.patch_safetensors_metadata(
            str(src),
            str(dst),
            json.dumps(normalized_updates, ensure_ascii=False),
            bool(remove_existing),
        )
    header, payload_offset = _read_header(src)
    metadata = {} if remove_existing else dict(header.get("__metadata__", {}) or {})
    metadata.update(normalized_updates)
    header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f_in, dst.open("wb") as f_out:
        f_in.seek(payload_offset)
        f_out.write(struct.pack("<Q", len(encoded)))
        f_out.write(encoded)
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)
    return read_model_metadata(str(dst))


def _read_safetensors_metadata(path: Path) -> dict[str, Any]:
    header, _ = _read_header(path)
    tensors = {
        key: value
        for key, value in header.items()
        if key != "__metadata__" and isinstance(value, dict)
    }
    return {
        "format": "safetensors",
        "metadata": header.get("__metadata__", {}) or {},
        "tensor_count": len(tensors),
        "tensors": list(tensors.keys())[:200],
        "editable": True,
    }


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError("Invalid safetensors header")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header, 8 + int(header_len)

