"""Optional native rowwise quantization bridge with torch fallback safety."""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


def native_rowwise_enabled() -> bool:
    value = os.environ.get("LULYNX_NATIVE_ROWWISE_QUANT", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def native_rowwise_api() -> Any | None:
    if not native_rowwise_enabled():
        return None
    _extend_native_search_path()
    try:
        native = importlib.import_module("lulynx_native")
    except Exception:
        return None
    if not hasattr(native, "quantize_rowwise_f32_bytes"):
        return None
    return native


def _extend_native_search_path() -> None:
    backend_root = Path(__file__).resolve().parents[2]
    for candidate in (
        backend_root / "native" / "target" / "release",
        backend_root / "native" / "target" / "debug",
    ):
        if not candidate.is_dir() or not (candidate / "lulynx_native.pyd").is_file():
            continue
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def native_rowwise_capability() -> dict[str, Any]:
    native = native_rowwise_api()
    if native is None:
        return {
            "available": False,
            "provider": "torch_fallback",
            "reason": "lulynx_native rowwise quantization is not available",
        }
    try:
        payload = native.rowwise_quant_capability()
        result = dict(payload) if isinstance(payload, dict) else {"available": True, "provider": "lulynx_native.rowwise_quant_v1"}
        formats = result.get("formats")
        if isinstance(formats, (list, tuple)):
            result["formats"] = [item for item in formats if str(item) != "lulynx_uint4_rowwise"]
        result["uint4_provider"] = "torch.affine_uint4_blockwise_v2"
        result["uint4_native_disabled_reason"] = "bundled native kernel emits the older symmetric uint4 layout"
        return result
    except Exception as exc:
        return {"available": False, "provider": "torch_fallback", "reason": str(exc)}


def try_quantize_rowwise_tensor_native(
    tensor: torch.Tensor,
    fmt: str,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]] | None:
    if str(fmt or "").strip().lower().replace("-", "_") == "lulynx_uint4_rowwise":
        # UINT4 now uses the Python affine blockwise v2 path.  The currently
        # bundled native kernel still emits the older symmetric layout, so let
        # Python own new UINT4 artifacts until native exposes offset payloads.
        return None
    native = native_rowwise_api()
    if native is None:
        return None
    shape = tuple(int(dim) for dim in tensor.shape)
    if len(shape) < 2 or tensor.numel() <= 0:
        return None
    rows = shape[0]
    flat = tensor.detach().cpu().float().contiguous().view(rows, -1)
    cols = int(flat.shape[1])
    try:
        payload = native.quantize_rowwise_f32_bytes(flat.numpy().tobytes(order="C"), rows, cols, fmt)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        q_bytes = bytes(payload["q_bytes"])
        scale_bytes = bytes(payload["scale_bytes"])
        original_cols = int(payload.get("original_cols") or cols)
        if fmt == "lulynx_int8_rowwise":
            q = torch.frombuffer(bytearray(q_bytes), dtype=torch.int8).clone().view(rows, cols).contiguous()
        elif fmt == "lulynx_uint4_rowwise":
            packed_cols = (cols + 1) // 2
            q = torch.frombuffer(bytearray(q_bytes), dtype=torch.uint8).clone().view(rows, packed_cols).contiguous()
        else:
            return None
        scale = torch.frombuffer(bytearray(scale_bytes), dtype=torch.float16).clone().view(rows, 1).contiguous()
    except Exception:
        return None
    metadata = {
        "provider": str(payload.get("provider") or "lulynx_native.rowwise_quant_v1"),
        "kernel": str(payload.get("kernel") or "native"),
    }
    return q, scale, original_cols, metadata


__all__ = [
    "native_rowwise_api",
    "native_rowwise_capability",
    "native_rowwise_enabled",
    "try_quantize_rowwise_tensor_native",
]
