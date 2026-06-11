# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Helpers for loading safetensors files while honouring ``disable_mmap_load_safetensors``.

Safetensors normally maps the file with mmap so the OS can stream tensors lazily.
That is the right default on local SSDs but causes pathological random-read
patterns on:

  * Network drives (SMB/NFS) where each random read costs a round-trip
  * Spinning HDDs with high seek latency
  * Some Windows path layouts that disable readahead for memory-mapped files

When ``disable_mmap=True`` we materialise the file fully into a single ``bytes``
buffer and parse it via ``safetensors.torch.load`` so the OS can perform a
single contiguous sequential read.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .lulynx_quantized_safetensors import (
    QUANT_PREFIX,
    decode_quantized_tensor,
    dequantize_state_dict,
    decode_dtype_to_torch,
    is_lulynx_quantized_metadata,
    parse_quantized_tensor_entries,
)

PathLike = Union[str, os.PathLike]


def load_safetensors(
    path: PathLike,
    *,
    device: Optional[str] = None,
    disable_mmap: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load a safetensors file into a flat ``{name: tensor}`` dict.

    Parameters
    ----------
    path
        Path to the ``.safetensors`` file.
    device
        Optional device string forwarded to ``safetensors.torch.load_file``;
        ignored in mmap-disabled mode (the bytes are decoded on CPU first
        and the caller is expected to ``.to(device)`` if needed).
    disable_mmap
        When ``True``, read the entire file into RAM with a single sequential
        read and then parse with ``safetensors.torch.load``. When ``False``
        (default), use ``safetensors.torch.load_file`` which maps the file.
    """
    path_str = str(Path(path))
    metadata = _read_metadata(path_str)
    state = _load_raw_safetensors(path_str, device=device, disable_mmap=disable_mmap)
    if is_lulynx_quantized_metadata(metadata):
        return dequantize_state_dict(state, metadata)
    return state


def _load_raw_safetensors(
    path: PathLike,
    *,
    device: Optional[str] = None,
    disable_mmap: bool = False,
) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load as st_load
    from safetensors.torch import load_file as st_load_file

    path_str = str(Path(path))
    if not disable_mmap:
        if device is None:
            return st_load_file(path_str)
        return st_load_file(path_str, device=device)

    with open(path_str, "rb") as f:
        data = f.read()
    state = st_load(data)
    if device is not None and device != "cpu":
        state = {k: v.to(device) for k, v in state.items()}
    return state


def _read_metadata(path: PathLike) -> Dict[str, str]:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except Exception:
        return {}


def open_safetensors(
    path: PathLike,
    *,
    framework: str = "pt",
    device: str = "cpu",
    disable_mmap: bool = False,
) -> Any:
    """Return either a ``safe_open`` context handle or an in-memory shim.

    The shim implements the subset of the ``safe_open`` API actually used by
    this project: ``keys()``, ``get_tensor(name)``, and ``metadata()``. When
    ``disable_mmap`` is True we eagerly load the file into a dict, then expose
    a context-manager façade so call sites need no branching.
    """
    metadata = _read_metadata(path)
    quantized = is_lulynx_quantized_metadata(metadata)
    if framework != "pt" and quantized:
        raise ValueError("Lulynx quantized safetensors can only be opened with framework='pt'")

    if not disable_mmap and not quantized:
        from safetensors import safe_open

        return safe_open(str(path), framework=framework, device=device)

    if not quantized:
        state = _load_raw_safetensors(path, device=device, disable_mmap=True)
        return _InMemorySafeOpen(state, metadata)

    if disable_mmap:
        state = _load_raw_safetensors(path, device=device, disable_mmap=True)
        raw = _InMemorySafeOpen(state, metadata)
    else:
        from safetensors import safe_open

        raw = safe_open(str(path), framework=framework, device=device)
    return _QuantizedSafeOpen(raw, metadata)


class _InMemorySafeOpen:
    """Context-manager shim over a fully-materialised state dict."""

    def __init__(self, state: Dict[str, torch.Tensor], metadata: Dict[str, str]) -> None:
        self._state = state
        self._metadata = metadata

    def __enter__(self) -> "_InMemorySafeOpen":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._state = {}

    def keys(self):
        return self._state.keys()

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._state[name]

    def get_slice(self, name: str) -> "_InMemorySliceView":
        return _InMemorySliceView(self._state[name])

    def metadata(self) -> Dict[str, str]:
        return dict(self._metadata)


class _InMemorySliceView:
    """Mimic the subset of ``safe_open(...).get_slice`` used by this project."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def get_shape(self):
        return list(self._tensor.shape)

    def get_dtype(self):
        return str(self._tensor.dtype).replace("torch.", "")


class _QuantizedSafeOpen:
    """Expose Lulynx rowwise safetensors through the usual safe_open shape."""

    def __init__(self, raw: Any, metadata: Dict[str, str]) -> None:
        self._raw = raw
        self._metadata = metadata
        self._entries = {entry.key: entry for entry in parse_quantized_tensor_entries(metadata)}
        self._reserved = {
            raw_key
            for entry in self._entries.values()
            for raw_key in (entry.q_key, entry.scale_key)
        }
        self._reserved.update(entry.offset_key for entry in self._entries.values() if entry.offset_key)
        self._handle: Any = None

    def __enter__(self) -> "_QuantizedSafeOpen":
        self._handle = self._raw.__enter__() if hasattr(self._raw, "__enter__") else self._raw
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self._raw, "__exit__"):
            self._raw.__exit__(exc_type, exc, tb)
        self._handle = None

    @property
    def _source(self) -> Any:
        return self._handle if self._handle is not None else self._raw

    def keys(self):
        visible = [key for key in self._source.keys() if self._is_visible_raw_key(str(key))]
        for key in self._entries:
            if key not in visible:
                visible.append(key)
        return visible

    def get_tensor(self, name: str) -> torch.Tensor:
        entry = self._entries.get(name)
        if entry is None:
            return self._source.get_tensor(name)
        payload = {
            "q": self._source.get_tensor(entry.q_key),
            "scale": self._source.get_tensor(entry.scale_key),
            "offset": self._source.get_tensor(entry.offset_key) if entry.offset_key else None,
        }
        return decode_quantized_tensor(entry, payload["q"], payload["scale"], payload["offset"])

    def get_slice(self, name: str) -> "_QuantizedSliceView":
        entry = self._entries.get(name)
        if entry is not None:
            return _QuantizedSliceView(entry.shape, entry.decode_dtype)
        return self._source.get_slice(name)

    def metadata(self) -> Dict[str, str]:
        return dict(self._metadata)

    def _is_visible_raw_key(self, key: str) -> bool:
        return key not in self._reserved and not key.startswith(f"{QUANT_PREFIX}.")


class _QuantizedSliceView:
    def __init__(self, shape: list[int], dtype_name: str) -> None:
        self._shape = list(shape)
        self._dtype = decode_dtype_to_torch(dtype_name)

    def get_shape(self):
        return list(self._shape)

    def get_dtype(self):
        return str(self._dtype).replace("torch.", "")


def resolve_disable_mmap(config: Any, default: bool = False) -> bool:
    """Read ``disable_mmap_load_safetensors`` from a config-like object."""
    if config is None:
        return default
    return bool(getattr(config, "disable_mmap_load_safetensors", default))
