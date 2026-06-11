# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Compact tensor layout metadata for lossless cache parity probes."""

from __future__ import annotations

from typing import Any, Mapping


DEFAULT_LAYOUT_TENSOR_KEYS = (
    "latents",
    "encoder_hidden_states",
    "attention_mask",
    "loss_mask",
    "loss_masks",
    "t5_input_ids",
    "t5_attention_mask",
    "qwen3_hidden_states",
    "qwen3_attention_mask",
)


def tensor_layout_metadata(
    tensor: Any,
    *,
    tensor_key: str,
    payload_source: str = "",
    copy_path: str = "",
    array_source: str = "",
    cache_file: str = "",
) -> dict[str, Any]:
    """Return JSON-safe layout facts for a tensor-like value."""

    shape = getattr(tensor, "shape", ())
    stride = tensor.stride() if hasattr(tensor, "stride") else ()
    is_contiguous = bool(tensor.is_contiguous()) if hasattr(tensor, "is_contiguous") else False
    storage_offset = int(tensor.storage_offset()) if hasattr(tensor, "storage_offset") else 0
    return {
        "tensor_key": str(tensor_key),
        "dtype": str(getattr(tensor, "dtype", "")).replace("torch.", ""),
        "shape": [int(item) for item in tuple(shape)],
        "stride": [int(item) for item in tuple(stride)],
        "is_contiguous": is_contiguous,
        "storage_offset": storage_offset,
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        "device": str(getattr(tensor, "device", "")),
        "payload_source": str(payload_source),
        "copy_path": str(copy_path),
        "array_source": str(array_source),
        "cache_file": str(cache_file),
    }


def mapping_tensor_layouts(
    values: Mapping[str, Any],
    *,
    sample_id: str = "",
    payload_source: str = "",
    copy_path: str = "",
    array_source: str = "",
    cache_file: str = "",
    tensor_keys: tuple[str, ...] = DEFAULT_LAYOUT_TENSOR_KEYS,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in tensor_keys:
        value = values.get(key)
        if not hasattr(value, "shape"):
            continue
        record = tensor_layout_metadata(
            value,
            tensor_key=key,
            payload_source=payload_source,
            copy_path=copy_path,
            array_source=array_source,
            cache_file=cache_file,
        )
        if sample_id:
            record["sample_id"] = str(sample_id)
        records.append(record)
    return records


def batch_tensor_layouts(
    batch: Mapping[str, Any],
    *,
    payload_source: str = "",
    copy_path: str = "",
    array_source: str = "collated_batch",
) -> list[dict[str, Any]]:
    return mapping_tensor_layouts(
        batch,
        payload_source=payload_source,
        copy_path=copy_path,
        array_source=array_source,
    )


__all__ = [
    "DEFAULT_LAYOUT_TENSOR_KEYS",
    "batch_tensor_layouts",
    "mapping_tensor_layouts",
    "tensor_layout_metadata",
]
