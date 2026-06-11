"""Payload ownership summaries for native cache reader dispatch shadows."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch


TEXT_PAYLOAD_FIELDS = {
    "encoder_hidden_states",
    "attention_mask",
    "pooled_prompt_embeds",
    "t5_input_ids",
    "t5_attention_mask",
    "qwen3_hidden_states",
    "qwen3_attention_mask",
}

LATENT_PAYLOAD_FIELDS = {"latents"}

_FNV_OFFSET_BASIS = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211


def _fnv1a_64(payload: bytes) -> int:
    checksum = _FNV_OFFSET_BASIS
    for value in payload:
        checksum ^= int(value)
        checksum = (checksum * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return checksum


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    element_count = int(tensor.numel())
    payload = tensor.numpy().tobytes(order="C") if element_count else b""
    return {
        "kind": "tensor",
        "shape": [int(dim) for dim in tensor.shape],
        "canonical_dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(value.device),
        "element_count": element_count,
        "payload_byte_count": int(tensor.element_size() * element_count),
        "payload_checksum": _fnv1a_64(payload),
        "is_contiguous": bool(value.is_contiguous()),
        "requires_grad": bool(value.requires_grad),
    }


def _value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if isinstance(value, (list, tuple)):
        type_counts: dict[str, int] = {}
        for item in value:
            type_name = type(item).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return {
            "kind": "sequence",
            "item_count": len(value),
            "item_type_counts": type_counts,
        }
    if isinstance(value, dict):
        return {"kind": "mapping", "key_count": len(value), "keys": sorted(str(key) for key in value.keys())[:16]}
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (str, bytes)):
        return {"kind": type(value).__name__, "item_count": 1, "length": len(value)}
    return {"kind": type(value).__name__, "item_count": 1}


def build_cache_reader_payload_ownership_shadow(
    batch: Dict[str, Any],
    *,
    sample_indices: Sequence[int],
    batch_payload_parity_passed: bool,
) -> Dict[str, Any]:
    """Describe which collated batch fields are still Python authoritative."""
    fields: dict[str, dict[str, Any]] = {}
    text_fields: list[str] = []
    aux_fields: list[str] = []
    latent_fields: list[str] = []
    for name in sorted(str(key) for key in batch.keys()):
        value = batch.get(name)
        if name in LATENT_PAYLOAD_FIELDS:
            category = "latent"
            latent_fields.append(name)
            ownership = "native_shadow_validated_python_batch_authoritative"
        elif name in TEXT_PAYLOAD_FIELDS:
            category = "text_conditioning"
            text_fields.append(name)
            ownership = "python_batch_authoritative"
        else:
            category = "auxiliary"
            aux_fields.append(name)
            ownership = "python_batch_authoritative"
        fields[name] = {
            "category": category,
            "ownership": ownership,
            "native_payload_promoted": False,
            "python_batch_authoritative": True,
            **_value_summary(value),
        }
    return {
        "schema_version": 1,
        "provider": "native_cache_reader_payload_ownership_shadow_v1",
        "ok": True,
        "debug_only": True,
        "shadow_run": True,
        "sample_indices": [int(index) for index in sample_indices],
        "sample_count": len(list(sample_indices)),
        "field_count": len(fields),
        "latent_fields": latent_fields,
        "text_payload_fields": text_fields,
        "aux_payload_fields": aux_fields,
        "python_authoritative_fields": sorted(text_fields + aux_fields + latent_fields),
        "native_latent_shadow_verified": bool(batch_payload_parity_passed),
        "text_payload_ownership_ready": bool(text_fields),
        "aux_payload_ownership_ready": bool(aux_fields),
        "native_text_payload_promoted": False,
        "native_aux_payload_promoted": False,
        "fields": fields,
        "would_allow_native_dispatch": False,
        "fallback_to_python_batch": True,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_dispatch": False,
        "training_path_enabled": False,
    }


def compact_payload_ownership_shadow(report: Any) -> Dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {}
    return {
        "provider": str(report.get("provider") or ""),
        "ok": bool(report.get("ok", False)),
        "field_count": int(report.get("field_count", 0) or 0),
        "latent_fields": [str(item) for item in list(report.get("latent_fields", []) or [])],
        "text_payload_fields": [str(item) for item in list(report.get("text_payload_fields", []) or [])],
        "aux_payload_fields": [str(item) for item in list(report.get("aux_payload_fields", []) or [])],
        "native_latent_shadow_verified": bool(report.get("native_latent_shadow_verified", False)),
        "text_payload_ownership_ready": bool(report.get("text_payload_ownership_ready", False)),
        "aux_payload_ownership_ready": bool(report.get("aux_payload_ownership_ready", False)),
        "native_text_payload_promoted": False,
        "native_aux_payload_promoted": False,
        "fallback_to_python_batch": True,
        "training_path_enabled": False,
    }


__all__ = [
    "build_cache_reader_payload_ownership_shadow",
    "compact_payload_ownership_shadow",
]
