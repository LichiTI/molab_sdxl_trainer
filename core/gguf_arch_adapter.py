"""Architecture-specific planning for Python GGUF exports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class GGUFExportPlan:
    arch: str
    name: str
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    skipped_tensors: int = 0
    warnings: list[str] | None = None


_LLAMA_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_GGUF_LLAMA_LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_LLAMA_TENSOR_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

_LLAMA_LAYER_TENSOR_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
}

_LLAMA_REQUIRED_SUFFIXES = {
    "attn_norm.weight",
    "ffn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_down.weight",
    "ffn_up.weight",
}


def plan_gguf_export(
    state: dict[str, torch.Tensor],
    *,
    arch: str,
    name: str,
    source_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> GGUFExportPlan:
    resolved_arch = normalize_gguf_arch(arch)
    clean_name = str(name or (source_path.stem if source_path else "model"))
    if resolved_arch == "llama":
        return _plan_llama_export(state, clean_name, source_path=source_path, metadata=metadata or {})
    if resolved_arch != "generic":
        raise ValueError(f"Unsupported GGUF architecture: {arch}")
    return GGUFExportPlan(
        arch="generic",
        name=clean_name,
        tensors={str(key): value for key, value in state.items() if torch.is_tensor(value)},
        metadata={"name": clean_name},
    )


def normalize_gguf_arch(value: Any) -> str:
    text = str(value or "generic").strip().lower().replace("-", "_")
    aliases = {
        "": "generic",
        "raw": "generic",
        "tensor": "generic",
        "tensors": "generic",
        "llama2": "llama",
        "llama3": "llama",
        "llama_hf": "llama",
        "hf_llama": "llama",
    }
    return aliases.get(text, text)


def _plan_llama_export(
    state: dict[str, torch.Tensor],
    name: str,
    *,
    source_path: Path | None,
    metadata: dict[str, Any],
) -> GGUFExportPlan:
    config = _load_sibling_config(source_path)
    tensors: dict[str, torch.Tensor] = {}
    skipped = 0

    for key, tensor in state.items():
        if not torch.is_tensor(tensor):
            skipped += 1
            continue
        mapped = _map_llama_tensor_name(str(key))
        if mapped is None:
            skipped += 1
            continue
        tensors[mapped] = tensor

    if "token_embd.weight" not in tensors:
        raise ValueError("GGUF llama export requires model.embed_tokens.weight or token_embd.weight")
    if "output_norm.weight" not in tensors:
        raise ValueError("GGUF llama export requires model.norm.weight or output_norm.weight")

    layers = _llama_layer_indexes(tensors)
    if not layers:
        raise ValueError("GGUF llama export requires at least one complete decoder layer")

    missing = _missing_llama_layer_tensors(tensors, layers)
    if missing:
        preview = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise ValueError(f"GGUF llama export is missing required tensors: {preview}{more}")

    embed = tensors["token_embd.weight"]
    hidden = _shape_dim(embed, 1, "token_embd.weight")
    vocab = _shape_dim(embed, 0, "token_embd.weight")
    first_layer = min(layers)
    feed_forward = _shape_dim(tensors[f"blk.{first_layer}.ffn_gate.weight"], 0, f"blk.{first_layer}.ffn_gate.weight")
    head_count = _positive_int(metadata.get("head_count") or config.get("num_attention_heads"), "head_count")
    if head_count <= 0:
        raise ValueError("GGUF llama export requires num_attention_heads in config.json or gguf_metadata.head_count")
    head_count_kv = _positive_int(metadata.get("head_count_kv") or config.get("num_key_value_heads") or head_count, "head_count_kv")
    rope_dim = _positive_int(metadata.get("rope_dimension_count") or config.get("head_dim") or hidden // head_count, "rope_dimension_count")
    context_length = _positive_int(
        metadata.get("context_length") or config.get("max_position_embeddings") or config.get("seq_length") or 2048,
        "context_length",
    )
    rms_eps = _float_value(metadata.get("layer_norm_rms_eps") or config.get("rms_norm_eps") or 1e-5, "layer_norm_rms_eps")
    rope_base = metadata.get("rope_freq_base") or config.get("rope_theta")

    if hidden % head_count != 0:
        raise ValueError("GGUF llama export requires embedding_length to be divisible by head_count")

    plan_metadata: dict[str, Any] = {
        "name": name,
        "context_length": context_length,
        "embedding_length": hidden,
        "block_count": len(layers),
        "feed_forward_length": feed_forward,
        "head_count": head_count,
        "head_count_kv": head_count_kv,
        "rope_dimension_count": rope_dim,
        "layer_norm_rms_eps": rms_eps,
        "vocab_size": vocab,
    }
    if rope_base is not None and rope_base != "":
        plan_metadata["rope_freq_base"] = _float_value(rope_base, "rope_freq_base")

    warnings: list[str] = []
    if "output.weight" not in tensors:
        warnings.append("output.weight is missing; exported GGUF relies on tied output embeddings if the consumer supports it")

    return GGUFExportPlan(
        arch="llama",
        name=name,
        tensors=tensors,
        metadata=plan_metadata,
        skipped_tensors=skipped,
        warnings=warnings,
    )


def _map_llama_tensor_name(key: str) -> str | None:
    if key in _LLAMA_TENSOR_MAP:
        return _LLAMA_TENSOR_MAP[key]
    if key in {"token_embd.weight", "output_norm.weight", "output.weight"}:
        return key
    gguf_match = _GGUF_LLAMA_LAYER_RE.match(key)
    if gguf_match:
        suffix = gguf_match.group(2)
        return key if suffix in _LLAMA_REQUIRED_SUFFIXES else None
    layer_match = _LLAMA_LAYER_RE.match(key)
    if not layer_match:
        return None
    mapped_tail = _LLAMA_LAYER_TENSOR_MAP.get(layer_match.group(2))
    return f"blk.{layer_match.group(1)}.{mapped_tail}" if mapped_tail else None


def _llama_layer_indexes(tensors: dict[str, torch.Tensor]) -> list[int]:
    indexes = set()
    for name in tensors:
        match = _GGUF_LLAMA_LAYER_RE.match(name)
        if match:
            indexes.add(int(match.group(1)))
    return sorted(indexes)


def _missing_llama_layer_tensors(tensors: dict[str, torch.Tensor], layers: list[int]) -> list[str]:
    missing: list[str] = []
    for index in layers:
        for suffix in sorted(_LLAMA_REQUIRED_SUFFIXES):
            name = f"blk.{index}.{suffix}"
            if name not in tensors:
                missing.append(name)
    return missing


def _load_sibling_config(source_path: Path | None) -> dict[str, Any]:
    if source_path is None:
        return {}
    config_path = source_path.parent / "config.json"
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _shape_dim(tensor: torch.Tensor, dim: int, name: str) -> int:
    shape = tuple(int(item) for item in tensor.shape)
    if len(shape) <= dim:
        raise ValueError(f"GGUF llama tensor {name} must have at least {dim + 1} dimensions")
    return shape[dim]


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except Exception:
        return 0
    if number < 0:
        raise ValueError(f"{name} must be positive")
    return number


def _float_value(value: Any, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a number") from exc


__all__ = ["GGUFExportPlan", "normalize_gguf_arch", "plan_gguf_export"]
