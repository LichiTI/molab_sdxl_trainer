# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Fixed text-token padding contract (#101).

For ``torch.compile`` and CUDAGraph capture to work, the input tensor
shapes must stay constant across batches.  Standard tokenisers produce
variable-length output (the longest caption in the batch), which causes
compile recompilations or graph capture failures.

This module wraps a tokenizer to *always* pad to a fixed maximum
length, and provides utilities to verify that the contract holds.

Usage::

    from .fixed_token_padding import FixedTokenPadder

    padder = FixedTokenPadder(tokenizer, fixed_length=512)
    encoded = padder.encode_batch(["a cat", "a dog photo"])
    # encoded["input_ids"].shape == [2, 512]
    # encoded["attention_mask"].shape == [2, 512]
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch

logger = logging.getLogger(__name__)


class TokenFieldBatch(dict):
    """Small token batch wrapper with dict and attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _get_token_field(encoded: Any, key: str) -> Any:
    if isinstance(encoded, Mapping):
        return encoded.get(key)
    return getattr(encoded, key, None)


def _iter_encoded_items(encoded: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(encoded, Mapping):
        return encoded.items()
    items = getattr(encoded, "items", None)
    if callable(items):
        try:
            return items()
        except Exception:
            pass
    fields = getattr(encoded, "__dict__", None)
    if isinstance(fields, dict):
        return fields.items()
    return ()


def _ensure_token_batch(encoded: Any) -> Any:
    if isinstance(encoded, TokenFieldBatch):
        return encoded
    if isinstance(encoded, Mapping) and not hasattr(encoded, "input_ids"):
        return TokenFieldBatch(encoded)
    return encoded


def _sane_positive_int(value: Any, *, max_reasonable: int = 1_000_000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if parsed <= 0 or parsed > max_reasonable:
        return 0
    return parsed


def _encoder_position_limit(text_encoder: Any) -> int:
    for owner in (getattr(text_encoder, "config", None), getattr(getattr(text_encoder, "text_model", None), "config", None)):
        limit = _sane_positive_int(getattr(owner, "max_position_embeddings", None))
        if limit > 0:
            return limit
    return 0


def resolve_fixed_token_length(
    tokenizer: Any,
    *,
    requested_length: int = 0,
    text_encoder: Any = None,
    default_length: int = 77,
) -> int:
    """Resolve a compile-safe token length without exceeding model limits."""

    requested = _sane_positive_int(requested_length)
    tokenizer_limit = _sane_positive_int(getattr(tokenizer, "model_max_length", None))
    encoder_limit = _encoder_position_limit(text_encoder)
    base = requested or tokenizer_limit or encoder_limit or _sane_positive_int(default_length) or 77
    limits = [value for value in (tokenizer_limit, encoder_limit) if value > 0]
    return max(min([base, *limits]) if limits else base, 1)


@dataclass
class FixedTokenPaddingConfig:
    """Configuration for fixed-length tokenisation."""

    fixed_length: int = 512        # exact output length
    truncation: bool = True        # truncate if input exceeds fixed_length
    pad_token_id: Optional[int] = None  # if None, infer from tokenizer
    return_tensors: str = "pt"


# ---------------------------------------------------------------------------
# Padder
# ---------------------------------------------------------------------------

class FixedTokenPadder:
    """Wrap a tokenizer to enforce a static output shape.

    Parameters
    ----------
    tokenizer : Any
        HuggingFace-style tokenizer with ``__call__`` returning ``input_ids`` /
        ``attention_mask``.
    fixed_length : int
        Output sequence length.  Inputs shorter are padded; longer are truncated.
    """

    def __init__(self, tokenizer: Any, fixed_length: int = 512) -> None:
        if fixed_length < 1:
            raise ValueError("fixed_length must be >= 1")
        self.tokenizer = tokenizer
        self.fixed_length = int(fixed_length)

    def encode_batch(
        self,
        captions: List[str],
        *,
        return_tensors: str = "pt",
    ) -> Any:
        """Encode a list of captions into ``[batch, fixed_length]`` tensors."""
        if not isinstance(captions, list):
            captions = list(captions)

        # Use the tokenizer's max_length / padding contract directly when supported
        try:
            encoded = self.tokenizer(
                captions,
                padding="max_length",
                max_length=self.fixed_length,
                truncation=True,
                return_tensors=return_tensors,
            )
            return self._validate_shape(encoded)
        except TypeError:
            # Older tokenizers might not accept these kwargs; fall back to manual padding
            return self._manual_encode(captions, return_tensors=return_tensors)

    def encode_one(self, caption: str) -> Any:
        return self.encode_batch([caption])

    def _validate_shape(self, encoded: Any) -> Any:
        """Confirm the shape contract; pad/truncate manually if violated."""
        input_ids = _get_token_field(encoded, "input_ids")
        if input_ids is None:
            return encoded
        if input_ids.shape[-1] == self.fixed_length:
            return _ensure_token_batch(encoded)
        # Tokenizer ignored max_length; renormalise manually
        return self._reshape_to_fixed(encoded)

    def _reshape_to_fixed(self, encoded: Any) -> TokenFieldBatch:
        out: TokenFieldBatch = TokenFieldBatch()
        pad_id = self._resolve_pad_id()
        for key, tensor in _iter_encoded_items(encoded):
            if not isinstance(tensor, torch.Tensor):
                out[key] = tensor
                continue
            cur = tensor.shape[-1]
            if cur == self.fixed_length:
                out[key] = tensor
            elif cur > self.fixed_length:
                out[key] = tensor[..., : self.fixed_length].contiguous()
            else:
                pad_amount = self.fixed_length - cur
                fill = pad_id if key == "input_ids" else 0
                pad = tensor.new_full((*tensor.shape[:-1], pad_amount), fill)
                out[key] = torch.cat([tensor, pad], dim=-1)
        return out

    def _manual_encode(self, captions: List[str], *, return_tensors: str = "pt") -> TokenFieldBatch:
        """Slow fallback: encode captions one at a time and pad in Python."""
        pad_id = self._resolve_pad_id()
        all_ids: List[List[int]] = []
        all_masks: List[List[int]] = []
        for cap in captions:
            ids = self.tokenizer.encode(cap, add_special_tokens=True)
            if len(ids) > self.fixed_length:
                ids = ids[: self.fixed_length]
            mask = [1] * len(ids)
            while len(ids) < self.fixed_length:
                ids.append(pad_id)
                mask.append(0)
            all_ids.append(ids)
            all_masks.append(mask)
        return TokenFieldBatch({
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        })

    def _resolve_pad_id(self) -> int:
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(self.tokenizer, attr, None)
            if isinstance(value, int):
                return value
        return 0


# ---------------------------------------------------------------------------
# Multi-encoder padding for Anima (CLIP + Qwen3 + T5)
# ---------------------------------------------------------------------------

@dataclass
class AnimaMultiEncoderPaddingConfig:
    """Configuration for multi-encoder fixed-length padding."""

    fixed_text_tokens: int = 0          # primary CLIP tokens (0 = no padding)
    fixed_qwen3_tokens: int = 0         # Qwen3 tokens (0 = no padding)
    fixed_t5_tokens: int = 0            # T5 tokens (0 = no padding)
    fixed_visual_tokens: int = 0        # visual tokens (0 = no padding)
    warn_on_truncation: bool = True     # log warning when truncating


def apply_anima_multi_encoder_padding(
    text_data: Dict[str, Optional[torch.Tensor]],
    config: AnimaMultiEncoderPaddingConfig,
) -> Dict[str, Optional[torch.Tensor]]:
    """Apply fixed-length padding to Anima multi-encoder text conditioning.

    Parameters
    ----------
    text_data : dict
        Dictionary with keys:
        - encoder_hidden_states: [seq, dim] primary CLIP embeddings
        - attention_mask: [seq] bool mask for CLIP
        - qwen3_hidden_states: [seq, dim] Qwen3 embeddings (optional)
        - qwen3_attention_mask: [seq] bool mask for Qwen3 (optional)
        - t5_input_ids: [seq] long T5 token IDs (optional)
        - t5_attention_mask: [seq] bool mask for T5 (optional)
    config : AnimaMultiEncoderPaddingConfig
        Padding configuration.

    Returns
    -------
    dict
        Padded text_data with fixed-length tensors.
    """
    result = {}

    # Primary CLIP text encoder
    if config.fixed_text_tokens > 0:
        result["encoder_hidden_states"] = _pad_or_truncate_embeddings(
            text_data.get("encoder_hidden_states"),
            config.fixed_text_tokens,
            "encoder_hidden_states",
            config.warn_on_truncation,
        )
        result["attention_mask"] = _pad_or_truncate_mask(
            text_data.get("attention_mask"),
            config.fixed_text_tokens,
            "attention_mask",
            config.warn_on_truncation,
        )
    else:
        result["encoder_hidden_states"] = text_data.get("encoder_hidden_states")
        result["attention_mask"] = text_data.get("attention_mask")

    # Qwen3 secondary encoder
    if config.fixed_qwen3_tokens > 0:
        result["qwen3_hidden_states"] = _pad_or_truncate_embeddings(
            text_data.get("qwen3_hidden_states"),
            config.fixed_qwen3_tokens,
            "qwen3_hidden_states",
            config.warn_on_truncation,
        )
        result["qwen3_attention_mask"] = _pad_or_truncate_mask(
            text_data.get("qwen3_attention_mask"),
            config.fixed_qwen3_tokens,
            "qwen3_attention_mask",
            config.warn_on_truncation,
        )
    else:
        result["qwen3_hidden_states"] = text_data.get("qwen3_hidden_states")
        result["qwen3_attention_mask"] = text_data.get("qwen3_attention_mask")

    # T5 encoder
    if config.fixed_t5_tokens > 0:
        result["t5_input_ids"] = _pad_or_truncate_ids(
            text_data.get("t5_input_ids"),
            config.fixed_t5_tokens,
            "t5_input_ids",
            config.warn_on_truncation,
        )
        result["t5_attention_mask"] = _pad_or_truncate_mask(
            text_data.get("t5_attention_mask"),
            config.fixed_t5_tokens,
            "t5_attention_mask",
            config.warn_on_truncation,
        )
    else:
        result["t5_input_ids"] = text_data.get("t5_input_ids")
        result["t5_attention_mask"] = text_data.get("t5_attention_mask")

    return result


def _pad_or_truncate_embeddings(
    tensor: Optional[torch.Tensor],
    target_length: int,
    name: str,
    warn: bool,
) -> Optional[torch.Tensor]:
    """Pad or truncate embedding tensor [seq, dim] to target_length."""
    if tensor is None:
        return None
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D [seq, dim], got shape {tensor.shape}")
    cur_len = tensor.shape[0]
    if cur_len == target_length:
        return tensor
    if cur_len > target_length:
        if warn:
            logger.warning(f"{name} truncated from {cur_len} to {target_length} tokens")
        return tensor[:target_length].contiguous()
    # Pad with zeros
    pad_amount = target_length - cur_len
    pad = torch.zeros(pad_amount, tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


def _pad_or_truncate_mask(
    tensor: Optional[torch.Tensor],
    target_length: int,
    name: str,
    warn: bool,
) -> Optional[torch.Tensor]:
    """Pad or truncate attention mask [seq] to target_length."""
    if tensor is None:
        return None
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D [seq], got shape {tensor.shape}")
    cur_len = tensor.shape[0]
    if cur_len == target_length:
        return tensor
    if cur_len > target_length:
        if warn:
            logger.warning(f"{name} truncated from {cur_len} to {target_length} tokens")
        return tensor[:target_length].contiguous()
    # Pad with False (0)
    pad_amount = target_length - cur_len
    pad = torch.zeros(pad_amount, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


def _pad_or_truncate_ids(
    tensor: Optional[torch.Tensor],
    target_length: int,
    name: str,
    warn: bool,
) -> Optional[torch.Tensor]:
    """Pad or truncate token ID tensor [seq] to target_length."""
    if tensor is None:
        return None
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1D [seq], got shape {tensor.shape}")
    cur_len = tensor.shape[0]
    if cur_len == target_length:
        return tensor
    if cur_len > target_length:
        if warn:
            logger.warning(f"{name} truncated from {cur_len} to {target_length} tokens")
        return tensor[:target_length].contiguous()
    # Pad with 0 (standard pad token ID)
    pad_amount = target_length - cur_len
    pad = torch.zeros(pad_amount, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


# ---------------------------------------------------------------------------
# Static-shape contract verification
# ---------------------------------------------------------------------------

def verify_static_shape(batches: List[Dict[str, torch.Tensor]]) -> bool:
    """Check that every batch has identical token shape — required for compile."""
    if not batches:
        return True
    reference: Optional[torch.Size] = None
    for b in batches:
        ids = _get_token_field(b, "input_ids")
        if ids is None:
            return False
        if reference is None:
            reference = ids.shape[1:]  # ignore batch dim
        elif ids.shape[1:] != reference:
            return False
    return True
