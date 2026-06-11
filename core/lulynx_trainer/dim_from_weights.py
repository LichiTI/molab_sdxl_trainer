"""Infer LoRA rank (dim) from existing weight files.

Reads a LoRA safetensors or .pt checkpoint and determines the rank
from the shape of the lora_down / lora_A matrices.  This supports the
``dim_from_weights`` config option: when enabled, the injector uses the
rank from the file rather than the config value, allowing fine-tuning
of an existing LoRA at its original rank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Dict

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DimFromWeightsResolution:
    enabled: bool
    weight_path: str
    original_rank: int
    inferred_rank: int
    applied: bool
    per_layer_ranks: Dict[str, int]
    error: str = ""


def resolve_dim_from_weights_path(config: Any, model_arch: str = "") -> str:
    """Return the adapter weight path used for rank inference."""
    weight_path = str(getattr(config, "network_weights_path", "") or "").strip()
    if not weight_path and str(model_arch or "").strip().lower() == "anima":
        weight_path = str(getattr(config, "anima_dit_adapter_path", "") or "").strip()
    return weight_path


def apply_dim_from_weights(
    config: Any,
    *,
    model_arch: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
) -> DimFromWeightsResolution:
    """Apply ``dim_from_weights`` rank inference to a training config.

    The trainer calls this immediately before adapter injection so the injector
    is built with the checkpoint's original rank.
    """
    original_rank = int(getattr(config, "network_dim", 32) or 32)
    if not bool(getattr(config, "dim_from_weights", False)):
        return DimFromWeightsResolution(False, "", original_rank, original_rank, False, {})

    weight_path = resolve_dim_from_weights_path(config, model_arch)
    if not weight_path:
        return DimFromWeightsResolution(True, "", original_rank, original_rank, False, {})

    try:
        inferred_rank, per_layer_ranks = infer_rank_from_weights(weight_path, default_rank=original_rank)
        applied = inferred_rank != original_rank
        if applied:
            if log_fn is not None:
                log_fn(
                    f"dim_from_weights: inferred rank={inferred_rank} "
                    f"(config was {original_rank}) before adapter injection"
                )
            setattr(config, "network_dim", inferred_rank)
        return DimFromWeightsResolution(
            True,
            weight_path,
            original_rank,
            inferred_rank,
            applied,
            per_layer_ranks,
        )
    except Exception as exc:
        message = f"dim_from_weights inference failed before adapter injection: {exc}"
        if log_fn is not None:
            log_fn(message)
        return DimFromWeightsResolution(True, weight_path, original_rank, original_rank, False, {}, message)


def infer_rank_from_weights(
    path: str,
    default_rank: int = 32,
) -> Tuple[int, Dict[str, int]]:
    """Infer the LoRA rank from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to a ``.safetensors`` or ``.pt`` / ``.bin`` LoRA checkpoint.
    default_rank : int
        Fallback rank if no lora_down / lora_A keys are found.

    Returns
    -------
    (rank, per_layer_ranks) : tuple
        *rank* is the most common rank across all LoRA layers (the one to
        use for injection). *per_layer_ranks* maps layer name patterns to
        their individual ranks for diagnostic logging.
    """
    state_dict = _load_state_dict(path)
    if state_dict is None:
        logger.warning("Could not load weights from %s; using default rank=%d", path, default_rank)
        return default_rank, {}

    per_layer_ranks: Dict[str, int] = {}
    rank_counts: Dict[int, int] = {}

    for key, tensor in state_dict.items():
        rank = _infer_rank_from_key(key, tensor)
        if rank is not None:
            per_layer_ranks[key] = rank
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

    if not rank_counts:
        logger.warning("No lora_down/lora_A keys found in %s; using default rank=%d", path, default_rank)
        return default_rank, {}

    # Use the most common rank as the inferred rank
    inferred_rank = max(rank_counts, key=rank_counts.get)
    logger.info(
        "Inferred LoRA rank=%d from %s (rank distribution: %s)",
        inferred_rank, path, rank_counts,
    )
    return inferred_rank, per_layer_ranks


def _load_state_dict(path: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load a state dict from safetensors or pt file."""
    p = Path(path)
    if not p.exists():
        logger.warning("Weight file not found: %s", path)
        return None

    if p.suffix in (".safetensors",):
        try:
            from safetensors.torch import load_file
            return load_file(str(p))
        except ImportError:
            logger.warning("safetensors library not installed; cannot read %s", path)
            return None
        except Exception as exc:
            logger.warning("Failed to read safetensors %s: %s", path, exc)
            return None

    try:
        from core.safe_pickle import safe_torch_load
        data = safe_torch_load(str(p))
        if isinstance(data, dict):
            if "state_dict" in data:
                return data["state_dict"]
            return data
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)

    return None


def _infer_rank_from_key(key: str, tensor: torch.Tensor) -> Optional[int]:
    """Extract rank from a single weight key/tensor pair.

    Convention: lora_down.weight has shape (rank, in_features),
    lora_A.weight has shape (rank, in_features).
    """
    name = key.lower()
    if "lora_down" in name or "lora_a" in name:
        if tensor.dim() >= 2:
            return tensor.shape[0]
    return None
