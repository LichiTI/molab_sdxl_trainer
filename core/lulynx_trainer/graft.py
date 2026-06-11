# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""GRAFT — multi-LoRA merging workflow (Phase 8.11 / #119).

Merge several independent LoRA checkpoints into one rank-r adapter
without naively summing them.  The pipeline:

  1. Load each LoRA checkpoint as a state dict.
  2. For each shared key compute the *full* delta weight ``W = up @ down * scale``.
  3. Sum the deltas (optionally weighted per checkpoint).
  4. SVD-truncate to rank ``r`` and write back as new ``up`` / ``down`` matrices.

The SVD step keeps the adapter at the requested rank while preserving
the dominant directions of the combined update.

Usage::

    from .graft import GRAFTConfig, graft_loras

    cfg = GRAFTConfig(target_rank=16)
    out = graft_loras(["a.safetensors", "b.safetensors"], cfg, weights=[1.0, 0.5])
    save_state_dict(out, "merged.safetensors")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class GRAFTConfig:
    """GRAFT merging hyper-parameters."""

    target_rank: int = 16
    alpha: Optional[float] = None      # if None, set to target_rank (scaling=1)
    skip_missing: bool = True          # skip keys not present in every checkpoint
    weight_normalize: bool = False     # divide weights so they sum to 1
    epsilon: float = 1e-6              # SVD truncation safety


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_lora_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a LoRA checkpoint to CPU regardless of file format."""
    p = Path(path)
    if p.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
            return load_file(str(p))
        except ImportError:
            return torch.load(str(p.with_suffix(".pt")), map_location="cpu", weights_only=True)
    return torch.load(str(p), map_location="cpu", weights_only=True)


def save_lora_state_dict(sd: Dict[str, torch.Tensor], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import save_file
            save_file(sd, str(p))
            return
        except ImportError:
            pass
    torch.save(sd, str(p))


# ---------------------------------------------------------------------------
# Key inspection
# ---------------------------------------------------------------------------

def _find_lora_pairs(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, str, str]]:
    """Discover (base_name, down_key, up_key) tuples in a state dict.

    Recognises both ``lora_down.weight``/``lora_up.weight`` and
    ``lora_A``/``lora_B`` naming conventions.
    """
    pairs: List[Tuple[str, str, str]] = []
    seen_bases: set = set()
    for key in state_dict:
        for down_suffix, up_suffix in (
            (".lora_down.weight", ".lora_up.weight"),
            (".lora_A.weight", ".lora_B.weight"),
            (".lora_A", ".lora_B"),
        ):
            if key.endswith(down_suffix):
                base = key[: -len(down_suffix)]
                if base in seen_bases:
                    continue
                up_key = base + up_suffix
                if up_key in state_dict:
                    pairs.append((base, key, up_key))
                    seen_bases.add(base)
                break
    return pairs


def _alpha_for_pair(state_dict: Dict[str, torch.Tensor], base: str) -> Optional[float]:
    for suffix in (".alpha", ".lora_alpha"):
        if (base + suffix) in state_dict:
            t = state_dict[base + suffix]
            try:
                return float(t.item() if torch.is_tensor(t) else t)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def graft_loras(
    paths: Sequence[str],
    config: GRAFTConfig,
    *,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Merge several LoRA state dicts into a single rank-r adapter."""
    if not paths:
        raise ValueError("graft_loras requires at least one input path")

    weights = list(weights) if weights is not None else [1.0] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("weights length must match paths length")

    if config.weight_normalize:
        total = sum(weights) or 1.0
        weights = [w / total for w in weights]

    state_dicts = [load_lora_state_dict(p) for p in paths]
    pair_lists = [_find_lora_pairs(sd) for sd in state_dicts]

    # Use the first checkpoint as the canonical key set
    canonical = pair_lists[0]
    canonical_bases = {b for b, _, _ in canonical}

    merged: Dict[str, torch.Tensor] = {}

    for base, down_key, up_key in canonical:
        # Aggregate deltas across checkpoints
        accumulated: Optional[torch.Tensor] = None

        skip = False
        for sd, w in zip(state_dicts, weights):
            ckpt_pairs = {b: (d, u) for b, d, u in _find_lora_pairs(sd)}
            if base not in ckpt_pairs:
                if config.skip_missing:
                    continue
                skip = True
                break

            d_key, u_key = ckpt_pairs[base]
            down = sd[d_key].to(torch.float32)
            up = sd[u_key].to(torch.float32)
            rank = down.shape[0]
            alpha = _alpha_for_pair(sd, base) or rank
            scale = alpha / max(rank, 1)
            delta = (up @ down) * scale * w
            accumulated = delta if accumulated is None else accumulated + delta

        if skip or accumulated is None:
            continue

        new_up, new_down = _svd_truncate(
            accumulated, target_rank=config.target_rank, epsilon=config.epsilon,
        )

        # Re-key with the down_suffix used in the first checkpoint
        suffix = down_key[len(base):]
        if suffix == ".lora_down.weight":
            up_suffix = ".lora_up.weight"
        elif suffix == ".lora_A.weight":
            up_suffix = ".lora_B.weight"
        else:
            up_suffix = ".lora_B"
        merged[base + suffix] = new_down
        merged[base + up_suffix] = new_up

        # Write back alpha if requested
        target_alpha = config.alpha if config.alpha is not None else config.target_rank
        merged[base + ".alpha"] = torch.tensor(float(target_alpha))

    return merged


def _svd_truncate(
    delta: torch.Tensor,
    target_rank: int,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SVD-decompose ``delta`` and return ``(up, down)`` of rank ``target_rank``."""
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    rank = min(target_rank, S.shape[0])

    # Filter out near-zero singular values
    keep = S[:rank] > epsilon
    if keep.any():
        U_r = U[:, :rank][:, keep]
        S_r = S[:rank][keep]
        V_r = Vh[:rank, :][keep, :]
    else:
        U_r = U[:, :rank]
        S_r = S[:rank]
        V_r = Vh[:rank, :]

    sqrt_s = torch.sqrt(S_r)
    up = U_r * sqrt_s.unsqueeze(0)        # [out, r]
    down = V_r * sqrt_s.unsqueeze(-1)     # [r, in]

    # Pad to target_rank if SVD trimmed early
    if up.shape[1] < target_rank:
        pad = target_rank - up.shape[1]
        up = torch.nn.functional.pad(up, (0, pad))
        down = torch.nn.functional.pad(down, (0, 0, 0, pad))

    return up, down
