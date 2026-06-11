# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Sparse-sampling attention entropy probe.

Two paths:
- **materialized** (torch backend): sample rows from existing attention weights.
- **non-materialized** (sdpa / flash2 / sageattn): compute Q_sampled @ K^T manually.

Overhead per probe: ~0.1 ms + ~2 MB temporary VRAM (seq_len=4096). Runs once
every N steps on a single randomly-chosen attention call.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import torch

_SAMPLE_ROWS = 128

# ── Probe state machine ─────────────────────────────────────────────────────

_probe_armed: bool = False
_probe_target_call: int = -1
_probe_call_counter: int = 0
_probe_result: Optional[float] = None


def arm_probe(total_attention_calls: int = 32) -> None:
    global _probe_armed, _probe_target_call, _probe_call_counter, _probe_result
    _probe_armed = True
    _probe_target_call = random.randint(0, max(total_attention_calls - 1, 0))
    _probe_call_counter = 0
    _probe_result = None


def disarm_probe() -> None:
    global _probe_armed, _probe_call_counter
    _probe_armed = False
    _probe_call_counter = 0


def should_probe() -> bool:
    global _probe_call_counter
    if not _probe_armed or _probe_result is not None:
        return False
    hit = _probe_call_counter == _probe_target_call
    _probe_call_counter += 1
    return hit


def collect_probe() -> Optional[float]:
    global _probe_result
    result = _probe_result
    disarm_probe()
    if result is not None:
        _accumulate(result)
    return result


# ── Entropy computation ──────────────────────────────────────────────────────

def _compute_entropy(weights: torch.Tensor) -> float:
    """Compute mean entropy from a probability matrix. weights: (..., T)."""
    eps = 1e-8
    ent = -(weights * torch.log(weights + eps)).sum(dim=-1)
    return float(ent.mean().item())


def probe_materialized(attn_weights: torch.Tensor) -> None:
    """Torch backend: sample from existing (B, H, T, T) attention weights."""
    global _probe_result
    B, H, T, _ = attn_weights.shape
    h = random.randint(0, max(H - 1, 0))
    n = min(_SAMPLE_ROWS, T)
    indices = torch.randperm(T, device=attn_weights.device)[:n]
    sampled = attn_weights[:, h, indices, :]  # (B, n, T)
    _probe_result = _compute_entropy(sampled.detach())


def probe_from_qk(q: torch.Tensor, k: torch.Tensor) -> None:
    """Non-materialized backend: compute sampled attention scores manually.

    q, k: (B, H, T, D)
    """
    global _probe_result
    B, H, T, D = q.shape
    h = random.randint(0, max(H - 1, 0))
    n = min(_SAMPLE_ROWS, T)
    indices = torch.randperm(T, device=q.device)[:n]

    q_sampled = q[:, h, indices, :].detach()  # (B, n, D)
    k_head = k[:, h].detach()                 # (B, T, D)

    scale = D ** -0.5
    scores = torch.matmul(q_sampled, k_head.transpose(-1, -2)) * scale  # (B, n, T)
    weights = scores.softmax(dim=-1)
    _probe_result = _compute_entropy(weights)


# ── Cumulative stats ─────────────────────────────────────────────────────────

_entropy_stats: Dict[str, float] = {
    "sum": 0.0,
    "count": 0,
    "min": float("inf"),
    "max": float("-inf"),
}


def _accumulate(value: float) -> None:
    _entropy_stats["sum"] += value
    _entropy_stats["count"] += 1
    if value < _entropy_stats["min"]:
        _entropy_stats["min"] = value
    if value > _entropy_stats["max"]:
        _entropy_stats["max"] = value


def snapshot_entropy_stats() -> Dict[str, float]:
    count = _entropy_stats["count"]
    return {
        "mean": _entropy_stats["sum"] / count if count > 0 else 0.0,
        "min": _entropy_stats["min"] if count > 0 else 0.0,
        "max": _entropy_stats["max"] if count > 0 else 0.0,
        "count": count,
    }


def reset_entropy_stats() -> None:
    _entropy_stats["sum"] = 0.0
    _entropy_stats["count"] = 0
    _entropy_stats["min"] = float("inf")
    _entropy_stats["max"] = float("-inf")
