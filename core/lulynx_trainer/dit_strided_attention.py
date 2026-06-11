"""Strided attention patterns for sparse DiT attention.

Extends the local window attention with strided patterns that skip every N tokens,
useful for long sequences and hierarchical attention patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch

from .dit_local_window_attention import DiTLocalWindowAttentionPlan


@dataclass(frozen=True)
class StridedAttentionPolicy:
    """Configuration for strided attention patterns.

    Attributes
    ----------
    enabled : bool
        Whether strided attention is enabled
    stride : int
        Stride value (attend to every N-th token)
    pattern : str
        Striding pattern:
        - "uniform": Every N-th token uniformly
        - "block": Blocks of size N with gaps
        - "dilated": Exponentially increasing stride
        - "local_global": Local window + strided global
    block_size : int
        For "block" pattern, size of each block
    local_window : int
        For "local_global" pattern, size of local window
    include_self : bool
        Always include self-attention (diagonal)
    """
    enabled: bool = False
    stride: int = 2
    pattern: Literal["uniform", "block", "dilated", "local_global"] = "uniform"
    block_size: int = 4
    local_window: int = 8
    include_self: bool = True

    def validate(self) -> None:
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.local_window < 0:
            raise ValueError("local_window must be >= 0")


def build_strided_attention_mask(
    token_count: int,
    policy: StridedAttentionPolicy | Mapping[str, Any] | None = None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build strided attention mask.

    Parameters
    ----------
    token_count : int
        Number of tokens
    policy : StridedAttentionPolicy or dict, optional
        Striding policy configuration
    device : torch.device or str, optional
        Device for mask tensor

    Returns
    -------
    torch.Tensor
        Boolean mask of shape (token_count, token_count)
        True = attend, False = mask out

    Examples
    --------
    >>> # Uniform stride: attend to every 2nd token
    >>> policy = StridedAttentionPolicy(enabled=True, stride=2, pattern="uniform")
    >>> mask = build_strided_attention_mask(8, policy)

    >>> # Local + global: local window of 4, global stride of 2
    >>> policy = StridedAttentionPolicy(
    ...     enabled=True, stride=2, pattern="local_global", local_window=4
    ... )
    >>> mask = build_strided_attention_mask(16, policy)
    """
    if isinstance(policy, Mapping):
        policy = StridedAttentionPolicy(**policy)
    elif policy is None:
        policy = StridedAttentionPolicy()

    policy.validate()

    if not policy.enabled or policy.stride <= 1:
        # Full attention
        return torch.ones((token_count, token_count), dtype=torch.bool, device=device)

    if policy.pattern == "uniform":
        mask = _build_uniform_strided_mask(token_count, policy.stride, device)
    elif policy.pattern == "block":
        mask = _build_block_strided_mask(token_count, policy.stride, policy.block_size, device)
    elif policy.pattern == "dilated":
        mask = _build_dilated_strided_mask(token_count, policy.stride, device)
    elif policy.pattern == "local_global":
        mask = _build_local_global_mask(token_count, policy.stride, policy.local_window, device)
    else:
        raise ValueError(f"Unknown strided pattern: {policy.pattern}")

    if policy.include_self:
        mask.fill_diagonal_(True)

    # Ensure every query can attend to at least one key
    if not bool(mask.any(dim=1).all()):
        raise RuntimeError("Strided attention mask produced empty query rows")

    return mask


def _build_uniform_strided_mask(
    token_count: int,
    stride: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Uniform stride: attend to every N-th token.

    Example with stride=2, token_count=8:
        Query 0 attends to: 0, 2, 4, 6
        Query 1 attends to: 1, 3, 5, 7
        Query 2 attends to: 0, 2, 4, 6
        ...
    """
    mask = torch.zeros((token_count, token_count), dtype=torch.bool, device=device)

    for i in range(token_count):
        # Attend to tokens at positions: i % stride, i % stride + stride, ...
        start = i % stride
        indices = torch.arange(start, token_count, stride, device=device)
        mask[i, indices] = True

    return mask


def _build_block_strided_mask(
    token_count: int,
    stride: int,
    block_size: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Block stride: blocks of size B with stride S.

    Example with block_size=2, stride=4, token_count=8:
        Block 0: [0, 1]
        Block 1: [4, 5]
        Each query attends to all tokens in its block
    """
    mask = torch.zeros((token_count, token_count), dtype=torch.bool, device=device)

    for i in range(token_count):
        # Find which block this token belongs to
        block_id = i // stride
        block_start = block_id * stride
        block_end = min(block_start + block_size, token_count)

        # Attend to all tokens in the same block
        mask[i, block_start:block_end] = True

        # Also attend to tokens in other blocks (strided)
        for j in range(0, token_count, stride):
            end = min(j + block_size, token_count)
            mask[i, j:end] = True

    return mask


def _build_dilated_strided_mask(
    token_count: int,
    base_stride: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Dilated stride: exponentially increasing distances.

    Example with base_stride=2:
        Query i attends to: i-1, i-2, i-4, i-8, i-16, ...
                            i+1, i+2, i+4, i+8, i+16, ...
    """
    mask = torch.zeros((token_count, token_count), dtype=torch.bool, device=device)

    for i in range(token_count):
        # Exponentially increasing offsets
        offset = base_stride
        while offset < token_count:
            # Look backward
            if i - offset >= 0:
                mask[i, i - offset] = True
            # Look forward
            if i + offset < token_count:
                mask[i, i + offset] = True
            offset *= base_stride

    return mask


def _build_local_global_mask(
    token_count: int,
    global_stride: int,
    local_window: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Local + global: local window + strided global attention.

    Combines dense local window with sparse global strided attention.

    Example with local_window=4, global_stride=2:
        Query 0 attends to:
            - Local: [0, 1, 2, 3]  (window)
            - Global: [0, 2, 4, 6, 8, ...]  (strided)
    """
    mask = torch.zeros((token_count, token_count), dtype=torch.bool, device=device)

    half_window = local_window // 2

    for i in range(token_count):
        # Local window
        start = max(0, i - half_window)
        end = min(token_count, i + half_window + 1)
        mask[i, start:end] = True

        # Global strided
        indices = torch.arange(0, token_count, global_stride, device=device)
        mask[i, indices] = True

    return mask


def build_strided_attention_plan(
    token_count: int,
    policy: StridedAttentionPolicy | Mapping[str, Any] | None = None,
    *,
    device: torch.device | str | None = None,
) -> DiTLocalWindowAttentionPlan:
    """Build a DiTLocalWindowAttentionPlan with strided mask.

    This creates a plan compatible with existing local window attention code.

    Parameters
    ----------
    token_count : int
        Number of tokens
    policy : StridedAttentionPolicy or dict, optional
        Striding policy
    device : torch.device or str, optional
        Device for tensors

    Returns
    -------
    DiTLocalWindowAttentionPlan
        Plan with strided attention mask
    """
    if isinstance(policy, Mapping):
        policy = StridedAttentionPolicy(**policy)
    elif policy is None:
        policy = StridedAttentionPolicy()

    mask = build_strided_attention_mask(token_count, policy, device=device)

    enabled = policy.enabled and (mask.sum() < token_count * token_count)
    reason = f"strided_{policy.pattern}_s{policy.stride}" if enabled else "full_attention"

    # Create a plan compatible with DiTLocalWindowAttentionPlan
    return DiTLocalWindowAttentionPlan(
        enabled=enabled,
        token_count=token_count,
        grid_h=token_count,  # Treat as 1D sequence
        grid_w=1,
        window_h=token_count,
        window_w=1,
        one_sided=False,
        shift_h=0,
        shift_w=0,
        mask=mask,
        reason=reason,
    )


def build_strided_attention_scorecard(
    *,
    policy: StridedAttentionPolicy | None = None,
    token_count: int = 0,
    pattern_tested: bool = False,
    sparsity_verified: bool = False,
    quality_maintained: bool = False,
) -> dict[str, Any]:
    """Build scorecard for strided attention readiness.

    Parameters
    ----------
    policy : StridedAttentionPolicy, optional
        Policy to validate
    token_count : int
        Token count for testing
    pattern_tested : bool
        Whether pattern has been tested
    sparsity_verified : bool
        Whether sparsity reduction is verified
    quality_maintained : bool
        Whether quality is maintained

    Returns
    -------
    dict
        Scorecard with readiness status
    """
    cfg = policy or StridedAttentionPolicy()
    blockers: list[str] = []

    try:
        cfg.validate()
    except ValueError as exc:
        blockers.append(f"invalid_config:{exc}")

    if not pattern_tested:
        blockers.append("pattern_not_tested")
    if not sparsity_verified:
        blockers.append("sparsity_not_verified")
    if not quality_maintained:
        blockers.append("quality_not_maintained")

    if token_count > 0:
        try:
            mask = build_strided_attention_mask(token_count, cfg)
            sparsity = 1.0 - (mask.sum().item() / (token_count * token_count))
        except Exception as exc:
            blockers.append(f"mask_build_failed:{exc}")
            sparsity = 0.0
    else:
        sparsity = 0.0

    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "strided_attention_v0",
        "ok": ready,
        "optimization_ready": ready,
        "pattern": cfg.pattern,
        "stride": cfg.stride,
        "block_size": cfg.block_size,
        "local_window": cfg.local_window,
        "estimated_sparsity": float(sparsity),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "integrate with dit_local_window_attention"
            if ready
            else "complete testing and quality verification"
        ),
    }


__all__ = [
    "StridedAttentionPolicy",
    "build_strided_attention_mask",
    "build_strided_attention_plan",
    "build_strided_attention_scorecard",
]
