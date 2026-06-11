"""Gathered local/window attention prototype for DiT training research."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .dit_local_window_attention import (
    DiTLocalWindowAttentionPlan,
    DiTLocalWindowAttentionPolicy,
    build_dit_local_window_attention_plan,
)


@dataclass(frozen=True)
class GatheredLocalWindowAttentionResult:
    output: torch.Tensor
    plan: DiTLocalWindowAttentionPlan
    dense_score_elements: int
    gathered_score_elements: int

    @property
    def estimated_score_fraction(self) -> float:
        total = max(int(self.dense_score_elements), 1)
        return float(self.gathered_score_elements / total)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.plan.enabled),
            "token_count": int(self.plan.token_count),
            "allowed_pairs": int(self.plan.allowed_pairs),
            "dense_score_elements": int(self.dense_score_elements),
            "gathered_score_elements": int(self.gathered_score_elements),
            "estimated_score_fraction": float(self.estimated_score_fraction),
            "estimated_score_reduction": float(1.0 - self.estimated_score_fraction),
            "reason": self.plan.reason,
        }


def gathered_dit_local_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    policy: DiTLocalWindowAttentionPolicy | Mapping[str, Any] | None = None,
    *,
    scale: float | None = None,
) -> GatheredLocalWindowAttentionResult:
    _validate_qkv(query, key, value)
    batch, heads, token_count, head_dim = query.shape
    plan = build_dit_local_window_attention_plan(token_count, policy, device=query.device)
    dense_score_elements = int(batch * heads * token_count * token_count)
    gathered_score_elements = int(batch * heads * plan.allowed_pairs)
    if not plan.enabled:
        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, scale=scale)
        return GatheredLocalWindowAttentionResult(output, plan, dense_score_elements, dense_score_elements)

    resolved_scale = float(scale) if scale is not None else 1.0 / math.sqrt(float(head_dim))
    rows: list[torch.Tensor] = []
    for query_index in range(token_count):
        key_indices = plan.mask[query_index].nonzero(as_tuple=False).flatten()
        local_key = key.index_select(dim=-2, index=key_indices)
        local_value = value.index_select(dim=-2, index=key_indices)
        local_query = query[:, :, query_index, :].unsqueeze(-2)
        scores = torch.matmul(local_query, local_key.transpose(-2, -1)) * resolved_scale
        weights = torch.softmax(scores, dim=-1)
        rows.append(torch.matmul(weights, local_value))
    output = torch.cat(rows, dim=-2)
    return GatheredLocalWindowAttentionResult(output, plan, dense_score_elements, gathered_score_elements)


def build_gathered_local_window_attention_scorecard(
    result: GatheredLocalWindowAttentionResult | Mapping[str, Any],
    *,
    dense_mask_parity_ok: bool = False,
    grad_flow_ok: bool = False,
) -> dict[str, Any]:
    payload = result.as_dict() if isinstance(result, GatheredLocalWindowAttentionResult) else dict(result)
    blockers: list[str] = []
    if not bool(payload.get("enabled")):
        blockers.append("local_window_not_enabled")
    if float(payload.get("estimated_score_fraction") or 1.0) >= 1.0:
        blockers.append("score_reduction_missing")
    if not dense_mask_parity_ok:
        blockers.append("dense_mask_parity_missing")
    if not grad_flow_ok:
        blockers.append("grad_flow_missing")
    blockers.append("production_cuda_kernel_missing")
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_gathered_sparse_prototype_v0",
        "reducer_id": "local_window_attention",
        "ok": not blockers[:-1],
        "prototype_ready": not blockers[:-1],
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "result": payload,
        "dense_mask_parity_ok": bool(dense_mask_parity_ok),
        "grad_flow_ok": bool(grad_flow_ok),
        "blocked_reasons": blockers,
        "recommended_next_step": "replace Python gathered prototype with production CUDA/window kernel",
    }


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, heads, tokens, head_dim]")
    if tuple(query.shape) != tuple(key.shape) or tuple(query.shape) != tuple(value.shape):
        raise ValueError("query, key, and value must have the same shape")
    if any(dim <= 0 for dim in query.shape):
        raise ValueError("query, key, and value must have non-empty dimensions")


__all__ = [
    "GatheredLocalWindowAttentionResult",
    "build_gathered_local_window_attention_scorecard",
    "gathered_dit_local_window_attention",
]
