"""Default-off local/window attention probe for DiT training.

This module is a small tensor primitive for high-resolution DiT compute
reducer experiments. It does not patch real Anima/Newbie blocks and it does not
change the existing attention backend dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DiTLocalWindowAttentionPolicy:
    enabled: bool = False
    grid_h: int = 0
    grid_w: int = 0
    window_h: int = 0
    window_w: int = 0
    one_sided: bool = False
    shift_h: int = 0
    shift_w: int = 0
    include_self: bool = True

    def normalized(self) -> "DiTLocalWindowAttentionPolicy":
        return DiTLocalWindowAttentionPolicy(
            enabled=bool(self.enabled),
            grid_h=max(int(self.grid_h or 0), 0),
            grid_w=max(int(self.grid_w or 0), 0),
            window_h=max(int(self.window_h or 0), 0),
            window_w=max(int(self.window_w or 0), 0),
            one_sided=bool(self.one_sided),
            shift_h=int(self.shift_h or 0),
            shift_w=int(self.shift_w or 0),
            include_self=bool(self.include_self),
        )


@dataclass(frozen=True)
class DiTLocalWindowAttentionPlan:
    enabled: bool
    token_count: int
    grid_h: int
    grid_w: int
    window_h: int
    window_w: int
    one_sided: bool
    shift_h: int
    shift_w: int
    mask: torch.Tensor
    reason: str

    @property
    def allowed_pairs(self) -> int:
        return int(self.mask.sum().item())

    @property
    def estimated_attention_fraction(self) -> float:
        total = max(int(self.token_count) * int(self.token_count), 1)
        return float(self.allowed_pairs / total)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "token_count": int(self.token_count),
            "grid_h": int(self.grid_h),
            "grid_w": int(self.grid_w),
            "window_h": int(self.window_h),
            "window_w": int(self.window_w),
            "one_sided": bool(self.one_sided),
            "shift_h": int(self.shift_h),
            "shift_w": int(self.shift_w),
            "allowed_pairs": int(self.allowed_pairs),
            "estimated_attention_fraction": float(self.estimated_attention_fraction),
            "reason": self.reason,
        }


def build_dit_local_window_attention_plan(
    token_count: int,
    policy: DiTLocalWindowAttentionPolicy | Mapping[str, Any] | None = None,
    *,
    device: torch.device | str | None = None,
) -> DiTLocalWindowAttentionPlan:
    count = int(token_count)
    if count <= 0:
        raise ValueError("token_count must be positive")
    cfg = _policy(policy)
    grid_h, grid_w = _resolve_grid(count, cfg.grid_h, cfg.grid_w)
    full_mask = torch.ones((count, count), dtype=torch.bool, device=device)
    if not cfg.enabled:
        return _plan(cfg, full_mask, count, grid_h, grid_w, enabled=False, reason="disabled")
    if cfg.window_h <= 0 or cfg.window_w <= 0:
        return _plan(cfg, full_mask, count, grid_h, grid_w, enabled=False, reason="window_disabled")

    mask = _build_2d_window_mask(
        grid_h=grid_h,
        grid_w=grid_w,
        window_h=cfg.window_h,
        window_w=cfg.window_w,
        one_sided=cfg.one_sided,
        shift_h=cfg.shift_h,
        shift_w=cfg.shift_w,
        include_self=cfg.include_self,
        device=device,
    )
    enabled = bool(mask.sum().item() < count * count)
    reason = "local_window" if enabled else "covers_full_attention"
    return _plan(cfg, mask, count, grid_h, grid_w, enabled=enabled, reason=reason)


def dit_local_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    policy: DiTLocalWindowAttentionPolicy | Mapping[str, Any] | None = None,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, DiTLocalWindowAttentionPlan]:
    _validate_qkv(query, key, value)
    token_count = int(query.shape[-2])
    plan = build_dit_local_window_attention_plan(token_count, policy, device=query.device)
    output = _masked_attention(query, key, value, plan.mask, scale=scale)
    return output, plan


def dit_local_window_attention_gather(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    policy: DiTLocalWindowAttentionPolicy | Mapping[str, Any] | None = None,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, DiTLocalWindowAttentionPlan, dict[str, Any]]:
    _validate_qkv(query, key, value)
    token_count = int(query.shape[-2])
    plan = build_dit_local_window_attention_plan(token_count, policy, device=query.device)
    output, kernel = _gathered_window_attention(query, key, value, plan.mask, scale=scale)
    return output, plan, kernel


def build_dit_local_window_attention_scorecard(
    plan: DiTLocalWindowAttentionPlan | Mapping[str, Any],
    *,
    shape_stable: bool = False,
    disabled_parity_ok: bool = False,
    observed_loss_delta: float | None = None,
    max_allowed_loss_delta: float = 0.0,
    observed_quality_drift: float | None = None,
    max_allowed_quality_drift: float = 0.0,
) -> dict[str, Any]:
    payload = plan.as_dict() if isinstance(plan, DiTLocalWindowAttentionPlan) else dict(plan)
    loss_ready = observed_loss_delta is not None
    quality_ready = observed_quality_drift is not None
    loss_ok = loss_ready and float(observed_loss_delta) <= max(float(max_allowed_loss_delta), 0.0)
    quality_ok = quality_ready and float(observed_quality_drift) <= max(float(max_allowed_quality_drift), 0.0)
    blockers: list[str] = []
    if not bool(payload.get("enabled")):
        blockers.append("local_window_not_enabled")
    if not shape_stable:
        blockers.append("shape_stability_evidence_missing")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_evidence_missing")
    if not loss_ready:
        blockers.append("loss_parity_gate_missing")
    elif not loss_ok:
        blockers.append("loss_delta_above_threshold")
    if not quality_ready:
        blockers.append("quality_drift_gate_missing")
    elif not quality_ok:
        blockers.append("quality_drift_above_threshold")
    blockers.append("real_anima_newbie_2k_ab_missing")
    return {
        "schema_version": 1,
        "scorecard": "dit_local_window_attention_probe_v0",
        "reducer_id": "local_window_attention",
        "ok": not blockers[:-1],
        "probe_ready": True,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "shape_stable": bool(shape_stable),
        "disabled_parity_ok": bool(disabled_parity_ok),
        "plan": payload,
        "loss_gate": {
            "observed_loss_delta": observed_loss_delta,
            "max_allowed_loss_delta": float(max_allowed_loss_delta),
            "ready": loss_ready,
            "ok": loss_ok,
        },
        "quality_gate": {
            "observed_quality_drift": observed_quality_drift,
            "max_allowed_quality_drift": float(max_allowed_quality_drift),
            "ready": quality_ready,
            "ok": quality_ok,
        },
        "blocked_reasons": blockers,
        "recommended_next_step": "run real 2K+ Anima/Newbie local-window attention A/B",
    }


def _build_2d_window_mask(
    *,
    grid_h: int,
    grid_w: int,
    window_h: int,
    window_w: int,
    one_sided: bool,
    shift_h: int,
    shift_w: int,
    include_self: bool,
    device: torch.device | str | None,
) -> torch.Tensor:
    rows = torch.arange(grid_h, device=device)
    cols = torch.arange(grid_w, device=device)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.stack(((yy + shift_h) % grid_h, (xx + shift_w) % grid_w), dim=-1).reshape(-1, 2)
    q = coords[:, None, :]
    k = coords[None, :, :]
    dr = q[..., 0] - k[..., 0]
    dc = q[..., 1] - k[..., 1]
    row_radius = max(int(window_h) // 2, 0)
    col_radius = max(int(window_w) // 2, 0)
    mask = (dr.abs() <= row_radius) & (dc.abs() <= col_radius)
    if one_sided:
        q_rank = torch.arange(grid_h * grid_w, device=device)[:, None]
        k_rank = torch.arange(grid_h * grid_w, device=device)[None, :]
        mask = mask & (k_rank <= q_rank)
    if include_self:
        mask.fill_diagonal_(True)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("local window attention mask produced an empty query row")
    return mask


def _masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale: float | None,
) -> torch.Tensor:
    head_dim = int(query.shape[-1])
    resolved_scale = float(scale) if scale is not None else head_dim ** -0.5
    attn = torch.matmul(query, key.transpose(-2, -1)) * resolved_scale
    bias = torch.where(mask, 0.0, float("-inf")).to(dtype=attn.dtype, device=attn.device)
    weights = F.softmax(attn + bias, dim=-1)
    return torch.matmul(weights, value)


def _gathered_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale: float | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    token_count = int(query.shape[-2])
    index, valid = _window_index_from_mask(mask)
    safe_index = index.clamp_min(0)
    gathered_key = key[:, :, safe_index, :]
    gathered_value = value[:, :, safe_index, :]
    head_dim = int(query.shape[-1])
    resolved_scale = float(scale) if scale is not None else head_dim ** -0.5
    scores = (query.unsqueeze(-2) * gathered_key).sum(dim=-1) * resolved_scale
    scores = scores.masked_fill(~valid.view(1, 1, token_count, -1), float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = (weights.unsqueeze(-1) * gathered_value).sum(dim=-2)
    return output, {
        "kernel": "torch_gather_window_attention_v0",
        "dense_score_matrix_built": False,
        "token_count": token_count,
        "max_window_tokens": int(index.shape[-1]),
        "score_elements": int(token_count * index.shape[-1]),
        "dense_score_elements": int(token_count * token_count),
        "estimated_score_fraction": float((token_count * index.shape[-1]) / max(token_count * token_count, 1)),
    }


def _window_index_from_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim != 2 or int(mask.shape[0]) != int(mask.shape[1]):
        raise ValueError("mask must have shape [tokens, tokens]")
    counts = mask.sum(dim=1)
    max_count = int(counts.max().item())
    if max_count <= 0:
        raise ValueError("local window attention mask produced an empty query row")
    token_count = int(mask.shape[0])
    index = torch.full((token_count, max_count), 0, dtype=torch.long, device=mask.device)
    valid = torch.zeros((token_count, max_count), dtype=torch.bool, device=mask.device)
    for row in range(token_count):
        cols = torch.nonzero(mask[row], as_tuple=False).flatten()
        width = int(cols.numel())
        index[row, :width] = cols
        valid[row, :width] = True
    return index, valid


def _plan(
    cfg: DiTLocalWindowAttentionPolicy,
    mask: torch.Tensor,
    token_count: int,
    grid_h: int,
    grid_w: int,
    *,
    enabled: bool,
    reason: str,
) -> DiTLocalWindowAttentionPlan:
    return DiTLocalWindowAttentionPlan(
        enabled=bool(enabled),
        token_count=int(token_count),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        window_h=int(cfg.window_h),
        window_w=int(cfg.window_w),
        one_sided=bool(cfg.one_sided),
        shift_h=int(cfg.shift_h),
        shift_w=int(cfg.shift_w),
        mask=mask,
        reason=reason,
    )


def _resolve_grid(token_count: int, grid_h: int, grid_w: int) -> tuple[int, int]:
    if grid_h > 0 and grid_w > 0:
        if grid_h * grid_w != token_count:
            raise ValueError("grid_h * grid_w must match token_count")
        return grid_h, grid_w
    side = int(math.isqrt(token_count))
    if side * side == token_count:
        return side, side
    return 1, token_count


def _policy(policy: DiTLocalWindowAttentionPolicy | Mapping[str, Any] | None) -> DiTLocalWindowAttentionPolicy:
    if isinstance(policy, Mapping):
        return DiTLocalWindowAttentionPolicy(**policy).normalized()
    return (policy or DiTLocalWindowAttentionPolicy()).normalized()


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
    "DiTLocalWindowAttentionPlan",
    "DiTLocalWindowAttentionPolicy",
    "build_dit_local_window_attention_plan",
    "build_dit_local_window_attention_scorecard",
    "dit_local_window_attention",
    "dit_local_window_attention_gather",
]
