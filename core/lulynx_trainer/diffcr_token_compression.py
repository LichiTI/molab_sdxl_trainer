"""DiffCR-style adaptive token compression probe for DiT routes.

This primitive clusters tokens by a deterministic score order, averages each
group, and can expand compressed outputs back to the original token count. It is
default-off research plumbing and is not wired into the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import torch


@dataclass(frozen=True)
class DiffCRTokenCompressionPolicy:
    enabled: bool = False
    compression_ratio: float = 1.0
    min_tokens: int = 1
    score_mode: str = "l2"
    layer_ratios: Mapping[int, float] | None = None
    timestep_ratios: Mapping[int, float] | None = None

    def normalized(self) -> "DiffCRTokenCompressionPolicy":
        score_mode = str(self.score_mode or "l2").strip().lower()
        if score_mode not in {"l2", "abs_mean", "provided", "sequential"}:
            score_mode = "l2"
        return DiffCRTokenCompressionPolicy(
            enabled=bool(self.enabled),
            compression_ratio=_clamp_ratio(self.compression_ratio),
            min_tokens=max(int(self.min_tokens or 1), 1),
            score_mode=score_mode,
            layer_ratios=_normalize_ratio_map(self.layer_ratios),
            timestep_ratios=_normalize_ratio_map(self.timestep_ratios),
        )


@dataclass(frozen=True)
class DiffCRCompressionPlan:
    enabled: bool
    batch_size: int
    token_count: int
    compressed_count: int
    assignment: torch.Tensor
    score_order: torch.Tensor
    ratio: float
    layer_index: int
    timestep_index: int
    score_mode: str
    reason: str

    @property
    def estimated_attention_fraction(self) -> float:
        if self.token_count <= 0:
            return 1.0
        kept = self.compressed_count / self.token_count
        return float(kept * kept)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "batch_size": int(self.batch_size),
            "token_count": int(self.token_count),
            "compressed_count": int(self.compressed_count),
            "ratio": float(self.ratio),
            "layer_index": int(self.layer_index),
            "timestep_index": int(self.timestep_index),
            "score_mode": self.score_mode,
            "estimated_attention_fraction": float(self.estimated_attention_fraction),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DiffCRCompressedTokens:
    tokens: torch.Tensor
    plan: DiffCRCompressionPlan


def build_diffcr_compression_plan(
    tokens: torch.Tensor,
    policy: DiffCRTokenCompressionPolicy | Mapping[str, Any] | None = None,
    *,
    layer_index: int = -1,
    timestep_index: int = -1,
    scores: torch.Tensor | None = None,
) -> DiffCRCompressionPlan:
    batch_size, token_count, _hidden_size = _validate_tokens(tokens)
    cfg = _policy(policy)
    layer = int(layer_index)
    timestep = int(timestep_index)
    ratio = _resolve_ratio(cfg, layer, timestep)
    compressed_count = min(max(int(math.ceil(token_count * ratio)), cfg.min_tokens), token_count)
    enabled = bool(cfg.enabled and compressed_count < token_count)
    reason = "compressed" if enabled else ("all_tokens_kept" if cfg.enabled else "disabled")
    order = _score_order(tokens, cfg.score_mode, scores)
    assignment = _assignment_from_order(order, compressed_count)
    return DiffCRCompressionPlan(
        enabled=enabled,
        batch_size=batch_size,
        token_count=token_count,
        compressed_count=compressed_count,
        assignment=assignment,
        score_order=order,
        ratio=float(compressed_count / token_count),
        layer_index=layer,
        timestep_index=timestep,
        score_mode=cfg.score_mode,
        reason=reason,
    )


def compress_diffcr_tokens(
    tokens: torch.Tensor,
    policy: DiffCRTokenCompressionPolicy | Mapping[str, Any] | None = None,
    *,
    layer_index: int = -1,
    timestep_index: int = -1,
    scores: torch.Tensor | None = None,
) -> DiffCRCompressedTokens:
    plan = build_diffcr_compression_plan(
        tokens,
        policy,
        layer_index=layer_index,
        timestep_index=timestep_index,
        scores=scores,
    )
    if not plan.enabled:
        return DiffCRCompressedTokens(tokens=tokens, plan=plan)
    compressed = _mean_by_assignment(tokens, plan.assignment, plan.compressed_count)
    return DiffCRCompressedTokens(tokens=compressed, plan=plan)


def expand_diffcr_tokens(compressed_tokens: torch.Tensor, plan: DiffCRCompressionPlan) -> torch.Tensor:
    if compressed_tokens.ndim != 3:
        raise ValueError("compressed_tokens must have shape [batch, tokens, hidden]")
    if compressed_tokens.shape[0] != plan.batch_size:
        raise ValueError("compressed token batch does not match the plan")
    if compressed_tokens.shape[1] != plan.compressed_count:
        raise ValueError("compressed token count does not match the plan")
    if not plan.enabled:
        return compressed_tokens
    hidden = int(compressed_tokens.shape[-1])
    expanded_index = plan.assignment.unsqueeze(-1).expand(-1, -1, hidden).to(device=compressed_tokens.device)
    return torch.gather(compressed_tokens, dim=1, index=expanded_index)


def run_diffcr_compressed_block(
    tokens: torch.Tensor,
    block: Callable[[torch.Tensor], torch.Tensor],
    policy: DiffCRTokenCompressionPolicy | Mapping[str, Any] | None = None,
    *,
    layer_index: int = -1,
    timestep_index: int = -1,
    scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, DiffCRCompressionPlan]:
    compressed = compress_diffcr_tokens(
        tokens,
        policy,
        layer_index=layer_index,
        timestep_index=timestep_index,
        scores=scores,
    )
    block_out = block(compressed.tokens)
    if not compressed.plan.enabled:
        return block_out, compressed.plan
    return expand_diffcr_tokens(block_out, compressed.plan), compressed.plan


def build_diffcr_token_compression_scorecard(
    plan: DiffCRCompressionPlan | Mapping[str, Any],
    *,
    shape_stable: bool = False,
    disabled_parity_ok: bool = False,
    observed_loss_delta: float | None = None,
    max_allowed_loss_delta: float = 0.0,
    max_allowed_quality_drift: float = 0.0,
    observed_quality_drift: float | None = None,
) -> dict[str, Any]:
    payload = plan.as_dict() if isinstance(plan, DiffCRCompressionPlan) else dict(plan)
    loss_ready = observed_loss_delta is not None
    loss_ok = loss_ready and float(observed_loss_delta) <= max(float(max_allowed_loss_delta), 0.0)
    drift_ready = observed_quality_drift is not None
    drift_ok = drift_ready and float(observed_quality_drift) <= max(float(max_allowed_quality_drift), 0.0)
    blockers: list[str] = []
    if not bool(payload.get("enabled")):
        blockers.append("compression_not_enabled")
    if not shape_stable:
        blockers.append("shape_stability_evidence_missing")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_evidence_missing")
    if not loss_ready:
        blockers.append("loss_parity_gate_missing")
    elif not loss_ok:
        blockers.append("loss_parity_gate_failed")
    if not drift_ready:
        blockers.append("real_quality_drift_measurement_missing")
    elif not drift_ok:
        blockers.append("quality_drift_above_threshold")
    return {
        "schema_version": 1,
        "scorecard": "diffcr_token_compression_probe_v0",
        "reducer_id": "diffcr",
        "ok": not blockers,
        "probe_ready": True,
        "training_path_enabled": False,
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
            "ready": drift_ready,
            "ok": drift_ok,
        },
        "blocked_reasons": blockers,
        "recommended_next_step": "run high-resolution Anima/Newbie quality-drift A/B",
    }


def _validate_tokens(tokens: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, hidden]")
    batch_size, token_count, hidden_size = tokens.shape
    if batch_size <= 0 or token_count <= 0 or hidden_size <= 0:
        raise ValueError("tokens must have non-empty dimensions")
    return int(batch_size), int(token_count), int(hidden_size)


def _policy(policy: DiffCRTokenCompressionPolicy | Mapping[str, Any] | None) -> DiffCRTokenCompressionPolicy:
    if isinstance(policy, Mapping):
        return DiffCRTokenCompressionPolicy(**policy).normalized()
    return (policy or DiffCRTokenCompressionPolicy()).normalized()


def _resolve_ratio(cfg: DiffCRTokenCompressionPolicy, layer_index: int, timestep_index: int) -> float:
    ratio = cfg.compression_ratio
    if cfg.layer_ratios and layer_index in cfg.layer_ratios:
        ratio = cfg.layer_ratios[layer_index]
    if cfg.timestep_ratios and timestep_index in cfg.timestep_ratios:
        ratio = cfg.timestep_ratios[timestep_index]
    return _clamp_ratio(ratio)


def _score_order(tokens: torch.Tensor, score_mode: str, scores: torch.Tensor | None) -> torch.Tensor:
    batch, count, _hidden = tokens.shape
    if score_mode == "sequential":
        return torch.arange(count, device=tokens.device, dtype=torch.long).expand(batch, -1).clone()
    if scores is not None:
        if scores.shape != tokens.shape[:2]:
            raise ValueError("scores must have shape [batch, tokens]")
        values = scores.to(device=tokens.device, dtype=torch.float32)
    elif score_mode == "abs_mean":
        values = tokens.detach().abs().float().mean(dim=-1)
    else:
        values = tokens.detach().float().pow(2).mean(dim=-1)
    return torch.argsort(values, dim=1, descending=True, stable=True)


def _assignment_from_order(order: torch.Tensor, compressed_count: int) -> torch.Tensor:
    batch, token_count = order.shape
    ranks = torch.arange(token_count, device=order.device, dtype=torch.long).expand(batch, -1)
    group_by_rank = torch.clamp((ranks * compressed_count) // token_count, max=compressed_count - 1)
    assignment = torch.empty_like(order)
    assignment.scatter_(1, order, group_by_rank)
    return assignment


def _mean_by_assignment(tokens: torch.Tensor, assignment: torch.Tensor, compressed_count: int) -> torch.Tensor:
    batch, _token_count, hidden = tokens.shape
    out = tokens.new_zeros((batch, compressed_count, hidden))
    counts = tokens.new_zeros((batch, compressed_count, 1))
    scatter_index = assignment.unsqueeze(-1).expand(-1, -1, hidden)
    out.scatter_add_(1, scatter_index, tokens)
    counts.scatter_add_(1, assignment.unsqueeze(-1), tokens.new_ones((batch, assignment.shape[1], 1)))
    return out / counts.clamp_min(1.0)


def _normalize_ratio_map(value: Mapping[int, float] | None) -> dict[int, float] | None:
    if not value:
        return None
    return {int(key): _clamp_ratio(val) for key, val in value.items()}


def _clamp_ratio(value: Any) -> float:
    ratio = 1.0 if value is None else float(value)
    return min(max(ratio, 0.0), 1.0)


__all__ = [
    "DiffCRCompressedTokens",
    "DiffCRCompressionPlan",
    "DiffCRTokenCompressionPolicy",
    "build_diffcr_compression_plan",
    "build_diffcr_token_compression_scorecard",
    "compress_diffcr_tokens",
    "expand_diffcr_tokens",
    "run_diffcr_compressed_block",
]
