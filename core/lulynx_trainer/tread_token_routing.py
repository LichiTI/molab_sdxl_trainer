"""Tiny TREAD-style token routing fixture for DiT training research.

This module is intentionally a primitive-only proof. It keeps the disabled path
exactly equivalent to the original block call and, when enabled, routes a fixed
fraction of high-score tokens through a provided block while carrying the
remaining tokens forward for later reintegration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import torch


@dataclass(frozen=True)
class TreadTokenRoutePolicy:
    enabled: bool = False
    keep_ratio: float = 1.0
    min_keep_tokens: int = 1
    score_mode: str = "l2"

    def normalized(self) -> "TreadTokenRoutePolicy":
        keep_ratio = 1.0 if self.keep_ratio is None else float(self.keep_ratio)
        min_keep_tokens = 1 if self.min_keep_tokens is None else int(self.min_keep_tokens)
        score_mode = str(self.score_mode or "l2").strip().lower()
        if score_mode not in {"l2", "abs_mean", "provided"}:
            score_mode = "l2"
        return TreadTokenRoutePolicy(
            enabled=bool(self.enabled),
            keep_ratio=min(max(keep_ratio, 0.0), 1.0),
            min_keep_tokens=max(min_keep_tokens, 1),
            score_mode=score_mode,
        )


@dataclass(frozen=True)
class TreadTokenRoutePlan:
    enabled: bool
    batch_size: int
    token_count: int
    keep_count: int
    keep_indices: torch.Tensor
    drop_indices: torch.Tensor
    keep_mask: torch.Tensor
    score_mode: str
    reason: str

    @property
    def drop_count(self) -> int:
        return max(self.token_count - self.keep_count, 0)

    @property
    def kept_fraction(self) -> float:
        if self.token_count <= 0:
            return 1.0
        return float(self.keep_count / self.token_count)

    @property
    def estimated_attention_fraction(self) -> float:
        fraction = self.kept_fraction
        return float(fraction * fraction)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "batch_size": int(self.batch_size),
            "token_count": int(self.token_count),
            "keep_count": int(self.keep_count),
            "drop_count": int(self.drop_count),
            "kept_fraction": float(self.kept_fraction),
            "estimated_attention_fraction": float(self.estimated_attention_fraction),
            "score_mode": self.score_mode,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TreadRoutedTokens:
    kept: torch.Tensor
    dropped: torch.Tensor
    plan: TreadTokenRoutePlan


def _validate_tokens(tokens: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, hidden]")
    batch_size, token_count, hidden_size = tokens.shape
    if batch_size <= 0 or token_count <= 0 or hidden_size <= 0:
        raise ValueError("tokens must have non-empty batch, token, and hidden dimensions")
    return int(batch_size), int(token_count), int(hidden_size)


def _score_tokens(tokens: torch.Tensor, policy: TreadTokenRoutePolicy, scores: torch.Tensor | None) -> torch.Tensor:
    if scores is not None:
        if scores.shape != tokens.shape[:2]:
            raise ValueError("scores must have shape [batch, tokens]")
        return scores.to(device=tokens.device, dtype=torch.float32)
    if policy.score_mode == "abs_mean":
        return tokens.detach().abs().float().mean(dim=-1)
    return tokens.detach().float().pow(2).mean(dim=-1)


def _gather_token_dim(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return tokens[:, :0, :]
    expanded = indices.to(device=tokens.device).unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=expanded)


def build_tread_token_route_plan(
    tokens: torch.Tensor,
    policy: TreadTokenRoutePolicy | Mapping[str, Any] | None = None,
    *,
    scores: torch.Tensor | None = None,
) -> TreadTokenRoutePlan:
    batch_size, token_count, _hidden_size = _validate_tokens(tokens)
    if isinstance(policy, Mapping):
        policy = TreadTokenRoutePolicy(**policy)
    normalized = (policy or TreadTokenRoutePolicy()).normalized()

    if not normalized.enabled:
        keep_count = token_count
        reason = "disabled"
    else:
        keep_count = int(math.ceil(token_count * normalized.keep_ratio))
        keep_count = min(max(keep_count, normalized.min_keep_tokens), token_count)
        reason = "all_tokens_kept" if keep_count >= token_count else "routed"

    if keep_count >= token_count:
        keep_indices = torch.arange(token_count, device=tokens.device, dtype=torch.long).expand(batch_size, -1).clone()
        drop_indices = torch.empty((batch_size, 0), device=tokens.device, dtype=torch.long)
        keep_mask = torch.ones((batch_size, token_count), device=tokens.device, dtype=torch.bool)
        return TreadTokenRoutePlan(
            enabled=bool(normalized.enabled and reason != "disabled"),
            batch_size=batch_size,
            token_count=token_count,
            keep_count=token_count,
            keep_indices=keep_indices,
            drop_indices=drop_indices,
            keep_mask=keep_mask,
            score_mode=normalized.score_mode,
            reason=reason,
        )

    token_scores = _score_tokens(tokens, normalized, scores)
    keep_indices = torch.topk(token_scores, k=keep_count, dim=1, largest=True, sorted=False).indices
    keep_indices = torch.sort(keep_indices, dim=1).values
    keep_mask = torch.zeros((batch_size, token_count), device=tokens.device, dtype=torch.bool)
    keep_mask.scatter_(1, keep_indices, True)
    drop_indices = (~keep_mask).nonzero(as_tuple=False)[:, 1].reshape(batch_size, token_count - keep_count)

    return TreadTokenRoutePlan(
        enabled=True,
        batch_size=batch_size,
        token_count=token_count,
        keep_count=keep_count,
        keep_indices=keep_indices,
        drop_indices=drop_indices,
        keep_mask=keep_mask,
        score_mode=normalized.score_mode,
        reason=reason,
    )


def apply_tread_token_route(
    tokens: torch.Tensor,
    policy: TreadTokenRoutePolicy | Mapping[str, Any] | None = None,
    *,
    scores: torch.Tensor | None = None,
) -> TreadRoutedTokens:
    plan = build_tread_token_route_plan(tokens, policy, scores=scores)
    return TreadRoutedTokens(
        kept=_gather_token_dim(tokens, plan.keep_indices),
        dropped=_gather_token_dim(tokens, plan.drop_indices),
        plan=plan,
    )


def restore_tread_tokens(
    kept_tokens: torch.Tensor,
    dropped_tokens: torch.Tensor,
    plan: TreadTokenRoutePlan,
) -> torch.Tensor:
    if kept_tokens.ndim != 3 or dropped_tokens.ndim != 3:
        raise ValueError("kept_tokens and dropped_tokens must have shape [batch, tokens, hidden]")
    if kept_tokens.shape[0] != plan.batch_size or dropped_tokens.shape[0] != plan.batch_size:
        raise ValueError("token batch size does not match the route plan")
    if kept_tokens.shape[1] != plan.keep_count or dropped_tokens.shape[1] != plan.drop_count:
        raise ValueError("token counts do not match the route plan")
    if kept_tokens.shape[2] != dropped_tokens.shape[2] and plan.drop_count:
        raise ValueError("kept and dropped hidden sizes must match")

    hidden_size = int(kept_tokens.shape[2])
    restored = kept_tokens.new_empty((plan.batch_size, plan.token_count, hidden_size))
    restored.scatter_(1, plan.keep_indices.unsqueeze(-1).expand(-1, -1, hidden_size), kept_tokens)
    if plan.drop_count:
        restored.scatter_(1, plan.drop_indices.unsqueeze(-1).expand(-1, -1, hidden_size), dropped_tokens)
    return restored


def run_tread_routed_block(
    tokens: torch.Tensor,
    block: Callable[[torch.Tensor], torch.Tensor],
    policy: TreadTokenRoutePolicy | Mapping[str, Any] | None = None,
    *,
    scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TreadTokenRoutePlan]:
    routed = apply_tread_token_route(tokens, policy, scores=scores)
    if not routed.plan.enabled:
        return block(tokens), routed.plan
    kept_out = block(routed.kept)
    return restore_tread_tokens(kept_out, routed.dropped, routed.plan), routed.plan


def build_tread_token_route_scorecard(
    plan: TreadTokenRoutePlan | Mapping[str, Any],
    *,
    shape_stable: bool = False,
    disabled_parity_ok: bool = False,
    observed_loss_delta: float | None = None,
    max_allowed_loss_delta: float = 0.0,
    observed_quality_drift: float | None = None,
    max_allowed_quality_drift: float = 0.0,
) -> dict[str, Any]:
    payload = plan.as_dict() if isinstance(plan, TreadTokenRoutePlan) else dict(plan)
    loss_ready = observed_loss_delta is not None
    quality_ready = observed_quality_drift is not None
    loss_ok = loss_ready and float(observed_loss_delta) <= max(float(max_allowed_loss_delta), 0.0)
    quality_ok = quality_ready and float(observed_quality_drift) <= max(float(max_allowed_quality_drift), 0.0)
    blockers: list[str] = []
    if not bool(payload.get("enabled")):
        blockers.append("route_not_enabled")
    if not shape_stable:
        blockers.append("shape_stability_evidence_missing")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_evidence_missing")
    if not loss_ready:
        blockers.append("loss_parity_gate_missing")
    elif not loss_ok:
        blockers.append("loss_parity_gate_failed")
    if not quality_ready:
        blockers.append("quality_gate_missing")
    elif not quality_ok:
        blockers.append("quality_gate_failed")
    blockers.append("real_anima_newbie_ab_missing")
    return {
        "schema_version": 1,
        "scorecard": "tread_token_route_fixture_v0",
        "reducer_id": "tread",
        "ok": not blockers[:-1],
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
            "ready": quality_ready,
            "ok": quality_ok,
        },
        "blocked_reasons": blockers,
        "recommended_next_step": "run real Anima/Newbie TREAD loss and quality A/B",
    }


__all__ = [
    "TreadRoutedTokens",
    "TreadTokenRoutePlan",
    "TreadTokenRoutePolicy",
    "apply_tread_token_route",
    "build_tread_token_route_plan",
    "build_tread_token_route_scorecard",
    "restore_tread_tokens",
    "run_tread_routed_block",
]
