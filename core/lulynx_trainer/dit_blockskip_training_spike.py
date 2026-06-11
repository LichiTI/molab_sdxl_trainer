"""Research-only DiT BlockSkip training primitive.

The primitive models deterministic training-time block skipping with residual
reuse. It is default-off, does not patch real DiT modules, and keeps promotion
blocked until real loss/quality evidence exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class DiTBlockSkipPolicy:
    enabled: bool = False
    skip_ratio: float = 0.0
    warmup_steps: int = 0
    stop_step: int = -1
    min_block: int = 0
    max_block: int = -1
    skip_every: int = 0
    reuse_residual: bool = True

    def normalized(self) -> "DiTBlockSkipPolicy":
        skip_ratio = _clamp(float(self.skip_ratio or 0.0), 0.0, 0.95)
        skip_every = max(int(self.skip_every or 0), 0)
        if skip_every == 0 and skip_ratio > 0.0:
            skip_every = max(int(round(1.0 / skip_ratio)), 1)
        return DiTBlockSkipPolicy(
            enabled=bool(self.enabled),
            skip_ratio=skip_ratio,
            warmup_steps=max(int(self.warmup_steps or 0), 0),
            stop_step=int(self.stop_step if self.stop_step is not None else -1),
            min_block=max(int(self.min_block or 0), 0),
            max_block=int(self.max_block if self.max_block is not None else -1),
            skip_every=skip_every,
            reuse_residual=bool(self.reuse_residual),
        )


@dataclass(frozen=True)
class DiTBlockSkipDecision:
    block_index: int
    step_index: int
    total_blocks: int
    total_steps: int
    skip: bool
    reuse_residual: bool
    reason: str
    estimated_block_compute_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_index": int(self.block_index),
            "step_index": int(self.step_index),
            "total_blocks": int(self.total_blocks),
            "total_steps": int(self.total_steps),
            "skip": bool(self.skip),
            "reuse_residual": bool(self.reuse_residual),
            "reason": self.reason,
            "estimated_block_compute_fraction": float(self.estimated_block_compute_fraction),
        }


@dataclass(frozen=True)
class DiTBlockSkipPlan:
    enabled: bool
    step_index: int
    total_blocks: int
    total_steps: int
    decisions: tuple[DiTBlockSkipDecision, ...]
    policy: DiTBlockSkipPolicy

    @property
    def skipped_blocks(self) -> int:
        return sum(1 for decision in self.decisions if decision.skip)

    @property
    def estimated_block_compute_fraction(self) -> float:
        if self.total_blocks <= 0:
            return 1.0
        return float((self.total_blocks - self.skipped_blocks) / self.total_blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "step_index": int(self.step_index),
            "total_blocks": int(self.total_blocks),
            "total_steps": int(self.total_steps),
            "skipped_blocks": int(self.skipped_blocks),
            "estimated_block_compute_fraction": float(self.estimated_block_compute_fraction),
            "policy": {
                "skip_ratio": float(self.policy.skip_ratio),
                "warmup_steps": int(self.policy.warmup_steps),
                "stop_step": int(self.policy.stop_step),
                "min_block": int(self.policy.min_block),
                "max_block": int(self.policy.max_block),
                "skip_every": int(self.policy.skip_every),
                "reuse_residual": bool(self.policy.reuse_residual),
            },
            "decisions": [decision.as_dict() for decision in self.decisions],
        }


def build_dit_blockskip_plan(
    *,
    total_blocks: int,
    step_index: int,
    total_steps: int = 0,
    policy: DiTBlockSkipPolicy | Mapping[str, Any] | None = None,
) -> DiTBlockSkipPlan:
    cfg = _policy(policy)
    blocks = int(total_blocks)
    if blocks <= 0:
        raise ValueError("total_blocks must be positive")
    step = int(step_index)
    steps = max(int(total_steps or 0), 0)
    max_block = blocks - 1 if cfg.max_block < 0 else min(cfg.max_block, blocks - 1)
    decisions = tuple(
        _decide_block(
            block_index=block_index,
            step_index=step,
            total_blocks=blocks,
            total_steps=steps,
            min_block=cfg.min_block,
            max_block=max_block,
            policy=cfg,
        )
        for block_index in range(blocks)
    )
    return DiTBlockSkipPlan(
        enabled=bool(cfg.enabled and any(decision.skip for decision in decisions)),
        step_index=step,
        total_blocks=blocks,
        total_steps=steps,
        decisions=decisions,
        policy=cfg,
    )


def apply_dit_blockskip_decision(
    tokens: torch.Tensor,
    block: Callable[[torch.Tensor], torch.Tensor],
    decision: DiTBlockSkipDecision,
    *,
    cached_residual: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_tokens(tokens, "tokens")
    if not decision.skip:
        output = block(tokens)
        _validate_same_shape(output, tokens, "block output")
        return output
    if not decision.reuse_residual:
        return tokens
    residual = cached_residual if cached_residual is not None else tokens
    _validate_same_shape(residual, tokens, "cached residual")
    return residual


def run_dit_blockskip_sequence(
    tokens: torch.Tensor,
    blocks: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    plan: DiTBlockSkipPlan,
    *,
    cached_residuals: Sequence[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    if len(blocks) != plan.total_blocks:
        raise ValueError("block count does not match the blockskip plan")
    if cached_residuals is not None and len(cached_residuals) != len(blocks):
        raise ValueError("cached_residuals must match the block count")
    output = tokens
    for idx, block in enumerate(blocks):
        residual = None if cached_residuals is None else cached_residuals[idx]
        output = apply_dit_blockskip_decision(output, block, plan.decisions[idx], cached_residual=residual)
    return output


def build_dit_blockskip_training_scorecard(
    plan: DiTBlockSkipPlan | Mapping[str, Any],
    *,
    shape_stable: bool = False,
    disabled_parity_ok: bool = False,
    observed_loss_delta: float | None = None,
    max_allowed_loss_delta: float = 0.0,
    observed_quality_drift: float | None = None,
    max_allowed_quality_drift: float = 0.0,
) -> dict[str, Any]:
    payload = plan.as_dict() if isinstance(plan, DiTBlockSkipPlan) else dict(plan)
    loss_ready = observed_loss_delta is not None
    quality_ready = observed_quality_drift is not None
    loss_ok = loss_ready and float(observed_loss_delta) <= max(float(max_allowed_loss_delta), 0.0)
    quality_ok = quality_ready and float(observed_quality_drift) <= max(float(max_allowed_quality_drift), 0.0)
    blockers: list[str] = []
    if not bool(payload.get("enabled")):
        blockers.append("no_blocks_skipped")
    if not shape_stable:
        blockers.append("shape_stability_evidence_missing")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_evidence_missing")
    if not loss_ready:
        blockers.append("real_loss_parity_missing")
    elif not loss_ok:
        blockers.append("loss_delta_above_threshold")
    if not quality_ready:
        blockers.append("quality_drift_gate_missing")
    elif not quality_ok:
        blockers.append("quality_drift_above_threshold")
    blockers.append("real_anima_newbie_ab_missing")
    return {
        "schema_version": 1,
        "scorecard": "dit_blockskip_training_spike_v0",
        "reducer_id": "blockskip",
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
        "recommended_next_step": "run real Anima/Newbie training loss and quality A/B",
    }


def _decide_block(
    *,
    block_index: int,
    step_index: int,
    total_blocks: int,
    total_steps: int,
    min_block: int,
    max_block: int,
    policy: DiTBlockSkipPolicy,
) -> DiTBlockSkipDecision:
    skip = False
    reason = "disabled"
    if policy.enabled:
        if policy.skip_every <= 0:
            reason = "skip_ratio_zero"
        elif step_index < policy.warmup_steps:
            reason = "warmup"
        elif policy.stop_step >= 0 and step_index >= policy.stop_step:
            reason = "tail_stop"
        elif block_index < min_block or block_index > max_block:
            reason = "block_outside_range"
        else:
            candidate = block_index - min_block
            skip = (candidate + step_index) % policy.skip_every == 0
            reason = "scheduled_skip" if skip else "scheduled_forward"
    estimated_fraction = (total_blocks - 1) / total_blocks if skip else 1.0
    return DiTBlockSkipDecision(
        block_index=int(block_index),
        step_index=int(step_index),
        total_blocks=int(total_blocks),
        total_steps=int(total_steps),
        skip=bool(skip),
        reuse_residual=bool(policy.reuse_residual),
        reason=reason,
        estimated_block_compute_fraction=float(estimated_fraction),
    )


def _policy(policy: DiTBlockSkipPolicy | Mapping[str, Any] | None) -> DiTBlockSkipPolicy:
    if isinstance(policy, Mapping):
        return DiTBlockSkipPolicy(**policy).normalized()
    return (policy or DiTBlockSkipPolicy()).normalized()


def _validate_tokens(tokens: torch.Tensor, name: str) -> None:
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tokens.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, tokens, hidden]")


def _validate_same_shape(value: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    _validate_tokens(value, name)
    if tuple(value.shape) != tuple(reference.shape):
        raise ValueError(f"{name} shape must match tokens")


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
