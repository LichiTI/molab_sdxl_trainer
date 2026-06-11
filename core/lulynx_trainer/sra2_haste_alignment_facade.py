"""Default-off SRA2 + HASTE alignment-loss facade.

The facade composes the existing VAE self-representation alignment primitive
with the HASTE schedule guard. It is intentionally trainer-neutral so future
runtime wiring can call one small contract instead of duplicating gate logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .haste_alignment_schedule import HasteAlignmentSchedulePolicy, resolve_haste_alignment_weight
from .sra2_vae_alignment import SRA2VaeAlignmentPolicy, sra2_vae_alignment_loss


@dataclass(frozen=True)
class SRA2HasteAlignmentPolicy:
    enabled: bool = False
    loss_type: str = "cosine"
    normalize_targets: bool = True
    stop_grad_target: bool = True
    base_weight: float = 1.0
    start_step: int = 0
    stop_step: int = -1
    decay_start_step: int = -1
    decay_end_step: int = -1
    min_weight: float = 0.0
    plateau_patience: int = 0
    min_relative_improvement: float = 0.0

    def normalized(self) -> "SRA2HasteAlignmentPolicy":
        schedule = self.schedule_policy().normalized()
        sra2 = self.sra2_policy(weight=schedule.base_weight).normalized()
        return SRA2HasteAlignmentPolicy(
            enabled=bool(self.enabled),
            loss_type=sra2.loss_type,
            normalize_targets=sra2.normalize_targets,
            stop_grad_target=sra2.stop_grad_target,
            base_weight=schedule.base_weight,
            start_step=schedule.start_step,
            stop_step=schedule.stop_step,
            decay_start_step=schedule.decay_start_step,
            decay_end_step=schedule.decay_end_step,
            min_weight=schedule.min_weight,
            plateau_patience=schedule.plateau_patience,
            min_relative_improvement=schedule.min_relative_improvement,
        )

    def schedule_policy(self) -> HasteAlignmentSchedulePolicy:
        return HasteAlignmentSchedulePolicy(
            enabled=self.enabled,
            base_weight=self.base_weight,
            start_step=self.start_step,
            stop_step=self.stop_step,
            decay_start_step=self.decay_start_step,
            decay_end_step=self.decay_end_step,
            min_weight=self.min_weight,
            plateau_patience=self.plateau_patience,
            min_relative_improvement=self.min_relative_improvement,
        )

    def sra2_policy(self, *, weight: float) -> SRA2VaeAlignmentPolicy:
        return SRA2VaeAlignmentPolicy(
            enabled=self.enabled,
            weight=weight,
            loss_type=self.loss_type,
            normalize_targets=self.normalize_targets,
            stop_grad_target=self.stop_grad_target,
        )


def sra2_haste_alignment_loss(
    hidden_states: torch.Tensor,
    vae_features: torch.Tensor,
    policy: SRA2HasteAlignmentPolicy | Mapping[str, Any] | None = None,
    *,
    step: int = 0,
    total_steps: int = 0,
    loss_history: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    cfg = _policy(policy)
    schedule = resolve_haste_alignment_weight(
        step=step,
        total_steps=total_steps,
        policy=cfg.schedule_policy(),
        loss_history=loss_history,
    )
    if not schedule.active:
        loss, sra2_profile = sra2_vae_alignment_loss(
            hidden_states,
            vae_features,
            cfg.sra2_policy(weight=0.0),
        )
        return loss, _profile(cfg, schedule.as_dict(), sra2_profile)

    loss, sra2_profile = sra2_vae_alignment_loss(
        hidden_states,
        vae_features,
        cfg.sra2_policy(weight=schedule.weight),
    )
    return loss, _profile(cfg, schedule.as_dict(), sra2_profile)


def build_sra2_haste_alignment_scorecard(profile: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(profile)
    active = bool(payload.get("active"))
    blockers = []
    if not active:
        blockers.append("alignment_not_active")
    blockers.extend(["trainer_wiring_missing", "real_quality_gate_missing"])
    return {
        "schema_version": 1,
        "scorecard": "sra2_haste_alignment_facade_v0",
        "ok": bool(payload.get("facade_ready")),
        "facade_ready": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "profile": payload,
        "blocked_reasons": blockers,
        "recommended_next_step": "wire default-off trainer loss accumulation after real quality gate design",
    }


def _policy(policy: SRA2HasteAlignmentPolicy | Mapping[str, Any] | None) -> SRA2HasteAlignmentPolicy:
    if isinstance(policy, Mapping):
        return SRA2HasteAlignmentPolicy(**policy).normalized()
    return (policy or SRA2HasteAlignmentPolicy()).normalized()


def _profile(
    cfg: SRA2HasteAlignmentPolicy,
    schedule_profile: Mapping[str, Any],
    sra2_profile: Mapping[str, Any],
) -> dict[str, Any]:
    active = bool(schedule_profile.get("active")) and bool(sra2_profile.get("enabled"))
    return {
        "facade": "sra2_haste_alignment_facade_v0",
        "facade_ready": True,
        "enabled": bool(cfg.enabled),
        "active": active,
        "reason": sra2_profile.get("reason") if active else schedule_profile.get("reason", "disabled"),
        "weight": float(schedule_profile.get("weight") or 0.0),
        "schedule": dict(schedule_profile),
        "sra2": dict(sra2_profile),
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


__all__ = [
    "SRA2HasteAlignmentPolicy",
    "build_sra2_haste_alignment_scorecard",
    "sra2_haste_alignment_loss",
]
