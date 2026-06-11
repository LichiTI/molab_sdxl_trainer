"""Default-off DP-DMD / Turbo distillation primitives for DiT routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DpDmdTurboConfig:
    model_family: str = "anima"
    objective: str = "dp_dmd_turbo"
    student_steps: int = 4
    teacher_steps: int = 28
    guidance_scale: float = 1.0
    prediction_type: str = "velocity"
    diversity_anchor_weight: float = 0.0
    fake_critic_weight: float = 0.0
    fake_critic_margin: float = 0.05
    metadata_version: int = 1
    training_path_enabled: bool = False

    def validate(self) -> None:
        if self.model_family not in {"anima", "newbie"}:
            raise ValueError("model_family must be anima or newbie")
        if self.objective not in {"dp_dmd_turbo", "turbo_dmd"}:
            raise ValueError("objective must be dp_dmd_turbo or turbo_dmd")
        if self.student_steps < 1:
            raise ValueError("student_steps must be >= 1")
        if self.teacher_steps < self.student_steps:
            raise ValueError("teacher_steps must be >= student_steps")
        if self.guidance_scale < 0.0:
            raise ValueError("guidance_scale must be >= 0")
        if self.prediction_type not in {"velocity", "epsilon", "x0"}:
            raise ValueError("prediction_type must be velocity, epsilon, or x0")
        if self.diversity_anchor_weight < 0.0:
            raise ValueError("diversity_anchor_weight must be >= 0")
        if self.fake_critic_weight < 0.0:
            raise ValueError("fake_critic_weight must be >= 0")
        if self.fake_critic_margin < 0.0:
            raise ValueError("fake_critic_margin must be >= 0")
        if self.training_path_enabled:
            raise ValueError("DP-DMD trainer wiring is not enabled by this primitive")


def bake_cfg_prediction(cond: torch.Tensor, uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    if cond.shape != uncond.shape:
        raise ValueError("cond and uncond predictions must have the same shape")
    return uncond + float(guidance_scale) * (cond - uncond)


def distillation_consistency_loss(
    student_pred: torch.Tensor,
    teacher_cond_pred: torch.Tensor,
    *,
    teacher_uncond_pred: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    detach_teacher: bool = True,
) -> torch.Tensor:
    teacher_target = teacher_cond_pred
    if teacher_uncond_pred is not None:
        teacher_target = bake_cfg_prediction(teacher_cond_pred, teacher_uncond_pred, guidance_scale)
    if detach_teacher:
        teacher_target = teacher_target.detach()
    if student_pred.shape != teacher_target.shape:
        raise ValueError("student_pred and teacher target must have the same shape")
    return F.mse_loss(student_pred, teacher_target)


def diversity_anchor_loss(
    student_latents: torch.Tensor,
    anchor_latents: torch.Tensor,
    *,
    weight: float = 1.0,
    detach_anchor: bool = True,
    mode: str = "l2",
) -> torch.Tensor:
    if student_latents.shape != anchor_latents.shape:
        raise ValueError("student_latents and anchor_latents must have the same shape")
    if detach_anchor:
        anchor_latents = anchor_latents.detach()
    if float(weight) <= 0.0:
        return student_latents.new_zeros(())
    if mode == "cosine":
        student_f = student_latents.flatten(start_dim=1).float()
        anchor_f = anchor_latents.flatten(start_dim=1).float()
        loss = 1.0 - F.cosine_similarity(student_f, anchor_f, dim=-1).mean()
    elif mode == "l2":
        loss = F.mse_loss(student_latents, anchor_latents)
    else:
        raise ValueError("diversity anchor mode must be l2 or cosine")
    return loss * float(weight)


def fake_critic_margin_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    negative_pred: torch.Tensor,
    *,
    margin: float = 0.05,
    weight: float = 1.0,
    detach_teacher: bool = True,
) -> torch.Tensor:
    if student_pred.shape != teacher_pred.shape or student_pred.shape != negative_pred.shape:
        raise ValueError("student, teacher, and negative predictions must have the same shape")
    if float(weight) <= 0.0:
        return student_pred.new_zeros(())
    if detach_teacher:
        teacher_pred = teacher_pred.detach()
        negative_pred = negative_pred.detach()
    student_distance = (student_pred - teacher_pred).flatten(start_dim=1).square().mean(dim=1)
    negative_distance = (negative_pred - teacher_pred).flatten(start_dim=1).square().mean(dim=1)
    return F.relu(float(margin) + student_distance - negative_distance).mean() * float(weight)


def compute_dp_dmd_turbo_loss(
    student_pred: torch.Tensor,
    teacher_cond_pred: torch.Tensor,
    *,
    config: DpDmdTurboConfig | Mapping[str, Any] | None = None,
    teacher_uncond_pred: torch.Tensor | None = None,
    student_latents: torch.Tensor | None = None,
    anchor_latents: torch.Tensor | None = None,
    negative_pred: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cfg = _coerce_config(config)
    cfg.validate()
    consistency = distillation_consistency_loss(
        student_pred,
        teacher_cond_pred,
        teacher_uncond_pred=teacher_uncond_pred,
        guidance_scale=cfg.guidance_scale,
    )
    diversity = student_pred.new_zeros(())
    if student_latents is not None and anchor_latents is not None:
        diversity = diversity_anchor_loss(
            student_latents,
            anchor_latents,
            weight=cfg.diversity_anchor_weight,
        )
    critic = student_pred.new_zeros(())
    teacher_target = teacher_cond_pred
    if teacher_uncond_pred is not None:
        teacher_target = bake_cfg_prediction(teacher_cond_pred, teacher_uncond_pred, cfg.guidance_scale)
    if negative_pred is not None:
        critic = fake_critic_margin_loss(
            student_pred,
            teacher_target,
            negative_pred,
            margin=cfg.fake_critic_margin,
            weight=cfg.fake_critic_weight,
        )
    total = consistency + diversity + critic
    return {
        "total": total,
        "consistency": consistency,
        "diversity_anchor": diversity,
        "fake_critic": critic,
    }


def build_dp_dmd_loop_plan(config: DpDmdTurboConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _coerce_config(config)
    cfg.validate()
    teacher_indices = _teacher_step_indices(cfg.student_steps, cfg.teacher_steps)
    return {
        "schema_version": 1,
        "plan": "dp_dmd_turbo_loop_plan_v0",
        "model_family": cfg.model_family,
        "objective": cfg.objective,
        "student_steps": cfg.student_steps,
        "teacher_steps": cfg.teacher_steps,
        "prediction_type": cfg.prediction_type,
        "guidance_scale": cfg.guidance_scale,
        "cfg_baked_into_student": cfg.guidance_scale != 1.0,
        "teacher_step_indices": teacher_indices,
        "student_step_plan": [
            {"student_step": idx, "teacher_step": teacher_step}
            for idx, teacher_step in enumerate(teacher_indices)
        ],
        "requires_teacher_sampler_contract": True,
        "requires_student_adapter_save_metadata": True,
        "request_fields_emitted": False,
        "trainer_wiring_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
    }


def build_dp_dmd_metadata(config: DpDmdTurboConfig | Mapping[str, Any] | None = None) -> dict[str, str]:
    cfg = _coerce_config(config)
    cfg.validate()
    return {
        "ss_feature_type": "dp_dmd_turbo",
        "ss_dp_dmd_version": str(cfg.metadata_version),
        "ss_dp_dmd_model_family": cfg.model_family,
        "ss_dp_dmd_objective": cfg.objective,
        "ss_dp_dmd_student_steps": str(cfg.student_steps),
        "ss_dp_dmd_teacher_steps": str(cfg.teacher_steps),
        "ss_dp_dmd_guidance_scale": _fmt_float(cfg.guidance_scale),
        "ss_dp_dmd_prediction_type": cfg.prediction_type,
        "ss_dp_dmd_diversity_anchor_weight": _fmt_float(cfg.diversity_anchor_weight),
        "ss_dp_dmd_fake_critic_weight": _fmt_float(cfg.fake_critic_weight),
        "ss_training_path_enabled": "false",
        "ss_default_behavior_changed": "false",
    }


def build_dp_dmd_scorecard(
    *,
    config: DpDmdTurboConfig | Mapping[str, Any] | None = None,
    loop_plan_ok: bool = False,
    cfg_target_ok: bool = False,
    consistency_loss_ok: bool = False,
    diversity_anchor_ok: bool = False,
    fake_critic_ok: bool = False,
    metadata_roundtrip_ok: bool = False,
    route_separation_ok: bool = False,
) -> dict[str, Any]:
    cfg = _coerce_config(config)
    blockers: list[str] = []
    try:
        cfg.validate()
    except ValueError as exc:
        blockers.append(f"invalid_config:{exc}")
    checks = {
        "loop_plan_missing": loop_plan_ok,
        "cfg_target_missing": cfg_target_ok,
        "consistency_loss_missing": consistency_loss_ok,
        "diversity_anchor_missing": diversity_anchor_ok,
        "fake_critic_missing": fake_critic_ok,
        "metadata_roundtrip_missing": metadata_roundtrip_ok,
        "route_separation_missing": route_separation_ok,
    }
    blockers.extend(reason for reason, passed in checks.items() if not passed)
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dp_dmd_turbo_primitive_v0",
        "ok": ready,
        "primitive_ready": ready,
        "model_family": cfg.model_family,
        "objective": cfg.objective,
        "student_steps": cfg.student_steps,
        "teacher_steps": cfg.teacher_steps,
        "request_fields_emitted": False,
        "trainer_wiring_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add DP-DMD trainer preflight and sampler contract before request wiring"
            if ready
            else "complete DP-DMD loop, target, loss, metadata, and route-separation proofs"
        ),
    }


def _coerce_config(config: DpDmdTurboConfig | Mapping[str, Any] | None) -> DpDmdTurboConfig:
    if isinstance(config, DpDmdTurboConfig):
        return config
    values = dict(config or {})
    return DpDmdTurboConfig(
        model_family=str(values.get("model_family", DpDmdTurboConfig.model_family)).strip().lower(),
        objective=str(values.get("objective", DpDmdTurboConfig.objective)).strip().lower(),
        student_steps=int(values.get("student_steps", DpDmdTurboConfig.student_steps)),
        teacher_steps=int(values.get("teacher_steps", DpDmdTurboConfig.teacher_steps)),
        guidance_scale=float(values.get("guidance_scale", DpDmdTurboConfig.guidance_scale)),
        prediction_type=str(values.get("prediction_type", DpDmdTurboConfig.prediction_type)).strip().lower(),
        diversity_anchor_weight=float(
            values.get("diversity_anchor_weight", DpDmdTurboConfig.diversity_anchor_weight)
        ),
        fake_critic_weight=float(values.get("fake_critic_weight", DpDmdTurboConfig.fake_critic_weight)),
        fake_critic_margin=float(values.get("fake_critic_margin", DpDmdTurboConfig.fake_critic_margin)),
        metadata_version=int(values.get("metadata_version", DpDmdTurboConfig.metadata_version)),
        training_path_enabled=_boolish(values.get("training_path_enabled", False)),
    )


def _teacher_step_indices(student_steps: int, teacher_steps: int) -> list[int]:
    if student_steps == 1:
        return [max(teacher_steps - 1, 0)]
    span = max(teacher_steps - 1, 1)
    return [int(round(idx * span / float(student_steps - 1))) for idx in range(student_steps)]


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _fmt_float(value: float) -> str:
    return ("%0.6f" % float(value)).rstrip("0").rstrip(".")


__all__ = [
    "DpDmdTurboConfig",
    "bake_cfg_prediction",
    "build_dp_dmd_loop_plan",
    "build_dp_dmd_metadata",
    "build_dp_dmd_scorecard",
    "compute_dp_dmd_turbo_loss",
    "distillation_consistency_loss",
    "diversity_anchor_loss",
    "fake_critic_margin_loss",
]
