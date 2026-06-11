"""Default-off SPD multi-resolution inference planning primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpdInferenceConfig:
    model_family: str = "anima"
    scale_factors: tuple[float, ...] = (0.5, 1.0)
    steps_per_level: tuple[int, ...] = (4, 8)
    resize_mode: str = "bilinear"
    trajectory_lora_enabled: bool = False
    metadata_version: int = 1
    runtime_activation_enabled: bool = False

    def validate(self) -> None:
        if self.model_family not in {"anima", "newbie"}:
            raise ValueError("model_family must be anima or newbie")
        if not self.scale_factors:
            raise ValueError("scale_factors must not be empty")
        if len(self.scale_factors) != len(self.steps_per_level):
            raise ValueError("scale_factors and steps_per_level must have equal length")
        last = 0.0
        for scale in self.scale_factors:
            if float(scale) <= 0.0 or float(scale) > 1.0:
                raise ValueError("scale_factors must be inside (0, 1]")
            if float(scale) < last:
                raise ValueError("scale_factors must be non-decreasing")
            last = float(scale)
        if abs(float(self.scale_factors[-1]) - 1.0) > 1e-6:
            raise ValueError("final SPD scale must be 1.0")
        if any(int(steps) < 1 for steps in self.steps_per_level):
            raise ValueError("steps_per_level values must be >= 1")
        if self.resize_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("resize_mode must be nearest, bilinear, or bicubic")
        if self.runtime_activation_enabled:
            raise ValueError("SPD runtime activation is not enabled by this primitive")


def build_spd_inference_plan(
    latent_shape: Sequence[int],
    config: SpdInferenceConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _coerce_config(config)
    cfg.validate()
    if len(tuple(latent_shape)) != 4:
        raise ValueError("latent_shape must be [batch, channels, height, width]")
    batch, channels, height, width = (int(item) for item in latent_shape)
    if min(batch, channels, height, width) < 1:
        raise ValueError("latent_shape dimensions must be >= 1")
    levels = []
    step_offset = 0
    for idx, (scale, steps) in enumerate(zip(cfg.scale_factors, cfg.steps_per_level)):
        latent_h = max(1, int(round(height * float(scale))))
        latent_w = max(1, int(round(width * float(scale))))
        levels.append(
            {
                "level": idx,
                "scale": float(scale),
                "steps": int(steps),
                "step_start": step_offset,
                "step_end_exclusive": step_offset + int(steps),
                "latent_shape": [batch, channels, latent_h, latent_w],
            }
        )
        step_offset += int(steps)
    return {
        "schema_version": 1,
        "plan": "spd_multiresolution_inference_plan_v0",
        "model_family": cfg.model_family,
        "resize_mode": cfg.resize_mode,
        "base_latent_shape": [batch, channels, height, width],
        "total_steps": step_offset,
        "level_count": len(levels),
        "levels": levels,
        "trajectory_lora_enabled": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
    }


def resize_latents_for_spd(
    latents: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    if latents.dim() != 4:
        raise ValueError("latents must be [batch, channels, height, width]")
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h < 1 or target_w < 1:
        raise ValueError("target height and width must be >= 1")
    if mode == "nearest":
        return F.interpolate(latents, size=(target_h, target_w), mode=mode)
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(latents, size=(target_h, target_w), mode=mode, align_corners=False)
    raise ValueError("resize mode must be nearest, bilinear, or bicubic")


def build_spd_trajectory_lora_decision(
    *,
    requested: bool,
    inference_route_ready: bool,
    checkpoint_metadata_ready: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    if requested and not inference_route_ready:
        blockers.append("spd_inference_route_missing")
    if requested and not checkpoint_metadata_ready:
        blockers.append("trajectory_lora_metadata_missing")
    allowed = bool(requested and not blockers)
    return {
        "schema_version": 1,
        "decision": "spd_trajectory_lora_boundary_v0",
        "requested": bool(requested),
        "allowed": allowed,
        "trajectory_lora_enabled": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add trajectory-LoRA save/load preflight after SPD inference route works"
            if allowed
            else "finish SPD inference route and checkpoint metadata before trajectory LoRA"
        ),
    }


def build_spd_metadata(config: SpdInferenceConfig | Mapping[str, Any] | None = None) -> dict[str, str]:
    cfg = _coerce_config(config)
    cfg.validate()
    return {
        "ss_feature_type": "spd_inference",
        "ss_spd_version": str(cfg.metadata_version),
        "ss_spd_model_family": cfg.model_family,
        "ss_spd_scale_factors": ",".join(_fmt_float(item) for item in cfg.scale_factors),
        "ss_spd_steps_per_level": ",".join(str(int(item)) for item in cfg.steps_per_level),
        "ss_spd_resize_mode": cfg.resize_mode,
        "ss_spd_trajectory_lora_enabled": "false",
        "ss_runtime_activation_enabled": "false",
        "ss_training_path_enabled": "false",
        "ss_default_behavior_changed": "false",
    }


def build_spd_scorecard(
    *,
    config: SpdInferenceConfig | Mapping[str, Any] | None = None,
    schedule_ok: bool = False,
    resize_ok: bool = False,
    no_op_final_scale_ok: bool = False,
    trajectory_lora_block_ok: bool = False,
    metadata_roundtrip_ok: bool = False,
) -> dict[str, Any]:
    cfg = _coerce_config(config)
    blockers: list[str] = []
    try:
        cfg.validate()
    except ValueError as exc:
        blockers.append(f"invalid_config:{exc}")
    checks = {
        "schedule_missing": schedule_ok,
        "resize_missing": resize_ok,
        "final_scale_noop_missing": no_op_final_scale_ok,
        "trajectory_lora_block_missing": trajectory_lora_block_ok,
        "metadata_roundtrip_missing": metadata_roundtrip_ok,
    }
    blockers.extend(reason for reason, passed in checks.items() if not passed)
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "spd_inference_primitive_v0",
        "ok": ready,
        "primitive_ready": ready,
        "model_family": cfg.model_family,
        "scale_factors": list(cfg.scale_factors),
        "steps_per_level": list(cfg.steps_per_level),
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add SPD sampler preflight before request/runtime wiring"
            if ready
            else "complete SPD schedule, resize, trajectory boundary, and metadata proofs"
        ),
    }


def _coerce_config(config: SpdInferenceConfig | Mapping[str, Any] | None) -> SpdInferenceConfig:
    if isinstance(config, SpdInferenceConfig):
        return config
    values = dict(config or {})
    return SpdInferenceConfig(
        model_family=str(values.get("model_family", SpdInferenceConfig.model_family)).strip().lower(),
        scale_factors=_float_tuple(values.get("scale_factors", SpdInferenceConfig.scale_factors)),
        steps_per_level=_int_tuple(values.get("steps_per_level", SpdInferenceConfig.steps_per_level)),
        resize_mode=str(values.get("resize_mode", SpdInferenceConfig.resize_mode)).strip().lower(),
        trajectory_lora_enabled=_boolish(values.get("trajectory_lora_enabled", False)),
        metadata_version=int(values.get("metadata_version", SpdInferenceConfig.metadata_version)),
        runtime_activation_enabled=_boolish(values.get("runtime_activation_enabled", False)),
    )


def _float_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return tuple(float(item) for item in value)


def _int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _fmt_float(value: float) -> str:
    return ("%0.6f" % float(value)).rstrip("0").rstrip(".")


__all__ = [
    "SpdInferenceConfig",
    "build_spd_inference_plan",
    "build_spd_metadata",
    "build_spd_scorecard",
    "build_spd_trajectory_lora_decision",
    "resize_latents_for_spd",
]
