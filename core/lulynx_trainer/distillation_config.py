"""Unified distillation framework configuration.

Supports multiple distillation modes:
- Mode D: Consistency Distillation (LCM-style, recommended)
- Mode C: Guided Dataset (pre-generated teacher outputs)
- Mode A: Online Adversarial (teacher + student + critic)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class DistillationMode(enum.Enum):
    """Distillation training modes."""

    CONSISTENCY = "consistency"           # Mode D - LCM-style (recommended)
    GUIDED_DATASET = "guided_dataset"     # Mode C - Pre-generated data
    ONLINE_ADVERSARIAL = "online_adv"     # Mode A - Teacher+Student+Critic
    HYBRID = "hybrid"                     # Mode D+C - Consistency with dataset


@dataclass(frozen=True)
class DistillationConfig:
    """Unified distillation configuration."""

    # Global settings
    enabled: bool = False
    mode: str = "consistency"

    # Step configuration
    teacher_steps: int = 50
    student_steps: int = 4
    weight: float = 1.0

    # Consistency mode (Mode D)
    consistency_use_perceptual: bool = False
    consistency_perceptual_weight: float = 0.1
    consistency_teacher_guidance: float = 7.5
    consistency_student_guidance: float = 7.5

    # Guided dataset mode (Mode C)
    guided_dataset_path: str = ""
    guided_dataset_pregenerate: bool = False
    guided_dataset_num_samples: int = 10000

    # Online adversarial mode (Mode A)
    online_adv_critic_enabled: bool = True
    online_adv_critic_weight: float = 0.1
    online_adv_critic_arch: str = "lightweight"
    online_adv_feature_matching: bool = False
    online_adv_feature_matching_weight: float = 0.0

    # Hybrid mode (Mode D+C)
    hybrid_cache_ratio: float = 0.5

    def normalized(self) -> "DistillationConfig":
        """Normalize and validate configuration."""
        mode = str(self.mode or "consistency").strip().lower().replace("-", "_")
        if mode not in {"consistency", "guided_dataset", "online_adv", "hybrid"}:
            mode = "consistency"

        teacher_steps = max(int(self.teacher_steps or 50), 1)
        student_steps = max(int(self.student_steps or 4), 1)

        if student_steps >= teacher_steps:
            raise ValueError(
                f"student_steps ({student_steps}) must be < teacher_steps ({teacher_steps})"
            )

        weight = max(float(self.weight or 1.0), 0.0)

        return DistillationConfig(
            enabled=bool(self.enabled),
            mode=mode,
            teacher_steps=teacher_steps,
            student_steps=student_steps,
            weight=weight,
            consistency_use_perceptual=bool(self.consistency_use_perceptual),
            consistency_perceptual_weight=max(float(self.consistency_perceptual_weight or 0.1), 0.0),
            consistency_teacher_guidance=max(float(self.consistency_teacher_guidance or 7.5), 0.0),
            consistency_student_guidance=max(float(self.consistency_student_guidance or 7.5), 0.0),
            guided_dataset_path=str(self.guided_dataset_path or "").strip(),
            guided_dataset_pregenerate=bool(self.guided_dataset_pregenerate),
            guided_dataset_num_samples=max(int(self.guided_dataset_num_samples or 10000), 1),
            online_adv_critic_enabled=bool(self.online_adv_critic_enabled),
            online_adv_critic_weight=max(float(self.online_adv_critic_weight or 0.1), 0.0),
            online_adv_critic_arch=str(self.online_adv_critic_arch or "lightweight").strip().lower(),
            online_adv_feature_matching=bool(self.online_adv_feature_matching),
            online_adv_feature_matching_weight=max(float(self.online_adv_feature_matching_weight or 0.0), 0.0),
            hybrid_cache_ratio=min(max(float(self.hybrid_cache_ratio or 0.5), 0.0), 1.0),
        )

    def get_mode_enum(self) -> DistillationMode:
        """Get distillation mode as enum."""
        mode_map = {
            "consistency": DistillationMode.CONSISTENCY,
            "guided_dataset": DistillationMode.GUIDED_DATASET,
            "online_adv": DistillationMode.ONLINE_ADVERSARIAL,
            "hybrid": DistillationMode.HYBRID,
        }
        return mode_map.get(self.mode, DistillationMode.CONSISTENCY)


def create_distillation_config_from_training_config(config) -> Optional[DistillationConfig]:
    """Create distillation config from training config.

    Parameters
    ----------
    config : object
        Training configuration object.

    Returns
    -------
    DistillationConfig or None
        Distillation config if enabled, None otherwise.
    """
    if not getattr(config, "distillation_enabled", False):
        return None

    distill_config = DistillationConfig(
        enabled=True,
        mode=getattr(config, "distillation_mode", "consistency"),
        teacher_steps=getattr(config, "distillation_teacher_steps", 50),
        student_steps=getattr(config, "distillation_student_steps", 4),
        weight=getattr(config, "distillation_weight", 1.0),
        consistency_use_perceptual=getattr(config, "distillation_consistency_use_perceptual", False),
        consistency_perceptual_weight=getattr(config, "distillation_consistency_perceptual_weight", 0.1),
        consistency_teacher_guidance=getattr(config, "distillation_consistency_teacher_guidance", 7.5),
        consistency_student_guidance=getattr(config, "distillation_consistency_student_guidance", 7.5),
        guided_dataset_path=getattr(config, "distillation_guided_dataset_path", ""),
        guided_dataset_pregenerate=getattr(config, "distillation_guided_dataset_pregenerate", False),
        guided_dataset_num_samples=getattr(config, "distillation_guided_dataset_num_samples", 10000),
        online_adv_critic_enabled=getattr(config, "distillation_online_adv_critic_enabled", True),
        online_adv_critic_weight=getattr(config, "distillation_online_adv_critic_weight", 0.1),
        online_adv_critic_arch=getattr(config, "distillation_online_adv_critic_arch", "lightweight"),
        online_adv_feature_matching=getattr(config, "distillation_online_adv_feature_matching", False),
        online_adv_feature_matching_weight=getattr(config, "distillation_online_adv_feature_matching_weight", 0.0),
        hybrid_cache_ratio=getattr(config, "distillation_hybrid_cache_ratio", 0.5),
    )

    return distill_config.normalized()


def get_distillation_config_help() -> str:
    """Get configuration help text.

    Returns
    -------
    str
        Help text with examples.
    """
    return """
Distillation Configuration Guide
=================================

Lulynx Trainer supports multiple distillation modes for training
fast-sampling LoRA models (e.g., 4-step Turbo LoRA).

## Recommended: Mode D - Consistency Distillation

This is the LCM-style approach, most suitable for training Turbo LoRA.

    [distillation]
    enabled = true
    mode = "consistency"
    teacher_steps = 50
    student_steps = 4
    weight = 1.0

    consistency_use_perceptual = false
    consistency_perceptual_weight = 0.1

## Mode C - Guided Dataset

Pre-generate teacher outputs, then train student offline.

    [distillation]
    enabled = true
    mode = "guided_dataset"
    teacher_steps = 50
    student_steps = 4

    guided_dataset_path = "./distilled_data"
    guided_dataset_pregenerate = false
    guided_dataset_num_samples = 10000

## Mode D+C - Hybrid

Best of both: online consistency + cached outputs.

    [distillation]
    enabled = true
    mode = "hybrid"
    teacher_steps = 50
    student_steps = 4

    hybrid_cache_ratio = 0.5  # 50% from cache

## Mode A - Online Adversarial (Experimental)

Full GAN-style distillation with critic.

    [distillation]
    enabled = true
    mode = "online_adv"
    teacher_steps = 50
    student_steps = 4

    online_adv_critic_enabled = true
    online_adv_critic_weight = 0.1
    online_adv_critic_arch = "lightweight"

## Parameters

- teacher_steps: Steps for teacher inference (e.g., 50)
- student_steps: Target steps for student (e.g., 4)
- weight: Distillation loss weight
- mode: "consistency" (recommended) | "guided_dataset" | "online_adv" | "hybrid"

See documentation for full parameter reference.
"""
