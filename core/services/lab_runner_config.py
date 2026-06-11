"""Runner config builders for Lulynx LAB request-native routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.contracts import DitFewStepLoraRequest, LabDistillerRequest, TurboLoraRequest


def _int_value(config: dict[str, Any], key: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}")
    return value


def _float_value(
    config: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}")
    return value


def build_turbo_lora_runner_config(
    *,
    config: dict[str, Any],
    request: TurboLoraRequest,
    run_id: str,
    base_model_path: Path,
    train_data_dir: Path,
    teacher_lora_path: Path | None,
    vae_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    """Build the existing Turbo/LCM LoRA runner config from a normalized request."""

    return {
        **config,
        "schema_id": "sdxl-turbo-lora",
        "run_id": run_id,
        "dry_run": request.dry_run,
        "confirm_real_run": request.confirm_real_run,
        "distill_method": request.distill_method,
        "real_objective": request.real_objective,
        "base_model_path": str(base_model_path),
        "train_data_dir": str(train_data_dir),
        "teacher_lora_path": str(teacher_lora_path) if teacher_lora_path else "",
        "teacher_lora_scope": request.teacher_lora_scope,
        "vae_path": str(vae_path) if vae_path else "",
        "output_path": str(output_path),
        "teacher_scheduler": str(config.get("teacher_scheduler") or "dpmpp_2m_karras"),
        "teacher_steps": _int_value(config, "teacher_steps", 30, minimum=1),
        "student_scheduler": str(config.get("student_scheduler") or "lcm"),
        "student_steps": _int_value(config, "student_steps", 4, minimum=1),
        "guidance_scale": _float_value(config, "guidance_scale", 1.5, minimum=0.0),
        "lcm_target_stride": _int_value(config, "lcm_target_stride", 80, minimum=1),
        "timestep_sampling": str(config.get("timestep_sampling") or "lcm"),
        "seed": _int_value(config, "seed", 42, minimum=0),
        "distillation_loss_weight": _float_value(config, "distillation_loss_weight", 1.0, minimum=0.0),
        "learning_rate": request.learning_rate,
        "max_train_steps": request.max_train_steps,
        "batch_size": request.batch_size,
        "mixed_precision": request.mixed_precision,
        "resolution": _int_value(config, "resolution", 512, minimum=256, maximum=512),
        "network_dim": request.network_dim,
        "network_alpha": request.network_alpha,
        "network_dropout": _float_value(config, "network_dropout", 0.0, minimum=0.0, maximum=1.0),
        "target_modules": str(config.get("target_modules") or "unet_attention"),
        "metadata_note": str(config.get("metadata_note") or ""),
    }


def build_lab_distiller_runner_config(
    *,
    config: dict[str, Any],
    request: LabDistillerRequest,
    run_id: str,
    unet_path: Path,
    lora_path: Path,
    projector_path: Path | None,
    teacher_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    """Build the existing LAB Distiller runner config from a normalized request."""

    return {
        **config,
        "schema_id": request.schema_id,
        "run_id": run_id,
        "unet_path": str(unet_path),
        "lora_path": str(lora_path),
        "llm_path": request.llm_path,
        "projector_path": str(projector_path) if projector_path else "",
        "teacher_path": str(teacher_path) if teacher_path else "",
        "output_path": str(output_path),
        "steps": request.steps,
        "batch_size": request.batch_size,
        "device": request.device,
        "dtype": request.dtype,
        "learning_rate": request.learning_rate,
        "dry_run": request.dry_run,
    }


def build_dit_few_step_lora_runner_config(
    *,
    config: dict[str, Any],
    request: DitFewStepLoraRequest,
    run_id: str,
    output_path: Path,
) -> dict[str, Any]:
    """Build the existing Anima/Newbie few-step LoRA runner config."""

    return {
        **config,
        "schema_id": request.schema_id,
        "run_id": run_id,
        "dry_run": True,
        "model_family": request.model_family,
        "output_path": str(output_path),
        "student_steps": request.student_steps,
        "teacher_steps": request.teacher_steps,
        "guidance_scale": request.guidance_scale,
        "network_dim": request.network_dim,
        "network_alpha": request.network_alpha,
        "seed": request.seed,
        "distill_method": request.distill_method,
        "few_step_objective": request.few_step_objective,
        "sigma_schedule": request.sigma_schedule,
    }


def build_artifact_validation_runner_config(*, output_path: Path) -> dict[str, Any]:
    """Build the existing artifact validation runner config."""

    return {"output_path": str(output_path)}


def build_artifact_report_runner_config(*, output_path: Path, samples_path: Path | None) -> dict[str, Any]:
    """Build the existing artifact sample-report runner config."""

    return {"output_path": str(output_path), "samples_dir": str(samples_path) if samples_path else ""}
