"""Prepared subprocess job specs for Lulynx LAB routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.services.lab_runner_config import (
    build_artifact_report_runner_config,
    build_artifact_validation_runner_config,
)
from backend.core.services.lab_subprocess_runner import LabSubprocessJobSpec


def _runner_path(backend_root: Path, filename: str) -> Path:
    return backend_root / "core" / "tools" / "lulynx_lab" / filename


def build_lab_distiller_job_spec(
    *,
    runner_config: dict[str, Any],
    runtime_id: str,
    backend_root: Path,
    run_id: str,
    output_path: Path,
) -> LabSubprocessJobSpec:
    """Build a subprocess job spec for the LAB Distiller runner."""

    return LabSubprocessJobSpec(
        name="LAB Distiller",
        config=runner_config,
        runtime_id=runtime_id,
        runner_path=_runner_path(backend_root, "run_distiller.py"),
        config_filename="lab_distiller_config.json",
        metadata={
            "schema_id": "lab-distiller",
            "run_id": run_id,
            "output_path": str(output_path),
            "dry_run": bool(runner_config.get("dry_run", True)),
        },
    )


def build_turbo_lora_job_spec(
    *,
    runner_config: dict[str, Any],
    runtime_id: str,
    backend_root: Path,
    run_id: str,
    output_path: Path,
) -> LabSubprocessJobSpec:
    """Build a subprocess job spec for the SDXL Turbo/LCM LoRA runner."""

    return LabSubprocessJobSpec(
        name="SDXL Turbo / LCM LoRA",
        config=runner_config,
        runtime_id=runtime_id,
        runner_path=_runner_path(backend_root, "run_turbo_lora.py"),
        config_filename="turbo_lora_config.json",
        metadata={
            "schema_id": "sdxl-turbo-lora",
            "run_id": run_id,
            "output_path": str(output_path),
            "dry_run": bool(runner_config.get("dry_run", True)),
            "distill_method": str(runner_config.get("distill_method") or ""),
            "real_objective": str(runner_config.get("real_objective") or ""),
        },
    )


def build_dit_few_step_lora_job_spec(
    *,
    runner_config: dict[str, Any],
    runtime_id: str,
    backend_root: Path,
    run_id: str,
    output_path: Path,
) -> LabSubprocessJobSpec:
    """Build a subprocess job spec for the Anima/Newbie few-step LoRA runner."""

    family = str(runner_config.get("model_family") or "anima")
    schema_id = str(runner_config.get("schema_id") or f"{family}-few-step-lora")
    return LabSubprocessJobSpec(
        name=f"{family.title()} few-step LoRA",
        config=runner_config,
        runtime_id=runtime_id,
        runner_path=_runner_path(backend_root, "run_dit_few_step_lora.py"),
        config_filename="dit_few_step_lora_config.json",
        metadata={
            "schema_id": schema_id,
            "run_id": run_id,
            "output_path": str(output_path),
            "dry_run": True,
            "model_family": family,
        },
    )


def build_artifact_validation_job_spec(
    *,
    runtime_id: str,
    backend_root: Path,
    output_path: Path,
) -> LabSubprocessJobSpec:
    """Build a subprocess job spec for Turbo/LCM LoRA artifact validation."""

    return LabSubprocessJobSpec(
        name="Turbo LoRA output validation",
        config=build_artifact_validation_runner_config(output_path=output_path),
        runtime_id=runtime_id,
        runner_path=_runner_path(backend_root, "validate_turbo_lora_output.py"),
        config_filename="turbo_lora_validate.json",
        metadata={"schema_id": "sdxl-turbo-lora", "output_path": str(output_path)},
        runner_args=[str(output_path), "--write-sidecar"],
    )


def build_artifact_report_job_spec(
    *,
    runtime_id: str,
    backend_root: Path,
    output_path: Path,
    samples_path: Path | None,
) -> LabSubprocessJobSpec:
    """Build a subprocess job spec for Turbo/LCM LoRA sample reports."""

    runner_args = [
        str(output_path),
        *(["--samples-dir", str(samples_path)] if samples_path else []),
        "--write-sidecar",
    ]
    return LabSubprocessJobSpec(
        name="Turbo LoRA sample report",
        config=build_artifact_report_runner_config(output_path=output_path, samples_path=samples_path),
        runtime_id=runtime_id,
        runner_path=_runner_path(backend_root, "report_turbo_lora_samples.py"),
        config_filename="turbo_lora_sample_report.json",
        metadata={
            "schema_id": "sdxl-turbo-lora",
            "output_path": str(output_path),
            "samples_dir": str(samples_path) if samples_path else "",
        },
        runner_args=runner_args,
    )


__all__ = [
    "build_artifact_report_job_spec",
    "build_artifact_validation_job_spec",
    "build_dit_few_step_lora_job_spec",
    "build_lab_distiller_job_spec",
    "build_turbo_lora_job_spec",
]
