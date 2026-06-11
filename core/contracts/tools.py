"""Tool and LAB request contracts."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import BaseRequest


class ToolRequest(BaseRequest):
    """Generic request for bounded utility/tool actions."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "tool.generic"
    tool_id: str = ""
    action: str = "run"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)
    output_dir: str = ""
    dry_run: bool = False
    permissions: List[str] = Field(default_factory=list)

    @field_validator("tool_id")
    @classmethod
    def _tool_id_required(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("tool_id is required")
        return value


class LabRequest(ToolRequest):
    """Base request for Lulynx LAB experimental runners."""

    schema_id: str = "lab.generic"
    lab_id: str = ""
    runtime_id: str = ""
    output_path: str = ""
    resources: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("lab_id")
    @classmethod
    def _lab_id_required(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("lab_id is required")
        return value


class TurboLoraRequest(LabRequest):
    """Request contract for SDXL Turbo/LCM LoRA experimental runner."""

    schema_id: str = "sdxl-turbo-lora"
    tool_id: str = "lulynx-lab"
    lab_id: str = "sdxl-turbo-lora"
    base_model_path: str = ""
    train_data_dir: str = ""
    teacher_lora_path: str = ""
    teacher_lora_scope: str = "unet_only"
    vae_path: str = ""
    output_path: str = "./output/turbo_lora/sdxl_lcm_lora.safetensors"
    distill_method: str = "lcm_lora"
    real_objective: str = "lcm_consistency_probe"
    dry_run: bool = True
    confirm_real_run: bool = False
    max_train_steps: int = 1000
    batch_size: int = 1
    mixed_precision: str = "bf16"
    resolution: int = 512
    network_dim: int = 16
    network_alpha: int = 16
    learning_rate: float = 1e-4
    metadata_note: str = ""

    @field_validator("teacher_lora_scope")
    @classmethod
    def _teacher_lora_scope_supported(cls, value: str) -> str:
        value = str(value or "").strip().lower()
        aliases = {"full": "unet_and_text_encoder_experimental", "all": "unet_and_text_encoder_experimental"}
        value = aliases.get(value, value)
        if value not in {"unet_only", "unet_and_text_encoder_experimental"}:
            raise ValueError("unsupported teacher_lora_scope")
        return value

    @field_validator("mixed_precision")
    @classmethod
    def _mixed_precision_supported(cls, value: str) -> str:
        value = str(value or "").strip().lower()
        if value not in {"bf16", "fp16", "fp32"}:
            raise ValueError("mixed_precision must be bf16, fp16, or fp32")
        return value

    @field_validator("distill_method")
    @classmethod
    def _distill_method_supported(cls, value: str) -> str:
        value = str(value or "").strip().lower()
        if value not in {"lcm_lora", "turbo_lora"}:
            raise ValueError("distill_method must be lcm_lora or turbo_lora")
        return value

    @field_validator("real_objective")
    @classmethod
    def _real_objective_supported(cls, value: str) -> str:
        value = str(value or "").strip().lower()
        if value not in {"epsilon_lora_probe", "lcm_consistency_probe"}:
            raise ValueError("unsupported real_objective")
        return value

    @field_validator("max_train_steps", "batch_size", "resolution", "network_dim", "network_alpha")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        value = int(value)
        if value < 1:
            raise ValueError("value must be >= 1")
        return value

    def real_run_allowed(self) -> bool:
        return bool(not self.dry_run and self.confirm_real_run and self.max_train_steps <= 4 and self.batch_size == 1)


class LabDistillerRequest(LabRequest):
    """Request contract for the experimental LAB sidecar distiller."""

    schema_id: str = "lab-distiller"
    tool_id: str = "lulynx-lab"
    lab_id: str = "lab-distiller"
    unet_path: str = ""
    lora_path: str = ""
    llm_path: str = "Qwen/Qwen2.5-0.5B"
    projector_path: str = ""
    teacher_path: str = ""
    output_path: str = "./output/lab_distiller/sidecar.safetensors"
    steps: int = 1000
    batch_size: int = 4
    device: str = "cuda"
    dtype: str = "bf16"
    learning_rate: float = 1e-5
    dry_run: bool = True

    @field_validator("dtype")
    @classmethod
    def _dtype_supported(cls, value: str) -> str:
        value = str(value or "").strip().lower()
        if value not in {"auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
            raise ValueError("unsupported dtype")
        return value

    @field_validator("steps", "batch_size")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        value = int(value)
        if value < 1:
            raise ValueError("value must be >= 1")
        return value


class DitFewStepLoraRequest(LabRequest):
    """Request contract for Anima/Newbie few-step acceleration LoRA probes."""

    schema_id: str = "anima-few-step-lora"
    tool_id: str = "lulynx-lab"
    lab_id: str = "dit-few-step-lora"
    model_train_type: str = ""
    model_family: str = "anima"
    output_path: str = "./output/dit_few_step_lora/anima_few_step_lora.safetensors"
    dry_run: bool = True
    student_steps: int = 4
    teacher_steps: int = 28
    guidance_scale: float = 1.0
    network_dim: int = 16
    network_alpha: int = 16
    seed: int = 42
    distill_method: str = "family_flow_consistency"
    few_step_objective: str = "contract_probe"
    sigma_schedule: str = "family_default"

    @field_validator("model_family")
    @classmethod
    def _family_supported(cls, value: str) -> str:
        value = str(value or "").strip().lower()
        if value not in {"anima", "newbie"}:
            raise ValueError("model_family must be anima or newbie")
        return value

    @field_validator("student_steps", "teacher_steps", "network_dim", "network_alpha")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        value = int(value)
        if value < 1:
            raise ValueError("value must be >= 1")
        return value


class ArtifactValidationRequest(ToolRequest):
    """Request contract for validating a generated artifact."""

    schema_id: str = "artifact.validation"
    tool_id: str = "artifact-validator"
    artifact_path: str = ""
    write_sidecar: bool = True

    @field_validator("artifact_path")
    @classmethod
    def _artifact_path_required(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("artifact_path is required")
        return value

    @model_validator(mode="after")
    def _artifact_path_default_required(self) -> "ArtifactValidationRequest":
        if not str(self.artifact_path or "").strip():
            raise ValueError("artifact_path is required")
        return self


class ArtifactReportRequest(ArtifactValidationRequest):
    """Request contract for sample/quality reports attached to an artifact."""

    schema_id: str = "artifact.report"
    tool_id: str = "artifact-reporter"
    samples_dir: str = ""

