"""Base-model TensorRT static runtime contracts.

These contracts describe the LAB/runtime handoff for static transformer
TensorRT engines. They are not generation requests and they do not enable the
product generation or training paths by themselves.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import BaseRequest, PlatformIssue, RunResult, RunStatus


BASE_MODEL_TENSORRT_RUNTIME_ABI = "base_model_static_tensorrt_runtime_v1"
BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID = "base-model.tensorrt.runtime"
BASE_MODEL_TENSORRT_RUNTIME_RESULT_SCHEMA_ID = "base-model.tensorrt.runtime.result"


class BaseModelTensorRtRuntimeMode(str, Enum):
    GATE = "gate"
    SMOKE = "smoke"


class BaseModelTensorRtRuntimeRequest(BaseRequest):
    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID
    abi: str = BASE_MODEL_TENSORRT_RUNTIME_ABI
    family: str = "newbie"
    component: str = "transformer"
    mode: BaseModelTensorRtRuntimeMode = BaseModelTensorRtRuntimeMode.GATE
    engine_path: str = ""
    precision: str = "fp32"
    layer_indices: List[int] = Field(default_factory=lambda: list(range(36)))
    shape: Dict[str, int] = Field(default_factory=lambda: {
        "batch": 1,
        "latent_channels": 16,
        "latent_height": 64,
        "latent_width": 64,
        "tokens": 512,
        "hidden_dim": 2304,
        "pooled_dim": 1024,
        "patch_size": 2,
    })
    device: str = "cuda"
    dtype: str = "float32"
    dry_run: bool = True

    @field_validator("abi")
    @classmethod
    def _abi_supported(cls, value: str) -> str:
        value = str(value or "").strip()
        if value != BASE_MODEL_TENSORRT_RUNTIME_ABI:
            raise ValueError("unsupported base-model TensorRT runtime ABI")
        return value

    @field_validator("family", "component", "precision", "device", "dtype")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return str(value or "").strip().lower().replace("-", "_")

    @field_validator("layer_indices", mode="before")
    @classmethod
    def _normalize_layer_indices(cls, value: object) -> list[int]:
        if value is None or value == "":
            return list(range(36))
        if isinstance(value, str):
            items: list[int] = []
            for part in value.replace(";", ",").split(","):
                token = part.strip()
                if not token:
                    continue
                if "-" in token:
                    start, end = token.split("-", 1)
                    items.extend(range(int(start), int(end) + 1))
                else:
                    items.append(int(token))
            return list(dict.fromkeys(items))
        if isinstance(value, list):
            return list(dict.fromkeys(int(item) for item in value))
        raise ValueError("layer_indices must be a list or range string")

    @field_validator("shape", mode="before")
    @classmethod
    def _normalize_shape(cls, value: object) -> dict[str, int]:
        data = dict(value or {}) if isinstance(value, dict) else {}
        defaults = {
            "batch": 1,
            "latent_channels": 16,
            "latent_height": 64,
            "latent_width": 64,
            "tokens": 512,
            "hidden_dim": 2304,
            "pooled_dim": 1024,
            "patch_size": 2,
        }
        defaults.update({str(key): int(val) for key, val in data.items() if val is not None})
        return defaults

    @model_validator(mode="after")
    def _validate_boundary(self) -> "BaseModelTensorRtRuntimeRequest":
        if self.family != "newbie":
            raise ValueError("base-model TensorRT runtime gate is currently validated only for newbie")
        if self.component != "transformer":
            raise ValueError("base-model TensorRT runtime gate currently supports transformer only")
        if self.precision != "fp32":
            raise ValueError("Newbie TensorRT runtime gate currently accepts FP32 only")
        if not str(self.engine_path or "").strip():
            raise ValueError("engine_path is required")
        return self


class BaseModelTensorRtRuntimeResult(RunResult):
    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = BASE_MODEL_TENSORRT_RUNTIME_RESULT_SCHEMA_ID
    abi: str = BASE_MODEL_TENSORRT_RUNTIME_ABI
    runtime_loadable: bool = False
    lab_runtime_allowed: bool = False
    generation_path_enabled: bool = False
    training_path_enabled: bool = False

    @classmethod
    def from_gate_report(
        cls,
        report: Dict[str, Any],
        *,
        request: BaseModelTensorRtRuntimeRequest | None = None,
    ) -> "BaseModelTensorRtRuntimeResult":
        blockers = [str(item) for item in report.get("blockers") or [] if str(item)]
        warnings = [str(item) for item in report.get("warnings") or [] if str(item)]
        issues = [
            PlatformIssue(code=f"base_model_tensorrt.{item}", message=item, severity="error")
            for item in blockers
        ] + [
            PlatformIssue(code=f"base_model_tensorrt.{item}", message=item, severity="warning")
            for item in warnings
        ]
        ok = bool(report.get("runtime_loadable")) and not blockers
        return cls(
            request_id=request.request_id if request else "",
            status=RunStatus.SUCCEEDED if ok else RunStatus.SKIPPED,
            message="Base-model TensorRT static transformer runtime gate passed." if ok else "Base-model TensorRT runtime gate is blocked.",
            issues=issues,
            runtime_loadable=bool(report.get("runtime_loadable")),
            lab_runtime_allowed=bool(report.get("lab_runtime_allowed")),
            generation_path_enabled=False,
            training_path_enabled=False,
            data={
                "runtime_report": report,
                "generation_path_enabled": False,
                "training_path_enabled": False,
            },
        )


__all__ = [
    "BASE_MODEL_TENSORRT_RUNTIME_ABI",
    "BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID",
    "BASE_MODEL_TENSORRT_RUNTIME_RESULT_SCHEMA_ID",
    "BaseModelTensorRtRuntimeMode",
    "BaseModelTensorRtRuntimeRequest",
    "BaseModelTensorRtRuntimeResult",
]
