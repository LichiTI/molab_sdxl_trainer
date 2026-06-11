"""Image GGUF runtime loader ABI contracts.

These models describe the handoff boundary for future image GGUF loaders. They
are report-only today: no tensor payloads are read, no model modules are built,
and training/runtime dispatch stays disabled until a real loader satisfies this
contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import BaseRequest, PlatformIssue, RunResult, RunStatus


IMAGE_GGUF_RUNTIME_LOADER_ABI = "image_gguf_runtime_loader_v1"
IMAGE_GGUF_RUNTIME_PROVIDER = "lulynx_image_runtime_loader"
IMAGE_GGUF_RUNTIME_REQUEST_SCHEMA_ID = "image.gguf.runtime.load"
IMAGE_GGUF_RUNTIME_RESULT_SCHEMA_ID = "image.gguf.runtime.load.result"
IMAGE_GGUF_SUPPORTED_COMPONENTS = {
    "anima_dit",
    "clip",
    "newbie_dit",
    "sd15_unet",
    "sdxl_unet",
    "t5",
    "vae",
}


class ImageGGUFLoadMode(str, Enum):
    """Supported runtime-loader request modes."""

    DESCRIPTOR_ONLY = "descriptor_only"
    MODULE_BUILD = "module_build"
    QUALITY_PROBE = "quality_probe"


class ImageGGUFRuntimeLoadRequest(BaseRequest):
    """Canonical request shape for a future image GGUF runtime loader."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = IMAGE_GGUF_RUNTIME_REQUEST_SCHEMA_ID
    abi: str = IMAGE_GGUF_RUNTIME_LOADER_ABI
    gguf_path: str = ""
    sidecar_path: str = ""
    component: str = ""
    family: str = ""
    target_runtime: str = IMAGE_GGUF_RUNTIME_PROVIDER
    load_mode: ImageGGUFLoadMode = ImageGGUFLoadMode.DESCRIPTOR_ONLY
    device: str = "auto"
    dtype_policy: str = "descriptor"
    allow_experimental_tensor_types: bool = False
    required_runtime_features: List[str] = Field(default_factory=list)
    quality_gates: List[str] = Field(default_factory=list)
    dry_run: bool = True

    @field_validator("abi")
    @classmethod
    def _abi_supported(cls, value: str) -> str:
        value = str(value or "").strip()
        if value != IMAGE_GGUF_RUNTIME_LOADER_ABI:
            raise ValueError("unsupported image GGUF runtime loader ABI")
        return value

    @field_validator("component")
    @classmethod
    def _component_normalized(cls, value: str) -> str:
        return str(value or "").strip().lower()

    @field_validator("target_runtime")
    @classmethod
    def _target_runtime_required(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("target_runtime is required")
        return value

    @field_validator("required_runtime_features", "quality_gates", mode="before")
    @classmethod
    def _string_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("value must be a list of strings")
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))

    @model_validator(mode="after")
    def _validate_boundary(self) -> "ImageGGUFRuntimeLoadRequest":
        path = str(self.gguf_path or "").strip()
        if not path:
            raise ValueError("gguf_path is required")
        if not path.lower().endswith(".gguf"):
            raise ValueError("gguf_path must end with .gguf")
        if self.sidecar_path and not str(self.sidecar_path).lower().endswith(".json"):
            raise ValueError("sidecar_path must be a JSON sidecar when provided")
        if self.component not in IMAGE_GGUF_SUPPORTED_COMPONENTS:
            raise ValueError("unsupported image GGUF runtime component")
        return self


class ImageGGUFRuntimeLoadResult(RunResult):
    """Report-only result envelope for image GGUF runtime-loader attempts."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = IMAGE_GGUF_RUNTIME_RESULT_SCHEMA_ID
    abi: str = IMAGE_GGUF_RUNTIME_LOADER_ABI
    runtime_loadable: bool = False
    loader_implemented: bool = False
    loadability: str = "shape_only_reference"

    @classmethod
    def report_only(
        cls,
        *,
        request: ImageGGUFRuntimeLoadRequest | None = None,
        runtime_contract: Dict[str, Any] | None = None,
    ) -> "ImageGGUFRuntimeLoadResult":
        contract = dict(runtime_contract or {})
        blockers = [str(item) for item in contract.get("blockers") or [] if str(item)]
        issues = [
            PlatformIssue(
                code="image_gguf.runtime_loader_blocked",
                message=message,
                severity="warning",
            )
            for message in blockers
        ]
        return cls(
            request_id=request.request_id if request else "",
            status=RunStatus.SKIPPED,
            message="Image GGUF runtime loader is report-only; no model module was loaded.",
            issues=issues,
            runtime_loadable=False,
            loader_implemented=False,
            loadability=str(contract.get("loadability") or "shape_only_reference"),
            data={
                "component": contract.get("component") or (request.component if request else ""),
                "runtime_contract": contract,
                "reads_tensor_payloads": False,
                "builds_model_modules": False,
                "training_path_enabled": False,
            },
        )


def build_image_gguf_runtime_loader_abi(
    *,
    component: str,
    tensor_type_policy: Dict[str, Any] | None = None,
    required_runtime_features: List[str] | None = None,
    quality_gates: List[str] | None = None,
    runtime_entrypoint: str = "",
    implemented: bool = False,
    blockers: List[str] | None = None,
) -> Dict[str, Any]:
    """Build the report-only ABI descriptor embedded in shape-loader reports."""

    return {
        "schema_version": 1,
        "abi": IMAGE_GGUF_RUNTIME_LOADER_ABI,
        "provider": IMAGE_GGUF_RUNTIME_PROVIDER,
        "request_schema_id": IMAGE_GGUF_RUNTIME_REQUEST_SCHEMA_ID,
        "result_schema_id": IMAGE_GGUF_RUNTIME_RESULT_SCHEMA_ID,
        "component": str(component or ""),
        "implemented": bool(implemented),
        "report_only": not bool(implemented),
        "runtime_entrypoint": str(runtime_entrypoint or ""),
        "supported_load_modes": [ImageGGUFLoadMode.DESCRIPTOR_ONLY.value],
        "declared_future_load_modes": [
            ImageGGUFLoadMode.MODULE_BUILD.value,
            ImageGGUFLoadMode.QUALITY_PROBE.value,
        ],
        "reads_tensor_payloads": False,
        "builds_model_modules": False,
        "runs_forward_pass": False,
        "training_path_enabled": False,
        "tensor_type_policy": dict(tensor_type_policy or {}),
        "required_runtime_features": list(required_runtime_features or []),
        "quality_gates": list(quality_gates or []),
        "blockers": list(dict.fromkeys(str(item) for item in blockers or [] if str(item))),
    }


__all__ = [
    "IMAGE_GGUF_RUNTIME_LOADER_ABI",
    "IMAGE_GGUF_RUNTIME_PROVIDER",
    "IMAGE_GGUF_RUNTIME_REQUEST_SCHEMA_ID",
    "IMAGE_GGUF_RUNTIME_RESULT_SCHEMA_ID",
    "IMAGE_GGUF_SUPPORTED_COMPONENTS",
    "ImageGGUFLoadMode",
    "ImageGGUFRuntimeLoadRequest",
    "ImageGGUFRuntimeLoadResult",
    "build_image_gguf_runtime_loader_abi",
]
