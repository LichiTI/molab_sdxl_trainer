"""Runtime, repair, and environment action request contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import ConfigDict, Field, field_validator

from .base import BaseRequest


class RuntimeAction(str, Enum):
    PROBE = "probe"
    INSTALL = "install"
    REPAIR = "repair"
    MIRROR_TEST = "mirror-test"
    START_BACKEND = "start-backend"
    STOP_BACKEND = "stop-backend"
    RESTART_BACKEND = "restart-backend"
    INSTALL_WD14_DEPS = "install-wd14-deps"


class RuntimeRequest(BaseRequest):
    """Canonical request for launcher/runtime environment actions."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "runtime.action"
    action: RuntimeAction = RuntimeAction.PROBE
    runtime_id: str = ""
    target: str = ""
    python_executable: str = ""
    working_dir: str = ""
    mirror_url: str = ""
    packages: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    dry_run: bool = False
    confirm_destructive: bool = False
    permissions: List[str] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> object:
        text = str(value or RuntimeAction.PROBE.value).strip().lower().replace("_", "-")
        aliases = {
            "mirror": "mirror-test",
            "test-mirror": "mirror-test",
            "wd14": "install-wd14-deps",
            "install-wd14": "install-wd14-deps",
            "backend-start": "start-backend",
            "backend-stop": "stop-backend",
            "backend-restart": "restart-backend",
        }
        return aliases.get(text, text)

    def requires_confirmation(self) -> bool:
        return self.action in {RuntimeAction.STOP_BACKEND.value, RuntimeAction.RESTART_BACKEND.value}


class RuntimeOptimizationRequest(BaseRequest):
    """Request boundary for cross-cutting runtime optimization intent.

    This stays outside training/generation hot paths. Adapters resolve it into
    plain config fields before model loading or step execution.
    """

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "runtime.optimization"
    route: str = ""
    compile_runtime: str = "off"
    torch_compile: bool | None = None
    torch_compile_backend: str = ""
    torch_compile_mode: str = ""
    torch_compile_dynamic: bool | None = None
    torch_compile_fullgraph: bool | None = None
    torch_compile_scope: str = ""
    anima_compile_scope: str = ""
    compile_cache_enabled: bool | None = None
    compile_shape_strategy: str = "auto"
    compile_target_strategy: str = "auto"
    compile_contract_strict: bool | None = None
    compile_static_shape_drop_last: bool | None = None
    compile_require_cache_first: bool | None = None
    native_token_bucket_compile: bool | None = None
    explicit_fields: List[str] = Field(default_factory=list)

    @field_validator("compile_runtime", mode="before")
    @classmethod
    def _normalize_compile_runtime(cls, value: object) -> str:
        text = str(value or "off").strip().lower().replace("-", "_").replace(" ", "")
        aliases = {
            "": "off",
            "default": "auto",
            "smart": "auto",
            "none": "off",
            "disabled": "off",
            "false": "off",
            "0": "off",
            "torch_compile": "compile",
            "compile_only": "compile",
            "cache": "compile_cache",
            "cached": "compile_cache",
            "compile_cached": "compile_cache",
            "cuda_graph": "cudagraph",
            "cuda_graphs": "cudagraph",
            "cudagraphs": "cudagraph",
            "compile_cuda_graph": "compile_cudagraph",
            "compile_cuda_graphs": "compile_cudagraph",
            "compile_cudagraphs": "compile_cudagraph",
        }
        normalized = aliases.get(text, text)
        return normalized if normalized in {"off", "auto", "compile", "compile_cache", "cudagraph", "compile_cudagraph"} else "off"

    @field_validator("compile_shape_strategy", mode="before")
    @classmethod
    def _normalize_shape_strategy(cls, value: object) -> str:
        text = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "")
        aliases = {
            "": "auto",
            "default": "auto",
            "pad": "fixed_pad",
            "static_pad": "fixed_pad",
            "flatten": "token_flatten",
            "tokenflatten": "token_flatten",
            "native_no_pad": "native",
            "no_pad": "native",
        }
        normalized = aliases.get(text, text)
        return normalized if normalized in {"auto", "fixed_pad", "token_flatten", "native"} else "auto"

    @field_validator("compile_target_strategy", mode="before")
    @classmethod
    def _normalize_target_strategy(cls, value: object) -> str:
        text = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "")
        aliases = {
            "": "auto",
            "default": "auto",
            "per_block": "block",
            "blocks": "block",
            "inner": "inner_forward",
            "innerforward": "inner_forward",
            "forward_impl": "inner_forward",
        }
        normalized = aliases.get(text, text)
        return normalized if normalized in {"auto", "block", "inner_forward"} else "auto"


class RuntimeOptimizationResolution(BaseRequest):
    """Resolved runtime optimization fields for compatibility config adapters."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "runtime.optimization.resolution"
    route: str = ""
    fields: Dict[str, Any] = Field(default_factory=dict)
    requested: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    fallback_reason: str = ""


class RepairAction(BaseRequest):
    """Small action object for notification center one-click repair flows."""

    schema_id: str = "runtime.repair-action"
    action_id: str = ""
    label: str = ""
    runtime_request: RuntimeRequest = Field(default_factory=RuntimeRequest)
    issue_codes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("action_id")
    @classmethod
    def _action_id_required(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("action_id is required")
        return value
