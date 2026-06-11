"""TurboCore request-native bridge contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import ConfigDict, Field, field_validator

from .base import BaseRequest, RunResult


class TurboCoreBridgeAction(str, Enum):
    PROBE = "probe"
    CAPABILITY = "capability"
    WORKSPACE_PIPELINE_LIFECYCLE = "workspace-pipeline-lifecycle"
    VALIDATE_NATIVE_ABI = "validate-native-abi"


class TurboCoreBridgeRequest(BaseRequest):
    """Canonical request for developer-only TurboCore bridge probes.

    This is an orchestration boundary. It must stay out of training hot paths
    and should be resolved through a runner/service runtime.
    """

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "turbocore.bridge"
    action: TurboCoreBridgeAction = TurboCoreBridgeAction.PROBE
    runtime_id: str = ""
    model_type: str = "anima"
    training_type: str = "lora"
    features: List[str] = Field(default_factory=lambda: ["workspace_pool", "data_pipeline"])
    workspace_mb: int = 0
    prefetch_depth: int = 2
    lifecycle_batches: int = 4
    device: str = "cpu"
    dtype: str = "float32"
    native_report: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> object:
        text = str(value or TurboCoreBridgeAction.PROBE.value).strip().lower().replace("_", "-")
        aliases = {
            "workspace": "workspace-pipeline-lifecycle",
            "workspace-pipeline": "workspace-pipeline-lifecycle",
            "pipeline": "workspace-pipeline-lifecycle",
            "native-abi": "validate-native-abi",
            "validate": "validate-native-abi",
            "capabilities": "capability",
        }
        return aliases.get(text, text)

    @field_validator("workspace_mb", "prefetch_depth", "lifecycle_batches")
    @classmethod
    def _non_negative_ints(cls, value: int) -> int:
        return max(int(value), 0)

    def normalized_features(self) -> List[str]:
        return list(dict.fromkeys(str(item).strip().lower() for item in self.features if str(item).strip()))


class TurboCoreBridgeResult(RunResult):
    """RunResult specialization for TurboCore bridge probes."""

    pass
