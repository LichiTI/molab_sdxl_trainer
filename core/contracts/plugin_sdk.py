"""Plugin SDK request and registration contracts."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import BaseRequest


class PluginPermissionRequest(BaseRequest):
    """Permission declaration requested by a plugin manifest."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "plugin.permission"
    plugin_id: str = ""
    permissions: List[str] = Field(default_factory=list)
    safe_root_roles: List[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("plugin_id")
    @classmethod
    def _plugin_id_required(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("plugin_id is required")
        return value


class PluginRunnerRegistration(BaseRequest):
    """Declarative runner registration exposed by a plugin."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "plugin.runner-registration"
    plugin_id: str = ""
    runner_id: str = ""
    request_schema_id: str = ""
    entrypoint: str = ""
    job_type: str = "tool"
    permissions: List[str] = Field(default_factory=list)
    artifact_types: List[str] = Field(default_factory=list)

    @field_validator("plugin_id", "runner_id", "request_schema_id", "entrypoint")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("plugin runner registration fields cannot be empty")
        return value

    @model_validator(mode="after")
    def _required_defaults_present(self) -> "PluginRunnerRegistration":
        missing = [
            name
            for name in ("plugin_id", "runner_id", "request_schema_id", "entrypoint")
            if not str(getattr(self, name, "") or "").strip()
        ]
        if missing:
            raise ValueError("plugin runner registration fields cannot be empty: " + ", ".join(missing))
        _validate_entrypoint_format(self.entrypoint, "plugin runner entrypoint")
        return self


class PluginRequestSchemaRegistration(BaseRequest):
    """Declarative request schema exposed by a plugin."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "plugin.request-schema-registration"
    plugin_id: str = ""
    request_schema_id: str = ""
    version: int = 1
    kind: str = "tool"
    schema_path: str = ""
    title: str = ""
    description: str = ""

    @field_validator("plugin_id", "request_schema_id", "schema_path")
    @classmethod
    def _required_schema_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("plugin request schema fields cannot be empty")
        return value

    @field_validator("version")
    @classmethod
    def _positive_version(cls, value: int) -> int:
        value = int(value)
        if value < 1:
            raise ValueError("request schema version must be >= 1")
        return value


class PluginResourceDetectorRegistration(BaseRequest):
    """Declarative resource detector/provider exposed by a plugin."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "plugin.resource-detector-registration"
    plugin_id: str = ""
    detector_id: str = ""
    resource_type: str = ""
    entrypoint: str = ""
    permissions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("plugin_id", "detector_id", "resource_type", "entrypoint")
    @classmethod
    def _required_detector_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("plugin resource detector fields cannot be empty")
        return value

    @model_validator(mode="after")
    def _entrypoint_format_present(self) -> "PluginResourceDetectorRegistration":
        _validate_entrypoint_format(self.entrypoint, "plugin resource detector entrypoint")
        return self


class PluginArtifactHandlerRegistration(BaseRequest):
    """Declarative artifact handler exposed by a plugin."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "plugin.artifact-handler-registration"
    plugin_id: str = ""
    handler_id: str = ""
    artifact_kind: str = ""
    entrypoint: str = ""
    permissions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("plugin_id", "handler_id", "artifact_kind", "entrypoint")
    @classmethod
    def _required_handler_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("plugin artifact handler fields cannot be empty")
        return value

    @model_validator(mode="after")
    def _entrypoint_format_present(self) -> "PluginArtifactHandlerRegistration":
        _validate_entrypoint_format(self.entrypoint, "plugin artifact handler entrypoint")
        return self


class PluginUiSlotRegistration(BaseRequest):
    """Declarative UI slot registration for Launcher/WebUI hosts."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "plugin.ui-slot-registration"
    plugin_id: str = ""
    slot_id: str = ""
    host: str = "launcher"
    label_key: str = ""
    icon: str = ""
    command_id: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("plugin_id", "slot_id")
    @classmethod
    def _required_slot_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("plugin ui slot fields cannot be empty")
        return value

    @model_validator(mode="after")
    def _required_slot_defaults_present(self) -> "PluginUiSlotRegistration":
        missing = [name for name in ("plugin_id", "slot_id") if not str(getattr(self, name, "") or "").strip()]
        if missing:
            raise ValueError("plugin ui slot fields cannot be empty: " + ", ".join(missing))
        return self


def _validate_entrypoint_format(value: str, label: str) -> None:
    entry_file, separator, function_name = str(value or "").strip().partition(":")
    if not separator or not entry_file.strip() or not function_name.strip():
        raise ValueError(f"{label} must use file.py:function format")
