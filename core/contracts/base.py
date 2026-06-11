"""Base request/result contracts for the request-native platform roadmap."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RequestSource(str, Enum):
    """Origin of a request before it enters a runner."""

    UNKNOWN = "unknown"
    WEBUI = "webui"
    LAUNCHER = "launcher"
    FASTAPI = "fastapi"
    CLI = "cli"
    PLUGIN = "plugin"
    TEST = "test"


class RunStatus(str, Enum):
    """Common run/job terminal and in-flight states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class RequestMetadata(BaseModel):
    """Metadata that travels with a request but is not business config."""

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    source: RequestSource = RequestSource.UNKNOWN
    source_version: str = ""
    user_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = Field(default_factory=dict)


class BaseRequest(BaseModel):
    """Base class for request-native API boundary models.

    Unknown fields are preserved deliberately. During migration, legacy payloads
    often contain extra keys; adapters can inspect ``unknown_fields`` instead of
    silently dropping data.
    """

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = ""
    schema_version: int = 1
    compat_mode: bool = False
    metadata: RequestMetadata = Field(default_factory=RequestMetadata)

    @field_validator("schema_version")
    @classmethod
    def _positive_schema_version(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError("schema_version must be >= 1")
        return int(value)

    @classmethod
    def from_legacy_payload(
        cls,
        payload: Dict[str, Any],
        *,
        source: RequestSource | str = RequestSource.UNKNOWN,
        compat_mode: bool = True,
    ) -> "BaseRequest":
        """Create a request from an existing untyped payload."""

        data = dict(payload or {})
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            metadata = {**metadata, "source": metadata.get("source") or source}
        else:
            metadata = {"source": source}
        data["metadata"] = metadata
        data.setdefault("compat_mode", compat_mode)
        return cls.model_validate(data)

    @property
    def request_id(self) -> str:
        return self.metadata.request_id

    @property
    def unknown_fields(self) -> List[str]:
        return sorted((self.model_extra or {}).keys())

    def to_legacy_config(self) -> Dict[str, Any]:
        """Return a plain dict suitable for compatibility adapters."""

        data = self.model_dump(mode="json")
        data.pop("metadata", None)
        data.pop("compat_mode", None)
        data.pop("schema_version", None)
        return data


class PlatformIssue(BaseModel):
    """Structured warning/error emitted by resolvers, preflight, or runners."""

    model_config = ConfigDict(extra="allow")

    code: str
    message: str
    severity: str = "warning"
    field: str = ""
    hint: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class ArtifactFile(BaseModel):
    """A concrete file produced or consumed by a run."""

    model_config = ConfigDict(extra="allow")

    path: str
    role: str = "output"
    media_type: str = ""
    size_bytes: Optional[int] = None
    sha256: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactManifest(BaseModel):
    """Common manifest envelope for outputs shown in Resource Center/Jobs."""

    model_config = ConfigDict(extra="allow")

    artifact_id: str = Field(default_factory=lambda: uuid4().hex)
    artifact_kind: str
    schema_id: str = ""
    producer: str = ""
    producer_version: str = ""
    request_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    files: List[ArtifactFile] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)


class JobEvent(BaseModel):
    """Uniform event emitted by jobs/runners for UI and logs."""

    model_config = ConfigDict(extra="allow")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    job_id: str = ""
    request_id: str = ""
    status: RunStatus = RunStatus.RUNNING
    message: str = ""
    progress_current: Optional[float] = None
    progress_total: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: Dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Common return envelope for runners and repair/tool actions."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    request_id: str = ""
    status: RunStatus = RunStatus.SUCCEEDED
    message: str = ""
    issues: List[PlatformIssue] = Field(default_factory=list)
    artifacts: List[ArtifactManifest] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.SUCCEEDED or self.status == RunStatus.SUCCEEDED.value

    @classmethod
    def success(cls, **kwargs: Any) -> "RunResult":
        return cls(status=RunStatus.SUCCEEDED, **kwargs)

    @classmethod
    def failure(cls, message: str, *, issues: List[PlatformIssue] | None = None, **kwargs: Any) -> "RunResult":
        return cls(status=RunStatus.FAILED, message=message, issues=issues or [], **kwargs)
