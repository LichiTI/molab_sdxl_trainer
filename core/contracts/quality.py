"""Quality report contracts for post-run artifact review."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .base import ArtifactFile, ArtifactManifest, BaseRequest


class QualityMetric(BaseModel):
    """One optional quality/evidence metric.

    Metrics are evidence, not automatic promotion gates. Heavy providers such as
    CLIP/Jina CLIP can add rows here later without changing runner envelopes.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    value: float | str | bool | None = None
    status: str = "recorded"
    level: str = "optional"
    source: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QualitySample(BaseModel):
    """One sample artifact considered by a quality report."""

    model_config = ConfigDict(extra="allow")

    path: str
    exists: bool = False
    role: str = "sample"
    prompt: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QualityReport(BaseModel):
    """Common report envelope for human and optional metric-based review."""

    model_config = ConfigDict(extra="allow")

    report_schema: str = "lulynx.quality-report.v1"
    schema_id: str = ""
    artifact_path: str = ""
    artifact_kind: str = "model"
    quality_status: str = "not_quality_validated"
    quality_boundary: str = "evidence_envelope_not_final_quality_gate"
    review_level: str = "none"
    manual_review_status: str = "not_requested"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    samples: List[QualitySample] = Field(default_factory=list)
    metrics: List[QualityMetric] = Field(default_factory=list)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    report_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("quality_status")
    @classmethod
    def _valid_quality_status(cls, value: str) -> str:
        value = str(value or "not_quality_validated").strip().lower()
        allowed = {
            "not_quality_validated",
            "manual_review_pending",
            "passed",
            "failed",
            "inconclusive",
        }
        if value not in allowed:
            raise ValueError(f"quality_status must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def _default_evidence_summary(self) -> "QualityReport":
        if not self.evidence_summary:
            self.evidence_summary = _quality_report_evidence_summary(
                samples=self.samples,
                metrics=self.metrics,
                issues=self.issues,
                manual_review_status=self.manual_review_status,
                quality_boundary=self.quality_boundary,
            )
        return self

    def to_artifact_manifest(self, *, request_id: str = "", producer: str = "quality-report") -> ArtifactManifest:
        files = [ArtifactFile(path=self.artifact_path, role="subject")] if self.artifact_path else []
        files.extend(ArtifactFile(path=sample.path, role=sample.role, media_type="image/png") for sample in self.samples)
        return ArtifactManifest(
            artifact_kind="quality-report",
            schema_id=self.schema_id,
            producer=producer,
            request_id=request_id,
            files=files,
            metadata={
                "quality_status": self.quality_status,
                "quality_boundary": self.quality_boundary,
                "review_level": self.review_level,
                "manual_review_status": self.manual_review_status,
                "metric_count": len(self.metrics),
                "sample_count": len(self.samples),
                "evidence_summary": dict(self.evidence_summary),
                **self.report_metadata,
            },
            validation={"quality_report": self.model_dump(mode="json")},
        )


def _quality_report_evidence_summary(
    *,
    samples: List[QualitySample],
    metrics: List[QualityMetric],
    issues: List[Dict[str, Any]],
    manual_review_status: str,
    quality_boundary: str,
) -> Dict[str, Any]:
    metric_status_counts: Dict[str, int] = {}
    for metric in metrics:
        status = str(metric.status or "recorded")
        metric_status_counts[status] = metric_status_counts.get(status, 0) + 1
    issue_severity_counts: Dict[str, int] = {}
    for issue in issues:
        severity = str(issue.get("severity") or "info")
        issue_severity_counts[severity] = issue_severity_counts.get(severity, 0) + 1
    existing_samples = sum(1 for sample in samples if sample.exists)
    return {
        "quality_boundary": quality_boundary,
        "sample_count": len(samples),
        "existing_sample_count": existing_samples,
        "missing_sample_count": len(samples) - existing_samples,
        "metric_count": len(metrics),
        "metric_status_counts": metric_status_counts,
        "issue_count": len(issues),
        "issue_severity_counts": issue_severity_counts,
        "manual_review_status": str(manual_review_status or "not_requested"),
    }


class QualityReportRequest(BaseRequest):
    """Request for building a lightweight quality report envelope."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = "quality.report"
    artifact_path: str = ""
    artifact_kind: str = "model"
    sample_paths: List[str] = Field(default_factory=list)
    quality_boundary: str = "evidence_envelope_not_final_quality_gate"
    manual_review_status: str = "not_requested"
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    report_metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _split_legacy_report_metadata(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            return raw

        platform_keys = {
            "request_id",
            "source",
            "source_version",
            "user_id",
            "session_id",
            "trace_id",
            "created_at",
            "labels",
        }
        platform_metadata = {key: value for key, value in metadata.items() if key in platform_keys}
        report_metadata = {key: value for key, value in metadata.items() if key not in platform_keys}
        if report_metadata and not raw.get("report_metadata"):
            raw["report_metadata"] = report_metadata
        if report_metadata:
            if platform_metadata:
                raw["metadata"] = platform_metadata
            else:
                raw.pop("metadata", None)
        return raw
