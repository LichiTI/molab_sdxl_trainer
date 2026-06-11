"""Lightweight builders for quality report envelopes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.contracts import QualityMetric, QualityReport, QualityReportRequest, QualitySample
from backend.core.contracts.quality import _quality_report_evidence_summary
from backend.core.services.quality_report_renderer import write_quality_report_exports


def build_quality_report(request: QualityReportRequest, *, project_root: Path | None = None) -> QualityReport:
    """Build a quality report without running heavy visual metrics.

    The report intentionally starts as ``not_quality_validated`` unless manual
    review or metric providers explicitly add stronger evidence.
    """

    samples = [
        QualitySample(path=path, exists=_path_exists(path, project_root=project_root))
        for path in request.sample_paths
        if str(path or "").strip()
    ]
    metrics = [QualityMetric.model_validate(item) for item in request.metrics if isinstance(item, dict)]
    report_metadata = dict(request.report_metadata or {})
    missing_samples = [sample.path for sample in samples if not sample.exists]
    issues = []
    if missing_samples:
        issues.append(
            {
                "code": "quality.sample_missing",
                "severity": "warning",
                "message": "Some sample files do not exist.",
                "paths": missing_samples,
            }
        )
    status = _quality_status(request.manual_review_status, metrics, missing_samples)
    evidence_summary = build_quality_evidence_summary(
        samples=samples,
        metrics=metrics,
        issues=issues,
        manual_review_status=request.manual_review_status,
        quality_boundary=request.quality_boundary,
    )
    return QualityReport(
        schema_id=str(report_metadata.get("target_schema_id") or request.schema_id or "quality.report"),
        artifact_path=request.artifact_path,
        artifact_kind=request.artifact_kind,
        quality_status=status,
        quality_boundary=request.quality_boundary,
        review_level="manual" if request.manual_review_status != "not_requested" else "evidence-envelope",
        manual_review_status=request.manual_review_status,
        samples=samples,
        metrics=metrics,
        issues=issues,
        evidence_summary=evidence_summary,
        report_metadata=report_metadata,
    )


def build_quality_evidence_summary(
    *,
    samples: list[QualitySample],
    metrics: list[QualityMetric],
    issues: list[dict[str, Any]],
    manual_review_status: str,
    quality_boundary: str,
) -> dict[str, Any]:
    return _quality_report_evidence_summary(
        samples=samples,
        metrics=metrics,
        issues=issues,
        manual_review_status=manual_review_status,
        quality_boundary=quality_boundary,
    )


def _quality_status(manual_review_status: str, metrics: list[QualityMetric], missing_samples: list[str]) -> str:
    manual = str(manual_review_status or "not_requested").strip().lower()
    if manual in {"passed", "failed", "inconclusive"}:
        return manual
    if manual in {"pending", "requested", "manual_review_pending"}:
        return "manual_review_pending"
    if any(metric.status == "failed" for metric in metrics):
        return "failed"
    if missing_samples:
        return "inconclusive"
    return "not_quality_validated"


def _path_exists(path: str, *, project_root: Path | None = None) -> bool:
    text = str(path or "").strip()
    if not text:
        return False
    candidate = Path(text)
    if not candidate.is_absolute() and project_root is not None:
        candidate = project_root / candidate
    try:
        return candidate.exists()
    except OSError:
        return False


def build_quality_report_payload(
    payload: dict[str, Any],
    *,
    project_root: Path | None = None,
    safe_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    request = QualityReportRequest.from_legacy_payload(payload, source="fastapi", compat_mode=True)
    report = build_quality_report(request, project_root=project_root)
    result = {
        "request": request.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "artifact": report.to_artifact_manifest(request_id=request.request_id).model_dump(mode="json"),
    }
    export_dir = str(payload.get("export_dir") or "").strip()
    if export_dir:
        base = str(payload.get("export_basename") or "quality_report").strip() or "quality_report"
        safe_export_dir = _resolve_safe_export_dir(export_dir, project_root=project_root, safe_roots=safe_roots)
        result["exports"] = write_quality_report_exports(report, safe_export_dir, basename=base)
    return result


def _resolve_safe_export_dir(export_dir: str, *, project_root: Path | None, safe_roots: tuple[Path, ...]) -> Path:
    candidate = Path(export_dir)
    if not candidate.is_absolute() and project_root is not None:
        candidate = project_root / candidate
    roots = tuple(root for root in ((project_root,) if project_root is not None else ()) + tuple(safe_roots) if root is not None)
    if not roots:
        return candidate
    if _path_is_under_any_root(candidate, roots):
        return candidate
    raise ValueError(f"Quality report export_dir points outside safe roots: {export_dir}")


def _path_is_under_any_root(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return True
        except (OSError, ValueError):
            continue
    return False
