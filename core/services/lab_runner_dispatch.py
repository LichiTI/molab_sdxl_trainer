"""Registry dispatch helpers for Lulynx LAB subprocess jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from backend.core.contracts import LabRequest, RequestSource, RunContext, RunResult
from backend.core.runners.lab import create_lab_registry
from backend.core.services.lab_subprocess_runner import LabSubprocessJobSpec


class LabRunnerDispatchError(RuntimeError):
    """Raised when the LAB registry boundary cannot produce a legacy payload."""

    def __init__(self, message: str, *, status_code: int = 500, result: RunResult | None = None) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.result = result


def build_lab_request_for_spec(spec: LabSubprocessJobSpec, *, source: RequestSource = RequestSource.FASTAPI) -> LabRequest:
    """Build the request-native LAB request for a prepared subprocess spec."""

    schema_id = str(spec.metadata.get("schema_id") or spec.config.get("schema_id") or "lulynx-lab")
    return LabRequest.from_legacy_payload(
        {
            "schema_id": schema_id,
            "tool_id": "lulynx-lab",
            "lab_id": schema_id,
            "runtime_id": spec.runtime_id,
            "dry_run": bool(spec.config.get("dry_run", True)),
        },
        source=source,
    )


def run_lab_subprocess_spec_via_registry(
    spec: LabSubprocessJobSpec,
    *,
    project_root: Path,
    backend_root: Path,
    job_manager: Any,
    permissions: Iterable[str] = ("start_training",),
    telemetry_reader_factory: Any | None = None,
) -> RunResult:
    """Dispatch a LAB subprocess spec through the request-native runner registry."""

    request = build_lab_request_for_spec(spec)
    context = RunContext(
        project_root=project_root,
        backend_root=backend_root,
        runtime_id=spec.runtime_id,
        metadata={
            "permissions": list(permissions),
            "lab_subprocess_job": spec.to_dict(),
        },
    )
    return create_lab_registry(
        job_manager=job_manager,
        telemetry_reader_factory=telemetry_reader_factory,
    ).run(request, context)


def submit_lab_subprocess_spec_for_legacy(
    spec: LabSubprocessJobSpec,
    *,
    project_root: Path,
    backend_root: Path,
    job_manager: Any,
    telemetry_reader_factory: Any | None = None,
) -> dict[str, Any]:
    """Run a LAB spec through the registry and return the legacy route payload."""

    result = run_lab_subprocess_spec_via_registry(
        spec,
        project_root=project_root,
        backend_root=backend_root,
        job_manager=job_manager,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    status = str(getattr(result.status, "value", result.status) or "")
    if status not in {"queued", "running", "succeeded"}:
        status_code = int(result.data.get("status_code") or 500)
        raise LabRunnerDispatchError(
            result.message or "LAB runner submission failed",
            status_code=status_code,
            result=result,
        )
    payload = result.data.get("legacy_payload")
    if not isinstance(payload, dict) or not payload:
        raise LabRunnerDispatchError(
            "LAB runner did not return a legacy payload",
            status_code=500,
            result=result,
        )
    return dict(payload)


__all__ = [
    "LabRunnerDispatchError",
    "build_lab_request_for_spec",
    "run_lab_subprocess_spec_via_registry",
    "submit_lab_subprocess_spec_for_legacy",
]
