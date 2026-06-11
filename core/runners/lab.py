"""Lulynx LAB runners for request-native orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.contracts import LabRequest, PlatformIssue, RunContext, RunResult, RunnerRegistry
from backend.core.services.lab_subprocess_runner import (
    LabSubprocessJobSpec,
    LabSubprocessSubmissionError,
    submit_lab_subprocess_job,
)


class LabSubprocessRunner:
    """Registry-backed adapter for existing LAB subprocess tools.

    Routes still prepare schema-specific configs for compatibility, but the
    actual JobManager/subprocess/result handoff goes through this runner boundary.
    """

    runner_id = "lab.subprocess-runner"
    schema_ids = (
        "lab-distiller",
        "sdxl-turbo-lora",
        "anima-few-step-lora",
        "newbie-few-step-lora",
        "artifact.validation",
        "artifact.report",
    )
    capability_metadata = {
        "supports_dry_run": True,
        "permissions": ["read_models", "read_dataset", "write_output", "start_training"],
        "resources": ["runtime", "runner_script", "config", "output"],
        "heavy_dependencies": [],
        "estimated_cost": "medium",
        "metadata": {
            "real_training": True,
            "note": "Subprocess bridge for existing Lulynx LAB runners.",
        },
    }

    def __init__(self, *, job_manager: Any | None = None, telemetry_reader_factory: Any | None = None) -> None:
        self._job_manager = job_manager
        self._telemetry_reader_factory = telemetry_reader_factory

    def run(self, request: Any, context: RunContext) -> RunResult:
        if not isinstance(request, LabRequest):
            request = LabRequest.from_legacy_payload(_coerce_lab_request_payload(request))
        if "start_training" not in set(context.metadata.get("permissions") or []):
            return RunResult.failure(
                "LAB subprocess execution requires start_training permission.",
                request_id=request.request_id,
                issues=[
                    PlatformIssue(
                        code="lab.permission_required",
                        message="start_training permission is required for LAB subprocess dispatch.",
                        severity="error",
                        field="permissions",
                    )
                ],
                data={"runner_id": self.runner_id, "schema_id": request.schema_id},
            )

        try:
            spec = LabSubprocessJobSpec.from_mapping(context.metadata.get("lab_subprocess_job") or {})
            payload = submit_lab_subprocess_job(
                spec,
                project_root=context.project_root,
                backend_root=context.backend_root or context.project_root / "backend",
                job_manager=self._job_manager or _default_job_manager(),
                telemetry_reader_factory=self._telemetry_reader_factory,
            )
        except LabSubprocessSubmissionError as exc:
            return _failure_from_error(request, self.runner_id, str(exc), status_code=exc.status_code)
        except Exception as exc:
            return _failure_from_error(request, self.runner_id, str(exc), status_code=500)

        run_result = payload.get("run_result")
        if isinstance(run_result, dict):
            result = RunResult.model_validate(run_result)
        else:
            result = RunResult(status="queued", run_id=str(payload.get("task_id") or ""), request_id=request.request_id)
        result.request_id = result.request_id or request.request_id
        result.data = {**dict(result.data or {}), "runner_id": self.runner_id, "legacy_payload": payload}
        return result


def _default_job_manager() -> Any:
    from resources.web.deps import get_job_manager

    return get_job_manager()


def _coerce_lab_request_payload(request: Any) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        data = request.model_dump(mode="json")
    elif isinstance(request, dict):
        data = dict(request)
    else:
        data = {"schema_id": getattr(request, "schema_id", "")}
    schema_id = str(data.get("schema_id") or data.get("lab_id") or "lulynx-lab")
    data.setdefault("schema_id", schema_id)
    data.setdefault("tool_id", "lulynx-lab")
    data.setdefault("lab_id", schema_id)
    return data


def _failure_from_error(request: LabRequest, runner_id: str, message: str, *, status_code: int) -> RunResult:
    return RunResult.failure(
        message,
        request_id=request.request_id,
        issues=[
            PlatformIssue(
                code="lab.subprocess_submission_failed",
                message=message,
                severity="error",
                details={"status_code": status_code},
            )
        ],
        data={"runner_id": runner_id, "schema_id": request.schema_id, "status_code": status_code},
    )


def create_lab_registry(*, job_manager: Any | None = None, telemetry_reader_factory: Any | None = None) -> RunnerRegistry:
    registry = RunnerRegistry()
    registry.register(LabSubprocessRunner(job_manager=job_manager, telemetry_reader_factory=telemetry_reader_factory))
    return registry
