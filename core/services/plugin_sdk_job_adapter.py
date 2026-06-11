"""Job submission adapter for plugin SDK runners."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.core.contracts import BaseRequest, PluginRunnerRegistration, RunContext

from .plugin_sdk_job_summary import build_plugin_sdk_job_summary


ApprovalSnapshotBuilder = Callable[[PluginRunnerRegistration | None], dict[str, Any]]


def plugin_sdk_default_safe_root_paths(project_root: Path, backend_root: Path | None) -> dict[str, list[str]]:
    """Return host-owned safe-root role paths for SDK runner jobs."""

    roots: dict[str, list[str]] = {
        "project": [str(project_root)],
        "models": [str(project_root / "models")],
        "output": [str(project_root / "output")],
        "cache": [str(project_root / "cache"), str(project_root / "backend" / "cache")],
        "plugin_data": [str(project_root / "data" / "plugins")],
    }
    if backend_root is not None:
        roots["backend"] = [str(backend_root)]
    return roots


def submit_sdk_runner_job(
    *,
    registry: Any,
    runner_id: str,
    payload: dict[str, Any] | None,
    project_root: Path,
    backend_root: Path | None,
    job_manager: Any,
    approval_snapshot_builder: ApprovalSnapshotBuilder,
    host_policy_metadata: dict[str, Any] | None = None,
    requested_by: str = "ui-user",
) -> dict[str, Any]:
    """Submit a declarative plugin runner through the common job manager."""

    from backend.core.job_manager import Job, JobType

    try:
        runner = registry.get(runner_id)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    schema_id = str(getattr(runner, "schema_ids", ("",))[0] or "")
    request_data = dict(payload or {})
    request_data.setdefault("schema_id", schema_id)
    request = BaseRequest.from_legacy_payload(request_data, source="plugin", compat_mode=True)
    approval_snapshot = approval_snapshot_builder(getattr(runner, "registration", None))
    job = Job(
        type=JobType.GENERIC,
        name=f"Plugin Runner: {runner_id}",
        total_items=1,
        metadata={
            "kind": "plugin_sdk_runner",
            "runner_id": runner_id,
            "schema_id": request.schema_id,
            "requested_by": requested_by,
            "request": request.model_dump(mode="json"),
            "approval_snapshot": approval_snapshot,
        },
    )

    def _worker(progress_callback=None, cancel_check=None):
        if progress_callback:
            progress_callback(0, 1)
        if cancel_check and cancel_check():
            raise RuntimeError("Plugin runner job cancelled")
        metadata = {
            "allow_plugin_execution": True,
            "plugin_permission_source": "approval-store",
            "plugin_safe_root_roles": ["project", "backend", "models", "output", "cache", "plugin_data"],
            "plugin_safe_root_paths": plugin_sdk_default_safe_root_paths(project_root, backend_root),
            "plugin_sdk_progress_callback": progress_callback,
        }
        metadata.update(dict(host_policy_metadata or {}))
        context = RunContext(
            project_root=project_root,
            backend_root=backend_root,
            safe_roots=(project_root,),
            metadata=metadata,
        )
        result = registry.run(request, context)
        sdk_summary = build_plugin_sdk_job_summary(
            result,
            runner_id=runner_id,
            schema_id=request.schema_id,
            approval_snapshot=approval_snapshot,
        )
        events = [event.model_dump(mode="json") for event in registry.events_for_result(result)]
        job.metadata["approval_snapshot"] = result.data.get("approval_snapshot") or approval_snapshot
        job.metadata["run_result"] = result.model_dump(mode="json")
        job.metadata["sdk_summary"] = sdk_summary
        job.metadata["events"] = events
        job.metadata["artifacts"] = [artifact.model_dump(mode="json") for artifact in result.artifacts]
        last_progress = sdk_summary.get("last_progress") if isinstance(sdk_summary, dict) else None
        if progress_callback and isinstance(last_progress, dict):
            current = last_progress.get("current")
            total = last_progress.get("total")
            if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
                progress_callback(current, total)
        if progress_callback:
            progress_callback(1, 1)
        if not result.ok:
            raise RuntimeError(result.message or "Plugin runner failed")
        return result

    job_id = job_manager.submit(job, worker_func=_worker)
    return {"success": True, "job_id": job_id, "runner_id": runner_id, "schema_id": request.schema_id}
