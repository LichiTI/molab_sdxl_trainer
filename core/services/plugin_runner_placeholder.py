"""Registry-visible SDK runner placeholders for plugin runtime."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from backend.core.contracts import (
    BaseRequest,
    PlatformIssue,
    PluginRunnerRegistration,
    RunContext,
    RunResult,
)

from .plugin_execution_guard import elevated_runner_permissions


class PluginRuntimeRunnerProtocol(Protocol):
    def run_sdk_runner(
        self,
        registration: PluginRunnerRegistration,
        request: BaseRequest,
        context: RunContext,
    ) -> RunResult:
        ...


class PluginSdkResultFactory:
    """Small result factory exposed to plugin runner functions."""

    def success(self, **kwargs: Any) -> RunResult:
        return RunResult.success(**kwargs)

    def failure(self, message: str, **kwargs: Any) -> RunResult:
        return RunResult.failure(message, **kwargs)


class PluginSdkReporter:
    """Collect lightweight logs/progress emitted by one plugin runner."""

    def __init__(self, progress_callback: Callable[[float, float], Any] | None = None) -> None:
        self.logs: list[dict[str, Any]] = []
        self.progress: list[dict[str, Any]] = []
        self._progress_callback = progress_callback

    def log(self, message: str, *, level: str = "info", **metadata: Any) -> None:
        text = str(message or "").strip()
        if not text:
            return
        self.logs.append({"level": str(level or "info"), "message": text, "metadata": dict(metadata)})

    def report_progress(self, current: float, total: float = 1, *, message: str = "") -> None:
        self.progress.append({"current": current, "total": total, "message": str(message or "")})
        if self._progress_callback is not None:
            try:
                self._progress_callback(float(current), float(total))
            except Exception:
                pass

    def attach_to_result(self, result: RunResult) -> RunResult:
        if self.logs:
            result.data.setdefault("sdk_logs", list(self.logs))
        if self.progress:
            result.data.setdefault("sdk_progress", list(self.progress))
        return result


class PluginSdkExecutionContext:
    """Narrow SDK context passed to permissioned plugin runners."""

    def __init__(self, run_context: RunContext, registration: PluginRunnerRegistration) -> None:
        self.run_context = run_context
        self.registration = registration
        self.results = PluginSdkResultFactory()
        progress_callback = run_context.metadata.get("plugin_sdk_progress_callback")
        self.reporter = PluginSdkReporter(progress_callback if callable(progress_callback) else None)
        self.metadata = {
            "plugin_id": registration.plugin_id,
            "runner_id": registration.runner_id,
            "schema_id": registration.request_schema_id,
            "safe_root_roles": list(run_context.metadata.get("plugin_safe_root_roles") or run_context.metadata.get("safe_root_roles") or []),
            "safe_root_paths": dict(run_context.metadata.get("plugin_safe_root_paths") or run_context.metadata.get("safe_root_paths") or {}),
        }

    def log(self, message: str, *, level: str = "info", **metadata: Any) -> None:
        self.reporter.log(message, level=level, **metadata)

    def progress(self, current: float, total: float = 1, *, message: str = "") -> None:
        self.reporter.report_progress(current, total, message=message)

    def attach_report(self, result: RunResult) -> RunResult:
        return self.reporter.attach_to_result(result)


class PluginRunnerPlaceholder:
    """Registry-visible placeholder for declarative plugin runners.

    It deliberately does not execute plugin code unless a permissioned runtime
    harness is explicitly attached.
    """

    def __init__(
        self,
        registration: PluginRunnerRegistration,
        *,
        runtime: PluginRuntimeRunnerProtocol | None = None,
        execution_enabled: bool = False,
    ) -> None:
        self.registration = registration
        self.runtime = runtime
        self.execution_enabled = execution_enabled
        self.runner_id = f"plugin.{registration.plugin_id}.{registration.runner_id}"
        self.schema_ids = (registration.request_schema_id,)
        self.capability_metadata = {
            "supports_dry_run": False,
            "permissions": list(registration.permissions),
            "resources": [],
            "heavy_dependencies": [],
            "estimated_cost": "plugin-declared",
            "metadata": {
                "plugin_id": registration.plugin_id,
                "plugin_runner_id": registration.runner_id,
                "entrypoint": registration.entrypoint,
                "job_type": registration.job_type,
                "artifact_types": list(registration.artifact_types),
                "elevated_permissions": elevated_runner_permissions(list(registration.permissions)),
                "execution_enabled": execution_enabled,
            },
        }

    def run(self, request: BaseRequest, context: RunContext) -> RunResult:
        if self.execution_enabled and self.runtime is not None:
            return self.runtime.run_sdk_runner(self.registration, request, context)
        return RunResult.failure(
            "Plugin runner execution is not enabled yet.",
            request_id=request.request_id,
            issues=[
                PlatformIssue(
                    code="plugin.runner_execution_unavailable",
                    message="The plugin runner is registered for discovery only until the safe execution harness is connected.",
                    severity="error",
                )
            ],
            data={
                "plugin_id": self.registration.plugin_id,
                "runner_id": self.registration.runner_id,
                "entrypoint": self.registration.entrypoint,
            },
        )
