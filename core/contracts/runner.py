"""Runner protocol and registry for request-native execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Protocol

from .base import BaseRequest, JobEvent, RunResult


@dataclass(frozen=True)
class RunContext:
    """Execution context passed to runners after request validation."""

    project_root: Path
    backend_root: Path | None = None
    work_dir: Path | None = None
    safe_roots: tuple[Path, ...] = ()
    runtime_id: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_safe_path(self, path: str | Path) -> bool:
        """Return whether a path stays under one of the configured safe roots."""

        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            return False
        roots = self.safe_roots or (self.project_root,)
        for root in roots:
            try:
                resolved.relative_to(Path(root).resolve())
                return True
            except ValueError:
                continue
        return False


class RunnerProtocol(Protocol):
    """Protocol implemented by request-native runners."""

    runner_id: str
    schema_ids: tuple[str, ...]

    def run(self, request: BaseRequest, context: RunContext) -> RunResult:
        """Execute a validated request and return a common result."""


class RunnerRegistry:
    """In-memory registry for request-native runners."""

    def __init__(self) -> None:
        self._runners: Dict[str, RunnerProtocol] = {}
        self._schema_to_runner: Dict[str, str] = {}

    def register(self, runner: RunnerProtocol, *, replace: bool = False) -> None:
        runner_id = str(getattr(runner, "runner_id", "") or "").strip()
        if not runner_id:
            raise ValueError("runner_id is required")
        schema_ids = tuple(str(item).strip() for item in getattr(runner, "schema_ids", ()) if str(item).strip())
        if not schema_ids:
            raise ValueError("runner must declare at least one schema id")
        if runner_id in self._runners and not replace:
            raise ValueError(f"runner already registered: {runner_id}")
        for schema_id in schema_ids:
            existing = self._schema_to_runner.get(schema_id)
            if existing and existing != runner_id and not replace:
                raise ValueError(f"schema already registered: {schema_id}")
        self._runners[runner_id] = runner
        for schema_id in schema_ids:
            self._schema_to_runner[schema_id] = runner_id

    def get(self, runner_id: str) -> RunnerProtocol:
        try:
            return self._runners[runner_id]
        except KeyError as exc:
            raise KeyError(f"runner not registered: {runner_id}") from exc

    def resolve(self, schema_id: str) -> RunnerProtocol:
        runner_id = self._schema_to_runner.get(str(schema_id or ""))
        if not runner_id:
            raise KeyError(f"no runner registered for schema: {schema_id}")
        return self.get(runner_id)

    def run(self, request: BaseRequest, context: RunContext) -> RunResult:
        return self.resolve(request.schema_id).run(request, context)

    def capabilities(self) -> list[dict[str, Any]]:
        capabilities: list[dict[str, Any]] = []
        for runner_id, runner in sorted(self._runners.items()):
            metadata = getattr(runner, "capability_metadata", {})
            if callable(metadata):
                metadata = metadata()
            if not isinstance(metadata, dict):
                metadata = {}
            capabilities.append(
                {
                    "runner_id": runner_id,
                    "schema_ids": list(getattr(runner, "schema_ids", ())),
                    "supports_dry_run": bool(metadata.get("supports_dry_run", False)),
                    "permissions": list(metadata.get("permissions") or []),
                    "resources": list(metadata.get("resources") or []),
                    "heavy_dependencies": list(metadata.get("heavy_dependencies") or []),
                    "estimated_cost": str(metadata.get("estimated_cost") or "unknown"),
                    "metadata": dict(metadata.get("metadata") or {}),
                }
            )
        return capabilities

    def events_for_result(self, result: RunResult) -> Iterable[JobEvent]:
        yield JobEvent(
            job_id=result.run_id,
            request_id=result.request_id,
            status=result.status,
            message=result.message,
            payload={"artifact_count": len(result.artifacts), "issue_count": len(result.issues)},
        )
