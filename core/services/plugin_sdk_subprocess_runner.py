"""Host-side subprocess harness for plugin SDK runners."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import BaseRequest, PlatformIssue, PluginRunnerRegistration, RunContext, RunResult

from .plugin_sdk_sandbox_policy import PluginSdkSandboxPolicy, is_sensitive_env_key


_BASE_ENV_KEYS = ("SystemRoot", "WINDIR", "TEMP", "TMP", "PATH")

def execute_plugin_sdk_runner_subprocess(
    *,
    registration: PluginRunnerRegistration,
    request: BaseRequest,
    context: RunContext,
    plugin_dir: Path,
    manifest_path: Path,
    approval_snapshot: dict[str, Any],
    sandbox_policy: PluginSdkSandboxPolicy,
    python_executable: str | None = None,
    subprocess_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> RunResult:
    """Run a plugin SDK function in a separate Python process."""

    python = python_executable or sys.executable
    if not python:
        return RunResult.failure(
            "Plugin subprocess execution requires a Python executable.",
            request_id=request.request_id,
            issues=[PlatformIssue(code="plugin.subprocess_python_missing", message="Python executable is not available.", severity="error")],
            data=_base_data(registration, approval_snapshot, sandbox_policy),
        )

    with tempfile.TemporaryDirectory(prefix="lulynx_plugin_sdk_") as tempdir:
        temp_root = Path(tempdir)
        input_path = temp_root / "input.json"
        result_path = temp_root / "result.json"
        input_path.write_text(
            json.dumps(
                {
                    "registration": registration.model_dump(mode="json"),
                    "request": request.model_dump(mode="json"),
                    "context": _context_payload(context, sandbox_policy=sandbox_policy),
                    "plugin_dir": str(plugin_dir),
                    "manifest_path": str(manifest_path),
                    "approval_snapshot": approval_snapshot,
                    "sandbox_policy": sandbox_policy.to_dict(),
                    "result_path": str(result_path),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        worker_path = Path(__file__).with_name("plugin_sdk_subprocess_worker.py")
        command = [python, str(worker_path), "--input", str(input_path)]
        try:
            completed = subprocess_runner(
                command,
                cwd=str(context.project_root),
                env=_subprocess_env(context, sandbox_policy),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=sandbox_policy.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RunResult.failure(
                "Plugin runner subprocess timed out.",
                request_id=request.request_id,
                issues=[
                    PlatformIssue(
                        code="plugin.subprocess_timeout",
                        message=f"Plugin runner exceeded {sandbox_policy.timeout_seconds} seconds.",
                        severity="error",
                    )
                ],
                data={**_base_data(registration, approval_snapshot, sandbox_policy), "stdout": str(exc.stdout or ""), "stderr": str(exc.stderr or "")},
            )

        if result_path.is_file():
            try:
                result = RunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
                if not result.request_id:
                    result.request_id = request.request_id
                result.data["approval_snapshot"] = approval_snapshot
                result.data["sandbox_policy"] = sandbox_policy.to_dict()
                result.data["subprocess"] = _completed_process_payload(completed)
                return result
            except Exception as exc:
                return RunResult.failure(
                    f"Plugin runner subprocess returned invalid result: {exc}",
                    request_id=request.request_id,
                    issues=[PlatformIssue(code="plugin.subprocess_result_invalid", message=str(exc), severity="error")],
                    data={**_base_data(registration, approval_snapshot, sandbox_policy), "subprocess": _completed_process_payload(completed)},
                )

        return RunResult.failure(
            "Plugin runner subprocess did not produce a result sidecar.",
            request_id=request.request_id,
            issues=[
                PlatformIssue(
                    code="plugin.subprocess_result_missing",
                    message="Plugin subprocess exited without writing result.json.",
                    severity="error",
                )
            ],
            data={**_base_data(registration, approval_snapshot, sandbox_policy), "subprocess": _completed_process_payload(completed)},
        )


def _context_payload(context: RunContext, *, sandbox_policy: PluginSdkSandboxPolicy | None = None) -> dict[str, Any]:
    return {
        "project_root": str(context.project_root),
        "backend_root": str(context.backend_root) if context.backend_root else "",
        "work_dir": str(context.work_dir) if context.work_dir else "",
        "safe_roots": [str(path) for path in context.safe_roots or ()],
        "runtime_id": context.runtime_id,
        "env": _context_env_payload(context, sandbox_policy),
        "metadata": _json_safe_metadata(dict(context.metadata or {})),
    }


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if callable(value):
            continue
        try:
            json.dumps(value)
        except TypeError:
            safe[str(key)] = str(value)
        else:
            safe[str(key)] = value
    return safe


def _subprocess_env(context: RunContext, sandbox_policy: PluginSdkSandboxPolicy) -> dict[str, str]:
    env: dict[str, str] = {}
    source = {**os.environ, **dict(context.env or {})}
    for key in _BASE_ENV_KEYS:
        value = source.get(key)
        if value is not None:
            env[key] = str(value)
    for key in sandbox_policy.env_allowlist:
        if is_sensitive_env_key(key):
            continue
        value = source.get(key)
        if value is not None:
            env[key] = str(value)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    repo_root = str(context.project_root)
    backend_root = str(context.backend_root or Path(context.project_root) / "backend")
    python_path_parts = [repo_root, backend_root]
    existing_pythonpath = source.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    return env


def _context_env_payload(context: RunContext, sandbox_policy: PluginSdkSandboxPolicy | None) -> dict[str, str]:
    if sandbox_policy is None:
        return {}
    visible: dict[str, str] = {}
    source = dict(context.env or {})
    for key in sandbox_policy.env_allowlist:
        if is_sensitive_env_key(key):
            continue
        value = source.get(key)
        if value is not None:
            visible[str(key)] = str(value)
    return visible


def _completed_process_payload(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "returncode": int(completed.returncode),
        "stdout_tail": str(completed.stdout or "")[-4000:],
        "stderr_tail": str(completed.stderr or "")[-4000:],
    }


def _base_data(
    registration: PluginRunnerRegistration,
    approval_snapshot: dict[str, Any],
    sandbox_policy: PluginSdkSandboxPolicy,
) -> dict[str, Any]:
    return {
        "plugin_id": registration.plugin_id,
        "runner_id": registration.runner_id,
        "approval_snapshot": approval_snapshot,
        "sandbox_policy": sandbox_policy.to_dict(),
    }
