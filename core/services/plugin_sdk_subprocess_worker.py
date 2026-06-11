"""Worker process entry point for plugin SDK runner subprocess execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = _BACKEND_ROOT.parent
_PLUGIN_CORE_SRC = _BACKEND_ROOT / "core" / "warehouse" / "lulynx_plugin_core" / "src"
_ROUTE_CONTRACT_SRC = _BACKEND_ROOT / "core" / "warehouse" / "lulynx_route_contract" / "src"
for _path in (_PROJECT_ROOT, _BACKEND_ROOT, _PLUGIN_CORE_SRC, _ROUTE_CONTRACT_SRC):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from backend.core.contracts import BaseRequest, PlatformIssue, PluginRunnerRegistration, RunContext, RunResult
from backend.core.services.plugin_loader import load_plugin_functions
from backend.core.services.plugin_runner_placeholder import PluginSdkExecutionContext
from backend.core.services.plugin_sdk_sandbox_policy import PluginSdkSandboxPolicy
from lulynx_plugin_core.manifest import load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one plugin SDK runner function in a subprocess.")
    parser.add_argument("--input", required=True, help="JSON input sidecar path")
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result_path = Path(payload["result_path"])
    try:
        result = run_worker_payload(payload)
    except Exception as exc:
        registration_payload = payload.get("registration") or {}
        request_payload = payload.get("request") or {}
        result = RunResult.failure(
            f"Plugin subprocess worker failed: {exc}",
            request_id=str(request_payload.get("metadata", {}).get("request_id") or ""),
            issues=[PlatformIssue(code="plugin.runner_failed", message=str(exc), severity="error")],
            data={
                "plugin_id": registration_payload.get("plugin_id", ""),
                "runner_id": registration_payload.get("runner_id", ""),
                "sandbox_policy": payload.get("sandbox_policy") or {},
            },
        )
    result_path.write_text(result.model_dump_json(), encoding="utf-8")
    return 0 if result.ok else 1


def run_worker_payload(payload: dict[str, Any]) -> RunResult:
    registration = PluginRunnerRegistration.model_validate(payload["registration"])
    request = BaseRequest.model_validate(payload["request"])
    context = _context_from_payload(payload.get("context") or {})
    policy_payload = payload.get("sandbox_policy") or {}
    sandbox_policy = PluginSdkSandboxPolicy(
        execution_mode=str(policy_payload.get("execution_mode") or "subprocess"),
        timeout_seconds=int(policy_payload.get("timeout_seconds") or 30),
        env_allowlist=tuple(str(item) for item in policy_payload.get("env_allowlist") or []),
        requested_by=str(policy_payload.get("requested_by") or "subprocess-worker"),
    )
    entry_file, _, function_name = str(registration.entrypoint).partition(":")
    if not entry_file or not function_name:
        return RunResult.failure(
            "Plugin runner entrypoint must use file.py:function format.",
            request_id=request.request_id,
            data={"plugin_id": registration.plugin_id, "runner_id": registration.runner_id},
        )
    manifest = load_manifest(Path(payload["manifest_path"]))
    functions = load_plugin_functions(
        Path(payload["plugin_dir"]),
        manifest,
        [function_name],
        entry=entry_file,
        permissions=list((payload.get("approval_snapshot") or {}).get("granted_permissions") or []),
    )
    sdk_context = PluginSdkExecutionContext(context, registration)
    raw_result = functions[function_name](request, sdk_context)
    if isinstance(raw_result, RunResult):
        result = raw_result
    elif isinstance(raw_result, dict):
        result = RunResult.model_validate({"request_id": request.request_id, **raw_result})
    else:
        result = RunResult.failure(
            "Plugin runner returned an unsupported result type.",
            request_id=request.request_id,
            data={"plugin_id": registration.plugin_id, "runner_id": registration.runner_id},
        )
    if not result.request_id:
        result.request_id = request.request_id
    result = sdk_context.attach_report(result)
    result.data["approval_snapshot"] = payload.get("approval_snapshot") or {}
    result.data["sandbox_policy"] = sandbox_policy.to_dict()
    return result


def _context_from_payload(payload: dict[str, Any]) -> RunContext:
    project_root = Path(payload.get("project_root") or ".").resolve()
    backend_root_text = str(payload.get("backend_root") or "").strip()
    work_dir_text = str(payload.get("work_dir") or "").strip()
    safe_roots = tuple(Path(item) for item in payload.get("safe_roots") or [])
    return RunContext(
        project_root=project_root,
        backend_root=Path(backend_root_text) if backend_root_text else None,
        work_dir=Path(work_dir_text) if work_dir_text else None,
        safe_roots=safe_roots,
        runtime_id=str(payload.get("runtime_id") or ""),
        env={str(k): str(v) for k, v in dict(payload.get("env") or {}).items()},
        metadata=dict(payload.get("metadata") or {}),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
