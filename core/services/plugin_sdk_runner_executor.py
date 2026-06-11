"""Execution helper for declared plugin SDK runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import BaseRequest, PlatformIssue, PluginRunnerRegistration, RunContext, RunResult

from .plugin_request_validator import PluginRequestValidator, granted_safe_root_roles
from .plugin_runner_placeholder import PluginSdkExecutionContext
from .plugin_sdk_sandbox_policy import build_plugin_sdk_sandbox_policy
from .plugin_sdk_subprocess_runner import execute_plugin_sdk_runner_subprocess


AuditAppend = Callable[..., Any]
ApprovalSnapshotBuilder = Callable[[PluginRunnerRegistration | None, dict[str, Any] | None], dict[str, Any]]
CollectRegistrations = Callable[[], dict[str, Any]]
FindPluginDir = Callable[[str], Path | None]
ReadManifestPayload = Callable[[str], dict[str, Any]]


def execute_plugin_sdk_runner(
    *,
    registration: PluginRunnerRegistration,
    request: BaseRequest,
    context: RunContext,
    find_plugin_dir: FindPluginDir,
    collect_sdk_registrations: CollectRegistrations,
    approval_snapshot_builder: ApprovalSnapshotBuilder,
    read_raw_manifest_payload: ReadManifestPayload,
    audit_append: AuditAppend,
    manifest_loader: Callable[[Path], Any],
    function_loader: Callable[..., dict[str, Callable[..., Any]]],
) -> RunResult:
    if not bool(context.metadata.get("allow_plugin_execution")):
        audit_append(
            event_type="plugin_runner_denied",
            plugin_id=registration.plugin_id,
            payload={"runner_id": registration.runner_id, "reason": "execution_not_allowed"},
        )
        return RunResult.failure(
            "Plugin runner execution requires allow_plugin_execution context metadata.",
            request_id=request.request_id,
            issues=[
                PlatformIssue(
                    code="plugin.execution_not_allowed",
                    message="Plugin runner execution was requested without an explicit execution gate.",
                    severity="error",
                )
            ],
            data={"plugin_id": registration.plugin_id, "runner_id": registration.runner_id},
        )

    validator = PluginRequestValidator(
        find_plugin_dir=find_plugin_dir,
        collect_sdk_registrations=collect_sdk_registrations,
    )
    schema_issue = validator.validate_schema(registration, request)
    if schema_issue is not None:
        audit_append(
            event_type="plugin_runner_denied",
            plugin_id=registration.plugin_id,
            payload={"runner_id": registration.runner_id, "reason": "schema_validation_failed", "error": schema_issue},
        )
        return RunResult.failure(
            "Plugin runner request schema validation failed.",
            request_id=request.request_id,
            issues=[PlatformIssue(code="plugin.schema_validation_failed", message=schema_issue, severity="error")],
            data={"plugin_id": registration.plugin_id, "runner_id": registration.runner_id},
        )

    path_issue = validator.validate_paths(request, context, registration)
    if path_issue is not None:
        audit_append(
            event_type="plugin_runner_denied",
            plugin_id=registration.plugin_id,
            payload={"runner_id": registration.runner_id, "reason": "unsafe_path", "error": path_issue},
        )
        return RunResult.failure(
            "Plugin runner request path validation failed.",
            request_id=request.request_id,
            issues=[PlatformIssue(code="plugin.path_outside_safe_roots", message=path_issue, severity="error")],
            data={"plugin_id": registration.plugin_id, "runner_id": registration.runner_id},
        )

    safe_root_role_issue = validator.validate_safe_root_roles(registration, context)
    if safe_root_role_issue is not None:
        required_roles = validator.required_safe_root_roles(registration)
        granted_roles = granted_safe_root_roles(context)
        audit_append(
            event_type="plugin_runner_denied",
            plugin_id=registration.plugin_id,
            payload={
                "runner_id": registration.runner_id,
                "reason": "safe_root_role_missing",
                "error": safe_root_role_issue,
                "required_safe_root_roles": required_roles,
                "granted_safe_root_roles": granted_roles,
            },
        )
        return RunResult.failure(
            "Plugin runner safe root role validation failed.",
            request_id=request.request_id,
            issues=[
                PlatformIssue(
                    code="plugin.safe_root_role_missing",
                    message=safe_root_role_issue,
                    severity="error",
                    details={
                        "required_safe_root_roles": required_roles,
                        "granted_safe_root_roles": granted_roles,
                    },
                )
            ],
            data={"plugin_id": registration.plugin_id, "runner_id": registration.runner_id},
        )

    approval_snapshot = approval_snapshot_builder(registration, dict(context.metadata or {}))
    sandbox_policy = build_plugin_sdk_sandbox_policy(registration, dict(context.metadata or {}), approval_snapshot)
    if (
        sandbox_policy.enforce_subprocess_for_elevated
        and sandbox_policy.elevated_permissions
        and sandbox_policy.execution_mode != "subprocess"
    ):
        audit_append(
            event_type="plugin_runner_denied",
            plugin_id=registration.plugin_id,
            payload={
                "runner_id": registration.runner_id,
                "reason": "subprocess_required_for_elevated_permissions",
                "sandbox_policy": sandbox_policy.to_dict(),
                "approval_snapshot": approval_snapshot,
            },
        )
        return RunResult.failure(
            "Plugin runner elevated permissions require subprocess execution.",
            request_id=request.request_id,
            issues=[
                PlatformIssue(
                    code="plugin.subprocess_required_for_elevated_permissions",
                    message="Plugin runner elevated permissions require subprocess execution.",
                    severity="error",
                    details={
                        "elevated_permissions": list(sandbox_policy.elevated_permissions),
                        "sandbox_policy": sandbox_policy.to_dict(),
                    },
                )
            ],
            data={
                "plugin_id": registration.plugin_id,
                "runner_id": registration.runner_id,
                "approval_snapshot": approval_snapshot,
                "sandbox_policy": sandbox_policy.to_dict(),
            },
        )
    granted_permissions = set(approval_snapshot.get("granted_permissions") or [])
    permission_source = str(approval_snapshot.get("permission_source") or "explicit-context").strip()
    missing_permissions = list(approval_snapshot.get("missing_permissions") or [])
    if missing_permissions:
        audit_append(
            event_type="plugin_runner_denied",
            plugin_id=registration.plugin_id,
            payload={
                "runner_id": registration.runner_id,
                "reason": "permission_missing",
                "permission_source": permission_source,
                "missing_permissions": missing_permissions,
                "approval_snapshot": approval_snapshot,
            },
        )
        return RunResult.failure(
            "Plugin runner permissions are missing.",
            request_id=request.request_id,
            issues=[
                PlatformIssue(
                    code="plugin.permission_missing",
                    message="Plugin runner permissions are missing: " + ", ".join(missing_permissions),
                    severity="error",
                    details={"missing_permissions": missing_permissions, "approval_snapshot": approval_snapshot},
                )
            ],
            data={
                "plugin_id": registration.plugin_id,
                "runner_id": registration.runner_id,
                "approval_snapshot": approval_snapshot,
            },
        )

    entry_file, _, function_name = str(registration.entrypoint).partition(":")
    if not entry_file or not function_name:
        return RunResult.failure(
            "Plugin runner entrypoint must use file.py:function format.",
            request_id=request.request_id,
            data={"entrypoint": registration.entrypoint},
        )

    plugin_dir = find_plugin_dir(registration.plugin_id)
    if plugin_dir is None:
        return RunResult.failure(
            f"Plugin is not available: {registration.plugin_id}",
            request_id=request.request_id,
            data={"plugin_id": registration.plugin_id},
        )
    if not read_raw_manifest_payload(registration.plugin_id):
        return RunResult.failure(
            f"Plugin manifest is not available: {registration.plugin_id}",
            request_id=request.request_id,
            data={"plugin_id": registration.plugin_id},
        )

    try:
        manifest_path = plugin_dir / "manifest.json"
        if not manifest_path.is_file():
            manifest_path = plugin_dir / "plugin_manifest.json"
        sdk_context = PluginSdkExecutionContext(context, registration)
        audit_append(
            event_type="plugin_runner_started",
            plugin_id=registration.plugin_id,
            payload={
                "runner_id": registration.runner_id,
                "schema_id": request.schema_id,
                "permission_source": permission_source,
                "permissions": sorted(granted_permissions),
                "approval_snapshot": approval_snapshot,
                "sandbox_policy": sandbox_policy.to_dict(),
            },
        )
        if sandbox_policy.execution_mode == "subprocess":
            raw_result = execute_plugin_sdk_runner_subprocess(
                registration=registration,
                request=request,
                context=context,
                plugin_dir=plugin_dir,
                manifest_path=manifest_path,
                approval_snapshot=approval_snapshot,
                sandbox_policy=sandbox_policy,
            )
        else:
            manifest = manifest_loader(manifest_path)
            functions = function_loader(
                plugin_dir,
                manifest,
                [function_name],
                entry=entry_file,
                permissions=list(approval_snapshot.get("granted_permissions") or []),
            )
            raw_result = functions[function_name](request, sdk_context)
    except Exception as exc:
        audit_append(
            event_type="plugin_runner_failed",
            plugin_id=registration.plugin_id,
            payload={"runner_id": registration.runner_id, "error": str(exc)},
        )
        return RunResult.failure(
            f"Plugin runner failed: {exc}",
            request_id=request.request_id,
            issues=[PlatformIssue(code="plugin.runner_failed", message=str(exc), severity="error")],
            data={"plugin_id": registration.plugin_id, "runner_id": registration.runner_id},
        )

    if isinstance(raw_result, RunResult):
        if not raw_result.request_id:
            raw_result.request_id = request.request_id
        raw_result = sdk_context.attach_report(raw_result)
        _attach_host_controlled_metadata(raw_result, approval_snapshot, sandbox_policy.to_dict())
        _audit_runner_finished(audit_append, registration, raw_result.status, approval_snapshot)
        return raw_result
    if isinstance(raw_result, dict):
        result = RunResult.model_validate({"request_id": request.request_id, **raw_result})
        result = sdk_context.attach_report(result)
        _attach_host_controlled_metadata(result, approval_snapshot, sandbox_policy.to_dict())
        _audit_runner_finished(audit_append, registration, result.status, approval_snapshot)
        return result

    audit_append(
        event_type="plugin_runner_failed",
        plugin_id=registration.plugin_id,
        payload={"runner_id": registration.runner_id, "error": "unsupported_result_type"},
    )
    return RunResult.failure(
        "Plugin runner returned an unsupported result type.",
        request_id=request.request_id,
        data={"plugin_id": registration.plugin_id, "runner_id": registration.runner_id},
    )


def _audit_runner_finished(
    audit_append: AuditAppend,
    registration: PluginRunnerRegistration,
    status: str,
    approval_snapshot: dict[str, Any],
) -> None:
    audit_append(
        event_type="plugin_runner_finished",
        plugin_id=registration.plugin_id,
        payload={
            "runner_id": registration.runner_id,
            "status": status,
            "approval_snapshot": approval_snapshot,
        },
    )


def _attach_host_controlled_metadata(
    result: RunResult,
    approval_snapshot: dict[str, Any],
    sandbox_policy: dict[str, Any],
) -> None:
    result.data["approval_snapshot"] = approval_snapshot
    result.data["sandbox_policy"] = sandbox_policy
