"""Plugin SDK review helpers for UI-facing permission summaries."""

from __future__ import annotations

from typing import Any


def build_runner_permission_review(
    *,
    runner_capability: dict[str, Any],
    execution_capability: dict[str, Any] | None = None,
    request_schemas: list[dict[str, Any]] | None = None,
    permission_requests: list[dict[str, Any]] | None = None,
    approval_ready: bool = False,
    schema_policy_summary: dict[str, Any] | None = None,
    entrypoint_policy_summary: dict[str, Any] | None = None,
    sandbox_policy_preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact review payload for one SDK runner.

    The plugin center should not need to infer permission state from several
    registration lists. This helper keeps the UI-facing summary deterministic
    while the raw declarations remain available for developer tooling.
    """

    capability = dict(runner_capability or {})
    execution = dict(execution_capability or {})
    metadata = dict(capability.get("metadata") or {})
    plugin_id = str(metadata.get("plugin_id") or "").strip()
    plugin_runner_id = str(metadata.get("plugin_runner_id") or "").strip()
    request_schema_id = _first_schema_id(capability)
    permissions = [str(item) for item in capability.get("permissions") or [] if str(item).strip()]
    artifact_types = [str(item) for item in metadata.get("artifact_types") or [] if str(item).strip()]
    schema = _find_request_schema(request_schemas or [], plugin_id, request_schema_id)
    declarations = _permission_declarations(permission_requests or [], plugin_id)
    approved = bool(approval_ready) or not permissions

    permission_rows = []
    for permission in permissions:
        declaration = declarations.get(permission, {})
        permission_rows.append(
            {
                "permission": permission,
                "approved": approved,
                "reason": str(declaration.get("reason") or ""),
                "safe_root_roles": list(declaration.get("safe_root_roles") or []),
                "declared_by_plugin": bool(declaration),
            }
        )

    warnings: list[dict[str, str]] = []
    if request_schema_id and not schema:
        warnings.append(
            {
                "code": "plugin.request_schema_missing",
                "message": f"Request schema declaration is missing: {request_schema_id}",
            }
        )
    if permissions and not approved:
        warnings.append(
            {
                "code": "plugin.approval_required",
                "message": "Plugin runner requires approval before execution.",
            }
        )
    policy_summary = dict(schema_policy_summary or _empty_schema_policy_summary())
    for warning in policy_summary.get("warnings") or []:
        if isinstance(warning, dict):
            warnings.append({"code": str(warning.get("code") or "plugin.schema_policy_warning"), "message": str(warning.get("message") or "")})
    entrypoint_policy = dict(entrypoint_policy_summary or _empty_entrypoint_policy_summary())
    for warning in entrypoint_policy.get("warnings") or []:
        if isinstance(warning, dict):
            warnings.append({"code": str(warning.get("code") or "plugin.entrypoint_policy_warning"), "message": str(warning.get("message") or "")})
    sandbox_policy = dict(sandbox_policy_preview or _empty_sandbox_policy_preview())
    if sandbox_policy.get("isolation_warning"):
        warnings.append(
            {
                "code": "plugin.sandbox_isolation_warning",
                "message": f"Plugin runner sandbox warning: {sandbox_policy.get('isolation_warning')}",
            }
        )

    return {
        "plugin_id": plugin_id,
        "runner_id": str(capability.get("runner_id") or ""),
        "plugin_runner_id": plugin_runner_id,
        "request_schema_id": request_schema_id,
        "request_schema": {
            "available": bool(schema),
            "schema_path": str(schema.get("schema_path") or "") if schema else "",
            "title": str(schema.get("title") or "") if schema else "",
            "description": str(schema.get("description") or "") if schema else "",
            "version": int(schema.get("version") or 1) if schema else 1,
        },
        "entrypoint": str(metadata.get("entrypoint") or ""),
        "job_type": str(metadata.get("job_type") or "tool"),
        "artifact_types": artifact_types,
        "schema_policy": policy_summary,
        "entrypoint_policy": entrypoint_policy,
        "sandbox_policy": sandbox_policy,
        "permissions": permission_rows,
        "required_permissions": permissions,
        "missing_approval": [] if approved else permissions,
        "approval_required": bool(permissions),
        "approval_ready": approved,
        "execution_available": bool((execution.get("metadata") or {}).get("execution_enabled")),
        "warnings": warnings,
    }


def _first_schema_id(capability: dict[str, Any]) -> str:
    schema_ids = capability.get("schema_ids") or []
    if not schema_ids:
        return ""
    return str(schema_ids[0] or "").strip()


def _empty_schema_policy_summary() -> dict[str, Any]:
    return {
        "schema": "plugin-request-schema-policy-summary-v1",
        "available": False,
        "schema_path": "",
        "safe_root_roles": [],
        "path_intents": [],
        "path_policy_count": 0,
        "warnings": [],
        "paths": [],
    }


def _empty_entrypoint_policy_summary() -> dict[str, Any]:
    return {
        "schema": "plugin-entrypoint-policy-summary-v1",
        "available": False,
        "entrypoint": "",
        "entry_file": "",
        "function_name": "",
        "file_exists": False,
        "inside_plugin_dir": False,
        "warnings": [],
    }


def _empty_sandbox_policy_preview() -> dict[str, Any]:
    return {
        "schema": "plugin-sdk-sandbox-policy-v1",
        "execution_mode": "in_process",
        "timeout_seconds": 30,
        "env_allowlist": [],
        "blocked_env_allowlist": [],
        "requested_by": "default",
        "elevated_permissions": [],
        "isolation_warning": "",
        "enforce_subprocess_for_elevated": False,
        "default_subprocess_for_untrusted": False,
        "force_subprocess_for_untrusted": False,
        "trust_state": "unknown",
    }


def _find_request_schema(
    request_schemas: list[dict[str, Any]],
    plugin_id: str,
    request_schema_id: str,
) -> dict[str, Any] | None:
    for item in request_schemas:
        if str(item.get("plugin_id") or "").strip() != plugin_id:
            continue
        if str(item.get("request_schema_id") or "").strip() == request_schema_id:
            return item
    return None


def _permission_declarations(permission_requests: list[dict[str, Any]], plugin_id: str) -> dict[str, dict[str, Any]]:
    declarations: dict[str, dict[str, Any]] = {}
    for item in permission_requests:
        if str(item.get("plugin_id") or "").strip() != plugin_id:
            continue
        for permission in item.get("permissions") or []:
            permission_id = str(permission or "").strip()
            if permission_id:
                declarations[permission_id] = item
    return declarations
