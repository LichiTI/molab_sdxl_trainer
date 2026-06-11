"""Registry and status adapters for declarative plugin SDK runners."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from backend.core.contracts import PluginRunnerRegistration, RunnerRegistry

from .plugin_entrypoint_policy import build_runner_entrypoint_policy_summary
from .plugin_runner_placeholder import PluginRunnerPlaceholder
from .plugin_schema_policy_summary import build_request_schema_policy_summary
from .plugin_sdk_review import build_runner_permission_review
from .plugin_sdk_sandbox_policy import build_plugin_sdk_sandbox_policy

logger = logging.getLogger(__name__)


ApprovalSnapshotBuilder = Callable[[PluginRunnerRegistration], dict[str, Any]]
FindPluginDir = Callable[[str], Path | None]


def build_sdk_runner_registry_from_registrations(
    registrations: Mapping[str, Any],
    *,
    runtime: Any | None = None,
    execution_enabled: bool = False,
) -> RunnerRegistry:
    """Build a registry from manifest-declared runner registrations."""

    registry = RunnerRegistry()
    for item in registrations.get("runners", []) or []:
        try:
            registration = PluginRunnerRegistration.model_validate(item)
            registry.register(
                PluginRunnerPlaceholder(
                    registration,
                    runtime=runtime,
                    execution_enabled=execution_enabled,
                )
            )
        except Exception:
            logger.debug("Skipping invalid plugin runner registration", exc_info=True)
    return registry


def build_sdk_status_payload(
    *,
    registrations: Mapping[str, Any],
    discovery_capabilities: list[dict[str, Any]],
    execution_capabilities: list[dict[str, Any]],
    approval_snapshot_builder: ApprovalSnapshotBuilder,
    find_plugin_dir: FindPluginDir | None = None,
    sandbox_context_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build UI-facing SDK status without owning runtime lifecycle state."""

    execution_by_id = {item.get("runner_id"): item for item in execution_capabilities}
    runners: list[dict[str, Any]] = []
    for item in discovery_capabilities:
        runner_id = str(item.get("runner_id") or "")
        executable = execution_by_id.get(runner_id) or {}
        metadata = dict(item.get("metadata") or {})
        registration_permissions = list(item.get("permissions") or [])
        registration = PluginRunnerRegistration(
            plugin_id=str(metadata.get("plugin_id") or ""),
            runner_id=str(metadata.get("plugin_runner_id") or "approval-probe"),
            request_schema_id=str((item.get("schema_ids") or ["approval.probe"])[0] or "approval.probe"),
            entrypoint=str(metadata.get("entrypoint") or "plugin.py:probe"),
            permissions=registration_permissions,
        )
        approval_snapshot = approval_snapshot_builder(registration)
        approval_ready = bool(approval_snapshot.get("approved"))
        request_schema_id = str((item.get("schema_ids") or [""])[0] or "")
        schema_registration = _find_request_schema(
            list(registrations.get("request_schemas") or []),
            str(metadata.get("plugin_id") or ""),
            request_schema_id,
        )
        schema_policy_summary = build_request_schema_policy_summary(
            schema_registration=schema_registration,
            find_plugin_dir=find_plugin_dir,
        )
        entrypoint_policy_summary = build_runner_entrypoint_policy_summary(
            registration=registration.model_dump(mode="json"),
            find_plugin_dir=find_plugin_dir,
        )
        sandbox_policy_preview = build_plugin_sdk_sandbox_policy(
            registration,
            dict(sandbox_context_metadata or {}),
            approval_snapshot,
        ).to_dict()
        review = build_runner_permission_review(
            runner_capability=item,
            execution_capability=executable,
            request_schemas=list(registrations.get("request_schemas") or []),
            permission_requests=list(registrations.get("permissions") or []),
            approval_ready=approval_ready,
            schema_policy_summary=schema_policy_summary,
            entrypoint_policy_summary=entrypoint_policy_summary,
            sandbox_policy_preview=sandbox_policy_preview,
        )
        runners.append(
            {
                **item,
                "execution_capability": executable,
                "execution_available": bool((executable.get("metadata") or {}).get("execution_enabled")),
                "approval_ready": approval_ready,
                "approval_required": bool(registration_permissions),
                "approval_snapshot": approval_snapshot,
                "permission_review": review,
            }
        )
    return {**dict(registrations), "runner_capabilities": runners}


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
