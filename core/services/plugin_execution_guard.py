"""Approval and permission snapshots for plugin SDK runner execution."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


_ELEVATED_PERMISSION_MARKERS = {
    "network",
    "network_access",
    "network_optional",
    "http_client",
    "process",
    "process_execute",
    "process:execute",
    "subprocess",
    "shell",
    "exec",
    "write_project",
    "write_runtime",
    "runtime.install_packages",
}


def _normalize_permission(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def elevated_runner_permissions(permissions: list[str]) -> list[str]:
    """Return runner permissions that must come from persisted plugin approval."""

    elevated: set[str] = set()
    for permission in permissions or []:
        raw = str(permission or "").strip()
        text = _normalize_permission(raw)
        if not text:
            continue
        if text in _ELEVATED_PERMISSION_MARKERS:
            elevated.add(raw)
            continue
        if text.startswith("network") or text.startswith("process"):
            elevated.add(raw)
            continue
        if "subprocess" in text or "download" in text or text.startswith("http"):
            elevated.add(raw)
            continue
    return sorted(elevated)


def build_plugin_identity(plugin_id: str, version: str, package_hash: str, signer: str = "") -> str:
    """Return the approval-store identity for one concrete plugin package."""

    return f"{plugin_id}|{version}|{package_hash}|{signer}"


def build_plugin_runner_identity(
    plugin_id: str,
    version: str,
    package_hash: str,
    signer: str = "",
    *,
    runner_id: str,
    request_schema_id: str,
    entrypoint: str,
) -> str:
    """Return the approval-store identity for one concrete SDK runner."""

    base = build_plugin_identity(plugin_id, version, package_hash, signer)
    return "|".join(
        [
            base,
            "runner",
            _identity_part(runner_id),
            _identity_part(request_schema_id),
            _identity_part(entrypoint),
        ]
    )


def collect_sdk_permission_ids(registrations: Mapping[str, Any], plugin_id: str) -> list[str]:
    """Collect all SDK permission ids declared for a plugin."""

    plugin_id = str(plugin_id or "").strip()
    permissions: set[str] = set()
    for group in ("permissions", "runners", "resource_detectors", "artifact_handlers"):
        for item in registrations.get(group, []) or []:
            if not isinstance(item, Mapping) or str(item.get("plugin_id") or "").strip() != plugin_id:
                continue
            for permission in item.get("permissions") or []:
                text = str(permission or "").strip()
                if text:
                    permissions.add(text)
    return sorted(permissions)


def runner_required_capabilities(
    *,
    manifest: Any | None,
    runner_permissions: list[str],
) -> list[str]:
    """Return capabilities that must be bound to approval before runner use."""

    capabilities = set(str(item) for item in getattr(manifest, "capabilities", ()) or [] if str(item).strip())
    capabilities.update(str(item) for item in runner_permissions or [] if str(item).strip())
    return sorted(capabilities)


def runner_scoped_capabilities(*, runner_permissions: list[str]) -> list[str]:
    """Return capabilities approved for exactly one SDK runner."""

    return sorted(str(item) for item in runner_permissions or [] if str(item).strip())


def build_runner_approval_snapshot(
    *,
    plugin_id: str,
    runner_id: str,
    request_schema_id: str,
    entrypoint: str,
    runner_permissions: list[str],
    context_metadata: Mapping[str, Any] | None,
    approval_store: Any,
    manifest: Any | None,
    package_hash: str,
    developer_mode: bool = False,
) -> dict[str, Any]:
    """Build a serializable permission decision for audit logs and jobs."""

    metadata = dict(context_metadata or {})
    required_permissions = sorted(str(item) for item in runner_permissions or [] if str(item).strip())
    context_permissions = sorted(str(item) for item in metadata.get("plugin_permissions") or [] if str(item).strip())
    context_source = str(metadata.get("plugin_permission_source") or "explicit-context").strip() or "explicit-context"
    elevated_permissions = elevated_runner_permissions(required_permissions)
    signer = getattr(getattr(manifest, "signature", None), "signer", "") if manifest is not None else ""
    version = str(getattr(manifest, "version", "") or "")
    identity = build_plugin_identity(plugin_id, version, package_hash, signer) if manifest is not None and package_hash else ""
    runner_identity = (
        build_plugin_runner_identity(
            plugin_id,
            version,
            package_hash,
            signer,
            runner_id=runner_id,
            request_schema_id=request_schema_id,
            entrypoint=entrypoint,
        )
        if manifest is not None and package_hash
        else ""
    )

    base = {
        "schema": "plugin-runner-approval-snapshot-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plugin_id": plugin_id,
        "runner_id": runner_id,
        "request_schema_id": request_schema_id,
        "entrypoint": entrypoint,
        "plugin_version": version,
        "package_hash": package_hash,
        "signer": signer,
        "approval_key": identity,
        "runner_approval_key": runner_identity,
        "approval_scope": "plugin",
        "approval_fallback_used": False,
        "developer_mode": bool(developer_mode),
        "required_permissions": required_permissions,
        "context_permissions": context_permissions,
        "elevated_permissions": elevated_permissions,
        "granted_permissions": [],
        "missing_permissions": list(required_permissions),
        "permission_source": context_source,
        "approved": False,
        "approval_reason": "permission_missing" if required_permissions else "",
        "approved_by": "",
        "approved_at": "",
    }

    if not required_permissions:
        return {
            **base,
            "approved": True,
            "approval_reason": "",
            "missing_permissions": [],
            "permission_source": "no-runner-permissions",
        }

    if context_permissions:
        elevated = set(elevated_permissions)
        granted = sorted(set(required_permissions).intersection(context_permissions).difference(elevated))
        missing = sorted(set(required_permissions).difference(granted))
        approval_reason = "elevated_permission_requires_approval_store" if elevated else "permission_missing"
        return {
            **base,
            "approved": not missing and not elevated,
            "approval_reason": "" if not missing and not elevated else approval_reason,
            "granted_permissions": granted,
            "missing_permissions": missing,
            "permission_source": context_source,
        }

    if manifest is None or not package_hash:
        return {
            **base,
            "approval_reason": "plugin_identity_unavailable",
            "permission_source": "approval-store",
        }

    required_capabilities = runner_scoped_capabilities(runner_permissions=required_permissions)
    approval = approval_store.check(approval_key=runner_identity, required_capabilities=required_capabilities)
    approved = bool(approval.get("approved"))
    record = find_approval_record(approval_store, runner_identity)
    scope = "runner"
    fallback_used = False
    if not approved and not record:
        plugin_required_capabilities = runner_required_capabilities(manifest=manifest, runner_permissions=required_permissions)
        plugin_approval = approval_store.check(approval_key=identity, required_capabilities=plugin_required_capabilities)
        plugin_record = find_approval_record(approval_store, identity)
        if bool(plugin_approval.get("approved")) and plugin_record:
            approval = plugin_approval
            approved = True
            record = plugin_record
            scope = "plugin"
            fallback_used = True
    approved_capabilities = set(record.get("capabilities") or []) if record else set()
    granted = sorted(set(required_permissions).intersection(approved_capabilities))
    missing = sorted(set(required_permissions).difference(granted))
    if not approved and approval.get("missing"):
        missing = sorted(set(missing).union(str(item) for item in approval.get("missing") or [] if str(item) in required_permissions))
    return {
        **base,
        "approved": approved and not missing,
        "approval_reason": str(approval.get("reason") or ""),
        "granted_permissions": granted if approved else granted,
        "missing_permissions": missing,
        "permission_source": "approval-store",
        "approval_scope": scope,
        "approval_fallback_used": fallback_used,
        "approved_by": str(record.get("approved_by") or "") if record else "",
        "approved_at": str(record.get("approved_at") or "") if record else "",
    }


def find_approval_record(approval_store: Any, approval_key: str) -> dict[str, Any]:
    try:
        records = approval_store.list_records()
    except Exception:
        return {}
    for record in records:
        if isinstance(record, Mapping) and record.get("approval_key") == approval_key:
            return dict(record)
    return {}


def _identity_part(value: object) -> str:
    text = str(value or "").strip().replace("|", "%7C")
    return text
