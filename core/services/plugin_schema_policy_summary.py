"""Read-only request-schema policy summaries for Plugin SDK review surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from backend.core.services.plugin_safe_root_policy import schema_path_intent_info, schema_safe_root_roles


FindPluginDir = Callable[[str], Path | None]


def build_request_schema_policy_summary(
    *,
    schema_registration: dict[str, Any] | None,
    find_plugin_dir: FindPluginDir | None = None,
) -> dict[str, Any]:
    """Return safe-root/intent metadata declared by one request schema."""

    if not schema_registration:
        return _empty_summary(available=False)
    plugin_id = str(schema_registration.get("plugin_id") or "").strip()
    schema_path = str(schema_registration.get("schema_path") or "").strip()
    if not plugin_id or not schema_path or find_plugin_dir is None:
        return _empty_summary(available=bool(schema_registration), schema_path=schema_path)

    plugin_dir = find_plugin_dir(plugin_id)
    if plugin_dir is None:
        return _empty_summary(available=False, schema_path=schema_path, error=f"Plugin is not available: {plugin_id}")
    path = plugin_dir / schema_path
    if not _is_relative_to(path, plugin_dir):
        return _empty_summary(
            available=False,
            schema_path=schema_path,
            error=f"Plugin request schema path must stay inside the plugin directory: {schema_path}",
        )
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _empty_summary(available=False, schema_path=schema_path, error=f"Unable to read plugin request schema: {exc}")
    if not isinstance(schema, dict):
        return _empty_summary(available=False, schema_path=schema_path, error="Plugin request schema must be a JSON object")
    return summarize_schema_policy(schema, schema_path=schema_path)


def summarize_schema_policy(schema: dict[str, Any], *, schema_path: str = "") -> dict[str, Any]:
    rows = list(iter_schema_policy_rows(schema))
    safe_root_roles = sorted({role for row in rows for role in row["safe_root_roles"]})
    path_intents = sorted({row["path_intent"] for row in rows if row["path_intent"]})
    warnings = [row for row in rows if row.get("intent_issue")]
    return {
        "schema": "plugin-request-schema-policy-summary-v1",
        "available": True,
        "schema_path": schema_path,
        "safe_root_roles": safe_root_roles,
        "path_intents": path_intents,
        "path_policy_count": len(rows),
        "warnings": [
            {
                "code": "plugin.schema_path_intent_invalid",
                "message": f"Field {row['field_path']} has invalid path intent: {row['intent_issue']}",
                "field_path": row["field_path"],
            }
            for row in warnings
        ],
        "paths": rows,
    }


def iter_schema_policy_rows(schema: dict[str, Any], *, path: str = "$"):
    roles = schema_safe_root_roles(schema)
    intent, intent_issue = schema_path_intent_info(schema)
    if roles:
        yield {
            "field_path": path,
            "safe_root_roles": list(roles),
            "path_intent": intent,
            "intent_issue": intent_issue or "",
        }

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for field, spec in properties.items():
            if isinstance(spec, dict):
                child_path = str(field) if path == "$" else f"{path}.{field}"
                yield from iter_schema_policy_rows(spec, path=child_path)

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        child_path = "*" if path == "$" else f"{path}.*"
        yield from iter_schema_policy_rows(additional, path=child_path)

    item_spec = schema.get("items")
    if isinstance(item_spec, dict):
        yield from iter_schema_policy_rows(item_spec, path=f"{path}[]")

    for key in ("allOf", "anyOf", "oneOf"):
        specs = schema.get(key)
        if isinstance(specs, list):
            for index, spec in enumerate(specs):
                if isinstance(spec, dict):
                    yield from iter_schema_policy_rows(spec, path=f"{path}.{key}[{index}]")


def _empty_summary(*, available: bool, schema_path: str = "", error: str = "") -> dict[str, Any]:
    warnings = []
    if error:
        warnings.append({"code": "plugin.schema_policy_unavailable", "message": error})
    return {
        "schema": "plugin-request-schema-policy-summary-v1",
        "available": available,
        "schema_path": schema_path,
        "safe_root_roles": [],
        "path_intents": [],
        "path_policy_count": 0,
        "warnings": warnings,
        "paths": [],
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False
