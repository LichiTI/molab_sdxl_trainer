"""Safe-root and path-intent policy helpers for plugin SDK requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.contracts import RunContext


def validate_schema_safe_root_paths(
    payload: Any,
    schema: dict[str, Any],
    context: RunContext,
    *,
    runner_permissions: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    for field_path, value, roles, intent, intent_issue in iter_schema_safe_root_constraints(payload, schema):
        if intent_issue:
            return f"Field {field_path} has invalid path intent: {intent_issue}"
        text = str(value or "").strip()
        if not text or "://" in text:
            continue
        matched_roles = safe_root_roles_for_path(text, roles, context)
        if not matched_roles:
            return f"Field {field_path} must stay inside safe root role(s) {', '.join(roles)}: {text}"
        intent_permission_issue = validate_path_intent_permissions(
            field_path,
            intent,
            matched_roles,
            runner_permissions,
        )
        if intent_permission_issue:
            return intent_permission_issue
    return None


def iter_schema_safe_root_values(value: Any, schema: dict[str, Any], *, path: str = "$"):
    for field_path, item, roles, _intent, _intent_issue in iter_schema_safe_root_constraints(value, schema, path=path):
        yield field_path, item, roles


def iter_schema_safe_root_constraints(value: Any, schema: dict[str, Any], *, path: str = "$"):
    roles = schema_safe_root_roles(schema)
    intent, intent_issue = schema_path_intent_info(schema)
    if roles:
        if isinstance(value, str):
            yield path, value, roles, intent, intent_issue
        elif isinstance(value, list):
            for index, item in enumerate(value):
                child_path = f"{path}[{index}]"
                if isinstance(item, str):
                    yield child_path, item, roles, intent, intent_issue
                elif isinstance(item, (dict, list)):
                    yield from iter_schema_safe_root_constraints(item, schema, path=child_path)
        elif isinstance(value, dict):
            for key, item in value.items():
                child_path = f"{path}.{key}"
                if isinstance(item, str):
                    yield child_path, item, roles, intent, intent_issue
                elif isinstance(item, (dict, list)):
                    yield from iter_schema_safe_root_constraints(item, schema, path=child_path)

    if isinstance(value, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for field, spec in properties.items():
                if field in value and isinstance(spec, dict):
                    child_path = str(field) if path == "$" else f"{path}.{field}"
                    yield from iter_schema_safe_root_constraints(value.get(field), spec, path=child_path)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            declared = set(properties.keys()) if isinstance(properties, dict) else set()
            for field, item in value.items():
                if field in declared:
                    continue
                child_path = str(field) if path == "$" else f"{path}.{field}"
                yield from iter_schema_safe_root_constraints(item, additional, path=child_path)
    elif isinstance(value, list):
        item_spec = schema.get("items")
        if isinstance(item_spec, dict):
            for index, item in enumerate(value):
                yield from iter_schema_safe_root_constraints(item, item_spec, path=f"{path}[{index}]")

    for key in ("allOf", "anyOf", "oneOf"):
        specs = schema.get(key)
        if isinstance(specs, list):
            for spec in specs:
                if isinstance(spec, dict):
                    yield from iter_schema_safe_root_constraints(value, spec, path=path)


def schema_safe_root_roles(schema: dict[str, Any]) -> tuple[str, ...]:
    value = _first_schema_extension(schema, "x-lulynx-safe-root-roles", "x-lulynx-safe-root-role")
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def schema_path_intent(schema: dict[str, Any]) -> str:
    intent, _issue = schema_path_intent_info(schema)
    return intent


def schema_path_intent_info(schema: dict[str, Any]) -> tuple[str, str | None]:
    value = _first_schema_extension(schema, "x-lulynx-path-intent", "x-lulynx-safe-root-intent")
    if value is None or value == "":
        return "", None
    if not isinstance(value, str):
        return "", "expected one of read, write, readwrite"
    normalized = value.strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    aliases = {
        "input": "read",
        "read": "read",
        "r": "read",
        "output": "write",
        "create": "write",
        "write": "write",
        "w": "write",
        "read_write": "readwrite",
        "readwrite": "readwrite",
        "rw": "readwrite",
    }
    intent = aliases.get(normalized)
    if intent is None:
        return "", "expected one of read, write, readwrite"
    return intent, None


def path_is_safe_for_any_role(path: str, roles: tuple[str, ...], context: RunContext) -> bool:
    return bool(safe_root_roles_for_path(path, roles, context))


def safe_root_roles_for_path(path: str, roles: tuple[str, ...], context: RunContext) -> tuple[str, ...]:
    role_roots = safe_root_paths_by_role(context)
    matched: list[str] = []
    for role in roles:
        roots = role_roots.get(role)
        if roots and _path_is_under_roots(path, roots, context.project_root):
            matched.append(role)
    return tuple(matched)


def validate_path_intent_permissions(
    field_path: str,
    intent: str,
    matched_roles: tuple[str, ...],
    runner_permissions: list[str] | tuple[str, ...] | None,
) -> str | None:
    if not intent or runner_permissions is None:
        return None
    actions = path_intent_actions(intent)
    if not actions:
        return None
    permissions = {_normalize_permission_token(item) for item in runner_permissions if str(item or "").strip()}
    for role in matched_roles:
        if all(path_permission_matches(action, role, permissions) for action in actions):
            return None
    expected = sorted({candidate for role in matched_roles for action in actions for candidate in path_permission_candidates(action, role)})
    return (
        f"Field {field_path} declares {intent} path intent for safe root role(s) "
        f"{', '.join(matched_roles)} but runner permissions do not include required permission(s): "
        f"{', '.join(expected)}"
    )


def path_intent_actions(intent: str) -> tuple[str, ...]:
    if intent == "read":
        return ("read",)
    if intent == "write":
        return ("write",)
    if intent == "readwrite":
        return ("read", "write")
    return ()


def path_permission_matches(action: str, role: str, normalized_permissions: set[str]) -> bool:
    return any(_normalize_permission_token(candidate) in normalized_permissions for candidate in path_permission_candidates(action, role))


def path_permission_candidates(action: str, role: str) -> tuple[str, ...]:
    action = _normalize_permission_token(action)
    role = _normalize_permission_token(role)
    return (
        f"{action}_{role}",
        f"{role}_{action}",
        f"plugin_{action}_{role}",
        f"{action}_safe_root_{role}",
    )


def safe_root_paths_by_role(context: RunContext) -> dict[str, tuple[Path, ...]]:
    metadata = dict(context.metadata or {})
    roots: dict[str, list[Path]] = {}
    for key in ("plugin_safe_root_paths", "safe_root_paths"):
        raw = metadata.get(key)
        if isinstance(raw, dict):
            for role, value in raw.items():
                for item in _path_values(value):
                    roots.setdefault(str(role), []).append(Path(item))
    roots.setdefault("project", []).append(context.project_root)
    if context.backend_root is not None:
        roots.setdefault("backend", []).append(context.backend_root)
    if context.work_dir is not None:
        roots.setdefault("work", []).append(context.work_dir)
    for root in context.safe_roots or ():
        roots.setdefault("safe", []).append(root)
    return {role: tuple(values) for role, values in roots.items()}


def granted_safe_root_roles(context: RunContext) -> list[str]:
    roles: set[str] = set()
    metadata = dict(context.metadata or {})
    for key in ("plugin_safe_root_roles", "safe_root_roles"):
        roles.update(_role_values(metadata.get(key)))
    return sorted(roles)


def _normalize_permission_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(":", "_").replace(".", "_")


def _role_values(value: Any) -> set[str]:
    roles: set[str] = set()
    if isinstance(value, str):
        roles.update(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, dict):
        roles.update(str(key).strip() for key, enabled in value.items() if enabled and str(key).strip())
    elif isinstance(value, (list, tuple, set)):
        roles.update(str(item).strip() for item in value if str(item).strip())
    return roles


def _path_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _path_is_under_roots(path: str, roots: tuple[Path, ...], project_root: Path) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def _first_schema_extension(schema: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in schema:
            return schema.get(key)
    return None
