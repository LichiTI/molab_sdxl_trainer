"""Validation helpers for plugin SDK runner requests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from backend.core.contracts import BaseRequest, RunContext
from backend.core.contracts.plugin_sdk import PluginRunnerRegistration
from backend.core.services.plugin_safe_root_policy import (
    granted_safe_root_roles,
    iter_schema_safe_root_constraints,
    iter_schema_safe_root_values,
    path_intent_actions,
    path_is_safe_for_any_role,
    path_permission_candidates,
    path_permission_matches,
    safe_root_paths_by_role,
    safe_root_roles_for_path,
    schema_path_intent,
    schema_path_intent_info,
    schema_safe_root_roles,
    validate_path_intent_permissions,
    validate_schema_safe_root_paths,
)


class PluginRequestValidator:
    """Validate plugin runner requests without coupling logic to PluginRuntime."""

    def __init__(
        self,
        *,
        find_plugin_dir: Callable[[str], Path | None],
        collect_sdk_registrations: Callable[[], dict[str, Any]],
    ) -> None:
        self._find_plugin_dir = find_plugin_dir
        self._collect_sdk_registrations = collect_sdk_registrations

    def validate_schema(self, registration: PluginRunnerRegistration, request: BaseRequest) -> str | None:
        schema, issue = self._load_schema(registration)
        if issue:
            return issue
        if schema is None:
            return None
        return validate_json_schema_subset(request.model_dump(mode="json"), schema)

    def validate_paths(
        self,
        request: BaseRequest,
        context: RunContext,
        registration: PluginRunnerRegistration | None = None,
    ) -> str | None:
        payload = request.model_dump(mode="json")
        if registration is not None:
            schema, issue = self._load_schema(registration)
            if issue:
                return issue
            if schema is not None:
                schema_issue = validate_schema_safe_root_paths(
                    payload,
                    schema,
                    context,
                    runner_permissions=registration.permissions,
                )
                if schema_issue:
                    return schema_issue
        for field_path, value in iter_path_like_values(payload):
            text = str(value or "").strip()
            if not text:
                continue
            if "://" in text:
                continue
            if not context.is_safe_path(text):
                return f"Field {field_path} points outside safe roots: {text}"
        return None

    def _load_schema(self, registration: PluginRunnerRegistration) -> tuple[dict[str, Any] | None, str | None]:
        plugin_dir = self._find_plugin_dir(registration.plugin_id)
        if plugin_dir is None:
            return None, f"Plugin is not available: {registration.plugin_id}"
        schema_decl = self._schema_declaration(registration)
        if schema_decl is None:
            return None, None
        schema_path_text = str(schema_decl.get("schema_path") or "").strip()
        schema_path = plugin_dir / schema_path_text
        if not _is_relative_to(schema_path, plugin_dir):
            return None, f"Plugin request schema path must stay inside the plugin directory: {schema_path_text}"
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, f"Unable to read plugin request schema: {exc}"
        if not isinstance(schema, dict):
            return None, "Plugin request schema must be a JSON object"
        return schema, None

    def validate_safe_root_roles(self, registration: PluginRunnerRegistration, context: RunContext) -> str | None:
        required_roles = self.required_safe_root_roles(registration)
        if not required_roles:
            return None
        granted_roles = granted_safe_root_roles(context)
        missing = sorted(set(required_roles).difference(granted_roles))
        if missing:
            return "Plugin runner safe root roles are missing: " + ", ".join(missing)
        return None

    def required_safe_root_roles(self, registration: PluginRunnerRegistration) -> list[str]:
        runner_permissions = {str(item or "").strip() for item in registration.permissions or [] if str(item or "").strip()}
        if not runner_permissions:
            return []
        required: set[str] = set()
        registrations = self._collect_sdk_registrations()
        for item in registrations.get("permissions", []) or []:
            if str(item.get("plugin_id") or "").strip() != registration.plugin_id:
                continue
            declared_permissions = {str(value or "").strip() for value in item.get("permissions") or [] if str(value or "").strip()}
            if not runner_permissions.intersection(declared_permissions):
                continue
            required.update(str(role or "").strip() for role in item.get("safe_root_roles") or [] if str(role or "").strip())
        return sorted(required)

    def _schema_declaration(self, registration: PluginRunnerRegistration) -> dict[str, Any] | None:
        registrations = self._collect_sdk_registrations()
        for item in registrations.get("request_schemas", []):
            if item.get("plugin_id") == registration.plugin_id and item.get("request_schema_id") == registration.request_schema_id:
                return item
        return None


def validate_json_schema_subset(payload: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Validate a small dependency-free JSON Schema subset for SDK requests."""

    issue = validate_json_schema_value("$", payload, schema)
    if issue:
        return issue
    return None


def validate_json_schema_value(path: str, value: Any, schema: dict[str, Any]) -> str | None:
    if not isinstance(schema, dict):
        return None

    issue = validate_json_schema_composition(path, value, schema)
    if issue:
        return issue

    if "const" in schema and value != schema.get("const"):
        return f"Field {path} must equal {schema.get('const')!r}"
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        return f"Field {path} must be one of {enum_values!r}"

    expected_type = schema.get("type")
    if expected_type and not json_type_matches(value, expected_type):
        return f"Field {path} must be {expected_type}"

    if isinstance(value, dict):
        issue = validate_json_schema_object(path, value, schema)
        if issue:
            return issue
    elif isinstance(value, list):
        issue = validate_json_schema_array(path, value, schema)
        if issue:
            return issue
    elif isinstance(value, str):
        issue = validate_json_schema_string(path, value, schema)
        if issue:
            return issue
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        issue = validate_json_schema_number(path, value, schema)
        if issue:
            return issue
    return None


def validate_json_schema_object(path: str, payload: dict[str, Any], schema: dict[str, Any]) -> str | None:
    min_properties = schema.get("minProperties")
    if isinstance(min_properties, int) and len(payload) < min_properties:
        return f"Field {path} properties must be >= {min_properties}"
    max_properties = schema.get("maxProperties")
    if isinstance(max_properties, int) and len(payload) > max_properties:
        return f"Field {path} properties must be <= {max_properties}"

    required = schema.get("required")
    if isinstance(required, list):
        for field in required:
            key = str(field)
            required_value = payload.get(key)
            if key not in payload or required_value is None or required_value == "":
                return f"Missing required field: {key}" if path == "$" else f"Missing required field: {path}.{key}"

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for field, spec in properties.items():
            if not isinstance(spec, dict) or field not in payload:
                continue
            child_path = str(field) if path == "$" else f"{path}.{field}"
            issue = validate_json_schema_value(child_path, payload.get(field), spec)
            if issue:
                return issue

    additional = schema.get("additionalProperties")
    if additional is False and isinstance(properties, dict):
        allowed = {str(key) for key in properties.keys()}
        for field in payload:
            if str(field) not in allowed:
                return f"Field {path} has unknown property: {field}" if path != "$" else f"Unknown property: {field}"
    elif isinstance(additional, dict):
        declared = {str(key) for key in properties.keys()} if isinstance(properties, dict) else set()
        for field, item in payload.items():
            if str(field) in declared:
                continue
            child_path = str(field) if path == "$" else f"{path}.{field}"
            issue = validate_json_schema_value(child_path, item, additional)
            if issue:
                return issue
    return None


def validate_json_schema_property(field: str, value: Any, spec: dict[str, Any]) -> str | None:
    return validate_json_schema_value(field, value, spec)


def validate_json_schema_number(field: str, value: int | float, spec: dict[str, Any]) -> str | None:
    minimum = spec.get("minimum")
    if isinstance(minimum, (int, float)) and value < minimum:
        return f"Field {field} must be >= {minimum}"
    maximum = spec.get("maximum")
    if isinstance(maximum, (int, float)) and value > maximum:
        return f"Field {field} must be <= {maximum}"
    exclusive_minimum = spec.get("exclusiveMinimum")
    if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
        return f"Field {field} must be > {exclusive_minimum}"
    exclusive_maximum = spec.get("exclusiveMaximum")
    if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
        return f"Field {field} must be < {exclusive_maximum}"
    return None


def validate_json_schema_string(field: str, value: str, spec: dict[str, Any]) -> str | None:
    min_length = spec.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        return f"Field {field} length must be >= {min_length}"
    max_length = spec.get("maxLength")
    if isinstance(max_length, int) and len(value) > max_length:
        return f"Field {field} length must be <= {max_length}"
    pattern = spec.get("pattern")
    if isinstance(pattern, str):
        try:
            if re.search(pattern, value) is None:
                return f"Field {field} must match pattern {pattern!r}"
        except re.error as exc:
            return f"Field {field} has invalid pattern: {exc}"
    fmt = spec.get("format")
    if isinstance(fmt, str):
        issue = validate_json_schema_string_format(field, value, fmt)
        if issue:
            return issue
    return None


def validate_json_schema_array(field: str, value: list[Any], spec: dict[str, Any]) -> str | None:
    min_items = spec.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        return f"Field {field} items must be >= {min_items}"
    max_items = spec.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        return f"Field {field} items must be <= {max_items}"
    if spec.get("uniqueItems") is True:
        seen: set[str] = set()
        for item in value:
            try:
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            except TypeError:
                key = repr(item)
            if key in seen:
                return f"Field {field} items must be unique"
            seen.add(key)
    item_spec = spec.get("items")
    if isinstance(item_spec, dict):
        for index, item in enumerate(value):
            issue = validate_json_schema_value(f"{field}[{index}]", item, item_spec)
            if issue:
                return issue
    return None


def validate_json_schema_composition(path: str, value: Any, schema: dict[str, Any]) -> str | None:
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for index, item in enumerate(all_of):
            if not isinstance(item, dict):
                continue
            issue = validate_json_schema_value(path, value, item)
            if issue:
                return f"Field {path} failed allOf[{index}]: {issue}"

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        issues: list[str] = []
        for item in any_of:
            if not isinstance(item, dict):
                continue
            issue = validate_json_schema_value(path, value, item)
            if issue is None:
                return None
            issues.append(issue)
        return f"Field {path} must match at least one anyOf schema" + (f": {issues[0]}" if issues else "")

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = 0
        first_issue = ""
        for item in one_of:
            if not isinstance(item, dict):
                continue
            issue = validate_json_schema_value(path, value, item)
            if issue is None:
                matches += 1
            elif not first_issue:
                first_issue = issue
        if matches != 1:
            if matches == 0:
                return f"Field {path} must match exactly one oneOf schema" + (f": {first_issue}" if first_issue else "")
            return f"Field {path} must match exactly one oneOf schema, matched {matches}"

    not_schema = schema.get("not")
    if isinstance(not_schema, dict) and validate_json_schema_value(path, value, not_schema) is None:
        return f"Field {path} must not match forbidden schema"
    return None


def validate_json_schema_string_format(field: str, value: str, fmt: str) -> str | None:
    normalized = fmt.strip().lower()
    if normalized == "uri":
        parsed = urlparse(value)
        if not parsed.scheme:
            return f"Field {field} must match format 'uri'"
    elif normalized == "uri-reference":
        return None
    elif normalized == "email":
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value):
            return f"Field {field} must match format 'email'"
    elif normalized == "hostname":
        if len(value) > 253 or not re.match(r"^[A-Za-z0-9.-]+$", value) or ".." in value:
            return f"Field {field} must match format 'hostname'"
    elif normalized in {"date-time", "date"}:
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$" if normalized == "date-time" else r"^\d{4}-\d{2}-\d{2}$"
        if not re.match(pattern, value):
            return f"Field {field} must match format {normalized!r}"
    return None


def json_type_matches(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(json_type_matches(value, item) for item in expected_type)
    expected = str(expected_type)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def iter_path_like_values(value: Any, *, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            lowered = key_text.lower()
            if _is_path_like_key(lowered):
                yield from _iter_values_under_path_key(item, child_prefix)
            elif isinstance(item, (dict, list)):
                yield from iter_path_like_values(item, prefix=child_prefix)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            if isinstance(item, (dict, list)):
                yield from iter_path_like_values(item, prefix=child_prefix)


def _iter_values_under_path_key(value: Any, prefix: str):
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            if isinstance(item, str):
                yield child_prefix, item
            elif isinstance(item, (dict, list)):
                yield from _iter_values_under_path_key(item, child_prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}"
            if isinstance(item, str):
                yield child_prefix, item
            elif isinstance(item, (dict, list)):
                yield from _iter_values_under_path_key(item, child_prefix)


def _is_path_like_key(lowered_key: str) -> bool:
    return any(token in lowered_key for token in ("path", "dir", "file", "folder", "output"))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False
