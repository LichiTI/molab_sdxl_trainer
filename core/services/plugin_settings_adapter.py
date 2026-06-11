"""Adapters for plugin settings schema and persisted values."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PYTORCH_OPTIMIZER_PLUGIN_ID = "lulynx.optimizer.pytorch_optimizer"


def discover_pytorch_optimizer_names(plugin_root: Path) -> list[str]:
    """Read optimizer names from the optional pytorch-optimizer plugin package."""

    init_file = plugin_root / "pytorch_optimizer-main" / "pytorch_optimizer" / "optimizer" / "__init__.py"
    try:
        text = init_file.read_text(encoding="utf-8")
    except Exception:
        return []
    marker = "OPTIMIZER_LIST"
    start = text.find(marker)
    if start < 0:
        return []
    equals = text.find("=", start)
    if equals < 0:
        return []
    start = text.find("[", equals)
    end = text.find("]", start)
    if start < 0 or end < 0:
        return []
    names: list[str] = []
    for raw in text[start + 1:end].splitlines():
        item = raw.split("#", 1)[0].strip().rstrip(",")
        if not item or not item.replace("_", "").replace(".", "").isalnum():
            continue
        if item not in names:
            names.append(item)
    return names


def schema_defaults(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract default values from a plugin settings schema."""

    defaults: dict[str, Any] = {}
    for key, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        if "default" in spec:
            default_value = spec.get("default")
            defaults[key] = list(default_value) if isinstance(default_value, list) else default_value
    return defaults


def _coerce_schema_version(value: Any) -> int:
    try:
        version = int(value or 1)
    except (TypeError, ValueError):
        return 1
    return max(version, 1)


def _coerce_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"", "0", "false", "no", "n", "off", "disabled"}:
        return False
    return bool(value)


def _schema_option_values(spec: dict[str, Any]) -> list[str]:
    values: list[str] = []
    raw_options = spec.get("options", [])
    if not isinstance(raw_options, list):
        return values
    for item in raw_options:
        if isinstance(item, dict):
            raw_value = item.get("value")
            if raw_value is None:
                raw_value = item.get("id")
            if raw_value is None:
                raw_value = item.get("key")
            if raw_value is None:
                raw_value = item.get("label")
        else:
            raw_value = item
        if raw_value is None:
            continue
        value = str(raw_value)
        if value and value not in values:
            values.append(value)
    return values


def _runtime_ids_from_manifest(manifest_payload: dict[str, Any]) -> list[str]:
    offline = manifest_payload.get("offline_runtime_pack") if isinstance(manifest_payload, dict) else None
    raw_ids = offline.get("runtime_ids") if isinstance(offline, dict) else []
    if not isinstance(raw_ids, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_ids:
        runtime_id = str(item or "").strip()
        if not runtime_id or runtime_id in seen:
            continue
        seen.add(runtime_id)
        result.append(runtime_id)
    return result


def _offline_pack_roots(manifest_payload: dict[str, Any], plugin_dir: Path | None) -> tuple[Path | None, Path | None]:
    if plugin_dir is None:
        return None, None
    offline = manifest_payload.get("offline_runtime_pack") if isinstance(manifest_payload, dict) else None
    offline = offline if isinstance(offline, dict) else {}
    locks_root = plugin_dir / Path(str(offline.get("locks_root") or "locks"))
    resources_root = plugin_dir / Path(str(offline.get("resources_root") or "resources"))
    return locks_root, resources_root


def _offline_pack_complete_count(manifest_payload: dict[str, Any], plugin_dir: Path | None) -> int:
    runtime_ids = _runtime_ids_from_manifest(manifest_payload)
    locks_root, resources_root = _offline_pack_roots(manifest_payload, plugin_dir)
    if locks_root is None or resources_root is None:
        return 0
    complete = 0
    for runtime_id in runtime_ids:
        runtime_lock_root = locks_root / runtime_id
        if (
            (runtime_lock_root / "requirements.lock.json").is_file()
            and (runtime_lock_root / "requirements.lock.txt").is_file()
            and resources_root.is_dir()
        ):
            complete += 1
    return complete


def _offline_pack_wheel_dir_count(manifest_payload: dict[str, Any], plugin_dir: Path | None) -> int:
    runtime_ids = _runtime_ids_from_manifest(manifest_payload)
    locks_root, _ = _offline_pack_roots(manifest_payload, plugin_dir)
    if locks_root is None:
        return 0
    wheel_dirs: set[str] = set()
    for runtime_id in runtime_ids:
        lock_json = locks_root / runtime_id / "requirements.lock.json"
        try:
            import json

            metadata = json.loads(lock_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_dirs = metadata.get("wheel_dirs") if isinstance(metadata, dict) else []
        if isinstance(raw_dirs, list):
            for item in raw_dirs:
                text = str(item or "").strip().replace("\\", "/")
                if text:
                    wheel_dirs.add(text)
    return len(wheel_dirs)


def resolve_pytorch_optimizer_visible_names(plugin_root: Path, values: dict[str, Any]) -> list[str]:
    """Return optimizer names exposed by the plugin settings."""

    all_names = discover_pytorch_optimizer_names(plugin_root)
    if not all_names:
        return []
    expose_all = _coerce_boolean(values.get("expose_all_optimizers"))
    raw_selected = values.get("visible_optimizers")
    selected = [str(item) for item in raw_selected] if isinstance(raw_selected, list) else []
    selected = [name for name in selected if name in all_names]
    candidates = all_names if expose_all else selected
    if not candidates:
        candidates = all_names[:6]
    try:
        limit = int(values.get("max_visible_optimizers") or len(candidates))
    except (TypeError, ValueError):
        limit = len(candidates)
    limit = max(1, min(limit, len(candidates)))
    return candidates[:limit]


def _dynamic_value(
    key: str,
    *,
    plugin_root: Path,
    manifest_payload: dict[str, Any],
    values: dict[str, Any],
    plugin_dir: Path | None,
) -> Any:
    if key == "pytorch_optimizer_summary":
        all_count = len(discover_pytorch_optimizer_names(plugin_root))
        visible_count = len(resolve_pytorch_optimizer_visible_names(plugin_root, values))
        mode = "全部候选" if _coerce_boolean(values.get("expose_all_optimizers")) else "手动勾选"
        return f"已发现 {all_count} 个 pytorch-optimizer 优化器；当前以{mode}模式向后端/UI 暴露 {visible_count} 个。"
    if key == "pytorch_optimizer_visible_count":
        return str(len(resolve_pytorch_optimizer_visible_names(plugin_root, values)))
    if key == "pytorch_optimizer_visible_names":
        return resolve_pytorch_optimizer_visible_names(plugin_root, values)
    if key == "offline_runtime_pack_summary":
        runtime_ids = _runtime_ids_from_manifest(manifest_payload)
        complete = _offline_pack_complete_count(manifest_payload, plugin_dir)
        return f"包含 {len(runtime_ids)} 条运行环境线路，其中 {complete} 条锁文件和资源目录完整。"
    if key == "offline_runtime_pack_routes":
        return _runtime_ids_from_manifest(manifest_payload)
    if key == "offline_runtime_pack_resource_summary":
        locks_root, resources_root = _offline_pack_roots(manifest_payload, plugin_dir)
        wheel_dir_count = _offline_pack_wheel_dir_count(manifest_payload, plugin_dir)
        locks_state = "存在" if locks_root is not None and locks_root.is_dir() else "缺失"
        resources_state = "存在" if resources_root is not None and resources_root.is_dir() else "缺失"
        return f"locks 目录{locks_state}，resources 目录{resources_state}，声明的 wheel 资源目录 {wheel_dir_count} 个。"
    return None


def extract_settings_schema(manifest_payload: dict[str, Any]) -> dict[str, Any]:
    """Return the settings schema declared by a plugin manifest."""

    if not isinstance(manifest_payload, dict):
        return {}
    raw_schema = manifest_payload.get("settings_schema")
    if isinstance(raw_schema, dict):
        return raw_schema
    panel = manifest_payload.get("advanced_panel")
    if isinstance(panel, dict) and isinstance(panel.get("settings_schema"), dict):
        return panel["settings_schema"]
    return {}


def build_plugin_settings_panel_payload(manifest_payload: dict[str, Any]) -> dict[str, Any]:
    """Build lightweight panel metadata for plugin inventory/runtime lists."""

    if not isinstance(manifest_payload, dict):
        return {"available": False}

    panel = manifest_payload.get("advanced_panel")
    panel = dict(panel) if isinstance(panel, dict) else {}
    schema = extract_settings_schema(manifest_payload)
    custom = panel.get("custom") if isinstance(panel.get("custom"), dict) else {}
    custom_entry = str((custom or {}).get("entry") or panel.get("custom_entry") or "").strip()
    has_schema = bool(schema)
    has_custom = bool(custom_entry)

    return {
        "available": has_schema or has_custom,
        "schema_version": _coerce_schema_version(panel.get("schema_version")),
        "mode": str(panel.get("mode") or ("schema" if has_schema else "custom" if has_custom else "none")),
        "title": str(panel.get("title") or panel.get("label") or ""),
        "description": str(panel.get("description") or ""),
        "settings_schema": schema,
        "custom": {"entry": custom_entry} if has_custom else {},
    }


def hydrate_settings_schema(
    schema: dict[str, Any],
    plugin_root: Path,
    *,
    manifest_payload: dict[str, Any] | None = None,
    values: dict[str, Any] | None = None,
    plugin_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve lightweight dynamic options for a settings schema."""

    hydrated: dict[str, Any] = {}
    manifest = manifest_payload if isinstance(manifest_payload, dict) else {}
    hydrated_values = values if isinstance(values, dict) else {}
    for key, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        item = dict(spec)
        if item.get("dynamic_options") == "pytorch_optimizer_names":
            item["options"] = discover_pytorch_optimizer_names(plugin_root)
        dynamic_value = str(item.get("dynamic_value") or "").strip()
        if dynamic_value:
            resolved = _dynamic_value(
                dynamic_value,
                plugin_root=plugin_root,
                manifest_payload=manifest,
                values=hydrated_values,
                plugin_dir=plugin_dir,
            )
            if resolved is not None:
                item["value"] = resolved
        hydrated[key] = item
    return hydrated


def build_plugin_settings_payload(
    *,
    plugin_id: str,
    manifest_payload: dict[str, Any],
    all_settings: dict[str, Any],
    plugin_root: Path,
    plugin_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the public settings response for a plugin."""

    schema = extract_settings_schema(manifest_payload)
    defaults = schema_defaults(schema)
    saved = all_settings.get("plugins", {}).get(plugin_id, {})
    if not isinstance(saved, dict):
        saved = {}
    values = {**defaults, **saved}
    return {
        "plugin_id": plugin_id,
        "schema": hydrate_settings_schema(
            schema,
            plugin_root,
            manifest_payload=manifest_payload,
            values=values,
            plugin_dir=plugin_dir,
        ),
        "defaults": defaults,
        "values": values,
    }


def sanitize_plugin_settings(schema: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Keep only schema-backed settings and coerce values to supported types."""

    sanitized: dict[str, Any] = {}
    for key, spec in schema.items():
        if key not in settings or not isinstance(spec, dict):
            continue
        if _coerce_boolean(spec.get("readonly")):
            continue
        value = settings.get(key)
        typ = str(spec.get("type") or "string").lower()
        if typ == "boolean":
            sanitized[key] = _coerce_boolean(value)
        elif typ in {"integer", "int"}:
            try:
                coerced = int(value)
            except (TypeError, ValueError):
                coerced = int(spec.get("default") or 0)
            if "min" in spec:
                min_value = spec.get("min")
                coerced = max(coerced, int(min_value if min_value is not None else coerced))
            if "max" in spec:
                max_value = spec.get("max")
                coerced = min(coerced, int(max_value if max_value is not None else coerced))
            sanitized[key] = coerced
        elif typ in {"number", "float"}:
            try:
                coerced_float = float(value)
            except (TypeError, ValueError):
                coerced_float = float(spec.get("default") or 0.0)
            if "min" in spec:
                min_value = spec.get("min")
                coerced_float = max(coerced_float, float(min_value if min_value is not None else coerced_float))
            if "max" in spec:
                max_value = spec.get("max")
                coerced_float = min(coerced_float, float(max_value if max_value is not None else coerced_float))
            sanitized[key] = coerced_float
        elif typ == "select":
            option_values = _schema_option_values(spec)
            allowed = set(option_values)
            text = str(value or "")
            if allowed and text not in allowed:
                default_text = str(spec.get("default") or "")
                text = default_text if default_text in allowed else option_values[0]
            sanitized[key] = text
        elif typ == "multiselect":
            allowed = set(_schema_option_values(spec))
            incoming = value if isinstance(value, list) else []
            sanitized[key] = [str(item) for item in incoming if not allowed or str(item) in allowed]
        else:
            sanitized[key] = str(value or "")
    return sanitized
