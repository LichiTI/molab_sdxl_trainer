"""Read-only entrypoint checks for Plugin SDK review surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


FindPluginDir = Callable[[str], Path | None]


def build_runner_entrypoint_policy_summary(
    *,
    registration: dict[str, Any] | None,
    find_plugin_dir: FindPluginDir | None = None,
) -> dict[str, Any]:
    """Return static entrypoint diagnostics without importing plugin code."""

    if not registration:
        return _empty_summary(available=False)
    plugin_id = str(registration.get("plugin_id") or "").strip()
    entrypoint = str(registration.get("entrypoint") or "").strip()
    entry_file, function_name = split_entrypoint(entrypoint)
    if not entry_file or not function_name:
        return _empty_summary(
            available=False,
            entrypoint=entrypoint,
            entry_file=entry_file,
            function_name=function_name,
            error="Plugin runner entrypoint must use file.py:function format.",
        )
    if not plugin_id or find_plugin_dir is None:
        return _empty_summary(
            available=bool(entrypoint),
            entrypoint=entrypoint,
            entry_file=entry_file,
            function_name=function_name,
        )
    plugin_dir = find_plugin_dir(plugin_id)
    if plugin_dir is None:
        return _empty_summary(
            available=False,
            entrypoint=entrypoint,
            entry_file=entry_file,
            function_name=function_name,
            error=f"Plugin is not available: {plugin_id}",
        )
    path = plugin_dir / entry_file
    if not _is_relative_to(path, plugin_dir):
        return _empty_summary(
            available=False,
            entrypoint=entrypoint,
            entry_file=entry_file,
            function_name=function_name,
            error=f"Plugin runner entry file must stay inside the plugin directory: {entry_file}",
        )
    exists = path.is_file()
    warnings = []
    if not exists:
        warnings.append(
            {
                "code": "plugin.entrypoint_file_missing",
                "message": f"Plugin runner entry file does not exist: {entry_file}",
            }
        )
    return {
        "schema": "plugin-entrypoint-policy-summary-v1",
        "available": exists,
        "entrypoint": entrypoint,
        "entry_file": entry_file,
        "function_name": function_name,
        "file_exists": exists,
        "inside_plugin_dir": True,
        "warnings": warnings,
    }


def split_entrypoint(entrypoint: str) -> tuple[str, str]:
    entry_file, separator, function_name = str(entrypoint or "").strip().partition(":")
    if not separator:
        return entry_file.strip(), ""
    return entry_file.strip(), function_name.strip()


def _empty_summary(
    *,
    available: bool,
    entrypoint: str = "",
    entry_file: str = "",
    function_name: str = "",
    error: str = "",
) -> dict[str, Any]:
    warnings = []
    if error:
        warnings.append({"code": "plugin.entrypoint_policy_unavailable", "message": error})
    return {
        "schema": "plugin-entrypoint-policy-summary-v1",
        "available": available,
        "entrypoint": entrypoint,
        "entry_file": entry_file,
        "function_name": function_name,
        "file_exists": False,
        "inside_plugin_dir": False,
        "warnings": warnings,
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False
