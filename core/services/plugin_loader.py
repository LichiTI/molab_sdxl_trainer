"""Sandboxed plugin loader.

Executes untrusted plugin code in a restricted namespace with safe builtins
and a guarded import function that only allows whitelisted modules.
"""

from __future__ import annotations

import ast
import builtins
import importlib
from pathlib import Path
from typing import Any, Callable, Iterable

from lulynx_plugin_core.manifest import PluginManifest


class PluginLoadError(Exception):
    """Raised when a plugin fails to load in the sandbox."""


# ── Safe builtins whitelist ────────────────────────────────────────────

_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset({
    # Type constructors
    "abs", "bool", "bytes", "complex", "dict", "float", "frozenset",
    "int", "list", "set", "str", "tuple",
    # Collection / iteration
    "enumerate", "filter", "iter", "len", "map", "max", "min", "next",
    "range", "reversed", "round", "sorted", "sum", "zip",
    # Type checks
    "callable", "hasattr", "getattr", "setattr", "isinstance", "issubclass",
    "id", "type", "super", "property", "classmethod", "staticmethod",
    # I/O
    "print", "format", "repr", "ascii", "bin", "hex", "oct",
    # Math
    "divmod", "hash", "pow",
    # Exceptions
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
    "OSError", "ImportError", "NameError",
    # Misc
    "None", "True", "False", "NotImplemented", "Ellipsis",
    "__name__", "__doc__",
})

# ── Safe import whitelist ──────────────────────────────────────────────

_BASE_IMPORT_MODULES: frozenset[str] = frozenset({
    # Standard library — safe, no system access
    "json", "math", "re", "datetime", "collections",
    "itertools", "functools", "dataclasses", "enum", "typing",
    "logging", "hashlib", "base64", "uuid", "copy", "textwrap",
    "string", "abc", "contextlib", "statistics", "operator",
    "numbers", "decimal", "fractions", "secrets", "hmac",
    "__future__",
    "struct", "array", "bisect", "heapq", "queue",
    "pprint",
    # Internal cleanroom packages (allowed)
    "lulynx_plugin_core",
    "lulynx_route_contract",
    "lulynx_compliance",
    "lulynx_trainer_registry",
})

_FILE_WRITE_IMPORT_MODULES: frozenset[str] = frozenset({
    "pathlib", "tempfile", "shutil", "glob", "fnmatch",
})

_NETWORK_IMPORT_MODULES: frozenset[str] = frozenset({
    "http", "urllib", "socket", "ssl", "ipaddress",
})

_PROCESS_IMPORT_MODULES: frozenset[str] = frozenset({
    "subprocess", "multiprocessing",
})

_BLOCKED_BUILTIN_CALLS: frozenset[str] = frozenset({
    "open", "eval", "exec", "compile", "input", "breakpoint", "__import__",
})

_PATH_WRITE_METHOD_NAMES: frozenset[str] = frozenset({
    "chmod", "hardlink_to", "link_to", "lchmod", "mkdir", "rename",
    "replace", "rmdir", "symlink_to", "touch", "unlink", "write_bytes",
    "write_text",
})

_PROCESS_METHOD_NAMES: frozenset[str] = frozenset({
    "call", "check_call", "check_output", "execv", "execve", "fork",
    "popen", "Popen", "run", "spawn", "spawnl", "spawnle", "spawnlp",
    "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "system",
})


def _normalize_permission(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(":", "_").replace(".", "_")


def _has_file_write_permission(permissions: Iterable[object]) -> bool:
    for permission in permissions:
        text = _normalize_permission(permission)
        if text in {"file_write", "filesystem_write", "write_file", "write_files", "write_output", "write_project"}:
            return True
        if text.startswith("write_") or text.endswith("_write"):
            return True
    return False


def _has_network_permission(permissions: Iterable[object]) -> bool:
    for permission in permissions:
        text = _normalize_permission(permission)
        if text in {"network", "network_optional", "network_access", "http_client"}:
            return True
        if text.startswith("network") or "download" in text or text.startswith("http"):
            return True
    return False


def _has_process_permission(permissions: Iterable[object]) -> bool:
    for permission in permissions:
        text = _normalize_permission(permission)
        if text in {"process", "process_execute", "subprocess", "shell", "exec"}:
            return True
        if "subprocess" in text or text.startswith("process"):
            return True
    return False


def allowed_import_modules(permissions: Iterable[object] | None = None) -> frozenset[str]:
    """Return top-level modules available for a sandboxed plugin load."""

    permission_list = list(permissions or [])
    modules = set(_BASE_IMPORT_MODULES)
    if _has_file_write_permission(permission_list):
        modules.update(_FILE_WRITE_IMPORT_MODULES)
    if _has_network_permission(permission_list):
        modules.update(_NETWORK_IMPORT_MODULES)
    if _has_process_permission(permission_list):
        modules.update(_PROCESS_IMPORT_MODULES)
    return frozenset(modules)


def import_gate_requirements(module_name: str) -> list[str]:
    top_level = module_name.split(".")[0]
    requirements: list[str] = []
    if top_level in _FILE_WRITE_IMPORT_MODULES:
        requirements.append("write_output")
    if top_level in _NETWORK_IMPORT_MODULES:
        requirements.append("network")
    if top_level in _PROCESS_IMPORT_MODULES:
        requirements.append("process:execute")
    return requirements


def validate_plugin_source_safety(source: str, *, permissions: Iterable[object] | None = None) -> None:
    """Reject obvious sandbox escapes before executing plugin source."""

    permission_list = list(permissions or [])
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PluginLoadError(f"Plugin source has invalid syntax: {exc}") from exc

    allowed_modules = allowed_import_modules(permission_list)
    process_aliases = _collect_import_aliases(tree, _PROCESS_IMPORT_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            issue = _validate_import_node(node, allowed_modules)
            if issue:
                raise PluginLoadError(issue)
            continue
        if isinstance(node, ast.Call):
            issue = _validate_call_node(node, permission_list, process_aliases=process_aliases)
            if issue:
                raise PluginLoadError(issue)


def _collect_import_aliases(tree: ast.AST, module_roots: frozenset[str]) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level in module_roots:
                    aliases.add(alias.asname or top_level)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level = node.module.split(".")[0]
            if top_level in module_roots:
                for alias in node.names:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _validate_import_node(node: ast.Import | ast.ImportFrom, allowed_modules: frozenset[str]) -> str | None:
    names: list[str] = []
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif node.module:
        names = [node.module]
    for name in names:
        top_level = name.split(".")[0]
        if top_level in allowed_modules:
            continue
        requirements = import_gate_requirements(name)
        hint = f" Requires one of permissions: {', '.join(requirements)}." if requirements else ""
        line = getattr(node, "lineno", "?")
        return f"Plugin sandbox blocked import of '{name}' at line {line}.{hint}"
    return None


def _validate_call_node(node: ast.Call, permissions: list[object], *, process_aliases: set[str]) -> str | None:
    line = getattr(node, "lineno", "?")
    if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTIN_CALLS:
        return f"Plugin sandbox blocked call to builtin '{node.func.id}' at line {line}."
    if not isinstance(node.func, ast.Attribute):
        return None
    attr = node.func.attr
    if attr in _PATH_WRITE_METHOD_NAMES and not _has_file_write_permission(permissions):
        return f"Plugin sandbox blocked file write method '{attr}' at line {line}. Requires one of permissions: write_output."
    if attr in _PROCESS_METHOD_NAMES and _attribute_root_name(node.func.value) in process_aliases and not _has_process_permission(permissions):
        return f"Plugin sandbox blocked process method '{attr}' at line {line}. Requires one of permissions: process:execute."
    return None


def _attribute_root_name(node: ast.AST) -> str:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return ""


def _build_restricted_builtins() -> dict[str, Any]:
    """Build a restricted builtins dict containing only safe names."""
    raw = builtins.__dict__
    return {name: raw[name] for name in _SAFE_BUILTIN_NAMES if name in raw}


def _guarded_import_factory(allowed_modules: frozenset[str]) -> Callable[..., Any]:
    """Replacement for __builtins__.__import__ that blocks unsafe modules."""

    def _guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        # Allow submodule imports of whitelisted top-level packages.
        top_level = name.split(".")[0]
        if top_level not in allowed_modules:
            requirements = import_gate_requirements(name)
            hint = f" Requires one of permissions: {', '.join(requirements)}." if requirements else ""
            raise ImportError(
                f"Plugin sandbox blocked import of '{name}'. "
                f"Only modules allowed by declared permissions can be imported.{hint}"
            )
        return importlib.import_module(name)

    return _guarded_import


def _build_sandbox(entry_path: Path, manifest: PluginManifest, permissions: Iterable[object] | None = None) -> dict[str, Any]:
    restricted_builtins = _build_restricted_builtins()
    restricted_builtins["__import__"] = _guarded_import_factory(allowed_import_modules(permissions))
    return {
        "__builtins__": restricted_builtins,
        "__name__": f"plugin_{manifest.plugin_id}",
        "__file__": str(entry_path),
        "__doc__": None,
    }


def _resolve_plugin_entry_path(plugin_dir: Path, entry_name: str) -> Path:
    """Resolve a plugin entry file while keeping it inside the plugin root."""

    root = plugin_dir.resolve()
    entry_text = str(entry_name or "").strip()
    if not entry_text:
        raise PluginLoadError("Plugin entry file cannot be empty.")
    entry_path = (root / entry_text).resolve()
    try:
        entry_path.relative_to(root)
    except ValueError as exc:
        raise PluginLoadError(f"Plugin entry file must stay inside the plugin directory: {entry_text}") from exc
    return entry_path


def load_plugin(
    plugin_dir: Path,
    manifest: PluginManifest,
    *,
    permissions: Iterable[object] | None = None,
) -> dict[str, Callable]:
    """Load a plugin's entry file in a sandboxed namespace.

    Returns a dict of handler_name -> callable extracted from the module
    globals.  Only callables whose names match hook handler declarations
    in the manifest are returned.

    Raises ``PluginLoadError`` on any load failure.
    """
    entry_path = _resolve_plugin_entry_path(plugin_dir, manifest.entry)
    if not entry_path.exists():
        raise PluginLoadError(
            f"Entry file '{manifest.entry}' not found in {plugin_dir}"
        )

    try:
        source = entry_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PluginLoadError(f"Cannot read entry file: {exc}") from exc
    validate_plugin_source_safety(source, permissions=permissions)

    sandbox = _build_sandbox(entry_path, manifest, permissions)

    # Execute in sandbox
    try:
        code = compile(source, str(entry_path), "exec")
        exec(code, sandbox)
    except PluginLoadError:
        raise
    except Exception as exc:
        raise PluginLoadError(
            f"Plugin '{manifest.plugin_id}' failed to execute: {exc}"
        ) from exc

    # Extract handler callables referenced in manifest hooks
    handlers: dict[str, Callable] = {}
    for binding in manifest.hooks:
        handler_name = binding.handler
        obj = sandbox.get(handler_name)
        if obj is None:
            raise PluginLoadError(
                f"Handler '{handler_name}' not found in plugin '{manifest.plugin_id}'"
            )
        if not callable(obj):
            raise PluginLoadError(
                f"'{handler_name}' in plugin '{manifest.plugin_id}' is not callable"
            )
        handlers[handler_name] = obj

    return handlers


def load_plugin_functions(
    plugin_dir: Path,
    manifest: PluginManifest,
    function_names: list[str],
    *,
    entry: str | None = None,
    permissions: Iterable[object] | None = None,
) -> dict[str, Callable]:
    """Load selected plugin functions in the same sandbox used for hooks."""

    entry_name = entry or manifest.entry
    entry_path = _resolve_plugin_entry_path(plugin_dir, entry_name)
    if not entry_path.exists():
        raise PluginLoadError(
            f"Entry file '{entry_name}' not found in {plugin_dir}"
        )

    try:
        source = entry_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PluginLoadError(f"Cannot read entry file: {exc}") from exc
    validate_plugin_source_safety(source, permissions=permissions)

    sandbox = _build_sandbox(entry_path, manifest, permissions)

    try:
        code = compile(source, str(entry_path), "exec")
        exec(code, sandbox)
    except Exception as exc:
        raise PluginLoadError(
            f"Plugin '{manifest.plugin_id}' failed to execute: {exc}"
        ) from exc

    functions: dict[str, Callable] = {}
    for function_name in function_names:
        obj = sandbox.get(function_name)
        if obj is None:
            raise PluginLoadError(
                f"Function '{function_name}' not found in plugin '{manifest.plugin_id}'"
            )
        if not callable(obj):
            raise PluginLoadError(
                f"'{function_name}' in plugin '{manifest.plugin_id}' is not callable"
            )
        functions[function_name] = obj
    return functions

