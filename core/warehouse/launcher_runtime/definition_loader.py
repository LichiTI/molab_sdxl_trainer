"""Load :class:`RuntimeDef` instances from JSON or TOML definition files.

Stdlib-only.  Supports:
- JSON via :mod:`json` (always available).
- TOML via :mod:`tomllib` (Python 3.11+); raises :class:`LoadError` if
  unavailable and TOML is requested.

No installers or external processes are executed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts.runtime import RuntimeCategory, RuntimeDef


# ── public exception ────────────────────────────────────────────


class LoadError(Exception):
    """Raised when a definition file cannot be parsed or validated."""


# ── dict → RuntimeDef ──────────────────────────────────────────

_REQUIRED_KEYS = frozenset({
    "id",
    "name_en",
    "name_zh",
    "description_en",
    "description_zh",
    "category",
})


def runtime_def_from_dict(data: dict[str, Any]) -> RuntimeDef:
    """Convert a plain dict (parsed from JSON/TOML) to a :class:`RuntimeDef`.

    Raises :class:`LoadError` on missing required keys or invalid values.
    """
    missing = _REQUIRED_KEYS - data.keys()
    if missing:
        raise LoadError(f"missing required keys: {', '.join(sorted(missing))}")

    try:
        category = RuntimeCategory(data["category"])
    except ValueError:
        raise LoadError(
            f"unknown category {data['category']!r}; "
            f"expected one of: {', '.join(c.value for c in RuntimeCategory)}"
        )

    raw_env_dir = data.get("env_dir_names", ())
    if isinstance(raw_env_dir, str):
        raw_env_dir = (raw_env_dir,)

    raw_scripts = data.get("install_scripts", ())
    if isinstance(raw_scripts, str):
        raw_scripts = (raw_scripts,)

    return RuntimeDef(
        id=data["id"],
        name_en=data["name_en"],
        name_zh=data["name_zh"],
        description_en=data["description_en"],
        description_zh=data["description_zh"],
        category=category,
        experimental=data.get("experimental", False),
        env_dir_names=tuple(raw_env_dir),
        python_rel_path=data.get("python_rel_path", "python.exe"),
        env_vars=dict(data.get("env_vars", {})),
        install_scripts=tuple(raw_scripts),
        attention_policy_default=data.get("attention_policy_default", "auto"),
        extra=dict(data.get("extra", {})),
    )


# ── JSON loading ────────────────────────────────────────────────


def load_json(source: str | Path) -> list[RuntimeDef]:
    """Load runtime definitions from a JSON file path or JSON string.

    Accepted top-level shapes:
    - ``{"runtimes": [...]}`` envelope
    - A bare ``[...]`` array
    - A single ``{...}`` object (returned as a one-element list)

    Raises :class:`LoadError` on parse or validation errors.
    """
    source_path = Path(source)
    if source_path.is_file():
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LoadError(f"cannot read JSON file {source_path}: {exc}") from exc
    else:
        try:
            raw = json.loads(source)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LoadError(f"cannot parse JSON string: {exc}") from exc

    return _normalize_and_build(raw, "JSON")


# ── TOML loading ────────────────────────────────────────────────


def load_toml(source: str | Path) -> list[RuntimeDef]:
    """Load runtime definitions from a TOML file path or TOML string.

    Accepted top-level shapes are the same as :func:`load_json`.

    Requires Python 3.11+ (``tomllib`` in the standard library).
    Raises :class:`LoadError` if ``tomllib`` is unavailable or parsing fails.
    """
    try:
        import tomllib
    except ImportError as exc:
        raise LoadError(
            "TOML support requires Python 3.11+ (tomllib not available)"
        ) from exc

    source_path = Path(source)
    if source_path.is_file():
        try:
            raw = tomllib.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise LoadError(f"cannot read TOML file {source_path}: {exc}") from exc
    else:
        try:
            raw = tomllib.loads(source)
        except (TypeError, tomllib.TOMLDecodeError) as exc:
            raise LoadError(f"cannot parse TOML string: {exc}") from exc

    return _normalize_and_build(raw, "TOML")


# ── file auto-detection ────────────────────────────────────────

_EXT_MAP: dict[str, Any] = {
    ".json": load_json,
    ".toml": load_toml,
}


def load_file(path: str | Path) -> list[RuntimeDef]:
    """Load runtime definitions, auto-detecting format from file extension.

    Supported extensions: ``.json``, ``.toml``.

    Raises :class:`LoadError` for unknown extensions or load failures.
    """
    p = Path(path)
    loader = _EXT_MAP.get(p.suffix.lower())
    if loader is None:
        supported = ", ".join(sorted(_EXT_MAP))
        raise LoadError(f"unsupported file extension {p.suffix!r}; expected one of: {supported}")
    return loader(p)


# ── already-parsed data ────────────────────────────────────────


def load_dict(data: dict[str, Any] | list[dict[str, Any]]) -> list[RuntimeDef]:
    """Convert an already-parsed dict or list of dicts into :class:`RuntimeDef` instances.

    Accepts the same top-level shapes as :func:`load_json`.
    """
    return _normalize_and_build(data, "dict")


# ── internal helpers ────────────────────────────────────────────


def _normalize_and_build(raw: Any, format_name: str) -> list[RuntimeDef]:
    """Extract a list of raw dicts from various envelope shapes, then build RuntimeDefs."""
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        if "runtimes" in raw:
            items = raw["runtimes"]
            if not isinstance(items, list):
                raise LoadError(
                    f"{format_name}: 'runtimes' key must contain an array, got {type(items).__name__}"
                )
        else:
            items = [raw]
    else:
        raise LoadError(
            f"{format_name}: expected a dict or list at top level, got {type(raw).__name__}"
        )

    defs: list[RuntimeDef] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise LoadError(f"{format_name}: entry {i} is not a dict (got {type(item).__name__})")
        try:
            defs.append(runtime_def_from_dict(item))
        except LoadError as exc:
            raise LoadError(f"{format_name}: entry {i} ({item.get('id', '?')}): {exc}") from exc

    return defs
