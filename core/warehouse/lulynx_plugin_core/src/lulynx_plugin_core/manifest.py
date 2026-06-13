"""Plugin manifest schema and JSON loader.

A manifest is a JSON file (``plugin_manifest.json``) that declares a
plugin's identity, capabilities, hook bindings, and optional signature.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_MANIFEST_FILENAME = "plugin_manifest.json"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+$")


@dataclass(frozen=True)
class PluginHookBinding:
    """Declares that a plugin handles a specific hook event."""

    event: str
    handler: str
    priority: int = 0
    training_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginSignature:
    """Optional signature attestation embedded in a manifest."""

    scheme: str
    signer: str
    signature: str
    content_hash: str
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginManifest:
    """Immutable plugin descriptor loaded from a manifest JSON file."""

    plugin_id: str
    name: str
    version: str
    entry: str
    description: str
    capabilities: tuple[str, ...]
    hooks: tuple[PluginHookBinding, ...]
    min_core_version: str
    signature: PluginSignature | None
    enabled_by_default: bool


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _norm(value: object) -> str:
    return str(value or "").strip()


def _norm_caps(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("capabilities must be a JSON array")
    seen: list[str] = []
    for item in raw:
        text = _norm(item)
        if text and text not in seen:
            seen.append(text)
    return tuple(seen)


def _norm_training_types(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("training_types must be a JSON array when present")
    out: list[str] = []
    for item in raw:
        text = _norm(item).lower()
        if text in {"*", "all"}:
            return ()  # empty = unrestricted
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _norm_hooks(raw: object) -> tuple[PluginHookBinding, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("hooks must be a JSON array")
    bindings: list[PluginHookBinding] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each hooks entry must be a JSON object")
        event = _norm(item.get("event"))
        handler = _norm(item.get("handler"))
        if not event or not handler:
            raise ValueError("hooks entries require non-empty event and handler")
        try:
            priority = int(item.get("priority", 0) or 0)
        except (TypeError, ValueError):
            priority = 0
        bindings.append(PluginHookBinding(
            event=event,
            handler=handler,
            priority=priority,
            training_types=_norm_training_types(item.get("training_types")),
        ))
    return tuple(bindings)


def _norm_signature(raw: object) -> PluginSignature | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("signature must be a JSON object")
    files_raw = raw.get("files", [])
    files: list[str] = []
    if isinstance(files_raw, list):
        for f in files_raw:
            text = _norm(f).replace("\\", "/")
            if text:
                files.append(text)
    return PluginSignature(
        scheme=_norm(raw.get("scheme")) or "none",
        signer=_norm(raw.get("signer")),
        signature=_norm(raw.get("signature")),
        content_hash=_norm(raw.get("hash")),
        files=tuple(files),
    )


def load_manifest(path: Path) -> PluginManifest:
    """Load and validate a plugin manifest from a JSON file.

    Raises ``ValueError`` on schema violations.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")

    plugin_id = _norm(payload.get("id"))
    if not plugin_id:
        raise ValueError("manifest.id is required")
    if not _ID_PATTERN.match(plugin_id):
        raise ValueError(f"manifest.id contains unsupported characters: {plugin_id}")

    version = _norm(payload.get("version"))
    if not version:
        raise ValueError("manifest.version is required")

    return PluginManifest(
        plugin_id=plugin_id,
        name=_norm(payload.get("name")) or plugin_id,
        version=version,
        entry=_norm(payload.get("entry")) or "plugin.py",
        description=_norm(payload.get("description")),
        capabilities=_norm_caps(payload.get("capabilities")),
        hooks=_norm_hooks(payload.get("hooks")),
        min_core_version=_norm(payload.get("min_core_version")),
        signature=_norm_signature(payload.get("signature")),
        enabled_by_default=bool(payload.get("enabled_by_default", True)),
    )


def identity_key(
    *,
    plugin_id: str,
    version: str,
    package_hash: str,
    signer: str,
) -> str:
    """Build a canonical identity key for an approval/trust record."""
    return (
        f"{_norm(plugin_id)}|{_norm(version)}"
        f"|{_norm(package_hash) or 'unhashed'}"
        f"|{_norm(signer) or 'unsigned'}"
    )
