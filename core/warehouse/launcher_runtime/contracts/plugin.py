"""Plugin manifest contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PluginManifest:
    """Schema for a plugin.json manifest file."""

    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    capabilities: tuple[str, ...] = ()
    hooks: tuple[str, ...] = ()
    enabled_by_default: bool = True


@dataclass
class PluginInfo:
    """Resolved plugin state (manifest + runtime toggle)."""

    manifest: PluginManifest
    enabled: bool = True
    path: str = ""

    @property
    def id(self) -> str:
        return self.manifest.id
