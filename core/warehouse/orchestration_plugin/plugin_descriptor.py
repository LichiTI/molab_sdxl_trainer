"""
Plugin descriptor and manifest structures for identifying and packaging plugins.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PluginSignature:
    """Cryptographic identity for a plugin artifact.

    Attributes:
        digest: Hex-encoded hash of the plugin content.
        algorithm: Hash algorithm used (default ``"sha256"``).
        signer: Optional identifier of the signing entity.
        signed_at: Optional ISO-8601 timestamp of signing.
    """

    digest: str
    algorithm: str = "sha256"
    signer: str = ""
    signed_at: str = ""

    @classmethod
    def from_bytes(cls, data: bytes, algorithm: str = "sha256", **kwargs: Any) -> PluginSignature:
        """Compute a signature by hashing *data*."""
        h = hashlib.new(algorithm)
        h.update(data)
        return cls(digest=h.hexdigest(), algorithm=algorithm, **kwargs)

    @classmethod
    def from_file(cls, path: Path | str, algorithm: str = "sha256", **kwargs: Any) -> PluginSignature:
        """Compute a signature by reading and hashing a file."""
        path = Path(path)
        return cls.from_bytes(path.read_bytes(), algorithm=algorithm, **kwargs)


@dataclass
class PluginDescriptor:
    """Full metadata for a single loadable plugin.

    Attributes:
        name: Unique plugin identifier.
        version: Semver string.
        description: Short human summary.
        entry_point: Dotted path or filename of the plugin entry module.
        signature: Integrity information.
        dependencies: Other plugin names required at load time.
        capabilities: Capability names this plugin provides.
        hooks: Hook names this plugin implements.
        config_schema: Optional JSON-Schema dict for plugin configuration.
        metadata: Arbitrary key-value metadata.
        is_system: Whether this is an internal/non-removable plugin.
        min_host_version: Minimum host version required (semver string).
    """

    name: str
    version: str = "0.0.0"
    description: str = ""
    entry_point: str = ""
    signature: Optional[PluginSignature] = None
    dependencies: Sequence[str] = field(default_factory=list)
    capabilities: Sequence[str] = field(default_factory=list)
    hooks: Sequence[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_system: bool = False
    min_host_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        d: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "entry_point": self.entry_point,
            "dependencies": list(self.dependencies),
            "capabilities": list(self.capabilities),
            "hooks": list(self.hooks),
            "is_system": self.is_system,
            "min_host_version": self.min_host_version,
            "metadata": self.metadata,
        }
        if self.signature is not None:
            d["signature"] = {
                "digest": self.signature.digest,
                "algorithm": self.signature.algorithm,
                "signer": self.signature.signer,
                "signed_at": self.signature.signed_at,
            }
        if self.config_schema is not None:
            d["config_schema"] = self.config_schema
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginDescriptor:
        """Reconstruct from a dict previously produced by ``to_dict``."""
        sig_data = data.get("signature")
        sig = PluginSignature(**sig_data) if sig_data else None
        return cls(
            name=data["name"],
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            entry_point=data.get("entry_point", ""),
            signature=sig,
            dependencies=data.get("dependencies", []),
            capabilities=data.get("capabilities", []),
            hooks=data.get("hooks", []),
            config_schema=data.get("config_schema"),
            metadata=data.get("metadata", {}),
            is_system=data.get("is_system", False),
            min_host_version=data.get("min_host_version", ""),
        )

    def validate(self) -> List[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: List[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.version:
            errors.append("version is required")
        if self.entry_point and not self.entry_point.replace("_", "").replace(".", "").isalnum():
            errors.append(f"entry_point '{self.entry_point}' contains invalid characters")
        return errors


@dataclass
class PluginManifest:
    """Container for a collection of plugin descriptors.

    Represents a manifest file (e.g. a JSON file listing all plugins
    in a plugin directory or distribution).

    Attributes:
        version: Manifest format version.
        plugins: Ordered list of plugin descriptors.
        author: Manifest author.
        license: License identifier.
        metadata: Arbitrary manifest-level metadata.
    """

    version: str = "1.0.0"
    plugins: List[PluginDescriptor] = field(default_factory=list)
    author: str = ""
    license: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, descriptor: PluginDescriptor) -> None:
        self.plugins.append(descriptor)

    def get(self, name: str) -> Optional[PluginDescriptor]:
        for p in self.plugins:
            if p.name == name:
                return p
        return None

    def remove(self, name: str) -> bool:
        """Remove a plugin by name.  Returns True if found."""
        for i, p in enumerate(self.plugins):
            if p.name == name:
                self.plugins.pop(i)
                return True
        return False

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.plugins]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "author": self.author,
            "license": self.license,
            "metadata": self.metadata,
            "plugins": [p.to_dict() for p in self.plugins],
        }

    def save_json(self, path: Path | str, indent: int = 2) -> None:
        """Write the manifest to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginManifest:
        return cls(
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            license=data.get("license", ""),
            metadata=data.get("metadata", {}),
            plugins=[PluginDescriptor.from_dict(p) for p in data.get("plugins", [])],
        )

    @classmethod
    def load_json(cls, path: Path | str) -> PluginManifest:
        """Read a manifest from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
