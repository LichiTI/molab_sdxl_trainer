"""
Schema registry — centralised storage and lookup for JSON-Schema documents
loaded from heterogeneous sources.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SchemaEntry:
    """A single schema document stored in the registry.

    Attributes:
        schema_id: Unique identifier (often the ``$id`` keyword).
        version: Version string for the schema itself.
        content: The JSON-Schema dict.
        content_type: MIME-style label (``"application/schema+json"``).
        source: Human-readable origin hint (file path, URL, ...).
        metadata: Arbitrary key-value metadata.
    """

    schema_id: str
    version: str = "1.0.0"
    content: Dict[str, Any] = field(default_factory=dict)
    content_type: str = "application/schema+json"
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.content.get("title", self.schema_id)

    @property
    def description(self) -> str:
        return self.content.get("description", "")


class SchemaSource(ABC):
    """Abstract base for schema loading backends."""

    @abstractmethod
    def load(self) -> List[SchemaEntry]:
        """Load and return all schema entries from this source."""
        ...

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Short label for this source (used in error messages)."""
        ...


class DirectorySchemaSource(SchemaSource):
    """Load JSON-Schema files from a directory.

    By default reads every ``*.json`` file.  Each file must be a valid
    JSON-Schema document; the ``$id`` keyword (or filename stem) becomes
    the ``schema_id``.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        glob_pattern: str = "*.json",
        version: str = "1.0.0",
    ) -> None:
        self._dir = Path(directory)
        self._glob = glob_pattern
        self._version = version

    @property
    def identifier(self) -> str:
        return f"directory:{self._dir}"

    def load(self) -> List[SchemaEntry]:
        entries: List[SchemaEntry] = []
        for path in sorted(self._dir.glob(self._glob)):
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            schema_id = content.get("$id", path.stem)
            entries.append(
                SchemaEntry(
                    schema_id=schema_id,
                    version=self._version,
                    content=content,
                    source=str(path),
                )
            )
        return entries


SchemaTransformer = Callable[[SchemaEntry], SchemaEntry]
"""A callable that transforms one SchemaEntry into another."""


class SchemaRegistry:
    """Central registry that aggregates schemas from multiple sources.

    Supports loading, querying, and transforming schemas.  The registry
    is source-agnostic: anything implementing ``SchemaSource`` can feed it.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, SchemaEntry] = {}
        self._sources: List[SchemaSource] = []
        self._transformers: List[SchemaTransformer] = []

    # -- Source management -----------------------------------------------------

    def add_source(self, source: SchemaSource) -> None:
        """Register a source (does not load immediately)."""
        self._sources.append(source)

    def load_all(self, *, apply_transformers: bool = True) -> int:
        """Load from every registered source.  Returns count of entries added."""
        count = 0
        for source in self._sources:
            for entry in source.load():
                if apply_transformers:
                    entry = self._apply_transformers(entry)
                self._entries[entry.schema_id] = entry
                count += 1
        return count

    def load_source(self, source: SchemaSource, *, apply_transformers: bool = True) -> int:
        """Load from a single ad-hoc source.  Returns count added."""
        count = 0
        for entry in source.load():
            if apply_transformers:
                entry = self._apply_transformers(entry)
            self._entries[entry.schema_id] = entry
            count += 1
        return count

    # -- Transformer management -----------------------------------------------

    def add_transformer(self, transformer: SchemaTransformer) -> None:
        """Register a post-load transformer applied to every entry."""
        self._transformers.append(transformer)

    def _apply_transformers(self, entry: SchemaEntry) -> SchemaEntry:
        for tx in self._transformers:
            entry = tx(entry)
        return entry

    # -- Query -----------------------------------------------------------------

    def get(self, schema_id: str) -> Optional[SchemaEntry]:
        return self._entries.get(schema_id)

    def __contains__(self, schema_id: str) -> bool:
        return schema_id in self._entries

    def all(self) -> List[SchemaEntry]:
        return list(self._entries.values())

    def ids(self) -> List[str]:
        return list(self._entries.keys())

    def by_version(self, version: str) -> List[SchemaEntry]:
        return [e for e in self._entries.values() if e.version == version]

    def search(self, query: str) -> List[SchemaEntry]:
        """Case-insensitive search across schema_id, title, and description."""
        q = query.lower()
        return [
            e for e in self._entries.values()
            if q in e.schema_id.lower()
            or q in e.title.lower()
            or q in e.description.lower()
        ]

    # -- Mutation --------------------------------------------------------------

    def put(self, entry: SchemaEntry, *, apply_transformers: bool = True) -> None:
        """Insert or replace a single entry."""
        if apply_transformers:
            entry = self._apply_transformers(entry)
        self._entries[entry.schema_id] = entry

    def remove(self, schema_id: str) -> bool:
        """Remove an entry by id.  Returns True if it existed."""
        return self._entries.pop(schema_id, None) is not None

    def clear(self) -> None:
        """Remove all entries (sources and transformers are kept)."""
        self._entries.clear()
