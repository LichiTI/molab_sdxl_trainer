"""Plugin approval persistence store.

Stores approval records as a JSON file.  Each record binds a plugin
identity key (id|version|hash|signer) to a set of approved capabilities.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = "plugin-approvals-v1"


class ApprovalStore:
    """Thread-safe JSON-file approval store."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return []
        if isinstance(data, dict):
            return [r for r in data.get("records", []) if isinstance(r, dict)]
        return []

    def _save(self, records: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump({"schema": _SCHEMA, "records": records}, fh, indent=2)

    def list_records(self) -> list[dict]:
        """Return all approval records."""
        return self._load()

    def grant(
        self,
        *,
        approval_key: str,
        plugin_id: str,
        version: str,
        package_hash: str,
        signer: str,
        capabilities: list[str],
        approved_by: str = "local-user",
    ) -> dict:
        """Grant or update approval for a plugin identity."""
        records = [r for r in self._load() if r.get("approval_key") != approval_key]
        record = {
            "approval_key": approval_key,
            "plugin_id": plugin_id,
            "version": version,
            "package_hash": package_hash,
            "signer": signer,
            "capabilities": sorted(set(capabilities)),
            "approved_by": approved_by,
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
        records.append(record)
        self._save(records)
        return record

    def revoke(self, plugin_id: str, *, all_versions: bool = True) -> int:
        """Revoke approval for a plugin.  Returns count of removed records."""
        records = self._load()
        kept: list[dict] = []
        removed = 0
        for r in records:
            if r.get("plugin_id") == plugin_id and (all_versions or removed == 0):
                removed += 1
            else:
                kept.append(r)
        self._save(kept)
        return removed

    def check(
        self,
        *,
        approval_key: str,
        required_capabilities: list[str],
    ) -> dict:
        """Check whether a plugin identity has all required capabilities approved."""
        records = self._load()
        matched = None
        for r in records:
            if r.get("approval_key") == approval_key:
                matched = r
                break
        if matched is None:
            return {"approved": False, "reason": "no_record", "missing": sorted(required_capabilities)}
        approved = set(matched.get("capabilities", []))
        missing = sorted(c for c in required_capabilities if c not in approved)
        return {
            "approved": not missing,
            "reason": "" if not missing else "capability_not_approved",
            "missing": missing,
        }
