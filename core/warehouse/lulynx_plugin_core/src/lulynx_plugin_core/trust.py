"""Signature verification and trust store.

Provides SHA-256 package hash computation for plugin directories,
signature verification against manifest declarations, and a JSON-file
trust store with deny-lists and allow-lists.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Sequence


_SCHEMA = "plugin-trust-v1"


def compute_package_hash(
    root: Path,
    files: Sequence[str],
) -> tuple[str, list[str], list[str]]:
    """Compute a SHA-256 digest over the listed files in a plugin directory.

    Returns ``(hash_string, normalized_files, missing_files)``.
    Files are sorted alphabetically before hashing.
    """
    normalized = sorted(set(files))
    digest = hashlib.sha256()
    missing: list[str] = []
    for rel in normalized:
        full = (root / rel).resolve()
        try:
            full.relative_to(root.resolve())
        except ValueError:
            missing.append(rel)
            continue
        if not full.exists() or not full.is_file():
            missing.append(rel)
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with open(full, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}", normalized, missing


def verify_attestation(
    *,
    scheme: str,
    declared_hash: str,
    computed_hash: str,
) -> dict:
    """Verify a manifest signature declaration against a computed hash.

    Returns a result dict with ``ok``, ``scheme``, and ``reason`` keys.
    """
    scheme = str(scheme or "").strip().lower()
    if scheme in {"none", ""}:
        return {"ok": True, "scheme": "none", "reason": "unsigned"}
    if scheme in {"community-attestation-v1", "attested-hash-v1"}:
        if not declared_hash:
            return {"ok": False, "scheme": scheme, "reason": "missing_declared_hash"}
        if declared_hash != computed_hash:
            return {"ok": False, "scheme": scheme, "reason": "hash_mismatch"}
        return {"ok": True, "scheme": scheme, "reason": "attested_hash_match"}
    return {"ok": False, "scheme": scheme, "reason": "unsupported_scheme"}


class TrustStore:
    """Thread-safe JSON-file trust store with allow-list and deny-list."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self._path.exists():
            return {"schema": _SCHEMA, "allowlist": [], "deny_hashes": [], "revoked_signers": []}
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return {"schema": _SCHEMA, "allowlist": [], "deny_hashes": [], "revoked_signers": []}
        if not isinstance(data, dict):
            return {"schema": _SCHEMA, "allowlist": [], "deny_hashes": [], "revoked_signers": []}
        return {
            "schema": str(data.get("schema") or _SCHEMA),
            "allowlist": [i for i in data.get("allowlist", []) if isinstance(i, dict)],
            "deny_hashes": [str(i).strip() for i in data.get("deny_hashes", []) if str(i).strip()],
            "revoked_signers": [str(i).strip() for i in data.get("revoked_signers", []) if str(i).strip()],
        }

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)

    def evaluate(
        self,
        *,
        plugin_id: str,
        version: str,
        package_hash: str,
        signer: str,
        required_tier: int,
    ) -> dict:
        """Evaluate trust policy for a plugin.

        Returns ``{"ok": bool, "reason": str}``.
        Tier < 3 plugins always pass (no trust verification needed).
        """
        data = self._load()
        if package_hash in set(data["deny_hashes"]):
            return {"ok": False, "reason": "hash_denied"}
        if signer and signer in set(data["revoked_signers"]):
            return {"ok": False, "reason": "signer_revoked"}
        if required_tier < 3:
            return {"ok": True, "reason": "not_required"}
        for entry in data["allowlist"]:
            if (
                entry.get("plugin_id") == plugin_id
                and entry.get("version") == version
                and entry.get("hash") == package_hash
                and entry.get("signer") == signer
            ):
                return {"ok": True, "reason": "allowlist_match"}
        return {"ok": False, "reason": "allowlist_miss"}
