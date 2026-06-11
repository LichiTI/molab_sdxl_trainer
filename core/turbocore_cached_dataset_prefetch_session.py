"""Persistent native cached dataset prefetch session wrapper."""

from __future__ import annotations

import weakref
from typing import Any, Dict

from core.turbocore_cached_dataset_prefetch_manifest import build_cached_dataset_prefetch_manifest
from core.turbocore_cached_dataset_prefetch_native import (
    load_native_cache_prefetch_fast_session_api,
    load_native_cache_prefetch_session_api,
    manifest_fingerprint,
    payload_signature,
    stable_json,
)


class NativeCachedDatasetPrefetchSession:
    """Debug-only native session that owns cached dataset manifest metadata."""

    def __init__(self, manifest: Dict[str, Any]) -> None:
        native = load_native_cache_prefetch_session_api()
        self._native = native
        self._manifest_signature = payload_signature(manifest)
        self._fingerprint = manifest_fingerprint(manifest)
        created = native.create_cache_prefetch_session(stable_json(manifest))
        if not bool(created.get("ok", False)):
            reason = str(created.get("reason") or "native_cache_prefetch_session_create_failed")
            raise RuntimeError(reason)
        self.session_id = int(created.get("session_id", 0) or 0)
        if self.session_id <= 0:
            raise RuntimeError("native_cache_prefetch_session_id_missing")
        self.created = created
        self._closed = False

    @property
    def manifest_signature(self) -> str:
        return self._manifest_signature

    @property
    def fingerprint(self) -> Dict[str, Any]:
        return dict(self._fingerprint)

    def stats(self) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_prefetch_session_closed", "training_path_enabled": False}
        return dict(self._native.cache_prefetch_session_stats(self.session_id))

    def run(
        self,
        *,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        seed: int,
        prefetch_depth: int,
        chunk_size: int = 256,
        max_preview: int = 16,
        manifest: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_prefetch_session_closed", "training_path_enabled": False}
        if manifest is not None and payload_signature(manifest) != self._manifest_signature:
            return _changed_manifest_report(
                "turbocore_cached_dataset_prefetch_session_probe",
                "native_cached_dataset_prefetch_session",
                self.session_id,
            )
        return dict(self._native.run_cache_prefetch_session_probe(
            self.session_id,
            int(batch_size),
            bool(drop_last),
            bool(shuffle),
            int(seed),
            max(int(prefetch_depth), 1),
            max(int(chunk_size), 1),
            max(int(max_preview), 1),
        ))

    def run_fast(
        self,
        *,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        seed: int,
        prefetch_depth: int,
        chunk_size: int = 256,
        max_preview: int = 16,
        manifest: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_prefetch_session_closed", "training_path_enabled": False}
        if manifest is not None and payload_signature(manifest) != self._manifest_signature:
            return _changed_manifest_report(
                "turbocore_cached_dataset_prefetch_session_fast_probe",
                "native_cached_dataset_prefetch_session_fast",
                self.session_id,
            )
        native = load_native_cache_prefetch_fast_session_api()
        return dict(native.run_cache_prefetch_session_fast_probe(
            self.session_id,
            int(batch_size),
            bool(drop_last),
            bool(shuffle),
            int(seed),
            max(int(prefetch_depth), 1),
            max(int(chunk_size), 1),
            max(int(max_preview), 1),
        ))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._native.destroy_cache_prefetch_session(self.session_id)
        finally:
            self._closed = True

    def __enter__(self) -> "NativeCachedDatasetPrefetchSession":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _changed_manifest_report(probe: str, provider: str, session_id: int) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": probe,
        "provider": provider,
        "native_runtime": True,
        "ok": False,
        "session_id": session_id,
        "reason": "reference_manifest_changed",
        "requires_rebuild": True,
        "training_path_enabled": False,
    }


def create_cached_dataset_prefetch_session(dataset: Any) -> NativeCachedDatasetPrefetchSession:
    return NativeCachedDatasetPrefetchSession(build_cached_dataset_prefetch_manifest(dataset))


def close_cached_dataset_prefetch_session(dataset: Any) -> bool:
    session = getattr(dataset, "_native_cache_prefetch_session", None)
    if session is None:
        return False
    try:
        session.close()
    finally:
        for name in (
            "_native_cache_prefetch_session",
            "_native_cache_prefetch_session_signature",
            "_native_cache_prefetch_session_fingerprint",
            "_native_cache_prefetch_session_finalizer",
        ):
            try:
                delattr(dataset, name)
            except Exception:
                pass
    return True


def get_or_create_dataset_prefetch_session(
    dataset: Any,
) -> tuple[NativeCachedDatasetPrefetchSession, bool, bool, Dict[str, Any]]:
    manifest = build_cached_dataset_prefetch_manifest(dataset)
    fingerprint = manifest_fingerprint(manifest)
    existing = getattr(dataset, "_native_cache_prefetch_session", None)
    if isinstance(existing, NativeCachedDatasetPrefetchSession):
        stats = existing.stats()
        previous_fingerprint = getattr(dataset, "_native_cache_prefetch_session_fingerprint", None)
        fingerprint_matches = previous_fingerprint == fingerprint
        sample_count_matches = int(stats.get("sample_count", -1) or -1) == len(getattr(dataset, "samples", []) or [])
        if bool(stats.get("ok", False)) and sample_count_matches and fingerprint_matches:
            return existing, True, False, fingerprint
        close_cached_dataset_prefetch_session(dataset)

    session = NativeCachedDatasetPrefetchSession(manifest)
    try:
        setattr(dataset, "_native_cache_prefetch_session", session)
        setattr(dataset, "_native_cache_prefetch_session_signature", session.manifest_signature)
        setattr(dataset, "_native_cache_prefetch_session_fingerprint", fingerprint)
        setattr(dataset, "_native_cache_prefetch_session_finalizer", weakref.finalize(dataset, session.close))
    except Exception:
        session.close()
        raise
    return session, False, existing is not None, fingerprint


__all__ = [
    "NativeCachedDatasetPrefetchSession",
    "close_cached_dataset_prefetch_session",
    "create_cached_dataset_prefetch_session",
    "get_or_create_dataset_prefetch_session",
]
