"""Persistent debug-only native cache reader header sessions."""

from __future__ import annotations

import weakref
from typing import Any, Dict

from core.turbocore_cache_reader_shadow_manifest import build_cache_reader_shadow_manifest
from core.turbocore_cache_reader_shadow_native import ENABLE_ENV, load_native_cache_reader_shadow_api
from core.turbocore_cache_reader_shadow_timing import _python_cache_reader_shadow_timing
from core.turbocore_cached_dataset_prefetch_native import (
    manifest_fingerprint,
    payload_signature,
    stable_json,
    truthy_env,
)


class NativeCacheReaderShadowSession:
    """Debug-only native session that owns cache reader header metadata."""

    def __init__(self, manifest: Dict[str, Any], *, max_files: int = 64, max_tensors_per_file: int = 128) -> None:
        native = load_native_cache_reader_shadow_api()
        self._native = native
        self._manifest_signature = payload_signature(manifest)
        self._fingerprint = manifest_fingerprint(manifest)
        self._max_files = max(int(max_files), 1)
        self._max_tensors_per_file = max(int(max_tensors_per_file), 1)
        self._closed = False
        created = native.create_cache_reader_shadow_session(
            stable_json(manifest),
            self._max_files,
            self._max_tensors_per_file,
        )
        if not bool(created.get("ok", False)):
            reason = str(created.get("reason") or "native_cache_reader_shadow_session_create_failed")
            raise RuntimeError(reason)
        self.session_id = int(created.get("session_id", 0) or 0)
        if self.session_id <= 0:
            raise RuntimeError("native_cache_reader_shadow_session_id_missing")
        self.created = dict(created)

    @property
    def manifest_signature(self) -> str:
        return self._manifest_signature

    @property
    def fingerprint(self) -> Dict[str, Any]:
        return dict(self._fingerprint)

    @property
    def max_files(self) -> int:
        return self._max_files

    @property
    def max_tensors_per_file(self) -> int:
        return self._max_tensors_per_file

    def stats(self) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_reader_shadow_session_closed", "training_path_enabled": False}
        return dict(self._native.cache_reader_shadow_session_stats(self.session_id))

    def run(self, *, max_preview: int = 16, manifest: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_reader_shadow_session_closed", "training_path_enabled": False}
        if manifest is not None and payload_signature(manifest) != self._manifest_signature:
            return {
                "schema_version": 1,
                "probe": "turbocore_cache_reader_shadow_session_probe",
                "provider": "native_cache_reader_shadow_session",
                "native_runtime": True,
                "ok": False,
                "session_id": self.session_id,
                "reason": "reference_manifest_changed",
                "requires_rebuild": True,
                "training_path_enabled": False,
            }
        return dict(self._native.run_cache_reader_shadow_session_probe(self.session_id, max(int(max_preview), 1)))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._native.destroy_cache_reader_shadow_session(self.session_id)
        finally:
            self._closed = True

    def __enter__(self) -> "NativeCacheReaderShadowSession":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def create_cache_reader_shadow_session(
    dataset: Any,
    *,
    max_files: int = 64,
    max_tensors_per_file: int = 128,
) -> NativeCacheReaderShadowSession:
    return NativeCacheReaderShadowSession(
        build_cache_reader_shadow_manifest(dataset, max_files=max_files),
        max_files=max_files,
        max_tensors_per_file=max_tensors_per_file,
    )


def close_cache_reader_shadow_session(dataset: Any) -> bool:
    session = getattr(dataset, "_native_cache_reader_shadow_session", None)
    if session is None:
        return False
    try:
        session.close()
    finally:
        for name in (
            "_native_cache_reader_shadow_session",
            "_native_cache_reader_shadow_session_signature",
            "_native_cache_reader_shadow_session_fingerprint",
            "_native_cache_reader_shadow_session_max_files",
            "_native_cache_reader_shadow_session_max_tensors_per_file",
            "_native_cache_reader_shadow_session_finalizer",
        ):
            try:
                delattr(dataset, name)
            except Exception:
                pass
    return True


def _get_or_create_cache_reader_shadow_session(
    dataset: Any,
    *,
    max_files: int = 64,
    max_tensors_per_file: int = 128,
) -> tuple[NativeCacheReaderShadowSession, bool, bool, Dict[str, Any]]:
    manifest = build_cache_reader_shadow_manifest(dataset, max_files=max_files)
    fingerprint = manifest_fingerprint(manifest)
    resolved_max_files = max(int(max_files), 1)
    resolved_max_tensors = max(int(max_tensors_per_file), 1)
    existing = getattr(dataset, "_native_cache_reader_shadow_session", None)
    if isinstance(existing, NativeCacheReaderShadowSession):
        stats = existing.stats()
        previous_fingerprint = getattr(dataset, "_native_cache_reader_shadow_session_fingerprint", None)
        previous_max_files = int(getattr(dataset, "_native_cache_reader_shadow_session_max_files", 0) or 0)
        previous_max_tensors = int(getattr(dataset, "_native_cache_reader_shadow_session_max_tensors_per_file", 0) or 0)
        if (
            bool(stats.get("ok", False))
            and previous_fingerprint == fingerprint
            and previous_max_files == resolved_max_files
            and previous_max_tensors == resolved_max_tensors
        ):
            return existing, True, False, fingerprint
        close_cache_reader_shadow_session(dataset)

    session = NativeCacheReaderShadowSession(
        manifest,
        max_files=resolved_max_files,
        max_tensors_per_file=resolved_max_tensors,
    )
    try:
        setattr(dataset, "_native_cache_reader_shadow_session", session)
        setattr(dataset, "_native_cache_reader_shadow_session_signature", session.manifest_signature)
        setattr(dataset, "_native_cache_reader_shadow_session_fingerprint", fingerprint)
        setattr(dataset, "_native_cache_reader_shadow_session_max_files", resolved_max_files)
        setattr(dataset, "_native_cache_reader_shadow_session_max_tensors_per_file", resolved_max_tensors)
        setattr(dataset, "_native_cache_reader_shadow_session_finalizer", weakref.finalize(dataset, session.close))
    except Exception:
        session.close()
        raise
    return session, False, existing is not None, fingerprint


def run_cache_reader_shadow_header_session(
    dataset: Any,
    *,
    max_files: int = 64,
    max_tensors_per_file: int = 128,
    max_preview: int = 16,
    prefer_native: bool = True,
    persist_session: bool = False,
) -> Dict[str, Any]:
    manifest = build_cache_reader_shadow_manifest(dataset, max_files=max_files)
    if not truthy_env(ENABLE_ENV):
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_session_probe",
            "provider": "native_cache_reader_shadow_policy",
            "ok": True,
            "skipped": True,
            "reason": "shadow_disabled_by_env",
            "enable_env": ENABLE_ENV,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    if not prefer_native:
        return _python_cache_reader_shadow_timing(
            manifest,
            max_files=max_files,
            max_bytes_per_file=4096,
            buffer_size=4096,
        )
    try:
        if persist_session:
            session, session_reused, requires_rebuild, fingerprint = _get_or_create_cache_reader_shadow_session(
                dataset,
                max_files=max_files,
                max_tensors_per_file=max_tensors_per_file,
            )
            probe = session.run(max_preview=max_preview, manifest=manifest)
            session_id = session.session_id
            session_create = session.created
        else:
            with NativeCacheReaderShadowSession(
                manifest,
                max_files=max_files,
                max_tensors_per_file=max_tensors_per_file,
            ) as session:
                session_reused = False
                requires_rebuild = False
                fingerprint = session.fingerprint
                probe = session.run(max_preview=max_preview, manifest=manifest)
                session_id = session.session_id
                session_create = session.created
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_adapter",
            "provider": "native_cache_reader_shadow_persistent_session_adapter" if persist_session else "native_cache_reader_shadow_session_adapter",
            "ok": bool(probe.get("ok", False)),
            "skipped": False,
            "shadow_run": True,
            "persistent_session": bool(persist_session),
            "session_reused_by_adapter": bool(session_reused),
            "requires_rebuild": bool(requires_rebuild),
            "fingerprint": fingerprint,
            "session_id": session_id,
            "session_create": session_create,
            "reader_probe": probe,
            "native_error": "",
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    except Exception as exc:
        fallback = _python_cache_reader_shadow_timing(
            manifest,
            max_files=max_files,
            max_bytes_per_file=4096,
            buffer_size=4096,
        )
        fallback["provider"] = "python_cache_reader_shadow_session_fallback"
        fallback["native_error"] = f"{type(exc).__name__}: {exc}"
        return fallback


__all__ = [
    "NativeCacheReaderShadowSession",
    "close_cache_reader_shadow_session",
    "create_cache_reader_shadow_session",
    "run_cache_reader_shadow_header_session",
]
