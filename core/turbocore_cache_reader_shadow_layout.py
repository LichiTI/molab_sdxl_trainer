"""Payload layout probes for debug-only cache reader shadow."""

from __future__ import annotations

from typing import Any, Dict

from core.turbocore_cache_reader_shadow_manifest import build_cache_reader_shadow_manifest
from core.turbocore_cache_reader_shadow_native import ENABLE_ENV, load_native_cache_reader_shadow_api
from core.turbocore_cached_dataset_prefetch_native import stable_json, truthy_env


def run_cache_reader_shadow_payload_layout(
    dataset: Any,
    *,
    max_files: int = 64,
    max_tensors_per_file: int = 128,
    prefer_native: bool = True,
) -> Dict[str, Any]:
    manifest = build_cache_reader_shadow_manifest(dataset, max_files=max_files)
    if not truthy_env(ENABLE_ENV):
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_payload_layout",
            "provider": "native_cache_reader_shadow_policy",
            "ok": True,
            "skipped": True,
            "reason": "shadow_disabled_by_env",
            "enable_env": ENABLE_ENV,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    if not prefer_native:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_payload_layout",
            "provider": "python_cache_reader_shadow_payload_layout_fallback",
            "ok": False,
            "reason": "python_payload_layout_shadow_not_implemented",
            "payload_layout_contract": False,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    try:
        native = load_native_cache_reader_shadow_api()
        return dict(native.run_cache_reader_shadow_payload_layout_probe(
            stable_json(manifest),
            max(int(max_files), 1),
            max(int(max_tensors_per_file), 1),
        ))
    except Exception as exc:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_payload_layout",
            "provider": "python_cache_reader_shadow_payload_layout_fallback",
            "ok": False,
            "reason": "native_payload_layout_probe_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
            "payload_layout_contract": False,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }


def run_cache_reader_shadow_payload_read(
    dataset: Any,
    *,
    max_files: int = 64,
    max_tensors_per_file: int = 128,
    max_payload_bytes_per_tensor: int = 4096,
    buffer_size: int = 4096,
    selected_only: bool = True,
    read_full_payload: bool = False,
    prefer_native: bool = True,
) -> Dict[str, Any]:
    manifest = build_cache_reader_shadow_manifest(dataset, max_files=max_files)
    if not truthy_env(ENABLE_ENV):
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_payload_read",
            "provider": "native_cache_reader_shadow_policy",
            "ok": True,
            "skipped": True,
            "reason": "shadow_disabled_by_env",
            "enable_env": ENABLE_ENV,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    if not prefer_native:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_payload_read",
            "provider": "python_cache_reader_shadow_payload_read_fallback",
            "ok": False,
            "reason": "python_payload_read_shadow_not_implemented",
            "payload_read_shadow": False,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    try:
        native = load_native_cache_reader_shadow_api()
        return dict(native.run_cache_reader_shadow_payload_read_probe(
            stable_json(manifest),
            max(int(max_files), 1),
            max(int(max_tensors_per_file), 1),
            max(int(max_payload_bytes_per_tensor), 1),
            max(int(buffer_size), 1),
            bool(selected_only),
            bool(read_full_payload),
        ))
    except Exception as exc:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_payload_read",
            "provider": "python_cache_reader_shadow_payload_read_fallback",
            "ok": False,
            "reason": "native_payload_read_probe_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
            "payload_read_shadow": False,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }


def run_cache_reader_shadow_tensor_decode_contract(
    dataset: Any,
    *,
    max_files: int = 64,
    max_tensors_per_file: int = 128,
    selected_only: bool = True,
    prefer_native: bool = True,
) -> Dict[str, Any]:
    manifest = build_cache_reader_shadow_manifest(dataset, max_files=max_files)
    if not truthy_env(ENABLE_ENV):
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_tensor_decode_contract",
            "provider": "native_cache_reader_shadow_policy",
            "ok": True,
            "skipped": True,
            "reason": "shadow_disabled_by_env",
            "enable_env": ENABLE_ENV,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    if not prefer_native:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_tensor_decode_contract",
            "provider": "python_cache_reader_shadow_tensor_decode_contract_fallback",
            "ok": False,
            "reason": "python_tensor_decode_contract_shadow_not_implemented",
            "tensor_decode_contract": False,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    try:
        native = load_native_cache_reader_shadow_api()
        return dict(native.run_cache_reader_shadow_tensor_decode_contract_probe(
            stable_json(manifest),
            max(int(max_files), 1),
            max(int(max_tensors_per_file), 1),
            bool(selected_only),
        ))
    except Exception as exc:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_tensor_decode_contract",
            "provider": "python_cache_reader_shadow_tensor_decode_contract_fallback",
            "ok": False,
            "reason": "native_tensor_decode_contract_probe_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
            "tensor_decode_contract": False,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }


def run_cache_reader_shadow_tensor_decode_parity(
    dataset: Any,
    *,
    max_files: int = 64,
    max_tensors_per_file: int = 128,
    max_decode_payload_bytes: int = 16 * 1024 * 1024,
    selected_only: bool = True,
    prefer_native: bool = True,
) -> Dict[str, Any]:
    manifest = build_cache_reader_shadow_manifest(dataset, max_files=max_files)
    if not truthy_env(ENABLE_ENV):
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_tensor_decode_parity",
            "provider": "native_cache_reader_shadow_policy",
            "ok": True,
            "skipped": True,
            "reason": "shadow_disabled_by_env",
            "enable_env": ENABLE_ENV,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    if not prefer_native:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_tensor_decode_parity",
            "provider": "python_cache_reader_shadow_tensor_decode_parity_fallback",
            "ok": False,
            "reason": "python_tensor_decode_parity_shadow_not_implemented",
            "tensor_decode_parity_shadow": False,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    try:
        native = load_native_cache_reader_shadow_api()
        return dict(native.run_cache_reader_shadow_tensor_decode_parity_probe(
            stable_json(manifest),
            max(int(max_files), 1),
            max(int(max_tensors_per_file), 1),
            max(int(max_decode_payload_bytes), 1),
            bool(selected_only),
        ))
    except Exception as exc:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_tensor_decode_parity",
            "provider": "python_cache_reader_shadow_tensor_decode_parity_fallback",
            "ok": False,
            "reason": "native_tensor_decode_parity_probe_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
            "tensor_decode_parity_shadow": False,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }


class NativeCacheReaderDecodeShadowSession:
    """Debug-only native session that owns tensor decode layout candidates."""

    def __init__(
        self,
        manifest: Dict[str, Any],
        *,
        max_files: int = 64,
        max_tensors_per_file: int = 128,
        max_decode_payload_bytes: int = 16 * 1024 * 1024,
        selected_only: bool = True,
    ) -> None:
        native = load_native_cache_reader_shadow_api()
        self._native = native
        self._closed = False
        self._max_files = max(int(max_files), 1)
        self._max_tensors_per_file = max(int(max_tensors_per_file), 1)
        self._max_decode_payload_bytes = max(int(max_decode_payload_bytes), 1)
        self._selected_only = bool(selected_only)
        created = native.create_cache_reader_shadow_tensor_decode_session(
            stable_json(manifest),
            self._max_files,
            self._max_tensors_per_file,
            self._max_decode_payload_bytes,
            self._selected_only,
        )
        if not bool(created.get("ok", False)):
            reason = str(created.get("reason") or "native_cache_reader_decode_shadow_session_create_failed")
            raise RuntimeError(reason)
        self.session_id = int(created.get("session_id", 0) or 0)
        if self.session_id <= 0:
            raise RuntimeError("native_cache_reader_decode_shadow_session_id_missing")
        self.created = dict(created)

    def stats(self) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_reader_decode_shadow_session_closed", "training_path_enabled": False}
        return dict(self._native.cache_reader_shadow_tensor_decode_session_stats(self.session_id))

    def run_chunk(self, *, cursor: int = 0, max_tensors: int = 16) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_reader_decode_shadow_session_closed", "training_path_enabled": False}
        return dict(self._native.run_cache_reader_shadow_tensor_decode_session_chunk(
            self.session_id,
            max(int(cursor), 0),
            max(int(max_tensors), 1),
        ))

    def run_cpu_payload_chunk(
        self,
        *,
        cursor: int = 0,
        max_tensors: int = 16,
        max_cpu_payload_buffer_bytes: int = 1024 * 1024,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_reader_decode_shadow_session_closed", "training_path_enabled": False}
        return dict(self._native.run_cache_reader_shadow_tensor_decode_session_cpu_payload_chunk(
            self.session_id,
            max(int(cursor), 0),
            max(int(max_tensors), 1),
            max(int(max_cpu_payload_buffer_bytes), 1),
        ))

    def run_batch_cpu_payload_chunk(
        self,
        *,
        cursor: int = 0,
        max_tensors: int = 16,
        max_batch_payload_buffer_bytes: int = 16 * 1024 * 1024,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_reader_decode_shadow_session_closed", "training_path_enabled": False}
        return dict(self._native.run_cache_reader_shadow_tensor_decode_session_batch_cpu_payload_chunk(
            self.session_id,
            max(int(cursor), 0),
            max(int(max_tensors), 1),
            max(int(max_batch_payload_buffer_bytes), 1),
        ))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._native.destroy_cache_reader_shadow_tensor_decode_session(self.session_id)
        finally:
            self._closed = True

    def __enter__(self) -> "NativeCacheReaderDecodeShadowSession":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def run_cache_reader_shadow_tensor_decode_session(
    dataset: Any,
    *,
    max_files: int = 64,
    max_tensors_per_file: int = 128,
    max_decode_payload_bytes: int = 16 * 1024 * 1024,
    selected_only: bool = True,
    chunk_size: int = 16,
    prefer_native: bool = True,
) -> Dict[str, Any]:
    manifest = build_cache_reader_shadow_manifest(dataset, max_files=max_files)
    if not truthy_env(ENABLE_ENV):
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_tensor_decode_session",
            "provider": "native_cache_reader_shadow_policy",
            "ok": True,
            "skipped": True,
            "reason": "shadow_disabled_by_env",
            "enable_env": ENABLE_ENV,
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    if not prefer_native:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_tensor_decode_session",
            "provider": "python_cache_reader_shadow_tensor_decode_session_fallback",
            "ok": False,
            "reason": "python_tensor_decode_session_shadow_not_implemented",
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }
    try:
        with NativeCacheReaderDecodeShadowSession(
            manifest,
            max_files=max_files,
            max_tensors_per_file=max_tensors_per_file,
            max_decode_payload_bytes=max_decode_payload_bytes,
            selected_only=selected_only,
        ) as session:
            cursor = 0
            chunks = []
            total_read = 0
            total_decoded = 0
            ok = True
            while True:
                chunk = session.run_chunk(cursor=cursor, max_tensors=chunk_size)
                chunks.append(chunk)
                ok = ok and bool(chunk.get("ok", False))
                total_read += int(chunk.get("data_payload_bytes_read", 0) or 0)
                total_decoded += int(chunk.get("tensor_decode_count", 0) or 0)
                cursor = int(chunk.get("next_cursor", cursor) or cursor)
                if bool(chunk.get("chunk_complete", True)):
                    break
            return {
                "schema_version": 1,
                "probe": "turbocore_cache_reader_shadow_tensor_decode_session",
                "provider": "native_cache_reader_shadow_tensor_decode_session_adapter",
                "native_runtime": True,
                "ok": ok,
                "debug_only": True,
                "shadow_run": True,
                "session_id": session.session_id,
                "session_create": session.created,
                "chunk_count": len(chunks),
                "tensor_decode_count": total_decoded,
                "data_payload_bytes_read": total_read,
                "chunks": chunks,
                "reads_tensor_payload_bytes": True,
                "parses_tensor_payloads": True,
                "decodes_tensor_payloads": True,
                "returns_tensor_payloads": False,
                "cache_reader_path_enabled": False,
                "training_path_enabled": False,
            }
    except Exception as exc:
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_tensor_decode_session",
            "provider": "python_cache_reader_shadow_tensor_decode_session_fallback",
            "ok": False,
            "reason": "native_tensor_decode_session_probe_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
            "cache_reader_path_enabled": False,
            "training_path_enabled": False,
        }


__all__ = [
    "NativeCacheReaderDecodeShadowSession",
    "run_cache_reader_shadow_payload_layout",
    "run_cache_reader_shadow_payload_read",
    "run_cache_reader_shadow_tensor_decode_contract",
    "run_cache_reader_shadow_tensor_decode_parity",
    "run_cache_reader_shadow_tensor_decode_session",
]
