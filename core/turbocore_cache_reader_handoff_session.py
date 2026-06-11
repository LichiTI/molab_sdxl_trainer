"""Debug-only long-lived batch handoff session for native cache reader payloads."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from core.turbocore_cache_reader_batch_gate import run_cache_reader_batch_parity_guard
from core.turbocore_cache_reader_shadow_layout import NativeCacheReaderDecodeShadowSession
from core.turbocore_cache_reader_shadow_manifest import build_cache_reader_shadow_manifest
from core.turbocore_cache_reader_training_probe import build_selected_cache_reader_dataset_view


class NativeCacheReaderBatchHandoffShadowSession:
    """Own a native decode session and expose repeated batch payload handoff probes."""

    def __init__(
        self,
        dataset: Any,
        sample_indices: Sequence[int],
        *,
        max_decode_payload_bytes: int,
        max_batch_cpu_payload_buffer_bytes: int,
    ) -> None:
        self.dataset = dataset
        self.sample_indices = [int(index) for index in sample_indices]
        self.max_batch_cpu_payload_buffer_bytes = max(int(max_batch_cpu_payload_buffer_bytes), 1)
        selected_view = build_selected_cache_reader_dataset_view(dataset, self.sample_indices)
        manifest = build_cache_reader_shadow_manifest(selected_view, max_files=max(len(self.sample_indices), 1) * 2)
        self._session = NativeCacheReaderDecodeShadowSession(
            manifest,
            max_files=max(len(self.sample_indices), 1) * 2,
            max_tensors_per_file=16,
            max_decode_payload_bytes=max(int(max_decode_payload_bytes), 1),
            selected_only=True,
        )
        self._cursor = 0
        self._closed = False
        self._run_count = 0
        self._total_payload_bytes = 0

    @property
    def session_id(self) -> int:
        return int(self._session.session_id)

    def stats(self) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_reader_batch_handoff_shadow_session_closed", "training_path_enabled": False}
        stats = dict(self._session.stats())
        return {
            "schema_version": 1,
            "provider": "native_cache_reader_batch_handoff_shadow_session",
            "ok": bool(stats.get("ok", False)),
            "session_id": self.session_id,
            "run_count": self._run_count,
            "cursor": self._cursor,
            "sample_count": len(self.sample_indices),
            "total_payload_bytes": self._total_payload_bytes,
            "native_decode_session_stats": stats,
            "returns_tensor_payloads": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "training_path_enabled": False,
        }

    def run_next(self, *, batch_size: int) -> Dict[str, Any]:
        if self._closed:
            return {"ok": False, "reason": "cache_reader_batch_handoff_shadow_session_closed", "training_path_enabled": False}
        start_cursor = self._cursor
        max_tensors = max(int(batch_size), 1)
        chunk = self._session.run_batch_cpu_payload_chunk(
            cursor=start_cursor,
            max_tensors=max_tensors,
            max_batch_payload_buffer_bytes=self.max_batch_cpu_payload_buffer_bytes,
        )
        self._run_count += 1
        self._cursor = int(chunk.get("next_cursor", start_cursor) or start_cursor)
        payload_shadow = dict(chunk.get("batch_cpu_payload_shadow", {}) or {})
        self._total_payload_bytes += int(payload_shadow.get("batch_cpu_payload_byte_count", 0) or 0)
        batch_indices = self.sample_indices[start_cursor:self._cursor]
        parity = run_cache_reader_batch_parity_guard(
            self.dataset,
            sample_indices=batch_indices,
            native_records=[dict(record) for record in list(chunk.get("records", []) or [])],
            native_batch_summary=dict(chunk.get("native_latent_batch_summary", {}) or {}),
            native_batch_payload_shadow=payload_shadow,
        )
        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_batch_handoff_shadow_session_run",
            "provider": "native_cache_reader_batch_handoff_shadow_session",
            "ok": bool(chunk.get("ok", False)) and bool(parity.get("ok", False)),
            "debug_only": True,
            "shadow_run": True,
            "session_id": self.session_id,
            "session_run": True,
            "session_reused": self._run_count > 1,
            "run_count": self._run_count,
            "cursor": start_cursor,
            "next_cursor": self._cursor,
            "chunk_complete": bool(chunk.get("chunk_complete", True)),
            "sample_indices": batch_indices,
            "batch_cpu_payload_byte_count": int(payload_shadow.get("batch_cpu_payload_byte_count", 0) or 0),
            "batch_cpu_payload_tensor_count": int(payload_shadow.get("source_tensor_count", 0) or 0),
            "batch_payload_parity_guard_passed": bool(parity.get("batch_payload_parity_guard_passed", False)),
            "torch_tensor_handoff_guard_passed": bool(parity.get("torch_tensor_handoff_guard_passed", False)),
            "torch_owned_tensor_handoff_guard_passed": bool(parity.get("torch_owned_tensor_handoff_guard_passed", False)),
            "batch_parity": parity,
            "returns_tensor_payloads": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "training_path_enabled": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._session.close()
        finally:
            self._closed = True

    def __enter__(self) -> "NativeCacheReaderBatchHandoffShadowSession":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def run_cache_reader_batch_handoff_shadow_session_probe(
    dataset: Any,
    *,
    sample_indices: Sequence[int],
    batch_size: int,
    max_decode_payload_bytes: int,
    max_batch_cpu_payload_buffer_bytes: int,
) -> Dict[str, Any]:
    runs: list[dict[str, Any]] = []
    with NativeCacheReaderBatchHandoffShadowSession(
        dataset,
        sample_indices,
        max_decode_payload_bytes=max_decode_payload_bytes,
        max_batch_cpu_payload_buffer_bytes=max_batch_cpu_payload_buffer_bytes,
    ) as session:
        stats_before = session.stats()
        while True:
            run = session.run_next(batch_size=batch_size)
            runs.append(run)
            if bool(run.get("chunk_complete", True)):
                break
        stats_after = session.stats()
    return {
        "schema_version": 1,
        "probe": "turbocore_cache_reader_batch_handoff_shadow_session",
        "provider": "native_cache_reader_batch_handoff_shadow_session",
        "ok": all(bool(run.get("ok", False)) for run in runs),
        "debug_only": True,
        "shadow_run": True,
        "session_id": int(stats_before.get("session_id", 0) or 0),
        "session_reused": any(bool(run.get("session_reused", False)) for run in runs),
        "run_count": len(runs),
        "batch_size": max(int(batch_size), 1),
        "sample_count": len([int(index) for index in sample_indices]),
        "total_payload_bytes": sum(int(run.get("batch_cpu_payload_byte_count", 0) or 0) for run in runs),
        "batch_payload_parity_guard_passed": all(bool(run.get("batch_payload_parity_guard_passed", False)) for run in runs),
        "torch_tensor_handoff_guard_passed": all(bool(run.get("torch_tensor_handoff_guard_passed", False)) for run in runs),
        "torch_owned_tensor_handoff_guard_passed": all(bool(run.get("torch_owned_tensor_handoff_guard_passed", False)) for run in runs),
        "stats_before": stats_before,
        "stats_after": stats_after,
        "runs": runs,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }


__all__ = [
    "NativeCacheReaderBatchHandoffShadowSession",
    "run_cache_reader_batch_handoff_shadow_session_probe",
]
