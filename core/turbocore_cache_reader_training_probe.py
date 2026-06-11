"""Native probe helpers for the experimental cache reader training gate."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from core.turbocore_cache_reader_shadow_layout import NativeCacheReaderDecodeShadowSession
from core.turbocore_cache_reader_shadow_manifest import build_cache_reader_shadow_manifest


def build_selected_cache_reader_dataset_view(dataset: Any, sample_indices: Sequence[int]) -> Any:
    view_type = type(type(dataset).__name__, (), {})
    view = view_type()
    view.data_dir = getattr(dataset, "data_dir", "")
    view.cache_mmap = bool(getattr(dataset, "cache_mmap", False))
    view.cache_lazy = bool(getattr(dataset, "cache_lazy", False))
    view.file_handle_cache_size = max(int(getattr(dataset, "file_handle_cache_size", 0) or 0), 0)
    all_samples = list(getattr(dataset, "samples", []) or [])
    view.samples = [all_samples[int(index)] for index in sample_indices if 0 <= int(index) < len(all_samples)]
    return view


def run_native_cache_reader_training_probe(
    dataset: Any,
    sample_indices: Sequence[int],
    *,
    max_decode_payload_bytes: int,
    max_cpu_payload_buffer_bytes: int = 0,
    max_batch_cpu_payload_buffer_bytes: int = 0,
) -> Dict[str, Any]:
    selected_view = build_selected_cache_reader_dataset_view(dataset, sample_indices)
    manifest = build_cache_reader_shadow_manifest(selected_view, max_files=max(len(sample_indices), 1) * 2)
    records: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    batch_payload_shadows: list[dict[str, Any]] = []
    with NativeCacheReaderDecodeShadowSession(
        manifest,
        max_files=max(len(sample_indices), 1) * 2,
        max_tensors_per_file=16,
        max_decode_payload_bytes=max(int(max_decode_payload_bytes), 1),
        selected_only=True,
    ) as session:
        cursor = 0
        chunk_count = 0
        total_read = 0
        while True:
            max_tensors = max(len(sample_indices), 1)
            if max_batch_cpu_payload_buffer_bytes > 0:
                chunk = session.run_batch_cpu_payload_chunk(
                    cursor=cursor,
                    max_tensors=max_tensors,
                    max_batch_payload_buffer_bytes=max_batch_cpu_payload_buffer_bytes,
                )
            elif max_cpu_payload_buffer_bytes > 0:
                chunk = session.run_cpu_payload_chunk(
                    cursor=cursor,
                    max_tensors=max_tensors,
                    max_cpu_payload_buffer_bytes=max_cpu_payload_buffer_bytes,
                )
            else:
                chunk = session.run_chunk(cursor=cursor, max_tensors=max_tensors)
            chunk_count += 1
            records.extend(list(chunk.get("records", []) or []))
            summary = chunk.get("native_latent_batch_summary")
            if isinstance(summary, dict):
                batch_summaries.append(dict(summary))
            batch_payload = chunk.get("batch_cpu_payload_shadow")
            if isinstance(batch_payload, dict):
                batch_payload_shadows.append(dict(batch_payload))
            total_read += int(chunk.get("data_payload_bytes_read", 0) or 0)
            cursor = int(chunk.get("next_cursor", cursor) or cursor)
            if bool(chunk.get("chunk_complete", True)):
                break
        return {
            "session_create": dict(session.created),
            "session_stats": dict(session.stats()),
            "chunk_count": chunk_count,
            "data_payload_bytes_read": total_read,
            "records": records,
            "native_latent_batch_summary": batch_summaries[-1] if batch_summaries else {},
            "native_latent_batch_summaries": batch_summaries,
            "batch_cpu_payload_shadow": batch_payload_shadows[-1] if batch_payload_shadows else {},
            "batch_cpu_payload_shadows": batch_payload_shadows,
        }


def compact_batch_cpu_payload_shadow(report: Any) -> Dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {}
    return {
        "provider": str(report.get("provider") or ""),
        "batch_cpu_payload_shadow": bool(report.get("batch_cpu_payload_shadow", False)),
        "batch_cpu_payload_ready": bool(report.get("batch_cpu_payload_ready", False)),
        "batch_cpu_payload_byte_count": int(report.get("batch_cpu_payload_byte_count", 0) or 0),
        "shape": list(report.get("shape", []) or []),
        "canonical_dtype": str(report.get("canonical_dtype") or ""),
        "batch_layout": str(report.get("batch_layout") or ""),
        "source_tensor_count": int(report.get("source_tensor_count", 0) or 0),
        "returns_cpu_payload_buffer": bool(report.get("returns_cpu_payload_buffer", False)),
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }


__all__ = [
    "build_selected_cache_reader_dataset_view",
    "compact_batch_cpu_payload_shadow",
    "run_native_cache_reader_training_probe",
]
