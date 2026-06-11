"""Smoke probe for debug-only native cache reader shadow timing."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.lulynx_trainer.anima_cached_dataset import AnimaCachedDataset, create_anima_cached_dataloader  # noqa: E402
from core.lulynx_trainer.newbie_cached_dataset import (  # noqa: E402
    NewbieCacheSchema,
    NewbieCachedDataset,
    create_newbie_cached_dataloader,
)
from core.turbocore_cache_reader_shadow import (  # noqa: E402
    ENABLE_ENV,
    NativeCacheReaderDecodeShadowSession,
    NativeCacheReaderShadowSession,
    cache_reader_shadow_header_cache_stats,
    cache_reader_shadow_payload_layout_cache_stats,
    close_cache_reader_shadow_session,
    clear_cache_reader_shadow_header_cache,
    clear_cache_reader_shadow_payload_layout_cache,
    run_cache_reader_shadow_header_session,
    run_cache_reader_shadow_payload_layout,
    run_cache_reader_shadow_payload_read,
    run_cache_reader_shadow_tensor_decode_contract,
    run_cache_reader_shadow_tensor_decode_parity,
    run_cache_reader_shadow_tensor_decode_session,
    run_cache_reader_shadow_timing,
)
from core.turbocore_cache_reader_shadow_manifest import build_cache_reader_shadow_manifest  # noqa: E402
from core.turbocore_cached_dataset_prefetch_manifest import build_cached_dataset_prefetch_manifest  # noqa: E402


def _set_env(values: dict[str, str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        os.environ[key] = value
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _write_newbie_cache(root: Path, stem: str, size: int) -> None:
    latents = np.arange(1 * 4 * size * size, dtype=np.float32).reshape(1, 4, size, size)
    np.savez(
        root / f"{stem}_newbie.npz",
        newbie_cache_schema_version=np.array([2], dtype=np.int64),
        latents=latents,
        encoder_hidden_states=np.zeros((1, 3, 8), dtype=np.float32),
        pooled_prompt_embeds=np.zeros((1, 6), dtype=np.float32),
        attention_mask=np.ones((1, 3), dtype=np.int64),
    )


def _write_anima_cache(root: Path, stem: str, size: int) -> None:
    latents = np.arange(4 * size * size, dtype=np.float32).reshape(4, size, size)
    np.savez(root / f"{stem}_0001_anima.npz", latents_4=latents)
    np.savez(
        root / f"{stem}_anima_te.npz",
        prompt_embeds=np.zeros((3, 8), dtype=np.float32),
        attn_mask=np.ones((3,), dtype=np.int64),
    )


def _write_safetensors_cache(root: Path, stem: str) -> Path:
    header = {
        "latents": {"dtype": "F32", "shape": [1, 4, 2, 2], "data_offsets": [0, 64]},
        "encoder_hidden_states": {"dtype": "F32", "shape": [1, 3, 8], "data_offsets": [64, 160]},
    }
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path = root / f"{stem}.safetensors"
    latents = np.arange(16, dtype="<f4").tobytes()
    embeds = np.zeros((1, 3, 8), dtype="<f4").tobytes()
    path.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header + latents + embeds)
    return path


class _CacheSample:
    def __init__(self, stem: str, path: Path) -> None:
        self.stem = stem
        self.cache_path = path


class _CacheDataset:
    cache_lazy = True
    cache_mmap = True
    file_handle_cache_size = 128

    def __init__(self, root: Path, path: Path) -> None:
        self.data_dir = root
        self.samples = [_CacheSample(path.stem, path)]


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lulynx_cache_reader_shadow_") as tmp:
        root = Path(tmp)
        _write_newbie_cache(root, "flat_a", 4)
        _write_newbie_cache(root, "flat_b", 4)
        newbie = NewbieCachedDataset(
            root,
            schema=NewbieCacheSchema(require_schema_version=True),
            cache_mmap=True,
            cache_lazy=True,
        )
        disabled = run_cache_reader_shadow_timing(newbie, max_files=2)
        assert disabled["skipped"] is True, disabled
        assert disabled["training_path_enabled"] is False, disabled

        old_env = _set_env({ENABLE_ENV: "1"})
        try:
            cleared_header_cache = clear_cache_reader_shadow_header_cache()
            cleared_layout_cache = clear_cache_reader_shadow_payload_layout_cache()
            newbie_report = run_cache_reader_shadow_timing(
                newbie,
                max_files=2,
                max_bytes_per_file=4096,
                buffer_size=4096,
            )
            newbie_session_report = run_cache_reader_shadow_header_session(
                newbie,
                max_files=2,
                max_tensors_per_file=16,
                max_preview=2,
            )
            newbie_layout_report = run_cache_reader_shadow_payload_layout(
                newbie,
                max_files=2,
                max_tensors_per_file=16,
            )
            newbie_payload_read_report = run_cache_reader_shadow_payload_read(
                newbie,
                max_files=2,
                max_tensors_per_file=16,
                max_payload_bytes_per_tensor=64,
                buffer_size=32,
                selected_only=True,
            )
            newbie_full_payload_read_report = run_cache_reader_shadow_payload_read(
                newbie,
                max_files=2,
                max_tensors_per_file=16,
                max_payload_bytes_per_tensor=64,
                buffer_size=32,
                selected_only=True,
                read_full_payload=True,
            )
            newbie_decode_contract_report = run_cache_reader_shadow_tensor_decode_contract(
                newbie,
                max_files=2,
                max_tensors_per_file=16,
                selected_only=True,
            )
            newbie_decode_parity_report = run_cache_reader_shadow_tensor_decode_parity(
                newbie,
                max_files=2,
                max_tensors_per_file=16,
                max_decode_payload_bytes=4096,
                selected_only=True,
            )
            newbie_decode_session_report = run_cache_reader_shadow_tensor_decode_session(
                newbie,
                max_files=2,
                max_tensors_per_file=16,
                max_decode_payload_bytes=4096,
                selected_only=True,
                chunk_size=1,
            )
            decode_manifest = build_cache_reader_shadow_manifest(newbie, max_files=2)
            with NativeCacheReaderDecodeShadowSession(
                decode_manifest,
                max_files=2,
                max_tensors_per_file=16,
                max_decode_payload_bytes=4096,
                selected_only=True,
            ) as decode_session:
                decode_session_stats = decode_session.stats()
                decode_session_first = decode_session.run_chunk(cursor=0, max_tensors=1)
                decode_session_second = decode_session.run_chunk(cursor=decode_session_first["next_cursor"], max_tensors=4)
            with NativeCacheReaderDecodeShadowSession(
                decode_manifest,
                max_files=2,
                max_tensors_per_file=16,
                max_decode_payload_bytes=4096,
                selected_only=True,
            ) as payload_session:
                decode_payload_chunk = payload_session.run_cpu_payload_chunk(
                    cursor=0,
                    max_tensors=1,
                    max_cpu_payload_buffer_bytes=4096,
                )
            with NativeCacheReaderDecodeShadowSession(
                decode_manifest,
                max_files=2,
                max_tensors_per_file=16,
                max_decode_payload_bytes=4096,
                selected_only=True,
            ) as batch_payload_session:
                decode_batch_payload_chunk = batch_payload_session.run_batch_cpu_payload_chunk(
                    cursor=0,
                    max_tensors=2,
                    max_batch_payload_buffer_bytes=4096,
                )
            layout_cache_after_newbie = cache_reader_shadow_payload_layout_cache_stats()
            manifest = build_cached_dataset_prefetch_manifest(newbie)
            with NativeCacheReaderShadowSession(manifest, max_files=2, max_tensors_per_file=16) as session:
                first = session.run(max_preview=2, manifest=manifest)
                second = session.run(max_preview=2, manifest=manifest)
                stats = session.stats()
            header_cache_after_manual = cache_reader_shadow_header_cache_stats()
            with NativeCacheReaderShadowSession(manifest, max_files=64, max_tensors_per_file=128) as session:
                adapter_warmup = session.stats()
            newbie_loader = create_newbie_cached_dataloader(
                newbie,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
            second_loader = create_newbie_cached_dataloader(
                newbie,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
            with np.load(root / "flat_a_newbie.npz") as data:
                arrays = {key: data[key] for key in data.files}
            arrays["latents"] = np.zeros((1, 4, 5, 5), dtype=np.float32)
            np.savez(root / "flat_a_newbie.npz", **arrays)
            third_loader = create_newbie_cached_dataloader(
                newbie,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
        finally:
            _restore_env(old_env)
        attached_newbie = getattr(newbie_loader, "native_cache_reader_shadow_timing")
        attached_newbie_session = getattr(newbie_loader, "native_cache_reader_shadow_session")
        attached_newbie_decode = getattr(newbie_loader, "native_cache_reader_decode_shadow_adapter")
        second_session = getattr(second_loader, "native_cache_reader_shadow_session")
        third_session = getattr(third_loader, "native_cache_reader_shadow_session")
        assert cleared_header_cache["ok"] is True, cleared_header_cache
        assert cleared_layout_cache["ok"] is True, cleared_layout_cache
        assert newbie_report["ok"] is True, newbie_report
        assert newbie_report["shadow_run"] is True, newbie_report
        assert newbie_report["file_count"] == 2, newbie_report
        assert newbie_report["ok_file_count"] == 2, newbie_report
        assert newbie_report["reads_file_bytes"] is True, newbie_report
        assert newbie_report["returns_tensor_payloads"] is False, newbie_report
        assert newbie_report["cache_reader_path_enabled"] is False, newbie_report
        assert newbie_report["training_path_enabled"] is False, newbie_report
        assert attached_newbie["ok"] is True, attached_newbie
        assert attached_newbie["training_path_enabled"] is False, attached_newbie
        assert newbie_session_report["ok"] is True, newbie_session_report
        assert newbie_session_report["reader_probe"]["summary"]["shape_count"] >= 2, newbie_session_report
        assert newbie_session_report["reader_probe"]["preview"][0]["parses_cache_headers"] is True, newbie_session_report
        assert newbie_session_report["reader_probe"]["returns_tensor_payloads"] is False, newbie_session_report
        assert newbie_session_report["reader_probe"]["training_path_enabled"] is False, newbie_session_report
        assert newbie_layout_report["ok"] is True, newbie_layout_report
        assert newbie_layout_report["payload_layout_contract"] is True, newbie_layout_report
        assert newbie_layout_report["data_payload_bytes_read"] == 0, newbie_layout_report
        assert newbie_layout_report["layout_cache"]["misses"] >= 2, newbie_layout_report
        assert newbie_layout_report["layout_cache"]["stored"] >= 2, newbie_layout_report
        assert newbie_layout_report["returns_tensor_payloads"] is False, newbie_layout_report
        assert newbie_layout_report["records"][0]["selected_latent_key"] == "latents", newbie_layout_report
        assert newbie_layout_report["records"][0]["selected_latent_shape"] == [1, 4, 4, 4], newbie_layout_report
        selected_layout = newbie_layout_report["records"][0]["selected_layout"]
        assert selected_layout["dtype"] == "<f4", selected_layout
        assert selected_layout["data_payload_bytes"] == 256, selected_layout
        assert selected_layout["data_payload_offset"] > 0, selected_layout
        assert newbie_payload_read_report["ok"] is True, newbie_payload_read_report
        assert newbie_payload_read_report["payload_read_shadow"] is True, newbie_payload_read_report
        assert newbie_payload_read_report["reads_tensor_payload_bytes"] is True, newbie_payload_read_report
        assert newbie_payload_read_report["parses_tensor_payloads"] is False, newbie_payload_read_report
        assert newbie_payload_read_report["returns_tensor_payloads"] is False, newbie_payload_read_report
        assert newbie_payload_read_report["data_payload_bytes_read"] == 128, newbie_payload_read_report
        assert newbie_payload_read_report["layout_cache"]["hits"] >= 2, newbie_payload_read_report
        assert newbie_payload_read_report["layout_report"]["total_header_bytes_read"] == 0, newbie_payload_read_report
        assert newbie_payload_read_report["tensor_read_count"] == 2, newbie_payload_read_report
        assert newbie_payload_read_report["records"][0]["tensor_key"] == "latents", newbie_payload_read_report
        assert newbie_payload_read_report["records"][0]["data_payload_bytes_read"] == 64, newbie_payload_read_report
        assert newbie_payload_read_report["records"][0]["read_limited"] is True, newbie_payload_read_report
        assert newbie_payload_read_report["training_path_enabled"] is False, newbie_payload_read_report
        assert newbie_full_payload_read_report["ok"] is True, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["payload_read_shadow"] is True, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["full_payload_read_shadow"] is True, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["payload_read_mode"] == "full_selected_payload", newbie_full_payload_read_report
        assert newbie_full_payload_read_report["layout_cache"]["hits"] >= 2, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["layout_report"]["total_header_bytes_read"] == 0, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["parses_tensor_payloads"] is False, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["returns_tensor_payloads"] is False, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["data_payload_bytes_read"] == 512, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["total_declared_payload_bytes"] == 512, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["full_payload_complete"] is True, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["records"][0]["data_payload_bytes_read"] == 256, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["records"][0]["read_limited"] is False, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["records"][0]["full_payload_complete"] is True, newbie_full_payload_read_report
        assert newbie_full_payload_read_report["training_path_enabled"] is False, newbie_full_payload_read_report
        assert newbie_decode_contract_report["ok"] is True, newbie_decode_contract_report
        assert newbie_decode_contract_report["tensor_decode_contract"] is True, newbie_decode_contract_report
        assert newbie_decode_contract_report["tensor_contract_count"] == 2, newbie_decode_contract_report
        assert newbie_decode_contract_report["data_payload_bytes_read"] == 0, newbie_decode_contract_report
        assert newbie_decode_contract_report["reads_tensor_payload_bytes"] is False, newbie_decode_contract_report
        assert newbie_decode_contract_report["parses_tensor_payloads"] is False, newbie_decode_contract_report
        assert newbie_decode_contract_report["returns_tensor_payloads"] is False, newbie_decode_contract_report
        assert newbie_decode_contract_report["layout_cache"]["hits"] >= 2, newbie_decode_contract_report
        newbie_contract = newbie_decode_contract_report["records"][0]
        assert newbie_contract["tensor_key"] == "latents", newbie_contract
        assert newbie_contract["canonical_dtype"] == "float32", newbie_contract
        assert newbie_contract["shape"] == [1, 4, 4, 4], newbie_contract
        assert newbie_contract["strides_elements"] == [64, 16, 4, 1], newbie_contract
        assert newbie_contract["expected_payload_bytes"] == 256, newbie_contract
        assert newbie_contract["payload_bytes_match"] is True, newbie_contract
        assert newbie_contract["contract_supported"] is True, newbie_contract
        assert newbie_decode_contract_report["training_path_enabled"] is False, newbie_decode_contract_report
        assert newbie_decode_parity_report["ok"] is True, newbie_decode_parity_report
        assert newbie_decode_parity_report["tensor_decode_count"] == 2, newbie_decode_parity_report
        assert newbie_decode_parity_report["data_payload_bytes_read"] == 512, newbie_decode_parity_report
        assert newbie_decode_parity_report["returns_tensor_payloads"] is False, newbie_decode_parity_report
        newbie_decode = newbie_decode_parity_report["records"][0]
        assert newbie_decode["tensor_key"] == "latents", newbie_decode
        assert newbie_decode["canonical_dtype"] == "float32", newbie_decode
        assert newbie_decode["decoded_element_count"] == 64, newbie_decode
        assert newbie_decode["decoded_finite_count"] == 64, newbie_decode
        assert newbie_decode["decoded_sum"] == 2016.0, newbie_decode
        assert newbie_decode["decoded_min"] == 0.0, newbie_decode
        assert newbie_decode["decoded_max"] == 63.0, newbie_decode
        assert newbie_decode["sample_values"] == [0.0, 1.0, 2.0, 3.0], newbie_decode
        assert newbie_decode["decode_ok"] is True, newbie_decode
        assert newbie_decode_parity_report["training_path_enabled"] is False, newbie_decode_parity_report
        assert newbie_decode_session_report["ok"] is True, newbie_decode_session_report
        assert newbie_decode_session_report["chunk_count"] == 2, newbie_decode_session_report
        assert newbie_decode_session_report["tensor_decode_count"] == 2, newbie_decode_session_report
        assert newbie_decode_session_report["data_payload_bytes_read"] == 512, newbie_decode_session_report
        assert newbie_decode_session_report["returns_tensor_payloads"] is False, newbie_decode_session_report
        assert decode_session_stats["ok"] is True, decode_session_stats
        assert decode_session_stats["summary"]["tensor_candidate_count"] == 2, decode_session_stats
        assert decode_session_stats["summary"]["data_payload_bytes_read"] == 0, decode_session_stats
        assert decode_session_first["ok"] is True, decode_session_first
        assert decode_session_first["cursor"] == 0, decode_session_first
        assert decode_session_first["next_cursor"] == 1, decode_session_first
        assert decode_session_first["chunk_complete"] is False, decode_session_first
        assert decode_session_first["data_payload_bytes_read"] == 256, decode_session_first
        assert decode_session_first["parses_cache_headers"] is False, decode_session_first
        assert decode_session_second["ok"] is True, decode_session_second
        assert decode_session_second["session_reused"] is True, decode_session_second
        assert decode_session_second["chunk_complete"] is True, decode_session_second
        assert decode_session_second["data_payload_bytes_read"] == 256, decode_session_second
        assert decode_payload_chunk["ok"] is True, decode_payload_chunk
        assert decode_payload_chunk["cpu_payload_buffer_shadow"] is True, decode_payload_chunk
        assert decode_payload_chunk["cpu_payload_buffer_tensor_count"] == 1, decode_payload_chunk
        assert decode_payload_chunk["cpu_payload_buffer_byte_count"] == 256, decode_payload_chunk
        payload_record = decode_payload_chunk["records"][0]
        payload_bytes = payload_record["cpu_payload_buffer_bytes"]
        assert isinstance(payload_bytes, bytes), decode_payload_chunk
        assert payload_record["returns_cpu_payload_buffer"] is True, payload_record
        assert payload_record["cpu_payload_buffer_byte_count"] == 256, payload_record
        payload_view = np.frombuffer(memoryview(payload_bytes), dtype=np.float32)
        assert payload_view.shape == (64,), payload_view.shape
        assert float(payload_view.sum()) == 2016.0, payload_view
        assert decode_payload_chunk["returns_tensor_payloads"] is False, decode_payload_chunk
        assert decode_payload_chunk["training_path_enabled"] is False, decode_payload_chunk
        assert decode_batch_payload_chunk["ok"] is True, decode_batch_payload_chunk
        batch_payload = decode_batch_payload_chunk["batch_cpu_payload_shadow"]
        assert batch_payload["batch_cpu_payload_ready"] is True, decode_batch_payload_chunk
        assert batch_payload["batch_cpu_payload_byte_count"] == 512, decode_batch_payload_chunk
        assert batch_payload["shape"] == [2, 4, 4, 4], decode_batch_payload_chunk
        assert batch_payload["batch_layout"] == "contiguous_nchw", decode_batch_payload_chunk
        batch_payload_bytes = batch_payload["batch_cpu_payload_bytes"]
        assert isinstance(batch_payload_bytes, bytes), decode_batch_payload_chunk
        batch_view = np.frombuffer(memoryview(batch_payload_bytes), dtype=np.float32).reshape(2, 4, 4, 4)
        assert batch_view.shape == (2, 4, 4, 4), batch_view.shape
        assert float(batch_view.sum()) == 4032.0, batch_view
        assert batch_payload["returns_cpu_payload_buffer"] is True, batch_payload
        assert decode_batch_payload_chunk["returns_cpu_payload_buffer"] is False, decode_batch_payload_chunk
        assert decode_batch_payload_chunk["returns_tensor_payloads"] is False, decode_batch_payload_chunk
        assert decode_batch_payload_chunk["training_path_enabled"] is False, decode_batch_payload_chunk
        assert layout_cache_after_newbie["entry_count"] >= 2, layout_cache_after_newbie
        first_header_cache = newbie_session_report["session_create"]["summary"]["header_cache"]
        assert first_header_cache["misses"] >= 2, first_header_cache
        assert first_header_cache["stored"] >= 2, first_header_cache
        assert first["ok"] is True, first
        assert second["session_reused"] is True, second
        assert stats["run_count"] == 2, stats
        assert first["summary"]["data_payload_bytes_read"] == 0, first
        manual_header_cache = stats["summary"]["header_cache"]
        assert manual_header_cache["hits"] >= 2, manual_header_cache
        assert manual_header_cache["reused_header_bytes"] > 0, manual_header_cache
        assert header_cache_after_manual["entry_count"] >= 2, header_cache_after_manual
        assert adapter_warmup["summary"]["header_cache"]["misses"] >= 2, adapter_warmup
        assert attached_newbie_session["ok"] is True, attached_newbie_session
        assert attached_newbie_session["persistent_session"] is True, attached_newbie_session
        assert attached_newbie_session["session_reused_by_adapter"] is False, attached_newbie_session
        assert attached_newbie_decode["ok"] is True, attached_newbie_decode
        assert attached_newbie_decode["sidecar_only"] is True, attached_newbie_decode
        assert attached_newbie_decode["batch_size"] == 1, attached_newbie_decode
        assert attached_newbie_decode["tensor_decode_count"] == 2, attached_newbie_decode
        assert attached_newbie_decode["chunk_count"] == 2, attached_newbie_decode
        assert attached_newbie_decode["data_payload_bytes_read"] == 512, attached_newbie_decode
        assert attached_newbie_decode["returns_tensor_payloads"] is False, attached_newbie_decode
        assert attached_newbie_decode["training_path_enabled"] is False, attached_newbie_decode
        assert second_session["persistent_session"] is True, second_session
        assert second_session["session_reused_by_adapter"] is True, second_session
        assert second_session["reader_probe"]["session_reused"] is True, second_session
        attached_header_cache = attached_newbie_session["session_create"]["summary"]["header_cache"]
        assert attached_header_cache["hits"] >= 2, attached_header_cache
        assert attached_newbie_session["reader_probe"]["summary"]["total_header_bytes_read"] == 0, attached_newbie_session
        assert third_session["persistent_session"] is True, third_session
        assert third_session["session_reused_by_adapter"] is False, third_session
        assert third_session["requires_rebuild"] is True, third_session
        assert third_session["reader_probe"]["session_reused"] is False, third_session
        rebuild_header_cache = third_session["session_create"]["summary"]["header_cache"]
        assert rebuild_header_cache["hits"] >= 1, rebuild_header_cache
        assert rebuild_header_cache["misses"] >= 1, rebuild_header_cache
        assert third_session["training_path_enabled"] is False, third_session
        close_cache_reader_shadow_session(newbie)

        safetensors_path = _write_safetensors_cache(root, "layout_sample")
        safetensors_dataset = _CacheDataset(root, safetensors_path)
        old_env = _set_env({ENABLE_ENV: "1"})
        try:
            safetensors_layout_report = run_cache_reader_shadow_payload_layout(
                safetensors_dataset,
                max_files=1,
                max_tensors_per_file=16,
            )
        finally:
            _restore_env(old_env)
        assert safetensors_layout_report["ok"] is True, safetensors_layout_report
        assert safetensors_layout_report["format_counts"][".safetensors"] == 1, safetensors_layout_report
        assert safetensors_layout_report["data_payload_bytes_read"] == 0, safetensors_layout_report
        assert safetensors_layout_report["records"][0]["selected_latent_key"] == "latents", safetensors_layout_report
        safetensors_selected = safetensors_layout_report["records"][0]["selected_layout"]
        assert safetensors_selected["dtype"] == "F32", safetensors_selected
        assert safetensors_selected["shape"] == [1, 4, 2, 2], safetensors_selected
        assert safetensors_selected["data_payload_bytes"] == 64, safetensors_selected
        assert safetensors_selected["data_payload_offset"] > 8, safetensors_selected
        old_env = _set_env({ENABLE_ENV: "1"})
        try:
            safetensors_payload_read_report = run_cache_reader_shadow_payload_read(
                safetensors_dataset,
                max_files=1,
                max_tensors_per_file=16,
                max_payload_bytes_per_tensor=32,
                buffer_size=16,
                selected_only=True,
            )
            safetensors_full_payload_read_report = run_cache_reader_shadow_payload_read(
                safetensors_dataset,
                max_files=1,
                max_tensors_per_file=16,
                max_payload_bytes_per_tensor=32,
                buffer_size=16,
                selected_only=True,
                read_full_payload=True,
            )
            safetensors_decode_contract_report = run_cache_reader_shadow_tensor_decode_contract(
                safetensors_dataset,
                max_files=1,
                max_tensors_per_file=16,
                selected_only=True,
            )
            safetensors_decode_parity_report = run_cache_reader_shadow_tensor_decode_parity(
                safetensors_dataset,
                max_files=1,
                max_tensors_per_file=16,
                max_decode_payload_bytes=4096,
                selected_only=True,
            )
        finally:
            _restore_env(old_env)
        assert safetensors_payload_read_report["ok"] is True, safetensors_payload_read_report
        assert safetensors_payload_read_report["tensor_read_count"] == 1, safetensors_payload_read_report
        assert safetensors_payload_read_report["data_payload_bytes_read"] == 32, safetensors_payload_read_report
        assert safetensors_payload_read_report["records"][0]["tensor_key"] == "latents", safetensors_payload_read_report
        assert safetensors_payload_read_report["records"][0]["read_limited"] is True, safetensors_payload_read_report
        assert safetensors_full_payload_read_report["ok"] is True, safetensors_full_payload_read_report
        assert safetensors_full_payload_read_report["full_payload_read_shadow"] is True, safetensors_full_payload_read_report
        assert safetensors_full_payload_read_report["tensor_read_count"] == 1, safetensors_full_payload_read_report
        assert safetensors_full_payload_read_report["data_payload_bytes_read"] == 64, safetensors_full_payload_read_report
        assert safetensors_full_payload_read_report["full_payload_complete"] is True, safetensors_full_payload_read_report
        assert safetensors_full_payload_read_report["records"][0]["read_limited"] is False, safetensors_full_payload_read_report
        assert safetensors_full_payload_read_report["parses_tensor_payloads"] is False, safetensors_full_payload_read_report
        assert safetensors_full_payload_read_report["returns_tensor_payloads"] is False, safetensors_full_payload_read_report
        assert safetensors_decode_contract_report["ok"] is True, safetensors_decode_contract_report
        assert safetensors_decode_contract_report["tensor_contract_count"] == 1, safetensors_decode_contract_report
        safetensors_contract = safetensors_decode_contract_report["records"][0]
        assert safetensors_contract["canonical_dtype"] == "float32", safetensors_contract
        assert safetensors_contract["shape"] == [1, 4, 2, 2], safetensors_contract
        assert safetensors_contract["strides_elements"] == [16, 4, 2, 1], safetensors_contract
        assert safetensors_contract["expected_payload_bytes"] == 64, safetensors_contract
        assert safetensors_contract["returns_tensor_payloads"] is False, safetensors_contract
        assert safetensors_decode_parity_report["ok"] is True, safetensors_decode_parity_report
        assert safetensors_decode_parity_report["tensor_decode_count"] == 1, safetensors_decode_parity_report
        assert safetensors_decode_parity_report["data_payload_bytes_read"] == 64, safetensors_decode_parity_report
        safetensors_decode = safetensors_decode_parity_report["records"][0]
        assert safetensors_decode["decoded_element_count"] == 16, safetensors_decode
        assert safetensors_decode["decoded_sum"] == 120.0, safetensors_decode
        assert safetensors_decode["decoded_max"] == 15.0, safetensors_decode
        assert safetensors_decode["returns_tensor_payloads"] is False, safetensors_decode
        assert safetensors_decode_parity_report["training_path_enabled"] is False, safetensors_decode_parity_report

        anima_root = root / "anima"
        anima_root.mkdir()
        _write_anima_cache(anima_root, "anime_a", 4)
        _write_anima_cache(anima_root, "anime_b", 5)
        anima = AnimaCachedDataset(anima_root, enable_bucket=False, cache_mmap=True, cache_lazy=True)
        old_env = _set_env({ENABLE_ENV: "1"})
        try:
            anima_loader = create_anima_cached_dataloader(
                anima,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            anima_layout_report = run_cache_reader_shadow_payload_layout(
                anima,
                max_files=4,
                max_tensors_per_file=16,
            )
            anima_payload_read_report = run_cache_reader_shadow_payload_read(
                anima,
                max_files=4,
                max_tensors_per_file=16,
                max_payload_bytes_per_tensor=32,
                buffer_size=16,
                selected_only=True,
            )
            anima_full_payload_read_report = run_cache_reader_shadow_payload_read(
                anima,
                max_files=4,
                max_tensors_per_file=16,
                max_payload_bytes_per_tensor=32,
                buffer_size=16,
                selected_only=True,
                read_full_payload=True,
            )
            anima_decode_contract_report = run_cache_reader_shadow_tensor_decode_contract(
                anima,
                max_files=4,
                max_tensors_per_file=16,
                selected_only=True,
            )
            anima_decode_parity_report = run_cache_reader_shadow_tensor_decode_parity(
                anima,
                max_files=4,
                max_tensors_per_file=16,
                max_decode_payload_bytes=4096,
                selected_only=True,
            )
        finally:
            _restore_env(old_env)
        anima_report = getattr(anima_loader, "native_cache_reader_shadow_timing")
        anima_session_report = getattr(anima_loader, "native_cache_reader_shadow_session")
        anima_decode_sidecar = getattr(anima_loader, "native_cache_reader_decode_shadow_adapter")
        assert anima_report["ok"] is True, anima_report
        assert anima_report["file_count"] == 4, anima_report
        assert anima_report["role_counts"]["latent"] == 2, anima_report
        assert anima_report["role_counts"]["text"] == 2, anima_report
        assert anima_report["parses_tensor_payloads"] is False, anima_report
        assert anima_report["training_path_enabled"] is False, anima_report
        assert anima_session_report["ok"] is True, anima_session_report
        assert anima_session_report["reader_probe"]["summary"]["file_count"] == 4, anima_session_report
        assert anima_session_report["reader_probe"]["summary"]["selected_latent_count"] >= 2, anima_session_report
        assert anima_session_report["reader_probe"]["parses_tensor_payloads"] is False, anima_session_report
        assert anima_session_report["reader_probe"]["training_path_enabled"] is False, anima_session_report
        assert anima_decode_sidecar["ok"] is True, anima_decode_sidecar
        assert anima_decode_sidecar["sidecar_only"] is True, anima_decode_sidecar
        assert anima_decode_sidecar["batch_size"] == 2, anima_decode_sidecar
        assert anima_decode_sidecar["tensor_decode_count"] == 2, anima_decode_sidecar
        assert anima_decode_sidecar["data_payload_bytes_read"] == 656, anima_decode_sidecar
        assert anima_decode_sidecar["training_path_enabled"] is False, anima_decode_sidecar
        assert anima_layout_report["ok"] is True, anima_layout_report
        assert anima_layout_report["selected_latent_count"] >= 2, anima_layout_report
        assert anima_layout_report["data_payload_bytes_read"] == 0, anima_layout_report
        assert anima_layout_report["records"][0]["selected_latent_key"] == "latents_4", anima_layout_report
        assert anima_layout_report["records"][0]["selected_latent_shape"] == [4, 4, 4], anima_layout_report
        assert anima_payload_read_report["ok"] is True, anima_payload_read_report
        assert anima_payload_read_report["tensor_read_count"] == 2, anima_payload_read_report
        assert anima_payload_read_report["skipped_no_selected_layout_count"] == 2, anima_payload_read_report
        assert anima_payload_read_report["data_payload_bytes_read"] == 64, anima_payload_read_report
        assert anima_payload_read_report["training_path_enabled"] is False, anima_payload_read_report
        assert anima_full_payload_read_report["ok"] is True, anima_full_payload_read_report
        assert anima_full_payload_read_report["tensor_read_count"] == 2, anima_full_payload_read_report
        assert anima_full_payload_read_report["skipped_no_selected_layout_count"] == 2, anima_full_payload_read_report
        assert anima_full_payload_read_report["data_payload_bytes_read"] == 656, anima_full_payload_read_report
        assert anima_full_payload_read_report["total_declared_payload_bytes"] == 656, anima_full_payload_read_report
        assert anima_full_payload_read_report["full_payload_complete"] is True, anima_full_payload_read_report
        assert anima_full_payload_read_report["returns_tensor_payloads"] is False, anima_full_payload_read_report
        assert anima_decode_contract_report["ok"] is True, anima_decode_contract_report
        assert anima_decode_contract_report["tensor_contract_count"] == 2, anima_decode_contract_report
        assert anima_decode_contract_report["skipped_no_selected_layout_count"] == 2, anima_decode_contract_report
        assert anima_decode_contract_report["data_payload_bytes_read"] == 0, anima_decode_contract_report
        anima_contract = anima_decode_contract_report["records"][0]
        assert anima_contract["tensor_key"] == "latents_4", anima_contract
        assert anima_contract["shape"] == [4, 4, 4], anima_contract
        assert anima_contract["strides_elements"] == [16, 4, 1], anima_contract
        assert anima_contract["expected_payload_bytes"] == 256, anima_contract
        assert anima_decode_parity_report["ok"] is True, anima_decode_parity_report
        assert anima_decode_parity_report["tensor_decode_count"] == 2, anima_decode_parity_report
        assert anima_decode_parity_report["skipped_no_selected_layout_count"] == 2, anima_decode_parity_report
        assert anima_decode_parity_report["data_payload_bytes_read"] == 656, anima_decode_parity_report
        anima_decode = anima_decode_parity_report["records"][0]
        assert anima_decode["tensor_key"] == "latents_4", anima_decode
        assert anima_decode["decoded_element_count"] == 64, anima_decode
        assert anima_decode["decoded_sum"] == 2016.0, anima_decode
        assert anima_decode["decoded_max"] == 63.0, anima_decode
        assert anima_decode_parity_report["training_path_enabled"] is False, anima_decode_parity_report

        return {
            "schema_version": 1,
            "probe": "turbocore_cache_reader_shadow_smoke",
            "ok": True,
            "newbie_decode_session_chunks": newbie_decode_session_report["chunk_count"],
            "newbie_decode_session_bytes": newbie_decode_session_report["data_payload_bytes_read"],
            "newbie_cpu_payload_buffer_bytes": decode_payload_chunk["cpu_payload_buffer_byte_count"],
            "newbie_batch_cpu_payload_buffer_bytes": batch_payload["batch_cpu_payload_byte_count"],
            "newbie_sidecar_bytes": attached_newbie_decode["data_payload_bytes_read"],
            "anima_decode_bytes": anima_decode_parity_report["data_payload_bytes_read"],
            "anima_sidecar_bytes": anima_decode_sidecar["data_payload_bytes_read"],
            "safetensors_decode_bytes": safetensors_decode_parity_report["data_payload_bytes_read"],
            "training_path_enabled": False,
            "disabled_reason": disabled.get("reason"),
        }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
