"""Compatibility facade for debug-only cache reader shadow probes."""

from __future__ import annotations

from core.turbocore_cache_reader_shadow_adapter import maybe_attach_cache_reader_shadow_timing
from core.turbocore_cache_reader_shadow_adapter import run_cache_reader_decode_sidecar_adapter
from core.turbocore_cache_reader_shadow_layout import (
    NativeCacheReaderDecodeShadowSession,
    run_cache_reader_shadow_payload_layout,
    run_cache_reader_shadow_payload_read,
    run_cache_reader_shadow_tensor_decode_contract,
    run_cache_reader_shadow_tensor_decode_parity,
    run_cache_reader_shadow_tensor_decode_session,
)
from core.turbocore_cache_reader_shadow_manifest import build_cache_reader_shadow_manifest
from core.turbocore_cache_reader_shadow_native import (
    DISABLE_ENV,
    ENABLE_ENV,
    cache_reader_shadow_header_cache_stats,
    cache_reader_shadow_payload_layout_cache_stats,
    clear_cache_reader_shadow_header_cache,
    clear_cache_reader_shadow_payload_layout_cache,
)
from core.turbocore_cache_reader_shadow_session import (
    NativeCacheReaderShadowSession,
    close_cache_reader_shadow_session,
    create_cache_reader_shadow_session,
    run_cache_reader_shadow_header_session,
)
from core.turbocore_cache_reader_shadow_timing import run_cache_reader_shadow_timing
from core.turbocore_cache_reader_training_gate import (
    BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV,
    CPU_PAYLOAD_BUFFER_BYTES_ENV,
    DISABLE_EXPERIMENTAL_ENV,
    ENABLE_EXPERIMENTAL_ENV,
    PARITY_BATCHES_ENV,
    PARITY_MAX_BYTES_ENV,
    maybe_attach_cache_reader_training_experimental_gate,
    run_cache_reader_training_experimental_gate,
)


__all__ = [
    "DISABLE_ENV",
    "DISABLE_EXPERIMENTAL_ENV",
    "ENABLE_ENV",
    "ENABLE_EXPERIMENTAL_ENV",
    "NativeCacheReaderShadowSession",
    "NativeCacheReaderDecodeShadowSession",
    "BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV",
    "CPU_PAYLOAD_BUFFER_BYTES_ENV",
    "PARITY_BATCHES_ENV",
    "PARITY_MAX_BYTES_ENV",
    "build_cache_reader_shadow_manifest",
    "cache_reader_shadow_header_cache_stats",
    "cache_reader_shadow_payload_layout_cache_stats",
    "close_cache_reader_shadow_session",
    "create_cache_reader_shadow_session",
    "clear_cache_reader_shadow_header_cache",
    "clear_cache_reader_shadow_payload_layout_cache",
    "maybe_attach_cache_reader_shadow_timing",
    "maybe_attach_cache_reader_training_experimental_gate",
    "run_cache_reader_decode_sidecar_adapter",
    "run_cache_reader_shadow_header_session",
    "run_cache_reader_shadow_payload_layout",
    "run_cache_reader_shadow_payload_read",
    "run_cache_reader_shadow_tensor_decode_contract",
    "run_cache_reader_shadow_tensor_decode_parity",
    "run_cache_reader_shadow_tensor_decode_session",
    "run_cache_reader_shadow_timing",
    "run_cache_reader_training_experimental_gate",
]
