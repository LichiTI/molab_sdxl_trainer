"""Compact reporting helpers for native cache reader decode sidecars."""

from __future__ import annotations

from typing import Any

from core.turbocore_cache_reader_payload_ownership import compact_payload_ownership_shadow


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _compact_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _as_int(count) for key, count in value.items()}


def compact_cache_reader_decode_sidecar_profile(
    report: Any,
    *,
    route: str = "",
    source: str = "",
) -> dict[str, Any]:
    """Return a telemetry-safe summary of a debug-only decode sidecar report."""
    if not isinstance(report, dict) or not report:
        return {}

    profile: dict[str, Any] = {
        "schema_version": 1,
        "provider": str(report.get("provider") or ""),
        "ok": bool(report.get("ok", False)),
        "route": str(route or ""),
        "source": str(source or ""),
        "native_runtime": bool(report.get("native_runtime", False)),
        "debug_only": bool(report.get("debug_only", True)),
        "shadow_run": bool(report.get("shadow_run", True)),
        "sidecar_only": bool(report.get("sidecar_only", False)),
        "batch_size": _as_int(report.get("batch_size")),
        "planned_shadow_batches": _as_int(report.get("planned_shadow_batches")),
        "chunk_count": _as_int(report.get("chunk_count")),
        "tensor_decode_count": _as_int(report.get("tensor_decode_count")),
        "data_payload_bytes_read": _as_int(report.get("data_payload_bytes_read")),
        "worker_count": _as_int(report.get("worker_count")),
        "reads_tensor_payload_bytes": bool(report.get("reads_tensor_payload_bytes", False)),
        "parses_tensor_payloads": bool(report.get("parses_tensor_payloads", False)),
        "decodes_tensor_payloads": bool(report.get("decodes_tensor_payloads", False)),
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }

    if report.get("prefetch_factor") is not None:
        profile["prefetch_factor"] = max(_as_int(report.get("prefetch_factor")), 1)
    if report.get("reason"):
        profile["reason"] = str(report.get("reason") or "")
    if report.get("native_error"):
        profile["native_error"] = str(report.get("native_error") or "")

    session_summary = report.get("session_summary")
    if isinstance(session_summary, dict):
        profile["tensor_candidate_count"] = _as_int(session_summary.get("tensor_candidate_count"))
        profile["total_declared_payload_bytes"] = _as_int(session_summary.get("total_declared_payload_bytes"))
        format_counts = _compact_counts(session_summary.get("format_counts"))
        if format_counts:
            profile["format_counts"] = format_counts
        role_counts = _compact_counts(session_summary.get("role_counts"))
        if role_counts:
            profile["role_counts"] = role_counts

        layout_cache = session_summary.get("layout_cache")
        if isinstance(layout_cache, dict):
            profile["layout_cache"] = {
                "hits": _as_int(layout_cache.get("hits")),
                "misses": _as_int(layout_cache.get("misses")),
                "stored": _as_int(layout_cache.get("stored")),
                "entry_count": _as_int(layout_cache.get("entry_count")),
                "reused_header_bytes": _as_int(layout_cache.get("reused_header_bytes")),
                "training_path_enabled": False,
            }

    return profile


def compact_cache_reader_training_gate_profile(
    report: Any,
    *,
    route: str = "",
    source: str = "",
) -> dict[str, Any]:
    """Return a telemetry-safe summary of the experimental training gate."""
    if not isinstance(report, dict) or not report:
        return {}

    profile: dict[str, Any] = {
        "schema_version": 1,
        "provider": str(report.get("provider") or ""),
        "ok": bool(report.get("ok", False)),
        "route": str(route or ""),
        "source": str(source or ""),
        "experimental_gate": bool(report.get("experimental_gate", False)),
        "native_runtime": bool(report.get("native_runtime", False)),
        "dataset_class": str(report.get("dataset_class") or ""),
        "sample_count": _as_int(report.get("sample_count")),
        "batch_size": _as_int(report.get("batch_size")),
        "planned_parity_batches": _as_int(report.get("planned_parity_batches")),
        "cpu_payload_buffer_shadow": bool(report.get("cpu_payload_buffer_shadow", False)),
        "max_cpu_payload_buffer_bytes": _as_int(report.get("max_cpu_payload_buffer_bytes")),
        "batch_cpu_payload_buffer_shadow": bool(report.get("batch_cpu_payload_buffer_shadow", False)),
        "max_batch_cpu_payload_buffer_bytes": _as_int(report.get("max_batch_cpu_payload_buffer_bytes")),
        "batch_handoff_session_shadow": bool(report.get("batch_handoff_session_shadow", False)),
        "batch_handoff_session_shadow_ran": bool(report.get("batch_handoff_session_shadow_ran", False)),
        "batch_handoff_session_shadow_passed": bool(report.get("batch_handoff_session_shadow_passed", False)),
        "text_payload_parity_shadow": bool(report.get("text_payload_parity_shadow", False)),
        "text_payload_parity_guard_ran": bool(report.get("text_payload_parity_guard_ran", False)),
        "text_payload_parity_guard_passed": bool(report.get("text_payload_parity_guard_passed", False)),
        "max_text_payload_buffer_bytes": _as_int(report.get("max_text_payload_buffer_bytes")),
        "batch_dispatch_contract_shadow": bool(report.get("batch_dispatch_contract_shadow", False)),
        "batch_dispatch_contract_shadow_ran": bool(report.get("batch_dispatch_contract_shadow_ran", False)),
        "batch_dispatch_contract_ready": bool(report.get("batch_dispatch_contract_ready", False)),
        "batch_dispatch_contract_would_allow_native_dispatch": False,
        "native_dispatch_eligible": False,
        "native_dispatch_blockers": [str(item) for item in list(report.get("native_dispatch_blockers", []) or [])],
        "dispatch_eligibility_shadow_gate_ready": bool(report.get("dispatch_eligibility_shadow_gate_ready", False)),
        "parity_guard_ran": bool(report.get("parity_guard_ran", False)),
        "parity_guard_passed": bool(report.get("parity_guard_passed", False)),
        "batch_parity_guard_ran": bool(report.get("batch_parity_guard_ran", False)),
        "batch_parity_guard_passed": bool(report.get("batch_parity_guard_passed", False)),
        "batch_payload_parity_guard_ran": bool(report.get("batch_payload_parity_guard_ran", False)),
        "batch_payload_parity_guard_passed": bool(report.get("batch_payload_parity_guard_passed", False)),
        "torch_tensor_handoff_guard_ran": bool(report.get("torch_tensor_handoff_guard_ran", False)),
        "torch_tensor_handoff_guard_passed": bool(report.get("torch_tensor_handoff_guard_passed", False)),
        "torch_owned_tensor_handoff_guard_ran": bool(report.get("torch_owned_tensor_handoff_guard_ran", False)),
        "torch_owned_tensor_handoff_guard_passed": bool(report.get("torch_owned_tensor_handoff_guard_passed", False)),
        "training_experimental_allowed": bool(report.get("training_experimental_allowed", False)),
        "tensor_parity_count": _as_int(report.get("tensor_parity_count")),
        "tensor_parity_matches": _as_int(report.get("tensor_parity_matches")),
        "mismatch_count": _as_int(report.get("mismatch_count")),
        "native_data_payload_bytes_read": _as_int(report.get("native_data_payload_bytes_read")),
        "python_data_payload_bytes_read": _as_int(report.get("python_data_payload_bytes_read")),
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }
    if report.get("reason"):
        profile["reason"] = str(report.get("reason") or "")
    blocked = [str(item) for item in list(report.get("blocked_reasons", []) or [])]
    if blocked:
        profile["blocked_reasons"] = blocked
    if report.get("native_error"):
        profile["native_error"] = str(report.get("native_error") or "")

    eligibility = report.get("dispatch_eligibility")
    if isinstance(eligibility, dict) and eligibility:
        profile["dispatch_eligibility"] = {
            "provider": str(eligibility.get("provider") or ""),
            "dataset_class": str(eligibility.get("dataset_class") or ""),
            "dataset_supported": bool(eligibility.get("dataset_supported", False)),
            "shadow_gate_ready": bool(eligibility.get("shadow_gate_ready", False)),
            "shadow_gate_blockers": [str(item) for item in list(eligibility.get("shadow_gate_blockers", []) or [])],
            "native_dispatch_eligible": False,
            "native_dispatch_blockers": [str(item) for item in list(eligibility.get("native_dispatch_blockers", []) or [])],
            "strict_fallback": bool(eligibility.get("strict_fallback", False)),
            "strict_fallback_passed": bool(eligibility.get("strict_fallback_passed", True)),
            "fallback_to_python_batch": True,
            "training_path_enabled": False,
        }

    native_probe = report.get("native_probe")
    if isinstance(native_probe, dict):
        profile["native_probe"] = {
            "chunk_count": _as_int(native_probe.get("chunk_count")),
            "tensor_decode_count": _as_int(native_probe.get("tensor_decode_count")),
            "data_payload_bytes_read": _as_int(native_probe.get("data_payload_bytes_read")),
            "training_path_enabled": False,
        }
        native_batch = native_probe.get("native_latent_batch_summary")
        if isinstance(native_batch, dict) and native_batch:
            profile["native_batch_summary"] = {
                "provider": str(native_batch.get("provider") or ""),
                "ready": bool(native_batch.get("batch_summary_ready", False)),
                "shape": list(native_batch.get("shape", []) or []),
                "canonical_dtype": str(native_batch.get("canonical_dtype") or ""),
                "source_tensor_count": _as_int(native_batch.get("source_tensor_count")),
                "native_batch_materialization_contract": bool(native_batch.get("native_batch_materialization_contract", False)),
                "materialization_contract_supported": bool(native_batch.get("materialization_contract_supported", False)),
                "cpu_buffer_bytes": _as_int(native_batch.get("cpu_buffer_bytes")),
                "cpu_payload_preview_shadow": bool(native_batch.get("cpu_payload_preview_shadow", False)),
                "payload_preview_byte_count": _as_int(native_batch.get("payload_preview_byte_count")),
                "payload_preview_tensor_count": _as_int(native_batch.get("payload_preview_tensor_count")),
                "cpu_payload_buffer_shadow": bool(native_batch.get("cpu_payload_buffer_shadow", False)),
                "cpu_payload_buffer_byte_count": _as_int(native_batch.get("cpu_payload_buffer_byte_count")),
                "cpu_payload_buffer_tensor_count": _as_int(native_batch.get("cpu_payload_buffer_tensor_count")),
                "returns_cpu_payload_buffer": bool(native_batch.get("returns_cpu_payload_buffer", False)),
                "training_path_enabled": False,
            }

    batch_parity = report.get("batch_parity")
    if isinstance(batch_parity, dict) and batch_parity:
        native_batch_ref = batch_parity.get("native_latent_batch_reference")
        provider = ""
        if isinstance(native_batch_ref, dict):
            provider = str(native_batch_ref.get("native_batch_summary_provider") or "")
        profile["batch_parity"] = {
            "ok": bool(batch_parity.get("ok", False)),
            "provider": str(batch_parity.get("provider") or ""),
            "batch_parity_guard_ran": bool(batch_parity.get("batch_parity_guard_ran", False)),
            "batch_parity_guard_passed": bool(batch_parity.get("batch_parity_guard_passed", False)),
            "batch_parity_field_count": _as_int(batch_parity.get("batch_parity_field_count")),
            "batch_parity_field_matches": _as_int(batch_parity.get("batch_parity_field_matches")),
            "batch_mismatch_count": _as_int(batch_parity.get("batch_mismatch_count")),
            "batch_payload_parity_guard_ran": bool(batch_parity.get("batch_payload_parity_guard_ran", False)),
            "batch_payload_parity_guard_passed": bool(batch_parity.get("batch_payload_parity_guard_passed", False)),
            "batch_payload_parity_field_count": _as_int(batch_parity.get("batch_payload_parity_field_count")),
            "batch_payload_parity_field_matches": _as_int(batch_parity.get("batch_payload_parity_field_matches")),
            "batch_payload_mismatch_count": _as_int(batch_parity.get("batch_payload_mismatch_count")),
            "torch_tensor_handoff_guard_ran": bool(batch_parity.get("torch_tensor_handoff_guard_ran", False)),
            "torch_tensor_handoff_guard_passed": bool(batch_parity.get("torch_tensor_handoff_guard_passed", False)),
            "torch_tensor_handoff_field_count": _as_int(batch_parity.get("torch_tensor_handoff_field_count")),
            "torch_tensor_handoff_field_matches": _as_int(batch_parity.get("torch_tensor_handoff_field_matches")),
            "torch_tensor_handoff_mismatch_count": _as_int(batch_parity.get("torch_tensor_handoff_mismatch_count")),
            "torch_owned_tensor_handoff_guard_ran": bool(batch_parity.get("torch_owned_tensor_handoff_guard_ran", False)),
            "torch_owned_tensor_handoff_guard_passed": bool(batch_parity.get("torch_owned_tensor_handoff_guard_passed", False)),
            "torch_owned_tensor_handoff_field_count": _as_int(batch_parity.get("torch_owned_tensor_handoff_field_count")),
            "torch_owned_tensor_handoff_field_matches": _as_int(batch_parity.get("torch_owned_tensor_handoff_field_matches")),
            "torch_owned_tensor_handoff_mismatch_count": _as_int(batch_parity.get("torch_owned_tensor_handoff_mismatch_count")),
            "native_batch_summary_provider": provider,
            "training_path_enabled": False,
        }
        owned_ref = batch_parity.get("native_torch_owned_tensor_handoff_reference")
        if isinstance(owned_ref, dict) and owned_ref:
            profile["batch_parity"]["torch_owned_tensor_handoff"] = {
                "provider": str(owned_ref.get("provider") or ""),
                "device": str(owned_ref.get("device") or ""),
                "is_contiguous": bool(owned_ref.get("is_contiguous", False)),
                "is_pinned": bool(owned_ref.get("is_pinned", False)),
                "source_buffer_read_only": bool(owned_ref.get("source_buffer_read_only", False)),
                "storage_aliases_source_payload": bool(owned_ref.get("storage_aliases_source_payload", True)),
                "storage_aliases_owned_payload": bool(owned_ref.get("storage_aliases_owned_payload", True)),
                "tensor_lifetime_guard_passed": bool(owned_ref.get("tensor_lifetime_guard_passed", False)),
                "torch_write_protection_enforced": bool(owned_ref.get("torch_write_protection_enforced", False)),
                "torch_frombuffer_warning_count": _as_int(owned_ref.get("torch_frombuffer_warning_count")),
                "pin_memory_attempted": bool(owned_ref.get("pin_memory_attempted", False)),
                "pin_memory_ready": bool(owned_ref.get("pin_memory_ready", False)),
                "device_transfer_probe_ran": bool(owned_ref.get("device_transfer_probe_ran", False)),
                "device_transfer_ms": float(owned_ref.get("device_transfer_ms", 0.0) or 0.0),
                "training_path_enabled": False,
            }
        ownership = compact_payload_ownership_shadow(batch_parity.get("payload_ownership_shadow"))
        if ownership:
            profile["batch_parity"]["payload_ownership_shadow"] = ownership
        batch_blocked = [str(item) for item in list(batch_parity.get("blocked_reasons", []) or [])]
        if batch_blocked:
            profile["batch_parity"]["blocked_reasons"] = batch_blocked

    handoff_session = report.get("batch_handoff_session")
    if isinstance(handoff_session, dict) and handoff_session:
        profile["batch_handoff_session"] = {
            "ok": bool(handoff_session.get("ok", False)),
            "provider": str(handoff_session.get("provider") or ""),
            "session_id": _as_int(handoff_session.get("session_id")),
            "session_reused": bool(handoff_session.get("session_reused", False)),
            "run_count": _as_int(handoff_session.get("run_count")),
            "batch_size": _as_int(handoff_session.get("batch_size")),
            "sample_count": _as_int(handoff_session.get("sample_count")),
            "total_payload_bytes": _as_int(handoff_session.get("total_payload_bytes")),
            "batch_payload_parity_guard_passed": bool(handoff_session.get("batch_payload_parity_guard_passed", False)),
            "torch_tensor_handoff_guard_passed": bool(handoff_session.get("torch_tensor_handoff_guard_passed", False)),
            "torch_owned_tensor_handoff_guard_passed": bool(handoff_session.get("torch_owned_tensor_handoff_guard_passed", False)),
            "returns_tensor_payloads": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "training_path_enabled": False,
        }

    text_payload = report.get("text_payload_parity")
    if isinstance(text_payload, dict) and text_payload:
        profile["text_payload_parity"] = {
            "ok": bool(text_payload.get("ok", False)),
            "provider": str(text_payload.get("provider") or ""),
            "text_payload_parity_guard_ran": bool(text_payload.get("text_payload_parity_guard_ran", False)),
            "text_payload_parity_guard_passed": bool(text_payload.get("text_payload_parity_guard_passed", False)),
            "text_payload_field_count": _as_int(text_payload.get("text_payload_field_count")),
            "text_payload_fields": [str(item) for item in list(text_payload.get("text_payload_fields", []) or [])],
            "native_tensor_decode_count": _as_int(text_payload.get("native_tensor_decode_count")),
            "native_data_payload_bytes_read": _as_int(text_payload.get("native_data_payload_bytes_read")),
            "returns_tensor_payloads": False,
            "training_path_enabled": False,
        }

    dispatch_contract = report.get("batch_dispatch_contract")
    if isinstance(dispatch_contract, dict) and dispatch_contract:
        profile["batch_dispatch_contract"] = {
            "ok": bool(dispatch_contract.get("ok", False)),
            "provider": str(dispatch_contract.get("provider") or ""),
            "dispatch_contract_ready": bool(dispatch_contract.get("dispatch_contract_ready", False)),
            "would_allow_native_dispatch": False,
            "fallback_to_python_batch": bool(dispatch_contract.get("fallback_to_python_batch", True)),
            "fallback_reasons": [str(item) for item in list(dispatch_contract.get("fallback_reasons", []) or [])],
            "native_dispatch_eligible": False,
            "native_dispatch_blockers": [str(item) for item in list(dispatch_contract.get("native_dispatch_blockers", []) or [])],
            "session_reused": bool(dispatch_contract.get("session_reused", False)),
            "run_count": _as_int(dispatch_contract.get("run_count")),
            "batch_handle_count": _as_int(dispatch_contract.get("batch_handle_count")),
            "total_payload_bytes": _as_int(dispatch_contract.get("total_payload_bytes")),
            "payload_ownership": str(dispatch_contract.get("payload_ownership") or ""),
            "tensor_ownership": str(dispatch_contract.get("tensor_ownership") or ""),
            "returns_tensor_payloads": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "training_dispatch": False,
            "training_path_enabled": False,
        }
        eligibility = dispatch_contract.get("dispatch_eligibility")
        if isinstance(eligibility, dict) and eligibility:
            profile["batch_dispatch_contract"]["dispatch_eligibility"] = {
                "shadow_gate_ready": bool(eligibility.get("shadow_gate_ready", False)),
                "native_dispatch_eligible": False,
                "native_dispatch_blockers": [str(item) for item in list(eligibility.get("native_dispatch_blockers", []) or [])],
                "strict_fallback": bool(eligibility.get("strict_fallback", False)),
                "strict_fallback_passed": bool(eligibility.get("strict_fallback_passed", True)),
                "training_path_enabled": False,
            }
        handles = [item for item in list(dispatch_contract.get("batch_handles", []) or []) if isinstance(item, dict)]
        if handles:
            first_ownership = compact_payload_ownership_shadow(handles[0].get("payload_ownership_shadow"))
            if first_ownership:
                profile["batch_dispatch_contract"]["first_payload_ownership_shadow"] = first_ownership

    return profile


__all__ = [
    "compact_cache_reader_decode_sidecar_profile",
    "compact_cache_reader_training_gate_profile",
]
