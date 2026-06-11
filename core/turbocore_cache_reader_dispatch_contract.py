"""Dispatch contract shadow for native cache reader batch handoff sessions."""

from __future__ import annotations

from typing import Any, Dict

from core.turbocore_cache_reader_payload_ownership import compact_payload_ownership_shadow


def _compact_dispatch_eligibility(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {
            "schema_version": 1,
            "provider": "native_cache_reader_dispatch_eligibility_policy_v1",
            "shadow_gate_ready": False,
            "native_dispatch_eligible": False,
            "native_dispatch_blockers": ["dispatch_eligibility_report_missing"],
            "would_allow_native_dispatch": False,
            "fallback_to_python_batch": True,
            "training_path_enabled": False,
        }
    return {
        "schema_version": int(value.get("schema_version", 1) or 1),
        "provider": str(value.get("provider") or "native_cache_reader_dispatch_eligibility_policy_v1"),
        "dataset_class": str(value.get("dataset_class") or ""),
        "dataset_supported": bool(value.get("dataset_supported", False)),
        "sample_count": int(value.get("sample_count", 0) or 0),
        "batch_size": int(value.get("batch_size", 0) or 0),
        "shuffle": bool(value.get("shuffle", False)),
        "drop_last": bool(value.get("drop_last", False)),
        "worker_count": int(value.get("worker_count", 0) or 0),
        "shadow_gate_ready": bool(value.get("shadow_gate_ready", False)),
        "shadow_gate_blockers": [str(item) for item in list(value.get("shadow_gate_blockers", []) or [])],
        "native_dispatch_eligible": False,
        "native_dispatch_blockers": [str(item) for item in list(value.get("native_dispatch_blockers", []) or [])],
        "strict_fallback": bool(value.get("strict_fallback", False)),
        "strict_fallback_passed": bool(value.get("strict_fallback_passed", True)),
        "would_allow_native_dispatch": False,
        "fallback_to_python_batch": True,
        "training_path_enabled": False,
    }


def build_cache_reader_batch_dispatch_contract_shadow(
    session_report: Dict[str, Any],
    *,
    dispatch_eligibility: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a compact no-dispatch envelope from a batch handoff session report."""
    if not isinstance(session_report, dict) or not session_report:
        return {
            "schema_version": 1,
            "provider": "native_cache_reader_batch_dispatch_contract_shadow",
            "ok": False,
            "reason": "batch_handoff_session_report_missing",
            "dispatch_contract_ready": False,
            "training_dispatch": False,
            "training_path_enabled": False,
        }

    runs = [dict(run) for run in list(session_report.get("runs", []) or []) if isinstance(run, dict)]
    blockers: list[str] = []
    if not bool(session_report.get("ok", False)):
        blockers.append("batch_handoff_session_not_ok")
    if not bool(session_report.get("session_reused", False)) and int(session_report.get("run_count", 0) or 0) > 1:
        blockers.append("batch_handoff_session_reuse_missing")
    if not bool(session_report.get("batch_payload_parity_guard_passed", False)):
        blockers.append("batch_payload_parity_guard_failed")
    if not bool(session_report.get("torch_owned_tensor_handoff_guard_passed", False)):
        blockers.append("torch_owned_tensor_handoff_guard_failed")

    batch_handles: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        parity = dict(run.get("batch_parity", {}) or {})
        owned_ref = dict(parity.get("native_torch_owned_tensor_handoff_reference", {}) or {})
        payload_ownership = compact_payload_ownership_shadow(parity.get("payload_ownership_shadow"))
        if not bool(run.get("ok", False)):
            blockers.append(f"batch_{index}_run_not_ok")
        if not bool(run.get("batch_payload_parity_guard_passed", False)):
            blockers.append(f"batch_{index}_payload_parity_failed")
        if not bool(run.get("torch_owned_tensor_handoff_guard_passed", False)):
            blockers.append(f"batch_{index}_owned_handoff_failed")
        batch_handles.append(
            {
                "handle_id": f"cache_reader_batch_shadow:{session_report.get('session_id', 0)}:{int(run.get('cursor', index) or 0)}",
                "session_id": int(session_report.get("session_id", 0) or 0),
                "cursor": int(run.get("cursor", 0) or 0),
                "next_cursor": int(run.get("next_cursor", 0) or 0),
                "sample_indices": [int(item) for item in list(run.get("sample_indices", []) or [])],
                "payload_byte_count": int(run.get("batch_cpu_payload_byte_count", 0) or 0),
                "tensor_count": int(run.get("batch_cpu_payload_tensor_count", 0) or 0),
                "owned_tensor_handoff_ready": bool(run.get("torch_owned_tensor_handoff_guard_passed", False)),
                "tensor_lifetime_guard_passed": bool(owned_ref.get("tensor_lifetime_guard_passed", False)),
                "pin_memory_ready": bool(owned_ref.get("pin_memory_ready", False)),
                "device_transfer_probe_ran": bool(owned_ref.get("device_transfer_probe_ran", False)),
                "payload_ownership_shadow": payload_ownership,
                "returns_tensor_payloads": False,
                "training_dispatch": False,
            }
        )

    eligibility = _compact_dispatch_eligibility(dispatch_eligibility)
    hard_blockers = [str(item) for item in list(eligibility.get("native_dispatch_blockers", []) or [])]
    if not hard_blockers:
        hard_blockers = [
            "native_cache_reader_training_dispatch_not_implemented",
            "python_dataloader_batch_remains_authoritative",
            "dispatch_eligibility_policy_missing_blockers",
        ]
    return {
        "schema_version": 1,
        "provider": "native_cache_reader_batch_dispatch_contract_shadow",
        "ok": not blockers,
        "debug_only": True,
        "shadow_run": True,
        "dispatch_contract_ready": not blockers,
        "would_allow_native_dispatch": False,
        "fallback_to_python_batch": True,
        "fallback_reasons": hard_blockers + blockers,
        "dispatch_eligibility": eligibility,
        "native_dispatch_eligible": False,
        "native_dispatch_blockers": hard_blockers,
        "session_id": int(session_report.get("session_id", 0) or 0),
        "session_reused": bool(session_report.get("session_reused", False)),
        "run_count": int(session_report.get("run_count", 0) or 0),
        "batch_handle_count": len(batch_handles),
        "total_payload_bytes": int(session_report.get("total_payload_bytes", 0) or 0),
        "batch_handles": batch_handles,
        "payload_ownership": "native_shadow_bytes_to_python_owned_tensor_staging",
        "tensor_ownership": "python_shadow_only_no_training_tensor_return",
        "sampler_policy": "python_sampler_authoritative",
        "worker_policy": "single_process_debug_shadow_only",
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_dispatch": False,
        "training_path_enabled": False,
    }


__all__ = ["build_cache_reader_batch_dispatch_contract_shadow"]
