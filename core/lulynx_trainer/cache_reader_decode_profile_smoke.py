"""Smoke coverage for training-side native cache reader decode sidecar profiles."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

from core.configs import ModelArch, UnifiedTrainingConfig
from core.lulynx_trainer.cache_reader_decode_profile import (
    compact_cache_reader_decode_sidecar_profile,
    compact_cache_reader_training_gate_profile,
)
from core.lulynx_trainer.trainer import LulynxTrainer


def _sidecar_report() -> dict[str, object]:
    return {
        "provider": "native_cache_reader_decode_sidecar_session_adapter",
        "native_runtime": True,
        "ok": True,
        "debug_only": True,
        "shadow_run": True,
        "sidecar_only": True,
        "batch_size": 2,
        "planned_shadow_batches": 2,
        "chunk_count": 2,
        "tensor_decode_count": 4,
        "data_payload_bytes_read": 1024,
        "worker_count": 0,
        "prefetch_factor": 2,
        "session_summary": {
            "tensor_candidate_count": 8,
            "total_declared_payload_bytes": 2048,
            "format_counts": {".npz": 8},
            "role_counts": {"latent": 8},
            "layout_cache": {"hits": 8, "misses": 0, "stored": 0, "entry_count": 8},
        },
        "reads_tensor_payload_bytes": True,
        "parses_tensor_payloads": True,
        "decodes_tensor_payloads": True,
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }


def _training_gate_report() -> dict[str, object]:
    return {
        "provider": "native_cache_reader_training_gate",
        "ok": True,
        "experimental_gate": True,
        "native_runtime": True,
        "dataset_class": "NewbieCachedDataset",
        "sample_count": 2,
        "batch_size": 2,
        "planned_parity_batches": 1,
        "cpu_payload_buffer_shadow": True,
        "max_cpu_payload_buffer_bytes": 4096,
        "batch_cpu_payload_buffer_shadow": True,
        "max_batch_cpu_payload_buffer_bytes": 4096,
        "batch_handoff_session_shadow": True,
        "batch_handoff_session_shadow_ran": True,
        "batch_handoff_session_shadow_passed": True,
        "batch_dispatch_contract_shadow": True,
        "batch_dispatch_contract_shadow_ran": True,
        "batch_dispatch_contract_ready": True,
        "batch_dispatch_contract_would_allow_native_dispatch": False,
        "native_dispatch_eligible": False,
        "native_dispatch_blockers": ["native_cache_reader_training_dispatch_not_implemented"],
        "dispatch_eligibility_shadow_gate_ready": True,
        "dispatch_eligibility": {
            "provider": "native_cache_reader_dispatch_eligibility_policy_v1",
            "dataset_class": "NewbieCachedDataset",
            "dataset_supported": True,
            "shadow_gate_ready": True,
            "shadow_gate_blockers": [],
            "native_dispatch_eligible": False,
            "native_dispatch_blockers": ["native_cache_reader_training_dispatch_not_implemented"],
            "strict_fallback": False,
            "strict_fallback_passed": True,
        },
        "parity_guard_ran": True,
        "parity_guard_passed": True,
        "batch_parity_guard_ran": True,
        "batch_parity_guard_passed": True,
        "batch_payload_parity_guard_ran": True,
        "batch_payload_parity_guard_passed": True,
        "torch_tensor_handoff_guard_ran": True,
        "torch_tensor_handoff_guard_passed": True,
        "torch_owned_tensor_handoff_guard_ran": True,
        "torch_owned_tensor_handoff_guard_passed": True,
        "training_experimental_allowed": True,
        "tensor_parity_count": 2,
        "tensor_parity_matches": 2,
        "mismatch_count": 0,
        "native_data_payload_bytes_read": 512,
        "python_data_payload_bytes_read": 512,
        "native_probe": {
            "chunk_count": 1,
            "tensor_decode_count": 2,
            "data_payload_bytes_read": 512,
            "native_latent_batch_summary": {
                "provider": "native_cache_reader_decode_session_batch_summary",
                "batch_summary_ready": True,
                "shape": [2, 4, 4, 4],
                "canonical_dtype": "float32",
                "source_tensor_count": 2,
                "native_batch_materialization_contract": True,
                "materialization_contract_supported": True,
                "cpu_buffer_bytes": 512,
                "cpu_payload_preview_shadow": True,
                "payload_preview_byte_count": 32,
                "payload_preview_tensor_count": 2,
                "cpu_payload_buffer_shadow": True,
                "cpu_payload_buffer_byte_count": 512,
                "cpu_payload_buffer_tensor_count": 2,
                "returns_cpu_payload_buffer": False,
            },
        },
        "batch_parity": {
            "ok": True,
            "provider": "native_cache_reader_training_batch_parity_guard",
            "batch_parity_guard_ran": True,
            "batch_parity_guard_passed": True,
            "batch_parity_field_count": 9,
            "batch_parity_field_matches": 9,
            "batch_mismatch_count": 0,
            "batch_payload_parity_guard_ran": True,
            "batch_payload_parity_guard_passed": True,
            "batch_payload_parity_field_count": 11,
            "batch_payload_parity_field_matches": 11,
            "batch_payload_mismatch_count": 0,
            "torch_tensor_handoff_guard_ran": True,
            "torch_tensor_handoff_guard_passed": True,
            "torch_tensor_handoff_field_count": 13,
            "torch_tensor_handoff_field_matches": 13,
            "torch_tensor_handoff_mismatch_count": 0,
            "torch_owned_tensor_handoff_guard_ran": True,
            "torch_owned_tensor_handoff_guard_passed": True,
            "torch_owned_tensor_handoff_field_count": 19,
            "torch_owned_tensor_handoff_field_matches": 19,
            "torch_owned_tensor_handoff_mismatch_count": 0,
            "native_torch_owned_tensor_handoff_reference": {
                "provider": "torch_owned_copy_batch_payload_shadow",
                "device": "cpu",
                "is_contiguous": True,
                "is_pinned": False,
                "source_buffer_read_only": True,
                "storage_aliases_source_payload": False,
                "storage_aliases_owned_payload": False,
                "tensor_lifetime_guard_passed": True,
                "torch_write_protection_enforced": True,
                "torch_frombuffer_warning_count": 0,
                "pin_memory_attempted": False,
                "pin_memory_ready": False,
                "device_transfer_probe_ran": False,
                "device_transfer_ms": 0.0,
            },
            "payload_ownership_shadow": {
                "provider": "native_cache_reader_payload_ownership_shadow_v1",
                "ok": True,
                "field_count": 4,
                "latent_fields": ["latents"],
                "text_payload_fields": ["encoder_hidden_states"],
                "aux_payload_fields": ["captions", "sample_id"],
                "native_latent_shadow_verified": True,
                "text_payload_ownership_ready": True,
                "aux_payload_ownership_ready": True,
                "native_text_payload_promoted": False,
                "native_aux_payload_promoted": False,
                "training_path_enabled": False,
            },
            "native_latent_batch_reference": {
                "native_batch_summary_provider": "native_cache_reader_decode_session_batch_summary",
            },
        },
        "batch_handoff_session": {
            "ok": True,
            "provider": "native_cache_reader_batch_handoff_shadow_session",
            "session_id": 42,
            "session_reused": True,
            "run_count": 2,
            "batch_size": 1,
            "sample_count": 2,
            "total_payload_bytes": 512,
            "batch_payload_parity_guard_passed": True,
            "torch_tensor_handoff_guard_passed": True,
            "torch_owned_tensor_handoff_guard_passed": True,
            "returns_tensor_payloads": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "training_path_enabled": False,
        },
        "batch_dispatch_contract": {
            "ok": True,
            "provider": "native_cache_reader_batch_dispatch_contract_shadow",
            "dispatch_contract_ready": True,
            "would_allow_native_dispatch": False,
            "fallback_to_python_batch": True,
            "fallback_reasons": ["native_cache_reader_training_dispatch_not_implemented"],
            "native_dispatch_eligible": False,
            "native_dispatch_blockers": ["native_cache_reader_training_dispatch_not_implemented"],
            "dispatch_eligibility": {
                "shadow_gate_ready": True,
                "native_dispatch_eligible": False,
                "native_dispatch_blockers": ["native_cache_reader_training_dispatch_not_implemented"],
                "strict_fallback": False,
                "strict_fallback_passed": True,
            },
            "session_reused": True,
            "run_count": 2,
            "batch_handle_count": 2,
            "batch_handles": [
                {
                    "payload_ownership_shadow": {
                        "provider": "native_cache_reader_payload_ownership_shadow_v1",
                        "ok": True,
                        "field_count": 4,
                        "latent_fields": ["latents"],
                        "text_payload_fields": ["encoder_hidden_states"],
                        "aux_payload_fields": ["captions", "sample_id"],
                        "native_latent_shadow_verified": True,
                        "text_payload_ownership_ready": True,
                        "aux_payload_ownership_ready": True,
                        "native_text_payload_promoted": False,
                        "native_aux_payload_promoted": False,
                        "training_path_enabled": False,
                    }
                }
            ],
            "total_payload_bytes": 512,
            "payload_ownership": "native_shadow_bytes_to_python_owned_tensor_staging",
            "tensor_ownership": "python_shadow_only_no_training_tensor_return",
            "returns_tensor_payloads": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "training_dispatch": False,
            "training_path_enabled": False,
        },
        "returns_tensor_payloads": False,
        "cache_reader_path_enabled": False,
        "prefetch_queue_training_path_enabled": False,
        "training_path_enabled": False,
    }


def test_compact_sidecar_profile_guards_training_path() -> None:
    profile = compact_cache_reader_decode_sidecar_profile(_sidecar_report(), route="newbie_cached")
    assert profile["ok"] is True, profile
    assert profile["sidecar_only"] is True, profile
    assert profile["tensor_decode_count"] == 4, profile
    assert profile["data_payload_bytes_read"] == 1024, profile
    assert profile["layout_cache"]["hits"] == 8, profile
    assert profile["returns_tensor_payloads"] is False, profile
    assert profile["cache_reader_path_enabled"] is False, profile
    assert profile["training_path_enabled"] is False, profile


def test_compact_training_gate_profile_guards_training_path() -> None:
    profile = compact_cache_reader_training_gate_profile(_training_gate_report(), route="newbie_cached")
    assert profile["ok"] is True, profile
    assert profile["parity_guard_passed"] is True, profile
    assert profile["batch_parity_guard_passed"] is True, profile
    assert profile["batch_payload_parity_guard_passed"] is True, profile
    assert profile["torch_tensor_handoff_guard_passed"] is True, profile
    assert profile["torch_owned_tensor_handoff_guard_passed"] is True, profile
    assert profile["training_experimental_allowed"] is True, profile
    assert profile["batch_cpu_payload_buffer_shadow"] is True, profile
    assert profile["max_batch_cpu_payload_buffer_bytes"] == 4096, profile
    assert profile["batch_handoff_session_shadow"] is True, profile
    assert profile["batch_handoff_session_shadow_passed"] is True, profile
    assert profile["batch_dispatch_contract_shadow"] is True, profile
    assert profile["batch_dispatch_contract_ready"] is True, profile
    assert profile["batch_dispatch_contract_would_allow_native_dispatch"] is False, profile
    assert profile["native_dispatch_eligible"] is False, profile
    assert "native_cache_reader_training_dispatch_not_implemented" in profile["native_dispatch_blockers"], profile
    assert profile["dispatch_eligibility"]["shadow_gate_ready"] is True, profile
    assert profile["dispatch_eligibility"]["native_dispatch_eligible"] is False, profile
    assert profile["native_batch_summary"]["provider"] == "native_cache_reader_decode_session_batch_summary", profile
    assert profile["native_batch_summary"]["native_batch_materialization_contract"] is True, profile
    assert profile["native_batch_summary"]["materialization_contract_supported"] is True, profile
    assert profile["native_batch_summary"]["cpu_payload_preview_shadow"] is True, profile
    assert profile["native_batch_summary"]["payload_preview_tensor_count"] == 2, profile
    assert profile["native_batch_summary"]["cpu_payload_buffer_shadow"] is True, profile
    assert profile["native_batch_summary"]["cpu_payload_buffer_byte_count"] == 512, profile
    assert profile["native_batch_summary"]["returns_cpu_payload_buffer"] is False, profile
    assert profile["batch_parity"]["native_batch_summary_provider"] == "native_cache_reader_decode_session_batch_summary", profile
    assert profile["batch_parity"]["batch_payload_parity_guard_passed"] is True, profile
    assert profile["batch_parity"]["torch_tensor_handoff_guard_passed"] is True, profile
    assert profile["batch_parity"]["torch_owned_tensor_handoff_guard_passed"] is True, profile
    assert profile["batch_parity"]["torch_owned_tensor_handoff"]["tensor_lifetime_guard_passed"] is True, profile
    assert profile["batch_parity"]["torch_owned_tensor_handoff"]["torch_write_protection_enforced"] is True, profile
    assert profile["batch_parity"]["torch_owned_tensor_handoff"]["storage_aliases_source_payload"] is False, profile
    assert profile["batch_parity"]["payload_ownership_shadow"]["text_payload_ownership_ready"] is True, profile
    assert "encoder_hidden_states" in profile["batch_parity"]["payload_ownership_shadow"]["text_payload_fields"], profile
    assert profile["batch_handoff_session"]["session_reused"] is True, profile
    assert profile["batch_handoff_session"]["torch_owned_tensor_handoff_guard_passed"] is True, profile
    assert profile["batch_dispatch_contract"]["dispatch_contract_ready"] is True, profile
    assert profile["batch_dispatch_contract"]["would_allow_native_dispatch"] is False, profile
    assert profile["batch_dispatch_contract"]["native_dispatch_eligible"] is False, profile
    assert profile["batch_dispatch_contract"]["dispatch_eligibility"]["shadow_gate_ready"] is True, profile
    assert profile["batch_dispatch_contract"]["first_payload_ownership_shadow"]["aux_payload_ownership_ready"] is True, profile
    assert profile["batch_dispatch_contract"]["fallback_to_python_batch"] is True, profile
    assert profile["returns_tensor_payloads"] is False, profile
    assert profile["cache_reader_path_enabled"] is False, profile
    assert profile["training_path_enabled"] is False, profile


def test_trainer_captures_sidecar_profile_for_manifest_and_runtime_event() -> None:
    cfg = UnifiedTrainingConfig(model_type=ModelArch.NEWBIE, output_dir=".")
    trainer = LulynxTrainer(cfg)
    trainer._data_backend_profile = {"effective_training_backend": "newbie_cached"}
    loader = SimpleNamespace(
        native_cache_reader_decode_shadow_adapter=_sidecar_report(),
        native_cache_reader_training_gate=_training_gate_report(),
    )

    captured = trainer._capture_cache_reader_decode_sidecar_profile(loader, route="newbie_cached")
    assert captured["training_path_enabled"] is False, captured
    gate_profile = trainer._capture_cache_reader_training_gate_profile(loader, route="newbie_cached")
    assert gate_profile["training_path_enabled"] is False, gate_profile
    assert gate_profile["batch_parity"]["native_batch_summary_provider"] == "native_cache_reader_decode_session_batch_summary"
    assert gate_profile["batch_parity"]["batch_payload_parity_guard_passed"] is True
    assert gate_profile["batch_parity"]["torch_tensor_handoff_guard_passed"] is True
    assert gate_profile["batch_parity"]["torch_owned_tensor_handoff_guard_passed"] is True
    assert gate_profile["batch_handoff_session"]["session_reused"] is True
    assert gate_profile["batch_dispatch_contract"]["dispatch_contract_ready"] is True
    assert gate_profile["batch_dispatch_contract"]["would_allow_native_dispatch"] is False
    assert gate_profile["batch_dispatch_contract"]["native_dispatch_eligible"] is False
    assert trainer._data_backend_profile["native_cache_reader"]["decode_sidecar"] == captured
    assert trainer._data_backend_profile["native_cache_reader"]["training_gate"] == gate_profile
    assert trainer._run_manifest_extra()["data_backend"]["native_cache_reader"]["decode_sidecar"] == captured
    assert trainer._run_manifest_extra()["data_backend"]["native_cache_reader"]["training_gate"] == gate_profile

    events: list[dict[str, object]] = []
    trainer.on_runtime_event = events.append
    trainer.training_loop = None
    trainer._dataset = None
    trainer._on_step_end(1, 0.5, {"lr": 0.001, "epoch": 0})

    event_profile = events[0]["data"]["native_cache_reader_decode_sidecar"]
    assert event_profile["tensor_decode_count"] == 4, event_profile
    assert event_profile["returns_tensor_payloads"] is False, event_profile
    assert event_profile["training_path_enabled"] is False, event_profile
    event_gate = events[0]["data"]["native_cache_reader_training_gate"]
    assert event_gate["training_experimental_allowed"] is True, event_gate
    assert event_gate["batch_parity"]["native_batch_summary_provider"] == "native_cache_reader_decode_session_batch_summary", event_gate
    assert event_gate["batch_parity"]["batch_payload_parity_guard_passed"] is True, event_gate
    assert event_gate["batch_parity"]["torch_tensor_handoff_guard_passed"] is True, event_gate
    assert event_gate["batch_parity"]["torch_owned_tensor_handoff_guard_passed"] is True, event_gate
    assert event_gate["batch_handoff_session"]["session_reused"] is True, event_gate
    assert event_gate["batch_dispatch_contract"]["dispatch_contract_ready"] is True, event_gate
    assert event_gate["batch_dispatch_contract"]["would_allow_native_dispatch"] is False, event_gate
    assert event_gate["batch_dispatch_contract"]["native_dispatch_eligible"] is False, event_gate
    assert event_gate["training_path_enabled"] is False, event_gate


if __name__ == "__main__":
    test_compact_sidecar_profile_guards_training_path()
    test_compact_training_gate_profile_guards_training_path()
    test_trainer_captures_sidecar_profile_for_manifest_and_runtime_event()
    print("PASS: cache reader decode sidecar profile smoke")
