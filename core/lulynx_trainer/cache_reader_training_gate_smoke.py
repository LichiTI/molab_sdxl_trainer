"""Smoke probe for the native cache reader training experimental gate."""

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
from core.turbocore_cache_reader_training_gate import (  # noqa: E402
    BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV,
    BATCH_DISPATCH_CONTRACT_ENV,
    BATCH_HANDOFF_SESSION_ENV,
    CPU_PAYLOAD_BUFFER_BYTES_ENV,
    DISABLE_EXPERIMENTAL_ENV,
    ENABLE_EXPERIMENTAL_ENV,
    PARITY_BATCHES_ENV,
    PARITY_MAX_BYTES_ENV,
    TEXT_PAYLOAD_BUFFER_BYTES_ENV,
    TEXT_PAYLOAD_PARITY_ENV,
    run_cache_reader_training_experimental_gate,
)
from core.turbocore_cache_reader_handoff_session import run_cache_reader_batch_handoff_shadow_session_probe  # noqa: E402


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


def _clear_env(keys: list[str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    return old


def _ensure_native_artifact_dir() -> None:
    if os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR"):
        return
    artifact_dir = REPO_ROOT / "backend" / "native" / "target" / "release"
    if artifact_dir.exists():
        os.environ["LULYNX_NATIVE_ARTIFACT_DIR"] = str(artifact_dir)


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


def _assert_training_path_closed(report: dict[str, Any]) -> None:
    assert report["returns_tensor_payloads"] is False, report
    assert report["cache_reader_path_enabled"] is False, report
    assert report["prefetch_queue_training_path_enabled"] is False, report
    assert report["training_path_enabled"] is False, report


def run_smoke() -> dict[str, Any]:
    native_artifact_env = {"LULYNX_NATIVE_ARTIFACT_DIR": os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR")}
    _ensure_native_artifact_dir()
    clean_env = _clear_env([
        ENABLE_EXPERIMENTAL_ENV,
        DISABLE_EXPERIMENTAL_ENV,
        PARITY_BATCHES_ENV,
        PARITY_MAX_BYTES_ENV,
        CPU_PAYLOAD_BUFFER_BYTES_ENV,
        BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV,
        BATCH_DISPATCH_CONTRACT_ENV,
        BATCH_HANDOFF_SESSION_ENV,
        TEXT_PAYLOAD_BUFFER_BYTES_ENV,
        TEXT_PAYLOAD_PARITY_ENV,
        "LULYNX_ENABLE_NATIVE_CACHE_READER_SHADOW",
        "LULYNX_DISABLE_NATIVE_CACHE_READER_SHADOW",
    ])
    env_keys = {
        ENABLE_EXPERIMENTAL_ENV: "1",
        PARITY_BATCHES_ENV: "1",
        PARITY_MAX_BYTES_ENV: "4096",
        BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV: "4096",
        TEXT_PAYLOAD_BUFFER_BYTES_ENV: "8192",
        BATCH_DISPATCH_CONTRACT_ENV: "1",
        BATCH_HANDOFF_SESSION_ENV: "1",
        TEXT_PAYLOAD_PARITY_ENV: "1",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="lulynx_cache_reader_training_gate_") as tmp:
            return _run_smoke_in_dir(Path(tmp), env_keys)
    finally:
        _restore_env(clean_env)
        _restore_env(native_artifact_env)


def _run_smoke_in_dir(root: Path, env_keys: dict[str, str]) -> dict[str, Any]:
    _write_newbie_cache(root, "flat_a", 4)
    _write_newbie_cache(root, "flat_b", 4)
    newbie = NewbieCachedDataset(
        root,
        schema=NewbieCacheSchema(require_schema_version=True),
        cache_mmap=True,
        cache_lazy=True,
    )

    disabled_loader = create_newbie_cached_dataloader(
        newbie,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    assert not hasattr(disabled_loader, "native_cache_reader_training_gate"), disabled_loader
    disabled_report = run_cache_reader_training_experimental_gate(
        newbie,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    assert disabled_report["skipped"] is True, disabled_report
    assert disabled_report["reason"] == "training_experimental_gate_disabled", disabled_report
    _assert_training_path_closed(disabled_report)

    old_env = _set_env(env_keys)
    try:
        newbie_report = run_cache_reader_training_experimental_gate(
            newbie,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            max_decode_payload_bytes=4096,
        )
        newbie_session_probe = run_cache_reader_batch_handoff_shadow_session_probe(
            newbie,
            sample_indices=[0, 1],
            batch_size=1,
            max_decode_payload_bytes=4096,
            max_batch_cpu_payload_buffer_bytes=4096,
        )
        newbie_loader = create_newbie_cached_dataloader(
            newbie,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        blocked_report = run_cache_reader_training_experimental_gate(
            newbie,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    finally:
        _restore_env(old_env)

    attached_report = getattr(newbie_loader, "native_cache_reader_training_gate")
    dataset_attached_report = getattr(newbie, "native_cache_reader_training_gate")
    attached_dispatch_shadow = getattr(newbie_loader, "native_cache_reader_batch_dispatch_contract_shadow")
    dataset_dispatch_shadow = getattr(newbie, "native_cache_reader_batch_dispatch_contract_shadow")
    assert newbie_report["ok"] is True, newbie_report
    assert newbie_report["experimental_gate"] is True, newbie_report
    assert newbie_report["training_experimental_allowed"] is True, newbie_report
    assert newbie_report["parity_guard_ran"] is True, newbie_report
    assert newbie_report["parity_guard_passed"] is True, newbie_report
    assert newbie_report["batch_parity_guard_ran"] is True, newbie_report
    assert newbie_report["batch_parity_guard_passed"] is True, newbie_report
    assert newbie_report["batch_payload_parity_guard_ran"] is True, newbie_report
    assert newbie_report["batch_payload_parity_guard_passed"] is True, newbie_report
    assert newbie_report["torch_tensor_handoff_guard_ran"] is True, newbie_report
    assert newbie_report["torch_tensor_handoff_guard_passed"] is True, newbie_report
    assert newbie_report["torch_owned_tensor_handoff_guard_ran"] is True, newbie_report
    assert newbie_report["torch_owned_tensor_handoff_guard_passed"] is True, newbie_report
    assert newbie_report["text_payload_parity_guard_ran"] is True, newbie_report
    assert newbie_report["text_payload_parity_guard_passed"] is True, newbie_report
    assert newbie_report["batch_handoff_session_shadow_ran"] is True, newbie_report
    assert newbie_report["batch_handoff_session_shadow_passed"] is True, newbie_report
    assert newbie_report["batch_dispatch_contract_shadow_ran"] is True, newbie_report
    assert newbie_report["batch_dispatch_contract_ready"] is True, newbie_report
    assert newbie_report["batch_dispatch_contract_would_allow_native_dispatch"] is False, newbie_report
    assert newbie_report["native_dispatch_eligible"] is False, newbie_report
    assert newbie_report["dispatch_eligibility_shadow_gate_ready"] is True, newbie_report
    assert "text_conditioning_payload_ownership_not_promoted" in newbie_report["native_dispatch_blockers"], newbie_report
    assert newbie_report["tensor_parity_count"] == 2, newbie_report
    assert newbie_report["tensor_parity_matches"] == 2, newbie_report
    assert newbie_report["mismatch_count"] == 0, newbie_report
    assert newbie_report["native_probe"]["tensor_decode_count"] == 2, newbie_report
    assert newbie_report["batch_parity"]["python_batch_reference"]["latents"]["shape"] == [2, 4, 4, 4], newbie_report
    assert newbie_report["batch_parity"]["batch_payload_parity_guard_passed"] is True, newbie_report
    assert newbie_report["batch_parity"]["batch_payload_parity_field_matches"] == 11, newbie_report
    assert newbie_report["batch_parity"]["native_batch_payload_reference"]["payload_byte_count"] == 512, newbie_report
    assert newbie_report["batch_parity"]["torch_tensor_handoff_guard_passed"] is True, newbie_report
    assert newbie_report["batch_parity"]["torch_tensor_handoff_field_matches"] == 13, newbie_report
    assert newbie_report["batch_parity"]["native_torch_tensor_handoff_reference"]["device"] == "cpu", newbie_report
    assert newbie_report["batch_parity"]["native_torch_tensor_handoff_reference"]["requires_grad"] is False, newbie_report
    assert newbie_report["batch_parity"]["native_torch_tensor_handoff_reference"]["torch_write_protection_enforced"] is False, newbie_report
    assert newbie_report["batch_parity"]["torch_owned_tensor_handoff_guard_passed"] is True, newbie_report
    assert newbie_report["batch_parity"]["torch_owned_tensor_handoff_field_matches"] == 19, newbie_report
    newbie_payload_ownership = newbie_report["batch_parity"]["payload_ownership_shadow"]
    assert newbie_payload_ownership["native_latent_shadow_verified"] is True, newbie_report
    assert "encoder_hidden_states" in newbie_payload_ownership["text_payload_fields"], newbie_report
    assert "captions" in newbie_payload_ownership["aux_payload_fields"], newbie_report
    assert newbie_payload_ownership["native_text_payload_promoted"] is False, newbie_report
    newbie_text_payload = newbie_report["text_payload_parity"]
    assert newbie_text_payload["text_payload_parity_guard_passed"] is True, newbie_report
    assert "encoder_hidden_states" in newbie_text_payload["text_payload_fields"], newbie_report
    assert "attention_mask" in newbie_text_payload["text_payload_fields"], newbie_report
    assert "pooled_prompt_embeds" in newbie_text_payload["text_payload_fields"], newbie_report
    newbie_owned_handoff = newbie_report["batch_parity"]["native_torch_owned_tensor_handoff_reference"]
    assert newbie_owned_handoff["device"] == "cpu", newbie_report
    assert newbie_owned_handoff["requires_grad"] is False, newbie_report
    assert newbie_owned_handoff["storage_aliases_source_payload"] is False, newbie_report
    assert newbie_owned_handoff["storage_aliases_owned_payload"] is False, newbie_report
    assert newbie_owned_handoff["tensor_lifetime_guard_passed"] is True, newbie_report
    assert newbie_owned_handoff["torch_write_protection_enforced"] is True, newbie_report
    assert newbie_owned_handoff["torch_frombuffer_warning_count"] == 0, newbie_report
    assert newbie_owned_handoff["returns_tensor_payloads"] is False, newbie_report
    newbie_handoff_session = newbie_report["batch_handoff_session"]
    assert newbie_handoff_session["ok"] is True, newbie_report
    assert newbie_handoff_session["run_count"] == 1, newbie_report
    assert newbie_handoff_session["batch_payload_parity_guard_passed"] is True, newbie_report
    assert newbie_handoff_session["torch_owned_tensor_handoff_guard_passed"] is True, newbie_report
    assert newbie_handoff_session["training_path_enabled"] is False, newbie_report
    newbie_dispatch_contract = newbie_report["batch_dispatch_contract"]
    assert newbie_dispatch_contract["ok"] is True, newbie_report
    assert newbie_dispatch_contract["dispatch_contract_ready"] is True, newbie_report
    assert newbie_dispatch_contract["would_allow_native_dispatch"] is False, newbie_report
    assert newbie_dispatch_contract["native_dispatch_eligible"] is False, newbie_report
    assert newbie_dispatch_contract["dispatch_eligibility"]["shadow_gate_ready"] is True, newbie_report
    assert "representative_training_matrix_not_passed" in newbie_dispatch_contract["native_dispatch_blockers"], newbie_report
    assert newbie_dispatch_contract["fallback_to_python_batch"] is True, newbie_report
    assert newbie_dispatch_contract["batch_handle_count"] == 1, newbie_report
    assert newbie_dispatch_contract["batch_handles"][0]["returns_tensor_payloads"] is False, newbie_report
    assert newbie_dispatch_contract["batch_handles"][0]["payload_ownership_shadow"]["text_payload_ownership_ready"] is True, newbie_report
    assert "native_cache_reader_training_dispatch_not_implemented" in newbie_dispatch_contract["fallback_reasons"], newbie_report
    assert newbie_dispatch_contract["training_path_enabled"] is False, newbie_report
    assert newbie_session_probe["ok"] is True, newbie_session_probe
    assert newbie_session_probe["run_count"] == 2, newbie_session_probe
    assert newbie_session_probe["session_reused"] is True, newbie_session_probe
    assert newbie_session_probe["batch_payload_parity_guard_passed"] is True, newbie_session_probe
    assert newbie_session_probe["torch_owned_tensor_handoff_guard_passed"] is True, newbie_session_probe
    assert newbie_session_probe["runs"][1]["session_reused"] is True, newbie_session_probe
    assert newbie_session_probe["training_path_enabled"] is False, newbie_session_probe
    newbie_native_batch = newbie_report["batch_parity"]["native_latent_batch_reference"]
    assert newbie_native_batch["native_batch_summary_provider"] == "native_cache_reader_decode_session_batch_summary", newbie_report
    native_summary = newbie_report["native_probe"]["native_latent_batch_summary"]
    assert native_summary["native_batch_materialization_contract"] is True, newbie_report
    assert native_summary["materialization_contract_supported"] is True, newbie_report
    assert native_summary["cpu_payload_preview_shadow"] is True, newbie_report
    assert native_summary["payload_preview_tensor_count"] == 2, newbie_report
    assert native_summary["cpu_payload_buffer_shadow"] is True, newbie_report
    assert native_summary["cpu_payload_buffer_tensor_count"] == 2, newbie_report
    assert native_summary["cpu_payload_buffer_byte_count"] == 512, newbie_report
    assert native_summary["returns_cpu_payload_buffer"] is False, newbie_report
    newbie_payload = newbie_report["native_probe"]["native_latent_batch_summaries"][-1]
    assert isinstance(newbie_report["native_probe"]["session_create"], dict), newbie_report
    assert newbie_payload["cpu_payload_buffer_byte_count"] == 512, newbie_report
    first_payload = newbie_report["native_probe"]["session_stats"]["summary"]
    assert first_payload["training_path_enabled"] is False, newbie_report
    _assert_training_path_closed(newbie_report)
    assert attached_report["parity_guard_passed"] is True, attached_report
    assert attached_report["batch_parity_guard_passed"] is True, attached_report
    assert dataset_attached_report["parity_guard_passed"] is True, dataset_attached_report
    _assert_training_path_closed(attached_report)
    assert attached_dispatch_shadow["ok"] is True, attached_dispatch_shadow
    assert attached_dispatch_shadow["boundary"] == "dataloader_attach_metadata_only", attached_dispatch_shadow
    assert attached_dispatch_shadow["dispatch_contract_ready"] is True, attached_dispatch_shadow
    assert attached_dispatch_shadow["would_allow_native_dispatch"] is False, attached_dispatch_shadow
    assert attached_dispatch_shadow["native_dispatch_eligible"] is False, attached_dispatch_shadow
    assert attached_dispatch_shadow["dispatch_eligibility"]["shadow_gate_ready"] is True, attached_dispatch_shadow
    assert attached_dispatch_shadow["fallback_to_python_batch"] is True, attached_dispatch_shadow
    assert attached_dispatch_shadow["batch_handle_count"] == 1, attached_dispatch_shadow
    assert attached_dispatch_shadow["training_path_enabled"] is False, attached_dispatch_shadow
    assert dataset_dispatch_shadow["dispatch_contract_ready"] is True, dataset_dispatch_shadow
    first_batch = next(iter(newbie_loader))
    assert "latents" in first_batch, first_batch
    assert first_batch["latents"].shape == (2, 4, 4, 4), first_batch
    assert "native_cache_reader_batch_dispatch_contract_shadow" not in first_batch, first_batch
    assert blocked_report["training_experimental_allowed"] is False, blocked_report
    assert blocked_report["parity_guard_ran"] is False, blocked_report
    assert "shuffle_order_parity_not_ready" in blocked_report["blocked_reasons"], blocked_report
    assert blocked_report["dispatch_eligibility_shadow_gate_ready"] is False, blocked_report
    assert "sampler_reseed_policy_not_promoted" in blocked_report["native_dispatch_blockers"], blocked_report
    _assert_training_path_closed(blocked_report)

    disabled_by_env = _set_env({ENABLE_EXPERIMENTAL_ENV: "1", DISABLE_EXPERIMENTAL_ENV: "1"})
    try:
        explicit_disabled = run_cache_reader_training_experimental_gate(
            newbie,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
    finally:
        _restore_env(disabled_by_env)
    assert explicit_disabled["skipped"] is True, explicit_disabled
    assert explicit_disabled["reason"] == "training_experimental_gate_disabled_by_env", explicit_disabled
    _assert_training_path_closed(explicit_disabled)

    anima_root = root / "anima"
    anima_root.mkdir()
    _write_anima_cache(anima_root, "anime_a", 4)
    _write_anima_cache(anima_root, "anime_b", 4)
    anima = AnimaCachedDataset(anima_root, enable_bucket=False, cache_mmap=True, cache_lazy=True)
    old_env = _set_env(env_keys)
    try:
        anima_report = run_cache_reader_training_experimental_gate(
            anima,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            max_decode_payload_bytes=4096,
        )
        anima_loader = create_anima_cached_dataloader(
            anima,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    finally:
        _restore_env(old_env)
    anima_attached_report = getattr(anima_loader, "native_cache_reader_training_gate")
    anima_dispatch_shadow = getattr(anima_loader, "native_cache_reader_batch_dispatch_contract_shadow")
    assert anima_report["ok"] is True, anima_report
    assert anima_report["parity_guard_passed"] is True, anima_report
    assert anima_report["batch_parity_guard_ran"] is True, anima_report
    assert anima_report["batch_parity_guard_passed"] is True, anima_report
    assert anima_report["batch_payload_parity_guard_ran"] is True, anima_report
    assert anima_report["batch_payload_parity_guard_passed"] is True, anima_report
    assert anima_report["torch_tensor_handoff_guard_ran"] is True, anima_report
    assert anima_report["torch_tensor_handoff_guard_passed"] is True, anima_report
    assert anima_report["torch_owned_tensor_handoff_guard_ran"] is True, anima_report
    assert anima_report["torch_owned_tensor_handoff_guard_passed"] is True, anima_report
    assert anima_report["text_payload_parity_guard_ran"] is True, anima_report
    assert anima_report["text_payload_parity_guard_passed"] is True, anima_report
    assert anima_report["batch_handoff_session_shadow_ran"] is True, anima_report
    assert anima_report["batch_handoff_session_shadow_passed"] is True, anima_report
    assert anima_report["batch_dispatch_contract_shadow_ran"] is True, anima_report
    assert anima_report["batch_dispatch_contract_ready"] is True, anima_report
    assert anima_report["batch_dispatch_contract_would_allow_native_dispatch"] is False, anima_report
    assert anima_report["native_dispatch_eligible"] is False, anima_report
    assert anima_report["dispatch_eligibility_shadow_gate_ready"] is True, anima_report
    assert "text_conditioning_payload_ownership_not_promoted" in anima_report["native_dispatch_blockers"], anima_report
    assert anima_report["training_experimental_allowed"] is True, anima_report
    assert anima_report["tensor_parity_count"] == 2, anima_report
    assert anima_report["tensor_parity_matches"] == 2, anima_report
    assert anima_report["native_probe"]["tensor_decode_count"] == 2, anima_report
    assert anima_report["batch_parity"]["python_batch_reference"]["latents"]["shape"] == [2, 4, 4, 4], anima_report
    assert anima_report["batch_parity"]["batch_payload_parity_guard_passed"] is True, anima_report
    assert anima_report["batch_parity"]["batch_payload_parity_field_matches"] == 11, anima_report
    assert anima_report["batch_parity"]["native_batch_payload_reference"]["payload_byte_count"] == 512, anima_report
    assert anima_report["batch_parity"]["torch_tensor_handoff_guard_passed"] is True, anima_report
    assert anima_report["batch_parity"]["torch_tensor_handoff_field_matches"] == 13, anima_report
    assert anima_report["batch_parity"]["native_torch_tensor_handoff_reference"]["device"] == "cpu", anima_report
    assert anima_report["batch_parity"]["native_torch_tensor_handoff_reference"]["requires_grad"] is False, anima_report
    assert anima_report["batch_parity"]["native_torch_tensor_handoff_reference"]["torch_write_protection_enforced"] is False, anima_report
    assert anima_report["batch_parity"]["torch_owned_tensor_handoff_guard_passed"] is True, anima_report
    assert anima_report["batch_parity"]["torch_owned_tensor_handoff_field_matches"] == 19, anima_report
    anima_payload_ownership = anima_report["batch_parity"]["payload_ownership_shadow"]
    assert anima_payload_ownership["native_latent_shadow_verified"] is True, anima_report
    assert "encoder_hidden_states" in anima_payload_ownership["text_payload_fields"], anima_report
    assert "attention_mask" in anima_payload_ownership["text_payload_fields"], anima_report
    assert "caption_weights" in anima_payload_ownership["aux_payload_fields"], anima_report
    assert anima_payload_ownership["native_aux_payload_promoted"] is False, anima_report
    anima_text_payload = anima_report["text_payload_parity"]
    assert anima_text_payload["text_payload_parity_guard_passed"] is True, anima_report
    assert "encoder_hidden_states" in anima_text_payload["text_payload_fields"], anima_report
    assert "attention_mask" in anima_text_payload["text_payload_fields"], anima_report
    anima_owned_handoff = anima_report["batch_parity"]["native_torch_owned_tensor_handoff_reference"]
    assert anima_owned_handoff["device"] == "cpu", anima_report
    assert anima_owned_handoff["requires_grad"] is False, anima_report
    assert anima_owned_handoff["storage_aliases_source_payload"] is False, anima_report
    assert anima_owned_handoff["storage_aliases_owned_payload"] is False, anima_report
    assert anima_owned_handoff["tensor_lifetime_guard_passed"] is True, anima_report
    assert anima_owned_handoff["torch_write_protection_enforced"] is True, anima_report
    assert anima_owned_handoff["torch_frombuffer_warning_count"] == 0, anima_report
    assert anima_owned_handoff["returns_tensor_payloads"] is False, anima_report
    anima_handoff_session = anima_report["batch_handoff_session"]
    assert anima_handoff_session["ok"] is True, anima_report
    assert anima_handoff_session["run_count"] == 1, anima_report
    assert anima_handoff_session["batch_payload_parity_guard_passed"] is True, anima_report
    assert anima_handoff_session["torch_owned_tensor_handoff_guard_passed"] is True, anima_report
    assert anima_handoff_session["training_path_enabled"] is False, anima_report
    anima_dispatch_contract = anima_report["batch_dispatch_contract"]
    assert anima_dispatch_contract["ok"] is True, anima_report
    assert anima_dispatch_contract["dispatch_contract_ready"] is True, anima_report
    assert anima_dispatch_contract["would_allow_native_dispatch"] is False, anima_report
    assert anima_dispatch_contract["native_dispatch_eligible"] is False, anima_report
    assert anima_dispatch_contract["dispatch_eligibility"]["shadow_gate_ready"] is True, anima_report
    assert anima_dispatch_contract["fallback_to_python_batch"] is True, anima_report
    assert anima_dispatch_contract["batch_handle_count"] == 1, anima_report
    assert anima_dispatch_contract["batch_handles"][0]["payload_ownership_shadow"]["text_payload_ownership_ready"] is True, anima_report
    assert anima_dispatch_contract["training_path_enabled"] is False, anima_report
    anima_native_batch = anima_report["batch_parity"]["native_latent_batch_reference"]
    assert anima_native_batch["native_batch_summary_provider"] == "native_cache_reader_decode_session_batch_summary", anima_report
    anima_native_summary = anima_report["native_probe"]["native_latent_batch_summary"]
    assert anima_native_summary["native_batch_materialization_contract"] is True, anima_report
    assert anima_native_summary["materialization_contract_supported"] is True, anima_report
    assert anima_native_summary["cpu_payload_preview_shadow"] is True, anima_report
    assert anima_native_summary["payload_preview_tensor_count"] == 2, anima_report
    assert anima_native_summary["cpu_payload_buffer_shadow"] is True, anima_report
    assert anima_native_summary["cpu_payload_buffer_tensor_count"] == 2, anima_report
    assert anima_native_summary["cpu_payload_buffer_byte_count"] == 512, anima_report
    assert anima_native_summary["returns_cpu_payload_buffer"] is False, anima_report
    assert anima_attached_report["parity_guard_passed"] is True, anima_attached_report
    assert anima_attached_report["batch_parity_guard_passed"] is True, anima_attached_report
    assert anima_dispatch_shadow["ok"] is True, anima_dispatch_shadow
    assert anima_dispatch_shadow["dispatch_contract_ready"] is True, anima_dispatch_shadow
    assert anima_dispatch_shadow["would_allow_native_dispatch"] is False, anima_dispatch_shadow
    assert anima_dispatch_shadow["native_dispatch_eligible"] is False, anima_dispatch_shadow
    assert "text_conditioning_payload_ownership_not_promoted" in anima_dispatch_shadow["native_dispatch_blockers"], anima_dispatch_shadow
    assert anima_dispatch_shadow["fallback_to_python_batch"] is True, anima_dispatch_shadow
    assert anima_dispatch_shadow["training_path_enabled"] is False, anima_dispatch_shadow
    _assert_training_path_closed(anima_report)
    _assert_training_path_closed(anima_attached_report)

    return {
        "schema_version": 1,
        "probe": "cache_reader_training_gate_smoke",
        "ok": True,
        "newbie_tensor_parity_count": newbie_report["tensor_parity_count"],
        "newbie_batch_parity_guard_passed": newbie_report["batch_parity_guard_passed"],
        "newbie_batch_payload_parity_guard_passed": newbie_report["batch_payload_parity_guard_passed"],
        "newbie_torch_tensor_handoff_guard_passed": newbie_report["torch_tensor_handoff_guard_passed"],
        "newbie_torch_owned_tensor_handoff_guard_passed": newbie_report["torch_owned_tensor_handoff_guard_passed"],
        "newbie_batch_handoff_session_shadow_passed": newbie_report["batch_handoff_session_shadow_passed"],
        "newbie_batch_dispatch_contract_ready": newbie_report["batch_dispatch_contract_ready"],
        "newbie_native_dispatch_eligible": newbie_report["native_dispatch_eligible"],
        "newbie_native_dispatch_blockers": newbie_report["native_dispatch_blockers"],
        "newbie_payload_text_fields": newbie_payload_ownership["text_payload_fields"],
        "newbie_payload_aux_fields": newbie_payload_ownership["aux_payload_fields"],
        "newbie_text_payload_parity_fields": newbie_text_payload["text_payload_fields"],
        "newbie_boundary_dispatch_contract_ready": attached_dispatch_shadow["dispatch_contract_ready"],
        "newbie_batch_handoff_session_reused": newbie_session_probe["session_reused"],
        "newbie_native_batch_summary_provider": newbie_native_batch["native_batch_summary_provider"],
        "newbie_materialization_contract_supported": native_summary["materialization_contract_supported"],
        "newbie_payload_preview_tensor_count": native_summary["payload_preview_tensor_count"],
        "newbie_cpu_payload_buffer_byte_count": native_summary["cpu_payload_buffer_byte_count"],
        "newbie_native_bytes": newbie_report["native_data_payload_bytes_read"],
        "anima_tensor_parity_count": anima_report["tensor_parity_count"],
        "anima_batch_parity_guard_passed": anima_report["batch_parity_guard_passed"],
        "anima_batch_payload_parity_guard_passed": anima_report["batch_payload_parity_guard_passed"],
        "anima_torch_tensor_handoff_guard_passed": anima_report["torch_tensor_handoff_guard_passed"],
        "anima_torch_owned_tensor_handoff_guard_passed": anima_report["torch_owned_tensor_handoff_guard_passed"],
        "anima_batch_handoff_session_shadow_passed": anima_report["batch_handoff_session_shadow_passed"],
        "anima_batch_dispatch_contract_ready": anima_report["batch_dispatch_contract_ready"],
        "anima_native_dispatch_eligible": anima_report["native_dispatch_eligible"],
        "anima_native_dispatch_blockers": anima_report["native_dispatch_blockers"],
        "anima_payload_text_fields": anima_payload_ownership["text_payload_fields"],
        "anima_payload_aux_fields": anima_payload_ownership["aux_payload_fields"],
        "anima_text_payload_parity_fields": anima_text_payload["text_payload_fields"],
        "anima_boundary_dispatch_contract_ready": anima_dispatch_shadow["dispatch_contract_ready"],
        "anima_native_batch_summary_provider": anima_native_batch["native_batch_summary_provider"],
        "anima_materialization_contract_supported": anima_native_summary["materialization_contract_supported"],
        "anima_payload_preview_tensor_count": anima_native_summary["payload_preview_tensor_count"],
        "anima_cpu_payload_buffer_byte_count": anima_native_summary["cpu_payload_buffer_byte_count"],
        "anima_native_bytes": anima_report["native_data_payload_bytes_read"],
        "blocked_reasons": blocked_report["blocked_reasons"],
        "training_path_enabled": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
