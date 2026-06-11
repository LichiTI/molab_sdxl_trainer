"""Smoke probe for the cache reader dispatch pre-promotion scorecard."""

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

from core.lulynx_trainer.newbie_cached_dataset import NewbieCacheSchema, NewbieCachedDataset  # noqa: E402
from core.lulynx_trainer.anima_cached_dataset import AnimaCachedDataset  # noqa: E402
from core.turbocore_cache_reader_dispatch_scorecard import build_cache_reader_dispatch_promotion_scorecard  # noqa: E402


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


def _ensure_native_artifact_dir() -> dict[str, str | None]:
    old = {"LULYNX_NATIVE_ARTIFACT_DIR": os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR")}
    if os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR"):
        return old
    artifact_dir = REPO_ROOT / "backend" / "native" / "target" / "release"
    if artifact_dir.exists():
        os.environ["LULYNX_NATIVE_ARTIFACT_DIR"] = str(artifact_dir)
    return old


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


def _assert_closed(value: dict[str, Any]) -> None:
    assert value["native_dispatch_eligible"] is False, value
    assert value["would_allow_native_dispatch"] is False, value
    assert value["fallback_to_python_batch"] is True, value
    assert value["returns_tensor_payloads"] is False, value
    assert value["cache_reader_path_enabled"] is False, value
    assert value["prefetch_queue_training_path_enabled"] is False, value
    assert value["training_path_enabled"] is False, value


def run_smoke() -> dict[str, Any]:
    native_env = _ensure_native_artifact_dir()
    guard_env = _set_env({"LULYNX_DISABLE_NATIVE_CACHE_READER_TRAINING_EXPERIMENTAL": "0"})
    try:
        with tempfile.TemporaryDirectory(prefix="lulynx_cache_reader_dispatch_scorecard_") as tmp:
            root = Path(tmp)
            _write_newbie_cache(root, "flat_a", 4)
            _write_newbie_cache(root, "flat_b", 4)
            anima_root = root / "anima"
            anima_root.mkdir()
            _write_anima_cache(anima_root, "anime_a", 4)
            _write_anima_cache(anima_root, "anime_b", 4)
            dataset = NewbieCachedDataset(
                root,
                schema=NewbieCacheSchema(require_schema_version=True),
                cache_mmap=True,
                cache_lazy=True,
            )
            anima = AnimaCachedDataset(anima_root, enable_bucket=False, cache_mmap=True, cache_lazy=True)
            try:
                scorecard = build_cache_reader_dispatch_promotion_scorecard(
                    dataset,
                    additional_supported_datasets=(("anima_flat_single_worker", anima),),
                    additional_supported_cases=(
                        {
                            "case_id": "newbie_single_item_batch",
                            "dataset": dataset,
                            "batch_size": 1,
                            "parity_batches": 1,
                            "max_decode_payload_bytes": 4096,
                            "max_batch_cpu_payload_buffer_bytes": 4096,
                            "max_text_payload_buffer_bytes": 8192,
                        },
                        {
                            "case_id": "newbie_two_batch_two_parity",
                            "dataset": dataset,
                            "batch_size": 2,
                            "parity_batches": 2,
                            "max_decode_payload_bytes": 8192,
                            "max_batch_cpu_payload_buffer_bytes": 8192,
                            "max_text_payload_buffer_bytes": 16384,
                        },
                    ),
                    batch_size=2,
                    parity_batches=1,
                    prefetch_factor=2,
                    max_decode_payload_bytes=4096,
                    max_batch_cpu_payload_buffer_bytes=4096,
                    max_text_payload_buffer_bytes=8192,
                    strict_fallback=True,
                )
            finally:
                close_handles = getattr(dataset, "close_file_handles", None)
                if callable(close_handles):
                    close_handles()
                close_anima_handles = getattr(anima, "close_file_handles", None)
                if callable(close_anima_handles):
                    close_anima_handles()
    finally:
        _restore_env(guard_env)
        _restore_env(native_env)

    assert scorecard["ok"] is True, scorecard
    assert scorecard["strict_fallback_matrix_passed"] is True, scorecard
    assert scorecard["representative_fallback_matrix_passed"] is True, scorecard
    assert scorecard["supported_shadow_probe_passed"] is True, scorecard
    assert scorecard["supported_probe_matrix_passed"] is True, scorecard
    assert scorecard["supported_probe_matrix_case_count"] == 4, scorecard
    assert scorecard["representative_supported_probe_matrix_passed"] is True, scorecard
    assert scorecard["promotion_ready"] is False, scorecard
    assert scorecard["representative_training_matrix_passed"] is False, scorecard
    assert "native_cache_reader_training_dispatch_not_implemented" in scorecard["promotion_blockers"], scorecard
    assert "cache_reader_dispatch_route_not_promoted" in scorecard["promotion_blockers"], scorecard
    _assert_closed(scorecard)

    matrix = scorecard["fallback_matrix"]
    assert matrix["ok"] is True, matrix
    assert matrix["failed_case_count"] == 0, matrix
    assert matrix["case_count"] >= 10, matrix
    _assert_closed(matrix)

    probe = scorecard["supported_shadow_probe"]
    assert probe["ok"] is True, probe
    assert probe["training_experimental_allowed"] is True, probe
    assert probe["parity_guard_passed"] is True, probe
    assert probe["batch_parity_guard_passed"] is True, probe
    assert probe["batch_payload_parity_guard_passed"] is True, probe
    assert probe["torch_owned_tensor_handoff_guard_passed"] is True, probe
    assert probe["batch_handoff_session_shadow_passed"] is True, probe
    assert probe["batch_dispatch_contract_ready"] is True, probe
    assert probe["text_payload_parity_guard_passed"] is True, probe
    assert "encoder_hidden_states" in probe["text_payload_fields"], probe
    assert "attention_mask" in probe["text_payload_fields"], probe
    assert probe["tensor_parity_count"] == 2, probe
    assert probe["tensor_parity_matches"] == 2, probe
    assert probe["batch_handle_count"] == 1, probe
    assert "representative_training_matrix_not_passed" in probe["native_dispatch_blockers"], probe
    _assert_closed(probe)

    supported_cases = scorecard["supported_probe_matrix"]
    case_ids = {str(case.get("case_id")) for case in supported_cases}
    assert {
        "primary_supported_single_worker",
        "anima_flat_single_worker",
        "newbie_single_item_batch",
        "newbie_two_batch_two_parity",
    }.issubset(case_ids), scorecard
    for case in supported_cases:
        assert case["supported_shadow_probe_passed"] is True, case
        assert case["failed_probe_checks"] == [], case
        assert case["probe"]["text_payload_parity_guard_passed"] is True, case
        _assert_closed(case)
        _assert_closed(case["probe"])

    return {
        "schema_version": 1,
        "probe": "cache_reader_dispatch_scorecard_smoke",
        "ok": True,
        "fallback_matrix_case_count": matrix["case_count"],
        "supported_shadow_probe_passed": scorecard["supported_shadow_probe_passed"],
        "supported_probe_matrix_case_count": scorecard["supported_probe_matrix_case_count"],
        "supported_probe_matrix_case_ids": scorecard["supported_probe_matrix_case_ids"],
        "promotion_ready": scorecard["promotion_ready"],
        "promotion_blockers": scorecard["promotion_blockers"],
        "text_payload_fields": probe["text_payload_fields"],
        "native_dispatch_eligible": False,
        "fallback_to_python_batch": True,
        "training_path_enabled": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
