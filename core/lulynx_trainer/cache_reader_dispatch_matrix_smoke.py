"""Smoke probe for the cache reader dispatch strict-fallback matrix."""

from __future__ import annotations

import json
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
from core.turbocore_cache_reader_dispatch_matrix import build_cache_reader_dispatch_fallback_matrix  # noqa: E402


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


def _case_by_id(report: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in list(report.get("cases", []) or []):
        if isinstance(case, dict) and case.get("case_id") == case_id:
            return case
    raise AssertionError(f"missing matrix case: {case_id}")


def _assert_closed(value: dict[str, Any]) -> None:
    assert value["native_dispatch_eligible"] is False, value
    assert value["would_allow_native_dispatch"] is False, value
    assert value["fallback_to_python_batch"] is True, value
    assert value["returns_tensor_payloads"] is False, value
    assert value["cache_reader_path_enabled"] is False, value
    assert value["prefetch_queue_training_path_enabled"] is False, value
    assert value["training_path_enabled"] is False, value


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lulynx_cache_reader_dispatch_matrix_") as tmp:
        root = Path(tmp)
        _write_newbie_cache(root, "flat_a", 4)
        _write_newbie_cache(root, "flat_b", 4)
        dataset = NewbieCachedDataset(
            root,
            schema=NewbieCacheSchema(require_schema_version=True),
            cache_mmap=True,
            cache_lazy=True,
        )
        try:
            report = build_cache_reader_dispatch_fallback_matrix(
                dataset,
                batch_size=2,
                prefetch_factor=2,
                strict_fallback=True,
            )
        finally:
            close_handles = getattr(dataset, "close_file_handles", None)
            if callable(close_handles):
                close_handles()

    assert report["ok"] is True, report
    assert report["strict_fallback_matrix_passed"] is True, report
    assert report["representative_fallback_matrix_passed"] is True, report
    assert report["representative_training_matrix_passed"] is False, report
    assert report["case_count"] >= 10, report
    assert report["failed_case_count"] == 0, report
    assert "native_cache_reader_training_dispatch_not_implemented" in report["native_dispatch_blockers"], report
    assert "representative_training_matrix_not_passed" in report["native_dispatch_blockers"], report
    _assert_closed(report)

    baseline = _case_by_id(report, "baseline_supported_single_worker")
    assert baseline["ok"] is True, baseline
    assert baseline["shadow_gate_ready"] is True, baseline
    assert baseline["shadow_gate_blockers"] == [], baseline
    assert "python_dataloader_batch_remains_authoritative" in baseline["native_dispatch_blockers"], baseline
    _assert_closed(baseline)

    shuffle = _case_by_id(report, "shuffle_sampler_reseed")
    assert shuffle["shadow_gate_ready"] is False, shuffle
    assert "shuffle_order_parity_not_ready" in shuffle["shadow_gate_blockers"], shuffle
    assert "sampler_reseed_policy_not_promoted" in shuffle["native_dispatch_blockers"], shuffle

    drop_last = _case_by_id(report, "drop_last_batch_boundary")
    assert "drop_last_parity_not_ready" in drop_last["shadow_gate_blockers"], drop_last
    assert "drop_last_batch_boundary_not_promoted" in drop_last["native_dispatch_blockers"], drop_last

    worker = _case_by_id(report, "multi_worker_sample_ownership")
    assert "multi_worker_cache_reader_parity_not_ready" in worker["shadow_gate_blockers"], worker
    assert "multi_worker_sample_ownership_not_promoted" in worker["native_dispatch_blockers"], worker

    bucket = _case_by_id(report, "bucket_sampler_ownership")
    assert "bucket_sampler_cache_reader_parity_not_ready" in bucket["shadow_gate_blockers"], bucket
    assert "bucket_sampler_ownership_not_promoted" in bucket["native_dispatch_blockers"], bucket

    crop = _case_by_id(report, "latent_crop_padding_ownership")
    assert "latent_crop_padding_parity_not_ready" in crop["shadow_gate_blockers"], crop
    assert "latent_crop_padding_ownership_not_promoted" in crop["native_dispatch_blockers"], crop

    text = _case_by_id(report, "text_token_shape_policy")
    assert text["shadow_gate_ready"] is True, text
    assert "text_token_shape_policy_not_promoted" in text["native_dispatch_blockers"], text

    caption = _case_by_id(report, "caption_runtime_transform_ownership")
    assert caption["shadow_gate_ready"] is True, caption
    assert "caption_runtime_transform_ownership_not_promoted" in caption["native_dispatch_blockers"], caption

    unsupported = _case_by_id(report, "unsupported_cached_dataset_class")
    assert "unsupported_cached_dataset_class" in unsupported["shadow_gate_blockers"], unsupported
    assert "unsupported_cached_dataset_class" in unsupported["native_dispatch_blockers"], unsupported

    return {
        "schema_version": 1,
        "probe": "cache_reader_dispatch_matrix_smoke",
        "ok": True,
        "case_count": report["case_count"],
        "passed_case_count": report["passed_case_count"],
        "failed_case_count": report["failed_case_count"],
        "baseline_shadow_gate_ready": baseline["shadow_gate_ready"],
        "shuffle_blockers": shuffle["native_dispatch_blockers"],
        "text_token_shadow_gate_ready": text["shadow_gate_ready"],
        "native_dispatch_eligible": False,
        "fallback_to_python_batch": True,
        "training_path_enabled": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
