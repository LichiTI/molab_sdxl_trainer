"""Smoke probe for the TurboCore dataset staging planner."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_capabilities import probe_native_training_bridge  # noqa: E402
from core.turbocore_dataset_staging import (  # noqa: E402
    plan_dataset_staging,
    run_native_dataset_staging_bulk_pipeline_probe,
    run_native_dataset_staging_handle_probe,
    run_native_dataset_staging_lazy_bulk_pipeline_probe,
    run_native_dataset_staging_lazy_fast_bulk_pipeline_probe,
    run_native_dataset_staging_pipeline_probe,
)
from core.turbocore_dataset_staging_session import (  # noqa: E402
    run_native_dataset_descriptor_session_probe,
    run_native_dataset_staging_lazy_affine_session_probe,
)


def _assert_common(payload: dict[str, Any]) -> None:
    assert payload["ok"] is True, payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["indices_returned"] is False, payload
    assert isinstance(payload["chunks"], list), payload
    assert payload["chunk_count"] == len(payload["chunks"]), payload


def run_smoke() -> dict[str, Any]:
    sequential = plan_dataset_staging(
        sample_count=10,
        batch_size=4,
        drop_last=False,
        shuffle=False,
        seed=123,
        prefetch_depth=2,
        chunk_size=2,
    )
    _assert_common(sequential)
    assert sequential["batch_count"] == 3, sequential
    assert sequential["provider"] == "python_dataset_staging", sequential
    assert sequential.get("native_skip_reason") == "native_dataset_staging_skipped:sequential_python_faster", sequential
    assert sequential["covered_samples"] == 10, sequential
    assert sequential["dropped_samples"] == 0, sequential
    assert sequential["chunks"][0] == {
        "start_batch": 0,
        "batch_count": 2,
        "start_sample": 0,
        "sample_count": 8,
        "order": "sequential_range",
    }, sequential

    dropped = plan_dataset_staging(
        sample_count=10,
        batch_size=4,
        drop_last=True,
        shuffle=False,
        seed=123,
        prefetch_depth=2,
        chunk_size=8,
    )
    _assert_common(dropped)
    assert dropped["batch_count"] == 2, dropped
    assert dropped["covered_samples"] == 8, dropped
    assert dropped["dropped_samples"] == 2, dropped

    shuffled_a = plan_dataset_staging(
        sample_count=33,
        batch_size=5,
        drop_last=True,
        shuffle=True,
        seed=77,
        prefetch_depth=4,
        chunk_size=3,
    )
    shuffled_b = plan_dataset_staging(
        sample_count=33,
        batch_size=5,
        drop_last=True,
        shuffle=True,
        seed=77,
        prefetch_depth=4,
        chunk_size=3,
    )
    shuffled_c = plan_dataset_staging(
        sample_count=33,
        batch_size=5,
        drop_last=True,
        shuffle=True,
        seed=78,
        prefetch_depth=4,
        chunk_size=3,
    )
    _assert_common(shuffled_a)
    assert shuffled_a["covered_samples"] == 30, shuffled_a
    assert shuffled_a["provider"] == "native_dataset_staging", shuffled_a
    assert shuffled_a["native_runtime"] is True, shuffled_a
    assert shuffled_a["dropped_samples"] == 3, shuffled_a
    assert shuffled_a["index_checksum"] == shuffled_b["index_checksum"], (shuffled_a, shuffled_b)
    assert shuffled_a["index_preview"] == shuffled_b["index_preview"], (shuffled_a, shuffled_b)
    assert shuffled_a["index_checksum"] != shuffled_c["index_checksum"], (shuffled_a, shuffled_c)
    assert shuffled_a["chunks"][0]["order"] == "deterministic_shuffle", shuffled_a

    with patch.dict(os.environ, {"LULYNX_DISABLE_NATIVE_DATA_STAGING": "1"}):
        fallback = plan_dataset_staging(
            sample_count=9,
            batch_size=4,
            drop_last=False,
            shuffle=True,
            seed=5,
            prefetch_depth=2,
            chunk_size=2,
            prefer_native=True,
        )
    _assert_common(fallback)
    assert fallback["provider"] == "python_dataset_staging", fallback
    assert fallback["native_runtime"] is False, fallback
    assert "native_fallback_reason" in fallback, fallback

    handle_probe = run_native_dataset_staging_handle_probe(
        sample_count=64,
        batch_size=4,
        drop_last=True,
        shuffle=True,
        seed=11,
        prefetch_depth=8,
        chunk_size=5,
    )
    assert handle_probe["ok"] is True, handle_probe
    assert handle_probe["batch_count"] == 16, handle_probe
    assert handle_probe["emitted_batches"] == 16, handle_probe
    assert handle_probe["chunk_count"] == 4, handle_probe
    assert handle_probe["training_path_enabled"] is False, handle_probe

    pipeline_probe = run_native_dataset_staging_pipeline_probe(
        sample_count=64,
        batch_size=4,
        drop_last=True,
        shuffle=True,
        seed=11,
        prefetch_depth=8,
        chunk_size=5,
    )
    assert pipeline_probe["ok"] is True, pipeline_probe
    assert pipeline_probe["batch_count"] == 16, pipeline_probe
    assert pipeline_probe["submitted_batches"] == 16, pipeline_probe
    assert pipeline_probe["consumed_batches"] == 16, pipeline_probe
    assert pipeline_probe["training_path_enabled"] is False, pipeline_probe

    bulk_probe = run_native_dataset_staging_bulk_pipeline_probe(
        sample_count=64,
        batch_size=4,
        drop_last=True,
        shuffle=True,
        seed=11,
        prefetch_depth=8,
        chunk_size=5,
    )
    assert bulk_probe["ok"] is True, bulk_probe
    assert bulk_probe["batch_count"] == 16, bulk_probe
    assert bulk_probe["submitted_batches"] == 16, bulk_probe
    assert bulk_probe["consumed_batches"] == 16, bulk_probe
    assert bulk_probe["released_batches"] == 16, bulk_probe
    assert bulk_probe["training_path_enabled"] is False, bulk_probe

    lazy_probe = run_native_dataset_staging_lazy_bulk_pipeline_probe(
        sample_count=64,
        batch_size=4,
        drop_last=True,
        seed=11,
        prefetch_depth=8,
        chunk_size=5,
    )
    assert lazy_probe["ok"] is True, lazy_probe
    assert lazy_probe["batch_count"] == 16, lazy_probe
    assert lazy_probe["submitted_batches"] == 16, lazy_probe
    assert lazy_probe["consumed_batches"] == 16, lazy_probe
    assert lazy_probe["released_batches"] == 16, lazy_probe
    assert lazy_probe["native_index_materialized"] is False, lazy_probe
    assert lazy_probe["shuffle_kind"] == "lazy_affine_permutation_v1", lazy_probe
    assert lazy_probe["training_path_enabled"] is False, lazy_probe

    lazy_fast_probe = run_native_dataset_staging_lazy_fast_bulk_pipeline_probe(
        sample_count=64,
        batch_size=4,
        drop_last=True,
        seed=11,
        prefetch_depth=8,
        chunk_size=5,
    )
    assert lazy_fast_probe["ok"] is True, lazy_fast_probe
    assert lazy_fast_probe["batch_count"] == 16, lazy_fast_probe
    assert lazy_fast_probe["submitted_batches"] == 16, lazy_fast_probe
    assert lazy_fast_probe["consumed_batches"] == 16, lazy_fast_probe
    assert lazy_fast_probe["released_batches"] == 16, lazy_fast_probe
    assert lazy_fast_probe["native_index_materialized"] is False, lazy_fast_probe
    assert lazy_fast_probe["runtime_summary_only"] is True, lazy_fast_probe
    assert lazy_fast_probe["checksum_covers_full_order"] is False, lazy_fast_probe
    assert lazy_fast_probe["shuffle_kind"] == "lazy_affine_permutation_v1", lazy_fast_probe
    assert lazy_fast_probe["training_path_enabled"] is False, lazy_fast_probe

    session_probe = run_native_dataset_staging_lazy_affine_session_probe(
        sample_count=64,
        batch_size=4,
        drop_last=True,
        seed=11,
        prefetch_depth=8,
        chunk_size=5,
        epochs=2,
    )
    assert session_probe["ok"] is True, session_probe
    assert session_probe["batch_count"] == 16, session_probe
    assert session_probe["epochs"] == 2, session_probe
    assert session_probe["native_index_materialized"] is False, session_probe
    assert session_probe["long_lived_descriptor"] is True, session_probe
    assert session_probe["training_path_enabled"] is False, session_probe
    assert session_probe["last_epoch"]["submitted_batches"] == 16, session_probe
    assert session_probe["final_stats"]["epochs_run"] == 2, session_probe

    descriptor_manifest = {
        "samples": [
            {
                "id": "img_0001",
                "path": "H:/lulynx-trainer/sucai/img_0001.png",
                "caption_path": "H:/lulynx-trainer/sucai/img_0001.txt",
                "width": 512,
                "height": 768,
                "bucket": "512x768",
            },
            {
                "id": "img_0002",
                "path": "H:/lulynx-trainer/sucai/img_0002.png",
                "caption_path": "H:/lulynx-trainer/sucai/img_0002.txt",
                "width": 768,
                "height": 512,
                "bucket": "768x512",
            },
            {
                "id": "img_0003",
                "path": "H:/lulynx-trainer/sucai/img_0003.png",
                "caption_path": "H:/lulynx-trainer/sucai/img_0003.txt",
                "width": 512,
                "height": 768,
                "bucket": "512x768",
            },
        ]
    }
    descriptor_probe = run_native_dataset_descriptor_session_probe(
        descriptor_manifest,
        batch_size=2,
        drop_last=False,
        prefetch_depth=4,
        chunk_size=2,
        epochs=2,
    )
    assert descriptor_probe["ok"] is True, descriptor_probe
    assert descriptor_probe["descriptor_count"] == 3, descriptor_probe
    assert descriptor_probe["batch_count"] == 2, descriptor_probe
    assert descriptor_probe["sample_descriptors_owned"] is True, descriptor_probe
    assert descriptor_probe["long_lived_descriptor"] is True, descriptor_probe
    assert descriptor_probe["training_path_enabled"] is False, descriptor_probe
    assert descriptor_probe["last_epoch"]["submitted_batches"] == 2, descriptor_probe
    assert descriptor_probe["first_chunk"]["batch_count"] == 2, descriptor_probe
    assert descriptor_probe["first_chunk"]["descriptor_preview"][0]["id"] == "img_0001", descriptor_probe
    assert descriptor_probe["worker_probe"]["ok"] is True, descriptor_probe
    assert descriptor_probe["worker_probe"]["submitted_batches"] == 2, descriptor_probe
    assert descriptor_probe["worker_probe"]["backpressure_mode"] == "bounded_worker_queue_v1", descriptor_probe
    assert descriptor_probe["worker_probe"]["worker_results_owned"] is True, descriptor_probe
    assert descriptor_probe["worker_probe"]["result_preview_returned"] is True, descriptor_probe
    assert descriptor_probe["worker_probe"]["result_preview"][0]["descriptor_preview"][0]["id"] == "img_0001", descriptor_probe
    assert descriptor_probe["worker_results_owned"] is True, descriptor_probe
    assert descriptor_probe["parity_probe"]["ok"] is True, descriptor_probe
    assert descriptor_probe["parity_probe"]["debug_only"] is True, descriptor_probe
    assert descriptor_probe["parity_probe"]["shadow_run"] is True, descriptor_probe
    assert descriptor_probe["parity_probe"]["mismatch_count"] == 0, descriptor_probe
    assert descriptor_probe["parity_probe"]["checksum_fast_path"] is True, descriptor_probe
    assert descriptor_probe["descriptor_parity_ok"] is True, descriptor_probe
    descriptor_stats = descriptor_probe["initial_stats"]
    assert descriptor_stats["descriptor_ownership"] == "native_explicit_samples", descriptor_probe
    assert descriptor_stats["bucket_counts"] == {"512x768": 2, "768x512": 1}, descriptor_probe
    assert descriptor_stats["descriptor_preview"][0]["id"] == "img_0001", descriptor_probe
    assert descriptor_probe["final_stats"]["epochs_run"] == 3, descriptor_probe

    bridge = probe_native_training_bridge()
    features = bridge.get("features") if isinstance(bridge.get("features"), dict) else {}
    dataset_feature = features.get("dataset_staging") if isinstance(features, dict) else {}
    descriptor_feature = dataset_feature.get("descriptor_session") if isinstance(dataset_feature, dict) else {}
    samplers = dataset_feature.get("samplers") if isinstance(dataset_feature, dict) else []
    lazy_sampler = next(
        (
            item
            for item in samplers
            if isinstance(item, dict) and item.get("name") == "lazy_affine_permutation_v1"
        ),
        {},
    )
    if lazy_sampler:
        assert lazy_sampler["status"] == "experimental_probe_ready", lazy_sampler
        assert lazy_sampler["materializes_index"] is False, lazy_sampler
        assert lazy_sampler["runtime_summary_only_supported"] is True, lazy_sampler
        assert lazy_sampler["long_lived_descriptor_supported"] is True, lazy_sampler
        assert lazy_sampler["training_path_enabled"] is False, lazy_sampler
    if descriptor_feature:
        assert descriptor_feature["status"] == "experimental_probe_ready", descriptor_feature
        assert descriptor_feature["sample_descriptors_owned"] is True, descriptor_feature
        assert descriptor_feature["supports_chunk_cursor"] is True, descriptor_feature
        assert descriptor_feature["supports_worker_backpressure_probe"] is True, descriptor_feature
        assert descriptor_feature["supports_descriptor_parity_shadow_probe"] is True, descriptor_feature
        assert descriptor_feature["supports_unified_shadow_lifecycle_probe"] is True, descriptor_feature
        assert descriptor_feature["training_path_enabled"] is False, descriptor_feature
    assert dataset_feature["supports_sampler_order_parity_shadow_probe"] is True, dataset_feature
    assert dataset_feature["supports_unified_shadow_lifecycle_probe"] is True, dataset_feature
    assert "validate_dataset_sampler_order_parity" in dataset_feature["entrypoints"], dataset_feature
    assert "run_dataset_shadow_lifecycle_probe" in dataset_feature["entrypoints"], dataset_feature

    return {
        "schema_version": 1,
        "probe": "turbocore_dataset_staging_smoke",
        "ok": True,
        "training_path_enabled": False,
        "sequential_provider": sequential.get("provider"),
        "shuffle_provider": shuffled_a.get("provider"),
        "fallback_provider": fallback.get("provider"),
        "dataset_staging_status": str((dataset_feature or {}).get("status", "")),
        "lazy_sampler_status": str((lazy_sampler or {}).get("status", "")),
        "sequential": sequential,
        "dropped": dropped,
        "shuffled": shuffled_a,
        "fallback": fallback,
        "handle_probe": handle_probe,
        "pipeline_probe": pipeline_probe,
        "bulk_probe": bulk_probe,
        "lazy_probe": lazy_probe,
        "lazy_fast_probe": lazy_fast_probe,
        "session_probe": session_probe,
        "descriptor_probe": descriptor_probe,
    }


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
