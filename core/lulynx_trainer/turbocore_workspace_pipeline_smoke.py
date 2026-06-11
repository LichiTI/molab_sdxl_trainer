"""Smoke tests for the TurboCore workspace/data-pipeline prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_workspace_pipeline import (  # noqa: E402
    NativeDataPipelinePrototype,
    StagedBatch,
    WorkspaceBufferSpec,
    build_workspace_pipeline_native_capability_stub,
    build_workspace_pipeline_prototype_report,
    run_workspace_pipeline_lifecycle_probe,
)


def _batch(batch_id: str) -> StagedBatch:
    return StagedBatch(
        batch_id=batch_id,
        payload={"sample_ids": [batch_id]},
        workspace=(
            WorkspaceBufferSpec.from_shape("latents", (1, 4, 8, 8), dtype=torch.float32),
            WorkspaceBufferSpec.from_shape("prompt", (1, 16, 32), dtype=torch.float32),
        ),
    )


def test_pipeline_reuses_workspace_buffers() -> None:
    pipeline = NativeDataPipelinePrototype(prefetch_depth=1)
    assert pipeline.submit(_batch("a")) is True
    first = pipeline.consume()
    assert first is not None
    first_latents = first.buffers["latents"]
    first.release()

    assert pipeline.submit(_batch("b")) is True
    second = pipeline.consume()
    assert second is not None
    assert second.buffers["latents"] is first_latents
    second.release()

    snapshot = pipeline.close()
    assert snapshot["training_path_enabled"] is False
    assert snapshot["workspace_pool"]["hits"] >= 2
    assert snapshot["workspace_pool"]["misses"] == 2
    assert snapshot["stats"]["released"] == 2


def test_pipeline_backpressure_and_empty_stall_stats() -> None:
    pipeline = NativeDataPipelinePrototype(prefetch_depth=1)
    assert pipeline.consume() is None
    assert pipeline.submit(_batch("a")) is True
    assert pipeline.submit(_batch("b")) is False
    snapshot = pipeline.close()
    assert snapshot["stats"]["queue_empty_stalls"] == 1
    assert snapshot["stats"]["queue_full_stalls"] == 1
    assert snapshot["stats"]["close_released"] == 1
    assert snapshot["ready"] == 0


def test_pipeline_context_manager_releases_lease() -> None:
    pipeline = NativeDataPipelinePrototype(prefetch_depth=2)
    assert pipeline.submit(_batch("a")) is True
    lease = pipeline.consume()
    assert lease is not None
    with lease:
        assert lease.batch.batch_id == "a"
    assert lease.released is True
    assert pipeline.snapshot()["in_flight"] == 0


def test_prototype_report_is_research_only() -> None:
    report = build_workspace_pipeline_prototype_report(prefetch_depth=3, workspace_mb=512)
    assert report["status"] == "python_abi_prototype"
    assert report["training_path_enabled"] is False
    assert report["workspace_mb"] == 512
    assert report["prefetch_depth"] == 3
    assert report["features"]["native_runtime"] is False


def test_native_capability_stub_is_inactive_schema() -> None:
    stub = build_workspace_pipeline_native_capability_stub()
    assert stub["status"] == "expected_native_schema"
    assert stub["training_path_enabled"] is False
    assert stub["features"]["workspace_pool"]["available"] is False
    assert stub["features"]["data_pipeline"]["available"] is False
    assert "workspace_acquire" in stub["features"]["workspace_pool"]["required_entrypoints"]
    assert "release_batch_lease" in stub["features"]["data_pipeline"]["required_entrypoints"]


def test_lifecycle_probe_reuses_and_drains() -> None:
    report = run_workspace_pipeline_lifecycle_probe(batches=5, prefetch_depth=2, workspace_mb=128)
    assert report["ok"] is True
    assert report["training_path_enabled"] is False
    assert report["submitted_batches"] == 5
    assert report["consumed_batches"] == 5
    assert report["in_flight"] == 0
    assert report["workspace_pool"]["hits"] > 0


if __name__ == "__main__":
    test_pipeline_reuses_workspace_buffers()
    test_pipeline_backpressure_and_empty_stall_stats()
    test_pipeline_context_manager_releases_lease()
    test_prototype_report_is_research_only()
    test_native_capability_stub_is_inactive_schema()
    test_lifecycle_probe_reuses_and_drains()
    print("turbocore_workspace_pipeline_smoke: ok")
