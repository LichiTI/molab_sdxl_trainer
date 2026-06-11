"""Smoke probe for the Rust TurboCore workspace/data-pipeline lifecycle ABI."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_capabilities import probe_native_training_bridge  # noqa: E402


def _inject_native_artifact_dir_from_env() -> None:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    if not path.is_dir():
        return
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _load_native() -> Any:
    _inject_native_artifact_dir_from_env()
    return importlib.import_module("lulynx_native")


def run_smoke() -> dict[str, Any]:
    native = _load_native()
    bridge = probe_native_training_bridge()
    features = bridge.get("features") if isinstance(bridge.get("features"), dict) else {}
    workspace_feature = features.get("workspace_pool") if isinstance(features, dict) else {}
    pipeline_feature = features.get("data_pipeline") if isinstance(features, dict) else {}

    pool_id = int(native.create_workspace_pool(64 * 1024 * 1024))
    assert pool_id > 0, pool_id
    assert native.workspace_acquire(pool_id, "cpu:float32:1x4x8x8") is True
    assert native.workspace_acquire(pool_id, "cpu:float32:1x4x8x8") is False
    assert native.workspace_release(pool_id, "cpu:float32:1x4x8x8") is True
    assert native.workspace_acquire(pool_id, "cpu:float32:1x4x8x8") is True
    workspace_stats = native.workspace_stats(pool_id)
    assert isinstance(workspace_stats, dict), workspace_stats
    assert workspace_stats["training_path_enabled"] is False
    assert workspace_stats["hits"] == 1, workspace_stats
    assert workspace_stats["misses"] == 1, workspace_stats
    assert workspace_stats["busy_stalls"] == 1, workspace_stats

    pipeline_id = int(native.create_data_pipeline(2, pool_id))
    assert pipeline_id > 0, pipeline_id
    assert native.submit_staged_batch(pipeline_id, '{"batch_id":"a"}') is True
    assert native.submit_staged_batch(pipeline_id, '{"batch_id":"b"}') is True
    assert native.submit_staged_batch(pipeline_id, '{"batch_id":"c"}') is False
    raw_lease = native.consume_ready_batch(pipeline_id)
    assert isinstance(raw_lease, str) and raw_lease, raw_lease
    lease = json.loads(raw_lease)
    assert lease["training_path_enabled"] is False
    assert lease["batch_descriptor"] == '{"batch_id":"a"}'
    assert native.release_batch_lease(pipeline_id, lease["lease_id"]) is True
    assert native.release_batch_lease(pipeline_id, lease["lease_id"]) is False
    close_stats = native.close_data_pipeline(pipeline_id)
    assert isinstance(close_stats, dict), close_stats
    assert close_stats["closed"] is True
    assert close_stats["training_path_enabled"] is False
    assert close_stats["stats"]["submitted"] == 2, close_stats
    assert close_stats["stats"]["queue_full_stalls"] == 1, close_stats
    assert close_stats["stats"]["released"] == 1, close_stats
    assert close_stats["stats"]["invalid_releases"] == 1, close_stats

    assert native.workspace_release(pool_id, "cpu:float32:1x4x8x8") is True

    bulk_pipeline_id = int(native.create_data_pipeline(8, pool_id))
    assert bulk_pipeline_id > 0, bulk_pipeline_id
    submitted_bulk = native.submit_staged_batches(
        bulk_pipeline_id,
        [f'{{"batch_id":"bulk-{index}"}}' for index in range(12)],
    )
    assert submitted_bulk == 8, submitted_bulk
    bulk_leases = native.consume_ready_batches(bulk_pipeline_id, 4)
    assert isinstance(bulk_leases, list), bulk_leases
    assert len(bulk_leases) == 4, bulk_leases
    assert native.release_batch_leases(
        bulk_pipeline_id,
        [str(lease["lease_id"]) for lease in bulk_leases],
    ) == 4
    bulk_close_stats = native.close_data_pipeline(bulk_pipeline_id)
    assert bulk_close_stats["stats"]["submitted"] == 8, bulk_close_stats
    assert bulk_close_stats["stats"]["queue_full_stalls"] == 4, bulk_close_stats
    assert bulk_close_stats["stats"]["released"] == 4, bulk_close_stats

    drain_pipeline_id = int(native.create_data_pipeline(4, pool_id))
    assert drain_pipeline_id > 0, drain_pipeline_id
    assert native.submit_staged_batches(drain_pipeline_id, ["x", "y", "z"]) == 3
    assert native.consume_and_release_ready_batches(drain_pipeline_id, 8) == 3
    drain_close_stats = native.close_data_pipeline(drain_pipeline_id)
    assert drain_close_stats["stats"]["consumed"] == 3, drain_close_stats
    assert drain_close_stats["stats"]["released"] == 3, drain_close_stats

    indexed_pipeline_id = int(native.create_data_pipeline(4, pool_id))
    assert indexed_pipeline_id > 0, indexed_pipeline_id
    assert native.submit_indexed_batches(indexed_pipeline_id, 0, 6) == 4
    assert native.consume_and_release_counted_batches(indexed_pipeline_id, 8) == 4
    indexed_close_stats = native.close_data_pipeline(indexed_pipeline_id)
    assert indexed_close_stats["stats"]["submitted"] == 4, indexed_close_stats
    assert indexed_close_stats["stats"]["queue_full_stalls"] == 2, indexed_close_stats
    assert indexed_close_stats["stats"]["consumed"] == 4, indexed_close_stats
    assert indexed_close_stats["stats"]["released"] == 4, indexed_close_stats

    assert native.destroy_workspace_pool(pool_id) is True

    probe = native.run_turbocore_pipeline_lifecycle_probe(128, 4)
    assert probe["ok"] is True, probe
    assert probe["training_path_enabled"] is False, probe

    payload = {
        "schema_version": 1,
        "probe": "turbocore_native_pipeline_lifecycle_smoke",
        "ok": True,
        "training_path_enabled": False,
        "workspace_status": str((workspace_feature or {}).get("status", "")),
        "data_pipeline_status": str((pipeline_feature or {}).get("status", "")),
        "workspace_stats": workspace_stats,
        "pipeline_close_stats": close_stats,
        "bulk_pipeline_close_stats": bulk_close_stats,
        "drain_pipeline_close_stats": drain_close_stats,
        "indexed_pipeline_close_stats": indexed_close_stats,
        "lifecycle_probe": probe,
    }
    assert payload["workspace_status"] in {"lifecycle_ready", "capability_stub"}
    assert payload["data_pipeline_status"] in {"lifecycle_ready", "capability_stub"}
    return payload


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
