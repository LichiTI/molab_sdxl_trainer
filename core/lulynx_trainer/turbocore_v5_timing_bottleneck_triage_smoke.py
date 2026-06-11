"""Smoke checks for V5 timing bottleneck triage."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_timing_bottleneck_triage import (  # noqa: E402
    build_v5_timing_bottleneck_triage,
)


def run_smoke() -> dict[str, Any]:
    sync = build_v5_timing_bottleneck_triage(matrix_summary=_matrix(_summary(runtime_sync=True)))
    assert sync["ok"] is True, sync
    assert sync["primary_bottleneck"] == "stream_event_chain_sync_fast_path", sync
    assert sync["metrics"]["global_context_sync"] is True, sync
    assert sync["default_rollout_allowed"] is False, sync
    assert sync["auto_rollout_allowed"] is False, sync

    prepare = build_v5_timing_bottleneck_triage(matrix_summary=_matrix(_summary(prepare_ms=8.0)))
    assert prepare["ok"] is True, prepare
    assert prepare["primary_bottleneck"] == "dispatch_prepare_cache_fast_path", prepare

    grad = build_v5_timing_bottleneck_triage(matrix_summary=_matrix(_summary(grad_sync_ms=4.0)))
    assert grad["ok"] is True, grad
    assert grad["primary_bottleneck"] == "direct_grad_owner_buffer_fast_path", grad

    copyback = build_v5_timing_bottleneck_triage(matrix_summary=_matrix(_summary(copyback_ms=4.0)))
    assert copyback["ok"] is True, copyback
    assert copyback["primary_bottleneck"] == "copyback_defer_or_owner_state_snapshot", copyback

    missing = build_v5_timing_bottleneck_triage(matrix_summary=_matrix({"native_dispatch_executed": True}))
    assert missing["ok"] is False, missing
    assert "v5_p7_timing_summary_missing" in missing["blocked_reasons"], missing
    assert missing["default_training_path_enabled"] is False, missing

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_timing_bottleneck_triage_smoke",
        "ok": True,
        "sync_primary": sync["primary_bottleneck"],
        "prepare_primary": prepare["primary_bottleneck"],
        "grad_primary": grad["primary_bottleneck"],
        "copyback_primary": copyback["primary_bottleneck"],
        "recommended_next_step": sync["recommended_next_step"],
    }


def _matrix(native_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "cases": [
            {
                "case": {"name": "baseline_phase"},
                "summary": {"native_dispatch_executed": False},
            },
            {
                "case": {"name": "native_update_dispatch_promotion_perf"},
                "summary": native_summary,
            },
        ],
    }


def _summary(
    *,
    runtime_sync: bool = False,
    prepare_ms: float = 1.0,
    grad_sync_ms: float = 0.5,
    copyback_ms: float = 0.5,
) -> dict[str, Any]:
    return {
        "native_dispatch_executed": True,
        "native_dispatch_training_executor_timing_present": True,
        "native_dispatch_training_executor_elapsed_ms_mean": 10.0,
        "native_dispatch_training_executor_state_sync_ms_mean": 0.2,
        "native_dispatch_training_executor_state_sync_ms_last": 0.1,
        "native_dispatch_update_executor_elapsed_ms_mean": 10.0,
        "native_dispatch_update_executor_grad_sync_ms_mean": grad_sync_ms,
        "native_dispatch_update_executor_owner_step_ms_mean": 1.0,
        "native_dispatch_update_executor_copyback_ms_mean": copyback_ms,
        "native_dispatch_loop_dispatch_runtime_prepare_ms_mean": prepare_ms,
        "native_dispatch_update_executor_used_direct_grad": False,
        "native_dispatch_update_executor_native_kernel_present": True,
        "native_dispatch_owner_native_runtime_synchronization": (
            "cuCtxSynchronize_after_native_step" if runtime_sync else "event_chain_bound"
        ),
        "native_dispatch_owner_native_runtime_stream_binding": (
            "cuda_driver_default_stream_null_synchronized" if runtime_sync else "bound_training_stream"
        ),
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
