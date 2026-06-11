"""Smoke checks for the V5-P12 ctx-sync-free benchmark canary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_update_benchmark_matrix import (  # noqa: E402
    OWNER_NATIVE_LAUNCH_PERF_MAX_NUMEL,
    _build_matrix_performance_report,
    _summarize_matrix,
    build_matrix_payload,
)


def _args(*, cases: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        run=False,
        write_dry_run=False,
        keep_going=False,
        family="anima",
        cases=cases,
        profiles=["standard"],
        steps=24,
        steady_warmup=4,
        samples=2,
        resolution=64,
        network_dim=1,
        train_batch_size=1,
        source_data="",
        python="python",
        optimizer_performance_report="",
        out=str(PROJECT_ROOT / "temp" / "matrix_ctx_sync_free_canary_smoke"),
    )


def test_matrix_dry_run_builds_ctx_sync_free_canary_case() -> None:
    payload = build_matrix_payload(_args(cases=["native_update_dispatch_ctx_sync_free_canary"]), run=False)
    case = payload["cases"][0]
    command = case["command"]
    assert case["case"]["name"] == "native_update_dispatch_ctx_sync_free_canary", case
    assert case["case"]["evidence_role"] == "performance_canary", case
    assert case["case"]["performance_sample"] is True, case
    assert "--turbocore-native-update-dispatch-enabled" in command, case
    assert "--turbocore-native-update-training-path-enabled" in command, case
    assert "--turbocore-native-update-require-native-cuda" in command, case
    assert "--turbocore-native-update-defer-state-sync" in command, case
    assert "--turbocore-update-shadow-owner-native-event-chain-probe" in command, case
    assert "--turbocore-native-update-runtime-synchronization-policy" in command, case
    policy_index = command.index("--turbocore-native-update-runtime-synchronization-policy")
    assert command[policy_index + 1] == "borrowed_stream_event_chain", case
    cap_index = command.index("--turbocore-update-shadow-owner-native-launch-max-numel")
    assert command[cap_index + 1] == OWNER_NATIVE_LAUNCH_PERF_MAX_NUMEL, case
    assert "--turbocore-native-update-diagnostic-executor-replay" not in command, case
    assert "--turbocore-update-shadow-checkpoint-contract" not in command, case


def test_matrix_summary_surfaces_ctx_sync_free_runtime_sync_case() -> None:
    payload = {
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "cases": [
            {
                "case": {"name": "baseline_phase"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steady_mean_step_ms": 1000.0,
                },
            },
            {
                "case": {"name": "native_update_dispatch_promotion_perf"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steady_mean_step_ms": 800.0,
                    "native_dispatch_executed": True,
                    "native_dispatch_owner_native_runtime_synchronization": "cuCtxSynchronize_after_native_step",
                },
            },
            {
                "case": {"name": "native_update_dispatch_ctx_sync_free_canary"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steady_mean_step_ms": 900.0,
                    "native_dispatch_executed": True,
                    "native_dispatch_owner_native_runtime_synchronization": (
                        "borrowed_stream_event_chain_no_ctx_sync"
                    ),
                },
            },
        ],
    }
    summary = _summarize_matrix(payload)
    assert summary["native_dispatch_owner_native_runtime_synchronization_by_case"] == {
        "native_update_dispatch_promotion_perf": "cuCtxSynchronize_after_native_step",
        "native_update_dispatch_ctx_sync_free_canary": "borrowed_stream_event_chain_no_ctx_sync",
    }, summary
    assert summary["native_dispatch_context_synchronize_cases"] == [
        "native_update_dispatch_promotion_perf"
    ], summary
    assert summary["native_dispatch_ctx_sync_free_cases"] == [
        "native_update_dispatch_ctx_sync_free_canary"
    ], summary
    comparison = summary["native_dispatch_ctx_sync_free_comparison"]
    assert comparison["ready"] is True, summary
    assert comparison["baseline_case"] == "baseline_phase", summary
    assert comparison["context_sync_case"] == "native_update_dispatch_promotion_perf", summary
    assert comparison["ctx_sync_free_case"] == "native_update_dispatch_ctx_sync_free_canary", summary
    assert comparison["ctx_sync_free_speedup_vs_baseline"] == 1.1111, summary
    assert comparison["ctx_sync_free_speedup_vs_context_sync_native"] == 0.8889, summary
    assert comparison["context_sync_speedup_vs_baseline"] == 1.25, summary
    assert comparison["representative_candidate_ready"] is False, summary
    assert summary["native_dispatch_ctx_sync_free_promotion_priority_unchanged"] is True, summary


def test_ctx_sync_free_canary_is_not_representative_promotion_case() -> None:
    payload = {
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "cases": [
            {
                "case": {"name": "baseline_phase"},
                "returncode": 0,
                "summary": {"success": True, "steps_completed": 24, "mean_step_ms": 1000.0},
            },
            {
                "case": {"name": "native_update_dispatch_ctx_sync_free_canary", "performance_sample": True},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 24,
                    "mean_step_ms": 900.0,
                    "native_dispatch_executed": True,
                },
            },
        ],
        "summary": {
            "executed_count": 2,
            "all_success": True,
            "mean_step_ms_by_case": {
                "baseline_phase": 1000.0,
                "native_update_dispatch_ctx_sync_free_canary": 900.0,
            },
        },
    }
    report = _build_matrix_performance_report(payload)
    training_matrix = report["performance_gate"]["evidence"]["training_matrix"]
    assert training_matrix["native_case"] == "", report
    assert "native_dispatch_benchmark_case_missing" in report["blocked_reasons"], report


def main() -> int:
    test_matrix_dry_run_builds_ctx_sync_free_canary_case()
    test_matrix_summary_surfaces_ctx_sync_free_runtime_sync_case()
    test_ctx_sync_free_canary_is_not_representative_promotion_case()
    print("turbocore_v5_ctx_sync_free_canary_matrix_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
