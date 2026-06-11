"""Smoke checks for TurboCore update benchmark matrix dry-run."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_update_benchmark_matrix import (  # noqa: E402
    OWNER_NATIVE_LAUNCH_PERF_MAX_NUMEL,
    OWNER_NATIVE_LAUNCH_SAFE_MAX_NUMEL,
    _build_matrix_performance_report,
    _summarize_benchmark_summary,
    _summarize_matrix,
    build_matrix_payload,
)


def test_matrix_dry_run_builds_expected_commands() -> None:
    args = argparse.Namespace(
        run=False,
        write_dry_run=False,
        keep_going=False,
        family="anima",
        cases=["baseline_phase", "gate_profile", "native_update_dispatch", "owner_native_launch_small"],
        profiles=["standard"],
        steps=2,
        steady_warmup=0,
        samples=2,
        resolution=64,
        network_dim=1,
        train_batch_size=1,
        source_data="",
        python="python",
        optimizer_performance_report="",
        out=str(PROJECT_ROOT / "temp" / "matrix_smoke"),
    )
    payload = build_matrix_payload(args, run=False)
    assert payload["run"] is False, payload
    assert payload["summary"]["case_count"] == 4, payload
    baseline = payload["cases"][0]
    gate = payload["cases"][1]
    dispatch = payload["cases"][2]
    owner = payload["cases"][3]
    assert baseline["case"]["name"] == "baseline_phase", baseline
    assert "--phase-profile" in baseline["command"], baseline
    assert "--turbocore-native-update-mode" not in baseline["command"], baseline
    assert gate["case"]["name"] == "gate_profile", gate
    assert "--turbocore-native-update-mode" in gate["command"], gate
    assert "--turbocore-update-shadow-copyback-probe" in gate["command"], gate
    assert "--turbocore-update-shadow-native-binding-probe" in gate["command"], gate
    assert "--turbocore-update-shadow-save-owner-state" in gate["command"], gate
    assert dispatch["case"]["name"] == "native_update_dispatch", dispatch
    assert "--turbocore-native-update-dispatch-enabled" in dispatch["command"], dispatch
    assert "--turbocore-native-update-training-path-enabled" in dispatch["command"], dispatch
    assert "--turbocore-native-update-require-native-cuda" in dispatch["command"], dispatch
    assert "--turbocore-native-update-allow-missing-kernel" in dispatch["command"], dispatch
    assert "--turbocore-native-update-diagnostic-executor-replay" in dispatch["command"], dispatch
    assert "--turbocore-update-shadow-stop-after-consecutive-passes" not in dispatch["command"], dispatch
    assert "native_experimental" in dispatch["command"], dispatch
    assert "--turbocore-update-shadow-owner-native-launch-probe" in dispatch["command"], dispatch
    dispatch_cap_index = dispatch["command"].index("--turbocore-update-shadow-owner-native-launch-max-numel")
    assert dispatch["command"][dispatch_cap_index + 1] == OWNER_NATIVE_LAUNCH_SAFE_MAX_NUMEL, dispatch
    assert "--turbocore-update-shadow-owner-native-event-chain-probe" in dispatch["command"], dispatch
    assert owner["case"]["name"] == "owner_native_launch_small", owner
    assert "--turbocore-update-shadow-owner-native-launch-max-numel" in owner["command"], owner
    assert OWNER_NATIVE_LAUNCH_SAFE_MAX_NUMEL in owner["command"], owner
    assert "--turbocore-native-update-dispatch-enabled" not in owner["command"], owner
    assert payload["summary"]["executed_count"] == 0, payload


def test_matrix_dry_run_builds_clean_perf_case() -> None:
    args = argparse.Namespace(
        run=False,
        write_dry_run=False,
        keep_going=False,
        family="anima",
        cases=["native_update_dispatch_perf"],
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
        out=str(PROJECT_ROOT / "temp" / "matrix_perf_smoke"),
    )
    payload = build_matrix_payload(args, run=False)
    perf = payload["cases"][0]
    assert perf["case"]["name"] == "native_update_dispatch_perf", perf
    assert perf["case"]["evidence_role"] == "performance", perf
    assert perf["case"]["performance_sample"] is True, perf
    assert "--turbocore-native-update-dispatch-enabled" in perf["command"], perf
    assert "--turbocore-native-update-training-path-enabled" in perf["command"], perf
    assert "--turbocore-update-shadow-stop-after-consecutive-passes" in perf["command"], perf
    assert "--turbocore-native-update-defer-state-sync" in perf["command"], perf
    assert "--turbocore-native-update-diagnostic-executor-replay" not in perf["command"], perf
    perf_cap_index = perf["command"].index("--turbocore-update-shadow-owner-native-launch-max-numel")
    assert perf["command"][perf_cap_index + 1] == OWNER_NATIVE_LAUNCH_PERF_MAX_NUMEL, perf


def test_matrix_dry_run_builds_promotion_perf_case() -> None:
    args = argparse.Namespace(
        run=False,
        write_dry_run=False,
        keep_going=False,
        family="anima",
        cases=["native_update_dispatch_promotion_perf"],
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
        out=str(PROJECT_ROOT / "temp" / "matrix_promotion_perf_smoke"),
    )
    payload = build_matrix_payload(args, run=False)
    perf = payload["cases"][0]
    command = perf["command"]
    assert perf["case"]["name"] == "native_update_dispatch_promotion_perf", perf
    assert perf["case"]["evidence_role"] == "performance", perf
    assert perf["case"]["performance_sample"] is True, perf
    assert "--turbocore-native-update-dispatch-enabled" in command, perf
    assert "--turbocore-native-update-training-path-enabled" in command, perf
    assert "--turbocore-native-update-defer-state-sync" in command, perf
    assert "--turbocore-update-shadow-save-owner-state" in command, perf
    assert "--turbocore-update-shadow-copyback-probe" in command, perf
    assert "--turbocore-update-shadow-copyback-dispatch-experimental" in command, perf
    assert "--turbocore-update-shadow-native-binding-probe" in command, perf
    assert "--turbocore-update-shadow-owner-native-launch-probe" in command, perf
    assert "--turbocore-update-shadow-owner-native-event-chain-probe" in command, perf
    assert "--turbocore-update-shadow-checkpoint-contract" not in command, perf
    assert "--turbocore-native-update-diagnostic-executor-replay" not in command, perf
    perf_cap_index = command.index("--turbocore-update-shadow-owner-native-launch-max-numel")
    assert command[perf_cap_index + 1] == OWNER_NATIVE_LAUNCH_PERF_MAX_NUMEL, perf
    sample_index = command.index("--turbocore-update-shadow-compare-sample-params")
    assert command[sample_index + 1] == "8", perf


def test_matrix_summary_surfaces_native_dispatch_status() -> None:
    payload = {
        "cases": [
            {
                "case": {"name": "native_update_dispatch"},
                "returncode": 0,
                "summary": {
                    "mean_step_ms": 1000.0,
                    "peak_vram_mb": 4096.0,
                    "native_dispatch_requested": True,
                    "native_dispatch_gate_requested": True,
                    "native_dispatch_executed": False,
                    "native_dispatch_disabled_for_run": True,
                    "native_dispatch_recovery_observation_bridge_ready": True,
                    "native_dispatch_training_dispatch_recovery_ready": False,
                    "native_dispatch_training_dispatch_recovery_blocked": True,
                    "native_dispatch_direct_gradient_write_boundary_ready": True,
                    "native_dispatch_direct_gradient_write_native_supported": False,
                    "native_dispatch_direct_gradient_write_lifecycle_ready": True,
                    "native_dispatch_direct_gradient_write_bound": False,
                    "native_dispatch_direct_gradient_write_default_off": True,
                    "native_dispatch_stream_ordering_ready": True,
                    "native_dispatch_stream_lifetime_ownership_boundary_ready": True,
                    "native_dispatch_stream_lifetime_ownership_evidence_bound": False,
                    "native_dispatch_stream_lifetime_ownership_default_off": True,
                    "native_dispatch_stream_lifetime_ownership_ready": False,
                    "native_dispatch_training_flat_owner_boundary_ready": True,
                    "native_dispatch_training_flat_owner_reference_ready": True,
                    "native_dispatch_training_flat_owner_bound": False,
                    "native_dispatch_training_flat_owner_default_off": True,
                    "native_dispatch_training_dispatch_kernel_boundary_ready": True,
                    "native_dispatch_training_dispatch_kernel_evidence_present": True,
                    "native_dispatch_training_dispatch_kernel_bound": False,
                    "native_dispatch_training_dispatch_kernel_default_off": True,
                    "native_dispatch_training_runtime_executor_boundary_ready": True,
                    "native_dispatch_training_runtime_executor_bound": False,
                    "native_dispatch_training_runtime_executor_default_off": True,
                    "native_dispatch_training_path_request_boundary_ready": True,
                    "native_dispatch_explicit_training_path_requested": False,
                    "native_dispatch_training_path_default_off": True,
                    "native_dispatch_rehearsal_evidence_ready": True,
                    "native_dispatch_training_promotion_preconditions_ready": False,
                    "native_dispatch_missing_for_training_promotion": [
                        "stream_lifetime_ownership_default_off",
                        "direct_gradient_write_default_off",
                        "native_training_flat_owner_default_off",
                        "native_training_dispatch_kernel_default_off",
                        "native_dispatch_training_runtime_executor_default_off",
                        "native_dispatch_training_path_default_off",
                    ],
                    "native_dispatch_training_executor_preconditions_ready": False,
                    "native_dispatch_diagnostic_executor_preconditions_ready": False,
                    "native_dispatch_execution_blocked_reasons": ["native_dispatch_runtime_default_off"],
                    "native_dispatch_diagnostic_executor_blocked_reasons": [
                        "native_dispatch_diagnostic_clone_context_disabled"
                    ],
                    "native_dispatch_executor_probe_blocked_reasons": [
                        "native_dispatch_diagnostic_executor_call_disabled"
                    ],
                    "native_dispatch_diagnostic_replay_reports": 1,
                    "native_dispatch_diagnostic_replay_called": True,
                    "native_dispatch_diagnostic_replay_ok": True,
                    "native_dispatch_diagnostic_replay_native_step_executed": False,
                    "native_dispatch_diagnostic_replay_training_path_enabled": False,
                    "native_dispatch_diagnostic_replay_preconditions_ready": True,
                    "native_dispatch_diagnostic_replay_blocked_reasons": [],
                    "owner_native_launch_probe_present": True,
                    "owner_native_launch_attempted": True,
                    "owner_native_launch_ok": True,
                    "owner_native_kernel_executed": True,
                    "owner_native_parity_ok": True,
                    "owner_native_numel": 4096,
                    "gate_blocked_reasons": ["native_dispatch_runtime_not_implemented"],
                    "performance_gate_blocked_reasons": ["native_dispatch_not_executed_in_benchmark_case"],
                },
            }
        ]
    }
    summary = _summarize_matrix(payload)
    assert summary["native_dispatch_requested_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_gate_requested_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_executed_cases"] == [], summary
    assert summary["native_dispatch_disabled_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_rehearsal_evidence_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_recovery_observation_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_training_recovery_ready_cases"] == [], summary
    assert summary["native_dispatch_training_recovery_blocked_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_direct_gradient_write_boundary_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_direct_gradient_write_native_supported_cases"] == [], summary
    assert summary["native_dispatch_direct_gradient_write_lifecycle_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_direct_gradient_write_bound_cases"] == [], summary
    assert summary["native_dispatch_direct_gradient_write_default_off_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_stream_ordering_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_stream_lifetime_ownership_boundary_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_stream_lifetime_ownership_evidence_bound_cases"] == [], summary
    assert summary["native_dispatch_stream_lifetime_ownership_default_off_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_stream_lifetime_ownership_ready_cases"] == [], summary
    assert summary["native_dispatch_training_flat_owner_boundary_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_training_flat_owner_reference_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_training_flat_owner_bound_cases"] == [], summary
    assert summary["native_dispatch_training_flat_owner_default_off_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_training_dispatch_kernel_boundary_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_training_dispatch_kernel_evidence_present_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_training_dispatch_kernel_bound_cases"] == [], summary
    assert summary["native_dispatch_training_dispatch_kernel_default_off_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_training_runtime_executor_boundary_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_training_runtime_executor_bound_cases"] == [], summary
    assert summary["native_dispatch_training_runtime_executor_default_off_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_training_path_request_boundary_ready_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_explicit_training_path_requested_cases"] == [], summary
    assert summary["native_dispatch_training_path_default_off_cases"] == ["native_update_dispatch"], summary
    missing = summary["native_dispatch_missing_for_training_promotion_by_case"]["native_update_dispatch"]
    assert "stream_event_chain_validation_missing" not in missing, summary
    assert "stream_lifetime_ownership_default_off" in missing, summary
    assert "stream_lifetime_ownership_not_promoted" not in missing, summary
    assert "direct_gradient_write_default_off" in missing, summary
    assert "direct_gradient_write_not_native_supported" not in missing, summary
    assert "native_training_flat_owner_default_off" in missing, summary
    assert "native_training_flat_owner_not_promoted" not in missing, summary
    assert "native_training_dispatch_kernel_default_off" in missing, summary
    assert "native_training_dispatch_kernel_not_promoted" not in missing, summary
    assert "native_dispatch_training_runtime_executor_missing" not in missing, summary
    assert "native_dispatch_training_runtime_executor_default_off" in missing, summary
    assert "native_dispatch_training_path_default_off" in missing, summary
    assert summary["native_dispatch_execution_blockers_by_case"]["native_update_dispatch"] == [
        "native_dispatch_runtime_default_off"
    ], summary
    assert summary["native_dispatch_training_executor_ready_cases"] == [], summary
    assert summary["native_dispatch_diagnostic_executor_ready_cases"] == [], summary
    assert summary["native_dispatch_diagnostic_executor_blockers_by_case"]["native_update_dispatch"] == [
        "native_dispatch_diagnostic_clone_context_disabled"
    ], summary
    assert summary["native_dispatch_executor_probe_blockers_by_case"]["native_update_dispatch"] == [
        "native_dispatch_diagnostic_executor_call_disabled"
    ], summary
    assert summary["native_dispatch_diagnostic_replay_cases"] == ["native_update_dispatch"], summary
    assert summary["native_dispatch_diagnostic_replay_ok_cases"] == ["native_update_dispatch"], summary
    assert "native_update_dispatch" not in summary["native_dispatch_diagnostic_replay_blockers_by_case"], summary
    assert "native_update_dispatch" in summary["performance_gate_blockers_by_case"], summary


def test_matrix_summary_surfaces_owner_native_status() -> None:
    payload = {
        "cases": [
            {
                "case": {"name": "owner_native_launch_small"},
                "returncode": 0,
                "summary": {
                    "mean_step_ms": 1000.0,
                    "peak_vram_mb": 4096.0,
                    "owner_native_launch_probe_present": True,
                    "owner_native_launch_attempted": True,
                    "owner_native_launch_ok": True,
                    "owner_native_kernel_executed": True,
                    "owner_native_parity_ok": True,
                    "owner_native_numel": 2171392,
                    "performance_gate_blocked_reasons": ["native_dispatch_benchmark_case_missing"],
                },
            }
        ]
    }
    report = _build_matrix_performance_report(payload)
    owner = report["performance_gate"]["evidence"]["owner_native_kernel"]
    assert owner["present"] is True, report
    assert owner["kernel_executed"] is True, report
    assert owner["parity_ok"] is True, report
    assert owner["owner_numel"] == 2171392, report


def test_matrix_performance_report_requires_executed_native_case() -> None:
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
                "case": {"name": "native_update_dispatch"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 24,
                    "mean_step_ms": 900.0,
                    "native_dispatch_executed": False,
                    "owner_native_launch_probe_present": True,
                    "owner_native_launch_attempted": True,
                    "owner_native_launch_ok": True,
                    "owner_native_kernel_executed": True,
                    "owner_native_parity_ok": True,
                    "owner_native_numel": 4096,
                },
            },
        ],
        "summary": {
            "executed_count": 2,
            "all_success": True,
            "mean_step_ms_by_case": {"baseline_phase": 1000.0, "native_update_dispatch": 900.0},
        },
    }
    report = _build_matrix_performance_report(payload)
    reasons = set(report["blocked_reasons"])
    training_matrix = report["performance_gate"]["evidence"]["training_matrix"]
    owner = report["performance_gate"]["evidence"]["owner_native_kernel"]
    assert training_matrix["native_case"] == "native_update_dispatch", report
    assert training_matrix["native_dispatch_executed"] is False, report
    assert owner["present"] is True, report
    assert owner["kernel_executed"] is True, report
    assert "native_dispatch_benchmark_case_missing" not in reasons, report
    assert "owner_backed_native_kernel_evidence_missing" not in reasons, report
    assert "native_dispatch_not_executed_in_benchmark_case" in reasons, report


def test_matrix_performance_report_uses_best_owner_native_probe() -> None:
    payload = {
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "cases": [
            {
                "case": {"name": "native_update_dispatch"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 24,
                    "mean_step_ms": 900.0,
                    "native_dispatch_executed": False,
                    "owner_native_launch_probe_present": True,
                    "owner_native_launch_attempted": False,
                    "owner_native_launch_ok": False,
                    "owner_native_kernel_executed": False,
                    "owner_native_parity_ok": False,
                    "owner_native_numel": 4096,
                },
            },
            {
                "case": {"name": "owner_native_launch_small"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 24,
                    "mean_step_ms": 950.0,
                    "owner_native_launch_probe_present": True,
                    "owner_native_launch_attempted": True,
                    "owner_native_launch_ok": True,
                    "owner_native_kernel_executed": True,
                    "owner_native_parity_ok": True,
                    "owner_native_numel": 4096,
                },
            },
        ],
        "summary": {
            "executed_count": 2,
            "all_success": True,
            "mean_step_ms_by_case": {"baseline_phase": 1000.0, "native_update_dispatch": 900.0},
        },
    }
    report = _build_matrix_performance_report(payload)
    owner = report["performance_gate"]["evidence"]["owner_native_kernel"]
    reasons = set(report["blocked_reasons"])
    assert owner["present"] is True, report
    assert owner["kernel_executed"] is True, report
    assert owner["parity_ok"] is True, report
    assert "owner_backed_native_kernel_not_executed" not in reasons, report


def test_matrix_performance_report_prefers_clean_perf_case() -> None:
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
                "case": {"name": "native_update_dispatch"},
                "returncode": 0,
                "summary": {"success": True, "steps_completed": 24, "mean_step_ms": 3000.0, "native_dispatch_executed": True},
            },
            {
                "case": {"name": "native_update_dispatch_perf", "performance_sample": True},
                "returncode": 0,
                "summary": {"success": True, "steps_completed": 24, "mean_step_ms": 900.0, "native_dispatch_executed": True},
            },
        ],
        "summary": {
            "executed_count": 3,
            "all_success": True,
            "mean_step_ms_by_case": {
                "baseline_phase": 1000.0,
                "native_update_dispatch": 3000.0,
                "native_update_dispatch_perf": 900.0,
            },
        },
        "optimizer_performance_artifact": {"optimizer_performance_gate": _optimizer_gate(speedup=1.25)},
    }
    report = _build_matrix_performance_report(payload)
    training_matrix = report["performance_gate"]["evidence"]["training_matrix"]
    assert training_matrix["native_case"] == "native_update_dispatch_perf", report
    assert training_matrix["native_mean_step_ms"] == 900.0, report
    assert "end_to_end_speedup_below_threshold" not in report["blocked_reasons"], report


def test_matrix_performance_report_prefers_promotion_perf_case() -> None:
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
                "case": {"name": "native_update_dispatch_perf", "performance_sample": True},
                "returncode": 0,
                "summary": {"success": True, "steps_completed": 24, "mean_step_ms": 1200.0, "native_dispatch_executed": True},
            },
            {
                "case": {"name": "native_update_dispatch_promotion_perf", "performance_sample": True},
                "returncode": 0,
                "summary": {"success": True, "steps_completed": 24, "mean_step_ms": 900.0, "native_dispatch_executed": True},
            },
            {
                "case": {"name": "owner_native_launch_small"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 24,
                    "mean_step_ms": 950.0,
                    "owner_native_launch_probe_present": True,
                    "owner_native_launch_attempted": True,
                    "owner_native_launch_ok": True,
                    "owner_native_kernel_executed": True,
                    "owner_native_parity_ok": True,
                    "owner_native_numel": 4096,
                },
            },
        ],
        "summary": {
            "executed_count": 4,
            "all_success": True,
            "mean_step_ms_by_case": {
                "baseline_phase": 1000.0,
                "native_update_dispatch_perf": 1200.0,
                "native_update_dispatch_promotion_perf": 900.0,
                "owner_native_launch_small": 950.0,
            },
        },
        "optimizer_performance_artifact": {"optimizer_performance_gate": _optimizer_gate(speedup=1.25)},
    }
    report = _build_matrix_performance_report(payload)
    training_matrix = report["performance_gate"]["evidence"]["training_matrix"]
    assert training_matrix["native_case"] == "native_update_dispatch_promotion_perf", report
    assert training_matrix["native_mean_step_ms"] == 900.0, report
    assert "end_to_end_speedup_below_threshold" not in report["blocked_reasons"], report


def test_matrix_performance_report_accepts_optimizer_artifact() -> None:
    payload = _representative_matrix_payload()
    payload["optimizer_performance_artifact"] = {"optimizer_performance_gate": _optimizer_gate(speedup=1.25)}
    report = _build_matrix_performance_report(payload)
    reasons = set(report["blocked_reasons"])
    optimizer = report["performance_gate"]["evidence"]["optimizer_microbenchmark"]
    assert optimizer["present"] is True, report
    assert optimizer["best_speedup_vs_baseline"] == 1.25, report
    assert "optimizer_microbenchmark_missing" not in reasons, report
    assert report["blocked_reasons"] == [], report


def test_matrix_performance_report_keeps_short_optimizer_artifact_blocked() -> None:
    payload = _representative_matrix_payload()
    payload["optimizer_performance_artifact"] = {
        "optimizer_performance_gate": _optimizer_gate(speedup=1.15, quality="short_benchmark", promotion_ok=False)
    }
    report = _build_matrix_performance_report(payload)
    reasons = set(report["blocked_reasons"])
    assert "optimizer_microbenchmark_missing" not in reasons, report
    assert "optimizer_microbenchmark_promotion_gate_not_ok" in reasons, report
    assert "optimizer_microbenchmark_not_promotion_grade" in reasons, report
    assert "optimizer_microbenchmark_speedup_below_promotion" in reasons, report


def test_matrix_dry_run_loads_optimizer_artifact_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = Path(tmp) / "optimizer_report.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "benchmark": "turbocore_triton_adamw_flat_v0",
                    "ok": True,
                    "iters": 20,
                    "warmup": 5,
                    "performance_gate": _optimizer_gate(speedup=1.3),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            run=False,
            write_dry_run=False,
            keep_going=False,
            family="anima",
            cases=["baseline_phase"],
            profiles=["standard"],
            steps=2,
            steady_warmup=0,
            samples=1,
            resolution=64,
            network_dim=1,
            train_batch_size=1,
            source_data="",
            python="python",
            optimizer_performance_report=str(artifact_path),
            out=str(Path(tmp) / "matrix"),
        )
        payload = build_matrix_payload(args, run=False)
    artifact = payload["optimizer_performance_artifact"]
    summary = payload["summary"]["optimizer_performance_artifact"]
    gate_summary = payload["summary"]["native_update_performance_gate"]
    assert artifact["gate_present"] is True, payload
    assert summary["gate_ok"] is True, payload
    assert summary["best_speedup_vs_baseline"] == 1.3, payload
    assert gate_summary["optimizer_evidence_present"] is True, payload
    assert gate_summary["optimizer_evidence_quality"] == "promotion_benchmark", payload


def test_profile_summary_can_inherit_optimizer_artifact_gate() -> None:
    summary = {
        "native_update_performance_report": {
            "benchmark_matrix": {
                "matrix": "turbocore_update_benchmark_matrix_v0",
                "run": True,
                "cases": [],
                "summary": {"executed_count": 0},
            },
            "performance_gate": {
                "blocked_reasons": ["optimizer_microbenchmark_missing"],
            },
        },
        "runs": {
            "standard": {
                "success": True,
                "steps_completed": 1,
                "mean_step_ms": 1000.0,
                "steady_mean_step_ms": 1000.0,
                "peak_vram_mb": 4096.0,
                "native_update_readiness": {"blocked_reasons": []},
                "update_shadow_reports": [
                    {
                        "after_optimizer": {"compared": True, "parity_ok_loose": True},
                        "owner_native_launch_probe": {
                            "ok": True,
                            "attempted": True,
                            "kernel_executed": True,
                            "parity_ok": True,
                            "persistent_owner_mutated": False,
                            "owner_numel": 4096,
                        },
                    }
                ],
            }
        },
    }
    compact = _summarize_benchmark_summary(
        summary,
        optimizer_performance_gate=_optimizer_gate(speedup=1.3),
    )
    reasons = set(compact["performance_gate_blocked_reasons"])
    assert "optimizer_microbenchmark_missing" not in reasons, compact


def test_profile_summary_keeps_best_shadow_evidence_after_autostop() -> None:
    summary = {
        "native_update_performance_report": {"performance_gate": {"blocked_reasons": []}},
        "runs": {
            "standard": {
                "success": True,
                "steps_completed": 4,
                "mean_step_ms": 1000.0,
                "steady_mean_step_ms": 900.0,
                "peak_vram_mb": 4096.0,
                "native_update_readiness": {"blocked_reasons": []},
                "update_shadow_reports": [
                    {
                        "after_optimizer": {
                            "compared": True,
                            "sampled": True,
                            "sample_parameter_tensors": 1,
                            "total_parameter_tensors": 8,
                            "parity_ok_loose": True,
                            "max_abs_param_diff": 0.0,
                            "auto_stopped_after_this_step": True,
                        },
                        "copyback_probe": {"scratch_copyback_validated": True, "real_parameters_mutated": False},
                        "native_binding_probe": {
                            "request_shape_ready": True,
                            "tensor_object_binding_ready": True,
                            "launch_plan_ready": True,
                            "event_chain_verified": True,
                        },
                        "owner_native_launch_probe": {
                            "ok": True,
                            "attempted": True,
                            "kernel_executed": True,
                            "parity_ok": True,
                            "persistent_owner_mutated": False,
                            "owner_numel": 4096,
                        },
                    },
                    {
                        "reason": "auto_stopped_after_consecutive_passes",
                        "after_optimizer": {
                            "compared": False,
                            "skipped": True,
                            "reason": "auto_stopped_after_consecutive_passes",
                        },
                    },
                ],
            }
        },
    }
    compact = _summarize_benchmark_summary(
        summary,
        optimizer_performance_gate=_optimizer_gate(speedup=1.3),
    )
    assert compact["shadow_auto_stopped"] is True, compact
    assert compact["shadow_sampled"] is True, compact
    assert compact["owner_native_launch_probe_present"] is True, compact
    assert compact["owner_native_kernel_executed"] is True, compact
    assert compact["owner_native_parity_ok"] is True, compact
    assert compact["copyback_scratch_validated"] is True, compact
    assert compact["native_binding_launch_plan_ready"] is True, compact


def test_profile_summary_surfaces_native_training_timing() -> None:
    summary = {
        "native_update_performance_report": {"performance_gate": {"blocked_reasons": []}},
        "runs": {
            "standard": {
                "success": True,
                "steps_completed": 2,
                "mean_step_ms": 1000.0,
                "steady_mean_step_ms": 900.0,
                "peak_vram_mb": 4096.0,
                "native_update_readiness": {"blocked_reasons": []},
                "native_update_dispatch_runtime_reports": [
                    {
                        "training_executor": {
                            "result": {
                                "native_step_executed": True,
                                "timing": {
                                    "elapsed_ms": 12.0,
                                    "state_sync_ms": 1.0,
                                    "param_sync_ms": 2.0,
                                    "executor_step_ms": 7.0,
                                    "optimizer_state_sync_ms": 2.0,
                                },
                                "update_report": {
                                    "owner_backend": "rust_cuda_adamw_v0",
                                    "used_direct_grad": False,
                                    "native_kernel_present": True,
                                    "timing": {
                                        "elapsed_ms": 7.0,
                                        "grad_sync_ms": 1.5,
                                        "owner_step_ms": 3.5,
                                        "copyback_ms": 1.0,
                                        "zero_grad_ms": 0.5,
                                    },
                                    "owner_step": {
                                        "native_report": {
                                            "runtime_synchronization": "cuCtxSynchronize_after_native_step",
                                            "runtime_launch_stream_binding": "cuda_driver_default_stream_null_synchronized",
                                            "stream_lifetime_bound": True,
                                            "stream_synchronization_bound": True,
                                        }
                                    },
                                },
                            }
                        }
                    }
                ],
                "native_update_loop_timings": [
                    {
                        "executor_get_ms": 0.25,
                        "dispatch_runtime_prepare_ms": 12.5,
                        "gate_update_ms": 0.4,
                    }
                ],
            }
        },
    }
    compact = _summarize_benchmark_summary(summary)
    assert compact["native_dispatch_training_executor_reports"] == 1, compact
    assert compact["native_dispatch_training_executor_executed_reports"] == 1, compact
    assert compact["native_dispatch_training_executor_timing_present"] is True, compact
    assert compact["native_dispatch_training_executor_executor_step_ms_mean"] == 7.0, compact
    assert compact["native_dispatch_update_executor_owner_step_ms_mean"] == 3.5, compact
    assert compact["native_dispatch_update_executor_copyback_ms_mean"] == 1.0, compact
    assert compact["native_dispatch_loop_dispatch_runtime_prepare_ms_mean"] == 12.5, compact
    assert compact["native_dispatch_update_executor_owner_backend"] == "rust_cuda_adamw_v0", compact
    assert compact["native_dispatch_owner_native_report_present"] is True, compact
    assert compact["native_dispatch_owner_native_runtime_synchronization"] == "cuCtxSynchronize_after_native_step", compact


def _representative_matrix_payload() -> dict:
    return {
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "cases": [
            {
                "case": {"name": "baseline_phase"},
                "returncode": 0,
                "summary": {"success": True, "steps_completed": 24, "mean_step_ms": 1000.0},
            },
            {
                "case": {"name": "native_update_dispatch"},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 24,
                    "mean_step_ms": 900.0,
                    "native_dispatch_executed": True,
                    "owner_native_launch_probe_present": True,
                    "owner_native_launch_attempted": True,
                    "owner_native_launch_ok": True,
                    "owner_native_kernel_executed": True,
                    "owner_native_parity_ok": True,
                    "owner_native_numel": 4096,
                },
            },
        ],
        "summary": {
            "executed_count": 2,
            "all_success": True,
            "mean_step_ms_by_case": {"baseline_phase": 1000.0, "native_update_dispatch": 900.0},
        },
    }


def _optimizer_gate(*, speedup: float, quality: str = "promotion_benchmark", promotion_ok: bool = True) -> dict:
    return {
        "gate": "turbocore_optimizer_performance_gate",
        "status": "promotion_candidate_needs_route_validation",
        "ok": True,
        "promotion_gate_ok": promotion_ok,
        "runtime_dispatch_allowed": False,
        "evidence_quality": quality,
        "best_candidate": {
            "optimizer": "triton_adamw_flat_v0",
            "speedup_vs_baseline": speedup,
            "promotion_gate_ok": promotion_ok,
        },
    }


def main() -> int:
    test_matrix_dry_run_builds_expected_commands()
    test_matrix_dry_run_builds_clean_perf_case()
    test_matrix_dry_run_builds_promotion_perf_case()
    test_matrix_summary_surfaces_native_dispatch_status()
    test_matrix_summary_surfaces_owner_native_status()
    test_matrix_performance_report_requires_executed_native_case()
    test_matrix_performance_report_uses_best_owner_native_probe()
    test_matrix_performance_report_prefers_clean_perf_case()
    test_matrix_performance_report_prefers_promotion_perf_case()
    test_matrix_performance_report_accepts_optimizer_artifact()
    test_matrix_performance_report_keeps_short_optimizer_artifact_blocked()
    test_matrix_dry_run_loads_optimizer_artifact_path()
    test_profile_summary_can_inherit_optimizer_artifact_gate()
    test_profile_summary_keeps_best_shadow_evidence_after_autostop()
    test_profile_summary_surfaces_native_training_timing()
    print("turbocore_update_benchmark_matrix_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
