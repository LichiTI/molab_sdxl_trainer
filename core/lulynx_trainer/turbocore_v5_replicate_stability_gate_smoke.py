"""Smoke checks for V5 replicate stability gate."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_replicate_stability_gate import (  # noqa: E402
    build_v5_replicate_stability_gate,
)


def run_smoke() -> dict[str, Any]:
    accepted = build_v5_replicate_stability_gate(
        matrix_payloads=[
            _matrix_payload(speedup=1.08),
            _matrix_payload(speedup=1.11),
            _matrix_payload(speedup=1.07),
        ]
    )
    assert accepted["ok"] is True, accepted
    assert accepted["manual_wider_canary_allowed"] is True, accepted
    assert accepted["default_training_path_enabled"] is False, accepted
    assert accepted["default_rollout_allowed"] is False, accepted
    assert accepted["auto_rollout_allowed"] is False, accepted
    assert accepted["aggregate"]["min_speedup"] == 1.07, accepted
    assert accepted["ready_run_count"] == 3, accepted

    single = build_v5_replicate_stability_gate(matrix_payloads=[_matrix_payload(speedup=1.21)])
    assert single["ok"] is False, single
    assert "v5_p3_replicate_runs_too_few" in single["blocked_reasons"], single
    assert single["manual_wider_canary_allowed"] is False, single

    slow = build_v5_replicate_stability_gate(
        matrix_payloads=[
            _matrix_payload(speedup=1.08),
            _matrix_payload(speedup=1.01),
            _matrix_payload(speedup=1.09),
        ]
    )
    assert slow["ok"] is False, slow
    assert "v5_p3_end_to_end_speedup_below_threshold" in slow["blocked_reasons"], slow

    missing_timing = build_v5_replicate_stability_gate(
        matrix_payloads=[
            _matrix_payload(speedup=1.08),
            _matrix_payload(speedup=1.09, timing=False),
            _matrix_payload(speedup=1.07),
        ]
    )
    assert missing_timing["ok"] is False, missing_timing
    assert "v5_p3_native_timing_summary_missing" in missing_timing["blocked_reasons"], missing_timing

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for index, speedup in enumerate((1.08, 1.09, 1.07)):
            path = Path(tmp) / f"matrix_{index}.json"
            path.write_text(json.dumps(_matrix_payload(speedup=speedup), ensure_ascii=False), encoding="utf-8")
            paths.append(path)
        loaded = build_v5_replicate_stability_gate(matrix_summary_paths=paths)
    assert loaded["ok"] is True, loaded
    assert loaded["run_count"] == 3, loaded

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_replicate_stability_gate_smoke",
        "ok": True,
        "accepted_speedups": accepted["aggregate"]["speedup_samples"],
        "single_blocked_reasons": single["blocked_reasons"],
        "slow_blocked_reasons": slow["blocked_reasons"],
        "missing_timing_blocked_reasons": missing_timing["blocked_reasons"],
        "recommended_next_step": accepted["recommended_next_step"],
    }


def _matrix_payload(*, speedup: float, timing: bool = True) -> dict[str, Any]:
    baseline_ms = 1000.0
    native_ms = baseline_ms / float(speedup)
    native_summary = {
        "success": True,
        "steps_completed": 24,
        "mean_step_ms": native_ms,
        "steady_mean_step_ms": native_ms,
        "peak_vram_mb": 4096.0,
        "native_dispatch_executed": True,
        "native_dispatch_probe_cache_retained": True,
        "owner_native_launch_probe_present": True,
        "owner_native_launch_attempted": True,
        "owner_native_launch_ok": True,
        "owner_native_kernel_executed": True,
        "owner_native_parity_ok": True,
        "owner_native_numel": 4096,
    }
    if timing:
        native_summary.update(
            {
                "native_dispatch_training_executor_timing_present": True,
                "native_dispatch_update_report_present": True,
                "native_dispatch_owner_native_report_present": True,
                "native_dispatch_owner_native_runtime_synchronization": "cuCtxSynchronize_after_native_step",
            }
        )
    return {
        "schema_version": 1,
        "matrix": "turbocore_update_benchmark_matrix_v0",
        "run": True,
        "family": "anima",
        "cases": [
            {
                "case": {"name": "baseline_phase", "performance_sample": False},
                "returncode": 0,
                "summary": {
                    "success": True,
                    "steps_completed": 24,
                    "mean_step_ms": baseline_ms,
                    "steady_mean_step_ms": baseline_ms,
                    "peak_vram_mb": 4096.0,
                },
            },
            {
                "case": {"name": "native_update_dispatch_promotion_perf", "performance_sample": True},
                "returncode": 0,
                "summary": native_summary,
            },
        ],
        "summary": {
            "case_count": 2,
            "executed_count": 2,
            "all_success": True,
            "mean_step_ms_by_case": {
                "baseline_phase": baseline_ms,
                "native_update_dispatch_promotion_perf": native_ms,
            },
        },
        "optimizer_performance_artifact": {
            "optimizer_performance_gate": {
                "gate": "turbocore_optimizer_performance_gate",
                "ok": True,
                "promotion_gate_ok": True,
                "runtime_dispatch_allowed": False,
                "evidence_quality": "promotion_benchmark",
                "best_candidate": {
                    "optimizer": "turbocore_adamw_cuda_runtime_session",
                    "speedup_vs_baseline": 22.0,
                    "promotion_gate_ok": True,
                },
            }
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
