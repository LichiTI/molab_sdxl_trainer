"""V4 representative benchmark manifest for exact AdamW native canary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

from core.lulynx_trainer.turbocore_update_benchmark_matrix import build_matrix_payload


BENCHMARK_CASES = ("baseline_phase", "native_update_dispatch_perf")
MIN_REPRESENTATIVE_STEPS = 20


def build_v4_representative_benchmark_manifest_scorecard(
    *,
    v3_audit: Mapping[str, Any] | None = None,
    family: str = "anima",
    steps: int = MIN_REPRESENTATIVE_STEPS,
    run_benchmark: bool = False,
) -> dict[str, Any]:
    """Build the representative benchmark manifest without enabling defaults."""

    args = _matrix_args(family=family, steps=steps)
    matrix = build_matrix_payload(args, run=bool(run_benchmark))
    summary = matrix.get("summary") if isinstance(matrix.get("summary"), Mapping) else {}
    perf_gate = summary.get("native_update_performance_gate") if isinstance(summary.get("native_update_performance_gate"), Mapping) else {}
    cases = matrix.get("cases") if isinstance(matrix.get("cases"), list) else []
    by_name = {
        str(case.get("case", {}).get("name", "")): case
        for case in cases
        if isinstance(case, Mapping)
    }
    baseline = by_name.get("baseline_phase", {})
    native = by_name.get("native_update_dispatch_perf", {})
    v3_ready = bool(v3_audit.get("roadmap_completed", False)) if isinstance(v3_audit, Mapping) else True
    progress_gates = {
        "v3_roadmap_complete": v3_ready,
        "benchmark_matrix_manifest_built": matrix.get("matrix") == "turbocore_update_benchmark_matrix_v0",
        "baseline_case_present": bool(baseline),
        "native_perf_case_present": bool(native)
        and bool(native.get("case", {}).get("performance_sample", False)),
        "representative_steps_configured": int(matrix.get("cases", [{}])[0].get("command", []).count("--steps")) >= 1
        and int(steps) >= MIN_REPRESENTATIVE_STEPS,
        "product_benchmark_path_used": all(
            any(str(part).endswith("native_runtime_profile_benchmark.py") for part in case.get("command", []))
            for case in (baseline, native)
            if isinstance(case, Mapping)
        ),
        "dry_run_default": matrix.get("run") is False and int(summary.get("executed_count", 0) or 0) == 0,
        "performance_gate_still_blocked_until_run": "representative_training_matrix_not_executed"
        in list(perf_gate.get("blocked_reasons", []) or []),
        "default_behavior_unchanged": True,
    }
    ready = all(progress_gates.values())
    blockers = [f"v4_p0_{name}_missing" for name, ok in progress_gates.items() if not ok]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v4_representative_benchmark_manifest_scorecard_v0",
        "gate": "v4_representative_benchmark_manifest",
        "ok": ready,
        "milestone_completed": ready,
        "benchmark_manifest_ready": ready,
        "family": str(family),
        "benchmark_cases": list(BENCHMARK_CASES),
        "representative_steps": int(steps),
        "run_benchmark": bool(run_benchmark),
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "matrix_summary": {
            "run": bool(matrix.get("run", False)),
            "case_count": int(summary.get("case_count", 0) or 0),
            "executed_count": int(summary.get("executed_count", 0) or 0),
            "all_success": summary.get("all_success"),
            "performance_gate_blocked_reasons": list(perf_gate.get("blocked_reasons", []) or []),
        },
        "case_commands": {
            name: {
                "summary_path": str(case.get("summary_path", "") or ""),
                "command_text": str(case.get("command_text", "") or ""),
            }
            for name, case in by_name.items()
            if name in BENCHMARK_CASES
        },
        "progress_gates": progress_gates,
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run V4-P1 representative benchmark matrix with --run when time budget allows"
            if ready
            else "complete V4-P0 representative benchmark manifest blockers"
        ),
        "notes": [
            "This gate builds the product benchmark command manifest only; it does not run the long benchmark by default.",
            "The native performance gate must remain blocked until the representative matrix is actually executed.",
            "Default and auto rollout stay disabled regardless of this manifest.",
        ],
    }


def _matrix_args(*, family: str, steps: int) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    return argparse.Namespace(
        run=False,
        write_dry_run=False,
        keep_going=False,
        family=str(family),
        cases=list(BENCHMARK_CASES),
        profiles=["standard"],
        steps=max(int(steps), MIN_REPRESENTATIVE_STEPS),
        steady_warmup=4,
        samples=2,
        resolution=64,
        network_dim=1,
        train_batch_size=1,
        source_data="",
        python=str(Path(sys.executable)),
        optimizer_performance_report="",
        out=str(repo / "temp" / "turbocore_v4_representative_benchmark_manifest"),
    )


__all__ = ["build_v4_representative_benchmark_manifest_scorecard"]
