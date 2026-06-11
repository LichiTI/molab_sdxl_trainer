"""Smoke checks for V4 representative benchmark manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_v4_representative_benchmark_manifest_scorecard import (  # noqa: E402
    build_v4_representative_benchmark_manifest_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_v4_representative_benchmark_manifest_scorecard()
    gates = report["progress_gates"]
    matrix = report["matrix_summary"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["benchmark_manifest_ready"] is True, report
    assert report["default_rollout_allowed"] is False, report
    assert report["auto_rollout_allowed"] is False, report
    assert gates["benchmark_matrix_manifest_built"] is True, gates
    assert gates["baseline_case_present"] is True, gates
    assert gates["native_perf_case_present"] is True, gates
    assert gates["representative_steps_configured"] is True, gates
    assert gates["product_benchmark_path_used"] is True, gates
    assert gates["dry_run_default"] is True, gates
    assert gates["performance_gate_still_blocked_until_run"] is True, gates
    assert matrix["executed_count"] == 0, matrix
    assert "representative_training_matrix_not_executed" in matrix["performance_gate_blocked_reasons"], matrix
    return {
        "schema_version": 1,
        "probe": "turbocore_v4_representative_benchmark_manifest_scorecard_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
