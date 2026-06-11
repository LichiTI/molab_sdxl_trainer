"""Smoke checks for the Native Training Performance V4-P0 audit."""

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

from devtools.audit_native_training_performance_v4_p0 import (  # noqa: E402
    build_v4_p0_representative_benchmark_manifest_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_v4_p0_representative_benchmark_manifest_audit()
    gates = report["progress_gates"]
    assert report["audit"] == "native_training_performance_v4_p0_audit_v0", report
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert gates["v3_roadmap_complete"] is True, gates
    assert gates["benchmark_manifest_ready"] is True, gates
    assert gates["baseline_case_present"] is True, gates
    assert gates["native_perf_case_present"] is True, gates
    assert gates["representative_steps_configured"] is True, gates
    assert gates["dry_run_default"] is True, gates
    assert gates["performance_gate_still_blocked_until_run"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v4_p0_audit_smoke",
        "ok": True,
        "milestone_completed": report["milestone_completed"],
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
