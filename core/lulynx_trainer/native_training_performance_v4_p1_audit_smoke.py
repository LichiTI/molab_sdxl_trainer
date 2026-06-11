"""Smoke checks for the Native Training Performance V4-P1 audit."""

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

from devtools.audit_native_training_performance_v4_p1 import (  # noqa: E402
    build_v4_p1_representative_benchmark_result_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_v4_p1_representative_benchmark_result_audit()
    gates = report["progress_gates"]
    assert report["audit"] == "native_training_performance_v4_p1_audit_v0", report
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["real_benchmark_result_ready"] is False, report
    assert gates["p0_manifest_complete"] is True, gates
    assert gates["result_ingestion_accepts_promotion_fixture"] is True, gates
    assert gates["result_ingestion_blocks_dry_run_manifest"] is True, gates
    assert gates["result_ingestion_surfaces_missing_or_current_input"] is True, gates
    assert gates["result_adapter_does_not_enable_rollout"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    current = report["sections"]["current_real_result_status"]
    assert current["ok"] is False, current
    assert "v4_p1_result_input_missing" in current["blocked_reasons"], current
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v4_p1_audit_smoke",
        "ok": True,
        "milestone_completed": report["milestone_completed"],
        "real_benchmark_result_ready": report["real_benchmark_result_ready"],
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
