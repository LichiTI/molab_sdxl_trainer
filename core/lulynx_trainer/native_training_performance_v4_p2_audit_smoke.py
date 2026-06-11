"""Smoke checks for the Native Training Performance V4-P2 audit."""

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

from devtools.audit_native_training_performance_v4_p2 import (  # noqa: E402
    build_v4_p2_checkpoint_resume_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_v4_p2_checkpoint_resume_audit()
    gates = report["progress_gates"]
    assert report["audit"] == "native_training_performance_v4_p2_audit_v0", report
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["real_benchmark_result_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_training_path_enabled"] is False, report
    assert report["default_rollout_allowed"] is False, report
    assert report["auto_rollout_allowed"] is False, report
    assert all(gates.values()), gates
    assert report["remaining_blockers"] == [], report
    checkpoint = report["sections"]["checkpoint_resume_boundary"]
    live = checkpoint["live_probe"]
    assert live["restore_compatible"] is True, live
    assert live["mismatch_compatible"] is False, live
    assert live["training_path_stays_default_off"] is True, live
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v4_p2_audit_smoke",
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
