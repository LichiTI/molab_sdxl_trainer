"""Smoke checks for the Native Training Performance V3-P3 audit."""

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

from devtools.audit_native_training_performance_v3_p3 import (  # noqa: E402
    build_v3_p3_exact_adamw_runtime_recovery_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_v3_p3_exact_adamw_runtime_recovery_audit()
    gates = report["progress_gates"]
    assert report["audit"] == "native_training_performance_v3_p3_audit_v0", report
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert gates["p2_short_matrix_complete"] is True, gates
    assert gates["runtime_recovery_hardened"] is True, gates
    assert gates["runtime_error_latches"] is True, gates
    assert gates["state_mismatch_latches"] is True, gates
    assert gates["resume_mismatch_latches"] is True, gates
    assert gates["optimizer_state_sync_failure_latches"] is True, gates
    assert gates["shadow_autostop_skip_not_latched"] is True, gates
    assert gates["clean_policy_not_latched"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v3_p3_audit_smoke",
        "ok": True,
        "milestone_completed": report["milestone_completed"],
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
