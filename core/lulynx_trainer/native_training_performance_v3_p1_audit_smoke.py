"""Smoke checks for the Native Training Performance V3-P1 audit."""

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

from devtools.audit_native_training_performance_v3_p1 import (  # noqa: E402
    build_v3_p1_exact_adamw_rollout_manifest_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_v3_p1_exact_adamw_rollout_manifest_audit(native_training_mode="canary")
    gates = report["progress_gates"]
    assert report["audit"] == "native_training_performance_v3_p1_audit_v0", report
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert gates["p0_real_canary_complete"] is True, gates
    assert gates["rollout_manifest_ready"] is True, gates
    assert gates["explicit_canary_only"] is True, gates
    assert gates["default_off"] is True, gates
    assert gates["fallback_rollback_ready"] is True, gates
    assert gates["route_request_fields_present"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v3_p1_audit_smoke",
        "ok": True,
        "milestone_completed": report["milestone_completed"],
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
