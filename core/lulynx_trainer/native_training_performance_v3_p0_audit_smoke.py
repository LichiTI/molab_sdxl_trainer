"""Smoke checks for the Native Training Performance V3-P0 audit."""

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

from devtools.audit_native_training_performance_v3_p0 import (  # noqa: E402
    build_v3_p0_exact_adamw_real_canary_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_v3_p0_exact_adamw_real_canary_audit(run_live_training=True)
    gates = report["progress_gates"]
    assert report["audit"] == "native_training_performance_v3_p0_audit_v0", report
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert gates["default_off"] is True, gates
    assert gates["explicit_opt_in_request"] is True, gates
    assert gates["live_training_native_step"] is True, gates
    assert gates["fallback_preserved"] is True, gates
    assert gates["state_sync_ready"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["default_training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v3_p0_audit_smoke",
        "ok": True,
        "milestone_completed": report["milestone_completed"],
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
