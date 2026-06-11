"""Smoke checks for Native Training Performance V2-P40 audit."""

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

from devtools.audit_native_training_performance_p40 import (  # noqa: E402
    build_p40_adafactor_real_dispatch_integration_review_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_p40_adafactor_real_dispatch_integration_review_audit()
    gates = report["progress_gates"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert gates["p39_explicit_canary_rollout_policy_dependency_named"] is True, gates
    assert gates["dispatch_integration_review"] is True, gates
    assert gates["manual_review_required"] is True, gates
    assert gates["canary_auto_blocked_until_review"] is True, gates
    assert gates["fallback_rollback_ready"] is True, gates
    assert gates["runtime_dispatch_not_enabled"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert gates["native_real_dispatch_enabled_false"] is True, gates
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p40_audit_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
