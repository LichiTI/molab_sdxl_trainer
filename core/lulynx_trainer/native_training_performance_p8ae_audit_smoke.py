"""Smoke checks for Native Training Performance V2-P8AE audit."""

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

from devtools.audit_native_training_performance_p8ae import (  # noqa: E402
    build_p8ae_paged_adamw8bit_dispatch_review_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_p8ae_paged_adamw8bit_dispatch_review_audit(quick=True)
    gates = report["progress_gates"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert gates["p8r_rollout_policy_complete"] is True, gates
    assert gates["dispatch_integration_review"] is True, gates
    assert gates["manual_review_required"] is True, gates
    assert gates["canary_auto_blocked_until_review"] is True, gates
    assert gates["fallback_rollback_ready"] is True, gates
    assert gates["training_loop_native_canary"] is True, gates
    assert gates["runtime_dispatch_not_enabled"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p8ae_audit_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
