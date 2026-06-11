"""Smoke checks for the Native Training Performance V3 roadmap audit."""

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

from devtools.audit_native_training_performance_roadmap_v3 import (  # noqa: E402
    EXPECTED_MILESTONES,
    build_v3_roadmap_audit,
)


def run_smoke() -> dict[str, Any]:
    audit = build_v3_roadmap_audit(run_live_training=True, include_full_sections=False)
    gates = audit["progress_gates"]
    assert audit["audit"] == "native_training_performance_roadmap_v3_audit_v0", audit
    assert audit["ok"] is True, audit
    assert audit["roadmap_completed"] is True, audit
    assert set(gates) == EXPECTED_MILESTONES, gates
    assert all(gates.values()), gates
    assert audit["default_training_path_enabled"] is False, audit
    assert audit["default_rollout_allowed"] is False, audit
    assert audit["auto_rollout_allowed"] is False, audit
    assert audit["missing_milestones"] == [], audit
    assert audit["remaining_blockers"] == [], audit
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v3_audit_smoke",
        "ok": True,
        "roadmap_completed": audit["roadmap_completed"],
        "progress_gates": gates,
        "recommended_next_step": audit["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
