"""Smoke checks for the Native Training Performance V4 roadmap audit."""

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

from devtools.audit_native_training_performance_roadmap_v4 import (  # noqa: E402
    EXPECTED_MILESTONES,
    build_v4_roadmap_audit,
)


def run_smoke() -> dict[str, Any]:
    audit = build_v4_roadmap_audit(run_live_training=True, run_live_probe=True, include_full_sections=False)
    gates = audit["progress_gates"]
    assert audit["audit"] == "native_training_performance_roadmap_v4_audit_v0", audit
    assert audit["ok"] is True, audit
    assert audit["roadmap_completed"] is True, audit
    assert audit["real_benchmark_result_ready"] is False, audit
    assert audit["promotion_decision"] == "hold_for_representative_benchmark", audit
    assert audit["manual_wider_canary_allowed"] is False, audit
    assert set(gates) == EXPECTED_MILESTONES, gates
    assert all(gates.values()), gates
    assert audit["default_training_path_enabled"] is False, audit
    assert audit["default_rollout_allowed"] is False, audit
    assert audit["auto_rollout_allowed"] is False, audit
    assert audit["missing_milestones"] == [], audit
    assert audit["remaining_blockers"] == [], audit
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v4_audit_smoke",
        "ok": True,
        "roadmap_completed": audit["roadmap_completed"],
        "real_benchmark_result_ready": audit["real_benchmark_result_ready"],
        "promotion_decision": audit["promotion_decision"],
        "progress_gates": gates,
        "recommended_next_step": audit["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
