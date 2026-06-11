"""Smoke checks for Native Training Performance V2-P38 audit."""

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

from devtools.audit_native_training_performance_p38 import build_p38_adafactor_e2e_shadow_matrix_audit  # noqa: E402


def run_smoke() -> dict[str, Any]:
    report = build_p38_adafactor_e2e_shadow_matrix_audit()
    gates = report["progress_gates"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert gates["p37_training_loop_canary_dependency_named"] is True, gates
    assert gates["e2e_shadow_matrix_scaffold"] is True, gates
    assert gates["fallback_backend_authoritative"] is True, gates
    assert gates["native_shadow_training_does_not_mutate_authority"] is True, gates
    assert gates["runtime_dispatch_not_enabled"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p38_audit_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
