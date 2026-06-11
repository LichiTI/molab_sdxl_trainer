"""Smoke checks for Native Training Performance V2-P41 audit."""

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

from devtools.audit_native_training_performance_p41 import (  # noqa: E402
    build_p41_plugin_adamlike_selected_optimizer_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_p41_plugin_adamlike_selected_optimizer_audit()
    gates = report["progress_gates"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert gates["p11_plugin_selector_complete"] is True, gates
    assert gates["selected_optimizer_abi_ready"] is True, gates
    assert gates["adamw_native_route_candidate_ready"] is True, gates
    assert gates["dedicated_kernel_queue_ready"] is True, gates
    assert gates["runtime_dispatch_not_enabled"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p41_audit_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
