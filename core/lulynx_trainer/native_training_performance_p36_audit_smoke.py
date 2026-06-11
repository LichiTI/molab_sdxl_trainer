"""Smoke checks for Native Training Performance V2-P36 audit."""

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

from devtools.audit_native_training_performance_p36 import (  # noqa: E402
    build_p36_adafactor_runtime_adapter_shadow_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_p36_adafactor_runtime_adapter_shadow_audit(quick=True)
    gates = report["progress_gates"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["report_only"] is True, report
    assert report["native_call_performed_by_p36"] is False, report
    assert report["p35_audit_builder"] == "build_p35_adafactor_training_tensor_binding_audit", report
    assert gates["p35_training_tensor_binding_complete"] is True, gates
    assert gates["runtime_dispatch_adapter_shadow"] is True, gates
    assert gates["fallback_backend_authoritative"] is True, gates
    assert gates["native_shadow_call_disabled"] is True, gates
    assert gates["runtime_dispatch_not_enabled"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p36_audit_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
