"""Smoke checks for Native Training Performance V2-P15 audit."""

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

from devtools.audit_native_training_performance_p15 import (  # noqa: E402
    build_p15_plugin_schedulefree_training_tensor_binding_canary_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_p15_plugin_schedulefree_training_tensor_binding_canary_audit()
    gates = report["progress_gates"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert gates["checkpoint_adapter_proof"] is True, gates
    assert gates["training_tensor_binding_canary"] is True, gates
    assert gates["binding_request_shape"] is True, gates
    assert gates["non_mutating_binding_probe"] is True, gates
    assert gates["e2e_no_regression"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p15_audit_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": report["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
