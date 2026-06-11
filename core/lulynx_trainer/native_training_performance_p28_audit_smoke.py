"""Smoke checks for Native Training Performance V2-P28 audit."""

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

from devtools.audit_native_training_performance_p28 import (  # noqa: E402
    P27_AUDIT_BUILDER,
    build_p28_anima_factored_adamw_e2e_shadow_matrix_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_p28_anima_factored_adamw_e2e_shadow_matrix_audit(quick=True)
    gates = report["progress_gates"]
    summary = report["summary"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["report_only"] is True, report
    assert report["native_call_performed_by_p28"] is False, report
    assert report["p27_audit_builder"] == P27_AUDIT_BUILDER, report
    assert gates["p27_training_loop_canary_dependency_named"] is True, gates
    assert gates["e2e_shadow_matrix_scaffold"] is True, gates
    assert gates["fallback_backend_authoritative"] is True, gates
    assert gates["native_shadow_training_does_not_mutate_authority"] is True, gates
    assert gates["runtime_dispatch_not_enabled"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    assert summary["fallback_backend"] == "python_anima_factored_adamw", summary
    assert summary["fallback_backend_authoritative"] is True, summary
    assert summary["native_shadow_training_mutates_authority"] is False, summary
    assert summary["runtime_dispatch_not_enabled"] is True, summary
    assert summary["default_behavior_unchanged"] is True, summary
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p28_audit_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": summary["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
