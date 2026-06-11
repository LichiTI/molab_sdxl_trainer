"""Smoke checks for Native Training Performance V2-P29 audit."""

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

from devtools.audit_native_training_performance_p29 import (  # noqa: E402
    P28_AUDIT_BUILDER,
    build_p29_anima_factored_adamw_explicit_canary_rollout_policy_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_p29_anima_factored_adamw_explicit_canary_rollout_policy_audit(quick=True)
    gates = report["progress_gates"]
    summary = report["summary"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["report_only"] is True, report
    assert report["native_call_performed_by_p29"] is False, report
    assert report["p28_audit_builder"] == P28_AUDIT_BUILDER, report
    assert report["canary_auto_enabled"] is False, report
    assert report["manual_review_required"] is True, report
    assert report["fallback_rollback_ready"] is True, report
    assert report["runtime_dispatch_not_enabled"] is True, report
    assert report["default_behavior_unchanged"] is True, report
    assert gates["p28_e2e_shadow_matrix_dependency_named"] is True, gates
    assert gates["explicit_canary_policy"] is True, gates
    assert gates["canary_auto_enabled_false"] is True, gates
    assert gates["manual_review_required"] is True, gates
    assert gates["fallback_rollback_ready"] is True, gates
    assert gates["runtime_dispatch_not_enabled"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert report["remaining_blockers"] == [], report
    assert report["manual_review_blockers"], report
    assert report["rollback_blockers"], report
    assert summary["canary_auto_enabled"] is False, summary
    assert summary["manual_review_required"] is True, summary
    assert summary["fallback_rollback_ready"] is True, summary
    assert summary["runtime_dispatch_not_enabled"] is True, summary
    assert summary["default_behavior_unchanged"] is True, summary
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p29_audit_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": summary["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
