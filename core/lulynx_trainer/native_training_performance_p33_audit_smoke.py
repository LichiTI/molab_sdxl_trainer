"""Smoke checks for Native Training Performance V2-P33 audit."""

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

from core.turbocore_automagicpp_real_dispatch_integration_review_scorecard import (  # noqa: E402
    P32_AUDIT_BUILDER,
)
from devtools.audit_native_training_performance_p33 import (  # noqa: E402
    build_p33_automagicpp_real_dispatch_integration_review_audit,
)


def run_smoke() -> dict[str, Any]:
    report = build_p33_automagicpp_real_dispatch_integration_review_audit(quick=True)
    gates = report["progress_gates"]
    summary = report["summary"]
    assert report["ok"] is True, report
    assert report["milestone_completed"] is True, report
    assert report["report_only"] is True, report
    assert report["native_call_performed_by_p33"] is False, report
    assert report["p32_audit_builder"] == P32_AUDIT_BUILDER, report
    assert report["dispatch_integration_review"] is True, report
    assert report["manual_review_required"] is True, report
    assert report["canary_auto_blocked_until_review"] is True, report
    assert report["fallback_rollback_ready"] is True, report
    assert report["runtime_dispatch_not_enabled"] is True, report
    assert report["default_behavior_unchanged"] is True, report
    assert report["native_real_dispatch_enabled"] is False, report
    assert gates["p32_explicit_canary_rollout_policy_dependency_named"] is True, gates
    assert gates["dispatch_integration_review"] is True, gates
    assert gates["manual_review_required"] is True, gates
    assert gates["canary_auto_blocked_until_review"] is True, gates
    assert gates["fallback_rollback_ready"] is True, gates
    assert gates["runtime_dispatch_not_enabled"] is True, gates
    assert gates["default_behavior_unchanged"] is True, gates
    assert gates["native_real_dispatch_enabled_false"] is True, gates
    assert report["remaining_blockers"] == [], report
    assert summary["dispatch_integration_review"] is True, summary
    assert summary["manual_review_required"] is True, summary
    assert summary["canary_auto_blocked_until_review"] is True, summary
    assert summary["fallback_rollback_ready"] is True, summary
    assert summary["runtime_dispatch_not_enabled"] is True, summary
    assert summary["default_behavior_unchanged"] is True, summary
    assert summary["native_real_dispatch_enabled"] is False, summary
    return {
        "schema_version": 1,
        "probe": "native_training_performance_p33_audit_smoke",
        "ok": True,
        "progress_gates": gates,
        "recommended_next_step": summary["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
