"""Smoke checks for Automagic++ P33 real dispatch review gate."""

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
    build_automagicpp_real_dispatch_integration_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_automagicpp_real_dispatch_integration_review_scorecard()
    review = report["review_package"]
    summary = report["summary"]
    dispatch = review["dispatch_contract"]
    rollback = review["rollback_policy"]
    assert report["ok"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["review_gate_ready"] is True, report
    assert report["dispatch_integration_review"] is True, report
    assert report["manual_review_required"] is True, report
    assert report["canary_auto_blocked_until_review"] is True, report
    assert report["fallback_rollback_ready"] is True, report
    assert report["runtime_dispatch_not_enabled"] is True, report
    assert report["native_real_dispatch_enabled"] is False, report
    assert report["default_behavior_unchanged"] is True, report
    assert report["p32_dependency"]["required_builder"] == P32_AUDIT_BUILDER, report
    assert report["p32_dependency"]["builder_name_recorded"] is True, report
    assert review["allowed_initial_modes"] == ["off", "observe"], review
    assert review["blocked_modes_until_review"] == ["canary", "auto"], review
    assert dispatch["fallback_update_authority"] == "python_automagicpp", dispatch
    assert dispatch["runtime_dispatch_enabled_by_this_gate"] is False, dispatch
    assert rollback["fallback_backend"] == "python_automagicpp", rollback
    assert rollback["fallback_authoritative"] is True, rollback
    assert summary["dispatch_integration_review"] is True, summary
    assert summary["manual_review_required"] is True, summary
    assert summary["canary_auto_blocked_until_review"] is True, summary
    assert summary["fallback_rollback_ready"] is True, summary
    assert summary["runtime_dispatch_not_enabled"] is True, summary
    assert summary["native_real_dispatch_enabled"] is False, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_automagicpp_real_dispatch_integration_review_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
