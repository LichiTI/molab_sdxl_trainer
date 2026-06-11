"""Smoke checks for Muon dispatch integration review scorecard."""

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

from core.turbocore_muon_canary_rollout_policy_scorecard import (  # noqa: E402
    build_muon_canary_rollout_policy_scorecard,
)
from core.turbocore_muon_dispatch_integration_review_scorecard import (  # noqa: E402
    TARGET_OPTIMIZER,
    build_muon_dispatch_integration_review_scorecard,
)


def run_smoke() -> dict[str, Any]:
    rollout = build_muon_canary_rollout_policy_scorecard(write_artifact=True)
    report = build_muon_dispatch_integration_review_scorecard(
        rollout_policy_report=rollout,
        write_artifact=True,
    )
    review = report["review_package"]
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_muon_dispatch_integration_review_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["review_gate_ready"] is True, report
    assert report["dispatch_integration_review"] is True, report
    assert report["manual_review_required"] is True, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert review["optimizer_types"] == [TARGET_OPTIMIZER], review
    assert review["allowed_initial_modes"] == ["off", "observe"], review
    assert review["blocked_modes_until_review"] == ["canary", "auto"], review
    assert review["dispatch_contract"]["runtime_dispatch_enabled_by_this_gate"] is False, review
    assert review["dispatch_contract"]["request_schema_ui_enabled_by_this_gate"] is False, review
    assert summary["optimizer_count"] == 1, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready"] is False, summary
    assert summary["native_dispatch_allowed"] is False, summary
    assert summary["training_path_enabled"] is False, summary
    return {
        "schema_version": 1,
        "probe": "turbocore_muon_dispatch_integration_review_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
