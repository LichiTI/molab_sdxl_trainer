"""Smoke checks for the v2 remaining-gate handoff scorecard."""

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

from core.turbocore_optimizer_v2_remaining_gate_handoff_scorecard import (  # noqa: E402
    build_optimizer_v2_remaining_gate_handoff_scorecard,
)


ARTIFACT = (
    REPO_ROOT
    / "temp"
    / "turbocore_optimizer"
    / "turbocore_optimizer_v2_remaining_gate_handoff_scorecard.json"
)


def run_smoke() -> dict[str, Any]:
    report = build_optimizer_v2_remaining_gate_handoff_scorecard(write_artifact=True)
    summary = report["summary"]
    rows = report["rows"]

    assert report["scorecard"] == "turbocore_optimizer_v2_remaining_gate_handoff_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["handoff_ready"] is True, report
    assert report["roadmap_complete"] is False, report
    assert report["promotion_ready"] is False, report
    assert report["manual_review_required"] is True, report
    assert report["approval_recorded"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["product_native_ready"] is False, report
    assert summary["v2_remaining_gate_total_count"] == 6, report
    assert summary["v2_remaining_gate_open_count"] == 6, report
    assert summary["v2_remaining_gate_closed_count"] == 0, report
    assert summary["v2_remaining_gate_owner_release_open_count"] == 5, report
    assert summary["v2_remaining_gate_product_exposure_open_count"] == 1, report
    assert summary["v2_remaining_gate_handoff_ready_count"] == 1, report
    assert summary["v2_remaining_gate_release_artifact_ready_count"] == 1, report
    assert summary["v2_remaining_gate_default_off_guard_count"] == 1, report
    assert summary["v2_remaining_gate_runtime_dispatch_ready_count"] == 0, report
    assert summary["v2_remaining_gate_native_dispatch_allowed_count"] == 0, report
    assert summary["v2_remaining_gate_training_path_enabled_count"] == 0, report
    assert summary["v2_remaining_gate_product_native_ready_count"] == 0, report
    assert summary["v2_remaining_gate_default_behavior_changed_count"] == 0, report
    assert summary["v2_remaining_gate_unsafe_claim_count"] == 0, report
    assert "v2_owner_release_approval_missing" in report["promotion_blockers"], report
    assert "v2_product_exposure_decision_not_recorded" in report["promotion_blockers"], report
    assert all(row["manual_record_required"] is True for row in rows), report
    assert ARTIFACT.exists(), ARTIFACT

    unsafe = build_optimizer_v2_remaining_gate_handoff_scorecard(
        source_reports={
            "native_readiness_gap": {
                "summary": {
                    "default_off_product_runtime_dispatch_ready_optimizer_count": 1,
                    "default_off_product_native_dispatch_allowed_optimizer_count": 0,
                    "default_off_product_training_path_enabled_optimizer_count": 0,
                    "default_off_product_product_native_ready_optimizer_count": 0,
                }
            },
            "release_artifact_first_validation": {
                "summary": {"release_artifact_first_validation_ready_count": 1}
            },
        },
        write_artifact=False,
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["summary"]["v2_remaining_gate_default_off_guard_count"] == 0, unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_remaining_gate_handoff_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
