"""Smoke for the TurboCore optimizer family follow-up scorecard."""

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

from core.turbocore_optimizer_family_follow_up_scorecard import (  # noqa: E402
    build_family_follow_up_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_family_follow_up_scorecard(write_artifact=True)
    summary = report["summary"]
    rows = {str(row["optimizer_name"]): row for row in report["rows"]}

    assert report["scorecard"] == "turbocore_optimizer_family_follow_up_scorecard_v0", report
    assert report["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design_v2.md", report
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert summary["family_follow_up_family_count"] == 4, summary
    assert summary["family_follow_up_base_canary_ready_count"] == 4, summary
    assert summary["family_follow_up_remaining_branch_count"] == 0, summary
    assert summary["family_follow_up_branch_contract_tracked_count"] == 7, summary
    assert summary["family_follow_up_branch_reference_ready_count"] == 6, summary
    assert summary["family_follow_up_branch_implementation_ready_count"] == 7, summary
    assert summary["family_follow_up_branch_native_gap_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert sorted(rows) == ["fromage", "pid", "rmsprop", "sgdp"], rows
    assert rows["rmsprop"]["remaining_branch_count"] == 0, rows["rmsprop"]
    assert rows["pid"]["remaining_branch_count"] == 0, rows["pid"]
    assert rows["sgdp"]["remaining_branch_count"] == 0, rows["sgdp"]
    assert rows["fromage"]["remaining_branch_count"] == 0, rows["fromage"]
    for row in rows.values():
        assert row["base_canary_ready"] is True, row
        assert row["optimizer_family"] == "simple_formula", row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["product_native_dispatch_ready"] is False, row
        assert row["native_step_count"] >= 1, row
        assert row["native_kernel_launch_count"] >= 1, row

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_family_follow_up_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
