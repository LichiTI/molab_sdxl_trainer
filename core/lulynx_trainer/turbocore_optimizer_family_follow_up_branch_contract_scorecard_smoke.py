"""Smoke for the TurboCore optimizer family follow-up branch contracts."""

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

from core.turbocore_optimizer_family_follow_up_branch_contract_scorecard import (  # noqa: E402
    build_family_follow_up_branch_contract_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_family_follow_up_branch_contract_scorecard(write_artifact=True)
    summary = report["summary"]
    rows = {str(row["branch_id"]): row for row in report["rows"]}

    assert report["scorecard"] == "turbocore_optimizer_family_follow_up_branch_contract_scorecard_v0", report
    assert report["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design_v2.md", report
    assert report["ok"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert summary["family_follow_up_branch_contract_tracked_count"] == 7, summary
    assert summary["family_follow_up_branch_reference_ready_count"] == 6, summary
    assert summary["family_follow_up_branch_implementation_ready_count"] == 7, summary
    assert summary["family_follow_up_branch_native_gap_count"] == 0, summary
    assert sorted(rows) == [
        "fromage_p_bound",
        "fromage_per_tensor_norm_matrix",
        "pid_momentum_three_buffer",
        "rmsprop_centered",
        "rmsprop_momentum",
        "sgdp_decoupled_decay",
        "sgdp_projection",
    ], rows
    ready_branches = {
        "rmsprop_centered",
        "rmsprop_momentum",
        "pid_momentum_three_buffer",
        "sgdp_decoupled_decay",
        "sgdp_projection",
        "fromage_p_bound",
        "fromage_per_tensor_norm_matrix",
    }
    for branch_id, row in rows.items():
        assert row["branch_contract_tracked"] is True, row
        assert row["branch_implementation_ready"] is (branch_id in ready_branches), row
        assert row["training_path_enabled"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["product_native_ready"] is False, row
        assert row["blocked_reasons"] == [], row

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_family_follow_up_branch_contract_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
