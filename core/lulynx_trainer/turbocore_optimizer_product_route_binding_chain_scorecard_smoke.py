"""Smoke checks for the v2 O4 product route-binding chain aggregate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (str(SCRIPT_ROOT), str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_product_route_binding_chain_scorecard import (  # noqa: E402
    build_optimizer_product_route_binding_chain_scorecard,
)
from turbocore_native_update_product_exposure_decision_smoke import run_smoke as run_product_exposure_smoke  # noqa: E402


def run_smoke() -> dict[str, Any]:
    product_exposure = run_product_exposure_smoke()
    assert product_exposure["ok"] is True, product_exposure
    assert product_exposure["real_artifact_checked"] is True, product_exposure
    report = build_optimizer_product_route_binding_chain_scorecard(write_artifact=True)
    summary = report["summary"]
    rows = {row["roadmap_item"]: row for row in report["rows"]}
    artifact_path = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_optimizer_product_route_binding_chain_scorecard.json"
    )

    assert report["scorecard"] == "turbocore_optimizer_product_route_binding_chain_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["product_route_binding_chain_ready"] is False, report
    assert report["promotion_ready"] is False, report
    assert report["product_training_route_bound"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["product_native_ready"] is False, report
    assert summary["product_route_binding_chain_stage_count"] == 7, report
    assert summary["product_route_binding_chain_ready_stage_count"] == 7, report
    assert summary["product_route_binding_chain_open_stage_count"] == 0, report
    assert summary["product_route_binding_chain_contract_ready_count"] == 7, report
    assert summary["product_route_binding_chain_approval_missing_count"] >= 3, report
    assert summary["product_route_binding_chain_product_training_route_bound_count"] == 0, report
    assert summary["product_route_binding_chain_runtime_dispatch_ready_count"] == 0, report
    assert summary["product_route_binding_chain_native_dispatch_allowed_count"] == 0, report
    assert summary["product_route_binding_chain_training_path_enabled_count"] == 0, report
    assert summary["product_route_binding_chain_default_behavior_changed_count"] == 0, report
    assert summary["product_route_binding_chain_product_native_ready_count"] == 0, report
    assert summary["product_training_route_binding_kwargs_wired_count"] == 4, report
    assert summary["product_launch_staging_wired_count"] == 2, report
    assert summary["run_local_adapter_staged_count"] == 0, report
    assert summary["runtime_config_patch_applied_count"] == 0, report
    assert summary["training_loop_contract_candidate_switch_count"] == 3, report
    assert summary["training_loop_contract_open_training_path_enabled_count"] == 1, report
    assert all(row["stage_ready"] is True for row in rows.values()), rows
    assert rows["O4-6"]["training_path_enabled"] is False, rows["O4-6"]
    assert "owner_release_approval_missing" in report["promotion_blockers"], report
    assert "product_exposure_decision_not_recorded" in report["promotion_blockers"], report
    assert artifact_path.exists(), artifact_path

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_product_route_binding_chain_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
