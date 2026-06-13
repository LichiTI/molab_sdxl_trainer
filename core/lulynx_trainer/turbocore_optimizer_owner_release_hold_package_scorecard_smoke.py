"""Smoke checks for the v2 O2 optimizer owner/release hold package."""

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

from core.turbocore_optimizer_owner_release_hold_package_scorecard import (  # noqa: E402
    build_optimizer_owner_release_hold_package_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_optimizer_owner_release_hold_package_scorecard(write_artifact=True)
    summary = report["summary"]
    rows = {row["family_id"]: row for row in report["rows"]}
    artifact_path = (
        REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_optimizer_owner_release_hold_package_scorecard.json"
    )

    assert report["scorecard"] == "turbocore_optimizer_owner_release_hold_package_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["owner_release_hold_package_ready"] is True, report
    assert report["promotion_ready"] is False, report
    assert report["manual_review_required"] is True, report
    assert report["owner_approval_recorded"] is False, report
    assert report["release_approval_recorded"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["product_native_ready_count"] == 0, report
    assert summary["owner_release_hold_package_family_count"] == 5, report
    assert summary["owner_release_hold_package_ready_family_count"] == 5, report
    assert summary["owner_release_hold_package_manual_review_required_count"] == 5, report
    assert summary["owner_release_hold_package_owner_approval_missing_count"] == 5, report
    assert summary["owner_release_hold_package_release_approval_missing_count"] == 5, report
    assert summary["owner_release_hold_package_runtime_dispatch_ready_count"] == 0, report
    assert summary["owner_release_hold_package_native_dispatch_allowed_count"] == 0, report
    assert summary["owner_release_hold_package_training_path_enabled_count"] == 0, report
    assert summary["owner_release_hold_package_default_behavior_changed_count"] == 0, report
    assert summary["owner_release_hold_package_product_native_ready_count"] == 0, report
    assert set(rows) == {
        "adam_like",
        "schedule_free",
        "factored_memory",
        "factored_custom",
        "simple_variant_selected_route",
    }, rows
    assert rows["adam_like"]["optimizer_count"] == 25, rows["adam_like"]
    assert rows["schedule_free"]["optimizer_count"] == 3, rows["schedule_free"]
    assert rows["factored_memory"]["optimizer_count"] == 8, rows["factored_memory"]
    assert rows["factored_custom"]["optimizer_count"] == 3, rows["factored_custom"]
    assert rows["simple_variant_selected_route"]["optimizer_count"] == 2, rows["simple_variant_selected_route"]
    assert "optimizer_owner_release_approval_missing" in report["promotion_blockers"], report
    assert "optimizer_release_approval_missing" in report["promotion_blockers"], report
    assert "optimizer_product_dispatch_not_approved" in report["promotion_blockers"], report
    assert artifact_path.exists(), artifact_path

    unsafe = build_optimizer_owner_release_hold_package_scorecard(
        family_reports={"adam_like": _unsafe_family_report()},
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["owner_release_hold_package_ready"] is False, unsafe
    unsafe_summary = unsafe["summary"]
    assert unsafe_summary["owner_release_hold_package_training_path_enabled_count"] == 1, unsafe
    assert unsafe_summary["owner_release_hold_package_product_native_ready_count"] == 1, unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_owner_release_hold_package_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _unsafe_family_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "fixture",
        "gate": "unsafe_fixture",
        "owner_release_hold_ready": False,
        "manual_review_required": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "runtime_dispatch_ready": True,
        "native_dispatch_allowed": True,
        "training_path_enabled": True,
        "default_behavior_changed": True,
        "request_fields_emitted": True,
        "schema_exposure_allowed": True,
        "ui_exposure_allowed": True,
        "blocked_reasons": ["unsafe_fixture_enabled_training_path"],
        "promotion_blockers": ["unsafe_fixture_enabled_training_path"],
        "summary": {"optimizer_count": 1, "product_native_ready_count": 1},
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
