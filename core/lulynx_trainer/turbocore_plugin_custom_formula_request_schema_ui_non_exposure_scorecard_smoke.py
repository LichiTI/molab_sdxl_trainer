"""Smoke checks for custom-formula request/schema/UI non-exposure."""

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

from core.turbocore_plugin_custom_formula_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    build_plugin_custom_formula_request_schema_ui_non_exposure_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_custom_formula_request_schema_ui_non_exposure_scorecard(write_artifact=True)
    summary = report["summary"]

    assert report["scorecard"] == (
        "turbocore_plugin_custom_formula_request_schema_ui_non_exposure_scorecard_v0"
    ), report
    assert report["ok"] is True, report
    assert report["request_schema_ui_non_exposure_ready"] is True, report
    assert report["owner_release_hold_ready"] is True, report
    assert report["family_batch_ready"] is True, report
    assert report["owner_approval_recorded"] is False, report
    assert report["release_approval_recorded"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["request_adapter_enabled"] is False, report
    assert report["backend_router_registered"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["product_native_ready_count"] == 0, report
    assert summary["optimizer_count"] == 47, report
    assert summary["present_boundary_path_count"] > 0, report
    assert summary["scanned_file_count"] > 0, report
    assert summary["forbidden_token_hit_count"] == 0, report
    assert summary["product_native_ready_count"] == 0, report
    assert "plugin_custom_formula_request_schema_ui_exposure_not_approved" in report[
        "promotion_blockers"
    ], report

    unsafe = build_plugin_custom_formula_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=_hold_fixture(native_dispatch_allowed=True),
    )
    assert unsafe["ok"] is False, unsafe
    assert "plugin_custom_formula_request_schema_ui_hold_enabled_boundary" in unsafe[
        "blocked_reasons"
    ], unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_custom_formula_request_schema_ui_non_exposure_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _hold_fixture(*, native_dispatch_allowed: bool = False) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "fixture",
        "owner_release_hold_ready": True,
        "family_batch_ready": True,
        "owner_approval_recorded": False,
        "release_approval_recorded": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": native_dispatch_allowed,
        "training_path_enabled": False,
        "summary": {"optimizer_count": 47},
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
