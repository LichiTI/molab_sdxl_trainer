"""Smoke checks for closure/second-order plugin owner/release hold."""

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

from core.turbocore_plugin_closure_second_order_owner_release_hold_scorecard import (  # noqa: E402
    build_plugin_closure_second_order_owner_release_hold_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_closure_second_order_owner_release_hold_scorecard(write_artifact=True)
    summary = report["summary"]

    assert report["scorecard"] == "turbocore_plugin_closure_second_order_owner_release_hold_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["owner_release_hold_ready"] is True, report
    assert report["family_batch_ready"] is True, report
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
    assert report["product_native_dispatch_ready"] is False, report
    assert report["product_native_ready_count"] == 0, report
    assert summary["optimizer_count"] == 5, report
    assert summary["training_loop_abi_implementation_ready_count"] == 5, report
    assert summary["resume_parity_matrix_implementation_ready_count"] == 5, report
    assert summary["closure_resume_replay_row_implementation_ready_count"] == 20, report
    assert summary["product_native_ready_count"] == 0, report
    assert "plugin_closure_second_order_owner_approval_missing" in report["promotion_blockers"], report
    assert "plugin_closure_second_order_release_approval_missing" in report["promotion_blockers"], report

    unsafe = build_plugin_closure_second_order_owner_release_hold_scorecard(
        family_batch_report=_family_batch_fixture(training_path_enabled=True),
    )
    assert unsafe["ok"] is False, unsafe
    assert "plugin_closure_second_order_batch_enabled_dispatch" in unsafe["blocked_reasons"], unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_closure_second_order_owner_release_hold_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _family_batch_fixture(*, training_path_enabled: bool = False) -> dict[str, Any]:
    rows = [{"selected_optimizer_name": name} for name in ("adahessian", "alig", "bsam", "kron", "lbfgs")]
    return {
        "schema_version": 1,
        "scorecard": "fixture",
        "selected_closure_second_order_family_batch_ready": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": training_path_enabled,
        "rows": rows,
        "summary": {
            "selected_optimizer_count": 5,
            "training_loop_abi_implementation_ready_count": 5,
            "resume_parity_matrix_implementation_ready_count": 5,
            "native_kernel_preconditions_implementation_ready_count": 5,
            "closure_resume_replay_row_implementation_ready_count": 20,
            "plugin_selected_native_ready_count": 0,
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
