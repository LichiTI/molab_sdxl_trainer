"""Smoke checks for selected plugin closure/second-order runtime precondition rehearsal."""

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

from core.turbocore_plugin_closure_second_order_family_batch_scorecard import TARGET_PLUGIN_OPTIMIZERS  # noqa: E402
from core.turbocore_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard import (  # noqa: E402
    ROADMAP,
    build_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard(write_artifact=True)
    summary = report["summary"]
    rows = {str(row["selected_optimizer_name"]): row for row in report["cases"]}
    assert report["scorecard"] == "turbocore_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard_v0", report
    assert report["roadmap"] == ROADMAP, report
    assert report["ok"] is True, report
    assert report["runtime_precondition_rehearsal_ready"] is True, report
    assert report["runtime_dispatch_rehearsal_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert summary["selected_optimizer_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["case_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["runtime_precondition_rehearsal_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["training_loop_abi_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["resume_parity_matrix_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["closure_resume_replay_artifact_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["closure_resume_replay_row_ready_count"] == 20, summary
    assert summary["state_resume_adapter_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["native_kernel_precondition_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["native_step_count"] == 0, summary
    assert summary["native_kernel_launch_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    for expected in TARGET_PLUGIN_OPTIMIZERS:
        row = rows[expected]
        assert row["runtime_precondition_rehearsal_ready"] is True, row
        assert row["runtime_dispatch_rehearsal_ready"] is False, row
        assert row["training_loop_abi_ready"] is True, row
        assert row["resume_parity_matrix_ready"] is True, row
        assert row["closure_resume_replay_artifact_ready"] is True, row
        assert row["native_kernel_precondition_ready"] is True, row
        assert row["native_step_executed"] is False, row
        assert row["native_kernel_launched"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["native_dispatch_allowed"] is False, row
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
