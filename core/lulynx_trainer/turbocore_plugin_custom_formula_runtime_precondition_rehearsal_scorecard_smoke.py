"""Smoke checks for selected plugin custom-formula runtime precondition rehearsal."""

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

from core.turbocore_plugin_custom_formula_runtime_precondition_rehearsal_scorecard import (  # noqa: E402
    ROADMAP,
    build_plugin_custom_formula_runtime_precondition_rehearsal_scorecard,
)


EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT = 47


def run_smoke() -> dict[str, Any]:
    report = build_plugin_custom_formula_runtime_precondition_rehearsal_scorecard(write_artifact=True)
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_plugin_custom_formula_runtime_precondition_rehearsal_scorecard_v0", report
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
    assert summary["selected_optimizer_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["case_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["runtime_precondition_rehearsal_ready_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["formula_spec_ready_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["state_inventory_ready_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["quality_guard_ready_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["formula_parity_ready_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["resume_parity_ready_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["formula_step_execution_ready_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["resume_next_step_replay_ready_count"] == EXPECTED_CUSTOM_FORMULA_OPTIMIZER_COUNT, summary
    assert summary["native_step_count"] == 0, summary
    assert summary["native_kernel_launch_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    for row in report["cases"]:
        assert row["runtime_precondition_rehearsal_ready"] is True, row
        assert row["runtime_dispatch_rehearsal_ready"] is False, row
        assert row["formula_step_execution_ready"] is True, row
        assert row["resume_next_step_replay_ready"] is True, row
        assert row["native_step_executed"] is False, row
        assert row["native_kernel_launched"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["native_dispatch_allowed"] is False, row
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_custom_formula_runtime_precondition_rehearsal_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
