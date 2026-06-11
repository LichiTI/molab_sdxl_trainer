"""Smoke checks for selected plugin factored-memory runtime precondition rehearsal."""

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

from core.turbocore_plugin_factored_memory_state_layout_scorecard import (  # noqa: E402
    FACTORED_MEMORY_PLUGIN_OPTIMIZERS,
)
from core.turbocore_plugin_factored_memory_runtime_precondition_rehearsal_scorecard import (  # noqa: E402
    ROADMAP,
    build_plugin_factored_memory_runtime_precondition_rehearsal_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_factored_memory_runtime_precondition_rehearsal_scorecard(write_artifact=True)
    summary = report["summary"]
    rows = {str(row["selected_optimizer_name"]): row for row in report["cases"]}
    assert report["scorecard"] == "turbocore_plugin_factored_memory_runtime_precondition_rehearsal_scorecard_v0", report
    assert report["roadmap"] == ROADMAP, report
    assert report["ok"] is True, report
    assert report["runtime_precondition_rehearsal_ready"] is True, report
    assert report["runtime_dispatch_rehearsal_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert summary["selected_optimizer_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["case_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["runtime_precondition_rehearsal_ready_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["native_layout_abi_ready_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["quality_matrix_ready_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["formula_tensor_binding_matrix_ready_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["dispatch_review_ready_count"] == len(FACTORED_MEMORY_PLUGIN_OPTIMIZERS), summary
    assert summary["native_step_count"] == 0, summary
    assert summary["native_kernel_launch_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    for expected in FACTORED_MEMORY_PLUGIN_OPTIMIZERS:
        row = rows[expected]
        assert row["runtime_precondition_rehearsal_ready"] is True, row
        assert row["runtime_dispatch_rehearsal_ready"] is False, row
        assert row["native_step_executed"] is False, row
        assert row["native_kernel_launched"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["native_dispatch_allowed"] is False, row
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_factored_memory_runtime_precondition_rehearsal_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
