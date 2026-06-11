"""Smoke checks for selected plugin factored-memory family batch scorecard."""

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

from core.turbocore_plugin_factored_memory_family_batch_scorecard import (  # noqa: E402
    build_plugin_factored_memory_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_factored_memory_family_batch_scorecard(write_artifact=True)
    summary = report["summary"]
    assert report["scorecard"] == "turbocore_plugin_factored_memory_family_batch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_factored_memory_family_batch_ready"] is True, report
    assert report["selected_native_layout_abi_ready"] is True, report
    assert report["layout_quality_matrix_ready"] is True, report
    assert report["native_kernel_entry_conditions_ready"] is True, report
    assert report["selected_optimizer_abi_ready"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["plugin_selected_native_ready_count"] == 0, report
    assert report["dispatch_integration_review"]["dispatch_review_gate_ready"] is True, report
    assert report["dispatch_integration_review"]["owner_approval_recorded"] is False, report
    assert report["dispatch_integration_review"]["release_approval_recorded"] is False, report
    assert report["dispatch_integration_review"]["request_fields_emitted"] is False, report
    assert report["dispatch_integration_review"]["schema_exposure_allowed"] is False, report
    assert report["dispatch_integration_review"]["ui_exposure_allowed"] is False, report
    assert summary["selected_optimizer_count"] == 8, summary
    assert summary["observed_resume_layout_count"] == 8, summary
    assert summary["manual_contract_pending_count"] == 0, summary
    assert summary["native_layout_abi_ready_count"] == 8, summary
    assert summary["quality_matrix_ready_count"] == 8, summary
    assert summary["native_kernel_entry_condition_ready_count"] == 8, summary
    assert summary["formula_tensor_binding_matrix_artifact_ready_count"] == 8, summary
    assert summary["formula_tensor_binding_matrix_implementation_ready_count"] == 8, summary
    assert summary["formula_step_execution_ready_count"] == 8, summary
    assert summary["resume_next_step_replay_ready_count"] == 8, summary
    assert summary["tensor_binding_ready_count"] == 8, summary
    assert summary["dispatch_review_gate_ready"] is True, summary
    assert summary["dispatch_review_ready_count"] == 8, summary
    assert summary["formula_parity_case_planned_count"] == 40, summary
    assert summary["tensor_binding_case_planned_count"] == 24, summary
    assert summary["native_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["plugin_selected_native_ready_count"] == 0, summary
    for row in report["rows"]:
        assert row["native_route_family"] == "factored_memory_layout", row
        assert row["native_layout_abi_ready"] is True, row
        assert row["quality_matrix_ready"] is True, row
        assert row["native_kernel_entry_condition_ready"] is True, row
        assert row["batch_status"] == "dispatch_review_ready_report_only", row
        assert row["dispatch_integration_review_ready"] is True, row
        assert row["state_key_count"] > 0, row
        assert row["tensor_state_count"] > 0, row
        matrix = row["formula_tensor_binding_matrix"]
        assert matrix["matrix_artifact_ready"] is True, row
        assert matrix["matrix_implementation_ready"] is True, row
        assert matrix["formula_step_execution_ready"] is True, row
        assert matrix["resume_next_step_replay_ready"] is True, row
        assert matrix["tensor_binding_ready"] is True, row
        assert len(matrix["planned_cases"]) == 5, row
        assert all(case["status"] == "implementation_ready" for case in matrix["planned_cases"]), row
        assert all(case["native_dispatch_allowed"] is False for case in matrix["planned_cases"]), row
        assert row["plugin_selected_native_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["training_path_enabled"] is False, row
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_factored_memory_family_batch_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
