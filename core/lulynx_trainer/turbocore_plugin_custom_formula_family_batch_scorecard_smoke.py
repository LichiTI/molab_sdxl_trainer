"""Smoke checks for selected plugin custom-formula family batch scorecard."""

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

from core.turbocore_plugin_custom_formula_family_batch_scorecard import (  # noqa: E402
    CUSTOM_FORMULA_ROUTE_FAMILY,
    build_plugin_custom_formula_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_custom_formula_family_batch_scorecard()
    rows = {str(row["selected_optimizer_name"]): row for row in report["rows"]}
    summary = report["summary"]
    selected_names = report["request_contract"]["selected_optimizer_names"]
    formula_ready_rows = {
        name: row for name, row in rows.items() if row["evidence_status"]["formula_spec"] == "ready"
    }
    state_inventory_ready_rows = {
        name: row for name, row in rows.items() if row["evidence_status"]["state_inventory"] == "ready"
    }
    quality_guard_ready_rows = {
        name: row for name, row in rows.items() if row["evidence_status"]["quality_guard_matrix"] == "ready"
    }
    ready_stage_total = sum(
        1 for row in rows.values() for status in row["evidence_status"].values() if status == "ready"
    )
    expected_formula_ready_names = selected_names

    assert report["scorecard"] == "turbocore_plugin_custom_formula_family_batch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_custom_formula_family_batch_ready"] is True, report
    assert report["report_only"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["plugin_selected_native_ready_count"] == 0, report

    assert selected_names == sorted(rows), report
    assert summary["selected_optimizer_count"] == len(rows), summary
    assert summary["selector_custom_formula_count"] == len(rows), summary
    assert summary["selector_custom_formula_count"] > 0, summary
    assert summary["selector_classified_count"] == len(rows), summary
    assert summary["resume_proven_count"] == len(rows), summary
    assert summary["formula_spec_required_count"] == len(rows), summary
    assert summary["state_inventory_required_count"] == len(rows), summary
    assert summary["quality_guard_required_count"] == len(rows), summary
    assert summary["formula_parity_required_count"] == len(rows), summary
    assert summary["resume_parity_required_count"] == len(rows), summary
    assert summary["backlog_ready_count"] == len(rows), summary
    assert summary["evidence_artifact_planned_count"] == len(rows) * 5, summary
    assert summary["evidence_status_pending_total"] == 0, summary
    assert summary["per_optimizer_next_action_count"] == 0, summary
    assert sorted(summary["evidence_stage_pending_counts"]) == [
        "formula_parity_matrix",
        "formula_spec",
        "quality_guard_matrix",
        "resume_parity_matrix",
        "state_inventory",
    ], summary
    assert summary["evidence_stage_pending_counts"]["formula_spec"] == len(rows) - len(formula_ready_rows), summary
    assert summary["evidence_stage_pending_counts"]["state_inventory"] == 0, summary
    assert summary["evidence_stage_pending_counts"]["quality_guard_matrix"] == 0, summary
    assert all(count == 0 for count in summary["evidence_stage_pending_counts"].values()), summary
    assert summary["evidence_stage_ready_counts"]["formula_spec"] == len(formula_ready_rows), summary
    assert summary["evidence_stage_ready_counts"]["state_inventory"] == len(state_inventory_ready_rows), summary
    assert summary["evidence_stage_ready_counts"]["quality_guard_matrix"] == len(quality_guard_ready_rows), summary
    assert all(count == len(rows) for count in summary["evidence_stage_ready_counts"].values()), summary
    assert summary["formula_spec_artifact_ready_count"] == len(formula_ready_rows), summary
    assert summary["formula_spec_artifact_ready_count"] == len(rows), summary
    assert summary["formula_spec_artifact_pending_count"] == 0, summary
    assert summary["formula_spec_ready_tier_counts"] == {
        "adaptive_lr_or_bound": 8,
        "adaptive_moment_variant": 14,
        "gradient_transform_or_projection": 11,
        "optimizer_specific_state_machine": 9,
        "quality_guard_sensitive": 5,
    }, summary
    assert summary["formula_spec_ready_optimizer_names"] == expected_formula_ready_names, summary
    assert summary["formula_state_inventory_skeleton_count"] == len(rows), summary
    assert summary["state_inventory_artifact_ready_count"] == len(rows), summary
    assert summary["state_inventory_artifact_pending_count"] == 0, summary
    assert summary["state_inventory_ready_optimizer_names"] == selected_names, summary
    assert summary["quality_guard_matrix_artifact_ready_count"] == len(rows), summary
    assert summary["quality_guard_matrix_artifact_pending_count"] == 0, summary
    assert summary["quality_guard_matrix_case_planned_count"] == len(rows) * 7, summary
    assert summary["formula_parity_matrix_artifact_planned_count"] == len(rows), summary
    assert summary["formula_parity_matrix_implementation_ready_count"] == len(rows), summary
    assert summary["formula_parity_case_planned_count"] == len(rows) * 6, summary
    assert summary["resume_parity_matrix_artifact_planned_count"] == len(rows), summary
    assert summary["resume_parity_matrix_implementation_ready_count"] == len(rows), summary
    assert summary["resume_parity_case_planned_count"] == len(rows) * 4, summary
    assert summary["execution_matrix_ready"] is True, summary
    assert summary["formula_step_execution_ready_count"] == len(rows), summary
    assert summary["resume_next_step_replay_ready_count"] == len(rows), summary
    assert summary["execution_failed_count"] == 0, summary
    assert set(summary["backlog_tier_counts"]) == {
        "adaptive_lr_or_bound",
        "adaptive_moment_variant",
        "gradient_transform_or_projection",
        "optimizer_specific_state_machine",
        "quality_guard_sensitive",
    }, summary
    assert sum(summary["backlog_tier_counts"].values()) == len(rows), summary
    assert summary["adamw_kernel_compatible_count"] == 0, summary
    assert summary["simple_kernel_compatible_count"] == 0, summary
    assert summary["native_ready_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["product_native_dispatch_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["default_behavior_changed_count"] == 0, summary
    assert summary["plugin_selected_native_ready_count"] == 0, summary
    assert summary["exact_adamw_product_native_route_count_delta"] == 0, summary
    assert summary["missing_selector_classification_count"] == 0, summary
    assert summary["unsafe_claim_count"] == 0, summary

    native = report["native_compatibility"]
    assert native["requires_per_optimizer_formula_spec"] is True, native
    assert native["requires_per_optimizer_state_inventory"] is True, native
    assert native["requires_quality_guard_matrix"] is True, native
    assert native["requires_formula_parity_matrix"] is True, native
    assert native["requires_resume_parity_matrix"] is True, native
    assert native["adamw_step_kernel_compatible"] is False, native
    assert native["simple_formula_kernel_compatible"] is False, native
    assert native["can_reuse_exact_adamw_native_dispatch"] is False, native
    backlog_plan = report["backlog_plan"]
    assert backlog_plan["status"] == "ready_for_parallel_per_optimizer_evidence", backlog_plan
    assert backlog_plan["native_work_policy"] == "blocked_until_optimizer_evidence_stages_pass", backlog_plan
    execution_matrix = report["execution_matrix"]
    assert execution_matrix["execution_matrix_ready"] is True, execution_matrix
    assert execution_matrix["formula_step_execution_ready_count"] == len(rows), execution_matrix
    assert execution_matrix["resume_next_step_replay_ready_count"] == len(rows), execution_matrix
    assert execution_matrix["execution_failed_count"] == 0, execution_matrix
    assert execution_matrix["training_path_enabled"] is False, execution_matrix
    assert execution_matrix["native_dispatch_allowed"] is False, execution_matrix
    assert execution_matrix["native_kernel_ready"] is False, execution_matrix

    for row in rows.values():
        assert row["native_route_family"] == CUSTOM_FORMULA_ROUTE_FAMILY, row
        assert row["selector_classified"] is True, row
        assert row["resume_proven"] is True, row
        assert row["batch_status"] == "formula_state_quality_parity_required_report_only", row
        assert row["native_route"] == "none_report_only", row
        assert row["adamw_state_schema_compatible"] is False, row
        assert row["adamw_kernel_compatible"] is False, row
        assert row["simple_formula_kernel_compatible"] is False, row
        assert row["can_reuse_exact_adamw_native_dispatch"] is False, row
        assert row["plugin_selected_native_ready"] is False, row
        assert row["product_native_ready"] is False, row
        assert row["product_native_dispatch_ready"] is False, row
        assert row["training_path_enabled"] is False, row
        assert row["default_behavior_changed"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert row["native_kernel_ready"] is False, row
        assert len(row["formula_state_quality_parity_work_items"]) == 5, row
        assert len(row["evidence_artifacts"]) == 5, row
        assert len(row["evidence_plan"]) == 5, row
        assert row["parallel_next_actions"] == [], row
        if row["selected_optimizer_name"] in expected_formula_ready_names:
            assert row["evidence_status"]["formula_spec"] == "ready", row
            assert row["formula_spec_artifact"]["status"] == "ready", row
            assert row["formula_spec_artifact"]["report_only"] is True, row
            assert row["formula_spec_artifact"]["state_inventory_status"] == "skeleton_only_pending_full_inventory", row
            assert row["formula_spec_artifact"]["formula_parity_status"] == "pending", row
            assert row["formula_spec_artifact"]["resume_parity_status"] == "pending", row
            assert row["formula_spec_artifact"]["native_kernel_ready"] is False, row
            assert row["formula_spec_artifact"]["state_inventory_skeleton"], row
            assert row["formula_spec_artifact"]["hparam_surface_skeleton"], row
            assert row["formula_spec_artifact"]["quality_guard_skeleton"], row
            assert "selected_plugin_custom_formula_spec_missing" not in row["blocked_reasons"], row
            assert row["formula_spec_artifact"]["source_review_target"].startswith("pytorch_optimizer:"), row
            assert row["evidence_status"]["state_inventory"] == "ready", row
            assert row["state_inventory_artifact"]["status"] == "ready", row
            assert row["state_inventory_artifact"]["report_only"] is True, row
            assert row["state_inventory_artifact"]["source_file"].startswith(
                "plugin/pytorch_optimizer-main/pytorch_optimizer/optimizer/"
            ), row
            assert row["state_inventory_artifact"]["source_class"], row
            assert row["state_inventory_artifact"]["state_dict_key_inventory"], row
            assert row["state_inventory_artifact"]["native_kernel_ready"] is False, row
            assert "selected_plugin_custom_state_inventory_missing" not in row["blocked_reasons"], row
            assert row["evidence_status"]["quality_guard_matrix"] == "ready", row
            assert row["quality_guard_artifact"]["status"] == "ready", row
            assert row["quality_guard_artifact"]["report_only"] is True, row
            assert row["quality_guard_artifact"]["guard_case_count"] == 7, row
            assert row["quality_guard_artifact"]["native_kernel_ready"] is False, row
            assert "selected_plugin_custom_quality_guard_matrix_missing" not in row["blocked_reasons"], row
            assert row["formula_parity_matrix_artifact"]["status"] == "implementation_ready", row
            assert row["formula_parity_matrix_artifact"]["case_count"] == 6, row
            assert row["formula_parity_matrix_artifact"]["implementation_ready"] is True, row
            assert row["formula_parity_matrix_artifact"]["implementation_evidence"]["formula_step_execution_ready"] is True, row
            assert row["formula_parity_matrix_artifact"]["native_kernel_ready"] is False, row
            assert row["resume_parity_matrix_artifact"]["status"] == "implementation_ready", row
            assert row["resume_parity_matrix_artifact"]["case_count"] == 4, row
            assert row["resume_parity_matrix_artifact"]["implementation_ready"] is True, row
            assert row["resume_parity_matrix_artifact"]["implementation_evidence"]["resume_next_step_replay_ready"] is True, row
            assert row["resume_parity_matrix_artifact"]["native_kernel_ready"] is False, row
        else:
            assert row["evidence_status"]["formula_spec"] == "pending", row
            assert row["formula_spec_artifact"] is None, row
            assert "selected_plugin_custom_formula_spec_missing" in row["blocked_reasons"], row
        assert all(status == "ready" for status in row["evidence_status"].values()), row
        assert "selected_plugin_custom_formula_parity_matrix_missing" not in row["blocked_reasons"], row
        assert "selected_plugin_custom_resume_parity_matrix_missing" not in row["blocked_reasons"], row
        assert all(
            item["status"] == row["evidence_status"][item["stage"]] for item in row["evidence_plan"]
        ), row
        assert all(item["blocks_native_kernel_work"] is True for item in row["evidence_plan"]), row
        backlog = row["custom_formula_backlog"]
        assert backlog["backlog_ready_for_owner"] is True, row
        assert backlog["native_kernel_work_allowed"] is False, row
        assert backlog["evidence_owner_status"] == "unassigned", row
        assert backlog["source_review_target"].startswith("pytorch_optimizer:"), row
        assert backlog["state_inventory_seed"], row
        assert backlog["hparam_surface_seed"], row
        assert backlog["quality_guard_seed"], row
        assert row["blocked_reasons"], row
        contract = row["custom_formula_contract"]
        assert contract["requires_backlog_tier_owner"] is True, row
        assert contract["requires_evidence_artifact_plan"] is True, row
        assert contract["requires_per_optimizer_formula_spec"] is True, row
        assert contract["requires_per_optimizer_state_inventory"] is True, row
        assert contract["requires_quality_guard_matrix"] is True, row
        assert contract["requires_formula_parity_matrix"] is True, row
        assert contract["requires_resume_parity_matrix"] is True, row
        assert contract["adamw_step_kernel_compatible"] is False, row
        assert contract["simple_formula_kernel_compatible"] is False, row
        assert contract["can_reuse_exact_adamw_native_dispatch"] is False, row

    unsafe = build_plugin_custom_formula_family_batch_scorecard(
        selector_report=_selector_fixture(["customprobe"], runtime_dispatch_ready=True),
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["summary"]["unsafe_claim_count"] == 1, unsafe
    assert unsafe["plugin_selected_native_ready_count"] == 0, unsafe
    assert unsafe["training_path_enabled"] is False, unsafe
    assert unsafe["default_behavior_changed"] is False, unsafe
    assert unsafe["runtime_dispatch_ready"] is False, unsafe
    assert unsafe["native_dispatch_allowed"] is False, unsafe
    assert unsafe["native_kernel_ready"] is False, unsafe
    assert unsafe["product_native_ready"] is False, unsafe
    assert unsafe["product_native_dispatch_ready"] is False, unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_custom_formula_family_batch_scorecard_smoke",
        "ok": True,
        "selected_optimizer_count": len(rows),
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _selector_fixture(names: list[str], *, runtime_dispatch_ready: bool = False) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scorecard": "selector_fixture",
        "ok": True,
        "plugin_selector_classification_ready": True,
        "selector_boundary_ready": True,
        "all_discovered_plugins_resume_proven": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": runtime_dispatch_ready,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
        "summary": {
            "plugin_optimizer_count": len(names),
            "missing_resume_count": 0,
            "route_family_counts": {CUSTOM_FORMULA_ROUTE_FAMILY: len(names)},
        },
        "rows": [_selector_row(name) for name in names],
    }


def _selector_row(name: str) -> dict[str, Any]:
    return {
        "optimizer_name": name,
        "selector": "PytorchOptimizer",
        "native_route_family": CUSTOM_FORMULA_ROUTE_FAMILY,
        "resume_proven": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "native_kernel_ready": False,
        "product_native_ready": False,
        "product_native_dispatch_ready": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
