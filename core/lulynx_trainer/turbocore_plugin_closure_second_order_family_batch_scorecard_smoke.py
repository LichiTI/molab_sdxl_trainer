"""Smoke checks for selected plugin closure/second-order family batch scorecard."""

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

from core.turbocore_plugin_closure_second_order_family_batch_scorecard import (  # noqa: E402
    TARGET_PLUGIN_OPTIMIZERS,
    build_plugin_closure_second_order_family_batch_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_plugin_closure_second_order_family_batch_scorecard()
    rows = {str(row["selected_optimizer_name"]): row for row in report["rows"]}
    summary = report["summary"]

    assert report["scorecard"] == "turbocore_plugin_closure_second_order_family_batch_scorecard_v0", report
    assert report["ok"] is True, report
    assert report["selected_closure_second_order_family_batch_ready"] is True, report
    assert report["report_only"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["native_kernel_ready"] is False, report
    assert report["product_native_ready"] is False, report
    assert report["product_native_dispatch_ready"] is False, report
    assert report["plugin_selected_native_ready_count"] == 0, report

    assert set(rows) == set(TARGET_PLUGIN_OPTIMIZERS), rows
    assert summary["selected_optimizer_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selector_closure_or_second_order_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["selector_classified_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["closure_required_count"] >= 4, summary
    assert summary["create_graph_required_count"] >= 3, summary
    assert summary["hessian_or_hvp_required_count"] >= 3, summary
    assert summary["higher_order_training_loop_abi_required_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["training_loop_abi_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["training_loop_abi_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["closure_replay_contract_required_count"] >= 4, summary
    assert summary["resume_parity_matrix_spec_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["resume_parity_matrix_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["closure_resume_replay_artifact_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["closure_resume_replay_artifact_row_count"] == 20, summary
    assert summary["closure_resume_replay_artifact_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["closure_resume_replay_row_implementation_ready_count"] == 20, summary
    assert summary["closure_replay_case_plan_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["closure_replay_case_planned_count"] >= len(TARGET_PLUGIN_OPTIMIZERS) * 3, summary
    assert summary["create_graph_hvp_lifetime_case_plan_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["create_graph_hvp_lifetime_case_planned_count"] >= len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["higher_order_graph_lifetime_required_count"] >= 3, summary
    assert summary["state_resume_adapter_required_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["state_resume_adapter_scope_plan_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["state_resume_adapter_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["unsafe_native_reuse_blocker_plan_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["unsafe_native_reuse_blocker_planned_count"] >= len(TARGET_PLUGIN_OPTIMIZERS) * 4, summary
    assert summary["native_kernel_precondition_plan_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["native_kernel_preconditions_implementation_ready_count"] == len(TARGET_PLUGIN_OPTIMIZERS), summary
    assert summary["adamw_kernel_compatible_count"] == 0, summary
    assert summary["simple_kernel_compatible_count"] == 0, summary
    assert summary["native_ready_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary
    assert summary["training_path_enabled_count"] == 0, summary
    assert summary["default_behavior_changed_count"] == 0, summary
    assert summary["plugin_selected_native_ready_count"] == 0, summary
    assert summary["exact_adamw_product_native_route_count_delta"] == 0, summary
    assert summary["unsafe_claim_count"] == 0, summary

    for row in rows.values():
        assert row["native_route_family"] == "closure_or_second_order", row
        assert row["selector_classified"] is True, row
        assert row["resume_proven"] is True, row
        assert row["batch_status"] == "training_loop_abi_replay_ready_report_only", row
        assert row["requires_higher_order_training_loop_abi"] is True, row
        assert row["training_loop_abi_spec_ready"] is True, row
        assert row["training_loop_abi_implementation_ready"] is True, row
        assert row["training_loop_abi_contract"]["state_resume_adapter_required"] is True, row
        assert row["training_loop_abi_contract"]["native_kernel_precondition"] == (
            "training_loop_abi_and_resume_parity_matrix_ready"
        ), row
        matrix = row["resume_parity_matrix_plan"]
        assert matrix["spec_ready"] is True, row
        assert matrix["implementation_ready"] is True, row
        assert matrix["closure_replay_cases"], row
        assert matrix["create_graph_hvp_lifetime_cases"], row
        assert matrix["state_resume_adapter_scope"], row
        assert matrix["state_resume_adapter_cases"], row
        artifact = matrix["closure_resume_replay_artifact"]
        assert artifact["artifact_kind"] == "closure_second_order_resume_replay_rows_v0", row
        assert artifact["artifact_ready"] is True, row
        assert artifact["artifact_status"] == "implementation_ready", row
        assert artifact["implementation_ready"] is True, row
        assert artifact["rows"], row
        for artifact_row in artifact["rows"]:
            assert artifact_row["artifact_status"] == "implementation_ready", artifact_row
            assert artifact_row["required_payload"], artifact_row
            assert artifact_row["replay_assertions"], artifact_row
            assert artifact_row["implementation_ready"] is True, artifact_row
            assert artifact_row["native_dispatch_allowed"] is False, artifact_row
        assert matrix["unsafe_native_reuse_blockers"], row
        assert "resume_parity_matrix_not_implemented" in matrix["unsafe_native_reuse_blockers"], row
        assert matrix["native_kernel_preconditions"], row
        assert "resume_parity_matrix_implementation_ready" in matrix["native_kernel_preconditions"], row
        assert "default_off_product_gate_preserved" in matrix["native_kernel_preconditions"], row
        assert matrix["evidence_status"] == "implementation_ready", row
        assert row["native_route"] == "none_report_only", row
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
        assert row["blocked_reasons"], row

    unsafe = build_plugin_closure_second_order_family_batch_scorecard(
        selector_report=_selector_fixture(runtime_dispatch_ready=True),
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["summary"]["unsafe_claim_count"] == 1, unsafe
    assert unsafe["plugin_selected_native_ready_count"] == 0, unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_closure_second_order_family_batch_scorecard_smoke",
        "ok": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _selector_fixture(*, runtime_dispatch_ready: bool = False) -> dict[str, Any]:
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
        "summary": {
            "plugin_optimizer_count": len(TARGET_PLUGIN_OPTIMIZERS),
            "missing_resume_count": 0,
            "route_family_counts": {"closure_or_second_order": len(TARGET_PLUGIN_OPTIMIZERS)},
        },
        "rows": [_selector_row(name) for name in TARGET_PLUGIN_OPTIMIZERS],
    }


def _selector_row(name: str) -> dict[str, Any]:
    return {
        "optimizer_name": name,
        "selector": "PytorchOptimizer",
        "native_route_family": "closure_or_second_order",
        "resume_proven": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
