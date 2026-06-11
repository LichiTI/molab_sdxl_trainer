"""Smoke checks for selected plugin optimizer family batch scorecard."""

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

from core.turbocore_plugin_optimizer_family_batch_scorecard import (  # noqa: E402
    build_plugin_optimizer_family_batch_scorecard,
)

def run_smoke() -> dict[str, Any]:
    report = build_plugin_optimizer_family_batch_scorecard(refresh_family_artifacts=True)
    rows = {str(row["native_route_family"]): row for row in report["family_rows"]}
    summary = report["summary"]
    assert report["ok"] is True, report
    assert report["plugin_optimizer_family_batch_ready"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["runtime_dispatch_ready"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert summary["plugin_optimizer_count"] >= 100, report
    assert summary["selected_optimizer_gate_ready_count"] == 10, report
    assert summary["selected_optimizer_gate_pending_count"] == 0, report
    assert summary["plugin_selected_native_ready_count"] == 0, report
    assert summary["plugin_factored_memory_layout_observed_count"] == 8, report
    assert summary["plugin_factored_memory_manual_pending_count"] == 0, report
    assert rows["adam_like_formula"]["selected_optimizer_gate_ready"] is True, rows["adam_like_formula"]
    assert rows["adam_like_formula"]["selected_native_canary_ready_count"] >= 25, rows["adam_like_formula"]
    assert summary["selected_adamlike_e2e_shadow_matrix_ready"] is True, report
    assert summary["selected_adamlike_canary_rollout_policy_ready"] is True, report
    assert summary["selected_adamlike_owner_release_hold_ready"] is True, report
    assert summary["selected_adamlike_owner_release_hold_optimizer_count"] == 25, report
    assert summary["selected_adamlike_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_adamlike_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_adamlike_request_schema_ui_optimizer_count"] == 25, report
    assert summary["selected_adamlike_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_adamlike_request_schema_ui_product_native_ready_count"] == 0, report
    assert rows["adam_like_formula"]["next_gate"] == (
        "keep Adam-like native dispatch unwired until explicit owner/release approval is recorded"
    ), rows["adam_like_formula"]
    assert rows["schedule_free_state_machine"]["selected_optimizer_gate_ready"] is True, rows["schedule_free_state_machine"]
    assert rows["schedule_free_state_machine"]["selected_native_canary_ready_count"] == 3, rows["schedule_free_state_machine"]
    assert summary["selected_schedulefree_family_batch_ready"] is True, report
    assert summary["selected_schedulefree_e2e_shadow_case_count"] == 6, report
    assert summary["selected_schedulefree_native_canary_ready_count"] == 3, report
    assert summary["selected_schedulefree_dispatch_review_gate_ready"] is True, report
    assert summary["selected_schedulefree_owner_release_hold_ready"] is True, report
    assert summary["selected_schedulefree_owner_release_hold_optimizer_count"] == 3, report
    assert summary["selected_schedulefree_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_schedulefree_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_schedulefree_request_schema_ui_optimizer_count"] == 3, report
    assert summary["selected_schedulefree_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_schedulefree_request_schema_ui_product_native_ready_count"] == 0, report
    assert rows["schedule_free_state_machine"]["next_gate"] == (
        "keep schedule-free native dispatch unwired until explicit owner/release approval is recorded"
    ), rows["schedule_free_state_machine"]
    assert rows["adaptive_lr_state_machine"]["selected_optimizer_gate_ready"] is True, rows["adaptive_lr_state_machine"]
    assert summary["selected_adaptivelr_family_batch_ready"] is True, report
    assert summary["selected_adaptivelr_reference_ready_count"] == 6, report
    assert summary["selected_adaptivelr_state_machine_abi_spec_ready_count"] == 6, report
    assert summary["selected_adaptivelr_state_machine_abi_implementation_ready_count"] == 6, report
    assert summary["selected_adaptivelr_native_kernel_preconditions_spec_ready_count"] == 6, report
    assert summary["selected_adaptivelr_native_kernel_preconditions_implementation_ready_count"] == 6, report
    assert summary["selected_adaptivelr_state_machine_replay_matrix_artifact_ready_count"] == 6, report
    assert summary["selected_adaptivelr_state_machine_replay_matrix_implementation_ready_count"] == 6, report
    assert summary["selected_adaptivelr_state_machine_replay_case_planned_count"] == 36, report
    assert summary["selected_adaptivelr_state_machine_replay_case_implementation_ready_count"] == 36, report
    assert summary["selected_adaptivelr_state_machine_replay_resume_case_planned_count"] == 24, report
    assert summary["selected_adaptivelr_state_machine_replay_resume_case_implementation_ready_count"] == 24, report
    assert summary["selected_adaptivelr_owner_release_hold_ready"] is True, report
    assert summary["selected_adaptivelr_owner_release_hold_optimizer_count"] == 6, report
    assert summary["selected_adaptivelr_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_adaptivelr_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_adaptivelr_request_schema_ui_optimizer_count"] == 6, report
    assert summary["selected_adaptivelr_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_adaptivelr_request_schema_ui_product_native_ready_count"] == 0, report
    assert rows["adaptive_lr_state_machine"]["next_gate"] == (
        "keep adaptive-LR native dispatch unwired until explicit owner/release approval is recorded"
    ), rows["adaptive_lr_state_machine"]
    assert rows["simple_formula"]["selected_optimizer_gate_ready"] is True, rows["simple_formula"]
    assert summary["selected_simple_formula_family_batch_ready"] is True, report
    assert summary["selected_simple_formula_optimizer_count"] == 18, report
    assert summary["selected_simple_formula_reference_canary_ready_count"] == 1, report
    assert summary["selected_simple_formula_native_canary_ready_count"] == 18, report
    assert rows["simple_formula"]["selected_native_canary_ready_count"] == 18, rows["simple_formula"]
    assert summary["selected_simple_formula_e2e_shadow_matrix_ready"] is True, report
    assert summary["selected_simple_formula_canary_rollout_policy_ready"] is True, report
    assert summary["selected_simple_formula_dispatch_review_gate_ready"] is True, report
    assert summary["selected_simple_formula_owner_release_hold_ready"] is True, report
    assert summary["selected_simple_formula_owner_release_hold_optimizer_count"] == 18, report
    assert summary["selected_simple_formula_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_simple_formula_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_simple_formula_request_schema_ui_optimizer_count"] == 18, report
    assert summary["selected_simple_formula_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_simple_formula_request_schema_ui_product_native_ready_count"] == 0, report
    assert rows["simple_formula"]["next_gate"] == (
        "keep simple-formula native dispatch unwired until explicit owner/release approval is recorded"
    ), rows["simple_formula"]
    assert rows["closure_or_second_order"]["selected_optimizer_gate_ready"] is True, rows["closure_or_second_order"]
    assert summary["selected_closure_second_order_family_batch_ready"] is True, report
    assert summary["selected_closure_second_order_optimizer_count"] == 5, report
    assert summary["selected_closure_second_order_higher_order_abi_required_count"] == 5, report
    assert summary["selected_closure_second_order_training_loop_abi_spec_ready_count"] == 5, report
    assert summary["selected_closure_second_order_training_loop_abi_implementation_ready_count"] == 5, report
    assert summary["selected_closure_second_order_resume_parity_matrix_spec_ready_count"] == 5, report
    assert summary["selected_closure_second_order_resume_parity_matrix_implementation_ready_count"] == 5, report
    assert summary["selected_closure_second_order_closure_replay_case_planned_count"] == 22, report
    assert summary["selected_closure_second_order_create_graph_hvp_lifetime_case_planned_count"] == 18, report
    assert summary["selected_closure_second_order_closure_resume_replay_artifact_ready_count"] == 5, report
    assert summary["selected_closure_second_order_closure_resume_replay_artifact_row_count"] == 20, report
    assert summary["selected_closure_second_order_closure_resume_replay_artifact_implementation_ready_count"] == 5, report
    assert summary["selected_closure_second_order_closure_resume_replay_row_implementation_ready_count"] == 20, report
    assert summary["selected_closure_second_order_native_kernel_precondition_plan_ready_count"] == 5, report
    assert summary["selected_closure_second_order_native_kernel_preconditions_implementation_ready_count"] == 5, report
    assert summary["selected_closure_second_order_owner_release_hold_ready"] is True, report
    assert summary["selected_closure_second_order_owner_release_hold_optimizer_count"] == 5, report
    assert summary["selected_closure_second_order_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_closure_second_order_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_closure_second_order_request_schema_ui_optimizer_count"] == 5, report
    assert summary["selected_closure_second_order_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_closure_second_order_request_schema_ui_product_native_ready_count"] == 0, report
    assert rows["custom_formula"]["selected_optimizer_gate_ready"] is True, rows["custom_formula"]
    assert summary["selected_custom_formula_family_batch_ready"] is True, report
    assert summary["selected_custom_formula_optimizer_count"] == 47, report
    assert summary["selected_custom_formula_parity_required_count"] == 47, report
    assert summary["selected_custom_formula_backlog_ready_count"] == 47, report
    assert summary["selected_custom_formula_evidence_artifact_planned_count"] == 235, report
    assert summary["selected_custom_formula_evidence_status_pending_total"] == 0, report
    assert summary["selected_custom_formula_formula_spec_artifact_ready_count"] == 47, report
    assert summary["selected_custom_formula_formula_spec_artifact_pending_count"] == 0, report
    assert summary["selected_custom_formula_state_inventory_skeleton_count"] == 47, report
    assert summary["selected_custom_formula_state_inventory_artifact_ready_count"] == 47, report
    assert summary["selected_custom_formula_state_inventory_artifact_pending_count"] == 0, report
    assert summary["selected_custom_formula_quality_guard_matrix_artifact_ready_count"] == 47, report
    assert summary["selected_custom_formula_quality_guard_matrix_artifact_pending_count"] == 0, report
    assert summary["selected_custom_formula_quality_guard_matrix_case_planned_count"] == 329, report
    assert summary["selected_custom_formula_formula_parity_matrix_artifact_planned_count"] == 47, report
    assert summary["selected_custom_formula_formula_parity_matrix_implementation_ready_count"] == 47, report
    assert summary["selected_custom_formula_formula_parity_case_planned_count"] == 282, report
    assert summary["selected_custom_formula_resume_parity_matrix_artifact_planned_count"] == 47, report
    assert summary["selected_custom_formula_resume_parity_matrix_implementation_ready_count"] == 47, report
    assert summary["selected_custom_formula_owner_release_hold_ready"] is True, report
    assert summary["selected_custom_formula_owner_release_hold_optimizer_count"] == 47, report
    assert summary["selected_custom_formula_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_custom_formula_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_custom_formula_request_schema_ui_optimizer_count"] == 47, report
    assert summary["selected_custom_formula_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_custom_formula_request_schema_ui_product_native_ready_count"] == 0, report
    assert (
        rows["custom_formula"]["next_gate"]
        == "keep custom-formula native dispatch unwired until explicit owner/release approval is recorded"
    ), rows["custom_formula"]
    assert summary["selected_custom_formula_resume_parity_case_planned_count"] == 188, report
    assert rows["factored_memory_layout"]["selected_optimizer_gate_ready"] is True, rows["factored_memory_layout"]
    assert summary["selected_factored_memory_family_batch_ready"] is True, report
    assert summary["selected_factored_memory_optimizer_count"] == 8, report
    assert summary["selected_factored_memory_observed_layout_count"] == 8, report
    assert summary["selected_factored_memory_native_layout_abi_ready_count"] == 8, report
    assert summary["selected_factored_memory_quality_matrix_ready_count"] == 8, report
    assert summary["selected_factored_memory_native_kernel_entry_condition_ready_count"] == 8, report
    assert summary["selected_factored_memory_formula_tensor_binding_matrix_artifact_ready_count"] == 8, report
    assert summary["selected_factored_memory_formula_tensor_binding_matrix_implementation_ready_count"] == 8, report
    assert summary["selected_factored_memory_formula_step_execution_ready_count"] == 8, report
    assert summary["selected_factored_memory_resume_next_step_replay_ready_count"] == 8, report
    assert summary["selected_factored_memory_tensor_binding_ready_count"] == 8, report
    assert summary["selected_factored_memory_dispatch_review_gate_ready"] is True, report
    assert summary["selected_factored_memory_dispatch_review_ready_count"] == 8, report
    assert summary["selected_factored_memory_formula_parity_case_planned_count"] == 40, report
    assert summary["selected_factored_memory_tensor_binding_case_planned_count"] == 24, report
    assert summary["selected_factored_memory_owner_release_hold_ready"] is True, report
    assert summary["selected_factored_memory_owner_release_hold_optimizer_count"] == 8, report
    assert summary["selected_factored_memory_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_factored_memory_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_factored_memory_request_schema_ui_optimizer_count"] == 8, report
    assert summary["selected_factored_memory_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_factored_memory_request_schema_ui_product_native_ready_count"] == 0, report
    assert (
        rows["factored_memory_layout"]["next_gate"]
        == "keep factored-memory native dispatch unwired until explicit owner/release approval is recorded"
    ), rows["factored_memory_layout"]
    assert rows["fused_backward"]["selected_optimizer_gate_ready"] is True, rows["fused_backward"]
    assert summary["selected_fused_backward_family_batch_ready"] is True, report
    assert summary["selected_fused_backward_optimizer_count"] == 2, report
    assert summary["selected_fused_backward_gradient_ownership_abi_required_count"] == 2, report
    assert summary["selected_fused_backward_per_optimizer_abi_spec_ready_count"] == 2, report
    assert summary["selected_fused_backward_abi_implementation_ready_count"] == 2, report
    assert summary["selected_fused_backward_native_kernel_preconditions_spec_ready_count"] == 2, report
    assert summary["selected_fused_backward_resume_parity_matrix_spec_ready_count"] == 2, report
    assert summary["selected_fused_backward_resume_parity_matrix_implementation_ready_count"] == 2, report
    assert summary["selected_fused_backward_replay_case_planned_count"] == 10, report
    assert summary["selected_fused_backward_replay_case_implementation_ready_count"] == 10, report
    assert summary["selected_fused_backward_owner_release_hold_ready"] is True, report
    assert summary["selected_fused_backward_owner_release_hold_optimizer_count"] == 2, report
    assert summary["selected_fused_backward_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_fused_backward_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_fused_backward_request_schema_ui_optimizer_count"] == 2, report
    assert summary["selected_fused_backward_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_fused_backward_request_schema_ui_product_native_ready_count"] == 0, report
    assert rows["fused_backward"]["next_gate"] == (
        "keep fused-backward native dispatch unwired until explicit owner/release approval is recorded"
    ), rows["fused_backward"]
    assert rows["model_or_shape_aware"]["selected_optimizer_gate_ready"] is True, rows["model_or_shape_aware"]
    assert summary["selected_model_shape_aware_family_batch_ready"] is True, report
    assert summary["selected_model_shape_aware_optimizer_count"] == 7, report
    assert summary["selected_model_shape_aware_param_group_contract_count"] == 7, report
    assert summary["selected_model_shape_aware_param_group_abi_spec_ready_count"] == 7, report
    assert summary["selected_model_shape_aware_param_group_abi_implementation_ready_count"] == 7, report
    assert summary["selected_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count"] == 7, report
    assert summary["selected_model_shape_aware_param_group_resume_replay_matrix_row_count"] == 29, report
    assert summary["selected_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count"] == 7, report
    assert summary["selected_model_shape_aware_param_group_resume_replay_row_implementation_ready_count"] == 29, report
    assert summary["selected_model_shape_aware_owner_release_hold_ready"] is True, report
    assert summary["selected_model_shape_aware_owner_release_hold_optimizer_count"] == 7, report
    assert summary["selected_model_shape_aware_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_model_shape_aware_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_model_shape_aware_request_schema_ui_optimizer_count"] == 7, report
    assert summary["selected_model_shape_aware_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_model_shape_aware_request_schema_ui_product_native_ready_count"] == 0, report
    assert rows["model_or_shape_aware"]["next_gate"] == (
        "keep model/shape-aware native dispatch unwired until explicit owner/release approval is recorded"
    ), rows["model_or_shape_aware"]
    assert rows["state_adapter_special"]["selected_optimizer_gate_ready"] is True, rows["state_adapter_special"]
    assert summary["selected_state_adapter_special_family_batch_ready"] is True, report
    assert summary["selected_state_adapter_special_optimizer_count"] == 3, report
    assert summary["selected_state_adapter_special_param_ownership_abi_required_count"] == 3, report
    assert summary["selected_state_adapter_special_adapter_abi_spec_ready_count"] == 3, report
    assert summary["selected_state_adapter_special_adapter_abi_implementation_ready_count"] == 3, report
    assert summary["selected_state_adapter_special_native_kernel_precondition_spec_ready_count"] == 3, report
    assert summary["selected_state_adapter_special_native_kernel_precondition_implementation_ready_count"] == 3, report
    assert summary["selected_state_adapter_special_resume_matrix_artifact_ready_count"] == 3, report
    assert summary["selected_state_adapter_special_resume_matrix_implementation_ready_count"] == 3, report
    assert summary["selected_state_adapter_special_resume_replay_case_planned_count"] == 15, report
    assert summary["selected_state_adapter_special_resume_replay_case_implementation_ready_count"] == 15, report
    assert summary["selected_state_adapter_special_resume_translation_case_planned_count"] == 12, report
    assert summary["selected_state_adapter_special_resume_translation_case_implementation_ready_count"] == 12, report
    assert summary["selected_state_adapter_special_owner_release_hold_ready"] is True, report
    assert summary["selected_state_adapter_special_owner_release_hold_optimizer_count"] == 3, report
    assert summary["selected_state_adapter_special_owner_release_hold_product_native_ready_count"] == 0, report
    assert summary["selected_state_adapter_special_request_schema_ui_non_exposure_ready"] is True, report
    assert summary["selected_state_adapter_special_request_schema_ui_optimizer_count"] == 3, report
    assert summary["selected_state_adapter_special_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert summary["selected_state_adapter_special_request_schema_ui_product_native_ready_count"] == 0, report
    assert rows["state_adapter_special"]["next_gate"] == (
        "keep state-adapter-special native dispatch unwired until explicit owner/release approval is recorded"
    ), rows["state_adapter_special"]
    assert rows["factored_memory_layout"]["builtin_layout_reference_available"] is True, rows["factored_memory_layout"]
    assert rows["factored_memory_layout"]["plugin_state_layout_observed_count"] == 8, rows["factored_memory_layout"]
    assert rows["factored_memory_layout"]["selected_optimizer_gate_ready"] is True, rows["factored_memory_layout"]
    assert report["recommended_next_step"] == (
        "prepare plugin selected-family owner/release hold with product dispatch still default-off"
    ), report
    assert all(row["native_dispatch_allowed"] is False for row in rows.values()), rows
    _write_real_artifact(report)
    return {
        "schema_version": 1,
        "probe": "turbocore_plugin_optimizer_family_batch_scorecard_smoke",
        "ok": True,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def _write_real_artifact(report: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_plugin_optimizer_family_batch_scorecard.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
