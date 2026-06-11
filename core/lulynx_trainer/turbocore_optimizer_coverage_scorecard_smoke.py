"""Smoke checks for TurboCore optimizer family coverage scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_optimizer_coverage_scorecard import build_optimizer_family_coverage_scorecard  # noqa: E402
from core.turbocore_adaptive_lr_state_machine_batch_scorecard import (  # noqa: E402
    build_adaptive_lr_state_machine_batch_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_matrix_scorecard import (  # noqa: E402
    build_adaptive_lr_state_machine_replay_matrix_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import (  # noqa: E402
    build_adaptive_lr_state_machine_replay_executor_scorecard,
)
from core.turbocore_adaptive_lr_native_state_machine_abi_preconditions_scorecard import (  # noqa: E402
    build_adaptive_lr_native_state_machine_abi_preconditions_scorecard,
)
from core.turbocore_adaptive_lr_native_state_machine_abi_skeleton_scorecard import (  # noqa: E402
    build_adaptive_lr_native_state_machine_abi_skeleton_scorecard,
)
from core.turbocore_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard import (  # noqa: E402
    build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard,
)
from core.turbocore_adaptive_lr_native_state_machine_implementation_stub_scorecard import (  # noqa: E402
    build_adaptive_lr_native_state_machine_implementation_stub_scorecard,
)
from core.turbocore_adaptive_lr_cuda_kernel_contract_plan_scorecard import (  # noqa: E402
    build_adaptive_lr_cuda_kernel_contract_plan_scorecard,
)
from core.turbocore_adaptive_lr_cuda_kernel_implementation_scorecard import (  # noqa: E402
    build_adaptive_lr_cuda_kernel_implementation_scorecard,
)
from core.turbocore_adaptive_lr_training_tensor_binding_canary_scorecard import (  # noqa: E402
    build_adaptive_lr_training_tensor_binding_canary_scorecard,
)
from core.turbocore_adaptive_lr_runtime_dispatch_shadow_scorecard import (  # noqa: E402
    build_adaptive_lr_runtime_dispatch_shadow_scorecard,
)
from core.turbocore_adaptive_lr_training_loop_canary_scorecard import (  # noqa: E402
    build_adaptive_lr_training_loop_canary_scorecard,
)
from core.turbocore_adaptive_lr_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_adaptive_lr_e2e_shadow_matrix_scorecard,
)
from core.turbocore_adaptive_lr_canary_rollout_policy_scorecard import (  # noqa: E402
    build_adaptive_lr_canary_rollout_policy_scorecard,
)
from core.turbocore_adaptive_lr_dispatch_integration_review_scorecard import (  # noqa: E402
    build_adaptive_lr_dispatch_integration_review_scorecard,
)
from core.turbocore_adaptive_lr_owner_release_hold_scorecard import (  # noqa: E402
    build_adaptive_lr_owner_release_hold_scorecard,
)
from core.turbocore_adaptive_lr_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    build_adaptive_lr_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_factored_custom_optimizer_state_layout_scorecard import (  # noqa: E402
    build_factored_custom_optimizer_state_layout_scorecard,
)
from core.turbocore_factored_custom_optimizer_family_batch_scorecard import (  # noqa: E402
    build_factored_custom_optimizer_family_batch_scorecard,
)
from core.turbocore_factored_custom_owner_release_hold_scorecard import (  # noqa: E402
    build_factored_custom_owner_release_hold_scorecard,
)
from core.turbocore_factored_custom_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    build_factored_custom_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_muon_model_shape_aware_family_batch_scorecard import (  # noqa: E402
    build_muon_model_shape_aware_family_batch_scorecard,
)
from core.turbocore_muon_native_scratch_kernel_scorecard import (  # noqa: E402
    build_muon_native_scratch_kernel_scorecard,
)
from core.turbocore_muon_training_tensor_binding_canary_scorecard import (  # noqa: E402
    build_muon_training_tensor_binding_canary_scorecard,
)
from core.turbocore_muon_training_loop_canary_scorecard import (  # noqa: E402
    build_muon_training_loop_canary_scorecard,
)
from core.turbocore_muon_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_muon_e2e_shadow_matrix_scorecard,
)
from core.turbocore_muon_canary_rollout_policy_scorecard import (  # noqa: E402
    build_muon_canary_rollout_policy_scorecard,
)
from core.turbocore_muon_dispatch_integration_review_scorecard import (  # noqa: E402
    build_muon_dispatch_integration_review_scorecard,
)
from core.turbocore_muon_owner_release_hold_scorecard import (  # noqa: E402
    build_muon_owner_release_hold_scorecard,
)
from core.turbocore_muon_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    build_muon_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_adamw_variant_family_batch_scorecard import (  # noqa: E402
    build_adamw_variant_family_batch_scorecard,
)
from core.turbocore_adamw_variant_product_training_canary_scorecard import (  # noqa: E402
    build_adamw_variant_product_training_canary_scorecard,
)
from core.turbocore_adamw_variant_owner_release_hold_scorecard import (  # noqa: E402
    build_adamw_variant_owner_release_hold_scorecard,
)
from core.turbocore_adamw_variant_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    build_adamw_variant_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_adamw_variant_e2e_shadow_matrix_scorecard import (  # noqa: E402
    build_adamw_variant_e2e_shadow_matrix_scorecard,
)
from core.turbocore_adamw_variant_canary_rollout_policy_scorecard import (  # noqa: E402
    build_adamw_variant_canary_rollout_policy_scorecard,
)
from core.turbocore_adamw_variant_dispatch_integration_review_scorecard import (  # noqa: E402
    build_adamw_variant_dispatch_integration_review_scorecard,
)
from core.turbocore_adamw_representative_route_matrix_scorecard import (  # noqa: E402
    build_adamw_representative_route_matrix_scorecard,
)
from core.turbocore_exact_adamw_stream_event_chain_abi_scorecard import (  # noqa: E402
    build_exact_adamw_stream_event_chain_abi_scorecard,
)
from core.turbocore_simple_optimizer_family_batch_scorecard import (  # noqa: E402
    build_simple_optimizer_family_batch_scorecard,
)
from core.turbocore_simple_optimizer_product_training_canary_scorecard import (  # noqa: E402
    build_simple_optimizer_product_training_canary_scorecard,
)
from core.turbocore_simple_optimizer_owner_release_hold_scorecard import (  # noqa: E402
    build_simple_optimizer_owner_release_hold_scorecard,
)
from core.turbocore_simple_optimizer_schedulefree_rollout_policy_scorecard import (  # noqa: E402
    build_simple_optimizer_schedulefree_rollout_policy_scorecard,
)
from core.turbocore_simple_optimizer_schedulefree_dispatch_integration_review_scorecard import (  # noqa: E402
    build_simple_optimizer_schedulefree_dispatch_integration_review_scorecard,
)
from core.turbocore_simple_optimizer_schedulefree_owner_release_hold_scorecard import (  # noqa: E402
    build_simple_optimizer_schedulefree_owner_release_hold_scorecard,
)
from core.turbocore_simple_optimizer_quantized_dispatch_integration_review_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_dispatch_integration_review_scorecard,
)
from core.turbocore_simple_optimizer_quantized_owner_approval_hold_scorecard import (  # noqa: E402
    build_simple_optimizer_quantized_owner_approval_hold_scorecard,
)
from core.turbocore_simple_optimizer_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    build_simple_optimizer_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_optimizer_family_batch_scorecard import (  # noqa: E402
    build_plugin_optimizer_family_batch_scorecard,
)
from core.turbocore_plugin_selected_family_owner_release_hold_scorecard import (  # noqa: E402
    build_plugin_selected_family_owner_release_hold_scorecard,
)
from core.turbocore_plugin_selected_family_request_schema_ui_non_exposure_scorecard import (  # noqa: E402
    build_plugin_selected_family_request_schema_ui_non_exposure_scorecard,
)
from core.turbocore_plugin_adamlike_family_batch_scorecard import (  # noqa: E402
    build_plugin_adamlike_family_batch_scorecard,
)
from core.turbocore_plugin_adaptivelr_family_batch_scorecard import (  # noqa: E402
    build_plugin_adaptivelr_family_batch_scorecard,
)
from core.turbocore_plugin_closure_second_order_family_batch_scorecard import (  # noqa: E402
    build_plugin_closure_second_order_family_batch_scorecard,
)
from core.turbocore_plugin_custom_formula_family_batch_scorecard import (  # noqa: E402
    build_plugin_custom_formula_family_batch_scorecard,
)
from core.turbocore_plugin_factored_memory_family_batch_scorecard import (  # noqa: E402
    build_plugin_factored_memory_family_batch_scorecard,
)
from core.turbocore_plugin_fused_backward_family_batch_scorecard import (  # noqa: E402
    build_plugin_fused_backward_family_batch_scorecard,
)
from core.turbocore_plugin_model_shape_aware_family_batch_scorecard import (  # noqa: E402
    build_plugin_model_shape_aware_family_batch_scorecard,
)
from core.turbocore_plugin_schedulefree_family_batch_scorecard import (  # noqa: E402
    build_plugin_schedulefree_family_batch_scorecard,
)
from core.turbocore_plugin_simple_formula_family_batch_scorecard import (  # noqa: E402
    build_plugin_simple_formula_family_batch_scorecard,
)
from core.turbocore_plugin_state_adapter_special_family_batch_scorecard import (  # noqa: E402
    build_plugin_state_adapter_special_family_batch_scorecard,
)
from core.turbocore_optimizer_family_kernel_contract_scorecard import (  # noqa: E402
    build_optimizer_family_kernel_contract_scorecard,
)


def run_smoke() -> dict[str, Any]:
    report = build_optimizer_family_coverage_scorecard()
    return _validate_coverage_report(report, artifact_mode="artifact_first")


def run_rebuild_smoke() -> dict[str, Any]:
    _refresh_adamw_variant_review_artifacts()
    adamw_route_matrix = build_adamw_representative_route_matrix_scorecard(steps=4)
    exact_adamw_stream_abi = build_exact_adamw_stream_event_chain_abi_scorecard(write_artifact=True)
    adamw_variant_batch = build_adamw_variant_family_batch_scorecard(include_live_training_loop_canaries=True)
    adamw_variant_product_canary = build_adamw_variant_product_training_canary_scorecard(
        family_batch_report=adamw_variant_batch
    )
    adamw_variant_owner_hold = build_adamw_variant_owner_release_hold_scorecard(
        product_training_canary_report=adamw_variant_product_canary,
        workspace_root=REPO_ROOT,
    )
    adamw_variant_request_schema_ui = build_adamw_variant_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=adamw_variant_owner_hold,
        workspace_root=REPO_ROOT,
    )
    _write_named_artifact(
        "turbocore_adamw_variant_product_training_canary_scorecard.json",
        adamw_variant_product_canary,
    )
    _write_named_artifact(
        "turbocore_adamw_variant_owner_release_hold_scorecard.json",
        adamw_variant_owner_hold,
    )
    _write_named_artifact(
        "turbocore_adamw_variant_request_schema_ui_non_exposure_scorecard.json",
        adamw_variant_request_schema_ui,
    )
    simple_batch = build_simple_optimizer_family_batch_scorecard(workspace_root=REPO_ROOT)
    simple_product_canary = build_simple_optimizer_product_training_canary_scorecard(
        family_batch_report=simple_batch,
        workspace_root=REPO_ROOT,
    )
    simple_owner_hold = build_simple_optimizer_owner_release_hold_scorecard(
        product_training_canary_report=simple_product_canary,
        workspace_root=REPO_ROOT,
    )
    simple_schedulefree_rollout = build_simple_optimizer_schedulefree_rollout_policy_scorecard()
    simple_schedulefree_review = build_simple_optimizer_schedulefree_dispatch_integration_review_scorecard(
        rollout_policy_report=simple_schedulefree_rollout
    )
    simple_schedulefree_hold = build_simple_optimizer_schedulefree_owner_release_hold_scorecard(
        dispatch_review_report=simple_schedulefree_review
    )
    simple_quantized_review = build_simple_optimizer_quantized_dispatch_integration_review_scorecard()
    simple_quantized_hold = build_simple_optimizer_quantized_owner_approval_hold_scorecard(
        dispatch_review_report=simple_quantized_review
    )
    simple_request_schema_ui = build_simple_optimizer_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=simple_owner_hold,
        quantized_owner_approval_hold_report=simple_quantized_hold,
        schedulefree_owner_release_hold_report=simple_schedulefree_hold,
        workspace_root=REPO_ROOT,
    )
    _write_named_artifact(
        "turbocore_simple_optimizer_request_schema_ui_non_exposure_scorecard.json",
        simple_request_schema_ui,
    )
    adaptive_batch = build_adaptive_lr_state_machine_batch_scorecard(write_artifact=True)
    adaptive_replay = build_adaptive_lr_state_machine_replay_matrix_scorecard(batch_report=adaptive_batch)
    adaptive_executor = build_adaptive_lr_state_machine_replay_executor_scorecard(
        replay_matrix_report=adaptive_replay
    )
    adaptive_preconditions = build_adaptive_lr_native_state_machine_abi_preconditions_scorecard(
        replay_executor_report=adaptive_executor
    )
    adaptive_skeleton = build_adaptive_lr_native_state_machine_abi_skeleton_scorecard(
        abi_preconditions_report=adaptive_preconditions
    )
    adaptive_cpu_guard = build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard(
        abi_skeleton_report=adaptive_skeleton
    )
    adaptive_impl_stub = build_adaptive_lr_native_state_machine_implementation_stub_scorecard(
        cpu_guard_report=adaptive_cpu_guard
    )
    adaptive_cuda_contract = build_adaptive_lr_cuda_kernel_contract_plan_scorecard(
        implementation_stub_report=adaptive_impl_stub
    )
    adaptive_cuda_impl = build_adaptive_lr_cuda_kernel_implementation_scorecard(
        contract_plan_report=adaptive_cuda_contract,
        workspace_root=REPO_ROOT,
    )
    adaptive_tensor_binding = build_adaptive_lr_training_tensor_binding_canary_scorecard(
        cuda_implementation_report=adaptive_cuda_impl,
        workspace_root=REPO_ROOT,
    )
    adaptive_runtime_shadow = build_adaptive_lr_runtime_dispatch_shadow_scorecard(
        training_tensor_binding_report=adaptive_tensor_binding
    )
    adaptive_training_loop = build_adaptive_lr_training_loop_canary_scorecard(
        runtime_dispatch_shadow_report=adaptive_runtime_shadow
    )
    adaptive_e2e_shadow = build_adaptive_lr_e2e_shadow_matrix_scorecard(
        training_loop_canary_report=adaptive_training_loop
    )
    adaptive_rollout = build_adaptive_lr_canary_rollout_policy_scorecard(
        shadow_matrix_report=adaptive_e2e_shadow
    )
    adaptive_dispatch_review = build_adaptive_lr_dispatch_integration_review_scorecard(
        rollout_policy_report=adaptive_rollout
    )
    adaptive_owner_hold = build_adaptive_lr_owner_release_hold_scorecard(
        dispatch_review_report=adaptive_dispatch_review
    )
    adaptive_request_schema_ui = build_adaptive_lr_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=adaptive_owner_hold,
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    factored_custom_layout = build_factored_custom_optimizer_state_layout_scorecard(write_artifact=True)
    factored_custom_batch = build_factored_custom_optimizer_family_batch_scorecard(
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    factored_custom_owner_hold = build_factored_custom_owner_release_hold_scorecard(
        family_batch_report=factored_custom_batch,
        write_artifact=True,
    )
    factored_custom_request_schema_ui = build_factored_custom_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=factored_custom_owner_hold,
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    muon_model_shape_batch = build_muon_model_shape_aware_family_batch_scorecard(write_artifact=True)
    muon_native_scratch = build_muon_native_scratch_kernel_scorecard(
        muon_model_shape_report=muon_model_shape_batch,
        write_artifact=True,
    )
    muon_tensor_binding = build_muon_training_tensor_binding_canary_scorecard(
        native_scratch_report=muon_native_scratch,
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    muon_training_loop = build_muon_training_loop_canary_scorecard(
        training_tensor_binding_report=muon_tensor_binding,
        write_artifact=True,
    )
    muon_e2e_shadow = build_muon_e2e_shadow_matrix_scorecard(
        training_loop_canary_report=muon_training_loop,
        write_artifact=True,
    )
    muon_rollout = build_muon_canary_rollout_policy_scorecard(
        shadow_matrix_report=muon_e2e_shadow,
        write_artifact=True,
    )
    muon_dispatch_review = build_muon_dispatch_integration_review_scorecard(
        rollout_policy_report=muon_rollout,
        write_artifact=True,
    )
    muon_owner_hold = build_muon_owner_release_hold_scorecard(
        family_batch_report=muon_model_shape_batch,
        rollout_policy_report=muon_rollout,
        dispatch_review_report=muon_dispatch_review,
        write_artifact=True,
    )
    muon_request_schema_ui = build_muon_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=muon_owner_hold,
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    plugin_batch = build_plugin_optimizer_family_batch_scorecard(
        write_artifact=True,
        refresh_family_artifacts=True,
    )
    plugin_owner_hold = build_plugin_selected_family_owner_release_hold_scorecard(
        family_batch_report=plugin_batch,
        write_artifact=True,
    )
    plugin_request_schema_ui = build_plugin_selected_family_request_schema_ui_non_exposure_scorecard(
        owner_release_hold_report=plugin_owner_hold,
        workspace_root=REPO_ROOT,
        write_artifact=True,
    )
    optimizer_family_kernel_contract = build_optimizer_family_kernel_contract_scorecard(write_artifact=True)
    report = build_optimizer_family_coverage_scorecard(
        adamw_representative_route_matrix_report=adamw_route_matrix,
        exact_adamw_stream_event_chain_abi_report=exact_adamw_stream_abi,
        adamw_variant_batch_report=adamw_variant_batch,
        adamw_variant_product_training_canary_report=adamw_variant_product_canary,
        adamw_variant_owner_release_hold_report=adamw_variant_owner_hold,
        adamw_variant_request_schema_ui_report=adamw_variant_request_schema_ui,
        simple_family_batch_report=simple_batch,
        simple_product_training_canary_report=simple_product_canary,
        simple_owner_release_hold_report=simple_owner_hold,
        simple_schedulefree_rollout_policy_report=simple_schedulefree_rollout,
        simple_schedulefree_dispatch_review_report=simple_schedulefree_review,
        simple_schedulefree_owner_release_hold_report=simple_schedulefree_hold,
        simple_request_schema_ui_report=simple_request_schema_ui,
        adaptive_lr_batch_report=adaptive_batch,
        adaptive_lr_replay_matrix_report=adaptive_replay,
        adaptive_lr_replay_executor_report=adaptive_executor,
        adaptive_lr_abi_preconditions_report=adaptive_preconditions,
        adaptive_lr_abi_skeleton_report=adaptive_skeleton,
        adaptive_lr_cpu_guard_report=adaptive_cpu_guard,
        adaptive_lr_implementation_stub_report=adaptive_impl_stub,
        adaptive_lr_cuda_contract_report=adaptive_cuda_contract,
        adaptive_lr_cuda_implementation_report=adaptive_cuda_impl,
        adaptive_lr_training_tensor_binding_report=adaptive_tensor_binding,
        adaptive_lr_runtime_dispatch_shadow_report=adaptive_runtime_shadow,
        adaptive_lr_training_loop_canary_report=adaptive_training_loop,
        adaptive_lr_e2e_shadow_matrix_report=adaptive_e2e_shadow,
        adaptive_lr_canary_rollout_policy_report=adaptive_rollout,
        adaptive_lr_dispatch_review_report=adaptive_dispatch_review,
        adaptive_lr_owner_release_hold_report=adaptive_owner_hold,
        adaptive_lr_request_schema_ui_report=adaptive_request_schema_ui,
        factored_custom_state_layout_report=factored_custom_layout,
        factored_custom_family_batch_report=factored_custom_batch,
        factored_custom_owner_release_hold_report=factored_custom_owner_hold,
        factored_custom_request_schema_ui_report=factored_custom_request_schema_ui,
        muon_model_shape_aware_batch_report=muon_model_shape_batch,
        muon_native_scratch_kernel_report=muon_native_scratch,
        muon_training_tensor_binding_report=muon_tensor_binding,
        muon_training_loop_canary_report=muon_training_loop,
        muon_e2e_shadow_matrix_report=muon_e2e_shadow,
        muon_canary_rollout_policy_report=muon_rollout,
        muon_dispatch_review_report=muon_dispatch_review,
        muon_owner_release_hold_report=muon_owner_hold,
        muon_request_schema_ui_report=muon_request_schema_ui,
        plugin_family_batch_report=plugin_batch,
        plugin_selected_family_owner_release_hold_report=plugin_owner_hold,
        plugin_selected_family_request_schema_ui_report=plugin_request_schema_ui,
        optimizer_family_kernel_contract_report=optimizer_family_kernel_contract,
    )
    return _validate_coverage_report(report, artifact_mode="rebuild")


def _validate_coverage_report(report: dict[str, Any], *, artifact_mode: str) -> dict[str, Any]:
    rows = {str(row["optimizer_type"]): row for row in report["rows"]}
    simple_batch = report.get("simple_optimizer_family_batch") or {}
    simple_state_machine_reference_ready = (
        int(simple_batch.get("variant_state_machine_reference_ready_count", 0) or 0) > 0
    )
    assert report["ok"] is True, report
    assert report["evidence_ready"] is True, report
    assert report["promotion_ready"] is True, report
    assert report["ready_for_optimizer_family_coverage_review"] is True, report
    assert report["manual_review_required"] is True, report
    assert report["training_path_enabled"] is False, report
    assert report["default_behavior_changed"] is False, report
    assert report["product_exposure_allowed"] is False, report
    assert report["runtime_dispatch_allowed"] is False, report
    assert report["native_dispatch_allowed"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["backend_router_registered"] is False, report
    assert report["summary"]["missing_classification_count"] == 0, report
    assert report["summary"]["native_ready_count"] == 1, report
    assert report["summary"]["optimizer_native_kernel_inventory_source_ready_count"] == 124, report
    assert report["summary"]["optimizer_native_kernel_inventory_probe_ready_count"] == 124, report
    assert report["summary"]["optimizer_native_kernel_inventory_product_native_ready_count"] == 0, report
    assert report["summary"]["optimizer_family_kernel_contract_entrypoint_present_count"] == 1, report
    assert report["summary"]["optimizer_family_kernel_contract_ready_count"] == 10, report
    assert report["summary"]["optimizer_family_kernel_contract_validation_ok_count"] == 10, report
    assert report["summary"]["optimizer_family_kernel_contract_source_ready_count"] == 10, report
    assert report["summary"]["optimizer_family_kernel_contract_training_path_enabled_count"] == 0, report
    assert report["summary"]["optimizer_family_kernel_contract_native_dispatch_allowed_count"] == 0, report
    assert report["summary"]["optimizer_family_kernel_contract_product_native_ready_count"] == 0, report
    assert report["optimizer_family_kernel_contract"]["ok"] is True, report
    assert report["optimizer_family_kernel_contract"]["entrypoint_present"] is True, report
    assert report["optimizer_family_kernel_contract"]["optimizer_family_contract_count"] == 10, report
    assert report["optimizer_family_kernel_contract"]["training_path_enabled"] is False, report
    assert report["optimizer_family_kernel_contract"]["native_dispatch_allowed"] is False, report
    assert report["optimizer_family_kernel_contract"]["product_native_ready"] is False, report
    assert report["summary"]["adamw_representative_route_matrix_ready_count"] == 1, report
    assert report["summary"]["adamw_representative_route_matrix_route_row_ready_count"] == 3, report
    assert report["summary"]["adamw_representative_route_matrix_product_native_ready_count"] == 0, report
    assert report["summary"]["exact_adamw_stream_event_chain_ownership_abi_ready_count"] == 1, report
    assert report["summary"]["exact_adamw_stream_lifetime_ownership_bound_evidence_count"] == 1, report
    assert report["summary"]["exact_adamw_stream_event_chain_verified_count"] == 1, report
    assert report["summary"]["exact_adamw_stream_event_chain_product_native_ready_count"] == 0, report
    exact_adamw_stream = report["exact_adamw_stream_event_chain_ownership_abi"]
    assert exact_adamw_stream["present"] is True, exact_adamw_stream
    assert exact_adamw_stream["ok"] is True, exact_adamw_stream
    assert exact_adamw_stream["stream_event_chain_ownership_abi_ready"] is True, exact_adamw_stream
    assert exact_adamw_stream["stream_lifetime_ownership_bound_evidence"] is True, exact_adamw_stream
    assert exact_adamw_stream["event_chain_verified"] is True, exact_adamw_stream
    assert exact_adamw_stream["training_path_enabled"] is False, exact_adamw_stream
    assert exact_adamw_stream["training_dispatch"] is False, exact_adamw_stream
    assert exact_adamw_stream["runtime_dispatch_allowed"] is False, exact_adamw_stream
    assert exact_adamw_stream["native_dispatch_allowed"] is False, exact_adamw_stream
    assert exact_adamw_stream["product_native_ready_count"] == 0, exact_adamw_stream
    assert rows["AdamW"]["turbocore_status"] == "adamw_representative_route_matrix_ready", rows["AdamW"]
    assert rows["AdamW"]["turbocore_route"] == "rust_cuda_adamw_v0", rows["AdamW"]
    assert rows["AdamW"]["training_path_enabled"] is False, rows["AdamW"]
    assert (
        rows["AdamW"]["next_gate"]
        == "record explicit owner/release approval for exact AdamW native dispatch"
    ), rows["AdamW"]
    expected_adamw_variant_status = "adamw_variant_request_schema_ui_non_exposure_ready"
    assert rows["AdamW8bit"]["turbocore_status"] == expected_adamw_variant_status, rows["AdamW8bit"]
    assert rows["PagedAdamW"]["turbocore_status"] == expected_adamw_variant_status, rows["PagedAdamW"]
    assert rows["PagedAdamW32bit"]["turbocore_status"] == expected_adamw_variant_status, rows["PagedAdamW32bit"]
    assert rows["PagedAdamW8bit"]["turbocore_status"] == expected_adamw_variant_status, rows["PagedAdamW8bit"]
    assert rows["KahanAdamW8bit"]["turbocore_status"] == expected_adamw_variant_status, rows["KahanAdamW8bit"]
    assert rows["AdamWScheduleFree"]["turbocore_status"] == expected_adamw_variant_status, rows["AdamWScheduleFree"]
    assert (
        rows["Automagic++"]["turbocore_status"]
        == "factored_custom_request_schema_ui_non_exposure_ready"
    ), rows["Automagic++"]
    assert (
        rows["AutoProdigy"]["turbocore_status"]
        == "adaptive_lr_request_schema_ui_non_exposure_ready"
    ), rows["AutoProdigy"]
    assert rows["Muon"]["turbocore_status"] == "model_shape_aware_request_schema_ui_non_exposure_ready", rows["Muon"]
    assert (
        rows["DAdaptAdam"]["turbocore_status"]
        == "adaptive_lr_request_schema_ui_non_exposure_ready"
    ), rows["DAdaptAdam"]
    assert (
        rows["prodigyplus.ProdigyPlusScheduleFree"]["turbocore_status"]
        == "adaptive_lr_request_schema_ui_non_exposure_ready"
    ), rows["prodigyplus.ProdigyPlusScheduleFree"]
    assert (
        rows["AnimaFactoredAdamW"]["turbocore_status"]
        == "factored_custom_request_schema_ui_non_exposure_ready"
    ), rows["AnimaFactoredAdamW"]
    assert (
        rows["adafactor"]["turbocore_status"]
        == "factored_custom_request_schema_ui_non_exposure_ready"
    ), rows["adafactor"]
    expected_simple_status = "simple_formula_request_schema_ui_non_exposure_ready"
    assert rows["Lion"]["turbocore_status"] == expected_simple_status, rows["Lion"]
    assert rows["SGDNesterov"]["turbocore_status"] == expected_simple_status, rows["SGDNesterov"]
    assert rows["Lion8bit"]["turbocore_status"] == expected_simple_status, rows["Lion8bit"]
    assert rows["PagedLion8bit"]["turbocore_status"] == expected_simple_status, rows["PagedLion8bit"]
    assert rows["SGDNesterov8bit"]["turbocore_status"] == expected_simple_status, rows["SGDNesterov8bit"]
    if simple_state_machine_reference_ready:
        assert rows["RAdamScheduleFree"]["turbocore_status"] == expected_simple_status, rows["RAdamScheduleFree"]
        assert rows["SGDScheduleFree"]["turbocore_status"] == expected_simple_status, rows["SGDScheduleFree"]
    else:
        assert rows["RAdamScheduleFree"]["turbocore_status"] == "simple_update_research", rows["RAdamScheduleFree"]
        assert rows["SGDScheduleFree"]["turbocore_status"] == "simple_update_research", rows["SGDScheduleFree"]
    assert report["summary"]["simple_formula_native_dispatch_canary_ready_count"] == 0, report
    assert report["summary"]["simple_formula_native_batch_canary_ready_count"] == 2, report
    assert report["summary"]["simple_formula_representative_product_training_canary_ready_count"] == 2, report
    assert report["summary"]["simple_formula_owner_release_hold_ready_count"] == 2, report
    assert report["summary"]["simple_formula_request_schema_ui_non_exposure_ready_count"] == 7, report
    assert report["summary"]["simple_formula_request_schema_ui_optimizer_count"] == 7, report
    assert report["summary"]["simple_formula_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["simple_formula_request_schema_ui_product_native_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_state_machine_reference_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_state_machine_reference_artifact_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_state_machine_replay_matrix_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_state_machine_replay_executor_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_state_machine_replay_executor_artifact_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_native_state_machine_abi_precondition_review_ready_count"] == 0, report
    assert (
        report["summary"]["adaptive_lr_native_state_machine_abi_precondition_review_artifact_ready_count"] == 11
    ), report
    assert report["summary"]["adaptive_lr_native_state_machine_abi_skeleton_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_native_state_machine_abi_skeleton_artifact_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_native_state_machine_cpu_reference_guard_ready_count"] == 0, report
    assert (
        report["summary"]["adaptive_lr_native_state_machine_cpu_reference_guard_artifact_ready_count"] == 11
    ), report
    assert report["summary"]["adaptive_lr_native_state_machine_implementation_stub_ready_count"] == 0, report
    assert (
        report["summary"]["adaptive_lr_native_state_machine_implementation_stub_artifact_ready_count"] == 11
    ), report
    assert report["summary"]["adaptive_lr_cuda_kernel_contract_plan_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_cuda_kernel_contract_plan_artifact_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_cuda_kernel_implementation_ready_row_count"] == 0, report
    assert report["summary"]["adaptive_lr_training_tensor_binding_canary_ready_row_count"] == 0, report
    assert report["summary"]["adaptive_lr_runtime_dispatch_shadow_ready_row_count"] == 0, report
    assert report["summary"]["adaptive_lr_training_loop_canary_ready_row_count"] == 0, report
    assert report["summary"]["adaptive_lr_e2e_shadow_matrix_ready_row_count"] == 0, report
    assert report["summary"]["adaptive_lr_canary_rollout_policy_ready_row_count"] == 0, report
    assert report["summary"]["adaptive_lr_dispatch_integration_review_ready_row_count"] == 0, report
    assert report["summary"]["adaptive_lr_owner_release_hold_ready_row_count"] == 0, report
    assert report["summary"]["adaptive_lr_request_schema_ui_non_exposure_ready_row_count"] == 11, report
    assert report["summary"]["adaptive_lr_state_machine_replay_matrix_artifact_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_state_machine_replay_matrix_implementation_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_state_machine_replay_case_planned_count"] == 66, report
    assert report["summary"]["adaptive_lr_state_machine_replay_resume_case_planned_count"] == 44, report
    assert report["summary"]["adaptive_lr_state_machine_replay_executor_reference_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_state_machine_replay_executor_resume_passed_count"] == 11, report
    assert report["summary"]["adaptive_lr_native_state_machine_abi_precondition_package_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_native_kernel_precondition_review_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_native_state_machine_abi_implementation_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_native_kernel_preconditions_implementation_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_state_machine_entrypoint_contract_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_launch_plan_schema_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_state_buffer_mapping_contract_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_skeleton_state_machine_abi_implementation_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_cpu_guard_valid_launch_plan_passed_count"] == 11, report
    assert report["summary"]["adaptive_lr_cpu_guard_bad_finite_scalar_rejected_count"] == 11, report
    assert report["summary"]["adaptive_lr_cpu_guard_bad_dispatch_rejected_count"] == 11, report
    assert report["summary"]["adaptive_lr_cpu_guard_state_machine_abi_implementation_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_implementation_stub_entrypoint_contract_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_implementation_stub_state_transition_contract_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_implementation_stub_dispatch_disabled_assertion_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_implementation_stub_state_machine_abi_implementation_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_cuda_kernel_contract_runtime_canary_manifest_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_cuda_kernel_implementation_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_cuda_kernel_executed_count"] == 11, report
    assert report["summary"]["adaptive_lr_cuda_runtime_canary_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_cuda_runtime_canary_hit_count"] == 0, report
    assert report["summary"]["adaptive_lr_training_tensor_binding_canary_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_training_tensor_binding_parity_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_training_tensor_binding_family_passed_case_count"] == 2, report
    assert report["summary"]["adaptive_lr_runtime_dispatch_shadow_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_runtime_dispatch_shadow_native_shadow_call_allowed_count"] == 0, report
    assert report["summary"]["adaptive_lr_training_loop_canary_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_training_loop_family_passed_case_count"] == 2, report
    assert report["summary"]["adaptive_lr_training_loop_native_step_count"] == 2, report
    assert report["summary"]["adaptive_lr_training_loop_native_kernel_launch_count"] == 2, report
    assert report["summary"]["adaptive_lr_e2e_shadow_matrix_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_e2e_shadow_matrix_case_count"] == 22, report
    assert report["summary"]["adaptive_lr_e2e_shadow_matrix_report_only_case_count"] == 22, report
    assert report["summary"]["adaptive_lr_canary_rollout_policy_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_canary_rollout_policy_runtime_dispatch_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_canary_rollout_policy_native_dispatch_allowed_count"] == 0, report
    assert report["summary"]["adaptive_lr_canary_rollout_policy_training_path_enabled_count"] == 0, report
    assert report["summary"]["adaptive_lr_dispatch_integration_review_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_dispatch_integration_review_product_native_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_owner_release_hold_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["adaptive_lr_request_schema_ui_non_exposure_ready_count"] == 11, report
    assert report["summary"]["adaptive_lr_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["factored_custom_state_layout_reference_ready_count"] == 3, report
    assert report["summary"]["factored_custom_state_layout_ready_row_count"] == 0, report
    assert report["summary"]["factored_custom_dispatch_integration_review_ready_row_count"] == 0, report
    assert report["summary"]["factored_custom_owner_release_hold_ready_row_count"] == 0, report
    assert report["summary"]["factored_custom_request_schema_ui_non_exposure_ready_row_count"] == 3, report
    assert report["summary"]["factored_custom_optimizer_count"] == 3, report
    assert report["summary"]["factored_custom_local_live_reference_count"] == 2, report
    assert report["summary"]["factored_custom_memory_saving_candidate_count"] == 3, report
    assert report["summary"]["factored_custom_passed_case_count"] == 3, report
    assert report["summary"]["factored_custom_required_case_count"] == 3, report
    assert report["summary"]["factored_custom_family_batch_ready"] is True, report
    assert report["summary"]["factored_custom_native_scratch_kernel_ready_count"] == 3, report
    assert report["summary"]["factored_custom_training_tensor_binding_canary_ready_count"] == 3, report
    assert report["summary"]["factored_custom_runtime_dispatch_adapter_shadow_ready_count"] == 3, report
    assert report["summary"]["factored_custom_training_loop_canary_ready_count"] == 3, report
    assert report["summary"]["factored_custom_e2e_shadow_matrix_ready_count"] == 3, report
    assert report["summary"]["factored_custom_canary_rollout_policy_ready_count"] == 3, report
    assert report["summary"]["factored_custom_dispatch_integration_review_ready_count"] == 3, report
    assert report["summary"]["factored_custom_owner_release_hold_ready_count"] == 3, report
    assert report["summary"]["factored_custom_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["factored_custom_request_schema_ui_non_exposure_ready_count"] == 3, report
    assert report["summary"]["factored_custom_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["factored_custom_unsafe_claim_count"] == 0, report
    assert report["summary"]["factored_custom_product_native_ready_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_family_batch_ready"] is True, report
    assert report["summary"]["muon_model_shape_aware_param_group_abi_ready_row_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_dispatch_review_ready_row_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_native_scratch_kernel_ready_row_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_training_tensor_binding_ready_row_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_training_loop_ready_row_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_e2e_shadow_matrix_ready_row_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_owner_release_hold_ready_row_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_request_schema_ui_non_exposure_ready_row_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_optimizer_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_param_group_abi_spec_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_param_group_abi_implementation_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_param_group_resume_replay_matrix_row_count"] == 3, report
    assert (
        report["summary"]["muon_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count"] == 1
    ), report
    assert report["summary"]["muon_model_shape_aware_param_group_resume_replay_row_implementation_ready_count"] == 3, report
    assert report["summary"]["muon_model_shape_aware_native_kernel_precondition_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_runtime_dispatch_shadow_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_dispatch_integration_review_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_native_scratch_kernel_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_native_scratch_kernel_executed_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_native_scratch_product_native_ready_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_training_tensor_binding_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_training_tensor_binding_parity_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_training_tensor_binding_kernel_executed_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_training_tensor_binding_product_native_ready_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_training_loop_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_training_loop_native_step_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_training_loop_native_kernel_launch_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_training_loop_product_native_ready_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_e2e_shadow_matrix_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_e2e_shadow_matrix_case_count"] == 2, report
    assert report["summary"]["muon_model_shape_aware_e2e_shadow_matrix_report_only_case_count"] == 2, report
    assert report["summary"]["muon_model_shape_aware_e2e_shadow_matrix_product_native_ready_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_canary_rollout_policy_ready_count"] == 1, report
    assert (
        report["summary"]["muon_model_shape_aware_canary_rollout_policy_runtime_dispatch_ready_count"] == 0
    ), report
    assert (
        report["summary"]["muon_model_shape_aware_canary_rollout_policy_native_dispatch_allowed_count"] == 0
    ), report
    assert (
        report["summary"]["muon_model_shape_aware_canary_rollout_policy_training_path_enabled_count"] == 0
    ), report
    assert (
        report["summary"]["muon_model_shape_aware_canary_rollout_policy_product_native_ready_count"] == 0
    ), report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["canary_rollout_policy_ready"] is True, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["optimizer_count"] == 1, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["manual_review_required"] is True, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["canary_auto_enabled"] is False, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["canary_enabled_by_default"] is False, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["explicit_opt_in_required"] is True, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["max_canary_fraction_default"] == 0.0, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["runtime_dispatch_ready_count"] == 0, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["native_dispatch_allowed_count"] == 0, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["training_path_enabled_count"] == 0, report
    assert report["muon_model_shape_aware_canary_rollout_policy"]["product_native_ready_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_dispatch_review_gate_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_dispatch_review_product_native_ready_count"] == 0, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["dispatch_integration_review"] is True, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["optimizer_count"] == 1, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["manual_review_required"] is True, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["runtime_dispatch_ready"] is False, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["native_dispatch_allowed"] is False, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["training_path_enabled"] is False, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["request_fields_emitted"] is False, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["schema_exposure_allowed"] is False, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["ui_exposure_allowed"] is False, report
    assert report["muon_model_shape_aware_dispatch_integration_review"]["product_native_ready_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_owner_release_hold_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_request_schema_ui_non_exposure_ready_count"] == 1, report
    assert report["summary"]["muon_model_shape_aware_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["muon_model_shape_aware_product_native_ready_count"] == 0, report
    assert report["muon_model_shape_aware_owner_release_hold"]["owner_release_hold_ready"] is True, report
    assert report["muon_model_shape_aware_owner_release_hold"]["product_native_ready_count"] == 0, report
    assert report["muon_model_shape_aware_request_schema_ui_non_exposure"][
        "request_schema_ui_non_exposure_ready"
    ] is True, report
    assert report["muon_model_shape_aware_request_schema_ui_non_exposure"]["optimizer_count"] == 1, report
    assert report["muon_model_shape_aware_request_schema_ui_non_exposure"]["forbidden_token_hit_count"] == 0, report
    assert report["muon_model_shape_aware_request_schema_ui_non_exposure"]["product_native_ready_count"] == 0, report
    assert report["muon_model_shape_aware_native_scratch_kernel"]["native_scratch_kernel_ready"] is True, report
    assert report["muon_model_shape_aware_native_scratch_kernel"]["native_scratch_kernel_ready_count"] == 1, report
    assert report["muon_model_shape_aware_native_scratch_kernel"]["kernel_executed_count"] == 1, report
    assert report["muon_model_shape_aware_native_scratch_kernel"]["product_native_ready_count"] == 0, report
    assert report["muon_model_shape_aware_training_tensor_binding_canary"][
        "training_tensor_binding_canary_ready"
    ] is True, report
    assert report["muon_model_shape_aware_training_tensor_binding_canary"][
        "training_tensor_binding_parity_ready"
    ] is True, report
    assert report["muon_model_shape_aware_training_tensor_binding_canary"]["kernel_executed_count"] == 1, report
    assert report["muon_model_shape_aware_training_tensor_binding_canary"]["product_native_ready_count"] == 0, report
    assert report["muon_model_shape_aware_training_loop_canary"]["training_loop_canary_ready"] is True, report
    assert report["muon_model_shape_aware_training_loop_canary"]["training_loop_canary_ready_count"] == 1, report
    assert report["muon_model_shape_aware_training_loop_canary"]["native_step_count"] == 1, report
    assert report["muon_model_shape_aware_training_loop_canary"]["native_kernel_launch_count"] == 1, report
    assert report["muon_model_shape_aware_training_loop_canary"]["product_native_ready_count"] == 0, report
    assert report["muon_model_shape_aware_e2e_shadow_matrix"]["e2e_shadow_matrix_ready"] is True, report
    assert report["muon_model_shape_aware_e2e_shadow_matrix"]["case_count"] == 2, report
    assert report["muon_model_shape_aware_e2e_shadow_matrix"]["report_only_case_count"] == 2, report
    assert report["muon_model_shape_aware_e2e_shadow_matrix"]["product_native_ready_count"] == 0, report
    assert report["summary"]["simple_formula_variant_layout_spec_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_native_kernel_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_native_abi_spec_ready_count"] == 5, report
    assert report["summary"]["simple_formula_variant_quantized_formula_parity_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_native_scratch_kernel_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_runtime_canary_manifest_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_training_loop_canary_manifest_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_training_loop_canary_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_e2e_no_regression_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_product_state_sync_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_rollout_policy_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_dispatch_integration_review_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_owner_approval_hold_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_product_state_sync_review_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_product_optimizer_state_sync_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_quantized_optimizer_state_sync_state_tensor_count"] == 6, report
    assert report["summary"]["simple_formula_variant_quantized_optimizer_state_sync_parameter_tensor_count"] == 3, report
    assert report["summary"]["simple_formula_variant_schedule_free_native_canary_ready_count"] == 2, report
    assert report["summary"]["simple_formula_variant_schedule_free_rollout_policy_ready_count"] == 2, report
    assert report["summary"]["simple_formula_variant_schedule_free_dispatch_integration_review_ready_count"] == 2, report
    assert report["summary"]["simple_formula_variant_schedule_free_owner_release_hold_ready_count"] == 2, report
    assert report["summary"]["simple_formula_variant_quantized_native_canary_pending_count"] == 0, report
    assert report["simple_optimizer_family_batch"]["variant_quantized_owner_approval_hold_ready_count"] == 3, report
    assert report["simple_optimizer_product_training_canary"][
        "representative_product_training_canary_ready_count"
    ] == 2, report
    assert report["simple_optimizer_owner_release_hold"]["owner_release_hold_ready"] is True, report
    assert report["simple_optimizer_owner_release_hold"]["optimizer_count"] == 2, report
    assert report["simple_optimizer_owner_release_hold"]["product_native_ready_count"] == 0, report
    assert report["simple_optimizer_schedulefree_rollout_policy"]["canary_rollout_policy_ready"] is True, report
    assert report["simple_optimizer_schedulefree_rollout_policy"]["optimizer_count"] == 2, report
    assert report["simple_optimizer_schedulefree_rollout_policy"]["product_native_ready_count"] == 0, report
    assert report["simple_optimizer_schedulefree_dispatch_review"]["review_gate_ready"] is True, report
    assert report["simple_optimizer_schedulefree_dispatch_review"]["optimizer_count"] == 2, report
    assert report["simple_optimizer_schedulefree_dispatch_review"]["product_native_ready_count"] == 0, report
    assert report["simple_optimizer_schedulefree_owner_release_hold"]["owner_release_hold_ready"] is True, report
    assert report["simple_optimizer_schedulefree_owner_release_hold"]["optimizer_count"] == 2, report
    assert report["simple_optimizer_schedulefree_owner_release_hold"]["product_native_ready_count"] == 0, report
    assert report["simple_optimizer_request_schema_ui_non_exposure"][
        "request_schema_ui_non_exposure_ready"
    ] is True, report
    assert report["simple_optimizer_request_schema_ui_non_exposure"]["optimizer_count"] == 7, report
    assert report["simple_optimizer_request_schema_ui_non_exposure"]["forbidden_token_hit_count"] == 0, report
    assert report["simple_optimizer_request_schema_ui_non_exposure"]["product_native_ready_count"] == 0, report
    assert report["simple_optimizer_request_schema_ui_non_exposure"]["request_fields_emitted"] is False, report
    assert report["simple_optimizer_request_schema_ui_non_exposure"]["schema_exposure_allowed"] is False, report
    assert report["simple_optimizer_request_schema_ui_non_exposure"]["ui_exposure_allowed"] is False, report
    assert report["summary"]["simple_formula_variant_formula_parity_matrix_artifact_ready_count"] == 5, report
    assert report["summary"]["simple_formula_variant_formula_parity_matrix_implementation_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_resume_parity_matrix_artifact_ready_count"] == 5, report
    assert report["summary"]["simple_formula_variant_resume_parity_matrix_implementation_ready_count"] == 5, report
    assert report["summary"]["simple_formula_variant_quantized_resume_parity_ready_count"] == 3, report
    assert report["summary"]["simple_formula_variant_schedule_free_resume_parity_ready_count"] == 2, report
    if simple_state_machine_reference_ready:
        assert report["summary"]["simple_formula_variant_state_machine_reference_ready_count"] == 2, report
    assert report["adaptive_lr_state_machine_replay_matrix"]["state_machine_replay_matrix_ready"] is True, report
    assert report["adaptive_lr_state_machine_replay_matrix"][
        "state_machine_replay_matrix_artifact_ready_count"
    ] == 11, report
    assert report["adaptive_lr_state_machine_replay_matrix"][
        "state_machine_replay_matrix_implementation_ready_count"
    ] == 0, report
    assert report["adaptive_lr_state_machine_replay_matrix"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_state_machine_replay_executor"]["state_machine_replay_executor_ready"] is True, report
    assert report["adaptive_lr_state_machine_replay_executor"]["reference_replay_executor_ready_count"] == 11, report
    assert report["adaptive_lr_state_machine_replay_executor"]["resume_next_step_parity_passed_count"] == 11, report
    assert report["adaptive_lr_state_machine_replay_executor"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_native_state_machine_abi_preconditions"][
        "native_state_machine_abi_preconditions_ready"
    ] is True, report
    assert report["adaptive_lr_native_state_machine_abi_preconditions"][
        "native_state_machine_abi_precondition_review_ready_count"
    ] == 11, report
    assert report["adaptive_lr_native_state_machine_abi_preconditions"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_native_state_machine_abi_skeleton"][
        "native_state_machine_abi_skeleton_ready"
    ] is True, report
    assert report["adaptive_lr_native_state_machine_abi_skeleton"][
        "native_state_machine_abi_skeleton_ready_count"
    ] == 11, report
    assert report["adaptive_lr_native_state_machine_abi_skeleton"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_native_state_machine_cpu_reference_guard"][
        "native_state_machine_cpu_reference_guard_ready"
    ] is True, report
    assert report["adaptive_lr_native_state_machine_cpu_reference_guard"][
        "cpu_reference_guard_ready_count"
    ] == 11, report
    assert report["adaptive_lr_native_state_machine_cpu_reference_guard"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_native_state_machine_implementation_stub"][
        "native_state_machine_implementation_stub_ready"
    ] is True, report
    assert report["adaptive_lr_native_state_machine_implementation_stub"][
        "implementation_stub_ready_count"
    ] == 11, report
    assert report["adaptive_lr_native_state_machine_implementation_stub"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_cuda_kernel_contract_plan"]["cuda_kernel_contract_plan_ready"] is True, report
    assert report["adaptive_lr_cuda_kernel_contract_plan"]["cuda_kernel_contract_plan_ready_count"] == 11, report
    assert report["adaptive_lr_cuda_kernel_contract_plan"]["runtime_canary_manifest_ready_count"] == 11, report
    assert report["adaptive_lr_cuda_kernel_contract_plan"]["cuda_kernel_implementation_ready_count"] == 0, report
    assert report["adaptive_lr_cuda_kernel_contract_plan"]["runtime_canary_ready_count"] == 0, report
    assert report["adaptive_lr_cuda_kernel_contract_plan"]["runtime_canary_hit_count"] == 0, report
    assert report["adaptive_lr_cuda_kernel_contract_plan"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_cuda_kernel_implementation"]["cuda_kernel_implementation_ready"] is True, report
    assert report["adaptive_lr_cuda_kernel_implementation"]["cuda_kernel_implementation_ready_count"] == 11, report
    assert report["adaptive_lr_cuda_kernel_implementation"]["kernel_executed_count"] == 11, report
    assert report["adaptive_lr_cuda_kernel_implementation"]["runtime_canary_ready_count"] == 0, report
    assert report["adaptive_lr_cuda_kernel_implementation"]["runtime_canary_hit_count"] == 0, report
    assert report["adaptive_lr_cuda_kernel_implementation"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_training_tensor_binding_canary"]["training_tensor_binding_canary_ready"] is True, report
    assert report["adaptive_lr_training_tensor_binding_canary"]["training_tensor_binding_canary_ready_count"] == 11, report
    assert report["adaptive_lr_training_tensor_binding_canary"]["training_tensor_binding_parity_ready_count"] == 11, report
    assert report["adaptive_lr_training_tensor_binding_canary"]["family_passed_case_count"] == 2, report
    assert report["adaptive_lr_training_tensor_binding_canary"]["runtime_canary_ready_count"] == 0, report
    assert report["adaptive_lr_training_tensor_binding_canary"]["runtime_canary_hit_count"] == 0, report
    assert report["adaptive_lr_training_tensor_binding_canary"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_runtime_dispatch_shadow"]["runtime_dispatch_shadow_ready"] is True, report
    assert report["adaptive_lr_runtime_dispatch_shadow"]["runtime_dispatch_shadow_ready_count"] == 11, report
    assert report["adaptive_lr_runtime_dispatch_shadow"]["fallback_backend_authoritative"] is True, report
    assert report["adaptive_lr_runtime_dispatch_shadow"]["native_shadow_call_allowed"] is False, report
    assert report["adaptive_lr_runtime_dispatch_shadow"]["native_shadow_call_allowed_count"] == 0, report
    assert report["adaptive_lr_runtime_dispatch_shadow"]["runtime_dispatch_ready_count"] == 0, report
    assert report["adaptive_lr_runtime_dispatch_shadow"]["native_dispatch_allowed_count"] == 0, report
    assert report["adaptive_lr_runtime_dispatch_shadow"]["training_path_enabled_count"] == 0, report
    assert report["adaptive_lr_runtime_dispatch_shadow"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_training_loop_canary"]["training_loop_canary_ready"] is True, report
    assert report["adaptive_lr_training_loop_canary"]["training_loop_canary_ready_count"] == 11, report
    assert report["adaptive_lr_training_loop_canary"]["family_passed_case_count"] == 2, report
    assert report["adaptive_lr_training_loop_canary"]["native_step_count"] == 2, report
    assert report["adaptive_lr_training_loop_canary"]["native_kernel_launch_count"] == 2, report
    assert report["adaptive_lr_training_loop_canary"]["runtime_dispatch_ready_count"] == 0, report
    assert report["adaptive_lr_training_loop_canary"]["native_dispatch_allowed_count"] == 0, report
    assert report["adaptive_lr_training_loop_canary"]["training_path_enabled_count"] == 0, report
    assert report["adaptive_lr_training_loop_canary"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["e2e_shadow_matrix_ready"] is True, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["e2e_shadow_matrix_passed"] is False, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["live_shadow_matrix_executed"] is False, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["e2e_shadow_matrix_ready_count"] == 11, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["case_count"] == 22, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["report_only_case_count"] == 22, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["failed_case_count"] == 0, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["runtime_dispatch_ready_count"] == 0, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["native_dispatch_allowed_count"] == 0, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["training_path_enabled_count"] == 0, report
    assert report["adaptive_lr_e2e_shadow_matrix"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_canary_rollout_policy"]["canary_rollout_policy_ready"] is True, report
    assert report["adaptive_lr_canary_rollout_policy"]["manual_review_required"] is True, report
    assert report["adaptive_lr_canary_rollout_policy"]["canary_auto_enabled"] is False, report
    assert report["adaptive_lr_canary_rollout_policy"]["canary_rollout_policy_ready_count"] == 11, report
    assert report["adaptive_lr_canary_rollout_policy"]["runtime_dispatch_ready_count"] == 0, report
    assert report["adaptive_lr_canary_rollout_policy"]["native_dispatch_allowed_count"] == 0, report
    assert report["adaptive_lr_canary_rollout_policy"]["training_path_enabled_count"] == 0, report
    assert report["adaptive_lr_canary_rollout_policy"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_dispatch_integration_review"]["review_gate_ready"] is True, report
    assert report["adaptive_lr_dispatch_integration_review"]["dispatch_integration_review"] is True, report
    assert report["adaptive_lr_dispatch_integration_review"]["manual_review_required"] is True, report
    assert report["adaptive_lr_dispatch_integration_review"]["optimizer_count"] == 11, report
    assert report["adaptive_lr_dispatch_integration_review"]["runtime_dispatch_ready"] is False, report
    assert report["adaptive_lr_dispatch_integration_review"]["native_dispatch_allowed"] is False, report
    assert report["adaptive_lr_dispatch_integration_review"]["training_path_enabled"] is False, report
    assert report["adaptive_lr_dispatch_integration_review"]["request_fields_emitted"] is False, report
    assert report["adaptive_lr_dispatch_integration_review"]["schema_exposure_allowed"] is False, report
    assert report["adaptive_lr_dispatch_integration_review"]["ui_exposure_allowed"] is False, report
    assert report["adaptive_lr_dispatch_integration_review"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_owner_release_hold"]["owner_release_hold_ready"] is True, report
    assert report["adaptive_lr_owner_release_hold"]["dispatch_integration_review"] is True, report
    assert report["adaptive_lr_owner_release_hold"]["owner_approval_recorded"] is False, report
    assert report["adaptive_lr_owner_release_hold"]["release_approval_recorded"] is False, report
    assert report["adaptive_lr_owner_release_hold"]["optimizer_count"] == 11, report
    assert report["adaptive_lr_owner_release_hold"]["runtime_dispatch_ready"] is False, report
    assert report["adaptive_lr_owner_release_hold"]["native_dispatch_allowed"] is False, report
    assert report["adaptive_lr_owner_release_hold"]["training_path_enabled"] is False, report
    assert report["adaptive_lr_owner_release_hold"]["request_fields_emitted"] is False, report
    assert report["adaptive_lr_owner_release_hold"]["schema_exposure_allowed"] is False, report
    assert report["adaptive_lr_owner_release_hold"]["ui_exposure_allowed"] is False, report
    assert report["adaptive_lr_owner_release_hold"]["product_native_ready_count"] == 0, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["request_schema_ui_non_exposure_ready"] is True, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["owner_release_hold_ready"] is True, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["optimizer_count"] == 11, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["present_boundary_path_count"] > 0, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["scanned_file_count"] > 0, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["forbidden_token_hit_count"] == 0, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["runtime_dispatch_ready"] is False, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["native_dispatch_allowed"] is False, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["training_path_enabled"] is False, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["request_fields_emitted"] is False, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["schema_exposure_allowed"] is False, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["ui_exposure_allowed"] is False, report
    assert report["adaptive_lr_request_schema_ui_non_exposure"]["product_native_ready_count"] == 0, report
    assert report["factored_custom_optimizer_state_layout"]["state_layout_reference_ready"] is True, report
    assert report["factored_custom_optimizer_state_layout"]["optimizer_count"] == 3, report
    assert report["factored_custom_optimizer_state_layout"]["local_live_reference_count"] == 2, report
    assert report["factored_custom_optimizer_state_layout"]["memory_saving_candidate_count"] == 3, report
    assert report["factored_custom_optimizer_state_layout"]["training_path_enabled"] is False, report
    assert report["factored_custom_optimizer_state_layout"]["native_dispatch_allowed"] is False, report
    assert report["factored_custom_optimizer_state_layout"]["runtime_dispatch_ready"] is False, report
    assert report["factored_custom_optimizer_family_batch"]["factored_custom_family_batch_ready"] is True, report
    assert report["factored_custom_optimizer_family_batch"]["optimizer_count"] == 3, report
    assert report["factored_custom_optimizer_family_batch"]["dispatch_integration_review_ready_count"] == 3, report
    assert report["factored_custom_optimizer_family_batch"]["product_native_ready_count"] == 0, report
    assert report["factored_custom_optimizer_family_batch"]["unsafe_claim_count"] == 0, report
    assert report["factored_custom_optimizer_family_batch"]["training_path_enabled"] is False, report
    assert report["factored_custom_optimizer_family_batch"]["native_dispatch_allowed"] is False, report
    assert report["factored_custom_optimizer_family_batch"]["runtime_dispatch_ready"] is False, report
    assert report["factored_custom_owner_release_hold"]["owner_release_hold_ready"] is True, report
    assert report["factored_custom_owner_release_hold"]["optimizer_count"] == 3, report
    assert report["factored_custom_owner_release_hold"]["owner_approval_recorded"] is False, report
    assert report["factored_custom_owner_release_hold"]["release_approval_recorded"] is False, report
    assert report["factored_custom_owner_release_hold"]["product_native_ready_count"] == 0, report
    assert report["factored_custom_owner_release_hold"]["training_path_enabled"] is False, report
    assert report["factored_custom_owner_release_hold"]["native_dispatch_allowed"] is False, report
    assert report["factored_custom_owner_release_hold"]["runtime_dispatch_ready"] is False, report
    assert (
        report["factored_custom_request_schema_ui_non_exposure"]["request_schema_ui_non_exposure_ready"]
        is True
    ), report
    assert report["factored_custom_request_schema_ui_non_exposure"]["optimizer_count"] == 3, report
    assert report["factored_custom_request_schema_ui_non_exposure"]["present_boundary_path_count"] > 0, report
    assert report["factored_custom_request_schema_ui_non_exposure"]["scanned_file_count"] > 0, report
    assert report["factored_custom_request_schema_ui_non_exposure"]["forbidden_token_hit_count"] == 0, report
    assert report["factored_custom_request_schema_ui_non_exposure"]["request_fields_emitted"] is False, report
    assert report["factored_custom_request_schema_ui_non_exposure"]["schema_exposure_allowed"] is False, report
    assert report["factored_custom_request_schema_ui_non_exposure"]["ui_exposure_allowed"] is False, report
    assert report["factored_custom_request_schema_ui_non_exposure"]["training_path_enabled"] is False, report
    assert report["factored_custom_request_schema_ui_non_exposure"]["native_dispatch_allowed"] is False, report
    assert report["factored_custom_request_schema_ui_non_exposure"]["runtime_dispatch_ready"] is False, report
    assert report["muon_model_shape_aware_family_batch"]["muon_model_shape_aware_family_batch_ready"] is True, report
    assert report["muon_model_shape_aware_family_batch"]["optimizer_count"] == 1, report
    assert report["muon_model_shape_aware_family_batch"]["param_group_abi_spec_ready_count"] == 1, report
    assert report["muon_model_shape_aware_family_batch"]["param_group_abi_implementation_ready_count"] == 1, report
    assert report["muon_model_shape_aware_family_batch"]["param_group_resume_replay_matrix_row_count"] == 3, report
    assert report["muon_model_shape_aware_family_batch"]["native_kernel_precondition_ready_count"] == 1, report
    assert report["muon_model_shape_aware_family_batch"]["runtime_dispatch_shadow_ready_count"] == 1, report
    assert report["muon_model_shape_aware_family_batch"]["dispatch_integration_review_ready_count"] == 1, report
    assert report["muon_model_shape_aware_family_batch"]["training_path_enabled"] is False, report
    assert report["muon_model_shape_aware_family_batch"]["native_dispatch_allowed"] is False, report
    assert report["muon_model_shape_aware_family_batch"]["runtime_dispatch_ready"] is False, report
    assert report["muon_model_shape_aware_family_batch"]["native_kernel_ready"] is False, report
    assert report["muon_model_shape_aware_family_batch"]["product_native_ready_count"] == 0, report
    assert report["summary"]["adamw_variant_family_batch_pending_count"] == 0, report
    assert report["summary"]["adamw_variant_native_canary_ready_count"] == 0, report
    assert report["summary"]["adamw_variant_representative_product_training_canary_ready_count"] == 6, report
    assert (
        report["summary"]["adamw_variant_representative_product_training_canary_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["adamw_variant_owner_release_hold_ready_count"] == 6, report
    assert report["summary"]["adamw_variant_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["adamw_variant_request_schema_ui_non_exposure_ready_count"] == 6, report
    assert report["summary"]["adamw_variant_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["adamw_variant_request_schema_ui_product_native_ready_count"] == 0, report
    assert report["summary"]["adamw_variant_native_canary_stage_evidence_ready_count"] == 6, report
    assert report["summary"]["adamw_variant_product_native_ready_count"] == 0, report
    assert report["summary"]["adamw_variant_native_abi_ready_count"] == 1, report
    assert report["summary"]["adamw_variant_scratch_formula_canary_ready_count"] == 1, report
    assert report["summary"]["adamw_variant_native_scratch_kernel_ready_count"] == 1, report
    assert report["summary"]["adamw_variant_runtime_canary_manifest_ready_count"] == 6, report
    assert report["summary"]["adamw_variant_training_loop_canary_manifest_ready_count"] == 6, report
    assert report["summary"]["adamw_variant_state_reference_ready_count"] == 6, report
    assert report["summary"]["adamw_variant_native_canary_manifest_ready_count"] == 6, report
    assert report["summary"]["adamw_variant_training_loop_canary_ready_count"] == 6, report
    assert report["summary"]["adamw_variant_schedule_free_native_abi_ready_count"] == 1, report
    assert report["summary"]["adamw_variant_schedule_free_scratch_formula_canary_ready_count"] == 1, report
    assert report["summary"]["adamw_variant_schedule_free_native_scratch_kernel_ready_count"] == 1, report
    assert report["summary"]["adamw_variant_e2e_shadow_matrix_ready"] is True, report
    assert report["summary"]["adamw_variant_canary_rollout_policy_ready"] is True, report
    assert report["summary"]["adamw_variant_dispatch_integration_review_ready"] is True, report
    assert report["adamw_variant_family_batch"]["native_ready_count"] == 6, report
    assert report["adamw_variant_family_batch"]["native_canary_stage_evidence_ready_count"] == 6, report
    assert report["adamw_variant_family_batch"]["product_native_ready_count"] == 0, report
    assert report["adamw_variant_family_batch"]["exact_adamw_included"] is False, report
    assert report["adamw_variant_family_batch"]["e2e_shadow_matrix_ready"] is True, report
    assert report["adamw_variant_family_batch"]["canary_rollout_policy_ready"] is True, report
    assert report["adamw_variant_family_batch"]["dispatch_integration_review_ready"] is True, report
    assert report["adamw_variant_family_batch"]["state_reference_ready_count"] == 6, report
    assert report["adamw_variant_family_batch"]["native_canary_manifest_ready_count"] == 6, report
    assert report["adamw_variant_family_batch"]["training_loop_canary_ready_count"] == 6, report
    assert report["adamw_variant_family_batch"]["schedule_free_native_abi_ready_count"] == 1, report
    assert report["adamw_variant_family_batch"]["schedule_free_scratch_formula_canary_ready_count"] == 1, report
    assert report["adamw_variant_family_batch"]["schedule_free_native_scratch_kernel_ready_count"] == 1, report
    assert report["adamw_variant_product_training_canary"][
        "representative_product_training_canary_ready"
    ] is True, report
    assert report["adamw_variant_product_training_canary"][
        "representative_product_training_canary_ready_count"
    ] == 6, report
    assert report["adamw_variant_product_training_canary"]["product_native_ready_count"] == 0, report
    assert report["adamw_variant_product_training_canary"]["training_path_enabled"] is False, report
    assert report["adamw_variant_product_training_canary"]["native_dispatch_allowed"] is False, report
    assert report["adamw_variant_product_training_canary"]["runtime_dispatch_ready"] is False, report
    assert report["adamw_variant_owner_release_hold"]["owner_release_hold_ready"] is True, report
    assert report["adamw_variant_owner_release_hold"]["representative_product_training_canary_ready"] is True, report
    assert report["adamw_variant_owner_release_hold"]["owner_approval_recorded"] is False, report
    assert report["adamw_variant_owner_release_hold"]["release_approval_recorded"] is False, report
    assert report["adamw_variant_owner_release_hold"]["optimizer_count"] == 6, report
    assert report["adamw_variant_owner_release_hold"]["product_native_ready_count"] == 0, report
    assert report["adamw_variant_owner_release_hold"]["training_path_enabled"] is False, report
    assert report["adamw_variant_owner_release_hold"]["native_dispatch_allowed"] is False, report
    assert report["adamw_variant_owner_release_hold"]["runtime_dispatch_ready"] is False, report
    assert report["adamw_variant_request_schema_ui_non_exposure"][
        "request_schema_ui_non_exposure_ready"
    ] is True, report
    assert report["adamw_variant_request_schema_ui_non_exposure"]["owner_release_hold_ready"] is True, report
    assert report["adamw_variant_request_schema_ui_non_exposure"]["optimizer_count"] == 6, report
    assert report["adamw_variant_request_schema_ui_non_exposure"]["forbidden_token_hit_count"] == 0, report
    assert report["adamw_variant_request_schema_ui_non_exposure"]["product_native_ready_count"] == 0, report
    assert report["adamw_variant_request_schema_ui_non_exposure"]["training_path_enabled"] is False, report
    assert report["adamw_variant_request_schema_ui_non_exposure"]["native_dispatch_allowed"] is False, report
    assert report["adamw_variant_request_schema_ui_non_exposure"]["runtime_dispatch_ready"] is False, report
    assert report["adaptive_lr_state_machine_batch"]["native_ready_count"] == 0, report
    assert report["summary"]["plugin_selector_classification_ready"] is True, report
    assert report["summary"]["plugin_selector_missing_classification_count"] == 0, report
    assert report["summary"]["plugin_selector_missing_resume_count"] == 0, report
    assert report["summary"]["plugin_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_optimizer_gate_ready_count"] == 10, report
    assert report["summary"]["plugin_selected_optimizer_gate_pending_count"] == 0, report
    assert report["summary"]["plugin_selected_adamlike_native_canary_ready_count"] >= 25, report
    assert report["summary"]["plugin_selected_adamlike_e2e_shadow_matrix_ready"] is True, report
    assert report["summary"]["plugin_selected_adamlike_canary_rollout_policy_ready"] is True, report
    assert report["summary"]["plugin_selected_adamlike_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_adamlike_owner_release_hold_optimizer_count"] == 25, report
    assert report["summary"]["plugin_selected_adamlike_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_adamlike_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_adamlike_request_schema_ui_optimizer_count"] == 25, report
    assert report["summary"]["plugin_selected_adamlike_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["plugin_selected_adamlike_request_schema_ui_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_schedulefree_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_schedulefree_e2e_shadow_case_count"] == 6, report
    assert report["summary"]["plugin_selected_schedulefree_native_canary_ready_count"] == 3, report
    assert report["summary"]["plugin_selected_schedulefree_dispatch_review_gate_ready"] is True, report
    assert report["summary"]["plugin_selected_schedulefree_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_schedulefree_owner_release_hold_optimizer_count"] == 3, report
    assert report["summary"]["plugin_selected_schedulefree_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_schedulefree_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_schedulefree_request_schema_ui_optimizer_count"] == 3, report
    assert report["summary"]["plugin_selected_schedulefree_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["plugin_selected_schedulefree_request_schema_ui_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_adaptivelr_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_adaptivelr_reference_ready_count"] == 6, report
    assert report["summary"]["plugin_selected_adaptivelr_state_machine_abi_spec_ready_count"] == 6, report
    assert report["summary"]["plugin_selected_adaptivelr_state_machine_abi_implementation_ready_count"] == 6, report
    assert report["summary"]["plugin_selected_adaptivelr_native_kernel_preconditions_spec_ready_count"] == 6, report
    assert report["summary"][
        "plugin_selected_adaptivelr_native_kernel_preconditions_implementation_ready_count"
    ] == 6, report
    assert report["summary"]["plugin_selected_adaptivelr_state_machine_replay_matrix_artifact_ready_count"] == 6, report
    assert report["summary"]["plugin_selected_adaptivelr_state_machine_replay_matrix_implementation_ready_count"] == 6, report
    assert report["summary"]["plugin_selected_adaptivelr_state_machine_replay_case_planned_count"] == 36, report
    assert report["summary"][
        "plugin_selected_adaptivelr_state_machine_replay_case_implementation_ready_count"
    ] == 36, report
    assert report["summary"]["plugin_selected_adaptivelr_state_machine_replay_resume_case_planned_count"] == 24, report
    assert report["summary"][
        "plugin_selected_adaptivelr_state_machine_replay_resume_case_implementation_ready_count"
    ] == 24, report
    assert report["summary"]["plugin_selected_adaptivelr_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_adaptivelr_owner_release_hold_optimizer_count"] == 6, report
    assert report["summary"]["plugin_selected_adaptivelr_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_adaptivelr_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_adaptivelr_request_schema_ui_optimizer_count"] == 6, report
    assert report["summary"]["plugin_selected_adaptivelr_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["plugin_selected_adaptivelr_request_schema_ui_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_simple_formula_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_simple_formula_optimizer_count"] == 18, report
    assert report["summary"]["plugin_selected_simple_formula_reference_canary_ready_count"] == 1, report
    assert report["summary"]["plugin_selected_simple_formula_native_canary_ready_count"] == 18, report
    assert report["summary"]["plugin_selected_simple_formula_e2e_shadow_matrix_ready"] is True, report
    assert report["summary"]["plugin_selected_simple_formula_e2e_shadow_case_count"] == 18, report
    assert report["summary"]["plugin_selected_simple_formula_canary_rollout_policy_ready"] is True, report
    assert report["summary"]["plugin_selected_simple_formula_canary_rollout_policy_ready_count"] == 18, report
    assert report["summary"]["plugin_selected_simple_formula_dispatch_review_gate_ready"] is True, report
    assert report["summary"]["plugin_selected_simple_formula_dispatch_review_ready_count"] == 18, report
    assert report["summary"]["plugin_selected_simple_formula_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_simple_formula_owner_release_hold_optimizer_count"] == 18, report
    assert report["summary"]["plugin_selected_simple_formula_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_simple_formula_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_simple_formula_request_schema_ui_optimizer_count"] == 18, report
    assert report["summary"]["plugin_selected_simple_formula_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["plugin_selected_simple_formula_request_schema_ui_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_closure_second_order_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_closure_second_order_optimizer_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_higher_order_abi_required_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_training_loop_abi_spec_ready_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_training_loop_abi_implementation_ready_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_resume_parity_matrix_spec_ready_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_resume_parity_matrix_implementation_ready_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_closure_replay_case_planned_count"] == 22, report
    assert report["summary"]["plugin_selected_closure_second_order_create_graph_hvp_lifetime_case_planned_count"] == 18, report
    assert report["summary"]["plugin_selected_closure_second_order_closure_resume_replay_artifact_ready_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_closure_resume_replay_artifact_row_count"] == 20, report
    assert report["summary"]["plugin_selected_closure_second_order_closure_resume_replay_artifact_implementation_ready_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_closure_resume_replay_row_implementation_ready_count"] == 20, report
    assert report["summary"]["plugin_selected_closure_second_order_native_kernel_precondition_plan_ready_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_native_kernel_preconditions_implementation_ready_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_closure_second_order_owner_release_hold_optimizer_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_closure_second_order_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_closure_second_order_request_schema_ui_optimizer_count"] == 5, report
    assert report["summary"]["plugin_selected_closure_second_order_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["plugin_selected_closure_second_order_request_schema_ui_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_custom_formula_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_custom_formula_optimizer_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_parity_required_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_backlog_ready_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_evidence_artifact_planned_count"] == 235, report
    assert report["summary"]["plugin_selected_custom_formula_evidence_status_pending_total"] == 0, report
    assert report["summary"]["plugin_selected_custom_formula_formula_spec_artifact_ready_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_formula_spec_artifact_pending_count"] == 0, report
    assert report["summary"]["plugin_selected_custom_formula_state_inventory_skeleton_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_state_inventory_artifact_ready_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_state_inventory_artifact_pending_count"] == 0, report
    assert report["summary"]["plugin_selected_custom_formula_quality_guard_matrix_artifact_ready_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_quality_guard_matrix_artifact_pending_count"] == 0, report
    assert report["summary"]["plugin_selected_custom_formula_quality_guard_matrix_case_planned_count"] == 329, report
    assert report["summary"]["plugin_selected_custom_formula_formula_parity_matrix_artifact_planned_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_formula_parity_matrix_implementation_ready_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_formula_parity_case_planned_count"] == 282, report
    assert report["summary"]["plugin_selected_custom_formula_resume_parity_matrix_artifact_planned_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_resume_parity_matrix_implementation_ready_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_resume_parity_case_planned_count"] == 188, report
    assert report["summary"]["plugin_selected_custom_formula_execution_matrix_ready"] is True, report
    assert report["summary"]["plugin_selected_custom_formula_step_execution_ready_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_resume_next_step_replay_ready_count"] == 47, report
    assert report["summary"]["plugin_selected_custom_formula_execution_failed_count"] == 0, report
    assert report["summary"]["plugin_selected_custom_formula_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_custom_formula_owner_release_hold_optimizer_count"] == 47, report
    assert (
        report["summary"]["plugin_selected_custom_formula_owner_release_hold_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["plugin_selected_custom_formula_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_custom_formula_request_schema_ui_optimizer_count"] == 47, report
    assert (
        report["summary"]["plugin_selected_custom_formula_request_schema_ui_forbidden_token_hit_count"] == 0
    ), report
    assert (
        report["summary"]["plugin_selected_custom_formula_request_schema_ui_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["plugin_selected_factored_memory_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_factored_memory_optimizer_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_observed_layout_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_native_layout_abi_ready_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_quality_matrix_ready_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_native_kernel_entry_condition_ready_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_formula_tensor_binding_matrix_artifact_ready_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_formula_tensor_binding_matrix_implementation_ready_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_formula_step_execution_ready_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_resume_next_step_replay_ready_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_tensor_binding_ready_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_dispatch_review_gate_ready"] is True, report
    assert report["summary"]["plugin_selected_factored_memory_dispatch_review_ready_count"] == 8, report
    assert report["summary"]["plugin_selected_factored_memory_formula_parity_case_planned_count"] == 40, report
    assert report["summary"]["plugin_selected_factored_memory_tensor_binding_case_planned_count"] == 24, report
    assert report["summary"]["plugin_selected_factored_memory_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_factored_memory_owner_release_hold_optimizer_count"] == 8, report
    assert (
        report["summary"]["plugin_selected_factored_memory_owner_release_hold_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["plugin_selected_factored_memory_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_factored_memory_request_schema_ui_optimizer_count"] == 8, report
    assert (
        report["summary"]["plugin_selected_factored_memory_request_schema_ui_forbidden_token_hit_count"] == 0
    ), report
    assert (
        report["summary"]["plugin_selected_factored_memory_request_schema_ui_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["plugin_selected_fused_backward_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_fused_backward_optimizer_count"] == 2, report
    assert report["summary"]["plugin_selected_fused_backward_gradient_ownership_abi_required_count"] == 2, report
    assert report["summary"]["plugin_selected_fused_backward_per_optimizer_abi_spec_ready_count"] == 2, report
    assert report["summary"]["plugin_selected_fused_backward_abi_implementation_ready_count"] == 2, report
    assert report["summary"]["plugin_selected_fused_backward_native_kernel_preconditions_spec_ready_count"] == 2, report
    assert report["summary"]["plugin_selected_fused_backward_resume_parity_matrix_spec_ready_count"] == 2, report
    assert report["summary"]["plugin_selected_fused_backward_resume_parity_matrix_implementation_ready_count"] == 2, report
    assert report["summary"]["plugin_selected_fused_backward_replay_case_planned_count"] == 10, report
    assert report["summary"]["plugin_selected_fused_backward_replay_case_implementation_ready_count"] == 10, report
    assert report["summary"]["plugin_selected_fused_backward_loss_scale_boundary_case_planned_count"] == 4, report
    assert report["summary"]["plugin_selected_fused_backward_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_fused_backward_owner_release_hold_optimizer_count"] == 2, report
    assert (
        report["summary"]["plugin_selected_fused_backward_owner_release_hold_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["plugin_selected_fused_backward_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_fused_backward_request_schema_ui_optimizer_count"] == 2, report
    assert (
        report["summary"]["plugin_selected_fused_backward_request_schema_ui_forbidden_token_hit_count"] == 0
    ), report
    assert (
        report["summary"]["plugin_selected_fused_backward_request_schema_ui_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["plugin_selected_model_shape_aware_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_model_shape_aware_optimizer_count"] == 7, report
    assert report["summary"]["plugin_selected_model_shape_aware_param_group_contract_count"] == 7, report
    assert report["summary"]["plugin_selected_model_shape_aware_param_group_abi_spec_ready_count"] == 7, report
    assert report["summary"]["plugin_selected_model_shape_aware_param_group_abi_implementation_ready_count"] == 7, report
    assert report["summary"]["plugin_selected_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count"] == 7, report
    assert report["summary"]["plugin_selected_model_shape_aware_param_group_resume_replay_matrix_row_count"] == 29, report
    assert report["summary"]["plugin_selected_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count"] == 7, report
    assert report["summary"]["plugin_selected_model_shape_aware_param_group_resume_replay_row_implementation_ready_count"] == 29, report
    assert report["summary"]["plugin_selected_model_shape_aware_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_model_shape_aware_owner_release_hold_optimizer_count"] == 7, report
    assert (
        report["summary"]["plugin_selected_model_shape_aware_owner_release_hold_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["plugin_selected_model_shape_aware_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_model_shape_aware_request_schema_ui_optimizer_count"] == 7, report
    assert (
        report["summary"]["plugin_selected_model_shape_aware_request_schema_ui_forbidden_token_hit_count"] == 0
    ), report
    assert (
        report["summary"]["plugin_selected_model_shape_aware_request_schema_ui_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["plugin_selected_state_adapter_special_family_batch_ready"] is True, report
    assert report["summary"]["plugin_selected_state_adapter_special_optimizer_count"] == 3, report
    assert report["summary"]["plugin_selected_state_adapter_special_param_ownership_abi_required_count"] == 3, report
    assert report["summary"]["plugin_selected_state_adapter_special_adapter_abi_spec_ready_count"] == 3, report
    assert report["summary"]["plugin_selected_state_adapter_special_adapter_abi_implementation_ready_count"] == 3, report
    assert report["summary"]["plugin_selected_state_adapter_special_native_kernel_precondition_spec_ready_count"] == 3, report
    assert report["summary"][
        "plugin_selected_state_adapter_special_native_kernel_precondition_implementation_ready_count"
    ] == 3, report
    assert report["summary"]["plugin_selected_state_adapter_special_resume_matrix_artifact_ready_count"] == 3, report
    assert report["summary"]["plugin_selected_state_adapter_special_resume_matrix_implementation_ready_count"] == 3, report
    assert report["summary"]["plugin_selected_state_adapter_special_resume_replay_case_planned_count"] == 15, report
    assert report["summary"][
        "plugin_selected_state_adapter_special_resume_replay_case_implementation_ready_count"
    ] == 15, report
    assert report["summary"]["plugin_selected_state_adapter_special_resume_translation_case_planned_count"] == 12, report
    assert report["summary"][
        "plugin_selected_state_adapter_special_resume_translation_case_implementation_ready_count"
    ] == 12, report
    assert report["summary"]["plugin_selected_state_adapter_special_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_state_adapter_special_owner_release_hold_optimizer_count"] == 3, report
    assert (
        report["summary"]["plugin_selected_state_adapter_special_owner_release_hold_product_native_ready_count"] == 0
    ), report
    assert (
        report["summary"]["plugin_selected_state_adapter_special_request_schema_ui_non_exposure_ready"] is True
    ), report
    assert report["summary"]["plugin_selected_state_adapter_special_request_schema_ui_optimizer_count"] == 3, report
    assert (
        report["summary"]["plugin_selected_state_adapter_special_request_schema_ui_forbidden_token_hit_count"] == 0
    ), report
    assert (
        report["summary"]["plugin_selected_state_adapter_special_request_schema_ui_product_native_ready_count"] == 0
    ), report
    assert report["summary"]["plugin_factored_memory_layout_observed_count"] == 8, report
    assert report["summary"]["plugin_selected_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_family_owner_release_hold_ready"] is True, report
    assert report["summary"]["plugin_selected_family_owner_release_hold_family_count"] == 10, report
    assert report["summary"]["plugin_selected_family_owner_release_hold_optimizer_count"] == 124, report
    assert report["summary"]["plugin_selected_family_owner_release_hold_product_native_ready_count"] == 0, report
    assert report["summary"]["plugin_selected_family_request_schema_ui_non_exposure_ready"] is True, report
    assert report["summary"]["plugin_selected_family_request_schema_ui_family_count"] == 10, report
    assert report["summary"]["plugin_selected_family_request_schema_ui_optimizer_count"] == 124, report
    assert report["summary"]["plugin_selected_family_request_schema_ui_forbidden_token_hit_count"] == 0, report
    assert report["summary"]["plugin_selected_family_request_schema_ui_product_native_ready_count"] == 0, report
    assert report["plugin_selector_scorecard"]["plugin_selector_classification_ready"] is True, report
    assert report["plugin_selector_scorecard"]["native_dispatch_allowed"] is False, report
    assert report["plugin_optimizer_family_batch"]["plugin_optimizer_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["native_dispatch_allowed"] is False, report
    assert report["plugin_optimizer_family_batch"]["runtime_dispatch_ready"] is False, report
    assert report["plugin_selected_family_owner_release_hold"]["owner_release_hold_ready"] is True, report
    assert report["plugin_selected_family_owner_release_hold"]["owner_approval_recorded"] is False, report
    assert report["plugin_selected_family_owner_release_hold"]["release_approval_recorded"] is False, report
    assert report["plugin_selected_family_owner_release_hold"]["native_dispatch_allowed"] is False, report
    assert report["plugin_selected_family_owner_release_hold"]["runtime_dispatch_ready"] is False, report
    assert report["plugin_selected_family_owner_release_hold"]["training_path_enabled"] is False, report
    assert report["plugin_selected_family_owner_release_hold"]["request_fields_emitted"] is False, report
    assert report["plugin_selected_family_owner_release_hold"]["schema_exposure_allowed"] is False, report
    assert report["plugin_selected_family_owner_release_hold"]["ui_exposure_allowed"] is False, report
    assert report["plugin_selected_family_owner_release_hold"]["family_count"] == 10, report
    assert report["plugin_selected_family_owner_release_hold"]["plugin_optimizer_count"] == 124, report
    assert report["plugin_selected_family_owner_release_hold"]["product_native_ready_count"] == 0, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"][
        "request_schema_ui_non_exposure_ready"
    ] is True, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["owner_approval_recorded"] is False, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["release_approval_recorded"] is False, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["native_dispatch_allowed"] is False, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["runtime_dispatch_ready"] is False, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["training_path_enabled"] is False, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["request_fields_emitted"] is False, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["schema_exposure_allowed"] is False, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["ui_exposure_allowed"] is False, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["family_count"] == 10, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["plugin_optimizer_count"] == 124, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["forbidden_token_hit_count"] == 0, report
    assert report["plugin_selected_family_request_schema_ui_non_exposure"]["product_native_ready_count"] == 0, report
    assert report["plugin_optimizer_family_batch"]["selected_adamlike_native_canary_ready_count"] >= 25, report
    assert report["plugin_optimizer_family_batch"]["selected_adamlike_e2e_shadow_matrix_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_adamlike_canary_rollout_policy_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_schedulefree_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_schedulefree_e2e_shadow_case_count"] == 6, report
    assert report["plugin_optimizer_family_batch"]["selected_schedulefree_native_canary_ready_count"] == 3, report
    assert report["plugin_optimizer_family_batch"]["selected_schedulefree_dispatch_review_gate_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_adaptivelr_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_adaptivelr_reference_ready_count"] == 6, report
    assert report["plugin_optimizer_family_batch"]["selected_adaptivelr_state_machine_abi_spec_ready_count"] == 6, report
    assert report["plugin_optimizer_family_batch"][
        "selected_adaptivelr_state_machine_abi_implementation_ready_count"
    ] == 6, report
    assert report["plugin_optimizer_family_batch"][
        "selected_adaptivelr_native_kernel_preconditions_spec_ready_count"
    ] == 6, report
    assert report["plugin_optimizer_family_batch"][
        "selected_adaptivelr_native_kernel_preconditions_implementation_ready_count"
    ] == 6, report
    assert report["plugin_optimizer_family_batch"][
        "selected_adaptivelr_state_machine_replay_matrix_artifact_ready_count"
    ] == 6, report
    assert report["plugin_optimizer_family_batch"][
        "selected_adaptivelr_state_machine_replay_matrix_implementation_ready_count"
    ] == 6, report
    assert report["plugin_optimizer_family_batch"][
        "selected_adaptivelr_state_machine_replay_case_planned_count"
    ] == 36, report
    assert report["plugin_optimizer_family_batch"][
        "selected_adaptivelr_state_machine_replay_case_implementation_ready_count"
    ] == 36, report
    assert report["plugin_optimizer_family_batch"][
        "selected_adaptivelr_state_machine_replay_resume_case_planned_count"
    ] == 24, report
    assert report["plugin_optimizer_family_batch"][
        "selected_adaptivelr_state_machine_replay_resume_case_implementation_ready_count"
    ] == 24, report
    assert report["plugin_optimizer_family_batch"]["selected_simple_formula_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_simple_formula_optimizer_count"] == 18, report
    assert report["plugin_optimizer_family_batch"]["selected_simple_formula_reference_canary_ready_count"] == 1, report
    assert report["plugin_optimizer_family_batch"]["selected_simple_formula_native_canary_ready_count"] == 18, report
    assert report["plugin_optimizer_family_batch"]["selected_simple_formula_e2e_shadow_matrix_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_simple_formula_e2e_shadow_case_count"] == 18, report
    assert report["plugin_optimizer_family_batch"]["selected_simple_formula_canary_rollout_policy_ready"] is True, report
    assert (
        report["plugin_optimizer_family_batch"]["selected_simple_formula_canary_rollout_policy_ready_count"] == 18
    ), report
    assert report["plugin_optimizer_family_batch"]["selected_simple_formula_dispatch_review_gate_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_simple_formula_dispatch_review_ready_count"] == 18, report
    assert report["plugin_optimizer_family_batch"]["selected_closure_second_order_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_closure_second_order_optimizer_count"] == 5, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_higher_order_abi_required_count"
    ] == 5, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_training_loop_abi_spec_ready_count"
    ] == 5, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_training_loop_abi_implementation_ready_count"
    ] == 5, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_resume_parity_matrix_spec_ready_count"
    ] == 5, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_resume_parity_matrix_implementation_ready_count"
    ] == 5, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_closure_replay_case_planned_count"
    ] == 22, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_create_graph_hvp_lifetime_case_planned_count"
    ] == 18, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_closure_resume_replay_artifact_ready_count"
    ] == 5, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_closure_resume_replay_artifact_row_count"
    ] == 20, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_closure_resume_replay_artifact_implementation_ready_count"
    ] == 5, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_closure_resume_replay_row_implementation_ready_count"
    ] == 20, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_native_kernel_precondition_plan_ready_count"
    ] == 5, report
    assert report["plugin_optimizer_family_batch"][
        "selected_closure_second_order_native_kernel_preconditions_implementation_ready_count"
    ] == 5, report
    assert report["plugin_optimizer_family_batch"]["selected_closure_second_order_owner_release_hold_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_closure_second_order_owner_release_hold_optimizer_count"] == 5, report
    assert (
        report["plugin_optimizer_family_batch"]["selected_closure_second_order_owner_release_hold_product_native_ready_count"]
        == 0
    ), report
    assert (
        report["plugin_optimizer_family_batch"]["selected_closure_second_order_request_schema_ui_non_exposure_ready"]
        is True
    ), report
    assert report["plugin_optimizer_family_batch"]["selected_closure_second_order_request_schema_ui_optimizer_count"] == 5, report
    assert (
        report["plugin_optimizer_family_batch"]["selected_closure_second_order_request_schema_ui_forbidden_token_hit_count"]
        == 0
    ), report
    assert (
        report["plugin_optimizer_family_batch"]["selected_closure_second_order_request_schema_ui_product_native_ready_count"]
        == 0
    ), report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_optimizer_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_parity_required_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_backlog_ready_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_evidence_artifact_planned_count"] == 235, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_evidence_status_pending_total"] == 0, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_formula_spec_artifact_ready_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_formula_spec_artifact_pending_count"] == 0, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_state_inventory_skeleton_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_state_inventory_artifact_ready_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_state_inventory_artifact_pending_count"] == 0, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_quality_guard_matrix_artifact_ready_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_quality_guard_matrix_artifact_pending_count"] == 0, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_quality_guard_matrix_case_planned_count"] == 329, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_formula_parity_matrix_artifact_planned_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_formula_parity_matrix_implementation_ready_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_formula_parity_case_planned_count"] == 282, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_resume_parity_matrix_artifact_planned_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_resume_parity_matrix_implementation_ready_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_resume_parity_case_planned_count"] == 188, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_execution_matrix_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_step_execution_ready_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_resume_next_step_replay_ready_count"] == 47, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_execution_failed_count"] == 0, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_owner_release_hold_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_owner_release_hold_optimizer_count"] == 47, report
    assert (
        report["plugin_optimizer_family_batch"][
            "selected_custom_formula_owner_release_hold_product_native_ready_count"
        ]
        == 0
    ), report
    assert (
        report["plugin_optimizer_family_batch"]["selected_custom_formula_request_schema_ui_non_exposure_ready"]
        is True
    ), report
    assert report["plugin_optimizer_family_batch"]["selected_custom_formula_request_schema_ui_optimizer_count"] == 47, report
    assert (
        report["plugin_optimizer_family_batch"][
            "selected_custom_formula_request_schema_ui_forbidden_token_hit_count"
        ]
        == 0
    ), report
    assert (
        report["plugin_optimizer_family_batch"][
            "selected_custom_formula_request_schema_ui_product_native_ready_count"
        ]
        == 0
    ), report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_optimizer_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_observed_layout_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_native_layout_abi_ready_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_quality_matrix_ready_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_native_kernel_entry_condition_ready_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_formula_tensor_binding_matrix_artifact_ready_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_formula_tensor_binding_matrix_implementation_ready_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_formula_step_execution_ready_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_resume_next_step_replay_ready_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_tensor_binding_ready_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_dispatch_review_gate_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_dispatch_review_ready_count"] == 8, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_formula_parity_case_planned_count"] == 40, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_tensor_binding_case_planned_count"] == 24, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_owner_release_hold_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_owner_release_hold_optimizer_count"] == 8, report
    assert (
        report["plugin_optimizer_family_batch"][
            "selected_factored_memory_owner_release_hold_product_native_ready_count"
        ]
        == 0
    ), report
    assert (
        report["plugin_optimizer_family_batch"]["selected_factored_memory_request_schema_ui_non_exposure_ready"]
        is True
    ), report
    assert report["plugin_optimizer_family_batch"]["selected_factored_memory_request_schema_ui_optimizer_count"] == 8, report
    assert (
        report["plugin_optimizer_family_batch"][
            "selected_factored_memory_request_schema_ui_forbidden_token_hit_count"
        ]
        == 0
    ), report
    assert (
        report["plugin_optimizer_family_batch"][
            "selected_factored_memory_request_schema_ui_product_native_ready_count"
        ]
        == 0
    ), report
    assert report["plugin_optimizer_family_batch"]["selected_fused_backward_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_fused_backward_optimizer_count"] == 2, report
    assert report["plugin_optimizer_family_batch"][
        "selected_fused_backward_gradient_ownership_abi_required_count"
    ] == 2, report
    assert report["plugin_optimizer_family_batch"]["selected_fused_backward_per_optimizer_abi_spec_ready_count"] == 2, report
    assert report["plugin_optimizer_family_batch"]["selected_fused_backward_abi_implementation_ready_count"] == 2, report
    assert report["plugin_optimizer_family_batch"][
        "selected_fused_backward_native_kernel_preconditions_spec_ready_count"
    ] == 2, report
    assert report["plugin_optimizer_family_batch"][
        "selected_fused_backward_resume_parity_matrix_implementation_ready_count"
    ] == 2, report
    assert report["plugin_optimizer_family_batch"][
        "selected_fused_backward_replay_case_implementation_ready_count"
    ] == 10, report
    assert report["plugin_optimizer_family_batch"]["selected_fused_backward_owner_release_hold_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_fused_backward_owner_release_hold_optimizer_count"] == 2, report
    assert (
        report["plugin_optimizer_family_batch"][
            "selected_fused_backward_owner_release_hold_product_native_ready_count"
        ]
        == 0
    ), report
    assert (
        report["plugin_optimizer_family_batch"]["selected_fused_backward_request_schema_ui_non_exposure_ready"]
        is True
    ), report
    assert report["plugin_optimizer_family_batch"]["selected_fused_backward_request_schema_ui_optimizer_count"] == 2, report
    assert (
        report["plugin_optimizer_family_batch"][
            "selected_fused_backward_request_schema_ui_forbidden_token_hit_count"
        ]
        == 0
    ), report
    assert (
        report["plugin_optimizer_family_batch"][
            "selected_fused_backward_request_schema_ui_product_native_ready_count"
        ]
        == 0
    ), report
    assert report["plugin_optimizer_family_batch"]["selected_model_shape_aware_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_model_shape_aware_optimizer_count"] == 7, report
    assert report["plugin_optimizer_family_batch"]["selected_model_shape_aware_param_group_contract_count"] == 7, report
    assert report["plugin_optimizer_family_batch"]["selected_model_shape_aware_param_group_abi_spec_ready_count"] == 7, report
    assert report["plugin_optimizer_family_batch"]["selected_model_shape_aware_param_group_abi_implementation_ready_count"] == 7, report
    assert report["plugin_optimizer_family_batch"]["selected_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count"] == 7, report
    assert report["plugin_optimizer_family_batch"]["selected_model_shape_aware_param_group_resume_replay_matrix_row_count"] == 29, report
    assert report["plugin_optimizer_family_batch"]["selected_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count"] == 7, report
    assert report["plugin_optimizer_family_batch"]["selected_model_shape_aware_param_group_resume_replay_row_implementation_ready_count"] == 29, report
    assert report["plugin_optimizer_family_batch"]["selected_state_adapter_special_family_batch_ready"] is True, report
    assert report["plugin_optimizer_family_batch"]["selected_state_adapter_special_optimizer_count"] == 3, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_param_ownership_abi_required_count"
    ] == 3, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_adapter_abi_spec_ready_count"
    ] == 3, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_adapter_abi_implementation_ready_count"
    ] == 3, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_native_kernel_precondition_spec_ready_count"
    ] == 3, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_native_kernel_precondition_implementation_ready_count"
    ] == 3, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_resume_matrix_artifact_ready_count"
    ] == 3, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_resume_matrix_implementation_ready_count"
    ] == 3, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_resume_replay_case_planned_count"
    ] == 15, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_resume_replay_case_implementation_ready_count"
    ] == 15, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_resume_translation_case_planned_count"
    ] == 12, report
    assert report["plugin_optimizer_family_batch"][
        "selected_state_adapter_special_resume_translation_case_implementation_ready_count"
    ] == 12, report
    assert report["plugin_optimizer_family_batch"]["plugin_factored_memory_layout_observed_count"] == 8, report
    assert "adam_like_formula" in report["plugin_optimizer_summary"]["plugin_selector_route_family_counts"], report
    for selector_type in ("PytorchOptimizer", "GenericOptimizer"):
        assert rows[selector_type]["turbocore_status"] == "plugin_selected_family_non_exposure_ready_default_off"
        assert rows[selector_type]["turbocore_route"] == "selected_plugin_family_non_exposure_default_off_v0"
        assert rows[selector_type]["training_path_enabled"] is False
        assert rows[selector_type]["default_behavior_changed"] is False
        assert (
            rows[selector_type]["next_gate"]
            == "keep plugin selected-family native dispatch unwired until explicit owner/release approval is recorded"
        )
    assert report["plugin_optimizer_summary"]["plugin_optimizer_count"] > 0, report
    assert report["priority_groups"], report
    assert (
        report["recommended_next_step"]
        == "keep native dispatch unwired until explicit owner/release approval is recorded"
    ), report
    for group in report["priority_groups"]:
        next_gate = str(group.get("next_gate", "")).lower()
        assert (
            "await explicit owner" in next_gate
            or "until explicit owner" in next_gate
            or "record explicit owner" in next_gate
            or "explicit owner/release approval" in next_gate
        ), group
    _write_real_artifact(report)
    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_coverage_scorecard_smoke",
        "ok": True,
        "roadmap": "devtools/docs/turbocore_optimizer_backend_design.md",
        "artifact_mode": artifact_mode,
        "real_artifact_checked": True,
        "summary": report["summary"],
        "recommended_next_step": report["recommended_next_step"],
    }


def _write_real_artifact(report: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    for filename in (
        "turbocore_optimizer_family_coverage_scorecard.json",
        "turbocore_optimizer_coverage_scorecard.json",
    ):
        (temp_dir / filename).write_text(payload, encoding="utf-8")


def _refresh_adamw_variant_review_artifacts() -> None:
    e2e = build_adamw_variant_e2e_shadow_matrix_scorecard(include_live_canaries=True)
    rollout = build_adamw_variant_canary_rollout_policy_scorecard(shadow_matrix_report=e2e)
    review = build_adamw_variant_dispatch_integration_review_scorecard(rollout_policy_report=rollout)
    _write_named_artifact("turbocore_adamw_variant_e2e_shadow_matrix_scorecard.json", e2e)
    _write_named_artifact("turbocore_adamw_variant_canary_rollout_policy_scorecard.json", rollout)
    _write_named_artifact("turbocore_adamw_variant_dispatch_integration_review_scorecard.json", review)


def _write_named_artifact(filename: str, payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-artifacts",
        action="store_true",
        help="Refresh nested optimizer-family artifacts before validating coverage.",
    )
    args = parser.parse_args(argv)
    payload = run_rebuild_smoke() if args.rebuild_artifacts else run_smoke()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
