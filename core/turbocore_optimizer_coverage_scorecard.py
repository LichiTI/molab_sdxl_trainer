"""Report-only TurboCore optimizer family coverage scorecard.

The current native update path is an exact AdamW route.  Other optimizer names
must not silently reuse that route unless their state and update semantics are
proven compatible.  This scorecard classifies optimizer families so roadmap
audits can track the next kernels without changing training behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.configs import OptimizerType
from core.lulynx_trainer.optimizer_capabilities import optimizer_capability_report
from core.turbocore_optimizer_native_kernel_inventory_scorecard import (
    build_optimizer_native_kernel_inventory_scorecard,
)
from core.turbocore_optimizer_family_kernel_contract_scorecard import (
    build_optimizer_family_kernel_contract_scorecard,
)
from core.turbocore_plugin_optimizer_selector_scorecard import build_plugin_optimizer_selector_scorecard


_NATIVE_ADAMW_READY = {OptimizerType.ADAMW}

_SIMPLE_FORMULA_NATIVE_DISPATCH_CANARY_READY = {
    OptimizerType.LION,
    OptimizerType.SGD_NESTEROV,
}

_ADAMW_VARIANT_RESEARCH = {
    OptimizerType.ADAMW_8BIT,
    OptimizerType.PAGED_ADAMW,
    OptimizerType.PAGED_ADAMW_32BIT,
    OptimizerType.PAGED_ADAMW_8BIT,
    OptimizerType.KAHAN_ADAMW_8BIT,
    OptimizerType.ADAMW_SCHEDULE_FREE,
}

_LION_SGD_RESEARCH = {
    OptimizerType.LION_8BIT,
    OptimizerType.PAGED_LION_8BIT,
    OptimizerType.SGD_NESTEROV_8BIT,
    OptimizerType.SGD_SCHEDULE_FREE,
    OptimizerType.RADAM_SCHEDULE_FREE,
}

_ADAPTIVE_LR_RESEARCH = {
    OptimizerType.PRODIGY,
    OptimizerType.AUTO_PRODIGY,
    OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
    OptimizerType.DADAPTATION,
    OptimizerType.DADAPT_ADAM_PREPRINT,
    OptimizerType.DADAPT_ADAGRAD,
    OptimizerType.DADAPT_ADAM,
    OptimizerType.DADAPT_ADAN,
    OptimizerType.DADAPT_ADAN_IP,
    OptimizerType.DADAPT_LION,
    OptimizerType.DADAPT_SGD,
}

_FACTORED_OR_CUSTOM_RESEARCH = {
    OptimizerType.ADAFACTOR,
    OptimizerType.AUTOMAGIC_PLUS_PLUS,
    OptimizerType.ANIMA_FACTORED_ADAMW,
}

_MODEL_SHAPE_AWARE_RESEARCH = {
    OptimizerType.MUON,
}

_SELECTOR_TYPES = {
    OptimizerType.PYTORCH_OPTIMIZER,
    OptimizerType.GENERIC,
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_optimizer_family_coverage_scorecard(
    *,
    optimizers: Iterable[OptimizerType | str] | None = None,
    adamw_representative_route_matrix_report: Mapping[str, Any] | None = None,
    exact_adamw_stream_event_chain_abi_report: Mapping[str, Any] | None = None,
    adamw_variant_batch_report: Mapping[str, Any] | None = None,
    adamw_variant_product_training_canary_report: Mapping[str, Any] | None = None,
    adamw_variant_owner_release_hold_report: Mapping[str, Any] | None = None,
    adamw_variant_request_schema_ui_report: Mapping[str, Any] | None = None,
    simple_family_batch_report: Mapping[str, Any] | None = None,
    simple_product_training_canary_report: Mapping[str, Any] | None = None,
    simple_owner_release_hold_report: Mapping[str, Any] | None = None,
    simple_schedulefree_rollout_policy_report: Mapping[str, Any] | None = None,
    simple_schedulefree_dispatch_review_report: Mapping[str, Any] | None = None,
    simple_schedulefree_owner_release_hold_report: Mapping[str, Any] | None = None,
    simple_request_schema_ui_report: Mapping[str, Any] | None = None,
    simple_runtime_dispatch_rehearsal_report: Mapping[str, Any] | None = None,
    adaptive_lr_batch_report: Mapping[str, Any] | None = None,
    adaptive_lr_replay_matrix_report: Mapping[str, Any] | None = None,
    adaptive_lr_replay_executor_report: Mapping[str, Any] | None = None,
    adaptive_lr_abi_preconditions_report: Mapping[str, Any] | None = None,
    adaptive_lr_abi_skeleton_report: Mapping[str, Any] | None = None,
    adaptive_lr_cpu_guard_report: Mapping[str, Any] | None = None,
    adaptive_lr_implementation_stub_report: Mapping[str, Any] | None = None,
    adaptive_lr_cuda_contract_report: Mapping[str, Any] | None = None,
    adaptive_lr_cuda_implementation_report: Mapping[str, Any] | None = None,
    adaptive_lr_training_tensor_binding_report: Mapping[str, Any] | None = None,
    adaptive_lr_runtime_dispatch_shadow_report: Mapping[str, Any] | None = None,
    adaptive_lr_training_loop_canary_report: Mapping[str, Any] | None = None,
    adaptive_lr_e2e_shadow_matrix_report: Mapping[str, Any] | None = None,
    adaptive_lr_canary_rollout_policy_report: Mapping[str, Any] | None = None,
    adaptive_lr_dispatch_review_report: Mapping[str, Any] | None = None,
    adaptive_lr_owner_release_hold_report: Mapping[str, Any] | None = None,
    adaptive_lr_request_schema_ui_report: Mapping[str, Any] | None = None,
    factored_custom_state_layout_report: Mapping[str, Any] | None = None,
    factored_custom_family_batch_report: Mapping[str, Any] | None = None,
    factored_custom_owner_release_hold_report: Mapping[str, Any] | None = None,
    factored_custom_request_schema_ui_report: Mapping[str, Any] | None = None,
    muon_model_shape_aware_batch_report: Mapping[str, Any] | None = None,
    muon_native_scratch_kernel_report: Mapping[str, Any] | None = None,
    muon_training_tensor_binding_report: Mapping[str, Any] | None = None,
    muon_training_loop_canary_report: Mapping[str, Any] | None = None,
    muon_e2e_shadow_matrix_report: Mapping[str, Any] | None = None,
    muon_canary_rollout_policy_report: Mapping[str, Any] | None = None,
    muon_dispatch_review_report: Mapping[str, Any] | None = None,
    muon_owner_release_hold_report: Mapping[str, Any] | None = None,
    muon_request_schema_ui_report: Mapping[str, Any] | None = None,
    plugin_family_batch_report: Mapping[str, Any] | None = None,
    plugin_selected_family_owner_release_hold_report: Mapping[str, Any] | None = None,
    plugin_selected_family_request_schema_ui_report: Mapping[str, Any] | None = None,
    plugin_adamlike_runtime_dispatch_rehearsal_report: Mapping[str, Any] | None = None,
    plugin_schedulefree_runtime_dispatch_rehearsal_report: Mapping[str, Any] | None = None,
    plugin_adaptivelr_runtime_dispatch_rehearsal_report: Mapping[str, Any] | None = None,
    plugin_closure_second_order_runtime_precondition_rehearsal_report: Mapping[str, Any] | None = None,
    plugin_custom_formula_runtime_precondition_rehearsal_report: Mapping[str, Any] | None = None,
    plugin_factored_memory_runtime_precondition_rehearsal_report: Mapping[str, Any] | None = None,
    plugin_model_shape_aware_runtime_precondition_rehearsal_report: Mapping[str, Any] | None = None,
    plugin_state_adapter_special_runtime_precondition_rehearsal_report: Mapping[str, Any] | None = None,
    plugin_fused_backward_runtime_precondition_rehearsal_report: Mapping[str, Any] | None = None,
    optimizer_native_kernel_inventory_report: Mapping[str, Any] | None = None,
    optimizer_family_kernel_contract_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify supported optimizer names by TurboCore implementation path."""

    capability_report = optimizer_capability_report(optimizers)
    adamw_route_matrix = _adamw_representative_route_matrix_report(adamw_representative_route_matrix_report)
    exact_adamw_stream_abi = _exact_adamw_stream_event_chain_abi_report(
        exact_adamw_stream_event_chain_abi_report
    )
    adamw_variant_batch = _adamw_variant_batch_report(adamw_variant_batch_report)
    adamw_variant_product_canary = _adamw_variant_product_training_canary_report(
        adamw_variant_product_training_canary_report
    )
    adamw_variant_owner_hold = _adamw_variant_owner_release_hold_report(
        adamw_variant_owner_release_hold_report
    )
    adamw_variant_request_schema_ui = _adamw_variant_request_schema_ui_report(
        adamw_variant_request_schema_ui_report
    )
    simple_batch = _simple_family_batch_report(simple_family_batch_report)
    simple_product_canary = _simple_product_training_canary_report(simple_product_training_canary_report)
    simple_owner_hold = _simple_owner_release_hold_report(simple_owner_release_hold_report)
    simple_schedulefree_rollout = _simple_schedulefree_rollout_policy_report(simple_schedulefree_rollout_policy_report)
    simple_schedulefree_review = _simple_schedulefree_dispatch_review_report(simple_schedulefree_dispatch_review_report)
    simple_schedulefree_hold = _simple_schedulefree_owner_release_hold_report(
        simple_schedulefree_owner_release_hold_report
    )
    simple_request_schema_ui = _simple_request_schema_ui_report(simple_request_schema_ui_report)
    simple_runtime_rehearsal = _simple_runtime_dispatch_rehearsal_report(
        simple_runtime_dispatch_rehearsal_report
    )
    adaptive_batch = _adaptive_lr_batch_report(adaptive_lr_batch_report)
    adaptive_replay = _adaptive_lr_replay_matrix_report(adaptive_lr_replay_matrix_report)
    adaptive_executor = _adaptive_lr_replay_executor_report(adaptive_lr_replay_executor_report)
    adaptive_preconditions = _adaptive_lr_abi_preconditions_report(adaptive_lr_abi_preconditions_report)
    adaptive_skeleton = _adaptive_lr_abi_skeleton_report(adaptive_lr_abi_skeleton_report)
    adaptive_cpu_guard = _adaptive_lr_cpu_guard_report(adaptive_lr_cpu_guard_report)
    adaptive_impl_stub = _adaptive_lr_implementation_stub_report(adaptive_lr_implementation_stub_report)
    adaptive_cuda_contract = _adaptive_lr_cuda_contract_report(adaptive_lr_cuda_contract_report)
    adaptive_cuda_impl = _adaptive_lr_cuda_implementation_report(adaptive_lr_cuda_implementation_report)
    adaptive_tensor_binding = _adaptive_lr_training_tensor_binding_report(adaptive_lr_training_tensor_binding_report)
    adaptive_runtime_shadow = _adaptive_lr_runtime_dispatch_shadow_report(
        adaptive_lr_runtime_dispatch_shadow_report
    )
    adaptive_training_loop = _adaptive_lr_training_loop_canary_report(adaptive_lr_training_loop_canary_report)
    adaptive_e2e_shadow = _adaptive_lr_e2e_shadow_matrix_report(adaptive_lr_e2e_shadow_matrix_report)
    adaptive_rollout = _adaptive_lr_canary_rollout_policy_report(adaptive_lr_canary_rollout_policy_report)
    adaptive_dispatch_review = _adaptive_lr_dispatch_review_report(adaptive_lr_dispatch_review_report)
    adaptive_owner_hold = _adaptive_lr_owner_release_hold_report(adaptive_lr_owner_release_hold_report)
    adaptive_request_schema_ui = _adaptive_lr_request_schema_ui_report(adaptive_lr_request_schema_ui_report)
    factored_custom_layout = _factored_custom_state_layout_report(factored_custom_state_layout_report)
    factored_custom_batch = _factored_custom_family_batch_report(factored_custom_family_batch_report)
    factored_custom_owner_hold = _factored_custom_owner_release_hold_report(
        factored_custom_owner_release_hold_report
    )
    factored_custom_request_schema_ui = _factored_custom_request_schema_ui_report(
        factored_custom_request_schema_ui_report
    )
    muon_model_shape_batch = _muon_model_shape_aware_batch_report(muon_model_shape_aware_batch_report)
    muon_native_scratch = _muon_native_scratch_kernel_report(muon_native_scratch_kernel_report)
    muon_tensor_binding = _muon_training_tensor_binding_report(muon_training_tensor_binding_report)
    muon_training_loop = _muon_training_loop_canary_report(muon_training_loop_canary_report)
    muon_e2e_shadow = _muon_e2e_shadow_matrix_report(muon_e2e_shadow_matrix_report)
    muon_rollout = _muon_canary_rollout_policy_report(muon_canary_rollout_policy_report)
    muon_dispatch_review = _muon_dispatch_review_report(muon_dispatch_review_report)
    muon_owner_hold = _muon_owner_release_hold_report(muon_owner_release_hold_report)
    muon_request_schema_ui = _muon_request_schema_ui_report(muon_request_schema_ui_report)
    plugin_batch = _plugin_family_batch_report(plugin_family_batch_report)
    plugin_owner_hold = _plugin_selected_family_owner_release_hold_report(
        plugin_selected_family_owner_release_hold_report
    )
    plugin_request_schema_ui = _plugin_selected_family_request_schema_ui_report(
        plugin_selected_family_request_schema_ui_report
    )
    plugin_adamlike_runtime_rehearsal = _plugin_adamlike_runtime_dispatch_rehearsal_report(
        plugin_adamlike_runtime_dispatch_rehearsal_report
    )
    plugin_schedulefree_runtime_rehearsal = _plugin_schedulefree_runtime_dispatch_rehearsal_report(
        plugin_schedulefree_runtime_dispatch_rehearsal_report
    )
    plugin_adaptivelr_runtime_rehearsal = _plugin_adaptivelr_runtime_dispatch_rehearsal_report(
        plugin_adaptivelr_runtime_dispatch_rehearsal_report
    )
    plugin_closure_second_order_precondition_rehearsal = (
        _plugin_closure_second_order_runtime_precondition_rehearsal_report(
            plugin_closure_second_order_runtime_precondition_rehearsal_report
        )
    )
    plugin_custom_formula_precondition_rehearsal = (
        _plugin_custom_formula_runtime_precondition_rehearsal_report(
            plugin_custom_formula_runtime_precondition_rehearsal_report
        )
    )
    plugin_factored_memory_precondition_rehearsal = (
        _plugin_factored_memory_runtime_precondition_rehearsal_report(
            plugin_factored_memory_runtime_precondition_rehearsal_report
        )
    )
    plugin_model_shape_precondition_rehearsal = (
        _plugin_model_shape_aware_runtime_precondition_rehearsal_report(
            plugin_model_shape_aware_runtime_precondition_rehearsal_report
        )
    )
    plugin_state_adapter_precondition_rehearsal = (
        _plugin_state_adapter_special_runtime_precondition_rehearsal_report(
            plugin_state_adapter_special_runtime_precondition_rehearsal_report
        )
    )
    plugin_fused_backward_precondition_rehearsal = (
        _plugin_fused_backward_runtime_precondition_rehearsal_report(
            plugin_fused_backward_runtime_precondition_rehearsal_report
        )
    )
    kernel_inventory = _optimizer_native_kernel_inventory_report(optimizer_native_kernel_inventory_report)
    family_kernel_contract = _optimizer_family_kernel_contract_report(optimizer_family_kernel_contract_report)
    selector_scorecard = _selector_scorecard_from_plugin_batch(plugin_batch) or build_plugin_optimizer_selector_scorecard()
    rows = [_coverage_row(item) for item in capability_report.get("optimizers", []) if isinstance(item, dict)]
    rows = _apply_adamw_representative_route_matrix(rows, adamw_route_matrix)
    rows = _apply_adamw_variant_batch(rows, adamw_variant_batch)
    rows = _apply_adamw_variant_product_training_canary(rows, adamw_variant_product_canary)
    rows = _apply_adamw_variant_owner_release_hold(rows, adamw_variant_owner_hold)
    rows = _apply_adamw_variant_request_schema_ui(rows, adamw_variant_request_schema_ui)
    rows = _apply_simple_family_batch(rows, simple_batch)
    rows = _apply_simple_product_training_canary(rows, simple_product_canary)
    rows = _apply_simple_owner_release_hold(rows, simple_owner_hold)
    rows = _apply_simple_schedulefree_rollout_policy(rows, simple_schedulefree_rollout)
    rows = _apply_simple_schedulefree_dispatch_review(rows, simple_schedulefree_review)
    rows = _apply_simple_schedulefree_owner_release_hold(rows, simple_schedulefree_hold)
    rows = _apply_simple_request_schema_ui(rows, simple_request_schema_ui)
    rows = _apply_adaptive_lr_batch(rows, adaptive_batch)
    rows = _apply_adaptive_lr_replay_matrix(rows, adaptive_replay)
    rows = _apply_adaptive_lr_replay_executor(rows, adaptive_executor)
    rows = _apply_adaptive_lr_abi_preconditions(rows, adaptive_preconditions)
    rows = _apply_adaptive_lr_abi_skeleton(rows, adaptive_skeleton)
    rows = _apply_adaptive_lr_cpu_guard(rows, adaptive_cpu_guard)
    rows = _apply_adaptive_lr_implementation_stub(rows, adaptive_impl_stub)
    rows = _apply_adaptive_lr_cuda_contract(rows, adaptive_cuda_contract)
    rows = _apply_adaptive_lr_cuda_implementation(rows, adaptive_cuda_impl)
    rows = _apply_adaptive_lr_training_tensor_binding(rows, adaptive_tensor_binding)
    rows = _apply_adaptive_lr_runtime_dispatch_shadow(rows, adaptive_runtime_shadow)
    rows = _apply_adaptive_lr_training_loop_canary(rows, adaptive_training_loop)
    rows = _apply_adaptive_lr_e2e_shadow_matrix(rows, adaptive_e2e_shadow)
    rows = _apply_adaptive_lr_canary_rollout_policy(rows, adaptive_rollout)
    rows = _apply_adaptive_lr_dispatch_review(rows, adaptive_dispatch_review)
    rows = _apply_adaptive_lr_owner_release_hold(rows, adaptive_owner_hold)
    rows = _apply_adaptive_lr_request_schema_ui(rows, adaptive_request_schema_ui)
    rows = _apply_factored_custom_state_layout(rows, factored_custom_layout)
    rows = _apply_factored_custom_family_batch(rows, factored_custom_batch)
    rows = _apply_factored_custom_owner_release_hold(rows, factored_custom_owner_hold)
    rows = _apply_factored_custom_request_schema_ui(rows, factored_custom_request_schema_ui)
    rows = _apply_muon_model_shape_aware_batch(rows, muon_model_shape_batch)
    rows = _apply_muon_native_scratch_kernel(rows, muon_native_scratch)
    rows = _apply_muon_training_tensor_binding(rows, muon_tensor_binding)
    rows = _apply_muon_training_loop_canary(rows, muon_training_loop)
    rows = _apply_muon_e2e_shadow_matrix(rows, muon_e2e_shadow)
    rows = _apply_muon_canary_rollout_policy(rows, muon_rollout)
    rows = _apply_muon_dispatch_review(rows, muon_dispatch_review)
    rows = _apply_muon_owner_release_hold(rows, muon_owner_hold)
    rows = _apply_muon_request_schema_ui(rows, muon_request_schema_ui)
    rows = _apply_plugin_selector_coverage(
        rows,
        selector_scorecard,
        plugin_batch,
        plugin_owner_hold,
        plugin_request_schema_ui,
    )
    missing = [row["optimizer_type"] for row in rows if row["turbocore_status"] == "unclassified"]
    native_ready = [
        row
        for row in rows
        if row["turbocore_status"] in {"native_adamw_ready", "adamw_representative_route_matrix_ready"}
    ]
    adamw_representative_route_matrix_ready = [
        row for row in rows if row["turbocore_status"] == "adamw_representative_route_matrix_ready"
    ]
    simple_native_canary_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_native_dispatch_canary_ready"
    ]
    simple_native_batch_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_native_batch_canary_ready"
    ]
    simple_product_training_canary_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_representative_product_training_canary_ready"
    ]
    simple_owner_release_hold_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_owner_release_hold_ready"
    ]
    simple_variant_layout_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_layout_spec_ready"
    ]
    simple_variant_state_machine_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_state_machine_reference_ready"
    ]
    simple_variant_native_abi_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_native_abi_spec_ready"
    ]
    simple_variant_quantized_formula_parity_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_quantized_formula_parity_ready"
    ]
    simple_variant_quantized_native_scratch_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_quantized_native_scratch_kernel_ready"
    ]
    simple_variant_quantized_runtime_manifest_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "simple_formula_variant_quantized_runtime_canary_manifest_ready"
    ]
    simple_variant_quantized_training_loop_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_quantized_training_loop_canary_ready"
    ]
    simple_variant_quantized_e2e_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_quantized_e2e_no_regression_ready"
    ]
    simple_variant_quantized_product_state_sync_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_quantized_product_state_sync_ready"
    ]
    simple_variant_quantized_rollout_policy_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_quantized_rollout_policy_ready"
    ]
    simple_variant_quantized_dispatch_review_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "simple_formula_variant_quantized_dispatch_integration_review_ready"
    ]
    simple_variant_quantized_owner_hold_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_quantized_owner_approval_hold_ready"
    ]
    simple_variant_quantized_training_loop_manifest_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "simple_formula_variant_quantized_training_loop_canary_manifest_ready"
    ]
    simple_variant_schedule_free_native_canary_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_schedule_free_native_canary_ready"
    ]
    simple_variant_schedule_free_rollout_policy_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_variant_schedule_free_rollout_policy_ready"
    ]
    simple_variant_schedule_free_dispatch_review_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "simple_formula_variant_schedule_free_dispatch_integration_review_ready"
    ]
    simple_variant_schedule_free_owner_hold_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "simple_formula_variant_schedule_free_owner_release_hold_ready"
    ]
    simple_request_schema_ui_ready = [
        row for row in rows if row["turbocore_status"] == "simple_formula_request_schema_ui_non_exposure_ready"
    ]
    simple_summary = _as_dict(simple_batch.get("summary"))
    simple_product_summary = _as_dict(simple_product_canary.get("summary"))
    simple_request_schema_ui_summary = _as_dict(simple_request_schema_ui.get("summary"))
    simple_runtime_rehearsal_summary = _as_dict(simple_runtime_rehearsal.get("summary"))
    simple_schedulefree_rollout_summary = _as_dict(simple_schedulefree_rollout.get("summary"))
    simple_schedulefree_review_summary = _as_dict(simple_schedulefree_review.get("summary"))
    simple_schedulefree_hold_summary = _as_dict(simple_schedulefree_hold.get("summary"))
    simple_schedulefree_rollout_count = (
        int(simple_schedulefree_rollout_summary.get("optimizer_count", 0) or 0)
        if simple_schedulefree_rollout.get("canary_rollout_policy_ready") is True
        else 0
    )
    simple_schedulefree_review_count = (
        int(simple_schedulefree_review_summary.get("optimizer_count", 0) or 0)
        if simple_schedulefree_review.get("review_gate_ready") is True
        else 0
    )
    simple_schedulefree_hold_count = (
        int(simple_schedulefree_hold_summary.get("optimizer_count", 0) or 0)
        if simple_schedulefree_hold.get("owner_release_hold_ready") is True
        else 0
    )
    adaptive_lr_reference_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_state_machine_reference_ready"
    ]
    adaptive_lr_replay_matrix_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_state_machine_replay_matrix_ready"
    ]
    adaptive_lr_replay_executor_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_state_machine_replay_executor_ready"
    ]
    adaptive_lr_abi_preconditions_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "adaptive_lr_native_state_machine_abi_precondition_review_ready"
    ]
    adaptive_lr_abi_skeleton_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_native_state_machine_abi_skeleton_ready"
    ]
    adaptive_lr_cpu_guard_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_native_state_machine_cpu_reference_guard_ready"
    ]
    adaptive_lr_impl_stub_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_native_state_machine_implementation_stub_ready"
    ]
    adaptive_lr_cuda_contract_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_cuda_kernel_contract_plan_ready"
    ]
    adaptive_lr_cuda_impl_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_cuda_kernel_implementation_ready"
    ]
    adaptive_lr_tensor_binding_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_training_tensor_binding_canary_ready"
    ]
    adaptive_lr_runtime_shadow_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_runtime_dispatch_shadow_ready"
    ]
    adaptive_lr_training_loop_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_training_loop_canary_ready"
    ]
    adaptive_lr_e2e_shadow_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_e2e_shadow_matrix_ready"
    ]
    adaptive_lr_rollout_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_canary_rollout_policy_ready"
    ]
    adaptive_lr_dispatch_review_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_dispatch_integration_review_ready"
    ]
    adaptive_lr_owner_hold_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_owner_release_hold_ready"
    ]
    adaptive_lr_request_schema_ui_ready = [
        row for row in rows if row["turbocore_status"] == "adaptive_lr_request_schema_ui_non_exposure_ready"
    ]
    factored_custom_state_layout_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_state_layout_reference_ready"
    ]
    factored_custom_native_scratch_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_native_scratch_kernel_ready"
    ]
    factored_custom_tensor_binding_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_training_tensor_binding_canary_ready"
    ]
    factored_custom_runtime_shadow_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_runtime_dispatch_adapter_shadow_ready"
    ]
    factored_custom_training_loop_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_training_loop_canary_ready"
    ]
    factored_custom_e2e_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_e2e_shadow_matrix_ready"
    ]
    factored_custom_rollout_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_canary_rollout_policy_ready"
    ]
    factored_custom_dispatch_review_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_dispatch_integration_review_ready"
    ]
    factored_custom_owner_hold_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_owner_release_hold_ready"
    ]
    factored_custom_request_schema_ui_ready = [
        row for row in rows if row["turbocore_status"] == "factored_custom_request_schema_ui_non_exposure_ready"
    ]
    model_shape_aware_param_group_abi_ready = [
        row
        for row in rows
        if row["turbocore_status"]
        in {
            "model_shape_aware_param_group_abi_ready",
            "model_shape_aware_dispatch_review_ready",
            "model_shape_aware_native_scratch_kernel_ready",
            "model_shape_aware_training_tensor_binding_canary_ready",
            "model_shape_aware_training_loop_canary_ready",
            "model_shape_aware_e2e_shadow_matrix_ready",
            "model_shape_aware_canary_rollout_policy_ready",
            "model_shape_aware_dispatch_integration_review_ready",
            "model_shape_aware_owner_release_hold_ready",
            "model_shape_aware_request_schema_ui_non_exposure_ready",
        }
    ]
    model_shape_aware_dispatch_review_ready = [
        row for row in rows if row["turbocore_status"] == "model_shape_aware_dispatch_review_ready"
    ]
    model_shape_aware_native_scratch_ready = [
        row for row in rows if row["turbocore_status"] == "model_shape_aware_native_scratch_kernel_ready"
    ]
    model_shape_aware_tensor_binding_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "model_shape_aware_training_tensor_binding_canary_ready"
    ]
    model_shape_aware_training_loop_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "model_shape_aware_training_loop_canary_ready"
    ]
    model_shape_aware_e2e_shadow_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "model_shape_aware_e2e_shadow_matrix_ready"
    ]
    model_shape_aware_rollout_ready = [
        row for row in rows if row["turbocore_status"] == "model_shape_aware_canary_rollout_policy_ready"
    ]
    model_shape_aware_dispatch_integration_ready = [
        row for row in rows if row["turbocore_status"] == "model_shape_aware_dispatch_integration_review_ready"
    ]
    model_shape_aware_owner_hold_ready = [
        row for row in rows if row["turbocore_status"] == "model_shape_aware_owner_release_hold_ready"
    ]
    model_shape_aware_request_schema_ui_ready = [
        row for row in rows if row["turbocore_status"] == "model_shape_aware_request_schema_ui_non_exposure_ready"
    ]
    adamw_variant_batch_pending = [
        row for row in rows if row["turbocore_status"] == "adamw_variant_family_batch_pending"
    ]
    adamw_variant_native_canary_ready = [
        row for row in rows if row["turbocore_status"] == "adamw_variant_native_canary_ready"
    ]
    adamw_variant_product_training_canary_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "adamw_variant_representative_product_training_canary_ready"
    ]
    adamw_variant_owner_release_hold_ready = [
        row for row in rows if row["turbocore_status"] == "adamw_variant_owner_release_hold_ready"
    ]
    adamw_variant_request_schema_ui_ready = [
        row for row in rows if row["turbocore_status"] == "adamw_variant_request_schema_ui_non_exposure_ready"
    ]
    adamw_variant_native_abi_ready = [
        row for row in rows if row["turbocore_status"] == "adamw_variant_native_abi_ready_kernel_pending"
    ]
    adamw_variant_scratch_formula_ready = [
        row for row in rows if row["turbocore_status"] == "adamw_variant_scratch_formula_canary_ready_kernel_pending"
    ]
    adamw_variant_native_scratch_ready = [
        row for row in rows if row["turbocore_status"] == "adamw_variant_native_scratch_kernel_ready_runtime_pending"
    ]
    adamw_variant_runtime_manifest_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "adamw_variant_runtime_canary_manifest_ready_training_loop_pending"
    ]
    adamw_variant_training_loop_manifest_ready = [
        row
        for row in rows
        if row["turbocore_status"] == "adamw_variant_training_loop_canary_manifest_ready_dispatch_pending"
    ]
    research = [row for row in rows if row["turbocore_status"].endswith("_research")]
    selectors = [row for row in rows if row["optimizer_type"] in {item.value for item in _SELECTOR_TYPES}]
    blocked = [row for row in rows if row["turbocore_status"] in {"unclassified", "standardcore_only"}]
    plugin = _plugin_summary(rows, selector_scorecard, plugin_batch, plugin_owner_hold, plugin_request_schema_ui)
    priority_groups = _priority_groups(
        rows,
        plugin,
        adamw_variant_batch,
        adamw_variant_product_canary,
        adamw_variant_owner_hold,
        adamw_variant_request_schema_ui,
        simple_product_canary,
        simple_owner_hold,
        simple_schedulefree_review,
        simple_schedulefree_hold,
        simple_request_schema_ui,
        adaptive_request_schema_ui,
        factored_custom_request_schema_ui,
        muon_request_schema_ui,
        plugin_owner_hold,
        plugin_request_schema_ui,
    )
    selector_blockers = _selector_blockers(selector_scorecard)
    plugin_batch_blockers = _plugin_batch_blockers(plugin_batch)
    plugin_owner_hold_blockers = _plugin_selected_family_owner_release_hold_blockers(plugin_owner_hold)
    plugin_request_schema_ui_blockers = _plugin_selected_family_request_schema_ui_blockers(plugin_request_schema_ui)
    exact_adamw_stream_abi_blockers = _exact_adamw_stream_event_chain_abi_blockers(
        exact_adamw_stream_abi
    )
    muon_model_shape_blockers = _muon_model_shape_aware_batch_blockers(muon_model_shape_batch)
    muon_native_scratch_blockers = _muon_native_scratch_kernel_blockers(muon_native_scratch)
    muon_tensor_binding_blockers = _muon_training_tensor_binding_blockers(muon_tensor_binding)
    muon_training_loop_blockers = _muon_training_loop_canary_blockers(muon_training_loop)
    muon_e2e_shadow_blockers = _muon_e2e_shadow_matrix_blockers(muon_e2e_shadow)
    muon_rollout_blockers = _muon_canary_rollout_policy_blockers(muon_rollout)
    muon_dispatch_review_blockers = _muon_dispatch_review_blockers(muon_dispatch_review)
    muon_owner_hold_blockers = _muon_owner_release_hold_blockers(muon_owner_hold)
    muon_request_schema_ui_blockers = _muon_request_schema_ui_blockers(muon_request_schema_ui)
    factored_custom_blockers = (
        _factored_custom_state_layout_blockers(factored_custom_layout)
        + _factored_custom_family_batch_blockers(factored_custom_batch)
        + _factored_custom_owner_release_hold_blockers(factored_custom_owner_hold)
        + _factored_custom_request_schema_ui_blockers(factored_custom_request_schema_ui)
    )
    blockers = (
        [f"unclassified_optimizer:{name}" for name in missing]
        + exact_adamw_stream_abi_blockers
        + selector_blockers
        + plugin_batch_blockers
        + plugin_owner_hold_blockers
        + plugin_request_schema_ui_blockers
        + muon_model_shape_blockers
        + muon_native_scratch_blockers
        + muon_tensor_binding_blockers
        + muon_training_loop_blockers
        + muon_e2e_shadow_blockers
        + muon_rollout_blockers
        + muon_dispatch_review_blockers
        + muon_owner_hold_blockers
        + muon_request_schema_ui_blockers
        + factored_custom_blockers
    )
    return {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_family_coverage_scorecard_v0",
        "gate": "optimizer_family_coverage",
        "ok": not blockers,
        "evidence_ready": not blockers,
        "promotion_ready": not blockers,
        "ready_for_optimizer_family_coverage_review": not blockers,
        "manual_review_required": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "product_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "native_adamw_current_scope": {
            "optimizer_types": [row["optimizer_type"] for row in native_ready],
            "route": "rust_cuda_adamw_v0",
            "exact_semantics_only": True,
        },
        "summary": {
            "total_optimizer_types": len(rows),
            "native_ready_count": len(native_ready),
            "adamw_representative_route_matrix_ready_count": len(adamw_representative_route_matrix_ready),
            "adamw_representative_route_matrix_product_native_ready_count": int(
                _as_dict(adamw_route_matrix.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "adamw_representative_route_matrix_route_row_ready_count": int(
                _as_dict(adamw_route_matrix.get("summary")).get("route_row_ready_count", 0) or 0
            ),
            "exact_adamw_stream_event_chain_ownership_abi_ready_count": int(
                _as_dict(exact_adamw_stream_abi.get("summary")).get(
                    "stream_event_chain_ownership_abi_ready_count",
                    0,
                )
                or 0
            ),
            "exact_adamw_stream_lifetime_ownership_bound_evidence_count": int(
                _as_dict(exact_adamw_stream_abi.get("summary")).get(
                    "stream_lifetime_ownership_bound_evidence_count",
                    0,
                )
                or 0
            ),
            "exact_adamw_stream_event_chain_verified_count": int(
                _as_dict(exact_adamw_stream_abi.get("summary")).get(
                    "event_chain_verified_count",
                    0,
                )
                or 0
            ),
            "exact_adamw_stream_event_chain_product_native_ready_count": int(
                _as_dict(exact_adamw_stream_abi.get("summary")).get("product_native_ready_count", 0)
                or 0
            ),
            "simple_formula_native_dispatch_canary_ready_count": len(simple_native_canary_ready),
            "simple_formula_native_batch_canary_ready_count": max(
                len(simple_native_batch_ready),
                int(simple_summary.get("batch_canary_ready_count", 0) or 0),
            ),
            "simple_formula_representative_product_training_canary_ready_count": max(
                len(simple_product_training_canary_ready),
                int(simple_product_summary.get("representative_product_training_canary_ready_count", 0) or 0),
            ),
            "simple_formula_owner_release_hold_ready_count": max(
                len(simple_owner_release_hold_ready),
                int(_as_dict(simple_owner_hold.get("summary")).get("optimizer_count", 0) or 0)
                if simple_owner_hold.get("owner_release_hold_ready") is True
                else 0,
            ),
            "simple_formula_variant_layout_spec_ready_count": max(
                len(simple_variant_layout_ready),
                int(simple_summary.get("variant_layout_spec_ready_count", 0) or 0),
            ),
            "simple_formula_variant_state_machine_reference_ready_count": max(
                len(simple_variant_state_machine_ready),
                int(simple_summary.get("variant_state_machine_reference_ready_count", 0) or 0),
            ),
            "simple_formula_variant_native_abi_spec_ready_count": max(
                len(simple_variant_native_abi_ready),
                int(simple_summary.get("variant_native_abi_spec_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_formula_parity_ready_count": max(
                len(simple_variant_quantized_formula_parity_ready),
                int(simple_summary.get("variant_quantized_formula_parity_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_native_scratch_kernel_ready_count": max(
                len(simple_variant_quantized_native_scratch_ready),
                int(simple_summary.get("variant_quantized_native_scratch_kernel_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_runtime_canary_manifest_ready_count": max(
                len(simple_variant_quantized_runtime_manifest_ready),
                int(simple_summary.get("variant_quantized_runtime_canary_manifest_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_training_loop_canary_ready_count": max(
                len(simple_variant_quantized_training_loop_ready),
                int(simple_summary.get("variant_quantized_training_loop_canary_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_e2e_no_regression_ready_count": max(
                len(simple_variant_quantized_e2e_ready),
                int(simple_summary.get("variant_quantized_e2e_no_regression_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_product_state_sync_ready_count": max(
                len(simple_variant_quantized_product_state_sync_ready),
                int(simple_summary.get("variant_quantized_product_optimizer_state_sync_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_rollout_policy_ready_count": max(
                len(simple_variant_quantized_rollout_policy_ready),
                int(simple_summary.get("variant_quantized_rollout_policy_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_dispatch_integration_review_ready_count": max(
                len(simple_variant_quantized_dispatch_review_ready),
                int(simple_summary.get("variant_quantized_dispatch_integration_review_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_owner_approval_hold_ready_count": max(
                len(simple_variant_quantized_owner_hold_ready),
                int(simple_summary.get("variant_quantized_owner_approval_hold_ready_count", 0) or 0),
            ),
            "simple_formula_variant_quantized_product_state_sync_review_ready_count": int(
                simple_summary.get(
                    "variant_quantized_product_state_sync_review_ready_count",
                    0,
                )
                or 0
            ),
            "simple_formula_variant_quantized_product_optimizer_state_sync_ready_count": int(
                simple_summary.get(
                    "variant_quantized_product_optimizer_state_sync_ready_count",
                    0,
                )
                or 0
            ),
            "simple_formula_variant_quantized_optimizer_state_sync_state_tensor_count": int(
                simple_summary.get(
                    "variant_quantized_optimizer_state_sync_state_tensor_count",
                    0,
                )
                or 0
            ),
            "simple_formula_variant_quantized_optimizer_state_sync_parameter_tensor_count": int(
                simple_summary.get(
                    "variant_quantized_optimizer_state_sync_parameter_tensor_count",
                    0,
                )
                or 0
            ),
            "simple_formula_variant_quantized_training_loop_canary_manifest_ready_count": max(
                len(simple_variant_quantized_training_loop_manifest_ready),
                int(simple_summary.get("variant_quantized_training_loop_canary_manifest_ready_count", 0) or 0),
            ),
            "simple_formula_variant_schedule_free_native_canary_ready_count": max(
                len(simple_variant_schedule_free_native_canary_ready),
                int(simple_summary.get("variant_schedule_free_native_canary_ready_count", 0) or 0),
            ),
            "simple_formula_variant_schedule_free_rollout_policy_ready_count": max(
                len(simple_variant_schedule_free_rollout_policy_ready),
                simple_schedulefree_rollout_count,
            ),
            "simple_formula_variant_schedule_free_dispatch_integration_review_ready_count": max(
                len(simple_variant_schedule_free_dispatch_review_ready),
                simple_schedulefree_review_count,
            ),
            "simple_formula_variant_schedule_free_owner_release_hold_ready_count": max(
                len(simple_variant_schedule_free_owner_hold_ready),
                simple_schedulefree_hold_count,
            ),
            "simple_formula_request_schema_ui_non_exposure_ready_count": len(simple_request_schema_ui_ready),
            "simple_formula_request_schema_ui_optimizer_count": int(
                simple_request_schema_ui_summary.get("optimizer_count", 0) or 0
            ),
            "simple_formula_request_schema_ui_forbidden_token_hit_count": int(
                simple_request_schema_ui_summary.get("forbidden_token_hit_count", 0) or 0
            ),
            "simple_formula_request_schema_ui_product_native_ready_count": int(
                simple_request_schema_ui_summary.get("product_native_ready_count", 0) or 0
            ),
            "simple_formula_runtime_dispatch_rehearsal_ready_count": int(
                simple_runtime_rehearsal_summary.get("runtime_dispatch_rehearsal_ready_count", 0) or 0
            ),
            "simple_formula_runtime_dispatch_rehearsal_case_count": int(
                simple_runtime_rehearsal_summary.get("case_count", 0) or 0
            ),
            "simple_formula_runtime_dispatch_rehearsal_native_step_count": int(
                simple_runtime_rehearsal_summary.get("native_step_count", 0) or 0
            ),
            "simple_formula_runtime_dispatch_rehearsal_product_native_ready_count": int(
                simple_runtime_rehearsal_summary.get("product_native_ready_count", 0) or 0
            ),
            "simple_formula_variant_quantized_native_canary_pending_count": int(
                simple_summary.get("variant_quantized_native_canary_pending_count", 0) or 0
            ),
            "simple_formula_variant_formula_parity_matrix_artifact_ready_count": int(
                simple_summary.get(
                    "variant_formula_parity_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "simple_formula_variant_formula_parity_matrix_implementation_ready_count": int(
                simple_summary.get(
                    "variant_formula_parity_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "simple_formula_variant_resume_parity_matrix_artifact_ready_count": int(
                simple_summary.get(
                    "variant_resume_parity_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "simple_formula_variant_resume_parity_matrix_implementation_ready_count": int(
                simple_summary.get(
                    "variant_resume_parity_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "simple_formula_variant_quantized_resume_parity_ready_count": int(
                simple_summary.get("variant_quantized_resume_parity_ready_count", 0) or 0
            ),
            "simple_formula_variant_schedule_free_resume_parity_ready_count": int(
                simple_summary.get("variant_schedule_free_resume_parity_ready_count", 0) or 0
            ),
            "simple_formula_variant_native_kernel_ready_count": int(
                simple_summary.get("variant_native_kernel_ready_count", 0) or 0
            ),
            "adamw_variant_family_batch_pending_count": len(adamw_variant_batch_pending),
            "adamw_variant_native_canary_ready_count": len(adamw_variant_native_canary_ready),
            "adamw_variant_representative_product_training_canary_ready_count": max(
                len(adamw_variant_product_training_canary_ready),
                int(
                    _as_dict(adamw_variant_product_canary.get("summary")).get(
                        "representative_product_training_canary_ready_count",
                        0,
                    )
                    or 0
                ),
            ),
            "adamw_variant_representative_product_training_canary_product_native_ready_count": int(
                _as_dict(adamw_variant_product_canary.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "adamw_variant_owner_release_hold_ready_count": max(
                len(adamw_variant_owner_release_hold_ready),
                int(_as_dict(adamw_variant_owner_hold.get("summary")).get("optimizer_count", 0) or 0)
                if adamw_variant_owner_hold.get("owner_release_hold_ready") is True
                else 0,
            ),
            "adamw_variant_owner_release_hold_product_native_ready_count": int(
                _as_dict(adamw_variant_owner_hold.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "adamw_variant_request_schema_ui_non_exposure_ready_count": len(
                adamw_variant_request_schema_ui_ready
            ),
            "adamw_variant_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(adamw_variant_request_schema_ui.get("summary")).get("forbidden_token_hit_count", 0)
                or 0
            ),
            "adamw_variant_request_schema_ui_product_native_ready_count": int(
                _as_dict(adamw_variant_request_schema_ui.get("summary")).get("product_native_ready_count", 0)
                or 0
            ),
            "adamw_variant_native_canary_stage_evidence_ready_count": int(
                _as_dict(adamw_variant_batch.get("summary")).get(
                    "native_canary_stage_evidence_ready_count",
                    _as_dict(adamw_variant_batch.get("summary")).get("native_ready_count", 0),
                )
                or 0
            ),
            "adamw_variant_product_native_ready_count": int(
                _as_dict(adamw_variant_batch.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "adamw_variant_native_abi_ready_count": max(
                len(adamw_variant_native_abi_ready),
                int(_as_dict(adamw_variant_batch.get("summary")).get("schedule_free_native_abi_ready_count", 0) or 0),
            ),
            "adamw_variant_scratch_formula_canary_ready_count": max(
                len(adamw_variant_scratch_formula_ready),
                int(
                    _as_dict(adamw_variant_batch.get("summary")).get(
                        "schedule_free_scratch_formula_canary_ready_count",
                        0,
                    )
                    or 0
                ),
            ),
            "adamw_variant_native_scratch_kernel_ready_count": max(
                len(adamw_variant_native_scratch_ready),
                int(
                    _as_dict(adamw_variant_batch.get("summary")).get(
                        "schedule_free_native_scratch_kernel_ready_count",
                        0,
                    )
                    or 0
                ),
            ),
            "adamw_variant_runtime_canary_manifest_ready_count": max(
                len(adamw_variant_runtime_manifest_ready),
                int(_as_dict(adamw_variant_batch.get("summary")).get("native_canary_manifest_ready_count", 0) or 0),
            ),
            "adamw_variant_training_loop_canary_manifest_ready_count": max(
                len(adamw_variant_training_loop_manifest_ready),
                int(_as_dict(adamw_variant_batch.get("summary")).get("training_loop_canary_ready_count", 0) or 0),
            ),
            "adamw_variant_state_reference_ready_count": int(
                _as_dict(adamw_variant_batch.get("summary")).get("state_reference_ready_count", 0) or 0
            ),
            "adamw_variant_native_canary_manifest_ready_count": int(
                _as_dict(adamw_variant_batch.get("summary")).get("native_canary_manifest_ready_count", 0) or 0
            ),
            "adamw_variant_training_loop_canary_ready_count": int(
                _as_dict(adamw_variant_batch.get("summary")).get("training_loop_canary_ready_count", 0) or 0
            ),
            "adamw_variant_schedule_free_native_abi_ready_count": int(
                _as_dict(adamw_variant_batch.get("summary")).get("schedule_free_native_abi_ready_count", 0) or 0
            ),
            "adamw_variant_schedule_free_scratch_formula_canary_ready_count": int(
                _as_dict(adamw_variant_batch.get("summary")).get(
                    "schedule_free_scratch_formula_canary_ready_count",
                    0,
                )
                or 0
            ),
            "adamw_variant_schedule_free_native_scratch_kernel_ready_count": int(
                _as_dict(adamw_variant_batch.get("summary")).get(
                    "schedule_free_native_scratch_kernel_ready_count",
                    0,
                )
                or 0
            ),
            "adamw_variant_e2e_shadow_matrix_ready": _as_dict(adamw_variant_batch.get("summary")).get(
                "e2e_shadow_matrix_ready"
            )
            is True,
            "adamw_variant_canary_rollout_policy_ready": _as_dict(adamw_variant_batch.get("summary")).get(
                "canary_rollout_policy_ready"
            )
            is True,
            "adamw_variant_dispatch_integration_review_ready": _as_dict(adamw_variant_batch.get("summary")).get(
                "dispatch_integration_review_ready"
            )
            is True,
            "adaptive_lr_state_machine_reference_ready_count": len(adaptive_lr_reference_ready),
            "adaptive_lr_state_machine_reference_artifact_ready_count": int(
                _as_dict(adaptive_batch.get("summary")).get("state_machine_reference_ready_count", 0) or 0
            ),
            "adaptive_lr_state_machine_replay_matrix_ready_count": len(adaptive_lr_replay_matrix_ready),
            "adaptive_lr_state_machine_replay_executor_ready_count": len(adaptive_lr_replay_executor_ready),
            "adaptive_lr_state_machine_replay_executor_artifact_ready_count": int(
                _as_dict(adaptive_executor.get("summary")).get("reference_replay_executor_ready_count", 0) or 0
            ),
            "adaptive_lr_native_state_machine_abi_precondition_review_ready_count": len(
                adaptive_lr_abi_preconditions_ready
            ),
            "adaptive_lr_native_state_machine_abi_precondition_review_artifact_ready_count": int(
                _as_dict(adaptive_preconditions.get("summary")).get(
                    "native_state_machine_abi_precondition_review_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_native_state_machine_abi_skeleton_ready_count": len(adaptive_lr_abi_skeleton_ready),
            "adaptive_lr_native_state_machine_abi_skeleton_artifact_ready_count": int(
                _as_dict(adaptive_skeleton.get("summary")).get("native_state_machine_abi_skeleton_ready_count", 0)
                or 0
            ),
            "adaptive_lr_native_state_machine_cpu_reference_guard_ready_count": len(adaptive_lr_cpu_guard_ready),
            "adaptive_lr_native_state_machine_cpu_reference_guard_artifact_ready_count": int(
                _as_dict(adaptive_cpu_guard.get("summary")).get("cpu_reference_guard_ready_count", 0) or 0
            ),
            "adaptive_lr_native_state_machine_implementation_stub_ready_count": len(adaptive_lr_impl_stub_ready),
            "adaptive_lr_native_state_machine_implementation_stub_artifact_ready_count": int(
                _as_dict(adaptive_impl_stub.get("summary")).get("implementation_stub_ready_count", 0) or 0
            ),
            "adaptive_lr_cuda_kernel_contract_plan_ready_count": len(adaptive_lr_cuda_contract_ready),
            "adaptive_lr_cuda_kernel_contract_plan_artifact_ready_count": int(
                _as_dict(adaptive_cuda_contract.get("summary")).get("cuda_kernel_contract_plan_ready_count", 0) or 0
            ),
            "adaptive_lr_cuda_kernel_implementation_ready_row_count": len(adaptive_lr_cuda_impl_ready),
            "adaptive_lr_training_tensor_binding_canary_ready_row_count": len(adaptive_lr_tensor_binding_ready),
            "adaptive_lr_runtime_dispatch_shadow_ready_row_count": len(adaptive_lr_runtime_shadow_ready),
            "adaptive_lr_training_loop_canary_ready_row_count": len(adaptive_lr_training_loop_ready),
            "adaptive_lr_e2e_shadow_matrix_ready_row_count": len(adaptive_lr_e2e_shadow_ready),
            "adaptive_lr_canary_rollout_policy_ready_row_count": len(adaptive_lr_rollout_ready),
            "adaptive_lr_dispatch_integration_review_ready_row_count": len(adaptive_lr_dispatch_review_ready),
            "adaptive_lr_owner_release_hold_ready_row_count": len(adaptive_lr_owner_hold_ready),
            "adaptive_lr_request_schema_ui_non_exposure_ready_row_count": len(adaptive_lr_request_schema_ui_ready),
            "adaptive_lr_state_machine_replay_matrix_artifact_ready_count": int(
                _as_dict(adaptive_replay.get("summary")).get(
                    "state_machine_replay_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_state_machine_replay_matrix_implementation_ready_count": int(
                _as_dict(adaptive_executor.get("summary")).get(
                    "state_machine_replay_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_state_machine_replay_case_planned_count": int(
                _as_dict(adaptive_replay.get("summary")).get("state_machine_replay_case_planned_count", 0) or 0
            ),
            "adaptive_lr_state_machine_replay_resume_case_planned_count": int(
                _as_dict(adaptive_replay.get("summary")).get(
                    "state_machine_replay_resume_case_planned_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_state_machine_replay_executor_reference_ready_count": int(
                _as_dict(adaptive_executor.get("summary")).get("reference_replay_executor_ready_count", 0) or 0
            ),
            "adaptive_lr_state_machine_replay_executor_resume_passed_count": int(
                _as_dict(adaptive_executor.get("summary")).get("resume_next_step_parity_passed_count", 0) or 0
            ),
            "adaptive_lr_native_state_machine_abi_precondition_package_ready_count": int(
                _as_dict(adaptive_preconditions.get("summary")).get(
                    "native_state_machine_abi_precondition_package_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_native_kernel_precondition_review_ready_count": int(
                _as_dict(adaptive_preconditions.get("summary")).get(
                    "native_kernel_precondition_review_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_native_state_machine_abi_implementation_ready_count": int(
                max(
                    int(
                        _as_dict(adaptive_preconditions.get("summary")).get(
                            "state_machine_abi_implementation_ready_count",
                            0,
                        )
                        or 0
                    ),
                    int(
                        _as_dict(adaptive_cuda_impl.get("summary")).get(
                            "state_machine_abi_implementation_ready_count",
                            0,
                        )
                        or 0
                    ),
                )
            ),
            "adaptive_lr_native_kernel_preconditions_implementation_ready_count": int(
                max(
                    int(
                        _as_dict(adaptive_preconditions.get("summary")).get(
                            "native_kernel_preconditions_implementation_ready_count",
                            0,
                        )
                        or 0
                    ),
                    int(
                        _as_dict(adaptive_cuda_impl.get("summary")).get(
                            "native_kernel_preconditions_implementation_ready_count",
                            0,
                        )
                        or 0
                    ),
                )
            ),
            "adaptive_lr_state_machine_entrypoint_contract_ready_count": int(
                _as_dict(adaptive_skeleton.get("summary")).get(
                    "state_machine_entrypoint_contract_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_launch_plan_schema_ready_count": int(
                _as_dict(adaptive_skeleton.get("summary")).get("launch_plan_schema_ready_count", 0) or 0
            ),
            "adaptive_lr_state_buffer_mapping_contract_ready_count": int(
                _as_dict(adaptive_skeleton.get("summary")).get(
                    "state_buffer_mapping_contract_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_skeleton_state_machine_abi_implementation_ready_count": int(
                _as_dict(adaptive_skeleton.get("summary")).get(
                    "state_machine_abi_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_cpu_guard_valid_launch_plan_passed_count": int(
                _as_dict(adaptive_cpu_guard.get("summary")).get(
                    "valid_launch_plan_guard_passed_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_cpu_guard_bad_finite_scalar_rejected_count": int(
                _as_dict(adaptive_cpu_guard.get("summary")).get(
                    "bad_finite_scalar_guard_rejected_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_cpu_guard_bad_dispatch_rejected_count": int(
                _as_dict(adaptive_cpu_guard.get("summary")).get(
                    "bad_dispatch_guard_rejected_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_cpu_guard_state_machine_abi_implementation_ready_count": int(
                _as_dict(adaptive_cpu_guard.get("summary")).get(
                    "state_machine_abi_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_implementation_stub_entrypoint_contract_ready_count": int(
                _as_dict(adaptive_impl_stub.get("summary")).get(
                    "stub_entrypoint_contract_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_implementation_stub_state_transition_contract_ready_count": int(
                _as_dict(adaptive_impl_stub.get("summary")).get(
                    "stub_state_transition_contract_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_implementation_stub_dispatch_disabled_assertion_ready_count": int(
                _as_dict(adaptive_impl_stub.get("summary")).get(
                    "stub_dispatch_disabled_assertion_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_implementation_stub_state_machine_abi_implementation_ready_count": int(
                _as_dict(adaptive_impl_stub.get("summary")).get(
                    "state_machine_abi_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_cuda_kernel_contract_runtime_canary_manifest_ready_count": int(
                _as_dict(adaptive_cuda_contract.get("summary")).get(
                    "runtime_canary_manifest_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_cuda_kernel_implementation_ready_count": int(
                _as_dict(adaptive_cuda_impl.get("summary")).get(
                    "cuda_kernel_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_cuda_kernel_executed_count": int(
                _as_dict(adaptive_cuda_impl.get("summary")).get("kernel_executed_count", 0) or 0
            ),
            "adaptive_lr_cuda_runtime_canary_ready_count": int(
                _as_dict(adaptive_tensor_binding.get("summary")).get("runtime_canary_ready_count", 0) or 0
            ),
            "adaptive_lr_cuda_runtime_canary_hit_count": int(
                _as_dict(adaptive_tensor_binding.get("summary")).get("runtime_canary_hit_count", 0) or 0
            ),
            "adaptive_lr_training_tensor_binding_canary_ready_count": int(
                _as_dict(adaptive_tensor_binding.get("summary")).get(
                    "training_tensor_binding_canary_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_training_tensor_binding_parity_ready_count": int(
                _as_dict(adaptive_tensor_binding.get("summary")).get(
                    "training_tensor_binding_parity_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_training_tensor_binding_family_passed_case_count": int(
                _as_dict(adaptive_tensor_binding.get("summary")).get("family_passed_case_count", 0) or 0
            ),
            "adaptive_lr_runtime_dispatch_shadow_ready_count": int(
                _as_dict(adaptive_runtime_shadow.get("summary")).get(
                    "runtime_dispatch_shadow_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_runtime_dispatch_shadow_native_shadow_call_allowed_count": int(
                _as_dict(adaptive_runtime_shadow.get("summary")).get(
                    "native_shadow_call_allowed_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_training_loop_canary_ready_count": int(
                _as_dict(adaptive_training_loop.get("summary")).get(
                    "training_loop_canary_ready_count",
                    0,
                )
                or 0
            ),
            "adaptive_lr_training_loop_family_passed_case_count": int(
                _as_dict(adaptive_training_loop.get("summary")).get("family_passed_case_count", 0) or 0
            ),
            "adaptive_lr_training_loop_native_step_count": int(
                _as_dict(adaptive_training_loop.get("summary")).get("native_step_count", 0) or 0
            ),
            "adaptive_lr_training_loop_native_kernel_launch_count": int(
                _as_dict(adaptive_training_loop.get("summary")).get("native_kernel_launch_count", 0) or 0
            ),
            "adaptive_lr_e2e_shadow_matrix_ready_count": int(
                _as_dict(adaptive_e2e_shadow.get("summary")).get("e2e_shadow_matrix_ready_count", 0) or 0
            ),
            "adaptive_lr_e2e_shadow_matrix_case_count": int(
                _as_dict(adaptive_e2e_shadow.get("summary")).get("case_count", 0) or 0
            ),
            "adaptive_lr_e2e_shadow_matrix_report_only_case_count": int(
                _as_dict(adaptive_e2e_shadow.get("summary")).get("report_only_case_count", 0) or 0
            ),
            "adaptive_lr_canary_rollout_policy_ready_count": int(
                _as_dict(adaptive_rollout.get("summary")).get("canary_rollout_policy_ready_count", 0) or 0
            ),
            "adaptive_lr_canary_rollout_policy_runtime_dispatch_ready_count": int(
                _as_dict(adaptive_rollout.get("summary")).get("runtime_dispatch_ready_count", 0) or 0
            ),
            "adaptive_lr_canary_rollout_policy_native_dispatch_allowed_count": int(
                _as_dict(adaptive_rollout.get("summary")).get("native_dispatch_allowed_count", 0) or 0
            ),
            "adaptive_lr_canary_rollout_policy_training_path_enabled_count": int(
                _as_dict(adaptive_rollout.get("summary")).get("training_path_enabled_count", 0) or 0
            ),
            "adaptive_lr_dispatch_integration_review_ready_count": int(
                _as_dict(adaptive_dispatch_review.get("summary")).get("optimizer_count", 0) or 0
            )
            if adaptive_dispatch_review.get("review_gate_ready") is True
            else 0,
            "adaptive_lr_dispatch_integration_review_product_native_ready_count": int(
                _as_dict(adaptive_dispatch_review.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "adaptive_lr_owner_release_hold_ready_count": int(
                _as_dict(adaptive_owner_hold.get("summary")).get("optimizer_count", 0) or 0
            )
            if adaptive_owner_hold.get("owner_release_hold_ready") is True
            else 0,
            "adaptive_lr_owner_release_hold_product_native_ready_count": int(
                _as_dict(adaptive_owner_hold.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "adaptive_lr_request_schema_ui_non_exposure_ready_count": int(
                _as_dict(adaptive_request_schema_ui.get("summary")).get("optimizer_count", 0) or 0
            )
            if adaptive_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True
            else 0,
            "adaptive_lr_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(adaptive_request_schema_ui.get("summary")).get("forbidden_token_hit_count", 0) or 0
            ),
            "factored_custom_state_layout_reference_ready_count": int(
                _as_dict(factored_custom_layout.get("summary")).get("optimizer_count", 0) or 0
            )
            if factored_custom_layout.get("state_layout_reference_ready") is True
            else 0,
            "factored_custom_state_layout_ready_row_count": len(factored_custom_state_layout_ready),
            "factored_custom_native_scratch_kernel_ready_row_count": len(factored_custom_native_scratch_ready),
            "factored_custom_training_tensor_binding_canary_ready_row_count": len(
                factored_custom_tensor_binding_ready
            ),
            "factored_custom_runtime_dispatch_adapter_shadow_ready_row_count": len(
                factored_custom_runtime_shadow_ready
            ),
            "factored_custom_training_loop_canary_ready_row_count": len(factored_custom_training_loop_ready),
            "factored_custom_e2e_shadow_matrix_ready_row_count": len(factored_custom_e2e_ready),
            "factored_custom_canary_rollout_policy_ready_row_count": len(factored_custom_rollout_ready),
            "factored_custom_dispatch_integration_review_ready_row_count": len(
                factored_custom_dispatch_review_ready
            ),
            "factored_custom_owner_release_hold_ready_row_count": len(factored_custom_owner_hold_ready),
            "factored_custom_request_schema_ui_non_exposure_ready_row_count": len(
                factored_custom_request_schema_ui_ready
            ),
            "factored_custom_optimizer_count": int(
                _as_dict(factored_custom_layout.get("summary")).get("optimizer_count", 0) or 0
            ),
            "factored_custom_local_live_reference_count": int(
                _as_dict(factored_custom_layout.get("summary")).get("local_live_reference_count", 0) or 0
            ),
            "factored_custom_memory_saving_candidate_count": int(
                _as_dict(factored_custom_layout.get("summary")).get("memory_saving_candidate_count", 0) or 0
            ),
            "factored_custom_passed_case_count": int(
                _as_dict(factored_custom_layout.get("summary")).get("passed_case_count", 0) or 0
            ),
            "factored_custom_required_case_count": int(
                _as_dict(factored_custom_layout.get("summary")).get("required_case_count", 0) or 0
            ),
            "factored_custom_family_batch_ready": factored_custom_batch.get("factored_custom_family_batch_ready")
            is True,
            "factored_custom_native_scratch_kernel_ready_count": int(
                _as_dict(factored_custom_batch.get("summary")).get("native_scratch_kernel_ready_count", 0) or 0
            ),
            "factored_custom_training_tensor_binding_canary_ready_count": int(
                _as_dict(factored_custom_batch.get("summary")).get(
                    "training_tensor_binding_canary_ready_count",
                    0,
                )
                or 0
            ),
            "factored_custom_runtime_dispatch_adapter_shadow_ready_count": int(
                _as_dict(factored_custom_batch.get("summary")).get(
                    "runtime_dispatch_adapter_shadow_ready_count",
                    0,
                )
                or 0
            ),
            "factored_custom_training_loop_canary_ready_count": int(
                _as_dict(factored_custom_batch.get("summary")).get("training_loop_canary_ready_count", 0) or 0
            ),
            "factored_custom_e2e_shadow_matrix_ready_count": int(
                _as_dict(factored_custom_batch.get("summary")).get("e2e_shadow_matrix_ready_count", 0) or 0
            ),
            "factored_custom_canary_rollout_policy_ready_count": int(
                _as_dict(factored_custom_batch.get("summary")).get("canary_rollout_policy_ready_count", 0) or 0
            ),
            "factored_custom_dispatch_integration_review_ready_count": int(
                _as_dict(factored_custom_batch.get("summary")).get(
                    "dispatch_integration_review_ready_count",
                    0,
                )
                or 0
            ),
            "factored_custom_owner_release_hold_ready_count": int(
                _as_dict(factored_custom_owner_hold.get("summary")).get("optimizer_count", 0) or 0
            )
            if factored_custom_owner_hold.get("owner_release_hold_ready") is True
            else 0,
            "factored_custom_owner_release_hold_product_native_ready_count": int(
                _as_dict(factored_custom_owner_hold.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "factored_custom_request_schema_ui_non_exposure_ready_count": int(
                _as_dict(factored_custom_request_schema_ui.get("summary")).get("optimizer_count", 0) or 0
            )
            if factored_custom_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True
            else 0,
            "factored_custom_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(factored_custom_request_schema_ui.get("summary")).get("forbidden_token_hit_count", 0)
                or 0
            ),
            "factored_custom_unsafe_claim_count": int(
                _as_dict(factored_custom_batch.get("summary")).get("unsafe_claim_count", 0) or 0
            ),
            "factored_custom_product_native_ready_count": 0,
            "muon_model_shape_aware_family_batch_ready": muon_model_shape_batch.get(
                "muon_model_shape_aware_family_batch_ready"
            )
            is True,
            "muon_model_shape_aware_param_group_abi_ready_row_count": len(
                model_shape_aware_param_group_abi_ready
            ),
            "muon_model_shape_aware_dispatch_review_ready_row_count": len(
                model_shape_aware_dispatch_review_ready
            ),
            "muon_model_shape_aware_native_scratch_kernel_ready_row_count": len(
                model_shape_aware_native_scratch_ready
            ),
            "muon_model_shape_aware_training_tensor_binding_ready_row_count": len(
                model_shape_aware_tensor_binding_ready
            ),
            "muon_model_shape_aware_training_loop_ready_row_count": len(
                model_shape_aware_training_loop_ready
            ),
            "muon_model_shape_aware_e2e_shadow_matrix_ready_row_count": len(
                model_shape_aware_e2e_shadow_ready
            ),
            "muon_model_shape_aware_canary_rollout_policy_ready_row_count": len(
                model_shape_aware_rollout_ready
            ),
            "muon_model_shape_aware_dispatch_integration_review_ready_row_count": len(
                model_shape_aware_dispatch_integration_ready
            ),
            "muon_model_shape_aware_owner_release_hold_ready_row_count": len(
                model_shape_aware_owner_hold_ready
            ),
            "muon_model_shape_aware_request_schema_ui_non_exposure_ready_row_count": len(
                model_shape_aware_request_schema_ui_ready
            ),
            "muon_model_shape_aware_optimizer_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get("optimizer_count", 0) or 0
            ),
            "muon_model_shape_aware_param_group_abi_spec_ready_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get("param_group_abi_spec_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_param_group_abi_implementation_ready_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get(
                    "param_group_abi_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get(
                    "param_group_resume_replay_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_param_group_resume_replay_matrix_row_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get(
                    "param_group_resume_replay_matrix_row_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get(
                    "param_group_resume_replay_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_param_group_resume_replay_row_implementation_ready_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get(
                    "param_group_resume_replay_row_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_native_kernel_precondition_ready_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get(
                    "native_kernel_precondition_ready_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_runtime_dispatch_shadow_ready_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get(
                    "runtime_dispatch_shadow_ready_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_dispatch_integration_review_ready_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get(
                    "dispatch_integration_review_ready_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_native_scratch_kernel_ready_count": int(
                _as_dict(muon_native_scratch.get("summary")).get("native_scratch_kernel_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_native_scratch_kernel_executed_count": int(
                _as_dict(muon_native_scratch.get("summary")).get("kernel_executed_count", 0) or 0
            ),
            "muon_model_shape_aware_native_scratch_product_native_ready_count": int(
                _as_dict(muon_native_scratch.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_training_tensor_binding_ready_count": int(
                _as_dict(muon_tensor_binding.get("summary")).get(
                    "training_tensor_binding_canary_ready_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_training_tensor_binding_parity_ready_count": int(
                _as_dict(muon_tensor_binding.get("summary")).get(
                    "training_tensor_binding_parity_ready_count",
                    0,
                )
                or 0
            ),
            "muon_model_shape_aware_training_tensor_binding_kernel_executed_count": int(
                _as_dict(muon_tensor_binding.get("summary")).get("kernel_executed_count", 0) or 0
            ),
            "muon_model_shape_aware_training_tensor_binding_product_native_ready_count": int(
                _as_dict(muon_tensor_binding.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_training_loop_ready_count": int(
                _as_dict(muon_training_loop.get("summary")).get("training_loop_canary_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_training_loop_native_step_count": int(
                _as_dict(muon_training_loop.get("summary")).get("native_step_count", 0) or 0
            ),
            "muon_model_shape_aware_training_loop_native_kernel_launch_count": int(
                _as_dict(muon_training_loop.get("summary")).get("native_kernel_launch_count", 0) or 0
            ),
            "muon_model_shape_aware_training_loop_product_native_ready_count": int(
                _as_dict(muon_training_loop.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_e2e_shadow_matrix_ready_count": int(
                _as_dict(muon_e2e_shadow.get("summary")).get("e2e_shadow_matrix_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_e2e_shadow_matrix_case_count": int(
                _as_dict(muon_e2e_shadow.get("summary")).get("case_count", 0) or 0
            ),
            "muon_model_shape_aware_e2e_shadow_matrix_report_only_case_count": int(
                _as_dict(muon_e2e_shadow.get("summary")).get("report_only_case_count", 0) or 0
            ),
            "muon_model_shape_aware_e2e_shadow_matrix_product_native_ready_count": int(
                _as_dict(muon_e2e_shadow.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_canary_rollout_policy_ready_count": int(
                _as_dict(muon_rollout.get("summary")).get("canary_rollout_policy_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_canary_rollout_policy_runtime_dispatch_ready_count": int(
                _as_dict(muon_rollout.get("summary")).get("runtime_dispatch_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_canary_rollout_policy_native_dispatch_allowed_count": int(
                _as_dict(muon_rollout.get("summary")).get("native_dispatch_allowed_count", 0) or 0
            ),
            "muon_model_shape_aware_canary_rollout_policy_training_path_enabled_count": int(
                _as_dict(muon_rollout.get("summary")).get("training_path_enabled_count", 0) or 0
            ),
            "muon_model_shape_aware_canary_rollout_policy_product_native_ready_count": int(
                _as_dict(muon_rollout.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_dispatch_review_gate_ready_count": int(
                _as_dict(muon_dispatch_review.get("summary")).get("optimizer_count", 0) or 0
            )
            if muon_dispatch_review.get("dispatch_integration_review") is True
            else 0,
            "muon_model_shape_aware_dispatch_review_product_native_ready_count": int(
                _as_dict(muon_dispatch_review.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_product_native_ready_count": int(
                _as_dict(muon_model_shape_batch.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_owner_release_hold_ready_count": int(
                _as_dict(muon_owner_hold.get("summary")).get("optimizer_count", 0) or 0
            )
            if muon_owner_hold.get("owner_release_hold_ready") is True
            else 0,
            "muon_model_shape_aware_owner_release_hold_product_native_ready_count": int(
                _as_dict(muon_owner_hold.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "muon_model_shape_aware_request_schema_ui_non_exposure_ready_count": int(
                _as_dict(muon_request_schema_ui.get("summary")).get("optimizer_count", 0) or 0
            )
            if muon_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True
            else 0,
            "muon_model_shape_aware_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(muon_request_schema_ui.get("summary")).get("forbidden_token_hit_count", 0) or 0
            ),
            "research_candidate_count": len(research),
            "selector_count": len(selectors),
            "standardcore_only_count": len(blocked),
            "missing_classification_count": len(missing),
            "plugin_optimizer_count": plugin["plugin_optimizer_count"],
            "plugin_resume_proven_count": plugin["plugin_resume_proven_count"],
            "route_family_counts": dict(plugin.get("plugin_selector_route_family_counts", {})),
            "route_family_count": len(plugin.get("plugin_selector_route_family_counts", {})),
            "plugin_selector_classification_ready": plugin["plugin_selector_classification_ready"],
            "plugin_selector_missing_classification_count": plugin["plugin_selector_missing_classification_count"],
            "plugin_selector_missing_resume_count": plugin["plugin_selector_missing_resume_count"],
            "plugin_family_batch_ready": plugin_batch.get("plugin_optimizer_family_batch_ready") is True,
            "plugin_selected_optimizer_gate_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_optimizer_gate_ready_count", 0) or 0
            ),
            "plugin_selected_optimizer_gate_pending_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_optimizer_gate_pending_count", 0) or 0
            ),
            "plugin_selected_adamlike_native_canary_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_adamlike_native_canary_ready_count", 0) or 0
            ),
            "plugin_selected_adamlike_e2e_shadow_matrix_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_adamlike_e2e_shadow_matrix_ready"
            )
            is True,
            "plugin_selected_adamlike_canary_rollout_policy_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_adamlike_canary_rollout_policy_ready"
            )
            is True,
            "plugin_selected_adamlike_owner_release_hold_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_adamlike_owner_release_hold_ready"
            )
            is True,
            "plugin_selected_adamlike_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_adamlike_owner_release_hold_optimizer_count", 0)
                or 0
            ),
            "plugin_selected_adamlike_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adamlike_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adamlike_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_adamlike_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_adamlike_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_adamlike_request_schema_ui_optimizer_count", 0)
                or 0
            ),
            "plugin_selected_adamlike_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adamlike_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adamlike_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adamlike_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adamlike_runtime_dispatch_rehearsal_ready_count": int(
                _as_dict(plugin_adamlike_runtime_rehearsal.get("summary")).get(
                    "runtime_dispatch_rehearsal_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adamlike_runtime_dispatch_rehearsal_case_count": int(
                _as_dict(plugin_adamlike_runtime_rehearsal.get("summary")).get("case_count", 0) or 0
            ),
            "plugin_selected_adamlike_runtime_dispatch_rehearsal_native_step_count": int(
                _as_dict(plugin_adamlike_runtime_rehearsal.get("summary")).get("native_step_count", 0) or 0
            ),
            "plugin_selected_adamlike_runtime_dispatch_rehearsal_product_native_ready_count": int(
                _as_dict(plugin_adamlike_runtime_rehearsal.get("summary")).get("product_native_ready_count", 0)
                or 0
            ),
            "plugin_selected_schedulefree_family_batch_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_schedulefree_family_batch_ready"
            )
            is True,
            "plugin_selected_schedulefree_e2e_shadow_case_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_schedulefree_e2e_shadow_case_count", 0) or 0
            ),
            "plugin_selected_schedulefree_native_canary_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_schedulefree_native_canary_ready_count", 0) or 0
            ),
            "plugin_selected_schedulefree_dispatch_review_gate_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_schedulefree_dispatch_review_gate_ready"
            )
            is True,
            "plugin_selected_schedulefree_owner_release_hold_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_schedulefree_owner_release_hold_ready"
            )
            is True,
            "plugin_selected_schedulefree_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_schedulefree_owner_release_hold_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_schedulefree_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_schedulefree_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_schedulefree_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_schedulefree_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_schedulefree_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_schedulefree_request_schema_ui_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_schedulefree_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_schedulefree_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_schedulefree_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_schedulefree_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_schedulefree_runtime_dispatch_rehearsal_ready_count": int(
                _as_dict(plugin_schedulefree_runtime_rehearsal.get("summary")).get(
                    "runtime_dispatch_rehearsal_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_schedulefree_runtime_dispatch_rehearsal_case_count": int(
                _as_dict(plugin_schedulefree_runtime_rehearsal.get("summary")).get("case_count", 0) or 0
            ),
            "plugin_selected_schedulefree_runtime_dispatch_rehearsal_native_step_count": int(
                _as_dict(plugin_schedulefree_runtime_rehearsal.get("summary")).get("native_step_count", 0) or 0
            ),
            "plugin_selected_schedulefree_runtime_dispatch_rehearsal_product_native_ready_count": int(
                _as_dict(plugin_schedulefree_runtime_rehearsal.get("summary")).get("product_native_ready_count", 0)
                or 0
            ),
            "plugin_selected_adaptivelr_family_batch_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_adaptivelr_family_batch_ready"
            )
            is True,
            "plugin_selected_adaptivelr_reference_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_adaptivelr_reference_ready_count", 0) or 0
            ),
            "plugin_selected_adaptivelr_state_machine_abi_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_state_machine_abi_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_state_machine_abi_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_state_machine_abi_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_native_kernel_preconditions_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_native_kernel_preconditions_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_native_kernel_preconditions_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_native_kernel_preconditions_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_state_machine_replay_matrix_artifact_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_state_machine_replay_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_state_machine_replay_matrix_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_state_machine_replay_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_state_machine_replay_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_state_machine_replay_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_state_machine_replay_case_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_state_machine_replay_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_state_machine_replay_resume_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_state_machine_replay_resume_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_state_machine_replay_resume_case_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_state_machine_replay_resume_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_owner_release_hold_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_adaptivelr_owner_release_hold_ready"
            )
            is True,
            "plugin_selected_adaptivelr_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_owner_release_hold_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_adaptivelr_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_adaptivelr_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_request_schema_ui_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_adaptivelr_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_runtime_dispatch_rehearsal_ready_count": int(
                _as_dict(plugin_adaptivelr_runtime_rehearsal.get("summary")).get(
                    "runtime_dispatch_rehearsal_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_runtime_dispatch_rehearsal_case_count": int(
                _as_dict(plugin_adaptivelr_runtime_rehearsal.get("summary")).get("case_count", 0) or 0
            ),
            "plugin_selected_adaptivelr_runtime_dispatch_rehearsal_native_step_count": int(
                _as_dict(plugin_adaptivelr_runtime_rehearsal.get("summary")).get("native_step_count", 0) or 0
            ),
            "plugin_selected_adaptivelr_runtime_dispatch_rehearsal_mapped_native_step_count": int(
                _as_dict(plugin_adaptivelr_runtime_rehearsal.get("summary")).get(
                    "mapped_selected_native_step_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_adaptivelr_runtime_dispatch_rehearsal_product_native_ready_count": int(
                _as_dict(plugin_adaptivelr_runtime_rehearsal.get("summary")).get("product_native_ready_count", 0)
                or 0
            ),
            "plugin_selected_simple_formula_family_batch_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_simple_formula_family_batch_ready"
            )
            is True,
            "plugin_selected_simple_formula_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_simple_formula_optimizer_count", 0) or 0
            ),
            "plugin_selected_simple_formula_reference_canary_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_simple_formula_reference_canary_ready_count", 0)
                or 0
            ),
            "plugin_selected_simple_formula_native_canary_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_simple_formula_native_canary_ready_count", 0)
                or 0
            ),
            "plugin_selected_simple_formula_e2e_shadow_matrix_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_simple_formula_e2e_shadow_matrix_ready"
            )
            is True,
            "plugin_selected_simple_formula_e2e_shadow_case_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_simple_formula_e2e_shadow_case_count", 0)
                or 0
            ),
            "plugin_selected_simple_formula_canary_rollout_policy_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_simple_formula_canary_rollout_policy_ready")
            is True,
            "plugin_selected_simple_formula_canary_rollout_policy_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_simple_formula_canary_rollout_policy_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_simple_formula_dispatch_review_gate_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_simple_formula_dispatch_review_gate_ready"
            )
            is True,
            "plugin_selected_simple_formula_dispatch_review_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_simple_formula_dispatch_review_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_simple_formula_owner_release_hold_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_simple_formula_owner_release_hold_ready"
            )
            is True,
            "plugin_selected_simple_formula_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_simple_formula_owner_release_hold_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_simple_formula_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_simple_formula_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_simple_formula_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_simple_formula_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_simple_formula_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_simple_formula_request_schema_ui_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_simple_formula_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_simple_formula_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_simple_formula_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_simple_formula_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_family_batch_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_closure_second_order_family_batch_ready"
            )
            is True,
            "plugin_selected_closure_second_order_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_closure_second_order_optimizer_count", 0) or 0
            ),
            "plugin_selected_closure_second_order_higher_order_abi_required_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_higher_order_abi_required_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_training_loop_abi_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_training_loop_abi_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_training_loop_abi_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_training_loop_abi_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_resume_parity_matrix_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_resume_parity_matrix_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_resume_parity_matrix_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_resume_parity_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_closure_replay_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_closure_replay_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_create_graph_hvp_lifetime_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_create_graph_hvp_lifetime_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_closure_resume_replay_artifact_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_closure_resume_replay_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_closure_resume_replay_artifact_row_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_closure_resume_replay_artifact_row_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_closure_resume_replay_artifact_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_closure_resume_replay_artifact_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_closure_resume_replay_row_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_closure_resume_replay_row_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_native_kernel_precondition_plan_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_native_kernel_precondition_plan_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_native_kernel_preconditions_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_native_kernel_preconditions_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_owner_release_hold_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_closure_second_order_owner_release_hold_ready"
            )
            is True,
            "plugin_selected_closure_second_order_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_owner_release_hold_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_closure_second_order_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_closure_second_order_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_request_schema_ui_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_closure_second_order_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_runtime_precondition_rehearsal_ready_count": int(
                _as_dict(plugin_closure_second_order_precondition_rehearsal.get("summary")).get(
                    "runtime_precondition_rehearsal_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_runtime_precondition_rehearsal_case_count": int(
                _as_dict(plugin_closure_second_order_precondition_rehearsal.get("summary")).get("case_count", 0)
                or 0
            ),
            "plugin_selected_closure_second_order_runtime_precondition_rehearsal_native_step_count": int(
                _as_dict(plugin_closure_second_order_precondition_rehearsal.get("summary")).get(
                    "native_step_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_closure_second_order_runtime_precondition_rehearsal_product_native_ready_count": int(
                _as_dict(plugin_closure_second_order_precondition_rehearsal.get("summary")).get(
                    "product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_family_batch_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_custom_formula_family_batch_ready"
            )
            is True,
            "plugin_selected_custom_formula_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_custom_formula_optimizer_count", 0) or 0
            ),
            "plugin_selected_custom_formula_parity_required_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_custom_formula_parity_required_count", 0) or 0
            ),
            "plugin_selected_custom_formula_backlog_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_custom_formula_backlog_ready_count", 0) or 0
            ),
            "plugin_selected_custom_formula_evidence_artifact_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_evidence_artifact_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_evidence_status_pending_total": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_evidence_status_pending_total",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_formula_spec_artifact_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_formula_spec_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_formula_spec_artifact_pending_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_formula_spec_artifact_pending_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_state_inventory_skeleton_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_state_inventory_skeleton_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_state_inventory_artifact_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_state_inventory_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_state_inventory_artifact_pending_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_state_inventory_artifact_pending_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_quality_guard_matrix_artifact_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_quality_guard_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_quality_guard_matrix_artifact_pending_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_quality_guard_matrix_artifact_pending_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_quality_guard_matrix_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_quality_guard_matrix_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_formula_parity_matrix_artifact_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_formula_parity_matrix_artifact_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_formula_parity_matrix_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_formula_parity_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_formula_parity_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_formula_parity_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_resume_parity_matrix_artifact_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_resume_parity_matrix_artifact_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_resume_parity_matrix_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_resume_parity_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_resume_parity_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_resume_parity_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_execution_matrix_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_custom_formula_execution_matrix_ready"
            )
            is True,
            "plugin_selected_custom_formula_step_execution_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_step_execution_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_resume_next_step_replay_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_resume_next_step_replay_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_execution_failed_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_execution_failed_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_owner_release_hold_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_custom_formula_owner_release_hold_ready")
            is True,
            "plugin_selected_custom_formula_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_owner_release_hold_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_custom_formula_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_custom_formula_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_request_schema_ui_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_custom_formula_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_runtime_precondition_rehearsal_ready_count": int(
                _as_dict(plugin_custom_formula_precondition_rehearsal.get("summary")).get(
                    "runtime_precondition_rehearsal_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_custom_formula_runtime_precondition_rehearsal_case_count": int(
                _as_dict(plugin_custom_formula_precondition_rehearsal.get("summary")).get("case_count", 0) or 0
            ),
            "plugin_selected_custom_formula_runtime_precondition_rehearsal_native_step_count": int(
                _as_dict(plugin_custom_formula_precondition_rehearsal.get("summary")).get("native_step_count", 0)
                or 0
            ),
            "plugin_selected_custom_formula_runtime_precondition_rehearsal_product_native_ready_count": int(
                _as_dict(plugin_custom_formula_precondition_rehearsal.get("summary")).get(
                    "product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_family_batch_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_factored_memory_family_batch_ready"
            )
            is True,
            "plugin_selected_factored_memory_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_factored_memory_optimizer_count", 0) or 0
            ),
            "plugin_selected_factored_memory_observed_layout_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_factored_memory_observed_layout_count", 0) or 0
            ),
            "plugin_selected_factored_memory_native_layout_abi_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_native_layout_abi_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_quality_matrix_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_quality_matrix_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_native_kernel_entry_condition_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_native_kernel_entry_condition_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_formula_tensor_binding_matrix_artifact_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_formula_tensor_binding_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_formula_tensor_binding_matrix_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_formula_tensor_binding_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_formula_step_execution_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_formula_step_execution_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_resume_next_step_replay_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_resume_next_step_replay_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_tensor_binding_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_tensor_binding_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_dispatch_review_gate_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_factored_memory_dispatch_review_gate_ready")
            is True,
            "plugin_selected_factored_memory_dispatch_review_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_dispatch_review_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_formula_parity_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_formula_parity_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_tensor_binding_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_tensor_binding_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_owner_release_hold_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_factored_memory_owner_release_hold_ready")
            is True,
            "plugin_selected_factored_memory_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_owner_release_hold_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_factored_memory_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_factored_memory_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_request_schema_ui_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_factored_memory_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_runtime_precondition_rehearsal_ready_count": int(
                _as_dict(plugin_factored_memory_precondition_rehearsal.get("summary")).get(
                    "runtime_precondition_rehearsal_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_factored_memory_runtime_precondition_rehearsal_case_count": int(
                _as_dict(plugin_factored_memory_precondition_rehearsal.get("summary")).get("case_count", 0) or 0
            ),
            "plugin_selected_factored_memory_runtime_precondition_rehearsal_native_step_count": int(
                _as_dict(plugin_factored_memory_precondition_rehearsal.get("summary")).get("native_step_count", 0)
                or 0
            ),
            "plugin_selected_factored_memory_runtime_precondition_rehearsal_product_native_ready_count": int(
                _as_dict(plugin_factored_memory_precondition_rehearsal.get("summary")).get(
                    "product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_family_batch_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_fused_backward_family_batch_ready"
            )
            is True,
            "plugin_selected_fused_backward_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_fused_backward_optimizer_count", 0) or 0
            ),
            "plugin_selected_fused_backward_gradient_ownership_abi_required_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_gradient_ownership_abi_required_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_per_optimizer_abi_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_per_optimizer_abi_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_abi_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_abi_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_native_kernel_preconditions_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_native_kernel_preconditions_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_resume_parity_matrix_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_resume_parity_matrix_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_resume_parity_matrix_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_resume_parity_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_replay_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_replay_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_replay_case_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_replay_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_loss_scale_boundary_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_loss_scale_boundary_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_owner_release_hold_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_fused_backward_owner_release_hold_ready")
            is True,
            "plugin_selected_fused_backward_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_owner_release_hold_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_fused_backward_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_fused_backward_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_request_schema_ui_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_fused_backward_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_runtime_precondition_rehearsal_ready_count": int(
                _as_dict(plugin_fused_backward_precondition_rehearsal.get("summary")).get(
                    "runtime_precondition_rehearsal_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_fused_backward_runtime_precondition_rehearsal_case_count": int(
                _as_dict(plugin_fused_backward_precondition_rehearsal.get("summary")).get("case_count", 0) or 0
            ),
            "plugin_selected_fused_backward_runtime_precondition_rehearsal_native_step_count": int(
                _as_dict(plugin_fused_backward_precondition_rehearsal.get("summary")).get("native_step_count", 0)
                or 0
            ),
            "plugin_selected_fused_backward_runtime_precondition_rehearsal_product_native_ready_count": int(
                _as_dict(plugin_fused_backward_precondition_rehearsal.get("summary")).get(
                    "product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_family_batch_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_model_shape_aware_family_batch_ready"
            )
            is True,
            "plugin_selected_model_shape_aware_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_model_shape_aware_optimizer_count", 0) or 0
            ),
            "plugin_selected_model_shape_aware_param_group_contract_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_param_group_contract_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_param_group_abi_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_param_group_abi_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_param_group_abi_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_param_group_abi_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_param_group_resume_replay_matrix_row_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_param_group_resume_replay_matrix_row_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_param_group_resume_replay_row_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_param_group_resume_replay_row_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_owner_release_hold_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_model_shape_aware_owner_release_hold_ready")
            is True,
            "plugin_selected_model_shape_aware_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_owner_release_hold_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_model_shape_aware_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_model_shape_aware_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_request_schema_ui_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_model_shape_aware_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_runtime_precondition_rehearsal_ready_count": int(
                _as_dict(plugin_model_shape_precondition_rehearsal.get("summary")).get(
                    "runtime_precondition_rehearsal_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_model_shape_aware_runtime_precondition_rehearsal_case_count": int(
                _as_dict(plugin_model_shape_precondition_rehearsal.get("summary")).get("case_count", 0) or 0
            ),
            "plugin_selected_model_shape_aware_runtime_precondition_rehearsal_native_step_count": int(
                _as_dict(plugin_model_shape_precondition_rehearsal.get("summary")).get("native_step_count", 0)
                or 0
            ),
            "plugin_selected_model_shape_aware_runtime_precondition_rehearsal_product_native_ready_count": int(
                _as_dict(plugin_model_shape_precondition_rehearsal.get("summary")).get(
                    "product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_family_batch_ready": _as_dict(plugin_batch.get("summary")).get(
                "selected_state_adapter_special_family_batch_ready"
            )
            is True,
            "plugin_selected_state_adapter_special_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get("selected_state_adapter_special_optimizer_count", 0) or 0
            ),
            "plugin_selected_state_adapter_special_param_ownership_abi_required_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_param_ownership_abi_required_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_adapter_abi_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_adapter_abi_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_adapter_abi_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_adapter_abi_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_native_kernel_precondition_spec_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_native_kernel_precondition_spec_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_native_kernel_precondition_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_native_kernel_precondition_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_resume_matrix_artifact_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_resume_matrix_artifact_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_resume_matrix_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_resume_matrix_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_resume_replay_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_resume_replay_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_resume_replay_case_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_resume_replay_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_resume_translation_case_planned_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_resume_translation_case_planned_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_resume_translation_case_implementation_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_resume_translation_case_implementation_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_owner_release_hold_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_state_adapter_special_owner_release_hold_ready")
            is True,
            "plugin_selected_state_adapter_special_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_owner_release_hold_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_owner_release_hold_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_request_schema_ui_non_exposure_ready": _as_dict(
                plugin_batch.get("summary")
            ).get("selected_state_adapter_special_request_schema_ui_non_exposure_ready")
            is True,
            "plugin_selected_state_adapter_special_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_request_schema_ui_optimizer_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_request_schema_ui_forbidden_token_hit_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get(
                    "selected_state_adapter_special_request_schema_ui_product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_runtime_precondition_rehearsal_ready_count": int(
                _as_dict(plugin_state_adapter_precondition_rehearsal.get("summary")).get(
                    "runtime_precondition_rehearsal_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_selected_state_adapter_special_runtime_precondition_rehearsal_case_count": int(
                _as_dict(plugin_state_adapter_precondition_rehearsal.get("summary")).get("case_count", 0) or 0
            ),
            "plugin_selected_state_adapter_special_runtime_precondition_rehearsal_native_step_count": int(
                _as_dict(plugin_state_adapter_precondition_rehearsal.get("summary")).get("native_step_count", 0)
                or 0
            ),
            "plugin_selected_state_adapter_special_runtime_precondition_rehearsal_product_native_ready_count": int(
                _as_dict(plugin_state_adapter_precondition_rehearsal.get("summary")).get(
                    "product_native_ready_count",
                    0,
                )
                or 0
            ),
            "plugin_factored_memory_layout_observed_count": int(
                _as_dict(plugin_batch.get("summary")).get("plugin_factored_memory_layout_observed_count", 0) or 0
            ),
            "plugin_selected_native_ready_count": int(
                _as_dict(plugin_batch.get("summary")).get("plugin_selected_native_ready_count", 0) or 0
            ),
            "plugin_selected_family_owner_release_hold_ready": plugin_owner_hold.get(
                "owner_release_hold_ready"
            )
            is True,
            "plugin_selected_family_owner_release_hold_family_count": int(
                _as_dict(plugin_owner_hold.get("summary")).get("family_count", 0) or 0
            ),
            "plugin_selected_family_owner_release_hold_optimizer_count": int(
                _as_dict(plugin_owner_hold.get("summary")).get("plugin_optimizer_count", 0) or 0
            ),
            "plugin_selected_family_owner_release_hold_product_native_ready_count": int(
                _as_dict(plugin_owner_hold.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "plugin_selected_family_request_schema_ui_non_exposure_ready": plugin_request_schema_ui.get(
                "request_schema_ui_non_exposure_ready"
            )
            is True,
            "plugin_selected_family_request_schema_ui_family_count": int(
                _as_dict(plugin_request_schema_ui.get("summary")).get("family_count", 0) or 0
            ),
            "plugin_selected_family_request_schema_ui_optimizer_count": int(
                _as_dict(plugin_request_schema_ui.get("summary")).get("plugin_optimizer_count", 0) or 0
            ),
            "plugin_selected_family_request_schema_ui_forbidden_token_hit_count": int(
                _as_dict(plugin_request_schema_ui.get("summary")).get("forbidden_token_hit_count", 0) or 0
            ),
            "plugin_selected_family_request_schema_ui_product_native_ready_count": int(
                _as_dict(plugin_request_schema_ui.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "optimizer_native_kernel_inventory_source_ready_count": int(
                _as_dict(kernel_inventory.get("summary")).get("kernel_source_present_count", 0) or 0
            ),
            "optimizer_native_kernel_inventory_probe_ready_count": int(
                _as_dict(kernel_inventory.get("summary")).get("rust_probe_present_count", 0) or 0
            ),
            "optimizer_native_kernel_inventory_product_native_ready_count": int(
                _as_dict(kernel_inventory.get("summary")).get("product_native_ready_count", 0) or 0
            ),
            "optimizer_family_kernel_contract_entrypoint_present_count": int(
                _as_dict(family_kernel_contract.get("summary")).get("entrypoint_present_count", 0) or 0
            ),
            "optimizer_family_kernel_contract_ready_count": int(
                _as_dict(family_kernel_contract.get("summary")).get("optimizer_family_contract_count", 0) or 0
            ),
            "optimizer_family_kernel_contract_validation_ok_count": int(
                _as_dict(family_kernel_contract.get("summary")).get("validation_ok_count", 0) or 0
            ),
            "optimizer_family_kernel_contract_source_ready_count": int(
                _as_dict(family_kernel_contract.get("summary")).get("kernel_source_ready_count", 0) or 0
            ),
            "optimizer_family_kernel_contract_training_path_enabled_count": int(
                _as_dict(family_kernel_contract.get("summary")).get("training_path_enabled_count", 0) or 0
            ),
            "optimizer_family_kernel_contract_native_dispatch_allowed_count": int(
                _as_dict(family_kernel_contract.get("summary")).get("native_dispatch_allowed_count", 0) or 0
            ),
            "optimizer_family_kernel_contract_product_native_ready_count": int(
                _as_dict(family_kernel_contract.get("summary")).get("product_native_ready_count", 0) or 0
            ),
        },
        "priority_groups": priority_groups,
        "rows": rows,
        "adamw_variant_family_batch": _compact_adamw_variant_batch(adamw_variant_batch),
        "exact_adamw_stream_event_chain_ownership_abi": _compact_exact_adamw_stream_event_chain_abi(
            exact_adamw_stream_abi
        ),
        "adamw_variant_product_training_canary": _compact_adamw_variant_product_training_canary(
            adamw_variant_product_canary
        ),
        "adamw_variant_owner_release_hold": _compact_adamw_variant_owner_release_hold(
            adamw_variant_owner_hold
        ),
        "adamw_variant_request_schema_ui_non_exposure": _compact_adamw_variant_request_schema_ui(
            adamw_variant_request_schema_ui
        ),
        "simple_optimizer_family_batch": _compact_simple_family_batch(simple_batch),
        "simple_optimizer_product_training_canary": _compact_simple_product_training_canary(simple_product_canary),
        "simple_optimizer_owner_release_hold": _compact_simple_owner_release_hold(simple_owner_hold),
        "simple_optimizer_schedulefree_rollout_policy": _compact_simple_schedulefree_rollout_policy(
            simple_schedulefree_rollout
        ),
        "simple_optimizer_schedulefree_dispatch_review": _compact_simple_schedulefree_dispatch_review(
            simple_schedulefree_review
        ),
        "simple_optimizer_schedulefree_owner_release_hold": _compact_simple_schedulefree_owner_release_hold(
            simple_schedulefree_hold
        ),
        "simple_optimizer_request_schema_ui_non_exposure": _compact_simple_request_schema_ui(
            simple_request_schema_ui
        ),
        "simple_optimizer_runtime_dispatch_rehearsal": _compact_simple_runtime_dispatch_rehearsal(
            simple_runtime_rehearsal
        ),
        "adaptive_lr_state_machine_batch": _compact_adaptive_lr_batch(adaptive_batch),
        "adaptive_lr_state_machine_replay_matrix": _compact_adaptive_lr_replay_matrix(adaptive_replay),
        "adaptive_lr_state_machine_replay_executor": _compact_adaptive_lr_replay_executor(adaptive_executor),
        "adaptive_lr_native_state_machine_abi_preconditions": _compact_adaptive_lr_abi_preconditions(
            adaptive_preconditions
        ),
        "adaptive_lr_native_state_machine_abi_skeleton": _compact_adaptive_lr_abi_skeleton(adaptive_skeleton),
        "adaptive_lr_native_state_machine_cpu_reference_guard": _compact_adaptive_lr_cpu_guard(
            adaptive_cpu_guard
        ),
        "adaptive_lr_native_state_machine_implementation_stub": _compact_adaptive_lr_implementation_stub(
            adaptive_impl_stub
        ),
        "adaptive_lr_cuda_kernel_contract_plan": _compact_adaptive_lr_cuda_contract(adaptive_cuda_contract),
        "adaptive_lr_cuda_kernel_implementation": _compact_adaptive_lr_cuda_implementation(adaptive_cuda_impl),
        "adaptive_lr_training_tensor_binding_canary": _compact_adaptive_lr_training_tensor_binding(
            adaptive_tensor_binding
        ),
        "adaptive_lr_runtime_dispatch_shadow": _compact_adaptive_lr_runtime_dispatch_shadow(
            adaptive_runtime_shadow
        ),
        "adaptive_lr_training_loop_canary": _compact_adaptive_lr_training_loop_canary(adaptive_training_loop),
        "adaptive_lr_e2e_shadow_matrix": _compact_adaptive_lr_e2e_shadow_matrix(adaptive_e2e_shadow),
        "adaptive_lr_canary_rollout_policy": _compact_adaptive_lr_canary_rollout_policy(adaptive_rollout),
        "adaptive_lr_dispatch_integration_review": _compact_adaptive_lr_dispatch_review(
            adaptive_dispatch_review
        ),
        "adaptive_lr_owner_release_hold": _compact_adaptive_lr_owner_release_hold(adaptive_owner_hold),
        "adaptive_lr_request_schema_ui_non_exposure": _compact_adaptive_lr_request_schema_ui(
            adaptive_request_schema_ui
        ),
        "factored_custom_optimizer_state_layout": _compact_factored_custom_state_layout(factored_custom_layout),
        "factored_custom_optimizer_family_batch": _compact_factored_custom_family_batch(factored_custom_batch),
        "factored_custom_owner_release_hold": _compact_factored_custom_owner_release_hold(
            factored_custom_owner_hold
        ),
        "factored_custom_request_schema_ui_non_exposure": _compact_factored_custom_request_schema_ui(
            factored_custom_request_schema_ui
        ),
        "muon_model_shape_aware_family_batch": _compact_muon_model_shape_aware_batch(muon_model_shape_batch),
        "muon_model_shape_aware_native_scratch_kernel": _compact_muon_native_scratch_kernel(
            muon_native_scratch
        ),
        "muon_model_shape_aware_training_tensor_binding_canary": _compact_muon_training_tensor_binding(
            muon_tensor_binding
        ),
        "muon_model_shape_aware_training_loop_canary": _compact_muon_training_loop_canary(
            muon_training_loop
        ),
        "muon_model_shape_aware_e2e_shadow_matrix": _compact_muon_e2e_shadow_matrix(
            muon_e2e_shadow
        ),
        "muon_model_shape_aware_canary_rollout_policy": _compact_muon_canary_rollout_policy(
            muon_rollout
        ),
        "muon_model_shape_aware_dispatch_integration_review": _compact_muon_dispatch_review(
            muon_dispatch_review
        ),
        "muon_model_shape_aware_owner_release_hold": _compact_muon_owner_release_hold(muon_owner_hold),
        "muon_model_shape_aware_request_schema_ui_non_exposure": _compact_muon_request_schema_ui(
            muon_request_schema_ui
        ),
        "plugin_optimizer_family_batch": _compact_plugin_family_batch(plugin_batch),
        "plugin_adamlike_runtime_dispatch_rehearsal": _compact_plugin_adamlike_runtime_dispatch_rehearsal(
            plugin_adamlike_runtime_rehearsal
        ),
        "plugin_schedulefree_runtime_dispatch_rehearsal": _compact_plugin_schedulefree_runtime_dispatch_rehearsal(
            plugin_schedulefree_runtime_rehearsal
        ),
        "plugin_adaptivelr_runtime_dispatch_rehearsal": _compact_plugin_adaptivelr_runtime_dispatch_rehearsal(
            plugin_adaptivelr_runtime_rehearsal
        ),
        "plugin_closure_second_order_runtime_precondition_rehearsal": (
            _compact_plugin_closure_second_order_runtime_precondition_rehearsal(
                plugin_closure_second_order_precondition_rehearsal
            )
        ),
        "plugin_custom_formula_runtime_precondition_rehearsal": (
            _compact_plugin_custom_formula_runtime_precondition_rehearsal(
                plugin_custom_formula_precondition_rehearsal
            )
        ),
        "plugin_factored_memory_runtime_precondition_rehearsal": (
            _compact_plugin_factored_memory_runtime_precondition_rehearsal(
                plugin_factored_memory_precondition_rehearsal
            )
        ),
        "plugin_model_shape_aware_runtime_precondition_rehearsal": (
            _compact_plugin_model_shape_aware_runtime_precondition_rehearsal(
                plugin_model_shape_precondition_rehearsal
            )
        ),
        "plugin_state_adapter_special_runtime_precondition_rehearsal": (
            _compact_plugin_state_adapter_special_runtime_precondition_rehearsal(
                plugin_state_adapter_precondition_rehearsal
            )
        ),
        "plugin_fused_backward_runtime_precondition_rehearsal": (
            _compact_plugin_fused_backward_runtime_precondition_rehearsal(
                plugin_fused_backward_precondition_rehearsal
            )
        ),
        "plugin_selected_family_owner_release_hold": _compact_plugin_selected_family_owner_release_hold(
            plugin_owner_hold
        ),
        "plugin_selected_family_request_schema_ui_non_exposure": _compact_plugin_selected_family_request_schema_ui(
            plugin_request_schema_ui
        ),
        "optimizer_native_kernel_inventory": _compact_optimizer_native_kernel_inventory(kernel_inventory),
        "optimizer_family_kernel_contract": _compact_optimizer_family_kernel_contract(family_kernel_contract),
        "plugin_optimizer_summary": plugin,
        "plugin_selector_scorecard": _compact_selector_scorecard(selector_scorecard),
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": _recommended_next_step(priority_groups),
        "notes": [
            "This scorecard is report-only and never enables native dispatch.",
            "Only exact AdamW is eligible for the current Rust/CUDA update route.",
            "Selector/plugin optimizers require per-selected-optimizer scorecards before native work.",
        ],
    }


def _coverage_row(capability: Mapping[str, Any]) -> dict[str, Any]:
    value = str(capability.get("optimizer_type", "") or "")
    try:
        optimizer = OptimizerType(value)
    except ValueError:
        return _row(capability, status="unclassified", route="unknown", next_gate="add_optimizer_capability_mapping")

    if optimizer in _NATIVE_ADAMW_READY:
        return _row(
            capability,
            status="native_adamw_ready",
            route="rust_cuda_adamw_v0",
            next_gate="extend_representative_route_matrix",
            notes=("Current exact AdamW native route.",),
        )
    if optimizer in _ADAMW_VARIANT_RESEARCH:
        return _row(
            capability,
            status="adamw_variant_research",
            route="new_kernel_or_state_layout_required",
            next_gate="prove_quantized_or_paged_state_parity",
            notes=("AdamW-like math, but state dtype, paging, Kahan, or schedule-free semantics differ.",),
        )
    if optimizer in _SIMPLE_FORMULA_NATIVE_DISPATCH_CANARY_READY:
        return _row(
            capability,
            status="simple_formula_native_dispatch_canary_ready",
            route="rust_cuda_simple_formula_runtime_v0",
            next_gate="representative_product_training_canary",
            notes=("Formula parity, runtime tensor launch, training executor, and dispatch runtime canary are proven.",),
        )
    if optimizer in _LION_SGD_RESEARCH:
        return _row(
            capability,
            status="simple_update_research",
            route="new_formula_kernel_required",
            next_gate="state_layout_or_variant_formula_parity_probe",
            notes=("Simple formula family variant, but dtype, paging, or schedule-free semantics still differ.",),
        )
    if optimizer in _ADAPTIVE_LR_RESEARCH:
        return _row(
            capability,
            status="adaptive_lr_research",
            route="new_state_machine_and_kernel_required",
            next_gate="model_dynamic_lr_state_before_cuda_kernel",
            notes=("Adaptive LR/state-machine optimizers need their own native semantics.",),
        )
    if optimizer in _FACTORED_OR_CUSTOM_RESEARCH:
        return _row(
            capability,
            status="factored_or_custom_research",
            route="custom_layout_and_kernel_required",
            next_gate="prove_state_layout_memory_and_quality_tradeoff",
            notes=("Factored/custom optimizer state cannot reuse exact AdamW dispatch.",),
        )
    if optimizer in _MODEL_SHAPE_AWARE_RESEARCH:
        return _row(
            capability,
            status="model_shape_aware_research",
            route="model_shape_param_group_abi_required",
            next_gate="param_group_abi_and_shape_replay_matrix",
            notes=("Muon requires shape-aware param-group splitting and cannot reuse exact AdamW dispatch.",),
        )
    if optimizer in _SELECTOR_TYPES:
        return _row(
            capability,
            status="selector_requires_selected_optimizer_scorecard",
            route="selected_optimizer_subroute",
            next_gate="classify_selected_plugin_optimizers",
            notes=("Selector is not one optimizer; each selected implementation needs a row.",),
        )
    return _row(capability, status="unclassified", route="unknown", next_gate="classify_optimizer_family")


def _simple_family_batch_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_simple_optimizer_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _simple_product_training_canary_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_simple_optimizer_product_training_canary_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _simple_owner_release_hold_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_simple_optimizer_owner_release_hold_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _simple_schedulefree_rollout_policy_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_simple_optimizer_schedulefree_rollout_policy_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _simple_schedulefree_dispatch_review_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_simple_optimizer_schedulefree_dispatch_integration_review_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _simple_schedulefree_owner_release_hold_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_simple_optimizer_schedulefree_owner_release_hold_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _simple_request_schema_ui_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_simple_optimizer_request_schema_ui_non_exposure_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _simple_runtime_dispatch_rehearsal_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_simple_formula_runtime_dispatch_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adamw_representative_route_matrix_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adamw_representative_route_matrix_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _exact_adamw_stream_event_chain_abi_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_exact_adamw_stream_event_chain_abi_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _apply_adamw_representative_route_matrix(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("representative_route_matrix_ready") is not True:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] == OptimizerType.ADAMW.value and row["turbocore_status"] == "native_adamw_ready":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adamw_representative_route_matrix_ready",
                    "turbocore_route": str(report.get("route") or "rust_cuda_adamw_v0"),
                    "next_gate": str(
                        report.get("recommended_next_step")
                        or "record explicit owner/release approval for exact AdamW native dispatch"
                    ),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Exact AdamW representative route matrix is ready.",
                        "Product native dispatch remains disabled until explicit owner/release approval is recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _adamw_variant_batch_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adamw_variant_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adamw_variant_product_training_canary_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_adamw_variant_product_training_canary_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adamw_variant_owner_release_hold_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_adamw_variant_owner_release_hold_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adamw_variant_request_schema_ui_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_adamw_variant_request_schema_ui_non_exposure_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _apply_adamw_variant_batch(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = _as_dict(report.get("summary"))
    if int(summary.get("target_count", 0) or 0) <= 0:
        return rows
    batch_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        batch = _as_dict(batch_rows.get(optimizer_type))
        if batch:
            updated = dict(row)
            if batch.get("native_ready") is True:
                status = "adamw_variant_native_canary_ready"
                next_gate = "representative_product_training_canary"
            elif batch.get("batch_status") == "training_loop_canary_manifest_ready_dispatch_pending":
                status = "adamw_variant_training_loop_canary_manifest_ready_dispatch_pending"
                next_gate = str(batch.get("next_gate") or "schedule_free_training_dispatch_canary")
            elif batch.get("batch_status") == "runtime_canary_manifest_ready_training_loop_pending":
                status = "adamw_variant_runtime_canary_manifest_ready_training_loop_pending"
                next_gate = str(batch.get("next_gate") or "schedule_free_training_loop_canary")
            elif batch.get("batch_status") == "native_scratch_kernel_ready_runtime_pending":
                status = "adamw_variant_native_scratch_kernel_ready_runtime_pending"
                next_gate = str(batch.get("next_gate") or "schedule_free_runtime_canary")
            elif batch.get("batch_status") == "scratch_formula_canary_ready_kernel_pending":
                status = "adamw_variant_scratch_formula_canary_ready_kernel_pending"
                next_gate = str(batch.get("next_gate") or "schedule_free_native_kernel_canary")
            elif batch.get("batch_status") == "native_abi_ready_kernel_pending":
                status = "adamw_variant_native_abi_ready_kernel_pending"
                next_gate = str(batch.get("next_gate") or "schedule_free_native_kernel_canary")
            else:
                status = "adamw_variant_family_batch_pending"
                next_gate = str(batch.get("next_gate") or "variant_specific_native_canary_manifest")
            updated.update(
                {
                    "turbocore_status": status,
                    "turbocore_route": "adamw_variant_dedicated_kernel_required",
                    "next_gate": next_gate,
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "AdamW variant family batch has been aggregated separately from exact AdamW.",
                        "Exact AdamW native readiness is not counted for variants.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adamw_variant_product_training_canary(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("representative_product_training_canary_ready") is not True:
        return rows
    canary_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        canary = _as_dict(canary_rows.get(optimizer_type))
        if canary and canary.get("canary_status") == "representative_product_training_canary_ready":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adamw_variant_representative_product_training_canary_ready",
                    "turbocore_route": "adamw_variant_dedicated_kernel_required",
                    "next_gate": "record explicit owner/release approval for AdamW variant native dispatch",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "AdamW variant representative product-training canary evidence is ready.",
                        "Product native dispatch remains blocked until explicit owner/release approval.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adamw_variant_owner_release_hold(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("owner_release_hold_ready") is not True:
        return rows
    hold = _as_dict(report.get("hold_manifest"))
    optimizer_types = set(_strings(hold.get("optimizer_types")))
    if not optimizer_types:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in optimizer_types:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adamw_variant_owner_release_hold_ready",
                    "turbocore_route": "adamw_variant_dedicated_kernel_required",
                    "next_gate": "record_explicit_adamw_variant_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Owner/release approval hold is ready for this AdamW variant.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adamw_variant_request_schema_ui(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("request_schema_ui_non_exposure_ready") is not True:
        return rows
    optimizer_count = int(_as_dict(report.get("summary")).get("optimizer_count", 0) or 0)
    if optimizer_count <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["turbocore_status"] == "adamw_variant_owner_release_hold_ready":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adamw_variant_request_schema_ui_non_exposure_ready",
                    "turbocore_route": "adamw_variant_non_exposure_default_off_v0",
                    "next_gate": "keep AdamW variant native dispatch unwired until explicit owner/release approval is recorded",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Request/schema/UI non-exposure has been audited for AdamW variants.",
                        "Product native dispatch remains disabled until owner/release approval is recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _adaptive_lr_batch_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_state_machine_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_replay_matrix_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_state_machine_replay_matrix_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_replay_executor_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_state_machine_replay_executor_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_abi_preconditions_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_native_state_machine_abi_preconditions_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_abi_skeleton_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_native_state_machine_abi_skeleton_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_cpu_guard_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_implementation_stub_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_native_state_machine_implementation_stub_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_cuda_contract_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_cuda_kernel_contract_plan_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_cuda_implementation_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_cuda_kernel_implementation_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_training_tensor_binding_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_training_tensor_binding_canary_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_runtime_dispatch_shadow_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_runtime_dispatch_shadow_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_training_loop_canary_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_training_loop_canary_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_e2e_shadow_matrix_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_e2e_shadow_matrix_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_canary_rollout_policy_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_canary_rollout_policy_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_dispatch_review_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_dispatch_integration_review_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_owner_release_hold_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_owner_release_hold_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _adaptive_lr_request_schema_ui_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_adaptive_lr_request_schema_ui_non_exposure_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _factored_custom_state_layout_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_factored_custom_optimizer_state_layout_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _factored_custom_family_batch_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_factored_custom_optimizer_family_batch_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _factored_custom_owner_release_hold_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_factored_custom_owner_release_hold_scorecard.json"
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _factored_custom_request_schema_ui_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_factored_custom_request_schema_ui_non_exposure_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _plugin_family_batch_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_plugin_optimizer_family_batch_scorecard.json"
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _plugin_adamlike_runtime_dispatch_rehearsal_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_adamlike_runtime_dispatch_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _plugin_schedulefree_runtime_dispatch_rehearsal_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_schedulefree_runtime_dispatch_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _plugin_adaptivelr_runtime_dispatch_rehearsal_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_adaptivelr_runtime_dispatch_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _plugin_closure_second_order_runtime_precondition_rehearsal_report(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_closure_second_order_runtime_precondition_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _plugin_custom_formula_runtime_precondition_rehearsal_report(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_custom_formula_runtime_precondition_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _plugin_factored_memory_runtime_precondition_rehearsal_report(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_factored_memory_runtime_precondition_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _plugin_model_shape_aware_runtime_precondition_rehearsal_report(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_model_shape_aware_runtime_precondition_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _plugin_state_adapter_special_runtime_precondition_rehearsal_report(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_state_adapter_special_runtime_precondition_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _plugin_fused_backward_runtime_precondition_rehearsal_report(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_fused_backward_runtime_precondition_rehearsal_scorecard.json"
    )
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_dict(payload)


def _optimizer_native_kernel_inventory_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_optimizer_native_kernel_inventory_scorecard.json"
    )
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    try:
        return _as_dict(build_optimizer_native_kernel_inventory_scorecard(write_artifact=True))
    except Exception:
        return {}


def _optimizer_family_kernel_contract_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_optimizer_family_kernel_contract_scorecard.json"
    )
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    try:
        return _as_dict(build_optimizer_family_kernel_contract_scorecard(write_artifact=True))
    except Exception:
        return {}


def _plugin_selected_family_owner_release_hold_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_selected_family_owner_release_hold_scorecard.json"
    )
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _plugin_selected_family_request_schema_ui_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_plugin_selected_family_request_schema_ui_non_exposure_scorecard.json"
    )
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _muon_model_shape_aware_batch_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_muon_model_shape_aware_family_batch_scorecard.json"
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _muon_native_scratch_kernel_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_muon_native_scratch_kernel_scorecard.json"
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _muon_training_tensor_binding_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_muon_training_tensor_binding_canary_scorecard.json"
    )
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _muon_training_loop_canary_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_muon_training_loop_canary_scorecard.json"
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _muon_e2e_shadow_matrix_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_muon_e2e_shadow_matrix_scorecard.json"
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _muon_canary_rollout_policy_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_muon_canary_rollout_policy_scorecard.json"
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _muon_dispatch_review_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_muon_dispatch_integration_review_scorecard.json"
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _muon_owner_release_hold_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_muon_owner_release_hold_scorecard.json"
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _muon_request_schema_ui_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is not None:
        return _as_dict(report)
    source = (
        REPO_ROOT
        / "temp"
        / "turbocore_optimizer"
        / "turbocore_muon_request_schema_ui_non_exposure_scorecard.json"
    )
    if source.exists():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _as_dict(payload)
    return {}


def _apply_simple_family_batch(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not report:
        return rows
    batch_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        batch = _as_dict(batch_rows.get(optimizer_type))
        if batch.get("batch_status") == "simple_formula_native_batch_canary_ready":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "simple_formula_native_batch_canary_ready",
                    "turbocore_route": str(batch.get("native_route") or "rust_cuda_simple_formula_runtime_v0"),
                    "next_gate": "representative_product_training_canary",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Family batch gate reports formula, ABI, registry, kernel, runtime, executor, dispatch, training-loop, and e2e canary evidence.",
                        "Product native dispatch remains disabled until a separate rollout decision.",
                    ],
                }
            )
            out.append(updated)
        elif batch.get("batch_status") in {
            "simple_formula_variant_schedule_free_native_canary_ready",
            "simple_formula_variant_quantized_dispatch_integration_review_ready",
            "simple_formula_variant_quantized_owner_approval_hold_ready",
            "simple_formula_variant_quantized_rollout_policy_ready",
            "simple_formula_variant_quantized_product_state_sync_ready",
            "simple_formula_variant_quantized_e2e_no_regression_ready",
            "simple_formula_variant_quantized_training_loop_canary_ready",
            "simple_formula_variant_quantized_training_loop_canary_manifest_ready",
            "simple_formula_variant_quantized_runtime_canary_manifest_ready",
            "simple_formula_variant_quantized_native_scratch_kernel_ready",
            "simple_formula_variant_quantized_formula_parity_ready",
            "simple_formula_variant_native_abi_spec_ready",
            "simple_formula_variant_layout_spec_ready",
            "simple_formula_variant_state_machine_reference_ready",
        }:
            batch_status = str(batch.get("batch_status") or "")
            is_state_machine = batch_status == "simple_formula_variant_state_machine_reference_ready"
            is_native_abi = batch_status == "simple_formula_variant_native_abi_spec_ready"
            is_schedule_free_native_canary = batch_status == "simple_formula_variant_schedule_free_native_canary_ready"
            is_quantized_product_state_sync_ready = (
                batch_status == "simple_formula_variant_quantized_product_state_sync_ready"
            )
            is_quantized_rollout_policy_ready = (
                batch_status == "simple_formula_variant_quantized_rollout_policy_ready"
            )
            is_quantized_dispatch_review_ready = (
                batch_status == "simple_formula_variant_quantized_dispatch_integration_review_ready"
            )
            is_quantized_owner_hold_ready = (
                batch_status == "simple_formula_variant_quantized_owner_approval_hold_ready"
            )
            is_quantized_training_loop_ready = (
                batch_status == "simple_formula_variant_quantized_training_loop_canary_ready"
            )
            is_quantized_e2e_ready = batch_status == "simple_formula_variant_quantized_e2e_no_regression_ready"
            is_quantized_native_scratch = (
                batch_status == "simple_formula_variant_quantized_native_scratch_kernel_ready"
            )
            is_quantized_runtime_manifest = (
                batch_status == "simple_formula_variant_quantized_runtime_canary_manifest_ready"
            )
            is_quantized_training_loop_manifest = (
                batch_status == "simple_formula_variant_quantized_training_loop_canary_manifest_ready"
            )
            is_quantized_formula_parity = batch_status == "simple_formula_variant_quantized_formula_parity_ready"
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": (
                        "simple_formula_variant_schedule_free_native_canary_ready"
                        if is_schedule_free_native_canary
                        else
                        "simple_formula_variant_quantized_dispatch_integration_review_ready"
                        if is_quantized_dispatch_review_ready
                        else
                        "simple_formula_variant_quantized_owner_approval_hold_ready"
                        if is_quantized_owner_hold_ready
                        else
                        "simple_formula_variant_quantized_rollout_policy_ready"
                        if is_quantized_rollout_policy_ready
                        else
                        "simple_formula_variant_quantized_product_state_sync_ready"
                        if is_quantized_product_state_sync_ready
                        else
                        "simple_formula_variant_quantized_e2e_no_regression_ready"
                        if is_quantized_e2e_ready
                        else
                        "simple_formula_variant_quantized_training_loop_canary_ready"
                        if is_quantized_training_loop_ready
                        else
                        "simple_formula_variant_quantized_training_loop_canary_manifest_ready"
                        if is_quantized_training_loop_manifest
                        else
                        "simple_formula_variant_quantized_runtime_canary_manifest_ready"
                        if is_quantized_runtime_manifest
                        else
                        "simple_formula_variant_quantized_native_scratch_kernel_ready"
                        if is_quantized_native_scratch
                        else
                        "simple_formula_variant_quantized_formula_parity_ready"
                        if is_quantized_formula_parity
                        else
                        "simple_formula_variant_native_abi_spec_ready"
                        if is_native_abi
                        else
                        "simple_formula_variant_state_machine_reference_ready"
                        if is_state_machine
                        else "simple_formula_variant_layout_spec_ready"
                    ),
                    "turbocore_route": str(batch.get("native_route") or "dedicated_variant_kernel_required"),
                    "next_gate": str(batch.get("next_gate") or "variant_native_abi_and_parity_matrix"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Simple-formula variant report-only state/layout evidence is ready.",
                        "Native ABI and parity matrix plans may be ready, but implementation is still pending.",
                        "Native kernel, runtime dispatch, and product exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_simple_product_training_canary(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("representative_product_training_canary_ready") is not True:
        return rows
    canary_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        canary = _as_dict(canary_rows.get(optimizer_type))
        if canary.get("canary_status") == "representative_product_training_canary_ready":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "simple_formula_representative_product_training_canary_ready",
                    "turbocore_route": str(canary.get("native_route") or "rust_cuda_simple_formula_runtime_v0"),
                    "next_gate": str(canary.get("next_gate") or "simple_formula_explicit_owner_release_review"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Representative product-training canary evidence is ready for this fp32 simple optimizer.",
                        "Product native dispatch remains disabled pending explicit owner/release approval.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_simple_owner_release_hold(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("owner_release_hold_ready") is not True:
        return rows
    hold = _as_dict(report.get("hold_manifest"))
    optimizer_types = set(_strings(hold.get("optimizer_types")))
    if not optimizer_types:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in optimizer_types:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "simple_formula_owner_release_hold_ready",
                    "turbocore_route": "rust_cuda_simple_formula_runtime_v0",
                    "next_gate": "record_explicit_simple_formula_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Owner/release approval hold is ready for this fp32 simple optimizer.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_simple_schedulefree_rollout_policy(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("canary_rollout_policy_ready") is not True:
        return rows
    rollout_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        rollout = _as_dict(rollout_rows.get(row["optimizer_type"]))
        if rollout.get("rollout_status") == "schedule_free_rollout_policy_ready":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "simple_formula_variant_schedule_free_rollout_policy_ready",
                    "turbocore_route": "schedulefree_state_machine_dedicated_kernel_required",
                    "next_gate": str(rollout.get("next_gate") or "simple_schedulefree_dispatch_integration_review"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Default-off rollout policy is ready for this built-in schedule-free variant.",
                        "Product native dispatch remains disabled pending dispatch review and owner approval.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_simple_schedulefree_dispatch_review(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("review_gate_ready") is not True or report.get("dispatch_integration_review") is not True:
        return rows
    review = _as_dict(report.get("review_package"))
    optimizer_types = set(_strings(review.get("optimizer_types")))
    if not optimizer_types:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in optimizer_types:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "simple_formula_variant_schedule_free_dispatch_integration_review_ready",
                    "turbocore_route": "schedulefree_state_machine_dedicated_kernel_required",
                    "next_gate": "record_explicit_simple_schedulefree_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Dispatch integration review evidence is ready for this built-in schedule-free variant.",
                        "Product native dispatch remains disabled until owner/release approval is recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_simple_schedulefree_owner_release_hold(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("owner_release_hold_ready") is not True:
        return rows
    hold = _as_dict(report.get("hold_manifest"))
    optimizer_types = set(_strings(hold.get("optimizer_types")))
    if not optimizer_types:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in optimizer_types:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "simple_formula_variant_schedule_free_owner_release_hold_ready",
                    "turbocore_route": "schedulefree_state_machine_dedicated_kernel_required",
                    "next_gate": "record_explicit_simple_schedulefree_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Owner/release approval hold is ready for this built-in schedule-free variant.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_simple_request_schema_ui(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("request_schema_ui_non_exposure_ready") is not True:
        return rows
    optimizer_types = set(_strings(report.get("target_optimizer_types")))
    if not optimizer_types:
        return rows
    hold_statuses = {
        "simple_formula_owner_release_hold_ready",
        "simple_formula_variant_quantized_owner_approval_hold_ready",
        "simple_formula_variant_schedule_free_owner_release_hold_ready",
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in optimizer_types and row["turbocore_status"] in hold_statuses:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "simple_formula_request_schema_ui_non_exposure_ready",
                    "next_gate": "record_explicit_simple_formula_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Request/schema/UI non-exposure audit is ready for this simple optimizer.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_batch(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = _as_dict(report.get("summary"))
    if int(summary.get("state_machine_reference_ready_count", 0) or 0) <= 0:
        return rows
    batch_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        batch = _as_dict(batch_rows.get(optimizer_type))
        if batch.get("state_machine_reference_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_state_machine_reference_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(batch.get("next_gate") or "adaptive_lr_native_state_machine_abi_review"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR state-machine reference batch is ready.",
                        "Native ABI, resume parity, and CUDA/runtime dispatch are still pending.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_replay_matrix(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("state_machine_replay_matrix_ready") is not True:
        return rows
    replay_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        replay = _as_dict(replay_rows.get(optimizer_type))
        if replay.get("state_machine_replay_matrix_artifact_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_state_machine_replay_matrix_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(replay.get("next_gate") or "adaptive_lr_state_machine_abi_replay_executor"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR replay/resume matrix artifact is ready.",
                        "Native ABI implementation, runtime dispatch, and CUDA kernels are still pending.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_replay_executor(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("state_machine_replay_executor_ready") is not True:
        return rows
    executor_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        executor = _as_dict(executor_rows.get(optimizer_type))
        if executor.get("reference_replay_executor_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_state_machine_replay_executor_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(executor.get("next_gate") or "adaptive_lr_native_state_machine_abi_preconditions"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR trainer-path replay executor parity is ready.",
                        "Native ABI implementation, runtime dispatch, and CUDA kernels are still pending.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_abi_preconditions(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("native_state_machine_abi_preconditions_ready") is not True:
        return rows
    precondition_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        preconditions = _as_dict(precondition_rows.get(optimizer_type))
        if preconditions.get("native_state_machine_abi_precondition_review_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_native_state_machine_abi_precondition_review_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(
                        preconditions.get("next_gate") or "adaptive_lr_native_state_machine_abi_skeleton"
                    ),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR native state-machine ABI precondition review is ready.",
                        "Native ABI implementation, runtime dispatch, CUDA kernels, request/schema/UI, and product exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_abi_skeleton(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("native_state_machine_abi_skeleton_ready") is not True:
        return rows
    skeleton_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        skeleton = _as_dict(skeleton_rows.get(optimizer_type))
        if skeleton.get("native_state_machine_abi_skeleton_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_native_state_machine_abi_skeleton_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(
                        skeleton.get("next_gate") or "adaptive_lr_native_state_machine_cpu_reference_guard"
                    ),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR native state-machine ABI skeleton is ready.",
                        "CPU reference guard, native ABI implementation, runtime dispatch, CUDA kernels, request/schema/UI, and product exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_cpu_guard(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("native_state_machine_cpu_reference_guard_ready") is not True:
        return rows
    guard_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        guard = _as_dict(guard_rows.get(optimizer_type))
        if guard.get("cpu_reference_guard_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_native_state_machine_cpu_reference_guard_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(
                        guard.get("next_gate") or "adaptive_lr_native_state_machine_implementation_stub"
                    ),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR native state-machine CPU reference guard evidence is ready.",
                        "Native ABI implementation, runtime dispatch, CUDA kernels, request/schema/UI, and product exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_implementation_stub(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("native_state_machine_implementation_stub_ready") is not True:
        return rows
    stub_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        stub = _as_dict(stub_rows.get(optimizer_type))
        if stub.get("implementation_stub_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_native_state_machine_implementation_stub_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(stub.get("next_gate") or "adaptive_lr_cuda_kernel_contract_plan"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR native state-machine implementation stub evidence is ready.",
                        "Registered native ABI, runtime dispatch, CUDA kernels, request/schema/UI, and product exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_cuda_contract(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("cuda_kernel_contract_plan_ready") is not True:
        return rows
    contract_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        contract = _as_dict(contract_rows.get(optimizer_type))
        if contract.get("cuda_kernel_contract_plan_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_cuda_kernel_contract_plan_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(contract.get("next_gate") or "adaptive_lr_cuda_kernel_implementation"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR CUDA kernel contract and runtime canary manifest plan are ready.",
                        "CUDA implementation, runtime canary execution, request/schema/UI, and product exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_cuda_implementation(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("cuda_kernel_implementation_ready") is not True:
        return rows
    impl_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        impl = _as_dict(impl_rows.get(optimizer_type))
        if impl.get("cuda_kernel_implementation_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_cuda_kernel_implementation_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(impl.get("next_gate") or "adaptive_lr_runtime_tensor_binding_canary"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR CUDA scratch-kernel implementation evidence is ready.",
                        "Runtime tensor binding, request/schema/UI, runtime dispatch, and product exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_training_tensor_binding(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("training_tensor_binding_canary_ready") is not True:
        return rows
    binding_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        binding = _as_dict(binding_rows.get(optimizer_type))
        if binding.get("training_tensor_binding_canary_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_training_tensor_binding_canary_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(binding.get("next_gate") or "adaptive_lr_training_loop_canary"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR live tensor-binding canary evidence is ready on isolated CUDA tensors.",
                        "TrainingLoop canary, request/schema/UI, runtime dispatch, and product exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_runtime_dispatch_shadow(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("runtime_dispatch_shadow_ready") is not True:
        return rows
    shadow_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        shadow = _as_dict(shadow_rows.get(optimizer_type))
        if shadow.get("runtime_dispatch_shadow_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_runtime_dispatch_shadow_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(shadow.get("next_gate") or "adaptive_lr_training_loop_canary"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR runtime/request shadow envelope is ready with Python fallback authoritative.",
                        "Native shadow calls, TrainingLoop dispatch, request/schema/UI, and product exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_training_loop_canary(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("training_loop_canary_ready") is not True:
        return rows
    loop_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        loop = _as_dict(loop_rows.get(optimizer_type))
        if loop.get("training_loop_canary_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_training_loop_canary_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(loop.get("next_gate") or "adaptive_lr_e2e_shadow_matrix"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR explicit TrainingLoop canary executed native family representative steps.",
                        "Product native dispatch, request/schema/UI, rollout policy, and release promotion remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_e2e_shadow_matrix(rows: list[dict[str, Any]], report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("e2e_shadow_matrix_ready") is not True:
        return rows
    matrix_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        matrix = _as_dict(matrix_rows.get(optimizer_type))
        if matrix.get("e2e_shadow_matrix_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_e2e_shadow_matrix_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(matrix.get("next_gate") or "adaptive_lr_canary_rollout_policy"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR e2e shadow matrix scaffold is ready with Python fallback authoritative.",
                        "Live shadow matrix execution, rollout policy, request/schema/UI, and product native dispatch remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_canary_rollout_policy(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("canary_rollout_policy_ready") is not True:
        return rows
    policy_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        optimizer_type = row["optimizer_type"]
        policy = _as_dict(policy_rows.get(optimizer_type))
        if policy.get("canary_rollout_policy_ready") is True:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_canary_rollout_policy_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": str(policy.get("next_gate") or "adaptive_lr_dispatch_integration_review"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Adaptive LR default-off canary rollout policy is ready.",
                        "Manual dispatch integration review, request/schema/UI, and product native dispatch remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_dispatch_review(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("review_gate_ready") is not True or report.get("dispatch_integration_review") is not True:
        return rows
    review = _as_dict(report.get("review_package"))
    optimizer_types = set(_strings(review.get("optimizer_types")))
    if not optimizer_types:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in optimizer_types:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_dispatch_integration_review_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": "record_explicit_adaptive_lr_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Dispatch integration review evidence is ready for this built-in adaptive-LR optimizer.",
                        "Product native dispatch remains disabled until owner/release approval is recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_owner_release_hold(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("owner_release_hold_ready") is not True:
        return rows
    hold = _as_dict(report.get("hold_manifest"))
    optimizer_types = set(_strings(hold.get("optimizer_types")))
    if not optimizer_types:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in optimizer_types:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_owner_release_hold_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": "record_explicit_adaptive_lr_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Owner/release approval hold is ready for this built-in adaptive-LR optimizer.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_adaptive_lr_request_schema_ui(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("request_schema_ui_non_exposure_ready") is not True:
        return rows
    optimizer_count = int(_as_dict(report.get("summary")).get("optimizer_count", 0) or 0)
    if optimizer_count <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["turbocore_status"] == "adaptive_lr_owner_release_hold_ready":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "adaptive_lr_request_schema_ui_non_exposure_ready",
                    "turbocore_route": "adaptive_lr_state_machine_reference_v0",
                    "next_gate": "record_explicit_adaptive_lr_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Request/schema/UI non-exposure audit is ready for this built-in adaptive-LR optimizer.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_factored_custom_state_layout(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("state_layout_reference_ready") is not True:
        return rows
    layout_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        layout = _as_dict(layout_rows.get(row["optimizer_type"]))
        if layout:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "factored_custom_state_layout_reference_ready",
                    "turbocore_route": "factored_custom_layout_reference_v0",
                    "next_gate": str(
                        report.get("recommended_next_step")
                        or "native layout ABI and short full-finetune quality matrix"
                    ),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Factored/custom state-layout reference is ready for this built-in optimizer.",
                        "Exact AdamW kernel reuse remains blocked; product native dispatch remains disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_factored_custom_family_batch(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not report:
        return rows
    batch_rows = {
        str(row.get("optimizer_type") or ""): row
        for row in report.get("rows", [])
        if isinstance(row, Mapping)
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        batch = _as_dict(batch_rows.get(row["optimizer_type"]))
        if batch:
            status = str(batch.get("batch_status") or "factored_custom_state_layout_reference_ready")
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": status,
                    "turbocore_route": "factored_custom_native_canary_chain_v0",
                    "next_gate": str(batch.get("next_gate") or "record_explicit_factored_custom_owner_release_approval"),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Factored/custom native canary chain evidence is aggregated by family batch.",
                        "Dispatch review readiness remains default-off and is not owner/release approval.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_factored_custom_owner_release_hold(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("owner_release_hold_ready") is not True:
        return rows
    hold = _as_dict(report.get("hold_manifest"))
    optimizer_types = set(_strings(hold.get("optimizer_types")))
    if not optimizer_types:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in optimizer_types:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "factored_custom_owner_release_hold_ready",
                    "turbocore_route": "factored_custom_native_canary_chain_v0",
                    "next_gate": "record_explicit_factored_custom_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Owner/release approval hold is ready for this built-in factored/custom optimizer.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_factored_custom_request_schema_ui(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("request_schema_ui_non_exposure_ready") is not True:
        return rows
    optimizer_count = int(_as_dict(report.get("summary")).get("optimizer_count", 0) or 0)
    if optimizer_count <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["turbocore_status"] == "factored_custom_owner_release_hold_ready":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "factored_custom_request_schema_ui_non_exposure_ready",
                    "turbocore_route": "factored_custom_native_canary_chain_v0",
                    "next_gate": "record_explicit_factored_custom_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Request/schema/UI non-exposure audit is ready for this built-in factored/custom optimizer.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_muon_model_shape_aware_batch(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("muon_model_shape_aware_family_batch_ready") is not True:
        return rows
    summary = _as_dict(report.get("summary"))
    dispatch_review_ready = int(summary.get("dispatch_integration_review_ready_count", 0) or 0) > 0
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] == OptimizerType.MUON.value and row["turbocore_status"] == "model_shape_aware_research":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": (
                        "model_shape_aware_dispatch_review_ready"
                        if dispatch_review_ready
                        else "model_shape_aware_param_group_abi_ready"
                    ),
                    "turbocore_route": (
                        "builtin_muon_model_shape_dispatch_review_v0"
                        if dispatch_review_ready
                        else "builtin_muon_model_shape_param_group_abi_v0"
                    ),
                    "next_gate": (
                        "explicit_owner_release_approval_for_builtin_muon_native_dispatch"
                        if dispatch_review_ready
                        else "owner_release_hold_for_builtin_muon_native_kernel"
                    ),
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Built-in Muon param-group ABI, native preconditions, and runtime shadow review are ready.",
                        "Product native dispatch remains disabled until owner/release approval and native kernel work.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_muon_native_scratch_kernel(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("native_scratch_kernel_ready") is not True:
        return rows
    summary = _as_dict(report.get("summary"))
    if int(summary.get("native_scratch_kernel_ready_count", 0) or 0) <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] == OptimizerType.MUON.value and row["turbocore_status"] in {
            "model_shape_aware_dispatch_review_ready",
            "model_shape_aware_param_group_abi_ready",
        }:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "model_shape_aware_native_scratch_kernel_ready",
                    "turbocore_route": "builtin_muon_model_shape_native_scratch_kernel_v0",
                    "next_gate": "muon_runtime_tensor_binding_canary_default_off",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Built-in Muon diagnostic CUDA scratch kernel parity is ready.",
                        "The probe uses synthetic buffers only; product native dispatch remains disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_muon_training_tensor_binding(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("training_tensor_binding_canary_ready") is not True:
        return rows
    summary = _as_dict(report.get("summary"))
    if int(summary.get("training_tensor_binding_canary_ready_count", 0) or 0) <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] == OptimizerType.MUON.value and row["turbocore_status"] in {
            "model_shape_aware_native_scratch_kernel_ready",
            "model_shape_aware_dispatch_review_ready",
            "model_shape_aware_param_group_abi_ready",
        }:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "model_shape_aware_training_tensor_binding_canary_ready",
                    "turbocore_route": "builtin_muon_training_tensor_binding_canary_v0",
                    "next_gate": "muon_training_loop_canary_default_off",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Built-in Muon live tensor-binding canary is ready on isolated CUDA tensors.",
                        "Product TrainingLoop/native dispatch remains disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_muon_training_loop_canary(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("training_loop_canary_ready") is not True:
        return rows
    summary = _as_dict(report.get("summary"))
    if int(summary.get("training_loop_canary_ready_count", 0) or 0) <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] == OptimizerType.MUON.value and row["turbocore_status"] in {
            "model_shape_aware_training_tensor_binding_canary_ready",
            "model_shape_aware_native_scratch_kernel_ready",
            "model_shape_aware_dispatch_review_ready",
            "model_shape_aware_param_group_abi_ready",
        }:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "model_shape_aware_training_loop_canary_ready",
                    "turbocore_route": "builtin_muon_training_loop_canary_v0",
                    "next_gate": "muon_e2e_shadow_matrix_default_off",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Built-in Muon TrainingLoop canary executed a native step on isolated CUDA tensors.",
                        "Product request/schema/UI, runtime dispatch, native dispatch, and release exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_muon_e2e_shadow_matrix(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("e2e_shadow_matrix_ready") is not True:
        return rows
    summary = _as_dict(report.get("summary"))
    if int(summary.get("e2e_shadow_matrix_ready_count", 0) or 0) <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] == OptimizerType.MUON.value and row["turbocore_status"] in {
            "model_shape_aware_training_loop_canary_ready",
            "model_shape_aware_training_tensor_binding_canary_ready",
            "model_shape_aware_native_scratch_kernel_ready",
            "model_shape_aware_dispatch_review_ready",
            "model_shape_aware_param_group_abi_ready",
        }:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "model_shape_aware_e2e_shadow_matrix_ready",
                    "turbocore_route": "builtin_muon_e2e_shadow_matrix_v0",
                    "next_gate": "muon_canary_rollout_policy_default_off",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Built-in Muon e2e shadow matrix scaffold is ready in report-only mode.",
                        "No native shadow call mutates product training state.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_muon_canary_rollout_policy(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("canary_rollout_policy_ready") is not True:
        return rows
    summary = _as_dict(report.get("summary"))
    if int(summary.get("canary_rollout_policy_ready_count", 0) or 0) <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] == OptimizerType.MUON.value and row["turbocore_status"] in {
            "model_shape_aware_e2e_shadow_matrix_ready",
            "model_shape_aware_training_loop_canary_ready",
            "model_shape_aware_training_tensor_binding_canary_ready",
            "model_shape_aware_native_scratch_kernel_ready",
            "model_shape_aware_dispatch_review_ready",
            "model_shape_aware_param_group_abi_ready",
        }:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "model_shape_aware_canary_rollout_policy_ready",
                    "turbocore_route": "builtin_muon_canary_rollout_policy_v0",
                    "next_gate": "muon_owner_release_hold_default_off",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Built-in Muon canary rollout policy is default-off and report-only.",
                        "Product native dispatch remains disabled until explicit owner/release approval.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_muon_dispatch_review(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("dispatch_integration_review") is not True:
        return rows
    summary = _as_dict(report.get("summary"))
    if int(summary.get("optimizer_count", 0) or 0) <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] == OptimizerType.MUON.value and row["turbocore_status"] in {
            "model_shape_aware_canary_rollout_policy_ready",
            "model_shape_aware_e2e_shadow_matrix_ready",
            "model_shape_aware_training_loop_canary_ready",
            "model_shape_aware_training_tensor_binding_canary_ready",
            "model_shape_aware_native_scratch_kernel_ready",
            "model_shape_aware_dispatch_review_ready",
            "model_shape_aware_param_group_abi_ready",
        }:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "model_shape_aware_dispatch_integration_review_ready",
                    "turbocore_route": "builtin_muon_dispatch_integration_review_v0",
                    "next_gate": "muon_owner_release_hold_default_off",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Built-in Muon dispatch integration review is ready in default-off mode.",
                        "Product runtime dispatch, native dispatch, and request/schema/UI exposure remain disabled.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_muon_owner_release_hold(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("owner_release_hold_ready") is not True:
        return rows
    hold = _as_dict(report.get("hold_manifest"))
    optimizer_types = set(_strings(hold.get("optimizer_types")))
    if not optimizer_types:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in optimizer_types:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "model_shape_aware_owner_release_hold_ready",
                    "turbocore_route": "builtin_muon_model_shape_owner_release_hold_v0",
                    "next_gate": "record_explicit_muon_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Owner/release approval hold is ready for built-in Muon.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_muon_request_schema_ui(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if report.get("request_schema_ui_non_exposure_ready") is not True:
        return rows
    optimizer_count = int(_as_dict(report.get("summary")).get("optimizer_count", 0) or 0)
    if optimizer_count <= 0:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["turbocore_status"] == "model_shape_aware_owner_release_hold_ready":
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": "model_shape_aware_request_schema_ui_non_exposure_ready",
                    "turbocore_route": "builtin_muon_model_shape_non_exposure_v0",
                    "next_gate": "record_explicit_muon_owner_release_approval",
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": [
                        "Request/schema/UI non-exposure audit is ready for built-in Muon.",
                        "Product native dispatch remains disabled until approval artifacts are recorded.",
                    ],
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _apply_plugin_selector_coverage(
    rows: list[dict[str, Any]],
    selector_scorecard: Mapping[str, Any],
    plugin_batch: Mapping[str, Any],
    plugin_owner_hold: Mapping[str, Any],
    plugin_request_schema_ui: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if selector_scorecard.get("plugin_selector_classification_ready") is not True:
        return rows
    selector_summary = _as_dict(selector_scorecard.get("summary"))
    if int(selector_scorecard.get("missing_classification_count", 0) or 0) > 0:
        return rows
    if int(selector_summary.get("missing_resume_count", 0) or 0) > 0:
        return rows

    plugin_summary = _as_dict(plugin_batch.get("summary"))
    selected_ready = int(plugin_summary.get("selected_optimizer_gate_ready_count", 0) or 0)
    selected_pending = int(plugin_summary.get("selected_optimizer_gate_pending_count", 0) or 0)
    next_gate = _plugin_next_gate(selector_scorecard, plugin_batch, plugin_owner_hold, plugin_request_schema_ui)
    status = "plugin_selector_classification_ready_default_off"
    route = "selected_optimizer_subroute_classified_default_off"
    notes = [
        "Plugin selector classification and resume evidence are ready.",
        "Each selected implementation still resolves through family-specific evidence before native dispatch.",
    ]
    if plugin_batch.get("plugin_optimizer_family_batch_ready") is True and selected_ready > 0 and selected_pending == 0:
        status = "plugin_selected_family_coverage_ready_default_off"
        route = "selected_plugin_family_coverage_default_off_v0"
        notes = [
            "All selected plugin optimizer families have coverage evidence.",
            "Native dispatch remains unwired until explicit owner/release approval is recorded.",
        ]
    if plugin_owner_hold.get("owner_release_hold_ready") is True:
        status = "plugin_selected_family_owner_release_hold_ready_default_off"
        route = "selected_plugin_family_owner_release_hold_default_off_v0"
        notes = [
            "Selected plugin optimizer families have owner/release hold evidence.",
            "Owner approval is not recorded and native dispatch remains disabled.",
        ]
    if plugin_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True:
        status = "plugin_selected_family_non_exposure_ready_default_off"
        route = "selected_plugin_family_non_exposure_default_off_v0"
        notes = [
            "Selected plugin optimizer families have request/schema/UI non-exposure evidence.",
            "Product native dispatch remains unwired until explicit owner/release approval is recorded.",
        ]

    out: list[dict[str, Any]] = []
    for row in rows:
        if row["optimizer_type"] in {item.value for item in _SELECTOR_TYPES}:
            updated = dict(row)
            updated.update(
                {
                    "turbocore_status": status,
                    "turbocore_route": route,
                    "next_gate": next_gate,
                    "training_path_enabled": False,
                    "default_behavior_changed": False,
                    "notes": notes,
                }
            )
            out.append(updated)
        else:
            out.append(row)
    return out


def _row(
    capability: Mapping[str, Any],
    *,
    status: str,
    route: str,
    next_gate: str,
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "optimizer_type": str(capability.get("optimizer_type", "") or ""),
        "capability_status": str(capability.get("status", "") or ""),
        "capability_family": str(capability.get("family", "") or ""),
        "implementation": str(capability.get("implementation", "") or ""),
        "dependency": str(capability.get("dependency", "") or ""),
        "dependency_available": bool(capability.get("dependency_available", False)),
        "turbocore_status": status,
        "turbocore_route": route,
        "next_gate": next_gate,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "notes": list(notes),
    }


def _plugin_summary(
    rows: list[dict[str, Any]],
    selector_scorecard: Mapping[str, Any],
    plugin_batch: Mapping[str, Any],
    plugin_owner_hold: Mapping[str, Any],
    plugin_request_schema_ui: Mapping[str, Any],
) -> dict[str, Any]:
    selector = next((row for row in rows if row["optimizer_type"] == OptimizerType.PYTORCH_OPTIMIZER.value), {})
    capability_report = optimizer_capability_report([OptimizerType.PYTORCH_OPTIMIZER])
    optimizer = {}
    for item in capability_report.get("optimizers", []):
        if isinstance(item, dict) and item.get("optimizer_type") == OptimizerType.PYTORCH_OPTIMIZER.value:
            optimizer = item
            break
    plugin_names = [str(item) for item in optimizer.get("plugin_optimizers", []) if str(item)]
    resume = [str(item) for item in optimizer.get("plugin_resume_smoke_passed", []) if str(item)]
    special = dict(optimizer.get("plugin_special_handling", {}) or {})
    return {
        "selector_present": bool(selector),
        "plugin_optimizer_count": len(plugin_names),
        "plugin_resume_proven_count": len(resume),
        "plugin_special_handling_count": len(special),
        "plugin_selector_classification_ready": bool(selector_scorecard.get("plugin_selector_classification_ready", False)),
        "plugin_selector_missing_classification_count": int(selector_scorecard.get("missing_classification_count", 0) or 0),
        "plugin_selector_missing_resume_count": int(_as_dict(selector_scorecard.get("summary")).get("missing_resume_count", 0) or 0),
        "plugin_selector_route_family_counts": dict(_as_dict(_as_dict(selector_scorecard.get("summary")).get("route_family_counts"))),
        "resume_proven_examples": sorted(resume)[:12],
        "special_handling_examples": sorted(special)[:12],
        "next_gate": _plugin_next_gate(selector_scorecard, plugin_batch, plugin_owner_hold, plugin_request_schema_ui),
    }


def _plugin_next_gate(
    selector_scorecard: Mapping[str, Any],
    plugin_batch: Mapping[str, Any],
    plugin_owner_hold: Mapping[str, Any],
    plugin_request_schema_ui: Mapping[str, Any],
) -> str:
    if plugin_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True:
        return "keep plugin selected-family native dispatch unwired until explicit owner/release approval is recorded"
    if plugin_owner_hold.get("owner_release_hold_ready") is True:
        return "add plugin selected-family request/schema/UI non-exposure guard"

    selector_summary = _as_dict(selector_scorecard.get("summary"))
    missing_classification = int(selector_scorecard.get("missing_classification_count", 0) or 0)
    missing_resume = int(selector_summary.get("missing_resume_count", 0) or 0)
    if missing_classification or missing_resume:
        return "classify high-use plugin optimizers into AdamW-like, simple formula, factored, closure, or unsupported families"

    summary = _as_dict(plugin_batch.get("summary"))
    count = int(summary.get("selected_custom_formula_optimizer_count", 0) or 0)
    ready = int(summary.get("selected_custom_formula_formula_parity_matrix_implementation_ready_count", 0) or 0)
    if ready < count:
        return f"implement custom-formula parity/resume matrices for {count} selected plugin optimizers"

    count = int(summary.get("selected_adaptivelr_optimizer_count", 0) or 0)
    abi = int(summary.get("selected_adaptivelr_state_machine_abi_implementation_ready_count", 0) or 0)
    replay = int(summary.get("selected_adaptivelr_state_machine_replay_matrix_implementation_ready_count", 0) or 0)
    if abi < count or replay < count:
        return f"implement adaptive-LR state-machine ABI replay matrices for {count} selected plugin optimizers"

    count = int(summary.get("selected_simple_formula_optimizer_count", 0) or 0)
    ready = int(summary.get("selected_simple_formula_native_canary_ready_count", 0) or 0)
    if ready < count:
        return f"complete simple-formula selected plugin canaries for {count} selected plugin optimizers"

    count = int(summary.get("selected_factored_memory_optimizer_count", 0) or 0)
    ready = int(summary.get("selected_factored_memory_formula_tensor_binding_matrix_implementation_ready_count", 0) or 0)
    if ready < count:
        return f"implement factored-memory tensor-binding matrices for {count} selected plugin optimizers"
    count = int(summary.get("selected_model_shape_aware_optimizer_count", 0) or 0)
    ready = int(summary.get("selected_model_shape_aware_param_group_abi_implementation_ready_count", 0) or 0)
    if ready < count:
        return f"implement model-shape-aware param-group ABI replay matrices for {count} selected plugin optimizers"
    count = int(summary.get("selected_closure_second_order_optimizer_count", 0) or 0)
    ready = int(summary.get("selected_closure_second_order_training_loop_abi_implementation_ready_count", 0) or 0)
    if ready < count:
        return f"implement closure/second-order training-loop ABI matrices for {count} selected plugin optimizers"
    count = int(summary.get("selected_fused_backward_optimizer_count", 0) or 0)
    abi = int(summary.get("selected_fused_backward_abi_implementation_ready_count", 0) or 0)
    resume = int(summary.get("selected_fused_backward_resume_parity_matrix_implementation_ready_count", 0) or 0)
    if abi < count or resume < count:
        return f"implement fused-backward ABI replay matrices for {count} selected plugin optimizers"
    count = int(summary.get("selected_state_adapter_special_optimizer_count", 0) or 0)
    abi = int(summary.get("selected_state_adapter_special_adapter_abi_implementation_ready_count", 0) or 0)
    resume = int(summary.get("selected_state_adapter_special_resume_matrix_implementation_ready_count", 0) or 0)
    if abi < count or resume < count:
        return f"implement state-adapter ABI replay matrices for {count} selected plugin optimizers"
    return "prepare plugin selected-family owner/release hold with product dispatch still default-off"


def _compact_selector_scorecard(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": bool(report.get("ok", False)),
        "plugin_selector_classification_ready": bool(report.get("plugin_selector_classification_ready", False)),
        "selector_boundary_ready": bool(report.get("selector_boundary_ready", False)),
        "all_discovered_plugins_resume_proven": bool(report.get("all_discovered_plugins_resume_proven", False)),
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "missing_resume_count": int(summary.get("missing_resume_count", 0) or 0),
        "missing_classification_count": int(report.get("missing_classification_count", 0) or 0),
        "route_family_counts": dict(_as_dict(summary.get("route_family_counts"))),
        "promotion_blockers": _strings(report.get("promotion_blockers")),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _selector_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return ["plugin_optimizer_selector_scorecard_missing"]
    blocked: list[str] = []
    if report.get("plugin_selector_classification_ready") is not True:
        blocked.append("plugin_optimizer_selector_classification_not_ready")
    if report.get("selector_boundary_ready") is not True:
        blocked.append("plugin_optimizer_selector_boundary_not_ready")
    if report.get("all_discovered_plugins_resume_proven") is not True:
        blocked.append("plugin_optimizer_selector_resume_not_proven")
    if report.get("native_dispatch_allowed") is True or report.get("runtime_dispatch_ready") is True:
        blocked.append("plugin_optimizer_selector_unsafe_dispatch_claim")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _selector_scorecard_from_plugin_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    selector = _as_dict(report.get("selector_scorecard"))
    if not selector:
        return {}
    summary = {
        "plugin_optimizer_count": int(selector.get("plugin_optimizer_count", 0) or 0),
        "missing_resume_count": int(selector.get("missing_resume_count", 0) or 0),
        "route_family_counts": dict(_as_dict(selector.get("route_family_counts"))),
    }
    return {
        "ok": selector.get("ok") is True,
        "plugin_selector_classification_ready": selector.get("plugin_selector_classification_ready") is True,
        "selector_boundary_ready": selector.get("selector_boundary_ready") is True,
        "all_discovered_plugins_resume_proven": selector.get("all_discovered_plugins_resume_proven") is True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "missing_classification_count": int(selector.get("missing_classification_count", 0) or 0),
        "summary": summary,
        "blocked_reasons": [],
        "promotion_blockers": [],
    }


def _plugin_batch_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("plugin_optimizer_family_batch_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("plugin_optimizer_family_batch_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("plugin_optimizer_family_batch_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("plugin_optimizer_family_batch_native_dispatch_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _plugin_selected_family_owner_release_hold_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("plugin_selected_family_owner_release_hold_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("plugin_selected_family_owner_release_hold_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("plugin_selected_family_owner_release_hold_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("plugin_selected_family_owner_release_hold_native_dispatch_allowed")
    if report.get("request_fields_emitted") is True:
        blocked.append("plugin_selected_family_owner_release_hold_request_fields_emitted")
    if report.get("schema_exposure_allowed") is True:
        blocked.append("plugin_selected_family_owner_release_hold_schema_exposure_allowed")
    if report.get("ui_exposure_allowed") is True:
        blocked.append("plugin_selected_family_owner_release_hold_ui_exposure_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _plugin_selected_family_request_schema_ui_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("plugin_selected_family_request_schema_ui_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("plugin_selected_family_request_schema_ui_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("plugin_selected_family_request_schema_ui_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("plugin_selected_family_request_schema_ui_native_dispatch_allowed")
    if report.get("request_fields_emitted") is True:
        blocked.append("plugin_selected_family_request_schema_ui_request_fields_emitted")
    if report.get("schema_exposure_allowed") is True:
        blocked.append("plugin_selected_family_request_schema_ui_schema_exposure_allowed")
    if report.get("ui_exposure_allowed") is True:
        blocked.append("plugin_selected_family_request_schema_ui_ui_exposure_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _exact_adamw_stream_event_chain_abi_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_training_path_enabled")
    if report.get("training_dispatch") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_training_dispatch_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_default_behavior_changed")
    if report.get("runtime_dispatch_allowed") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_runtime_dispatch_allowed")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_native_dispatch_allowed")
    if report.get("product_exposure_allowed") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_product_exposure_allowed")
    if report.get("request_fields_emitted") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_request_fields_emitted")
    if report.get("schema_exposure_allowed") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_schema_exposure_allowed")
    if report.get("ui_exposure_allowed") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_ui_exposure_allowed")
    if report.get("backend_router_registered") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_backend_router_registered")
    if report.get("sync_fast_path_allowed") is True:
        blocked.append("exact_adamw_stream_event_chain_abi_sync_fast_path_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _muon_model_shape_aware_batch_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("muon_model_shape_aware_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("muon_model_shape_aware_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("muon_model_shape_aware_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("muon_model_shape_aware_native_dispatch_allowed")
    if report.get("native_kernel_ready") is True:
        blocked.append("muon_model_shape_aware_native_kernel_ready")
    if report.get("product_native_ready") is True or report.get("product_native_dispatch_ready") is True:
        blocked.append("muon_model_shape_aware_product_native_ready")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _muon_native_scratch_kernel_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("muon_native_scratch_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("muon_native_scratch_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("muon_native_scratch_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("muon_native_scratch_native_dispatch_allowed")
    if report.get("product_native_ready") is True or int(report.get("product_native_ready_count", 0) or 0) > 0:
        blocked.append("muon_native_scratch_product_native_ready")
    case = _as_dict(report.get("case"))
    if case.get("training_tensor_binding") is True:
        blocked.append("muon_native_scratch_training_tensor_binding")
    if case.get("training_dispatch") is True:
        blocked.append("muon_native_scratch_training_dispatch")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _muon_training_tensor_binding_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("muon_training_tensor_binding_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("muon_training_tensor_binding_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("muon_training_tensor_binding_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("muon_training_tensor_binding_native_dispatch_allowed")
    if report.get("product_native_ready") is True or int(report.get("product_native_ready_count", 0) or 0) > 0:
        blocked.append("muon_training_tensor_binding_product_native_ready")
    live = _as_dict(report.get("live_probe"))
    if live.get("training_dispatch") is True:
        blocked.append("muon_training_tensor_binding_training_dispatch")
    if live.get("training_path_enabled") is True:
        blocked.append("muon_training_tensor_binding_live_training_path_enabled")
    if live.get("native_dispatch_allowed") is True:
        blocked.append("muon_training_tensor_binding_live_native_dispatch_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _muon_training_loop_canary_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("muon_training_loop_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("muon_training_loop_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("muon_training_loop_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("muon_training_loop_native_dispatch_allowed")
    if report.get("product_native_ready") is True:
        blocked.append("muon_training_loop_product_native_ready")
    summary = _as_dict(report.get("summary"))
    if int(summary.get("product_native_ready_count", 0) or 0) > 0:
        blocked.append("muon_training_loop_product_native_ready_count")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _muon_e2e_shadow_matrix_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("muon_e2e_shadow_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("muon_e2e_shadow_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("muon_e2e_shadow_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("muon_e2e_shadow_native_dispatch_allowed")
    if report.get("product_native_ready") is True:
        blocked.append("muon_e2e_shadow_product_native_ready")
    if report.get("live_shadow_matrix_executed") is True:
        blocked.append("muon_e2e_shadow_live_matrix_executed")
    if report.get("native_shadow_updates_original") is True:
        blocked.append("muon_e2e_shadow_mutated_original")
    if report.get("native_shadow_training_mutates_authority") is True:
        blocked.append("muon_e2e_shadow_mutated_authority")
    summary = _as_dict(report.get("summary"))
    if int(summary.get("product_native_ready_count", 0) or 0) > 0:
        blocked.append("muon_e2e_shadow_product_native_ready_count")
    if int(summary.get("failed_case_count", 0) or 0) > 0:
        blocked.append("muon_e2e_shadow_failed_cases")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _muon_canary_rollout_policy_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("muon_canary_rollout_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("muon_canary_rollout_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("muon_canary_rollout_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("muon_canary_rollout_native_dispatch_allowed")
    if report.get("product_native_ready") is True:
        blocked.append("muon_canary_rollout_product_native_ready")
    if report.get("canary_auto_enabled") is True:
        blocked.append("muon_canary_rollout_auto_enabled")
    policy = _as_dict(report.get("policy"))
    if policy.get("canary_enabled_by_default") is True:
        blocked.append("muon_canary_rollout_enabled_by_default")
    if float(policy.get("max_canary_fraction_default", 0.0) or 0.0) != 0.0:
        blocked.append("muon_canary_rollout_default_fraction_nonzero")
    if policy.get("explicit_opt_in_required") is not True:
        blocked.append("muon_canary_rollout_missing_explicit_opt_in")
    summary = _as_dict(report.get("summary"))
    if int(summary.get("runtime_dispatch_ready_count", 0) or 0) > 0:
        blocked.append("muon_canary_rollout_runtime_dispatch_ready_count")
    if int(summary.get("native_dispatch_allowed_count", 0) or 0) > 0:
        blocked.append("muon_canary_rollout_native_dispatch_allowed_count")
    if int(summary.get("training_path_enabled_count", 0) or 0) > 0:
        blocked.append("muon_canary_rollout_training_path_enabled_count")
    if int(summary.get("product_native_ready_count", 0) or 0) > 0:
        blocked.append("muon_canary_rollout_product_native_ready_count")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _muon_dispatch_review_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("muon_dispatch_review_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("muon_dispatch_review_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("muon_dispatch_review_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("muon_dispatch_review_native_dispatch_allowed")
    if report.get("request_fields_emitted") is True:
        blocked.append("muon_dispatch_review_request_fields_emitted")
    if report.get("schema_exposure_allowed") is True:
        blocked.append("muon_dispatch_review_schema_exposure_allowed")
    if report.get("ui_exposure_allowed") is True:
        blocked.append("muon_dispatch_review_ui_exposure_allowed")
    if report.get("product_native_ready") is True or report.get("product_native_dispatch_ready") is True:
        blocked.append("muon_dispatch_review_product_native_ready")
    summary = _as_dict(report.get("summary"))
    if int(summary.get("product_native_ready_count", 0) or 0) > 0:
        blocked.append("muon_dispatch_review_product_native_ready_count")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _muon_owner_release_hold_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("muon_owner_release_hold_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("muon_owner_release_hold_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("muon_owner_release_hold_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("muon_owner_release_hold_native_dispatch_allowed")
    if report.get("native_kernel_ready") is True:
        blocked.append("muon_owner_release_hold_native_kernel_ready")
    if report.get("request_fields_emitted") is True:
        blocked.append("muon_owner_release_hold_request_fields_emitted")
    if report.get("schema_exposure_allowed") is True:
        blocked.append("muon_owner_release_hold_schema_exposure_allowed")
    if report.get("ui_exposure_allowed") is True:
        blocked.append("muon_owner_release_hold_ui_exposure_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _muon_request_schema_ui_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("muon_request_schema_ui_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("muon_request_schema_ui_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("muon_request_schema_ui_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("muon_request_schema_ui_native_dispatch_allowed")
    if report.get("native_kernel_ready") is True:
        blocked.append("muon_request_schema_ui_native_kernel_ready")
    if report.get("request_fields_emitted") is True:
        blocked.append("muon_request_schema_ui_request_fields_emitted")
    if report.get("schema_exposure_allowed") is True:
        blocked.append("muon_request_schema_ui_schema_exposure_allowed")
    if report.get("ui_exposure_allowed") is True:
        blocked.append("muon_request_schema_ui_ui_exposure_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _factored_custom_state_layout_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("factored_custom_state_layout_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("factored_custom_state_layout_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("factored_custom_state_layout_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("factored_custom_state_layout_native_dispatch_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _factored_custom_family_batch_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("factored_custom_family_batch_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("factored_custom_family_batch_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("factored_custom_family_batch_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("factored_custom_family_batch_native_dispatch_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _factored_custom_owner_release_hold_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("factored_custom_owner_release_hold_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("factored_custom_owner_release_hold_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("factored_custom_owner_release_hold_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("factored_custom_owner_release_hold_native_dispatch_allowed")
    if report.get("request_fields_emitted") is True:
        blocked.append("factored_custom_owner_release_hold_request_fields_emitted")
    if report.get("schema_exposure_allowed") is True:
        blocked.append("factored_custom_owner_release_hold_schema_exposure_allowed")
    if report.get("ui_exposure_allowed") is True:
        blocked.append("factored_custom_owner_release_hold_ui_exposure_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _factored_custom_request_schema_ui_blockers(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return []
    blocked: list[str] = []
    if report.get("training_path_enabled") is True:
        blocked.append("factored_custom_request_schema_ui_training_path_enabled")
    if report.get("default_behavior_changed") is True:
        blocked.append("factored_custom_request_schema_ui_default_behavior_changed")
    if report.get("runtime_dispatch_ready") is True:
        blocked.append("factored_custom_request_schema_ui_runtime_dispatch_ready")
    if report.get("native_dispatch_allowed") is True:
        blocked.append("factored_custom_request_schema_ui_native_dispatch_allowed")
    if report.get("request_fields_emitted") is True:
        blocked.append("factored_custom_request_schema_ui_request_fields_emitted")
    if report.get("schema_exposure_allowed") is True:
        blocked.append("factored_custom_request_schema_ui_schema_exposure_allowed")
    if report.get("ui_exposure_allowed") is True:
        blocked.append("factored_custom_request_schema_ui_ui_exposure_allowed")
    blocked.extend(_strings(report.get("blocked_reasons")))
    return _dedupe(blocked)


def _priority_groups(
    rows: list[dict[str, Any]],
    plugin: Mapping[str, Any],
    adamw_variant_batch: Mapping[str, Any],
    adamw_variant_product_canary: Mapping[str, Any],
    adamw_variant_owner_hold: Mapping[str, Any],
    adamw_variant_request_schema_ui: Mapping[str, Any],
    simple_product_canary: Mapping[str, Any],
    simple_owner_hold: Mapping[str, Any],
    simple_schedulefree_review: Mapping[str, Any],
    simple_schedulefree_hold: Mapping[str, Any],
    simple_request_schema_ui: Mapping[str, Any],
    adaptive_lr_request_schema_ui: Mapping[str, Any],
    factored_custom_request_schema_ui: Mapping[str, Any],
    muon_request_schema_ui: Mapping[str, Any],
    plugin_selected_family_owner_release_hold: Mapping[str, Any],
    plugin_selected_family_request_schema_ui: Mapping[str, Any],
) -> list[dict[str, Any]]:
    def names(status: str) -> list[str]:
        return [row["optimizer_type"] for row in rows if row["turbocore_status"] == status]

    groups = [
        {
            "group": "exact_adamw_native_route",
            "priority": "P0",
            "optimizer_types": names("adamw_representative_route_matrix_ready") + names("native_adamw_ready"),
            "why": "Exact AdamW is the only optimizer currently eligible for the Rust/CUDA update route.",
            "next_gate": _exact_adamw_next_gate(rows),
        },
        {
            "group": "adamw_variants_quantized_paged",
            "priority": "P1",
            "optimizer_types": names("adamw_variant_request_schema_ui_non_exposure_ready")
            + names("adamw_variant_owner_release_hold_ready")
            + names("adamw_variant_representative_product_training_canary_ready")
            + names("adamw_variant_native_canary_ready")
            + names("adamw_variant_training_loop_canary_manifest_ready_dispatch_pending")
            + names("adamw_variant_runtime_canary_manifest_ready_training_loop_pending")
            + names("adamw_variant_family_batch_pending")
            + names("adamw_variant_research"),
            "why": "Closest to current AdamW route and important for low-VRAM users.",
            "next_gate": _adamw_variant_next_gate(
                adamw_variant_batch,
                adamw_variant_product_canary,
                adamw_variant_owner_hold,
                adamw_variant_request_schema_ui,
            ),
        },
        {
            "group": "simple_formula_lion_sgd",
            "priority": "P2",
            "optimizer_types": (
                names("simple_formula_request_schema_ui_non_exposure_ready")
                +
                names("simple_formula_owner_release_hold_ready")
                + names("simple_formula_representative_product_training_canary_ready")
                + names("simple_formula_native_batch_canary_ready")
                + names("simple_formula_native_dispatch_canary_ready")
                + names("simple_formula_variant_schedule_free_owner_release_hold_ready")
                + names("simple_formula_variant_schedule_free_dispatch_integration_review_ready")
                + names("simple_formula_variant_schedule_free_rollout_policy_ready")
                + names("simple_formula_variant_schedule_free_native_canary_ready")
                + names("simple_formula_variant_quantized_owner_approval_hold_ready")
                + names("simple_formula_variant_quantized_dispatch_integration_review_ready")
                + names("simple_formula_variant_quantized_rollout_policy_ready")
                + names("simple_formula_variant_quantized_product_state_sync_ready")
                + names("simple_formula_variant_quantized_e2e_no_regression_ready")
                + names("simple_formula_variant_quantized_training_loop_canary_ready")
                + names("simple_formula_variant_quantized_training_loop_canary_manifest_ready")
                + names("simple_formula_variant_quantized_runtime_canary_manifest_ready")
                + names("simple_formula_variant_quantized_native_scratch_kernel_ready")
                + names("simple_formula_variant_quantized_formula_parity_ready")
                + names("simple_formula_variant_native_abi_spec_ready")
                + names("simple_formula_variant_layout_spec_ready")
                + names("simple_formula_variant_state_machine_reference_ready")
                + names("simple_update_research")
            ),
            "why": "Small formulas can validate the multi-kernel optimizer ABI.",
            "next_gate": _simple_formula_next_gate(
                simple_product_canary,
                simple_owner_hold,
                simple_schedulefree_review,
                simple_schedulefree_hold,
                simple_request_schema_ui,
            ),
        },
        {
            "group": "adaptive_lr_prodigy_dadapt",
            "priority": "P3",
            "optimizer_types": (
                names("adaptive_lr_request_schema_ui_non_exposure_ready")
                +
                names("adaptive_lr_owner_release_hold_ready")
                +
                names("adaptive_lr_dispatch_integration_review_ready")
                +
                names("adaptive_lr_canary_rollout_policy_ready")
                +
                names("adaptive_lr_e2e_shadow_matrix_ready")
                +
                names("adaptive_lr_training_loop_canary_ready")
                +
                names("adaptive_lr_runtime_dispatch_shadow_ready")
                +
                names("adaptive_lr_training_tensor_binding_canary_ready")
                +
                names("adaptive_lr_cuda_kernel_implementation_ready")
                +
                names("adaptive_lr_cuda_kernel_contract_plan_ready")
                +
                names("adaptive_lr_native_state_machine_implementation_stub_ready")
                +
                names("adaptive_lr_native_state_machine_cpu_reference_guard_ready")
                +
                names("adaptive_lr_native_state_machine_abi_skeleton_ready")
                +
                names("adaptive_lr_native_state_machine_abi_precondition_review_ready")
                +
                names("adaptive_lr_state_machine_replay_executor_ready")
                +
                names("adaptive_lr_state_machine_replay_matrix_ready")
                + names("adaptive_lr_state_machine_reference_ready")
                + names("adaptive_lr_research")
            ),
            "why": "Popular for LoRA but requires dynamic LR/state-machine semantics.",
            "next_gate": _adaptive_lr_next_gate(adaptive_lr_request_schema_ui),
        },
        {
            "group": "factored_custom_memory_optimizers",
            "priority": "P4",
            "optimizer_types": (
                names("factored_custom_request_schema_ui_non_exposure_ready")
                + names("factored_custom_owner_release_hold_ready")
                + names("factored_custom_dispatch_integration_review_ready")
                + names("factored_custom_canary_rollout_policy_ready")
                + names("factored_custom_e2e_shadow_matrix_ready")
                + names("factored_custom_training_loop_canary_ready")
                + names("factored_custom_runtime_dispatch_adapter_shadow_ready")
                + names("factored_custom_training_tensor_binding_canary_ready")
                + names("factored_custom_native_scratch_kernel_ready")
                +
                names("factored_custom_state_layout_reference_ready")
                + names("factored_or_custom_research")
            ),
            "why": "Potential memory win for full finetune, but quality and layout risk is higher.",
            "next_gate": _factored_custom_next_gate(factored_custom_request_schema_ui),
        },
        {
            "group": "plugin_selector_expansion",
            "priority": "P5",
            "optimizer_types": [OptimizerType.PYTORCH_OPTIMIZER.value, OptimizerType.GENERIC.value]
            + names("model_shape_aware_request_schema_ui_non_exposure_ready")
            + names("model_shape_aware_owner_release_hold_ready")
            + names("model_shape_aware_e2e_shadow_matrix_ready")
            + names("model_shape_aware_training_loop_canary_ready")
            + names("model_shape_aware_training_tensor_binding_canary_ready")
            + names("model_shape_aware_native_scratch_kernel_ready")
            + names("model_shape_aware_dispatch_review_ready")
            + names("model_shape_aware_param_group_abi_ready")
            + names("model_shape_aware_research"),
            "why": f"{plugin.get('plugin_optimizer_count', 0)} discovered plugin optimizers cannot share one native path.",
            "next_gate": _plugin_or_muon_next_gate(plugin, muon_request_schema_ui),
        },
    ]
    return [group for group in groups if group["optimizer_types"]]


def _exact_adamw_next_gate(rows: list[dict[str, Any]]) -> str:
    if names := [row for row in rows if row["turbocore_status"] == "adamw_representative_route_matrix_ready"]:
        return "record explicit owner/release approval for exact AdamW native dispatch"
    return "extend representative route matrix for exact AdamW native dispatch"


def _adamw_variant_next_gate(
    report: Mapping[str, Any],
    product_canary: Mapping[str, Any],
    owner_hold: Mapping[str, Any],
    request_schema_ui: Mapping[str, Any],
) -> str:
    if request_schema_ui.get("request_schema_ui_non_exposure_ready") is True:
        return "keep AdamW variant native dispatch unwired until explicit owner/release approval is recorded"
    if owner_hold.get("owner_release_hold_ready") is True:
        return "request/schema/UI non-exposure gate for AdamW variants with dispatch still default-off"
    if product_canary.get("representative_product_training_canary_ready") is True:
        return "prepare AdamW variant owner/release hold with product dispatch still default-off"
    summary = _as_dict(report.get("summary"))
    if summary.get("dispatch_integration_review_ready") is True:
        return "representative product-training canary for ready AdamW variants"
    if summary.get("e2e_shadow_matrix_ready") is True and summary.get("canary_rollout_policy_ready") is True:
        return "AdamW variant owner/release hold package with product dispatch still default-off"
    if summary.get("e2e_shadow_matrix_ready") is True:
        return "default-off canary rollout policy for ready AdamW variants"
    return "quantized or paged state parity, resume, and memory/speed matrix"


def _simple_formula_next_gate(
    simple_product_canary: Mapping[str, Any],
    simple_owner_hold: Mapping[str, Any],
    simple_schedulefree_review: Mapping[str, Any],
    simple_schedulefree_hold: Mapping[str, Any],
    simple_request_schema_ui: Mapping[str, Any],
) -> str:
    if simple_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True:
        return "keep simple-formula native dispatch unwired until explicit owner/release approval is recorded"
    if simple_schedulefree_hold.get("owner_release_hold_ready") is True:
        return "record explicit owner/release approval artifacts for fp32, quantized, and schedule-free simple variants"
    if simple_schedulefree_review.get("review_gate_ready") is True:
        return "record explicit owner/release approval artifacts for fp32 Lion/SGDNesterov, quantized variants, and built-in schedule-free variants"
    if simple_owner_hold.get("owner_release_hold_ready") is True:
        return "record explicit owner/release approval artifacts for fp32 Lion/SGDNesterov; quantized owner approval hold; dispatch review for schedule-free variants"
    if simple_product_canary.get("representative_product_training_canary_ready") is True:
        return "explicit owner/release approval for fp32 Lion/SGDNesterov; quantized owner approval hold; rollout review for schedule-free variants"
    return "representative product training canary for fp32 Lion/SGD; quantized owner approval hold; rollout review for schedule-free variants"


def _adaptive_lr_next_gate(adaptive_lr_request_schema_ui: Mapping[str, Any]) -> str:
    if adaptive_lr_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True:
        return "keep adaptive-LR native dispatch unwired until explicit owner/release approval is recorded"
    return "request/schema/UI non-exposure gate after owner/release hold, with dispatch still default-off"


def _factored_custom_next_gate(factored_custom_request_schema_ui: Mapping[str, Any]) -> str:
    if factored_custom_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True:
        return "keep factored/custom native dispatch unwired until explicit owner/release approval is recorded"
    return "request/schema/UI non-exposure gate after owner/release hold, with dispatch still default-off"


def _plugin_or_muon_next_gate(plugin: Mapping[str, Any], muon_request_schema_ui: Mapping[str, Any]) -> str:
    plugin_next = str(plugin.get("next_gate") or "")
    if muon_request_schema_ui.get("request_schema_ui_non_exposure_ready") is True:
        return plugin_next or "keep Muon native dispatch unwired until explicit owner/release approval is recorded"
    return "complete Muon owner/release hold and request/schema/UI non-exposure before plugin selector expansion"


def _manual_approval_gate(next_gate: object) -> bool:
    text = str(next_gate).lower()
    return (
        "await explicit owner" in text
        or "until explicit owner" in text
        or "record explicit owner" in text
        or "explicit owner/release approval" in text
    )


def _recommended_next_step(priority_groups: list[Mapping[str, Any]]) -> str:
    if not priority_groups:
        return "no optimizer expansion candidates classified"
    actionable = [
        group
        for group in priority_groups
        if not _manual_approval_gate(group.get("next_gate", ""))
    ]
    if not actionable:
        return "keep native dispatch unwired until explicit owner/release approval is recorded"
    first = actionable[0]
    return f"start {first.get('group')} with {first.get('next_gate')}"


def _compact_simple_family_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "simple_formula_native_batch_canary_ready": report.get("simple_formula_native_batch_canary_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "exact_target_count": int(summary.get("exact_target_count", 0) or 0),
        "batch_canary_ready_count": int(summary.get("batch_canary_ready_count", 0) or 0),
        "pending_variant_count": int(summary.get("pending_variant_count", 0) or 0),
        "variant_layout_spec_ready_count": int(summary.get("variant_layout_spec_ready_count", 0) or 0),
        "variant_state_machine_reference_ready_count": int(
            summary.get("variant_state_machine_reference_ready_count", 0) or 0
        ),
        "variant_native_abi_spec_ready_count": int(summary.get("variant_native_abi_spec_ready_count", 0) or 0),
        "variant_schedule_free_native_canary_ready_count": int(
            summary.get("variant_schedule_free_native_canary_ready_count", 0) or 0
        ),
        "variant_quantized_formula_parity_ready_count": int(
            summary.get("variant_quantized_formula_parity_ready_count", 0) or 0
        ),
        "variant_quantized_native_scratch_kernel_ready_count": int(
            summary.get("variant_quantized_native_scratch_kernel_ready_count", 0) or 0
        ),
        "variant_quantized_runtime_canary_manifest_ready_count": int(
            summary.get("variant_quantized_runtime_canary_manifest_ready_count", 0) or 0
        ),
        "variant_quantized_training_loop_canary_manifest_ready_count": int(
            summary.get("variant_quantized_training_loop_canary_manifest_ready_count", 0) or 0
        ),
        "variant_quantized_training_loop_canary_ready_count": int(
            summary.get("variant_quantized_training_loop_canary_ready_count", 0) or 0
        ),
        "variant_quantized_e2e_no_regression_ready_count": int(
            summary.get("variant_quantized_e2e_no_regression_ready_count", 0) or 0
        ),
        "variant_quantized_product_state_sync_review_ready_count": int(
            summary.get("variant_quantized_product_state_sync_review_ready_count", 0) or 0
        ),
        "variant_quantized_product_optimizer_state_sync_ready_count": int(
            summary.get("variant_quantized_product_optimizer_state_sync_ready_count", 0) or 0
        ),
        "variant_quantized_optimizer_state_sync_state_tensor_count": int(
            summary.get("variant_quantized_optimizer_state_sync_state_tensor_count", 0) or 0
        ),
        "variant_quantized_optimizer_state_sync_parameter_tensor_count": int(
            summary.get("variant_quantized_optimizer_state_sync_parameter_tensor_count", 0) or 0
        ),
        "variant_quantized_rollout_policy_ready_count": int(
            summary.get("variant_quantized_rollout_policy_ready_count", 0) or 0
        ),
        "variant_quantized_dispatch_integration_review_ready_count": int(
            summary.get("variant_quantized_dispatch_integration_review_ready_count", 0) or 0
        ),
        "variant_quantized_owner_approval_hold_ready_count": int(
            summary.get("variant_quantized_owner_approval_hold_ready_count", 0) or 0
        ),
        "variant_quantized_native_canary_pending_count": int(
            summary.get("variant_quantized_native_canary_pending_count", 0) or 0
        ),
        "variant_formula_parity_matrix_artifact_ready_count": int(
            summary.get("variant_formula_parity_matrix_artifact_ready_count", 0) or 0
        ),
        "variant_formula_parity_matrix_implementation_ready_count": int(
            summary.get("variant_formula_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "variant_resume_parity_matrix_artifact_ready_count": int(
            summary.get("variant_resume_parity_matrix_artifact_ready_count", 0) or 0
        ),
        "variant_resume_parity_matrix_implementation_ready_count": int(
            summary.get("variant_resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "variant_native_kernel_ready_count": int(summary.get("variant_native_kernel_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_simple_product_training_canary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "representative_product_training_canary_ready": report.get(
            "representative_product_training_canary_ready"
        ) is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_optimizer_count": int(summary.get("target_optimizer_count", 0) or 0),
        "representative_product_training_canary_ready_count": int(
            summary.get("representative_product_training_canary_ready_count", 0) or 0
        ),
        "ready_required_stage_count": int(summary.get("ready_required_stage_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_simple_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "representative_product_training_canary_ready": report.get(
            "representative_product_training_canary_ready"
        ) is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_simple_schedulefree_rollout_policy(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "canary_rollout_policy_ready": report.get("canary_rollout_policy_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_simple_schedulefree_dispatch_review(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "review_gate_ready": report.get("review_gate_ready") is True,
        "dispatch_integration_review": report.get("dispatch_integration_review") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_simple_schedulefree_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "dispatch_integration_review": report.get("dispatch_integration_review") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_simple_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "quantized_owner_approval_hold_ready": report.get("quantized_owner_approval_hold_ready") is True,
        "schedulefree_owner_release_hold_ready": report.get("schedulefree_owner_release_hold_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "present_boundary_path_count": int(summary.get("present_boundary_path_count", 0) or 0),
        "scanned_file_count": int(summary.get("scanned_file_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_simple_runtime_dispatch_rehearsal(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_dispatch_rehearsal_ready_count": int(
            summary.get("runtime_dispatch_rehearsal_ready_count", 0) or 0
        ),
        "training_executor_called_count": int(summary.get("training_executor_called_count", 0) or 0),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "skip_pytorch_count": int(summary.get("skip_pytorch_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_exact_adamw_stream_event_chain_abi(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "stream_event_chain_ownership_abi_ready": report.get(
            "stream_event_chain_ownership_abi_ready"
        )
        is True,
        "stream_lifetime_ownership_boundary_ready": report.get(
            "stream_lifetime_ownership_boundary_ready"
        )
        is True,
        "stream_lifetime_ownership_bound_evidence": report.get(
            "stream_lifetime_ownership_bound_evidence"
        )
        is True,
        "stream_ordering_verified": report.get("stream_ordering_verified") is True,
        "event_chain_verified": report.get("event_chain_verified") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "training_dispatch": bool(report.get("training_dispatch", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_allowed": bool(report.get("runtime_dispatch_allowed", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_exposure_allowed": bool(report.get("product_exposure_allowed", False)),
        "sync_fast_path_allowed": bool(report.get("sync_fast_path_allowed", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "stream_event_chain_ownership_abi_ready_count": int(
            summary.get("stream_event_chain_ownership_abi_ready_count", 0) or 0
        ),
        "stream_lifetime_ownership_bound_evidence_count": int(
            summary.get("stream_lifetime_ownership_bound_evidence_count", 0) or 0
        ),
        "event_chain_verified_count": int(summary.get("event_chain_verified_count", 0) or 0),
        "sync_fast_path_allowed_count": int(summary.get("sync_fast_path_allowed_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
        "promotion_blockers": _strings(report.get("promotion_blockers")),
    }


def _compact_adamw_variant_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "native_canary_stage_evidence_ready_count": int(
            summary.get("native_canary_stage_evidence_ready_count", summary.get("native_ready_count", 0)) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "native_canary_manifest_count": int(summary.get("native_canary_manifest_count", 0) or 0),
        "state_reference_ready_count": int(summary.get("state_reference_ready_count", 0) or 0),
        "native_canary_manifest_ready_count": int(summary.get("native_canary_manifest_ready_count", 0) or 0),
        "training_loop_canary_ready_count": int(summary.get("training_loop_canary_ready_count", 0) or 0),
        "schedule_free_native_abi_ready_count": int(summary.get("schedule_free_native_abi_ready_count", 0) or 0),
        "schedule_free_scratch_formula_canary_ready_count": int(
            summary.get("schedule_free_scratch_formula_canary_ready_count", 0) or 0
        ),
        "schedule_free_native_scratch_kernel_ready_count": int(
            summary.get("schedule_free_native_scratch_kernel_ready_count", 0) or 0
        ),
        "pending_count": int(summary.get("pending_count", 0) or 0),
        "exact_adamw_included": bool(summary.get("exact_adamw_included", False)),
        "e2e_shadow_matrix_ready": summary.get("e2e_shadow_matrix_ready") is True,
        "canary_rollout_policy_ready": summary.get("canary_rollout_policy_ready") is True,
        "dispatch_integration_review_ready": summary.get("dispatch_integration_review_ready") is True,
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adamw_variant_product_training_canary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "representative_product_training_canary_ready": report.get(
            "representative_product_training_canary_ready"
        )
        is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "target_optimizer_count": int(summary.get("target_optimizer_count", 0) or 0),
        "representative_product_training_canary_ready_count": int(
            summary.get("representative_product_training_canary_ready_count", 0) or 0
        ),
        "ready_required_family_gate_count": int(summary.get("ready_required_family_gate_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adamw_variant_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "representative_product_training_canary_ready": report.get(
            "representative_product_training_canary_ready"
        )
        is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adamw_variant_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "present_boundary_path_count": int(summary.get("present_boundary_path_count", 0) or 0),
        "scanned_file_count": int(summary.get("scanned_file_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "state_machine_reference_ready_count": int(
            summary.get("state_machine_reference_ready_count", 0) or 0
        ),
        "native_ready_count": int(summary.get("native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_replay_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "state_machine_replay_matrix_ready": report.get("state_machine_replay_matrix_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "state_machine_replay_matrix_artifact_ready_count": int(
            summary.get("state_machine_replay_matrix_artifact_ready_count", 0) or 0
        ),
        "state_machine_replay_matrix_implementation_ready_count": int(
            summary.get("state_machine_replay_matrix_implementation_ready_count", 0) or 0
        ),
        "state_machine_replay_case_planned_count": int(
            summary.get("state_machine_replay_case_planned_count", 0) or 0
        ),
        "state_machine_replay_resume_case_planned_count": int(
            summary.get("state_machine_replay_resume_case_planned_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_replay_executor(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "state_machine_replay_executor_ready": report.get("state_machine_replay_executor_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "reference_replay_executor_ready_count": int(
            summary.get("reference_replay_executor_ready_count", 0) or 0
        ),
        "resume_next_step_parity_passed_count": int(
            summary.get("resume_next_step_parity_passed_count", 0) or 0
        ),
        "state_machine_abi_implementation_ready_count": int(
            summary.get("state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_abi_preconditions(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "native_state_machine_abi_preconditions_ready": (
            report.get("native_state_machine_abi_preconditions_ready") is True
        ),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "native_state_machine_abi_precondition_review_ready_count": int(
            summary.get("native_state_machine_abi_precondition_review_ready_count", 0) or 0
        ),
        "native_state_machine_abi_precondition_package_ready_count": int(
            summary.get("native_state_machine_abi_precondition_package_ready_count", 0) or 0
        ),
        "native_kernel_precondition_review_ready_count": int(
            summary.get("native_kernel_precondition_review_ready_count", 0) or 0
        ),
        "state_machine_abi_implementation_ready_count": int(
            summary.get("state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_abi_skeleton(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "native_state_machine_abi_skeleton_ready": report.get("native_state_machine_abi_skeleton_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "native_state_machine_abi_skeleton_ready_count": int(
            summary.get("native_state_machine_abi_skeleton_ready_count", 0) or 0
        ),
        "state_machine_entrypoint_contract_ready_count": int(
            summary.get("state_machine_entrypoint_contract_ready_count", 0) or 0
        ),
        "launch_plan_schema_ready_count": int(summary.get("launch_plan_schema_ready_count", 0) or 0),
        "state_buffer_mapping_contract_ready_count": int(
            summary.get("state_buffer_mapping_contract_ready_count", 0) or 0
        ),
        "state_machine_abi_implementation_ready_count": int(
            summary.get("state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_cpu_guard(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "native_state_machine_cpu_reference_guard_ready": (
            report.get("native_state_machine_cpu_reference_guard_ready") is True
        ),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "cpu_reference_guard_ready_count": int(summary.get("cpu_reference_guard_ready_count", 0) or 0),
        "valid_launch_plan_guard_passed_count": int(
            summary.get("valid_launch_plan_guard_passed_count", 0) or 0
        ),
        "bad_finite_scalar_guard_rejected_count": int(
            summary.get("bad_finite_scalar_guard_rejected_count", 0) or 0
        ),
        "bad_dispatch_guard_rejected_count": int(summary.get("bad_dispatch_guard_rejected_count", 0) or 0),
        "state_machine_abi_implementation_ready_count": int(
            summary.get("state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_implementation_stub(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "native_state_machine_implementation_stub_ready": (
            report.get("native_state_machine_implementation_stub_ready") is True
        ),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "implementation_stub_ready_count": int(summary.get("implementation_stub_ready_count", 0) or 0),
        "stub_entrypoint_contract_ready_count": int(summary.get("stub_entrypoint_contract_ready_count", 0) or 0),
        "stub_state_transition_contract_ready_count": int(
            summary.get("stub_state_transition_contract_ready_count", 0) or 0
        ),
        "stub_dispatch_disabled_assertion_ready_count": int(
            summary.get("stub_dispatch_disabled_assertion_ready_count", 0) or 0
        ),
        "state_machine_abi_implementation_ready_count": int(
            summary.get("state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_cuda_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "cuda_kernel_contract_plan_ready": report.get("cuda_kernel_contract_plan_ready") is True,
        "runtime_canary_manifest_ready": report.get("runtime_canary_manifest_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "cuda_kernel_contract_plan_ready_count": int(
            summary.get("cuda_kernel_contract_plan_ready_count", 0) or 0
        ),
        "runtime_canary_manifest_ready_count": int(
            summary.get("runtime_canary_manifest_ready_count", 0) or 0
        ),
        "cuda_kernel_implementation_ready_count": int(
            summary.get("cuda_kernel_implementation_ready_count", 0) or 0
        ),
        "runtime_canary_ready_count": int(summary.get("runtime_canary_ready_count", 0) or 0),
        "runtime_canary_hit_count": int(summary.get("runtime_canary_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_cuda_implementation(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "cuda_kernel_implementation_ready": report.get("cuda_kernel_implementation_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "cuda_kernel_implementation_ready_count": int(
            summary.get("cuda_kernel_implementation_ready_count", 0) or 0
        ),
        "state_machine_abi_implementation_ready_count": int(
            summary.get("state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "native_kernel_preconditions_implementation_ready_count": int(
            summary.get("native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "kernel_executed_count": int(summary.get("kernel_executed_count", 0) or 0),
        "contract_plan_ready_count": int(summary.get("contract_plan_ready_count", 0) or 0),
        "runtime_canary_ready_count": int(summary.get("runtime_canary_ready_count", 0) or 0),
        "runtime_canary_hit_count": int(summary.get("runtime_canary_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_training_tensor_binding(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "training_tensor_binding_canary_ready": report.get("training_tensor_binding_canary_ready") is True,
        "training_tensor_binding_parity_ready": report.get("training_tensor_binding_parity_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "training_tensor_binding_canary_ready_count": int(
            summary.get("training_tensor_binding_canary_ready_count", 0) or 0
        ),
        "training_tensor_binding_parity_ready_count": int(
            summary.get("training_tensor_binding_parity_ready_count", 0) or 0
        ),
        "kernel_executed_count": int(summary.get("kernel_executed_count", 0) or 0),
        "family_case_count": int(summary.get("family_case_count", 0) or 0),
        "family_passed_case_count": int(summary.get("family_passed_case_count", 0) or 0),
        "runtime_canary_ready_count": int(summary.get("runtime_canary_ready_count", 0) or 0),
        "runtime_canary_hit_count": int(summary.get("runtime_canary_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_runtime_dispatch_shadow(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "runtime_dispatch_shadow_ready": report.get("runtime_dispatch_shadow_ready") is True,
        "fallback_backend_authoritative": report.get("fallback_backend_authoritative") is True,
        "native_shadow_call_allowed": bool(report.get("native_shadow_call_allowed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "runtime_dispatch_shadow_ready_count": int(
            summary.get("runtime_dispatch_shadow_ready_count", 0) or 0
        ),
        "training_tensor_binding_canary_ready_count": int(
            summary.get("training_tensor_binding_canary_ready_count", 0) or 0
        ),
        "training_tensor_binding_parity_ready_count": int(
            summary.get("training_tensor_binding_parity_ready_count", 0) or 0
        ),
        "native_shadow_call_allowed_count": int(summary.get("native_shadow_call_allowed_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_training_loop_canary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "training_loop_canary_ready": report.get("training_loop_canary_ready") is True,
        "training_loop_canary_hit": report.get("training_loop_canary_hit") is True,
        "runtime_dispatch_shadow_ready": report.get("runtime_dispatch_shadow_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "family_case_count": int(summary.get("family_case_count", 0) or 0),
        "family_passed_case_count": int(summary.get("family_passed_case_count", 0) or 0),
        "training_loop_canary_ready_count": int(summary.get("training_loop_canary_ready_count", 0) or 0),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_e2e_shadow_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "e2e_shadow_matrix_ready": report.get("e2e_shadow_matrix_ready") is True,
        "e2e_shadow_matrix_passed": report.get("e2e_shadow_matrix_passed") is True,
        "report_only_matrix_scaffold_ready": report.get("report_only_matrix_scaffold_ready") is True,
        "live_shadow_matrix_executed": bool(report.get("live_shadow_matrix_executed", False)),
        "fallback_backend_authoritative": report.get("fallback_backend_authoritative") is True,
        "native_shadow_updates_original": bool(report.get("native_shadow_updates_original", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "report_only_case_count": int(summary.get("report_only_case_count", 0) or 0),
        "failed_case_count": int(summary.get("failed_case_count", 0) or 0),
        "e2e_shadow_matrix_ready_count": int(summary.get("e2e_shadow_matrix_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_canary_rollout_policy(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "canary_rollout_policy_ready": report.get("canary_rollout_policy_ready") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "canary_auto_enabled": bool(report.get("canary_auto_enabled", False)),
        "fallback_backend_authoritative": report.get("fallback_backend_authoritative") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "target_count": int(summary.get("target_count", 0) or 0),
        "canary_rollout_policy_ready_count": int(summary.get("canary_rollout_policy_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_dispatch_review(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "review_gate_ready": report.get("review_gate_ready") is True,
        "dispatch_integration_review": report.get("dispatch_integration_review") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "dispatch_integration_review": report.get("dispatch_integration_review") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_adaptive_lr_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "request_adapter_enabled": bool(report.get("request_adapter_enabled", False)),
        "backend_router_registered": bool(report.get("backend_router_registered", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "present_boundary_path_count": int(summary.get("present_boundary_path_count", 0) or 0),
        "scanned_file_count": int(summary.get("scanned_file_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_factored_custom_state_layout(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "state_layout_reference_ready": report.get("state_layout_reference_ready") is True,
        "factored_custom_family_classified": report.get("factored_custom_family_classified") is True,
        "quality_guard_documented": report.get("quality_guard_documented") is True,
        "adamw_kernel_reuse_blocked": report.get("adamw_kernel_reuse_blocked") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "local_live_reference_count": int(summary.get("local_live_reference_count", 0) or 0),
        "memory_saving_candidate_count": int(summary.get("memory_saving_candidate_count", 0) or 0),
        "passed_case_count": int(summary.get("passed_case_count", 0) or 0),
        "required_case_count": int(summary.get("required_case_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_factored_custom_family_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "factored_custom_family_batch_ready": report.get("factored_custom_family_batch_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "native_scratch_kernel_ready_count": int(summary.get("native_scratch_kernel_ready_count", 0) or 0),
        "training_tensor_binding_canary_ready_count": int(
            summary.get("training_tensor_binding_canary_ready_count", 0) or 0
        ),
        "runtime_dispatch_adapter_shadow_ready_count": int(
            summary.get("runtime_dispatch_adapter_shadow_ready_count", 0) or 0
        ),
        "training_loop_canary_ready_count": int(summary.get("training_loop_canary_ready_count", 0) or 0),
        "e2e_shadow_matrix_ready_count": int(summary.get("e2e_shadow_matrix_ready_count", 0) or 0),
        "canary_rollout_policy_ready_count": int(summary.get("canary_rollout_policy_ready_count", 0) or 0),
        "dispatch_integration_review_ready_count": int(
            summary.get("dispatch_integration_review_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "unsafe_claim_count": int(summary.get("unsafe_claim_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_factored_custom_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "family_batch_ready": report.get("family_batch_ready") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_factored_custom_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "request_adapter_enabled": bool(report.get("request_adapter_enabled", False)),
        "backend_router_registered": bool(report.get("backend_router_registered", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "present_boundary_path_count": int(summary.get("present_boundary_path_count", 0) or 0),
        "scanned_file_count": int(summary.get("scanned_file_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_optimizer_native_kernel_inventory(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    family_rows = report.get("family_rows") if isinstance(report.get("family_rows"), list) else []
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "static_inventory_only": bool(report.get("static_inventory_only", False)),
        "cuda_executed": bool(report.get("cuda_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "family_count": int(summary.get("family_count", 0) or 0),
        "kernel_source_present_count": int(summary.get("kernel_source_present_count", 0) or 0),
        "rust_probe_present_count": int(summary.get("rust_probe_present_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "family_rows": [
            {
                "native_route_family": str(_as_dict(row).get("native_route_family", "") or ""),
                "optimizer_count": int(_as_dict(row).get("optimizer_count", 0) or 0),
                "kernel_source_present_count": int(_as_dict(row).get("kernel_source_present_count", 0) or 0),
                "rust_probe_present_count": int(_as_dict(row).get("rust_probe_present_count", 0) or 0),
                "product_native_ready_count": int(_as_dict(row).get("product_native_ready_count", 0) or 0),
            }
            for row in family_rows
            if isinstance(row, Mapping)
        ],
    }


def _compact_optimizer_family_kernel_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "native_importable": bool(report.get("native_importable", False)),
        "entrypoint_present": bool(report.get("entrypoint_present", False)),
        "entrypoint": str(report.get("entrypoint", "") or ""),
        "kernel_source": str(report.get("kernel_source", "") or ""),
        "shared_by_family": bool(report.get("shared_by_family", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "kernel_executed": bool(report.get("kernel_executed", False)),
        "product_native_ready": bool(report.get("product_native_ready", False)),
        "required_family_count": int(summary.get("required_family_count", 0) or 0),
        "required_family_present_count": int(summary.get("required_family_present_count", 0) or 0),
        "optimizer_family_contract_count": int(summary.get("optimizer_family_contract_count", 0) or 0),
        "validation_ok_count": int(summary.get("validation_ok_count", 0) or 0),
        "kernel_source_ready_count": int(summary.get("kernel_source_ready_count", 0) or 0),
        "native_kernel_present_count": int(summary.get("native_kernel_present_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "kernel_executed_count": int(summary.get("kernel_executed_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "missing_families": _strings(report.get("missing_families")),
    }


def _compact_plugin_adamlike_runtime_dispatch_rehearsal(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_dispatch_rehearsal_ready_count": int(
            summary.get("runtime_dispatch_rehearsal_ready_count", 0) or 0
        ),
        "training_executor_called_count": int(summary.get("training_executor_called_count", 0) or 0),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "skip_pytorch_count": int(summary.get("skip_pytorch_count", 0) or 0),
        "exact_adamw_route_canary_ready_count": int(
            summary.get("exact_adamw_route_canary_ready_count", 0) or 0
        ),
        "dedicated_route_canary_ready_count": int(summary.get("dedicated_route_canary_ready_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_schedulefree_runtime_dispatch_rehearsal(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_dispatch_rehearsal_ready_count": int(
            summary.get("runtime_dispatch_rehearsal_ready_count", 0) or 0
        ),
        "training_executor_called_count": int(summary.get("training_executor_called_count", 0) or 0),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "skip_pytorch_count": int(summary.get("skip_pytorch_count", 0) or 0),
        "e2e_shadow_case_count": int(summary.get("e2e_shadow_case_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_adaptivelr_runtime_dispatch_rehearsal(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_dispatch_rehearsal_ready_count": int(
            summary.get("runtime_dispatch_rehearsal_ready_count", 0) or 0
        ),
        "training_executor_called_count": int(summary.get("training_executor_called_count", 0) or 0),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "mapped_selected_native_step_count": int(summary.get("mapped_selected_native_step_count", 0) or 0),
        "mapped_selected_native_kernel_launch_count": int(
            summary.get("mapped_selected_native_kernel_launch_count", 0) or 0
        ),
        "representative_family_case_count": int(summary.get("representative_family_case_count", 0) or 0),
        "skip_pytorch_count": int(summary.get("skip_pytorch_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_closure_second_order_runtime_precondition_rehearsal(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_precondition_rehearsal_ready": report.get("runtime_precondition_rehearsal_ready") is True,
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_kernel_ready": bool(report.get("native_kernel_ready", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_precondition_rehearsal_ready_count": int(
            summary.get("runtime_precondition_rehearsal_ready_count", 0) or 0
        ),
        "training_loop_abi_ready_count": int(summary.get("training_loop_abi_ready_count", 0) or 0),
        "resume_parity_matrix_ready_count": int(summary.get("resume_parity_matrix_ready_count", 0) or 0),
        "closure_resume_replay_artifact_ready_count": int(
            summary.get("closure_resume_replay_artifact_ready_count", 0) or 0
        ),
        "closure_resume_replay_row_ready_count": int(
            summary.get("closure_resume_replay_row_ready_count", 0) or 0
        ),
        "state_resume_adapter_ready_count": int(summary.get("state_resume_adapter_ready_count", 0) or 0),
        "native_kernel_precondition_ready_count": int(
            summary.get("native_kernel_precondition_ready_count", 0) or 0
        ),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_custom_formula_runtime_precondition_rehearsal(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_precondition_rehearsal_ready": report.get("runtime_precondition_rehearsal_ready") is True,
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_kernel_ready": bool(report.get("native_kernel_ready", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_precondition_rehearsal_ready_count": int(
            summary.get("runtime_precondition_rehearsal_ready_count", 0) or 0
        ),
        "formula_spec_ready_count": int(summary.get("formula_spec_ready_count", 0) or 0),
        "state_inventory_ready_count": int(summary.get("state_inventory_ready_count", 0) or 0),
        "quality_guard_ready_count": int(summary.get("quality_guard_ready_count", 0) or 0),
        "formula_parity_ready_count": int(summary.get("formula_parity_ready_count", 0) or 0),
        "resume_parity_ready_count": int(summary.get("resume_parity_ready_count", 0) or 0),
        "formula_step_execution_ready_count": int(summary.get("formula_step_execution_ready_count", 0) or 0),
        "resume_next_step_replay_ready_count": int(summary.get("resume_next_step_replay_ready_count", 0) or 0),
        "quality_guard_case_ready_count": int(summary.get("quality_guard_case_ready_count", 0) or 0),
        "formula_parity_case_ready_count": int(summary.get("formula_parity_case_ready_count", 0) or 0),
        "resume_parity_case_ready_count": int(summary.get("resume_parity_case_ready_count", 0) or 0),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_factored_memory_runtime_precondition_rehearsal(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_precondition_rehearsal_ready": report.get("runtime_precondition_rehearsal_ready") is True,
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_kernel_ready": bool(report.get("native_kernel_ready", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_precondition_rehearsal_ready_count": int(
            summary.get("runtime_precondition_rehearsal_ready_count", 0) or 0
        ),
        "native_layout_abi_ready_count": int(summary.get("native_layout_abi_ready_count", 0) or 0),
        "quality_matrix_ready_count": int(summary.get("quality_matrix_ready_count", 0) or 0),
        "formula_tensor_binding_matrix_ready_count": int(
            summary.get("formula_tensor_binding_matrix_ready_count", 0) or 0
        ),
        "dispatch_review_ready_count": int(summary.get("dispatch_review_ready_count", 0) or 0),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_model_shape_aware_runtime_precondition_rehearsal(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_precondition_rehearsal_ready": report.get("runtime_precondition_rehearsal_ready") is True,
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_kernel_ready": bool(report.get("native_kernel_ready", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_precondition_rehearsal_ready_count": int(
            summary.get("runtime_precondition_rehearsal_ready_count", 0) or 0
        ),
        "param_group_abi_ready_count": int(summary.get("param_group_abi_ready_count", 0) or 0),
        "resume_replay_matrix_ready_count": int(summary.get("resume_replay_matrix_ready_count", 0) or 0),
        "resume_replay_row_ready_count": int(summary.get("resume_replay_row_ready_count", 0) or 0),
        "model_structure_contract_count": int(summary.get("model_structure_contract_count", 0) or 0),
        "shape_partition_contract_count": int(summary.get("shape_partition_contract_count", 0) or 0),
        "distributed_collective_contract_count": int(
            summary.get("distributed_collective_contract_count", 0) or 0
        ),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_state_adapter_special_runtime_precondition_rehearsal(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_precondition_rehearsal_ready": report.get("runtime_precondition_rehearsal_ready") is True,
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_kernel_ready": bool(report.get("native_kernel_ready", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_precondition_rehearsal_ready_count": int(
            summary.get("runtime_precondition_rehearsal_ready_count", 0) or 0
        ),
        "adapter_abi_ready_count": int(summary.get("adapter_abi_ready_count", 0) or 0),
        "adapter_resume_matrix_ready_count": int(summary.get("adapter_resume_matrix_ready_count", 0) or 0),
        "resume_replay_case_ready_count": int(summary.get("resume_replay_case_ready_count", 0) or 0),
        "translation_case_ready_count": int(summary.get("translation_case_ready_count", 0) or 0),
        "native_kernel_precondition_ready_count": int(
            summary.get("native_kernel_precondition_ready_count", 0) or 0
        ),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_fused_backward_runtime_precondition_rehearsal(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "roadmap": str(report.get("roadmap", "") or ""),
        "runtime_precondition_rehearsal_ready": report.get("runtime_precondition_rehearsal_ready") is True,
        "runtime_dispatch_rehearsal_ready": report.get("runtime_dispatch_rehearsal_ready") is True,
        "internal_rehearsal_executed": bool(report.get("internal_rehearsal_executed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_kernel_ready": bool(report.get("native_kernel_ready", False)),
        "product_native_dispatch_ready": bool(report.get("product_native_dispatch_ready", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "selected_optimizer_count": int(summary.get("selected_optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "runtime_precondition_rehearsal_ready_count": int(
            summary.get("runtime_precondition_rehearsal_ready_count", 0) or 0
        ),
        "fused_backward_abi_ready_count": int(summary.get("fused_backward_abi_ready_count", 0) or 0),
        "resume_parity_matrix_ready_count": int(summary.get("resume_parity_matrix_ready_count", 0) or 0),
        "replay_case_ready_count": int(summary.get("replay_case_ready_count", 0) or 0),
        "native_kernel_precondition_ready_count": int(
            summary.get("native_kernel_precondition_ready_count", 0) or 0
        ),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_family_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "plugin_optimizer_family_batch_ready": report.get("plugin_optimizer_family_batch_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "selected_optimizer_gate_ready_count": int(summary.get("selected_optimizer_gate_ready_count", 0) or 0),
        "selected_optimizer_gate_pending_count": int(summary.get("selected_optimizer_gate_pending_count", 0) or 0),
        "selected_adamlike_native_canary_ready_count": int(
            summary.get("selected_adamlike_native_canary_ready_count", 0) or 0
        ),
        "selected_adamlike_e2e_shadow_matrix_ready": summary.get(
            "selected_adamlike_e2e_shadow_matrix_ready"
        )
        is True,
        "selected_adamlike_canary_rollout_policy_ready": summary.get(
            "selected_adamlike_canary_rollout_policy_ready"
        )
        is True,
        "selected_adamlike_owner_release_hold_ready": summary.get(
            "selected_adamlike_owner_release_hold_ready"
        )
        is True,
        "selected_adamlike_owner_release_hold_optimizer_count": int(
            summary.get("selected_adamlike_owner_release_hold_optimizer_count", 0) or 0
        ),
        "selected_adamlike_owner_release_hold_product_native_ready_count": int(
            summary.get("selected_adamlike_owner_release_hold_product_native_ready_count", 0) or 0
        ),
        "selected_adamlike_request_schema_ui_non_exposure_ready": summary.get(
            "selected_adamlike_request_schema_ui_non_exposure_ready"
        )
        is True,
        "selected_adamlike_request_schema_ui_optimizer_count": int(
            summary.get("selected_adamlike_request_schema_ui_optimizer_count", 0) or 0
        ),
        "selected_adamlike_request_schema_ui_forbidden_token_hit_count": int(
            summary.get("selected_adamlike_request_schema_ui_forbidden_token_hit_count", 0) or 0
        ),
        "selected_adamlike_request_schema_ui_product_native_ready_count": int(
            summary.get("selected_adamlike_request_schema_ui_product_native_ready_count", 0) or 0
        ),
        "selected_schedulefree_family_batch_ready": summary.get("selected_schedulefree_family_batch_ready") is True,
        "selected_schedulefree_e2e_shadow_case_count": int(
            summary.get("selected_schedulefree_e2e_shadow_case_count", 0) or 0
        ),
        "selected_schedulefree_native_canary_ready_count": int(
            summary.get("selected_schedulefree_native_canary_ready_count", 0) or 0
        ),
        "selected_schedulefree_dispatch_review_gate_ready": summary.get(
            "selected_schedulefree_dispatch_review_gate_ready"
        )
        is True,
        "selected_schedulefree_owner_release_hold_ready": summary.get(
            "selected_schedulefree_owner_release_hold_ready"
        )
        is True,
        "selected_schedulefree_owner_release_hold_optimizer_count": int(
            summary.get("selected_schedulefree_owner_release_hold_optimizer_count", 0) or 0
        ),
        "selected_schedulefree_owner_release_hold_product_native_ready_count": int(
            summary.get("selected_schedulefree_owner_release_hold_product_native_ready_count", 0) or 0
        ),
        "selected_schedulefree_request_schema_ui_non_exposure_ready": summary.get(
            "selected_schedulefree_request_schema_ui_non_exposure_ready"
        )
        is True,
        "selected_schedulefree_request_schema_ui_optimizer_count": int(
            summary.get("selected_schedulefree_request_schema_ui_optimizer_count", 0) or 0
        ),
        "selected_schedulefree_request_schema_ui_forbidden_token_hit_count": int(
            summary.get("selected_schedulefree_request_schema_ui_forbidden_token_hit_count", 0) or 0
        ),
        "selected_schedulefree_request_schema_ui_product_native_ready_count": int(
            summary.get("selected_schedulefree_request_schema_ui_product_native_ready_count", 0) or 0
        ),
        "selected_adaptivelr_family_batch_ready": summary.get("selected_adaptivelr_family_batch_ready") is True,
        "selected_adaptivelr_reference_ready_count": int(
            summary.get("selected_adaptivelr_reference_ready_count", 0) or 0
        ),
        "selected_adaptivelr_state_machine_abi_spec_ready_count": int(
            summary.get("selected_adaptivelr_state_machine_abi_spec_ready_count", 0) or 0
        ),
        "selected_adaptivelr_state_machine_abi_implementation_ready_count": int(
            summary.get("selected_adaptivelr_state_machine_abi_implementation_ready_count", 0) or 0
        ),
        "selected_adaptivelr_native_kernel_preconditions_spec_ready_count": int(
            summary.get("selected_adaptivelr_native_kernel_preconditions_spec_ready_count", 0) or 0
        ),
        "selected_adaptivelr_native_kernel_preconditions_implementation_ready_count": int(
            summary.get("selected_adaptivelr_native_kernel_preconditions_implementation_ready_count", 0) or 0
        ),
        "selected_adaptivelr_state_machine_replay_matrix_artifact_ready_count": int(
            summary.get("selected_adaptivelr_state_machine_replay_matrix_artifact_ready_count", 0) or 0
        ),
        "selected_adaptivelr_state_machine_replay_matrix_implementation_ready_count": int(
            summary.get("selected_adaptivelr_state_machine_replay_matrix_implementation_ready_count", 0) or 0
        ),
        "selected_adaptivelr_state_machine_replay_case_planned_count": int(
            summary.get("selected_adaptivelr_state_machine_replay_case_planned_count", 0) or 0
        ),
        "selected_adaptivelr_state_machine_replay_case_implementation_ready_count": int(
            summary.get("selected_adaptivelr_state_machine_replay_case_implementation_ready_count", 0) or 0
        ),
        "selected_adaptivelr_state_machine_replay_resume_case_planned_count": int(
            summary.get("selected_adaptivelr_state_machine_replay_resume_case_planned_count", 0) or 0
        ),
        "selected_adaptivelr_state_machine_replay_resume_case_implementation_ready_count": int(
            summary.get("selected_adaptivelr_state_machine_replay_resume_case_implementation_ready_count", 0) or 0
        ),
        "selected_adaptivelr_owner_release_hold_ready": summary.get(
            "selected_adaptivelr_owner_release_hold_ready"
        )
        is True,
        "selected_adaptivelr_owner_release_hold_optimizer_count": int(
            summary.get("selected_adaptivelr_owner_release_hold_optimizer_count", 0) or 0
        ),
        "selected_adaptivelr_owner_release_hold_product_native_ready_count": int(
            summary.get("selected_adaptivelr_owner_release_hold_product_native_ready_count", 0) or 0
        ),
        "selected_adaptivelr_request_schema_ui_non_exposure_ready": summary.get(
            "selected_adaptivelr_request_schema_ui_non_exposure_ready"
        )
        is True,
        "selected_adaptivelr_request_schema_ui_optimizer_count": int(
            summary.get("selected_adaptivelr_request_schema_ui_optimizer_count", 0) or 0
        ),
        "selected_adaptivelr_request_schema_ui_forbidden_token_hit_count": int(
            summary.get("selected_adaptivelr_request_schema_ui_forbidden_token_hit_count", 0) or 0
        ),
        "selected_adaptivelr_request_schema_ui_product_native_ready_count": int(
            summary.get("selected_adaptivelr_request_schema_ui_product_native_ready_count", 0) or 0
        ),
        "selected_simple_formula_family_batch_ready": summary.get("selected_simple_formula_family_batch_ready") is True,
        "selected_simple_formula_optimizer_count": int(summary.get("selected_simple_formula_optimizer_count", 0) or 0),
        "selected_simple_formula_reference_canary_ready_count": int(
            summary.get("selected_simple_formula_reference_canary_ready_count", 0) or 0
        ),
        "selected_simple_formula_native_canary_ready_count": int(
            summary.get("selected_simple_formula_native_canary_ready_count", 0) or 0
        ),
        "selected_simple_formula_e2e_shadow_matrix_ready": summary.get(
            "selected_simple_formula_e2e_shadow_matrix_ready"
        )
        is True,
        "selected_simple_formula_e2e_shadow_case_count": int(
            summary.get("selected_simple_formula_e2e_shadow_case_count", 0) or 0
        ),
        "selected_simple_formula_canary_rollout_policy_ready": summary.get(
            "selected_simple_formula_canary_rollout_policy_ready"
        )
        is True,
        "selected_simple_formula_canary_rollout_policy_ready_count": int(
            summary.get("selected_simple_formula_canary_rollout_policy_ready_count", 0) or 0
        ),
        "selected_simple_formula_dispatch_review_gate_ready": summary.get(
            "selected_simple_formula_dispatch_review_gate_ready"
        )
        is True,
        "selected_simple_formula_dispatch_review_ready_count": int(
            summary.get("selected_simple_formula_dispatch_review_ready_count", 0) or 0
        ),
        "selected_simple_formula_owner_release_hold_ready": summary.get(
            "selected_simple_formula_owner_release_hold_ready"
        )
        is True,
        "selected_simple_formula_owner_release_hold_optimizer_count": int(
            summary.get("selected_simple_formula_owner_release_hold_optimizer_count", 0) or 0
        ),
        "selected_simple_formula_owner_release_hold_product_native_ready_count": int(
            summary.get("selected_simple_formula_owner_release_hold_product_native_ready_count", 0) or 0
        ),
        "selected_simple_formula_request_schema_ui_non_exposure_ready": summary.get(
            "selected_simple_formula_request_schema_ui_non_exposure_ready"
        )
        is True,
        "selected_simple_formula_request_schema_ui_optimizer_count": int(
            summary.get("selected_simple_formula_request_schema_ui_optimizer_count", 0) or 0
        ),
        "selected_simple_formula_request_schema_ui_forbidden_token_hit_count": int(
            summary.get("selected_simple_formula_request_schema_ui_forbidden_token_hit_count", 0) or 0
        ),
        "selected_simple_formula_request_schema_ui_product_native_ready_count": int(
            summary.get("selected_simple_formula_request_schema_ui_product_native_ready_count", 0) or 0
        ),
        "selected_closure_second_order_family_batch_ready": summary.get(
            "selected_closure_second_order_family_batch_ready"
        )
        is True,
        "selected_closure_second_order_optimizer_count": int(
            summary.get("selected_closure_second_order_optimizer_count", 0) or 0
        ),
        "selected_closure_second_order_higher_order_abi_required_count": int(
            summary.get("selected_closure_second_order_higher_order_abi_required_count", 0) or 0
        ),
        "selected_closure_second_order_training_loop_abi_spec_ready_count": int(
            summary.get("selected_closure_second_order_training_loop_abi_spec_ready_count", 0) or 0
        ),
        "selected_closure_second_order_training_loop_abi_implementation_ready_count": int(
            summary.get("selected_closure_second_order_training_loop_abi_implementation_ready_count", 0) or 0
        ),
        "selected_closure_second_order_resume_parity_matrix_spec_ready_count": int(
            summary.get("selected_closure_second_order_resume_parity_matrix_spec_ready_count", 0) or 0
        ),
        "selected_closure_second_order_resume_parity_matrix_implementation_ready_count": int(
            summary.get("selected_closure_second_order_resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "selected_closure_second_order_closure_replay_case_planned_count": int(
            summary.get("selected_closure_second_order_closure_replay_case_planned_count", 0) or 0
        ),
        "selected_closure_second_order_create_graph_hvp_lifetime_case_planned_count": int(
            summary.get("selected_closure_second_order_create_graph_hvp_lifetime_case_planned_count", 0) or 0
        ),
        "selected_closure_second_order_closure_resume_replay_artifact_ready_count": int(
            summary.get("selected_closure_second_order_closure_resume_replay_artifact_ready_count", 0) or 0
        ),
        "selected_closure_second_order_closure_resume_replay_artifact_row_count": int(
            summary.get("selected_closure_second_order_closure_resume_replay_artifact_row_count", 0) or 0
        ),
        "selected_closure_second_order_closure_resume_replay_artifact_implementation_ready_count": int(
            summary.get(
                "selected_closure_second_order_closure_resume_replay_artifact_implementation_ready_count",
                0,
            )
            or 0
        ),
        "selected_closure_second_order_closure_resume_replay_row_implementation_ready_count": int(
            summary.get("selected_closure_second_order_closure_resume_replay_row_implementation_ready_count", 0) or 0
        ),
        "selected_closure_second_order_native_kernel_precondition_plan_ready_count": int(
            summary.get("selected_closure_second_order_native_kernel_precondition_plan_ready_count", 0) or 0
        ),
        "selected_closure_second_order_native_kernel_preconditions_implementation_ready_count": int(
            summary.get(
                "selected_closure_second_order_native_kernel_preconditions_implementation_ready_count",
                0,
            )
            or 0
        ),
        "selected_closure_second_order_owner_release_hold_ready": summary.get(
            "selected_closure_second_order_owner_release_hold_ready"
        )
        is True,
        "selected_closure_second_order_owner_release_hold_optimizer_count": int(
            summary.get("selected_closure_second_order_owner_release_hold_optimizer_count", 0) or 0
        ),
        "selected_closure_second_order_owner_release_hold_product_native_ready_count": int(
            summary.get("selected_closure_second_order_owner_release_hold_product_native_ready_count", 0) or 0
        ),
        "selected_closure_second_order_request_schema_ui_non_exposure_ready": summary.get(
            "selected_closure_second_order_request_schema_ui_non_exposure_ready"
        )
        is True,
        "selected_closure_second_order_request_schema_ui_optimizer_count": int(
            summary.get("selected_closure_second_order_request_schema_ui_optimizer_count", 0) or 0
        ),
        "selected_closure_second_order_request_schema_ui_forbidden_token_hit_count": int(
            summary.get("selected_closure_second_order_request_schema_ui_forbidden_token_hit_count", 0) or 0
        ),
        "selected_closure_second_order_request_schema_ui_product_native_ready_count": int(
            summary.get("selected_closure_second_order_request_schema_ui_product_native_ready_count", 0) or 0
        ),
        "selected_custom_formula_family_batch_ready": summary.get("selected_custom_formula_family_batch_ready") is True,
        "selected_custom_formula_optimizer_count": int(summary.get("selected_custom_formula_optimizer_count", 0) or 0),
        "selected_custom_formula_parity_required_count": int(
            summary.get("selected_custom_formula_parity_required_count", 0) or 0
        ),
        "selected_custom_formula_backlog_ready_count": int(
            summary.get("selected_custom_formula_backlog_ready_count", 0) or 0
        ),
        "selected_custom_formula_evidence_artifact_planned_count": int(
            summary.get("selected_custom_formula_evidence_artifact_planned_count", 0) or 0
        ),
        "selected_custom_formula_evidence_status_pending_total": int(
            summary.get("selected_custom_formula_evidence_status_pending_total", 0) or 0
        ),
        "selected_custom_formula_formula_spec_artifact_ready_count": int(
            summary.get("selected_custom_formula_formula_spec_artifact_ready_count", 0) or 0
        ),
        "selected_custom_formula_formula_spec_artifact_pending_count": int(
            summary.get("selected_custom_formula_formula_spec_artifact_pending_count", 0) or 0
        ),
        "selected_custom_formula_state_inventory_skeleton_count": int(
            summary.get("selected_custom_formula_state_inventory_skeleton_count", 0) or 0
        ),
        "selected_custom_formula_state_inventory_artifact_ready_count": int(
            summary.get("selected_custom_formula_state_inventory_artifact_ready_count", 0) or 0
        ),
        "selected_custom_formula_state_inventory_artifact_pending_count": int(
            summary.get("selected_custom_formula_state_inventory_artifact_pending_count", 0) or 0
        ),
        "selected_custom_formula_quality_guard_matrix_artifact_ready_count": int(
            summary.get("selected_custom_formula_quality_guard_matrix_artifact_ready_count", 0) or 0
        ),
        "selected_custom_formula_quality_guard_matrix_artifact_pending_count": int(
            summary.get("selected_custom_formula_quality_guard_matrix_artifact_pending_count", 0) or 0
        ),
        "selected_custom_formula_quality_guard_matrix_case_planned_count": int(
            summary.get("selected_custom_formula_quality_guard_matrix_case_planned_count", 0) or 0
        ),
        "selected_custom_formula_formula_parity_matrix_artifact_planned_count": int(
            summary.get("selected_custom_formula_formula_parity_matrix_artifact_planned_count", 0) or 0
        ),
        "selected_custom_formula_formula_parity_matrix_implementation_ready_count": int(
            summary.get("selected_custom_formula_formula_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "selected_custom_formula_formula_parity_case_planned_count": int(
            summary.get("selected_custom_formula_formula_parity_case_planned_count", 0) or 0
        ),
        "selected_custom_formula_resume_parity_matrix_artifact_planned_count": int(
            summary.get("selected_custom_formula_resume_parity_matrix_artifact_planned_count", 0) or 0
        ),
        "selected_custom_formula_resume_parity_matrix_implementation_ready_count": int(
            summary.get("selected_custom_formula_resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "selected_custom_formula_resume_parity_case_planned_count": int(
            summary.get("selected_custom_formula_resume_parity_case_planned_count", 0) or 0
        ),
        "selected_custom_formula_execution_matrix_ready": summary.get(
            "selected_custom_formula_execution_matrix_ready"
        )
        is True,
        "selected_custom_formula_step_execution_ready_count": int(
            summary.get("selected_custom_formula_step_execution_ready_count", 0) or 0
        ),
        "selected_custom_formula_resume_next_step_replay_ready_count": int(
            summary.get("selected_custom_formula_resume_next_step_replay_ready_count", 0) or 0
        ),
        "selected_custom_formula_execution_failed_count": int(
            summary.get("selected_custom_formula_execution_failed_count", 0) or 0
        ),
        "selected_custom_formula_owner_release_hold_ready": summary.get(
            "selected_custom_formula_owner_release_hold_ready"
        )
        is True,
        "selected_custom_formula_owner_release_hold_optimizer_count": int(
            summary.get("selected_custom_formula_owner_release_hold_optimizer_count", 0) or 0
        ),
        "selected_custom_formula_owner_release_hold_product_native_ready_count": int(
            summary.get("selected_custom_formula_owner_release_hold_product_native_ready_count", 0) or 0
        ),
        "selected_custom_formula_request_schema_ui_non_exposure_ready": summary.get(
            "selected_custom_formula_request_schema_ui_non_exposure_ready"
        )
        is True,
        "selected_custom_formula_request_schema_ui_optimizer_count": int(
            summary.get("selected_custom_formula_request_schema_ui_optimizer_count", 0) or 0
        ),
        "selected_custom_formula_request_schema_ui_forbidden_token_hit_count": int(
            summary.get("selected_custom_formula_request_schema_ui_forbidden_token_hit_count", 0) or 0
        ),
        "selected_custom_formula_request_schema_ui_product_native_ready_count": int(
            summary.get("selected_custom_formula_request_schema_ui_product_native_ready_count", 0) or 0
        ),
        "selected_factored_memory_family_batch_ready": summary.get("selected_factored_memory_family_batch_ready")
        is True,
        "selected_factored_memory_optimizer_count": int(
            summary.get("selected_factored_memory_optimizer_count", 0) or 0
        ),
        "selected_factored_memory_observed_layout_count": int(
            summary.get("selected_factored_memory_observed_layout_count", 0) or 0
        ),
        "selected_factored_memory_native_layout_abi_ready_count": int(
            summary.get("selected_factored_memory_native_layout_abi_ready_count", 0) or 0
        ),
        "selected_factored_memory_quality_matrix_ready_count": int(
            summary.get("selected_factored_memory_quality_matrix_ready_count", 0) or 0
        ),
        "selected_factored_memory_native_kernel_entry_condition_ready_count": int(
            summary.get("selected_factored_memory_native_kernel_entry_condition_ready_count", 0) or 0
        ),
        "selected_factored_memory_formula_tensor_binding_matrix_artifact_ready_count": int(
            summary.get("selected_factored_memory_formula_tensor_binding_matrix_artifact_ready_count", 0) or 0
        ),
        "selected_factored_memory_formula_tensor_binding_matrix_implementation_ready_count": int(
            summary.get("selected_factored_memory_formula_tensor_binding_matrix_implementation_ready_count", 0) or 0
        ),
        "selected_factored_memory_formula_step_execution_ready_count": int(
            summary.get("selected_factored_memory_formula_step_execution_ready_count", 0) or 0
        ),
        "selected_factored_memory_resume_next_step_replay_ready_count": int(
            summary.get("selected_factored_memory_resume_next_step_replay_ready_count", 0) or 0
        ),
        "selected_factored_memory_tensor_binding_ready_count": int(
            summary.get("selected_factored_memory_tensor_binding_ready_count", 0) or 0
        ),
        "selected_factored_memory_dispatch_review_gate_ready": summary.get(
            "selected_factored_memory_dispatch_review_gate_ready"
        )
        is True,
        "selected_factored_memory_dispatch_review_ready_count": int(
            summary.get("selected_factored_memory_dispatch_review_ready_count", 0) or 0
        ),
        "selected_factored_memory_formula_parity_case_planned_count": int(
            summary.get("selected_factored_memory_formula_parity_case_planned_count", 0) or 0
        ),
        "selected_factored_memory_tensor_binding_case_planned_count": int(
            summary.get("selected_factored_memory_tensor_binding_case_planned_count", 0) or 0
        ),
        "selected_factored_memory_owner_release_hold_ready": summary.get(
            "selected_factored_memory_owner_release_hold_ready"
        )
        is True,
        "selected_factored_memory_owner_release_hold_optimizer_count": int(
            summary.get("selected_factored_memory_owner_release_hold_optimizer_count", 0) or 0
        ),
        "selected_factored_memory_owner_release_hold_product_native_ready_count": int(
            summary.get("selected_factored_memory_owner_release_hold_product_native_ready_count", 0) or 0
        ),
        "selected_factored_memory_request_schema_ui_non_exposure_ready": summary.get(
            "selected_factored_memory_request_schema_ui_non_exposure_ready"
        )
        is True,
        "selected_factored_memory_request_schema_ui_optimizer_count": int(
            summary.get("selected_factored_memory_request_schema_ui_optimizer_count", 0) or 0
        ),
        "selected_factored_memory_request_schema_ui_forbidden_token_hit_count": int(
            summary.get("selected_factored_memory_request_schema_ui_forbidden_token_hit_count", 0) or 0
        ),
        "selected_factored_memory_request_schema_ui_product_native_ready_count": int(
            summary.get("selected_factored_memory_request_schema_ui_product_native_ready_count", 0) or 0
        ),
        "selected_fused_backward_family_batch_ready": summary.get("selected_fused_backward_family_batch_ready") is True,
        "selected_fused_backward_optimizer_count": int(summary.get("selected_fused_backward_optimizer_count", 0) or 0),
        "selected_fused_backward_gradient_ownership_abi_required_count": int(
            summary.get("selected_fused_backward_gradient_ownership_abi_required_count", 0) or 0
        ),
        "selected_fused_backward_per_optimizer_abi_spec_ready_count": int(
            summary.get("selected_fused_backward_per_optimizer_abi_spec_ready_count", 0) or 0
        ),
        "selected_fused_backward_abi_implementation_ready_count": int(
            summary.get("selected_fused_backward_abi_implementation_ready_count", 0) or 0
        ),
        "selected_fused_backward_native_kernel_preconditions_spec_ready_count": int(
            summary.get("selected_fused_backward_native_kernel_preconditions_spec_ready_count", 0) or 0
        ),
        "selected_fused_backward_resume_parity_matrix_spec_ready_count": int(
            summary.get("selected_fused_backward_resume_parity_matrix_spec_ready_count", 0) or 0
        ),
        "selected_fused_backward_resume_parity_matrix_implementation_ready_count": int(
            summary.get("selected_fused_backward_resume_parity_matrix_implementation_ready_count", 0) or 0
        ),
        "selected_fused_backward_replay_case_planned_count": int(
            summary.get("selected_fused_backward_replay_case_planned_count", 0) or 0
        ),
        "selected_fused_backward_replay_case_implementation_ready_count": int(
            summary.get("selected_fused_backward_replay_case_implementation_ready_count", 0) or 0
        ),
        "selected_fused_backward_loss_scale_boundary_case_planned_count": int(
            summary.get("selected_fused_backward_loss_scale_boundary_case_planned_count", 0) or 0
        ),
        "selected_fused_backward_owner_release_hold_ready": summary.get(
            "selected_fused_backward_owner_release_hold_ready"
        )
        is True,
        "selected_fused_backward_owner_release_hold_optimizer_count": int(
            summary.get("selected_fused_backward_owner_release_hold_optimizer_count", 0) or 0
        ),
        "selected_fused_backward_owner_release_hold_product_native_ready_count": int(
            summary.get("selected_fused_backward_owner_release_hold_product_native_ready_count", 0) or 0
        ),
        "selected_fused_backward_request_schema_ui_non_exposure_ready": summary.get(
            "selected_fused_backward_request_schema_ui_non_exposure_ready"
        )
        is True,
        "selected_fused_backward_request_schema_ui_optimizer_count": int(
            summary.get("selected_fused_backward_request_schema_ui_optimizer_count", 0) or 0
        ),
        "selected_fused_backward_request_schema_ui_forbidden_token_hit_count": int(
            summary.get("selected_fused_backward_request_schema_ui_forbidden_token_hit_count", 0) or 0
        ),
        "selected_fused_backward_request_schema_ui_product_native_ready_count": int(
            summary.get("selected_fused_backward_request_schema_ui_product_native_ready_count", 0) or 0
        ),
        "selected_model_shape_aware_family_batch_ready": summary.get("selected_model_shape_aware_family_batch_ready")
        is True,
        "selected_model_shape_aware_optimizer_count": int(
            summary.get("selected_model_shape_aware_optimizer_count", 0) or 0
        ),
        "selected_model_shape_aware_param_group_contract_count": int(
            summary.get("selected_model_shape_aware_param_group_contract_count", 0) or 0
        ),
        "selected_model_shape_aware_param_group_abi_spec_ready_count": int(
            summary.get("selected_model_shape_aware_param_group_abi_spec_ready_count", 0) or 0
        ),
        "selected_model_shape_aware_param_group_abi_implementation_ready_count": int(
            summary.get("selected_model_shape_aware_param_group_abi_implementation_ready_count", 0) or 0
        ),
        "selected_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count": int(
            summary.get(
                "selected_model_shape_aware_param_group_resume_replay_matrix_artifact_ready_count",
                0,
            )
            or 0
        ),
        "selected_model_shape_aware_param_group_resume_replay_matrix_row_count": int(
            summary.get("selected_model_shape_aware_param_group_resume_replay_matrix_row_count", 0) or 0
        ),
        "selected_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count": int(
            summary.get(
                "selected_model_shape_aware_param_group_resume_replay_matrix_implementation_ready_count",
                0,
            )
            or 0
        ),
        "selected_model_shape_aware_param_group_resume_replay_row_implementation_ready_count": int(
            summary.get("selected_model_shape_aware_param_group_resume_replay_row_implementation_ready_count", 0) or 0
        ),
        "selected_state_adapter_special_family_batch_ready": summary.get(
            "selected_state_adapter_special_family_batch_ready"
        )
        is True,
        "selected_state_adapter_special_optimizer_count": int(
            summary.get("selected_state_adapter_special_optimizer_count", 0) or 0
        ),
        "selected_state_adapter_special_param_ownership_abi_required_count": int(
            summary.get("selected_state_adapter_special_param_ownership_abi_required_count", 0) or 0
        ),
        "selected_state_adapter_special_adapter_abi_spec_ready_count": int(
            summary.get("selected_state_adapter_special_adapter_abi_spec_ready_count", 0) or 0
        ),
        "selected_state_adapter_special_adapter_abi_implementation_ready_count": int(
            summary.get("selected_state_adapter_special_adapter_abi_implementation_ready_count", 0) or 0
        ),
        "selected_state_adapter_special_native_kernel_precondition_spec_ready_count": int(
            summary.get("selected_state_adapter_special_native_kernel_precondition_spec_ready_count", 0) or 0
        ),
        "selected_state_adapter_special_native_kernel_precondition_implementation_ready_count": int(
            summary.get(
                "selected_state_adapter_special_native_kernel_precondition_implementation_ready_count",
                0,
            )
            or 0
        ),
        "selected_state_adapter_special_resume_matrix_artifact_ready_count": int(
            summary.get("selected_state_adapter_special_resume_matrix_artifact_ready_count", 0) or 0
        ),
        "selected_state_adapter_special_resume_matrix_implementation_ready_count": int(
            summary.get("selected_state_adapter_special_resume_matrix_implementation_ready_count", 0) or 0
        ),
        "selected_state_adapter_special_resume_replay_case_planned_count": int(
            summary.get("selected_state_adapter_special_resume_replay_case_planned_count", 0) or 0
        ),
        "selected_state_adapter_special_resume_replay_case_implementation_ready_count": int(
            summary.get("selected_state_adapter_special_resume_replay_case_implementation_ready_count", 0) or 0
        ),
        "selected_state_adapter_special_resume_translation_case_planned_count": int(
            summary.get("selected_state_adapter_special_resume_translation_case_planned_count", 0) or 0
        ),
        "selected_state_adapter_special_resume_translation_case_implementation_ready_count": int(
            summary.get("selected_state_adapter_special_resume_translation_case_implementation_ready_count", 0) or 0
        ),
        "plugin_factored_memory_layout_observed_count": int(
            summary.get("plugin_factored_memory_layout_observed_count", 0) or 0
        ),
        "plugin_selected_native_ready_count": int(summary.get("plugin_selected_native_ready_count", 0) or 0),
        "route_family_counts": dict(_as_dict(summary.get("route_family_counts"))),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_selected_family_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "plugin_optimizer_family_batch_ready": report.get("plugin_optimizer_family_batch_ready") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "family_count": int(summary.get("family_count", 0) or 0),
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_muon_model_shape_aware_batch(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "muon_model_shape_aware_family_batch_ready": report.get(
            "muon_model_shape_aware_family_batch_ready"
        )
        is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_kernel_ready": bool(report.get("native_kernel_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "param_group_abi_spec_ready_count": int(summary.get("param_group_abi_spec_ready_count", 0) or 0),
        "param_group_abi_implementation_ready_count": int(
            summary.get("param_group_abi_implementation_ready_count", 0) or 0
        ),
        "param_group_resume_replay_matrix_artifact_ready_count": int(
            summary.get("param_group_resume_replay_matrix_artifact_ready_count", 0) or 0
        ),
        "param_group_resume_replay_matrix_row_count": int(
            summary.get("param_group_resume_replay_matrix_row_count", 0) or 0
        ),
        "param_group_resume_replay_matrix_implementation_ready_count": int(
            summary.get("param_group_resume_replay_matrix_implementation_ready_count", 0) or 0
        ),
        "param_group_resume_replay_row_implementation_ready_count": int(
            summary.get("param_group_resume_replay_row_implementation_ready_count", 0) or 0
        ),
        "native_kernel_precondition_ready_count": int(
            summary.get("native_kernel_precondition_ready_count", 0) or 0
        ),
        "runtime_dispatch_shadow_ready_count": int(
            summary.get("runtime_dispatch_shadow_ready_count", 0) or 0
        ),
        "dispatch_integration_review_ready_count": int(
            summary.get("dispatch_integration_review_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_muon_native_scratch_kernel(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "native_scratch_kernel_ready": report.get("native_scratch_kernel_ready") is True,
        "native_kernel_ready": report.get("native_kernel_ready") is True,
        "runtime_canary_ready": report.get("runtime_canary_ready") is True,
        "runtime_canary_hit": report.get("runtime_canary_hit") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_native_ready": bool(report.get("product_native_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "native_scratch_kernel_ready_count": int(
            summary.get("native_scratch_kernel_ready_count", 0) or 0
        ),
        "kernel_executed_count": int(summary.get("kernel_executed_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_muon_training_tensor_binding(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "training_tensor_binding_canary_ready": report.get("training_tensor_binding_canary_ready") is True,
        "training_tensor_binding_parity_ready": report.get("training_tensor_binding_parity_ready") is True,
        "training_tensor_binding_probe_ready": report.get("training_tensor_binding_probe_ready") is True,
        "native_scratch_kernel_ready": report.get("native_scratch_kernel_ready") is True,
        "runtime_canary_ready": report.get("runtime_canary_ready") is True,
        "runtime_canary_hit": report.get("runtime_canary_hit") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_native_ready": bool(report.get("product_native_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "training_tensor_binding_canary_ready_count": int(
            summary.get("training_tensor_binding_canary_ready_count", 0) or 0
        ),
        "training_tensor_binding_parity_ready_count": int(
            summary.get("training_tensor_binding_parity_ready_count", 0) or 0
        ),
        "kernel_executed_count": int(summary.get("kernel_executed_count", 0) or 0),
        "training_parameters_mutated_count": int(
            summary.get("training_parameters_mutated_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_muon_training_loop_canary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "training_loop_canary_ready": report.get("training_loop_canary_ready") is True,
        "training_loop_canary_hit": report.get("training_loop_canary_hit") is True,
        "training_tensor_binding_canary_ready": report.get("training_tensor_binding_canary_ready") is True,
        "training_tensor_binding_parity_ready": report.get("training_tensor_binding_parity_ready") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_native_ready": bool(report.get("product_native_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "training_loop_canary_ready_count": int(
            summary.get("training_loop_canary_ready_count", 0) or 0
        ),
        "native_step_count": int(summary.get("native_step_count", 0) or 0),
        "native_kernel_launch_count": int(summary.get("native_kernel_launch_count", 0) or 0),
        "training_executor_called_count": int(summary.get("training_executor_called_count", 0) or 0),
        "training_parameters_mutated_count": int(
            summary.get("training_parameters_mutated_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_muon_e2e_shadow_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "report_only": report.get("report_only") is True,
        "e2e_shadow_matrix_ready": report.get("e2e_shadow_matrix_ready") is True,
        "e2e_shadow_matrix_passed": report.get("e2e_shadow_matrix_passed") is True,
        "report_only_matrix_scaffold_ready": report.get("report_only_matrix_scaffold_ready") is True,
        "live_shadow_matrix_executed": report.get("live_shadow_matrix_executed") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_native_ready": bool(report.get("product_native_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "case_count": int(summary.get("case_count", 0) or 0),
        "report_only_case_count": int(summary.get("report_only_case_count", 0) or 0),
        "failed_case_count": int(summary.get("failed_case_count", 0) or 0),
        "e2e_shadow_matrix_ready_count": int(
            summary.get("e2e_shadow_matrix_ready_count", 0) or 0
        ),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_muon_canary_rollout_policy(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    policy = _as_dict(report.get("policy"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "canary_rollout_policy_ready": report.get("canary_rollout_policy_ready") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "canary_auto_enabled": report.get("canary_auto_enabled") is True,
        "canary_enabled_by_default": policy.get("canary_enabled_by_default") is True,
        "explicit_opt_in_required": policy.get("explicit_opt_in_required") is True,
        "max_canary_fraction_default": float(policy.get("max_canary_fraction_default", 0.0) or 0.0),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "product_native_ready": bool(report.get("product_native_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "canary_rollout_policy_ready_count": int(
            summary.get("canary_rollout_policy_ready_count", 0) or 0
        ),
        "runtime_dispatch_ready_count": int(summary.get("runtime_dispatch_ready_count", 0) or 0),
        "native_dispatch_allowed_count": int(summary.get("native_dispatch_allowed_count", 0) or 0),
        "training_path_enabled_count": int(summary.get("training_path_enabled_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_muon_dispatch_review(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "review_gate_ready": report.get("review_gate_ready") is True,
        "dispatch_integration_review": report.get("dispatch_integration_review") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "default_behavior_changed": bool(report.get("default_behavior_changed", False)),
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "product_native_ready": bool(report.get("product_native_ready", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_muon_owner_release_hold(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "family_batch_ready": report.get("family_batch_ready") is True,
        "canary_rollout_policy_ready": report.get("canary_rollout_policy_ready") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_kernel_ready": bool(report.get("native_kernel_ready", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_muon_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "native_kernel_ready": bool(report.get("native_kernel_ready", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "optimizer_count": int(summary.get("optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _compact_plugin_selected_family_request_schema_ui(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    return {
        "present": bool(report),
        "ok": report.get("ok") is True,
        "request_schema_ui_non_exposure_ready": report.get("request_schema_ui_non_exposure_ready") is True,
        "owner_release_hold_ready": report.get("owner_release_hold_ready") is True,
        "manual_review_required": report.get("manual_review_required") is True,
        "owner_approval_recorded": report.get("owner_approval_recorded") is True,
        "release_approval_recorded": report.get("release_approval_recorded") is True,
        "runtime_dispatch_ready": bool(report.get("runtime_dispatch_ready", False)),
        "native_dispatch_allowed": bool(report.get("native_dispatch_allowed", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "request_fields_emitted": bool(report.get("request_fields_emitted", False)),
        "schema_exposure_allowed": bool(report.get("schema_exposure_allowed", False)),
        "ui_exposure_allowed": bool(report.get("ui_exposure_allowed", False)),
        "family_count": int(summary.get("family_count", 0) or 0),
        "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
        "forbidden_token_hit_count": int(summary.get("forbidden_token_hit_count", 0) or 0),
        "product_native_ready_count": int(summary.get("product_native_ready_count", 0) or 0),
        "blocked_reasons": _strings(report.get("blocked_reasons")),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = ["build_optimizer_family_coverage_scorecard"]
