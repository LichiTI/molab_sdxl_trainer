"""First-release stable-baseline readiness audit for Lulynx.

This module is JSON-only. It summarizes whether the current stable baseline
can ship as the first release while keeping research and experimental paths
fail-closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIRST_RELEASE_READINESS_REPORT = "lulynx_first_release_readiness_v0"

DEFAULT_GATE_JSON = "temp/lossless_resource_center_gate.json"
DEFAULT_FULL_TRAINER_READINESS_JSON = "temp/lossless_full_trainer_ab_readiness.json"
DEFAULT_RELEASE_SMOKE_JSON = "temp/lulynx_release_smoke.json"
DEFAULT_BATCH1_PARITY_SMOKE_JSON = "temp/lulynx_batch1_handler_parity_smoke.json"
DEFAULT_PIPELINE_REFACTOR_READINESS_JSON = (
    "temp/lulynx_training_pipeline_refactor_readiness_evidence.json"
)
DEFAULT_CACHE_CONTAINER_PLAN_JSON = "temp/lossless_cache_container_format_research_plan.json"
DEFAULT_CACHE_CONTAINER_READ_PATH_GATE_JSON = "temp/lossless_cache_container_read_path_gate.json"
DEFAULT_CACHE_CONTAINER_ADMISSION_CONTRACT_JSON = (
    "temp/lossless_cache_container_admission_contract.json"
)
DEFAULT_CACHE_CONTAINER_NEXT_AXIS_CONTRACT_JSON = (
    "temp/lossless_cache_container_next_axis_contract.json"
)
DEFAULT_CACHE_CONTAINER_PAYLOAD_LAYOUT_AXIS_CONTRACT_JSON = (
    "temp/lossless_cache_container_payload_layout_axis_contract.json"
)
DEFAULT_CACHE_CONTAINER_BASELINE_COMPARISON_CONTRACT_JSON = (
    "temp/lossless_cache_container_baseline_comparison_contract.json"
)
DEFAULT_CACHE_CONTAINER_COMPUTE_TAIL_NON_REGRESSION_CONTRACT_JSON = (
    "temp/lossless_cache_container_compute_tail_non_regression_contract.json"
)
DEFAULT_CACHE_CONTAINER_DECODE_IMPLEMENTATION_CONTRACT_JSON = (
    "temp/lossless_cache_container_decode_implementation_contract.json"
)
DEFAULT_CACHE_CONTAINER_DECODE_PATH_IMPLEMENTATION_PLAN_JSON = (
    "temp/lossless_cache_container_decode_path_implementation_plan.json"
)
DEFAULT_P4_P6_ACTION_PLAN_JSON = "temp/lossless_p4_p6_action_plan.json"
DEFAULT_P5_D3D12_CUDA_INTEROP_CONTRACT_JSON = (
    "temp/lossless_p5_d3d12_cuda_interop_contract.json"
)
DEFAULT_P5_D3D12_CUDA_FUNCTIONAL_INTEROP_PROBE_JSON = (
    "temp/lossless_p5_d3d12_cuda_functional_interop_probe.json"
)
DEFAULT_P5_D3D12_CUDA_FUNCTIONAL_PROBE_HARNESS_MANIFEST_JSON = (
    "temp/lossless_p5_d3d12_cuda_functional_probe_harness_manifest.json"
)
DEFAULT_P5_P6_CUDA_TENSOR_VIEW_BINDING_PREFLIGHT_JSON = (
    "temp/lossless_p5_p6_cuda_tensor_view_binding_preflight.json"
)
DEFAULT_P5_P6_GUARDED_TRAINER_RUNTIME_AB_PREFLIGHT_JSON = (
    "temp/lossless_p5_p6_guarded_trainer_runtime_ab_preflight.json"
)
DEFAULT_P5_P6_RUNTIME_AB_APPROVAL_PACKET_JSON = (
    "temp/lossless_p5_p6_runtime_ab_approval_packet.json"
)
DEFAULT_P6_COMPUTE_DECODE_READINESS_JSON = (
    "temp/lossless_p6_d3d12_cuda_compute_decode_readiness.json"
)
DEFAULT_P5_P6_NATIVE_BLUEPRINT_JSON = (
    "temp/lossless_p5_p6_native_implementation_blueprint.json"
)
DEFAULT_PHASE_JITTER_PRODUCT_GATE_JSON = (
    "temp/lossless_full_trainer_phase_jitter_product_gate.json"
)
DEFAULT_PHASE_JITTER_CONTROL_DELTA_JSON = (
    "temp/lossless_full_trainer_phase_jitter_control_delta.json"
)
DEFAULT_PHASE_JITTER_PHASE_ATTRIBUTION_JSON = (
    "temp/lossless_full_trainer_phase_jitter_phase_attribution.json"
)
DEFAULT_PHASE_JITTER_OPTIMIZER_UPDATE_PLAN_JSON = (
    "temp/lossless_full_trainer_phase_jitter_optimizer_update_plan.json"
)
DEFAULT_PHASE_JITTER_OPTIMIZER_UPDATE_DELTA_JSON = (
    "temp/lossless_full_trainer_phase_jitter_optimizer_update_delta.json"
)
DEFAULT_PHASE_JITTER_OPTIMIZER_UPDATE_REPEAT_PLAN_JSON = (
    "temp/lossless_full_trainer_phase_jitter_optimizer_update_repeat_plan.json"
)
DEFAULT_COMPUTE_TAIL_FOCUSED_RERUN_RESULTS_JSON = (
    "temp/lossless_compute_tail_focused_rerun_results.json"
)
DEFAULT_COMPUTE_TAIL_RAW_ORDER_JITTER_REPEAT_PHASE_ATTRIBUTION_JSON = (
    "temp/lossless_compute_tail_raw_order_jitter_repeat_phase_attribution.json"
)
DEFAULT_COMPUTE_TAIL_RAW_ORDER_JITTER_ORDER_NEUTRAL_EXCESS_MITIGATION_CONTRACT_JSON = (
    "temp/lossless_compute_tail_raw_order_jitter_order_neutral_excess_mitigation_contract.json"
)
DEFAULT_GUARDED_OPTIMIZER_UPDATE_TAIL_MITIGATION_DESIGN_JSON = (
    "temp/lossless_full_trainer_guarded_optimizer_update_tail_mitigation_design.json"
)
DEFAULT_GUARDED_OPTIMIZER_UPDATE_TAIL_VALIDATION_MANIFEST_JSON = (
    "temp/lossless_full_trainer_guarded_optimizer_update_tail_validation_manifest.json"
)
DEFAULT_RAW_ORDER_JITTER_GUARDED_COMPUTE_PHASE_MITIGATION_DESIGN_JSON = (
    "temp/lossless_compute_tail_raw_order_jitter_guarded_compute_phase_mitigation_design.json"
)
DEFAULT_RAW_ORDER_JITTER_GUARDED_COMPUTE_PHASE_VALIDATION_MANIFEST_JSON = (
    "temp/lossless_compute_tail_raw_order_jitter_guarded_compute_phase_validation_manifest.json"
)
DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_RUN_MANIFEST_JSON = (
    "temp/lossless_compute_tail_raw_order_compute_phase_validation_run_manifest.json"
)
DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_PLAYBOOK_JSON = (
    "temp/lossless_compute_tail_raw_order_compute_phase_validation_playbook.json"
)
DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_RESULTS_JSON = (
    "temp/lossless_compute_tail_raw_order_compute_phase_validation_results.json"
)
DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_VERDICT_JSON = (
    "temp/lossless_compute_tail_raw_order_compute_phase_validation_verdict.json"
)
DEFAULT_P7_NON_HEAVY_NEXT_ACTION_JSON = (
    "temp/lossless_p7_non_heavy_next_action.json"
)
DEFAULT_P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_JSON = (
    "temp/lossless_p7_manual_heavy_authorization_bundle.json"
)
DEFAULT_P7_VALIDATION_FAILURE_TRIAGE_JSON = (
    "temp/lossless_p7_validation_failure_triage.json"
)
DEFAULT_P7_FAILED_ITEM_BLOCKER_RESOLUTION_MATRIX_JSON = (
    "temp/lossless_p7_failed_item_blocker_resolution_matrix.json"
)
DEFAULT_P7_BACKWARD_FORWARD_PHASE_GUARD_CONTRACT_JSON = (
    "temp/lossless_p7_backward_forward_phase_guard_contract.json"
)
DEFAULT_P7_BACKWARD_FORWARD_PHASE_GUARD_VALIDATION_RECOMPUTE_JSON = (
    "temp/lossless_p7_backward_forward_phase_guard_validation_recompute.json"
)
DEFAULT_P7_GUARDED_RAW_ORDER_COMPUTE_PHASE_VARIANT_CONTRACT_JSON = (
    "temp/lossless_p7_guarded_raw_order_compute_phase_variant_contract.json"
)
DEFAULT_P7_GUARDED_VARIANT_CROSS_DOMAIN_EVIDENCE_CONTRACT_JSON = (
    "temp/lossless_p7_guarded_variant_cross_domain_evidence_contract.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_RESIDUAL_GUARD_CONTRACT_JSON = (
    "temp/lossless_p7_optimizer_update_residual_guard_contract.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_RESIDUAL_GUARD_RECHECK_AFTER_NONREPRO_JSON = (
    "temp/lossless_p7_optimizer_update_residual_guard_recheck_after_nonrepro.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_REPEAT_POSITIVE_CONTROL_RECLASSIFICATION_JSON = (
    "temp/lossless_p7_optimizer_update_repeat_positive_control_reclassification_contract.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_REPEAT_POSITIVE_CONTROL_RESOLUTION_JSON = (
    "temp/lossless_p7_optimizer_update_repeat_positive_control_resolution_contract.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_REPEAT_POSITIVE_OPTIMIZER_INTERNAL_RESOLUTION_JSON = (
    "temp/lossless_p7_optimizer_update_repeat_positive_optimizer_internal_resolution_contract.json"
)
DEFAULT_P7_OPTIMIZER_STEP_MICRO_ATTRIBUTION_JSON = (
    "temp/lossless_p7_optimizer_step_micro_attribution_contract.json"
)
DEFAULT_P7_OPTIMIZER_STEP_MICRO_PROFILE_INSTRUMENTATION_JSON = (
    "temp/lossless_p7_optimizer_step_micro_profile_instrumentation_contract.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_RESIDUAL_GUARD_VALIDATION_RECOMPUTE_JSON = (
    "temp/lossless_p7_optimizer_update_residual_guard_validation_recompute.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_UNACCOUNTED_TAIL_ISOLATION_JSON = (
    "temp/lossless_p7_optimizer_update_unaccounted_tail_isolation.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_OUTER_PHASE_SUBSTAGE_INSTRUMENTATION_CONTRACT_JSON = (
    "temp/lossless_p7_optimizer_update_outer_phase_substage_instrumentation_contract.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_OUTER_PHASE_SUBSTAGE_PROFILE_IMPLEMENTATION_JSON = (
    "temp/lossless_p7_optimizer_update_outer_phase_substage_profile_implementation.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_OUTER_SUBSTAGE_TAIL_ATTRIBUTION_JSON = (
    "temp/lossless_p7_optimizer_update_outer_substage_tail_attribution.json"
)
DEFAULT_P7_ORDER_NEUTRAL_RESIDUAL_CONTRACT_SCOPE_CONTRACT_JSON = (
    "temp/lossless_p7_order_neutral_residual_contract_scope_contract.json"
)
DEFAULT_GUARDED_VARIANT_REGRESSION_ACTION_PLAN_JSON = (
    "temp/lossless_compute_tail_raw_order_guarded_variant_regression_action_plan.json"
)
DEFAULT_GUARDED_VARIANT_MITIGATION_BLUEPRINT_JSON = (
    "temp/lossless_compute_tail_raw_order_guarded_variant_mitigation_blueprint.json"
)
DEFAULT_REPLACEMENT_PHASE_GUARD_CONTRACT_JSON = (
    "temp/lossless_replacement_phase_guard_contract.json"
)
DEFAULT_GUARDED_VARIANT_RUNTIME_CONTRACT_JSON = (
    "temp/lossless_guarded_variant_runtime_contract.json"
)
DEFAULT_GUARDED_VARIANT_REQUEST_ADAPTER_CONTRACT_JSON = (
    "temp/lossless_guarded_variant_request_adapter_contract.json"
)
DEFAULT_DATALOADER_GUARD_METADATA_JSON = (
    "temp/lossless_dataloader_guard_metadata.json"
)

_RESEARCH_ARTIFACT_GATE_SPECS = [
    (
        "cache_container_format_research_plan",
        "lossless_cache_container_format_research_plan",
        "lossless_cache_container_research_gate_open",
        "cache_container_plan_checked",
        "cache_container_plan",
    ),
    (
        "cache_container_read_path_gate",
        "lossless_cache_container_read_path_gate",
        "lossless_cache_container_read_path_gate_open",
        "cache_container_read_path_gate_checked",
        "cache_container_read_path_gate",
    ),
    (
        "cache_container_admission_contract",
        "lossless_cache_container_admission_contract",
        "lossless_cache_container_admission_contract_gate_open",
        "cache_container_admission_contract_checked",
        "cache_container_admission_contract",
    ),
    (
        "cache_container_next_axis_contract",
        "lossless_cache_container_next_axis_contract",
        "lossless_cache_container_next_axis_contract_gate_open",
        "cache_container_next_axis_contract_checked",
        "cache_container_next_axis_contract",
    ),
    (
        "cache_container_payload_layout_axis_contract",
        "lossless_cache_container_payload_layout_axis_contract",
        "lossless_cache_container_payload_layout_axis_contract_gate_open",
        "cache_container_payload_layout_axis_contract_checked",
        "cache_container_payload_layout_axis_contract",
    ),
    (
        "cache_container_baseline_comparison_contract",
        "lossless_cache_container_baseline_comparison_contract",
        "lossless_cache_container_baseline_comparison_contract_gate_open",
        "cache_container_baseline_comparison_contract_checked",
        "cache_container_baseline_comparison_contract",
    ),
    (
        "cache_container_compute_tail_non_regression_contract",
        "lossless_cache_container_compute_tail_non_regression_contract",
        "lossless_cache_container_compute_tail_non_regression_contract_gate_open",
        "cache_container_compute_tail_non_regression_contract_checked",
        "cache_container_compute_tail_non_regression_contract",
    ),
    (
        "cache_container_decode_implementation_contract",
        "lossless_cache_container_decode_implementation_contract",
        "lossless_cache_container_decode_implementation_contract_gate_open",
        "cache_container_decode_implementation_contract_checked",
        "cache_container_decode_implementation_contract",
    ),
    (
        "cache_container_decode_path_implementation_plan",
        "lossless_cache_container_decode_path_implementation_plan",
        "lossless_cache_container_decode_path_implementation_plan_gate_open",
        "cache_container_decode_path_implementation_plan_checked",
        "cache_container_decode_path_implementation_plan",
    ),
    (
        "p4_p6_action_plan",
        "lossless_p4_p6_action_plan",
        "lossless_p4_p6_research_gate_open",
        "p4_p6_action_plan_checked",
        "p4_p6_action_plan",
    ),
    (
        "p5_d3d12_cuda_interop_contract",
        "lossless_p5_d3d12_cuda_interop_contract",
        "lossless_p5_d3d12_cuda_interop_contract_gate_open",
        "p5_d3d12_cuda_interop_contract_checked",
        "p5_d3d12_cuda_interop_contract",
    ),
    (
        "p5_d3d12_cuda_functional_interop_probe",
        "lossless_p5_d3d12_cuda_functional_interop_probe",
        "lossless_p5_d3d12_cuda_functional_interop_probe_gate_open",
        "p5_d3d12_cuda_functional_interop_probe_checked",
        "p5_d3d12_cuda_functional_interop_probe",
    ),
    (
        "p5_d3d12_cuda_functional_probe_harness_manifest",
        "lossless_p5_d3d12_cuda_functional_probe_harness_manifest",
        "lossless_p5_d3d12_cuda_functional_probe_harness_manifest_gate_open",
        "p5_d3d12_cuda_functional_probe_harness_manifest_checked",
        "p5_d3d12_cuda_functional_probe_harness_manifest",
    ),
    (
        "p5_p6_cuda_tensor_view_binding_preflight",
        "lossless_p5_p6_cuda_tensor_view_binding_preflight",
        "lossless_p5_p6_cuda_tensor_view_binding_preflight_gate_open",
        "p5_p6_cuda_tensor_view_binding_preflight_checked",
        "p5_p6_cuda_tensor_view_binding_preflight",
    ),
    (
        "p5_p6_guarded_trainer_runtime_ab_preflight",
        "lossless_p5_p6_guarded_trainer_runtime_ab_preflight",
        "lossless_p5_p6_guarded_trainer_runtime_ab_preflight_gate_open",
        "p5_p6_guarded_trainer_runtime_ab_preflight_checked",
        "p5_p6_guarded_trainer_runtime_ab_preflight",
    ),
    (
        "p5_p6_runtime_ab_approval_packet",
        "lossless_p5_p6_runtime_ab_approval_packet",
        "lossless_p5_p6_runtime_ab_approval_packet_gate_open",
        "p5_p6_runtime_ab_approval_packet_checked",
        "p5_p6_runtime_ab_approval_packet",
    ),
    (
        "p6_compute_decode_readiness",
        "lossless_p6_d3d12_cuda_compute_decode_readiness",
        "lossless_p6_compute_decode_readiness_gate_open",
        "p6_compute_decode_readiness_checked",
        "p6_compute_decode_readiness",
    ),
    (
        "p5_p6_native_blueprint",
        "lossless_p5_p6_native_implementation_blueprint",
        "lossless_p5_p6_native_blueprint_gate_open",
        "p5_p6_native_blueprint_checked",
        "p5_p6_native_blueprint",
    ),
    (
        "phase_jitter_product_gate",
        "lossless_full_trainer_phase_jitter_product_gate",
        "lossless_phase_jitter_product_gate_open",
        "phase_jitter_product_gate_checked",
        "phase_jitter_product_gate",
    ),
    (
        "phase_jitter_control_delta",
        "lossless_full_trainer_phase_jitter_control_delta",
        "lossless_phase_jitter_control_delta_gate_open",
        "phase_jitter_control_delta_checked",
        "phase_jitter_control_delta",
    ),
    (
        "phase_jitter_phase_attribution",
        "lossless_full_trainer_phase_jitter_phase_attribution",
        "lossless_phase_jitter_phase_attribution_gate_open",
        "phase_jitter_phase_attribution_checked",
        "phase_jitter_phase_attribution",
    ),
    (
        "phase_jitter_optimizer_update_plan",
        "lossless_full_trainer_phase_jitter_optimizer_update_plan",
        "lossless_phase_jitter_optimizer_update_plan_gate_open",
        "phase_jitter_optimizer_update_plan_checked",
        "phase_jitter_optimizer_update_plan",
    ),
    (
        "phase_jitter_optimizer_update_delta",
        "lossless_full_trainer_phase_jitter_optimizer_update_delta",
        "lossless_phase_jitter_optimizer_update_delta_gate_open",
        "phase_jitter_optimizer_update_delta_checked",
        "phase_jitter_optimizer_update_delta",
    ),
    (
        "phase_jitter_optimizer_update_repeat_plan",
        "lossless_full_trainer_phase_jitter_optimizer_update_repeat_plan",
        "lossless_phase_jitter_optimizer_update_repeat_plan_gate_open",
        "phase_jitter_optimizer_update_repeat_plan_checked",
        "phase_jitter_optimizer_update_repeat_plan",
    ),
    (
        "compute_tail_focused_rerun_results",
        "lossless_compute_tail_focused_rerun_results",
        "lossless_compute_tail_focused_rerun_results_gate_open",
        "compute_tail_focused_rerun_results_checked",
        "compute_tail_focused_rerun_results",
    ),
    (
        "compute_tail_raw_order_jitter_repeat_phase_attribution",
        "lossless_compute_tail_raw_order_jitter_repeat_phase_attribution",
        "lossless_compute_tail_raw_order_jitter_repeat_phase_attribution_gate_open",
        "compute_tail_raw_order_jitter_repeat_phase_attribution_checked",
        "compute_tail_raw_order_jitter_repeat_phase_attribution",
    ),
    (
        "guarded_optimizer_update_tail_mitigation_design",
        "lossless_full_trainer_guarded_optimizer_update_tail_mitigation_design",
        "lossless_guarded_optimizer_update_tail_mitigation_design_gate_open",
        "guarded_optimizer_update_tail_mitigation_design_checked",
        "guarded_optimizer_update_tail_mitigation_design",
    ),
    (
        "guarded_optimizer_update_tail_validation_manifest",
        "lossless_full_trainer_guarded_optimizer_update_tail_validation_manifest",
        "lossless_guarded_optimizer_update_tail_validation_manifest_gate_open",
        "guarded_optimizer_update_tail_validation_manifest_checked",
        "guarded_optimizer_update_tail_validation_manifest",
    ),
    (
        "raw_order_jitter_guarded_compute_phase_mitigation_design",
        "lossless_compute_tail_raw_order_jitter_guarded_compute_phase_mitigation_design",
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_gate_open",
        "raw_order_jitter_guarded_compute_phase_mitigation_design_checked",
        "raw_order_jitter_guarded_compute_phase_mitigation_design",
    ),
    (
        "raw_order_jitter_guarded_compute_phase_validation_manifest",
        "lossless_compute_tail_raw_order_jitter_guarded_compute_phase_validation_manifest",
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_gate_open",
        "raw_order_jitter_guarded_compute_phase_validation_manifest_checked",
        "raw_order_jitter_guarded_compute_phase_validation_manifest",
    ),
    (
        "raw_order_compute_phase_validation_run_manifest",
        "lossless_compute_tail_raw_order_compute_phase_validation_run_manifest",
        "lossless_raw_order_compute_phase_validation_run_manifest_gate_open",
        "raw_order_compute_phase_validation_run_manifest_checked",
        "raw_order_compute_phase_validation_run_manifest",
    ),
    (
        "raw_order_compute_phase_validation_playbook",
        "lossless_compute_tail_raw_order_compute_phase_validation_playbook",
        "lossless_raw_order_compute_phase_validation_playbook_gate_open",
        "raw_order_compute_phase_validation_playbook_checked",
        "raw_order_compute_phase_validation_playbook",
    ),
    (
        "raw_order_compute_phase_validation_results",
        "lossless_compute_tail_raw_order_compute_phase_validation_results",
        "lossless_raw_order_compute_phase_validation_results_gate_open",
        "raw_order_compute_phase_validation_results_checked",
        "raw_order_compute_phase_validation_results",
    ),
    (
        "raw_order_compute_phase_validation_verdict",
        "lossless_compute_tail_raw_order_compute_phase_validation_verdict",
        "lossless_raw_order_compute_phase_validation_verdict_gate_open",
        "raw_order_compute_phase_validation_verdict_checked",
        "raw_order_compute_phase_validation_verdict",
    ),
    (
        "p7_non_heavy_next_action",
        "lossless_p7_non_heavy_next_action",
        "lossless_p7_non_heavy_next_action_gate_open",
        "p7_non_heavy_next_action_checked",
        "p7_non_heavy_next_action",
    ),
    (
        "p7_manual_heavy_authorization_bundle",
        "lossless_p7_manual_heavy_authorization_bundle",
        "lossless_p7_manual_heavy_authorization_bundle_gate_open",
        "p7_manual_heavy_authorization_bundle_checked",
        "p7_manual_heavy_authorization_bundle",
    ),
    (
        "p7_validation_failure_triage",
        "lossless_p7_validation_failure_triage",
        "lossless_p7_validation_failure_triage_gate_open",
        "p7_validation_failure_triage_checked",
        "p7_validation_failure_triage",
    ),
    (
        "p7_failed_item_blocker_resolution_matrix",
        "lossless_p7_failed_item_blocker_resolution_matrix",
        "lossless_p7_failed_item_blocker_resolution_matrix_gate_open",
        "p7_failed_item_blocker_resolution_matrix_checked",
        "p7_failed_item_blocker_resolution_matrix",
    ),
    (
        "p7_backward_forward_phase_guard_contract",
        "lossless_p7_backward_forward_phase_guard_contract",
        "lossless_p7_backward_forward_phase_guard_contract_gate_open",
        "p7_backward_forward_phase_guard_contract_checked",
        "p7_backward_forward_phase_guard_contract",
    ),
    (
        "p7_backward_forward_phase_guard_validation_recompute",
        "lossless_p7_backward_forward_phase_guard_validation_recompute",
        "lossless_p7_backward_forward_phase_guard_validation_recompute_gate_open",
        "p7_backward_forward_phase_guard_validation_recompute_checked",
        "p7_backward_forward_phase_guard_validation_recompute",
    ),
    (
        "p7_guarded_raw_order_compute_phase_variant_contract",
        "lossless_p7_guarded_raw_order_compute_phase_variant_contract",
        "lossless_p7_guarded_raw_order_compute_phase_variant_contract_gate_open",
        "p7_guarded_raw_order_compute_phase_variant_contract_checked",
        "p7_guarded_raw_order_compute_phase_variant_contract",
    ),
    (
        "p7_guarded_variant_cross_domain_evidence_contract",
        "lossless_p7_guarded_variant_cross_domain_evidence_contract",
        "lossless_p7_guarded_variant_cross_domain_evidence_contract_gate_open",
        "p7_guarded_variant_cross_domain_evidence_contract_checked",
        "p7_guarded_variant_cross_domain_evidence_contract",
    ),
    (
        "p7_optimizer_update_residual_guard_contract",
        "lossless_p7_optimizer_update_residual_guard_contract",
        "lossless_p7_optimizer_update_residual_guard_contract_gate_open",
        "p7_optimizer_update_residual_guard_contract_checked",
        "p7_optimizer_update_residual_guard_contract",
    ),
    (
        "p7_optimizer_update_residual_guard_recheck_after_nonrepro",
        "lossless_p7_optimizer_update_residual_guard_recheck_after_nonrepro",
        "lossless_p7_optimizer_update_residual_guard_recheck_after_nonrepro_gate_open",
        "p7_optimizer_update_residual_guard_recheck_after_nonrepro_checked",
        "p7_optimizer_update_residual_guard_recheck_after_nonrepro",
    ),
    (
        "p7_optimizer_update_repeat_positive_control_reclassification",
        "lossless_p7_optimizer_update_repeat_positive_control_reclassification",
        "lossless_p7_optimizer_update_repeat_positive_control_reclassification_gate_open",
        "p7_optimizer_update_repeat_positive_control_reclassification_checked",
        "p7_optimizer_update_repeat_positive_control_reclassification",
    ),
    (
        "p7_optimizer_update_repeat_positive_control_resolution",
        "lossless_p7_optimizer_update_repeat_positive_control_resolution",
        "lossless_p7_optimizer_update_repeat_positive_control_resolution_gate_open",
        "p7_optimizer_update_repeat_positive_control_resolution_checked",
        "p7_optimizer_update_repeat_positive_control_resolution",
    ),
    (
        "p7_optimizer_update_repeat_positive_optimizer_internal_resolution",
        "lossless_p7_optimizer_update_repeat_positive_optimizer_internal_resolution",
        "lossless_p7_optimizer_update_repeat_positive_optimizer_internal_resolution_gate_open",
        "p7_optimizer_update_repeat_positive_optimizer_internal_resolution_checked",
        "p7_optimizer_update_repeat_positive_optimizer_internal_resolution",
    ),
    (
        "p7_optimizer_step_micro_attribution",
        "lossless_p7_optimizer_step_micro_attribution",
        "lossless_p7_optimizer_step_micro_attribution_gate_open",
        "p7_optimizer_step_micro_attribution_checked",
        "p7_optimizer_step_micro_attribution",
    ),
    (
        "p7_optimizer_step_micro_profile_instrumentation",
        "lossless_p7_optimizer_step_micro_profile_instrumentation",
        "lossless_p7_optimizer_step_micro_profile_instrumentation_gate_open",
        "p7_optimizer_step_micro_profile_instrumentation_checked",
        "p7_optimizer_step_micro_profile_instrumentation",
    ),
    (
        "p7_optimizer_update_residual_guard_validation_recompute",
        "lossless_p7_optimizer_update_residual_guard_validation_recompute",
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_gate_open",
        "p7_optimizer_update_residual_guard_validation_recompute_checked",
        "p7_optimizer_update_residual_guard_validation_recompute",
    ),
    (
        "p7_optimizer_update_unaccounted_tail_isolation",
        "lossless_p7_optimizer_update_unaccounted_tail_isolation",
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_gate_open",
        "p7_optimizer_update_unaccounted_tail_isolation_checked",
        "p7_optimizer_update_unaccounted_tail_isolation",
    ),
    (
        "p7_optimizer_update_outer_phase_substage_instrumentation_contract",
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_contract",
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_contract_gate_open",
        "p7_optimizer_update_outer_phase_substage_instrumentation_contract_checked",
        "p7_optimizer_update_outer_phase_substage_instrumentation_contract",
    ),
    (
        "p7_optimizer_update_outer_phase_substage_profile_implementation",
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation",
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_gate_open",
        "p7_optimizer_update_outer_phase_substage_profile_implementation_checked",
        "p7_optimizer_update_outer_phase_substage_profile_implementation",
    ),
    (
        "p7_optimizer_update_outer_substage_tail_attribution",
        "lossless_p7_optimizer_update_outer_substage_tail_attribution",
        "lossless_p7_optimizer_update_outer_substage_tail_attribution_gate_open",
        "p7_optimizer_update_outer_substage_tail_attribution_checked",
        "p7_optimizer_update_outer_substage_tail_attribution",
    ),
    (
        "p7_order_neutral_residual_contract_scope_contract",
        "lossless_p7_order_neutral_residual_contract_scope_contract",
        "lossless_p7_order_neutral_residual_contract_scope_contract_gate_open",
        "p7_order_neutral_residual_contract_scope_contract_checked",
        "p7_order_neutral_residual_contract_scope_contract",
    ),
    (
        "guarded_variant_regression_action_plan",
        "lossless_compute_tail_raw_order_guarded_variant_regression_action_plan",
        "lossless_guarded_variant_regression_action_plan_gate_open",
        "guarded_variant_regression_action_plan_checked",
        "guarded_variant_regression_action_plan",
    ),
    (
        "guarded_variant_mitigation_blueprint",
        "lossless_compute_tail_raw_order_guarded_variant_mitigation_blueprint",
        "lossless_guarded_variant_mitigation_blueprint_gate_open",
        "guarded_variant_mitigation_blueprint_checked",
        "guarded_variant_mitigation_blueprint",
    ),
    (
        "replacement_phase_guard_contract",
        "lossless_replacement_phase_guard_contract",
        "lossless_replacement_phase_guard_contract_gate_open",
        "replacement_phase_guard_contract_checked",
        "replacement_phase_guard_contract",
    ),
    (
        "guarded_variant_runtime_contract",
        "lossless_guarded_variant_runtime_contract",
        "lossless_guarded_variant_runtime_contract_gate_open",
        "guarded_variant_runtime_contract_checked",
        "guarded_variant_runtime_contract",
    ),
    (
        "guarded_variant_request_adapter_contract",
        "lossless_guarded_variant_request_adapter_contract",
        "lossless_guarded_variant_request_adapter_contract_gate_open",
        "guarded_variant_request_adapter_contract_checked",
        "guarded_variant_request_adapter_contract",
    ),
    (
        "dataloader_guard_metadata",
        "lossless_dataloader_guard_metadata",
        "lossless_dataloader_guard_metadata_gate_open",
        "dataloader_guard_metadata_checked",
        "dataloader_guard_metadata",
    ),
]


def resolve_first_release_readiness_path(repo_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def default_first_release_readiness_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "gate_path": resolve_first_release_readiness_path(repo_root, DEFAULT_GATE_JSON),
        "full_trainer_readiness_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_FULL_TRAINER_READINESS_JSON,
        ),
        "release_smoke_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_RELEASE_SMOKE_JSON,
        ),
        "batch1_parity_smoke_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_BATCH1_PARITY_SMOKE_JSON,
        ),
        "pipeline_refactor_readiness_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_PIPELINE_REFACTOR_READINESS_JSON,
        ),
        "cache_container_plan_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_CACHE_CONTAINER_PLAN_JSON,
        ),
        "cache_container_read_path_gate_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_CACHE_CONTAINER_READ_PATH_GATE_JSON,
        ),
        "cache_container_admission_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_CACHE_CONTAINER_ADMISSION_CONTRACT_JSON,
            )
        ),
        "cache_container_next_axis_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_CACHE_CONTAINER_NEXT_AXIS_CONTRACT_JSON,
            )
        ),
        "cache_container_payload_layout_axis_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_CACHE_CONTAINER_PAYLOAD_LAYOUT_AXIS_CONTRACT_JSON,
            )
        ),
        "cache_container_baseline_comparison_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_CACHE_CONTAINER_BASELINE_COMPARISON_CONTRACT_JSON,
            )
        ),
        "cache_container_compute_tail_non_regression_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_CACHE_CONTAINER_COMPUTE_TAIL_NON_REGRESSION_CONTRACT_JSON,
            )
        ),
        "cache_container_decode_implementation_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_CACHE_CONTAINER_DECODE_IMPLEMENTATION_CONTRACT_JSON,
            )
        ),
        "cache_container_decode_path_implementation_plan_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_CACHE_CONTAINER_DECODE_PATH_IMPLEMENTATION_PLAN_JSON,
            )
        ),
        "p4_p6_action_plan_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_P4_P6_ACTION_PLAN_JSON,
        ),
        "p5_d3d12_cuda_interop_contract_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_P5_D3D12_CUDA_INTEROP_CONTRACT_JSON,
        ),
        "p5_d3d12_cuda_functional_interop_probe_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P5_D3D12_CUDA_FUNCTIONAL_INTEROP_PROBE_JSON,
            )
        ),
        "p5_d3d12_cuda_functional_probe_harness_manifest_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P5_D3D12_CUDA_FUNCTIONAL_PROBE_HARNESS_MANIFEST_JSON,
            )
        ),
        "p5_p6_cuda_tensor_view_binding_preflight_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P5_P6_CUDA_TENSOR_VIEW_BINDING_PREFLIGHT_JSON,
            )
        ),
        "p5_p6_guarded_trainer_runtime_ab_preflight_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P5_P6_GUARDED_TRAINER_RUNTIME_AB_PREFLIGHT_JSON,
            )
        ),
        "p5_p6_runtime_ab_approval_packet_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P5_P6_RUNTIME_AB_APPROVAL_PACKET_JSON,
            )
        ),
        "p6_compute_decode_readiness_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_P6_COMPUTE_DECODE_READINESS_JSON,
        ),
        "p5_p6_native_blueprint_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_P5_P6_NATIVE_BLUEPRINT_JSON,
        ),
        "phase_jitter_product_gate_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_PHASE_JITTER_PRODUCT_GATE_JSON,
        ),
        "phase_jitter_control_delta_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_PHASE_JITTER_CONTROL_DELTA_JSON,
        ),
        "phase_jitter_phase_attribution_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_PHASE_JITTER_PHASE_ATTRIBUTION_JSON,
        ),
        "phase_jitter_optimizer_update_plan_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_PHASE_JITTER_OPTIMIZER_UPDATE_PLAN_JSON,
        ),
        "phase_jitter_optimizer_update_delta_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_PHASE_JITTER_OPTIMIZER_UPDATE_DELTA_JSON,
        ),
        "phase_jitter_optimizer_update_repeat_plan_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_PHASE_JITTER_OPTIMIZER_UPDATE_REPEAT_PLAN_JSON,
        ),
        "compute_tail_focused_rerun_results_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_COMPUTE_TAIL_FOCUSED_RERUN_RESULTS_JSON,
        ),
        "compute_tail_raw_order_jitter_repeat_phase_attribution_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_COMPUTE_TAIL_RAW_ORDER_JITTER_REPEAT_PHASE_ATTRIBUTION_JSON,
            )
        ),
        "compute_tail_raw_order_jitter_order_neutral_excess_mitigation_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_COMPUTE_TAIL_RAW_ORDER_JITTER_ORDER_NEUTRAL_EXCESS_MITIGATION_CONTRACT_JSON,
            )
        ),
        "guarded_optimizer_update_tail_mitigation_design_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_GUARDED_OPTIMIZER_UPDATE_TAIL_MITIGATION_DESIGN_JSON,
            )
        ),
        "guarded_optimizer_update_tail_validation_manifest_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_GUARDED_OPTIMIZER_UPDATE_TAIL_VALIDATION_MANIFEST_JSON,
            )
        ),
        "raw_order_jitter_guarded_compute_phase_mitigation_design_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_RAW_ORDER_JITTER_GUARDED_COMPUTE_PHASE_MITIGATION_DESIGN_JSON,
            )
        ),
        "raw_order_jitter_guarded_compute_phase_validation_manifest_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_RAW_ORDER_JITTER_GUARDED_COMPUTE_PHASE_VALIDATION_MANIFEST_JSON,
            )
        ),
        "raw_order_compute_phase_validation_run_manifest_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_RUN_MANIFEST_JSON,
            )
        ),
        "raw_order_compute_phase_validation_playbook_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_PLAYBOOK_JSON,
            )
        ),
        "raw_order_compute_phase_validation_results_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_RESULTS_JSON,
            )
        ),
        "raw_order_compute_phase_validation_verdict_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_VERDICT_JSON,
            )
        ),
        "p7_non_heavy_next_action_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_P7_NON_HEAVY_NEXT_ACTION_JSON,
        ),
        "p7_manual_heavy_authorization_bundle_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_JSON,
        ),
        "p7_validation_failure_triage_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_P7_VALIDATION_FAILURE_TRIAGE_JSON,
        ),
        "p7_failed_item_blocker_resolution_matrix_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_FAILED_ITEM_BLOCKER_RESOLUTION_MATRIX_JSON,
            )
        ),
        "p7_backward_forward_phase_guard_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_BACKWARD_FORWARD_PHASE_GUARD_CONTRACT_JSON,
            )
        ),
        "p7_backward_forward_phase_guard_validation_recompute_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_BACKWARD_FORWARD_PHASE_GUARD_VALIDATION_RECOMPUTE_JSON,
            )
        ),
        "p7_guarded_raw_order_compute_phase_variant_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_GUARDED_RAW_ORDER_COMPUTE_PHASE_VARIANT_CONTRACT_JSON,
            )
        ),
        "p7_guarded_variant_cross_domain_evidence_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_GUARDED_VARIANT_CROSS_DOMAIN_EVIDENCE_CONTRACT_JSON,
            )
        ),
        "p7_optimizer_update_residual_guard_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_RESIDUAL_GUARD_CONTRACT_JSON,
            )
        ),
        "p7_optimizer_update_residual_guard_recheck_after_nonrepro_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_RESIDUAL_GUARD_RECHECK_AFTER_NONREPRO_JSON,
            )
        ),
        "p7_optimizer_update_repeat_positive_control_reclassification_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_REPEAT_POSITIVE_CONTROL_RECLASSIFICATION_JSON,
            )
        ),
        "p7_optimizer_update_repeat_positive_control_resolution_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_REPEAT_POSITIVE_CONTROL_RESOLUTION_JSON,
            )
        ),
        "p7_optimizer_update_repeat_positive_optimizer_internal_resolution_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_REPEAT_POSITIVE_OPTIMIZER_INTERNAL_RESOLUTION_JSON,
            )
        ),
        "p7_optimizer_step_micro_attribution_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_STEP_MICRO_ATTRIBUTION_JSON,
            )
        ),
        "p7_optimizer_step_micro_profile_instrumentation_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_STEP_MICRO_PROFILE_INSTRUMENTATION_JSON,
            )
        ),
        "p7_optimizer_update_residual_guard_validation_recompute_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_RESIDUAL_GUARD_VALIDATION_RECOMPUTE_JSON,
            )
        ),
        "p7_optimizer_update_unaccounted_tail_isolation_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_UNACCOUNTED_TAIL_ISOLATION_JSON,
            )
        ),
        "p7_optimizer_update_outer_phase_substage_instrumentation_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_OUTER_PHASE_SUBSTAGE_INSTRUMENTATION_CONTRACT_JSON,
            )
        ),
        "p7_optimizer_update_outer_phase_substage_profile_implementation_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_OUTER_PHASE_SUBSTAGE_PROFILE_IMPLEMENTATION_JSON,
            )
        ),
        "p7_optimizer_update_outer_substage_tail_attribution_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_OUTER_SUBSTAGE_TAIL_ATTRIBUTION_JSON,
            )
        ),
        "p7_order_neutral_residual_contract_scope_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_ORDER_NEUTRAL_RESIDUAL_CONTRACT_SCOPE_CONTRACT_JSON,
            )
        ),
        "guarded_variant_regression_action_plan_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_GUARDED_VARIANT_REGRESSION_ACTION_PLAN_JSON,
            )
        ),
        "guarded_variant_mitigation_blueprint_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_GUARDED_VARIANT_MITIGATION_BLUEPRINT_JSON,
            )
        ),
        "replacement_phase_guard_contract_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_REPLACEMENT_PHASE_GUARD_CONTRACT_JSON,
        ),
        "guarded_variant_runtime_contract_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_GUARDED_VARIANT_RUNTIME_CONTRACT_JSON,
        ),
        "guarded_variant_request_adapter_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_GUARDED_VARIANT_REQUEST_ADAPTER_CONTRACT_JSON,
            )
        ),
        "dataloader_guard_metadata_path": resolve_first_release_readiness_path(
            repo_root,
            DEFAULT_DATALOADER_GUARD_METADATA_JSON,
        ),
    }


def build_default_first_release_readiness(repo_root: Path) -> dict[str, Any]:
    return build_first_release_readiness(**default_first_release_readiness_paths(repo_root))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_load_error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}"}
    if isinstance(payload, dict):
        return payload
    return {"_load_error": "not_a_json_object"}


def _mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _repo_rel(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def _source_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": "",
            "exists": False,
            "size_bytes": 0,
            "mtime_ns": 0,
        }
    exists = path.is_file()
    return {
        "path": _repo_rel(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "mtime_ns": path.stat().st_mtime_ns if exists else 0,
    }


def _source_files_for_paths(paths: dict[str, Path | None]) -> dict[str, dict[str, Any]]:
    return {source_id: _source_info(path) for source_id, path in paths.items()}


def _source_files_digest(source_files: dict[str, Any]) -> dict[str, Any]:
    rows = {
        str(source_id): row
        for source_id, row in source_files.items()
        if isinstance(row, dict)
    }
    missing_ids = [
        source_id for source_id, row in rows.items() if not bool(row.get("exists"))
    ]
    load_error_ids = [
        source_id
        for source_id, row in rows.items()
        if str(row.get("load_error") or row.get("_load_error") or "")
    ]
    mtimes: list[int] = []
    for row in rows.values():
        try:
            mtimes.append(int(row.get("mtime_ns") or 0))
        except (TypeError, ValueError):
            mtimes.append(0)
    return {
        "source_count": len(rows),
        "source_exists_count": sum(1 for row in rows.values() if row.get("exists")),
        "source_missing_count": len(missing_ids),
        "source_missing_ids": missing_ids,
        "source_load_error_ids": load_error_ids,
        "source_newest_mtime_ns": max(mtimes) if mtimes else 0,
    }


def _source_summary_fields(source_files: dict[str, Any]) -> dict[str, Any]:
    return {
        **{
            f"source_{source_id}_path": str(row.get("path") or "")
            for source_id, row in source_files.items()
            if isinstance(row, dict)
        },
        **_source_files_digest(source_files),
    }


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _approval_candidate_projection(
    prefix: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    bool_fields = (
        "changed_implementation_path_ready",
        "functional_native_probe_ready",
        "manual_gpu_validation_recorded",
        "compute_decode_candidate_route_ready",
        "baseline_comparison_contract_ready",
        "baseline_data_wait_win_recorded",
        "compute_tail_non_regression_contract_ready",
        "compute_tail_non_regression_recorded",
        "compute_decode_candidate_ready",
        "candidate_evidence_ready",
        "manual_runtime_ab_review_ready",
    )
    projection = {f"{prefix}_{key}": bool(summary.get(key)) for key in bool_fields}
    projection[f"{prefix}_candidate_evidence_blocker_count"] = int(
        summary.get("candidate_evidence_blocker_count") or 0
    )
    projection[f"{prefix}_candidate_evidence_blocker_ids"] = _string_list(
        summary.get("candidate_evidence_blocker_ids")
    )
    return projection


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _visible_dependency_ids(gate: dict[str, Any]) -> list[str]:
    exposure = _as_dict(gate.get("resource_center_exposure"))
    return [str(item) for item in _as_list(exposure.get("visible_dependency_ids")) if item]


def _blocked_dependency_ids(gate: dict[str, Any]) -> list[str]:
    exposure = _as_dict(gate.get("resource_center_exposure"))
    return [str(item) for item in _as_list(exposure.get("blocked_dependency_ids")) if item]


def _unique_strings(*values: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _as_list(value):
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            output.append(text)
    return output


_MISSING = object()


def _first_present_value(*sources: tuple[dict[str, Any], str], default: Any = None) -> Any:
    for source, key in sources:
        value = source.get(key, _MISSING)
        if value is not _MISSING:
            return value
    return default


_P7_PRODUCT_PROOF_DETAIL_BOOL_SUFFIXES = (
    "positive_phase_digest_ready",
    "positive_phase_validation_recompute_ready",
    "positive_phase_contract_ready",
    "optimizer_update_residual_digest_ready",
    "guarded_variant_group_denial_ready",
    "guarded_variant_denial_matrix_ready",
    "validation_results_digest_ready",
    "validation_context_ready",
    "validation_context_product_unlock_ready",
    "validation_context_runtime_product_or_resource_gate_open",
    "core_source_freshness_ready",
    "core_source_resource_gate_not_older_than_audit",
    "detail_source_freshness_ready",
    "detail_source_not_newer_than_core_sources",
    "json_only",
    "manifest_only",
    "does_not_run_training",
    "does_not_run_cuda",
    "does_not_run_nvcomp",
    "does_not_run_cache_scan",
    "does_not_run_runtime",
    "does_not_mutate_runtime",
    "case_order_sensitivity_digest_ready",
    "case_order_sensitivity_detected",
    "balanced_case_order_proof_digest_ready",
    "balanced_case_order_proof_passed",
    "order_neutral_aggregate_ready",
    "order_neutral_aggregate_digest_ready",
    "case_order_confounding_resolution_plan_ready",
    "case_order_confounding_requires_heavy",
    "case_order_confounding_requires_runtime_change",
    "order_neutral_excess_mitigation_contract_ready",
    "order_neutral_excess_mitigation_requires_heavy",
    "order_neutral_excess_mitigation_next_heavy_validation_required",
    "report_only_gaps_closed",
    "heavy_only_all_require_manual_or_heavy_validation",
    "failed_item_matrix_ready",
    "order_neutral_residual_scope_contract_ready",
    "order_neutral_residual_scope_contract_requires_manual_heavy_validation",
    "guarded_variant_cross_domain_contract_ready",
    "guarded_variant_cross_domain_substitution_allowed",
    "guarded_variant_cross_domain_group_regression_decrement_allowed",
)
_P7_PRODUCT_PROOF_DETAIL_INT_SUFFIXES = (
    "positive_phase_contract_row_count",
    "source_blocker_count",
    "positive_phase_original_backward_positive_count",
    "positive_phase_original_forward_positive_count",
    "positive_phase_effective_positive_count",
    "positive_phase_raw_control_positive_count",
    "positive_phase_top_row_count",
    "optimizer_update_residual_positive_count",
    "optimizer_update_residual_target_optimizer_positive_row_count",
    "guarded_variant_regressed_group_count",
    "guarded_variant_denial_unit_count",
    "validation_results_opened_gate_output_count",
    "validation_context_declared_validation_passed_output_count",
    "validation_context_real_heavy_evidence_output_count",
    "validation_context_passed_item_count",
    "validation_context_failed_item_count",
    "validation_context_inconclusive_item_count",
    "validation_context_pending_item_count",
    "validation_context_missing_output_count",
    "validation_context_output_parse_error_count",
    "detail_source_count",
    "detail_source_exists_count",
    "detail_source_missing_count",
    "validation_issue_count",
    "core_source_validation_issue_count",
    "detail_source_validation_issue_count",
    "detail_source_gate_open_count",
    "matrix_audit_field_mismatch_count",
    "resolution_row_count",
    "core_source_audit_mtime_ns",
    "core_source_resource_gate_mtime_ns",
    "core_source_min_mtime_ns",
    "detail_source_newest_mtime_ns",
    "core_source_freshness_issue_count",
    "detail_source_newer_than_core_source_count",
    "detail_source_freshness_issue_count",
    "freshness_issue_count",
    "case_order_sensitivity_group_count",
    "case_order_sensitivity_ready_pair_count",
    "case_order_sensitivity_step_flip_count",
    "balanced_case_order_proof_group_count",
    "balanced_case_order_proof_ready_group_count",
    "balanced_case_order_proof_missing_output_count",
    "order_neutral_adjusted_excess_group_count",
    "order_neutral_aggregate_adjusted_excess_group_count",
    "order_neutral_excess_mitigation_target_group_count",
    "order_neutral_excess_mitigation_contract_row_count",
)
_P7_PRODUCT_PROOF_DETAIL_STR_SUFFIXES = (
    "positive_phase_boundary_classification",
    "positive_phase_validation_recompute_classification",
    "optimizer_update_residual_classification",
    "optimizer_update_residual_breakdown_tail_field",
    "validation_context_verdict",
    "validation_context_decision",
    "case_order_sensitivity_classification",
    "case_order_sensitivity_next_recommended",
    "balanced_case_order_proof_classification",
    "balanced_case_order_proof_next_recommended",
    "order_neutral_aggregate_classification",
    "order_neutral_next_recommended",
    "order_neutral_aggregate_next_recommended",
    "case_order_confounding_selected_next_action_id",
    "case_order_confounding_selected_next_action_kind",
    "case_order_confounding_next_recommended",
    "order_neutral_excess_mitigation_contract_id",
    "order_neutral_excess_mitigation_next_recommended",
    "next_work_state",
    "primary_heavy_only_evidence_required",
    "primary_blocker_resolution_class",
    "primary_heavy_only_resolution_class",
    "next_recommended",
    "order_neutral_residual_scope_contract_classification",
    "order_neutral_residual_scope_contract_next_recommended",
)
_P7_PRODUCT_PROOF_DETAIL_STRING_LIST_SUFFIXES = (
    "positive_phase_target_labels",
    "positive_phase_target_report_keys",
    "optimizer_update_residual_target_group_labels",
    "optimizer_update_residual_raw_order_confounded_group_labels",
    "guarded_variant_regressed_group_ids",
    "guarded_variant_denial_unit_ids",
    "guarded_variant_runtime_denied_unit_ids",
    "guarded_variant_product_denied_unit_ids",
    "guarded_variant_resource_center_denied_unit_ids",
    "guarded_variant_raw_control_or_mixed_unit_ids",
    "guarded_variant_denial_blocker_ids",
    "validation_results_passed_item_ids",
    "validation_results_failed_item_ids",
    "validation_results_real_heavy_evidence_item_ids",
    "core_source_validation_issue_reasons",
    "detail_source_validation_issue_reasons",
    "validation_issue_reasons",
    "matrix_audit_field_mismatch_reasons",
    "detail_source_load_error_ids",
    "detail_source_gate_open_ids",
    "detail_source_newer_than_core_source_ids",
    "source_blocker_ids",
    "failed_item_matrix_modeled_ready_ids",
    "resolution_classes",
    "case_order_sensitivity_group_labels",
    "balanced_case_order_proof_confounded_group_labels",
    "balanced_case_order_proof_replacement_regressed_group_labels",
    "order_neutral_adjusted_excess_group_labels",
    "order_neutral_aggregate_adjusted_excess_group_labels",
    "order_neutral_raw_control_regressed_group_labels",
    "order_neutral_aggregate_raw_control_regressed_group_labels",
    "case_order_confounding_confounded_group_labels",
    "case_order_confounding_blocked_downstream_probe_ids",
    "order_neutral_excess_mitigation_target_group_labels",
    "order_neutral_excess_mitigation_target_metric_ids",
    "order_neutral_excess_mitigation_blocked_downstream_probe_ids",
    "base_failed_item_contract_ready_ids",
    "guarded_variant_report_only_contract_ready_ids",
    "order_neutral_residual_scope_contract_raw_control_regressed_group_labels",
    "guarded_variant_cross_domain_phase_jitter_required_group_ids",
    "guarded_variant_cross_domain_manual_heavy_raw_order_labels",
    "guarded_variant_cross_domain_forbidden_consumption",
)
_P7_PRODUCT_PROOF_DETAIL_DICT_SUFFIXES = (
    "positive_phase_rows_by_phase",
    "positive_phase_rows_by_label",
    "positive_phase_dominant_source_counts",
    "positive_phase_dominant_compute_phase_counts",
    "optimizer_update_residual_compute_phase_counts",
    "guarded_variant_group_blockers",
    "resolution_class_counts",
    "heavy_only_resolution_class_counts",
    "heavy_only_evidence_required_counts",
    "evidence_required_counts",
)
_P7_PRODUCT_PROOF_DETAIL_LIST_SUFFIXES = (
    "positive_phase_top_rows_digest",
    "optimizer_update_residual_top_rows_digest",
    "optimizer_update_residual_top_optimizer_update_rows_digest",
    "guarded_variant_activation_denial_matrix_digest",
    "core_source_validation_issues",
    "detail_source_validation_issues",
    "validation_issues",
    "core_source_freshness_issues",
    "detail_source_freshness_issues",
    "freshness_issues",
)
_P7_PRODUCT_PROOF_CORE_SOURCE_IDS = (
    "audit",
    "resource_gate",
)
_P7_PRODUCT_PROOF_DETAIL_SOURCE_IDS = (
    "guarded_variant_contract",
    "backward_forward_contract",
    "backward_forward_validation_recompute",
    "optimizer_residual_contract",
    "order_neutral_scope_contract",
    "optimizer_attribution",
    "repeat_phase_attribution",
    "case_order_sensitivity",
    "balanced_case_order_proof",
    "order_neutral_aggregate",
    "case_order_confounding_resolution_plan",
    "order_neutral_excess_mitigation_contract",
    "guarded_variant_heavy_evidence",
    "guarded_variant_cross_domain_contract",
    "failed_item_matrix",
    "validation_results",
    "validation_verdict",
)
_P7_PRODUCT_PROOF_SOURCE_IDS = (
    *_P7_PRODUCT_PROOF_CORE_SOURCE_IDS,
    *_P7_PRODUCT_PROOF_DETAIL_SOURCE_IDS,
)


_AUTHORIZATION_APPROVAL_PACKET_BOOL_SUFFIXES = (
    "ready",
    "json_only",
    "manifest_only",
    "report_only_allowed",
    "does_not_run_training",
    "does_not_run_cuda",
    "does_not_run_nvcomp",
    "does_not_run_cache_scan",
    "does_not_run_runtime",
    "does_not_mutate_runtime",
    "authorization_followups_complete",
    "action_available",
    "action_execute_allowed_by_default",
    "action_runner_has_execute_flag",
    "action_runner_has_allow_heavy_flag",
    "action_heavy_execute_armed",
    "action_requires_explicit_user_approval",
    "action_requires_explicit_heavy_authorization",
    "action_gpu_heavy",
    "action_cache_scan_heavy",
    "execute_requested",
    "training_path_enabled",
    "resource_center_allowed",
    "resource_center_candidate",
    "candidate",
    "default_enabled",
    "product_ready",
    "safe_to_auto_execute",
    "known_manual_heavy_execute_allowed_by_default",
    "known_manual_heavy_requires_explicit_authorization",
    "known_manual_heavy_training_path_enabled",
    "known_manual_heavy_resource_center_allowed",
    "product_proof_context_available",
    "product_proof_context_ready",
    "product_proof_report_only_gaps_closed",
    "product_proof_heavy_only_all_require_manual_or_heavy_validation",
    "freshness_ready",
    "freshness_next_action_not_older_than_inputs",
    "stage_readiness_ready",
    "selected_stage_heavy_outputs_require_allow_heavy",
)
_AUTHORIZATION_APPROVAL_PACKET_INT_SUFFIXES = (
    "product_proof_heavy_only_blocker_count",
    "product_proof_validation_issue_count",
    "product_proof_freshness_issue_count",
    "freshness_stale_pair_count",
    "stage_readiness_incomplete_stage_count",
    "validation_issue_count",
    "followup_output_missing_count",
    "followup_authorization_required_output_missing_count",
    "followup_non_heavy_prerequisite_output_missing_count",
    "selected_stage_pending_output_count",
    "selected_stage_authorization_required_output_count",
    "selected_stage_gpu_heavy_output_count",
    "selected_stage_cache_scan_heavy_output_count",
)
_AUTHORIZATION_APPROVAL_PACKET_STR_SUFFIXES = (
    "action_stage_id",
    "known_manual_heavy_blocked_action_id",
    "known_manual_heavy_blocked_action_kind",
    "known_manual_heavy_blocker_detail",
    "product_proof_next_work_state",
    "product_proof_primary_heavy_only_evidence_required",
    "next_recommended",
)
_AUTHORIZATION_APPROVAL_PACKET_STRING_LIST_SUFFIXES = (
    "product_proof_gate_open_keys",
    "validation_issue_reasons",
)
_AUTHORIZATION_APPROVAL_PACKET_DICT_SUFFIXES = (
    "product_proof_resolution_class_counts",
    "product_proof_heavy_only_evidence_required_counts",
    "followup_phase_output_missing_counts",
    "followup_kind_output_missing_counts",
    "followup_label_output_missing_counts",
)
_AUTHORIZATION_APPROVAL_PACKET_LIST_SUFFIXES = (
    "pending_outputs",
    "selected_stage_pending_outputs",
    "stage_readiness",
)


def _lossless_p7_product_proof_detail_projection(
    gate_summary: dict[str, Any],
    gate_evidence: dict[str, Any],
) -> dict[str, Any]:
    def first(suffix: str, default: Any = None) -> Any:
        key = f"p7_product_proof_{suffix}"
        return _first_present_value(
            (gate_summary, key),
            (gate_evidence, key),
            default=default,
        )

    output: dict[str, Any] = {}
    for suffix in _P7_PRODUCT_PROOF_DETAIL_BOOL_SUFFIXES:
        output[f"lossless_p7_product_proof_{suffix}"] = bool(first(suffix))
    for suffix in _P7_PRODUCT_PROOF_DETAIL_INT_SUFFIXES:
        output[f"lossless_p7_product_proof_{suffix}"] = int(first(suffix, 0) or 0)
    for suffix in _P7_PRODUCT_PROOF_DETAIL_STR_SUFFIXES:
        output[f"lossless_p7_product_proof_{suffix}"] = str(first(suffix, "") or "")
    for suffix in _P7_PRODUCT_PROOF_DETAIL_STRING_LIST_SUFFIXES:
        output[f"lossless_p7_product_proof_{suffix}"] = _unique_strings(
            first(suffix, [])
        )
    for suffix in _P7_PRODUCT_PROOF_DETAIL_DICT_SUFFIXES:
        output[f"lossless_p7_product_proof_{suffix}"] = _as_dict(first(suffix, {}))
    for suffix in _P7_PRODUCT_PROOF_DETAIL_LIST_SUFFIXES:
        output[f"lossless_p7_product_proof_{suffix}"] = _as_list(first(suffix, []))
    for source_id in _P7_PRODUCT_PROOF_SOURCE_IDS:
        suffix = f"source_{source_id}_path"
        output[f"lossless_p7_product_proof_{suffix}"] = str(first(suffix, "") or "")
    for source_id in _P7_PRODUCT_PROOF_CORE_SOURCE_IDS:
        suffix = f"core_source_{source_id}_path"
        output[f"lossless_p7_product_proof_{suffix}"] = str(first(suffix, "") or "")
    for source_id in _P7_PRODUCT_PROOF_DETAIL_SOURCE_IDS:
        suffix = f"detail_source_{source_id}_path"
        output[f"lossless_p7_product_proof_{suffix}"] = str(first(suffix, "") or "")
    for source_prefix in ("source", "core_source"):
        for suffix in (
            "count",
            "exists_count",
            "missing_count",
            "newest_mtime_ns",
        ):
            key = f"{source_prefix}_{suffix}"
            output[f"lossless_p7_product_proof_{key}"] = int(first(key, 0) or 0)
        for suffix in ("missing_ids", "load_error_ids"):
            key = f"{source_prefix}_{suffix}"
            output[f"lossless_p7_product_proof_{key}"] = _unique_strings(
                first(key, [])
            )
    output["lossless_p7_product_proof_detail_source_missing_ids"] = _unique_strings(
        first("detail_source_missing_ids", [])
    )
    return output


def _lossless_authorization_approval_packet_projection(
    gate_summary: dict[str, Any],
    gate_evidence: dict[str, Any],
) -> dict[str, Any]:
    def first(suffix: str, default: Any = None) -> Any:
        key = f"authorization_approval_packet_{suffix}"
        resource_key = f"resource_gate_{key}"
        return _first_present_value(
            (gate_summary, key),
            (gate_evidence, key),
            (gate_summary, resource_key),
            (gate_evidence, resource_key),
            default=default,
        )

    output: dict[str, Any] = {}
    for suffix in _AUTHORIZATION_APPROVAL_PACKET_BOOL_SUFFIXES:
        output[f"lossless_authorization_approval_packet_{suffix}"] = bool(
            first(suffix)
        )
    for suffix in _AUTHORIZATION_APPROVAL_PACKET_INT_SUFFIXES:
        output[f"lossless_authorization_approval_packet_{suffix}"] = int(
            first(suffix, 0) or 0
        )
    for suffix in _AUTHORIZATION_APPROVAL_PACKET_STR_SUFFIXES:
        output[f"lossless_authorization_approval_packet_{suffix}"] = str(
            first(suffix, "") or ""
        )
    for suffix in _AUTHORIZATION_APPROVAL_PACKET_STRING_LIST_SUFFIXES:
        output[f"lossless_authorization_approval_packet_{suffix}"] = _unique_strings(
            first(suffix, [])
        )
    for suffix in _AUTHORIZATION_APPROVAL_PACKET_DICT_SUFFIXES:
        output[f"lossless_authorization_approval_packet_{suffix}"] = _as_dict(
            first(suffix, {})
        )
    for suffix in _AUTHORIZATION_APPROVAL_PACKET_LIST_SUFFIXES:
        output[f"lossless_authorization_approval_packet_{suffix}"] = _as_list(
            first(suffix, [])
        )
    return output


_DATALOADER_GUARD_METADATA_BOOL_SUFFIXES = (
    "report_ready",
    "report_only",
    "runtime_activation_allowed",
    "request_adapter_activation_allowed",
    "runtime_default_change_allowed",
    "training_path_enabled",
    "resource_center_allowed",
    "resource_center_candidate",
    "default_enabled",
    "product_ready",
    "safe_to_auto_execute",
    "requires_manual_heavy_validation",
)
_DATALOADER_GUARD_METADATA_INT_SUFFIXES = (
    "loader_facade_count",
    "validation_issue_count",
)
_DATALOADER_GUARD_METADATA_LIST_SUFFIXES = (
    "source_config_ids",
    "false_gate_issues",
)


def _lossless_dataloader_guard_metadata_projection(
    summary: dict[str, Any],
) -> dict[str, Any]:
    def first(suffix: str, default: Any = None) -> Any:
        return _first_present_value(
            (summary, f"dataloader_guard_metadata_{suffix}"),
            (summary, suffix),
            default=default,
        )

    output: dict[str, Any] = {}
    for suffix in _DATALOADER_GUARD_METADATA_BOOL_SUFFIXES:
        output[f"lossless_dataloader_guard_metadata_{suffix}"] = bool(first(suffix))
    for suffix in _DATALOADER_GUARD_METADATA_INT_SUFFIXES:
        output[f"lossless_dataloader_guard_metadata_{suffix}"] = int(
            first(suffix, 0) or 0
        )
    for suffix in _DATALOADER_GUARD_METADATA_LIST_SUFFIXES:
        output[f"lossless_dataloader_guard_metadata_{suffix}"] = _unique_strings(
            first(suffix, [])
        )
    return output


_P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_BOOL_SUFFIXES = (
    "ready",
    "action_agreement_ready",
    "component_runner_contract_ready",
    "manual_heavy_command_inventory_ready",
    "execute_allowed_by_default",
    "does_not_run_training",
    "does_not_run_cuda",
    "does_not_run_nvcomp",
    "does_not_run_cache_scan",
    "does_not_run_runtime",
    "does_not_mutate_runtime",
    "requires_explicit_heavy_authorization",
    "safe_to_auto_execute",
    "training_path_enabled",
    "resource_center_allowed",
    "resource_center_candidate",
    "candidate",
    "default_enabled",
    "product_ready",
)
_P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_INT_SUFFIXES = (
    "source_ready_count",
    "source_count",
    "source_exists_count",
    "source_missing_count",
    "source_newest_mtime_ns",
    "component_packet_ready_count",
    "component_packet_count",
    "component_runner_contract_ready_count",
    "component_runner_contract_count",
    "source_command_count",
    "gpu_heavy_source_command_count",
    "open_gate_source_count",
    "validation_issue_count",
)
_P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_TEXT_SUFFIXES = (
    "selected_next_action_id",
    "selected_next_action_kind",
    "known_manual_heavy_action_id",
    "known_manual_heavy_action_kind",
    "source_approval_packet_path",
    "source_authorization_next_action_path",
    "source_p7_next_action_path",
    "source_backward_forward_packet_path",
    "source_guarded_variant_packet_path",
    "next_recommended",
)
_P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_LIST_SUFFIXES = (
    "target_validation_item_ids",
    "source_missing_ids",
    "source_load_error_ids",
)


def _lossless_p7_manual_heavy_authorization_bundle_projection(
    summary: dict[str, Any],
) -> dict[str, Any]:
    def first(suffix: str, default: Any = None) -> Any:
        return _first_present_value(
            (summary, f"p7_manual_heavy_authorization_bundle_{suffix}"),
            (summary, suffix),
            default=default,
        )

    output: dict[str, Any] = {}
    for suffix in _P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_BOOL_SUFFIXES:
        output[f"lossless_p7_manual_heavy_authorization_bundle_{suffix}"] = bool(
            first(suffix)
        )
    for suffix in _P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_INT_SUFFIXES:
        output[f"lossless_p7_manual_heavy_authorization_bundle_{suffix}"] = int(
            first(suffix, 0) or 0
        )
    for suffix in _P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_TEXT_SUFFIXES:
        output[f"lossless_p7_manual_heavy_authorization_bundle_{suffix}"] = str(
            first(suffix, "") or ""
        )
    for suffix in _P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_LIST_SUFFIXES:
        output[f"lossless_p7_manual_heavy_authorization_bundle_{suffix}"] = (
            _unique_strings(first(suffix, []))
        )
    return output


_P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_BOOL_SUFFIXES = (
    "ready",
    "activation_denial_matrix_ready",
    "real_heavy_manifest_ready",
    "manual_heavy_run_manifest_consumed",
    "manual_heavy_run_manifest_ready",
    "manual_heavy_run_execute_requested",
    "manual_heavy_run_allow_heavy",
    "manual_heavy_run_existing_output_coverage_ready",
    "mitigation_blueprint_ready",
    "execute_allowed_by_default",
    "requires_explicit_heavy_authorization",
    "report_only_allowed",
    "declares_validation_passed",
    "does_not_run_training",
    "does_not_run_cuda",
    "does_not_run_runtime",
    "does_not_mutate_runtime",
    "training_path_enabled",
    "resource_center_allowed",
    "product_ready",
    "safe_to_auto_execute",
    "runtime_activation_allowed",
)
_P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_INT_SUFFIXES = (
    "target_group_count",
    "observed_ready_group_count",
    "observed_regressed_group_count",
    "source_command_count",
    "source_count",
    "source_exists_count",
    "source_missing_count",
    "source_newest_mtime_ns",
    "gpu_heavy_source_command_count",
    "regressed_group_source_file_count",
    "manual_heavy_run_source_command_count",
    "manual_heavy_run_gpu_heavy_source_command_count",
    "manual_heavy_run_executed_count",
    "manual_heavy_run_execution_failure_count",
    "manual_heavy_run_output_exists_before_count",
    "manual_heavy_run_pending_output_count",
    "manual_heavy_run_validation_issue_count",
    "mitigation_implementation_unit_count",
    "validation_issue_count",
)
_P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_TEXT_SUFFIXES = (
    "active_roadmap",
    "selected_next_action_id",
    "target_validation_item_id",
    "target_bucket",
    "manual_heavy_run_fail_closed_status",
    "source_guarded_variant_heavy_evidence_path",
    "source_real_heavy_manifest_path",
    "source_manual_heavy_run_manifest_path",
    "source_regression_action_plan_path",
    "source_mitigation_blueprint_path",
    "next_recommended",
)
_P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_LIST_SUFFIXES = (
    "target_group_ids",
    "activation_denial_blocker_ids",
    "target_group_rows",
    "followup_runner_argv_without_execute",
    "followup_runner_argv_requires_authorization",
    "runner_argv_without_execute",
    "runner_argv_requires_authorization",
    "validation_issues",
    "source_missing_ids",
    "source_load_error_ids",
)
_P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_DICT_SUFFIXES = (
    "group_blockers",
    "required_fields",
)


def _lossless_p7_guarded_variant_manual_heavy_packet_projection(
    gate_summary: dict[str, Any],
    gate_evidence: dict[str, Any],
) -> dict[str, Any]:
    def first(suffix: str, default: Any = None) -> Any:
        key = f"p7_full_trainer_p7_guarded_variant_manual_heavy_packet_{suffix}"
        return _first_present_value(
            (gate_summary, key),
            (gate_evidence, key),
            default=default,
        )

    output: dict[str, Any] = {}
    for suffix in _P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_BOOL_SUFFIXES:
        output[f"lossless_p7_guarded_variant_manual_heavy_packet_{suffix}"] = bool(
            first(suffix)
        )
    for suffix in _P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_INT_SUFFIXES:
        output[f"lossless_p7_guarded_variant_manual_heavy_packet_{suffix}"] = int(
            first(suffix, 0) or 0
        )
    for suffix in _P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_TEXT_SUFFIXES:
        output[f"lossless_p7_guarded_variant_manual_heavy_packet_{suffix}"] = str(
            first(suffix, "") or ""
        )
    for suffix in _P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_LIST_SUFFIXES:
        output[f"lossless_p7_guarded_variant_manual_heavy_packet_{suffix}"] = (
            _as_list(first(suffix, []))
        )
    for suffix in _P7_GUARDED_VARIANT_MANUAL_HEAVY_PACKET_DICT_SUFFIXES:
        output[f"lossless_p7_guarded_variant_manual_heavy_packet_{suffix}"] = (
            _as_dict(first(suffix, {}))
        )
    return output


_P7_BACKWARD_FORWARD_MANUAL_HEAVY_PACKET_BOOL_SUFFIXES = (
    "ready",
    "execute_allowed_by_default",
    "requires_explicit_heavy_authorization",
    "report_only_allowed",
    "declares_validation_passed",
    "does_not_run_runtime",
    "does_not_mutate_runtime",
    "training_path_enabled",
    "resource_center_allowed",
    "product_ready",
    "safe_to_auto_execute",
)
_P7_BACKWARD_FORWARD_MANUAL_HEAVY_PACKET_INT_SUFFIXES = (
    "target_bucket_row_count",
    "effective_positive_count",
    "source_command_count",
    "gpu_heavy_source_command_count",
    "validation_issue_count",
    "source_count",
    "source_exists_count",
    "source_missing_count",
    "source_newest_mtime_ns",
)
_P7_BACKWARD_FORWARD_MANUAL_HEAVY_PACKET_TEXT_SUFFIXES = (
    "active_roadmap",
    "selected_next_action_id",
    "target_validation_item_id",
    "target_bucket",
    "source_backward_forward_validation_recompute_path",
    "source_real_heavy_manifest_path",
    "source_raw_order_repeat_plan_path",
    "next_recommended",
)
_P7_BACKWARD_FORWARD_MANUAL_HEAVY_PACKET_LIST_SUFFIXES = (
    "target_labels",
    "target_report_keys",
    "source_missing_ids",
    "source_load_error_ids",
)


def _lossless_p7_backward_forward_manual_heavy_packet_projection(
    gate_summary: dict[str, Any],
    gate_evidence: dict[str, Any],
) -> dict[str, Any]:
    def first(suffix: str, default: Any = None) -> Any:
        key = (
            "p7_full_trainer_p7_backward_forward_phase_guard_manual_heavy_packet_"
            f"{suffix}"
        )
        return _first_present_value(
            (gate_summary, key),
            (gate_evidence, key),
            default=default,
        )

    output: dict[str, Any] = {}
    prefix = "lossless_p7_backward_forward_phase_guard_manual_heavy_packet"
    for suffix in _P7_BACKWARD_FORWARD_MANUAL_HEAVY_PACKET_BOOL_SUFFIXES:
        output[f"{prefix}_{suffix}"] = bool(first(suffix))
    for suffix in _P7_BACKWARD_FORWARD_MANUAL_HEAVY_PACKET_INT_SUFFIXES:
        output[f"{prefix}_{suffix}"] = int(first(suffix, 0) or 0)
    for suffix in _P7_BACKWARD_FORWARD_MANUAL_HEAVY_PACKET_TEXT_SUFFIXES:
        output[f"{prefix}_{suffix}"] = str(first(suffix, "") or "")
    for suffix in _P7_BACKWARD_FORWARD_MANUAL_HEAVY_PACKET_LIST_SUFFIXES:
        output[f"{prefix}_{suffix}"] = _as_list(first(suffix, []))
    return output


_GUARDED_VARIANT_SOURCE_INT_SUFFIXES = (
    "source_count",
    "source_exists_count",
    "source_missing_count",
    "source_newest_mtime_ns",
)
_GUARDED_VARIANT_SOURCE_LIST_SUFFIXES = (
    "source_missing_ids",
    "source_load_error_ids",
)


def _lossless_source_files_digest(source_files: dict[str, Any]) -> dict[str, Any]:
    rows = {
        str(source_id): row
        for source_id, row in source_files.items()
        if isinstance(row, dict)
    }
    missing_ids = [
        source_id for source_id, row in rows.items() if not bool(row.get("exists"))
    ]
    load_error_ids = [
        source_id
        for source_id, row in rows.items()
        if str(row.get("load_error") or row.get("_load_error") or "")
    ]
    newest_mtime_ns = 0
    for row in rows.values():
        try:
            newest_mtime_ns = max(newest_mtime_ns, int(row.get("mtime_ns") or 0))
        except Exception:
            continue
    return {
        "source_count": len(rows),
        "source_exists_count": len(rows) - len(missing_ids),
        "source_missing_count": len(missing_ids),
        "source_missing_ids": missing_ids,
        "source_load_error_ids": load_error_ids,
        "source_newest_mtime_ns": newest_mtime_ns,
    }


def _lossless_guarded_variant_source_projection(
    gate_summary: dict[str, Any],
    gate_evidence: dict[str, Any],
    source_prefix: str,
    output_prefix: str,
) -> dict[str, Any]:
    def first(suffix: str, default: Any = None) -> Any:
        key = f"{source_prefix}_{suffix}"
        return _first_present_value(
            (gate_summary, key),
            (gate_evidence, key),
            default=default,
        )

    output: dict[str, Any] = {}
    for suffix in _GUARDED_VARIANT_SOURCE_INT_SUFFIXES:
        output[f"{output_prefix}_{suffix}"] = int(first(suffix, 0) or 0)
    for suffix in _GUARDED_VARIANT_SOURCE_LIST_SUFFIXES:
        output[f"{output_prefix}_{suffix}"] = _unique_strings(first(suffix, []))
    return output


def _lossless_p7_optimizer_update_detail_projection(
    unaccounted_summary: dict[str, Any],
    unaccounted_source_files: dict[str, Any],
    instrumentation_summary: dict[str, Any],
    instrumentation_source_files: dict[str, Any],
    implementation_summary: dict[str, Any],
    implementation_source_files: dict[str, Any],
    attribution_summary: dict[str, Any],
    attribution_source_files: dict[str, Any],
) -> dict[str, Any]:
    unaccounted_source_projection = {
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_source_optimizer_update_tail_attribution_path": str(
            unaccounted_summary.get("source_optimizer_update_tail_attribution_path")
            or (
                unaccounted_source_files.get("optimizer_update_tail_attribution")
                or {}
            ).get("path")
            or ""
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_source_optimizer_update_residual_guard_contract_path": str(
            unaccounted_summary.get(
                "source_optimizer_update_residual_guard_contract_path"
            )
            or (
                unaccounted_source_files.get(
                    "optimizer_update_residual_guard_contract"
                )
                or {}
            ).get("path")
            or ""
        ),
        **{
            f"lossless_p7_optimizer_update_unaccounted_tail_isolation_{suffix}": value
            for suffix, value in _lossless_source_files_digest(
                unaccounted_source_files
            ).items()
        },
    }
    instrumentation_source_projection = {
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_contract_source_unaccounted_tail_isolation_path": str(
            instrumentation_summary.get("source_unaccounted_tail_isolation_path")
            or (
                instrumentation_source_files.get("unaccounted_tail_isolation")
                or {}
            ).get("path")
            or ""
        ),
        **{
            f"lossless_p7_optimizer_update_outer_phase_substage_instrumentation_contract_{suffix}": value
            for suffix, value in _lossless_source_files_digest(
                instrumentation_source_files
            ).items()
        },
    }
    implementation_source_projection = {
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_source_outer_substage_contract_path": str(
            implementation_summary.get("source_outer_substage_contract_path")
            or (
                implementation_source_files.get("outer_substage_contract")
                or {}
            ).get("path")
            or ""
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_source_targeted_heavy_manifest_path": str(
            implementation_summary.get("source_targeted_heavy_manifest_path")
            or (
                implementation_source_files.get("targeted_heavy_manifest")
                or {}
            ).get("path")
            or ""
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_source_training_loop_path": str(
            implementation_summary.get("source_training_loop_path")
            or (implementation_source_files.get("training_loop") or {}).get("path")
            or ""
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_source_housekeeping_handler_path": str(
            implementation_summary.get("source_housekeeping_handler_path")
            or (
                implementation_source_files.get("housekeeping_handler")
                or {}
            ).get("path")
            or ""
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_source_step_phase_profile_path": str(
            implementation_summary.get("source_step_phase_profile_path")
            or (
                implementation_source_files.get("step_phase_profile")
                or {}
            ).get("path")
            or ""
        ),
        **{
            f"lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_{suffix}": value
            for suffix, value in _lossless_source_files_digest(
                implementation_source_files
            ).items()
        },
    }
    attribution_source_projection = {
        "lossless_p7_optimizer_update_outer_substage_tail_attribution_source_unaccounted_tail_isolation_path": str(
            attribution_summary.get("source_unaccounted_tail_isolation_path")
            or (
                attribution_source_files.get("unaccounted_tail_isolation")
                or {}
            ).get("path")
            or ""
        ),
        "lossless_p7_optimizer_update_outer_substage_tail_attribution_source_outer_substage_implementation_path": str(
            attribution_summary.get("source_outer_substage_implementation_path")
            or (
                attribution_source_files.get("outer_substage_implementation")
                or {}
            ).get("path")
            or ""
        ),
        **{
            f"lossless_p7_optimizer_update_outer_substage_tail_attribution_{suffix}": value
            for suffix, value in _lossless_source_files_digest(
                attribution_source_files
            ).items()
        },
    }
    return {
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_ready": bool(
            unaccounted_summary.get(
                "p7_optimizer_update_unaccounted_tail_isolation_ready"
            )
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_classification": str(
            unaccounted_summary.get("classification") or ""
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_optimizer_attribution_classification": str(
            unaccounted_summary.get("optimizer_attribution_classification") or ""
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_no_target_unaccounted_tail": bool(
            unaccounted_summary.get("no_target_unaccounted_tail")
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_target_unaccounted_positive_row_count": int(
            unaccounted_summary.get("target_unaccounted_positive_row_count") or 0
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_repeat_unaccounted_positive_row_count": int(
            unaccounted_summary.get("repeat_unaccounted_positive_row_count") or 0
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_replacement_specific_unaccounted_positive_count": int(
            unaccounted_summary.get("replacement_specific_unaccounted_positive_count")
            or 0
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_top_unaccounted_rows_count": int(
            unaccounted_summary.get("top_unaccounted_rows_count") or 0
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_next_action_id": str(
            unaccounted_summary.get("next_action_id") or ""
        ),
        "lossless_p7_optimizer_update_unaccounted_tail_isolation_requires_manual_heavy_validation": bool(
            unaccounted_summary.get("requires_manual_heavy_validation")
        ),
        **unaccounted_source_projection,
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_contract_ready": bool(
            instrumentation_summary.get(
                "p7_optimizer_update_outer_phase_substage_instrumentation_contract_ready"
            )
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_contract_required": bool(
            instrumentation_summary.get("instrumentation_contract_required")
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_contract_row_count": int(
            instrumentation_summary.get("contract_row_count") or 0
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_required_measured_unaccounted_substage_count": int(
            instrumentation_summary.get(
                "required_measured_unaccounted_substage_count"
            )
            or 0
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_reconciliation_required": bool(
            instrumentation_summary.get("reconciliation_required")
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_runtime_implementation_required": bool(
            instrumentation_summary.get("runtime_implementation_required")
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_next_action_id": str(
            instrumentation_summary.get("next_action_id") or ""
        ),
        **instrumentation_source_projection,
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_ready": bool(
            implementation_summary.get(
                "p7_optimizer_update_outer_phase_substage_profile_implementation_ready"
            )
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_implemented_label_count": int(
            implementation_summary.get("implemented_label_count") or 0
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_required_label_count": int(
            implementation_summary.get("required_label_count") or 0
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_missing_required_label_count": int(
            implementation_summary.get("missing_required_label_count") or 0
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_missing_required_label_ids": _unique_strings(
            implementation_summary.get("missing_required_label_ids")
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_targeted_heavy_refresh_ready": bool(
            implementation_summary.get("targeted_heavy_refresh_ready")
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_targeted_heavy_report_count": int(
            implementation_summary.get("targeted_heavy_report_count") or 0
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_targeted_heavy_row_count": int(
            implementation_summary.get("targeted_heavy_row_count") or 0
        ),
        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_next_action_id": str(
            implementation_summary.get("next_action_id") or ""
        ),
        **implementation_source_projection,
        "lossless_p7_optimizer_update_outer_substage_tail_attribution_ready": bool(
            attribution_summary.get(
                "p7_optimizer_update_outer_substage_tail_attribution_ready"
            )
        ),
        "lossless_p7_optimizer_update_outer_substage_tail_attribution_classification": str(
            attribution_summary.get("classification") or ""
        ),
        "lossless_p7_optimizer_update_outer_substage_tail_attribution_no_target_unaccounted_tail": bool(
            attribution_summary.get("no_target_unaccounted_tail")
        ),
        "lossless_p7_optimizer_update_outer_substage_tail_attribution_target_unaccounted_row_count": int(
            attribution_summary.get("target_unaccounted_row_count") or 0
        ),
        "lossless_p7_optimizer_update_outer_substage_tail_attribution_attributed_row_count": int(
            attribution_summary.get("attributed_row_count") or 0
        ),
        "lossless_p7_optimizer_update_outer_substage_tail_attribution_next_action_id": str(
            attribution_summary.get("next_action_id") or ""
        ),
        **attribution_source_projection,
    }


def _lossless_p7_failed_item_blocker_projection(
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lossless_p7_failed_item_blocker_resolution_matrix_ready": bool(
            summary.get("p7_failed_item_blocker_resolution_matrix_ready")
        ),
        "lossless_p7_failed_item_blocker_validation_decision": str(
            summary.get("validation_decision") or ""
        ),
        "lossless_p7_failed_item_blocker_verdict_decision": str(
            summary.get("verdict_decision") or ""
        ),
        "lossless_p7_failed_item_blocker_failed_item_count": int(
            summary.get("failed_item_count") or 0
        ),
        "lossless_p7_failed_item_blocker_failed_item_ids": _unique_strings(
            summary.get("failed_item_ids")
        ),
        "lossless_p7_failed_item_blocker_modeled_failed_item_count": int(
            summary.get("modeled_failed_item_count") or 0
        ),
        "lossless_p7_failed_item_blocker_unmodeled_failed_item_count": len(
            _as_list(summary.get("unmodeled_failed_item_ids"))
        ),
        "lossless_p7_failed_item_blocker_inconclusive_item_count": int(
            summary.get("inconclusive_item_count") or 0
        ),
        "lossless_p7_failed_item_blocker_resolved_scope_item_count": int(
            summary.get("resolved_scope_item_count") or 0
        ),
        "lossless_p7_failed_item_blocker_strongest_blocker_count": int(
            summary.get("strongest_blocker_count") or 0
        ),
        "lossless_p7_failed_item_blocker_next_manual_heavy_requirement_count": int(
            summary.get("next_manual_heavy_requirement_count") or 0
        ),
        "lossless_p7_failed_item_blocker_next_manual_heavy_requirements": _as_list(
            summary.get("next_manual_heavy_requirements")
        ),
        "lossless_p7_failed_item_blocker_product_unlock_ready": bool(
            summary.get("product_unlock_ready")
        ),
        "lossless_p7_failed_item_blocker_training_path_enabled": bool(
            summary.get("training_path_enabled")
        ),
        "lossless_p7_failed_item_blocker_resource_center_allowed": bool(
            summary.get("resource_center_allowed")
        ),
        "lossless_p7_failed_item_blocker_product_ready": bool(
            summary.get("product_ready")
        ),
        "lossless_p7_failed_item_blocker_safe_to_auto_execute": bool(
            summary.get("safe_to_auto_execute")
        ),
        "lossless_p7_failed_item_blocker_validation_issue_count": int(
            summary.get("validation_issue_count") or 0
        ),
        "lossless_p7_failed_item_blocker_next_recommended": str(
            summary.get("next_recommended") or ""
        ),
    }


def _lossless_p7_artifact_source_projection(
    prefix: str,
    summary: dict[str, Any],
    source_files: dict[str, Any],
    source_ids: tuple[str, ...],
) -> dict[str, Any]:
    digest = _lossless_source_files_digest(source_files)

    def source_path(source_id: str) -> str:
        return str(
            summary.get(f"source_{source_id}_path")
            or _as_dict(source_files.get(source_id)).get("path")
            or ""
        )

    return {
        **{
            f"{prefix}_source_{source_id}_path": source_path(source_id)
            for source_id in source_ids
        },
        f"{prefix}_source_count": int(
            summary.get("source_count") or digest.get("source_count") or 0
        ),
        f"{prefix}_source_exists_count": int(
            summary.get("source_exists_count")
            or digest.get("source_exists_count")
            or 0
        ),
        f"{prefix}_source_missing_count": int(
            summary.get("source_missing_count")
            or digest.get("source_missing_count")
            or 0
        ),
        f"{prefix}_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids") or digest.get("source_missing_ids")
        ),
        f"{prefix}_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
            or digest.get("source_load_error_ids")
        ),
        f"{prefix}_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns")
            or digest.get("source_newest_mtime_ns")
            or 0
        ),
    }


def _lossless_report_artifact_source_projection(
    prefix: str,
    summary: dict[str, Any],
    source_files: dict[str, Any],
    source_ids: tuple[str, ...],
) -> dict[str, Any]:
    digest = _lossless_source_files_digest(source_files)

    def source_path(source_id: str) -> str:
        return str(
            summary.get(f"source_{source_id}_path")
            or _as_dict(source_files.get(source_id)).get("path")
            or ""
        )

    return {
        **{
            f"{prefix}_source_{source_id}_path": source_path(source_id)
            for source_id in source_ids
        },
        f"{prefix}_source_count": int(
            summary.get("source_count") or digest.get("source_count") or 0
        ),
        f"{prefix}_source_exists_count": int(
            summary.get("source_exists_count")
            or digest.get("source_exists_count")
            or 0
        ),
        f"{prefix}_source_missing_count": int(
            summary.get("source_missing_count")
            or digest.get("source_missing_count")
            or 0
        ),
        f"{prefix}_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids") or digest.get("source_missing_ids")
        ),
        f"{prefix}_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
            or digest.get("source_load_error_ids")
        ),
        f"{prefix}_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns")
            or digest.get("source_newest_mtime_ns")
            or 0
        ),
    }


def _lossless_all_report_artifact_source_projection(
    prefix: str,
    summary: dict[str, Any],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    source_ids = tuple(
        str(source_id)
        for source_id, row in source_files.items()
        if isinstance(row, dict)
    )
    return _lossless_report_artifact_source_projection(
        prefix,
        summary,
        source_files,
        source_ids,
    )


def _lossless_p7_backward_forward_validation_recompute_projection(
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lossless_p7_backward_forward_phase_guard_validation_recompute_ready": bool(
            summary.get("p7_backward_forward_phase_guard_validation_recompute_ready")
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_validation_decision": str(
            summary.get("validation_decision") or ""
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_verdict_decision": str(
            summary.get("verdict_decision") or ""
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_validation_failed_item_present": bool(
            summary.get("validation_failed_item_present")
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_classification": str(
            summary.get("classification") or ""
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_effective_backward_forward_positive_count": int(
            summary.get("effective_backward_forward_positive_count") or 0
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_raw_control_positive_phase_count": int(
            summary.get("raw_control_positive_phase_count") or 0
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_raw_control_positive_rows_removed_from_effective_count": int(
            summary.get("raw_control_positive_rows_removed_from_effective_count") or 0
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_backward_forward_product_blocker_still_present": bool(
            summary.get("backward_forward_product_blocker_still_present")
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_requires_manual_heavy_validation": bool(
            summary.get("requires_manual_heavy_validation")
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_next_action_id": str(
            summary.get("next_action_id") or ""
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_next_recommended": str(
            summary.get("next_recommended") or ""
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_training_path_enabled": bool(
            summary.get("training_path_enabled")
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_resource_center_allowed": bool(
            summary.get("resource_center_allowed")
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_product_ready": bool(
            summary.get("product_ready")
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_safe_to_auto_execute": bool(
            summary.get("safe_to_auto_execute")
        ),
        "lossless_p7_backward_forward_phase_guard_validation_recompute_validation_issue_count": int(
            summary.get("validation_issue_count") or 0
        ),
    }


def _lossless_p7_guarded_variant_cross_domain_projection(
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lossless_p7_guarded_variant_cross_domain_contract_ready": bool(
            summary.get("p7_guarded_variant_cross_domain_evidence_contract_ready")
        ),
        "lossless_p7_guarded_variant_cross_domain_report_only_allowed": bool(
            summary.get("report_only_allowed")
        ),
        "lossless_p7_guarded_variant_cross_domain_manifest_only": bool(
            summary.get("manifest_only")
        ),
        "lossless_p7_guarded_variant_cross_domain_substitution_allowed": bool(
            summary.get("cross_domain_substitution_allowed")
        ),
        "lossless_p7_guarded_variant_cross_domain_group_regression_decrement_allowed": bool(
            summary.get("group_regression_decrement_allowed")
        ),
        "lossless_p7_guarded_variant_cross_domain_product_ready_allowed": bool(
            summary.get(
                "guarded_variant_product_ready_allowed_by_cross_domain_contract"
            )
        ),
        "lossless_p7_guarded_variant_cross_domain_declares_validation_passed": bool(
            summary.get("declares_validation_passed")
        ),
        "lossless_p7_guarded_variant_cross_domain_runtime_activation_allowed": bool(
            summary.get("runtime_activation_allowed")
        ),
        "lossless_p7_guarded_variant_cross_domain_training_path_enabled": bool(
            summary.get("training_path_enabled")
        ),
        "lossless_p7_guarded_variant_cross_domain_resource_center_allowed": bool(
            summary.get("resource_center_allowed")
        ),
        "lossless_p7_guarded_variant_cross_domain_resource_center_candidate": bool(
            summary.get("resource_center_candidate")
        ),
        "lossless_p7_guarded_variant_cross_domain_candidate": bool(
            summary.get("candidate")
        ),
        "lossless_p7_guarded_variant_cross_domain_default_enabled": bool(
            summary.get("default_enabled")
        ),
        "lossless_p7_guarded_variant_cross_domain_product_ready": bool(
            summary.get("product_ready")
        ),
        "lossless_p7_guarded_variant_cross_domain_safe_to_auto_execute": bool(
            summary.get("safe_to_auto_execute")
        ),
        "lossless_p7_guarded_variant_cross_domain_validation_issue_count": int(
            summary.get("validation_issue_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_source_guarded_variant_heavy_evidence_path": str(
            summary.get("source_guarded_variant_heavy_evidence_path") or ""
        ),
        "lossless_p7_guarded_variant_cross_domain_source_guarded_manual_heavy_packet_path": str(
            summary.get("source_guarded_manual_heavy_packet_path") or ""
        ),
        "lossless_p7_guarded_variant_cross_domain_source_guarded_manual_heavy_run_manifest_path": str(
            summary.get("source_guarded_manual_heavy_run_manifest_path") or ""
        ),
        "lossless_p7_guarded_variant_cross_domain_source_count": int(
            summary.get("source_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_source_exists_count": int(
            summary.get("source_exists_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_source_missing_count": int(
            summary.get("source_missing_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids")
        ),
        "lossless_p7_guarded_variant_cross_domain_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
        ),
        "lossless_p7_guarded_variant_cross_domain_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_allowed_consumption": _unique_strings(
            summary.get("allowed_consumption")
        ),
        "lossless_p7_guarded_variant_cross_domain_forbidden_consumption": _unique_strings(
            summary.get("forbidden_consumption")
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_evidence_domain": str(
            summary.get("manual_heavy_evidence_domain") or ""
        ),
        "lossless_p7_guarded_variant_cross_domain_phase_jitter_evidence_domain": str(
            summary.get("phase_jitter_evidence_domain") or ""
        ),
        "lossless_p7_guarded_variant_cross_domain_phase_jitter_required_group_count": int(
            summary.get("phase_jitter_required_group_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_phase_jitter_required_group_ids": _unique_strings(
            summary.get("phase_jitter_required_group_ids")
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_packet_target_group_ids": _unique_strings(
            summary.get("manual_heavy_packet_target_group_ids")
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_raw_order_label_count": int(
            summary.get("manual_heavy_raw_order_label_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_raw_order_labels": _unique_strings(
            summary.get("manual_heavy_raw_order_labels")
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_raw_order_command_kind_count": int(
            summary.get("manual_heavy_raw_order_command_kind_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_raw_order_command_kinds": _unique_strings(
            summary.get("manual_heavy_raw_order_command_kinds")
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_run_manifest_ready": bool(
            summary.get("manual_heavy_run_manifest_ready")
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_run_manifest_consumed": bool(
            summary.get("manual_heavy_run_manifest_consumed")
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_run_existing_output_coverage_ready": bool(
            summary.get("manual_heavy_run_existing_output_coverage_ready")
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_run_executed_count": int(
            summary.get("manual_heavy_run_executed_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_run_execution_failure_count": int(
            summary.get("manual_heavy_run_execution_failure_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_run_source_command_count": int(
            summary.get("manual_heavy_run_source_command_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_run_gpu_heavy_source_command_count": int(
            summary.get("manual_heavy_run_gpu_heavy_source_command_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_manual_heavy_run_fail_closed_status": str(
            summary.get("manual_heavy_run_fail_closed_status") or ""
        ),
        "lossless_p7_guarded_variant_cross_domain_raw_order_repeat_can_replace_phase_jitter_group_regression": bool(
            summary.get("raw_order_repeat_can_replace_phase_jitter_group_regression")
        ),
        "lossless_p7_guarded_variant_cross_domain_non_substitutable_reason_count": int(
            summary.get("non_substitutable_reason_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_non_substitutable_reasons": _unique_strings(
            summary.get("non_substitutable_reasons")
        ),
        "lossless_p7_guarded_variant_cross_domain_observed_guarded_ready_group_count": int(
            summary.get("observed_guarded_ready_group_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_observed_guarded_regressed_group_count": int(
            summary.get("observed_guarded_regressed_group_count") or 0
        ),
        "lossless_p7_guarded_variant_cross_domain_next_recommended": str(
            summary.get("next_recommended") or ""
        ),
    }


def _lossless_p7_optimizer_residual_validation_recompute_projection(
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_ready": bool(
            summary.get("p7_optimizer_update_residual_guard_validation_recompute_ready")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_classification": str(
            summary.get("classification") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_validation_decision": str(
            summary.get("validation_decision") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_verdict_decision": str(
            summary.get("verdict_decision") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_validation_failed_item_present": bool(
            summary.get("validation_failed_item_present")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_validation_results_path": str(
            summary.get("source_validation_results_path") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_validation_verdict_path": str(
            summary.get("source_validation_verdict_path") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_optimizer_residual_guard_contract_path": str(
            summary.get("source_optimizer_residual_guard_contract_path") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_repeat_positive_control_reclassification_path": str(
            summary.get("source_repeat_positive_control_reclassification_path") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_repeat_positive_control_resolution_path": str(
            summary.get("source_repeat_positive_control_resolution_path") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_optimizer_internal_resolution_path": str(
            summary.get("source_optimizer_internal_resolution_path") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_optimizer_step_micro_attribution_path": str(
            summary.get("source_optimizer_step_micro_attribution_path") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_optimizer_step_micro_profile_instrumentation_path": str(
            summary.get("source_optimizer_step_micro_profile_instrumentation_path")
            or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_optimizer_update_tail_validation_manifest_path": str(
            summary.get("source_optimizer_update_tail_validation_manifest_path") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_count": int(
            summary.get("source_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_exists_count": int(
            summary.get("source_exists_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_missing_count": int(
            summary.get("source_missing_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_target_positive_step_wall_count": int(
            summary.get("optimizer_target_positive_step_wall_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_target_positive_row_count": int(
            summary.get("optimizer_target_positive_row_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_optimizer_update_positive_count_original": int(
            summary.get("repeat_optimizer_update_positive_count_original") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_optimizer_update_positive_count_recomputed": int(
            summary.get("repeat_optimizer_update_positive_count_recomputed") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_optimizer_update_positive_count": int(
            summary.get("repeat_optimizer_update_positive_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_control_row_count": int(
            summary.get("repeat_control_row_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_control_resolved_row_count": int(
            summary.get("repeat_control_resolved_row_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_control_unresolved_row_count": int(
            summary.get("repeat_control_unresolved_row_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_control_row_count_matches_repeat_positive_count": bool(
            summary.get("repeat_control_row_count_matches_repeat_positive_count")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_control_row_count_compatible_with_repeat_positive_count": bool(
            summary.get("repeat_control_row_count_compatible_with_repeat_positive_count")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_positive_control_covered_by_phase_guard_count": int(
            summary.get("repeat_positive_control_covered_by_phase_guard_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_positive_control_nonpositive_step_wall_count": int(
            summary.get("repeat_positive_control_nonpositive_step_wall_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_positive_control_unresolved_optimizer_internal_positive_wall_count": int(
            summary.get(
                "repeat_positive_control_unresolved_optimizer_internal_positive_wall_count"
            )
            or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_repeat_positive_control_unresolved_mixed_repeat_control_positive_count": int(
            summary.get(
                "repeat_positive_control_unresolved_mixed_repeat_control_positive_count"
            )
            or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_internal_resolution_ready": bool(
            summary.get("optimizer_internal_resolution_ready")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_internal_resolution_classification": str(
            summary.get("optimizer_internal_resolution_classification") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_step_micro_attribution_ready": bool(
            summary.get("optimizer_step_micro_attribution_ready")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_step_micro_attribution_unresolved_count": int(
            summary.get("optimizer_step_micro_attribution_unresolved_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_step_micro_attribution_profile_missing_row_count": int(
            summary.get("optimizer_step_micro_attribution_profile_missing_row_count")
            or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_step_micro_profile_instrumentation_ready": bool(
            summary.get("optimizer_step_micro_profile_instrumentation_ready")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_step_micro_profile_targeted_heavy_refresh_ready": bool(
            summary.get("optimizer_step_micro_profile_targeted_heavy_refresh_ready")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_followup_attribution_complete": bool(
            summary.get("optimizer_followup_attribution_complete")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_update_internal_breakdown_original_next_action_id": str(
            summary.get("optimizer_update_internal_breakdown_original_next_action_id")
            or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_optimizer_update_internal_breakdown_next_action_superseded": bool(
            summary.get("optimizer_update_internal_breakdown_next_action_superseded")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_effective_optimizer_update_internal_breakdown_next_action_id": str(
            summary.get("effective_optimizer_update_internal_breakdown_next_action_id")
            or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_residual_guard_product_blocker_still_present": bool(
            summary.get("residual_guard_product_blocker_still_present")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_requires_optimizer_positive_count_zero": bool(
            summary.get("requires_optimizer_positive_count_zero")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_requires_manual_heavy_validation": bool(
            summary.get("requires_manual_heavy_validation")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_product_proof_still_failed": bool(
            summary.get("product_proof_still_failed")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_product_unlock_ready": bool(
            summary.get("product_unlock_ready")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_report_only_allowed": bool(
            summary.get("report_only_allowed")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_declares_validation_passed": bool(
            summary.get("declares_validation_passed")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_training_path_enabled": bool(
            summary.get("training_path_enabled")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_resource_center_allowed": bool(
            summary.get("resource_center_allowed")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_resource_center_candidate": bool(
            summary.get("resource_center_candidate")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_candidate": bool(
            summary.get("candidate")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_default_enabled": bool(
            summary.get("default_enabled")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_product_ready": bool(
            summary.get("product_ready")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_safe_to_auto_execute": bool(
            summary.get("safe_to_auto_execute")
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_validation_issue_count": int(
            summary.get("validation_issue_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_next_action_id": str(
            summary.get("next_action_id") or ""
        ),
        "lossless_p7_optimizer_update_residual_guard_validation_recompute_next_recommended": str(
            summary.get("next_recommended") or ""
        ),
    }


def _lossless_p7_optimizer_residual_contract_source_projection(
    summary: dict[str, Any],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    digest = _lossless_source_files_digest(source_files)

    def source_path(source_id: str, summary_key: str) -> str:
        return str(
            summary.get(summary_key)
            or _as_dict(source_files.get(source_id)).get("path")
            or ""
        )

    return {
        "lossless_p7_optimizer_update_residual_guard_contract_source_validation_failure_triage_path": source_path(
            "validation_failure_triage",
            "source_validation_failure_triage_path",
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_validation_results_path": source_path(
            "validation_results",
            "source_validation_results_path",
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_validation_verdict_path": source_path(
            "validation_verdict",
            "source_validation_verdict_path",
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_optimizer_update_tail_attribution_path": source_path(
            "optimizer_update_tail_attribution",
            "source_optimizer_update_tail_attribution_path",
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_raw_order_repeat_phase_attribution_path": source_path(
            "raw_order_repeat_phase_attribution",
            "source_raw_order_repeat_phase_attribution_path",
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_guarded_optimizer_update_tail_mitigation_design_path": source_path(
            "guarded_optimizer_update_tail_mitigation_design",
            "source_guarded_optimizer_update_tail_mitigation_design_path",
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_guarded_optimizer_update_tail_validation_manifest_path": source_path(
            "guarded_optimizer_update_tail_validation_manifest",
            "source_guarded_optimizer_update_tail_validation_manifest_path",
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_count": int(
            summary.get("source_count") or digest.get("source_count") or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_exists_count": int(
            summary.get("source_exists_count")
            or digest.get("source_exists_count")
            or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_missing_count": int(
            summary.get("source_missing_count")
            or digest.get("source_missing_count")
            or 0
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids") or digest.get("source_missing_ids")
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
            or digest.get("source_load_error_ids")
        ),
        "lossless_p7_optimizer_update_residual_guard_contract_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns")
            or digest.get("source_newest_mtime_ns")
            or 0
        ),
    }


def _lossless_p7_optimizer_residual_recheck_projection(
    summary: dict[str, Any],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    digest = _lossless_source_files_digest(source_files)

    def source_path(source_id: str, summary_key: str) -> str:
        return str(
            summary.get(summary_key)
            or _as_dict(source_files.get(source_id)).get("path")
            or ""
        )

    prefix = "lossless_p7_optimizer_update_residual_guard_recheck_after_nonrepro"
    return {
        f"{prefix}_ready": bool(
            summary.get("p7_optimizer_update_residual_guard_recheck_after_nonrepro_ready")
        ),
        f"{prefix}_source_optimizer_residual_guard_contract_path": source_path(
            "optimizer_residual_guard_contract",
            "source_optimizer_residual_guard_contract_path",
        ),
        f"{prefix}_source_unaccounted_tail_isolation_path": source_path(
            "unaccounted_tail_isolation",
            "source_unaccounted_tail_isolation_path",
        ),
        f"{prefix}_source_outer_substage_tail_attribution_path": source_path(
            "outer_substage_tail_attribution",
            "source_outer_substage_tail_attribution_path",
        ),
        f"{prefix}_source_outer_substage_profile_implementation_path": source_path(
            "outer_substage_profile_implementation",
            "source_outer_substage_profile_implementation_path",
        ),
        f"{prefix}_source_count": int(
            summary.get("source_count") or digest.get("source_count") or 0
        ),
        f"{prefix}_source_exists_count": int(
            summary.get("source_exists_count")
            or digest.get("source_exists_count")
            or 0
        ),
        f"{prefix}_source_missing_count": int(
            summary.get("source_missing_count")
            or digest.get("source_missing_count")
            or 0
        ),
        f"{prefix}_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids") or digest.get("source_missing_ids")
        ),
        f"{prefix}_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
            or digest.get("source_load_error_ids")
        ),
        f"{prefix}_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns")
            or digest.get("source_newest_mtime_ns")
            or 0
        ),
        f"{prefix}_target_tail_not_reproduced": bool(
            summary.get("target_tail_not_reproduced")
        ),
        f"{prefix}_target_positive_step_wall_count": int(
            summary.get("target_positive_step_wall_count") or 0
        ),
        f"{prefix}_repeat_optimizer_update_positive_count": int(
            summary.get("repeat_optimizer_update_positive_count") or 0
        ),
        f"{prefix}_next_action_id": str(summary.get("next_action_id") or ""),
    }


def _lossless_p7_optimizer_repeat_reclassification_projection(
    summary: dict[str, Any],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    digest = _lossless_source_files_digest(source_files)

    def source_path(source_id: str, summary_key: str) -> str:
        return str(
            summary.get(summary_key)
            or _as_dict(source_files.get(source_id)).get("path")
            or ""
        )

    prefix = "lossless_p7_optimizer_update_repeat_positive_control_reclassification"
    return {
        f"{prefix}_ready": bool(
            summary.get(
                "p7_optimizer_update_repeat_positive_control_reclassification_contract_ready"
            )
        ),
        f"{prefix}_source_optimizer_residual_guard_contract_path": source_path(
            "optimizer_residual_guard_contract",
            "source_optimizer_residual_guard_contract_path",
        ),
        f"{prefix}_source_optimizer_update_tail_attribution_path": source_path(
            "optimizer_update_tail_attribution",
            "source_optimizer_update_tail_attribution_path",
        ),
        f"{prefix}_source_unaccounted_tail_isolation_path": source_path(
            "unaccounted_tail_isolation",
            "source_unaccounted_tail_isolation_path",
        ),
        f"{prefix}_source_optimizer_update_residual_guard_nonrepro_recheck_path": source_path(
            "optimizer_update_residual_guard_nonrepro_recheck",
            "source_optimizer_update_residual_guard_nonrepro_recheck_path",
        ),
        f"{prefix}_source_count": int(
            summary.get("source_count") or digest.get("source_count") or 0
        ),
        f"{prefix}_source_exists_count": int(
            summary.get("source_exists_count")
            or digest.get("source_exists_count")
            or 0
        ),
        f"{prefix}_source_missing_count": int(
            summary.get("source_missing_count")
            or digest.get("source_missing_count")
            or 0
        ),
        f"{prefix}_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids") or digest.get("source_missing_ids")
        ),
        f"{prefix}_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
            or digest.get("source_load_error_ids")
        ),
        f"{prefix}_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns")
            or digest.get("source_newest_mtime_ns")
            or 0
        ),
        f"{prefix}_classification": str(summary.get("classification") or ""),
        f"{prefix}_repeat_optimizer_update_positive_count": int(
            summary.get("repeat_optimizer_update_positive_count") or 0
        ),
        f"{prefix}_repeat_control_row_count": int(
            summary.get("repeat_control_row_count") or 0
        ),
        f"{prefix}_next_action_id": str(summary.get("next_action_id") or ""),
    }


def _lossless_p7_optimizer_repeat_resolution_projection(
    summary: dict[str, Any],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    digest = _lossless_source_files_digest(source_files)

    def source_path(source_id: str, summary_key: str) -> str:
        return str(
            summary.get(summary_key)
            or _as_dict(source_files.get(source_id)).get("path")
            or ""
        )

    prefix = "lossless_p7_optimizer_update_repeat_positive_control_resolution"
    return {
        f"{prefix}_ready": bool(
            summary.get(
                "p7_optimizer_update_repeat_positive_control_resolution_contract_ready"
            )
        ),
        f"{prefix}_source_repeat_positive_control_reclassification_path": source_path(
            "repeat_positive_control_reclassification",
            "source_repeat_positive_control_reclassification_path",
        ),
        f"{prefix}_source_backward_forward_phase_guard_contract_path": source_path(
            "backward_forward_phase_guard_contract",
            "source_backward_forward_phase_guard_contract_path",
        ),
        f"{prefix}_source_repeat_phase_attribution_path": source_path(
            "repeat_phase_attribution",
            "source_repeat_phase_attribution_path",
        ),
        f"{prefix}_source_validation_results_path": source_path(
            "validation_results",
            "source_validation_results_path",
        ),
        f"{prefix}_source_count": int(
            summary.get("source_count") or digest.get("source_count") or 0
        ),
        f"{prefix}_source_exists_count": int(
            summary.get("source_exists_count")
            or digest.get("source_exists_count")
            or 0
        ),
        f"{prefix}_source_missing_count": int(
            summary.get("source_missing_count")
            or digest.get("source_missing_count")
            or 0
        ),
        f"{prefix}_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids") or digest.get("source_missing_ids")
        ),
        f"{prefix}_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
            or digest.get("source_load_error_ids")
        ),
        f"{prefix}_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns")
            or digest.get("source_newest_mtime_ns")
            or 0
        ),
        f"{prefix}_classification": str(summary.get("classification") or ""),
        f"{prefix}_unresolved_optimizer_internal_positive_wall_count": int(
            summary.get("unresolved_optimizer_internal_positive_wall_count") or 0
        ),
        f"{prefix}_next_action_id": str(summary.get("next_action_id") or ""),
    }


def _lossless_p7_optimizer_repeat_internal_resolution_projection(
    summary: dict[str, Any],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    digest = _lossless_source_files_digest(source_files)

    def source_path(source_id: str, summary_key: str) -> str:
        return str(
            summary.get(summary_key)
            or _as_dict(source_files.get(source_id)).get("path")
            or ""
        )

    prefix = (
        "lossless_p7_optimizer_update_repeat_positive_optimizer_internal_resolution"
    )
    return {
        f"{prefix}_ready": bool(
            summary.get(
                "p7_optimizer_update_repeat_positive_optimizer_internal_resolution_contract_ready"
            )
        ),
        f"{prefix}_source_repeat_positive_control_resolution_path": source_path(
            "repeat_positive_control_resolution",
            "source_repeat_positive_control_resolution_path",
        ),
        f"{prefix}_source_count": int(
            summary.get("source_count") or digest.get("source_count") or 0
        ),
        f"{prefix}_source_exists_count": int(
            summary.get("source_exists_count")
            or digest.get("source_exists_count")
            or 0
        ),
        f"{prefix}_source_missing_count": int(
            summary.get("source_missing_count")
            or digest.get("source_missing_count")
            or 0
        ),
        f"{prefix}_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids") or digest.get("source_missing_ids")
        ),
        f"{prefix}_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
            or digest.get("source_load_error_ids")
        ),
        f"{prefix}_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns")
            or digest.get("source_newest_mtime_ns")
            or 0
        ),
        f"{prefix}_classification": str(summary.get("classification") or ""),
        f"{prefix}_unresolved_optimizer_step_micro_attribution_count": int(
            summary.get("unresolved_optimizer_step_micro_attribution_count") or 0
        ),
        f"{prefix}_next_action_id": str(summary.get("next_action_id") or ""),
    }


def _lossless_p7_optimizer_step_micro_attribution_projection(
    summary: dict[str, Any],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    digest = _lossless_source_files_digest(source_files)

    def source_path(source_id: str, summary_key: str) -> str:
        return str(
            summary.get(summary_key)
            or _as_dict(source_files.get(source_id)).get("path")
            or ""
        )

    prefix = "lossless_p7_optimizer_step_micro_attribution"
    return {
        f"{prefix}_ready": bool(
            summary.get("p7_optimizer_step_micro_attribution_contract_ready")
        ),
        f"{prefix}_source_optimizer_internal_resolution_path": source_path(
            "optimizer_internal_resolution",
            "source_optimizer_internal_resolution_path",
        ),
        f"{prefix}_source_count": int(
            summary.get("source_count") or digest.get("source_count") or 0
        ),
        f"{prefix}_source_exists_count": int(
            summary.get("source_exists_count")
            or digest.get("source_exists_count")
            or 0
        ),
        f"{prefix}_source_missing_count": int(
            summary.get("source_missing_count")
            or digest.get("source_missing_count")
            or 0
        ),
        f"{prefix}_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids") or digest.get("source_missing_ids")
        ),
        f"{prefix}_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
            or digest.get("source_load_error_ids")
        ),
        f"{prefix}_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns")
            or digest.get("source_newest_mtime_ns")
            or 0
        ),
        f"{prefix}_classification": str(summary.get("classification") or ""),
        f"{prefix}_target_optimizer_step_micro_row_count": int(
            summary.get("target_optimizer_step_micro_row_count") or 0
        ),
        f"{prefix}_optimizer_step_micro_profile_missing_row_count": int(
            summary.get("optimizer_step_micro_profile_missing_row_count") or 0
        ),
        f"{prefix}_unresolved_optimizer_step_micro_attribution_count": int(
            summary.get("unresolved_optimizer_step_micro_attribution_count") or 0
        ),
        f"{prefix}_next_action_id": str(summary.get("next_action_id") or ""),
    }


def _lossless_p7_optimizer_step_micro_profile_instrumentation_projection(
    summary: dict[str, Any],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    digest = _lossless_source_files_digest(source_files)

    def source_path(source_id: str, summary_key: str) -> str:
        return str(
            summary.get(summary_key)
            or _as_dict(source_files.get(source_id)).get("path")
            or ""
        )

    prefix = "lossless_p7_optimizer_step_micro_profile_instrumentation"
    return {
        f"{prefix}_ready": bool(
            summary.get(
                "p7_optimizer_step_micro_profile_instrumentation_contract_ready"
            )
        ),
        f"{prefix}_source_optimizer_step_micro_attribution_path": source_path(
            "optimizer_step_micro_attribution",
            "source_optimizer_step_micro_attribution_path",
        ),
        f"{prefix}_source_step_phase_profile_path": source_path(
            "step_phase_profile",
            "source_step_phase_profile_path",
        ),
        f"{prefix}_source_optimizer_handler_path": source_path(
            "optimizer_handler",
            "source_optimizer_handler_path",
        ),
        f"{prefix}_source_count": int(
            summary.get("source_count") or digest.get("source_count") or 0
        ),
        f"{prefix}_source_exists_count": int(
            summary.get("source_exists_count")
            or digest.get("source_exists_count")
            or 0
        ),
        f"{prefix}_source_missing_count": int(
            summary.get("source_missing_count")
            or digest.get("source_missing_count")
            or 0
        ),
        f"{prefix}_source_missing_ids": _unique_strings(
            summary.get("source_missing_ids") or digest.get("source_missing_ids")
        ),
        f"{prefix}_source_load_error_ids": _unique_strings(
            summary.get("source_load_error_ids")
            or digest.get("source_load_error_ids")
        ),
        f"{prefix}_source_newest_mtime_ns": int(
            summary.get("source_newest_mtime_ns")
            or digest.get("source_newest_mtime_ns")
            or 0
        ),
        f"{prefix}_targeted_heavy_refresh_ready": bool(
            summary.get("targeted_heavy_refresh_ready")
        ),
        f"{prefix}_targeted_heavy_refresh_required": bool(
            summary.get("targeted_heavy_refresh_required")
        ),
        f"{prefix}_next_action_id": str(summary.get("next_action_id") or ""),
    }


def _lossless_manifest_contract_summary_projection(
    *sources: dict[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            normalized = key.removeprefix("resource_gate_")
            if not normalized.startswith(
                (
                    "cache_container_manifest_",
                    "cache_container_precision_",
                    "cache_container_parity_",
                    "cache_container_readonly_matrix_",
                    "cache_container_worker_safety_",
                    "cache_container_persistent_worker_safety_",
                    "cache_container_decode_path_implementation_",
                )
            ):
                continue
            projection.setdefault(normalized, value)
    return projection


def _p3_p7_taxonomy_summary_projection(
    *sources: dict[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            normalized = key.removeprefix("resource_gate_")
            if not normalized.startswith("p3_p7_blocker_taxonomy_"):
                continue
            projection.setdefault(normalized, value)
            projection.setdefault(f"resource_gate_{normalized}", value)
    return projection


_RESOURCE_GATE_HIGH_RISK_SOURCE_PREFIXES = (
    "p3_full_trainer_ab_source_",
    "p3_full_trainer_reverse_ab_source_",
    "p3_full_trainer_reverse_nosave_ab_source_",
    "p3_full_trainer_reverse_nosave_diag_ab_source_",
    "p3_lynx_manifest_full_trainer_ab_source_",
    "p3_lynx_manifest_full_trainer_reverse_ab_source_",
    "p3_lynx_manifest_full_trainer_24step_nocopy_ab_source_",
    "p3_lynx_manifest_full_trainer_24step_nocopy_reverse_ab_source_",
    "p3_lynx_manifest_full_trainer_48step_nocopy_ab_source_",
    "p3_lynx_manifest_full_trainer_48step_nocopy_reverse_ab_source_",
    "p3_real_training_matrix_source_",
    "p3_real_cuda_training_matrix_source_",
    "p4_nvcomp_gate_smoke_source_",
    "p4_nvcomp_real_newbie_matrix_source_",
    "p4_nvcomp_anima_coalesced_matrix_source_",
    "p4_nvcomp_real_newbie_coalesced_matrix_source_",
    "p4_nvcomp_anima_batch_coalesced_matrix_source_",
    "p4_nvcomp_real_newbie_batch_coalesced_matrix_source_",
    "p6_gpu_decode_primitives_smoke_source_",
    "p6_gpu_decode_primitives_large_source_",
    "p6_gpu_decode_primitives_pinned_source_",
    "p6_gpu_decode_fused_primitives_source_",
)


_RESOURCE_GATE_P3_REPORT_ONLY_SOURCE_PREFIXES = (
    "p3_full_trainer_multirun_source_",
    "p3_full_trainer_tail_risk_source_",
    "p3_full_trainer_compute_tail_source_",
    "p3_full_trainer_jitter_delta_source_",
    "p3_full_trainer_queue_cuda_tail_source_",
    "p3_full_trainer_focus_contrast_source_",
    "p3_full_trainer_geometry_focus_plan_source_",
    "p3_full_trainer_compute_tail_focused_rerun_manifest_source_",
    "p3_full_trainer_cuda_phase_attribution_source_",
    "p3_full_trainer_compute_amplification_boundary_source_",
    "p3_full_trainer_hotspot_negative_alignment_source_",
    "p3_full_trainer_readiness_source_",
    "p3_full_trainer_pivot_trigger_classification_source_",
    "p3_full_trainer_pivot_trigger_isolation_source_",
    "p3_full_trainer_pivot_isolation_plan_sanity_source_",
    "p3_full_trainer_pivot_first_phase_delta_source_",
    "p3_full_trainer_pivot_sample_order_manifest_source_",
    "p3_full_trainer_pivot_sample_order_source_",
    "p3_full_trainer_pivot_confirmation_manifest_source_",
    "p3_full_trainer_pivot_confirmation_source_",
    "p3_full_trainer_pivot_confirmation_delta_source_",
    "p3_full_trainer_pivot_confirmation_phase_attribution_source_",
    "p3_full_trainer_pivot_ranked_hotspot_plan_source_",
    "p3_full_trainer_pivot_ranked_sample_order_manifest_source_",
    "p3_full_trainer_pivot_ranked_sample_order_source_",
    "p3_full_trainer_pivot_ranked_confirmation_manifest_source_",
    "p3_full_trainer_pivot_ranked_confirmation_source_",
    "p3_full_trainer_pivot_ranked_confirmation_delta_source_",
    "p3_full_trainer_pivot_ranked_confirmation_phase_attribution_source_",
    "p3_full_trainer_pivot_ranked_optimizer_update_drilldown_source_",
    "p3_full_trainer_pivot_ranked_optimizer_update_step_source_boundary_source_",
    "p3_full_trainer_pivot_ranked_optimizer_update_strict_alignment_manifest_source_",
    "p3_full_trainer_pivot_hotspot_micro_scorecard_source_",
    "p3_lynx_guarded_trainer_ab_preflight_source_",
    "p3_lynx_manifest_trainer_ab_scorecard_source_",
    "p3_lynx_nocopy_scorecard_source_",
    "p3_lynx_nocopy_48step_scorecard_source_",
    "p3_real_candidate_gate_source_",
    "p3_anima_candidate_gate_source_",
)


def _resource_gate_high_risk_source_projection(
    *sources: dict[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            normalized = key.removeprefix("resource_gate_")
            if not normalized.startswith(_RESOURCE_GATE_HIGH_RISK_SOURCE_PREFIXES):
                continue
            projection.setdefault(f"resource_gate_{normalized}", value)
    return projection


def _resource_gate_p3_report_only_source_projection(
    *sources: dict[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            normalized = key.removeprefix("resource_gate_")
            if not normalized.startswith(_RESOURCE_GATE_P3_REPORT_ONLY_SOURCE_PREFIXES):
                continue
            projection.setdefault(f"resource_gate_{normalized}", value)
    return projection


def _lossless_p5_p6_p7_boundary_summary_projection(
    *sources: dict[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for source in sources:
        for key, value in source.items():
            normalized = key.removeprefix("resource_gate_")
            if not normalized.startswith("p5_p6_p7_boundary_summary_"):
                continue
            projection.setdefault(f"lossless_{normalized}", value)
    prefix = "lossless_p5_p6_p7_boundary_summary"
    reason_ids = _unique_strings(
        projection.get(f"{prefix}_blockers"),
        projection.get(f"{prefix}_d3d12_cuda_validation_gap_ids"),
        projection.get(f"{prefix}_runtime_ab_blocked_precondition_ids"),
        projection.get(f"{prefix}_native_blueprint_gap_ids"),
        projection.get(f"{prefix}_compute_decode_blockers"),
        projection.get(f"{prefix}_remaining_report_only_gap_ids"),
    )
    manual_heavy_reason_ids = _unique_strings(
        projection.get(f"{prefix}_manual_heavy_action_ids"),
        [
            reason_id
            for reason_id in reason_ids
            if (
                "manual" in reason_id
                or "heavy" in reason_id
                or "gpu_validation" in reason_id
            )
        ],
    )
    projection[f"{prefix}_closed_not_productized_reason_ids"] = reason_ids
    projection[f"{prefix}_closed_not_productized_reason_count"] = len(reason_ids)
    projection[f"{prefix}_manual_heavy_required_reason_ids"] = manual_heavy_reason_ids
    projection[f"{prefix}_manual_heavy_required_reason_count"] = len(
        manual_heavy_reason_ids
    )
    return projection


def _resource_gate_blocker_groups(gate: dict[str, Any]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    top_level = _unique_strings(gate.get("blockers"))
    if top_level:
        groups["resource_center_gate"] = top_level
    for phase_id, values in _as_dict(gate.get("blockers_by_phase")).items():
        phase_blockers = _unique_strings(values)
        if phase_blockers:
            groups[f"resource_gate_{str(phase_id).lower()}"] = phase_blockers
    return groups


def _release_smoke_status(report: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    failed = int(summary.get("failed") or 0)
    total = int(summary.get("total") or 0)
    return {
        "ok": not report.get("_load_error")
        and str(report.get("schema") or "") == "lulynx.release-smoke-report.v1"
        and total > 0
        and failed == 0,
        "failed": failed,
        "total": total,
        "schema": str(report.get("schema") or ""),
        "load_error": str(report.get("_load_error") or ""),
    }


def _batch1_parity_status(report: dict[str, Any]) -> dict[str, Any]:
    checks = _as_dict(report.get("checks"))
    return {
        "ok": not report.get("_load_error")
        and bool(report.get("passed"))
        and not bool(report.get("release_claim_allowed"))
        and bool(checks.get("release_claim_closed")),
        "passed": bool(report.get("passed")),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "release_claim_closed": bool(checks.get("release_claim_closed")),
        "load_error": str(report.get("_load_error") or ""),
    }


def _pipeline_refactor_readiness_status(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": not report.get("_load_error")
        and bool(report.get("does_not_add_training_entrypoint"))
        and not bool(report.get("release_claim_allowed"))
        and not bool(report.get("internal_orchestrator_gate_enabled")),
        "does_not_add_training_entrypoint": bool(
            report.get("does_not_add_training_entrypoint")
        ),
        "release_claim_allowed": bool(report.get("release_claim_allowed")),
        "internal_orchestrator_gate_enabled": bool(
            report.get("internal_orchestrator_gate_enabled")
        ),
        "blocker_count": len(_as_list(report.get("blockers"))),
        "load_error": str(report.get("_load_error") or ""),
    }


def _research_artifact_closed_status(report: dict[str, Any], *, artifact: str) -> dict[str, Any]:
    load_error = str(report.get("_load_error") or "")
    summary = _as_dict(report.get("summary"))
    missing = load_error == "missing"
    json_only = bool(report.get("json_only") or summary.get("json_only"))
    manifest_only = bool(
        report.get("manifest_only") or summary.get("manifest_only")
    )
    does_not_run_training = bool(
        report.get("does_not_run_training") or summary.get("does_not_run_training")
    )
    does_not_run_cuda = bool(
        report.get("does_not_run_cuda") or summary.get("does_not_run_cuda")
    )
    does_not_run_nvcomp = bool(
        report.get("does_not_run_nvcomp") or summary.get("does_not_run_nvcomp")
    )
    does_not_run_cache_scan = bool(
        report.get("does_not_run_cache_scan") or summary.get("does_not_run_cache_scan")
    )
    does_not_run_runtime = bool(
        report.get("does_not_run_runtime") or summary.get("does_not_run_runtime")
    )
    does_not_mutate_runtime = bool(
        report.get("does_not_mutate_runtime")
        or summary.get("does_not_mutate_runtime")
    )
    training_enabled = bool(
        report.get("training_path_enabled") or summary.get("training_path_enabled")
    )
    resource_center_allowed = bool(
        report.get("resource_center_allowed")
        or summary.get("resource_center_allowed")
    )
    resource_center_candidate = bool(
        report.get("resource_center_candidate")
        or summary.get("resource_center_candidate")
    )
    candidate = bool(report.get("candidate") or summary.get("candidate"))
    gate_open = bool(
        report.get("gate_open")
        or report.get("open_gate")
        or report.get("contract_gate_open")
        or summary.get("gate_open")
        or summary.get("open_gate")
        or summary.get("contract_gate_open")
        or int(report.get("gate_open_count") or 0) > 0
        or int(summary.get("gate_open_count") or 0) > 0
        or int(report.get("child_gate_open_count") or 0) > 0
        or int(summary.get("child_gate_open_count") or 0) > 0
        or int(report.get("opened_gate_output_count") or 0) > 0
        or int(summary.get("opened_gate_output_count") or 0) > 0
        or bool(report.get("gate_keys_open") or [])
        or bool(summary.get("gate_keys_open") or [])
        or bool(report.get("runtime_ab_ready") or summary.get("runtime_ab_ready"))
        or bool(
            report.get("execute_allowed_by_default")
            or summary.get("execute_allowed_by_default")
        )
        or bool(
            report.get("repeat_current_heavy_search_allowed")
            or summary.get("repeat_current_heavy_search_allowed")
        )
        or bool(report.get("heavy_rerun_allowed") or summary.get("heavy_rerun_allowed"))
    )
    if artifact in {
        "guarded_variant_request_adapter_contract",
        "lossless_guarded_variant_request_adapter_contract",
    }:
        gate_open = bool(
            gate_open
            or int(summary.get("validation_issue_count") or 0) > 0
            or bool(summary.get("unsafe_request_claim_keys") or [])
            or bool(report.get("unsafe_request_claim_keys") or [])
            or bool(summary.get("unsafe_runtime_claim_keys") or [])
            or bool(report.get("unsafe_runtime_claim_keys") or [])
            or bool(summary.get("runtime_ab_ready") or report.get("runtime_ab_ready"))
            or bool(
                summary.get("execute_allowed_by_default")
                or report.get("execute_allowed_by_default")
            )
            or bool(summary.get("execute_requested") or report.get("execute_requested"))
            or bool(summary.get("validation_passed") or report.get("validation_passed"))
            or bool(
                summary.get("product_unlock_ready")
                or report.get("product_unlock_ready")
            )
            or bool(
                summary.get("declares_validation_passed")
                or report.get("declares_validation_passed")
            )
        )
    default_enabled = bool(
        report.get("default_enabled") or summary.get("default_enabled")
    )
    product_ready = bool(report.get("product_ready") or summary.get("product_ready"))
    safe_to_auto_execute = bool(
        report.get("safe_to_auto_execute") or summary.get("safe_to_auto_execute")
    )
    runtime_activation_allowed = bool(
        report.get("runtime_activation_allowed")
        or summary.get("runtime_activation_allowed")
    )
    product_unlock_allowed = bool(
        report.get("product_unlock_allowed") or summary.get("product_unlock_allowed")
    )
    resource_center_unlock_allowed = bool(
        report.get("resource_center_unlock_allowed")
        or summary.get("resource_center_unlock_allowed")
    )
    request_adapter_activation_allowed = bool(
        report.get("request_adapter_activation_allowed")
        or summary.get("request_adapter_activation_allowed")
    )
    request_activation_denied = bool(
        report.get("request_activation_denied")
        or summary.get("request_activation_denied")
    )
    validation_issue_count = int(summary.get("validation_issue_count") or 0)
    unsafe_request_claim_keys = _as_list(
        summary.get("unsafe_request_claim_keys")
        or report.get("unsafe_request_claim_keys")
    )
    unsafe_runtime_claim_keys = _as_list(
        summary.get("unsafe_runtime_claim_keys")
        or report.get("unsafe_runtime_claim_keys")
    )
    return {
        "artifact": artifact,
        "ok": missing
        or (
            not load_error
            and not training_enabled
            and not resource_center_allowed
            and not resource_center_candidate
            and not candidate
            and not gate_open
            and not default_enabled
            and not product_ready
            and not safe_to_auto_execute
            and not runtime_activation_allowed
            and not product_unlock_allowed
            and not resource_center_unlock_allowed
            and not request_adapter_activation_allowed
        ),
        "checked": not missing,
        "load_error": load_error,
        "json_only": json_only,
        "manifest_only": manifest_only,
        "does_not_run_training": does_not_run_training,
        "does_not_run_cuda": does_not_run_cuda,
        "does_not_run_nvcomp": does_not_run_nvcomp,
        "does_not_run_cache_scan": does_not_run_cache_scan,
        "does_not_run_runtime": does_not_run_runtime,
        "does_not_mutate_runtime": does_not_mutate_runtime,
        "training_path_enabled": training_enabled,
        "resource_center_allowed": resource_center_allowed,
        "resource_center_candidate": resource_center_candidate,
        "candidate": candidate,
        "gate_open": gate_open,
        "default_enabled": default_enabled,
        "product_ready": product_ready,
        "safe_to_auto_execute": safe_to_auto_execute,
        "runtime_activation_allowed": runtime_activation_allowed,
        "product_unlock_allowed": product_unlock_allowed,
        "resource_center_unlock_allowed": resource_center_unlock_allowed,
        "request_adapter_activation_allowed": request_adapter_activation_allowed,
        "request_activation_denied": request_activation_denied,
        "validation_issue_count": validation_issue_count,
        "unsafe_request_claim_keys": unsafe_request_claim_keys,
        "unsafe_runtime_claim_keys": unsafe_runtime_claim_keys,
    }


def build_first_release_readiness(
    gate_path: Path,
    full_trainer_readiness_path: Path,
    release_smoke_path: Path | None = None,
    batch1_parity_smoke_path: Path | None = None,
    pipeline_refactor_readiness_path: Path | None = None,
    cache_container_plan_path: Path | None = None,
    cache_container_read_path_gate_path: Path | None = None,
    cache_container_admission_contract_path: Path | None = None,
    cache_container_next_axis_contract_path: Path | None = None,
    cache_container_payload_layout_axis_contract_path: Path | None = None,
    cache_container_baseline_comparison_contract_path: Path | None = None,
    cache_container_compute_tail_non_regression_contract_path: Path | None = None,
    cache_container_decode_implementation_contract_path: Path | None = None,
    cache_container_decode_path_implementation_plan_path: Path | None = None,
    p4_p6_action_plan_path: Path | None = None,
    p5_d3d12_cuda_interop_contract_path: Path | None = None,
    p5_d3d12_cuda_functional_interop_probe_path: Path | None = None,
    p5_d3d12_cuda_functional_probe_harness_manifest_path: Path | None = None,
    p5_p6_cuda_tensor_view_binding_preflight_path: Path | None = None,
    p5_p6_guarded_trainer_runtime_ab_preflight_path: Path | None = None,
    p5_p6_runtime_ab_approval_packet_path: Path | None = None,
    p6_compute_decode_readiness_path: Path | None = None,
    p5_p6_native_blueprint_path: Path | None = None,
    phase_jitter_product_gate_path: Path | None = None,
    phase_jitter_control_delta_path: Path | None = None,
    phase_jitter_phase_attribution_path: Path | None = None,
    phase_jitter_optimizer_update_plan_path: Path | None = None,
    phase_jitter_optimizer_update_delta_path: Path | None = None,
    phase_jitter_optimizer_update_repeat_plan_path: Path | None = None,
    compute_tail_focused_rerun_results_path: Path | None = None,
    compute_tail_raw_order_jitter_repeat_phase_attribution_path: Path | None = None,
    compute_tail_raw_order_jitter_order_neutral_excess_mitigation_contract_path: Path | None = None,
    guarded_optimizer_update_tail_mitigation_design_path: Path | None = None,
    guarded_optimizer_update_tail_validation_manifest_path: Path | None = None,
    raw_order_jitter_guarded_compute_phase_mitigation_design_path: Path | None = None,
    raw_order_jitter_guarded_compute_phase_validation_manifest_path: Path | None = None,
    raw_order_compute_phase_validation_run_manifest_path: Path | None = None,
    raw_order_compute_phase_validation_playbook_path: Path | None = None,
    raw_order_compute_phase_validation_results_path: Path | None = None,
    raw_order_compute_phase_validation_verdict_path: Path | None = None,
    p7_non_heavy_next_action_path: Path | None = None,
    p7_manual_heavy_authorization_bundle_path: Path | None = None,
    p7_validation_failure_triage_path: Path | None = None,
    p7_failed_item_blocker_resolution_matrix_path: Path | None = None,
    p7_backward_forward_phase_guard_contract_path: Path | None = None,
    p7_backward_forward_phase_guard_validation_recompute_path: Path | None = None,
    p7_guarded_raw_order_compute_phase_variant_contract_path: Path | None = None,
    p7_guarded_variant_cross_domain_evidence_contract_path: Path | None = None,
    p7_optimizer_update_residual_guard_contract_path: Path | None = None,
    p7_optimizer_update_residual_guard_recheck_after_nonrepro_path: Path | None = None,
    p7_optimizer_update_repeat_positive_control_reclassification_path: Path | None = None,
    p7_optimizer_update_repeat_positive_control_resolution_path: Path | None = None,
    p7_optimizer_update_repeat_positive_optimizer_internal_resolution_path: Path | None = None,
    p7_optimizer_step_micro_attribution_path: Path | None = None,
    p7_optimizer_step_micro_profile_instrumentation_path: Path | None = None,
    p7_optimizer_update_residual_guard_validation_recompute_path: Path | None = None,
    p7_optimizer_update_unaccounted_tail_isolation_path: Path | None = None,
    p7_optimizer_update_outer_phase_substage_instrumentation_contract_path: Path | None = None,
    p7_optimizer_update_outer_phase_substage_profile_implementation_path: Path | None = None,
    p7_optimizer_update_outer_substage_tail_attribution_path: Path | None = None,
    p7_order_neutral_residual_contract_scope_contract_path: Path | None = None,
    guarded_variant_regression_action_plan_path: Path | None = None,
    guarded_variant_mitigation_blueprint_path: Path | None = None,
    replacement_phase_guard_contract_path: Path | None = None,
    guarded_variant_runtime_contract_path: Path | None = None,
    guarded_variant_request_adapter_contract_path: Path | None = None,
    dataloader_guard_metadata_path: Path | None = None,
) -> dict[str, Any]:
    gate = _read_json(gate_path)
    readiness = _read_json(full_trainer_readiness_path)
    release_smoke = _read_json(release_smoke_path) if release_smoke_path else {}
    batch1_parity_smoke = (
        _read_json(batch1_parity_smoke_path) if batch1_parity_smoke_path else {}
    )
    pipeline_refactor_readiness = (
        _read_json(pipeline_refactor_readiness_path)
        if pipeline_refactor_readiness_path
        else {}
    )
    artifact_paths = {
        "cache_container_format_research_plan": cache_container_plan_path,
        "cache_container_read_path_gate": cache_container_read_path_gate_path,
        "cache_container_admission_contract": (
            cache_container_admission_contract_path
        ),
        "cache_container_next_axis_contract": (
            cache_container_next_axis_contract_path
        ),
        "cache_container_payload_layout_axis_contract": (
            cache_container_payload_layout_axis_contract_path
        ),
        "cache_container_baseline_comparison_contract": (
            cache_container_baseline_comparison_contract_path
        ),
        "cache_container_compute_tail_non_regression_contract": (
            cache_container_compute_tail_non_regression_contract_path
        ),
        "cache_container_decode_implementation_contract": (
            cache_container_decode_implementation_contract_path
        ),
        "cache_container_decode_path_implementation_plan": (
            cache_container_decode_path_implementation_plan_path
        ),
        "p4_p6_action_plan": p4_p6_action_plan_path,
        "p5_d3d12_cuda_interop_contract": p5_d3d12_cuda_interop_contract_path,
        "p5_d3d12_cuda_functional_interop_probe": (
            p5_d3d12_cuda_functional_interop_probe_path
        ),
        "p5_d3d12_cuda_functional_probe_harness_manifest": (
            p5_d3d12_cuda_functional_probe_harness_manifest_path
        ),
        "p5_p6_cuda_tensor_view_binding_preflight": (
            p5_p6_cuda_tensor_view_binding_preflight_path
        ),
        "p5_p6_guarded_trainer_runtime_ab_preflight": (
            p5_p6_guarded_trainer_runtime_ab_preflight_path
        ),
        "p5_p6_runtime_ab_approval_packet": p5_p6_runtime_ab_approval_packet_path,
        "p6_compute_decode_readiness": p6_compute_decode_readiness_path,
        "p5_p6_native_blueprint": p5_p6_native_blueprint_path,
        "phase_jitter_product_gate": phase_jitter_product_gate_path,
        "phase_jitter_control_delta": phase_jitter_control_delta_path,
        "phase_jitter_phase_attribution": phase_jitter_phase_attribution_path,
        "phase_jitter_optimizer_update_plan": phase_jitter_optimizer_update_plan_path,
        "phase_jitter_optimizer_update_delta": phase_jitter_optimizer_update_delta_path,
        "phase_jitter_optimizer_update_repeat_plan": (
            phase_jitter_optimizer_update_repeat_plan_path
        ),
        "compute_tail_focused_rerun_results": compute_tail_focused_rerun_results_path,
        "compute_tail_raw_order_jitter_repeat_phase_attribution": (
            compute_tail_raw_order_jitter_repeat_phase_attribution_path
        ),
        "compute_tail_raw_order_jitter_order_neutral_excess_mitigation_contract": (
            compute_tail_raw_order_jitter_order_neutral_excess_mitigation_contract_path
        ),
        "guarded_optimizer_update_tail_mitigation_design": (
            guarded_optimizer_update_tail_mitigation_design_path
        ),
        "guarded_optimizer_update_tail_validation_manifest": (
            guarded_optimizer_update_tail_validation_manifest_path
        ),
        "raw_order_jitter_guarded_compute_phase_mitigation_design": (
            raw_order_jitter_guarded_compute_phase_mitigation_design_path
        ),
        "raw_order_jitter_guarded_compute_phase_validation_manifest": (
            raw_order_jitter_guarded_compute_phase_validation_manifest_path
        ),
        "raw_order_compute_phase_validation_run_manifest": (
            raw_order_compute_phase_validation_run_manifest_path
        ),
        "raw_order_compute_phase_validation_playbook": (
            raw_order_compute_phase_validation_playbook_path
        ),
        "raw_order_compute_phase_validation_results": (
            raw_order_compute_phase_validation_results_path
        ),
        "raw_order_compute_phase_validation_verdict": (
            raw_order_compute_phase_validation_verdict_path
        ),
        "p7_non_heavy_next_action": p7_non_heavy_next_action_path,
        "p7_manual_heavy_authorization_bundle": (
            p7_manual_heavy_authorization_bundle_path
        ),
        "p7_validation_failure_triage": p7_validation_failure_triage_path,
        "p7_failed_item_blocker_resolution_matrix": (
            p7_failed_item_blocker_resolution_matrix_path
        ),
        "p7_backward_forward_phase_guard_contract": (
            p7_backward_forward_phase_guard_contract_path
        ),
        "p7_backward_forward_phase_guard_validation_recompute": (
            p7_backward_forward_phase_guard_validation_recompute_path
        ),
        "p7_guarded_raw_order_compute_phase_variant_contract": (
            p7_guarded_raw_order_compute_phase_variant_contract_path
        ),
        "p7_guarded_variant_cross_domain_evidence_contract": (
            p7_guarded_variant_cross_domain_evidence_contract_path
        ),
        "p7_optimizer_update_residual_guard_contract": (
            p7_optimizer_update_residual_guard_contract_path
        ),
        "p7_optimizer_update_residual_guard_recheck_after_nonrepro": (
            p7_optimizer_update_residual_guard_recheck_after_nonrepro_path
        ),
        "p7_optimizer_update_repeat_positive_control_reclassification": (
            p7_optimizer_update_repeat_positive_control_reclassification_path
        ),
        "p7_optimizer_update_repeat_positive_control_resolution": (
            p7_optimizer_update_repeat_positive_control_resolution_path
        ),
        "p7_optimizer_update_repeat_positive_optimizer_internal_resolution": (
            p7_optimizer_update_repeat_positive_optimizer_internal_resolution_path
        ),
        "p7_optimizer_step_micro_attribution": (
            p7_optimizer_step_micro_attribution_path
        ),
        "p7_optimizer_step_micro_profile_instrumentation": (
            p7_optimizer_step_micro_profile_instrumentation_path
        ),
        "p7_optimizer_update_residual_guard_validation_recompute": (
            p7_optimizer_update_residual_guard_validation_recompute_path
        ),
        "p7_optimizer_update_unaccounted_tail_isolation": (
            p7_optimizer_update_unaccounted_tail_isolation_path
        ),
        "p7_optimizer_update_outer_phase_substage_instrumentation_contract": (
            p7_optimizer_update_outer_phase_substage_instrumentation_contract_path
        ),
        "p7_optimizer_update_outer_phase_substage_profile_implementation": (
            p7_optimizer_update_outer_phase_substage_profile_implementation_path
        ),
        "p7_optimizer_update_outer_substage_tail_attribution": (
            p7_optimizer_update_outer_substage_tail_attribution_path
        ),
        "p7_order_neutral_residual_contract_scope_contract": (
            p7_order_neutral_residual_contract_scope_contract_path
        ),
        "guarded_variant_regression_action_plan": (
            guarded_variant_regression_action_plan_path
        ),
        "guarded_variant_mitigation_blueprint": (
            guarded_variant_mitigation_blueprint_path
        ),
        "replacement_phase_guard_contract": replacement_phase_guard_contract_path,
        "guarded_variant_runtime_contract": guarded_variant_runtime_contract_path,
        "guarded_variant_request_adapter_contract": (
            guarded_variant_request_adapter_contract_path
        ),
        "dataloader_guard_metadata": dataloader_guard_metadata_path,
    }
    artifact_reports = {
        key: _read_json(path) if path else {} for key, path in artifact_paths.items()
    }
    artifact_statuses = {
        key: _research_artifact_closed_status(artifact_reports[key], artifact=artifact)
        for key, artifact, _blocker, _checked, _source in _RESEARCH_ARTIFACT_GATE_SPECS
    }
    p4_p6_action_summary = _as_dict(
        artifact_reports.get("p4_p6_action_plan", {}).get("summary")
    )
    cache_container_format_plan_report = _as_dict(
        artifact_reports.get("cache_container_format_research_plan")
    )
    cache_container_format_plan_summary = _as_dict(
        cache_container_format_plan_report.get("summary")
    )
    cache_container_format_plan_source_files = _as_dict(
        cache_container_format_plan_report.get("source_files")
    )
    cache_container_read_path_gate_report = _as_dict(
        artifact_reports.get("cache_container_read_path_gate")
    )
    cache_container_read_path_gate_summary = _as_dict(
        cache_container_read_path_gate_report.get("summary")
    )
    cache_container_read_path_gate_source_files = _as_dict(
        cache_container_read_path_gate_report.get("source_files")
    )
    cache_container_admission_contract_report = _as_dict(
        artifact_reports.get("cache_container_admission_contract")
    )
    cache_container_admission_contract_summary = _as_dict(
        cache_container_admission_contract_report.get("summary")
    )
    cache_container_admission_contract_source_files = _as_dict(
        cache_container_admission_contract_report.get("source_files")
    )
    cache_container_next_axis_contract_report = _as_dict(
        artifact_reports.get("cache_container_next_axis_contract")
    )
    cache_container_next_axis_contract_summary = _as_dict(
        cache_container_next_axis_contract_report.get("summary")
    )
    cache_container_next_axis_contract_source_files = _as_dict(
        cache_container_next_axis_contract_report.get("source_files")
    )
    cache_container_payload_layout_axis_contract_report = _as_dict(
        artifact_reports.get("cache_container_payload_layout_axis_contract")
    )
    cache_container_payload_layout_axis_contract_summary = _as_dict(
        cache_container_payload_layout_axis_contract_report.get("summary")
    )
    cache_container_payload_layout_axis_contract_source_files = _as_dict(
        cache_container_payload_layout_axis_contract_report.get("source_files")
    )
    cache_container_baseline_comparison_contract_report = _as_dict(
        artifact_reports.get("cache_container_baseline_comparison_contract")
    )
    cache_container_baseline_comparison_contract_summary = _as_dict(
        cache_container_baseline_comparison_contract_report.get("summary")
    )
    cache_container_baseline_comparison_contract_source_files = _as_dict(
        cache_container_baseline_comparison_contract_report.get("source_files")
    )
    cache_container_compute_tail_non_regression_contract_report = _as_dict(
        artifact_reports.get("cache_container_compute_tail_non_regression_contract")
    )
    cache_container_compute_tail_non_regression_contract_summary = _as_dict(
        cache_container_compute_tail_non_regression_contract_report.get("summary")
    )
    cache_container_compute_tail_non_regression_contract_source_files = _as_dict(
        cache_container_compute_tail_non_regression_contract_report.get("source_files")
    )
    cache_container_decode_implementation_contract_report = _as_dict(
        artifact_reports.get("cache_container_decode_implementation_contract")
    )
    cache_container_decode_implementation_contract_summary = _as_dict(
        cache_container_decode_implementation_contract_report.get("summary")
    )
    cache_container_decode_implementation_contract_source_files = _as_dict(
        cache_container_decode_implementation_contract_report.get("source_files")
    )
    cache_container_decode_path_implementation_plan_report = _as_dict(
        artifact_reports.get("cache_container_decode_path_implementation_plan")
    )
    cache_container_decode_path_implementation_plan_summary = _as_dict(
        cache_container_decode_path_implementation_plan_report.get("summary")
    )
    cache_container_decode_path_implementation_plan_source_files = _as_dict(
        cache_container_decode_path_implementation_plan_report.get("source_files")
    )
    p5_d3d12_cuda_interop_contract_report = _as_dict(
        artifact_reports.get("p5_d3d12_cuda_interop_contract")
    )
    p5_d3d12_cuda_interop_contract_summary = _as_dict(
        p5_d3d12_cuda_interop_contract_report.get("summary")
    )
    p5_d3d12_cuda_interop_contract_source_files = _as_dict(
        p5_d3d12_cuda_interop_contract_report.get("source_files")
    )
    p5_d3d12_cuda_functional_interop_probe_report = _as_dict(
        artifact_reports.get("p5_d3d12_cuda_functional_interop_probe")
    )
    p5_d3d12_cuda_functional_interop_probe_summary = _as_dict(
        p5_d3d12_cuda_functional_interop_probe_report.get("summary")
    )
    p5_d3d12_cuda_functional_interop_probe_source_files = _as_dict(
        p5_d3d12_cuda_functional_interop_probe_report.get("source_files")
    )
    p5_d3d12_cuda_functional_probe_harness_manifest_report = _as_dict(
        artifact_reports.get("p5_d3d12_cuda_functional_probe_harness_manifest")
    )
    p5_d3d12_cuda_functional_probe_harness_manifest_summary = _as_dict(
        p5_d3d12_cuda_functional_probe_harness_manifest_report.get("summary")
    )
    p5_d3d12_cuda_functional_probe_harness_manifest_source_files = _as_dict(
        p5_d3d12_cuda_functional_probe_harness_manifest_report.get("source_files")
    )
    p5_p6_cuda_tensor_view_binding_preflight_report = _as_dict(
        artifact_reports.get("p5_p6_cuda_tensor_view_binding_preflight")
    )
    p5_p6_cuda_tensor_view_binding_preflight_summary = _as_dict(
        p5_p6_cuda_tensor_view_binding_preflight_report.get("summary")
    )
    p5_p6_cuda_tensor_view_binding_preflight_source_files = _as_dict(
        p5_p6_cuda_tensor_view_binding_preflight_report.get("source_files")
    )
    p5_p6_guarded_trainer_runtime_ab_preflight_report = _as_dict(
        artifact_reports.get("p5_p6_guarded_trainer_runtime_ab_preflight")
    )
    p5_p6_guarded_trainer_runtime_ab_preflight_summary = _as_dict(
        p5_p6_guarded_trainer_runtime_ab_preflight_report.get("summary")
    )
    p5_p6_guarded_trainer_runtime_ab_preflight_source_files = _as_dict(
        p5_p6_guarded_trainer_runtime_ab_preflight_report.get("source_files")
    )
    p6_compute_decode_report = _as_dict(
        artifact_reports.get("p6_compute_decode_readiness")
    )
    p6_compute_decode_summary = _as_dict(
        p6_compute_decode_report.get("summary")
    )
    p6_compute_decode_source_files = _as_dict(
        p6_compute_decode_report.get("source_files")
    )
    p5_p6_runtime_ab_approval_packet_report = _as_dict(
        artifact_reports.get("p5_p6_runtime_ab_approval_packet")
    )
    p5_p6_runtime_ab_approval_packet_summary = _as_dict(
        p5_p6_runtime_ab_approval_packet_report.get("summary")
    )
    p5_p6_runtime_ab_approval_packet_source_files = _as_dict(
        p5_p6_runtime_ab_approval_packet_report.get("source_files")
    )
    p5_p6_native_blueprint_report = _as_dict(
        artifact_reports.get("p5_p6_native_blueprint")
    )
    p5_p6_native_blueprint_summary = _as_dict(
        p5_p6_native_blueprint_report.get("summary")
    )
    p5_p6_native_blueprint_source_files = _as_dict(
        p5_p6_native_blueprint_report.get("source_files")
    )
    p7_non_heavy_report = _as_dict(
        artifact_reports.get("p7_non_heavy_next_action")
    )
    p7_non_heavy_summary = _as_dict(p7_non_heavy_report.get("summary"))
    p7_non_heavy_source_files = _as_dict(p7_non_heavy_report.get("source_files"))
    p7_manual_heavy_authorization_bundle_summary = _as_dict(
        artifact_reports.get("p7_manual_heavy_authorization_bundle", {}).get(
            "summary"
        )
    )
    p7_validation_verdict_report = _as_dict(
        artifact_reports.get("raw_order_compute_phase_validation_verdict")
    )
    p7_validation_verdict_summary = _as_dict(
        p7_validation_verdict_report.get("summary")
    )
    p7_validation_verdict_source_files = _as_dict(
        p7_validation_verdict_report.get("source_files")
    )
    p7_validation_failure_triage_report = _as_dict(
        artifact_reports.get("p7_validation_failure_triage")
    )
    p7_validation_failure_triage_summary = _as_dict(
        p7_validation_failure_triage_report.get("summary")
    )
    p7_validation_failure_triage_source_files = _as_dict(
        p7_validation_failure_triage_report.get("source_files")
    )
    p7_validation_failure_triage_rows = [
        row
        for row in _as_list(p7_validation_failure_triage_report.get("triage_rows"))
        if isinstance(row, dict)
    ]
    p7_validation_failure_triage_row_evidence_artifact_ids = _unique_strings(
        *[
            row.get("evidence_artifact_ids")
            for row in p7_validation_failure_triage_rows
        ]
    )
    p7_failed_item_blocker_resolution_report = _as_dict(
        artifact_reports.get("p7_failed_item_blocker_resolution_matrix")
    )
    p7_failed_item_blocker_resolution_summary = _as_dict(
        p7_failed_item_blocker_resolution_report.get("summary")
    )
    p7_failed_item_blocker_resolution_source_files = _as_dict(
        p7_failed_item_blocker_resolution_report.get("source_files")
    )
    p7_backward_forward_phase_guard_contract_report = _as_dict(
        artifact_reports.get("p7_backward_forward_phase_guard_contract")
    )
    p7_backward_forward_phase_guard_contract_summary = _as_dict(
        p7_backward_forward_phase_guard_contract_report.get("summary")
    )
    p7_backward_forward_phase_guard_contract_source_files = _as_dict(
        p7_backward_forward_phase_guard_contract_report.get("source_files")
    )
    p7_backward_forward_phase_guard_validation_recompute_summary = _as_dict(
        artifact_reports.get(
            "p7_backward_forward_phase_guard_validation_recompute", {}
        ).get("summary")
    )
    p7_backward_forward_phase_guard_validation_recompute_source_files = _as_dict(
        artifact_reports.get(
            "p7_backward_forward_phase_guard_validation_recompute", {}
        ).get("source_files")
    )
    p7_guarded_raw_order_contract_report = _as_dict(
        artifact_reports.get("p7_guarded_raw_order_compute_phase_variant_contract")
    )
    p7_guarded_raw_order_contract_summary = _as_dict(
        p7_guarded_raw_order_contract_report.get("summary")
    )
    p7_guarded_raw_order_contract_source_files = _as_dict(
        p7_guarded_raw_order_contract_report.get("source_files")
    )
    p7_guarded_variant_cross_domain_summary = _as_dict(
        artifact_reports.get(
            "p7_guarded_variant_cross_domain_evidence_contract", {}
        ).get("summary")
    )
    p7_optimizer_residual_contract_report = _as_dict(
        artifact_reports.get("p7_optimizer_update_residual_guard_contract")
    )
    p7_optimizer_residual_contract_summary = _as_dict(
        p7_optimizer_residual_contract_report.get("summary")
    )
    p7_optimizer_residual_contract_source_files = _as_dict(
        p7_optimizer_residual_contract_report.get("source_files")
    )
    p7_optimizer_residual_recheck_report = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_residual_guard_recheck_after_nonrepro"
        )
    )
    p7_optimizer_residual_recheck_summary = _as_dict(
        p7_optimizer_residual_recheck_report.get("summary")
    )
    p7_optimizer_residual_recheck_source_files = _as_dict(
        p7_optimizer_residual_recheck_report.get("source_files")
    )
    p7_optimizer_repeat_reclassification_report = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_repeat_positive_control_reclassification"
        )
    )
    p7_optimizer_repeat_reclassification_summary = _as_dict(
        p7_optimizer_repeat_reclassification_report.get("summary")
    )
    p7_optimizer_repeat_reclassification_source_files = _as_dict(
        p7_optimizer_repeat_reclassification_report.get("source_files")
    )
    p7_optimizer_repeat_resolution_report = _as_dict(
        artifact_reports.get("p7_optimizer_update_repeat_positive_control_resolution")
    )
    p7_optimizer_repeat_resolution_summary = _as_dict(
        p7_optimizer_repeat_resolution_report.get("summary")
    )
    p7_optimizer_repeat_resolution_source_files = _as_dict(
        p7_optimizer_repeat_resolution_report.get("source_files")
    )
    p7_optimizer_repeat_internal_resolution_report = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_repeat_positive_optimizer_internal_resolution"
        )
    )
    p7_optimizer_repeat_internal_resolution_summary = _as_dict(
        p7_optimizer_repeat_internal_resolution_report.get("summary")
    )
    p7_optimizer_repeat_internal_resolution_source_files = _as_dict(
        p7_optimizer_repeat_internal_resolution_report.get("source_files")
    )
    p7_optimizer_step_micro_attribution_report = _as_dict(
        artifact_reports.get("p7_optimizer_step_micro_attribution")
    )
    p7_optimizer_step_micro_attribution_summary = _as_dict(
        p7_optimizer_step_micro_attribution_report.get("summary")
    )
    p7_optimizer_step_micro_attribution_source_files = _as_dict(
        p7_optimizer_step_micro_attribution_report.get("source_files")
    )
    p7_optimizer_step_micro_profile_instrumentation_report = _as_dict(
        artifact_reports.get("p7_optimizer_step_micro_profile_instrumentation")
    )
    p7_optimizer_step_micro_profile_instrumentation_summary = _as_dict(
        p7_optimizer_step_micro_profile_instrumentation_report.get("summary")
    )
    p7_optimizer_step_micro_profile_instrumentation_source_files = _as_dict(
        p7_optimizer_step_micro_profile_instrumentation_report.get("source_files")
    )
    p7_optimizer_residual_validation_recompute_summary = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_residual_guard_validation_recompute", {}
        ).get("summary")
    )
    p7_optimizer_unaccounted_tail_isolation_report = _as_dict(
        artifact_reports.get("p7_optimizer_update_unaccounted_tail_isolation")
    )
    p7_optimizer_unaccounted_tail_isolation_summary = _as_dict(
        p7_optimizer_unaccounted_tail_isolation_report.get("summary")
    )
    p7_optimizer_unaccounted_tail_isolation_source_files = _as_dict(
        p7_optimizer_unaccounted_tail_isolation_report.get("source_files")
    )
    p7_optimizer_outer_phase_substage_instrumentation_summary = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_outer_phase_substage_instrumentation_contract", {}
        ).get("summary")
    )
    p7_optimizer_outer_phase_substage_instrumentation_source_files = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_outer_phase_substage_instrumentation_contract", {}
        ).get("source_files")
    )
    p7_optimizer_outer_phase_substage_profile_implementation_summary = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_outer_phase_substage_profile_implementation", {}
        ).get("summary")
    )
    p7_optimizer_outer_phase_substage_profile_implementation_source_files = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_outer_phase_substage_profile_implementation", {}
        ).get("source_files")
    )
    p7_optimizer_outer_substage_tail_attribution_summary = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_outer_substage_tail_attribution", {}
        ).get("summary")
    )
    p7_optimizer_outer_substage_tail_attribution_source_files = _as_dict(
        artifact_reports.get(
            "p7_optimizer_update_outer_substage_tail_attribution", {}
        ).get("source_files")
    )
    raw_order_jitter_guarded_compute_phase_mitigation_design_report = _as_dict(
        artifact_reports.get("raw_order_jitter_guarded_compute_phase_mitigation_design")
    )
    raw_order_jitter_guarded_compute_phase_mitigation_design_summary = _as_dict(
        raw_order_jitter_guarded_compute_phase_mitigation_design_report.get("summary")
    )
    raw_order_jitter_guarded_compute_phase_validation_manifest_report = _as_dict(
        artifact_reports.get("raw_order_jitter_guarded_compute_phase_validation_manifest")
    )
    raw_order_jitter_guarded_compute_phase_validation_manifest_summary = _as_dict(
        raw_order_jitter_guarded_compute_phase_validation_manifest_report.get(
            "summary"
        )
    )
    raw_order_jitter_guarded_compute_phase_validation_manifest_source_files = _as_dict(
        raw_order_jitter_guarded_compute_phase_validation_manifest_report.get(
            "source_files"
        )
    )
    raw_order_jitter_guarded_compute_phase_validation_manifest_mitigation_source = (
        _as_dict(
            raw_order_jitter_guarded_compute_phase_validation_manifest_source_files.get(
                "mitigation_design"
            )
        )
    )
    raw_order_jitter_guarded_compute_phase_validation_manifest_phase_source = (
        _as_dict(
            raw_order_jitter_guarded_compute_phase_validation_manifest_source_files.get(
                "phase_attribution"
            )
        )
    )
    raw_order_compute_phase_validation_run_manifest_report = _as_dict(
        artifact_reports.get("raw_order_compute_phase_validation_run_manifest")
    )
    raw_order_compute_phase_validation_run_manifest_summary = _as_dict(
        raw_order_compute_phase_validation_run_manifest_report.get("summary")
    )
    raw_order_compute_phase_validation_results_report = _as_dict(
        artifact_reports.get("raw_order_compute_phase_validation_results")
    )
    raw_order_compute_phase_validation_results_summary = _as_dict(
        raw_order_compute_phase_validation_results_report.get("summary")
    )
    raw_order_compute_phase_validation_results_source_files = _as_dict(
        raw_order_compute_phase_validation_results_report.get("source_files")
    )
    raw_order_compute_phase_validation_results_outputs = [
        item
        for item in _as_list(
            raw_order_compute_phase_validation_results_report.get("outputs")
        )
        if isinstance(item, dict)
    ]
    raw_order_compute_phase_validation_results_output_source_paths = _unique_strings(
        [
            str(_as_dict(item.get("source")).get("path") or "")
            for item in raw_order_compute_phase_validation_results_outputs
        ]
    )
    raw_order_jitter_order_neutral_excess_mitigation_contract_report = _as_dict(
        artifact_reports.get("raw_order_jitter_order_neutral_excess_mitigation_contract")
    )
    raw_order_jitter_order_neutral_excess_mitigation_contract_summary = _as_dict(
        raw_order_jitter_order_neutral_excess_mitigation_contract_report.get("summary")
    )
    p7_order_neutral_scope_contract_report = _as_dict(
        artifact_reports.get("p7_order_neutral_residual_contract_scope_contract")
    )
    p7_order_neutral_scope_contract_summary = _as_dict(
        p7_order_neutral_scope_contract_report.get("summary")
    )
    p7_order_neutral_scope_contract_source_files = _as_dict(
        p7_order_neutral_scope_contract_report.get("source_files")
    )
    guarded_variant_request_adapter_contract_summary = _as_dict(
        artifact_reports.get("guarded_variant_request_adapter_contract", {}).get(
            "summary"
        )
    )
    guarded_variant_runtime_contract_report = _as_dict(
        artifact_reports.get("guarded_variant_runtime_contract")
    )
    guarded_variant_runtime_contract_summary = _as_dict(
        guarded_variant_runtime_contract_report.get("summary")
    )
    guarded_variant_runtime_contract_source_files = _as_dict(
        guarded_variant_runtime_contract_report.get("source_files")
    )
    guarded_variant_request_adapter_contract_report = _as_dict(
        artifact_reports.get("guarded_variant_request_adapter_contract")
    )
    guarded_variant_request_adapter_contract_source_files = _as_dict(
        guarded_variant_request_adapter_contract_report.get("source_files")
    )
    dataloader_guard_metadata_summary = _as_dict(
        artifact_reports.get("dataloader_guard_metadata", {}).get("summary")
    )
    release_smoke_status = _release_smoke_status(release_smoke)
    batch1_parity_status = _batch1_parity_status(batch1_parity_smoke)
    pipeline_refactor_status = _pipeline_refactor_readiness_status(
        pipeline_refactor_readiness
    )
    visible_dependency_ids = _visible_dependency_ids(gate)
    blocked_dependency_ids = _blocked_dependency_ids(gate)
    gate_summary = _as_dict(gate.get("summary"))
    gate_evidence = _as_dict(gate.get("evidence"))
    readiness_summary = _as_dict(readiness.get("summary"))
    gate_source_mtime = _mtime(gate_path)
    gate_audit_source_mtime = float(
        _first_present_value(
            (gate_summary, "resource_gate_audit_source_mtime"),
            (gate_evidence, "resource_gate_audit_source_mtime"),
            default=0.0,
        )
        or 0.0
    )
    gate_not_older_than_audit = bool(
        gate_source_mtime > 0
        and gate_audit_source_mtime > 0
        and gate_source_mtime >= gate_audit_source_mtime
    )
    report_only_projection_gate_open_keys = _unique_strings(
        _first_present_value(
            (gate_summary, "report_only_projection_gate_open_keys"),
            (gate_evidence, "report_only_projection_gate_open_keys"),
            default=[],
        )
    )
    report_only_projection_gate_open_count = int(
        _first_present_value(
            (gate_summary, "report_only_projection_gate_open_count"),
            (gate_evidence, "report_only_projection_gate_open_count"),
            default=len(report_only_projection_gate_open_keys),
        )
        or 0
    )
    report_only_projection_all_gates_closed = bool(
        _first_present_value(
            (gate_summary, "report_only_projection_all_gates_closed"),
            (gate_evidence, "report_only_projection_all_gates_closed"),
            default=not report_only_projection_gate_open_keys,
        )
    )
    p7_product_proof_blocker_ids = _unique_strings(
        _first_present_value(
            (gate_summary, "p7_product_proof_blocker_ids"),
            (gate_evidence, "p7_product_proof_blocker_ids"),
            default=[],
        )
    )
    p7_product_proof_validation_failed_item_ids = _unique_strings(
        _first_present_value(
            (gate_summary, "p7_product_proof_validation_failed_item_ids"),
            (gate_evidence, "p7_product_proof_validation_failed_item_ids"),
            default=[],
        )
    )
    p7_manual_heavy_target_validation_item_ids = _unique_strings(
        _first_present_value(
            (
                gate_summary,
                "p7_manual_heavy_authorization_bundle_target_validation_item_ids",
            ),
            (
                gate_evidence,
                "p7_manual_heavy_authorization_bundle_target_validation_item_ids",
            ),
            (
                p7_manual_heavy_authorization_bundle_summary,
                "target_validation_item_ids",
            ),
            default=[],
        )
    )
    p7_manual_heavy_missing_failed_validation_item_ids = [
        item
        for item in p7_product_proof_validation_failed_item_ids
        if item not in p7_manual_heavy_target_validation_item_ids
    ]
    p7_manual_heavy_extra_target_validation_item_ids = [
        item
        for item in p7_manual_heavy_target_validation_item_ids
        if item not in p7_product_proof_validation_failed_item_ids
    ]
    p7_manual_heavy_target_failed_validation_item_coverage_exact = bool(
        p7_product_proof_validation_failed_item_ids
        and not p7_manual_heavy_missing_failed_validation_item_ids
        and not p7_manual_heavy_extra_target_validation_item_ids
    )
    p7_product_proof_product_blocker_details = _unique_strings(
        _first_present_value(
            (gate_summary, "p7_product_proof_product_blocker_details"),
            (gate_evidence, "p7_product_proof_product_blocker_details"),
            default=[],
        )
    )
    p7_product_proof_manual_heavy_action_ids = _unique_strings(
        _first_present_value(
            (gate_summary, "p7_product_proof_manual_heavy_action_ids"),
            (gate_evidence, "p7_product_proof_manual_heavy_action_ids"),
            default=[],
        )
    )
    p5_p6_p7_boundary_projection = _lossless_p5_p6_p7_boundary_summary_projection(
        gate_summary,
        gate_evidence,
        readiness_summary,
    )
    p5_p6_p7_boundary_has_projection = bool(p5_p6_p7_boundary_projection)
    p5_p6_p7_boundary_unexpected_open_keys = _unique_strings(
        p5_p6_p7_boundary_projection.get(
            "lossless_p5_p6_p7_boundary_summary_unexpected_open_gate_keys"
        ),
        p5_p6_p7_boundary_projection.get(
            "lossless_p5_p6_p7_boundary_summary_gate_open_keys"
        ),
        p5_p6_p7_boundary_projection.get(
            "lossless_p5_p6_p7_boundary_summary_report_only_open_keys"
        ),
        p5_p6_p7_boundary_projection.get(
            "lossless_p5_p6_p7_boundary_summary_report_only_gate_open_keys"
        ),
    )
    p5_p6_p7_boundary_unexpected_open_count = int(
        p5_p6_p7_boundary_projection.get(
            "lossless_p5_p6_p7_boundary_summary_unexpected_open_gate_count",
            p5_p6_p7_boundary_projection.get(
                "lossless_p5_p6_p7_boundary_summary_gate_open_count",
                0,
            ),
        )
        or 0
    )
    p5_p6_p7_boundary_report_only_open_count = int(
        p5_p6_p7_boundary_projection.get(
            "lossless_p5_p6_p7_boundary_summary_report_only_open_count",
            p5_p6_p7_boundary_projection.get(
                "lossless_p5_p6_p7_boundary_summary_report_only_gate_open_count",
                0,
            ),
        )
        or 0
    )
    p5_p6_p7_boundary_fail_closed = bool(
        p5_p6_p7_boundary_projection.get(
            "lossless_p5_p6_p7_boundary_summary_fail_closed"
        )
    )

    blockers: list[str] = []
    if gate.get("_load_error"):
        blockers.append("lossless_resource_center_gate_missing_or_invalid")
    elif gate_audit_source_mtime > 0 and not gate_not_older_than_audit:
        blockers.append("lossless_resource_center_gate_stale_for_audit")
    if report_only_projection_gate_open_count:
        blockers.append("lossless_report_only_projection_gate_open")
    if (
        p5_p6_p7_boundary_unexpected_open_count
        or p5_p6_p7_boundary_unexpected_open_keys
        or p5_p6_p7_boundary_report_only_open_count
        or (p5_p6_p7_boundary_has_projection and not p5_p6_p7_boundary_fail_closed)
    ):
        blockers.append("lossless_p5_p6_p7_boundary_summary_gate_open")
    if bool(gate.get("allowed")):
        blockers.append("lossless_resource_center_allowed")
    if bool(gate.get("training_path_enabled")):
        blockers.append("lossless_training_path_enabled")
    if bool(gate.get("default_enabled")):
        blockers.append("lossless_default_enabled")
    if visible_dependency_ids:
        blockers.append("lossless_resource_center_dependencies_visible")
    if not release_smoke_status["ok"]:
        blockers.append("core_release_smoke_missing_or_failed")
    if not batch1_parity_status["ok"]:
        blockers.append("batch1_handler_parity_smoke_missing_or_failed")
    if not pipeline_refactor_status["ok"]:
        blockers.append("experimental_claim_gate_evidence_missing_or_open")
    for key, _artifact, blocker, _checked, _source in _RESEARCH_ARTIFACT_GATE_SPECS:
        if not artifact_statuses[key]["ok"]:
            blockers.append(blocker)

    full_trainer_deferred_blockers = _unique_strings(readiness.get("product_blockers"))
    resource_gate_groups = _resource_gate_blocker_groups(gate)
    resource_gate_deferred_blockers = _unique_strings(
        gate.get("blockers"),
        *resource_gate_groups.values(),
    )
    deferred_research_blockers = _unique_strings(
        full_trainer_deferred_blockers,
        resource_gate_deferred_blockers,
    )
    deferred_research_blocker_groups = {
        key: _unique_strings(value)
        for key, value in _as_dict(readiness.get("product_blocker_groups")).items()
    }
    deferred_research_blocker_groups.update(resource_gate_groups)
    if readiness.get("_load_error"):
        deferred_research_blockers.append("lossless_full_trainer_readiness_missing")

    release_ready = not blockers
    core_release_smoke_covered = bool(
        release_smoke_status["ok"] and batch1_parity_status["ok"]
    )
    experimental_claim_gates_closed = bool(pipeline_refactor_status["ok"])
    research_artifact_gates_closed = all(
        artifact_statuses[key]["ok"]
        for key, _artifact, _blocker, _checked, _source in _RESEARCH_ARTIFACT_GATE_SPECS
    )
    decode_implementation_status = artifact_statuses[
        "cache_container_decode_implementation_contract"
    ]
    release_validation_todo: list[str] = []
    if not release_smoke_status["ok"]:
        release_validation_todo.append("run core release smoke")
    if not batch1_parity_status["ok"]:
        release_validation_todo.append("run batch1 handler parity smoke")
    if not pipeline_refactor_status["ok"]:
        release_validation_todo.append("refresh experimental claim gate evidence")
    summary = {
        "release_ready": release_ready,
        "release_strategy": "stable_baseline_first_release",
        "default_training_baseline": "current_raw_dataloader_and_current_trainer",
        "lossless_resource_center_allowed": bool(gate.get("allowed")),
        "lossless_training_path_enabled": bool(gate.get("training_path_enabled")),
        "lossless_default_enabled": bool(gate.get("default_enabled")),
        "lossless_visible_dependency_count": len(visible_dependency_ids),
        "lossless_blocked_dependency_count": len(blocked_dependency_ids),
        "lossless_resource_center_gate_source_mtime": gate_source_mtime,
        "lossless_resource_center_gate_audit_source": str(
            _first_present_value(
                (gate_summary, "resource_gate_audit_source"),
                (gate_evidence, "resource_gate_audit_source"),
                default="",
            )
            or ""
        ),
        "lossless_resource_center_gate_audit_source_exists": bool(
            _first_present_value(
                (gate_summary, "resource_gate_audit_source_exists"),
                (gate_evidence, "resource_gate_audit_source_exists"),
            )
        ),
        "lossless_resource_center_gate_audit_source_mtime": gate_audit_source_mtime,
        "lossless_resource_center_gate_not_older_than_audit": (
            gate_not_older_than_audit
        ),
        "lossless_resource_center_gate_freshness_ready": gate_not_older_than_audit,
        "lossless_report_only_projection_gate_open_keys": (
            report_only_projection_gate_open_keys
        ),
        "lossless_report_only_projection_gate_open_count": (
            report_only_projection_gate_open_count
        ),
        "lossless_report_only_projection_all_gates_closed": (
            report_only_projection_all_gates_closed
        ),
        **_lossless_dataloader_guard_metadata_projection(
            dataloader_guard_metadata_summary
        ),
        **_lossless_p7_manual_heavy_authorization_bundle_projection(
            p7_manual_heavy_authorization_bundle_summary
        ),
        **_lossless_p7_guarded_variant_manual_heavy_packet_projection(
            gate_summary,
            gate_evidence,
        ),
        **_lossless_p7_backward_forward_manual_heavy_packet_projection(
            gate_summary,
            gate_evidence,
        ),
        "lossless_p7_manual_heavy_authorization_bundle_target_validation_item_count": len(
            p7_manual_heavy_target_validation_item_ids
        ),
        "lossless_p7_manual_heavy_authorization_bundle_target_failed_validation_item_coverage_exact": (
            p7_manual_heavy_target_failed_validation_item_coverage_exact
        ),
        "lossless_p7_manual_heavy_authorization_bundle_missing_failed_validation_item_ids": (
            p7_manual_heavy_missing_failed_validation_item_ids
        ),
        "lossless_p7_manual_heavy_authorization_bundle_extra_target_validation_item_ids": (
            p7_manual_heavy_extra_target_validation_item_ids
        ),
        **_lossless_p7_optimizer_update_detail_projection(
            p7_optimizer_unaccounted_tail_isolation_summary,
            p7_optimizer_unaccounted_tail_isolation_source_files,
            p7_optimizer_outer_phase_substage_instrumentation_summary,
            p7_optimizer_outer_phase_substage_instrumentation_source_files,
            p7_optimizer_outer_phase_substage_profile_implementation_summary,
            p7_optimizer_outer_phase_substage_profile_implementation_source_files,
            p7_optimizer_outer_substage_tail_attribution_summary,
            p7_optimizer_outer_substage_tail_attribution_source_files,
        ),
        **_lossless_p7_failed_item_blocker_projection(
            p7_failed_item_blocker_resolution_summary
        ),
        **_lossless_p7_artifact_source_projection(
            "lossless_p7_non_heavy_next_action",
            p7_non_heavy_summary,
            p7_non_heavy_source_files,
            (
                "strict_delta",
                "phase_boundary",
                "raw_order_repeat_phase",
                "guarded_validation",
                "compute_phase_validation_results",
                "guarded_variant_heavy_evidence",
                "phase_jitter_product_gate",
                "authorization_stage_readiness",
                "p4_p6_action_plan",
                "p5_p7_platform_boundary",
                "cache_container_next_axis_contract",
            ),
        ),
        **_lossless_p7_artifact_source_projection(
            "lossless_p7_failed_item_blocker",
            p7_failed_item_blocker_resolution_summary,
            p7_failed_item_blocker_resolution_source_files,
            (
                "validation_results",
                "validation_verdict",
                "guarded_variant_heavy_evidence",
                "backward_forward_phase_guard_contract",
                "backward_forward_phase_guard_validation_recompute",
                "backward_forward_phase_guard_manual_heavy_packet",
                "repeat_phase_attribution",
                "optimizer_residual_guard_contract",
                "optimizer_update_tail_attribution",
                "optimizer_update_unaccounted_tail_isolation",
                "optimizer_update_outer_phase_substage_contract",
                "optimizer_update_outer_phase_substage_implementation",
                "optimizer_update_outer_substage_tail_attribution",
                "optimizer_update_residual_guard_nonrepro_recheck",
                "optimizer_update_repeat_positive_control_reclassification",
                "optimizer_update_repeat_positive_control_resolution",
                "optimizer_update_repeat_positive_optimizer_internal_resolution",
                "optimizer_step_micro_attribution",
                "optimizer_step_micro_profile_instrumentation",
                "optimizer_residual_guard_validation_recompute",
                "optimizer_update_tail_validation_manifest",
                "order_neutral_excess_mitigation_contract",
                "guarded_variant_manual_heavy_packet",
                "guarded_variant_manual_heavy_run_manifest",
                "guarded_variant_cross_domain_evidence_contract",
            ),
        ),
        **_lossless_p7_artifact_source_projection(
            "lossless_p7_validation_failure_triage",
            p7_validation_failure_triage_summary,
            p7_validation_failure_triage_source_files,
            (
                "validation_results",
                "guarded_variant_heavy_evidence",
                "raw_order_repeat_phase_attribution",
                "phase_jitter_product_gate",
                "phase_boundary",
                "optimizer_update_tail_attribution",
            ),
        ),
        **_lossless_p7_artifact_source_projection(
            "lossless_p7_backward_forward_phase_guard_contract",
            p7_backward_forward_phase_guard_contract_summary,
            p7_backward_forward_phase_guard_contract_source_files,
            (
                "raw_order_repeat_phase_attribution",
                "phase_boundary",
                "validation_failure_triage",
                "validation_results",
                "validation_verdict",
            ),
        ),
        **_lossless_p7_backward_forward_validation_recompute_projection(
            p7_backward_forward_phase_guard_validation_recompute_summary
        ),
        **_lossless_p7_artifact_source_projection(
            "lossless_p7_backward_forward_phase_guard_validation_recompute",
            p7_backward_forward_phase_guard_validation_recompute_summary,
            p7_backward_forward_phase_guard_validation_recompute_source_files,
            (
                "validation_results",
                "validation_verdict",
                "backward_forward_phase_guard_contract",
                "raw_order_repeat_phase_attribution",
            ),
        ),
        **_lossless_p7_artifact_source_projection(
            "lossless_p7_order_neutral_residual_contract_scope_contract",
            p7_order_neutral_scope_contract_summary,
            p7_order_neutral_scope_contract_source_files,
            (
                "validation_failure_triage",
                "validation_results",
                "validation_verdict",
                "order_neutral_aggregate",
                "case_order_confounding_resolution_plan",
                "order_neutral_excess_mitigation_contract",
                "guarded_compute_phase_validation_manifest",
            ),
        ),
        **_lossless_p7_guarded_variant_cross_domain_projection(
            p7_guarded_variant_cross_domain_summary
        ),
        **_lossless_p7_optimizer_residual_contract_source_projection(
            p7_optimizer_residual_contract_summary,
            p7_optimizer_residual_contract_source_files,
        ),
        **_lossless_p7_optimizer_residual_recheck_projection(
            p7_optimizer_residual_recheck_summary,
            p7_optimizer_residual_recheck_source_files,
        ),
        **_lossless_p7_optimizer_repeat_reclassification_projection(
            p7_optimizer_repeat_reclassification_summary,
            p7_optimizer_repeat_reclassification_source_files,
        ),
        **_lossless_p7_optimizer_repeat_resolution_projection(
            p7_optimizer_repeat_resolution_summary,
            p7_optimizer_repeat_resolution_source_files,
        ),
        **_lossless_p7_optimizer_repeat_internal_resolution_projection(
            p7_optimizer_repeat_internal_resolution_summary,
            p7_optimizer_repeat_internal_resolution_source_files,
        ),
        **_lossless_p7_optimizer_step_micro_attribution_projection(
            p7_optimizer_step_micro_attribution_summary,
            p7_optimizer_step_micro_attribution_source_files,
        ),
        **_lossless_p7_optimizer_step_micro_profile_instrumentation_projection(
            p7_optimizer_step_micro_profile_instrumentation_summary,
            p7_optimizer_step_micro_profile_instrumentation_source_files,
        ),
        **_lossless_p7_optimizer_residual_validation_recompute_projection(
            p7_optimizer_residual_validation_recompute_summary
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_ready": bool(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "guarded_compute_phase_mitigation_design_ready"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_does_not_run_runtime": bool(
            raw_order_jitter_guarded_compute_phase_mitigation_design_report.get(
                "does_not_run_runtime"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_does_not_mutate_runtime": bool(
            raw_order_jitter_guarded_compute_phase_mitigation_design_report.get(
                "does_not_mutate_runtime"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_selected_design_id": str(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "selected_design_id"
            )
            or ""
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_selected_candidate_count": int(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "selected_candidate_count"
            )
            or 0
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_validation_gate_count": int(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "validation_gate_count"
            )
            or 0
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_target_positive_step_wall_count": int(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "target_positive_step_wall_count"
            )
            or 0
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_target_dominant_source_counts": _as_dict(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "target_dominant_positive_source_counts"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_source_order_neutral_excess_contract_ready": bool(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "source_order_neutral_excess_contract_ready"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_source_order_neutral_excess_contract_id": str(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "source_order_neutral_excess_contract_id"
            )
            or ""
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_order_neutral_residual_contract_target_group_labels": _unique_strings(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "order_neutral_residual_contract_target_group_labels"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_order_neutral_residual_contract_target_metric_ids": _unique_strings(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "order_neutral_residual_contract_target_metric_ids"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_mitigation_design_next_recommended": str(
            raw_order_jitter_guarded_compute_phase_mitigation_design_summary.get(
                "next_recommended"
            )
            or ""
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_ready": bool(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "guarded_compute_phase_validation_manifest_ready"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_does_not_run_runtime": bool(
            raw_order_jitter_guarded_compute_phase_validation_manifest_report.get(
                "does_not_run_runtime"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_does_not_mutate_runtime": bool(
            raw_order_jitter_guarded_compute_phase_validation_manifest_report.get(
                "does_not_mutate_runtime"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_source_mitigation_design_ready": bool(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "source_mitigation_design_ready"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_source_phase_attribution_ready": bool(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "source_phase_attribution_ready"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_source_mitigation_design_path": str(
            raw_order_jitter_guarded_compute_phase_validation_manifest_mitigation_source.get(
                "path"
            )
            or ""
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_source_phase_attribution_path": str(
            raw_order_jitter_guarded_compute_phase_validation_manifest_phase_source.get(
                "path"
            )
            or ""
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_source_selected_design_id": str(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "source_selected_design_id"
            )
            or ""
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_expected_selected_design_id": str(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "source_expected_selected_design_id"
            )
            or ""
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_required_validation_item_ids": _unique_strings(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "required_validation_item_ids"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_manual_heavy_validation_required": bool(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "manual_heavy_validation_required"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_item_count": int(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "validation_item_count"
            )
            or 0
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_target_positive_step_wall_count": int(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "target_positive_step_wall_count"
            )
            or 0
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_target_dominant_source_counts": _as_dict(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "target_dominant_positive_source_counts"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_group_classifications": _as_dict(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "group_classifications"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_safe_to_auto_execute": bool(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "safe_to_auto_execute"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_declares_validation_passed": bool(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "declares_validation_passed"
            )
        ),
        "lossless_raw_order_jitter_guarded_compute_phase_validation_manifest_next_recommended": str(
            raw_order_jitter_guarded_compute_phase_validation_manifest_summary.get(
                "next_recommended"
            )
            or ""
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_ready": bool(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "raw_order_compute_phase_validation_run_manifest_ready"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_selected_command_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "selected_command_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_ready_to_execute_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "ready_to_execute_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_structured_argv_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "structured_argv_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_order_neutral_contract_ready": bool(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "source_order_neutral_excess_contract_ready"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_order_neutral_contract_id": str(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "source_order_neutral_excess_contract_id"
            )
            or ""
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_order_neutral_contract_consumed": bool(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "order_neutral_residual_contract_consumed"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_order_neutral_scope_ready": bool(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "order_neutral_residual_contract_scope_ready"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_order_neutral_contract_row_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "order_neutral_residual_contract_row_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_order_neutral_target_group_labels": _unique_strings(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "order_neutral_residual_contract_target_group_labels"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_order_neutral_target_metric_ids": _unique_strings(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "order_neutral_residual_contract_target_metric_ids"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_required_validation_item_ids": _unique_strings(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "required_validation_item_ids"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_manual_heavy_validation_required": bool(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "manual_heavy_validation_required"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_safe_to_auto_execute": bool(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "safe_to_auto_execute"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_execute_requested": bool(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "execute_requested"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_allow_heavy": bool(
            raw_order_compute_phase_validation_run_manifest_summary.get("allow_heavy")
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_executed_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "executed_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_execution_skipped_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "execution_skipped_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_execution_failure_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "execution_failure_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_output_exists_before_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "output_exists_before_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_pending_output_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "pending_output_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_validation_issue_count": int(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "validation_issue_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_declares_validation_passed": bool(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "declares_validation_passed"
            )
        ),
        "lossless_raw_order_compute_phase_validation_run_manifest_next_recommended": str(
            raw_order_compute_phase_validation_run_manifest_summary.get(
                "next_recommended"
            )
            or ""
        ),
        "lossless_raw_order_compute_phase_validation_results_ready": bool(
            raw_order_compute_phase_validation_results_summary.get(
                "raw_order_compute_phase_validation_results_ready"
            )
        ),
        "lossless_raw_order_compute_phase_validation_results_source_run_manifest_path": str(
            _as_dict(
                raw_order_compute_phase_validation_results_source_files.get(
                    "run_manifest"
                )
            ).get("path")
            or ""
        ),
        "lossless_raw_order_compute_phase_validation_results_source_optimizer_recompute_path": str(
            _as_dict(
                raw_order_compute_phase_validation_results_source_files.get(
                    "optimizer_recompute"
                )
            ).get("path")
            or ""
        ),
        "lossless_raw_order_compute_phase_validation_results_source_backward_forward_recompute_path": str(
            _as_dict(
                raw_order_compute_phase_validation_results_source_files.get(
                    "backward_forward_recompute"
                )
            ).get("path")
            or ""
        ),
        "lossless_raw_order_compute_phase_validation_results_source_backward_forward_manual_heavy_run_manifest_path": str(
            _as_dict(
                raw_order_compute_phase_validation_results_source_files.get(
                    "backward_forward_manual_heavy_run_manifest"
                )
            ).get("path")
            or ""
        ),
        "lossless_raw_order_compute_phase_validation_results_source_count_mismatch_item_count": int(
            raw_order_compute_phase_validation_results_summary.get(
                "source_count_mismatch_item_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_results_source_count_mismatch_item_ids": _unique_strings(
            raw_order_compute_phase_validation_results_summary.get(
                "source_count_mismatch_item_ids"
            )
        ),
        "lossless_raw_order_compute_phase_validation_results_output_parse_ok_count": int(
            raw_order_compute_phase_validation_results_summary.get(
                "output_parse_ok_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_results_output_parse_error_count": int(
            raw_order_compute_phase_validation_results_summary.get(
                "output_parse_error_count"
            )
            or 0
        ),
        "lossless_raw_order_compute_phase_validation_results_output_source_path_count": len(
            raw_order_compute_phase_validation_results_output_source_paths
        ),
        "lossless_raw_order_compute_phase_validation_results_output_source_paths": (
            raw_order_compute_phase_validation_results_output_source_paths
        ),
        "lossless_raw_order_compute_phase_validation_results_source_command_count_total": sum(
            int(item.get("source_command_count") or 0)
            for item in raw_order_compute_phase_validation_results_outputs
        ),
        "lossless_raw_order_compute_phase_validation_results_gpu_heavy_source_command_count_total": sum(
            int(item.get("gpu_heavy_source_command_count") or 0)
            for item in raw_order_compute_phase_validation_results_outputs
        ),
        "lossless_raw_order_compute_phase_validation_results_real_heavy_source_command_count_total": sum(
            int(item.get("output_real_heavy_source_command_count") or 0)
            for item in raw_order_compute_phase_validation_results_outputs
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_contract_ready": bool(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "order_neutral_residual_mitigation_contract_ready"
            )
            or raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "replacement_order_neutral_excess_mitigation_contract_ready"
            )
            or raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "raw_control_order_neutral_residual_contract_ready"
            )
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_contract_id": str(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "contract_id"
            )
            or ""
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_target_group_count": int(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "target_group_count"
            )
            or 0
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_target_group_labels": _unique_strings(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "target_group_labels"
            )
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_target_metric_ids": _unique_strings(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "target_metric_ids"
            )
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_contract_row_count": int(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "contract_row_count"
            )
            or 0
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_requires_heavy": bool(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "requires_heavy"
            )
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_next_heavy_validation_required": bool(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "next_heavy_validation_required"
            )
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_blocked_downstream_probe_ids": _unique_strings(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "blocked_downstream_probe_ids"
            )
        ),
        "lossless_raw_order_jitter_order_neutral_excess_mitigation_next_recommended": str(
            raw_order_jitter_order_neutral_excess_mitigation_contract_summary.get(
                "next_recommended"
            )
            or ""
        ),
        "lossless_p7_product_proof_ready": bool(
            _first_present_value(
                (gate_summary, "p7_product_proof_ready"),
                (gate_evidence, "p7_product_proof_ready"),
            )
        ),
        "lossless_p7_product_proof_passed": bool(
            _first_present_value(
                (gate_summary, "p7_product_proof_passed"),
                (gate_evidence, "p7_product_proof_passed"),
            )
        ),
        "lossless_p7_product_proof_blocker_ids": p7_product_proof_blocker_ids,
        "lossless_p7_product_proof_blocker_count": int(
            _first_present_value(
                (gate_summary, "p7_product_proof_blocker_count"),
                (gate_evidence, "p7_product_proof_blocker_count"),
                default=len(p7_product_proof_blocker_ids),
            )
            or 0
        ),
        "lossless_p7_product_proof_validation_failed_item_ids": (
            p7_product_proof_validation_failed_item_ids
        ),
        "lossless_p7_product_proof_product_blocker_details": (
            p7_product_proof_product_blocker_details
        ),
        "lossless_p7_product_proof_product_blocker_count": int(
            _first_present_value(
                (gate_summary, "p7_product_proof_product_blocker_count"),
                (gate_evidence, "p7_product_proof_product_blocker_count"),
                default=len(p7_product_proof_product_blocker_details),
            )
            or 0
        ),
        "lossless_p7_product_proof_manual_heavy_action_ids": (
            p7_product_proof_manual_heavy_action_ids
        ),
        "lossless_p7_product_proof_failed_item_contract_ready_ids": _unique_strings(
            _first_present_value(
                (gate_summary, "p7_product_proof_failed_item_contract_ready_ids"),
                (gate_evidence, "p7_product_proof_failed_item_contract_ready_ids"),
                default=[],
            )
        ),
        "lossless_p7_product_proof_failed_item_contract_ready_count": int(
            _first_present_value(
                (gate_summary, "p7_product_proof_failed_item_contract_ready_count"),
                (gate_evidence, "p7_product_proof_failed_item_contract_ready_count"),
                default=0,
            )
            or 0
        ),
        "lossless_p7_product_proof_failed_item_contract_missing_ids": _unique_strings(
            _first_present_value(
                (gate_summary, "p7_product_proof_failed_item_contract_missing_ids"),
                (gate_evidence, "p7_product_proof_failed_item_contract_missing_ids"),
                default=[],
            )
        ),
        "lossless_p7_product_proof_failed_item_contract_missing_count": int(
            _first_present_value(
                (gate_summary, "p7_product_proof_failed_item_contract_missing_count"),
                (gate_evidence, "p7_product_proof_failed_item_contract_missing_count"),
                default=0,
            )
            or 0
        ),
        "lossless_p7_product_proof_backward_positive_count": int(
            _first_present_value(
                (gate_summary, "p7_product_proof_backward_positive_count"),
                (gate_evidence, "p7_product_proof_backward_positive_count"),
                default=0,
            )
            or 0
        ),
        "lossless_p7_product_proof_forward_positive_count": int(
            _first_present_value(
                (gate_summary, "p7_product_proof_forward_positive_count"),
                (gate_evidence, "p7_product_proof_forward_positive_count"),
                default=0,
            )
            or 0
        ),
        "lossless_p7_product_proof_gate_keys_open": _unique_strings(
            _first_present_value(
                (gate_summary, "p7_product_proof_gate_keys_open"),
                (gate_evidence, "p7_product_proof_gate_keys_open"),
                default=[],
            )
        ),
        "lossless_p7_product_proof_blocker_resolution_matrix_ready": bool(
            _first_present_value(
                (gate_summary, "p7_product_proof_blocker_resolution_matrix_ready"),
                (gate_evidence, "p7_product_proof_blocker_resolution_matrix_ready"),
            )
        ),
        "lossless_p7_product_proof_remaining_report_only_gap_ids": _unique_strings(
            _first_present_value(
                (gate_summary, "p7_product_proof_remaining_report_only_gap_ids"),
                (gate_evidence, "p7_product_proof_remaining_report_only_gap_ids"),
                default=[],
            )
        ),
        "lossless_p7_product_proof_remaining_report_only_gap_count": int(
            _first_present_value(
                (gate_summary, "p7_product_proof_remaining_report_only_gap_count"),
                (gate_evidence, "p7_product_proof_remaining_report_only_gap_count"),
                default=0,
            )
            or 0
        ),
        "lossless_p7_product_proof_heavy_only_blocker_ids": _unique_strings(
            _first_present_value(
                (gate_summary, "p7_product_proof_heavy_only_blocker_ids"),
                (gate_evidence, "p7_product_proof_heavy_only_blocker_ids"),
                default=[],
            )
        ),
        "lossless_p7_product_proof_heavy_only_blocker_count": int(
            _first_present_value(
                (gate_summary, "p7_product_proof_heavy_only_blocker_count"),
                (gate_evidence, "p7_product_proof_heavy_only_blocker_count"),
                default=0,
            )
            or 0
        ),
        "lossless_p7_product_proof_report_only_closed_failed_item_ids": (
            _unique_strings(
                _first_present_value(
                    (
                        gate_summary,
                        "p7_product_proof_report_only_closed_failed_item_ids",
                    ),
                    (
                        gate_evidence,
                        "p7_product_proof_report_only_closed_failed_item_ids",
                    ),
                    default=[],
                )
            )
        ),
        "lossless_p7_product_proof_report_only_closed_failed_item_count": int(
            _first_present_value(
                (
                    gate_summary,
                    "p7_product_proof_report_only_closed_failed_item_count",
                ),
                (
                    gate_evidence,
                    "p7_product_proof_report_only_closed_failed_item_count",
                ),
                default=0,
            )
            or 0
        ),
        "lossless_p7_product_proof_blocker_resolution_classes": _unique_strings(
            _first_present_value(
                (gate_summary, "p7_product_proof_blocker_resolution_classes"),
                (gate_evidence, "p7_product_proof_blocker_resolution_classes"),
                default=[],
            )
        ),
        "lossless_p7_product_proof_resolution_classes": _unique_strings(
            _first_present_value(
                (gate_summary, "p7_product_proof_resolution_classes"),
                (gate_evidence, "p7_product_proof_resolution_classes"),
                (gate_summary, "p7_product_proof_blocker_resolution_classes"),
                (gate_evidence, "p7_product_proof_blocker_resolution_classes"),
                default=[],
            )
        ),
        **_lossless_p7_product_proof_detail_projection(gate_summary, gate_evidence),
        **_lossless_authorization_approval_packet_projection(
            gate_summary,
            gate_evidence,
        ),
        **_resource_gate_high_risk_source_projection(gate_summary, gate_evidence),
        **_resource_gate_p3_report_only_source_projection(
            gate_summary,
            gate_evidence,
        ),
        **p5_p6_p7_boundary_projection,
        "deferred_research_blocker_count": len(deferred_research_blockers),
        "deferred_full_trainer_blocker_count": len(full_trainer_deferred_blockers),
        "deferred_resource_gate_blocker_count": len(resource_gate_deferred_blockers),
        "deferred_resource_gate_blocker_phase_count": len(
            [
                key
                for key in resource_gate_groups
                if key.startswith("resource_gate_p")
            ]
        ),
        "lossless_product_ready": bool(readiness.get("product_ready")),
        "release_blocker_count": len(blockers),
        "core_release_smoke_covered": core_release_smoke_covered,
        "experimental_claim_gates_closed": experimental_claim_gates_closed,
        "research_artifact_gates_closed": research_artifact_gates_closed,
        "lossless_cache_container_decode_implementation_json_only": bool(
            decode_implementation_status["json_only"]
        ),
        "lossless_cache_container_decode_implementation_manifest_only": bool(
            decode_implementation_status["manifest_only"]
        ),
        "lossless_cache_container_decode_implementation_does_not_run_training": bool(
            decode_implementation_status["does_not_run_training"]
        ),
        "lossless_cache_container_decode_implementation_does_not_run_cuda": bool(
            decode_implementation_status["does_not_run_cuda"]
        ),
        "lossless_cache_container_decode_implementation_does_not_run_nvcomp": bool(
            decode_implementation_status["does_not_run_nvcomp"]
        ),
        "lossless_cache_container_decode_implementation_does_not_run_cache_scan": bool(
            decode_implementation_status["does_not_run_cache_scan"]
        ),
        "lossless_cache_container_decode_implementation_does_not_run_runtime": bool(
            decode_implementation_status["does_not_run_runtime"]
        ),
        "lossless_cache_container_decode_implementation_does_not_mutate_runtime": bool(
            decode_implementation_status["does_not_mutate_runtime"]
        ),
        "lossless_cache_container_decode_implementation_training_path_enabled": bool(
            decode_implementation_status["training_path_enabled"]
        ),
        "lossless_cache_container_decode_implementation_resource_center_allowed": bool(
            decode_implementation_status["resource_center_allowed"]
        ),
        "lossless_cache_container_decode_implementation_resource_center_candidate": bool(
            decode_implementation_status["resource_center_candidate"]
        ),
        "lossless_cache_container_decode_implementation_candidate": bool(
            decode_implementation_status["candidate"]
        ),
        "lossless_cache_container_decode_implementation_default_enabled": bool(
            decode_implementation_status["default_enabled"]
        ),
        "lossless_cache_container_decode_implementation_product_ready": bool(
            decode_implementation_status["product_ready"]
        ),
        "lossless_cache_container_decode_implementation_safe_to_auto_execute": bool(
            decode_implementation_status["safe_to_auto_execute"]
        ),
    }
    summary.update(
        {
            checked: bool(artifact_statuses[key]["checked"])
            for key, _artifact, _blocker, checked, _source in _RESEARCH_ARTIFACT_GATE_SPECS
        }
    )
    summary.update(
        {
            "p4_p6_no_repeat_current_heavy_search": bool(
                p4_p6_action_summary.get("no_repeat_current_heavy_search")
            ),
            "p4_p6_no_heavy_restart_reason": str(
                p4_p6_action_summary.get("no_heavy_restart_reason") or ""
            ),
            "p6_no_repeat_current_cache_scan": bool(
                p6_compute_decode_summary.get("no_repeat_current_cache_scan")
                or p4_p6_action_summary.get("p6_no_repeat_current_cache_scan")
            ),
            "p6_compute_decode_no_heavy_rerun_reason": str(
                p6_compute_decode_summary.get("no_heavy_rerun_reason") or ""
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_cache_container_format_plan",
                cache_container_format_plan_summary,
                cache_container_format_plan_source_files,
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_cache_container_read_path_gate",
                cache_container_read_path_gate_summary,
                cache_container_read_path_gate_source_files,
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_cache_container_admission",
                cache_container_admission_contract_summary,
                cache_container_admission_contract_source_files,
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_cache_container_next_axis",
                cache_container_next_axis_contract_summary,
                cache_container_next_axis_contract_source_files,
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_cache_container_payload_layout_axis",
                cache_container_payload_layout_axis_contract_summary,
                cache_container_payload_layout_axis_contract_source_files,
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_cache_container_baseline_comparison",
                cache_container_baseline_comparison_contract_summary,
                cache_container_baseline_comparison_contract_source_files,
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_cache_container_compute_tail_non_regression",
                cache_container_compute_tail_non_regression_contract_summary,
                cache_container_compute_tail_non_regression_contract_source_files,
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_cache_container_decode_implementation",
                cache_container_decode_implementation_contract_summary,
                cache_container_decode_implementation_contract_source_files,
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_cache_container_decode_path_implementation",
                cache_container_decode_path_implementation_plan_summary,
                cache_container_decode_path_implementation_plan_source_files,
            ),
            **_lossless_all_report_artifact_source_projection(
                "lossless_p4_p6_action_plan",
                p4_p6_action_summary,
                _as_dict(
                    artifact_reports.get("p4_p6_action_plan", {}).get(
                        "source_files"
                    )
                ),
            ),
            **_lossless_report_artifact_source_projection(
                "lossless_p5_d3d12_cuda_interop_contract",
                p5_d3d12_cuda_interop_contract_summary,
                p5_d3d12_cuda_interop_contract_source_files,
                ("actionable_plan", "platform_boundary"),
            ),
            **_lossless_report_artifact_source_projection(
                "lossless_p5_d3d12_cuda_functional_interop_probe",
                p5_d3d12_cuda_functional_interop_probe_summary,
                p5_d3d12_cuda_functional_interop_probe_source_files,
                (
                    "actionable_plan",
                    "interop_contract",
                    "platform_boundary",
                    "tensor_view_binding_preflight",
                ),
            ),
            **_lossless_report_artifact_source_projection(
                "lossless_p5_d3d12_cuda_functional_probe_harness_manifest",
                p5_d3d12_cuda_functional_probe_harness_manifest_summary,
                p5_d3d12_cuda_functional_probe_harness_manifest_source_files,
                (
                    "functional_interop_probe",
                    "native_interop",
                    "external_memory",
                    "fence_bridge",
                ),
            ),
            **_lossless_report_artifact_source_projection(
                "lossless_p5_p6_cuda_tensor_view_binding_preflight",
                p5_p6_cuda_tensor_view_binding_preflight_summary,
                p5_p6_cuda_tensor_view_binding_preflight_source_files,
                (
                    "functional_interop_probe",
                    "functional_probe_harness_manifest",
                ),
            ),
            **_lossless_report_artifact_source_projection(
                "lossless_p5_p6_guarded_trainer_runtime_ab_preflight",
                p5_p6_guarded_trainer_runtime_ab_preflight_summary,
                p5_p6_guarded_trainer_runtime_ab_preflight_source_files,
                ("functional_harness", "p6_readiness", "runtime_contract"),
            ),
            **_lossless_report_artifact_source_projection(
                "lossless_p6_compute_decode",
                p6_compute_decode_summary,
                p6_compute_decode_source_files,
                (
                    "interop_contract",
                    "functional_interop_probe",
                    "platform_boundary",
                    "fused_primitive",
                    "real_cache_scan",
                    "broad_cache_scan",
                    "candidate_search_results",
                    "action_plan",
                    "lynx_nocopy_48step_scorecard",
                ),
            ),
            **_lossless_report_artifact_source_projection(
                "lossless_p5_p6_runtime_ab_approval_packet",
                p5_p6_runtime_ab_approval_packet_summary,
                p5_p6_runtime_ab_approval_packet_source_files,
                (
                    "runtime_ab_preflight",
                    "tensor_view_preflight",
                    "native_blueprint",
                    "p6_readiness",
                ),
            ),
            **_lossless_report_artifact_source_projection(
                "lossless_p5_p6_native_blueprint",
                p5_p6_native_blueprint_summary,
                p5_p6_native_blueprint_source_files,
                (
                    "platform_boundary",
                    "interop_contract",
                    "functional_interop_probe",
                    "functional_probe_harness_manifest",
                    "tensor_view_binding_preflight",
                    "runtime_ab_preflight",
                    "p6_readiness",
                    "action_plan",
                ),
            ),
            "lossless_p6_compute_decode_readiness_ready": bool(
                p6_compute_decode_summary.get("p6_compute_decode_readiness_ready")
            ),
            "lossless_p6_compute_decode_runtime_ab_ready": bool(
                p6_compute_decode_summary.get("runtime_ab_ready")
            ),
            "lossless_p6_compute_decode_training_path_enabled": bool(
                p6_compute_decode_summary.get("training_path_enabled")
            ),
            "lossless_p6_compute_decode_resource_center_allowed": bool(
                p6_compute_decode_summary.get("resource_center_allowed")
            ),
            "lossless_p6_compute_decode_resource_center_candidate": bool(
                p6_compute_decode_summary.get("resource_center_candidate")
            ),
            "lossless_p6_compute_decode_candidate": bool(
                p6_compute_decode_summary.get("candidate")
            ),
            "lossless_p6_compute_decode_default_enabled": bool(
                p6_compute_decode_summary.get("default_enabled")
            ),
            "lossless_p6_compute_decode_product_ready": bool(
                p6_compute_decode_summary.get("product_ready")
            ),
            "lossless_p6_compute_decode_safe_to_auto_execute": bool(
                p6_compute_decode_summary.get("safe_to_auto_execute")
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_ready": bool(
                p5_p6_runtime_ab_approval_packet_summary.get(
                    "p5_p6_runtime_ab_approval_packet_ready"
                )
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_runtime_ab_ready": bool(
                p5_p6_runtime_ab_approval_packet_summary.get("runtime_ab_ready")
            ),
            **_approval_candidate_projection(
                "lossless_p5_p6_runtime_ab_approval_packet",
                p5_p6_runtime_ab_approval_packet_summary,
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_execute_allowed_by_default": bool(
                p5_p6_runtime_ab_approval_packet_summary.get(
                    "execute_allowed_by_default"
                )
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_execute_requested": bool(
                p5_p6_runtime_ab_approval_packet_summary.get("execute_requested")
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_requires_explicit_user_approval": bool(
                p5_p6_runtime_ab_approval_packet_summary.get(
                    "requires_explicit_user_approval"
                )
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_requires_explicit_heavy_authorization": bool(
                p5_p6_runtime_ab_approval_packet_summary.get(
                    "requires_explicit_heavy_authorization"
                )
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_manual_heavy_required": bool(
                p5_p6_runtime_ab_approval_packet_summary.get("manual_heavy_required")
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_training_path_enabled": bool(
                p5_p6_runtime_ab_approval_packet_summary.get("training_path_enabled")
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_resource_center_allowed": bool(
                p5_p6_runtime_ab_approval_packet_summary.get(
                    "resource_center_allowed"
                )
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_resource_center_candidate": bool(
                p5_p6_runtime_ab_approval_packet_summary.get(
                    "resource_center_candidate"
                )
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_candidate": bool(
                p5_p6_runtime_ab_approval_packet_summary.get("candidate")
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_default_enabled": bool(
                p5_p6_runtime_ab_approval_packet_summary.get("default_enabled")
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_product_ready": bool(
                p5_p6_runtime_ab_approval_packet_summary.get("product_ready")
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_safe_to_auto_execute": bool(
                p5_p6_runtime_ab_approval_packet_summary.get("safe_to_auto_execute")
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_blocked_precondition_count": int(
                p5_p6_runtime_ab_approval_packet_summary.get(
                    "blocked_precondition_count"
                )
                or 0
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_source_gate_open_count": int(
                p5_p6_runtime_ab_approval_packet_summary.get("source_gate_open_count")
                or 0
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_validation_issue_count": int(
                p5_p6_runtime_ab_approval_packet_summary.get("validation_issue_count")
                or 0
            ),
            "lossless_p5_p6_runtime_ab_approval_packet_next_recommended": str(
                p5_p6_runtime_ab_approval_packet_summary.get("next_recommended") or ""
            ),
            "p7_primary_blocker_detail": str(
                p7_non_heavy_summary.get("primary_blocker_detail") or ""
            ),
            "p7_real_heavy_evidence_output_count": int(
                p7_non_heavy_summary.get(
                    "compute_phase_validation_real_heavy_evidence_output_count"
                )
                or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_ready": bool(
                p7_validation_verdict_summary.get(
                    "raw_order_compute_phase_validation_verdict_ready"
                )
            ),
            "lossless_raw_order_compute_phase_validation_verdict_json_only": bool(
                p7_validation_verdict_report.get("json_only")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_manifest_only": bool(
                p7_validation_verdict_report.get("manifest_only")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_does_not_run_training": bool(
                p7_validation_verdict_report.get("does_not_run_training")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_does_not_run_cuda": bool(
                p7_validation_verdict_report.get("does_not_run_cuda")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_does_not_run_nvcomp": bool(
                p7_validation_verdict_report.get("does_not_run_nvcomp")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_does_not_run_cache_scan": bool(
                p7_validation_verdict_report.get("does_not_run_cache_scan")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_does_not_run_runtime": bool(
                p7_validation_verdict_report.get("does_not_run_runtime")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_does_not_mutate_runtime": bool(
                p7_validation_verdict_report.get("does_not_mutate_runtime")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_execute_requested": bool(
                p7_validation_verdict_report.get("execute_requested")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_source_results_path": str(
                _as_dict(p7_validation_verdict_source_files.get("results")).get(
                    "path"
                )
                or ""
            ),
            "lossless_raw_order_compute_phase_validation_verdict_source_resource_gate_path": str(
                _as_dict(
                    p7_validation_verdict_source_files.get("resource_gate")
                ).get("path")
                or ""
            ),
            "lossless_raw_order_compute_phase_validation_verdict_source_first_release_readiness_path": str(
                _as_dict(
                    p7_validation_verdict_source_files.get(
                        "first_release_readiness"
                    )
                ).get("path")
                or ""
            ),
            "p7_validation_verdict": str(
                p7_validation_verdict_summary.get("verdict") or ""
            ),
            "lossless_raw_order_compute_phase_validation_verdict": str(
                p7_validation_verdict_summary.get("verdict") or ""
            ),
            "p7_validation_product_unlock_ready": bool(
                p7_validation_verdict_summary.get("product_unlock_ready")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_product_unlock_ready": bool(
                p7_validation_verdict_summary.get("product_unlock_ready")
            ),
            "p7_validation_decision": str(
                p7_validation_verdict_summary.get("validation_decision") or ""
            ),
            "lossless_raw_order_compute_phase_validation_verdict_validation_decision": str(
                p7_validation_verdict_summary.get("validation_decision") or ""
            ),
            "lossless_raw_order_compute_phase_validation_verdict_manual_heavy_validation_required": bool(
                p7_validation_verdict_summary.get(
                    "manual_heavy_validation_required"
                )
            ),
            "lossless_raw_order_compute_phase_validation_verdict_selected_next_action_id": str(
                p7_validation_verdict_summary.get("selected_next_action_id") or ""
            ),
            "lossless_raw_order_compute_phase_validation_verdict_validation_item_count": int(
                p7_validation_verdict_summary.get("validation_item_count") or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_real_heavy_evidence_output_count": int(
                p7_validation_verdict_summary.get("real_heavy_evidence_output_count")
                or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_declared_validation_passed_output_count": int(
                p7_validation_verdict_summary.get(
                    "declared_validation_passed_output_count"
                )
                or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_opened_gate_output_count": int(
                p7_validation_verdict_summary.get("opened_gate_output_count") or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_runtime_product_or_resource_gate_open": bool(
                p7_validation_verdict_summary.get(
                    "runtime_product_or_resource_gate_open"
                )
            ),
            "lossless_raw_order_compute_phase_validation_verdict_training_path_enabled": bool(
                p7_validation_verdict_summary.get("training_path_enabled")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_resource_center_allowed": bool(
                p7_validation_verdict_summary.get("resource_center_allowed")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_resource_center_candidate": bool(
                p7_validation_verdict_summary.get("resource_center_candidate")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_candidate": bool(
                p7_validation_verdict_summary.get("candidate")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_default_enabled": bool(
                p7_validation_verdict_summary.get("default_enabled")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_product_ready": bool(
                p7_validation_verdict_summary.get("product_ready")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_safe_to_auto_execute": bool(
                p7_validation_verdict_summary.get("safe_to_auto_execute")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_source_selected_design_id": str(
                p7_validation_verdict_summary.get("source_selected_design_id") or ""
            ),
            "lossless_raw_order_compute_phase_validation_verdict_expected_selected_design_id": str(
                p7_validation_verdict_summary.get("expected_selected_design_id") or ""
            ),
            "lossless_raw_order_compute_phase_validation_verdict_order_neutral_contract_ready": bool(
                p7_validation_verdict_summary.get(
                    "source_order_neutral_excess_contract_ready"
                )
            ),
            "lossless_raw_order_compute_phase_validation_verdict_order_neutral_scope_ready": bool(
                p7_validation_verdict_summary.get("order_neutral_scope_ready")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_order_neutral_contract_id": str(
                p7_validation_verdict_summary.get(
                    "source_order_neutral_excess_contract_id"
                )
                or ""
            ),
            "lossless_raw_order_compute_phase_validation_verdict_order_neutral_contract_row_count": int(
                p7_validation_verdict_summary.get(
                    "order_neutral_residual_contract_row_count"
                )
                or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_order_neutral_target_group_labels": _unique_strings(
                p7_validation_verdict_summary.get(
                    "order_neutral_residual_contract_target_group_labels"
                )
            ),
            "lossless_raw_order_compute_phase_validation_verdict_order_neutral_target_metric_ids": _unique_strings(
                p7_validation_verdict_summary.get(
                    "order_neutral_residual_contract_target_metric_ids"
                )
            ),
            "lossless_raw_order_compute_phase_validation_verdict_required_validation_item_ids": _unique_strings(
                p7_validation_verdict_summary.get("required_validation_item_ids")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_real_heavy_evidence_item_ids": _unique_strings(
                p7_validation_verdict_summary.get("real_heavy_evidence_item_ids")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_contract_only_item_ids": _unique_strings(
                p7_validation_verdict_summary.get("contract_only_item_ids")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_missing_item_ids": _unique_strings(
                p7_validation_verdict_summary.get("missing_item_ids")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_missing_output_count": int(
                p7_validation_verdict_summary.get("missing_output_count") or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_output_parse_error_count": int(
                p7_validation_verdict_summary.get("output_parse_error_count") or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_validation_issue_count": int(
                p7_validation_verdict_summary.get("validation_issue_count") or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_validation_passed_item_count": int(
                p7_validation_verdict_summary.get("validation_passed_item_count")
                or 0
            ),
            "p7_validation_failed_item_count": int(
                p7_validation_verdict_summary.get("validation_failed_item_count")
                or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_validation_failed_item_count": int(
                p7_validation_verdict_summary.get("validation_failed_item_count")
                or 0
            ),
            "p7_validation_inconclusive_item_count": int(
                p7_validation_verdict_summary.get("validation_inconclusive_item_count")
                or 0
            ),
            "lossless_raw_order_compute_phase_validation_verdict_validation_inconclusive_item_count": int(
                p7_validation_verdict_summary.get("validation_inconclusive_item_count")
                or 0
            ),
            "p7_validation_passed_item_ids": _as_list(
                p7_validation_verdict_summary.get("passed_item_ids")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_passed_item_ids": _unique_strings(
                p7_validation_verdict_summary.get("passed_item_ids")
            ),
            "p7_validation_failed_item_ids": _as_list(
                p7_validation_verdict_summary.get("failed_item_ids")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_failed_item_ids": _unique_strings(
                p7_validation_verdict_summary.get("failed_item_ids")
            ),
            "p7_validation_inconclusive_item_ids": _as_list(
                p7_validation_verdict_summary.get("inconclusive_item_ids")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_inconclusive_item_ids": _unique_strings(
                p7_validation_verdict_summary.get("inconclusive_item_ids")
            ),
            "p7_validation_blocker_count": int(
                p7_validation_verdict_summary.get("blocker_count")
                or len(_as_list(p7_validation_verdict_summary.get("blockers")))
            ),
            "lossless_raw_order_compute_phase_validation_verdict_blocker_count": int(
                p7_validation_verdict_summary.get("blocker_count")
                or len(_as_list(p7_validation_verdict_summary.get("blockers")))
            ),
            "lossless_raw_order_compute_phase_validation_verdict_blockers": _unique_strings(
                p7_validation_verdict_summary.get("blockers")
            ),
            "lossless_raw_order_compute_phase_validation_verdict_next_recommended": str(
                p7_validation_verdict_summary.get("next_recommended") or ""
            ),
            "p7_validation_failure_triage_ready": bool(
                p7_validation_failure_triage_summary.get(
                    "p7_validation_failure_triage_ready"
                )
            ),
            "p7_validation_failure_triage_json_only": bool(
                p7_validation_failure_triage_report.get("json_only")
            ),
            "p7_validation_failure_triage_manifest_only": bool(
                p7_validation_failure_triage_report.get("manifest_only")
            ),
            "p7_validation_failure_triage_does_not_run_training": bool(
                p7_validation_failure_triage_report.get("does_not_run_training")
            ),
            "p7_validation_failure_triage_does_not_run_cuda": bool(
                p7_validation_failure_triage_report.get("does_not_run_cuda")
            ),
            "p7_validation_failure_triage_does_not_run_nvcomp": bool(
                p7_validation_failure_triage_report.get("does_not_run_nvcomp")
            ),
            "p7_validation_failure_triage_does_not_run_cache_scan": bool(
                p7_validation_failure_triage_report.get("does_not_run_cache_scan")
            ),
            "p7_validation_failure_triage_does_not_run_runtime": bool(
                p7_validation_failure_triage_report.get("does_not_run_runtime")
            ),
            "p7_validation_failure_triage_does_not_mutate_runtime": bool(
                p7_validation_failure_triage_report.get("does_not_mutate_runtime")
            ),
            "p7_validation_failure_triage_execute_requested": bool(
                p7_validation_failure_triage_report.get("execute_requested")
            ),
            "p7_validation_failure_triage_training_path_enabled": bool(
                p7_validation_failure_triage_summary.get("training_path_enabled")
            ),
            "p7_validation_failure_triage_resource_center_allowed": bool(
                p7_validation_failure_triage_summary.get("resource_center_allowed")
            ),
            "p7_validation_failure_triage_resource_center_candidate": bool(
                p7_validation_failure_triage_summary.get("resource_center_candidate")
            ),
            "p7_validation_failure_triage_candidate": bool(
                p7_validation_failure_triage_summary.get("candidate")
            ),
            "p7_validation_failure_triage_default_enabled": bool(
                p7_validation_failure_triage_summary.get("default_enabled")
            ),
            "p7_validation_failure_triage_product_ready": bool(
                p7_validation_failure_triage_summary.get("product_ready")
            ),
            "p7_validation_failure_triage_safe_to_auto_execute": bool(
                p7_validation_failure_triage_summary.get("safe_to_auto_execute")
            ),
            "p7_validation_failure_triage_source_validation_results_path": str(
                _as_dict(
                    p7_validation_failure_triage_source_files.get(
                        "validation_results"
                    )
                ).get("path")
                or ""
            ),
            "p7_validation_failure_triage_source_guarded_variant_heavy_evidence_path": str(
                _as_dict(
                    p7_validation_failure_triage_source_files.get(
                        "guarded_variant_heavy_evidence"
                    )
                ).get("path")
                or ""
            ),
            "p7_validation_failure_triage_source_raw_order_repeat_phase_attribution_path": str(
                _as_dict(
                    p7_validation_failure_triage_source_files.get(
                        "raw_order_repeat_phase_attribution"
                    )
                ).get("path")
                or ""
            ),
            "p7_validation_failure_triage_source_phase_jitter_product_gate_path": str(
                _as_dict(
                    p7_validation_failure_triage_source_files.get(
                        "phase_jitter_product_gate"
                    )
                ).get("path")
                or ""
            ),
            "p7_validation_failure_triage_source_phase_boundary_path": str(
                _as_dict(
                    p7_validation_failure_triage_source_files.get("phase_boundary")
                ).get("path")
                or ""
            ),
            "p7_validation_failure_triage_source_optimizer_update_tail_attribution_path": str(
                _as_dict(
                    p7_validation_failure_triage_source_files.get(
                        "optimizer_update_tail_attribution"
                    )
                ).get("path")
                or ""
            ),
            **_lossless_p7_artifact_source_projection(
                "p7_validation_failure_triage",
                p7_validation_failure_triage_summary,
                p7_validation_failure_triage_source_files,
                (
                    "validation_results",
                    "guarded_variant_heavy_evidence",
                    "raw_order_repeat_phase_attribution",
                    "phase_jitter_product_gate",
                    "phase_boundary",
                    "optimizer_update_tail_attribution",
                ),
            ),
            "p7_validation_failure_triage_validation_decision": str(
                p7_validation_failure_triage_summary.get("validation_decision") or ""
            ),
            "p7_validation_failure_triage_passed_item_count": int(
                p7_validation_failure_triage_summary.get("passed_item_count") or 0
            ),
            "p7_validation_failure_triage_failed_item_count": int(
                p7_validation_failure_triage_summary.get("failed_item_count") or 0
            ),
            "p7_validation_failure_triage_triaged_failed_item_count": int(
                p7_validation_failure_triage_summary.get("triaged_failed_item_count")
                or 0
            ),
            "p7_validation_failure_triage_untriaged_failed_item_count": int(
                p7_validation_failure_triage_summary.get("untriaged_failed_item_count")
                or 0
            ),
            "p7_validation_failure_triage_failed_item_ids": _unique_strings(
                p7_validation_failure_triage_summary.get("failed_item_ids")
            ),
            "p7_validation_failure_triage_triaged_failed_item_ids": _unique_strings(
                p7_validation_failure_triage_summary.get("triaged_failed_item_ids")
            ),
            "p7_validation_failure_triage_untriaged_failed_item_ids": _unique_strings(
                p7_validation_failure_triage_summary.get("untriaged_failed_item_ids")
            ),
            "p7_validation_failure_triage_blocker_groups": _as_list(
                p7_validation_failure_triage_summary.get("blocker_groups")
            ),
            "p7_validation_failure_triage_blocker_group_count": int(
                p7_validation_failure_triage_summary.get("blocker_group_count") or 0
            ),
            "p7_validation_failure_triage_next_action_ids": _as_list(
                p7_validation_failure_triage_summary.get("next_action_ids")
            ),
            "p7_validation_failure_triage_non_heavy_next_action_count": int(
                p7_validation_failure_triage_summary.get(
                    "non_heavy_next_action_count"
                )
                or 0
            ),
            "p7_validation_failure_triage_manual_heavy_restart_blocked_until_triaged": bool(
                p7_validation_failure_triage_summary.get(
                    "manual_heavy_restart_blocked_until_triaged"
                )
            ),
            "p7_validation_failure_triage_requires_explicit_heavy_authorization": bool(
                p7_validation_failure_triage_summary.get(
                    "requires_explicit_heavy_authorization"
                )
            ),
            "p7_validation_failure_triage_validation_issue_count": int(
                p7_validation_failure_triage_summary.get("validation_issue_count") or 0
            ),
            "p7_validation_failure_triage_row_count": len(
                p7_validation_failure_triage_rows
            ),
            "p7_validation_failure_triage_row_item_ids": _unique_strings(
                [row.get("item_id") for row in p7_validation_failure_triage_rows]
            ),
            "p7_validation_failure_triage_row_statuses": _unique_strings(
                [row.get("status") for row in p7_validation_failure_triage_rows]
            ),
            "p7_validation_failure_triage_row_failure_reasons": _unique_strings(
                [
                    row.get("failure_reason")
                    for row in p7_validation_failure_triage_rows
                ]
            ),
            "p7_validation_failure_triage_row_next_action_kinds": _unique_strings(
                [
                    row.get("next_action_kind")
                    for row in p7_validation_failure_triage_rows
                ]
            ),
            "p7_validation_failure_triage_row_evidence_artifact_ids": (
                p7_validation_failure_triage_row_evidence_artifact_ids
            ),
            "p7_validation_failure_triage_next_recommended": str(
                p7_validation_failure_triage_summary.get("next_recommended") or ""
            ),
            "p7_backward_forward_phase_guard_contract_ready": bool(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "p7_backward_forward_phase_guard_contract_ready"
                )
            ),
            "p7_backward_forward_phase_guard_contract_json_only": bool(
                p7_backward_forward_phase_guard_contract_report.get("json_only")
            ),
            "p7_backward_forward_phase_guard_contract_manifest_only": bool(
                p7_backward_forward_phase_guard_contract_report.get("manifest_only")
            ),
            "p7_backward_forward_phase_guard_contract_does_not_run_training": bool(
                p7_backward_forward_phase_guard_contract_report.get(
                    "does_not_run_training"
                )
            ),
            "p7_backward_forward_phase_guard_contract_does_not_run_cuda": bool(
                p7_backward_forward_phase_guard_contract_report.get(
                    "does_not_run_cuda"
                )
            ),
            "p7_backward_forward_phase_guard_contract_does_not_run_nvcomp": bool(
                p7_backward_forward_phase_guard_contract_report.get(
                    "does_not_run_nvcomp"
                )
            ),
            "p7_backward_forward_phase_guard_contract_does_not_run_cache_scan": bool(
                p7_backward_forward_phase_guard_contract_report.get(
                    "does_not_run_cache_scan"
                )
            ),
            "p7_backward_forward_phase_guard_contract_does_not_run_runtime": bool(
                p7_backward_forward_phase_guard_contract_report.get(
                    "does_not_run_runtime"
                )
            ),
            "p7_backward_forward_phase_guard_contract_does_not_mutate_runtime": bool(
                p7_backward_forward_phase_guard_contract_report.get(
                    "does_not_mutate_runtime"
                )
            ),
            "p7_backward_forward_phase_guard_contract_execute_requested": bool(
                p7_backward_forward_phase_guard_contract_report.get(
                    "execute_requested"
                )
            ),
            "p7_backward_forward_phase_guard_contract_source_raw_order_repeat_phase_attribution_path": str(
                _as_dict(
                    p7_backward_forward_phase_guard_contract_source_files.get(
                        "raw_order_repeat_phase_attribution"
                    )
                ).get("path")
                or ""
            ),
            "p7_backward_forward_phase_guard_contract_source_phase_boundary_path": str(
                _as_dict(
                    p7_backward_forward_phase_guard_contract_source_files.get(
                        "phase_boundary"
                    )
                ).get("path")
                or ""
            ),
            "p7_backward_forward_phase_guard_contract_source_validation_failure_triage_path": str(
                _as_dict(
                    p7_backward_forward_phase_guard_contract_source_files.get(
                        "validation_failure_triage"
                    )
                ).get("path")
                or ""
            ),
            "p7_backward_forward_phase_guard_contract_source_validation_results_path": str(
                _as_dict(
                    p7_backward_forward_phase_guard_contract_source_files.get(
                        "validation_results"
                    )
                ).get("path")
                or ""
            ),
            "p7_backward_forward_phase_guard_contract_source_validation_verdict_path": str(
                _as_dict(
                    p7_backward_forward_phase_guard_contract_source_files.get(
                        "validation_verdict"
                    )
                ).get("path")
                or ""
            ),
            **_lossless_p7_artifact_source_projection(
                "p7_backward_forward_phase_guard_contract",
                p7_backward_forward_phase_guard_contract_summary,
                p7_backward_forward_phase_guard_contract_source_files,
                (
                    "raw_order_repeat_phase_attribution",
                    "phase_boundary",
                    "validation_failure_triage",
                    "validation_results",
                    "validation_verdict",
                ),
            ),
            "p7_backward_forward_phase_guard_guarded_phase_ids": _unique_strings(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "guarded_phase_ids"
                )
            ),
            "p7_backward_forward_phase_guard_backward_positive_count": int(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "backward_positive_count"
                )
                or 0
            ),
            "p7_backward_forward_phase_guard_forward_positive_count": int(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "forward_positive_count"
                )
                or 0
            ),
            "p7_backward_forward_phase_guard_contract_row_count": int(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "contract_row_count"
                )
                or 0
            ),
            "p7_backward_forward_phase_guard_target_labels": _as_list(
                p7_backward_forward_phase_guard_contract_summary.get("target_labels")
            ),
            "p7_backward_forward_phase_guard_target_report_keys": _unique_strings(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "target_report_keys"
                )
            ),
            "p7_backward_forward_phase_guard_boundary_classification": str(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "boundary_classification"
                )
                or ""
            ),
            "p7_backward_forward_phase_guard_sample_overlap_count": int(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "sample_overlap_count"
                )
                or 0
            ),
            "p7_backward_forward_phase_guard_step_overlap_count": int(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "step_overlap_count"
                )
                or 0
            ),
            "p7_backward_forward_phase_guard_triage_next_action_present": bool(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "triage_next_action_present"
                )
            ),
            "p7_backward_forward_phase_guard_validation_failed_item_present": bool(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "validation_failed_item_present"
                )
            ),
            "p7_backward_forward_phase_guard_opened_gate_output_count": int(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "opened_gate_output_count"
                )
                or 0
            ),
            "p7_backward_forward_phase_guard_product_unlock_ready": bool(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "product_unlock_ready"
                )
            ),
            "p7_backward_forward_phase_guard_validation_decision": str(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "validation_decision"
                )
                or ""
            ),
            "p7_backward_forward_phase_guard_requires_manual_heavy_validation": bool(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "requires_manual_heavy_validation"
                )
            ),
            "p7_backward_forward_phase_guard_requires_positive_phase_counts_zero": bool(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "requires_positive_phase_counts_zero"
                )
            ),
            "p7_backward_forward_phase_guard_next_recommended": str(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "next_recommended"
                )
                or ""
            ),
            "p7_backward_forward_phase_guard_validation_issue_count": int(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "validation_issue_count"
                )
                or 0
            ),
            **_lossless_p7_artifact_source_projection(
                "p7_backward_forward_phase_guard_validation_recompute",
                p7_backward_forward_phase_guard_validation_recompute_summary,
                p7_backward_forward_phase_guard_validation_recompute_source_files,
                (
                    "validation_results",
                    "validation_verdict",
                    "backward_forward_phase_guard_contract",
                    "raw_order_repeat_phase_attribution",
                ),
            ),
            "p7_guarded_raw_order_compute_phase_variant_contract_checked": bool(
                artifact_statuses[
                    "p7_guarded_raw_order_compute_phase_variant_contract"
                ]["checked"]
            ),
            "p7_guarded_raw_order_compute_phase_variant_contract_ready": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "p7_guarded_raw_order_compute_phase_variant_contract_ready"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_report_only_contract_ready": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "p7_guarded_raw_order_compute_phase_variant_report_only_contract_ready"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_runtime_product_contract_ready": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "runtime_product_contract_ready"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_runtime_contract_ready": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "runtime_contract_ready"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_runtime_contract_blocked_reasons": _unique_strings(
                p7_guarded_raw_order_contract_summary.get(
                    "runtime_contract_blocked_reasons"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_runtime_contract_activation_blockers": _unique_strings(
                p7_guarded_raw_order_contract_summary.get(
                    "runtime_contract_activation_blockers"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_freshness_ready": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "source_freshness_ready"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_newer_than_contract_count": int(
                p7_guarded_raw_order_contract_summary.get(
                    "source_newer_than_contract_count"
                )
                or 0
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_newer_than_contract_ids": _unique_strings(
                p7_guarded_raw_order_contract_summary.get(
                    "source_newer_than_contract_ids"
                )
            ),
            **{
                f"p7_guarded_raw_order_compute_phase_variant_{suffix}": value
                for suffix, value in _lossless_source_files_digest(
                    p7_guarded_raw_order_contract_source_files
                ).items()
            },
            "p7_guarded_raw_order_compute_phase_variant_generated_after_action_plan": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "guarded_raw_order_contract_generated_after_action_plan"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_generated_after_mitigation_blueprint": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "guarded_raw_order_contract_generated_after_mitigation_blueprint"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_generated_after_runtime_contract": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "guarded_raw_order_contract_generated_after_runtime_contract"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_does_not_run_runtime": bool(
                p7_guarded_raw_order_contract_report.get("does_not_run_runtime")
                or p7_guarded_raw_order_contract_summary.get(
                    "does_not_run_runtime"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_report_only_allowed": bool(
                p7_guarded_raw_order_contract_report.get("report_only_allowed")
                or p7_guarded_raw_order_contract_summary.get(
                    "report_only_allowed"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_declares_validation_passed": bool(
                p7_guarded_raw_order_contract_report.get(
                    "declares_validation_passed"
                )
                or p7_guarded_raw_order_contract_summary.get(
                    "declares_validation_passed"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_safe_to_auto_execute": bool(
                p7_guarded_raw_order_contract_report.get("safe_to_auto_execute")
                or p7_guarded_raw_order_contract_summary.get(
                    "safe_to_auto_execute"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_training_path_enabled": bool(
                p7_guarded_raw_order_contract_report.get("training_path_enabled")
                or p7_guarded_raw_order_contract_summary.get(
                    "training_path_enabled"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_resource_center_allowed": bool(
                p7_guarded_raw_order_contract_report.get(
                    "resource_center_allowed"
                )
                or p7_guarded_raw_order_contract_summary.get(
                    "resource_center_allowed"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_product_ready": bool(
                p7_guarded_raw_order_contract_report.get("product_ready")
                or p7_guarded_raw_order_contract_summary.get("product_ready")
            ),
            "p7_guarded_raw_order_compute_phase_variant_validation_issue_count": int(
                p7_guarded_raw_order_contract_summary.get(
                    "validation_issue_count"
                )
                or 0
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_validation_failure_triage_path": str(
                (
                    p7_guarded_raw_order_contract_source_files.get(
                        "validation_failure_triage"
                    )
                    or {}
                ).get("path")
                or ""
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_validation_results_path": str(
                (
                    p7_guarded_raw_order_contract_source_files.get(
                        "validation_results"
                    )
                    or {}
                ).get("path")
                or ""
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_validation_verdict_path": str(
                (
                    p7_guarded_raw_order_contract_source_files.get(
                        "validation_verdict"
                    )
                    or {}
                ).get("path")
                or ""
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_guarded_variant_heavy_evidence_path": str(
                (
                    p7_guarded_raw_order_contract_source_files.get(
                        "guarded_variant_heavy_evidence"
                    )
                    or {}
                ).get("path")
                or ""
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_phase_jitter_product_gate_path": str(
                (
                    p7_guarded_raw_order_contract_source_files.get(
                        "phase_jitter_product_gate"
                    )
                    or {}
                ).get("path")
                or ""
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_regression_action_plan_path": str(
                (
                    p7_guarded_raw_order_contract_source_files.get(
                        "guarded_variant_regression_action_plan"
                    )
                    or {}
                ).get("path")
                or ""
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_mitigation_blueprint_path": str(
                (
                    p7_guarded_raw_order_contract_source_files.get(
                        "guarded_variant_mitigation_blueprint"
                    )
                    or {}
                ).get("path")
                or ""
            ),
            "p7_guarded_raw_order_compute_phase_variant_source_runtime_contract_path": str(
                (
                    p7_guarded_raw_order_contract_source_files.get(
                        "guarded_variant_runtime_contract"
                    )
                    or {}
                ).get("path")
                or ""
            ),
            "p7_guarded_raw_order_compute_phase_variant_regressed_group_count": int(
                p7_guarded_raw_order_contract_summary.get("regressed_group_count")
                or 0
            ),
            "p7_guarded_raw_order_compute_phase_variant_validation_failed_item_present": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "validation_failed_item_present"
                )
            ),
            "p7_guarded_raw_order_compute_phase_variant_requires_manual_heavy_validation": bool(
                p7_guarded_raw_order_contract_summary.get(
                    "requires_manual_heavy_validation"
                )
            ),
            "guarded_variant_regression_action_plan_checked": bool(
                artifact_statuses["guarded_variant_regression_action_plan"][
                    "checked"
                ]
            ),
            "guarded_variant_regression_action_plan_ready": bool(
                _as_dict(
                    artifact_reports.get(
                        "guarded_variant_regression_action_plan", {}
                    ).get("summary")
                ).get("guarded_variant_regression_action_plan_ready")
            ),
            "guarded_variant_runtime_contract_ready": bool(
                guarded_variant_runtime_contract_summary.get(
                    "guarded_variant_runtime_contract_ready"
                )
            ),
            "guarded_variant_runtime_activation_allowed": bool(
                guarded_variant_runtime_contract_summary.get(
                    "runtime_activation_allowed"
                )
            ),
            "guarded_variant_runtime_selected_unit_count": int(
                guarded_variant_runtime_contract_summary.get("selected_unit_count")
                or 0
            ),
            "guarded_variant_runtime_activation_blockers": _as_list(
                guarded_variant_runtime_contract_summary.get("activation_blockers")
            ),
            **{
                f"guarded_variant_runtime_{suffix}": value
                for suffix, value in _lossless_source_files_digest(
                    guarded_variant_runtime_contract_source_files
                ).items()
            },
            **_lossless_guarded_variant_source_projection(
                gate_summary,
                gate_evidence,
                "p7_guarded_variant_regression",
                "lossless_p7_guarded_variant_regression",
            ),
            **_lossless_guarded_variant_source_projection(
                gate_summary,
                gate_evidence,
                "p7_guarded_variant_mitigation",
                "lossless_p7_guarded_variant_mitigation",
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_optimizer_update_detail_projection(
                    p7_optimizer_unaccounted_tail_isolation_summary,
                    p7_optimizer_unaccounted_tail_isolation_source_files,
                    p7_optimizer_outer_phase_substage_instrumentation_summary,
                    p7_optimizer_outer_phase_substage_instrumentation_source_files,
                    p7_optimizer_outer_phase_substage_profile_implementation_summary,
                    p7_optimizer_outer_phase_substage_profile_implementation_source_files,
                    p7_optimizer_outer_substage_tail_attribution_summary,
                    p7_optimizer_outer_substage_tail_attribution_source_files,
                ).items()
                if key.startswith(
                    (
                        "lossless_p7_optimizer_update_unaccounted_tail_isolation_source",
                        "lossless_p7_optimizer_update_outer_phase_substage_instrumentation_contract_source",
                        "lossless_p7_optimizer_update_outer_phase_substage_profile_implementation_source",
                        "lossless_p7_optimizer_update_outer_substage_tail_attribution_source",
                    )
                )
            },
            "p7_optimizer_update_residual_guard_contract_checked": bool(
                artifact_statuses["p7_optimizer_update_residual_guard_contract"][
                    "checked"
                ]
            ),
            "p7_optimizer_update_residual_guard_contract_ready": bool(
                p7_optimizer_residual_contract_summary.get(
                    "p7_optimizer_update_residual_guard_contract_ready"
                )
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_optimizer_residual_contract_source_projection(
                    p7_optimizer_residual_contract_summary,
                    p7_optimizer_residual_contract_source_files,
                ).items()
            },
            "p7_optimizer_update_residual_guard_recheck_after_nonrepro_checked": bool(
                artifact_statuses[
                    "p7_optimizer_update_residual_guard_recheck_after_nonrepro"
                ]["checked"]
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_optimizer_residual_recheck_projection(
                    p7_optimizer_residual_recheck_summary,
                    p7_optimizer_residual_recheck_source_files,
                ).items()
            },
            "p7_optimizer_update_repeat_positive_control_reclassification_checked": bool(
                artifact_statuses[
                    "p7_optimizer_update_repeat_positive_control_reclassification"
                ]["checked"]
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_optimizer_repeat_reclassification_projection(
                    p7_optimizer_repeat_reclassification_summary,
                    p7_optimizer_repeat_reclassification_source_files,
                ).items()
            },
            "p7_optimizer_update_repeat_positive_control_resolution_checked": bool(
                artifact_statuses[
                    "p7_optimizer_update_repeat_positive_control_resolution"
                ]["checked"]
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_optimizer_repeat_resolution_projection(
                    p7_optimizer_repeat_resolution_summary,
                    p7_optimizer_repeat_resolution_source_files,
                ).items()
            },
            "p7_optimizer_update_repeat_positive_optimizer_internal_resolution_checked": bool(
                artifact_statuses[
                    "p7_optimizer_update_repeat_positive_optimizer_internal_resolution"
                ]["checked"]
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_optimizer_repeat_internal_resolution_projection(
                    p7_optimizer_repeat_internal_resolution_summary,
                    p7_optimizer_repeat_internal_resolution_source_files,
                ).items()
            },
            "p7_optimizer_step_micro_attribution_checked": bool(
                artifact_statuses["p7_optimizer_step_micro_attribution"]["checked"]
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_optimizer_step_micro_attribution_projection(
                    p7_optimizer_step_micro_attribution_summary,
                    p7_optimizer_step_micro_attribution_source_files,
                ).items()
            },
            "p7_optimizer_step_micro_profile_instrumentation_checked": bool(
                artifact_statuses[
                    "p7_optimizer_step_micro_profile_instrumentation"
                ]["checked"]
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_optimizer_step_micro_profile_instrumentation_projection(
                    p7_optimizer_step_micro_profile_instrumentation_summary,
                    p7_optimizer_step_micro_profile_instrumentation_source_files,
                ).items()
            },
            "p7_optimizer_update_residual_guard_validation_recompute_checked": bool(
                artifact_statuses[
                    "p7_optimizer_update_residual_guard_validation_recompute"
                ]["checked"]
            ),
            "p7_optimizer_update_residual_guard_validation_recompute_ready": bool(
                p7_optimizer_residual_validation_recompute_summary.get(
                    "p7_optimizer_update_residual_guard_validation_recompute_ready"
                )
            ),
            "p7_optimizer_update_residual_guard_validation_recompute_verdict_decision": str(
                p7_optimizer_residual_validation_recompute_summary.get(
                    "verdict_decision"
                )
                or ""
            ),
            "p7_optimizer_update_residual_guard_validation_recompute_repeat_optimizer_update_positive_count": int(
                p7_optimizer_residual_validation_recompute_summary.get(
                    "repeat_optimizer_update_positive_count"
                )
                or 0
            ),
            "p7_optimizer_update_residual_guard_validation_recompute_product_proof_still_failed": bool(
                p7_optimizer_residual_validation_recompute_summary.get(
                    "product_proof_still_failed"
                )
            ),
            "p7_optimizer_update_residual_guard_positive_count": int(
                p7_optimizer_residual_contract_summary.get(
                    "repeat_optimizer_update_positive_count"
                )
                or 0
            ),
            "p7_optimizer_update_residual_guard_validation_failed_item_present": bool(
                p7_optimizer_residual_contract_summary.get(
                    "validation_failed_item_present"
                )
            ),
            "p7_optimizer_update_residual_guard_requires_manual_heavy_validation": bool(
                p7_optimizer_residual_contract_summary.get(
                    "requires_manual_heavy_validation"
                )
            ),
            "p7_optimizer_update_residual_guard_internal_breakdown_required": bool(
                p7_optimizer_residual_contract_summary.get(
                    "optimizer_update_internal_breakdown_required"
                )
            ),
            "p7_optimizer_update_residual_guard_internal_breakdown_profiled": bool(
                p7_optimizer_residual_contract_summary.get(
                    "optimizer_update_internal_breakdown_profiled"
                )
            ),
            "p7_optimizer_update_residual_guard_internal_breakdown_missing": bool(
                p7_optimizer_residual_contract_summary.get(
                    "optimizer_update_internal_breakdown_missing"
                )
            ),
            "p7_optimizer_update_residual_guard_internal_breakdown_profiled_row_count": int(
                p7_optimizer_residual_contract_summary.get(
                    "optimizer_update_internal_breakdown_profiled_row_count"
                )
                or 0
            ),
            "p7_optimizer_update_residual_guard_internal_breakdown_target_row_count": int(
                p7_optimizer_residual_contract_summary.get(
                    "optimizer_update_internal_breakdown_target_row_count"
                )
                or 0
            ),
            "p7_optimizer_update_residual_guard_internal_breakdown_target_group_labels": _unique_strings(
                p7_optimizer_residual_contract_summary.get(
                    "optimizer_update_internal_breakdown_target_group_labels"
                )
            ),
            "p7_optimizer_update_residual_guard_internal_breakdown_next_action_id": str(
                p7_optimizer_residual_contract_summary.get(
                    "optimizer_update_internal_breakdown_next_action_id"
                )
                or ""
            ),
            "p7_optimizer_update_residual_guard_internal_breakdown_dominant_positive_source": str(
                p7_optimizer_residual_contract_summary.get(
                    "optimizer_update_internal_breakdown_dominant_positive_source"
                )
                or ""
            ),
            "p7_optimizer_update_residual_guard_internal_breakdown_source_counts": _as_dict(
                p7_optimizer_residual_contract_summary.get(
                    "optimizer_update_internal_breakdown_source_counts"
                )
            ),
            "p7_non_heavy_next_action_checked": bool(
                artifact_statuses["p7_non_heavy_next_action"]["checked"]
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_artifact_source_projection(
                    "lossless_p7_non_heavy_next_action",
                    p7_non_heavy_summary,
                    p7_non_heavy_source_files,
                    (
                        "strict_delta",
                        "phase_boundary",
                        "raw_order_repeat_phase",
                        "guarded_validation",
                        "compute_phase_validation_results",
                        "guarded_variant_heavy_evidence",
                        "phase_jitter_product_gate",
                        "authorization_stage_readiness",
                        "p4_p6_action_plan",
                        "p5_p7_platform_boundary",
                        "cache_container_next_axis_contract",
                    ),
                ).items()
            },
            "p7_failed_item_blocker_resolution_matrix_checked": bool(
                artifact_statuses["p7_failed_item_blocker_resolution_matrix"][
                    "checked"
                ]
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_artifact_source_projection(
                    "lossless_p7_failed_item_blocker",
                    p7_failed_item_blocker_resolution_summary,
                    p7_failed_item_blocker_resolution_source_files,
                    (
                        "validation_results",
                        "validation_verdict",
                        "guarded_variant_heavy_evidence",
                        "backward_forward_phase_guard_contract",
                        "backward_forward_phase_guard_validation_recompute",
                        "backward_forward_phase_guard_manual_heavy_packet",
                        "repeat_phase_attribution",
                        "optimizer_residual_guard_contract",
                        "optimizer_update_tail_attribution",
                        "optimizer_update_unaccounted_tail_isolation",
                        "optimizer_update_outer_phase_substage_contract",
                        "optimizer_update_outer_phase_substage_implementation",
                        "optimizer_update_outer_substage_tail_attribution",
                        "optimizer_update_residual_guard_nonrepro_recheck",
                        "optimizer_update_repeat_positive_control_reclassification",
                        "optimizer_update_repeat_positive_control_resolution",
                        "optimizer_update_repeat_positive_optimizer_internal_resolution",
                        "optimizer_step_micro_attribution",
                        "optimizer_step_micro_profile_instrumentation",
                        "optimizer_residual_guard_validation_recompute",
                        "optimizer_update_tail_validation_manifest",
                        "order_neutral_excess_mitigation_contract",
                        "guarded_variant_manual_heavy_packet",
                        "guarded_variant_manual_heavy_run_manifest",
                        "guarded_variant_cross_domain_evidence_contract",
                    ),
                ).items()
            },
            "p7_order_neutral_residual_contract_scope_contract_checked": bool(
                artifact_statuses[
                    "p7_order_neutral_residual_contract_scope_contract"
                ]["checked"]
            ),
            "p7_order_neutral_residual_contract_scope_contract_ready": bool(
                p7_order_neutral_scope_contract_summary.get(
                    "p7_order_neutral_residual_contract_scope_contract_ready"
                )
            ),
            "p7_order_neutral_residual_contract_scope_classification": str(
                p7_order_neutral_scope_contract_summary.get(
                    "contract_failure_classification"
                )
                or ""
            ),
            "p7_order_neutral_residual_contract_scope_validation_failed_item_present": bool(
                p7_order_neutral_scope_contract_summary.get(
                    "validation_failed_item_present"
                )
            ),
            "p7_order_neutral_residual_contract_scope_requires_manual_heavy_validation": bool(
                p7_order_neutral_scope_contract_summary.get(
                    "requires_manual_heavy_validation"
                )
            ),
            **{
                key.removeprefix("lossless_"): value
                for key, value in _lossless_p7_artifact_source_projection(
                    "lossless_p7_order_neutral_residual_contract_scope_contract",
                    p7_order_neutral_scope_contract_summary,
                    p7_order_neutral_scope_contract_source_files,
                    (
                        "validation_failure_triage",
                        "validation_results",
                        "validation_verdict",
                        "order_neutral_aggregate",
                        "case_order_confounding_resolution_plan",
                        "order_neutral_excess_mitigation_contract",
                        "guarded_compute_phase_validation_manifest",
                    ),
                ).items()
            },
            "guarded_variant_request_adapter_contract_ready": bool(
                guarded_variant_request_adapter_contract_summary.get(
                    "guarded_variant_request_adapter_contract_ready"
                )
            ),
            "guarded_variant_request_adapter_request_schema_exposed_for_audit": bool(
                guarded_variant_request_adapter_contract_summary.get(
                    "request_schema_exposed_for_audit"
                )
            ),
            "guarded_variant_request_activation_denied": bool(
                guarded_variant_request_adapter_contract_summary.get(
                    "request_activation_denied"
                )
            ),
            "guarded_variant_request_adapter_requested_activation_keys": _as_list(
                guarded_variant_request_adapter_contract_summary.get(
                    "requested_activation_keys"
                )
            ),
            "guarded_variant_request_adapter_activation_blockers": _as_list(
                guarded_variant_request_adapter_contract_summary.get(
                    "activation_blockers"
                )
            ),
            "guarded_variant_request_adapter_activation_allowed": bool(
                guarded_variant_request_adapter_contract_summary.get(
                    "request_adapter_activation_allowed"
                )
            ),
            **{
                f"guarded_variant_request_adapter_{suffix}": value
                for suffix, value in _lossless_source_files_digest(
                    guarded_variant_request_adapter_contract_source_files
                ).items()
            },
            "p3_p7_blocker_taxonomy_ready": bool(
                _first_present_value(
                    (gate_summary, "p3_p7_blocker_taxonomy_ready"),
                    (gate_evidence, "p3_p7_blocker_taxonomy_ready"),
                    (readiness_summary, "p3_p7_blocker_taxonomy_ready"),
                    (
                        readiness_summary,
                        "resource_gate_p3_p7_blocker_taxonomy_ready",
                    ),
                )
            ),
            "p3_p7_blocker_taxonomy_activation_denial_matrix_ready": bool(
                _first_present_value(
                    (
                        gate_summary,
                        "p3_p7_blocker_taxonomy_activation_denial_matrix_ready",
                    ),
                    (
                        gate_evidence,
                        "p3_p7_blocker_taxonomy_activation_denial_matrix_ready",
                    ),
                    (
                        readiness_summary,
                        "p3_p7_blocker_taxonomy_activation_denial_matrix_ready",
                    ),
                    (
                        readiness_summary,
                        "resource_gate_p3_p7_blocker_taxonomy_activation_denial_matrix_ready",
                    ),
                )
            ),
            "p3_p7_blocker_taxonomy_activation_denial_unit_count": int(
                _first_present_value(
                    (
                        gate_summary,
                        "p3_p7_blocker_taxonomy_activation_denial_unit_count",
                    ),
                    (
                        gate_evidence,
                        "p3_p7_blocker_taxonomy_activation_denial_unit_count",
                    ),
                    (
                        readiness_summary,
                        "p3_p7_blocker_taxonomy_activation_denial_unit_count",
                    ),
                    (
                        readiness_summary,
                        "resource_gate_p3_p7_blocker_taxonomy_activation_denial_unit_count",
                    ),
                )
                or 0
            ),
            "p3_p7_blocker_taxonomy_gate_keys_open": _as_list(
                _first_present_value(
                    (gate_summary, "p3_p7_blocker_taxonomy_gate_keys_open"),
                    (gate_evidence, "p3_p7_blocker_taxonomy_gate_keys_open"),
                    (readiness_summary, "p3_p7_blocker_taxonomy_gate_keys_open"),
                    (
                        readiness_summary,
                        "resource_gate_p3_p7_blocker_taxonomy_gate_keys_open",
                    ),
                    default=[],
                )
            ),
            "p3_p7_blocker_taxonomy_gate_closure_confirmed": bool(
                _first_present_value(
                    (
                        gate_summary,
                        "p3_p7_blocker_taxonomy_gate_closure_confirmed",
                    ),
                    (
                        gate_evidence,
                        "p3_p7_blocker_taxonomy_gate_closure_confirmed",
                    ),
                    (
                        readiness_summary,
                        "p3_p7_blocker_taxonomy_gate_closure_confirmed",
                    ),
                    (
                        readiness_summary,
                        "resource_gate_p3_p7_blocker_taxonomy_gate_closure_confirmed",
                    ),
                )
            ),
        }
    )
    summary.update(
        _p3_p7_taxonomy_summary_projection(
            gate_summary,
            gate_evidence,
            readiness_summary,
        )
    )
    summary.update(
        _lossless_manifest_contract_summary_projection(
            gate_summary,
            gate_evidence,
            readiness_summary,
        )
    )
    sources = {
        "lossless_resource_center_gate": str(gate_path),
        "lossless_full_trainer_readiness": str(full_trainer_readiness_path),
        "release_smoke": str(release_smoke_path) if release_smoke_path else "",
        "batch1_parity_smoke": str(batch1_parity_smoke_path)
        if batch1_parity_smoke_path
        else "",
        "pipeline_refactor_readiness": str(pipeline_refactor_readiness_path)
        if pipeline_refactor_readiness_path
        else "",
    }
    sources.update(
        {
            source: str(artifact_paths[key]) if artifact_paths[key] else ""
            for key, _artifact, _blocker, _checked, source in _RESEARCH_ARTIFACT_GATE_SPECS
        }
    )
    source_files = _source_files_for_paths(
        {
            "lossless_resource_center_gate": gate_path,
            "lossless_full_trainer_readiness": full_trainer_readiness_path,
            "release_smoke": release_smoke_path,
            "batch1_parity_smoke": batch1_parity_smoke_path,
            "pipeline_refactor_readiness": pipeline_refactor_readiness_path,
            **{
                source: artifact_paths[key]
                for key, _artifact, _blocker, _checked, source in _RESEARCH_ARTIFACT_GATE_SPECS
            },
        }
    )
    summary.update(_source_summary_fields(source_files))
    return {
        "report": FIRST_RELEASE_READINESS_REPORT,
        "ok": True,
        "release_ready": release_ready,
        "release_scope": "first_release_stable_baseline",
        "release_blockers": blockers,
        "deferred_research_blockers": deferred_research_blockers,
        "deferred_research_blocker_groups": deferred_research_blocker_groups,
        "release_validation_todo": release_validation_todo,
        "core_release_smoke": release_smoke_status,
        "batch1_handler_parity_smoke": batch1_parity_status,
        "experimental_claim_gate_evidence": pipeline_refactor_status,
        "research_artifact_gates": artifact_statuses,
        "gated_experimental_features": [
            "batch2/4/8 native multi-batch release claims",
            "TurboCore native-update product path",
            "torch.compile/attention backend generalized acceleration claims",
            "GPU 98/99 utilization claims",
            "lossless LXTB/LXCS/LXFS replacement training path",
            "Resource Center lossless optional dependencies",
            "nvCOMP/GPU decode",
            "DirectStorage/GDS/KvikIO/cuFile",
            "native sparse/bitmask product path",
            "manual GPU-heavy followup runners",
            ".lynx tensor-container research plan",
            ".lynx tensor-container admission contract",
            ".lynx new payload/layout axis contract",
            ".lynx payload/layout axis definition contract",
            ".lynx compute-tail non-regression contract",
            ".lynx decode/container implementation contract",
            ".lynx decode-path implementation plan",
            "P4/P6 action-plan research route",
            "P5 D3D12/CUDA interop contract",
            "P5 D3D12/CUDA functional interop probe",
            "P5 D3D12/CUDA functional probe harness manifest",
            "P5/P6 CUDA tensor view binding preflight",
            "P5/P6 guarded trainer runtime A/B preflight",
            "P5/P6 runtime A/B approval packet",
            "P6 D3D12/CUDA compute-decode readiness",
            "P5/P6 native implementation blueprint",
            "P3/P7 phase-jitter product gate",
            "P3/P7 phase-jitter control-adjusted delta",
            "P3/P7 phase-jitter phase attribution",
            "P3/P7 phase-jitter optimizer-update plan",
            "P3/P7 phase-jitter optimizer-update delta",
            "P7 raw/order compute-phase validation runner",
            "P7 raw/order compute-phase validation playbook",
            "P7 raw/order compute-phase validation results",
            "P7 raw/order compute-phase validation verdict",
            "P7 non-heavy next-action decision",
            "P7 manual-heavy authorization bundle",
            "P7 validation failure triage",
            "P7 failed-item blocker resolution matrix",
            "P7 backward/forward phase guard contract",
            "P7 backward/forward phase guard validation recompute",
            "P7 guarded raw-order compute-phase variant contract",
            "P7 guarded variant cross-domain evidence contract",
            "P7 optimizer-update residual guard contracts",
            "P7 optimizer-update residual validation recompute",
            "P7 optimizer-update outer-substage attribution contracts",
            "P7 order-neutral residual contract scope",
            "P7 guarded variant request-adapter contract",
        ],
        "summary": summary,
        "sources": sources,
        "source_files": source_files,
    }


__all__ = [
    "FIRST_RELEASE_READINESS_REPORT",
    "DEFAULT_GATE_JSON",
    "DEFAULT_FULL_TRAINER_READINESS_JSON",
    "DEFAULT_RELEASE_SMOKE_JSON",
    "DEFAULT_BATCH1_PARITY_SMOKE_JSON",
    "DEFAULT_PIPELINE_REFACTOR_READINESS_JSON",
    "DEFAULT_CACHE_CONTAINER_PLAN_JSON",
    "DEFAULT_CACHE_CONTAINER_READ_PATH_GATE_JSON",
    "DEFAULT_CACHE_CONTAINER_ADMISSION_CONTRACT_JSON",
    "DEFAULT_CACHE_CONTAINER_NEXT_AXIS_CONTRACT_JSON",
    "DEFAULT_CACHE_CONTAINER_PAYLOAD_LAYOUT_AXIS_CONTRACT_JSON",
    "DEFAULT_CACHE_CONTAINER_BASELINE_COMPARISON_CONTRACT_JSON",
    "DEFAULT_CACHE_CONTAINER_COMPUTE_TAIL_NON_REGRESSION_CONTRACT_JSON",
    "DEFAULT_CACHE_CONTAINER_DECODE_IMPLEMENTATION_CONTRACT_JSON",
    "DEFAULT_CACHE_CONTAINER_DECODE_PATH_IMPLEMENTATION_PLAN_JSON",
    "DEFAULT_P4_P6_ACTION_PLAN_JSON",
    "DEFAULT_P5_D3D12_CUDA_INTEROP_CONTRACT_JSON",
    "DEFAULT_P5_D3D12_CUDA_FUNCTIONAL_INTEROP_PROBE_JSON",
    "DEFAULT_P5_D3D12_CUDA_FUNCTIONAL_PROBE_HARNESS_MANIFEST_JSON",
    "DEFAULT_P5_P6_CUDA_TENSOR_VIEW_BINDING_PREFLIGHT_JSON",
    "DEFAULT_P5_P6_GUARDED_TRAINER_RUNTIME_AB_PREFLIGHT_JSON",
    "DEFAULT_P5_P6_RUNTIME_AB_APPROVAL_PACKET_JSON",
    "DEFAULT_P6_COMPUTE_DECODE_READINESS_JSON",
    "DEFAULT_P5_P6_NATIVE_BLUEPRINT_JSON",
    "DEFAULT_PHASE_JITTER_PRODUCT_GATE_JSON",
    "DEFAULT_PHASE_JITTER_CONTROL_DELTA_JSON",
    "DEFAULT_PHASE_JITTER_PHASE_ATTRIBUTION_JSON",
    "DEFAULT_PHASE_JITTER_OPTIMIZER_UPDATE_PLAN_JSON",
    "DEFAULT_PHASE_JITTER_OPTIMIZER_UPDATE_DELTA_JSON",
    "DEFAULT_PHASE_JITTER_OPTIMIZER_UPDATE_REPEAT_PLAN_JSON",
    "DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_RUN_MANIFEST_JSON",
    "DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_PLAYBOOK_JSON",
    "DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_RESULTS_JSON",
    "DEFAULT_RAW_ORDER_COMPUTE_PHASE_VALIDATION_VERDICT_JSON",
    "DEFAULT_P7_NON_HEAVY_NEXT_ACTION_JSON",
    "DEFAULT_P7_MANUAL_HEAVY_AUTHORIZATION_BUNDLE_JSON",
    "DEFAULT_GUARDED_VARIANT_REQUEST_ADAPTER_CONTRACT_JSON",
    "resolve_first_release_readiness_path",
    "default_first_release_readiness_paths",
    "build_default_first_release_readiness",
    "build_first_release_readiness",
]
