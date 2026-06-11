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
DEFAULT_P7_BACKWARD_FORWARD_PHASE_GUARD_CONTRACT_JSON = (
    "temp/lossless_p7_backward_forward_phase_guard_contract.json"
)
DEFAULT_P7_GUARDED_RAW_ORDER_COMPUTE_PHASE_VARIANT_CONTRACT_JSON = (
    "temp/lossless_p7_guarded_raw_order_compute_phase_variant_contract.json"
)
DEFAULT_P7_OPTIMIZER_UPDATE_RESIDUAL_GUARD_CONTRACT_JSON = (
    "temp/lossless_p7_optimizer_update_residual_guard_contract.json"
)
DEFAULT_P7_ORDER_NEUTRAL_RESIDUAL_CONTRACT_SCOPE_CONTRACT_JSON = (
    "temp/lossless_p7_order_neutral_residual_contract_scope_contract.json"
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
        "p7_backward_forward_phase_guard_contract",
        "lossless_p7_backward_forward_phase_guard_contract",
        "lossless_p7_backward_forward_phase_guard_contract_gate_open",
        "p7_backward_forward_phase_guard_contract_checked",
        "p7_backward_forward_phase_guard_contract",
    ),
    (
        "p7_guarded_raw_order_compute_phase_variant_contract",
        "lossless_p7_guarded_raw_order_compute_phase_variant_contract",
        "lossless_p7_guarded_raw_order_compute_phase_variant_contract_gate_open",
        "p7_guarded_raw_order_compute_phase_variant_contract_checked",
        "p7_guarded_raw_order_compute_phase_variant_contract",
    ),
    (
        "p7_optimizer_update_residual_guard_contract",
        "lossless_p7_optimizer_update_residual_guard_contract",
        "lossless_p7_optimizer_update_residual_guard_contract_gate_open",
        "p7_optimizer_update_residual_guard_contract_checked",
        "p7_optimizer_update_residual_guard_contract",
    ),
    (
        "p7_order_neutral_residual_contract_scope_contract",
        "lossless_p7_order_neutral_residual_contract_scope_contract",
        "lossless_p7_order_neutral_residual_contract_scope_contract_gate_open",
        "p7_order_neutral_residual_contract_scope_contract_checked",
        "p7_order_neutral_residual_contract_scope_contract",
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
        "p7_backward_forward_phase_guard_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_BACKWARD_FORWARD_PHASE_GUARD_CONTRACT_JSON,
            )
        ),
        "p7_guarded_raw_order_compute_phase_variant_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_GUARDED_RAW_ORDER_COMPUTE_PHASE_VARIANT_CONTRACT_JSON,
            )
        ),
        "p7_optimizer_update_residual_guard_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_OPTIMIZER_UPDATE_RESIDUAL_GUARD_CONTRACT_JSON,
            )
        ),
        "p7_order_neutral_residual_contract_scope_contract_path": (
            resolve_first_release_readiness_path(
                repo_root,
                DEFAULT_P7_ORDER_NEUTRAL_RESIDUAL_CONTRACT_SCOPE_CONTRACT_JSON,
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


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


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
)
_P7_PRODUCT_PROOF_DETAIL_INT_SUFFIXES = (
    "positive_phase_contract_row_count",
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
    "detail_source_validation_issue_count",
    "detail_source_gate_open_count",
    "core_source_audit_mtime_ns",
    "core_source_resource_gate_mtime_ns",
    "core_source_min_mtime_ns",
    "detail_source_newest_mtime_ns",
    "core_source_freshness_issue_count",
    "detail_source_newer_than_core_source_count",
    "detail_source_freshness_issue_count",
    "freshness_issue_count",
)
_P7_PRODUCT_PROOF_DETAIL_STR_SUFFIXES = (
    "positive_phase_boundary_classification",
    "optimizer_update_residual_classification",
    "optimizer_update_residual_breakdown_tail_field",
    "validation_context_verdict",
    "validation_context_decision",
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
    "detail_source_load_error_ids",
    "detail_source_gate_open_ids",
    "detail_source_newer_than_core_source_ids",
)
_P7_PRODUCT_PROOF_DETAIL_DICT_SUFFIXES = (
    "positive_phase_rows_by_phase",
    "positive_phase_rows_by_label",
    "positive_phase_dominant_source_counts",
    "positive_phase_dominant_compute_phase_counts",
    "optimizer_update_residual_compute_phase_counts",
    "guarded_variant_group_blockers",
)
_P7_PRODUCT_PROOF_DETAIL_LIST_SUFFIXES = (
    "positive_phase_top_rows_digest",
    "optimizer_update_residual_top_rows_digest",
    "guarded_variant_activation_denial_matrix_digest",
    "core_source_freshness_issues",
    "detail_source_freshness_issues",
    "freshness_issues",
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
    return output


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
    p7_backward_forward_phase_guard_contract_path: Path | None = None,
    p7_guarded_raw_order_compute_phase_variant_contract_path: Path | None = None,
    p7_optimizer_update_residual_guard_contract_path: Path | None = None,
    p7_order_neutral_residual_contract_scope_contract_path: Path | None = None,
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
        "p7_backward_forward_phase_guard_contract": (
            p7_backward_forward_phase_guard_contract_path
        ),
        "p7_guarded_raw_order_compute_phase_variant_contract": (
            p7_guarded_raw_order_compute_phase_variant_contract_path
        ),
        "p7_optimizer_update_residual_guard_contract": (
            p7_optimizer_update_residual_guard_contract_path
        ),
        "p7_order_neutral_residual_contract_scope_contract": (
            p7_order_neutral_residual_contract_scope_contract_path
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
    p6_compute_decode_summary = _as_dict(
        artifact_reports.get("p6_compute_decode_readiness", {}).get("summary")
    )
    p7_non_heavy_summary = _as_dict(
        artifact_reports.get("p7_non_heavy_next_action", {}).get("summary")
    )
    p7_validation_verdict_summary = _as_dict(
        artifact_reports.get("raw_order_compute_phase_validation_verdict", {}).get(
            "summary"
        )
    )
    p7_validation_failure_triage_summary = _as_dict(
        artifact_reports.get("p7_validation_failure_triage", {}).get("summary")
    )
    p7_backward_forward_phase_guard_contract_summary = _as_dict(
        artifact_reports.get("p7_backward_forward_phase_guard_contract", {}).get(
            "summary"
        )
    )
    p7_guarded_raw_order_contract_summary = _as_dict(
        artifact_reports.get(
            "p7_guarded_raw_order_compute_phase_variant_contract", {}
        ).get("summary")
    )
    p7_optimizer_residual_contract_summary = _as_dict(
        artifact_reports.get("p7_optimizer_update_residual_guard_contract", {}).get(
            "summary"
        )
    )
    p7_order_neutral_scope_contract_summary = _as_dict(
        artifact_reports.get(
            "p7_order_neutral_residual_contract_scope_contract", {}
        ).get("summary")
    )
    guarded_variant_request_adapter_contract_summary = _as_dict(
        artifact_reports.get("guarded_variant_request_adapter_contract", {}).get(
            "summary"
        )
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
        **_lossless_p7_product_proof_detail_projection(gate_summary, gate_evidence),
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
            "p7_primary_blocker_detail": str(
                p7_non_heavy_summary.get("primary_blocker_detail") or ""
            ),
            "p7_real_heavy_evidence_output_count": int(
                p7_non_heavy_summary.get(
                    "compute_phase_validation_real_heavy_evidence_output_count"
                )
                or 0
            ),
            "p7_validation_verdict": str(
                p7_validation_verdict_summary.get("verdict") or ""
            ),
            "p7_validation_product_unlock_ready": bool(
                p7_validation_verdict_summary.get("product_unlock_ready")
            ),
            "p7_validation_decision": str(
                p7_validation_verdict_summary.get("validation_decision") or ""
            ),
            "p7_validation_failed_item_count": int(
                p7_validation_verdict_summary.get("validation_failed_item_count")
                or 0
            ),
            "p7_validation_inconclusive_item_count": int(
                p7_validation_verdict_summary.get("validation_inconclusive_item_count")
                or 0
            ),
            "p7_validation_passed_item_ids": _as_list(
                p7_validation_verdict_summary.get("passed_item_ids")
            ),
            "p7_validation_failed_item_ids": _as_list(
                p7_validation_verdict_summary.get("failed_item_ids")
            ),
            "p7_validation_inconclusive_item_ids": _as_list(
                p7_validation_verdict_summary.get("inconclusive_item_ids")
            ),
            "p7_validation_blocker_count": int(
                p7_validation_verdict_summary.get("blocker_count")
                or len(_as_list(p7_validation_verdict_summary.get("blockers")))
            ),
            "p7_validation_failure_triage_ready": bool(
                p7_validation_failure_triage_summary.get(
                    "p7_validation_failure_triage_ready"
                )
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
            "p7_validation_failure_triage_blocker_groups": _as_list(
                p7_validation_failure_triage_summary.get("blocker_groups")
            ),
            "p7_validation_failure_triage_next_action_ids": _as_list(
                p7_validation_failure_triage_summary.get("next_action_ids")
            ),
            "p7_validation_failure_triage_manual_heavy_restart_blocked_until_triaged": bool(
                p7_validation_failure_triage_summary.get(
                    "manual_heavy_restart_blocked_until_triaged"
                )
            ),
            "p7_validation_failure_triage_next_recommended": str(
                p7_validation_failure_triage_summary.get("next_recommended") or ""
            ),
            "p7_backward_forward_phase_guard_contract_ready": bool(
                p7_backward_forward_phase_guard_contract_summary.get(
                    "p7_backward_forward_phase_guard_contract_ready"
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
            "P7 guarded variant request-adapter contract",
        ],
        "summary": summary,
        "sources": sources,
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
