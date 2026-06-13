"""Profiled TurboCore optimizer smoke suite."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (str(SCRIPT_ROOT), str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from turbocore_optimizer_smoke_scaffold import (  # noqa: E402
    build_scaffold_audit as build_scaffold_audit_payload,
    specialized_individual_turbocore_smoke_file_count,
    turbocore_related_smoke_files,
)
from turbocore_optimizer_smoke_suite_summary import payload_summary, suite_status_summary  # noqa: E402


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
REAL_RECORD_INPUT_MANIFEST_SCHEMA = "turbocore_real_record_gate_input_manifest_v1"
REAL_RECORD_INPUT_MANIFEST_REQUIRED_TRUE_FIELDS = (
    "allow_real_record_gate_execution",
    "real_reviewer_returned_signed_bundle_ready",
    "real_record_inputs_ready",
)
REAL_RECORD_INPUT_MANIFEST_REQUIRED_ZERO_FIELDS = (
    "training_path_enabled_count",
    "native_dispatch_allowed_count",
    "product_native_ready_count",
)
REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_PATH_FIELDS = (
    "record_input_checklist_path",
    "signed_bundle_extraction_record_path",
    "approval_execution_preflight_path",
)
REAL_RECORD_INPUT_MANIFEST_ARTIFACT_SHA256_FIELDS = {
    "record_input_checklist_path": "record_input_checklist_sha256",
    "signed_bundle_extraction_record_path": "signed_bundle_extraction_record_sha256",
    "approval_execution_preflight_path": "approval_execution_preflight_sha256",
}
REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_SHA256_FIELDS = tuple(
    REAL_RECORD_INPUT_MANIFEST_ARTIFACT_SHA256_FIELDS[field]
    for field in REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_PATH_FIELDS
)
REAL_RECORD_INPUT_MANIFEST_ARTIFACT_READY_REQUIREMENTS = {
    "record_input_checklist_path": {
        "exact": {
            "v2_record_input_checklist_phase1_ready_count": 1,
            "v2_record_input_checklist_phase2_ready_count": 1,
            "v2_record_input_checklist_full_ready_count": 1,
            "v2_record_input_checklist_unsafe_claim_count": 0,
        },
        "minimum": {
            "v2_record_input_checklist_support_ready_count": 3,
        },
    },
    "signed_bundle_extraction_record_path": {
        "exact": {
            "v2_signed_bundle_extraction_full_ready_for_record_count": 1,
            "v2_signed_bundle_extraction_source_digest_match_count": 1,
            "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count": 3,
            "v2_signed_bundle_extraction_missing_signature_count": 0,
            "v2_signed_bundle_extraction_unsafe_claim_count": 0,
        },
        "minimum": {},
    },
    "approval_execution_preflight_path": {
        "exact": {
            "v2_approval_preflight_phase1_ready_count": 1,
            "v2_approval_preflight_phase2_ready_count": 1,
            "v2_approval_preflight_full_ready_count": 1,
            "v2_approval_preflight_signed_bundle_valid_count": 1,
            "v2_approval_preflight_extracted_entry_digest_match_count": 3,
            "v2_approval_preflight_support_ready_count": 3,
            "v2_approval_preflight_hard_fail_count": 0,
        },
        "minimum": {},
    },
}
REAL_RECORD_PHASE1_INPUT_MANIFEST_ARTIFACT_READY_REQUIREMENTS = {
    "record_input_checklist_path": {
        "exact": {
            "v2_record_input_checklist_phase1_ready_count": 1,
            "v2_record_input_checklist_unsafe_claim_count": 0,
        },
        "minimum": {
            "v2_record_input_checklist_support_ready_count": 2,
        },
    },
    "signed_bundle_extraction_record_path": {
        "exact": {
            "v2_signed_bundle_extraction_phase1_ready_for_record_count": 1,
            "v2_signed_bundle_extraction_source_digest_match_count": 1,
            "v2_signed_bundle_extraction_phase1_missing_signature_count": 0,
            "v2_signed_bundle_extraction_unsafe_claim_count": 0,
        },
        "minimum": {
            "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count": 2,
        },
    },
    "approval_execution_preflight_path": {
        "exact": {
            "v2_approval_preflight_phase1_ready_count": 1,
            "v2_approval_preflight_signed_bundle_valid_count": 1,
            "v2_approval_preflight_owner_review_ready_count": 1,
            "v2_approval_preflight_product_exposure_ready_count": 1,
            "v2_approval_preflight_hard_fail_count": 0,
            "v2_approval_preflight_runtime_dispatch_ready_count": 0,
            "v2_approval_preflight_native_dispatch_allowed_count": 0,
            "v2_approval_preflight_training_path_enabled_count": 0,
            "v2_approval_preflight_product_native_ready_count": 0,
            "v2_approval_preflight_default_behavior_changed_count": 0,
            "v2_approval_preflight_unsafe_claim_count": 0,
        },
        "minimum": {
            "v2_approval_preflight_extracted_entry_digest_match_count": 2,
            "v2_approval_preflight_support_ready_count": 2,
        },
    },
}
REAL_RECORD_PHASE1_REQUIRED_SIGNATURE_IDS = (
    "owner_release_review",
    "product_exposure_review",
)
REAL_RECORD_PHASE2_DEFERRED_SIGNATURE_IDS = ("owner_release_direction",)
REAL_RECORD_PHASE2_REQUIRED_SIGNATURE_IDS = ("owner_release_direction",)
REAL_RECORD_PHASE1_REQUIRED_INPUT_IDS = (
    "signature_bundle",
    "signed_bundle",
    "owner_release_review",
    "product_exposure_review",
    "training_launch_contract",
    "product_exposure_evidence",
)
REAL_RECORD_PHASE1_DEFAULT_INPUT_PATHS = {
    "signature_bundle": DEFAULT_ARTIFACT_DIR / "turbocore_optimizer_v2_signature_bundle_packet.json",
    "signed_bundle": DEFAULT_ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle.reviewed.json",
    "owner_release_review": DEFAULT_ARTIFACT_DIR / "signed_owner_release_review.json",
    "product_exposure_review": DEFAULT_ARTIFACT_DIR / "signed_product_exposure_review.json",
    "training_launch_contract": DEFAULT_ARTIFACT_DIR / "native_update_training_launch_contract.json",
    "product_exposure_evidence": DEFAULT_ARTIFACT_DIR / "native_update_product_exposure_evidence.json",
}
REAL_RECORD_PHASE1_REVIEWER_HANDOFF_PATH = (
    DEFAULT_ARTIFACT_DIR / "turbocore_optimizer_v2_reviewer_handoff_packet.json"
)
REAL_RECORD_PHASE2_REQUIRED_INPUT_IDS = (
    "release_review_archive",
    "product_exposure_decision",
    "owner_release_direction_packet",
    "signature_bundle",
    "reviewer_handoff",
    "signed_bundle",
    "owner_release_direction",
)
REAL_RECORD_PHASE2_DEFAULT_INPUT_PATHS = {
    "release_review_archive": DEFAULT_ARTIFACT_DIR / "native_update_release_review_archive.json",
    "product_exposure_decision": DEFAULT_ARTIFACT_DIR / "native_update_product_exposure_decision.json",
    "owner_release_direction_packet": DEFAULT_ARTIFACT_DIR / "native_update_owner_release_direction_packet.json",
    "signature_bundle": DEFAULT_ARTIFACT_DIR / "turbocore_optimizer_v2_signature_bundle_packet.json",
    "reviewer_handoff": DEFAULT_ARTIFACT_DIR / "turbocore_optimizer_v2_reviewer_handoff_packet.json",
    "signed_bundle": DEFAULT_ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle.reviewed.json",
    "owner_release_direction": DEFAULT_ARTIFACT_DIR / "signed_owner_release_direction.json",
}


@dataclass(frozen=True)
class SmokeSpec:
    smoke_id: str
    module: str
    tier: str
    description: str


SMOKES: tuple[SmokeSpec, ...] = (
    SmokeSpec("selected_default_off_matrix", "turbocore_plugin_selected_default_off_matrix_scorecard_smoke", "quick", "Selected plugin family default-off matrix"),
    SmokeSpec(
        "runtime_rehearsal_matrix",
        "turbocore_plugin_runtime_rehearsal_matrix_smoke",
        "runtime",
        "Selected plugin runtime/precondition rehearsal matrix",
    ),
    SmokeSpec(
        "factored_custom_optimizer_family_batch",
        "turbocore_factored_custom_optimizer_family_batch_scorecard_smoke",
        "runtime",
        "Built-in factored/custom native canary chain batch",
    ),
    SmokeSpec(
        "stream_lifetime_lease_evidence",
        "turbocore_v5_stream_lifetime_lease_evidence_smoke",
        "runtime",
        "Borrowed-stream lifetime lease evidence remains explicit and default-off",
    ),
    SmokeSpec("exact_adamw_training_loop_dispatch", "turbocore_native_update_training_loop_dispatch_smoke", "runtime", "Exact AdamW TrainingLoop native dispatch and direct-gradient canary"),
    SmokeSpec("exact_adamw_training_loop_recovery", "turbocore_native_update_training_loop_recovery_state_sync_smoke", "runtime", "Exact AdamW TrainingLoop native failure recovery and state sync"),
    SmokeSpec("exact_adamw_training_loop_resume_state_guard", "turbocore_native_update_training_loop_resume_state_guard_smoke", "runtime", "Exact AdamW TrainingLoop resume owner-state guard"),
    SmokeSpec("native_kernel_inventory", "turbocore_optimizer_native_kernel_inventory_scorecard_smoke", "quick", "Selected plugin native kernel/probe inventory"),
    SmokeSpec("optimizer_family_kernel_contract", "turbocore_optimizer_family_kernel_contract_scorecard_smoke", "quick", "Shared optimizer family kernel contract native entrypoint"),
    SmokeSpec("optimizer_coverage_artifact", "turbocore_optimizer_coverage_artifact_scorecard_smoke", "coverage", "Optimizer coverage artifact validation"),
    SmokeSpec(
        "plugin_actual_training_coverage",
        "turbocore_plugin_actual_training_coverage_scorecard_smoke",
        "coverage",
        "Selected plugin actual-training coverage gap matrix",
    ),
    SmokeSpec(
        "plugin_training_loop_matrix",
        "optimizer_plugin_training_loop_matrix_smoke",
        "coverage",
        "Selected plugin optimizer TrainingLoop boundary matrix",
    ),
    SmokeSpec(
        "plugin_adaptivelr_training_loop_canary",
        "turbocore_plugin_adaptivelr_training_loop_canary_scorecard_smoke",
        "coverage",
        "Selected plugin adaptive-LR per-optimizer TrainingLoop native canaries",
    ),
    SmokeSpec(
        "plugin_fused_backward_hook_canary",
        "turbocore_lomo_fused_backward_hook_canary_scorecard_smoke",
        "coverage",
        "Selected plugin LOMO-family fused-backward native hook canaries",
    ),
    SmokeSpec(
        "plugin_bridge_training_loop_canary",
        "turbocore_plugin_bridge_training_loop_canary_scorecard_smoke",
        "coverage",
        "Bridge-created selected plugin TrainingLoop native canaries",
    ),
    SmokeSpec(
        "optimizer_family_follow_up",
        "turbocore_optimizer_family_follow_up_scorecard_smoke",
        "coverage",
        "V2 family follow-up canary and remaining branch summary",
    ),
    SmokeSpec(
        "optimizer_family_follow_up_branch_contract",
        "turbocore_optimizer_family_follow_up_branch_contract_scorecard_smoke",
        "coverage",
        "V2 family follow-up branch ABI gap contracts",
    ),
    SmokeSpec(
        "optimizer_owner_release_hold_package",
        "turbocore_optimizer_owner_release_hold_package_scorecard_smoke",
        "coverage",
        "V2 O2 owner/release hold package",
    ),
    SmokeSpec(
        "adaptive_lr_chain",
        "turbocore_adaptive_lr_chain_scorecard_smoke",
        "coverage",
        "V2 O3 adaptive-LR chain aggregate",
    ),
    SmokeSpec(
        "rmsprop_branch_reference",
        "turbocore_optimizer_rmsprop_branch_reference_scorecard_smoke",
        "coverage",
        "RMSProp centered/momentum reference matrix",
    ),
    SmokeSpec(
        "sgdp_branch_reference",
        "turbocore_optimizer_sgdp_branch_reference_scorecard_smoke",
        "coverage",
        "SGDP projection/decoupled decay reference matrix",
    ),
    SmokeSpec(
        "fromage_branch_reference",
        "turbocore_optimizer_fromage_branch_reference_scorecard_smoke",
        "coverage",
        "Fromage per-tensor norm/p_bound reference matrix",
    ),
    SmokeSpec(
        "rmsprop_launch_config",
        "turbocore_simple_optimizer_rmsprop_launch_config_smoke",
        "coverage",
        "RMSProp centered/momentum launch-config propagation",
    ),
    SmokeSpec(
        "rmsprop_kernel_contract",
        "turbocore_simple_optimizer_rmsprop_kernel_contract_smoke",
        "coverage",
        "RMSProp first-class simple optimizer kernel contract visibility",
    ),
    SmokeSpec(
        "rmsprop_native_abi_guard",
        "turbocore_simple_optimizer_rmsprop_native_abi_guard_smoke",
        "coverage",
        "RMSProp centered/momentum state layout and native ABI guard",
    ),
    SmokeSpec(
        "pid_launch_config",
        "turbocore_simple_optimizer_pid_launch_config_smoke",
        "coverage",
        "PID momentum three-buffer launch-config propagation",
    ),
    SmokeSpec(
        "pid_native_abi_guard",
        "turbocore_simple_optimizer_pid_native_abi_guard_smoke",
        "coverage",
        "PID momentum three-buffer state layout and native ABI guard",
    ),
    SmokeSpec(
        "sgdp_launch_config",
        "turbocore_simple_optimizer_sgdp_launch_config_smoke",
        "coverage",
        "SGDP projection/decoupled decay launch-config propagation",
    ),
    SmokeSpec(
        "sgdp_native_abi_guard",
        "turbocore_simple_optimizer_sgdp_native_abi_guard_smoke",
        "coverage",
        "SGDP projection/decoupled decay state layout and native ABI guard",
    ),
    SmokeSpec(
        "fromage_launch_config",
        "turbocore_simple_optimizer_fromage_launch_config_smoke",
        "coverage",
        "Fromage per-tensor norm/p_bound launch-config propagation",
    ),
    SmokeSpec(
        "fromage_native_abi_guard",
        "turbocore_simple_optimizer_fromage_native_abi_guard_smoke",
        "coverage",
        "Fromage per-tensor norm/p_bound state layout and native ABI guard",
    ),
    SmokeSpec(
        "optimizer_native_readiness_gap",
        "turbocore_optimizer_native_readiness_gap_scorecard_smoke",
        "coverage",
        "Optimizer native readiness gap by selected route family",
    ),
    SmokeSpec(
        "native_update_rollout_review_package",
        "turbocore_native_update_rollout_review_package_smoke",
        "release",
        "Native-update default-off rollout review package",
    ),
    SmokeSpec(
        "native_update_training_dispatch_integration_contract",
        "turbocore_native_update_training_dispatch_integration_contract_smoke",
        "release",
        "Native-update training dispatch integration contract",
    ),
    SmokeSpec(
        "native_update_activation_contract",
        "turbocore_native_update_activation_contract_smoke",
        "release",
        "Native-update activation contract",
    ),
    SmokeSpec(
        "native_update_runtime_execution_contract",
        "turbocore_native_update_runtime_execution_contract_smoke",
        "release",
        "Native-update runtime execution contract",
    ),
    SmokeSpec(
        "native_update_runtime_dispatch_contract",
        "turbocore_native_update_runtime_dispatch_contract_smoke",
        "release",
        "Native-update runtime dispatch contract",
    ),
    SmokeSpec(
        "native_update_native_dispatch_execution_contract",
        "turbocore_native_update_native_dispatch_execution_contract_smoke",
        "release",
        "Native-update native dispatch execution contract",
    ),
    SmokeSpec(
        "native_update_kernel_launch_execution_contract",
        "turbocore_native_update_kernel_launch_execution_contract_smoke",
        "release",
        "Native-update kernel launch execution contract",
    ),
    SmokeSpec(
        "native_update_parity_execution_contract",
        "turbocore_native_update_parity_execution_contract_smoke",
        "release",
        "Native-update parity execution contract",
    ),
    SmokeSpec(
        "native_update_training_step_execution_contract",
        "turbocore_native_update_training_step_execution_contract_smoke",
        "release",
        "Native-update training step execution contract",
    ),
    SmokeSpec(
        "native_update_training_launch_contract",
        "turbocore_native_update_training_launch_contract_smoke",
        "release",
        "Native-update training launch contract",
    ),
    SmokeSpec(
        "phase1_success_review",
        "turbocore_phase1_success_review_smoke",
        "release",
        "TurboCore Phase 1 success review contract",
    ),
    SmokeSpec(
        "optimizer_multitensor_release_hold",
        "turbocore_native_update_optimizer_multitensor_release_hold_scorecard_smoke",
        "release",
        "Multi-tensor native optimizer release-hold evidence",
    ),
    SmokeSpec(
        "representative_performance_importer",
        "turbocore_native_update_representative_performance_importer_smoke",
        "release",
        "Imported representative performance standard artifacts",
    ),
    SmokeSpec(
        "representative_performance_summary",
        "turbocore_native_update_representative_performance_summary_smoke",
        "release",
        "Representative performance evidence summary",
    ),
    SmokeSpec("product_exposure_decision", "turbocore_native_update_product_exposure_decision_smoke", "release", "Default-off product exposure decision gate"),
    SmokeSpec("release_review_package", "turbocore_native_update_release_review_package_smoke", "release", "Native-update release review package"),
    SmokeSpec("owner_release_handoff_summary", "turbocore_native_update_owner_release_handoff_summary_smoke", "release", "Compact owner-release handoff summary"),
    SmokeSpec("owner_release_review_packet", "turbocore_native_update_owner_release_review_packet_smoke", "release", "Owner release-review signable action packet"),
    SmokeSpec("owner_release_review_record", "turbocore_native_update_owner_release_review_record_smoke", "release", "Signed owner release-review record validator"),
    SmokeSpec("product_training_route_binding_preflight", "turbocore_optimizer_product_training_route_binding_preflight_smoke", "release", "Default-off product training-route binding preflight"),
    SmokeSpec("product_route_binding_chain", "turbocore_optimizer_product_route_binding_chain_scorecard_smoke", "release", "V2 O4 product route-binding contract chain"),
    SmokeSpec("product_training_route_binding_training_loop_contract", "turbocore_optimizer_product_training_route_binding_training_loop_contract_smoke", "release", "TrainingLoop post-approval route-binding context contract"),
    SmokeSpec("product_training_route_binding_config_adapter", "turbocore_optimizer_product_training_route_binding_config_adapter_smoke", "release", "TrainingLoop constructor kwargs route-binding adapter"),
    SmokeSpec("product_training_route_binding_product_route_adapter", "turbocore_optimizer_product_training_route_binding_product_route_adapter_smoke", "release", "Product trainer route kwargs adapter coverage"),
    SmokeSpec("product_training_route_binding_runtime_applier", "turbocore_optimizer_product_training_route_binding_runtime_applier_smoke", "release", "entry_train runtime config applier"),
    SmokeSpec("product_training_route_binding_run_local_staging", "turbocore_optimizer_product_training_route_binding_run_local_staging_smoke", "release", "Run-local route-binding artifact staging"),
    SmokeSpec("stable_first_release_scope", "turbocore_optimizer_stable_first_release_scope_smoke", "release", "Stable first-release TurboCore optimizer default-off scope"),
    SmokeSpec("release_review_archive", "turbocore_native_update_release_review_archive_smoke", "release", "Native-update release review archive"),
    SmokeSpec("owner_release_direction_packet", "turbocore_native_update_owner_release_direction_packet_smoke", "release", "Owner release-direction signable packet"),
    SmokeSpec("owner_release_direction_record", "turbocore_native_update_owner_release_direction_record_smoke", "release", "Signed owner release-direction record validator"),
    SmokeSpec("owner_direction_end_to_end", "turbocore_native_update_owner_direction_end_to_end_smoke", "release", "Owner-direction end-to-end count propagation"),
    SmokeSpec("v2_remaining_gate_handoff", "turbocore_optimizer_v2_remaining_gate_handoff_scorecard_smoke", "release", "V2 remaining approval/product-exposure gate handoff"),
    SmokeSpec("v2_signature_bundle", "turbocore_optimizer_v2_signature_bundle_packet_smoke", "release", "V2 signable template bundle for remaining gates"),
    SmokeSpec("v2_reviewer_handoff", "turbocore_optimizer_v2_reviewer_handoff_packet_smoke", "release", "V2 reviewer handoff packet for signed-bundle templates"),
    SmokeSpec("v2_approval_execution", "turbocore_optimizer_v2_approval_execution_plan_smoke", "release", "V2 real-approval execution plan"),
    SmokeSpec("v2_approval_command_audit", "turbocore_optimizer_v2_approval_command_audit_smoke", "release", "V2 approval execution command-chain audit"),
    SmokeSpec("v2_record_input_checklist", "turbocore_optimizer_v2_record_input_checklist_smoke", "release", "V2 approval record-input checklist"),
    SmokeSpec("release_artifact_first_validation", "turbocore_optimizer_release_artifact_first_validation_scorecard_smoke", "release", "V2 O5 artifact-first release validation"),
    SmokeSpec("v2_signed_bundle_extraction", "turbocore_optimizer_v2_signed_bundle_extractor_smoke", "release", "V2 signed-bundle extraction for approval validators"),
    SmokeSpec("v2_approval_execution_preflight", "turbocore_optimizer_v2_approval_execution_preflight_smoke", "release", "V2 approval execution input preflight"),
    SmokeSpec("v2_signed_bundle_intake", "turbocore_optimizer_v2_signed_bundle_intake_record_smoke", "release", "V2 signed template-bundle intake validator"),
    SmokeSpec("v2_signed_bundle_freshness", "turbocore_optimizer_v2_signed_bundle_freshness_guard_smoke", "release", "V2 signed-bundle source freshness guard"),
    SmokeSpec("v2_signed_bundle_roundtrip", "turbocore_optimizer_v2_signed_bundle_roundtrip_scorecard_smoke", "release", "V2 signed-bundle dry-run roundtrip"),
    SmokeSpec("v2_approval_state", "turbocore_optimizer_v2_approval_state_scorecard_smoke", "release", "V2 approval-chain state scorecard"),
    SmokeSpec("v2_completion_audit", "turbocore_optimizer_v2_completion_audit_smoke", "release", "V2 completion readiness audit"),
    SmokeSpec("promotion_scorecard", "turbocore_native_update_promotion_scorecard_smoke", "release", "Native-update promotion scorecard"),
    SmokeSpec("p6_audit", "native_training_performance_p6_audit_smoke", "full", "Native training performance P6 audit"),
)

INCLUDE_GROUPS: dict[str, tuple[str, ...]] = {
    "v2_runtime_boundary_batch": (
        "product_training_route_binding_preflight",
        "product_route_binding_chain",
        "product_training_route_binding_training_loop_contract",
        "product_training_route_binding_config_adapter",
        "product_training_route_binding_product_route_adapter",
        "product_training_route_binding_runtime_applier",
        "product_training_route_binding_run_local_staging",
        "stable_first_release_scope",
    ),
    "v2_release_handoff_batch": (
        "release_review_package",
        "owner_release_handoff_summary",
        "owner_release_review_packet",
        "release_review_archive",
        "owner_release_direction_packet",
        "v2_remaining_gate_handoff",
        "v2_signature_bundle",
        "v2_reviewer_handoff",
        "v2_approval_execution",
        "v2_approval_command_audit",
    ),
    "v2_real_gate_record_batch": (
        "product_exposure_decision",
        "owner_release_review_record",
        "owner_release_direction_record",
        "owner_direction_end_to_end",
    ),
    "v2_real_gate_phase1_record_batch": (
        "product_exposure_decision",
        "owner_release_review_record",
    ),
    "v2_approval_chain": (
        "v2_remaining_gate_handoff",
        "v2_signature_bundle",
        "v2_reviewer_handoff",
        "v2_approval_execution",
        "v2_approval_command_audit",
        "v2_record_input_checklist",
        "v2_signed_bundle_extraction",
        "v2_approval_execution_preflight",
        "v2_signed_bundle_intake",
        "v2_signed_bundle_freshness",
        "v2_signed_bundle_roundtrip",
        "v2_approval_state",
        "release_artifact_first_validation",
        "v2_completion_audit",
    ),
    "v2_freshness_sequence_batch": (
        "v2_approval_command_audit",
        "v2_signed_bundle_freshness",
        "v2_signed_bundle_roundtrip",
        "v2_signed_bundle_extraction",
        "v2_approval_execution_preflight",
        "v2_approval_state",
        "v2_completion_audit",
    ),
    "v2_support_guard_batch": (
        "v2_approval_command_audit",
        "v2_record_input_checklist",
        "v2_approval_state",
        "release_artifact_first_validation",
        "v2_completion_audit",
    ),
}

INCLUDE_GROUP_POLICIES: dict[str, dict[str, Any]] = {
    "v2_real_gate_record_batch": {
        "real_reviewer_input_required": True,
        "approval_artifact_allowed": True,
        "policy": "run_only_after_real_reviewer_signed_bundle_and_record_inputs_are_available",
        "release_evidence_role": "real_record_gate_validation_only",
        "real_record_gate_phase": "full",
        "execution_confirmation_flag": "--allow-real-record-gate",
        "real_record_input_manifest_required": True,
        "real_record_input_manifest_flag": "--real-record-input-manifest",
        "real_record_input_manifest_schema": REAL_RECORD_INPUT_MANIFEST_SCHEMA,
        "real_record_input_manifest_required_true_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_TRUE_FIELDS
        ),
        "real_record_input_manifest_required_zero_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_ZERO_FIELDS
        ),
        "real_record_input_manifest_required_artifact_path_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_PATH_FIELDS
        ),
        "real_record_input_manifest_required_artifact_sha256_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_SHA256_FIELDS
        ),
    },
    "v2_real_gate_phase1_record_batch": {
        "real_reviewer_input_required": True,
        "approval_artifact_allowed": True,
        "policy": "run_only_after_phase1_real_reviewer_signed_bundle_and_record_inputs_are_available",
        "release_evidence_role": "real_record_gate_validation_only",
        "real_record_gate_phase": "phase1",
        "execution_confirmation_flag": "--allow-real-record-gate",
        "real_record_input_manifest_required": True,
        "real_record_input_manifest_flag": "--real-record-input-manifest",
        "real_record_input_manifest_schema": REAL_RECORD_INPUT_MANIFEST_SCHEMA,
        "real_record_input_manifest_required_true_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_TRUE_FIELDS
        ),
        "real_record_input_manifest_required_zero_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_ZERO_FIELDS
        ),
        "real_record_input_manifest_required_artifact_path_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_PATH_FIELDS
        ),
        "real_record_input_manifest_required_artifact_sha256_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_SHA256_FIELDS
        ),
    },
}
REAL_RECORD_GATE_GROUP = "v2_real_gate_record_batch"
REAL_RECORD_PHASE1_GATE_GROUP = "v2_real_gate_phase1_record_batch"
REAL_RECORD_GATE_GROUPS = (REAL_RECORD_GATE_GROUP, REAL_RECORD_PHASE1_GATE_GROUP)
REAL_RECORD_GATE_SMOKE_IDS = frozenset(
    smoke_id
    for group_name in REAL_RECORD_GATE_GROUPS
    for smoke_id in INCLUDE_GROUPS[group_name]
)
REAL_RECORD_GATE_CLI_SMOKE_IDS = frozenset(
    (
        "product_exposure_decision",
        "owner_release_review_record",
        "owner_release_direction_record",
    )
)
REAL_RECORD_FULL_CONTEXT_INPUT_IDS = (
    "owner_release_review",
    "product_exposure_review",
    "owner_release_direction",
    "training_launch_contract",
    "product_exposure_evidence",
    "owner_release_direction_packet",
)
REAL_RECORD_PHASE1_CONTEXT_INPUT_IDS = (
    "owner_release_review",
    "product_exposure_review",
    "training_launch_contract",
    "product_exposure_evidence",
)

PROFILE_TIERS = {
    "quick": ("quick",),
    "runtime": ("quick", "runtime"),
    "coverage": ("quick", "coverage"),
    "release": ("quick", "release"),
    "batch": ("quick", "runtime", "coverage", "release"),
    "full": ("quick", "runtime", "coverage", "release", "full"),
}

PROFILE_GUIDANCE = {
    "quick": "Daily optimizer backend loop; validates default-off, inventory, and native family contract artifacts.",
    "runtime": "Use after optimizer runtime/precondition wiring changes.",
    "coverage": "Use after optimizer family coverage changes; validates existing coverage aggregation artifacts.",
    "release": "Use before owner/release handoff; keeps P6 and heavy coverage rebuilds out of the default path.",
    "batch": "Use after a batch of optimizer work; runs quick, runtime, coverage, and release without P6.",
    "full": "Use only when P6 native-training audit evidence is required.",
}

INDIVIDUAL_SMOKE_POLICY = [
    "Do not run per-family or per-optimizer smoke files during the normal loop.",
    "Start with this suite and a profile: quick, runtime, coverage, release, batch, or full.",
    "Accumulate small guard/summary-field changes and validate them as one module batch.",
    "Use an individual smoke only after the suite identifies a failing smoke_id/module.",
    "Use rebuild entrypoints only when the underlying evidence artifact intentionally needs refresh.",
]


def build_suite(profile: str, *, include: set[str] | None = None, exclude: set[str] | None = None) -> list[SmokeSpec]:
    tiers = set(PROFILE_TIERS[profile])
    selected = [spec for spec in SMOKES if spec.tier in tiers]
    expanded_exclude = _expand_filters(exclude)
    if include:
        profile_smokes = selected
        profile_by_smoke_id = {spec.smoke_id: spec for spec in profile_smokes}
        profile_by_module = {spec.module: spec for spec in profile_smokes}
        selected = []
        seen: set[str] = set()
        for key in _ordered_include_filters(include):
            spec = profile_by_smoke_id.get(key) or profile_by_module.get(key)
            if spec is None or spec.smoke_id in seen:
                continue
            selected.append(spec)
            seen.add(spec.smoke_id)
    if exclude:
        selected = [
            spec for spec in selected if spec.smoke_id not in expanded_exclude and spec.module not in expanded_exclude
        ]
    return selected


def unmatched_include_filters(profile: str, include: set[str] | None = None) -> list[str]:
    if not include:
        return []
    profile_keys = _profile_include_keys(profile)
    profile_group_keys = _profile_include_group_keys(profile)
    return sorted(item for item in include if item not in profile_keys and item not in profile_group_keys)


def build_suite_plan(
    profile: str = "quick",
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    real_record_input_manifest: Path | None = None,
) -> dict[str, Any]:
    selected = build_suite(profile, include=include, exclude=exclude)
    unmatched_includes = unmatched_include_filters(profile, include)
    group_policies = _selected_include_group_policies(profile, include, selected=selected)
    real_record_gate_confirmation_required = _real_record_gate_confirmation_required(group_policies)
    manifest_status = _real_record_input_manifest_status(
        real_record_gate_confirmation_required,
        real_record_input_manifest,
        artifact_ready_requirements=_real_record_input_manifest_artifact_ready_requirements(group_policies),
    )
    real_record_gate_execution_allowed = bool(
        not real_record_gate_confirmation_required or manifest_status["real_record_input_manifest_ready"]
    )
    return {
        "schema_version": 1,
        "suite": "turbocore_optimizer_smoke_suite",
        "roadmap": ROADMAP,
        "profile": profile,
        "guidance": PROFILE_GUIDANCE[profile],
        "include_filters": sorted(include or []),
        "expanded_include_filters": sorted(_expand_filters(include) or []),
        "include_groups": {
            name: list(items)
            for name, items in INCLUDE_GROUPS.items()
            if name in _profile_include_group_keys(profile)
        },
        "include_group_policies": group_policies,
        "real_reviewer_input_required": _real_reviewer_input_required(group_policies),
        "real_reviewer_input_required_groups": _real_reviewer_input_required_groups(group_policies),
        "real_record_gate_confirmation_required": real_record_gate_confirmation_required,
        "real_record_gate_confirmation_flag": "--allow-real-record-gate",
        **manifest_status,
        "real_record_phase1_readiness": build_real_record_phase1_readiness_packet(
            real_record_input_manifest=real_record_input_manifest
        )
        if real_record_gate_confirmation_required
        else {},
        "allow_real_record_gate": False,
        "real_record_gate_execution_allowed": False if real_record_gate_confirmation_required else real_record_gate_execution_allowed,
        "real_record_gate_execution_blocked_by_confirmation_count": 1 if real_record_gate_confirmation_required else 0,
        "real_record_gate_execution_blocked_by_input_manifest_count": 1 if real_record_gate_confirmation_required else 0,
        "real_record_gate_execution_blocked_count": 1 if real_record_gate_confirmation_required else 0,
        "unmatched_include_filters": unmatched_includes,
        "unmatched_include_count": len(unmatched_includes),
        "selected_count": len(selected),
        "selected_smokes": [spec.__dict__ for spec in selected],
        "profiles": {
            name: {
                "tiers": list(tiers),
                "guidance": PROFILE_GUIDANCE[name],
                "smoke_count": len(build_suite(name)),
            }
            for name, tiers in PROFILE_TIERS.items()
        },
        "artifact_policy": [
            *INDIVIDUAL_SMOKE_POLICY,
            "quick/coverage/release are artifact-first by default.",
        ],
        "explicit_rebuild_entrypoints": [
            "turbocore_plugin_selected_default_off_matrix_scorecard_smoke.py --rebuild-artifact",
            "turbocore_optimizer_native_kernel_inventory_scorecard_smoke.py --rebuild-artifact",
            "turbocore_optimizer_coverage_scorecard_smoke.py",
            "turbocore_optimizer_coverage_scorecard_smoke.py --rebuild-artifacts",
        ],
    }


def build_suite_list_report(
    profile: str = "quick",
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    real_record_input_manifest: Path | None = None,
) -> dict[str, Any]:
    selected = build_suite(profile, include=include, exclude=exclude)
    unmatched_includes = unmatched_include_filters(profile, include)
    group_policies = _selected_include_group_policies(profile, include, selected=selected)
    real_record_gate_confirmation_required = _real_record_gate_confirmation_required(group_policies)
    manifest_status = _real_record_input_manifest_status(
        real_record_gate_confirmation_required,
        real_record_input_manifest,
        artifact_ready_requirements=_real_record_input_manifest_artifact_ready_requirements(group_policies),
    )
    real_record_gate_execution_allowed = bool(
        not real_record_gate_confirmation_required or manifest_status["real_record_input_manifest_ready"]
    )
    return {
        "schema_version": 1,
        "suite": "turbocore_optimizer_smoke_suite",
        "roadmap": ROADMAP,
        "profile": profile,
        "mode": "list",
        "include_filters": sorted(include or []),
        "expanded_include_filters": sorted(_expand_filters(include) or []),
        "include_groups": {
            name: list(items)
            for name, items in INCLUDE_GROUPS.items()
            if name in _profile_include_group_keys(profile)
        },
        "include_group_policies": group_policies,
        "real_reviewer_input_required": _real_reviewer_input_required(group_policies),
        "real_reviewer_input_required_groups": _real_reviewer_input_required_groups(group_policies),
        "real_record_gate_confirmation_required": real_record_gate_confirmation_required,
        "real_record_gate_confirmation_flag": "--allow-real-record-gate",
        **manifest_status,
        "real_record_phase1_readiness": build_real_record_phase1_readiness_packet(
            real_record_input_manifest=real_record_input_manifest
        )
        if real_record_gate_confirmation_required
        else {},
        "allow_real_record_gate": False,
        "real_record_gate_execution_allowed": False if real_record_gate_confirmation_required else real_record_gate_execution_allowed,
        "real_record_gate_execution_blocked_by_confirmation_count": 1 if real_record_gate_confirmation_required else 0,
        "real_record_gate_execution_blocked_by_input_manifest_count": 1 if real_record_gate_confirmation_required else 0,
        "real_record_gate_execution_blocked_count": 1 if real_record_gate_confirmation_required else 0,
        "exclude_filters": sorted(exclude or []),
        "unmatched_include_filters": unmatched_includes,
        "unmatched_include_count": len(unmatched_includes),
        "selected_count": len(selected),
        "selected_smokes": [spec.__dict__ for spec in selected],
        "notes": [
            *(
                [
                    "One or more --include filters did not match a smoke_id or module in the selected profile."
                ]
                if unmatched_includes
                else []
            ),
            *(
                [
                    "No smokes were selected; empty selections are not valid release evidence."
                ]
                if not selected
                else []
            ),
            "List output is selection metadata only; use run output with unmatched_include_count=0 and selected_count>0 as release evidence.",
            *(
                [
                    "Selected include group requires real reviewer-returned inputs; synthetic smoke success is not an approval record."
                ]
                if _real_reviewer_input_required(group_policies)
                else []
            ),
            *(
                [
                    "Run output for real record gate groups requires --allow-real-record-gate after real reviewer-returned inputs exist."
                ]
                if _real_record_gate_confirmation_required(group_policies)
                else []
            ),
            *(
                [
                    "Real record gate execution also requires --real-record-input-manifest with ready real signed-bundle and record-input fields."
                ]
                if manifest_status["real_record_input_manifest_required"]
                else []
            ),
        ],
    }


def build_real_record_input_manifest_template() -> dict[str, Any]:
    return {
        "schema": REAL_RECORD_INPUT_MANIFEST_SCHEMA,
        "allow_real_record_gate_execution": False,
        "real_reviewer_returned_signed_bundle_ready": False,
        "real_record_inputs_ready": False,
        "record_input_checklist_path": "temp\\turbocore_optimizer\\turbocore_optimizer_v2_record_input_checklist.json",
        "record_input_checklist_sha256": "",
        "signed_bundle_extraction_record_path": (
            "temp\\turbocore_optimizer\\turbocore_optimizer_v2_signed_bundle_extraction_record.json"
        ),
        "signed_bundle_extraction_record_sha256": "",
        "approval_execution_preflight_path": (
            "temp\\turbocore_optimizer\\turbocore_optimizer_v2_approval_execution_preflight.json"
        ),
        "approval_execution_preflight_sha256": "",
        "training_path_enabled_count": 0,
        "native_dispatch_allowed_count": 0,
        "product_native_ready_count": 0,
        "notes": [
            "Set the three ready fields to true only after real reviewer-returned signed bundle and real record inputs are present.",
            "The checklist, extraction, and preflight paths must exist, match sha256, and report required ready counts.",
            "Keep all default-off counts at 0; nonzero counts keep v2_real_gate_record_batch blocked.",
            "Use with --allow-real-record-gate --real-record-input-manifest only for real record validator execution.",
        ],
    }


def build_real_record_phase1_readiness_packet(
    *,
    real_record_input_manifest: Path | None = None,
) -> dict[str, Any]:
    manifest_status = _real_record_input_manifest_status(
        True,
        real_record_input_manifest,
        artifact_ready_requirements=REAL_RECORD_PHASE1_INPUT_MANIFEST_ARTIFACT_READY_REQUIREMENTS,
    )
    handoff = _read_json_object_if_exists(REAL_RECORD_PHASE1_REVIEWER_HANDOFF_PATH)
    handoff_manifest = handoff.get("phase1_handoff_manifest") if isinstance(handoff.get("phase1_handoff_manifest"), Mapping) else {}
    preflight = _phase1_preflight_payload(real_record_input_manifest)
    preflight_summary = preflight.get("summary") if isinstance(preflight.get("summary"), Mapping) else {}
    input_paths = _phase1_input_paths(preflight)
    input_checks = {
        input_id: _phase1_input_check(input_id, input_paths[input_id])
        for input_id in REAL_RECORD_PHASE1_REQUIRED_INPUT_IDS
    }
    missing_inputs = [input_id for input_id, check in input_checks.items() if check["present"] is not True]
    signature_ids = _strings(handoff_manifest.get("required_signature_ids")) or list(
        REAL_RECORD_PHASE1_REQUIRED_SIGNATURE_IDS
    )
    deferred_signature_ids = _strings(handoff_manifest.get("deferred_signature_ids")) or list(
        REAL_RECORD_PHASE2_DEFERRED_SIGNATURE_IDS
    )
    handoff_phase_ready = bool(
        signature_ids == list(REAL_RECORD_PHASE1_REQUIRED_SIGNATURE_IDS)
        and deferred_signature_ids == list(REAL_RECORD_PHASE2_DEFERRED_SIGNATURE_IDS)
    )
    phase1_preflight_ready = _phase1_preflight_ready(preflight_summary)
    default_off_ready = _phase1_default_off_ready(preflight_summary)
    ready = bool(handoff_phase_ready and not missing_inputs and phase1_preflight_ready and default_off_ready)
    blockers = _dedupe(
        ([] if handoff else ["reviewer_handoff_packet_missing"])
        + ([] if handoff_phase_ready else ["phase1_handoff_manifest_not_ready"])
        + [f"{input_id}_missing" for input_id in missing_inputs]
        + ([] if phase1_preflight_ready else ["phase1_preflight_not_ready"])
        + ([] if default_off_ready else ["phase1_default_off_counts_not_zero"])
    )
    return {
        "schema_version": 1,
        "packet": "turbocore_real_record_phase1_readiness_v1",
        "roadmap_milestone": 75,
        "ready": ready,
        "ready_count": 1 if ready else 0,
        "blockers": blockers,
        "blocker_count": len(blockers),
        "required_signature_ids": list(REAL_RECORD_PHASE1_REQUIRED_SIGNATURE_IDS),
        "deferred_signature_ids": list(REAL_RECORD_PHASE2_DEFERRED_SIGNATURE_IDS),
        "handoff_required_signature_ids": signature_ids,
        "handoff_deferred_signature_ids": deferred_signature_ids,
        "handoff_phase_metadata_ready_count": 1 if handoff_phase_ready else 0,
        "required_input_ids": list(REAL_RECORD_PHASE1_REQUIRED_INPUT_IDS),
        "input_checks": list(input_checks.values()),
        "required_input_count": len(REAL_RECORD_PHASE1_REQUIRED_INPUT_IDS),
        "present_input_count": len(REAL_RECORD_PHASE1_REQUIRED_INPUT_IDS) - len(missing_inputs),
        "missing_input_count": len(missing_inputs),
        "missing_input_ids": missing_inputs,
        "phase1_preflight_ready_count": 1 if phase1_preflight_ready else 0,
        "phase1_default_off_ready_count": 1 if default_off_ready else 0,
        "preflight_summary": dict(preflight_summary),
        "real_record_input_manifest_status": manifest_status,
        "notes": [
            "Milestone 75 only covers phase1 owner-release review and product-exposure records.",
            "Phase2 owner-direction remains deferred and is not required for this packet to become ready.",
            "This packet is read-only; it does not run record validators or enable native dispatch.",
        ],
    }


def build_scaffold_audit(profile: str = "quick") -> dict[str, Any]:
    selected = build_suite(profile)
    report = build_scaffold_audit_payload(
        repo_root=REPO_ROOT,
        script_root=SCRIPT_ROOT,
        roadmap=ROADMAP,
        profile=profile,
        profile_guidance=PROFILE_GUIDANCE[profile],
        selected_smokes=selected,
        all_smokes=SMOKES,
        profiles={
            name: {
                "tiers": list(tiers),
                "guidance": PROFILE_GUIDANCE[name],
                "smoke_count": len(build_suite(name)),
            }
            for name, tiers in PROFILE_TIERS.items()
        },
        include_groups={
            name: list(items)
            for name, items in INCLUDE_GROUPS.items()
            if name in _profile_include_group_keys(profile)
        },
        available_include_groups={name: list(items) for name, items in INCLUDE_GROUPS.items()},
        include_group_policies=INCLUDE_GROUP_POLICIES,
        workflow=INDIVIDUAL_SMOKE_POLICY,
    )
    report["real_record_input_manifest_template_command"] = (
        "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py "
        "--real-record-input-manifest-template"
    )
    report["real_record_input_manifest_template"] = build_real_record_input_manifest_template()
    report["real_record_phase1_readiness_command"] = (
        "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py "
        "--real-record-phase1-readiness --real-record-input-manifest <real_record_input_manifest.json>"
    )
    report["real_record_phase1_readiness"] = build_real_record_phase1_readiness_packet()
    return report



def run_suite(
    profile: str = "quick",
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    continue_on_failure: bool = False,
    write_artifact: bool = True,
    artifact_path: Path | None = None,
    allow_unmatched_include: bool = False,
    allow_empty_selection: bool = False,
    allow_real_record_gate: bool = False,
    real_record_input_manifest: Path | None = None,
) -> dict[str, Any]:
    selected = build_suite(profile, include=include, exclude=exclude)
    unmatched_includes = unmatched_include_filters(profile, include)
    group_policies = _selected_include_group_policies(profile, include, selected=selected)
    real_record_gate_confirmation_required = _real_record_gate_confirmation_required(group_policies)
    manifest_status = _real_record_input_manifest_status(
        real_record_gate_confirmation_required,
        real_record_input_manifest,
        artifact_ready_requirements=_real_record_input_manifest_artifact_ready_requirements(group_policies),
    )
    real_record_gate_execution_allowed = bool(
        not real_record_gate_confirmation_required
        or (allow_real_record_gate and manifest_status["real_record_input_manifest_ready"])
    )
    real_record_gate_execution_blocked = bool(
        real_record_gate_confirmation_required and not real_record_gate_execution_allowed
    )
    manifest_blocked = bool(
        real_record_gate_confirmation_required
        and allow_real_record_gate
        and not manifest_status["real_record_input_manifest_ready"]
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    manifest_context = (
        _real_record_gate_manifest_context(
            real_record_input_manifest,
            required_input_ids=_real_record_manifest_context_input_ids(group_policies),
        )
        if real_record_gate_confirmation_required
        and real_record_gate_execution_allowed
        and any(spec.smoke_id in REAL_RECORD_GATE_CLI_SMOKE_IDS for spec in selected)
        else {}
    )
    if real_record_gate_execution_allowed:
        for spec in selected:
            if spec.smoke_id in REAL_RECORD_GATE_CLI_SMOKE_IDS:
                result = _run_real_record_gate(spec, manifest_context, write_record_artifacts=write_artifact)
            else:
                result = _run_one(spec)
            results.append(result)
            if not result["ok"] and not continue_on_failure:
                break
    failed = [item for item in results if not item["ok"]]
    status_summary = suite_status_summary(
        results,
        roadmap=ROADMAP,
        script_root=SCRIPT_ROOT,
        registered_smoke_count=len(SMOKES),
        turbocore_related_smoke_file_count=len(turbocore_related_smoke_files(SCRIPT_ROOT)),
        specialized_individual_turbocore_smoke_file_count=specialized_individual_turbocore_smoke_file_count(
            SCRIPT_ROOT,
            (spec.module for spec in SMOKES),
        ),
        include_group_policies=group_policies,
    )
    confirmation_blocked = bool(real_record_gate_confirmation_required and not allow_real_record_gate)
    include_filter_ok = bool(allow_unmatched_include or not unmatched_includes)
    selection_ok = bool(allow_empty_selection or selected)
    empty_selection_blocked = bool(not selection_ok and include_filter_ok)
    failed_count = (
        len(failed)
        + (0 if include_filter_ok else 1)
        + (1 if empty_selection_blocked else 0)
        + (1 if real_record_gate_execution_blocked else 0)
    )
    report = {
        "schema_version": 1,
        "suite": "turbocore_optimizer_smoke_suite",
        "profile": profile,
        "roadmap": ROADMAP,
        "ok": (
            include_filter_ok
            and selection_ok
            and real_record_gate_execution_allowed
            and not failed
            and len(results) == len(selected)
        ),
        "include_filters": sorted(include or []),
        "expanded_include_filters": sorted(_expand_filters(include) or []),
        "include_group_policies": group_policies,
        "real_reviewer_input_required": _real_reviewer_input_required(group_policies),
        "real_reviewer_input_required_groups": _real_reviewer_input_required_groups(group_policies),
        "real_record_gate_confirmation_required": real_record_gate_confirmation_required,
        "real_record_gate_confirmation_flag": "--allow-real-record-gate",
        **manifest_status,
        "real_record_phase1_readiness": build_real_record_phase1_readiness_packet(
            real_record_input_manifest=real_record_input_manifest
        )
        if real_record_gate_confirmation_required
        else {},
        "allow_real_record_gate": bool(allow_real_record_gate),
        "real_record_gate_execution_allowed": real_record_gate_execution_allowed,
        "real_record_gate_execution_blocked_by_confirmation_count": 1 if confirmation_blocked else 0,
        "real_record_gate_execution_blocked_by_input_manifest_count": 1 if manifest_blocked else 0,
        "real_record_gate_execution_blocked_count": 1 if real_record_gate_execution_blocked else 0,
        "real_record_gate_manifest_blocked_count": 1 if manifest_blocked else 0,
        "exclude_filters": sorted(exclude or []),
        "unmatched_include_filters": unmatched_includes,
        "unmatched_include_count": len(unmatched_includes),
        "allow_unmatched_include": bool(allow_unmatched_include),
        "allow_empty_selection": bool(allow_empty_selection),
        "include_filter_ok": include_filter_ok,
        "selection_ok": selection_ok,
        "empty_selection_blocked_count": 1 if empty_selection_blocked else 0,
        "selected_count": len(selected),
        "executed_count": len(results),
        "passed_count": sum(1 for item in results if item["ok"]),
        "failed_count": failed_count,
        "skipped_count": len(selected) - len(results),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "status_summary": status_summary,
        "results": results,
        "profile_guidance": PROFILE_GUIDANCE[profile],
        "artifact_policy": [
            *INDIVIDUAL_SMOKE_POLICY,
            "quick/coverage/release are artifact-first by default.",
        ],
        "notes": [
            "Use profile=batch after batched optimizer work; individual smokes follow suite failures only.",
            *(
                [
                    "Selected include group requires real reviewer-returned inputs; synthetic smoke success is not an approval record."
                ]
                if _real_reviewer_input_required(group_policies)
                else []
            ),
            *(
                [
                    "Real record gate execution is blocked until --allow-real-record-gate is passed after real reviewer-returned inputs exist."
                ]
                if real_record_gate_execution_blocked and not allow_real_record_gate
                else []
            ),
            *(
                [
                    "Real record gate execution is blocked until --real-record-input-manifest proves real signed-bundle and record-input readiness."
                ]
                if manifest_blocked
                else []
            ),
            *(
                [
                    "One or more --include filters did not match a smoke_id or module in the selected profile."
                ]
                if unmatched_includes
                else []
            ),
            *(
                [
                    "Use --allow-unmatched-include only for list/probe workflows; unmatched includes are not valid release evidence."
                ]
                if unmatched_includes and not allow_unmatched_include
                else []
            ),
            *(
                [
                    "No smokes were selected for execution; empty run selections are not valid release evidence."
                ]
                if not selected
                else []
            ),
            *(
                [
                    "Use --allow-empty-selection only for list/probe workflows; empty selections are not valid release evidence."
                ]
                if empty_selection_blocked
                else []
            ),
        ],
    }
    if write_artifact:
        path = artifact_path or DEFAULT_ARTIFACT_DIR / f"turbocore_optimizer_smoke_suite_{profile}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["artifact_path"] = str(path)
    return report


def _run_one(spec: SmokeSpec) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        module = importlib.import_module(spec.module)
        payload = _call_smoke(module)
        ok = _payload_ok(payload)
        return {
            "smoke_id": spec.smoke_id,
            "module": spec.module,
            "tier": spec.tier,
            "description": spec.description,
            "ok": ok,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "payload_summary": payload_summary(payload),
        }
    except Exception as exc:
        return {
            "smoke_id": spec.smoke_id,
            "module": spec.module,
            "tier": spec.tier,
            "description": spec.description,
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def _run_real_record_gate(
    spec: SmokeSpec,
    manifest_context: Mapping[str, Any],
    *,
    write_record_artifacts: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if not manifest_context:
            raise ValueError("real_record_manifest_context_missing")
        payload = _call_real_record_gate_cli(
            spec.smoke_id,
            manifest_context,
            write_record_artifacts=write_record_artifacts,
        )
        ok = _real_record_gate_payload_ok(spec.smoke_id, payload)
        return {
            "smoke_id": spec.smoke_id,
            "module": spec.module,
            "tier": spec.tier,
            "description": spec.description,
            "ok": ok,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "real_record_manifest_driven": True,
            "payload_summary": payload_summary(payload),
        }
    except Exception as exc:
        return {
            "smoke_id": spec.smoke_id,
            "module": spec.module,
            "tier": spec.tier,
            "description": spec.description,
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "real_record_manifest_driven": True,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def _call_real_record_gate_cli(
    smoke_id: str,
    manifest_context: Mapping[str, Any],
    *,
    write_record_artifacts: bool,
) -> dict[str, Any]:
    argv = _real_record_gate_cli_args(
        smoke_id,
        manifest_context,
        write_record_artifacts=write_record_artifacts,
    )
    module_name = {
        "product_exposure_decision": "core.turbocore_native_update_product_exposure_decision",
        "owner_release_review_record": "core.turbocore_native_update_owner_release_review_record",
        "owner_release_direction_record": "core.turbocore_native_update_owner_release_direction_record",
    }[smoke_id]
    module = importlib.import_module(module_name)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = int(module.main(argv))
    text = stdout.getvalue().strip()
    payload = json.loads(text) if text else {}
    if not isinstance(payload, dict):
        raise ValueError(f"{smoke_id}_cli_payload_not_object")
    payload["real_record_cli_exit_code"] = code
    payload["real_record_cli_args"] = argv
    if code != 0:
        payload.setdefault("ok", False)
    return payload


def _real_record_gate_cli_args(
    smoke_id: str,
    manifest_context: Mapping[str, Any],
    *,
    write_record_artifacts: bool,
) -> list[str]:
    artifact_dir = str(manifest_context["artifact_dir"])
    preflight = str(manifest_context["approval_execution_preflight_path"])
    input_paths = manifest_context["input_paths"]
    if smoke_id == "product_exposure_decision":
        args = [
            "--training-launch-contract",
            str(input_paths["training_launch_contract"]),
            "--product-exposure-evidence",
            str(input_paths["product_exposure_evidence"]),
            "--product-exposure-review",
            str(input_paths["product_exposure_review"]),
            "--approval-preflight",
            preflight,
        ]
        if write_record_artifacts:
            args.extend(["--out", str(Path(artifact_dir) / "native_update_product_exposure_decision.json")])
        return args
    if smoke_id == "owner_release_review_record":
        args = [
            "--signed-review",
            str(input_paths["owner_release_review"]),
            "--approval-preflight",
            preflight,
            "--artifact-dir",
            artifact_dir,
        ]
        if not write_record_artifacts:
            args.append("--no-artifact")
        return args
    if smoke_id == "owner_release_direction_record":
        args = [
            "--signed-direction",
            str(input_paths["owner_release_direction"]),
            "--owner-direction-packet",
            str(input_paths["owner_release_direction_packet"]),
            "--approval-preflight",
            preflight,
            "--artifact-dir",
            artifact_dir,
        ]
        if not write_record_artifacts:
            args.append("--no-artifact")
        return args
    raise ValueError(f"unsupported_real_record_gate_smoke:{smoke_id}")


def _real_record_gate_payload_ok(smoke_id: str, payload: Mapping[str, Any]) -> bool:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    if payload.get("real_record_cli_exit_code") != 0:
        return False
    if smoke_id == "product_exposure_decision":
        return bool(
            payload.get("product_exposure_decision_recorded") is True
            and summary.get("product_exposure_decision_recorded_count") == 1
            and summary.get("approval_preflight_binding_ready_count") == 1
        )
    if smoke_id == "owner_release_review_record":
        return bool(
            payload.get("release_review_recorded") is True
            and summary.get("release_review_recorded_count") == 1
            and summary.get("approval_preflight_binding_ready_count") == 1
        )
    if smoke_id == "owner_release_direction_record":
        return bool(
            payload.get("owner_release_direction_recorded") is True
            and summary.get("owner_release_direction_recorded_count") == 1
            and summary.get("approval_preflight_binding_ready_count") == 1
        )
    return _payload_ok(payload)


def _expand_filters(filters: set[str] | None) -> set[str]:
    expanded: set[str] = set()
    for item in filters or set():
        expanded.add(item)
        expanded.update(INCLUDE_GROUPS.get(item, ()))
    return expanded


def _ordered_include_filters(filters: set[str] | None) -> list[str]:
    if not filters:
        return []
    ordered: list[str] = []
    for group_name, group_items in INCLUDE_GROUPS.items():
        if group_name in filters:
            ordered.extend(group_items)
    for spec in SMOKES:
        if spec.smoke_id in filters:
            ordered.append(spec.smoke_id)
        if spec.module in filters:
            ordered.append(spec.module)
    return ordered


def _selected_include_group_policies(
    profile: str,
    filters: set[str] | None,
    *,
    selected: Sequence[SmokeSpec] | None = None,
) -> dict[str, dict[str, Any]]:
    profile_groups = _profile_include_group_keys(profile)
    policies = {
        group_name: _include_group_policy(group_name)
        for group_name in sorted(filters or set())
        if group_name in profile_groups
    }
    implicit_group = _selected_real_record_gate_policy_group(filters, selected)
    if implicit_group:
        policies.setdefault(implicit_group, _include_group_policy(implicit_group))
    return policies


def _selected_real_record_gate_smokes(selected: Sequence[SmokeSpec] | None) -> list[str]:
    return [
        spec.smoke_id
        for spec in selected or ()
        if spec.smoke_id in REAL_RECORD_GATE_SMOKE_IDS
    ]


def _selected_real_record_gate_policy_group(
    filters: set[str] | None,
    selected: Sequence[SmokeSpec] | None,
) -> str | None:
    selected_gate_smokes = set(_selected_real_record_gate_smokes(selected))
    if not selected_gate_smokes:
        return None
    if filters and REAL_RECORD_PHASE1_GATE_GROUP in filters:
        return REAL_RECORD_PHASE1_GATE_GROUP
    full_only_smokes = set(INCLUDE_GROUPS[REAL_RECORD_GATE_GROUP]) - set(
        INCLUDE_GROUPS[REAL_RECORD_PHASE1_GATE_GROUP]
    )
    if selected_gate_smokes & full_only_smokes:
        return REAL_RECORD_GATE_GROUP
    return REAL_RECORD_PHASE1_GATE_GROUP


def _include_group_policy(group_name: str) -> dict[str, Any]:
    policy = {
        "real_reviewer_input_required": False,
        "approval_artifact_allowed": False,
        "policy": "module_batch_validation",
        "release_evidence_role": "synthetic_or_read_only_validation",
    }
    policy.update(INCLUDE_GROUP_POLICIES.get(group_name, {}))
    return policy


def _real_reviewer_input_required(group_policies: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(policy.get("real_reviewer_input_required") is True for policy in group_policies.values())


def _real_reviewer_input_required_groups(group_policies: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        group_name
        for group_name, policy in sorted(group_policies.items())
        if policy.get("real_reviewer_input_required") is True
    ]


def _real_record_gate_confirmation_required(group_policies: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        policy.get("release_evidence_role") == "real_record_gate_validation_only"
        and policy.get("real_reviewer_input_required") is True
        for policy in group_policies.values()
    )


def _real_record_gate_phase(group_policies: Mapping[str, Mapping[str, Any]]) -> str:
    phases = {
        str(policy.get("real_record_gate_phase") or "")
        for policy in group_policies.values()
        if policy.get("release_evidence_role") == "real_record_gate_validation_only"
        and policy.get("real_reviewer_input_required") is True
    }
    if "full" in phases:
        return "full"
    if "phase1" in phases:
        return "phase1"
    return ""


def _real_record_input_manifest_artifact_ready_requirements(
    group_policies: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Mapping[str, Mapping[str, int]]]:
    if _real_record_gate_phase(group_policies) == "phase1":
        return REAL_RECORD_PHASE1_INPUT_MANIFEST_ARTIFACT_READY_REQUIREMENTS
    return REAL_RECORD_INPUT_MANIFEST_ARTIFACT_READY_REQUIREMENTS


def _real_record_manifest_context_input_ids(
    group_policies: Mapping[str, Mapping[str, Any]]
) -> tuple[str, ...]:
    if _real_record_gate_phase(group_policies) == "phase1":
        return REAL_RECORD_PHASE1_CONTEXT_INPUT_IDS
    return REAL_RECORD_FULL_CONTEXT_INPUT_IDS


def _real_record_input_manifest_status(
    required: bool,
    manifest_path: Path | None,
    *,
    artifact_ready_requirements: Mapping[str, Mapping[str, Mapping[str, int]]] | None = None,
) -> dict[str, Any]:
    artifact_ready_requirements = artifact_ready_requirements or REAL_RECORD_INPUT_MANIFEST_ARTIFACT_READY_REQUIREMENTS
    path_text = str(manifest_path) if manifest_path is not None else None
    status = {
        "real_record_input_manifest_required": bool(required),
        "real_record_input_manifest_flag": "--real-record-input-manifest",
        "real_record_input_manifest_schema": REAL_RECORD_INPUT_MANIFEST_SCHEMA,
        "real_record_input_manifest_path": path_text,
        "real_record_input_manifest_present": False,
        "real_record_input_manifest_valid": False,
        "real_record_input_manifest_ready": not required,
        "real_record_input_manifest_required_true_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_TRUE_FIELDS
        ),
        "real_record_input_manifest_required_zero_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_ZERO_FIELDS
        ),
        "real_record_input_manifest_required_artifact_path_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_PATH_FIELDS
        ),
        "real_record_input_manifest_required_artifact_sha256_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_SHA256_FIELDS
        ),
        "real_record_input_manifest_artifact_status": {},
        "real_record_input_manifest_missing_fields": list(
            REAL_RECORD_INPUT_MANIFEST_REQUIRED_TRUE_FIELDS
            + REAL_RECORD_INPUT_MANIFEST_REQUIRED_ZERO_FIELDS
            + REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_PATH_FIELDS
            + REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_SHA256_FIELDS
        )
        if required
        else [],
        "real_record_input_manifest_blockers": ["manifest_path_required"] if required else [],
        "real_record_input_manifest_error": None,
    }
    if not required:
        return status
    if manifest_path is None:
        status["real_record_input_manifest_error"] = "manifest_path_required"
        return status
    if not manifest_path.exists():
        status["real_record_input_manifest_blockers"] = ["manifest_path_not_found"]
        status["real_record_input_manifest_error"] = "manifest_path_not_found"
        return status
    status["real_record_input_manifest_present"] = True
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        status["real_record_input_manifest_blockers"] = ["manifest_json_error"]
        status["real_record_input_manifest_error"] = f"manifest_json_error:{type(exc).__name__}"
        return status
    if not isinstance(payload, dict):
        status["real_record_input_manifest_blockers"] = ["manifest_payload_not_object"]
        status["real_record_input_manifest_error"] = "manifest_payload_not_object"
        return status
    status["real_record_input_manifest_valid"] = True
    blockers: list[str] = []
    if payload.get("schema") != REAL_RECORD_INPUT_MANIFEST_SCHEMA:
        blockers.append("schema_mismatch")
    missing_true = [
        field
        for field in REAL_RECORD_INPUT_MANIFEST_REQUIRED_TRUE_FIELDS
        if payload.get(field) is not True
    ]
    missing_zero = [
        field
        for field in REAL_RECORD_INPUT_MANIFEST_REQUIRED_ZERO_FIELDS
        if payload.get(field) != 0
    ]
    missing_paths = [
        field
        for field in REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_PATH_FIELDS
        if not str(payload.get(field) or "").strip()
    ]
    missing_hashes = [
        field
        for field in REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_SHA256_FIELDS
        if not str(payload.get(field) or "").strip()
    ]
    artifact_status = _real_record_input_manifest_artifact_status(
        payload,
        artifact_ready_requirements=artifact_ready_requirements,
    )
    if missing_true:
        blockers.append("required_true_fields_not_ready")
    if missing_zero:
        blockers.append("default_off_counts_not_zero")
    if missing_paths:
        blockers.append("required_artifact_paths_missing")
    if missing_hashes:
        blockers.append("required_artifact_sha256_missing")
    for artifact in artifact_status.values():
        blockers.extend(_strings(artifact.get("blockers")))
    missing = missing_true + missing_zero + missing_paths + missing_hashes
    status["real_record_input_manifest_artifact_status"] = artifact_status
    status["real_record_input_manifest_missing_fields"] = missing
    status["real_record_input_manifest_blockers"] = _dedupe(blockers)
    status["real_record_input_manifest_ready"] = not blockers
    if blockers:
        status["real_record_input_manifest_error"] = ",".join(_dedupe(blockers))
    return status


def _real_record_input_manifest_artifact_status(
    payload: Mapping[str, Any],
    *,
    artifact_ready_requirements: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, Any]]:
    return {
        field: _manifest_artifact_status(
            field,
            payload,
            artifact_ready_requirements=artifact_ready_requirements,
        )
        for field in REAL_RECORD_INPUT_MANIFEST_REQUIRED_ARTIFACT_PATH_FIELDS
        if str(payload.get(field) or "").strip()
    }


def _manifest_artifact_status(
    field: str,
    manifest: Mapping[str, Any],
    *,
    artifact_ready_requirements: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, Any]:
    raw_path = str(manifest.get(field) or "")
    sha256_field = REAL_RECORD_INPUT_MANIFEST_ARTIFACT_SHA256_FIELDS[field]
    expected_sha256 = str(manifest.get(sha256_field) or "").strip().lower()
    path = _manifest_artifact_path(raw_path)
    requirements = artifact_ready_requirements[field]
    exact_requirements = dict(requirements.get("exact", {}))
    minimum_requirements = dict(requirements.get("minimum", {}))
    required_summary_fields = tuple(exact_requirements) + tuple(minimum_requirements)
    blockers: list[str] = []
    status = {
        "path": raw_path,
        "resolved_path": str(path),
        "sha256_field": sha256_field,
        "expected_sha256": expected_sha256,
        "sha256": None,
        "sha256_match": False,
        "present": False,
        "valid_json": False,
        "ready": False,
        "required_summary_ready_fields": list(required_summary_fields),
        "required_summary_exact_fields": exact_requirements,
        "required_summary_minimum_fields": minimum_requirements,
        "missing_summary_ready_fields": list(required_summary_fields),
        "blockers": [],
    }
    if not expected_sha256:
        blockers.append(f"{sha256_field}_missing")
    if not path.exists():
        blockers.append(f"{field}_not_found")
        status["blockers"] = _dedupe(blockers)
        return status
    status["present"] = True
    try:
        artifact_bytes = path.read_bytes()
    except Exception as exc:
        blockers.append(f"{field}_read_error:{type(exc).__name__}")
        status["blockers"] = _dedupe(blockers)
        return status
    sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    status["sha256"] = sha256
    status["sha256_match"] = bool(expected_sha256 and sha256 == expected_sha256)
    if expected_sha256 and sha256 != expected_sha256:
        blockers.append(f"{field}_sha256_mismatch")
    try:
        payload = json.loads(artifact_bytes.decode("utf-8"))
    except Exception as exc:
        blockers.append(f"{field}_json_error:{type(exc).__name__}")
        status["blockers"] = _dedupe(blockers)
        return status
    if not isinstance(payload, Mapping):
        blockers.append(f"{field}_payload_not_object")
        status["blockers"] = _dedupe(blockers)
        return status
    status["valid_json"] = True
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    missing_ready = _missing_manifest_summary_requirements(
        summary,
        exact=exact_requirements,
        minimum=minimum_requirements,
    )
    status["missing_summary_ready_fields"] = missing_ready
    if missing_ready:
        blockers.append(f"{field}_not_full_ready")
    status["blockers"] = _dedupe(blockers)
    status["ready"] = not blockers
    return status


def _missing_manifest_summary_requirements(
    summary: Mapping[str, Any],
    *,
    exact: Mapping[str, int],
    minimum: Mapping[str, int],
) -> list[str]:
    missing = [
        field
        for field, expected in exact.items()
        if summary.get(field) != expected
    ]
    missing.extend(
        field
        for field, expected_minimum in minimum.items()
        if not isinstance(summary.get(field), int) or summary.get(field) < expected_minimum
    )
    return missing


def _real_record_gate_manifest_context(
    manifest_path: Path | None,
    *,
    required_input_ids: Sequence[str],
) -> dict[str, Any]:
    if manifest_path is None:
        return {}
    manifest = _read_json_object(manifest_path)
    preflight_path = _manifest_artifact_path(str(manifest.get("approval_execution_preflight_path") or ""))
    preflight = _read_json_object(preflight_path)
    raw_input_paths = preflight.get("input_paths") if isinstance(preflight.get("input_paths"), Mapping) else {}
    missing = [input_id for input_id in required_input_ids if not str(raw_input_paths.get(input_id) or "").strip()]
    if missing:
        raise ValueError(f"real_record_preflight_input_paths_missing:{','.join(missing)}")
    return {
        "manifest_path": str(manifest_path),
        "record_input_checklist_path": _manifest_artifact_path(
            str(manifest.get("record_input_checklist_path") or "")
        ),
        "signed_bundle_extraction_record_path": _manifest_artifact_path(
            str(manifest.get("signed_bundle_extraction_record_path") or "")
        ),
        "approval_execution_preflight_path": preflight_path,
        "artifact_dir": preflight_path.parent,
        "input_paths": {
            input_id: _manifest_artifact_path(str(raw_input_paths[input_id]))
            for input_id in required_input_ids
        },
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_payload_not_object:{path}")
    return payload


def _read_json_object_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _read_json_object(path)
    except Exception:
        return {}


def _phase1_preflight_payload(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is not None and manifest_path.exists():
        manifest = _read_json_object_if_exists(manifest_path)
        raw_preflight_path = str(manifest.get("approval_execution_preflight_path") or "")
        if raw_preflight_path:
            payload = _read_json_object_if_exists(_manifest_artifact_path(raw_preflight_path))
            if payload:
                return payload
    return _read_json_object_if_exists(
        DEFAULT_ARTIFACT_DIR / "turbocore_optimizer_v2_approval_execution_preflight.json"
    )


def _phase1_input_paths(preflight: Mapping[str, Any]) -> dict[str, Path]:
    raw_input_paths = preflight.get("input_paths") if isinstance(preflight.get("input_paths"), Mapping) else {}
    return {
        input_id: _manifest_artifact_path(str(raw_input_paths.get(input_id) or default_path))
        for input_id, default_path in REAL_RECORD_PHASE1_DEFAULT_INPUT_PATHS.items()
    }


def _phase1_input_check(input_id: str, path: Path) -> dict[str, Any]:
    valid_json = False
    parse_error = None
    if path.exists():
        try:
            valid_json = isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
        except Exception as exc:
            parse_error = type(exc).__name__
    return {
        "id": input_id,
        "path": str(path),
        "present": path.exists(),
        "valid_json": valid_json,
        "parse_error": parse_error,
    }


def _phase1_preflight_ready(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("v2_approval_preflight_phase1_ready_count") == 1
        and summary.get("v2_approval_preflight_signed_bundle_valid_count") == 1
        and int(summary.get("v2_approval_preflight_extracted_entry_digest_match_count", 0) or 0) >= 2
        and int(summary.get("v2_approval_preflight_support_ready_count", 0) or 0) >= 2
        and summary.get("v2_approval_preflight_owner_review_ready_count") == 1
        and summary.get("v2_approval_preflight_product_exposure_ready_count") == 1
        and summary.get("v2_approval_preflight_hard_fail_count") == 0
    )


def _phase1_default_off_ready(summary: Mapping[str, Any]) -> bool:
    return all(
        summary.get(field, 0) == 0
        for field in (
            "v2_approval_preflight_runtime_dispatch_ready_count",
            "v2_approval_preflight_native_dispatch_allowed_count",
            "v2_approval_preflight_training_path_enabled_count",
            "v2_approval_preflight_product_native_ready_count",
            "v2_approval_preflight_default_behavior_changed_count",
            "v2_approval_preflight_unsafe_claim_count",
        )
    )


def _manifest_artifact_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _profile_include_keys(profile: str) -> set[str]:
    return {
        key
        for spec in build_suite(profile)
        for key in (spec.smoke_id, spec.module)
    }


def _profile_include_group_keys(profile: str) -> set[str]:
    profile_keys = _profile_include_keys(profile)
    return {
        name
        for name, items in INCLUDE_GROUPS.items()
        if any(item in profile_keys for item in items)
    }


def _call_smoke(module: Any) -> dict[str, Any]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        if hasattr(module, "run_smoke"):
            payload = module.run_smoke()
            if isinstance(payload, dict):
                payload = dict(payload)
                output = stdout.getvalue().strip()
                if output:
                    payload["_captured_stdout"] = output[-2000:]
                return payload
            return {"ok": False, "payload_type": type(payload).__name__}
        if hasattr(module, "main"):
            try:
                code = module.main()
            except SystemExit as exc:
                code = exc.code
            ok = code in (None, 0)
            payload = {
                "ok": ok,
                "probe": getattr(module, "__name__", ""),
                "main_exit_code": 0 if code is None else code,
            }
            output = stdout.getvalue().strip()
            if output:
                payload["_captured_stdout"] = output[-2000:]
            return payload
    return {"ok": False, "error": "smoke_module_has_no_run_smoke_or_main"}


def _payload_ok(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("ok") is not False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_TIERS), default="quick")
    parser.add_argument("--include", action="append", default=[], help="Smoke id or module to include.")
    parser.add_argument("--exclude", action="append", default=[], help="Smoke id or module to exclude.")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--allow-unmatched-include",
        action="store_true",
        help="Keep exit status green when an --include filter does not match; use only for probe workflows.",
    )
    parser.add_argument(
        "--allow-empty-selection",
        action="store_true",
        help="Keep exit status green when include/exclude filters select no smokes; use only for probe workflows.",
    )
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument(
        "--allow-real-record-gate",
        action="store_true",
        help=(
            "Permit v2_real_gate_record_batch execution; use only after real reviewer-returned signed bundles "
            "and record inputs exist."
        ),
    )
    parser.add_argument(
        "--real-record-input-manifest",
        type=Path,
        help=(
            "Required with --allow-real-record-gate; JSON manifest whose ready fields prove real "
            "reviewer-returned signed bundle and record inputs exist."
        ),
    )
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--list", action="store_true", help="List selected smokes without running them.")
    parser.add_argument("--plan", action="store_true", help="Print profile guidance without running smokes.")
    parser.add_argument("--audit-scaffold", action="store_true", help="Inspect smoke fragmentation without running smokes.")
    parser.add_argument(
        "--real-record-input-manifest-template",
        action="store_true",
        help="Print the JSON template required by --real-record-input-manifest without running smokes.",
    )
    parser.add_argument(
        "--real-record-phase1-readiness",
        action="store_true",
        help="Print the read-only Milestone 75 phase1 real-return readiness packet without running smokes.",
    )
    args = parser.parse_args(argv)
    include = set(args.include) or None
    exclude = set(args.exclude) or None
    if args.real_record_input_manifest_template:
        print(json.dumps(build_real_record_input_manifest_template(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.real_record_phase1_readiness:
        print(
            json.dumps(
                build_real_record_phase1_readiness_packet(
                    real_record_input_manifest=args.real_record_input_manifest,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.plan:
        print(
            json.dumps(
                build_suite_plan(
                    args.profile,
                    include=include,
                    exclude=exclude,
                    real_record_input_manifest=args.real_record_input_manifest,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.audit_scaffold:
        print(json.dumps(build_scaffold_audit(args.profile), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.list:
        print(
            json.dumps(
                build_suite_list_report(
                    args.profile,
                    include=include,
                    exclude=exclude,
                    real_record_input_manifest=args.real_record_input_manifest,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = run_suite(
        args.profile,
        include=include,
        exclude=exclude,
        continue_on_failure=bool(args.continue_on_failure),
        write_artifact=not bool(args.no_artifact),
        artifact_path=args.artifact_path,
        allow_unmatched_include=bool(args.allow_unmatched_include),
        allow_empty_selection=bool(args.allow_empty_selection),
        allow_real_record_gate=bool(args.allow_real_record_gate),
        real_record_input_manifest=args.real_record_input_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
