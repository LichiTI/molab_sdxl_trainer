"""Smoke checks for native-update release review package."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_native_update_release_review_package import (  # noqa: E402
    BLOCKED_DECISION,
    EXPECTED_GATES,
    HOLD_DECISION,
    READY_DECISION,
    SUPPLEMENTAL_GATES,
    build_native_update_release_review_package,
    load_gate_artifacts,
)


DEFAULT_OFF_FIELDS = (
    "default_behavior_changed",
    "product_exposure_allowed",
    "release_gate_open",
    "training_launch_allowed",
    "training_launch_enabled",
    "training_launch_executed",
    "training_path_enabled",
    "training_dispatch",
    "training_activation_allowed",
    "runtime_dispatch_allowed",
    "native_dispatch_allowed",
    "native_dispatch_enabled",
    "native_dispatch_executed",
    "kernel_launch_executed",
    "parity_executed",
    "training_step_executed",
    "request_submitted",
    "job_created",
    "queue_enqueued",
    "run_record_written",
    "ready_for_ui",
    "ui_exposure_allowed",
    "launcher_exposure_allowed",
    "webui_exposure_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
    "schema_exposure_allowed",
    "backend_router_registered",
    "rollout_authorization_allowed",
)


EXPECTED_OPTIMIZER_FAMILY_COUNTS = {
    "adamw_variant_owner_release_hold_product_native_ready_count": 0,
    "adamw_variant_owner_release_hold_ready_count": 6,
    "adamw_variant_request_schema_ui_non_exposure_ready_count": 6,
    "adamw_variant_request_schema_ui_product_native_ready_count": 0,
    "adaptive_lr_dispatch_integration_review_product_native_ready_count": 0,
    "adaptive_lr_request_schema_ui_non_exposure_ready_count": 11,
    "exact_adamw_stream_event_chain_ownership_abi_ready_count": 1,
    "exact_adamw_stream_event_chain_product_native_ready_count": 0,
    "exact_adamw_stream_event_chain_verified_count": 1,
    "exact_adamw_stream_lifetime_ownership_bound_evidence_count": 1,
    "factored_custom_product_native_ready_count": 0,
    "factored_custom_request_schema_ui_non_exposure_ready_count": 3,
    "missing_classification_count": 0,
    "muon_model_shape_aware_canary_rollout_policy_native_dispatch_allowed_count": 0,
    "muon_model_shape_aware_canary_rollout_policy_product_native_ready_count": 0,
    "muon_model_shape_aware_canary_rollout_policy_ready_count": 1,
    "muon_model_shape_aware_canary_rollout_policy_runtime_dispatch_ready_count": 0,
    "muon_model_shape_aware_canary_rollout_policy_training_path_enabled_count": 0,
    "muon_model_shape_aware_dispatch_integration_review_ready_count": 1,
    "muon_model_shape_aware_dispatch_review_gate_ready_count": 1,
    "muon_model_shape_aware_dispatch_review_product_native_ready_count": 0,
    "muon_model_shape_aware_e2e_shadow_matrix_case_count": 2,
    "muon_model_shape_aware_e2e_shadow_matrix_product_native_ready_count": 0,
    "muon_model_shape_aware_e2e_shadow_matrix_ready_count": 1,
    "muon_model_shape_aware_e2e_shadow_matrix_report_only_case_count": 2,
    "muon_model_shape_aware_native_scratch_kernel_executed_count": 1,
    "muon_model_shape_aware_native_scratch_kernel_ready_count": 1,
    "muon_model_shape_aware_native_scratch_product_native_ready_count": 0,
    "muon_model_shape_aware_owner_release_hold_product_native_ready_count": 0,
    "muon_model_shape_aware_owner_release_hold_ready_count": 1,
    "muon_model_shape_aware_product_native_ready_count": 0,
    "muon_model_shape_aware_request_schema_ui_forbidden_token_hit_count": 0,
    "muon_model_shape_aware_request_schema_ui_non_exposure_ready_count": 1,
    "muon_model_shape_aware_training_loop_native_kernel_launch_count": 1,
    "muon_model_shape_aware_training_loop_native_step_count": 1,
    "muon_model_shape_aware_training_loop_product_native_ready_count": 0,
    "muon_model_shape_aware_training_loop_ready_count": 1,
    "muon_model_shape_aware_training_tensor_binding_kernel_executed_count": 1,
    "muon_model_shape_aware_training_tensor_binding_parity_ready_count": 1,
    "muon_model_shape_aware_training_tensor_binding_product_native_ready_count": 0,
    "muon_model_shape_aware_training_tensor_binding_ready_count": 1,
    "native_ready_count": 1,
    "plugin_optimizer_count": 124,
    "plugin_selected_family_owner_release_hold_family_count": 10,
    "plugin_selected_family_owner_release_hold_optimizer_count": 124,
    "plugin_selected_family_owner_release_hold_product_native_ready_count": 0,
    "plugin_selected_family_request_schema_ui_family_count": 10,
    "plugin_selected_family_request_schema_ui_forbidden_token_hit_count": 0,
    "plugin_selected_family_request_schema_ui_optimizer_count": 124,
    "plugin_selected_family_request_schema_ui_product_native_ready_count": 0,
    "plugin_selected_native_ready_count": 0,
    "plugin_selected_closure_second_order_owner_release_hold_optimizer_count": 5,
    "plugin_selected_closure_second_order_owner_release_hold_product_native_ready_count": 0,
    "plugin_selected_closure_second_order_request_schema_ui_forbidden_token_hit_count": 0,
    "plugin_selected_closure_second_order_request_schema_ui_optimizer_count": 5,
    "plugin_selected_closure_second_order_request_schema_ui_product_native_ready_count": 0,
    "plugin_selected_custom_formula_owner_release_hold_optimizer_count": 47,
    "plugin_selected_custom_formula_owner_release_hold_product_native_ready_count": 0,
    "plugin_selected_custom_formula_request_schema_ui_forbidden_token_hit_count": 0,
    "plugin_selected_custom_formula_request_schema_ui_optimizer_count": 47,
    "plugin_selected_custom_formula_request_schema_ui_product_native_ready_count": 0,
    "plugin_selected_factored_memory_owner_release_hold_optimizer_count": 8,
    "plugin_selected_factored_memory_owner_release_hold_product_native_ready_count": 0,
    "plugin_selected_factored_memory_request_schema_ui_forbidden_token_hit_count": 0,
    "plugin_selected_factored_memory_request_schema_ui_optimizer_count": 8,
    "plugin_selected_factored_memory_request_schema_ui_product_native_ready_count": 0,
    "plugin_selected_fused_backward_owner_release_hold_optimizer_count": 2,
    "plugin_selected_fused_backward_owner_release_hold_product_native_ready_count": 0,
    "plugin_selected_fused_backward_request_schema_ui_forbidden_token_hit_count": 0,
    "plugin_selected_fused_backward_request_schema_ui_optimizer_count": 2,
    "plugin_selected_fused_backward_request_schema_ui_product_native_ready_count": 0,
    "plugin_selected_optimizer_gate_pending_count": 0,
    "plugin_selected_optimizer_gate_ready_count": 10,
    "plugin_selected_simple_formula_canary_rollout_policy_ready_count": 18,
    "plugin_selected_simple_formula_dispatch_review_ready_count": 18,
    "plugin_selected_simple_formula_e2e_shadow_case_count": 18,
    "plugin_selected_simple_formula_request_schema_ui_optimizer_count": 18,
    "plugin_selected_simple_formula_request_schema_ui_product_native_ready_count": 0,
    "simple_formula_request_schema_ui_product_native_ready_count": 0,
    "total_optimizer_types": 31,
}

LEGACY_SIMPLE_FORMULA_SUMMARY_DRIFT_COUNTS = {
    "simple_formula_native_dispatch_canary_ready_count": {0, 2},
    "simple_formula_native_batch_canary_ready_count": {0, 2},
    "simple_formula_representative_product_training_canary_ready_count": {0, 2},
    "simple_formula_owner_release_hold_ready_count": {0, 2},
    "simple_formula_request_schema_ui_non_exposure_ready_count": {0, 7},
}


def run_smoke() -> dict[str, Any]:
    artifacts = _artifact_map()
    pending = build_native_update_release_review_package(gate_artifacts=artifacts)
    assert pending["ok"] is True, pending
    assert pending["evidence_ready"] is True, pending
    assert pending["ready_for_review"] is True, pending
    assert pending["ready_for_owner_release_review"] is True, pending
    assert pending["default_off"] is True, pending
    assert pending["decision"] == HOLD_DECISION, pending
    assert pending["expected_gate_count"] == len(EXPECTED_GATES), pending
    assert pending["present_gate_count"] == len(EXPECTED_GATES), pending
    assert pending["default_off_gate_count"] == len(EXPECTED_GATES), pending
    assert pending["supplemental_gate_count"] == len(SUPPLEMENTAL_GATES), pending
    assert pending["present_supplemental_gate_count"] == len(SUPPLEMENTAL_GATES), pending
    assert pending["default_off_supplemental_gate_count"] == len(SUPPLEMENTAL_GATES), pending
    assert pending["supplemental_gate_summaries"]["optimizer_family_coverage"]["present"] is True, pending
    pending_multitensor_summary = pending["supplemental_gate_summaries"]["native_update_optimizer_multitensor_release_hold"]
    assert pending_multitensor_summary["present"] is True, pending
    assert pending_multitensor_summary["ok"] is True, pending
    assert pending_multitensor_summary["evidence_ready"] is True, pending
    assert pending_multitensor_summary["ready_for_review"] is True, pending
    assert pending_multitensor_summary["default_off"] is True, pending
    pending_gate_template = pending["release_review_template"]["acknowledged_gates"]
    assert len(pending_gate_template) == len(EXPECTED_GATES), pending
    assert "turbocore_phase1_success_review" in pending_gate_template, pending
    assert pending_gate_template["turbocore_phase1_success_review"]["evidence_ready"] is True, pending
    assert pending_gate_template["turbocore_phase1_success_review"]["ready_for_review"] is True, pending
    assert pending_gate_template["turbocore_phase1_success_review"]["default_off"] is True, pending
    pending_optimizer_template = pending["release_review_template"]["acknowledged_supplemental_gates"][
        "optimizer_family_coverage"
    ]
    assert pending_optimizer_template["recommended_next_step"] == (
        "keep native dispatch unwired until explicit owner/release approval is recorded"
    ), pending
    assert pending_optimizer_template["evidence_ready"] is True, pending
    assert pending_optimizer_template["ready_for_review"] is True, pending
    assert pending_optimizer_template["default_off"] is True, pending
    assert pending_optimizer_template["priority_next_gates"], pending
    assert "optimizer_family_counts" in pending_optimizer_template, pending
    assert pending_optimizer_template["source_count"] == 1, pending
    assert pending_optimizer_template["source_names"] == [], pending
    assert pending_optimizer_template["source_payload_digest_match"] is True, pending
    assert pending["post_release_request_fields"] == {}, pending
    assert "native_update_release_owner_review_missing" in pending["blocked_reasons"], pending
    handoff = pending["owner_release_review_handoff"]
    assert handoff["handoff"] == "native_update_release_owner_review_handoff_v0", pending
    assert handoff["ready_for_owner_release_review"] is True, pending
    assert handoff["decision"] == HOLD_DECISION, pending
    assert handoff["action_required"] == "collect_signed_native_update_release_review", pending
    assert "native_update_release_owner_review_missing" in handoff["blocked_reasons"], pending
    assert handoff["required_requested_scope"] == "native_update_release_review_package", pending
    assert "turbocore_phase1_success_review" in handoff["required_gate_acknowledgements"], pending
    assert "optimizer_family_coverage" in handoff["required_supplemental_acknowledgements"], pending
    assert "native_update_optimizer_multitensor_release_hold" in handoff[
        "required_supplemental_acknowledgements"
    ], pending
    assert handoff["supplemental_acknowledgement_counts"]["optimizer_family_coverage"] == pending_optimizer_template[
        "optimizer_family_counts"
    ], pending
    assert handoff["supplemental_acknowledgement_sources"]["optimizer_family_coverage"] == {
        "source_count": pending_optimizer_template["source_count"],
        "source_names": pending_optimizer_template["source_names"],
        "source_payload_digest_match": pending_optimizer_template["source_payload_digest_match"],
    }, pending
    assert handoff["release_review_template_digest"], pending
    assert "training_launch_executed" in handoff["must_remain_false"], pending
    assert "post_release_request_fields" in handoff["must_remain_empty"], pending
    _assert_default_off(pending)

    missing_supplemental_ack = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(approve=True),
    )
    assert missing_supplemental_ack["ok"] is False, missing_supplemental_ack
    assert missing_supplemental_ack["decision"] == BLOCKED_DECISION, missing_supplemental_ack
    assert "supplemental_ack_missing:optimizer_family_coverage" in "\n".join(
        missing_supplemental_ack["blocked_reasons"]
    ), missing_supplemental_ack
    _assert_default_off(missing_supplemental_ack)

    gate_ack = pending["release_review_template"]["acknowledged_gates"]
    supplemental_ack = pending["release_review_template"]["acknowledged_supplemental_gates"]
    signed = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_ack,
            acknowledged_supplemental_gates=supplemental_ack,
        ),
    )
    assert signed["ok"] is True, signed
    assert signed["ready_for_review"] is True, signed
    assert signed["default_off"] is True, signed
    assert signed["release_review_recorded"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["blocked_reasons"] == [], signed
    signed_handoff = signed["owner_release_review_handoff"]
    assert signed_handoff["ready_for_owner_release_review"] is True, signed
    assert signed_handoff["decision"] == READY_DECISION, signed
    assert signed_handoff["blocked_reasons"] == [], signed
    assert signed_handoff["supplemental_acknowledgement_counts"]["optimizer_family_coverage"] == signed[
        "release_review_template"
    ]["acknowledged_supplemental_gates"]["optimizer_family_coverage"]["optimizer_family_counts"], signed
    assert signed_handoff["supplemental_acknowledgement_sources"]["optimizer_family_coverage"] == {
        "source_count": signed["release_review_template"]["acknowledged_supplemental_gates"][
            "optimizer_family_coverage"
        ]["source_count"],
        "source_names": signed["release_review_template"]["acknowledged_supplemental_gates"][
            "optimizer_family_coverage"
        ]["source_names"],
        "source_payload_digest_match": signed["release_review_template"]["acknowledged_supplemental_gates"][
            "optimizer_family_coverage"
        ]["source_payload_digest_match"],
    }, signed
    _assert_default_off(signed)

    missing_gate_ack = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(approve=True, acknowledged_supplemental_gates=supplemental_ack),
    )
    assert missing_gate_ack["ok"] is False, missing_gate_ack
    assert missing_gate_ack["decision"] == BLOCKED_DECISION, missing_gate_ack
    assert "gate_ack_missing:turbocore_phase1_success_review" in "\n".join(
        missing_gate_ack["blocked_reasons"]
    ), missing_gate_ack
    _assert_default_off(missing_gate_ack)

    gate_digest_mismatch_ack = json.loads(json.dumps(gate_ack))
    gate_digest_mismatch_ack["turbocore_phase1_success_review"]["digest"] = "wrong_phase1_digest"
    gate_digest_mismatch = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_digest_mismatch_ack,
            acknowledged_supplemental_gates=supplemental_ack,
        ),
    )
    assert gate_digest_mismatch["ok"] is False, gate_digest_mismatch
    assert gate_digest_mismatch["decision"] == BLOCKED_DECISION, gate_digest_mismatch
    assert "gate_ack_digest_mismatch:turbocore_phase1_success_review" in "\n".join(
        gate_digest_mismatch["blocked_reasons"]
    ), gate_digest_mismatch
    _assert_default_off(gate_digest_mismatch)

    gate_default_off_mismatch_ack = json.loads(json.dumps(gate_ack))
    gate_default_off_mismatch_ack["turbocore_phase1_success_review"]["default_off"] = False
    gate_default_off_mismatch = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_default_off_mismatch_ack,
            acknowledged_supplemental_gates=supplemental_ack,
        ),
    )
    assert gate_default_off_mismatch["ok"] is False, gate_default_off_mismatch
    assert gate_default_off_mismatch["decision"] == BLOCKED_DECISION, gate_default_off_mismatch
    assert "gate_ack_default_off_mismatch:turbocore_phase1_success_review" in "\n".join(
        gate_default_off_mismatch["blocked_reasons"]
    ), gate_default_off_mismatch
    _assert_default_off(gate_default_off_mismatch)

    digest_mismatch_ack = json.loads(json.dumps(supplemental_ack))
    digest_mismatch_ack["optimizer_family_coverage"]["digest"] = "wrong_optimizer_coverage_digest"
    digest_mismatch = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_ack,
            acknowledged_supplemental_gates=digest_mismatch_ack,
        ),
    )
    assert digest_mismatch["ok"] is False, digest_mismatch
    assert digest_mismatch["decision"] == BLOCKED_DECISION, digest_mismatch
    assert "supplemental_ack_digest_mismatch:optimizer_family_coverage" in "\n".join(
        digest_mismatch["blocked_reasons"]
    ), digest_mismatch
    _assert_default_off(digest_mismatch)

    supplemental_default_off_mismatch_ack = json.loads(json.dumps(supplemental_ack))
    supplemental_default_off_mismatch_ack["optimizer_family_coverage"]["default_off"] = False
    supplemental_default_off_mismatch = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_ack,
            acknowledged_supplemental_gates=supplemental_default_off_mismatch_ack,
        ),
    )
    assert supplemental_default_off_mismatch["ok"] is False, supplemental_default_off_mismatch
    assert supplemental_default_off_mismatch["decision"] == BLOCKED_DECISION, supplemental_default_off_mismatch
    assert "supplemental_ack_default_off_mismatch:optimizer_family_coverage" in "\n".join(
        supplemental_default_off_mismatch["blocked_reasons"]
    ), supplemental_default_off_mismatch
    _assert_default_off(supplemental_default_off_mismatch)

    supplemental_evidence_not_ready_artifacts = dict(artifacts)
    supplemental_evidence_not_ready_artifacts["optimizer_family_coverage"] = {
        **supplemental_evidence_not_ready_artifacts["optimizer_family_coverage"],
        "evidence_ready": False,
    }
    supplemental_evidence_not_ready = build_native_update_release_review_package(
        gate_artifacts=supplemental_evidence_not_ready_artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_ack,
            acknowledged_supplemental_gates=build_native_update_release_review_package(
                gate_artifacts=supplemental_evidence_not_ready_artifacts
            )["release_review_template"]["acknowledged_supplemental_gates"],
        ),
    )
    assert supplemental_evidence_not_ready["ok"] is False, supplemental_evidence_not_ready
    assert supplemental_evidence_not_ready["decision"] == BLOCKED_DECISION, supplemental_evidence_not_ready
    assert "supplemental_gate_evidence_not_ready:optimizer_family_coverage" in "\n".join(
        supplemental_evidence_not_ready["blocked_reasons"]
    ), supplemental_evidence_not_ready
    _assert_default_off(supplemental_evidence_not_ready)

    supplemental_not_ready_for_review_artifacts = dict(artifacts)
    supplemental_not_ready_for_review_artifacts["optimizer_family_coverage"] = {
        **supplemental_not_ready_for_review_artifacts["optimizer_family_coverage"],
        "ready_for_optimizer_family_coverage_review": False,
    }
    supplemental_not_ready_for_review = build_native_update_release_review_package(
        gate_artifacts=supplemental_not_ready_for_review_artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_ack,
            acknowledged_supplemental_gates=build_native_update_release_review_package(
                gate_artifacts=supplemental_not_ready_for_review_artifacts
            )["release_review_template"]["acknowledged_supplemental_gates"],
        ),
    )
    assert supplemental_not_ready_for_review["ok"] is False, supplemental_not_ready_for_review
    assert supplemental_not_ready_for_review["decision"] == BLOCKED_DECISION, supplemental_not_ready_for_review
    assert "supplemental_gate_not_ready_for_review:optimizer_family_coverage" in "\n".join(
        supplemental_not_ready_for_review["blocked_reasons"]
    ), supplemental_not_ready_for_review
    _assert_default_off(supplemental_not_ready_for_review)

    next_step_mismatch_ack = json.loads(json.dumps(supplemental_ack))
    next_step_mismatch_ack["optimizer_family_coverage"][
        "recommended_next_step"
    ] = "await explicit owner/release approval for a different artifact"
    next_step_mismatch = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_ack,
            acknowledged_supplemental_gates=next_step_mismatch_ack,
        ),
    )
    assert next_step_mismatch["ok"] is False, next_step_mismatch
    assert next_step_mismatch["decision"] == BLOCKED_DECISION, next_step_mismatch
    assert "supplemental_ack_next_step_mismatch:optimizer_family_coverage" in "\n".join(
        next_step_mismatch["blocked_reasons"]
    ), next_step_mismatch
    _assert_default_off(next_step_mismatch)

    priority_gate_mismatch_ack = json.loads(json.dumps(supplemental_ack))
    priority_gate_mismatch_ack["optimizer_family_coverage"]["priority_next_gates"] = [
        "keep native dispatch unwired until explicit owner/release approval is recorded",
        "wire optimizer native dispatch automatically",
    ]
    priority_gate_mismatch = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_ack,
            acknowledged_supplemental_gates=priority_gate_mismatch_ack,
        ),
    )
    assert priority_gate_mismatch["ok"] is False, priority_gate_mismatch
    assert priority_gate_mismatch["decision"] == BLOCKED_DECISION, priority_gate_mismatch
    assert "supplemental_ack_priority_gates_mismatch:optimizer_family_coverage" in "\n".join(
        priority_gate_mismatch["blocked_reasons"]
    ), priority_gate_mismatch
    _assert_default_off(priority_gate_mismatch)

    counts_mismatch_ack = json.loads(json.dumps(supplemental_ack))
    counts_mismatch_ack["optimizer_family_coverage"]["optimizer_family_counts"][
        "plugin_selected_native_ready_count"
    ] = 1
    counts_mismatch = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_ack,
            acknowledged_supplemental_gates=counts_mismatch_ack,
        ),
    )
    assert counts_mismatch["ok"] is False, counts_mismatch
    assert counts_mismatch["decision"] == BLOCKED_DECISION, counts_mismatch
    assert "supplemental_ack_optimizer_family_counts_mismatch:optimizer_family_coverage" in "\n".join(
        counts_mismatch["blocked_reasons"]
    ), counts_mismatch
    _assert_default_off(counts_mismatch)

    source_payload_mismatch_ack = json.loads(json.dumps(supplemental_ack))
    source_payload_mismatch_ack["optimizer_family_coverage"]["source_payload_digest_match"] = False
    source_payload_mismatch = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=gate_ack,
            acknowledged_supplemental_gates=source_payload_mismatch_ack,
        ),
    )
    assert source_payload_mismatch["ok"] is False, source_payload_mismatch
    assert source_payload_mismatch["decision"] == BLOCKED_DECISION, source_payload_mismatch
    assert "supplemental_ack_source_payload_digest_match_mismatch:optimizer_family_coverage" in "\n".join(
        source_payload_mismatch["blocked_reasons"]
    ), source_payload_mismatch
    _assert_default_off(source_payload_mismatch)

    missing_gate = dict(artifacts)
    missing_gate.pop("turbocore_phase1_success_review")
    _assert_blocked(
        build_native_update_release_review_package(gate_artifacts=missing_gate),
        "gate_missing",
        "phase1_success",
    )

    missing_gate = dict(artifacts)
    missing_gate.pop("native_update_product_exposure_decision")
    _assert_blocked(
        build_native_update_release_review_package(gate_artifacts=missing_gate),
        "gate_missing",
        "product_exposure",
    )

    unsafe_gate = dict(artifacts)
    unsafe_gate["optimizer_family_coverage"] = {
        **unsafe_gate["optimizer_family_coverage"],
        "runtime_dispatch_allowed": True,
    }
    unsafe_supplemental = build_native_update_release_review_package(gate_artifacts=unsafe_gate)
    assert unsafe_supplemental["ok"] is False, unsafe_supplemental
    assert unsafe_supplemental["evidence_ready"] is True, unsafe_supplemental
    assert unsafe_supplemental["decision"] == BLOCKED_DECISION, unsafe_supplemental
    haystack = "\n".join(unsafe_supplemental["blocked_reasons"])
    assert "supplemental" in haystack and "optimizer_family_coverage" in haystack, unsafe_supplemental
    assert "runtime_dispatch_allowed" in haystack, unsafe_supplemental
    _assert_default_off(unsafe_supplemental)

    automatic_optimizer_gate = dict(artifacts)
    automatic_optimizer_gate["optimizer_family_coverage"] = {
        **automatic_optimizer_gate["optimizer_family_coverage"],
        "recommended_next_step": "wire optimizer native dispatch automatically",
        "priority_groups": [
            {
                "group": "optimizer_family_coverage",
                "next_gate": "wire optimizer native dispatch automatically",
                "optimizer_types": ["AdamW"],
                "priority": "P0",
                "why": "This unsafe recommendation must not pass release review.",
            }
        ],
    }
    automatic_supplemental = build_native_update_release_review_package(gate_artifacts=automatic_optimizer_gate)
    assert automatic_supplemental["ok"] is False, automatic_supplemental
    assert automatic_supplemental["evidence_ready"] is True, automatic_supplemental
    assert automatic_supplemental["decision"] == BLOCKED_DECISION, automatic_supplemental
    haystack = "\n".join(automatic_supplemental["blocked_reasons"])
    assert "optimizer_family_coverage_next_step_not_owner_release_hold" in haystack, automatic_supplemental
    assert "optimizer_family_coverage_priority_gate_not_owner_release_hold" in haystack, automatic_supplemental
    _assert_default_off(automatic_supplemental)

    missing_optimizer_next_step = dict(artifacts)
    missing_optimizer_next_step["optimizer_family_coverage"] = {
        k: v for k, v in missing_optimizer_next_step["optimizer_family_coverage"].items()
        if k != "recommended_next_step"
    }
    missing_next_step_package = build_native_update_release_review_package(
        gate_artifacts=missing_optimizer_next_step
    )
    assert missing_next_step_package["ok"] is False, missing_next_step_package
    haystack = "\n".join(missing_next_step_package["blocked_reasons"])
    assert "optimizer_family_coverage_next_step_missing" in haystack, missing_next_step_package
    _assert_default_off(missing_next_step_package)

    missing_optimizer_priority_gates = dict(artifacts)
    missing_optimizer_priority_gates["optimizer_family_coverage"] = {
        k: v for k, v in missing_optimizer_priority_gates["optimizer_family_coverage"].items()
        if k != "priority_groups"
    }
    missing_priority_package = build_native_update_release_review_package(
        gate_artifacts=missing_optimizer_priority_gates
    )
    assert missing_priority_package["ok"] is False, missing_priority_package
    haystack = "\n".join(missing_priority_package["blocked_reasons"])
    assert "optimizer_family_coverage_priority_gates_missing" in haystack, missing_priority_package
    _assert_default_off(missing_priority_package)

    unsafe_gate = dict(artifacts)
    unsafe_gate["turbocore_phase1_success_review"] = {
        **unsafe_gate["turbocore_phase1_success_review"],
        "runtime_dispatch_allowed": True,
    }
    _assert_blocked(
        build_native_update_release_review_package(gate_artifacts=unsafe_gate),
        "unsafe_claim",
        "phase1_success",
        "runtime_dispatch_allowed",
    )

    unsafe_gate = dict(artifacts)
    unsafe_gate["native_update_training_launch_contract"] = {
        **unsafe_gate["native_update_training_launch_contract"],
        "request_submitted": True,
    }
    _assert_blocked(
        build_native_update_release_review_package(gate_artifacts=unsafe_gate),
        "unsafe_claim",
        "request_submitted",
    )

    request_fields = dict(artifacts)
    request_fields["native_update_product_exposure_decision"] = {
        **request_fields["native_update_product_exposure_decision"],
        "post_product_exposure_request_fields": {"native_update": True},
    }
    _assert_blocked(
        build_native_update_release_review_package(gate_artifacts=request_fields),
        "not_default_off",
        "product_exposure",
    )

    unsafe_review = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review={
            **_release_review(
                approve=True,
                acknowledged_gates=gate_ack,
                acknowledged_supplemental_gates=supplemental_ack,
            ),
            "approve_request_submitted": True,
        },
    )
    assert unsafe_review["ok"] is False, unsafe_review
    assert unsafe_review["evidence_ready"] is True, unsafe_review
    assert unsafe_review["decision"] == BLOCKED_DECISION, unsafe_review
    assert "approve_request_submitted" in "\n".join(unsafe_review["blocked_reasons"]), unsafe_review
    _assert_default_off(unsafe_review)

    real_artifact = _write_real_artifact_case(artifacts)
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_release_review_package_smoke",
        "ok": True,
        "pending_decision": pending["decision"],
        "signed_decision": signed["decision"],
        "real_artifact_checked": bool(real_artifact),
        "recommended_next_step": pending["recommended_next_step"],
    }


def _write_real_artifact_case(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for gate, artifact in artifacts.items():
        path = temp_dir / f"{gate}.json"
        if not path.exists():
            path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    package = build_native_update_release_review_package(gate_artifacts=artifacts)
    loaded_artifacts = load_gate_artifacts(temp_dir)
    loaded_package = build_native_update_release_review_package(gate_artifacts=loaded_artifacts)
    assert package["ok"] is True, package
    assert package["evidence_ready"] is True, package
    assert package["decision"] == HOLD_DECISION, package
    assert package["post_release_request_fields"] == {}, package
    assert loaded_package["expected_gate_count"] == len(EXPECTED_GATES), loaded_package
    assert loaded_package["present_gate_count"] == len(EXPECTED_GATES), loaded_package
    assert loaded_package["supplemental_gate_count"] == len(SUPPLEMENTAL_GATES), loaded_package
    assert loaded_package["present_supplemental_gate_count"] == len(SUPPLEMENTAL_GATES), loaded_package
    assert loaded_package["default_off_supplemental_gate_count"] == len(SUPPLEMENTAL_GATES), loaded_package
    assert "turbocore_phase1_success_review" in loaded_package["gate_summaries"], loaded_package
    optimizer_summary = loaded_package["supplemental_gate_summaries"]["optimizer_family_coverage"]
    optimizer_template = loaded_package["release_review_template"]["acknowledged_supplemental_gates"][
        "optimizer_family_coverage"
    ]
    assert optimizer_summary["present"] is True, loaded_package
    assert optimizer_summary["ready_for_review"] is True, loaded_package
    assert optimizer_summary["default_off"] is True, loaded_package
    assert optimizer_summary["unsafe_claims"] == [], loaded_package
    assert optimizer_summary["source_count"] >= 1, loaded_package
    assert optimizer_summary["source_payload_digest_match"] is True, loaded_package
    assert optimizer_summary["recommended_next_step"] == (
        "keep native dispatch unwired until explicit owner/release approval is recorded"
    ), loaded_package
    _assert_optimizer_family_counts(optimizer_summary["optimizer_family_counts"], allow_extra=True)
    _assert_summary_counts_preserved(
        loaded_artifacts["optimizer_family_coverage"],
        optimizer_summary["optimizer_family_counts"],
    )
    assert optimizer_template["digest"] == optimizer_summary["digest"], loaded_package
    assert optimizer_template["recommended_next_step"] == optimizer_summary["recommended_next_step"], loaded_package
    assert optimizer_template["priority_next_gates"] == optimizer_summary["priority_next_gates"], loaded_package
    assert optimizer_template["optimizer_family_counts"] == optimizer_summary["optimizer_family_counts"], loaded_package
    assert optimizer_template["source_count"] == optimizer_summary["source_count"], loaded_package
    assert optimizer_template["source_names"] == optimizer_summary["source_names"], loaded_package
    assert optimizer_template["source_payload_digest_match"] == optimizer_summary[
        "source_payload_digest_match"
    ], loaded_package
    _assert_optimizer_coverage_alias_loader(artifacts["optimizer_family_coverage"])
    _assert_optimizer_coverage_duplicate_loader(artifacts["optimizer_family_coverage"])
    handoff = loaded_package["owner_release_review_handoff"]
    assert handoff["release_review_template_digest"], loaded_package
    assert handoff["supplemental_acknowledgement_counts"]["optimizer_family_coverage"] == optimizer_summary[
        "optimizer_family_counts"
    ], loaded_package
    assert handoff["supplemental_acknowledgement_sources"]["optimizer_family_coverage"] == {
        "source_count": optimizer_summary["source_count"],
        "source_names": optimizer_summary["source_names"],
        "source_payload_digest_match": optimizer_summary["source_payload_digest_match"],
    }, loaded_package
    assert optimizer_summary["priority_group_count"] >= 1, loaded_package
    for next_gate in optimizer_summary["priority_next_gates"]:
        lowered = str(next_gate).lower()
        assert (
            "await explicit owner" in lowered
            or "until explicit owner" in lowered
            or "record explicit owner" in lowered
            or "explicit owner/release approval" in lowered
        ), loaded_package
    _assert_default_off(package)
    _assert_default_off(loaded_package)
    (temp_dir / "native_update_release_review_package.json").write_text(
        json.dumps(loaded_package, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return loaded_package


def _assert_optimizer_coverage_alias_loader(report: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="turbocore_optimizer_alias_") as tmp:
        path = Path(tmp) / "turbocore_optimizer_coverage_scorecard.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        loaded = load_gate_artifacts(path.parent)
    assert "optimizer_family_coverage" in loaded, loaded
    assert loaded["optimizer_family_coverage"]["gate"] == "optimizer_family_coverage", loaded


def _assert_optimizer_coverage_duplicate_loader(report: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="turbocore_optimizer_duplicate_") as tmp:
        directory = Path(tmp)
        canonical = directory / "turbocore_optimizer_family_coverage_scorecard.json"
        alias = directory / "turbocore_optimizer_coverage_scorecard.json"
        canonical.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        alias.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        loaded = load_gate_artifacts(directory)
    optimizer = loaded["optimizer_family_coverage"]
    assert optimizer["_loader_source_count"] == 2, optimizer
    assert optimizer["_loader_payload_digest_match"] is True, optimizer
    assert {
        source["source_name"] for source in optimizer["_loader_sources"]
    } == {
        "turbocore_optimizer_family_coverage_scorecard.json",
        "turbocore_optimizer_coverage_scorecard.json",
    }, optimizer
    duplicate_package = build_native_update_release_review_package(
        gate_artifacts={**_artifact_map(), "optimizer_family_coverage": optimizer}
    )
    duplicate_summary = duplicate_package["supplemental_gate_summaries"]["optimizer_family_coverage"]
    assert duplicate_summary["source_count"] == 2, duplicate_package
    assert duplicate_summary["source_payload_digest_match"] is True, duplicate_package
    assert duplicate_package["decision"] == HOLD_DECISION, duplicate_package

    with tempfile.TemporaryDirectory(prefix="turbocore_optimizer_duplicate_mismatch_") as tmp:
        directory = Path(tmp)
        canonical = directory / "turbocore_optimizer_family_coverage_scorecard.json"
        alias = directory / "turbocore_optimizer_coverage_scorecard.json"
        mismatch = json.loads(json.dumps(report))
        mismatch["summary"]["plugin_selected_native_ready_count"] = 1
        canonical.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        alias.write_text(
            json.dumps(mismatch, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        loaded_mismatch = load_gate_artifacts(directory)
    mismatch_package = build_native_update_release_review_package(
        gate_artifacts={**_artifact_map(), "optimizer_family_coverage": loaded_mismatch["optimizer_family_coverage"]}
    )
    assert mismatch_package["ok"] is False, mismatch_package
    assert mismatch_package["decision"] == BLOCKED_DECISION, mismatch_package
    assert "native_update_release_supplemental_gate_source_payload_digest_mismatch:optimizer_family_coverage" in "\n".join(
        mismatch_package["blocked_reasons"]
    ), mismatch_package


def _artifact_map() -> dict[str, dict[str, Any]]:
    return {gate: _artifact(gate) for gate in EXPECTED_GATES + SUPPLEMENTAL_GATES}


def _artifact(gate: str) -> dict[str, Any]:
    ready_field = {
        "optimizer_family_coverage": "ready_for_optimizer_family_coverage_review",
        "turbocore_phase1_success_review": "ready_for_phase1_success_review",
        "native_update_rollout_review_package": "ready_for_owner_review",
        "native_update_training_dispatch_integration_contract": "ready_for_integration_review",
        "native_update_activation_contract": "ready_for_activation_review",
        "native_update_runtime_execution_contract": "ready_for_runtime_execution_review",
        "native_update_runtime_dispatch_contract": "ready_for_runtime_dispatch_review",
        "native_update_native_dispatch_execution_contract": "ready_for_native_dispatch_execution_review",
        "native_update_kernel_launch_execution_contract": "ready_for_kernel_launch_execution_review",
        "native_update_parity_execution_contract": "ready_for_parity_execution_review",
        "native_update_training_step_execution_contract": "ready_for_training_step_execution_review",
        "native_update_training_launch_contract": "ready_for_training_launch_review",
        "native_update_product_exposure_decision": "ready_for_product_exposure_review",
        "native_update_optimizer_multitensor_release_hold": "ready_for_optimizer_multitensor_release_review",
    }[gate]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "package": f"turbocore_{gate}_v0",
        "gate": gate,
        "ok": True,
        "evidence_ready": True,
        ready_field: True,
        "manual_review_required": True,
        "decision": f"{gate}_hold_for_owner_review_default_off",
        "blocked_reasons": ["owner_review_missing"],
    }
    if gate == "native_update_rollout_review_package":
        payload["evidence_package_ready"] = True
    if gate == "optimizer_family_coverage":
        payload["scorecard"] = "turbocore_optimizer_family_coverage_scorecard_v0"
        payload["promotion_ready"] = True
        payload[
            "recommended_next_step"
        ] = "keep native dispatch unwired until explicit owner/release approval is recorded"
        payload["priority_groups"] = [
            {
                "group": "optimizer_family_coverage",
                "next_gate": "keep native dispatch unwired until explicit owner/release approval is recorded",
                "optimizer_types": ["AdamW"],
                "priority": "P0",
                "why": "Synthetic release-smoke coverage keeps product dispatch default-off.",
            }
        ]
        payload["summary"] = {
            "total_optimizer_types": 30,
            "plugin_optimizer_count": 124,
            "plugin_selector_classification_ready": True,
            "plugin_selector_missing_classification_count": 0,
            "plugin_selector_missing_resume_count": 0,
        }
    for field in DEFAULT_OFF_FIELDS:
        payload[field] = False
    for field in (
        "post_native_update_request_fields",
        "post_integration_request_fields",
        "post_activation_request_fields",
        "post_runtime_execution_request_fields",
        "post_runtime_dispatch_request_fields",
        "post_native_dispatch_execution_request_fields",
        "post_kernel_launch_execution_request_fields",
        "post_parity_execution_request_fields",
        "post_training_step_execution_request_fields",
        "post_training_launch_request_fields",
        "post_product_exposure_request_fields",
        "post_phase1_request_fields",
    ):
        payload[field] = {}
    return payload


def _release_review(
    *,
    approve: bool,
    acknowledged_supplemental_gates: dict[str, Any] | None = None,
    acknowledged_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    review = {
        "reviewer": "owner",
        "reviewed_at": "2026-06-05",
        "requested_scope": "native_update_release_review_package",
        "approve_native_update_release_review_package": bool(approve),
    }
    if acknowledged_gates is not None:
        review["acknowledged_gates"] = acknowledged_gates
    if acknowledged_supplemental_gates is not None:
        review["acknowledged_supplemental_gates"] = acknowledged_supplemental_gates
    for field in (
        "acknowledge_all_expected_gates_present",
        "acknowledge_all_gates_default_off",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_no_training_launch_or_native_execution",
        "acknowledge_product_exposure_requires_separate_owner_direction",
    ):
        review[field] = True
    return review


def _assert_default_off(package: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert package[field] is False, (field, package)
    assert package["default_off"] is True, package


def _assert_optimizer_family_counts(counts: dict[str, Any], *, allow_extra: bool = False) -> None:
    for key, expected in EXPECTED_OPTIMIZER_FAMILY_COUNTS.items():
        assert counts[key] == expected, (key, counts)
    for key, allowed_values in LEGACY_SIMPLE_FORMULA_SUMMARY_DRIFT_COUNTS.items():
        if key in counts:
            assert int(counts[key]) in allowed_values, (key, counts)
    assert counts.get("plugin_selected_simple_formula_request_schema_ui_non_exposure_ready") in {
        True,
        1,
    }, ("plugin_selected_simple_formula_request_schema_ui_non_exposure_ready", counts)
    if not allow_extra:
        expected_keys = set(EXPECTED_OPTIMIZER_FAMILY_COUNTS)
        expected_keys.update(
            {
                "plugin_selected_simple_formula_request_schema_ui_non_exposure_ready",
                *LEGACY_SIMPLE_FORMULA_SUMMARY_DRIFT_COUNTS,
            }
        )
        assert set(counts) == expected_keys, counts


def _assert_summary_counts_preserved(
    coverage_report: dict[str, Any],
    optimizer_family_counts: dict[str, Any],
) -> None:
    summary = coverage_report.get("summary")
    assert isinstance(summary, dict), coverage_report
    summary_counts = {
        key: int(value)
        for key, value in summary.items()
        if (key.endswith("_count") or key.endswith("_ready")) and isinstance(value, (bool, int))
    }
    assert summary_counts, coverage_report
    for key, expected in summary_counts.items():
        actual = optimizer_family_counts[key]
        if key in LEGACY_SIMPLE_FORMULA_SUMMARY_DRIFT_COUNTS:
            allowed_values = LEGACY_SIMPLE_FORMULA_SUMMARY_DRIFT_COUNTS[key]
            assert int(expected) in allowed_values, (key, expected, optimizer_family_counts)
            assert int(actual) in allowed_values, (key, optimizer_family_counts)
            continue
        assert actual == expected, (key, optimizer_family_counts)


def _assert_selected_plugin_counts_preserved(
    coverage_report: dict[str, Any],
    optimizer_family_counts: dict[str, Any],
) -> None:
    _assert_summary_counts_preserved(coverage_report, optimizer_family_counts)


def _assert_blocked(package: dict[str, Any], *needles: str) -> None:
    assert package["ok"] is False, package
    assert package["evidence_ready"] is False, package
    assert package["decision"] == BLOCKED_DECISION, package
    haystack = "\n".join(package["blocked_reasons"])
    for needle in needles:
        assert needle in haystack, package
    _assert_default_off(package)


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
