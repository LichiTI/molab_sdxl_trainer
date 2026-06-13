"""Smoke checks for the v2 completion audit."""

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

from core.turbocore_optimizer_v2_completion_audit import (  # noqa: E402
    ARTIFACT,
    build_optimizer_v2_completion_audit,
)


def run_smoke() -> dict[str, Any]:
    artifact_payload = build_optimizer_v2_completion_audit(write_artifact=True)
    assert artifact_payload["ok"] is True, artifact_payload
    assert artifact_payload["completion_ready"] is False, artifact_payload
    assert artifact_payload["roadmap_complete"] is False, artifact_payload
    assert artifact_payload["approval_recorded"] is False, artifact_payload
    assert artifact_payload["runtime_dispatch_ready"] is False, artifact_payload
    assert artifact_payload["native_dispatch_allowed"] is False, artifact_payload
    assert artifact_payload["training_path_enabled"] is False, artifact_payload
    assert artifact_payload["product_native_ready"] is False, artifact_payload
    assert ARTIFACT.exists(), ARTIFACT

    audit = build_optimizer_v2_completion_audit(
        source_reports=_source_reports_waiting_for_signature(),
        write_artifact=False,
    )
    summary = audit["summary"]
    assert audit["ok"] is True, audit
    assert audit["completion_audit_ready"] is True, audit
    assert audit["completion_ready"] is False, audit
    assert audit["roadmap_complete"] is False, audit
    assert summary["v2_completion_audit_requirement_count"] == 15, audit
    assert summary["v2_completion_audit_passed_requirement_count"] == 4, audit
    assert summary["v2_completion_audit_failed_requirement_count"] == 11, audit
    assert summary["v2_completion_audit_remaining_gate_open_count"] == 6, audit
    assert summary["v2_completion_audit_command_audit_ready_count"] == 1, audit
    assert summary["v2_completion_audit_command_audit_expected_arg_binding_count"] == 35, audit
    assert summary["v2_completion_audit_command_audit_expected_arg_mismatch_count"] == 0, audit
    assert summary["v2_completion_audit_command_audit_expected_path_binding_ready_count"] == 1, audit
    assert summary["v2_completion_audit_command_audit_phase1_handoff_post_return_command_count"] == 3, audit
    assert summary["v2_completion_audit_command_audit_phase1_handoff_post_return_pre_record_command_count"] == 3, (
        audit
    )
    assert summary["v2_completion_audit_command_audit_phase1_handoff_post_return_approval_record_command_count"] == 0, (
        audit
    )
    assert summary["v2_completion_audit_command_audit_phase1_handoff_post_return_command_match_count"] == 3, audit
    assert summary["v2_completion_audit_command_audit_phase1_handoff_post_return_command_mismatch_count"] == 0, (
        audit
    )
    assert summary["v2_completion_audit_command_audit_phase1_handoff_post_return_ready_count"] == 1, audit
    assert summary["v2_completion_audit_phase1_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase2_ready_count"] == 0, audit
    assert summary["v2_completion_audit_record_input_item_count"] == 13, audit
    assert summary["v2_completion_audit_record_input_phase1_item_count"] == 6, audit
    assert summary["v2_completion_audit_record_input_phase2_item_count"] == 7, audit
    assert summary["v2_completion_audit_phase2_refresh_inputs_tracked_count"] == 2, audit
    assert summary["v2_completion_audit_record_input_checklist_shape_ready_count"] == 1, audit
    assert summary["v2_completion_audit_record_input_support_check_count"] == 3, audit
    assert summary["v2_completion_audit_record_input_support_ready_count"] == 2, audit
    assert summary["v2_completion_audit_record_input_support_blocked_item_count"] == 1, audit
    assert summary["v2_completion_audit_record_input_support_blocker_count"] == 1, audit
    assert summary["v2_completion_audit_record_input_support_source_binding_ready_count"] == 2, audit
    assert summary["v2_completion_audit_record_input_support_source_binding_blocker_count"] == 0, audit
    assert summary["v2_completion_audit_record_input_support_shape_ready_count"] == 1, audit
    assert summary["v2_completion_audit_phase1_signed_bundle_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase2_signed_bundle_ready_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_missing_signature_count"] == 3, audit
    assert summary["v2_completion_audit_phase1_signed_bundle_missing_signature_count"] == 2, audit
    assert summary["v2_completion_audit_phase2_signed_bundle_missing_signature_count"] == 1, audit
    assert summary["v2_completion_audit_signed_bundle_freshness_guard_ready_count"] == 1, audit
    assert summary["v2_completion_audit_signed_bundle_freshness_template_digest_match_count"] == 1, audit
    assert summary["v2_completion_audit_signed_bundle_freshness_stale_signed_bundle_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_freshness_unknown_signed_entry_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_freshness_unsafe_claim_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_not_ready_entry_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_source_digest_match_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_source_digest_stale_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_template_digest_mismatch_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_unknown_entry_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_unsigned_template_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_intake_integrity_ready_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_manual_field_check_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_manual_field_ready_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_manual_field_missing_count"] == 0, audit
    assert summary["v2_completion_audit_signed_bundle_manual_field_shape_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase1_signed_bundle_manual_field_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase2_signed_bundle_manual_field_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase1_signed_bundle_manual_shape_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase2_signed_bundle_manual_shape_ready_count"] == 0, audit
    assert summary["v2_completion_audit_full_signed_bundle_manual_shape_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase1_extraction_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase2_extraction_ready_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_missing_signature_count"] == 3, audit
    assert summary["v2_completion_audit_phase1_extraction_missing_signature_count"] == 2, audit
    assert summary["v2_completion_audit_phase2_extraction_missing_signature_count"] == 1, audit
    assert summary["v2_completion_audit_extraction_not_ready_entry_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_source_digest_match_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_source_digest_stale_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_template_digest_mismatch_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_unknown_entry_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_unsigned_template_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_integrity_ready_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_signed_entry_digest_present_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_extractable_signed_entry_digest_present_count"] == 0, audit
    assert summary["v2_completion_audit_extraction_digest_shape_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase1_preflight_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase2_preflight_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase1_record_chain_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase2_record_chain_ready_count"] == 0, audit
    assert summary["v2_completion_audit_full_record_chain_ready_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_full_ready_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_integrity_ready_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_signed_bundle_valid_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_source_digest_match_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_source_digest_stale_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_template_digest_mismatch_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_unknown_entry_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_not_ready_entry_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_source_integrity_ready_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_support_integrity_ready_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_unsigned_template_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_signed_payload_digest_present_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_signed_payload_digest_missing_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_signed_bundle_entry_digest_present_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_signed_payload_bundle_digest_match_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_extracted_entry_digest_match_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_extracted_entry_digest_mismatch_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_extracted_entry_source_missing_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_support_ready_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_support_invalid_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_support_source_binding_ready_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_support_source_binding_blocker_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_post_record_request_field_emission_count"] == 0, audit
    assert summary["v2_completion_audit_preflight_hard_fail_count"] == 0, audit
    assert summary["v2_completion_audit_owner_review_recorded_count"] == 0, audit
    assert summary["v2_completion_audit_product_exposure_recorded_count"] == 0, audit
    assert summary["v2_completion_audit_owner_direction_recorded_count"] == 0, audit
    assert summary["v2_completion_audit_phase1_record_preflight_binding_ready_count"] == 0, audit
    assert summary["v2_completion_audit_phase2_record_preflight_binding_ready_count"] == 0, audit
    assert summary["v2_completion_audit_full_record_preflight_binding_ready_count"] == 0, audit
    assert summary["v2_completion_audit_record_preflight_binding_ready_count"] == 0, audit
    assert summary["v2_completion_audit_artifact_first_required_artifact_count"] == 12, audit
    assert summary["v2_completion_audit_artifact_first_ready_artifact_count"] == 12, audit
    assert summary["v2_completion_audit_artifact_first_ready_count"] == 1, audit
    assert summary["v2_completion_audit_artifact_first_record_input_checklist_shape_ready_count"] == 1, audit
    assert summary["v2_completion_audit_artifact_first_record_input_support_shape_ready_count"] == 1, audit
    assert summary["v2_completion_audit_artifact_first_reviewer_handoff_phase_metadata_ready_count"] == 1, audit
    assert summary["v2_completion_audit_artifact_first_reviewer_handoff_phase1_template_entry_count"] == 2, audit
    assert summary["v2_completion_audit_artifact_first_reviewer_handoff_phase2_deferred_signature_count"] == 1, audit
    assert summary["v2_completion_audit_artifact_first_reviewer_handoff_phase_shape_ready_count"] == 1, audit
    assert summary["v2_completion_audit_artifact_first_command_audit_expected_path_binding_ready_count"] == 1, audit
    assert (
        summary["v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_count"] == 3
    ), audit
    assert (
        summary[
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_pre_record_command_count"
        ]
        == 3
    ), audit
    assert (
        summary[
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_approval_record_command_count"
        ]
        == 0
    ), audit
    assert (
        summary["v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_match_count"]
        == 3
    ), audit
    assert (
        summary["v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_mismatch_count"]
        == 0
    ), audit
    assert summary["v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_ready_count"] == 1, (
        audit
    )
    assert (
        summary["v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_shape_ready_count"] == 1
    ), audit
    assert summary["v2_completion_audit_default_off_guard_count"] == 1, audit
    assert summary["v2_completion_audit_completion_ready_count"] == 0, audit
    assert summary["v2_completion_audit_approval_recorded_count"] == 0, audit
    assert summary["v2_completion_audit_runtime_dispatch_ready_count"] == 0, audit
    assert summary["v2_completion_audit_native_dispatch_allowed_count"] == 0, audit
    assert summary["v2_completion_audit_training_path_enabled_count"] == 0, audit
    assert summary["v2_completion_audit_product_native_ready_count"] == 0, audit
    assert summary["v2_completion_audit_default_behavior_changed_count"] == 0, audit
    assert summary["v2_completion_audit_unsafe_claim_count"] == 0, audit

    unsafe = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_waiting_for_signature(),
            "approval_state": {
                "runtime_dispatch_ready": True,
                "summary": {
                    "v2_approval_state_remaining_gate_open_count": 6,
                    "v2_approval_state_recorded_stage_count": 0,
                },
            },
        },
        write_artifact=False,
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["summary"]["v2_completion_audit_unsafe_claim_count"] == 1, unsafe
    assert unsafe["summary"]["v2_completion_audit_default_off_guard_count"] == 0, unsafe

    stale_checklist = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "record_input_checklist": {
                "summary": {
                    "v2_record_input_checklist_item_count": 11,
                    "v2_record_input_checklist_phase1_item_count": 6,
                    "v2_record_input_checklist_phase1_ready_count": 1,
                    "v2_record_input_checklist_phase2_item_count": 5,
                    "v2_record_input_checklist_phase2_ready_count": 1,
                },
            },
        },
        write_artifact=False,
    )
    stale_summary = stale_checklist["summary"]
    assert stale_checklist["ok"] is True, stale_checklist
    assert stale_checklist["completion_ready"] is False, stale_checklist
    assert stale_checklist["roadmap_complete"] is False, stale_checklist
    assert stale_summary["v2_completion_audit_record_input_item_count"] == 11, stale_checklist
    assert stale_summary["v2_completion_audit_record_input_phase2_item_count"] == 5, stale_checklist
    assert stale_summary["v2_completion_audit_phase2_refresh_inputs_tracked_count"] == 0, stale_checklist
    assert stale_summary["v2_completion_audit_record_input_checklist_shape_ready_count"] == 0, stale_checklist
    assert stale_summary["v2_completion_audit_record_input_support_shape_ready_count"] == 0, stale_checklist
    assert stale_summary["v2_completion_audit_failed_requirement_count"] == 2, stale_checklist

    stale_command_audit = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "approval_command_audit": {
                "summary": {
                    "v2_approval_command_audit_ready_count": 1,
                    "v2_approval_command_audit_expected_arg_binding_count": 35,
                    "v2_approval_command_audit_expected_arg_mismatch_count": 1,
                    "v2_approval_command_audit_expected_path_binding_ready_count": 0,
                    "v2_approval_command_audit_preflight_artifact_write_count": 2,
                    "v2_approval_command_audit_preflight_no_artifact_blocker_count": 0,
                    "v2_approval_command_audit_record_command_preflight_arg_count": 3,
                },
            },
        },
        write_artifact=False,
    )
    stale_command_summary = stale_command_audit["summary"]
    assert stale_command_audit["ok"] is True, stale_command_audit
    assert stale_command_audit["completion_ready"] is False, stale_command_audit
    assert stale_command_audit["roadmap_complete"] is False, stale_command_audit
    assert stale_command_summary["v2_completion_audit_failed_requirement_count"] == 1, stale_command_audit
    assert stale_command_summary["v2_completion_audit_command_audit_expected_arg_mismatch_count"] == 1, (
        stale_command_audit
    )
    assert "approval_command_audit_path_binding_not_ready" in stale_command_audit["blocked_reasons"], (
        stale_command_audit
    )

    stale_command_handoff = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "approval_command_audit": {
                "summary": {
                    "v2_approval_command_audit_ready_count": 1,
                    "v2_approval_command_audit_expected_arg_binding_count": 35,
                    "v2_approval_command_audit_expected_arg_mismatch_count": 0,
                    "v2_approval_command_audit_expected_path_binding_ready_count": 1,
                    "v2_approval_command_audit_preflight_artifact_write_count": 2,
                    "v2_approval_command_audit_preflight_no_artifact_blocker_count": 0,
                    "v2_approval_command_audit_record_command_preflight_arg_count": 3,
                },
            },
        },
        write_artifact=False,
    )
    stale_handoff_summary = stale_command_handoff["summary"]
    assert stale_command_handoff["ok"] is True, stale_command_handoff
    assert stale_command_handoff["completion_ready"] is False, stale_command_handoff
    assert stale_command_handoff["roadmap_complete"] is False, stale_command_handoff
    assert stale_handoff_summary["v2_completion_audit_failed_requirement_count"] == 1, stale_command_handoff
    assert stale_handoff_summary["v2_completion_audit_command_audit_expected_path_binding_ready_count"] == 1, (
        stale_command_handoff
    )
    assert stale_handoff_summary["v2_completion_audit_command_audit_phase1_handoff_post_return_ready_count"] == 0, (
        stale_command_handoff
    )
    assert "approval_command_audit_path_binding_not_ready" in stale_command_handoff["blocked_reasons"], (
        stale_command_handoff
    )

    stale_preflight = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "approval_execution_preflight": {
                "summary": {
                    "v2_approval_preflight_full_ready_count": 1,
                    "v2_approval_preflight_signed_bundle_valid_count": 1,
                    "v2_approval_preflight_source_digest_match_count": 1,
                    "v2_approval_preflight_source_digest_stale_count": 0,
                    "v2_approval_preflight_template_digest_mismatch_count": 0,
                    "v2_approval_preflight_unknown_entry_count": 0,
                    "v2_approval_preflight_unsigned_template_count": 0,
                    "v2_approval_preflight_signed_payload_digest_present_count": 2,
                    "v2_approval_preflight_signed_payload_digest_missing_count": 1,
                    "v2_approval_preflight_signed_bundle_entry_digest_present_count": 3,
                    "v2_approval_preflight_signed_payload_bundle_digest_match_count": 2,
                    "v2_approval_preflight_extracted_entry_digest_match_count": 3,
                    "v2_approval_preflight_extracted_entry_digest_mismatch_count": 0,
                    "v2_approval_preflight_extracted_entry_source_missing_count": 0,
                    "v2_approval_preflight_support_ready_count": 3,
                    "v2_approval_preflight_support_invalid_count": 0,
                    "v2_approval_preflight_post_record_request_field_emission_count": 0,
                    "v2_approval_preflight_hard_fail_count": 0,
                },
            },
        },
        write_artifact=False,
    )
    stale_preflight_summary = stale_preflight["summary"]
    assert stale_preflight["ok"] is True, stale_preflight
    assert stale_preflight["completion_ready"] is False, stale_preflight
    assert stale_preflight["roadmap_complete"] is False, stale_preflight
    assert stale_preflight_summary["v2_completion_audit_failed_requirement_count"] == 1, stale_preflight
    assert stale_preflight_summary["v2_completion_audit_preflight_full_ready_count"] == 1, stale_preflight
    assert stale_preflight_summary["v2_completion_audit_preflight_integrity_ready_count"] == 0, stale_preflight
    assert stale_preflight_summary["v2_completion_audit_preflight_signed_payload_digest_present_count"] == 2, (
        stale_preflight
    )
    assert stale_preflight_summary["v2_completion_audit_preflight_signed_payload_digest_missing_count"] == 1, (
        stale_preflight
    )
    assert stale_preflight_summary["v2_completion_audit_preflight_signed_payload_bundle_digest_match_count"] == 2, (
        stale_preflight
    )
    assert "approval_preflight_integrity_guards_not_ready" in stale_preflight["blocked_reasons"], (
        stale_preflight
    )

    stale_not_ready_preflight = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "approval_execution_preflight": {
                "summary": {
                    "v2_approval_preflight_full_ready_count": 1,
                    "v2_approval_preflight_signed_bundle_valid_count": 1,
                    "v2_approval_preflight_source_digest_match_count": 1,
                    "v2_approval_preflight_source_digest_stale_count": 0,
                    "v2_approval_preflight_template_digest_mismatch_count": 0,
                    "v2_approval_preflight_unknown_entry_count": 0,
                    "v2_approval_preflight_not_ready_entry_count": 1,
                    "v2_approval_preflight_unsigned_template_count": 0,
                    "v2_approval_preflight_signed_payload_digest_present_count": 3,
                    "v2_approval_preflight_signed_payload_digest_missing_count": 0,
                    "v2_approval_preflight_signed_bundle_entry_digest_present_count": 3,
                    "v2_approval_preflight_signed_payload_bundle_digest_match_count": 3,
                    "v2_approval_preflight_extracted_entry_digest_match_count": 3,
                    "v2_approval_preflight_extracted_entry_digest_mismatch_count": 0,
                    "v2_approval_preflight_extracted_entry_source_missing_count": 0,
                    "v2_approval_preflight_support_ready_count": 3,
                    "v2_approval_preflight_support_invalid_count": 0,
                    "v2_approval_preflight_post_record_request_field_emission_count": 0,
                    "v2_approval_preflight_hard_fail_count": 0,
                },
            },
        },
        write_artifact=False,
    )
    stale_not_ready_summary = stale_not_ready_preflight["summary"]
    assert stale_not_ready_preflight["ok"] is True, stale_not_ready_preflight
    assert stale_not_ready_preflight["completion_ready"] is False, stale_not_ready_preflight
    assert stale_not_ready_summary["v2_completion_audit_failed_requirement_count"] == 1, stale_not_ready_preflight
    assert stale_not_ready_summary["v2_completion_audit_preflight_integrity_ready_count"] == 0, (
        stale_not_ready_preflight
    )
    assert stale_not_ready_summary["v2_completion_audit_preflight_not_ready_entry_count"] == 1, (
        stale_not_ready_preflight
    )
    assert "approval_preflight_integrity_guards_not_ready" in stale_not_ready_preflight["blocked_reasons"], (
        stale_not_ready_preflight
    )

    stale_record_binding = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "approval_state": {
                "summary": {
                    **_source_reports_complete_default_off()["approval_state"]["summary"],
                    "v2_approval_state_record_preflight_binding_ready_count": 2,
                    "v2_approval_state_full_record_preflight_binding_ready_count": 0,
                },
            },
        },
        write_artifact=False,
    )
    stale_record_binding_summary = stale_record_binding["summary"]
    assert stale_record_binding["ok"] is True, stale_record_binding
    assert stale_record_binding["completion_ready"] is False, stale_record_binding
    assert stale_record_binding_summary["v2_completion_audit_failed_requirement_count"] == 1, (
        stale_record_binding
    )
    assert stale_record_binding_summary["v2_completion_audit_record_preflight_binding_ready_count"] == 2, (
        stale_record_binding
    )
    assert "approval_records_preflight_binding_not_ready" in stale_record_binding["blocked_reasons"], (
        stale_record_binding
    )

    stale_signed_bundle_integrity = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "approval_state": {
                "summary": {
                    **_source_reports_complete_default_off()["approval_state"]["summary"],
                    "v2_approval_state_signed_bundle_source_digest_stale_count": 1,
                    "v2_approval_state_signed_bundle_intake_integrity_ready_count": 0,
                },
            },
        },
        write_artifact=False,
    )
    stale_signed_bundle_summary = stale_signed_bundle_integrity["summary"]
    assert stale_signed_bundle_integrity["ok"] is True, stale_signed_bundle_integrity
    assert stale_signed_bundle_integrity["completion_ready"] is False, stale_signed_bundle_integrity
    assert stale_signed_bundle_summary["v2_completion_audit_failed_requirement_count"] == 1, (
        stale_signed_bundle_integrity
    )
    assert stale_signed_bundle_summary["v2_completion_audit_signed_bundle_source_digest_stale_count"] == 1, (
        stale_signed_bundle_integrity
    )
    assert stale_signed_bundle_summary["v2_completion_audit_signed_bundle_intake_integrity_ready_count"] == 0, (
        stale_signed_bundle_integrity
    )
    assert "signed_bundle_integrity_not_ready" in stale_signed_bundle_integrity["blocked_reasons"], (
        stale_signed_bundle_integrity
    )

    missing_phase2_signature = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "approval_state": {
                "summary": {
                    **_source_reports_complete_default_off()["approval_state"]["summary"],
                    "v2_approval_state_signed_bundle_missing_signature_count": 1,
                    "v2_approval_state_phase2_signed_bundle_missing_signature_count": 1,
                    "v2_approval_state_phase2_record_chain_ready_count": 0,
                    "v2_approval_state_full_record_chain_ready_count": 0,
                },
            },
        },
        write_artifact=False,
    )
    missing_phase2_signature_summary = missing_phase2_signature["summary"]
    assert missing_phase2_signature["ok"] is True, missing_phase2_signature
    assert missing_phase2_signature["completion_ready"] is False, missing_phase2_signature
    assert missing_phase2_signature_summary["v2_completion_audit_failed_requirement_count"] == 1, (
        missing_phase2_signature
    )
    assert missing_phase2_signature_summary["v2_completion_audit_signed_bundle_missing_signature_count"] == 1, (
        missing_phase2_signature
    )
    assert missing_phase2_signature_summary["v2_completion_audit_phase2_signed_bundle_missing_signature_count"] == 1, (
        missing_phase2_signature
    )
    assert "signed_bundle_integrity_not_ready" in missing_phase2_signature["blocked_reasons"], (
        missing_phase2_signature
    )

    missing_phase2_extraction_signature = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "approval_state": {
                "summary": {
                    **_source_reports_complete_default_off()["approval_state"]["summary"],
                    "v2_approval_state_extraction_missing_signature_count": 1,
                    "v2_approval_state_phase2_extraction_missing_signature_count": 1,
                    "v2_approval_state_phase2_record_chain_ready_count": 0,
                    "v2_approval_state_full_record_chain_ready_count": 0,
                },
            },
        },
        write_artifact=False,
    )
    missing_phase2_extraction_summary = missing_phase2_extraction_signature["summary"]
    assert missing_phase2_extraction_signature["ok"] is True, missing_phase2_extraction_signature
    assert missing_phase2_extraction_signature["completion_ready"] is False, missing_phase2_extraction_signature
    assert missing_phase2_extraction_summary["v2_completion_audit_failed_requirement_count"] == 1, (
        missing_phase2_extraction_signature
    )
    assert missing_phase2_extraction_summary["v2_completion_audit_extraction_missing_signature_count"] == 1, (
        missing_phase2_extraction_signature
    )
    assert (
        missing_phase2_extraction_summary["v2_completion_audit_phase2_extraction_missing_signature_count"] == 1
    ), missing_phase2_extraction_signature
    assert "signed_bundle_integrity_not_ready" in missing_phase2_extraction_signature["blocked_reasons"], (
        missing_phase2_extraction_signature
    )

    stale_freshness = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "approval_state": {
                "summary": {
                    **_source_reports_complete_default_off()["approval_state"]["summary"],
                    "v2_approval_state_signed_bundle_freshness_guard_ready_count": 0,
                    "v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count": 1,
                },
            },
        },
        write_artifact=False,
    )
    stale_freshness_summary = stale_freshness["summary"]
    assert stale_freshness["ok"] is True, stale_freshness
    assert stale_freshness["completion_ready"] is False, stale_freshness
    assert stale_freshness_summary["v2_completion_audit_failed_requirement_count"] == 1, stale_freshness
    assert stale_freshness_summary["v2_completion_audit_signed_bundle_freshness_guard_ready_count"] == 0, (
        stale_freshness
    )
    assert stale_freshness_summary["v2_completion_audit_signed_bundle_freshness_stale_signed_bundle_count"] == 1, (
        stale_freshness
    )
    assert "signed_bundle_freshness_not_ready" in stale_freshness["blocked_reasons"], stale_freshness

    stale_artifact_first = build_optimizer_v2_completion_audit(
        source_reports={
            **_source_reports_complete_default_off(),
            "release_artifact_first_validation": {
                "summary": {
                    "release_artifact_first_validation_ready_count": 1,
                }
            },
        },
        write_artifact=False,
    )
    stale_artifact_first_summary = stale_artifact_first["summary"]
    assert stale_artifact_first["ok"] is True, stale_artifact_first
    assert stale_artifact_first["completion_ready"] is False, stale_artifact_first
    assert stale_artifact_first["roadmap_complete"] is False, stale_artifact_first
    assert stale_artifact_first_summary["v2_completion_audit_failed_requirement_count"] == 1, stale_artifact_first
    assert stale_artifact_first_summary["v2_completion_audit_artifact_first_ready_count"] == 1, stale_artifact_first
    assert (
        stale_artifact_first_summary["v2_completion_audit_artifact_first_required_artifact_count"] == 0
    ), stale_artifact_first
    assert (
        stale_artifact_first_summary[
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_shape_ready_count"
        ]
        == 0
    ), stale_artifact_first
    assert "release_artifact_first_validation_shape_not_ready" in stale_artifact_first["blocked_reasons"], (
        stale_artifact_first
    )

    complete = build_optimizer_v2_completion_audit(
        source_reports=_source_reports_complete_default_off(),
        write_artifact=False,
    )
    complete_summary = complete["summary"]
    assert complete["ok"] is True, complete
    assert complete["completion_ready"] is True, complete
    assert complete["roadmap_complete"] is True, complete
    assert complete["runtime_dispatch_ready"] is False, complete
    assert complete["native_dispatch_allowed"] is False, complete
    assert complete["training_path_enabled"] is False, complete
    assert complete["product_native_ready"] is False, complete
    assert complete_summary["v2_completion_audit_requirement_count"] == 15, complete
    assert complete_summary["v2_completion_audit_passed_requirement_count"] == 15, complete
    assert complete_summary["v2_completion_audit_failed_requirement_count"] == 0, complete
    assert complete_summary["v2_completion_audit_remaining_gate_open_count"] == 0, complete
    assert complete_summary["v2_completion_audit_command_audit_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_command_audit_expected_arg_binding_count"] == 35, complete
    assert complete_summary["v2_completion_audit_command_audit_expected_arg_mismatch_count"] == 0, complete
    assert complete_summary["v2_completion_audit_command_audit_expected_path_binding_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_command_audit_phase1_handoff_post_return_command_count"] == 3, complete
    assert (
        complete_summary["v2_completion_audit_command_audit_phase1_handoff_post_return_pre_record_command_count"] == 3
    ), complete
    assert (
        complete_summary[
            "v2_completion_audit_command_audit_phase1_handoff_post_return_approval_record_command_count"
        ]
        == 0
    ), complete
    assert (
        complete_summary["v2_completion_audit_command_audit_phase1_handoff_post_return_command_match_count"] == 3
    ), complete
    assert (
        complete_summary["v2_completion_audit_command_audit_phase1_handoff_post_return_command_mismatch_count"] == 0
    ), complete
    assert complete_summary["v2_completion_audit_command_audit_phase1_handoff_post_return_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase1_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase2_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_record_input_item_count"] == 13, complete
    assert complete_summary["v2_completion_audit_record_input_phase1_item_count"] == 6, complete
    assert complete_summary["v2_completion_audit_record_input_phase2_item_count"] == 7, complete
    assert complete_summary["v2_completion_audit_phase2_refresh_inputs_tracked_count"] == 2, complete
    assert complete_summary["v2_completion_audit_record_input_checklist_shape_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_record_input_support_check_count"] == 3, complete
    assert complete_summary["v2_completion_audit_record_input_support_ready_count"] == 3, complete
    assert complete_summary["v2_completion_audit_record_input_support_blocked_item_count"] == 0, complete
    assert complete_summary["v2_completion_audit_record_input_support_blocker_count"] == 0, complete
    assert complete_summary["v2_completion_audit_record_input_support_source_binding_ready_count"] == 3, complete
    assert complete_summary["v2_completion_audit_record_input_support_source_binding_blocker_count"] == 0, complete
    assert complete_summary["v2_completion_audit_record_input_support_shape_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase1_signed_bundle_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase2_signed_bundle_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_signed_bundle_missing_signature_count"] == 0, complete
    assert complete_summary["v2_completion_audit_phase1_signed_bundle_missing_signature_count"] == 0, complete
    assert complete_summary["v2_completion_audit_phase2_signed_bundle_missing_signature_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_freshness_guard_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_signed_bundle_freshness_template_digest_match_count"] == 1, complete
    assert complete_summary["v2_completion_audit_signed_bundle_freshness_stale_signed_bundle_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_freshness_unknown_signed_entry_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_freshness_unsafe_claim_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_not_ready_entry_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_source_digest_match_count"] == 1, complete
    assert complete_summary["v2_completion_audit_signed_bundle_source_digest_stale_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_template_digest_mismatch_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_unknown_entry_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_unsigned_template_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_intake_integrity_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_signed_bundle_manual_field_check_count"] == 3, complete
    assert complete_summary["v2_completion_audit_signed_bundle_manual_field_ready_count"] == 3, complete
    assert complete_summary["v2_completion_audit_signed_bundle_manual_field_missing_count"] == 0, complete
    assert complete_summary["v2_completion_audit_signed_bundle_manual_field_shape_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase1_signed_bundle_manual_field_ready_count"] == 2, complete
    assert complete_summary["v2_completion_audit_phase2_signed_bundle_manual_field_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase1_signed_bundle_manual_shape_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase2_signed_bundle_manual_shape_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_full_signed_bundle_manual_shape_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase1_extraction_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase2_extraction_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_extraction_missing_signature_count"] == 0, complete
    assert complete_summary["v2_completion_audit_phase1_extraction_missing_signature_count"] == 0, complete
    assert complete_summary["v2_completion_audit_phase2_extraction_missing_signature_count"] == 0, complete
    assert complete_summary["v2_completion_audit_extraction_not_ready_entry_count"] == 0, complete
    assert complete_summary["v2_completion_audit_extraction_source_digest_match_count"] == 1, complete
    assert complete_summary["v2_completion_audit_extraction_source_digest_stale_count"] == 0, complete
    assert complete_summary["v2_completion_audit_extraction_template_digest_mismatch_count"] == 0, complete
    assert complete_summary["v2_completion_audit_extraction_unknown_entry_count"] == 0, complete
    assert complete_summary["v2_completion_audit_extraction_unsigned_template_count"] == 0, complete
    assert complete_summary["v2_completion_audit_extraction_integrity_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_extraction_signed_entry_digest_present_count"] == 3, complete
    assert complete_summary["v2_completion_audit_extraction_extractable_signed_entry_digest_present_count"] == 3, complete
    assert complete_summary["v2_completion_audit_extraction_digest_shape_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase1_preflight_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase2_preflight_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase1_record_chain_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase2_record_chain_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_full_record_chain_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_preflight_full_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_preflight_integrity_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_preflight_signed_bundle_valid_count"] == 1, complete
    assert complete_summary["v2_completion_audit_preflight_source_digest_match_count"] == 1, complete
    assert complete_summary["v2_completion_audit_preflight_source_digest_stale_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_template_digest_mismatch_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_unknown_entry_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_not_ready_entry_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_source_integrity_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_preflight_support_integrity_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_preflight_unsigned_template_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_signed_payload_digest_present_count"] == 3, complete
    assert complete_summary["v2_completion_audit_preflight_signed_payload_digest_missing_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_signed_bundle_entry_digest_present_count"] == 3, complete
    assert complete_summary["v2_completion_audit_preflight_signed_payload_bundle_digest_match_count"] == 3, complete
    assert complete_summary["v2_completion_audit_preflight_extracted_entry_digest_match_count"] == 3, complete
    assert complete_summary["v2_completion_audit_preflight_extracted_entry_digest_mismatch_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_extracted_entry_source_missing_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_support_ready_count"] == 3, complete
    assert complete_summary["v2_completion_audit_preflight_support_invalid_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_support_source_binding_ready_count"] == 3, complete
    assert complete_summary["v2_completion_audit_preflight_support_source_binding_blocker_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_post_record_request_field_emission_count"] == 0, complete
    assert complete_summary["v2_completion_audit_preflight_hard_fail_count"] == 0, complete
    assert complete_summary["v2_completion_audit_owner_review_recorded_count"] == 1, complete
    assert complete_summary["v2_completion_audit_product_exposure_recorded_count"] == 1, complete
    assert complete_summary["v2_completion_audit_owner_direction_recorded_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase1_record_preflight_binding_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_phase2_record_preflight_binding_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_full_record_preflight_binding_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_record_preflight_binding_ready_count"] == 3, complete
    assert complete_summary["v2_completion_audit_artifact_first_required_artifact_count"] == 12, complete
    assert complete_summary["v2_completion_audit_artifact_first_ready_artifact_count"] == 12, complete
    assert complete_summary["v2_completion_audit_artifact_first_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_artifact_first_record_input_checklist_shape_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_artifact_first_record_input_support_shape_ready_count"] == 1, complete
    assert (
        complete_summary["v2_completion_audit_artifact_first_reviewer_handoff_phase_metadata_ready_count"] == 1
    ), complete
    assert (
        complete_summary["v2_completion_audit_artifact_first_reviewer_handoff_phase1_template_entry_count"] == 2
    ), complete
    assert (
        complete_summary["v2_completion_audit_artifact_first_reviewer_handoff_phase2_deferred_signature_count"] == 1
    ), complete
    assert complete_summary["v2_completion_audit_artifact_first_reviewer_handoff_phase_shape_ready_count"] == 1, complete
    assert (
        complete_summary["v2_completion_audit_artifact_first_command_audit_expected_path_binding_ready_count"] == 1
    ), complete
    assert (
        complete_summary["v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_count"]
        == 3
    ), complete
    assert (
        complete_summary[
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_pre_record_command_count"
        ]
        == 3
    ), complete
    assert (
        complete_summary[
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_approval_record_command_count"
        ]
        == 0
    ), complete
    assert (
        complete_summary[
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_match_count"
        ]
        == 3
    ), complete
    assert (
        complete_summary[
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_mismatch_count"
        ]
        == 0
    ), complete
    assert (
        complete_summary["v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_ready_count"]
        == 1
    ), complete
    assert (
        complete_summary[
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_shape_ready_count"
        ]
        == 1
    ), complete
    assert complete_summary["v2_completion_audit_default_off_guard_count"] == 1, complete
    assert complete_summary["v2_completion_audit_completion_ready_count"] == 1, complete
    assert complete_summary["v2_completion_audit_runtime_dispatch_ready_count"] == 0, complete
    assert complete_summary["v2_completion_audit_native_dispatch_allowed_count"] == 0, complete
    assert complete_summary["v2_completion_audit_training_path_enabled_count"] == 0, complete
    assert complete_summary["v2_completion_audit_product_native_ready_count"] == 0, complete
    assert complete_summary["v2_completion_audit_unsafe_claim_count"] == 0, complete

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_completion_audit_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "deterministic_waiting_state_checked": True,
        "stale_checklist_guard_checked": True,
        "stale_command_audit_guard_checked": True,
        "stale_preflight_integrity_guard_checked": True,
        "stale_preflight_not_ready_guard_checked": True,
        "stale_record_preflight_binding_guard_checked": True,
        "stale_signed_bundle_integrity_guard_checked": True,
        "stale_signed_bundle_freshness_guard_checked": True,
        "stale_artifact_first_guard_checked": True,
        "synthetic_completion_state_checked": True,
        "summary": summary,
        "recommended_next_step": audit["recommended_next_step"],
    }


def _source_reports_waiting_for_signature() -> dict[str, dict[str, Any]]:
    return {
        "remaining_gate_handoff": {
            "summary": {
                "v2_remaining_gate_total_count": 6,
                "v2_remaining_gate_open_count": 6,
            }
        },
        "approval_command_audit": {
            "summary": {
                "v2_approval_command_audit_ready_count": 1,
                "v2_approval_command_audit_expected_arg_binding_count": 35,
                "v2_approval_command_audit_expected_arg_mismatch_count": 0,
                "v2_approval_command_audit_expected_path_binding_ready_count": 1,
                "v2_approval_command_audit_preflight_artifact_write_count": 2,
                "v2_approval_command_audit_preflight_no_artifact_blocker_count": 0,
                "v2_approval_command_audit_record_command_preflight_arg_count": 3,
                "v2_approval_command_audit_phase1_handoff_post_return_command_count": 3,
                "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count": 3,
                "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count": 0,
                "v2_approval_command_audit_phase1_handoff_post_return_command_match_count": 3,
                "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count": 0,
                "v2_approval_command_audit_phase1_handoff_post_return_ready_count": 1,
            }
        },
        "record_input_checklist": {
            "summary": {
                "v2_record_input_checklist_item_count": 13,
                "v2_record_input_checklist_phase1_item_count": 6,
                "v2_record_input_checklist_phase1_ready_count": 0,
                "v2_record_input_checklist_phase2_item_count": 7,
                "v2_record_input_checklist_phase2_ready_count": 0,
                "v2_record_input_checklist_support_check_count": 3,
                "v2_record_input_checklist_support_ready_count": 2,
                "v2_record_input_checklist_support_blocked_item_count": 1,
                "v2_record_input_checklist_support_blocker_count": 1,
                "v2_record_input_checklist_support_source_binding_ready_count": 2,
                "v2_record_input_checklist_support_source_binding_blocker_count": 0,
            }
        },
        "approval_execution_preflight": {
            "summary": {
                "v2_approval_preflight_full_ready_count": 0,
                "v2_approval_preflight_signed_bundle_valid_count": 0,
                "v2_approval_preflight_source_digest_match_count": 0,
                "v2_approval_preflight_source_digest_stale_count": 0,
                "v2_approval_preflight_template_digest_mismatch_count": 0,
                "v2_approval_preflight_unknown_entry_count": 0,
                "v2_approval_preflight_unsigned_template_count": 0,
                "v2_approval_preflight_signed_payload_digest_present_count": 0,
                "v2_approval_preflight_signed_payload_digest_missing_count": 0,
                "v2_approval_preflight_signed_bundle_entry_digest_present_count": 0,
                "v2_approval_preflight_signed_payload_bundle_digest_match_count": 0,
                "v2_approval_preflight_extracted_entry_digest_match_count": 0,
                "v2_approval_preflight_extracted_entry_digest_mismatch_count": 0,
                "v2_approval_preflight_extracted_entry_source_missing_count": 0,
                "v2_approval_preflight_support_ready_count": 0,
                "v2_approval_preflight_support_invalid_count": 0,
                "v2_approval_preflight_support_source_binding_ready_count": 0,
                "v2_approval_preflight_support_source_binding_blocker_count": 0,
                "v2_approval_preflight_post_record_request_field_emission_count": 0,
                "v2_approval_preflight_hard_fail_count": 0,
            }
        },
        "approval_state": {
            "summary": {
                "v2_approval_state_remaining_gate_open_count": 6,
                "v2_approval_state_recorded_stage_count": 0,
                "v2_approval_state_phase1_signed_bundle_ready_count": 0,
                "v2_approval_state_phase2_signed_bundle_ready_count": 0,
                "v2_approval_state_signed_bundle_missing_signature_count": 3,
                "v2_approval_state_phase1_signed_bundle_missing_signature_count": 2,
                "v2_approval_state_phase2_signed_bundle_missing_signature_count": 1,
                "v2_approval_state_signed_bundle_freshness_guard_ready_count": 1,
                "v2_approval_state_signed_bundle_freshness_template_digest_match_count": 1,
                "v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count": 0,
                "v2_approval_state_signed_bundle_freshness_unknown_signed_entry_count": 0,
                "v2_approval_state_signed_bundle_freshness_unsafe_claim_count": 0,
                "v2_approval_state_signed_bundle_source_digest_match_count": 0,
                "v2_approval_state_signed_bundle_source_digest_stale_count": 0,
                "v2_approval_state_signed_bundle_template_digest_mismatch_count": 0,
                "v2_approval_state_signed_bundle_unknown_entry_count": 0,
                "v2_approval_state_signed_bundle_unsigned_template_count": 0,
                "v2_approval_state_signed_bundle_intake_integrity_ready_count": 0,
                "v2_approval_state_signed_bundle_manual_field_check_count": 0,
                "v2_approval_state_signed_bundle_manual_field_ready_count": 0,
                "v2_approval_state_signed_bundle_manual_field_missing_count": 0,
                "v2_approval_state_signed_bundle_manual_field_shape_ready_count": 0,
                "v2_approval_state_phase1_signed_bundle_manual_field_ready_count": 0,
                "v2_approval_state_phase2_signed_bundle_manual_field_ready_count": 0,
                "v2_approval_state_phase1_signed_bundle_manual_shape_ready_count": 0,
                "v2_approval_state_phase2_signed_bundle_manual_shape_ready_count": 0,
                "v2_approval_state_full_signed_bundle_manual_shape_ready_count": 0,
                "v2_approval_state_phase1_extraction_ready_count": 0,
                "v2_approval_state_phase2_extraction_ready_count": 0,
                "v2_approval_state_extraction_missing_signature_count": 3,
                "v2_approval_state_phase1_extraction_missing_signature_count": 2,
                "v2_approval_state_phase2_extraction_missing_signature_count": 1,
                "v2_approval_state_extraction_source_digest_match_count": 0,
                "v2_approval_state_extraction_source_digest_stale_count": 0,
                "v2_approval_state_extraction_template_digest_mismatch_count": 0,
                "v2_approval_state_extraction_unknown_entry_count": 0,
                "v2_approval_state_extraction_unsigned_template_count": 0,
                "v2_approval_state_extraction_integrity_ready_count": 0,
                "v2_approval_state_extraction_signed_entry_digest_present_count": 0,
                "v2_approval_state_extraction_extractable_signed_entry_digest_present_count": 0,
                "v2_approval_state_extraction_digest_shape_ready_count": 0,
                "v2_approval_state_phase1_preflight_ready_count": 0,
                "v2_approval_state_phase2_preflight_ready_count": 0,
                "v2_approval_state_preflight_source_integrity_ready_count": 0,
                "v2_approval_state_preflight_support_integrity_ready_count": 0,
                "v2_approval_state_preflight_support_source_binding_ready_count": 0,
                "v2_approval_state_preflight_support_source_binding_blocker_count": 0,
                "v2_approval_state_phase1_record_chain_ready_count": 0,
                "v2_approval_state_phase2_record_chain_ready_count": 0,
                "v2_approval_state_full_record_chain_ready_count": 0,
                "v2_approval_state_owner_review_recorded_count": 0,
                "v2_approval_state_product_exposure_recorded_count": 0,
                "v2_approval_state_owner_direction_recorded_count": 0,
                "v2_approval_state_phase1_record_preflight_binding_ready_count": 0,
                "v2_approval_state_phase2_record_preflight_binding_ready_count": 0,
                "v2_approval_state_full_record_preflight_binding_ready_count": 0,
                "v2_approval_state_record_preflight_binding_ready_count": 0,
            }
        },
        "release_artifact_first_validation": {
            "summary": _artifact_first_release_summary(),
        },
    }


def _source_reports_complete_default_off() -> dict[str, dict[str, Any]]:
    return {
        "remaining_gate_handoff": {
            "summary": {
                "v2_remaining_gate_total_count": 6,
                "v2_remaining_gate_open_count": 0,
            }
        },
        "approval_command_audit": {
            "summary": {
                "v2_approval_command_audit_ready_count": 1,
                "v2_approval_command_audit_expected_arg_binding_count": 35,
                "v2_approval_command_audit_expected_arg_mismatch_count": 0,
                "v2_approval_command_audit_expected_path_binding_ready_count": 1,
                "v2_approval_command_audit_preflight_artifact_write_count": 2,
                "v2_approval_command_audit_preflight_no_artifact_blocker_count": 0,
                "v2_approval_command_audit_record_command_preflight_arg_count": 3,
                "v2_approval_command_audit_phase1_handoff_post_return_command_count": 3,
                "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count": 3,
                "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count": 0,
                "v2_approval_command_audit_phase1_handoff_post_return_command_match_count": 3,
                "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count": 0,
                "v2_approval_command_audit_phase1_handoff_post_return_ready_count": 1,
            }
        },
        "record_input_checklist": {
            "summary": {
                "v2_record_input_checklist_item_count": 13,
                "v2_record_input_checklist_phase1_item_count": 6,
                "v2_record_input_checklist_phase1_ready_count": 1,
                "v2_record_input_checklist_phase2_item_count": 7,
                "v2_record_input_checklist_phase2_ready_count": 1,
                "v2_record_input_checklist_support_check_count": 3,
                "v2_record_input_checklist_support_ready_count": 3,
                "v2_record_input_checklist_support_blocked_item_count": 0,
                "v2_record_input_checklist_support_blocker_count": 0,
                "v2_record_input_checklist_support_source_binding_ready_count": 3,
                "v2_record_input_checklist_support_source_binding_blocker_count": 0,
            }
        },
        "approval_execution_preflight": {
            "summary": {
                "v2_approval_preflight_full_ready_count": 1,
                "v2_approval_preflight_signed_bundle_valid_count": 1,
                "v2_approval_preflight_source_digest_match_count": 1,
                "v2_approval_preflight_source_digest_stale_count": 0,
                "v2_approval_preflight_template_digest_mismatch_count": 0,
                "v2_approval_preflight_unknown_entry_count": 0,
                "v2_approval_preflight_unsigned_template_count": 0,
                "v2_approval_preflight_signed_payload_digest_present_count": 3,
                "v2_approval_preflight_signed_payload_digest_missing_count": 0,
                "v2_approval_preflight_signed_bundle_entry_digest_present_count": 3,
                "v2_approval_preflight_signed_payload_bundle_digest_match_count": 3,
                "v2_approval_preflight_extracted_entry_digest_match_count": 3,
                "v2_approval_preflight_extracted_entry_digest_mismatch_count": 0,
                "v2_approval_preflight_extracted_entry_source_missing_count": 0,
                "v2_approval_preflight_support_ready_count": 3,
                "v2_approval_preflight_support_invalid_count": 0,
                "v2_approval_preflight_support_source_binding_ready_count": 3,
                "v2_approval_preflight_support_source_binding_blocker_count": 0,
                "v2_approval_preflight_post_record_request_field_emission_count": 0,
                "v2_approval_preflight_hard_fail_count": 0,
            }
        },
        "approval_state": {
            "summary": {
                "v2_approval_state_remaining_gate_open_count": 0,
                "v2_approval_state_recorded_stage_count": 3,
                "v2_approval_state_phase1_signed_bundle_ready_count": 1,
                "v2_approval_state_phase2_signed_bundle_ready_count": 1,
                "v2_approval_state_signed_bundle_missing_signature_count": 0,
                "v2_approval_state_phase1_signed_bundle_missing_signature_count": 0,
                "v2_approval_state_phase2_signed_bundle_missing_signature_count": 0,
                "v2_approval_state_signed_bundle_freshness_guard_ready_count": 1,
                "v2_approval_state_signed_bundle_freshness_template_digest_match_count": 1,
                "v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count": 0,
                "v2_approval_state_signed_bundle_freshness_unknown_signed_entry_count": 0,
                "v2_approval_state_signed_bundle_freshness_unsafe_claim_count": 0,
                "v2_approval_state_signed_bundle_source_digest_match_count": 1,
                "v2_approval_state_signed_bundle_source_digest_stale_count": 0,
                "v2_approval_state_signed_bundle_template_digest_mismatch_count": 0,
                "v2_approval_state_signed_bundle_unknown_entry_count": 0,
                "v2_approval_state_signed_bundle_unsigned_template_count": 0,
                "v2_approval_state_signed_bundle_intake_integrity_ready_count": 1,
                "v2_approval_state_signed_bundle_manual_field_check_count": 3,
                "v2_approval_state_signed_bundle_manual_field_ready_count": 3,
                "v2_approval_state_signed_bundle_manual_field_missing_count": 0,
                "v2_approval_state_signed_bundle_manual_field_shape_ready_count": 1,
                "v2_approval_state_phase1_signed_bundle_manual_field_ready_count": 2,
                "v2_approval_state_phase2_signed_bundle_manual_field_ready_count": 1,
                "v2_approval_state_phase1_signed_bundle_manual_shape_ready_count": 1,
                "v2_approval_state_phase2_signed_bundle_manual_shape_ready_count": 1,
                "v2_approval_state_full_signed_bundle_manual_shape_ready_count": 1,
                "v2_approval_state_phase1_extraction_ready_count": 1,
                "v2_approval_state_phase2_extraction_ready_count": 1,
                "v2_approval_state_extraction_missing_signature_count": 0,
                "v2_approval_state_phase1_extraction_missing_signature_count": 0,
                "v2_approval_state_phase2_extraction_missing_signature_count": 0,
                "v2_approval_state_extraction_source_digest_match_count": 1,
                "v2_approval_state_extraction_source_digest_stale_count": 0,
                "v2_approval_state_extraction_template_digest_mismatch_count": 0,
                "v2_approval_state_extraction_unknown_entry_count": 0,
                "v2_approval_state_extraction_unsigned_template_count": 0,
                "v2_approval_state_extraction_integrity_ready_count": 1,
                "v2_approval_state_extraction_signed_entry_digest_present_count": 3,
                "v2_approval_state_extraction_extractable_signed_entry_digest_present_count": 3,
                "v2_approval_state_extraction_digest_shape_ready_count": 1,
                "v2_approval_state_phase1_preflight_ready_count": 1,
                "v2_approval_state_phase2_preflight_ready_count": 1,
                "v2_approval_state_preflight_source_integrity_ready_count": 1,
                "v2_approval_state_preflight_support_integrity_ready_count": 1,
                "v2_approval_state_preflight_support_source_binding_ready_count": 3,
                "v2_approval_state_preflight_support_source_binding_blocker_count": 0,
                "v2_approval_state_phase1_record_chain_ready_count": 1,
                "v2_approval_state_phase2_record_chain_ready_count": 1,
                "v2_approval_state_full_record_chain_ready_count": 1,
                "v2_approval_state_owner_review_recorded_count": 1,
                "v2_approval_state_product_exposure_recorded_count": 1,
                "v2_approval_state_owner_direction_recorded_count": 1,
                "v2_approval_state_phase1_record_preflight_binding_ready_count": 1,
                "v2_approval_state_phase2_record_preflight_binding_ready_count": 1,
                "v2_approval_state_full_record_preflight_binding_ready_count": 1,
                "v2_approval_state_record_preflight_binding_ready_count": 3,
            }
        },
        "release_artifact_first_validation": {
            "summary": _artifact_first_release_summary(),
        },
    }


def _artifact_first_release_summary() -> dict[str, int]:
    return {
        "release_artifact_first_required_artifact_count": 12,
        "release_artifact_first_ready_artifact_count": 12,
        "release_artifact_first_missing_artifact_count": 0,
        "release_artifact_first_parse_error_count": 0,
        "release_artifact_first_v2_record_input_checklist_shape_ready_count": 1,
        "release_artifact_first_v2_record_input_support_shape_ready_count": 1,
        "release_artifact_first_v2_reviewer_handoff_phase_metadata_ready_count": 1,
        "release_artifact_first_v2_reviewer_handoff_phase1_template_entry_count": 2,
        "release_artifact_first_v2_reviewer_handoff_phase2_deferred_signature_count": 1,
        "release_artifact_first_v2_reviewer_handoff_phase_shape_ready_count": 1,
        "release_artifact_first_v2_command_audit_expected_path_binding_ready_count": 1,
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_count": 3,
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_pre_record_command_count": 3,
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_approval_record_command_count": 0,
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_match_count": 3,
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_mismatch_count": 0,
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_ready_count": 1,
        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_shape_ready_count": 1,
        "release_artifact_first_validation_ready_count": 1,
        "release_artifact_first_runtime_dispatch_ready_count": 0,
        "release_artifact_first_native_dispatch_allowed_count": 0,
        "release_artifact_first_training_path_enabled_count": 0,
        "release_artifact_first_default_behavior_changed_count": 0,
        "release_artifact_first_product_native_ready_count": 0,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
