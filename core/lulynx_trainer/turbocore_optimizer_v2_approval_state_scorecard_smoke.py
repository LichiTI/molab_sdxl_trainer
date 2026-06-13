"""Smoke checks for the v2 approval-state scorecard."""

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

from core.turbocore_optimizer_v2_approval_state_scorecard import (  # noqa: E402
    ARTIFACT,
    build_optimizer_v2_approval_state_scorecard,
)


def run_smoke() -> dict[str, Any]:
    artifact_payload = build_optimizer_v2_approval_state_scorecard(write_artifact=True)
    assert artifact_payload["ok"] is True, artifact_payload
    assert artifact_payload["approval_recorded"] is False, artifact_payload
    assert artifact_payload["runtime_dispatch_ready"] is False, artifact_payload
    assert artifact_payload["native_dispatch_allowed"] is False, artifact_payload
    assert artifact_payload["training_path_enabled"] is False, artifact_payload
    assert artifact_payload["product_native_ready"] is False, artifact_payload
    assert ARTIFACT.exists(), ARTIFACT

    scorecard = build_optimizer_v2_approval_state_scorecard(
        source_reports=_source_reports_waiting_for_signature(),
        write_artifact=False,
    )
    summary = scorecard["summary"]
    stage_states = {stage["id"]: stage["state"] for stage in scorecard["stages"]}
    assert scorecard["ok"] is True, scorecard
    assert scorecard["approval_recorded"] is False, scorecard
    assert scorecard["runtime_dispatch_ready"] is False, scorecard
    assert scorecard["native_dispatch_allowed"] is False, scorecard
    assert scorecard["training_path_enabled"] is False, scorecard
    assert scorecard["product_native_ready"] is False, scorecard
    assert summary["v2_approval_state_stage_count"] == 15, scorecard
    assert summary["v2_approval_state_ready_stage_count"] == 6, scorecard
    assert summary["v2_approval_state_waiting_signature_stage_count"] == 8, scorecard
    assert summary["v2_approval_state_ready_for_record_stage_count"] == 0, scorecard
    assert summary["v2_approval_state_recorded_stage_count"] == 0, scorecard
    assert summary["v2_approval_state_missing_stage_count"] == 0, scorecard
    assert summary["v2_approval_state_blocked_stage_count"] == 1, scorecard
    assert summary["v2_approval_state_remaining_gate_open_count"] == 6, scorecard
    assert summary["v2_approval_state_signature_ready_count"] == 2, scorecard
    assert summary["v2_approval_state_signed_bundle_present_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_freshness_guard_ready_count"] == 1, scorecard
    assert summary["v2_approval_state_signed_bundle_freshness_template_digest_match_count"] == 1, scorecard
    assert summary["v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_freshness_unknown_signed_entry_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_freshness_unsafe_claim_count"] == 0, scorecard
    assert summary["v2_approval_state_phase1_signed_bundle_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase2_signed_bundle_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_full_signed_bundle_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_missing_signature_count"] == 3, scorecard
    assert summary["v2_approval_state_phase1_signed_bundle_missing_signature_count"] == 2, scorecard
    assert summary["v2_approval_state_phase2_signed_bundle_missing_signature_count"] == 1, scorecard
    assert summary["v2_approval_state_signed_bundle_not_ready_entry_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_source_digest_match_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_source_digest_stale_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_template_digest_mismatch_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_unknown_entry_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_unsigned_template_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_intake_integrity_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_manual_field_check_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_manual_field_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_manual_field_missing_count"] == 0, scorecard
    assert summary["v2_approval_state_signed_bundle_manual_field_shape_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase1_signed_bundle_manual_field_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase2_signed_bundle_manual_field_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase1_signed_bundle_manual_shape_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase2_signed_bundle_manual_shape_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_full_signed_bundle_manual_shape_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase1_extraction_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase2_extraction_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_full_extraction_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_missing_signature_count"] == 3, scorecard
    assert summary["v2_approval_state_phase1_extraction_missing_signature_count"] == 2, scorecard
    assert summary["v2_approval_state_phase2_extraction_missing_signature_count"] == 1, scorecard
    assert summary["v2_approval_state_extraction_not_ready_entry_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_source_digest_match_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_source_digest_stale_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_template_digest_mismatch_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_unknown_entry_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_unsigned_template_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_integrity_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_signed_entry_digest_present_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_extractable_signed_entry_digest_present_count"] == 0, scorecard
    assert summary["v2_approval_state_extraction_digest_shape_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_command_audit_ready_count"] == 1, scorecard
    assert summary["v2_approval_state_command_audit_expected_arg_binding_count"] == 35, scorecard
    assert summary["v2_approval_state_command_audit_expected_arg_mismatch_count"] == 0, scorecard
    assert summary["v2_approval_state_command_audit_expected_path_binding_ready_count"] == 1, scorecard
    assert summary["v2_approval_state_command_audit_preflight_artifact_write_count"] == 2, scorecard
    assert summary["v2_approval_state_command_audit_preflight_no_artifact_blocker_count"] == 0, scorecard
    assert summary["v2_approval_state_command_audit_record_command_preflight_arg_count"] == 3, scorecard
    assert summary["v2_approval_state_command_audit_phase1_handoff_post_return_command_count"] == 3, scorecard
    assert summary["v2_approval_state_command_audit_phase1_handoff_post_return_pre_record_command_count"] == 3, (
        scorecard
    )
    assert summary["v2_approval_state_command_audit_phase1_handoff_post_return_approval_record_command_count"] == 0, (
        scorecard
    )
    assert summary["v2_approval_state_command_audit_phase1_handoff_post_return_command_match_count"] == 3, scorecard
    assert summary["v2_approval_state_command_audit_phase1_handoff_post_return_command_mismatch_count"] == 0, (
        scorecard
    )
    assert summary["v2_approval_state_command_audit_phase1_handoff_post_return_ready_count"] == 1, scorecard
    assert summary["v2_approval_state_record_input_checklist_ready_count"] == 1, scorecard
    assert summary["v2_approval_state_record_input_support_check_count"] == 3, scorecard
    assert summary["v2_approval_state_record_input_support_ready_count"] == 2, scorecard
    assert summary["v2_approval_state_record_input_support_blocked_item_count"] == 1, scorecard
    assert summary["v2_approval_state_record_input_support_blocker_count"] == 1, scorecard
    assert summary["v2_approval_state_record_input_support_source_binding_ready_count"] == 2, scorecard
    assert summary["v2_approval_state_record_input_support_source_binding_blocker_count"] == 0, scorecard
    assert summary["v2_approval_state_record_input_support_shape_ready_count"] == 1, scorecard
    assert summary["v2_approval_state_phase1_preflight_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase2_preflight_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_full_preflight_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_not_ready_entry_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_source_digest_match_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_source_digest_stale_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_template_digest_mismatch_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_unknown_entry_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_unsigned_template_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_source_integrity_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_support_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_support_invalid_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_support_source_binding_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_support_source_binding_blocker_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_post_record_request_field_emission_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_hard_fail_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_support_integrity_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase1_preflight_support_integrity_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase2_preflight_support_integrity_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_signed_payload_digest_present_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_signed_payload_digest_missing_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_signed_bundle_entry_digest_present_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_signed_payload_bundle_digest_match_count"] == 0, scorecard
    assert summary["v2_approval_state_preflight_digest_shape_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase1_record_chain_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_phase2_record_chain_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_full_record_chain_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_owner_review_recorded_count"] == 0, scorecard
    assert summary["v2_approval_state_product_exposure_recorded_count"] == 0, scorecard
    assert summary["v2_approval_state_owner_direction_recorded_count"] == 0, scorecard
    assert summary["v2_approval_state_approval_recorded_count"] == 0, scorecard
    assert summary["v2_approval_state_runtime_dispatch_ready_count"] == 0, scorecard
    assert summary["v2_approval_state_native_dispatch_allowed_count"] == 0, scorecard
    assert summary["v2_approval_state_training_path_enabled_count"] == 0, scorecard
    assert summary["v2_approval_state_product_native_ready_count"] == 0, scorecard
    assert stage_states["reviewer_handoff"] == "waiting_for_external_signature", scorecard
    assert stage_states["signed_bundle_intake"] == "waiting_for_external_signature", scorecard
    assert stage_states["owner_release_direction_packet"] == "waiting_for_external_signature", scorecard

    phase1_scorecard = build_optimizer_v2_approval_state_scorecard(
        source_reports=_source_reports_phase1_ready_for_record(),
        write_artifact=False,
    )
    phase1_summary = phase1_scorecard["summary"]
    assert phase1_scorecard["ok"] is True, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase1_signed_bundle_ready_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_signed_bundle_missing_signature_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase1_signed_bundle_missing_signature_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase2_signed_bundle_missing_signature_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_signed_bundle_not_ready_entry_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_signed_bundle_source_digest_match_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_signed_bundle_intake_integrity_ready_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_signed_bundle_manual_field_check_count"] == 2, phase1_scorecard
    assert phase1_summary["v2_approval_state_signed_bundle_manual_field_ready_count"] == 2, phase1_scorecard
    assert phase1_summary["v2_approval_state_signed_bundle_manual_field_missing_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase1_signed_bundle_manual_field_ready_count"] == 2, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase1_signed_bundle_manual_shape_ready_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase2_signed_bundle_manual_shape_ready_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_full_signed_bundle_manual_shape_ready_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase1_extraction_ready_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_extraction_missing_signature_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase1_extraction_missing_signature_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase2_extraction_missing_signature_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_extraction_not_ready_entry_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_extraction_source_digest_match_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_extraction_integrity_ready_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_extraction_signed_entry_digest_present_count"] == 2, (
        phase1_scorecard
    )
    assert phase1_summary["v2_approval_state_extraction_extractable_signed_entry_digest_present_count"] == 2, (
        phase1_scorecard
    )
    assert phase1_summary["v2_approval_state_phase1_preflight_ready_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_preflight_not_ready_entry_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_preflight_source_digest_match_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_preflight_source_integrity_ready_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_preflight_support_ready_count"] == 2, phase1_scorecard
    assert phase1_summary["v2_approval_state_preflight_support_source_binding_ready_count"] == 2, (
        phase1_scorecard
    )
    assert phase1_summary["v2_approval_state_preflight_support_source_binding_blocker_count"] == 0, (
        phase1_scorecard
    )
    assert phase1_summary["v2_approval_state_preflight_support_integrity_ready_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase1_preflight_support_integrity_ready_count"] == 1, (
        phase1_scorecard
    )
    assert phase1_summary["v2_approval_state_phase2_preflight_support_integrity_ready_count"] == 0, (
        phase1_scorecard
    )
    assert phase1_summary["v2_approval_state_preflight_signed_payload_digest_present_count"] == 2, (
        phase1_scorecard
    )
    assert phase1_summary["v2_approval_state_preflight_signed_payload_digest_missing_count"] == 1, (
        phase1_scorecard
    )
    assert phase1_summary["v2_approval_state_preflight_signed_payload_bundle_digest_match_count"] == 2, (
        phase1_scorecard
    )
    assert phase1_summary["v2_approval_state_phase1_record_chain_ready_count"] == 1, phase1_scorecard
    assert phase1_summary["v2_approval_state_phase2_record_chain_ready_count"] == 0, phase1_scorecard
    assert phase1_summary["v2_approval_state_full_record_chain_ready_count"] == 0, phase1_scorecard

    stale_manual_fields = build_optimizer_v2_approval_state_scorecard(
        source_reports={
            **_source_reports_phase1_ready_for_record(),
            "signed_bundle_intake": {
                "summary": {
                    "v2_signed_bundle_present_count": 1,
                    "v2_signed_bundle_valid_record_count": 2,
                    "v2_signed_bundle_phase1_ready_count": 1,
                    "v2_signed_bundle_phase2_ready_count": 0,
                    "v2_signed_bundle_full_ready_count": 0,
                }
            },
        },
        write_artifact=False,
    )
    stale_manual_summary = stale_manual_fields["summary"]
    assert stale_manual_fields["ok"] is True, stale_manual_fields
    assert stale_manual_summary["v2_approval_state_phase1_signed_bundle_ready_count"] == 1, stale_manual_fields
    assert stale_manual_summary["v2_approval_state_phase1_signed_bundle_manual_shape_ready_count"] == 0, (
        stale_manual_fields
    )
    assert stale_manual_summary["v2_approval_state_phase1_record_chain_ready_count"] == 0, (
        stale_manual_fields
    )

    stale_command_audit = build_optimizer_v2_approval_state_scorecard(
        source_reports={
            **_source_reports_waiting_for_signature(),
            "approval_command_audit": {
                "summary": {
                    "v2_approval_command_audit_ready_count": 1,
                    "v2_approval_command_audit_missing_entrypoint_count": 0,
                    "v2_approval_command_audit_record_before_preflight_count": 0,
                    "v2_approval_command_audit_phase_marker_valid_count": 1,
                    "v2_approval_command_audit_expected_arg_binding_count": 35,
                    "v2_approval_command_audit_expected_arg_mismatch_count": 1,
                    "v2_approval_command_audit_expected_path_binding_ready_count": 0,
                    "v2_approval_command_audit_preflight_artifact_write_count": 2,
                    "v2_approval_command_audit_preflight_no_artifact_blocker_count": 0,
                    "v2_approval_command_audit_record_command_preflight_arg_count": 3,
                }
            },
        },
        write_artifact=False,
    )
    stale_summary = stale_command_audit["summary"]
    stale_stage_states = {stage["id"]: stage["state"] for stage in stale_command_audit["stages"]}
    assert stale_command_audit["ok"] is True, stale_command_audit
    assert stale_stage_states["approval_command_audit"] == "blocked", stale_command_audit
    assert stale_summary["v2_approval_state_ready_stage_count"] == 5, stale_command_audit
    assert stale_summary["v2_approval_state_blocked_stage_count"] == 2, stale_command_audit
    assert stale_summary["v2_approval_state_command_audit_expected_arg_mismatch_count"] == 1, (
        stale_command_audit
    )
    assert stale_summary["v2_approval_state_command_audit_expected_path_binding_ready_count"] == 0, (
        stale_command_audit
    )
    assert "v2_approval_command_audit_not_ready" in stale_command_audit["blocked_reasons"], (
        stale_command_audit
    )

    stale_command_handoff = build_optimizer_v2_approval_state_scorecard(
        source_reports={
            **_source_reports_waiting_for_signature(),
            "approval_command_audit": {
                "summary": {
                    "v2_approval_command_audit_ready_count": 1,
                    "v2_approval_command_audit_missing_entrypoint_count": 0,
                    "v2_approval_command_audit_record_before_preflight_count": 0,
                    "v2_approval_command_audit_phase_marker_valid_count": 1,
                    "v2_approval_command_audit_expected_arg_binding_count": 35,
                    "v2_approval_command_audit_expected_arg_mismatch_count": 0,
                    "v2_approval_command_audit_expected_path_binding_ready_count": 1,
                    "v2_approval_command_audit_preflight_artifact_write_count": 2,
                    "v2_approval_command_audit_preflight_no_artifact_blocker_count": 0,
                    "v2_approval_command_audit_record_command_preflight_arg_count": 3,
                }
            },
        },
        write_artifact=False,
    )
    stale_handoff_summary = stale_command_handoff["summary"]
    stale_handoff_stage_states = {stage["id"]: stage["state"] for stage in stale_command_handoff["stages"]}
    assert stale_command_handoff["ok"] is True, stale_command_handoff
    assert stale_handoff_stage_states["approval_command_audit"] == "blocked", stale_command_handoff
    assert stale_handoff_summary["v2_approval_state_command_audit_expected_path_binding_ready_count"] == 1, (
        stale_command_handoff
    )
    assert stale_handoff_summary["v2_approval_state_command_audit_phase1_handoff_post_return_ready_count"] == 0, (
        stale_command_handoff
    )
    assert "v2_approval_command_audit_not_ready" in stale_command_handoff["blocked_reasons"], (
        stale_command_handoff
    )

    stale_record_input = build_optimizer_v2_approval_state_scorecard(
        source_reports={
            **_source_reports_waiting_for_signature(),
            "record_input_checklist": {
                "summary": {
                    "v2_record_input_checklist_artifact_ready_count": 1,
                    "v2_record_input_checklist_unsafe_claim_count": 0,
                }
            },
        },
        write_artifact=False,
    )
    stale_record_summary = stale_record_input["summary"]
    stale_record_stage_states = {stage["id"]: stage["state"] for stage in stale_record_input["stages"]}
    assert stale_record_input["ok"] is True, stale_record_input
    assert stale_record_stage_states["record_input_checklist"] == "blocked", stale_record_input
    assert stale_record_summary["v2_approval_state_ready_stage_count"] == 5, stale_record_input
    assert stale_record_summary["v2_approval_state_blocked_stage_count"] == 2, stale_record_input
    assert stale_record_summary["v2_approval_state_record_input_checklist_ready_count"] == 0, stale_record_input
    assert stale_record_summary["v2_approval_state_record_input_support_shape_ready_count"] == 0, stale_record_input
    assert "v2_record_input_checklist_not_ready" in stale_record_input["blocked_reasons"], stale_record_input

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_approval_state_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "deterministic_waiting_state_checked": True,
        "stale_command_audit_path_binding_guard_checked": True,
        "stale_record_input_support_shape_guard_checked": True,
        "stale_signed_bundle_manual_field_shape_guard_checked": True,
        "summary": scorecard["summary"],
        "recommended_next_step": scorecard["recommended_next_step"],
    }


def _source_reports_waiting_for_signature() -> dict[str, dict[str, Any]]:
    return {
        "remaining_gate_handoff": {
            "summary": {
                "v2_remaining_gate_handoff_ready_count": 1,
                "v2_remaining_gate_open_count": 6,
            }
        },
        "signature_bundle": {
            "summary": {
                "v2_signature_bundle_ready_for_signature_count": 2,
            }
        },
        "reviewer_handoff": {
            "summary": {
                "v2_reviewer_handoff_packet_ready_count": 1,
            }
        },
        "approval_execution_plan": {
            "summary": {
                "v2_approval_execution_step_count": 16,
                "v2_approval_execution_phase1_step_count": 6,
                "v2_approval_execution_phase2_step_count": 9,
                "v2_approval_execution_plan_ready_count": 1,
            }
        },
        "approval_command_audit": {
            "summary": {
                "v2_approval_command_audit_ready_count": 1,
                "v2_approval_command_audit_missing_entrypoint_count": 0,
                "v2_approval_command_audit_record_before_preflight_count": 0,
                "v2_approval_command_audit_phase_marker_valid_count": 1,
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
                "v2_record_input_checklist_artifact_ready_count": 1,
                "v2_record_input_checklist_support_check_count": 3,
                "v2_record_input_checklist_support_ready_count": 2,
                "v2_record_input_checklist_support_blocked_item_count": 1,
                "v2_record_input_checklist_support_blocker_count": 1,
                "v2_record_input_checklist_support_source_binding_ready_count": 2,
                "v2_record_input_checklist_support_source_binding_blocker_count": 0,
                "v2_record_input_checklist_unsafe_claim_count": 0,
            }
        },
        "signed_bundle_freshness": {
            "summary": {
                "v2_signed_bundle_freshness_guard_ready_count": 1,
                "v2_signed_bundle_freshness_current_digest_present_count": 1,
                "v2_signed_bundle_freshness_template_digest_match_count": 1,
                "v2_signed_bundle_freshness_stale_signed_bundle_count": 0,
                "v2_signed_bundle_freshness_unknown_signed_entry_count": 0,
                "v2_signed_bundle_freshness_unsafe_claim_count": 0,
            }
        },
        "signed_bundle_intake": {
            "summary": {
                "v2_signed_bundle_present_count": 0,
                "v2_signed_bundle_valid_record_count": 0,
                "v2_signed_bundle_phase1_ready_count": 0,
                "v2_signed_bundle_phase2_ready_count": 0,
                "v2_signed_bundle_full_ready_count": 0,
                "v2_signed_bundle_missing_signature_count": 3,
                "v2_signed_bundle_phase1_missing_signature_count": 2,
                "v2_signed_bundle_phase2_missing_signature_count": 1,
                "v2_signed_bundle_source_digest_match_count": 0,
                "v2_signed_bundle_source_digest_stale_count": 0,
                "v2_signed_bundle_template_digest_mismatch_count": 0,
                "v2_signed_bundle_unknown_entry_count": 0,
                "v2_signed_bundle_unsigned_template_count": 0,
                "v2_signed_bundle_manual_field_check_count": 0,
                "v2_signed_bundle_manual_field_ready_count": 0,
                "v2_signed_bundle_manual_field_missing_count": 0,
                "v2_signed_bundle_manual_field_shape_ready_count": 0,
                "v2_signed_bundle_phase1_manual_field_ready_count": 0,
                "v2_signed_bundle_phase2_manual_field_ready_count": 0,
            },
            "blocked_reasons": ["signed_bundle_missing"],
        },
        "signed_bundle_extraction": {
            "summary": {
                "v2_signed_bundle_extraction_ready_for_record_count": 0,
                "v2_signed_bundle_extraction_phase1_ready_for_record_count": 0,
                "v2_signed_bundle_extraction_phase2_ready_for_record_count": 0,
                "v2_signed_bundle_extraction_full_ready_for_record_count": 0,
                "v2_signed_bundle_extraction_missing_signature_count": 3,
                "v2_signed_bundle_extraction_phase1_missing_signature_count": 2,
                "v2_signed_bundle_extraction_phase2_missing_signature_count": 1,
                "v2_signed_bundle_extraction_source_digest_match_count": 0,
                "v2_signed_bundle_extraction_source_digest_stale_count": 0,
                "v2_signed_bundle_extraction_template_digest_mismatch_count": 0,
                "v2_signed_bundle_extraction_unknown_entry_count": 0,
                "v2_signed_bundle_extraction_unsigned_template_count": 0,
                "v2_signed_bundle_extraction_signed_entry_digest_present_count": 0,
                "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count": 0,
            },
            "blocked_reasons": ["signed_bundle_missing"],
        },
        "approval_execution_preflight": {
            "summary": {
                "v2_approval_preflight_phase1_ready_count": 0,
                "v2_approval_preflight_phase2_ready_count": 0,
                "v2_approval_preflight_full_ready_count": 0,
                "v2_approval_preflight_source_digest_match_count": 0,
                "v2_approval_preflight_source_digest_stale_count": 0,
                "v2_approval_preflight_template_digest_mismatch_count": 0,
                "v2_approval_preflight_unknown_entry_count": 0,
                "v2_approval_preflight_unsigned_template_count": 0,
                "v2_approval_preflight_support_ready_count": 0,
                "v2_approval_preflight_support_invalid_count": 0,
                "v2_approval_preflight_support_source_binding_ready_count": 0,
                "v2_approval_preflight_support_source_binding_blocker_count": 0,
                "v2_approval_preflight_post_record_request_field_emission_count": 0,
                "v2_approval_preflight_hard_fail_count": 0,
                "v2_approval_preflight_signed_payload_digest_present_count": 0,
                "v2_approval_preflight_signed_payload_digest_missing_count": 0,
                "v2_approval_preflight_signed_bundle_entry_digest_present_count": 0,
                "v2_approval_preflight_signed_payload_bundle_digest_match_count": 0,
            },
            "blocked_reasons": ["signed_bundle_missing"],
        },
        "owner_release_review_record": {"release_review_recorded": False},
        "product_exposure_decision": {
            "ready_for_product_exposure_review": True,
            "product_exposure_decision_recorded": False,
        },
        "release_review_archive": {"archive_ready": False},
        "owner_release_direction_packet": {
            "summary": {
                "owner_release_direction_ready_for_signature_count": 1,
            }
        },
        "owner_release_direction_record": {"owner_release_direction_recorded": False},
    }


def _source_reports_phase1_ready_for_record() -> dict[str, dict[str, Any]]:
    reports = _source_reports_waiting_for_signature()
    reports["signed_bundle_intake"] = {
        "summary": {
            "v2_signed_bundle_present_count": 1,
            "v2_signed_bundle_valid_record_count": 2,
            "v2_signed_bundle_phase1_ready_count": 1,
            "v2_signed_bundle_phase2_ready_count": 0,
            "v2_signed_bundle_full_ready_count": 0,
            "v2_signed_bundle_missing_signature_count": 1,
            "v2_signed_bundle_phase1_missing_signature_count": 0,
            "v2_signed_bundle_phase2_missing_signature_count": 1,
            "v2_signed_bundle_source_digest_match_count": 1,
            "v2_signed_bundle_source_digest_stale_count": 0,
            "v2_signed_bundle_template_digest_mismatch_count": 0,
            "v2_signed_bundle_unknown_entry_count": 0,
            "v2_signed_bundle_unsigned_template_count": 0,
            "v2_signed_bundle_manual_field_check_count": 2,
            "v2_signed_bundle_manual_field_ready_count": 2,
            "v2_signed_bundle_manual_field_missing_count": 0,
            "v2_signed_bundle_manual_field_shape_ready_count": 1,
            "v2_signed_bundle_phase1_manual_field_ready_count": 2,
            "v2_signed_bundle_phase2_manual_field_ready_count": 0,
        }
    }
    reports["signed_bundle_extraction"] = {
        "summary": {
            "v2_signed_bundle_extraction_ready_for_record_count": 0,
            "v2_signed_bundle_extraction_phase1_ready_for_record_count": 1,
            "v2_signed_bundle_extraction_phase2_ready_for_record_count": 0,
            "v2_signed_bundle_extraction_full_ready_for_record_count": 0,
            "v2_signed_bundle_extraction_missing_signature_count": 1,
            "v2_signed_bundle_extraction_phase1_missing_signature_count": 0,
            "v2_signed_bundle_extraction_phase2_missing_signature_count": 1,
            "v2_signed_bundle_extraction_source_digest_match_count": 1,
            "v2_signed_bundle_extraction_source_digest_stale_count": 0,
            "v2_signed_bundle_extraction_template_digest_mismatch_count": 0,
            "v2_signed_bundle_extraction_unknown_entry_count": 0,
            "v2_signed_bundle_extraction_unsigned_template_count": 0,
            "v2_signed_bundle_extraction_signed_entry_digest_present_count": 2,
            "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count": 2,
        }
    }
    reports["approval_execution_preflight"] = {
        "summary": {
            "v2_approval_preflight_phase1_ready_count": 1,
            "v2_approval_preflight_phase2_ready_count": 0,
            "v2_approval_preflight_full_ready_count": 0,
            "v2_approval_preflight_source_digest_match_count": 1,
            "v2_approval_preflight_source_digest_stale_count": 0,
            "v2_approval_preflight_template_digest_mismatch_count": 0,
            "v2_approval_preflight_unknown_entry_count": 0,
            "v2_approval_preflight_unsigned_template_count": 0,
            "v2_approval_preflight_support_ready_count": 2,
            "v2_approval_preflight_support_invalid_count": 0,
            "v2_approval_preflight_support_source_binding_ready_count": 2,
            "v2_approval_preflight_support_source_binding_blocker_count": 0,
            "v2_approval_preflight_post_record_request_field_emission_count": 0,
            "v2_approval_preflight_hard_fail_count": 0,
            "v2_approval_preflight_signed_payload_digest_present_count": 2,
            "v2_approval_preflight_signed_payload_digest_missing_count": 1,
            "v2_approval_preflight_signed_bundle_entry_digest_present_count": 2,
            "v2_approval_preflight_signed_payload_bundle_digest_match_count": 2,
        }
    }
    return reports


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
