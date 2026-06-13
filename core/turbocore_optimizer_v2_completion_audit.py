"""Completion audit for the TurboCore optimizer v2 roadmap.

This audit is intentionally conservative: it proves whether the remaining
approval/product gates are closed before anyone can call the roadmap complete.
It does not record approvals, bind product routes, or enable native optimizer
dispatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_completion_audit.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"
EXPECTED_RECORD_INPUT_ITEM_COUNT = 13
EXPECTED_PHASE1_RECORD_INPUT_ITEM_COUNT = 6
EXPECTED_PHASE2_RECORD_INPUT_ITEM_COUNT = 7
EXPECTED_RECORD_INPUT_SUPPORT_CHECK_COUNT = 3
EXPECTED_RELEASE_ARTIFACT_FIRST_REQUIRED_ARTIFACT_COUNT = 12
PHASE2_REFRESH_INPUT_IDS = frozenset(
    {
        "phase2_signature_bundle",
        "phase2_reviewer_handoff",
    }
)

SOURCE_ARTIFACTS = {
    "remaining_gate_handoff": ARTIFACT_DIR / "turbocore_optimizer_v2_remaining_gate_handoff_scorecard.json",
    "approval_command_audit": ARTIFACT_DIR / "turbocore_optimizer_v2_approval_command_audit.json",
    "record_input_checklist": ARTIFACT_DIR / "turbocore_optimizer_v2_record_input_checklist.json",
    "approval_execution_preflight": ARTIFACT_DIR / "turbocore_optimizer_v2_approval_execution_preflight.json",
    "approval_state": ARTIFACT_DIR / "turbocore_optimizer_v2_approval_state_scorecard.json",
    "release_artifact_first_validation": ARTIFACT_DIR
    / "turbocore_optimizer_release_artifact_first_validation_scorecard.json",
}


def build_optimizer_v2_completion_audit(
    *,
    source_reports: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    sources = _load_sources(source_reports, directory)
    unsafe = _unsafe_claims(sources)
    requirements = _requirements(sources, unsafe)
    failed = [item for item in requirements if not item["ok"]]
    completion_ready = not failed and not unsafe
    payload = {
        "schema_version": 1,
        "audit": "turbocore_optimizer_v2_completion_audit_v0",
        "gate": "optimizer_v2_completion_audit",
        "roadmap": ROADMAP,
        "ok": not unsafe,
        "completion_audit_ready": not unsafe,
        "roadmap_complete": completion_ready,
        "completion_ready": completion_ready,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "source_artifacts": {name: str(directory / path.name) for name, path in SOURCE_ARTIFACTS.items()},
        "requirements": requirements,
        "summary": {
            "v2_completion_audit_requirement_count": len(requirements),
            "v2_completion_audit_passed_requirement_count": len(requirements) - len(failed),
            "v2_completion_audit_failed_requirement_count": len(failed),
            "v2_completion_audit_remaining_gate_open_count": _summary_int(
                sources["remaining_gate_handoff"],
                "v2_remaining_gate_open_count",
            ),
            "v2_completion_audit_command_audit_ready_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_ready_count",
            ),
            "v2_completion_audit_command_audit_expected_arg_binding_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_expected_arg_binding_count",
            ),
            "v2_completion_audit_command_audit_expected_arg_mismatch_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_expected_arg_mismatch_count",
            ),
            "v2_completion_audit_command_audit_expected_path_binding_ready_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_expected_path_binding_ready_count",
            ),
            "v2_completion_audit_command_audit_phase1_handoff_post_return_command_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_command_count",
            ),
            "v2_completion_audit_command_audit_phase1_handoff_post_return_pre_record_command_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count",
            ),
            "v2_completion_audit_command_audit_phase1_handoff_post_return_approval_record_command_count": (
                _summary_int(
                    sources["approval_command_audit"],
                    "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count",
                )
            ),
            "v2_completion_audit_command_audit_phase1_handoff_post_return_command_match_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_command_match_count",
            ),
            "v2_completion_audit_command_audit_phase1_handoff_post_return_command_mismatch_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count",
            ),
            "v2_completion_audit_command_audit_phase1_handoff_post_return_ready_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_ready_count",
            ),
            "v2_completion_audit_phase1_ready_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_phase1_ready_count",
            ),
            "v2_completion_audit_phase2_ready_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_phase2_ready_count",
            ),
            "v2_completion_audit_record_input_item_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_item_count",
            ),
            "v2_completion_audit_record_input_phase1_item_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_phase1_item_count",
            ),
            "v2_completion_audit_record_input_phase2_item_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_phase2_item_count",
            ),
            "v2_completion_audit_phase2_refresh_inputs_tracked_count": _phase2_refresh_inputs_tracked_count(
                sources["record_input_checklist"],
            ),
            "v2_completion_audit_record_input_checklist_shape_ready_count": (
                1 if _record_input_checklist_shape_ready(sources["record_input_checklist"]) else 0
            ),
            "v2_completion_audit_record_input_support_check_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_check_count",
            ),
            "v2_completion_audit_record_input_support_ready_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_ready_count",
            ),
            "v2_completion_audit_record_input_support_blocked_item_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_blocked_item_count",
            ),
            "v2_completion_audit_record_input_support_blocker_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_blocker_count",
            ),
            "v2_completion_audit_record_input_support_source_binding_ready_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_source_binding_ready_count",
            ),
            "v2_completion_audit_record_input_support_source_binding_blocker_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_source_binding_blocker_count",
            ),
            "v2_completion_audit_record_input_support_shape_ready_count": (
                1 if _record_input_support_shape_ready(sources["record_input_checklist"]) else 0
            ),
            "v2_completion_audit_phase1_signed_bundle_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase1_signed_bundle_ready_count",
            ),
            "v2_completion_audit_phase2_signed_bundle_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase2_signed_bundle_ready_count",
            ),
            "v2_completion_audit_signed_bundle_missing_signature_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_missing_signature_count",
            ),
            "v2_completion_audit_phase1_signed_bundle_missing_signature_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase1_signed_bundle_missing_signature_count",
            ),
            "v2_completion_audit_phase2_signed_bundle_missing_signature_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase2_signed_bundle_missing_signature_count",
            ),
            "v2_completion_audit_signed_bundle_freshness_guard_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_freshness_guard_ready_count",
            ),
            "v2_completion_audit_signed_bundle_freshness_template_digest_match_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_freshness_template_digest_match_count",
            ),
            "v2_completion_audit_signed_bundle_freshness_stale_signed_bundle_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count",
            ),
            "v2_completion_audit_signed_bundle_freshness_unknown_signed_entry_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_freshness_unknown_signed_entry_count",
            ),
            "v2_completion_audit_signed_bundle_freshness_unsafe_claim_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_freshness_unsafe_claim_count",
            ),
            "v2_completion_audit_signed_bundle_not_ready_entry_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_not_ready_entry_count",
            ),
            "v2_completion_audit_signed_bundle_source_digest_match_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_source_digest_match_count",
            ),
            "v2_completion_audit_signed_bundle_source_digest_stale_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_source_digest_stale_count",
            ),
            "v2_completion_audit_signed_bundle_template_digest_mismatch_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_template_digest_mismatch_count",
            ),
            "v2_completion_audit_signed_bundle_unknown_entry_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_unknown_entry_count",
            ),
            "v2_completion_audit_signed_bundle_unsigned_template_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_unsigned_template_count",
            ),
            "v2_completion_audit_signed_bundle_intake_integrity_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_intake_integrity_ready_count",
            ),
            "v2_completion_audit_signed_bundle_manual_field_check_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_manual_field_check_count",
            ),
            "v2_completion_audit_signed_bundle_manual_field_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_manual_field_ready_count",
            ),
            "v2_completion_audit_signed_bundle_manual_field_missing_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_manual_field_missing_count",
            ),
            "v2_completion_audit_signed_bundle_manual_field_shape_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_signed_bundle_manual_field_shape_ready_count",
            ),
            "v2_completion_audit_phase1_signed_bundle_manual_field_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase1_signed_bundle_manual_field_ready_count",
            ),
            "v2_completion_audit_phase2_signed_bundle_manual_field_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase2_signed_bundle_manual_field_ready_count",
            ),
            "v2_completion_audit_phase1_signed_bundle_manual_shape_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase1_signed_bundle_manual_shape_ready_count",
            ),
            "v2_completion_audit_phase2_signed_bundle_manual_shape_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase2_signed_bundle_manual_shape_ready_count",
            ),
            "v2_completion_audit_full_signed_bundle_manual_shape_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_full_signed_bundle_manual_shape_ready_count",
            ),
            "v2_completion_audit_phase1_extraction_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase1_extraction_ready_count",
            ),
            "v2_completion_audit_phase2_extraction_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase2_extraction_ready_count",
            ),
            "v2_completion_audit_extraction_missing_signature_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_missing_signature_count",
            ),
            "v2_completion_audit_phase1_extraction_missing_signature_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase1_extraction_missing_signature_count",
            ),
            "v2_completion_audit_phase2_extraction_missing_signature_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase2_extraction_missing_signature_count",
            ),
            "v2_completion_audit_extraction_not_ready_entry_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_not_ready_entry_count",
            ),
            "v2_completion_audit_extraction_source_digest_match_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_source_digest_match_count",
            ),
            "v2_completion_audit_extraction_source_digest_stale_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_source_digest_stale_count",
            ),
            "v2_completion_audit_extraction_template_digest_mismatch_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_template_digest_mismatch_count",
            ),
            "v2_completion_audit_extraction_unknown_entry_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_unknown_entry_count",
            ),
            "v2_completion_audit_extraction_unsigned_template_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_unsigned_template_count",
            ),
            "v2_completion_audit_extraction_integrity_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_integrity_ready_count",
            ),
            "v2_completion_audit_extraction_signed_entry_digest_present_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_signed_entry_digest_present_count",
            ),
            "v2_completion_audit_extraction_extractable_signed_entry_digest_present_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_extractable_signed_entry_digest_present_count",
            ),
            "v2_completion_audit_extraction_digest_shape_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_extraction_digest_shape_ready_count",
            ),
            "v2_completion_audit_phase1_preflight_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase1_preflight_ready_count",
            ),
            "v2_completion_audit_phase2_preflight_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase2_preflight_ready_count",
            ),
            "v2_completion_audit_phase1_record_chain_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase1_record_chain_ready_count",
            ),
            "v2_completion_audit_phase2_record_chain_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase2_record_chain_ready_count",
            ),
            "v2_completion_audit_full_record_chain_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_full_record_chain_ready_count",
            ),
            "v2_completion_audit_preflight_full_ready_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_full_ready_count",
            ),
            "v2_completion_audit_preflight_integrity_ready_count": (
                1 if _approval_preflight_integrity_guards_ready(sources["approval_execution_preflight"]) else 0
            ),
            "v2_completion_audit_preflight_signed_bundle_valid_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_signed_bundle_valid_count",
            ),
            "v2_completion_audit_preflight_source_digest_match_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_source_digest_match_count",
            ),
            "v2_completion_audit_preflight_source_digest_stale_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_source_digest_stale_count",
            ),
            "v2_completion_audit_preflight_template_digest_mismatch_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_template_digest_mismatch_count",
            ),
            "v2_completion_audit_preflight_unknown_entry_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_unknown_entry_count",
            ),
            "v2_completion_audit_preflight_not_ready_entry_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_not_ready_entry_count",
            ),
            "v2_completion_audit_preflight_source_integrity_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_preflight_source_integrity_ready_count",
            ),
            "v2_completion_audit_preflight_support_integrity_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_preflight_support_integrity_ready_count",
            ),
            "v2_completion_audit_preflight_unsigned_template_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_unsigned_template_count",
            ),
            "v2_completion_audit_preflight_signed_payload_digest_present_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_signed_payload_digest_present_count",
            ),
            "v2_completion_audit_preflight_signed_payload_digest_missing_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_signed_payload_digest_missing_count",
            ),
            "v2_completion_audit_preflight_signed_bundle_entry_digest_present_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_signed_bundle_entry_digest_present_count",
            ),
            "v2_completion_audit_preflight_signed_payload_bundle_digest_match_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_signed_payload_bundle_digest_match_count",
            ),
            "v2_completion_audit_preflight_extracted_entry_digest_match_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_extracted_entry_digest_match_count",
            ),
            "v2_completion_audit_preflight_extracted_entry_digest_mismatch_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_extracted_entry_digest_mismatch_count",
            ),
            "v2_completion_audit_preflight_extracted_entry_source_missing_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_extracted_entry_source_missing_count",
            ),
            "v2_completion_audit_preflight_support_ready_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_support_ready_count",
            ),
            "v2_completion_audit_preflight_support_invalid_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_support_invalid_count",
            ),
            "v2_completion_audit_preflight_support_source_binding_ready_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_support_source_binding_ready_count",
            ),
            "v2_completion_audit_preflight_support_source_binding_blocker_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_support_source_binding_blocker_count",
            ),
            "v2_completion_audit_preflight_post_record_request_field_emission_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_post_record_request_field_emission_count",
            ),
            "v2_completion_audit_preflight_hard_fail_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_hard_fail_count",
            ),
            "v2_completion_audit_owner_review_recorded_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_owner_review_recorded_count",
            ),
            "v2_completion_audit_product_exposure_recorded_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_product_exposure_recorded_count",
            ),
            "v2_completion_audit_owner_direction_recorded_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_owner_direction_recorded_count",
            ),
            "v2_completion_audit_phase1_record_preflight_binding_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase1_record_preflight_binding_ready_count",
            ),
            "v2_completion_audit_phase2_record_preflight_binding_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_phase2_record_preflight_binding_ready_count",
            ),
            "v2_completion_audit_full_record_preflight_binding_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_full_record_preflight_binding_ready_count",
            ),
            "v2_completion_audit_record_preflight_binding_ready_count": _summary_int(
                sources["approval_state"],
                "v2_approval_state_record_preflight_binding_ready_count",
            ),
            "v2_completion_audit_artifact_first_required_artifact_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_required_artifact_count",
            ),
            "v2_completion_audit_artifact_first_ready_artifact_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_ready_artifact_count",
            ),
            "v2_completion_audit_artifact_first_ready_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_validation_ready_count",
            ),
            "v2_completion_audit_artifact_first_record_input_checklist_shape_ready_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_v2_record_input_checklist_shape_ready_count",
            ),
            "v2_completion_audit_artifact_first_record_input_support_shape_ready_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_v2_record_input_support_shape_ready_count",
            ),
            "v2_completion_audit_artifact_first_reviewer_handoff_phase_metadata_ready_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_v2_reviewer_handoff_phase_metadata_ready_count",
            ),
            "v2_completion_audit_artifact_first_reviewer_handoff_phase1_template_entry_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_v2_reviewer_handoff_phase1_template_entry_count",
            ),
            "v2_completion_audit_artifact_first_reviewer_handoff_phase2_deferred_signature_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_v2_reviewer_handoff_phase2_deferred_signature_count",
            ),
            "v2_completion_audit_artifact_first_reviewer_handoff_phase_shape_ready_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_v2_reviewer_handoff_phase_shape_ready_count",
            ),
            "v2_completion_audit_artifact_first_command_audit_expected_path_binding_ready_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_v2_command_audit_expected_path_binding_ready_count",
            ),
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_count",
            ),
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_pre_record_command_count": (
                _summary_int(
                    sources["release_artifact_first_validation"],
                    (
                        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_"
                        "pre_record_command_count"
                    ),
                )
            ),
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_approval_record_command_count": (
                _summary_int(
                    sources["release_artifact_first_validation"],
                    (
                        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_"
                        "approval_record_command_count"
                    ),
                )
            ),
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_match_count": (
                _summary_int(
                    sources["release_artifact_first_validation"],
                    (
                        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_"
                        "command_match_count"
                    ),
                )
            ),
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_command_mismatch_count": (
                _summary_int(
                    sources["release_artifact_first_validation"],
                    (
                        "release_artifact_first_v2_command_audit_phase1_handoff_post_return_"
                        "command_mismatch_count"
                    ),
                )
            ),
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_ready_count": _summary_int(
                sources["release_artifact_first_validation"],
                "release_artifact_first_v2_command_audit_phase1_handoff_post_return_ready_count",
            ),
            "v2_completion_audit_artifact_first_command_audit_phase1_handoff_post_return_shape_ready_count": (
                _summary_int(
                    sources["release_artifact_first_validation"],
                    "release_artifact_first_v2_command_audit_phase1_handoff_post_return_shape_ready_count",
                )
            ),
            "v2_completion_audit_default_off_guard_count": 1 if _default_off_preserved(sources, unsafe) else 0,
            "v2_completion_audit_completion_ready_count": 1 if completion_ready else 0,
            "v2_completion_audit_approval_recorded_count": 0,
            "v2_completion_audit_runtime_dispatch_ready_count": 0,
            "v2_completion_audit_native_dispatch_allowed_count": 0,
            "v2_completion_audit_training_path_enabled_count": 0,
            "v2_completion_audit_product_native_ready_count": 0,
            "v2_completion_audit_default_behavior_changed_count": 0,
            "v2_completion_audit_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe([reason for item in failed for reason in item["blocked_reasons"]] + unsafe),
        "promotion_blockers": _dedupe(
            [
                "v2_completion_audit_not_ready",
                "real_owner_release_review_signature_missing",
                "real_product_exposure_review_signature_missing",
                "real_owner_release_direction_signature_missing",
            ]
            + [reason for item in failed for reason in item["blocked_reasons"]]
            + unsafe
        ),
        "recommended_next_step": _recommended_next_step(sources),
        "notes": [
            "This audit is a completion guard only.",
            "A passed release suite is not enough unless all completion requirements pass.",
            "Default product training remains PyTorch authoritative until explicit approval and route binding.",
        ],
    }
    if write_artifact:
        output = directory / ARTIFACT.name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _requirements(sources: Mapping[str, Mapping[str, Any]], unsafe: list[str]) -> list[dict[str, Any]]:
    remaining = sources["remaining_gate_handoff"]
    command_audit = sources["approval_command_audit"]
    checklist = sources["record_input_checklist"]
    preflight = sources["approval_execution_preflight"]
    state = sources["approval_state"]
    release = sources["release_artifact_first_validation"]
    return [
        _requirement(
            "remaining_gates_closed",
            _summary_int(remaining, "v2_remaining_gate_total_count") == 6
            and _summary_int(remaining, "v2_remaining_gate_open_count") == 0,
            "v2_remaining_gate_open",
        ),
        _requirement(
            "approval_command_audit_path_bound",
            _approval_command_audit_path_bound(command_audit),
            "approval_command_audit_path_binding_not_ready",
        ),
        _requirement(
            "phase1_record_inputs_ready",
            _phase1_record_inputs_ready(checklist),
            "phase1_record_inputs_not_ready",
        ),
        _requirement(
            "phase2_record_inputs_ready",
            _phase2_record_inputs_ready(checklist),
            "phase2_record_inputs_not_ready",
        ),
        _requirement(
            "approval_preflight_full_ready",
            _summary_int(preflight, "v2_approval_preflight_full_ready_count") == 1,
            "approval_preflight_not_full_ready",
        ),
        _requirement(
            "approval_preflight_integrity_guards_ready",
            _approval_preflight_integrity_guards_ready(preflight),
            "approval_preflight_integrity_guards_not_ready",
        ),
        _requirement(
            "signed_bundle_integrity_bound",
            _signed_bundle_integrity_bound(state),
            "signed_bundle_integrity_not_ready",
        ),
        _requirement(
            "signed_bundle_freshness_bound",
            _signed_bundle_freshness_bound(state),
            "signed_bundle_freshness_not_ready",
        ),
        _requirement(
            "owner_release_review_recorded",
            _summary_int(state, "v2_approval_state_owner_review_recorded_count") == 1,
            "owner_release_review_not_recorded",
        ),
        _requirement(
            "product_exposure_decision_recorded",
            _summary_int(state, "v2_approval_state_product_exposure_recorded_count") == 1,
            "product_exposure_decision_not_recorded",
        ),
        _requirement(
            "owner_release_direction_recorded",
            _summary_int(state, "v2_approval_state_owner_direction_recorded_count") == 1,
            "owner_release_direction_not_recorded",
        ),
        _requirement(
            "approval_records_preflight_bound",
            _summary_int(state, "v2_approval_state_record_preflight_binding_ready_count") == 3
            and _summary_int(state, "v2_approval_state_phase1_record_preflight_binding_ready_count") == 1
            and _summary_int(state, "v2_approval_state_phase2_record_preflight_binding_ready_count") == 1
            and _summary_int(state, "v2_approval_state_full_record_preflight_binding_ready_count") == 1,
            "approval_records_preflight_binding_not_ready",
        ),
        _requirement(
            "approval_state_has_no_open_remaining_gate",
            _summary_int(state, "v2_approval_state_remaining_gate_open_count") == 0
            and _summary_int(state, "v2_approval_state_recorded_stage_count") >= 3,
            "approval_state_remaining_gate_open",
        ),
        _requirement(
            "artifact_first_release_validation_ready",
            _artifact_first_release_validation_ready(release),
            "release_artifact_first_validation_shape_not_ready",
        ),
        _requirement("default_off_preserved", _default_off_preserved(sources, unsafe), "default_off_guard_failed"),
    ]


def _requirement(requirement_id: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": requirement_id,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


def _recommended_next_step(sources: Mapping[str, Mapping[str, Any]]) -> str:
    if not _phase1_record_inputs_ready(sources["record_input_checklist"]):
        return "collect real phase1 signed bundle and extracted owner/product review files"
    if _summary_int(sources["approval_execution_preflight"], "v2_approval_preflight_full_ready_count") == 0:
        return "run approval execution preflight before any record validator"
    return "record real owner/product decisions and owner direction while keeping default dispatch off"


def _load_sources(
    source_reports: Mapping[str, Mapping[str, Any]] | None,
    directory: Path,
) -> dict[str, dict[str, Any]]:
    overrides = source_reports or {}
    return {
        name: _as_dict(overrides.get(name)) or _read_json(directory / path.name)
        for name, path in SOURCE_ARTIFACTS.items()
    }


def _default_off_preserved(sources: Mapping[str, Mapping[str, Any]], unsafe: list[str]) -> bool:
    if unsafe:
        return False
    for report in sources.values():
        for field in (
            "default_behavior_changed",
            "runtime_dispatch_ready",
            "runtime_dispatch_allowed",
            "native_dispatch_allowed",
            "training_path_enabled",
            "product_native_ready",
            "product_exposure_allowed",
            "request_fields_emitted",
            "schema_exposure_allowed",
            "ui_exposure_allowed",
            "backend_router_registered",
        ):
            if report.get(field) is True:
                return False
    return True


def _unsafe_claims(sources: Mapping[str, Mapping[str, Any]]) -> list[str]:
    claims: list[str] = []
    for name, report in sources.items():
        for field in (
            "approval_artifact_written",
            "default_behavior_changed",
            "runtime_dispatch_ready",
            "runtime_dispatch_allowed",
            "native_dispatch_allowed",
            "training_path_enabled",
            "product_native_ready",
            "product_exposure_allowed",
            "request_fields_emitted",
            "schema_exposure_allowed",
            "ui_exposure_allowed",
            "backend_router_registered",
        ):
            if report.get(field) is True:
                claims.append(f"{name}_unsafe:{field}")
    return _dedupe(claims)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _as_dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return {}


def _summary_int(report: Mapping[str, Any], key: str) -> int:
    summary = _as_dict(report.get("summary"))
    try:
        return int(summary.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _phase1_record_inputs_ready(checklist: Mapping[str, Any]) -> bool:
    return (
        _summary_int(checklist, "v2_record_input_checklist_phase1_ready_count") == 1
        and _record_input_checklist_shape_ready(checklist)
        and _record_input_support_shape_ready(checklist)
        and _summary_int(checklist, "v2_record_input_checklist_support_ready_count") >= 2
    )


def _approval_command_audit_path_bound(command_audit: Mapping[str, Any]) -> bool:
    return (
        _summary_int(command_audit, "v2_approval_command_audit_ready_count") == 1
        and _summary_int(command_audit, "v2_approval_command_audit_expected_arg_binding_count") >= 35
        and _summary_int(command_audit, "v2_approval_command_audit_expected_arg_mismatch_count") == 0
        and _summary_int(command_audit, "v2_approval_command_audit_expected_path_binding_ready_count") == 1
        and _summary_int(command_audit, "v2_approval_command_audit_preflight_artifact_write_count") == 2
        and _summary_int(command_audit, "v2_approval_command_audit_preflight_no_artifact_blocker_count") == 0
        and _summary_int(command_audit, "v2_approval_command_audit_record_command_preflight_arg_count") == 3
        and _approval_command_audit_handoff_ready(command_audit)
    )


def _approval_command_audit_handoff_ready(command_audit: Mapping[str, Any]) -> bool:
    return (
        _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_command_count") == 3
        and _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count")
        == 3
        and _summary_int(
            command_audit,
            "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count",
        )
        == 0
        and _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_command_match_count")
        == 3
        and _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count")
        == 0
        and _summary_int(command_audit, "v2_approval_command_audit_phase1_handoff_post_return_ready_count") == 1
    )


def _artifact_first_release_validation_ready(release: Mapping[str, Any]) -> bool:
    return (
        _summary_int(release, "release_artifact_first_validation_ready_count") == 1
        and _summary_int(release, "release_artifact_first_required_artifact_count")
        >= EXPECTED_RELEASE_ARTIFACT_FIRST_REQUIRED_ARTIFACT_COUNT
        and _summary_int(release, "release_artifact_first_ready_artifact_count")
        >= EXPECTED_RELEASE_ARTIFACT_FIRST_REQUIRED_ARTIFACT_COUNT
        and _summary_int(release, "release_artifact_first_missing_artifact_count") == 0
        and _summary_int(release, "release_artifact_first_parse_error_count") == 0
        and _summary_int(release, "release_artifact_first_v2_record_input_checklist_shape_ready_count") == 1
        and _summary_int(release, "release_artifact_first_v2_record_input_support_shape_ready_count") == 1
        and _summary_int(release, "release_artifact_first_v2_reviewer_handoff_phase_metadata_ready_count") == 1
        and _summary_int(release, "release_artifact_first_v2_reviewer_handoff_phase1_template_entry_count") == 2
        and _summary_int(release, "release_artifact_first_v2_reviewer_handoff_phase2_deferred_signature_count") == 1
        and _summary_int(release, "release_artifact_first_v2_reviewer_handoff_phase_shape_ready_count") == 1
        and _summary_int(release, "release_artifact_first_v2_command_audit_expected_path_binding_ready_count") == 1
        and _summary_int(release, "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_count")
        == 3
        and _summary_int(
            release,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_pre_record_command_count",
        )
        == 3
        and _summary_int(
            release,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_approval_record_command_count",
        )
        == 0
        and _summary_int(
            release,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_match_count",
        )
        == 3
        and _summary_int(
            release,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_command_mismatch_count",
        )
        == 0
        and _summary_int(release, "release_artifact_first_v2_command_audit_phase1_handoff_post_return_ready_count")
        == 1
        and _summary_int(
            release,
            "release_artifact_first_v2_command_audit_phase1_handoff_post_return_shape_ready_count",
        )
        == 1
        and _summary_int(release, "release_artifact_first_runtime_dispatch_ready_count") == 0
        and _summary_int(release, "release_artifact_first_native_dispatch_allowed_count") == 0
        and _summary_int(release, "release_artifact_first_training_path_enabled_count") == 0
        and _summary_int(release, "release_artifact_first_default_behavior_changed_count") == 0
        and _summary_int(release, "release_artifact_first_product_native_ready_count") == 0
    )


def _approval_preflight_integrity_guards_ready(preflight: Mapping[str, Any]) -> bool:
    return (
        _summary_int(preflight, "v2_approval_preflight_signed_bundle_valid_count") == 1
        and _summary_int(preflight, "v2_approval_preflight_source_digest_match_count") == 1
        and _summary_int(preflight, "v2_approval_preflight_source_digest_stale_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_template_digest_mismatch_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_unknown_entry_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_not_ready_entry_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_unsigned_template_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_signed_payload_digest_present_count") >= 3
        and _summary_int(preflight, "v2_approval_preflight_signed_payload_digest_missing_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_signed_bundle_entry_digest_present_count") >= 3
        and _summary_int(preflight, "v2_approval_preflight_signed_payload_bundle_digest_match_count") >= 3
        and _summary_int(preflight, "v2_approval_preflight_extracted_entry_digest_match_count") >= 3
        and _summary_int(preflight, "v2_approval_preflight_extracted_entry_digest_mismatch_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_extracted_entry_source_missing_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_support_ready_count") >= 3
        and _summary_int(preflight, "v2_approval_preflight_support_invalid_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_support_source_binding_ready_count") >= 3
        and _summary_int(preflight, "v2_approval_preflight_support_source_binding_blocker_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_post_record_request_field_emission_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_hard_fail_count") == 0
    )


def _signed_bundle_integrity_bound(state: Mapping[str, Any]) -> bool:
    return (
        _summary_int(state, "v2_approval_state_signed_bundle_intake_integrity_ready_count") == 1
        and _summary_int(state, "v2_approval_state_extraction_integrity_ready_count") == 1
        and _summary_int(state, "v2_approval_state_preflight_source_integrity_ready_count") == 1
        and _summary_int(state, "v2_approval_state_preflight_support_integrity_ready_count") == 1
        and _summary_int(state, "v2_approval_state_signed_bundle_missing_signature_count") == 0
        and _summary_int(state, "v2_approval_state_phase1_signed_bundle_missing_signature_count") == 0
        and _summary_int(state, "v2_approval_state_phase2_signed_bundle_missing_signature_count") == 0
        and _summary_int(state, "v2_approval_state_extraction_missing_signature_count") == 0
        and _summary_int(state, "v2_approval_state_phase1_extraction_missing_signature_count") == 0
        and _summary_int(state, "v2_approval_state_phase2_extraction_missing_signature_count") == 0
        and _summary_int(state, "v2_approval_state_signed_bundle_manual_field_check_count") >= 3
        and _summary_int(state, "v2_approval_state_signed_bundle_manual_field_ready_count") >= 3
        and _summary_int(state, "v2_approval_state_signed_bundle_manual_field_missing_count") == 0
        and _summary_int(state, "v2_approval_state_signed_bundle_manual_field_shape_ready_count") == 1
        and _summary_int(state, "v2_approval_state_phase1_signed_bundle_manual_shape_ready_count") == 1
        and _summary_int(state, "v2_approval_state_phase2_signed_bundle_manual_shape_ready_count") == 1
        and _summary_int(state, "v2_approval_state_full_signed_bundle_manual_shape_ready_count") == 1
    )


def _signed_bundle_freshness_bound(state: Mapping[str, Any]) -> bool:
    return (
        _summary_int(state, "v2_approval_state_signed_bundle_freshness_guard_ready_count") == 1
        and _summary_int(state, "v2_approval_state_signed_bundle_freshness_template_digest_match_count") == 1
        and _summary_int(state, "v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count") == 0
        and _summary_int(state, "v2_approval_state_signed_bundle_freshness_unknown_signed_entry_count") == 0
        and _summary_int(state, "v2_approval_state_signed_bundle_freshness_unsafe_claim_count") == 0
    )


def _phase2_record_inputs_ready(checklist: Mapping[str, Any]) -> bool:
    return (
        _summary_int(checklist, "v2_record_input_checklist_phase2_ready_count") == 1
        and _record_input_checklist_shape_ready(checklist)
        and _record_input_support_shape_ready(checklist)
        and _summary_int(checklist, "v2_record_input_checklist_support_ready_count")
        >= EXPECTED_RECORD_INPUT_SUPPORT_CHECK_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_support_blocker_count") == 0
        and _summary_int(checklist, "v2_record_input_checklist_support_source_binding_ready_count")
        >= EXPECTED_RECORD_INPUT_SUPPORT_CHECK_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_support_source_binding_blocker_count") == 0
    )


def _record_input_support_shape_ready(checklist: Mapping[str, Any]) -> bool:
    return (
        _summary_int(checklist, "v2_record_input_checklist_support_check_count")
        >= EXPECTED_RECORD_INPUT_SUPPORT_CHECK_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_support_source_binding_ready_count") >= 2
        and _summary_int(checklist, "v2_record_input_checklist_support_source_binding_blocker_count") <= 1
    )


def _record_input_checklist_shape_ready(checklist: Mapping[str, Any]) -> bool:
    return (
        _summary_int(checklist, "v2_record_input_checklist_item_count") >= EXPECTED_RECORD_INPUT_ITEM_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_phase1_item_count")
        >= EXPECTED_PHASE1_RECORD_INPUT_ITEM_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_phase2_item_count")
        >= EXPECTED_PHASE2_RECORD_INPUT_ITEM_COUNT
        and _phase2_refresh_inputs_tracked_count(checklist) == len(PHASE2_REFRESH_INPUT_IDS)
    )


def _phase2_refresh_inputs_tracked_count(checklist: Mapping[str, Any]) -> int:
    items = checklist.get("items")
    if isinstance(items, list):
        seen = {
            str(item.get("id"))
            for item in items
            if isinstance(item, Mapping) and str(item.get("phase")) == "phase2_direction"
        }
        return len(PHASE2_REFRESH_INPUT_IDS.intersection(seen))
    if (
        _summary_int(checklist, "v2_record_input_checklist_item_count") >= EXPECTED_RECORD_INPUT_ITEM_COUNT
        and _summary_int(checklist, "v2_record_input_checklist_phase2_item_count")
        >= EXPECTED_PHASE2_RECORD_INPUT_ITEM_COUNT
    ):
        return len(PHASE2_REFRESH_INPUT_IDS)
    return 0


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="", help="Directory containing v2 approval-chain artifacts.")
    parser.add_argument("--no-artifact", action="store_true", help="Print audit without writing artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_completion_audit(
        artifact_dir=args.artifact_dir or None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_optimizer_v2_completion_audit"]


if __name__ == "__main__":
    raise SystemExit(main())
