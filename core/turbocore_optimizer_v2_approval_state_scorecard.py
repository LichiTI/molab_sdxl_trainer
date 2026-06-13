"""Aggregate TurboCore optimizer v2 approval-chain state.

The scorecard is a read-only release handoff view.  It gathers the v2 handoff,
signature, intake, extraction, preflight, and record artifacts into one status
table without recording approval or enabling native optimizer dispatch.
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
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_approval_state_scorecard.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"

SOURCE_ARTIFACTS = {
    "remaining_gate_handoff": ARTIFACT_DIR / "turbocore_optimizer_v2_remaining_gate_handoff_scorecard.json",
    "signature_bundle": ARTIFACT_DIR / "turbocore_optimizer_v2_signature_bundle_packet.json",
    "reviewer_handoff": ARTIFACT_DIR / "turbocore_optimizer_v2_reviewer_handoff_packet.json",
    "approval_execution_plan": ARTIFACT_DIR / "turbocore_optimizer_v2_approval_execution_plan.json",
    "approval_command_audit": ARTIFACT_DIR / "turbocore_optimizer_v2_approval_command_audit.json",
    "record_input_checklist": ARTIFACT_DIR / "turbocore_optimizer_v2_record_input_checklist.json",
    "signed_bundle_freshness": ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle_freshness_guard.json",
    "signed_bundle_intake": ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle_intake_record.json",
    "signed_bundle_extraction": ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle_extraction_record.json",
    "approval_execution_preflight": ARTIFACT_DIR / "turbocore_optimizer_v2_approval_execution_preflight.json",
    "owner_release_review_record": ARTIFACT_DIR / "native_update_owner_release_review_record.json",
    "product_exposure_decision": ARTIFACT_DIR / "native_update_product_exposure_decision.json",
    "release_review_archive": ARTIFACT_DIR / "native_update_release_review_archive.json",
    "owner_release_direction_packet": ARTIFACT_DIR / "native_update_owner_release_direction_packet.json",
    "owner_release_direction_record": ARTIFACT_DIR / "native_update_owner_release_direction_record.json",
}


def build_optimizer_v2_approval_state_scorecard(
    *,
    source_reports: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    sources = _load_sources(source_reports, directory)
    stages = _stages(sources)
    unsafe = _unsafe_claims(sources)
    ready = [stage for stage in stages if stage["state"] == "ready"]
    waiting = [stage for stage in stages if stage["state"] == "waiting_for_external_signature"]
    recordable = [stage for stage in stages if stage["state"] == "ready_for_record"]
    recorded = [stage for stage in stages if stage["state"] == "recorded"]
    missing = [stage for stage in stages if stage["state"] == "missing"]
    blocked = [stage for stage in stages if stage["state"] == "blocked"]
    signed_bundle_freshness_guard_ready = _signed_bundle_freshness_guard_ready(sources["signed_bundle_freshness"])
    phase1_signed_bundle_ready = _summary_int(sources["signed_bundle_intake"], "v2_signed_bundle_phase1_ready_count")
    phase2_signed_bundle_ready = _summary_int(sources["signed_bundle_intake"], "v2_signed_bundle_phase2_ready_count")
    full_signed_bundle_ready = _summary_int(sources["signed_bundle_intake"], "v2_signed_bundle_full_ready_count")
    signed_bundle_missing_signature_count = _summary_int(
        sources["signed_bundle_intake"],
        "v2_signed_bundle_missing_signature_count",
    )
    phase1_signed_bundle_missing_signature_count = _summary_int(
        sources["signed_bundle_intake"],
        "v2_signed_bundle_phase1_missing_signature_count",
    )
    phase2_signed_bundle_missing_signature_count = _summary_int(
        sources["signed_bundle_intake"],
        "v2_signed_bundle_phase2_missing_signature_count",
    )
    signed_bundle_not_ready_entry_count = _summary_int(
        sources["signed_bundle_intake"],
        "v2_signed_bundle_not_ready_entry_count",
    )
    signed_bundle_intake_integrity_ready = _signed_bundle_intake_integrity_ready(sources["signed_bundle_intake"])
    phase1_signed_bundle_manual_shape_ready = _signed_bundle_manual_field_shape_ready(
        sources["signed_bundle_intake"],
        expected_present=2,
        ready_key="v2_signed_bundle_phase1_manual_field_ready_count",
    )
    phase2_signed_bundle_manual_shape_ready = _signed_bundle_manual_field_shape_ready(
        sources["signed_bundle_intake"],
        expected_present=1,
        ready_key="v2_signed_bundle_phase2_manual_field_ready_count",
    )
    full_signed_bundle_manual_shape_ready = _signed_bundle_manual_field_shape_ready(
        sources["signed_bundle_intake"],
        expected_present=3,
        ready_key="v2_signed_bundle_manual_field_ready_count",
    )
    phase1_extraction_ready = _summary_int(
        sources["signed_bundle_extraction"],
        "v2_signed_bundle_extraction_phase1_ready_for_record_count",
    )
    phase2_extraction_ready = _summary_int(
        sources["signed_bundle_extraction"],
        "v2_signed_bundle_extraction_phase2_ready_for_record_count",
    )
    full_extraction_ready = _summary_int(
        sources["signed_bundle_extraction"],
        "v2_signed_bundle_extraction_full_ready_for_record_count",
    )
    extraction_missing_signature_count = _summary_int(
        sources["signed_bundle_extraction"],
        "v2_signed_bundle_extraction_missing_signature_count",
    )
    phase1_extraction_missing_signature_count = _summary_int(
        sources["signed_bundle_extraction"],
        "v2_signed_bundle_extraction_phase1_missing_signature_count",
    )
    phase2_extraction_missing_signature_count = _summary_int(
        sources["signed_bundle_extraction"],
        "v2_signed_bundle_extraction_phase2_missing_signature_count",
    )
    extraction_not_ready_entry_count = _summary_int(
        sources["signed_bundle_extraction"],
        "v2_signed_bundle_extraction_not_ready_entry_count",
    )
    signed_bundle_extraction_integrity_ready = _signed_bundle_extraction_integrity_ready(
        sources["signed_bundle_extraction"]
    )
    phase1_extraction_digest_shape_ready = _signed_bundle_extraction_digest_shape_ready(
        sources["signed_bundle_extraction"],
        expected_present=2,
    )
    phase2_extraction_digest_shape_ready = _signed_bundle_extraction_digest_shape_ready(
        sources["signed_bundle_extraction"],
        expected_present=1,
    )
    full_extraction_digest_shape_ready = _signed_bundle_extraction_digest_shape_ready(
        sources["signed_bundle_extraction"],
        expected_present=3,
    )
    phase1_preflight_ready = _summary_int(
        sources["approval_execution_preflight"],
        "v2_approval_preflight_phase1_ready_count",
    )
    phase2_preflight_ready = _summary_int(
        sources["approval_execution_preflight"],
        "v2_approval_preflight_phase2_ready_count",
    )
    full_preflight_ready = _summary_int(
        sources["approval_execution_preflight"],
        "v2_approval_preflight_full_ready_count",
    )
    preflight_not_ready_entry_count = _summary_int(
        sources["approval_execution_preflight"],
        "v2_approval_preflight_not_ready_entry_count",
    )
    approval_preflight_source_integrity_ready = _approval_preflight_source_integrity_ready(
        sources["approval_execution_preflight"]
    )
    phase1_approval_preflight_support_integrity_ready = _approval_preflight_support_integrity_ready(
        sources["approval_execution_preflight"],
        expected_ready=2,
    )
    phase2_approval_preflight_support_integrity_ready = _approval_preflight_support_integrity_ready(
        sources["approval_execution_preflight"],
        expected_ready=3,
    )
    full_approval_preflight_support_integrity_ready = _approval_preflight_support_integrity_ready(
        sources["approval_execution_preflight"],
        expected_ready=3,
    )
    phase1_preflight_digest_shape_ready = _approval_preflight_digest_shape_ready(
        sources["approval_execution_preflight"],
        expected_present=2,
    )
    phase2_preflight_digest_shape_ready = _approval_preflight_digest_shape_ready(
        sources["approval_execution_preflight"],
        expected_present=1,
    )
    full_preflight_digest_shape_ready = _approval_preflight_digest_shape_ready(
        sources["approval_execution_preflight"],
        expected_present=3,
    )
    owner_review_record_bound = _record_preflight_binding_ready(sources["owner_release_review_record"])
    product_exposure_record_bound = _record_preflight_binding_ready(sources["product_exposure_decision"])
    owner_direction_record_bound = _record_preflight_binding_ready(sources["owner_release_direction_record"])
    phase1_record_preflight_binding_ready = owner_review_record_bound and product_exposure_record_bound
    phase2_record_preflight_binding_ready = owner_direction_record_bound
    full_record_preflight_binding_ready = (
        phase1_record_preflight_binding_ready and phase2_record_preflight_binding_ready
    )
    record_input_ready = _record_input_checklist_ready(sources["record_input_checklist"])
    payload = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_v2_approval_state_scorecard_v0",
        "gate": "optimizer_v2_approval_state_scorecard",
        "roadmap": ROADMAP,
        "ok": not unsafe,
        "state_scorecard_ready": not unsafe,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "roadmap_complete": False,
        "promotion_ready": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "source_artifacts": {name: str(directory / path.name) for name, path in SOURCE_ARTIFACTS.items()},
        "stages": stages,
        "summary": {
            "v2_approval_state_stage_count": len(stages),
            "v2_approval_state_ready_stage_count": len(ready),
            "v2_approval_state_waiting_signature_stage_count": len(waiting),
            "v2_approval_state_ready_for_record_stage_count": len(recordable),
            "v2_approval_state_recorded_stage_count": len(recorded),
            "v2_approval_state_missing_stage_count": len(missing),
            "v2_approval_state_blocked_stage_count": len(blocked),
            "v2_approval_state_remaining_gate_open_count": _summary_int(
                sources["remaining_gate_handoff"],
                "v2_remaining_gate_open_count",
            ),
            "v2_approval_state_signature_ready_count": _summary_int(
                sources["signature_bundle"],
                "v2_signature_bundle_ready_for_signature_count",
            ),
            "v2_approval_state_signed_bundle_present_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_present_count",
            ),
            "v2_approval_state_signed_bundle_freshness_guard_ready_count": _summary_int(
                sources["signed_bundle_freshness"],
                "v2_signed_bundle_freshness_guard_ready_count",
            ),
            "v2_approval_state_signed_bundle_freshness_template_digest_match_count": _summary_int(
                sources["signed_bundle_freshness"],
                "v2_signed_bundle_freshness_template_digest_match_count",
            ),
            "v2_approval_state_signed_bundle_freshness_stale_signed_bundle_count": _summary_int(
                sources["signed_bundle_freshness"],
                "v2_signed_bundle_freshness_stale_signed_bundle_count",
            ),
            "v2_approval_state_signed_bundle_freshness_unknown_signed_entry_count": _summary_int(
                sources["signed_bundle_freshness"],
                "v2_signed_bundle_freshness_unknown_signed_entry_count",
            ),
            "v2_approval_state_signed_bundle_freshness_unsafe_claim_count": _summary_int(
                sources["signed_bundle_freshness"],
                "v2_signed_bundle_freshness_unsafe_claim_count",
            ),
            "v2_approval_state_phase1_signed_bundle_ready_count": phase1_signed_bundle_ready,
            "v2_approval_state_phase2_signed_bundle_ready_count": phase2_signed_bundle_ready,
            "v2_approval_state_full_signed_bundle_ready_count": full_signed_bundle_ready,
            "v2_approval_state_signed_bundle_missing_signature_count": signed_bundle_missing_signature_count,
            "v2_approval_state_phase1_signed_bundle_missing_signature_count": (
                phase1_signed_bundle_missing_signature_count
            ),
            "v2_approval_state_phase2_signed_bundle_missing_signature_count": (
                phase2_signed_bundle_missing_signature_count
            ),
            "v2_approval_state_signed_bundle_not_ready_entry_count": signed_bundle_not_ready_entry_count,
            "v2_approval_state_signed_bundle_source_digest_match_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_source_digest_match_count",
            ),
            "v2_approval_state_signed_bundle_source_digest_stale_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_source_digest_stale_count",
            ),
            "v2_approval_state_signed_bundle_template_digest_mismatch_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_template_digest_mismatch_count",
            ),
            "v2_approval_state_signed_bundle_unknown_entry_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_unknown_entry_count",
            ),
            "v2_approval_state_signed_bundle_unsigned_template_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_unsigned_template_count",
            ),
            "v2_approval_state_signed_bundle_intake_integrity_ready_count": _bool_count(
                signed_bundle_intake_integrity_ready
            ),
            "v2_approval_state_signed_bundle_manual_field_check_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_manual_field_check_count",
            ),
            "v2_approval_state_signed_bundle_manual_field_ready_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_manual_field_ready_count",
            ),
            "v2_approval_state_signed_bundle_manual_field_missing_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_manual_field_missing_count",
            ),
            "v2_approval_state_signed_bundle_manual_field_shape_ready_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_manual_field_shape_ready_count",
            ),
            "v2_approval_state_phase1_signed_bundle_manual_field_ready_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_phase1_manual_field_ready_count",
            ),
            "v2_approval_state_phase2_signed_bundle_manual_field_ready_count": _summary_int(
                sources["signed_bundle_intake"],
                "v2_signed_bundle_phase2_manual_field_ready_count",
            ),
            "v2_approval_state_phase1_signed_bundle_manual_shape_ready_count": _bool_count(
                phase1_signed_bundle_manual_shape_ready
            ),
            "v2_approval_state_phase2_signed_bundle_manual_shape_ready_count": _bool_count(
                phase2_signed_bundle_manual_shape_ready
            ),
            "v2_approval_state_full_signed_bundle_manual_shape_ready_count": _bool_count(
                full_signed_bundle_manual_shape_ready
            ),
            "v2_approval_state_phase1_extraction_ready_count": phase1_extraction_ready,
            "v2_approval_state_phase2_extraction_ready_count": phase2_extraction_ready,
            "v2_approval_state_full_extraction_ready_count": full_extraction_ready,
            "v2_approval_state_extraction_missing_signature_count": extraction_missing_signature_count,
            "v2_approval_state_phase1_extraction_missing_signature_count": (
                phase1_extraction_missing_signature_count
            ),
            "v2_approval_state_phase2_extraction_missing_signature_count": (
                phase2_extraction_missing_signature_count
            ),
            "v2_approval_state_extraction_not_ready_entry_count": extraction_not_ready_entry_count,
            "v2_approval_state_extraction_source_digest_match_count": _summary_int(
                sources["signed_bundle_extraction"],
                "v2_signed_bundle_extraction_source_digest_match_count",
            ),
            "v2_approval_state_extraction_source_digest_stale_count": _summary_int(
                sources["signed_bundle_extraction"],
                "v2_signed_bundle_extraction_source_digest_stale_count",
            ),
            "v2_approval_state_extraction_template_digest_mismatch_count": _summary_int(
                sources["signed_bundle_extraction"],
                "v2_signed_bundle_extraction_template_digest_mismatch_count",
            ),
            "v2_approval_state_extraction_unknown_entry_count": _summary_int(
                sources["signed_bundle_extraction"],
                "v2_signed_bundle_extraction_unknown_entry_count",
            ),
            "v2_approval_state_extraction_unsigned_template_count": _summary_int(
                sources["signed_bundle_extraction"],
                "v2_signed_bundle_extraction_unsigned_template_count",
            ),
            "v2_approval_state_extraction_integrity_ready_count": _bool_count(
                signed_bundle_extraction_integrity_ready
            ),
            "v2_approval_state_extraction_signed_entry_digest_present_count": _summary_int(
                sources["signed_bundle_extraction"],
                "v2_signed_bundle_extraction_signed_entry_digest_present_count",
            ),
            "v2_approval_state_extraction_extractable_signed_entry_digest_present_count": _summary_int(
                sources["signed_bundle_extraction"],
                "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count",
            ),
            "v2_approval_state_extraction_digest_shape_ready_count": _bool_count(
                full_extraction_digest_shape_ready
            ),
            "v2_approval_state_command_audit_ready_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_ready_count",
            ),
            "v2_approval_state_command_audit_expected_arg_binding_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_expected_arg_binding_count",
            ),
            "v2_approval_state_command_audit_expected_arg_mismatch_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_expected_arg_mismatch_count",
            ),
            "v2_approval_state_command_audit_expected_path_binding_ready_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_expected_path_binding_ready_count",
            ),
            "v2_approval_state_command_audit_preflight_artifact_write_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_preflight_artifact_write_count",
            ),
            "v2_approval_state_command_audit_preflight_no_artifact_blocker_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_preflight_no_artifact_blocker_count",
            ),
            "v2_approval_state_command_audit_record_command_preflight_arg_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_record_command_preflight_arg_count",
            ),
            "v2_approval_state_command_audit_phase1_handoff_post_return_command_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_command_count",
            ),
            "v2_approval_state_command_audit_phase1_handoff_post_return_pre_record_command_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_pre_record_command_count",
            ),
            "v2_approval_state_command_audit_phase1_handoff_post_return_approval_record_command_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_approval_record_command_count",
            ),
            "v2_approval_state_command_audit_phase1_handoff_post_return_command_match_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_command_match_count",
            ),
            "v2_approval_state_command_audit_phase1_handoff_post_return_command_mismatch_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_command_mismatch_count",
            ),
            "v2_approval_state_command_audit_phase1_handoff_post_return_ready_count": _summary_int(
                sources["approval_command_audit"],
                "v2_approval_command_audit_phase1_handoff_post_return_ready_count",
            ),
            "v2_approval_state_record_input_checklist_ready_count": _bool_count(record_input_ready),
            "v2_approval_state_record_input_support_check_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_check_count",
            ),
            "v2_approval_state_record_input_support_ready_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_ready_count",
            ),
            "v2_approval_state_record_input_support_blocked_item_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_blocked_item_count",
            ),
            "v2_approval_state_record_input_support_blocker_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_blocker_count",
            ),
            "v2_approval_state_record_input_support_source_binding_ready_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_source_binding_ready_count",
            ),
            "v2_approval_state_record_input_support_source_binding_blocker_count": _summary_int(
                sources["record_input_checklist"],
                "v2_record_input_checklist_support_source_binding_blocker_count",
            ),
            "v2_approval_state_record_input_support_shape_ready_count": _bool_count(
                _record_input_support_shape_ready(sources["record_input_checklist"])
            ),
            "v2_approval_state_phase1_preflight_ready_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_phase1_ready_count",
            ),
            "v2_approval_state_phase2_preflight_ready_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_phase2_ready_count",
            ),
            "v2_approval_state_full_preflight_ready_count": full_preflight_ready,
            "v2_approval_state_preflight_not_ready_entry_count": preflight_not_ready_entry_count,
            "v2_approval_state_preflight_source_digest_match_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_source_digest_match_count",
            ),
            "v2_approval_state_preflight_source_digest_stale_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_source_digest_stale_count",
            ),
            "v2_approval_state_preflight_template_digest_mismatch_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_template_digest_mismatch_count",
            ),
            "v2_approval_state_preflight_unknown_entry_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_unknown_entry_count",
            ),
            "v2_approval_state_preflight_unsigned_template_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_unsigned_template_count",
            ),
            "v2_approval_state_preflight_source_integrity_ready_count": _bool_count(
                approval_preflight_source_integrity_ready
            ),
            "v2_approval_state_preflight_support_ready_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_support_ready_count",
            ),
            "v2_approval_state_preflight_support_invalid_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_support_invalid_count",
            ),
            "v2_approval_state_preflight_support_source_binding_ready_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_support_source_binding_ready_count",
            ),
            "v2_approval_state_preflight_support_source_binding_blocker_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_support_source_binding_blocker_count",
            ),
            "v2_approval_state_preflight_post_record_request_field_emission_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_post_record_request_field_emission_count",
            ),
            "v2_approval_state_preflight_hard_fail_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_hard_fail_count",
            ),
            "v2_approval_state_preflight_support_integrity_ready_count": _bool_count(
                full_approval_preflight_support_integrity_ready
            ),
            "v2_approval_state_phase1_preflight_support_integrity_ready_count": _bool_count(
                phase1_approval_preflight_support_integrity_ready
            ),
            "v2_approval_state_phase2_preflight_support_integrity_ready_count": _bool_count(
                phase2_approval_preflight_support_integrity_ready
            ),
            "v2_approval_state_preflight_signed_payload_digest_present_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_signed_payload_digest_present_count",
            ),
            "v2_approval_state_preflight_signed_payload_digest_missing_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_signed_payload_digest_missing_count",
            ),
            "v2_approval_state_preflight_signed_bundle_entry_digest_present_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_signed_bundle_entry_digest_present_count",
            ),
            "v2_approval_state_preflight_signed_payload_bundle_digest_match_count": _summary_int(
                sources["approval_execution_preflight"],
                "v2_approval_preflight_signed_payload_bundle_digest_match_count",
            ),
            "v2_approval_state_preflight_digest_shape_ready_count": _bool_count(
                full_preflight_digest_shape_ready
            ),
            "v2_approval_state_phase1_record_chain_ready_count": _all_ready(
                _bool_count(signed_bundle_freshness_guard_ready),
                phase1_signed_bundle_ready,
                _bool_count(phase1_signed_bundle_missing_signature_count == 0),
                _bool_count(signed_bundle_intake_integrity_ready),
                _bool_count(signed_bundle_not_ready_entry_count == 0),
                _bool_count(phase1_signed_bundle_manual_shape_ready),
                phase1_extraction_ready,
                _bool_count(phase1_extraction_missing_signature_count == 0),
                _bool_count(signed_bundle_extraction_integrity_ready),
                _bool_count(extraction_not_ready_entry_count == 0),
                phase1_extraction_digest_shape_ready,
                phase1_preflight_ready,
                _bool_count(approval_preflight_source_integrity_ready),
                _bool_count(phase1_approval_preflight_support_integrity_ready),
                _bool_count(preflight_not_ready_entry_count == 0),
                phase1_preflight_digest_shape_ready,
            ),
            "v2_approval_state_phase2_record_chain_ready_count": _all_ready(
                _bool_count(signed_bundle_freshness_guard_ready),
                phase2_signed_bundle_ready,
                _bool_count(phase2_signed_bundle_missing_signature_count == 0),
                _bool_count(signed_bundle_intake_integrity_ready),
                _bool_count(signed_bundle_not_ready_entry_count == 0),
                _bool_count(phase2_signed_bundle_manual_shape_ready),
                phase2_extraction_ready,
                _bool_count(phase2_extraction_missing_signature_count == 0),
                _bool_count(signed_bundle_extraction_integrity_ready),
                _bool_count(extraction_not_ready_entry_count == 0),
                phase2_extraction_digest_shape_ready,
                phase2_preflight_ready,
                _bool_count(approval_preflight_source_integrity_ready),
                _bool_count(phase2_approval_preflight_support_integrity_ready),
                _bool_count(preflight_not_ready_entry_count == 0),
                phase2_preflight_digest_shape_ready,
            ),
            "v2_approval_state_full_record_chain_ready_count": _all_ready(
                _bool_count(signed_bundle_freshness_guard_ready),
                full_signed_bundle_ready,
                _bool_count(signed_bundle_missing_signature_count == 0),
                _bool_count(signed_bundle_intake_integrity_ready),
                _bool_count(signed_bundle_not_ready_entry_count == 0),
                _bool_count(full_signed_bundle_manual_shape_ready),
                full_extraction_ready,
                _bool_count(extraction_missing_signature_count == 0),
                _bool_count(signed_bundle_extraction_integrity_ready),
                _bool_count(extraction_not_ready_entry_count == 0),
                full_extraction_digest_shape_ready,
                full_preflight_ready,
                _bool_count(approval_preflight_source_integrity_ready),
                _bool_count(full_approval_preflight_support_integrity_ready),
                _bool_count(preflight_not_ready_entry_count == 0),
                full_preflight_digest_shape_ready,
            ),
            "v2_approval_state_owner_review_recorded_count": _bool_count(
                sources["owner_release_review_record"].get("release_review_recorded")
            ),
            "v2_approval_state_product_exposure_recorded_count": _bool_count(
                sources["product_exposure_decision"].get("product_exposure_decision_recorded")
            ),
            "v2_approval_state_owner_direction_recorded_count": _bool_count(
                sources["owner_release_direction_record"].get("owner_release_direction_recorded")
            ),
            "v2_approval_state_phase1_record_preflight_binding_ready_count": _bool_count(
                phase1_record_preflight_binding_ready
            ),
            "v2_approval_state_phase2_record_preflight_binding_ready_count": _bool_count(
                phase2_record_preflight_binding_ready
            ),
            "v2_approval_state_full_record_preflight_binding_ready_count": _bool_count(
                full_record_preflight_binding_ready
            ),
            "v2_approval_state_record_preflight_binding_ready_count": sum(
                1
                for item in (
                    owner_review_record_bound,
                    product_exposure_record_bound,
                    owner_direction_record_bound,
                )
                if item
            ),
            "v2_approval_state_approval_recorded_count": 0,
            "v2_approval_state_runtime_dispatch_ready_count": 0,
            "v2_approval_state_native_dispatch_allowed_count": 0,
            "v2_approval_state_training_path_enabled_count": 0,
            "v2_approval_state_product_native_ready_count": 0,
            "v2_approval_state_default_behavior_changed_count": 0,
            "v2_approval_state_unsafe_claim_count": len(unsafe),
        },
        "blocked_reasons": _dedupe(
            [reason for stage in stages for reason in _strings(stage.get("blocked_reasons"))] + unsafe
        ),
        "promotion_blockers": _dedupe(
            [
                "v2_approval_state_waiting_for_real_signatures",
                "product_dispatch_still_requires_explicit_route_binding",
            ]
            + [reason for stage in stages if stage["state"] != "recorded" for reason in _strings(stage.get("blocked_reasons"))]
            + unsafe
        ),
        "recommended_next_step": _recommended_next_step(stages),
        "notes": [
            "This scorecard aggregates approval-chain status only.",
            "Ready-for-record and recorded stages still keep product/native dispatch disabled.",
            "Synthetic smoke signatures are not counted as real owner/release/product approval.",
        ],
    }
    if write_artifact:
        output = directory / ARTIFACT.name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load_sources(
    source_reports: Mapping[str, Mapping[str, Any]] | None,
    directory: Path,
) -> dict[str, dict[str, Any]]:
    overrides = source_reports or {}
    return {
        name: _as_dict(overrides.get(name)) or _read_json(directory / path.name)
        for name, path in SOURCE_ARTIFACTS.items()
    }


def _stages(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _stage(
            "remaining_gate_handoff",
            sources["remaining_gate_handoff"],
            "ready",
            _summary_int(sources["remaining_gate_handoff"], "v2_remaining_gate_handoff_ready_count") == 1,
            "v2_remaining_gate_handoff_not_ready",
        ),
        _stage(
            "signature_bundle",
            sources["signature_bundle"],
            "ready",
            _summary_int(sources["signature_bundle"], "v2_signature_bundle_ready_for_signature_count") >= 2,
            "v2_signature_bundle_not_ready",
        ),
        _stage(
            "reviewer_handoff",
            sources["reviewer_handoff"],
            "waiting_for_external_signature",
            _summary_int(sources["reviewer_handoff"], "v2_reviewer_handoff_packet_ready_count") == 1,
            "v2_reviewer_handoff_not_ready",
        ),
        _stage(
            "approval_execution_plan",
            sources["approval_execution_plan"],
            "ready",
            _summary_int(sources["approval_execution_plan"], "v2_approval_execution_step_count") == 16
            and _summary_int(sources["approval_execution_plan"], "v2_approval_execution_phase1_step_count") == 6
            and _summary_int(sources["approval_execution_plan"], "v2_approval_execution_phase2_step_count") == 9
            and _summary_int(sources["approval_execution_plan"], "v2_approval_execution_plan_ready_count") == 1,
            "v2_approval_execution_plan_not_ready",
        ),
        _stage(
            "approval_command_audit",
            sources["approval_command_audit"],
            "ready",
            _approval_command_audit_ready(sources["approval_command_audit"]),
            "v2_approval_command_audit_not_ready",
        ),
        _stage(
            "record_input_checklist",
            sources["record_input_checklist"],
            "ready",
            _record_input_checklist_ready(sources["record_input_checklist"]),
            "v2_record_input_checklist_not_ready",
        ),
        _stage(
            "signed_bundle_freshness",
            sources["signed_bundle_freshness"],
            "ready",
            _signed_bundle_freshness_guard_ready(sources["signed_bundle_freshness"]),
            "v2_signed_bundle_freshness_guard_not_ready",
        ),
        _stage(
            "signed_bundle_intake",
            sources["signed_bundle_intake"],
            "ready_for_record",
            _summary_int(sources["signed_bundle_intake"], "v2_signed_bundle_valid_record_count") == 3
            and _signed_bundle_intake_integrity_ready(sources["signed_bundle_intake"])
            and _signed_bundle_manual_field_shape_ready(
                sources["signed_bundle_intake"],
                expected_present=3,
                ready_key="v2_signed_bundle_manual_field_ready_count",
            ),
            "v2_signed_bundle_not_validated",
        ),
        _stage(
            "signed_bundle_extraction",
            sources["signed_bundle_extraction"],
            "ready_for_record",
            _summary_int(sources["signed_bundle_extraction"], "v2_signed_bundle_extraction_ready_for_record_count")
            == 1
            and _signed_bundle_extraction_integrity_ready(sources["signed_bundle_extraction"])
            and _signed_bundle_extraction_digest_shape_ready(
                sources["signed_bundle_extraction"],
                expected_present=3,
            ),
            "v2_signed_bundle_not_extracted",
        ),
        _stage(
            "approval_execution_preflight",
            sources["approval_execution_preflight"],
            "ready_for_record",
            _summary_int(sources["approval_execution_preflight"], "v2_approval_preflight_full_ready_count") == 1
            and _approval_preflight_source_integrity_ready(sources["approval_execution_preflight"])
            and _approval_preflight_support_integrity_ready(
                sources["approval_execution_preflight"],
                expected_ready=3,
            )
            and _approval_preflight_digest_shape_ready(
                sources["approval_execution_preflight"],
                expected_present=3,
            ),
            "v2_approval_preflight_not_full_ready",
        ),
        _stage(
            "owner_release_review_record",
            sources["owner_release_review_record"],
            "recorded",
            sources["owner_release_review_record"].get("release_review_recorded") is True
            and _record_preflight_binding_ready(sources["owner_release_review_record"]),
            "owner_release_review_not_recorded",
        ),
        _stage(
            "product_exposure_decision",
            sources["product_exposure_decision"],
            "recorded",
            sources["product_exposure_decision"].get("product_exposure_decision_recorded") is True
            and _record_preflight_binding_ready(sources["product_exposure_decision"]),
            "product_exposure_decision_not_recorded",
        ),
        _stage(
            "release_review_archive",
            sources["release_review_archive"],
            "ready",
            sources["release_review_archive"].get("archive_ready") is True,
            "release_review_archive_not_ready",
        ),
        _stage(
            "owner_release_direction_packet",
            sources["owner_release_direction_packet"],
            "waiting_for_external_signature",
            _summary_int(sources["owner_release_direction_packet"], "owner_release_direction_ready_for_signature_count")
            == 1,
            "owner_release_direction_packet_not_ready",
        ),
        _stage(
            "owner_release_direction_record",
            sources["owner_release_direction_record"],
            "recorded",
            sources["owner_release_direction_record"].get("owner_release_direction_recorded") is True
            and _record_preflight_binding_ready(sources["owner_release_direction_record"]),
            "owner_release_direction_not_recorded",
        ),
    ]


def _stage(
    stage_id: str,
    report: Mapping[str, Any],
    success_state: str,
    ready: bool,
    blocker: str,
) -> dict[str, Any]:
    if not report:
        state = "missing"
        blockers = [f"{stage_id}_artifact_missing"]
    elif ready:
        state = success_state
        blockers = []
    else:
        state = "blocked" if success_state == "ready" else "waiting_for_external_signature"
        blockers = [blocker] + _strings(report.get("blocked_reasons"))
    return {
        "schema_version": 1,
        "id": stage_id,
        "state": state,
        "ready": ready,
        "approval_recorded": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "blocked_reasons": _dedupe(blockers),
    }


def _recommended_next_step(stages: list[Mapping[str, Any]]) -> str:
    for stage in stages:
        if stage["id"] == "reviewer_handoff" and stage["state"] == "waiting_for_external_signature":
            return "send signed-bundle template to real reviewers and wait for returned signatures"
        if stage["id"] == "signed_bundle_intake" and stage["state"] != "ready_for_record":
            return "validate the returned signed bundle with the intake CLI"
        if stage["id"] == "approval_execution_preflight" and stage["state"] != "ready_for_record":
            return "run extraction and approval execution preflight before record validators"
    return "continue the explicit owner/release/product approval record sequence"


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
    return _as_dict(json.loads(path.read_text(encoding="utf-8")))


def _summary_int(report: Mapping[str, Any], key: str) -> int:
    summary = _as_dict(report.get("summary"))
    try:
        return int(summary.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _approval_command_audit_ready(command_audit: Mapping[str, Any]) -> bool:
    return (
        _summary_int(command_audit, "v2_approval_command_audit_ready_count") == 1
        and _summary_int(command_audit, "v2_approval_command_audit_missing_entrypoint_count") == 0
        and _summary_int(command_audit, "v2_approval_command_audit_record_before_preflight_count") == 0
        and _summary_int(command_audit, "v2_approval_command_audit_phase_marker_valid_count") == 1
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


def _record_input_checklist_ready(record_input_checklist: Mapping[str, Any]) -> bool:
    return (
        _summary_int(record_input_checklist, "v2_record_input_checklist_artifact_ready_count") == 1
        and _record_input_support_shape_ready(record_input_checklist)
        and _summary_int(record_input_checklist, "v2_record_input_checklist_unsafe_claim_count") == 0
    )


def _record_input_support_shape_ready(record_input_checklist: Mapping[str, Any]) -> bool:
    return (
        _summary_int(record_input_checklist, "v2_record_input_checklist_support_check_count") == 3
        and _summary_int(record_input_checklist, "v2_record_input_checklist_support_ready_count") >= 2
        and _summary_int(record_input_checklist, "v2_record_input_checklist_support_source_binding_ready_count")
        >= 2
        and _summary_int(record_input_checklist, "v2_record_input_checklist_support_source_binding_blocker_count")
        <= 1
    )


def _approval_preflight_digest_shape_ready(preflight: Mapping[str, Any], *, expected_present: int) -> bool:
    expected_missing = max(0, 3 - expected_present)
    return (
        _summary_int(preflight, "v2_approval_preflight_not_ready_entry_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_signed_payload_digest_present_count") >= expected_present
        and _summary_int(preflight, "v2_approval_preflight_signed_payload_digest_missing_count") <= expected_missing
        and _summary_int(preflight, "v2_approval_preflight_signed_bundle_entry_digest_present_count")
        >= expected_present
        and _summary_int(preflight, "v2_approval_preflight_signed_payload_bundle_digest_match_count")
        >= expected_present
    )


def _approval_preflight_source_integrity_ready(preflight: Mapping[str, Any]) -> bool:
    return (
        _summary_int(preflight, "v2_approval_preflight_source_digest_match_count") == 1
        and _summary_int(preflight, "v2_approval_preflight_source_digest_stale_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_template_digest_mismatch_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_unknown_entry_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_unsigned_template_count") == 0
    )


def _approval_preflight_support_integrity_ready(preflight: Mapping[str, Any], *, expected_ready: int) -> bool:
    return (
        _summary_int(preflight, "v2_approval_preflight_support_ready_count") >= expected_ready
        and _summary_int(preflight, "v2_approval_preflight_support_invalid_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_support_source_binding_ready_count") >= expected_ready
        and _summary_int(preflight, "v2_approval_preflight_support_source_binding_blocker_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_post_record_request_field_emission_count") == 0
        and _summary_int(preflight, "v2_approval_preflight_hard_fail_count") == 0
    )


def _signed_bundle_freshness_guard_ready(freshness: Mapping[str, Any]) -> bool:
    return (
        _summary_int(freshness, "v2_signed_bundle_freshness_guard_ready_count") == 1
        and _summary_int(freshness, "v2_signed_bundle_freshness_current_digest_present_count") == 1
        and _summary_int(freshness, "v2_signed_bundle_freshness_template_digest_match_count") == 1
        and _summary_int(freshness, "v2_signed_bundle_freshness_stale_signed_bundle_count") == 0
        and _summary_int(freshness, "v2_signed_bundle_freshness_unknown_signed_entry_count") == 0
        and _summary_int(freshness, "v2_signed_bundle_freshness_unsafe_claim_count") == 0
    )


def _signed_bundle_intake_integrity_ready(signed_bundle: Mapping[str, Any]) -> bool:
    return (
        _summary_int(signed_bundle, "v2_signed_bundle_source_digest_match_count") == 1
        and _summary_int(signed_bundle, "v2_signed_bundle_source_digest_stale_count") == 0
        and _summary_int(signed_bundle, "v2_signed_bundle_template_digest_mismatch_count") == 0
        and _summary_int(signed_bundle, "v2_signed_bundle_unknown_entry_count") == 0
        and _summary_int(signed_bundle, "v2_signed_bundle_unsigned_template_count") == 0
    )


def _signed_bundle_extraction_integrity_ready(extraction: Mapping[str, Any]) -> bool:
    return (
        _summary_int(extraction, "v2_signed_bundle_extraction_source_digest_match_count") == 1
        and _summary_int(extraction, "v2_signed_bundle_extraction_source_digest_stale_count") == 0
        and _summary_int(extraction, "v2_signed_bundle_extraction_template_digest_mismatch_count") == 0
        and _summary_int(extraction, "v2_signed_bundle_extraction_unknown_entry_count") == 0
        and _summary_int(extraction, "v2_signed_bundle_extraction_unsigned_template_count") == 0
    )


def _signed_bundle_extraction_digest_shape_ready(extraction: Mapping[str, Any], *, expected_present: int) -> bool:
    return (
        _summary_int(extraction, "v2_signed_bundle_extraction_not_ready_entry_count") == 0
        and _summary_int(extraction, "v2_signed_bundle_extraction_signed_entry_digest_present_count") >= expected_present
        and _summary_int(extraction, "v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count")
        >= expected_present
    )


def _signed_bundle_manual_field_shape_ready(
    signed_bundle: Mapping[str, Any],
    *,
    expected_present: int,
    ready_key: str,
) -> bool:
    expected_missing = max(0, 3 - expected_present)
    return (
        _summary_int(signed_bundle, "v2_signed_bundle_manual_field_check_count") >= expected_present
        and _summary_int(signed_bundle, ready_key) >= expected_present
        and _summary_int(signed_bundle, "v2_signed_bundle_manual_field_missing_count") <= expected_missing
        and _summary_int(signed_bundle, "v2_signed_bundle_manual_field_shape_ready_count") == 1
    )


def _record_preflight_binding_ready(record: Mapping[str, Any]) -> bool:
    return (
        record.get("approval_preflight_binding_ready") is True
        and bool(record.get("approval_preflight_digest"))
        and bool(record.get("record_signed_payload_digest"))
        and bool(record.get("approval_preflight_signed_payload_digest"))
        and bool(record.get("approval_preflight_signed_bundle_entry_digest"))
        and record.get("approval_preflight_signed_payload_digest_match") is True
        and record.get("approval_preflight_signed_bundle_entry_digest_match") is True
    )


def _bool_count(value: Any) -> int:
    return 1 if value is True else 0


def _all_ready(*values: int) -> int:
    return 1 if values and all(value == 1 for value in values) else 0


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="", help="Directory containing v2 approval-chain artifacts.")
    parser.add_argument("--no-artifact", action="store_true", help="Print scorecard without writing artifact.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_optimizer_v2_approval_state_scorecard(
        artifact_dir=args.artifact_dir or None,
        write_artifact=not bool(args.no_artifact),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") is not False else 1


__all__ = ["build_optimizer_v2_approval_state_scorecard"]


if __name__ == "__main__":
    raise SystemExit(main())
