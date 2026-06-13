"""Smoke checks for the v2 approval record-input checklist."""

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

from core.turbocore_optimizer_v2_record_input_checklist import (  # noqa: E402
    ARTIFACT,
    build_optimizer_v2_record_input_checklist,
)


def run_smoke() -> dict[str, Any]:
    artifact_payload = build_optimizer_v2_record_input_checklist(write_artifact=True)
    assert artifact_payload["ok"] is True, artifact_payload
    assert artifact_payload["approval_recorded"] is False, artifact_payload
    assert artifact_payload["approval_artifact_written"] is False, artifact_payload
    assert artifact_payload["runtime_dispatch_ready"] is False, artifact_payload
    assert artifact_payload["native_dispatch_allowed"] is False, artifact_payload
    assert artifact_payload["training_path_enabled"] is False, artifact_payload
    assert artifact_payload["product_native_ready"] is False, artifact_payload
    assert ARTIFACT.exists(), ARTIFACT

    checklist = build_optimizer_v2_record_input_checklist(
        source_reports=_source_reports_waiting_for_signature(),
        write_artifact=False,
    )
    summary = checklist["summary"]
    assert checklist["ok"] is True, checklist
    assert checklist["record_input_checklist_artifact_ready"] is True, checklist
    assert checklist["record_input_checklist_full_ready"] is False, checklist
    assert checklist["approval_recorded"] is False, checklist
    assert checklist["runtime_dispatch_ready"] is False, checklist
    assert checklist["native_dispatch_allowed"] is False, checklist
    assert checklist["training_path_enabled"] is False, checklist
    assert checklist["product_native_ready"] is False, checklist
    assert checklist["missing_item_ids_by_phase"] == {
        "phase1_records": [
            "signed_bundle",
            "signed_owner_release_review",
            "signed_product_exposure_review",
        ],
        "phase2_direction": ["signed_owner_release_direction"],
    }, checklist
    assert summary["v2_record_input_checklist_item_count"] == 13, checklist
    assert summary["v2_record_input_checklist_present_item_count"] == 9, checklist
    assert summary["v2_record_input_checklist_valid_json_item_count"] == 9, checklist
    assert summary["v2_record_input_checklist_missing_item_count"] == 4, checklist
    assert summary["v2_record_input_checklist_ready_item_count"] == 3, checklist
    assert summary["v2_record_input_checklist_phase1_item_count"] == 6, checklist
    assert summary["v2_record_input_checklist_phase1_ready_item_count"] == 3, checklist
    assert summary["v2_record_input_checklist_phase1_missing_item_count"] == 3, checklist
    assert summary["v2_record_input_checklist_phase1_ready_count"] == 0, checklist
    assert summary["v2_record_input_checklist_phase2_item_count"] == 7, checklist
    assert summary["v2_record_input_checklist_phase2_ready_item_count"] == 0, checklist
    assert summary["v2_record_input_checklist_phase2_missing_item_count"] == 1, checklist
    assert summary["v2_record_input_checklist_phase2_ready_count"] == 0, checklist
    assert summary["v2_record_input_checklist_full_ready_count"] == 0, checklist
    assert summary["v2_record_input_checklist_support_check_count"] == 3, checklist
    assert summary["v2_record_input_checklist_support_ready_count"] == 2, checklist
    assert summary["v2_record_input_checklist_support_blocked_item_count"] == 1, checklist
    assert summary["v2_record_input_checklist_support_blocker_count"] == 1, checklist
    assert summary["v2_record_input_checklist_support_source_binding_ready_count"] == 2, checklist
    assert summary["v2_record_input_checklist_support_source_binding_blocker_count"] == 0, checklist
    assert summary["v2_record_input_checklist_artifact_ready_count"] == 1, checklist
    assert summary["v2_record_input_checklist_approval_recorded_count"] == 0, checklist
    assert summary["v2_record_input_checklist_runtime_dispatch_ready_count"] == 0, checklist
    assert summary["v2_record_input_checklist_native_dispatch_allowed_count"] == 0, checklist
    assert summary["v2_record_input_checklist_training_path_enabled_count"] == 0, checklist
    assert summary["v2_record_input_checklist_product_native_ready_count"] == 0, checklist
    assert summary["v2_record_input_checklist_default_behavior_changed_count"] == 0, checklist
    assert summary["v2_record_input_checklist_unsafe_claim_count"] == 0, checklist

    unsafe_checklist = build_optimizer_v2_record_input_checklist(
        source_reports={
            **_source_reports_waiting_for_signature(),
            "product_exposure_decision": {
                "product_exposure_decision_recorded": False,
                "training_path_enabled": True,
            },
        },
        write_artifact=False,
    )
    assert unsafe_checklist["ok"] is False, unsafe_checklist
    assert unsafe_checklist["summary"]["v2_record_input_checklist_unsafe_claim_count"] == 1, unsafe_checklist

    invalid_support_checklist = build_optimizer_v2_record_input_checklist(
        source_reports={
            **_source_reports_waiting_for_signature(),
            "signed_bundle": {"signed_entries": {}},
            "signed_owner_release_review": {"reviewer": "owner"},
            "signed_product_exposure_review": {"reviewer": "product"},
            "training_launch_contract": {"ok": True},
        },
        write_artifact=False,
    )
    assert invalid_support_checklist["ok"] is True, invalid_support_checklist
    assert invalid_support_checklist["summary"]["v2_record_input_checklist_phase1_ready_count"] == 0, (
        invalid_support_checklist
    )
    assert invalid_support_checklist["summary"]["v2_record_input_checklist_support_check_count"] == 3, (
        invalid_support_checklist
    )
    assert invalid_support_checklist["summary"]["v2_record_input_checklist_support_ready_count"] == 1, (
        invalid_support_checklist
    )
    assert invalid_support_checklist["summary"]["v2_record_input_checklist_support_blocked_item_count"] == 2, (
        invalid_support_checklist
    )
    assert invalid_support_checklist["summary"]["v2_record_input_checklist_support_blocker_count"] == 5, (
        invalid_support_checklist
    )
    assert (
        invalid_support_checklist["summary"]["v2_record_input_checklist_support_source_binding_ready_count"] == 1
    ), invalid_support_checklist
    assert (
        invalid_support_checklist["summary"]["v2_record_input_checklist_support_source_binding_blocker_count"] == 0
    ), invalid_support_checklist
    assert "training_launch_contract_package_mismatch" in invalid_support_checklist["blocked_reasons"], (
        invalid_support_checklist
    )
    invalid_items = {item["id"]: item for item in invalid_support_checklist["items"]}
    assert "training_launch_contract_package_mismatch" in invalid_items["training_launch_contract"][
        "support_blockers"
    ], invalid_support_checklist

    missing_source_reports = _source_reports_waiting_for_signature()
    missing_source_reports["training_launch_contract"] = {
        key: value
        for key, value in missing_source_reports["training_launch_contract"].items()
        if key
        not in (
            "training_step_execution_contract_summary",
            "training_launch_evidence_summary",
        )
    }
    missing_source_checklist = build_optimizer_v2_record_input_checklist(
        source_reports=missing_source_reports,
        write_artifact=False,
    )
    missing_source_summary = missing_source_checklist["summary"]
    missing_source_items = {item["id"]: item for item in missing_source_checklist["items"]}
    assert missing_source_checklist["ok"] is True, missing_source_checklist
    assert missing_source_summary["v2_record_input_checklist_support_ready_count"] == 1, (
        missing_source_checklist
    )
    assert missing_source_summary["v2_record_input_checklist_support_source_binding_ready_count"] == 1, (
        missing_source_checklist
    )
    assert missing_source_summary["v2_record_input_checklist_support_source_binding_blocker_count"] == 4, (
        missing_source_checklist
    )
    assert "training_launch_contract_training_step_source_missing" in missing_source_items[
        "training_launch_contract"
    ]["support_source_binding_blockers"], missing_source_checklist

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_record_input_checklist_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "deterministic_waiting_state_checked": True,
        "invalid_support_artifact_rejected": True,
        "missing_support_source_binding_rejected": True,
        "summary": summary,
        "recommended_next_step": checklist["recommended_next_step"],
    }


def _source_reports_waiting_for_signature() -> dict[str, dict[str, Any]]:
    return {
        "signature_bundle": {
            "ok": True,
            "summary": {"v2_signature_bundle_ready_for_signature_count": 2},
        },
        "training_launch_contract": {
            "package": "turbocore_native_update_training_launch_contract_v0",
            "gate": "native_update_training_launch_contract",
            "ok": True,
            "evidence_ready": True,
            "ready_for_training_launch_review": True,
            "post_training_launch_request_fields": {},
            "training_step_execution_contract_summary": {
                "source": "smoke://training_step_execution_contract",
                "digest": "training-step-digest",
            },
            "training_launch_evidence_summary": {
                "source": "smoke://training_launch_evidence",
                "digest": "training-launch-digest",
            },
        },
        "product_exposure_evidence": {
            "evidence": "native_update_product_exposure_evidence_v0",
            "source": "smoke://native_update_product_exposure_evidence",
            "ok": True,
            "default_off": True,
            "report_only": True,
            "contract_only": True,
            "product_exposure_decision_only": True,
            "records_evidence_only": True,
            "manual_only": True,
            "internal_only": True,
            "requires_explicit_owner_approval": True,
            "requires_explicit_operator_opt_in": True,
            "product_exposure_decision_ready": True,
        },
        "owner_release_review_record": {"release_review_recorded": False},
        "product_exposure_decision": {"product_exposure_decision_recorded": False},
        "release_review_archive": {"archive_ready": False},
        "owner_release_direction_packet": {
            "package": "turbocore_native_update_owner_release_direction_packet_v0",
            "gate": "native_update_owner_release_direction",
            "ok": True,
            "ready_for_owner_direction_signature": False,
            "post_owner_release_request_fields": {},
        },
        "phase2_signature_bundle": {
            "summary": {"v2_signature_bundle_owner_direction_ready_count": 0},
        },
        "phase2_reviewer_handoff": {
            "summary": {"v2_reviewer_handoff_signed_template_entry_count": 2},
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
