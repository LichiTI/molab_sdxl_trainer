"""Smoke checks for signed v2 signature-bundle intake."""

from __future__ import annotations

import contextlib
import hashlib
import io
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

from core.turbocore_native_update_owner_release_direction_packet import (  # noqa: E402
    build_native_update_owner_release_direction_packet,
)
from core.turbocore_native_update_release_review_archive import (  # noqa: E402
    build_native_update_release_review_archive,
)
from core.turbocore_optimizer_v2_signed_bundle_intake_record import (  # noqa: E402
    build_optimizer_v2_signed_bundle_intake_record,
    main as intake_cli_main,
)
from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    build_optimizer_v2_signature_bundle_packet,
)


ARTIFACT = (
    REPO_ROOT
    / "temp"
    / "turbocore_optimizer"
    / "turbocore_optimizer_v2_signed_bundle_intake_record.json"
)


def run_smoke() -> dict[str, Any]:
    pending = build_optimizer_v2_signed_bundle_intake_record(write_artifact=True)
    pending_summary = pending["summary"]
    assert pending["ok"] is True, pending
    assert pending["signed_bundle_present"] is False, pending
    assert pending["signed_bundle_valid"] is False, pending
    assert pending["approval_recorded"] is False, pending
    assert pending["approval_artifact_written"] is False, pending
    assert pending["runtime_dispatch_ready"] is False, pending
    assert pending["native_dispatch_allowed"] is False, pending
    assert pending["training_path_enabled"] is False, pending
    assert pending["product_native_ready"] is False, pending
    assert pending_summary["v2_signed_bundle_entry_count"] == 3, pending
    assert pending_summary["v2_signed_bundle_present_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_valid_record_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_phase1_valid_record_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_phase1_ready_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_phase2_valid_record_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_phase2_ready_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_full_ready_count"] == 0, pending
    assert pending["missing_signed_signature_ids"] == [
        "owner_release_review",
        "product_exposure_review",
        "owner_release_direction",
    ], pending
    assert pending_summary["v2_signed_bundle_missing_signature_count"] == 3, pending
    assert pending_summary["v2_signed_bundle_phase1_missing_signature_count"] == 2, pending
    assert pending_summary["v2_signed_bundle_phase2_missing_signature_count"] == 1, pending
    assert pending_summary["v2_signed_bundle_not_ready_entry_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_manual_field_check_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_manual_field_ready_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_manual_field_missing_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_manual_field_shape_ready_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_approval_artifact_written_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_runtime_dispatch_ready_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_native_dispatch_allowed_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_training_path_enabled_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_product_native_ready_count"] == 0, pending
    assert ARTIFACT.exists(), ARTIFACT

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = Path(temp_dir)
        base_bundle = build_optimizer_v2_signature_bundle_packet(write_artifact=False)
        owner_packet = _read_artifact("native_update_owner_release_review_packet.json")
        product_exposure = _read_artifact("native_update_product_exposure_decision.json")
        owner_review_signed = _signed_entry(base_bundle, "owner_release_review")
        product_exposure_signed = _signed_entry(base_bundle, "product_exposure_review")
        archive = build_native_update_release_review_archive(
            release_review_package=_recorded_release_review_package(),
            owner_release_review_record=_owner_release_review_record_ready(),
            stable_first_release_scope=_stable_first_release_scope(),
            write_artifact=True,
            artifact_path=artifact_dir / "native_update_release_review_archive.json",
        )
        direction_packet = build_native_update_owner_release_direction_packet(
            release_review_archive=archive,
            product_exposure_decision=_product_exposure_decision_ready(),
            stable_first_release_scope=_stable_first_release_scope(),
            artifact_dir=artifact_dir,
            write_artifact=False,
        )
        direction_bundle = build_optimizer_v2_signature_bundle_packet(
            source_reports={
                "remaining_gate_handoff": {
                    "handoff_ready": True,
                    "summary": {
                        "v2_remaining_gate_total_count": 6,
                        "v2_remaining_gate_default_off_guard_count": 1,
                    },
                },
                "owner_release_review_packet": owner_packet,
                "product_exposure_decision": product_exposure,
                "owner_release_direction_packet": direction_packet,
            },
            write_artifact=False,
        )
        signed_bundle = {
            "source_signature_bundle_digest": _digest_payload(direction_bundle),
            "signed_entries": {
                "owner_release_review": owner_review_signed,
                "product_exposure_review": product_exposure_signed,
                "owner_release_direction": _signed_entry(direction_bundle, "owner_release_direction"),
            }
        }
        phase1_signed_bundle = {
            "source_signature_bundle_digest": _digest_payload(direction_bundle),
            "signed_entries": {
                "owner_release_review": owner_review_signed,
                "product_exposure_review": product_exposure_signed,
            },
        }
        recorded = build_optimizer_v2_signed_bundle_intake_record(
            signature_bundle=direction_bundle,
            signed_bundle=signed_bundle,
            owner_review_packet=owner_packet,
            product_exposure_decision=product_exposure,
            owner_direction_packet=direction_packet,
            write_artifact=False,
            write_approval_artifacts=False,
        )
        phase1_recorded = build_optimizer_v2_signed_bundle_intake_record(
            signature_bundle=direction_bundle,
            signed_bundle=phase1_signed_bundle,
            owner_review_packet=owner_packet,
            product_exposure_decision=product_exposure,
            owner_direction_packet=direction_packet,
            write_artifact=False,
            write_approval_artifacts=False,
        )
        premature_direction_bundle = json.loads(json.dumps(direction_bundle))
        for entry in premature_direction_bundle["signature_entries"]:
            if entry["signature_id"] == "owner_release_direction":
                entry["ready_for_signature"] = False
                entry["blocked_reasons"] = ["owner_release_direction_phase2_not_ready_for_signature"]
        premature_direction_signed_bundle = json.loads(json.dumps(signed_bundle))
        premature_direction_signed_bundle["source_signature_bundle_digest"] = _digest_payload(premature_direction_bundle)
        premature_direction_recorded = build_optimizer_v2_signed_bundle_intake_record(
            signature_bundle=premature_direction_bundle,
            signed_bundle=premature_direction_signed_bundle,
            owner_review_packet=owner_packet,
            product_exposure_decision=product_exposure,
            owner_direction_packet=direction_packet,
            write_artifact=False,
            write_approval_artifacts=False,
        )
        stale_signed_bundle = dict(signed_bundle)
        stale_signed_bundle["source_signature_bundle_digest"] = "stale-digest-for-intake-smoke"
        stale_recorded = build_optimizer_v2_signed_bundle_intake_record(
            signature_bundle=direction_bundle,
            signed_bundle=stale_signed_bundle,
            owner_review_packet=owner_packet,
            product_exposure_decision=product_exposure,
            owner_direction_packet=direction_packet,
            write_artifact=False,
            write_approval_artifacts=False,
        )
        unsigned_template_signed_bundle = dict(signed_bundle)
        unsigned_template_signed_bundle["unsigned_template"] = True
        unsigned_template_recorded = build_optimizer_v2_signed_bundle_intake_record(
            signature_bundle=direction_bundle,
            signed_bundle=unsigned_template_signed_bundle,
            owner_review_packet=owner_packet,
            product_exposure_decision=product_exposure,
            owner_direction_packet=direction_packet,
            write_artifact=False,
            write_approval_artifacts=False,
        )
        template_digest_mismatch_bundle = json.loads(json.dumps(signed_bundle))
        template_digest_mismatch_bundle["signed_entries"]["product_exposure_review"][
            "source_v2_signature_template_digest"
        ] = "mismatched-product-exposure-template-digest"
        template_digest_mismatch_recorded = build_optimizer_v2_signed_bundle_intake_record(
            signature_bundle=direction_bundle,
            signed_bundle=template_digest_mismatch_bundle,
            owner_review_packet=owner_packet,
            product_exposure_decision=product_exposure,
            owner_direction_packet=direction_packet,
            write_artifact=False,
            write_approval_artifacts=False,
        )
        unknown_entry_bundle = json.loads(json.dumps(signed_bundle))
        unknown_entry_bundle["signed_entries"]["unexpected_review"] = dict(owner_review_signed)
        unknown_entry_recorded = build_optimizer_v2_signed_bundle_intake_record(
            signature_bundle=direction_bundle,
            signed_bundle=unknown_entry_bundle,
            owner_review_packet=owner_packet,
            product_exposure_decision=product_exposure,
            owner_direction_packet=direction_packet,
            write_artifact=False,
            write_approval_artifacts=False,
        )
        manual_field_missing_bundle = json.loads(json.dumps(signed_bundle))
        manual_field_missing_bundle["signed_entries"]["owner_release_review"]["reviewer"] = ""
        manual_field_missing_bundle["signed_entries"]["product_exposure_review"][
            "acknowledge_no_backend_router_registration"
        ] = False
        manual_field_missing_recorded = build_optimizer_v2_signed_bundle_intake_record(
            signature_bundle=direction_bundle,
            signed_bundle=manual_field_missing_bundle,
            owner_review_packet=owner_packet,
            product_exposure_decision=product_exposure,
            owner_direction_packet=direction_packet,
            write_artifact=False,
            write_approval_artifacts=False,
        )
        signature_bundle_path = artifact_dir / "turbocore_optimizer_v2_signature_bundle_packet.json"
        signed_bundle_path = artifact_dir / "turbocore_optimizer_v2_signed_bundle.json"
        stale_signed_bundle_path = artifact_dir / "turbocore_optimizer_v2_signed_bundle.stale.json"
        unsigned_template_signed_bundle_path = artifact_dir / "turbocore_optimizer_v2_signed_bundle.unsigned_template.json"
        template_digest_mismatch_bundle_path = artifact_dir / "turbocore_optimizer_v2_signed_bundle.template_mismatch.json"
        unknown_entry_bundle_path = artifact_dir / "turbocore_optimizer_v2_signed_bundle.unknown_entry.json"
        premature_direction_bundle_path = artifact_dir / "turbocore_optimizer_v2_signature_bundle_packet.not_ready_direction.json"
        premature_direction_signed_bundle_path = artifact_dir / "turbocore_optimizer_v2_signed_bundle.not_ready_direction.json"
        owner_packet_path = artifact_dir / "native_update_owner_release_review_packet.json"
        product_exposure_path = artifact_dir / "native_update_product_exposure_decision.json"
        direction_packet_path = artifact_dir / "native_update_owner_release_direction_packet.json"
        _write_json(signature_bundle_path, direction_bundle)
        _write_json(signed_bundle_path, signed_bundle)
        _write_json(stale_signed_bundle_path, stale_signed_bundle)
        _write_json(unsigned_template_signed_bundle_path, unsigned_template_signed_bundle)
        _write_json(template_digest_mismatch_bundle_path, template_digest_mismatch_bundle)
        _write_json(unknown_entry_bundle_path, unknown_entry_bundle)
        _write_json(premature_direction_bundle_path, premature_direction_bundle)
        _write_json(premature_direction_signed_bundle_path, premature_direction_signed_bundle)
        _write_json(owner_packet_path, owner_packet)
        _write_json(product_exposure_path, product_exposure)
        _write_json(direction_packet_path, direction_packet)
        cli_payload = _run_intake_cli(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(signed_bundle_path),
                "--owner-review-packet",
                str(owner_packet_path),
                "--product-exposure-decision",
                str(product_exposure_path),
                "--owner-direction-packet",
                str(direction_packet_path),
                "--no-artifact",
            ]
        )
        stale_cli_payload, stale_cli_exit = _run_intake_cli_allow_failure(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(stale_signed_bundle_path),
                "--owner-review-packet",
                str(owner_packet_path),
                "--product-exposure-decision",
                str(product_exposure_path),
                "--owner-direction-packet",
                str(direction_packet_path),
                "--no-artifact",
            ]
        )
        unsigned_template_cli_payload, unsigned_template_cli_exit = _run_intake_cli_allow_failure(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(unsigned_template_signed_bundle_path),
                "--owner-review-packet",
                str(owner_packet_path),
                "--product-exposure-decision",
                str(product_exposure_path),
                "--owner-direction-packet",
                str(direction_packet_path),
                "--no-artifact",
            ]
        )
        template_digest_mismatch_cli_payload, template_digest_mismatch_cli_exit = _run_intake_cli_allow_failure(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(template_digest_mismatch_bundle_path),
                "--owner-review-packet",
                str(owner_packet_path),
                "--product-exposure-decision",
                str(product_exposure_path),
                "--owner-direction-packet",
                str(direction_packet_path),
                "--no-artifact",
            ]
        )
        unknown_entry_cli_payload, unknown_entry_cli_exit = _run_intake_cli_allow_failure(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(unknown_entry_bundle_path),
                "--owner-review-packet",
                str(owner_packet_path),
                "--product-exposure-decision",
                str(product_exposure_path),
                "--owner-direction-packet",
                str(direction_packet_path),
                "--no-artifact",
            ]
        )
        premature_direction_cli_payload, premature_direction_cli_exit = _run_intake_cli_allow_failure(
            [
                "--signature-bundle",
                str(premature_direction_bundle_path),
                "--signed-bundle",
                str(premature_direction_signed_bundle_path),
                "--owner-review-packet",
                str(owner_packet_path),
                "--product-exposure-decision",
                str(product_exposure_path),
                "--owner-direction-packet",
                str(direction_packet_path),
                "--no-artifact",
            ]
        )
    recorded_summary = recorded["summary"]
    assert recorded["ok"] is True, recorded
    assert recorded["signed_bundle_present"] is True, recorded
    assert recorded["signed_bundle_source_digest_match"] is True, recorded
    assert recorded["signed_bundle_source_digest_stale"] is False, recorded
    assert recorded["signed_bundle_valid"] is True, recorded
    assert recorded["records_all_valid"] is True, recorded
    assert recorded["approval_recorded"] is False, recorded
    assert recorded["approval_artifact_written"] is False, recorded
    assert recorded_summary["v2_signed_bundle_present_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_source_digest_match_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_source_digest_stale_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_unsigned_template_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_template_digest_mismatch_count"] == 0, recorded
    assert recorded["missing_signed_signature_ids"] == [], recorded
    assert recorded_summary["v2_signed_bundle_missing_signature_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_phase1_missing_signature_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_phase2_missing_signature_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_not_ready_entry_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_manual_field_check_count"] == 3, recorded
    assert recorded_summary["v2_signed_bundle_manual_field_ready_count"] == 3, recorded
    assert recorded_summary["v2_signed_bundle_manual_field_missing_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_manual_field_missing_signature_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_manual_field_shape_ready_count"] == 1, recorded
    assert recorded["missing_manual_fields_by_signature_id"] == {}, recorded
    assert recorded_summary["v2_signed_bundle_phase1_manual_field_ready_count"] == 2, recorded
    assert recorded_summary["v2_signed_bundle_phase2_manual_field_ready_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_valid_record_count"] == 3, recorded
    assert recorded_summary["v2_signed_bundle_phase1_valid_record_count"] == 2, recorded
    assert recorded_summary["v2_signed_bundle_phase1_ready_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_phase2_valid_record_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_phase2_ready_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_full_ready_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_owner_review_recorded_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_product_exposure_recorded_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_owner_direction_recorded_count"] == 1, recorded
    assert recorded_summary["v2_signed_bundle_approval_artifact_written_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_runtime_dispatch_ready_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_native_dispatch_allowed_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_training_path_enabled_count"] == 0, recorded
    assert recorded_summary["v2_signed_bundle_product_native_ready_count"] == 0, recorded
    phase1_summary = phase1_recorded["summary"]
    assert phase1_recorded["ok"] is True, phase1_recorded
    assert phase1_recorded["signed_bundle_present"] is True, phase1_recorded
    assert phase1_recorded["signed_bundle_source_digest_match"] is True, phase1_recorded
    assert phase1_recorded["signed_bundle_valid"] is False, phase1_recorded
    assert phase1_recorded["records_all_valid"] is False, phase1_recorded
    assert phase1_summary["v2_signed_bundle_valid_record_count"] == 2, phase1_recorded
    assert phase1_summary["v2_signed_bundle_not_ready_entry_count"] == 0, phase1_recorded
    assert phase1_summary["v2_signed_bundle_manual_field_check_count"] == 2, phase1_recorded
    assert phase1_summary["v2_signed_bundle_manual_field_ready_count"] == 2, phase1_recorded
    assert phase1_summary["v2_signed_bundle_manual_field_missing_count"] == 0, phase1_recorded
    assert phase1_summary["v2_signed_bundle_manual_field_missing_signature_count"] == 0, phase1_recorded
    assert phase1_summary["v2_signed_bundle_manual_field_shape_ready_count"] == 1, phase1_recorded
    assert phase1_summary["v2_signed_bundle_phase1_manual_field_ready_count"] == 2, phase1_recorded
    assert phase1_summary["v2_signed_bundle_phase2_manual_field_ready_count"] == 0, phase1_recorded
    assert phase1_summary["v2_signed_bundle_phase1_valid_record_count"] == 2, phase1_recorded
    assert phase1_summary["v2_signed_bundle_phase1_ready_count"] == 1, phase1_recorded
    assert phase1_summary["v2_signed_bundle_phase2_valid_record_count"] == 0, phase1_recorded
    assert phase1_summary["v2_signed_bundle_phase2_ready_count"] == 0, phase1_recorded
    assert phase1_summary["v2_signed_bundle_full_ready_count"] == 0, phase1_recorded
    assert phase1_recorded["missing_signed_signature_ids"] == ["owner_release_direction"], phase1_recorded
    assert phase1_summary["v2_signed_bundle_missing_signature_count"] == 1, phase1_recorded
    assert phase1_summary["v2_signed_bundle_phase1_missing_signature_count"] == 0, phase1_recorded
    assert phase1_summary["v2_signed_bundle_phase2_missing_signature_count"] == 1, phase1_recorded
    assert phase1_summary["v2_signed_bundle_approval_artifact_written_count"] == 0, phase1_recorded
    premature_direction_summary = premature_direction_recorded["summary"]
    assert premature_direction_recorded["ok"] is False, premature_direction_recorded
    assert premature_direction_recorded["signed_bundle_present"] is True, premature_direction_recorded
    assert premature_direction_recorded["signed_bundle_source_digest_match"] is True, premature_direction_recorded
    assert premature_direction_recorded["signed_bundle_valid"] is False, premature_direction_recorded
    assert premature_direction_recorded["records_all_valid"] is False, premature_direction_recorded
    assert (
        "signed_entry_not_ready_for_signature:owner_release_direction"
        in premature_direction_recorded["blocked_reasons"]
    ), premature_direction_recorded
    assert premature_direction_summary["v2_signed_bundle_not_ready_entry_count"] == 1, premature_direction_recorded
    assert premature_direction_summary["v2_signed_bundle_valid_record_count"] == 0, premature_direction_recorded
    assert premature_direction_summary["v2_signed_bundle_phase1_ready_count"] == 0, premature_direction_recorded
    assert premature_direction_summary["v2_signed_bundle_phase2_ready_count"] == 0, premature_direction_recorded
    assert premature_direction_summary["v2_signed_bundle_full_ready_count"] == 0, premature_direction_recorded
    assert premature_direction_summary["v2_signed_bundle_approval_artifact_written_count"] == 0, premature_direction_recorded
    stale_summary = stale_recorded["summary"]
    assert stale_recorded["ok"] is False, stale_recorded
    assert stale_recorded["signed_bundle_present"] is True, stale_recorded
    assert stale_recorded["signed_bundle_source_digest_match"] is False, stale_recorded
    assert stale_recorded["signed_bundle_source_digest_stale"] is True, stale_recorded
    assert stale_recorded["signed_bundle_valid"] is False, stale_recorded
    assert stale_recorded["records_all_valid"] is False, stale_recorded
    assert stale_summary["v2_signed_bundle_source_digest_match_count"] == 0, stale_recorded
    assert stale_summary["v2_signed_bundle_source_digest_stale_count"] == 1, stale_recorded
    assert stale_summary["v2_signed_bundle_valid_record_count"] == 0, stale_recorded
    assert stale_summary["v2_signed_bundle_phase1_ready_count"] == 0, stale_recorded
    assert stale_summary["v2_signed_bundle_phase2_ready_count"] == 0, stale_recorded
    assert stale_summary["v2_signed_bundle_full_ready_count"] == 0, stale_recorded
    assert stale_summary["v2_signed_bundle_approval_artifact_written_count"] == 0, stale_recorded
    unsigned_template_summary = unsigned_template_recorded["summary"]
    assert unsigned_template_recorded["ok"] is False, unsigned_template_recorded
    assert unsigned_template_recorded["signed_bundle_present"] is True, unsigned_template_recorded
    assert unsigned_template_recorded["signed_bundle_source_digest_match"] is True, unsigned_template_recorded
    assert unsigned_template_recorded["signed_bundle_unsigned_template_marker"] is True, unsigned_template_recorded
    assert unsigned_template_recorded["signed_bundle_valid"] is False, unsigned_template_recorded
    assert unsigned_template_recorded["records_all_valid"] is False, unsigned_template_recorded
    assert "signed_bundle_unsigned_template_marker_present" in unsigned_template_recorded["blocked_reasons"], unsigned_template_recorded
    assert unsigned_template_summary["v2_signed_bundle_unsigned_template_count"] == 1, unsigned_template_recorded
    assert unsigned_template_summary["v2_signed_bundle_valid_record_count"] == 0, unsigned_template_recorded
    assert unsigned_template_summary["v2_signed_bundle_phase1_ready_count"] == 0, unsigned_template_recorded
    assert unsigned_template_summary["v2_signed_bundle_phase2_ready_count"] == 0, unsigned_template_recorded
    assert unsigned_template_summary["v2_signed_bundle_full_ready_count"] == 0, unsigned_template_recorded
    assert unsigned_template_summary["v2_signed_bundle_approval_artifact_written_count"] == 0, unsigned_template_recorded
    template_mismatch_summary = template_digest_mismatch_recorded["summary"]
    assert template_digest_mismatch_recorded["ok"] is False, template_digest_mismatch_recorded
    assert template_digest_mismatch_recorded["signed_bundle_present"] is True, template_digest_mismatch_recorded
    assert template_digest_mismatch_recorded["signed_bundle_source_digest_match"] is True, template_digest_mismatch_recorded
    assert template_digest_mismatch_recorded["signed_bundle_valid"] is False, template_digest_mismatch_recorded
    assert template_digest_mismatch_recorded["records_all_valid"] is False, template_digest_mismatch_recorded
    assert (
        "product_exposure_review_source_v2_signature_template_digest_mismatch"
        in template_digest_mismatch_recorded["blocked_reasons"]
    ), template_digest_mismatch_recorded
    assert template_mismatch_summary["v2_signed_bundle_template_digest_mismatch_count"] == 1, template_digest_mismatch_recorded
    assert template_mismatch_summary["v2_signed_bundle_valid_record_count"] == 0, template_digest_mismatch_recorded
    assert template_mismatch_summary["v2_signed_bundle_phase1_ready_count"] == 0, template_digest_mismatch_recorded
    assert template_mismatch_summary["v2_signed_bundle_phase2_ready_count"] == 0, template_digest_mismatch_recorded
    assert template_mismatch_summary["v2_signed_bundle_full_ready_count"] == 0, template_digest_mismatch_recorded
    assert template_mismatch_summary["v2_signed_bundle_approval_artifact_written_count"] == 0, template_digest_mismatch_recorded
    unknown_entry_summary = unknown_entry_recorded["summary"]
    assert unknown_entry_recorded["ok"] is False, unknown_entry_recorded
    assert unknown_entry_recorded["signed_bundle_present"] is True, unknown_entry_recorded
    assert unknown_entry_recorded["signed_bundle_source_digest_match"] is True, unknown_entry_recorded
    assert unknown_entry_recorded["signed_bundle_valid"] is False, unknown_entry_recorded
    assert unknown_entry_recorded["records_all_valid"] is False, unknown_entry_recorded
    assert unknown_entry_recorded["unknown_signed_signature_ids"] == ["unexpected_review"], unknown_entry_recorded
    assert "unknown_signed_entry:unexpected_review" in unknown_entry_recorded["blocked_reasons"], unknown_entry_recorded
    assert unknown_entry_summary["v2_signed_bundle_unknown_entry_count"] == 1, unknown_entry_recorded
    assert unknown_entry_summary["v2_signed_bundle_valid_record_count"] == 0, unknown_entry_recorded
    assert unknown_entry_summary["v2_signed_bundle_phase1_ready_count"] == 0, unknown_entry_recorded
    assert unknown_entry_summary["v2_signed_bundle_phase2_ready_count"] == 0, unknown_entry_recorded
    assert unknown_entry_summary["v2_signed_bundle_full_ready_count"] == 0, unknown_entry_recorded
    assert unknown_entry_summary["v2_signed_bundle_approval_artifact_written_count"] == 0, unknown_entry_recorded
    manual_field_missing_summary = manual_field_missing_recorded["summary"]
    assert manual_field_missing_recorded["ok"] is False, manual_field_missing_recorded
    assert manual_field_missing_recorded["signed_bundle_present"] is True, manual_field_missing_recorded
    assert manual_field_missing_recorded["signed_bundle_source_digest_match"] is True, manual_field_missing_recorded
    assert manual_field_missing_recorded["signed_bundle_valid"] is False, manual_field_missing_recorded
    assert manual_field_missing_recorded["records_all_valid"] is False, manual_field_missing_recorded
    assert (
        "owner_release_review_manual_field_missing:reviewer" in manual_field_missing_recorded["blocked_reasons"]
    ), manual_field_missing_recorded
    assert (
        "product_exposure_review_manual_field_missing:acknowledge_no_backend_router_registration"
        in manual_field_missing_recorded["blocked_reasons"]
    ), manual_field_missing_recorded
    assert manual_field_missing_summary["v2_signed_bundle_manual_field_check_count"] == 3, manual_field_missing_recorded
    assert manual_field_missing_summary["v2_signed_bundle_manual_field_ready_count"] == 1, manual_field_missing_recorded
    assert manual_field_missing_summary["v2_signed_bundle_manual_field_missing_count"] == 2, manual_field_missing_recorded
    assert manual_field_missing_summary["v2_signed_bundle_manual_field_missing_signature_count"] == 2, manual_field_missing_recorded
    assert manual_field_missing_summary["v2_signed_bundle_manual_field_shape_ready_count"] == 0, manual_field_missing_recorded
    assert manual_field_missing_recorded["missing_manual_fields_by_signature_id"] == {
        "owner_release_review": ["reviewer"],
        "product_exposure_review": ["acknowledge_no_backend_router_registration"],
    }, manual_field_missing_recorded
    assert manual_field_missing_summary["v2_signed_bundle_phase1_ready_count"] == 0, manual_field_missing_recorded
    assert manual_field_missing_summary["v2_signed_bundle_phase2_ready_count"] == 0, manual_field_missing_recorded
    assert manual_field_missing_summary["v2_signed_bundle_full_ready_count"] == 0, manual_field_missing_recorded
    assert manual_field_missing_summary["v2_signed_bundle_approval_artifact_written_count"] == 0, manual_field_missing_recorded
    assert cli_payload["ok"] is True, cli_payload
    assert cli_payload["signed_bundle_present"] is True, cli_payload
    assert cli_payload["signed_bundle_source_digest_match"] is True, cli_payload
    assert cli_payload["signed_bundle_valid"] is True, cli_payload
    assert cli_payload["approval_artifact_written"] is False, cli_payload
    assert cli_payload["summary"]["v2_signed_bundle_approval_artifact_written_count"] == 0, cli_payload
    assert stale_cli_exit == 1, stale_cli_payload
    assert stale_cli_payload["ok"] is False, stale_cli_payload
    assert stale_cli_payload["summary"]["v2_signed_bundle_source_digest_stale_count"] == 1, stale_cli_payload
    assert unsigned_template_cli_exit == 1, unsigned_template_cli_payload
    assert unsigned_template_cli_payload["ok"] is False, unsigned_template_cli_payload
    assert unsigned_template_cli_payload["summary"]["v2_signed_bundle_unsigned_template_count"] == 1, unsigned_template_cli_payload
    assert template_digest_mismatch_cli_exit == 1, template_digest_mismatch_cli_payload
    assert template_digest_mismatch_cli_payload["ok"] is False, template_digest_mismatch_cli_payload
    assert template_digest_mismatch_cli_payload["signed_bundle_valid"] is False, template_digest_mismatch_cli_payload
    assert template_digest_mismatch_cli_payload["summary"]["v2_signed_bundle_template_digest_mismatch_count"] == 1, template_digest_mismatch_cli_payload
    assert unknown_entry_cli_exit == 1, unknown_entry_cli_payload
    assert unknown_entry_cli_payload["ok"] is False, unknown_entry_cli_payload
    assert unknown_entry_cli_payload["signed_bundle_valid"] is False, unknown_entry_cli_payload
    assert unknown_entry_cli_payload["summary"]["v2_signed_bundle_unknown_entry_count"] == 1, unknown_entry_cli_payload
    assert premature_direction_cli_exit == 1, premature_direction_cli_payload
    assert premature_direction_cli_payload["ok"] is False, premature_direction_cli_payload
    assert premature_direction_cli_payload["signed_bundle_valid"] is False, premature_direction_cli_payload
    assert (
        premature_direction_cli_payload["summary"]["v2_signed_bundle_not_ready_entry_count"] == 1
    ), premature_direction_cli_payload

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_signed_bundle_intake_record_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "synthetic_signed_bundle_validated_in_memory": True,
        "stale_signed_bundle_rejected": True,
        "unsigned_template_signed_bundle_rejected": True,
        "template_digest_mismatch_rejected": True,
        "unknown_signed_entry_rejected": True,
        "not_ready_signed_entry_rejected": True,
        "cli_signed_bundle_validated_without_writing_artifact": True,
        "approval_artifact_written": False,
        "summary": pending_summary,
        "recommended_next_step": pending["recommended_next_step"],
    }


def _digest_payload(value: dict[str, Any]) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _signed_entry(bundle: dict[str, Any], signature_id: str) -> dict[str, Any]:
    entries = {entry["signature_id"]: entry for entry in bundle["signature_entries"]}
    entry = entries[signature_id]
    signed = dict(entry["template"])
    signed["source_v2_signature_template_digest"] = entry["source_template_digest"]
    signed["reviewer"] = f"synthetic_{signature_id}_smoke"
    signed["reviewed_at"] = "2026-06-09"
    if signature_id == "owner_release_review":
        signed["approve_native_update_release_review_package"] = True
    elif signature_id == "product_exposure_review":
        signed["approve_native_update_product_exposure_decision"] = True
    elif signature_id == "owner_release_direction":
        signed["approve_native_update_owner_release_direction"] = True
        signed["source_owner_release_direction_template_digest"] = entry["source_template_digest"]
    for field in entry["required_acknowledgement_fields"]:
        signed[field] = True
    return signed


def _read_artifact(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "temp" / "turbocore_optimizer" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_intake_cli(args: list[str]) -> dict[str, Any]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = int(intake_cli_main(args))
    assert exit_code == 0, stdout.getvalue()
    return json.loads(stdout.getvalue())


def _run_intake_cli_allow_failure(args: list[str]) -> tuple[dict[str, Any], int]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = int(intake_cli_main(args))
    return json.loads(stdout.getvalue()), exit_code


def _owner_release_review_record_ready() -> dict[str, Any]:
    return {
        "owner_packet_ready": True,
        "signed_review_present": True,
        "signed_review_valid": True,
        "approval_recorded": True,
        "release_review_recorded": True,
        "signed_review_digest_match": True,
        "release_package_decision": "native_update_release_review_recorded_default_off",
        "blocked_reasons": [],
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
    }


def _recorded_release_review_package() -> dict[str, Any]:
    package = _read_artifact("native_update_release_review_package.json")
    package["release_review_recorded"] = True
    package["decision"] = "native_update_release_review_recorded_default_off"
    package["blocked_reasons"] = []
    package["promotion_blockers"] = []
    package["runtime_dispatch_allowed"] = False
    package["native_dispatch_allowed"] = False
    package["training_path_enabled"] = False
    package["training_launch_executed"] = False
    return package


def _product_exposure_decision_ready() -> dict[str, Any]:
    decision = _read_artifact("native_update_product_exposure_decision.json")
    decision["product_exposure_decision_recorded"] = True
    decision["decision"] = "native_update_product_exposure_decision_recorded_default_off"
    decision["blocked_reasons"] = []
    return decision


def _stable_first_release_scope() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "turbocore_optimizer_stable_first_release_scope_v0",
        "gate": "optimizer_stable_first_release_default_off_scope",
        "ok": True,
        "stable_first_release_blocked_by_turbocore_optimizer": False,
        "turbocore_optimizer_default_off_release_scope_ready": True,
        "release_claim_allowed": True,
        "native_training_claim_allowed": False,
        "product_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "blocked_reasons": [],
        "summary": {
            "stable_first_release_turbocore_optimizer_blocker_count": 0,
            "turbocore_optimizer_default_off_release_scope_ready_count": 1,
            "owner_release_approval_recorded_count": 0,
            "owner_release_direction_recorded_count": 0,
            "owner_release_direction_approval_recorded_count": 0,
            "product_exposure_decision_recorded_count": 0,
            "product_training_route_binding_ready_count": 0,
            "run_local_adapter_staged_count": 0,
            "runtime_config_patch_applied_count": 0,
            "training_path_enabled_count": 0,
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
