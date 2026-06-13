"""Smoke checks for the v2 reviewer handoff packet."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    build_optimizer_v2_reviewer_handoff_packet,
)
from core.turbocore_optimizer_v2_signed_bundle_intake_record import (  # noqa: E402
    main as intake_cli_main,
)
from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    build_optimizer_v2_signature_bundle_packet,
)


ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_v2_reviewer_handoff_packet.json"
SIGNATURE_BUNDLE = ARTIFACT_DIR / "turbocore_optimizer_v2_signature_bundle_packet.json"
SIGNED_TEMPLATE = ARTIFACT_DIR / "turbocore_optimizer_v2_signed_bundle_template.json"


def run_smoke() -> dict[str, Any]:
    signature_bundle = build_optimizer_v2_signature_bundle_packet(write_artifact=True)
    handoff = build_optimizer_v2_reviewer_handoff_packet(
        signature_bundle=signature_bundle,
        write_artifact=True,
        write_signed_bundle_template=True,
    )
    summary = handoff["summary"]
    signed_template = handoff["signed_bundle_template"]
    signed_entries = signed_template["signed_entries"]
    required_fields = signed_template["required_manual_fields_by_signature_id"]
    phase1_manifest = handoff["phase1_handoff_manifest"]
    post_return_commands = handoff["phase1_post_return_operator_commands"]

    assert handoff["ok"] is True, handoff
    assert handoff["reviewer_handoff_ready"] is True, handoff
    assert handoff["approval_recorded"] is False, handoff
    assert handoff["approval_artifact_written"] is False, handoff
    assert handoff["runtime_dispatch_ready"] is False, handoff
    assert handoff["native_dispatch_allowed"] is False, handoff
    assert handoff["training_path_enabled"] is False, handoff
    assert handoff["product_native_ready"] is False, handoff
    assert {entry["signature_id"] for entry in handoff["ready_signature_entries"]} == {
        "owner_release_review",
        "product_exposure_review",
    }, handoff
    assert {entry["phase"] for entry in handoff["ready_signature_entries"]} == {"phase1"}, handoff
    assert [entry["signature_id"] for entry in handoff["blocked_signature_entries"]] == [
        "owner_release_direction"
    ], handoff
    assert [entry["phase"] for entry in handoff["blocked_signature_entries"]] == ["phase2"], handoff
    assert set(signed_entries) == {"owner_release_review", "product_exposure_review"}, handoff
    assert {
        signature_id: entry["phase"]
        for signature_id, entry in signed_entries.items()
    } == {
        "owner_release_review": "phase1",
        "product_exposure_review": "phase1",
    }, handoff
    assert {
        signature_id: entry["signature_id"]
        for signature_id, entry in signed_entries.items()
    } == {
        "owner_release_review": "owner_release_review",
        "product_exposure_review": "product_exposure_review",
    }, handoff
    assert phase1_manifest["phase"] == "phase1", handoff
    assert phase1_manifest["required_signature_ids"] == [
        "owner_release_review",
        "product_exposure_review",
    ], handoff
    assert phase1_manifest["deferred_signature_ids"] == ["owner_release_direction"], handoff
    assert phase1_manifest["phase1_required_signature_ids"] == [
        "owner_release_review",
        "product_exposure_review",
    ], handoff
    assert phase1_manifest["phase2_deferred_signature_ids"] == ["owner_release_direction"], handoff
    assert phase1_manifest["reviewer_returned_signed_bundle_artifact"].endswith(
        "turbocore_optimizer_v2_signed_bundle.reviewed.json"
    ), handoff
    assert [command["id"] for command in post_return_commands] == [
        "validate_phase1_signed_bundle",
        "extract_phase1_signed_reviews",
        "preflight_phase1_record_inputs",
    ], handoff
    assert all(command["writes_approval_record"] is False for command in post_return_commands), handoff
    assert all(command["approval_artifact_written"] is False for command in post_return_commands), handoff
    assert "turbocore_optimizer_v2_signed_bundle.reviewed.json" in post_return_commands[0]["command"], handoff
    assert "--write-extracted-artifacts" in post_return_commands[1]["command"], handoff
    assert "signed_owner_release_review.json" in post_return_commands[2]["command"], handoff
    assert "signed_product_exposure_review.json" in post_return_commands[2]["command"], handoff
    assert signed_entries["owner_release_review"]["reviewer"] == "", handoff
    assert signed_entries["product_exposure_review"]["reviewer"] == "", handoff
    assert required_fields["owner_release_review"] == [
        "reviewer",
        "reviewed_at",
        "approve_native_update_release_review_package",
        "acknowledge_all_expected_gates_present",
        "acknowledge_all_gates_default_off",
        "acknowledge_no_request_ui_schema_exposure",
        "acknowledge_no_training_launch_or_native_execution",
        "acknowledge_product_exposure_requires_separate_owner_direction",
    ], handoff
    assert required_fields["product_exposure_review"] == [
        "reviewer",
        "reviewed_at",
        "approve_native_update_product_exposure_decision",
        "acknowledge_no_backend_router_registration",
        "acknowledge_no_launcher_or_webui_exposure",
        "acknowledge_no_request_adapter_or_schema_change",
        "acknowledge_no_training_launch_or_request_submission",
        "acknowledge_product_exposure_default_off",
        "acknowledge_release_requires_separate_owner_decision",
        "acknowledge_training_launch_contract_ready",
    ], handoff
    assert phase1_manifest["required_manual_fields_by_signature_id"] == required_fields, handoff
    assert handoff["ready_signature_entries"][0]["required_manual_fields"] == required_fields["owner_release_review"], handoff
    assert signed_template["unsigned_template"] is True, handoff
    assert summary["v2_reviewer_handoff_entry_count"] == 3, handoff
    assert summary["v2_reviewer_handoff_ready_entry_count"] == 2, handoff
    assert summary["v2_reviewer_handoff_blocked_entry_count"] == 1, handoff
    assert summary["v2_reviewer_handoff_signed_template_entry_count"] == 2, handoff
    assert summary["v2_reviewer_handoff_phase_metadata_ready_count"] == 1, handoff
    assert summary["v2_reviewer_handoff_phase1_template_entry_count"] == 2, handoff
    assert summary["v2_reviewer_handoff_phase2_deferred_signature_count"] == 1, handoff
    assert summary["v2_reviewer_handoff_required_manual_field_signature_count"] == 2, handoff
    assert summary["v2_reviewer_handoff_required_manual_field_count"] == 18, handoff
    assert summary["v2_reviewer_handoff_phase1_signature_count"] == 2, handoff
    assert summary["v2_reviewer_handoff_phase2_blocked_signature_count"] == 1, handoff
    assert summary["v2_reviewer_handoff_phase1_required_manual_field_count"] == 18, handoff
    assert summary["v2_reviewer_handoff_template_dry_run_command_count"] == 1, handoff
    assert summary["v2_reviewer_handoff_real_return_validation_command_count"] == 1, handoff
    assert summary["v2_reviewer_handoff_phase1_post_return_command_count"] == 3, handoff
    assert summary["v2_reviewer_handoff_phase1_post_return_pre_record_command_count"] == 3, handoff
    assert summary["v2_reviewer_handoff_phase1_post_return_approval_record_command_count"] == 0, handoff
    assert summary["v2_reviewer_handoff_packet_ready_count"] == 1, handoff
    assert summary["v2_reviewer_handoff_approval_recorded_count"] == 0, handoff
    assert summary["v2_reviewer_handoff_runtime_dispatch_ready_count"] == 0, handoff
    assert summary["v2_reviewer_handoff_native_dispatch_allowed_count"] == 0, handoff
    assert summary["v2_reviewer_handoff_training_path_enabled_count"] == 0, handoff
    assert summary["v2_reviewer_handoff_product_native_ready_count"] == 0, handoff
    assert summary["v2_reviewer_handoff_default_behavior_changed_count"] == 0, handoff
    assert summary["v2_reviewer_handoff_unsafe_claim_count"] == 0, handoff
    assert "--signature-bundle" in handoff["validation_commands"][0], handoff
    assert "--allow-unsigned-template" in handoff["validation_commands"][0], handoff
    assert "--allow-unsigned-template" not in handoff["real_return_validation_commands"][0], handoff
    assert "turbocore_optimizer_v2_signed_bundle.reviewed.json" in handoff["real_return_validation_commands"][0], handoff
    assert phase1_manifest["real_return_validation_command"] == handoff["real_return_validation_commands"][0], handoff
    assert post_return_commands[0]["command"] == handoff["real_return_validation_commands"][0], handoff
    assert ARTIFACT.exists(), ARTIFACT
    assert SIGNATURE_BUNDLE.exists(), SIGNATURE_BUNDLE
    assert SIGNED_TEMPLATE.exists(), SIGNED_TEMPLATE

    dry_run = _run_intake_cli(
        [
            "--signature-bundle",
            str(SIGNATURE_BUNDLE),
            "--signed-bundle",
            str(SIGNED_TEMPLATE),
            "--allow-unsigned-template",
            "--no-artifact",
        ]
    )
    assert dry_run["ok"] is True, dry_run
    assert dry_run["signed_bundle_present"] is True, dry_run
    assert dry_run["signed_bundle_unsigned_template_marker"] is True, dry_run
    assert dry_run["signed_bundle_valid"] is False, dry_run
    assert dry_run["approval_recorded"] is False, dry_run
    assert dry_run["approval_artifact_written"] is False, dry_run
    assert dry_run["summary"]["v2_signed_bundle_unsigned_template_count"] == 1, dry_run
    assert dry_run["summary"]["v2_signed_bundle_approval_artifact_written_count"] == 0, dry_run

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_reviewer_handoff_packet_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "unsigned_template_intake_dry_run_checked": True,
        "summary": summary,
        "recommended_next_step": handoff["recommended_next_step"],
    }


def _run_intake_cli(args: list[str]) -> dict[str, Any]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = int(intake_cli_main(args))
    assert exit_code == 0, stdout.getvalue()
    return json.loads(stdout.getvalue())


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
