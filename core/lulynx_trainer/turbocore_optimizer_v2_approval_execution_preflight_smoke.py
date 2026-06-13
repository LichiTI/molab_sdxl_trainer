"""Smoke checks for v2 approval execution preflight."""

from __future__ import annotations

import contextlib
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

from core.turbocore_optimizer_v2_approval_execution_preflight import (  # noqa: E402
    ARTIFACT,
    build_optimizer_v2_approval_execution_preflight,
    main as preflight_cli_main,
)
from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    build_optimizer_v2_reviewer_handoff_packet,
)
from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    build_optimizer_v2_signature_bundle_packet,
)
from core.turbocore_optimizer_v2_signed_bundle_extractor import (  # noqa: E402
    build_optimizer_v2_signed_bundle_extraction_record,
)


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        pending_dir = Path(temp_dir) / "pending"
        pending = build_optimizer_v2_approval_execution_preflight(
            paths=_paths(pending_dir),
            write_artifact=True,
        )
    pending_summary = pending["summary"]
    assert pending["ok"] is True, pending
    assert pending["phase1_record_inputs_ready"] is False, pending
    assert pending["phase2_direction_inputs_ready"] is False, pending
    assert pending["full_record_execution_ready"] is False, pending
    assert pending["approval_recorded"] is False, pending
    assert pending["approval_artifact_written"] is False, pending
    assert pending["runtime_dispatch_ready"] is False, pending
    assert pending["native_dispatch_allowed"] is False, pending
    assert pending["training_path_enabled"] is False, pending
    assert pending["product_native_ready"] is False, pending
    assert pending["missing_input_ids_by_phase"] == {
        "shared": ["signature_bundle", "signed_bundle"],
        "phase1": [
            "owner_release_review",
            "product_exposure_review",
            "training_launch_contract",
            "product_exposure_evidence",
        ],
        "phase2": ["owner_release_direction", "owner_release_direction_packet"],
    }, pending
    assert pending_summary["v2_approval_preflight_file_count"] == 8, pending
    assert pending_summary["v2_approval_preflight_present_file_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_missing_file_count"] == 8, pending
    assert pending_summary["v2_approval_preflight_missing_shared_input_count"] == 2, pending
    assert pending_summary["v2_approval_preflight_missing_phase1_input_count"] == 4, pending
    assert pending_summary["v2_approval_preflight_missing_phase2_input_count"] == 2, pending
    assert pending_summary["v2_approval_preflight_signed_payload_digest_present_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_signed_payload_digest_missing_count"] == 3, pending
    assert pending_summary["v2_approval_preflight_signed_bundle_entry_digest_present_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_signed_payload_bundle_digest_match_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_phase1_ready_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_phase2_ready_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_full_ready_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_support_source_binding_ready_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_support_source_binding_blocker_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_runtime_dispatch_ready_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_native_dispatch_allowed_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_training_path_enabled_count"] == 0, pending
    assert pending_summary["v2_approval_preflight_product_native_ready_count"] == 0, pending
    assert ARTIFACT.exists(), ARTIFACT

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        signature_bundle = build_optimizer_v2_signature_bundle_packet(write_artifact=False)
        handoff = build_optimizer_v2_reviewer_handoff_packet(
            signature_bundle=signature_bundle,
            write_artifact=False,
            write_signed_bundle_template=False,
        )
        signed_bundle = {
            "schema_version": 1,
            "package": "turbocore_optimizer_v2_signed_bundle_preflight_smoke_v0",
            "source_signature_bundle_digest": str(handoff.get("source_signature_bundle_digest") or ""),
            "signed_entries": {
                signature_id: _sign_template(signature_id, template)
                for signature_id, template in handoff["signed_bundle_template"]["signed_entries"].items()
            },
        }
        paths = _paths(output_dir)
        _write_json(paths["signature_bundle"], signature_bundle)
        _write_json(paths["signed_bundle"], signed_bundle)
        extraction = build_optimizer_v2_signed_bundle_extraction_record(
            signature_bundle=signature_bundle,
            signed_bundle=signed_bundle,
            output_dir=output_dir,
            write_artifact=False,
            write_extracted_artifacts=True,
        )
        _write_json(paths["training_launch_contract"], _support_artifact("training_launch_contract"))
        _write_json(paths["product_exposure_evidence"], _support_artifact("product_exposure_evidence"))
        ready = build_optimizer_v2_approval_execution_preflight(paths=paths, write_artifact=False)
        cli_payload = _run_preflight_cli(paths)
        stale_bundle = dict(signed_bundle)
        stale_bundle["source_signature_bundle_digest"] = "stale-digest-for-preflight-smoke"
        stale_paths = dict(paths)
        stale_paths["signed_bundle"] = output_dir / "turbocore_optimizer_v2_signed_bundle.stale.json"
        _write_json(stale_paths["signed_bundle"], stale_bundle)
        stale = build_optimizer_v2_approval_execution_preflight(paths=stale_paths, write_artifact=False)
        template_mismatch_bundle = json.loads(json.dumps(signed_bundle))
        template_mismatch_bundle["signed_entries"]["product_exposure_review"][
            "source_v2_signature_template_digest"
        ] = "mismatched-product-exposure-template-digest"
        template_mismatch_paths = dict(paths)
        template_mismatch_paths["signed_bundle"] = (
            output_dir / "turbocore_optimizer_v2_signed_bundle.template_mismatch.json"
        )
        _write_json(template_mismatch_paths["signed_bundle"], template_mismatch_bundle)
        template_mismatch = build_optimizer_v2_approval_execution_preflight(
            paths=template_mismatch_paths,
            write_artifact=False,
        )
        unknown_entry_bundle = json.loads(json.dumps(signed_bundle))
        unknown_entry_bundle["signed_entries"]["unexpected_review"] = dict(
            unknown_entry_bundle["signed_entries"]["owner_release_review"]
        )
        unknown_entry_paths = dict(paths)
        unknown_entry_paths["signed_bundle"] = output_dir / "turbocore_optimizer_v2_signed_bundle.unknown_entry.json"
        _write_json(unknown_entry_paths["signed_bundle"], unknown_entry_bundle)
        unknown_entry = build_optimizer_v2_approval_execution_preflight(
            paths=unknown_entry_paths,
            write_artifact=False,
        )
        unknown_entry_cli_payload, unknown_entry_cli_exit = _run_preflight_cli_allow_failure(unknown_entry_paths)
        premature_direction_bundle = json.loads(json.dumps(signed_bundle))
        signature_entries = {entry["signature_id"]: entry for entry in signature_bundle["signature_entries"]}
        premature_direction_signed = _sign_template(
            "owner_release_direction",
            signature_entries["owner_release_direction"]["template"],
        )
        premature_direction_bundle["signed_entries"]["owner_release_direction"] = premature_direction_signed
        premature_direction_paths = dict(paths)
        premature_direction_paths["signed_bundle"] = (
            output_dir / "turbocore_optimizer_v2_signed_bundle.not_ready_direction.json"
        )
        premature_direction_paths["owner_release_direction"] = (
            output_dir / "signed_owner_release_direction.not_ready.json"
        )
        _write_json(premature_direction_paths["signed_bundle"], premature_direction_bundle)
        _write_json(premature_direction_paths["owner_release_direction"], premature_direction_signed)
        premature_direction = build_optimizer_v2_approval_execution_preflight(
            paths=premature_direction_paths,
            write_artifact=False,
        )
        premature_direction_cli_payload, premature_direction_cli_exit = _run_preflight_cli_allow_failure(
            premature_direction_paths
        )
        tampered_extracted_paths = dict(paths)
        tampered_extracted_paths["owner_release_review"] = output_dir / "signed_owner_release_review.tampered.json"
        tampered_owner_review = json.loads(paths["owner_release_review"].read_text(encoding="utf-8"))
        tampered_owner_review["reviewer"] = "tampered_preflight_smoke_reviewer"
        _write_json(tampered_extracted_paths["owner_release_review"], tampered_owner_review)
        tampered_extracted = build_optimizer_v2_approval_execution_preflight(
            paths=tampered_extracted_paths,
            write_artifact=False,
        )
        tampered_extracted_cli_payload, tampered_extracted_cli_exit = _run_preflight_cli_allow_failure(
            tampered_extracted_paths
        )
        invalid_support_paths = dict(paths)
        invalid_support_paths["training_launch_contract"] = (
            output_dir / "native_update_training_launch_contract.invalid.json"
        )
        _write_json(invalid_support_paths["training_launch_contract"], _support_artifact("wrong_support_shape"))
        invalid_support = build_optimizer_v2_approval_execution_preflight(
            paths=invalid_support_paths,
            write_artifact=False,
        )
        invalid_support_cli_payload, invalid_support_cli_exit = _run_preflight_cli_allow_failure(
            invalid_support_paths
        )
        missing_source_support_paths = dict(paths)
        missing_source_support_paths["training_launch_contract"] = (
            output_dir / "native_update_training_launch_contract.missing_source.json"
        )
        _write_json(
            missing_source_support_paths["training_launch_contract"],
            _support_artifact("training_launch_contract_missing_source"),
        )
        missing_source_support = build_optimizer_v2_approval_execution_preflight(
            paths=missing_source_support_paths,
            write_artifact=False,
        )
    ready_summary = ready["summary"]
    ready_signed_checks = {item["id"]: item for item in ready["signed_checks"]}
    assert extraction["signed_bundle_source_digest_match"] is True, extraction
    assert extraction["summary"]["v2_signed_bundle_extraction_source_digest_stale_count"] == 0, extraction
    assert extraction["summary"]["v2_signed_bundle_extraction_artifact_written_count"] == 2, extraction
    assert ready["ok"] is True, ready
    assert ready["phase1_record_inputs_ready"] is True, ready
    assert ready["phase2_direction_inputs_ready"] is False, ready
    assert ready["full_record_execution_ready"] is False, ready
    assert ready["missing_input_ids_by_phase"] == {
        "shared": [],
        "phase1": [],
        "phase2": ["owner_release_direction", "owner_release_direction_packet"],
    }, ready
    assert ready["approval_recorded"] is False, ready
    assert ready["approval_artifact_written"] is False, ready
    assert ready_summary["v2_approval_preflight_file_count"] == 8, ready
    assert ready_summary["v2_approval_preflight_present_file_count"] == 6, ready
    assert ready_summary["v2_approval_preflight_valid_json_file_count"] == 6, ready
    assert ready_summary["v2_approval_preflight_missing_file_count"] == 2, ready
    assert ready_summary["v2_approval_preflight_missing_shared_input_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_missing_phase1_input_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_missing_phase2_input_count"] == 2, ready
    assert ready_summary["v2_approval_preflight_signature_bundle_valid_count"] == 1, ready
    assert ready_summary["v2_approval_preflight_signed_bundle_valid_count"] == 1, ready
    assert ready_summary["v2_approval_preflight_source_digest_match_count"] == 1, ready
    assert ready_summary["v2_approval_preflight_source_digest_stale_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_template_digest_mismatch_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_unknown_entry_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_unsigned_template_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_signed_payload_digest_present_count"] == 2, ready
    assert ready_summary["v2_approval_preflight_signed_payload_digest_missing_count"] == 1, ready
    assert ready_summary["v2_approval_preflight_signed_bundle_entry_digest_present_count"] == 2, ready
    assert ready_summary["v2_approval_preflight_signed_payload_bundle_digest_match_count"] == 2, ready
    assert ready_summary["v2_approval_preflight_extracted_entry_digest_match_count"] == 2, ready
    assert ready_summary["v2_approval_preflight_extracted_entry_digest_mismatch_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_extracted_entry_source_missing_count"] == 0, ready
    assert ready_signed_checks["owner_release_review"]["signed_payload_digest"], ready
    assert ready_signed_checks["owner_release_review"]["signed_bundle_entry_digest"], ready
    assert (
        ready_signed_checks["owner_release_review"]["signed_payload_digest"]
        == ready_signed_checks["owner_release_review"]["signed_bundle_entry_digest"]
    ), ready
    assert ready_signed_checks["product_exposure_review"]["signed_payload_digest"], ready
    assert ready_signed_checks["product_exposure_review"]["extracted_entry_digest_match"] is True, ready
    assert ready_summary["v2_approval_preflight_support_ready_count"] == 2, ready
    assert ready_summary["v2_approval_preflight_support_invalid_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_support_source_binding_ready_count"] == 2, ready
    assert ready_summary["v2_approval_preflight_support_source_binding_blocker_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_hard_fail_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_owner_review_ready_count"] == 1, ready
    assert ready_summary["v2_approval_preflight_product_exposure_ready_count"] == 1, ready
    assert ready_summary["v2_approval_preflight_owner_direction_ready_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_phase1_ready_count"] == 1, ready
    assert ready_summary["v2_approval_preflight_phase2_ready_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_full_ready_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_runtime_dispatch_ready_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_native_dispatch_allowed_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_training_path_enabled_count"] == 0, ready
    assert ready_summary["v2_approval_preflight_product_native_ready_count"] == 0, ready
    assert cli_payload["summary"]["v2_approval_preflight_phase1_ready_count"] == 1, cli_payload
    assert cli_payload["summary"]["v2_approval_preflight_approval_recorded_count"] == 0, cli_payload
    assert stale["ok"] is False, stale
    assert stale["phase1_record_inputs_ready"] is False, stale
    assert stale["summary"]["v2_approval_preflight_source_digest_stale_count"] == 1, stale
    assert stale["summary"]["v2_approval_preflight_hard_fail_count"] == 1, stale
    assert template_mismatch["ok"] is False, template_mismatch
    assert template_mismatch["phase1_record_inputs_ready"] is False, template_mismatch
    assert (
        template_mismatch["summary"]["v2_approval_preflight_template_digest_mismatch_count"] == 1
    ), template_mismatch
    assert template_mismatch["summary"]["v2_approval_preflight_hard_fail_count"] == 1, template_mismatch
    assert unknown_entry["ok"] is False, unknown_entry
    assert unknown_entry["phase1_record_inputs_ready"] is False, unknown_entry
    assert unknown_entry["summary"]["v2_approval_preflight_unknown_entry_count"] == 1, unknown_entry
    assert unknown_entry["summary"]["v2_approval_preflight_hard_fail_count"] == 1, unknown_entry
    assert unknown_entry_cli_exit == 1, unknown_entry_cli_payload
    assert unknown_entry_cli_payload["ok"] is False, unknown_entry_cli_payload
    assert unknown_entry_cli_payload["summary"]["v2_approval_preflight_unknown_entry_count"] == 1, unknown_entry_cli_payload
    assert premature_direction["ok"] is False, premature_direction
    assert premature_direction["phase1_record_inputs_ready"] is False, premature_direction
    assert premature_direction["phase2_direction_inputs_ready"] is False, premature_direction
    assert premature_direction["summary"]["v2_approval_preflight_not_ready_entry_count"] == 1, premature_direction
    assert premature_direction["summary"]["v2_approval_preflight_hard_fail_count"] == 1, premature_direction
    assert premature_direction_cli_exit == 1, premature_direction_cli_payload
    assert premature_direction_cli_payload["ok"] is False, premature_direction_cli_payload
    assert (
        premature_direction_cli_payload["summary"]["v2_approval_preflight_not_ready_entry_count"] == 1
    ), premature_direction_cli_payload
    assert tampered_extracted["ok"] is False, tampered_extracted
    assert tampered_extracted["phase1_record_inputs_ready"] is False, tampered_extracted
    assert (
        tampered_extracted["summary"]["v2_approval_preflight_extracted_entry_digest_mismatch_count"] == 1
    ), tampered_extracted
    assert tampered_extracted["summary"]["v2_approval_preflight_hard_fail_count"] == 1, tampered_extracted
    assert tampered_extracted_cli_exit == 1, tampered_extracted_cli_payload
    assert tampered_extracted_cli_payload["ok"] is False, tampered_extracted_cli_payload
    assert (
        tampered_extracted_cli_payload["summary"]["v2_approval_preflight_extracted_entry_digest_mismatch_count"] == 1
    ), tampered_extracted_cli_payload
    assert invalid_support["ok"] is False, invalid_support
    assert invalid_support["phase1_record_inputs_ready"] is False, invalid_support
    assert invalid_support["summary"]["v2_approval_preflight_support_invalid_count"] == 1, invalid_support
    assert (
        invalid_support["summary"]["v2_approval_preflight_support_source_binding_ready_count"] == 1
    ), invalid_support
    assert (
        invalid_support["summary"]["v2_approval_preflight_support_source_binding_blocker_count"] == 0
    ), invalid_support
    assert invalid_support["summary"]["v2_approval_preflight_hard_fail_count"] == 1, invalid_support
    assert invalid_support_cli_exit == 1, invalid_support_cli_payload
    assert invalid_support_cli_payload["ok"] is False, invalid_support_cli_payload
    assert invalid_support_cli_payload["summary"]["v2_approval_preflight_support_invalid_count"] == 1, invalid_support_cli_payload
    assert missing_source_support["ok"] is False, missing_source_support
    assert missing_source_support["phase1_record_inputs_ready"] is False, missing_source_support
    assert (
        missing_source_support["summary"]["v2_approval_preflight_support_source_binding_ready_count"] == 1
    ), missing_source_support
    assert (
        missing_source_support["summary"]["v2_approval_preflight_support_source_binding_blocker_count"] == 4
    ), missing_source_support
    assert missing_source_support["summary"]["v2_approval_preflight_support_invalid_count"] == 1, (
        missing_source_support
    )
    assert missing_source_support["summary"]["v2_approval_preflight_hard_fail_count"] == 1, (
        missing_source_support
    )

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_approval_execution_preflight_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "synthetic_phase1_preflight_checked": True,
        "stale_signed_bundle_rejected": True,
        "template_digest_mismatch_rejected": True,
        "unknown_signed_entry_rejected": True,
        "not_ready_signed_entry_rejected": True,
        "tampered_extracted_entry_rejected": True,
        "invalid_support_artifact_rejected": True,
        "missing_support_source_binding_rejected": True,
        "approval_artifact_written": False,
        "summary": pending_summary,
        "recommended_next_step": pending["recommended_next_step"],
    }


def _paths(root: Path) -> dict[str, Path]:
    return {
        "signature_bundle": root / "turbocore_optimizer_v2_signature_bundle_packet.json",
        "signed_bundle": root / "turbocore_optimizer_v2_signed_bundle.reviewed.json",
        "owner_release_review": root / "signed_owner_release_review.json",
        "product_exposure_review": root / "signed_product_exposure_review.json",
        "owner_release_direction": root / "signed_owner_release_direction.json",
        "training_launch_contract": root / "native_update_training_launch_contract.json",
        "product_exposure_evidence": root / "native_update_product_exposure_evidence.json",
        "owner_release_direction_packet": root / "native_update_owner_release_direction_packet.json",
    }


def _sign_template(signature_id: str, template: dict[str, Any]) -> dict[str, Any]:
    signed = dict(template)
    signed["reviewer"] = f"synthetic_{signature_id}_preflight_smoke"
    signed["reviewed_at"] = "2026-06-09"
    for key in list(signed):
        if key.startswith("approve_") or key.startswith("acknowledge_"):
            signed[key] = True
    return signed


def _support_artifact(artifact_id: str) -> dict[str, Any]:
    if artifact_id == "training_launch_contract":
        return {
            **_default_off_flags(),
            "schema_version": 1,
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
        }
    if artifact_id == "training_launch_contract_missing_source":
        missing_source = _support_artifact("training_launch_contract")
        missing_source.pop("training_step_execution_contract_summary", None)
        missing_source.pop("training_launch_evidence_summary", None)
        return missing_source
    if artifact_id == "product_exposure_evidence":
        return {
            **_default_off_flags(),
            "schema_version": 1,
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
        }
    return {
        **_default_off_flags(),
        "schema_version": 1,
        "artifact": artifact_id,
        "ok": True,
        "default_off": True,
    }


def _default_off_flags() -> dict[str, Any]:
    return {
        "approval_recorded": False,
        "approval_artifact_written": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_preflight_cli(paths: dict[str, Path]) -> dict[str, Any]:
    payload, exit_code = _run_preflight_cli_allow_failure(paths)
    assert exit_code == 0, payload
    return payload


def _run_preflight_cli_allow_failure(paths: dict[str, Path]) -> tuple[dict[str, Any], int]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = int(
            preflight_cli_main(
                [
                    "--signature-bundle",
                    str(paths["signature_bundle"]),
                    "--signed-bundle",
                    str(paths["signed_bundle"]),
                    "--owner-review",
                    str(paths["owner_release_review"]),
                    "--product-exposure-review",
                    str(paths["product_exposure_review"]),
                    "--owner-direction",
                    str(paths["owner_release_direction"]),
                    "--training-launch-contract",
                    str(paths["training_launch_contract"]),
                    "--product-exposure-evidence",
                    str(paths["product_exposure_evidence"]),
                    "--owner-direction-packet",
                    str(paths["owner_release_direction_packet"]),
                    "--no-artifact",
                ]
            )
        )
    return json.loads(stdout.getvalue()), exit_code


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
