"""Smoke for validating signed TurboCore native-update owner release reviews."""

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

from core.turbocore_native_update_owner_release_handoff_summary import (  # noqa: E402
    build_native_update_owner_release_handoff_summary,
)
from core.turbocore_native_update_owner_release_review_packet import (  # noqa: E402
    build_native_update_owner_release_review_packet,
)
from core.turbocore_native_update_owner_release_review_record import (  # noqa: E402
    build_native_update_owner_release_review_record,
    main as record_cli_main,
)
from core.turbocore_native_update_release_review_package import (  # noqa: E402
    build_native_update_release_review_package,
    load_gate_artifacts,
)
from core.turbocore_native_update_representative_performance_importer import (  # noqa: E402
    build_native_update_representative_performance_import,
)


def run_smoke() -> dict[str, Any]:
    artifact_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    build_native_update_representative_performance_import(write_artifacts=True)
    package = build_native_update_release_review_package(gate_artifacts=load_gate_artifacts(artifact_dir))
    handoff = build_native_update_owner_release_handoff_summary(
        release_package=package,
        artifact_dir=artifact_dir,
        write_artifact=True,
    )
    packet = build_native_update_owner_release_review_packet(
        release_package=package,
        handoff_summary=handoff,
        artifact_dir=artifact_dir,
        write_artifact=True,
    )

    missing = build_native_update_owner_release_review_record(
        owner_packet=packet,
        artifact_dir=artifact_dir,
        write_artifact=True,
    )
    assert missing["ok"] is True, missing
    assert missing["owner_packet_ready"] is True, missing
    assert missing["signed_review_present"] is False, missing
    assert missing["approval_recorded"] is False, missing
    assert missing["release_review_recorded"] is False, missing
    assert "signed_owner_release_review_missing" in missing["blocked_reasons"], missing
    _assert_default_off(missing)

    signed = _signed_review(packet)
    tampered = dict(signed)
    tampered["source_release_review_template_digest"] = "stale"
    tampered_record = build_native_update_owner_release_review_record(
        signed_review=tampered,
        owner_packet=packet,
        approval_preflight=_phase1_preflight_ready(tampered),
        artifact_dir=artifact_dir,
        write_artifact=False,
    )
    assert tampered_record["ok"] is False, tampered_record
    assert tampered_record["signed_review_present"] is True, tampered_record
    assert tampered_record["approval_recorded"] is False, tampered_record
    assert "signed_owner_release_review_template_digest_mismatch" in tampered_record["blocked_reasons"], tampered_record
    _assert_default_off(tampered_record)

    missing_preflight_record = build_native_update_owner_release_review_record(
        signed_review=signed,
        owner_packet=packet,
        artifact_dir=artifact_dir,
        write_artifact=False,
    )
    assert missing_preflight_record["ok"] is False, missing_preflight_record
    assert missing_preflight_record["approval_recorded"] is False, missing_preflight_record
    assert "approval_execution_preflight_missing" in missing_preflight_record["blocked_reasons"], (
        missing_preflight_record
    )

    swapped = dict(signed)
    swapped["reviewer"] = "swapped_owner_record_smoke"
    swapped_preflight_record = build_native_update_owner_release_review_record(
        signed_review=swapped,
        owner_packet=packet,
        approval_preflight=_phase1_preflight_ready(signed),
        artifact_dir=artifact_dir,
        write_artifact=False,
    )
    assert swapped_preflight_record["ok"] is False, swapped_preflight_record
    assert swapped_preflight_record["approval_recorded"] is False, swapped_preflight_record
    assert (
        "approval_execution_preflight_owner_release_review_signed_payload_digest_mismatch"
        in swapped_preflight_record["blocked_reasons"]
    ), swapped_preflight_record

    missing_signed_check_record = build_native_update_owner_release_review_record(
        signed_review=signed,
        owner_packet=packet,
        approval_preflight=_phase1_preflight_ready(),
        artifact_dir=artifact_dir,
        write_artifact=False,
    )
    assert missing_signed_check_record["ok"] is False, missing_signed_check_record
    assert missing_signed_check_record["approval_recorded"] is False, missing_signed_check_record
    assert (
        "approval_execution_preflight_owner_release_review_signed_check_missing"
        in missing_signed_check_record["blocked_reasons"]
    ), missing_signed_check_record

    recorded = build_native_update_owner_release_review_record(
        signed_review=signed,
        owner_packet=packet,
        approval_preflight=_phase1_preflight_ready(signed),
        artifact_dir=artifact_dir,
        write_artifact=False,
    )
    assert recorded["ok"] is True, recorded
    assert recorded["signed_review_present"] is True, recorded
    assert recorded["signed_review_valid"] is True, recorded
    assert recorded["signed_review_digest_match"] is True, recorded
    assert recorded["approval_recorded"] is True, recorded
    assert recorded["release_review_recorded"] is True, recorded
    assert recorded["approval_preflight_binding_ready"] is True, recorded
    assert recorded["approval_preflight_digest"], recorded
    assert recorded["record_signed_payload_digest"], recorded
    assert recorded["approval_preflight_signed_payload_digest"] == recorded["record_signed_payload_digest"], recorded
    assert recorded["approval_preflight_signed_bundle_entry_digest"] == recorded["record_signed_payload_digest"], recorded
    assert recorded["blocked_reasons"] == [], recorded
    assert recorded["summary"]["release_review_recorded_count"] == 1, recorded
    assert recorded["summary"]["approval_preflight_binding_ready_count"] == 1, recorded
    _assert_default_off(recorded)

    assert _run_record_cli(["--no-artifact"]) == 0
    with tempfile.TemporaryDirectory() as temp_dir:
        signed_path = Path(temp_dir) / "signed_owner_release_review.json"
        preflight_path = Path(temp_dir) / "approval_preflight.json"
        signed_path.write_text(json.dumps(signed, ensure_ascii=False, indent=2), encoding="utf-8")
        preflight_path.write_text(
            json.dumps(_phase1_preflight_ready(signed), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        assert _run_record_cli(
            [
                "--signed-review",
                str(signed_path),
                "--approval-preflight",
                str(preflight_path),
                "--no-artifact",
            ]
        ) == 0

    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_owner_release_review_record_smoke",
        "ok": True,
        "roadmap": recorded["roadmap"],
        "missing_signed_review_blocked": True,
        "tampered_digest_blocked": True,
        "missing_preflight_blocked": True,
        "preflight_signed_payload_swap_blocked": True,
        "preflight_signed_check_missing_blocked": True,
        "synthetic_signed_review_validated_in_memory": True,
        "cli_signed_review_validated_without_writing_artifact": True,
        "approval_recorded_artifact_written": False,
        "release_review_recorded": False,
    }


def _signed_review(packet: dict[str, Any]) -> dict[str, Any]:
    signed = dict(packet["signable_review_record_template"])
    signed.update(
        {
            "reviewer": "synthetic_owner_record_smoke",
            "reviewed_at": "2026-06-07",
            "approve_native_update_release_review_package": True,
        }
    )
    for field in packet["required_acknowledgement_fields"]:
        signed[field] = True
    return signed


def _phase1_preflight_ready(signed_review: dict[str, Any] | None = None) -> dict[str, Any]:
    preflight = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_approval_execution_preflight_v0",
        "gate": "optimizer_v2_approval_execution_preflight",
        "ok": True,
        "phase1_record_inputs_ready": True,
        "phase2_direction_inputs_ready": False,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "summary": {
            "v2_approval_preflight_phase1_ready_count": 1,
            "v2_approval_preflight_phase2_ready_count": 0,
        },
    }
    if signed_review is not None:
        digest = _digest_payload(signed_review)
        preflight["signed_checks"] = [
            {
                "schema_version": 1,
                "id": "owner_release_review",
                "valid": True,
                "signed_payload_digest": digest,
                "signed_bundle_entry_digest": digest,
                "extracted_entry_digest_match": True,
                "extracted_entry_digest_mismatch": False,
                "extracted_entry_source_missing": False,
            }
        ]
    return preflight


def _digest_payload(value: dict[str, Any]) -> str:
    payload = {str(key): item for key, item in value.items() if not str(key).startswith("_source_")}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in (
        "product_exposure_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "training_path_enabled",
        "training_launch_executed",
    ):
        assert report[field] is False, report


def _run_record_cli(args: list[str]) -> int:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        return int(record_cli_main(args))


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
