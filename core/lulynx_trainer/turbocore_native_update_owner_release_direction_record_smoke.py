"""Smoke for validating signed TurboCore owner release directions."""

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
from core.turbocore_native_update_owner_release_direction_record import (  # noqa: E402
    build_native_update_owner_release_direction_record,
    main as record_cli_main,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def run_smoke() -> dict[str, Any]:
    artifact_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    real_packet = build_native_update_owner_release_direction_packet(
        artifact_dir=artifact_dir,
        write_artifact=True,
    )
    missing = build_native_update_owner_release_direction_record(
        owner_direction_packet=real_packet,
        artifact_dir=artifact_dir,
        write_artifact=True,
    )
    assert missing["ok"] is True, missing
    assert missing["roadmap"] == ROADMAP, missing
    assert missing["signed_direction_present"] is False, missing
    assert missing["owner_release_direction_recorded"] is False, missing
    assert missing["owner_release_approval_recorded"] is False, missing
    assert "signed_owner_release_direction_missing" in missing["blocked_reasons"], missing
    assert missing["summary"]["owner_release_direction_recorded_count"] == 0, missing
    assert missing["summary"]["owner_release_direction_approval_recorded_count"] == 0, missing
    _assert_default_off(missing)

    ready_packet = build_native_update_owner_release_direction_packet(
        release_review_archive=_archive_ready(),
        product_exposure_decision=_product_exposure_recorded(),
        stable_first_release_scope=_stable_scope_ready(),
        write_artifact=False,
    )
    signed = _signed_direction(ready_packet)
    recorded = build_native_update_owner_release_direction_record(
        signed_direction=signed,
        owner_direction_packet=ready_packet,
        approval_preflight=_phase2_preflight_ready(signed),
        write_artifact=False,
    )
    assert recorded["ok"] is True, recorded
    assert recorded["owner_direction_packet_ready"] is True, recorded
    assert recorded["signed_direction_present"] is True, recorded
    assert recorded["signed_direction_valid"] is True, recorded
    assert recorded["signed_owner_release_direction_digest_match"] is True, recorded
    assert recorded["owner_release_direction_recorded"] is True, recorded
    assert recorded["owner_release_approval_recorded"] is True, recorded
    assert recorded["approval_preflight_binding_ready"] is True, recorded
    assert recorded["approval_preflight_digest"], recorded
    assert recorded["record_signed_payload_digest"], recorded
    assert recorded["approval_preflight_signed_payload_digest"] == recorded["record_signed_payload_digest"], recorded
    assert recorded["approval_preflight_signed_bundle_entry_digest"] == recorded["record_signed_payload_digest"], recorded
    assert recorded["blocked_reasons"] == [], recorded
    assert recorded["summary"]["owner_release_direction_recorded_count"] == 1, recorded
    assert recorded["summary"]["owner_release_direction_approval_recorded_count"] == 1, recorded
    assert recorded["summary"]["approval_preflight_binding_ready_count"] == 1, recorded
    _assert_default_off(recorded)

    tampered = dict(signed)
    tampered["source_owner_release_direction_template_digest"] = "stale"
    tampered_record = build_native_update_owner_release_direction_record(
        signed_direction=tampered,
        owner_direction_packet=ready_packet,
        approval_preflight=_phase2_preflight_ready(tampered),
        write_artifact=False,
    )
    assert tampered_record["ok"] is False, tampered_record
    assert tampered_record["owner_release_direction_recorded"] is False, tampered_record
    assert "signed_owner_release_direction_template_digest_mismatch" in tampered_record["blocked_reasons"], (
        tampered_record
    )
    _assert_default_off(tampered_record)

    missing_preflight_record = build_native_update_owner_release_direction_record(
        signed_direction=signed,
        owner_direction_packet=ready_packet,
        write_artifact=False,
    )
    assert missing_preflight_record["ok"] is False, missing_preflight_record
    assert missing_preflight_record["owner_release_direction_recorded"] is False, missing_preflight_record
    assert "approval_execution_preflight_missing" in missing_preflight_record["blocked_reasons"], (
        missing_preflight_record
    )
    _assert_default_off(missing_preflight_record)

    swapped = dict(signed)
    swapped["reviewer"] = "swapped_owner_direction_record_smoke"
    swapped_preflight_record = build_native_update_owner_release_direction_record(
        signed_direction=swapped,
        owner_direction_packet=ready_packet,
        approval_preflight=_phase2_preflight_ready(signed),
        write_artifact=False,
    )
    assert swapped_preflight_record["ok"] is False, swapped_preflight_record
    assert swapped_preflight_record["owner_release_direction_recorded"] is False, swapped_preflight_record
    assert (
        "approval_execution_preflight_owner_release_direction_signed_payload_digest_mismatch"
        in swapped_preflight_record["blocked_reasons"]
    ), swapped_preflight_record
    _assert_default_off(swapped_preflight_record)

    missing_signed_check_record = build_native_update_owner_release_direction_record(
        signed_direction=signed,
        owner_direction_packet=ready_packet,
        approval_preflight=_phase2_preflight_ready(),
        write_artifact=False,
    )
    assert missing_signed_check_record["ok"] is False, missing_signed_check_record
    assert missing_signed_check_record["owner_release_direction_recorded"] is False, missing_signed_check_record
    assert (
        "approval_execution_preflight_owner_release_direction_signed_check_missing"
        in missing_signed_check_record["blocked_reasons"]
    ), missing_signed_check_record
    _assert_default_off(missing_signed_check_record)

    unsafe = dict(signed)
    unsafe["training_path_enabled"] = True
    unsafe_record = build_native_update_owner_release_direction_record(
        signed_direction=unsafe,
        owner_direction_packet=ready_packet,
        approval_preflight=_phase2_preflight_ready(unsafe),
        write_artifact=False,
    )
    assert unsafe_record["ok"] is False, unsafe_record
    assert "signed_owner_release_direction_unsafe:training_path_enabled" in unsafe_record["blocked_reasons"], (
        unsafe_record
    )

    assert _run_record_cli(["--no-artifact"]) == 0
    with tempfile.TemporaryDirectory() as temp_dir:
        packet_path = Path(temp_dir) / "owner_direction_packet.json"
        signed_path = Path(temp_dir) / "signed_owner_release_direction.json"
        preflight_path = Path(temp_dir) / "approval_preflight.json"
        packet_path.write_text(json.dumps(ready_packet, ensure_ascii=False, indent=2), encoding="utf-8")
        signed_path.write_text(json.dumps(signed, ensure_ascii=False, indent=2), encoding="utf-8")
        preflight_path.write_text(
            json.dumps(_phase2_preflight_ready(signed), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        assert _run_record_cli(
            [
                "--owner-direction-packet",
                str(packet_path),
                "--signed-direction",
                str(signed_path),
                "--approval-preflight",
                str(preflight_path),
                "--no-artifact",
            ]
        ) == 0

    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_owner_release_direction_record_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "missing_signed_direction_blocked": True,
        "tampered_digest_blocked": True,
        "missing_preflight_blocked": True,
        "preflight_signed_payload_swap_blocked": True,
        "preflight_signed_check_missing_blocked": True,
        "unsafe_signed_direction_blocked": True,
        "synthetic_signed_direction_validated_in_memory": True,
        "cli_signed_direction_validated_without_writing_artifact": True,
        "approval_recorded_artifact_written": False,
        "summary": missing["summary"],
        "recommended_next_step": missing["recommended_next_step"],
    }


def _signed_direction(packet: dict[str, Any]) -> dict[str, Any]:
    signed = dict(packet["signable_owner_release_direction_template"])
    signed.update(
        {
            "reviewer": "synthetic_owner_release_direction_record_smoke",
            "reviewed_at": "2026-06-07",
            "approve_native_update_owner_release_direction": True,
            "source_owner_release_direction_template_digest": packet[
                "source_owner_release_direction_template_digest"
            ],
        }
    )
    for field in packet["required_acknowledgement_fields"]:
        signed[field] = True
    return signed


def _phase2_preflight_ready(signed_direction: dict[str, Any] | None = None) -> dict[str, Any]:
    preflight = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_approval_execution_preflight_v0",
        "gate": "optimizer_v2_approval_execution_preflight",
        "ok": True,
        "phase1_record_inputs_ready": True,
        "phase2_direction_inputs_ready": True,
        "approval_recorded": False,
        "approval_artifact_written": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "summary": {
            "v2_approval_preflight_phase1_ready_count": 1,
            "v2_approval_preflight_phase2_ready_count": 1,
        },
    }
    if signed_direction is not None:
        digest = _digest_payload(signed_direction)
        preflight["signed_checks"] = [
            {
                "schema_version": 1,
                "id": "owner_release_direction",
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


def _archive_ready() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_release_review_archive_v0",
        "gate": "native_update_release_review_archive",
        "ok": True,
        "archive_ready": True,
        "archive_recorded": True,
        "ready_for_owner_release_direction": True,
        "default_off": True,
        "blocked_reasons": [],
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "post_release_request_fields": {},
    }


def _product_exposure_recorded() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_product_exposure_decision_v0",
        "gate": "native_update_product_exposure_decision",
        "ok": True,
        "product_exposure_decision_recorded": True,
        "decision": "native_update_product_exposure_decision_recorded_default_off",
        "blocked_reasons": [],
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "post_product_exposure_request_fields": {},
    }


def _stable_scope_ready() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "turbocore_optimizer_stable_first_release_scope_v0",
        "gate": "optimizer_stable_first_release_default_off_scope",
        "ok": True,
        "turbocore_optimizer_default_off_release_scope_ready": True,
        "product_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "blocked_reasons": [],
        "summary": {
            "stable_first_release_turbocore_optimizer_blocker_count": 0,
            "turbocore_optimizer_default_off_release_scope_ready_count": 1,
        },
    }


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in (
        "product_exposure_allowed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "backend_router_registered",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "training_path_enabled",
        "training_launch_executed",
    ):
        assert report[field] is False, (field, report)
    assert report["post_owner_release_request_fields"] == {}, report


def _run_record_cli(args: list[str]) -> int:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        return int(record_cli_main(args))


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
