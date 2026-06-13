"""Smoke for TurboCore owner release-direction packet."""

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

from core.turbocore_native_update_owner_release_direction_packet import (  # noqa: E402
    RECORDED_DECISION,
    WAITING_DECISION,
    build_native_update_owner_release_direction_packet,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def run_smoke() -> dict[str, Any]:
    artifact_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    real = build_native_update_owner_release_direction_packet(
        artifact_dir=artifact_dir,
        write_artifact=True,
    )
    assert real["roadmap"] == ROADMAP, real
    assert real["ok"] is True, real
    assert real["decision"] == WAITING_DECISION, real
    assert real["owner_release_direction_recorded"] is False, real
    assert real["owner_release_approval_recorded"] is False, real
    assert real["summary"]["owner_release_direction_recorded_count"] == 0, real
    assert real["summary"]["owner_release_direction_approval_recorded_count"] == 0, real
    assert real["summary"]["owner_release_approval_recorded_count"] == 0, real
    assert real["summary"]["runtime_dispatch_ready_count"] == 0, real
    assert real["summary"]["native_dispatch_allowed_count"] == 0, real
    assert real["summary"]["training_path_enabled_count"] == 0, real
    _assert_default_off(real)

    ready_packet = build_native_update_owner_release_direction_packet(
        release_review_archive=_archive_ready(),
        product_exposure_decision=_product_exposure_recorded(),
        stable_first_release_scope=_stable_scope_ready(),
        write_artifact=False,
    )
    assert ready_packet["ok"] is True, ready_packet
    assert ready_packet["ready_for_owner_direction_signature"] is True, ready_packet
    assert ready_packet["owner_release_direction_recorded"] is False, ready_packet
    assert ready_packet["summary"]["owner_release_direction_ready_for_signature_count"] == 1, ready_packet
    assert ready_packet["summary"]["owner_release_direction_approval_recorded_count"] == 0, ready_packet
    assert "owner_release_direction_not_recorded" in ready_packet["promotion_blockers"], ready_packet
    _assert_default_off(ready_packet)

    signed = _signed_direction(ready_packet)
    recorded = build_native_update_owner_release_direction_packet(
        release_review_archive=_archive_ready(),
        product_exposure_decision=_product_exposure_recorded(),
        stable_first_release_scope=_stable_scope_ready(),
        signed_direction=signed,
        write_artifact=False,
    )
    assert recorded["ok"] is True, recorded
    assert recorded["decision"] == RECORDED_DECISION, recorded
    assert recorded["owner_release_direction_recorded"] is True, recorded
    assert recorded["owner_release_approval_recorded"] is True, recorded
    assert recorded["summary"]["owner_release_direction_recorded_count"] == 1, recorded
    assert recorded["summary"]["owner_release_direction_approval_recorded_count"] == 1, recorded
    assert recorded["summary"]["owner_release_approval_recorded_count"] == 1, recorded
    assert recorded["blocked_reasons"] == [], recorded
    _assert_default_off(recorded)

    tampered = dict(signed)
    tampered["source_owner_release_direction_template_digest"] = "stale"
    tampered_record = build_native_update_owner_release_direction_packet(
        release_review_archive=_archive_ready(),
        product_exposure_decision=_product_exposure_recorded(),
        stable_first_release_scope=_stable_scope_ready(),
        signed_direction=tampered,
        write_artifact=False,
    )
    assert tampered_record["owner_release_direction_recorded"] is False, tampered_record
    assert "owner_release_direction_template_digest_mismatch" in tampered_record["blocked_reasons"], (
        tampered_record
    )
    _assert_default_off(tampered_record)

    unsafe_stable = _stable_scope_ready()
    unsafe_stable["training_path_enabled"] = True
    unsafe_record = build_native_update_owner_release_direction_packet(
        release_review_archive=_archive_ready(),
        product_exposure_decision=_product_exposure_recorded(),
        stable_first_release_scope=unsafe_stable,
        write_artifact=False,
    )
    assert unsafe_record["ok"] is False, unsafe_record
    assert "stable_first_release_scope_unsafe:training_path_enabled" in unsafe_record["blocked_reasons"], (
        unsafe_record
    )

    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_owner_release_direction_packet_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "real_artifact_checked": True,
        "synthetic_signed_direction_validated_in_memory": True,
        "approval_recorded_artifact_written": False,
        "summary": real["summary"],
        "recommended_next_step": real["recommended_next_step"],
    }


def _signed_direction(packet: dict[str, Any]) -> dict[str, Any]:
    signed = dict(packet["signable_owner_release_direction_template"])
    signed.update(
        {
            "reviewer": "synthetic_owner_release_direction_smoke",
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


def _archive_ready() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_release_review_archive_v0",
        "gate": "native_update_release_review_archive",
        "ok": True,
        "evidence_ready": True,
        "ready_for_review": True,
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
        "evidence_ready": True,
        "ready_for_product_exposure_review": True,
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
        "stable_first_release_scope": "stable_baseline_with_turbocore_optimizer_default_off",
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


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
