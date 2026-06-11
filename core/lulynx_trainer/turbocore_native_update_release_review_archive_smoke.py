from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (str(SCRIPT_ROOT), str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_native_update_release_review_archive import (  # noqa: E402
    ARCHIVE_BLOCKED_DECISION,
    ARCHIVE_HOLD_DECISION,
    ARCHIVE_READY_DECISION,
    build_native_update_release_review_archive,
)
from core.turbocore_native_update_release_review_package import (  # noqa: E402
    READY_DECISION,
    build_native_update_release_review_package,
)
from turbocore_native_update_release_review_package_smoke import (  # noqa: E402
    _artifact_map,
    _assert_optimizer_family_counts,
    _release_review,
    _write_real_artifact_case,
)


def run_smoke() -> dict[str, Any]:
    artifacts = _artifact_map()
    pending_package = build_native_update_release_review_package(gate_artifacts=artifacts)
    pending_archive = build_native_update_release_review_archive(
        release_review_package=pending_package,
        owner_release_review_record={},
        write_artifact=False,
    )
    assert pending_archive["ok"] is False, pending_archive
    assert pending_archive["evidence_ready"] is True, pending_archive
    assert pending_archive["ready_for_review"] is True, pending_archive
    assert pending_archive["default_off"] is True, pending_archive
    assert pending_archive["archive_ready"] is False, pending_archive
    assert pending_archive["archive_recorded"] is False, pending_archive
    assert pending_archive["ready_for_owner_release_direction"] is False, pending_archive
    assert pending_archive["decision"] == ARCHIVE_HOLD_DECISION, pending_archive
    assert pending_archive["release_review_package_summary"]["optimizer_family_counts"] == pending_package[
        "supplemental_gate_summaries"
    ]["optimizer_family_coverage"]["optimizer_family_counts"], pending_archive
    assert pending_archive["release_review_package_summary"]["optimizer_family_source_payload_digest_match"] is True, (
        pending_archive
    )
    assert pending_archive["release_review_package_summary"]["optimizer_family_handoff_sources_match"] is True, (
        pending_archive
    )
    pending_multitensor = pending_archive["release_review_package_summary"]["supplemental_gate_summaries"][
        "native_update_optimizer_multitensor_release_hold"
    ]
    assert pending_multitensor["present"] is True, pending_archive
    assert pending_multitensor["ok"] is True, pending_archive
    assert pending_multitensor["evidence_ready"] is True, pending_archive
    assert pending_multitensor["ready_for_review"] is True, pending_archive
    assert pending_multitensor["default_off"] is True, pending_archive
    assert pending_archive["release_review_package_summary"][
        "native_update_optimizer_multitensor_release_hold_default_off"
    ] is True, pending_archive
    assert pending_archive["owner_release_review_record_summary"]["present"] is False, pending_archive
    assert pending_archive["owner_release_review_record_summary"]["release_review_recorded"] is False, pending_archive
    assert "archive_review_not_recorded" in "\n".join(pending_archive["blocked_reasons"]), pending_archive
    _assert_default_off(pending_archive)

    template = pending_package["release_review_template"]
    signed_package = build_native_update_release_review_package(
        gate_artifacts=artifacts,
        release_review=_release_review(
            approve=True,
            acknowledged_gates=template["acknowledged_gates"],
            acknowledged_supplemental_gates=template["acknowledged_supplemental_gates"],
        ),
    )
    assert signed_package["decision"] == READY_DECISION, signed_package
    signed_archive = build_native_update_release_review_archive(
        release_review_package=signed_package,
        owner_release_review_record=_signed_owner_release_review_record(),
        write_artifact=False,
    )
    assert signed_archive["ok"] is True, signed_archive
    assert signed_archive["evidence_ready"] is True, signed_archive
    assert signed_archive["ready_for_review"] is True, signed_archive
    assert signed_archive["default_off"] is True, signed_archive
    assert signed_archive["archive_ready"] is True, signed_archive
    assert signed_archive["archive_recorded"] is True, signed_archive
    assert signed_archive["ready_for_owner_release_direction"] is True, signed_archive
    assert signed_archive["decision"] == ARCHIVE_READY_DECISION, signed_archive
    assert signed_archive["release_review_package_summary"]["optimizer_family_counts"] == signed_package[
        "supplemental_gate_summaries"
    ]["optimizer_family_coverage"]["optimizer_family_counts"], signed_archive
    assert signed_archive["release_review_package_summary"]["optimizer_family_source_payload_digest_match"] is True, (
        signed_archive
    )
    assert signed_archive["release_review_package_summary"]["optimizer_family_handoff_sources_match"] is True, (
        signed_archive
    )
    signed_multitensor = signed_archive["release_review_package_summary"]["supplemental_gate_summaries"][
        "native_update_optimizer_multitensor_release_hold"
    ]
    assert signed_multitensor["present"] is True, signed_archive
    assert signed_multitensor["evidence_ready"] is True, signed_archive
    assert signed_multitensor["default_off"] is True, signed_archive
    assert signed_archive["allowed_next_actions"] == ["await_owner_release_direction"], signed_archive
    assert signed_archive["owner_release_review_record_summary"]["present"] is True, signed_archive
    assert signed_archive["owner_release_review_record_summary"]["signed_review_valid"] is True, signed_archive
    assert signed_archive["owner_release_review_record_summary"]["release_review_recorded"] is True, signed_archive
    _assert_default_off(signed_archive)

    unsafe_package = {
        **signed_package,
        "native_dispatch_allowed": True,
    }
    unsafe_archive = build_native_update_release_review_archive(
        release_review_package=unsafe_package,
        write_artifact=False,
    )
    assert unsafe_archive["ok"] is False, unsafe_archive
    assert unsafe_archive["evidence_ready"] is False, unsafe_archive
    assert unsafe_archive["ready_for_review"] is False, unsafe_archive
    assert unsafe_archive["default_off"] is True, unsafe_archive
    assert unsafe_archive["decision"] == ARCHIVE_BLOCKED_DECISION, unsafe_archive
    assert "native_dispatch_allowed" in "\n".join(unsafe_archive["blocked_reasons"]), unsafe_archive
    _assert_default_off(unsafe_archive)

    incomplete_supplemental = json.loads(json.dumps(signed_package))
    incomplete_supplemental["supplemental_gate_summaries"]["optimizer_family_coverage"][
        "ready_for_review"
    ] = False
    incomplete_archive = build_native_update_release_review_archive(
        release_review_package=incomplete_supplemental,
        write_artifact=False,
    )
    assert incomplete_archive["ok"] is False, incomplete_archive
    assert incomplete_archive["evidence_ready"] is False, incomplete_archive
    assert incomplete_archive["ready_for_review"] is False, incomplete_archive
    assert incomplete_archive["default_off"] is True, incomplete_archive
    assert incomplete_archive["decision"] == ARCHIVE_BLOCKED_DECISION, incomplete_archive
    assert "optimizer_family_coverage_ready_for_review_failed" in "\n".join(
        incomplete_archive["blocked_reasons"]
    ), incomplete_archive
    _assert_default_off(incomplete_archive)

    incomplete_multitensor = json.loads(json.dumps(signed_package))
    incomplete_multitensor["supplemental_gate_summaries"]["native_update_optimizer_multitensor_release_hold"][
        "ready_for_review"
    ] = False
    incomplete_multitensor_archive = build_native_update_release_review_archive(
        release_review_package=incomplete_multitensor,
        write_artifact=False,
    )
    assert incomplete_multitensor_archive["ok"] is False, incomplete_multitensor_archive
    assert incomplete_multitensor_archive["evidence_ready"] is False, incomplete_multitensor_archive
    assert incomplete_multitensor_archive["ready_for_review"] is False, incomplete_multitensor_archive
    assert incomplete_multitensor_archive["decision"] == ARCHIVE_BLOCKED_DECISION, incomplete_multitensor_archive
    assert "supplemental_gate_ready_for_review_failed:native_update_optimizer_multitensor_release_hold" in "\n".join(
        incomplete_multitensor_archive["blocked_reasons"]
    ), incomplete_multitensor_archive
    _assert_default_off(incomplete_multitensor_archive)

    source_payload_mismatch = json.loads(json.dumps(signed_package))
    source_payload_mismatch["supplemental_gate_summaries"]["optimizer_family_coverage"][
        "source_payload_digest_match"
    ] = False
    source_payload_mismatch_archive = build_native_update_release_review_archive(
        release_review_package=source_payload_mismatch,
        write_artifact=False,
    )
    assert source_payload_mismatch_archive["ok"] is False, source_payload_mismatch_archive
    assert source_payload_mismatch_archive["evidence_ready"] is False, source_payload_mismatch_archive
    assert source_payload_mismatch_archive["ready_for_review"] is False, source_payload_mismatch_archive
    assert source_payload_mismatch_archive["default_off"] is True, source_payload_mismatch_archive
    assert source_payload_mismatch_archive["decision"] == ARCHIVE_BLOCKED_DECISION, source_payload_mismatch_archive
    assert "optimizer_family_source_payload_digest_match_failed" in "\n".join(
        source_payload_mismatch_archive["blocked_reasons"]
    ), source_payload_mismatch_archive
    _assert_default_off(source_payload_mismatch_archive)

    package_state_mismatch = json.loads(json.dumps(signed_package))
    package_state_mismatch["ready_for_review"] = False
    package_state_mismatch["default_off"] = False
    package_state_mismatch_archive = build_native_update_release_review_archive(
        release_review_package=package_state_mismatch,
        write_artifact=False,
    )
    assert package_state_mismatch_archive["ok"] is False, package_state_mismatch_archive
    assert package_state_mismatch_archive["evidence_ready"] is False, package_state_mismatch_archive
    assert package_state_mismatch_archive["ready_for_review"] is False, package_state_mismatch_archive
    assert package_state_mismatch_archive["default_off"] is True, package_state_mismatch_archive
    assert package_state_mismatch_archive["decision"] == ARCHIVE_BLOCKED_DECISION, package_state_mismatch_archive
    package_state_haystack = "\n".join(package_state_mismatch_archive["blocked_reasons"])
    assert "native_update_release_review_archive_not_ready_for_review" in package_state_haystack, (
        package_state_mismatch_archive
    )
    assert "native_update_release_review_archive_reported_default_off_failed" in package_state_haystack, (
        package_state_mismatch_archive
    )
    _assert_default_off(package_state_mismatch_archive)

    handoff_counts_mismatch = json.loads(json.dumps(signed_package))
    handoff_counts_mismatch["owner_release_review_handoff"]["supplemental_acknowledgement_counts"][
        "optimizer_family_coverage"
    ]["plugin_selected_native_ready_count"] = 1
    handoff_counts_mismatch_archive = build_native_update_release_review_archive(
        release_review_package=handoff_counts_mismatch,
        write_artifact=False,
    )
    assert handoff_counts_mismatch_archive["ok"] is False, handoff_counts_mismatch_archive
    assert handoff_counts_mismatch_archive["evidence_ready"] is False, handoff_counts_mismatch_archive
    assert handoff_counts_mismatch_archive["ready_for_review"] is False, handoff_counts_mismatch_archive
    assert handoff_counts_mismatch_archive["default_off"] is True, handoff_counts_mismatch_archive
    assert handoff_counts_mismatch_archive["decision"] == ARCHIVE_BLOCKED_DECISION, handoff_counts_mismatch_archive
    assert "optimizer_family_handoff_counts_match_failed" in "\n".join(
        handoff_counts_mismatch_archive["blocked_reasons"]
    ), handoff_counts_mismatch_archive
    _assert_default_off(handoff_counts_mismatch_archive)

    supplemental_counts_mismatch = json.loads(json.dumps(signed_package))
    supplemental_counts_mismatch["supplemental_gate_summaries"]["optimizer_family_coverage"][
        "optimizer_family_counts"
    ]["plugin_selected_native_ready_count"] = 1
    supplemental_counts_mismatch_archive = build_native_update_release_review_archive(
        release_review_package=supplemental_counts_mismatch,
        write_artifact=False,
    )
    assert supplemental_counts_mismatch_archive["ok"] is False, supplemental_counts_mismatch_archive
    assert supplemental_counts_mismatch_archive["evidence_ready"] is False, supplemental_counts_mismatch_archive
    assert supplemental_counts_mismatch_archive["ready_for_review"] is False, supplemental_counts_mismatch_archive
    assert supplemental_counts_mismatch_archive["default_off"] is True, supplemental_counts_mismatch_archive
    assert supplemental_counts_mismatch_archive["decision"] == ARCHIVE_BLOCKED_DECISION, (
        supplemental_counts_mismatch_archive
    )
    assert "optimizer_family_handoff_counts_match_failed" in "\n".join(
        supplemental_counts_mismatch_archive["blocked_reasons"]
    ), supplemental_counts_mismatch_archive
    _assert_default_off(supplemental_counts_mismatch_archive)

    handoff_sources_mismatch = json.loads(json.dumps(signed_package))
    handoff_sources_mismatch["owner_release_review_handoff"]["supplemental_acknowledgement_sources"][
        "optimizer_family_coverage"
    ]["source_count"] = 99
    handoff_sources_mismatch_archive = build_native_update_release_review_archive(
        release_review_package=handoff_sources_mismatch,
        write_artifact=False,
    )
    assert handoff_sources_mismatch_archive["ok"] is False, handoff_sources_mismatch_archive
    assert handoff_sources_mismatch_archive["evidence_ready"] is False, handoff_sources_mismatch_archive
    assert handoff_sources_mismatch_archive["ready_for_review"] is False, handoff_sources_mismatch_archive
    assert handoff_sources_mismatch_archive["default_off"] is True, handoff_sources_mismatch_archive
    assert handoff_sources_mismatch_archive["decision"] == ARCHIVE_BLOCKED_DECISION, handoff_sources_mismatch_archive
    assert "optimizer_family_handoff_sources_match_failed" in "\n".join(
        handoff_sources_mismatch_archive["blocked_reasons"]
    ), handoff_sources_mismatch_archive
    _assert_default_off(handoff_sources_mismatch_archive)

    handoff_digest_mismatch = json.loads(json.dumps(signed_package))
    handoff_digest_mismatch["owner_release_review_handoff"]["release_review_template_digest"] = (
        "wrong_release_review_template_digest"
    )
    handoff_digest_mismatch_archive = build_native_update_release_review_archive(
        release_review_package=handoff_digest_mismatch,
        write_artifact=False,
    )
    assert handoff_digest_mismatch_archive["ok"] is False, handoff_digest_mismatch_archive
    assert handoff_digest_mismatch_archive["evidence_ready"] is False, handoff_digest_mismatch_archive
    assert handoff_digest_mismatch_archive["ready_for_review"] is False, handoff_digest_mismatch_archive
    assert handoff_digest_mismatch_archive["default_off"] is True, handoff_digest_mismatch_archive
    assert handoff_digest_mismatch_archive["decision"] == ARCHIVE_BLOCKED_DECISION, handoff_digest_mismatch_archive
    assert "handoff_release_review_template_digest_match_failed" in "\n".join(
        handoff_digest_mismatch_archive["blocked_reasons"]
    ), handoff_digest_mismatch_archive
    _assert_default_off(handoff_digest_mismatch_archive)

    real_package = _write_real_artifact_case(artifacts)
    real_pending_archive = build_native_update_release_review_archive(
        release_review_package=real_package,
        write_artifact=True,
    )
    artifact = REPO_ROOT / "temp" / "turbocore_optimizer" / "native_update_release_review_archive.json"
    loaded = json.loads(artifact.read_text(encoding="utf-8"))
    assert loaded["decision"] == ARCHIVE_HOLD_DECISION, loaded
    assert loaded["decision"] == real_pending_archive["decision"], loaded
    assert loaded["runtime_dispatch_allowed"] is False, loaded
    assert loaded["native_dispatch_allowed"] is False, loaded
    assert loaded["training_path_enabled"] is False, loaded
    assert loaded["evidence_ready"] is True, loaded
    assert loaded["ready_for_review"] is True, loaded
    assert loaded["default_off"] is True, loaded
    assert loaded["post_archive_request_fields"] == {}, loaded
    assert loaded["release_review_package_summary"]["handoff_release_review_template_digest_match"] is True, loaded
    assert loaded["release_review_package_summary"]["optimizer_family_source_payload_digest_match"] is True, loaded
    assert loaded["release_review_package_summary"]["optimizer_family_handoff_sources_match"] is True, loaded
    assert "owner_release_review_record_summary" in loaded, loaded
    loaded_multitensor = loaded["release_review_package_summary"]["supplemental_gate_summaries"][
        "native_update_optimizer_multitensor_release_hold"
    ]
    assert loaded_multitensor["present"] is True, loaded
    assert loaded_multitensor["evidence_ready"] is True, loaded
    assert loaded_multitensor["ready_for_review"] is True, loaded
    assert loaded_multitensor["default_off"] is True, loaded
    assert loaded["release_review_package_summary"]["optimizer_family_counts"] == real_package[
        "supplemental_gate_summaries"
    ]["optimizer_family_coverage"]["optimizer_family_counts"], loaded
    _assert_optimizer_family_counts(
        loaded["release_review_package_summary"]["optimizer_family_counts"],
        allow_extra=True,
    )
    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_release_review_archive_smoke",
        "ok": True,
        "pending_decision": pending_archive["decision"],
        "signed_decision": signed_archive["decision"],
        "real_artifact_checked": True,
        "recommended_next_step": signed_archive["recommended_next_step"],
    }


def _signed_owner_release_review_record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_native_update_owner_release_review_record_v0",
        "gate": "native_update_owner_release_review_record",
        "ok": True,
        "owner_packet_ready": True,
        "signed_review_present": True,
        "signed_review_valid": True,
        "approval_recorded": True,
        "release_review_recorded": True,
        "decision": "native_update_release_review_recorded_default_off",
        "source_release_review_template_digest": "synthetic_smoke_digest",
        "signed_review_template_digest": "synthetic_smoke_digest",
        "signed_review_digest_match": True,
        "release_package_decision": READY_DECISION,
        "release_package_digest": "synthetic_smoke_release_package_digest",
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


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in (
        "default_behavior_changed",
        "product_exposure_allowed",
        "release_gate_open",
        "training_launch_allowed",
        "training_launch_enabled",
        "training_launch_executed",
        "training_path_enabled",
        "training_dispatch",
        "runtime_dispatch_allowed",
        "native_dispatch_allowed",
        "native_dispatch_enabled",
        "native_dispatch_executed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "backend_router_registered",
    ):
        assert report[field] is False, (field, report)
    assert report["post_archive_request_fields"] == {}, report
    assert report["post_release_request_fields"] == {}, report
    assert report["default_off"] is True, report


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
