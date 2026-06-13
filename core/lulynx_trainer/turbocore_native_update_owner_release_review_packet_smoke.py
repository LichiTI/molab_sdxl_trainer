"""Smoke for the TurboCore native-update owner release-review action packet."""

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

from core.turbocore_native_update_owner_release_handoff_summary import (  # noqa: E402
    build_native_update_owner_release_handoff_summary,
)
from core.turbocore_native_update_owner_release_direction_packet import (  # noqa: E402
    build_native_update_owner_release_direction_packet,
)
from core.turbocore_native_update_owner_release_direction_record import (  # noqa: E402
    build_native_update_owner_release_direction_record,
)
from core.turbocore_native_update_owner_release_review_packet import (  # noqa: E402
    build_native_update_owner_release_review_packet,
    main as packet_cli_main,
)
from core.turbocore_native_update_release_review_package import (  # noqa: E402
    EXPECTED_GATES,
    READY_DECISION,
    SUPPLEMENTAL_GATES,
    build_native_update_release_review_package,
    load_gate_artifacts,
)
from core.turbocore_native_update_representative_performance_importer import (  # noqa: E402
    build_native_update_representative_performance_import,
)
from core.turbocore_optimizer_product_training_route_binding_config_adapter import (  # noqa: E402
    build_optimizer_product_training_route_binding_config_adapter,
)
from core.turbocore_optimizer_product_training_route_binding_product_route_adapter import (  # noqa: E402
    build_optimizer_product_training_route_binding_product_route_adapter,
)
from core.turbocore_optimizer_product_training_route_binding_runtime_applier import (  # noqa: E402
    apply_optimizer_product_training_route_binding_runtime_patch,
)
from core.turbocore_optimizer_product_training_route_binding_run_local_staging import (  # noqa: E402
    build_optimizer_product_training_route_binding_run_local_staging,
)
from core.turbocore_optimizer_product_training_route_binding_preflight import (  # noqa: E402
    build_optimizer_product_training_route_binding_preflight,
)
from core.turbocore_optimizer_product_training_route_binding_training_loop_contract import (  # noqa: E402
    build_optimizer_product_training_route_binding_training_loop_contract,
)
from core.turbocore_optimizer_stable_first_release_scope import (  # noqa: E402
    build_turbocore_optimizer_stable_first_release_scope,
)


def run_smoke() -> dict[str, Any]:
    artifact_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    build_native_update_representative_performance_import(write_artifacts=True)
    build_optimizer_product_training_route_binding_preflight(write_artifact=True)
    build_optimizer_product_training_route_binding_training_loop_contract(write_artifact=True)
    build_optimizer_product_training_route_binding_config_adapter(write_artifact=True)
    build_optimizer_product_training_route_binding_product_route_adapter(write_artifact=True)
    apply_optimizer_product_training_route_binding_runtime_patch({}, write_artifact=True)
    build_optimizer_product_training_route_binding_run_local_staging(write_artifact=True, write_run_local_adapter=False)
    build_turbocore_optimizer_stable_first_release_scope(write_artifact=True)
    direction_packet = build_native_update_owner_release_direction_packet(write_artifact=True)
    build_native_update_owner_release_direction_record(
        owner_direction_packet=direction_packet,
        write_artifact=True,
    )
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
    template = packet["signable_review_record_template"]
    compact = packet["compact_evidence"]

    assert packet["roadmap"] == "devtools/docs/turbocore_optimizer_backend_design_v2.md", packet
    assert packet["ok"] is True, packet
    assert packet["ready_for_owner_signature"] is True, packet
    assert packet["approval_recorded"] is False, packet
    assert packet["release_review_recorded"] is False, packet
    assert packet["digest_match"] is True, packet
    expected_decision = "native_update_release_review_hold_for_owner_review_default_off"
    assert packet["source_release_package_decision"] == expected_decision, packet
    assert packet["blocked_reasons"] == ["native_update_release_owner_review_missing"], packet
    assert packet["required_gate_acknowledgement_count"] == len(EXPECTED_GATES), packet
    assert len(packet["required_supplemental_acknowledgements"]) == len(SUPPLEMENTAL_GATES), packet
    assert template["approve_native_update_release_review_package"] is False, packet
    assert template["reviewer"] == "", packet
    assert template["reviewed_at"] == "", packet
    assert template["source_release_review_template_digest"] == packet["source_release_review_template_digest"], packet
    assert template["source_release_package_digest"] == packet["source_release_package_digest"], packet
    for field in packet["required_acknowledgement_fields"]:
        assert template[field] is False, packet
    assert len(template["acknowledged_gates"]) == len(EXPECTED_GATES), packet
    assert len(template["acknowledged_supplemental_gates"]) == len(SUPPLEMENTAL_GATES), packet
    assert compact["expected_gate_count"] == len(EXPECTED_GATES), packet
    assert compact["present_gate_count"] == len(EXPECTED_GATES), packet
    assert compact["default_off_gate_count"] == len(EXPECTED_GATES), packet
    assert compact["representative_performance_artifact_present_count"] == 1, packet
    assert compact["representative_performance_gate_ready_count"] == 1, packet
    assert compact["representative_performance_fresh_live_run_count"] == 0, packet
    assert compact["plugin_optimizer_count"] == 124, packet
    assert compact["plugin_selected_native_ready_count"] == 0, packet
    assert compact["native_readiness_runtime_launch_coverage_ready_family_count"] == 10, packet
    assert compact["native_readiness_runtime_launch_adapter_ready_optimizer_count"] == 72, packet
    assert compact["native_readiness_owner_release_hold_ready_family_count"] == 10, packet
    assert compact["native_readiness_request_schema_ui_non_exposure_ready_family_count"] == 10, packet
    assert compact["native_readiness_family_specific_runtime_launch_missing_count"] == 0, packet
    assert compact["product_route_binding_preflight_ready_count"] == 0, packet
    assert compact["product_route_binding_candidate_count"] == 0, packet
    assert compact["owner_release_direction_ready_for_signature_count"] == 0, packet
    assert compact["owner_release_direction_recorded_count"] == 0, packet
    assert compact["owner_release_direction_approval_recorded_count"] == 0, packet
    synthetic_direction_packet = build_native_update_owner_release_review_packet(
        release_package=_release_package_ready(),
        handoff_summary=_handoff_with_direction_approval(),
        write_artifact=False,
    )
    synthetic_compact = synthetic_direction_packet["compact_evidence"]
    assert synthetic_direction_packet["ok"] is True, synthetic_direction_packet
    assert synthetic_compact["owner_release_direction_ready_for_signature_count"] == 1, synthetic_direction_packet
    assert synthetic_compact["owner_release_direction_recorded_count"] == 1, synthetic_direction_packet
    assert synthetic_compact["owner_release_direction_approval_recorded_count"] == 1, synthetic_direction_packet
    _assert_packet_default_off(synthetic_direction_packet)
    assert compact["training_loop_route_candidate_switch_count"] == 3, packet
    assert compact["training_loop_route_open_training_path_enabled_count"] == 1, packet
    assert compact["training_loop_route_request_fields_emitted_count"] == 0, packet
    assert compact["route_binding_config_patch_ready_count"] == 0, packet
    assert compact["route_binding_constructor_switch_field_count"] == 3, packet
    assert compact["route_binding_kwargs_patch_field_count"] == 0, packet
    assert compact["product_route_binding_product_route_count"] == 4, packet
    assert compact["product_route_binding_kwargs_wired_count"] == 4, packet
    assert compact["route_binding_runtime_config_patch_applied_count"] == 0, packet
    assert compact["route_binding_runtime_config_patch_field_count"] == 0, packet
    assert compact["route_binding_run_local_adapter_staged_count"] == 0, packet
    assert compact["product_launch_staging_wired_count"] == 2, packet
    assert compact["stable_first_release_turbocore_optimizer_blocker_count"] == 0, packet
    assert compact["turbocore_optimizer_default_off_release_scope_ready_count"] == 1, packet
    assert packet["product_exposure_allowed"] is False, packet
    assert packet["request_fields_emitted"] is False, packet
    assert packet["schema_exposure_allowed"] is False, packet
    assert packet["ui_exposure_allowed"] is False, packet
    assert packet["runtime_dispatch_allowed"] is False, packet
    assert packet["native_dispatch_allowed"] is False, packet
    assert packet["training_path_enabled"] is False, packet
    assert packet["training_launch_executed"] is False, packet
    assert _run_packet_cli(["--no-artifact"]) == 0, packet

    signed_review = dict(template)
    signed_review.update(
        {
            "reviewer": "synthetic_owner_packet_smoke",
            "reviewed_at": "2026-06-07",
            "approve_native_update_release_review_package": True,
        }
    )
    for field in packet["required_acknowledgement_fields"]:
        signed_review[field] = True
    signed = build_native_update_release_review_package(
        gate_artifacts=load_gate_artifacts(artifact_dir),
        release_review=signed_review,
    )
    assert signed["ok"] is True, signed
    assert signed["decision"] == READY_DECISION, signed
    assert signed["release_review_recorded"] is True, signed
    assert signed["blocked_reasons"] == [], signed
    _assert_default_off(signed)

    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_owner_release_review_packet_smoke",
        "ok": True,
        "roadmap": packet["roadmap"],
        "ready_for_owner_signature": packet["ready_for_owner_signature"],
        "approval_recorded": packet["approval_recorded"],
        "summary": {
            "expected_gate_count": compact["expected_gate_count"],
            "supplemental_gate_count": compact["supplemental_gate_count"],
            "representative_performance_gate_ready_count": compact[
                "representative_performance_gate_ready_count"
            ],
            "native_readiness_runtime_launch_coverage_ready_family_count": compact[
                "native_readiness_runtime_launch_coverage_ready_family_count"
            ],
            "plugin_optimizer_count": compact["plugin_optimizer_count"],
            "product_route_binding_preflight_ready_count": compact[
                "product_route_binding_preflight_ready_count"
            ],
            "owner_release_direction_ready_for_signature_count": compact[
                "owner_release_direction_ready_for_signature_count"
            ],
            "owner_release_direction_recorded_count": compact[
                "owner_release_direction_recorded_count"
            ],
            "owner_release_direction_approval_recorded_count": compact[
                "owner_release_direction_approval_recorded_count"
            ],
            "training_loop_route_candidate_switch_count": compact[
                "training_loop_route_candidate_switch_count"
            ],
            "route_binding_config_patch_ready_count": compact[
                "route_binding_config_patch_ready_count"
            ],
            "product_route_binding_kwargs_wired_count": compact[
                "product_route_binding_kwargs_wired_count"
            ],
            "route_binding_runtime_config_patch_applied_count": compact[
                "route_binding_runtime_config_patch_applied_count"
            ],
            "route_binding_run_local_adapter_staged_count": compact[
                "route_binding_run_local_adapter_staged_count"
            ],
            "product_launch_staging_wired_count": compact[
                "product_launch_staging_wired_count"
            ],
            "stable_first_release_turbocore_optimizer_blocker_count": compact[
                "stable_first_release_turbocore_optimizer_blocker_count"
            ],
            "turbocore_optimizer_default_off_release_scope_ready_count": compact[
                "turbocore_optimizer_default_off_release_scope_ready_count"
            ],
        },
    }


def _release_package_ready() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": True,
        "evidence_ready": True,
        "ready_for_owner_release_review": True,
        "default_off": True,
        "release_review_recorded": False,
        "decision": "native_update_release_review_hold_for_owner_review_default_off",
        "expected_gate_count": 0,
        "present_gate_count": 0,
        "default_off_gate_count": 0,
        "supplemental_gate_count": 0,
        "present_supplemental_gate_count": 0,
        "default_off_supplemental_gate_count": 0,
        "blocked_reasons": ["native_update_release_owner_review_missing"],
        "owner_release_review_handoff": {"release_review_template_digest": "synthetic-digest"},
        "release_review_template": {},
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "post_release_request_fields": {},
    }


def _handoff_with_direction_approval() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": True,
        "technical_evidence_ready": True,
        "source_release_package_digest": "synthetic-digest",
        "required_review_fields": [],
        "required_requested_scope": "",
        "required_gate_acknowledgement_count": 0,
        "required_supplemental_acknowledgements": [],
        "review_template_for_owner": {},
        "must_remain_false": [],
        "must_remain_empty": [],
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "summary": {
            "owner_release_direction_ready_for_signature_count": 1,
            "owner_release_direction_recorded_count": 1,
            "owner_release_direction_approval_recorded_count": 1,
            "owner_release_approval_recorded_count": 0,
            "training_path_enabled_count": 0,
        },
    }


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
    assert report["post_release_request_fields"] == {}, report


def _assert_packet_default_off(report: dict[str, Any]) -> None:
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


def _run_packet_cli(args: list[str]) -> int:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        return int(packet_cli_main(args))


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
