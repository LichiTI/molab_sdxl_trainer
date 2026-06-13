"""End-to-end owner-direction count propagation smoke for TurboCore."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (str(SCRIPT_ROOT), str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_native_update_owner_release_direction_packet import (  # noqa: E402
    build_native_update_owner_release_direction_packet,
)
from core.turbocore_native_update_owner_release_direction_record import (  # noqa: E402
    build_native_update_owner_release_direction_record,
)
from core.turbocore_native_update_owner_release_handoff_summary import (  # noqa: E402
    build_native_update_owner_release_handoff_summary,
)
from core.turbocore_native_update_owner_release_review_packet import (  # noqa: E402
    build_native_update_owner_release_review_packet,
)
from core.turbocore_native_update_release_review_archive import (  # noqa: E402
    build_native_update_release_review_archive,
)
from core.turbocore_optimizer_v2_approval_execution_preflight import (  # noqa: E402
    build_optimizer_v2_approval_execution_preflight,
)
from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    build_optimizer_v2_reviewer_handoff_packet,
)
from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    build_optimizer_v2_signature_bundle_packet,
)
from core.turbocore_optimizer_stable_first_release_scope import (  # noqa: E402
    build_turbocore_optimizer_stable_first_release_scope,
)
from turbocore_native_update_promotion_scorecard_smoke import (  # noqa: E402
    _owner_release_review_record_ready,
    _product_exposure_decision_ready,
    _release_review_package_ready,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = Path(temp_dir)
        archive = build_native_update_release_review_archive(
            release_review_package=_recorded_release_review_package(),
            owner_release_review_record=_owner_release_review_record_ready(),
            stable_first_release_scope=_stable_first_release_scope(counts=(0, 0)),
            write_artifact=True,
            artifact_path=artifact_dir / "native_update_release_review_archive.json",
        )
        assert archive["archive_ready"] is True, archive
        assert archive["ready_for_owner_release_direction"] is True, archive
        _assert_default_off(archive)

        ready_packet = build_native_update_owner_release_direction_packet(
            release_review_archive=archive,
            product_exposure_decision=_product_exposure_decision_ready(),
            stable_first_release_scope=_stable_first_release_scope(counts=(0, 0)),
            artifact_dir=artifact_dir,
            write_artifact=True,
        )
        assert ready_packet["ready_for_owner_direction_signature"] is True, ready_packet
        assert ready_packet["owner_release_direction_recorded"] is False, ready_packet
        assert ready_packet["summary"]["owner_release_direction_recorded_count"] == 0, ready_packet

        preflight = _phase2_preflight(ready_packet, artifact_dir)
        assert preflight["phase2_direction_inputs_ready"] is True, preflight
        assert preflight["summary"]["v2_approval_preflight_phase2_ready_count"] == 1, preflight
        assert preflight["summary"]["v2_approval_preflight_phase1_ready_count"] == 0, preflight
        assert preflight["summary"]["v2_approval_preflight_approval_recorded_count"] == 0, preflight
        _assert_default_off(preflight)

        signed_direction = json.loads((artifact_dir / "signed_owner_release_direction.json").read_text(encoding="utf-8"))
        direction_record = build_native_update_owner_release_direction_record(
            signed_direction=signed_direction,
            owner_direction_packet=ready_packet,
            approval_preflight=preflight,
            artifact_dir=artifact_dir,
            write_artifact=True,
        )
        assert direction_record["owner_release_direction_recorded"] is True, direction_record
        assert direction_record["summary"]["owner_release_direction_recorded_count"] == 1, direction_record
        assert direction_record["summary"]["owner_release_direction_approval_recorded_count"] == 1, direction_record
        _assert_default_off(direction_record)

        handoff = build_native_update_owner_release_handoff_summary(
            release_package=_hold_release_review_package(),
            artifact_dir=artifact_dir,
            write_artifact=False,
        )
        _assert_direction_counts(handoff["summary"], expected=1)
        _assert_default_off(handoff)

        owner_packet = build_native_update_owner_release_review_packet(
            release_package=_hold_release_review_package(),
            handoff_summary=handoff,
            artifact_dir=artifact_dir,
            write_artifact=False,
        )
        _assert_direction_counts(owner_packet["compact_evidence"], expected=1)
        _assert_default_off(owner_packet)

        stable_scope = build_turbocore_optimizer_stable_first_release_scope(
            owner_packet=owner_packet,
            route_preflight=_route_preflight_default_off(),
            run_local_staging=_run_local_staging_default_off(),
            artifact_dir=artifact_dir,
            write_artifact=False,
        )
        _assert_direction_counts(stable_scope["summary"], expected=1, owner_count_expected=0)
        assert stable_scope["turbocore_optimizer_default_off_release_scope_ready"] is False, stable_scope
        assert stable_scope["stable_first_release_blocked_by_turbocore_optimizer"] is True, stable_scope
        assert "owner_release_direction_recorded_count_before_stable_first_release" in stable_scope[
            "blocked_reasons"
        ], stable_scope
        assert "owner_release_direction_approval_recorded_count_before_stable_first_release" in stable_scope[
            "blocked_reasons"
        ], stable_scope
        _assert_default_off(stable_scope)

    return {
        "schema_version": 1,
        "probe": "turbocore_native_update_owner_direction_end_to_end_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "synthetic_owner_direction_end_to_end_validated": True,
        "real_artifact_approval_recorded_count": 0,
        "summary": {
            "synthetic_owner_release_direction_recorded_count": 1,
            "synthetic_owner_release_direction_approval_recorded_count": 1,
            "owner_release_direction_recorded_count": 0,
            "owner_release_direction_approval_recorded_count": 0,
            "owner_release_approval_recorded_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
        },
        "recommended_next_step": (
            "keep real owner direction unrecorded until a real signed owner direction is supplied"
        ),
    }


def _phase2_preflight(packet: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    signature_bundle = build_optimizer_v2_signature_bundle_packet(
        source_reports={
            "remaining_gate_handoff": _remaining_gate_handoff_ready(),
            "owner_release_review_packet": {},
            "product_exposure_decision": {},
            "owner_release_direction_packet": packet,
        },
        write_artifact=False,
    )
    handoff = build_optimizer_v2_reviewer_handoff_packet(
        signature_bundle=signature_bundle,
        write_artifact=False,
        write_signed_bundle_template=False,
    )
    signed_direction = _signed_direction(handoff["signed_bundle_template"]["signed_entries"]["owner_release_direction"])
    signed_direction["source_owner_release_direction_template_digest"] = packet[
        "source_owner_release_direction_template_digest"
    ]
    signed_bundle = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signed_bundle_phase2_end_to_end_smoke_v0",
        "source_signature_bundle_digest": handoff["source_signature_bundle_digest"],
        "approval_recorded": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "signed_entries": {"owner_release_direction": signed_direction},
    }
    paths = {
        "signature_bundle": artifact_dir / "turbocore_optimizer_v2_signature_bundle_packet.json",
        "signed_bundle": artifact_dir / "turbocore_optimizer_v2_signed_bundle.phase2.json",
        "owner_release_review": artifact_dir / "signed_owner_release_review.missing.json",
        "product_exposure_review": artifact_dir / "signed_product_exposure_review.missing.json",
        "owner_release_direction": artifact_dir / "signed_owner_release_direction.json",
        "training_launch_contract": artifact_dir / "native_update_training_launch_contract.missing.json",
        "product_exposure_evidence": artifact_dir / "native_update_product_exposure_evidence.missing.json",
        "owner_release_direction_packet": artifact_dir / "native_update_owner_release_direction_packet.json",
    }
    _write_json(paths["signature_bundle"], signature_bundle)
    _write_json(paths["signed_bundle"], signed_bundle)
    _write_json(paths["owner_release_direction"], signed_direction)
    _write_json(paths["owner_release_direction_packet"], packet)
    return build_optimizer_v2_approval_execution_preflight(paths=paths, write_artifact=False)


def _remaining_gate_handoff_ready() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_remaining_gate_handoff_scorecard_v0",
        "gate": "optimizer_v2_remaining_gate_handoff",
        "handoff_ready": True,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "product_native_ready": False,
        "summary": {
            "v2_remaining_gate_total_count": 6,
            "v2_remaining_gate_default_off_guard_count": 1,
        },
    }


def _signed_direction(template: dict[str, Any]) -> dict[str, Any]:
    signed = dict(template)
    signed.update(
        {
            "reviewer": "synthetic_owner_direction_end_to_end_smoke",
            "reviewed_at": "2026-06-07",
            "approve_native_update_owner_release_direction": True,
        }
    )
    for field in list(signed):
        if field.startswith("acknowledge_"):
            signed[field] = True
    return signed


def _hold_release_review_package() -> dict[str, Any]:
    return {
        **_recorded_release_review_package(),
        "release_review_recorded": False,
        "decision": "native_update_release_review_hold_for_owner_review_default_off",
        "blocked_reasons": ["native_update_release_owner_review_missing"],
        "promotion_blockers": ["native_update_release_owner_review_missing"],
    }


def _recorded_release_review_package() -> dict[str, Any]:
    package = dict(_release_review_package_ready())
    package["runtime_dispatch_allowed"] = False
    package["native_dispatch_allowed"] = False
    package["training_path_enabled"] = False
    package["training_launch_executed"] = False
    return package


def _stable_first_release_scope(*, counts: tuple[int, int]) -> dict[str, Any]:
    recorded_count, approval_count = counts
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
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "blocked_reasons": [],
        "summary": {
            "stable_first_release_turbocore_optimizer_blocker_count": 0,
            "turbocore_optimizer_default_off_release_scope_ready_count": 1,
            "owner_release_approval_recorded_count": 0,
            "owner_release_direction_recorded_count": recorded_count,
            "owner_release_direction_approval_recorded_count": approval_count,
            "product_exposure_decision_recorded_count": 0,
            "product_training_route_binding_ready_count": 0,
            "run_local_adapter_staged_count": 0,
            "runtime_config_patch_applied_count": 0,
            "training_path_enabled_count": 0,
        },
    }


def _route_preflight_default_off() -> dict[str, Any]:
    return {
        "product_training_route_binding_preflight_ready": False,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "summary": {"product_training_route_binding_ready_count": 0},
    }


def _run_local_staging_default_off() -> dict[str, Any]:
    return {
        "run_local_adapter_staged": False,
        "product_exposure_allowed": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "runtime_dispatch_allowed": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "training_launch_executed": False,
        "summary": {
            "run_local_adapter_staged_count": 0,
            "runtime_config_patch_applied_count": 0,
        },
    }


def _assert_direction_counts(
    summary: dict[str, Any],
    *,
    expected: int,
    owner_count_expected: int | None = None,
) -> None:
    assert summary["owner_release_direction_recorded_count"] == expected, summary
    assert summary["owner_release_direction_approval_recorded_count"] == expected, summary
    if owner_count_expected is not None:
        assert summary["owner_release_approval_recorded_count"] == owner_count_expected, summary


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
        if field in report:
            assert report[field] is False, (field, report)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
