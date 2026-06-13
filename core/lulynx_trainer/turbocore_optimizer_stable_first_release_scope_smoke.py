"""Smoke for TurboCore optimizer stable first-release default-off scope."""

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

from core.turbocore_native_update_owner_release_review_packet import (  # noqa: E402
    build_native_update_owner_release_review_packet,
)
from core.turbocore_native_update_representative_performance_importer import (  # noqa: E402
    build_native_update_representative_performance_import,
)
from core.turbocore_optimizer_product_training_route_binding_preflight import (  # noqa: E402
    build_optimizer_product_training_route_binding_preflight,
)
from core.turbocore_optimizer_product_training_route_binding_run_local_staging import (  # noqa: E402
    build_optimizer_product_training_route_binding_run_local_staging,
)
from core.turbocore_optimizer_stable_first_release_scope import (  # noqa: E402
    build_turbocore_optimizer_stable_first_release_scope,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def run_smoke() -> dict[str, Any]:
    build_native_update_representative_performance_import(write_artifacts=True)
    packet = build_native_update_owner_release_review_packet(write_artifact=True)
    preflight = build_optimizer_product_training_route_binding_preflight(write_artifact=True)
    staging = build_optimizer_product_training_route_binding_run_local_staging(
        write_artifact=True,
        write_run_local_adapter=False,
    )
    report = build_turbocore_optimizer_stable_first_release_scope(
        owner_packet=packet,
        route_preflight=preflight,
        run_local_staging=staging,
        write_artifact=True,
    )
    assert report["roadmap"] == ROADMAP, report
    assert report["ok"] is True, report
    assert report["stable_first_release_scope"] == "stable_baseline_with_turbocore_optimizer_default_off", report
    assert report["stable_first_release_blocked_by_turbocore_optimizer"] is False, report
    assert report["turbocore_optimizer_default_off_release_scope_ready"] is True, report
    assert report["owner_signature_ready"] is True, report
    assert report["release_claim_allowed"] is True, report
    assert report["native_training_claim_allowed"] is False, report
    assert report["blocked_reasons"] == [], report
    _assert_default_off(report)
    summary = report["summary"]
    assert summary["stable_first_release_turbocore_optimizer_blocker_count"] == 0, report
    assert summary["turbocore_optimizer_default_off_release_scope_ready_count"] == 1, report
    assert summary["owner_signature_ready_count"] == 1, report
    assert summary["owner_release_approval_recorded_count"] == 0, report
    assert summary["owner_release_direction_recorded_count"] == 0, report
    assert summary["owner_release_direction_approval_recorded_count"] == 0, report
    assert summary["product_exposure_decision_recorded_count"] == 0, report
    assert summary["product_training_route_binding_ready_count"] == 0, report
    assert summary["run_local_adapter_staged_count"] == 0, report
    assert summary["runtime_config_patch_applied_count"] == 0, report
    assert summary["training_path_enabled_count"] == 0, report

    unsafe = build_turbocore_optimizer_stable_first_release_scope(
        owner_packet={**packet, "ready_for_owner_signature": False},
        route_preflight={**preflight, "product_training_route_binding_preflight_ready": True},
        run_local_staging={**staging, "run_local_adapter_staged": True},
        write_artifact=False,
    )
    assert unsafe["stable_first_release_blocked_by_turbocore_optimizer"] is True, unsafe
    assert unsafe["release_claim_allowed"] is False, unsafe
    assert "owner_signature_packet_not_ready" in unsafe["blocked_reasons"], unsafe
    assert "route_binding_ready_before_stable_first_release" in unsafe["blocked_reasons"], unsafe
    assert "run_local_adapter_staged_before_stable_first_release" in unsafe["blocked_reasons"], unsafe

    signed_direction_scope = build_turbocore_optimizer_stable_first_release_scope(
        owner_packet={
            **packet,
            "compact_evidence": {
                **dict(packet.get("compact_evidence") or {}),
                "owner_release_direction_recorded_count": 1,
                "owner_release_direction_approval_recorded_count": 1,
            },
        },
        route_preflight=preflight,
        run_local_staging=staging,
        write_artifact=False,
    )
    signed_direction_summary = signed_direction_scope["summary"]
    assert signed_direction_scope["release_claim_allowed"] is False, signed_direction_scope
    assert signed_direction_scope["stable_first_release_blocked_by_turbocore_optimizer"] is True, (
        signed_direction_scope
    )
    assert "owner_release_direction_recorded_count_before_stable_first_release" in signed_direction_scope[
        "blocked_reasons"
    ], signed_direction_scope
    assert "owner_release_direction_approval_recorded_count_before_stable_first_release" in signed_direction_scope[
        "blocked_reasons"
    ], signed_direction_scope
    assert signed_direction_summary["stable_first_release_turbocore_optimizer_blocker_count"] == 1, (
        signed_direction_scope
    )
    assert signed_direction_summary["turbocore_optimizer_default_off_release_scope_ready_count"] == 0, (
        signed_direction_scope
    )
    assert signed_direction_summary["owner_release_direction_recorded_count"] == 1, signed_direction_scope
    assert signed_direction_summary["owner_release_direction_approval_recorded_count"] == 1, signed_direction_scope
    _assert_default_off(signed_direction_scope)

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_stable_first_release_scope_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
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
        "owner_release_approval_recorded",
        "product_exposure_decision_recorded",
        "product_training_route_binding_ready",
    ):
        assert report[field] is False, (field, report)


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
