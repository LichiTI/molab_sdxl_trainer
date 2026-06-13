"""Smoke for the artifact-first TurboCore optimizer native readiness gap report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_native_readiness_gap_scorecard import (  # noqa: E402
    build_optimizer_native_readiness_gap_scorecard,
)
from core.turbocore_optimizer_product_training_route_binding_preflight import (  # noqa: E402
    build_optimizer_product_training_route_binding_preflight,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design_v2.md"


def run_smoke() -> dict[str, Any]:
    report = build_optimizer_native_readiness_gap_scorecard(write_artifact=True)
    summary = _as_dict(report.get("summary"))
    rows = report.get("rows")
    signed_route_binding_preflight = _signed_route_binding_preflight()
    signed_report = build_optimizer_native_readiness_gap_scorecard(
        route_binding_preflight_report=signed_route_binding_preflight,
        write_artifact=False,
    )
    signed_summary = _as_dict(signed_report.get("summary"))
    signed_post_approval = _as_dict(signed_report.get("post_approval_summary"))
    default_off_product = _as_dict(report.get("default_off_product_summary"))
    signed_preview = _as_dict(report.get("signed_post_approval_preview_summary"))

    assert report.get("ok") is True, report
    assert report.get("roadmap") == ROADMAP, report
    assert report.get("artifact_first") is True, report
    assert report.get("promotion_ready") is False, report
    assert report.get("cuda_executed") is False, report
    assert report.get("runtime_dispatch_ready") is False, report
    assert report.get("native_dispatch_allowed") is False, report
    assert report.get("training_path_enabled") is False, report
    assert report.get("product_native_ready") is False, report
    assert isinstance(rows, list), report
    assert summary.get("route_family_count") == 10, report
    assert summary.get("plugin_optimizer_count") == 124, report
    assert summary.get("selected_optimizer_gate_ready_family_count") == 10, report
    assert summary.get("kernel_source_ready_optimizer_count") == 124, report
    assert summary.get("rust_probe_ready_optimizer_count") == 124, report
    assert summary.get("family_contract_ready_count") == 10, report
    assert summary.get("family_evidence_ready_count") == 10, report
    assert summary.get("runtime_rehearsal_ready_family_count") == 4, report
    assert summary.get("runtime_precondition_ready_family_count") == 6, report
    assert summary.get("family_specific_runtime_launch_adapter_ready_family_count") == 6, report
    assert summary.get("family_specific_runtime_launch_adapter_ready_optimizer_count") == 72, report
    assert summary.get("roadmap_v2_open_work_category_count") == 5, report
    assert summary.get("roadmap_v2_open_work_item_count") == 30, report
    assert summary.get("roadmap_v2_open_work_open_category_count") == 2, report
    assert summary.get("roadmap_v2_open_work_open_item_count") == 6, report
    assert summary.get("runtime_launch_coverage_ready_family_count") == 10, report
    assert summary.get("runtime_launch_coverage_ready_optimizer_count") == 124, report
    assert summary.get("runtime_launch_coverage_mode_counts") == {
        "dispatch": 52,
        "missing": 0,
        "precondition_adapter": 72,
    }, report
    assert summary.get("owner_release_hold_ready_family_count") == 10, report
    assert summary.get("request_schema_ui_non_exposure_ready_family_count") == 10, report
    assert summary.get("runtime_dispatch_ready_family_count") == 0, report
    assert summary.get("native_dispatch_allowed_family_count") == 0, report
    assert summary.get("training_path_enabled_family_count") == 0, report
    assert summary.get("product_native_ready_family_count") == 0, report
    assert default_off_product.get("runtime_dispatch_ready_family_count") == 0, report
    assert default_off_product.get("runtime_dispatch_ready_optimizer_count") == 0, report
    assert default_off_product.get("native_dispatch_allowed_family_count") == 0, report
    assert default_off_product.get("native_dispatch_allowed_optimizer_count") == 0, report
    assert default_off_product.get("training_path_enabled_family_count") == 0, report
    assert default_off_product.get("training_path_enabled_optimizer_count") == 0, report
    assert default_off_product.get("product_native_ready_family_count") == 0, report
    assert default_off_product.get("product_native_ready_optimizer_count") == 0, report
    assert summary.get("default_off_product_runtime_dispatch_ready_optimizer_count") == 0, report
    assert summary.get("default_off_product_native_dispatch_allowed_optimizer_count") == 0, report
    assert summary.get("default_off_product_training_path_enabled_optimizer_count") == 0, report
    assert summary.get("default_off_product_product_native_ready_optimizer_count") == 0, report
    assert signed_preview.get("post_approval_ready") is True, report
    assert signed_preview.get("preview_only") is True, report
    assert signed_preview.get("approval_gated") is True, report
    assert signed_preview.get("default_behavior_changed") is False, report
    assert signed_preview.get("runtime_dispatch_ready_family_count") == 10, report
    assert signed_preview.get("runtime_dispatch_ready_optimizer_count") == 124, report
    assert signed_preview.get("native_dispatch_allowed_family_count") == 10, report
    assert signed_preview.get("native_dispatch_allowed_optimizer_count") == 124, report
    assert signed_preview.get("training_path_enabled_family_count") == 10, report
    assert signed_preview.get("training_path_enabled_optimizer_count") == 124, report
    assert signed_preview.get("product_native_ready_family_count") == 10, report
    assert signed_preview.get("product_native_ready_optimizer_count") == 124, report
    assert summary.get("signed_post_approval_preview_ready_count") == 1, report
    assert summary.get("signed_post_approval_preview_runtime_dispatch_ready_optimizer_count") == 124, report
    assert summary.get("signed_post_approval_preview_native_dispatch_allowed_optimizer_count") == 124, report
    assert summary.get("signed_post_approval_preview_training_path_enabled_optimizer_count") == 124, report
    assert summary.get("signed_post_approval_preview_product_native_ready_optimizer_count") == 124, report
    assert summary.get("family_specific_runtime_launch_missing_count") == 0, report
    assert summary.get("product_training_route_missing_count") == 10, report
    assert summary.get("owner_release_approval_missing_count") == 10, report
    assert signed_report.get("post_approval_ready") is True, signed_report
    assert signed_post_approval.get("route_binding_preflight_ready") is True, signed_report
    assert signed_post_approval.get("candidate_ready") is True, signed_report
    assert signed_post_approval.get("runtime_dispatch_ready_family_count") == 10, signed_report
    assert signed_post_approval.get("runtime_dispatch_ready_optimizer_count") == 124, signed_report
    assert signed_post_approval.get("native_dispatch_allowed_family_count") == 10, signed_report
    assert signed_post_approval.get("native_dispatch_allowed_optimizer_count") == 124, signed_report
    assert signed_post_approval.get("training_path_enabled_family_count") == 10, signed_report
    assert signed_post_approval.get("training_path_enabled_optimizer_count") == 124, signed_report
    assert signed_post_approval.get("product_native_ready_family_count") == 10, signed_report
    assert signed_post_approval.get("product_native_ready_optimizer_count") == 124, signed_report
    assert signed_summary.get("runtime_dispatch_ready_family_count") == 0, signed_report
    assert signed_summary.get("native_dispatch_allowed_family_count") == 0, signed_report
    assert signed_summary.get("product_native_ready_family_count") == 0, signed_report
    assert signed_summary.get("roadmap_v2_open_work_category_count") == 5, signed_report
    assert signed_summary.get("roadmap_v2_open_work_item_count") == 30, signed_report
    assert signed_summary.get("roadmap_v2_open_work_open_item_count") == 6, signed_report
    for row in rows:
        assert isinstance(row, dict), row
        assert row.get("family_evidence_ready") is True, row
        assert row.get("runtime_rehearsal_mode") in {"dispatch", "precondition"}, row
        assert row.get("runtime_launch_coverage_mode") in {"dispatch", "precondition_adapter"}, row
        assert row.get("runtime_launch_coverage_ready_count") == row.get("optimizer_count"), row
        if row.get("runtime_rehearsal_mode") == "precondition":
            assert row.get("family_specific_runtime_launch_adapter_ready") is True, row
        assert row.get("owner_release_hold_ready") is True, row
        assert row.get("request_schema_ui_non_exposure_ready") is True, row
        assert row.get("runtime_dispatch_ready") is False, row
        assert row.get("native_dispatch_allowed") is False, row
        assert row.get("training_path_enabled") is False, row
        assert row.get("product_native_ready") is False, row

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_native_readiness_gap_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "artifact_mode": "artifact_first",
        "summary": summary,
        "default_off_product_summary": default_off_product,
        "roadmap_v2_open_work": report.get("roadmap_v2_open_work"),
        "post_approval_summary": signed_post_approval,
        "signed_post_approval_preview_summary": signed_preview,
        "recommended_next_step": report.get("recommended_next_step", ""),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _signed_route_binding_preflight() -> dict[str, Any]:
    artifact_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    return build_optimizer_product_training_route_binding_preflight(
        native_readiness_gap=_read_json(artifact_dir / "turbocore_optimizer_native_readiness_gap_scorecard.json"),
        owner_release_review_record={
            "schema_version": 1,
            "gate": "native_update_owner_release_review_record",
            "ok": True,
            "approval_recorded": True,
            "release_review_recorded": True,
            "product_exposure_allowed": False,
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ui_exposure_allowed": False,
            "runtime_dispatch_allowed": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "training_launch_executed": False,
        },
        owner_release_direction_record={
            "schema_version": 1,
            "gate": "native_update_owner_release_direction_record",
            "ok": True,
            "owner_release_direction_recorded": True,
            "owner_release_approval_recorded": True,
            "product_exposure_allowed": False,
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ui_exposure_allowed": False,
            "runtime_dispatch_allowed": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "training_launch_executed": False,
        },
        product_exposure_decision={
            "schema_version": 1,
            "gate": "native_update_product_exposure_decision",
            "ok": True,
            "evidence_ready": True,
            "ready_for_product_exposure_review": True,
            "product_exposure_decision_recorded": True,
            "post_product_exposure_request_fields": {},
            "product_exposure_allowed": False,
            "training_launch_allowed": False,
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ready_for_ui": False,
            "backend_router_registered": False,
        },
        release_review_package={
            "schema_version": 1,
            "gate": "native_update_release_review_package",
            "ok": True,
            "evidence_ready": True,
            "ready_for_review": True,
            "ready_for_owner_release_review": True,
            "release_review_recorded": True,
            "default_off": True,
            "post_release_request_fields": {},
            "release_gate_open": False,
            "training_launch_allowed": False,
            "runtime_dispatch_allowed": False,
            "native_dispatch_allowed": False,
            "training_path_enabled": False,
            "request_fields_emitted": False,
            "schema_exposure_allowed": False,
            "ui_exposure_allowed": False,
            "backend_router_registered": False,
        },
        write_artifact=False,
    )


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
