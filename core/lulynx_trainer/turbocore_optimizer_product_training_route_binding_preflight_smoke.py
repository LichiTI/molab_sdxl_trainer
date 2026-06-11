"""Smoke for TurboCore optimizer product training-route binding preflight."""

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

from core.turbocore_optimizer_product_training_route_binding_preflight import (  # noqa: E402
    build_optimizer_product_training_route_binding_preflight,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def run_smoke() -> dict[str, Any]:
    artifact_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    current = build_optimizer_product_training_route_binding_preflight(
        artifact_dir=artifact_dir,
        write_artifact=True,
    )
    summary = _as_dict(current.get("summary"))
    assert current["ok"] is True, current
    assert current["roadmap"] == ROADMAP, current
    assert current["artifact_first"] is True, current
    assert current["product_training_route_binding_preflight_ready"] is False, current
    assert current["product_training_route_bound"] is False, current
    assert "owner_release_approval_missing" in current["blocked_reasons"], current
    assert "product_exposure_decision_not_recorded" in current["blocked_reasons"], current
    assert summary["route_family_count"] == 10, current
    assert summary["plugin_optimizer_count"] == 124, current
    assert summary["runtime_launch_coverage_ready_family_count"] == 10, current
    assert summary["family_specific_runtime_launch_adapter_ready_optimizer_count"] == 72, current
    assert summary["owner_release_hold_ready_family_count"] == 10, current
    assert summary["request_schema_ui_non_exposure_ready_family_count"] == 10, current
    assert summary["product_training_route_binding_ready_count"] == 0, current
    assert summary["post_approval_training_route_binding_candidate_count"] == 0, current
    assert current["post_approval_training_route_binding_candidate"] == {}, current
    _assert_default_off(current)

    signed_candidate = build_optimizer_product_training_route_binding_preflight(
        native_readiness_gap=_read_json(artifact_dir / "turbocore_optimizer_native_readiness_gap_scorecard.json"),
        owner_release_review_record=_signed_owner_record(),
        product_exposure_decision=_signed_product_exposure_decision(),
        release_review_package=_signed_release_package(),
        write_artifact=False,
    )
    signed_summary = _as_dict(signed_candidate.get("summary"))
    assert signed_candidate["ok"] is True, signed_candidate
    assert signed_candidate["product_training_route_binding_preflight_ready"] is True, signed_candidate
    assert signed_candidate["product_training_route_bound"] is False, signed_candidate
    assert signed_candidate["blocked_reasons"] == [], signed_candidate
    assert "training_path_dispatch_not_enabled" in signed_candidate["promotion_blockers"], signed_candidate
    assert signed_summary["owner_release_approval_recorded_count"] == 1, signed_candidate
    assert signed_summary["release_review_recorded_count"] == 1, signed_candidate
    assert signed_summary["product_exposure_decision_recorded_count"] == 1, signed_candidate
    assert signed_summary["product_training_route_binding_ready_count"] == 10, signed_candidate
    assert signed_summary["post_approval_training_route_binding_candidate_count"] == 1, signed_candidate
    _assert_candidate_contract(signed_candidate["post_approval_training_route_binding_candidate"])
    _assert_default_off(signed_candidate)

    unsafe = build_optimizer_product_training_route_binding_preflight(
        native_readiness_gap={**_read_json(artifact_dir / "turbocore_optimizer_native_readiness_gap_scorecard.json")},
        owner_release_review_record={**_signed_owner_record(), "training_path_enabled": True},
        product_exposure_decision=_signed_product_exposure_decision(),
        release_review_package=_signed_release_package(),
        write_artifact=False,
    )
    assert unsafe["ok"] is False, unsafe
    assert unsafe["product_training_route_binding_preflight_ready"] is False, unsafe
    assert any("training_path_enabled" in item for item in unsafe["blocked_reasons"]), unsafe

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_product_training_route_binding_preflight_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "current_preflight_ready": current["product_training_route_binding_preflight_ready"],
        "synthetic_signed_candidate_ready": signed_candidate["product_training_route_binding_preflight_ready"],
        "unsafe_training_path_claim_blocked": True,
        "summary": summary,
        "recommended_next_step": current["recommended_next_step"],
    }


def _signed_owner_record() -> dict[str, Any]:
    return {
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
    }


def _signed_product_exposure_decision() -> dict[str, Any]:
    return {
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
        "blocked_reasons": [],
    }


def _signed_release_package() -> dict[str, Any]:
    return {
        "schema_version": 1,
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
        "blocked_reasons": [],
    }


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in (
        "product_training_route_bound",
        "runtime_dispatch_ready",
        "native_dispatch_allowed",
        "training_path_enabled",
        "training_launch_executed",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "backend_router_registered",
    ):
        assert report[field] is False, (field, report)
    assert report["post_training_route_request_fields"] == {}, report


def _assert_candidate_contract(candidate: dict[str, Any]) -> None:
    assert candidate["candidate"] == "post_approval_optimizer_training_route_binding_v0", candidate
    assert candidate["existing_training_loop_switches"] == {
        "turbocore_native_update_dispatch_enabled": True,
        "turbocore_native_update_training_path_enabled": True,
        "turbocore_native_update_require_native_cuda": True,
    }, candidate
    surface = candidate["request_ui_schema_contract"]
    assert surface["request_fields_emitted"] is False, candidate
    assert surface["schema_exposure_allowed"] is False, candidate
    assert surface["ui_exposure_allowed"] is False, candidate
    assert surface["backend_router_registered"] is False, candidate
    assert surface["post_training_route_request_fields"] == {}, candidate
    assert candidate["optimizer_scope"]["route_family_count"] == 10, candidate
    assert candidate["optimizer_scope"]["plugin_optimizer_count"] == 124, candidate


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
