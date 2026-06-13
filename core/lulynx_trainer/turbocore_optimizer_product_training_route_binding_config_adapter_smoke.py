"""Smoke for TurboCore optimizer product route-binding config adapter."""

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

from core.turbocore_optimizer_product_training_route_binding_config_adapter import (  # noqa: E402
    build_optimizer_product_training_route_binding_config_adapter,
)
from core.turbocore_optimizer_product_training_route_binding_preflight import (  # noqa: E402
    build_optimizer_product_training_route_binding_preflight,
)
from core.turbocore_optimizer_product_training_route_binding_training_loop_contract import (  # noqa: E402
    build_optimizer_product_training_route_binding_training_loop_contract,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def run_smoke() -> dict[str, Any]:
    artifact_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    current = build_optimizer_product_training_route_binding_config_adapter(
        artifact_dir=artifact_dir,
        write_artifact=True,
    )
    current_summary = _as_dict(current.get("summary"))
    assert current["ok"] is True, current
    assert current["roadmap"] == ROADMAP, current
    assert current["product_training_route_binding_config_patch_ready"] is False, current
    assert current["training_loop_kwargs_patch"] == {}, current
    assert current_summary["training_loop_constructor_switch_field_count"] == 3, current
    assert current_summary["training_loop_constructor_mode_field_present_count"] == 1, current
    assert current_summary["owner_release_direction_recorded_count"] == 0, current
    assert current_summary["owner_release_direction_approval_recorded_count"] == 0, current
    assert current_summary["product_training_route_binding_config_patch_ready_count"] == 0, current
    assert "product_training_route_binding_preflight_not_ready" in current["blocked_reasons"], current
    _assert_default_off(current)

    signed_preflight = _signed_preflight(artifact_dir)
    signed_contract = build_optimizer_product_training_route_binding_training_loop_contract(
        synthetic_signed_preflight_report=signed_preflight,
        write_artifact=False,
    )
    signed = build_optimizer_product_training_route_binding_config_adapter(
        preflight_report=signed_preflight,
        training_loop_contract=signed_contract,
        write_artifact=False,
    )
    signed_summary = _as_dict(signed.get("summary"))
    assert signed["ok"] is True, signed
    assert signed["product_training_route_binding_config_patch_ready"] is True, signed
    assert signed["blocked_reasons"] == [], signed
    assert signed_summary["product_training_route_binding_config_patch_ready_count"] == 1, signed
    assert signed_summary["owner_release_direction_recorded_count"] == 1, signed
    assert signed_summary["owner_release_direction_approval_recorded_count"] == 1, signed
    assert signed_summary["training_loop_kwargs_patch_field_count"] == 4, signed
    patch = signed["training_loop_kwargs_patch"]
    assert patch["turbocore_native_update_mode"] == "native_experimental", signed
    assert patch["turbocore_native_update_dispatch_enabled"] is True, signed
    assert patch["turbocore_native_update_training_path_enabled"] is True, signed
    assert patch["turbocore_native_update_require_native_cuda"] is True, signed
    _assert_default_off(signed)

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_product_training_route_binding_config_adapter_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": current_summary,
        "synthetic_signed_kwargs_patch_ready": True,
        "recommended_next_step": current["recommended_next_step"],
    }


def _signed_preflight(artifact_dir: Path) -> dict[str, Any]:
    return build_optimizer_product_training_route_binding_preflight(
        native_readiness_gap=_read_json(artifact_dir / "turbocore_optimizer_native_readiness_gap_scorecard.json"),
        owner_release_review_record={"ok": True, "approval_recorded": True, "release_review_recorded": True},
        owner_release_direction_packet={
            "ok": True,
            "owner_release_direction_recorded": True,
            "owner_release_approval_recorded": True,
        },
        product_exposure_decision={
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


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in (
        "product_training_route_bound",
        "training_path_enabled",
        "request_fields_emitted",
        "schema_exposure_allowed",
        "ui_exposure_allowed",
        "backend_router_registered",
    ):
        assert report[field] is False, (field, report)
    assert report["post_training_route_request_fields"] == {}, report


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
