"""Smoke for product trainer route adapter coverage of TurboCore optimizer binding."""

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

from core.turbocore_optimizer_product_training_route_binding_product_route_adapter import (  # noqa: E402
    build_optimizer_product_training_route_binding_product_route_adapter,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def run_smoke() -> dict[str, Any]:
    report = build_optimizer_product_training_route_binding_product_route_adapter(write_artifact=True)
    summary = report["summary"]
    assert report["ok"] is True, report
    assert report["roadmap"] == ROADMAP, report
    assert report["product_training_route_binding_kwargs_wired"] is True, report
    assert report["product_training_route_bound"] is False, report
    assert report["training_path_enabled"] is False, report
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["post_training_route_request_fields"] == {}, report
    assert summary["product_training_route_count"] == 4, report
    assert summary["product_training_route_binding_kwargs_wired_count"] == 4, report
    assert summary["training_loop_constructor_switch_field_count"] == 4, report
    assert summary["unified_config_switch_field_count"] == 4, report
    assert summary["owner_release_direction_recorded_count"] == 0, report
    assert summary["owner_release_direction_approval_recorded_count"] == 0, report
    assert summary["product_training_route_binding_config_patch_ready_count"] == 0, report
    assert summary["training_path_enabled_count"] == 0, report
    assert report["default_training_loop_kwargs_patch"]["turbocore_native_update_mode"] == "off", report
    default_patch = report["default_training_loop_kwargs_patch"]
    signed_patch = report["synthetic_signed_training_loop_kwargs_patch"]
    assert default_patch["turbocore_native_update_training_path_enabled"] is False, report
    assert signed_patch["turbocore_native_update_mode"] == "native_experimental", report
    assert signed_patch["turbocore_native_update_training_path_enabled"] is True, report
    signed = build_optimizer_product_training_route_binding_product_route_adapter(
        config_adapter_report={
            "summary": {
                "owner_release_direction_recorded_count": 1,
                "owner_release_direction_approval_recorded_count": 1,
                "product_training_route_binding_config_patch_ready_count": 1,
            }
        },
        write_artifact=False,
    )
    signed_summary = signed["summary"]
    assert signed["ok"] is True, signed
    assert signed["config_adapter_source"] == "supplied", signed
    assert signed_summary["owner_release_direction_recorded_count"] == 1, signed
    assert signed_summary["owner_release_direction_approval_recorded_count"] == 1, signed
    assert signed_summary["product_training_route_binding_config_patch_ready_count"] == 1, signed
    assert signed["product_training_route_bound"] is False, signed
    assert signed["training_path_enabled"] is False, signed
    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_product_training_route_binding_product_route_adapter_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": summary,
        "recommended_next_step": report["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
