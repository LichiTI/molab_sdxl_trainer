"""Smoke for TurboCore optimizer route-binding runtime config applier."""

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

from core.turbocore_optimizer_product_training_route_binding_runtime_applier import (  # noqa: E402
    apply_optimizer_product_training_route_binding_runtime_patch,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def run_smoke() -> dict[str, Any]:
    current_config: dict[str, Any] = {"optimizer_type": "AdamW"}
    current = apply_optimizer_product_training_route_binding_runtime_patch(
        current_config,
        write_artifact=True,
    )
    assert current["ok"] is True, current
    assert current["roadmap"] == ROADMAP, current
    assert current["runtime_config_patch_applied"] is False, current
    assert current["product_training_route_bound"] is False, current
    assert current["training_path_enabled"] is False, current
    assert current["summary"]["owner_release_direction_recorded_count"] == 0, current
    assert current["summary"]["owner_release_direction_approval_recorded_count"] == 0, current
    assert current["training_loop_kwargs_patch"] == {}, current
    assert current_config == {"optimizer_type": "AdamW"}, current
    assert "route_binding_config_patch_not_ready" in current["blocked_reasons"], current
    _assert_closed(current)

    run_local_dir = REPO_ROOT / "temp" / "turbocore_optimizer" / "runtime_applier_missing_adapter_smoke"
    run_local = apply_optimizer_product_training_route_binding_runtime_patch(
        {"optimizer_type": "AdamW"},
        artifact_dir=run_local_dir,
        refresh_config_adapter_artifact=False,
        write_artifact=False,
    )
    assert run_local["runtime_config_patch_applied"] is False, run_local
    assert run_local["config_adapter_source"] == "missing_existing_artifact", run_local
    assert run_local["config_adapter_artifact_refreshed"] is False, run_local

    signed_config: dict[str, Any] = {}
    signed = apply_optimizer_product_training_route_binding_runtime_patch(
        signed_config,
        config_adapter_report=_signed_adapter_report(),
        write_artifact=False,
    )
    assert signed["ok"] is True, signed
    assert signed["runtime_config_patch_applied"] is True, signed
    assert signed["summary"]["owner_release_direction_recorded_count"] == 1, signed
    assert signed["summary"]["owner_release_direction_approval_recorded_count"] == 1, signed
    assert signed["summary"]["runtime_config_patch_applied_count"] == 1, signed
    assert signed_config["turbocore_native_update_mode"] == "native_experimental", signed
    assert signed_config["turbocore_native_update_dispatch_enabled"] is True, signed
    assert signed_config["turbocore_native_update_training_path_enabled"] is True, signed
    assert signed_config["turbocore_native_update_require_native_cuda"] is True, signed
    _assert_closed(signed, allow_training_path=True)

    unsafe_config: dict[str, Any] = {}
    unsafe = apply_optimizer_product_training_route_binding_runtime_patch(
        unsafe_config,
        config_adapter_report={**_signed_adapter_report(), "request_fields_emitted": True},
        write_artifact=False,
    )
    assert unsafe["runtime_config_patch_applied"] is False, unsafe
    assert unsafe_config == {}, unsafe
    assert "route_binding_adapter_request_fields_not_closed" in unsafe["blocked_reasons"], unsafe
    _assert_entry_train_applies_before_config_adapter()

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_product_training_route_binding_runtime_applier_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": current["summary"],
        "entry_train_runtime_applier_wired": True,
        "synthetic_signed_runtime_patch_applied": True,
        "recommended_next_step": current["recommended_next_step"],
    }


def _signed_adapter_report() -> dict[str, Any]:
    return {
        "product_training_route_binding_config_patch_ready": True,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "post_training_route_request_fields": {},
        "training_loop_kwargs_patch": {
            "turbocore_native_update_mode": "native_experimental",
            "turbocore_native_update_dispatch_enabled": True,
            "turbocore_native_update_training_path_enabled": True,
            "turbocore_native_update_require_native_cuda": True,
        },
        "summary": {
            "owner_release_direction_recorded_count": 1,
            "owner_release_direction_approval_recorded_count": 1,
        },
    }


def _assert_closed(report: dict[str, Any], *, allow_training_path: bool = False) -> None:
    assert report["request_fields_emitted"] is False, report
    assert report["schema_exposure_allowed"] is False, report
    assert report["ui_exposure_allowed"] is False, report
    assert report["backend_router_registered"] is False, report
    assert report["post_training_route_request_fields"] == {}, report
    if not allow_training_path:
        assert report["training_path_enabled"] is False, report


def _assert_entry_train_applies_before_config_adapter() -> None:
    text = (REPO_ROOT / "backend" / "core" / "entry_train.py").read_text(encoding="utf-8")
    apply_index = text.index("apply_optimizer_product_training_route_binding_runtime_patch(")
    adapter_index = text.index("ConfigAdapter.from_frontend_dict(config_dict)")
    assert apply_index < adapter_index, "runtime applier must run before ConfigAdapter conversion"
    call = text[apply_index:adapter_index]
    assert "artifact_dir=run_dir" in call, "entry_train runtime applier must use run-local artifacts"
    assert "refresh_config_adapter_artifact=False" in call, "entry_train must not refresh global adapter artifacts"


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
