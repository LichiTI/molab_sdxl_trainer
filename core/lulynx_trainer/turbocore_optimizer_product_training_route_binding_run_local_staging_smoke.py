"""Smoke for run-local TurboCore optimizer route-binding artifact staging."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_product_training_route_binding_run_local_staging import (  # noqa: E402
    RUN_LOCAL_ADAPTER_NAME,
    build_optimizer_product_training_route_binding_run_local_staging,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir)
        current = build_optimizer_product_training_route_binding_run_local_staging(
            run_dir=run_dir,
            write_artifact=True,
        )
        assert current["ok"] is True, current
        assert current["roadmap"] == ROADMAP, current
        assert current["run_local_adapter_staged"] is False, current
        assert current["run_local_adapter_path"] == "", current
        assert not (run_dir / RUN_LOCAL_ADAPTER_NAME).exists(), current
        assert current["summary"]["owner_release_direction_recorded_count"] == 0, current
        assert current["summary"]["owner_release_direction_approval_recorded_count"] == 0, current
        assert current["summary"]["run_local_adapter_staged_count"] == 0, current
        assert "route_binding_config_patch_not_ready" in current["blocked_reasons"], current
        _assert_default_off(current)

        readonly = build_optimizer_product_training_route_binding_run_local_staging(
            run_dir=run_dir / "readonly",
            artifact_dir=run_dir / "empty_artifacts",
            refresh_config_adapter_artifact=False,
            write_artifact=False,
        )
        assert readonly["run_local_adapter_staged"] is False, readonly
        assert "route_binding_config_adapter_artifact_missing" in readonly["blocked_reasons"], readonly
        assert (run_dir / "readonly" / "turbocore_optimizer_route_binding_staging.json").exists(), readonly
        assert not (run_dir / "readonly" / RUN_LOCAL_ADAPTER_NAME).exists(), readonly
        _assert_default_off(readonly)

        signed = build_optimizer_product_training_route_binding_run_local_staging(
            run_dir=run_dir,
            config_adapter_report=_signed_adapter_report(),
            write_artifact=False,
        )
        staged_path = run_dir / RUN_LOCAL_ADAPTER_NAME
        assert signed["run_local_adapter_staged"] is True, signed
        assert signed["summary"]["owner_release_direction_recorded_count"] == 1, signed
        assert signed["summary"]["owner_release_direction_approval_recorded_count"] == 1, signed
        assert signed["summary"]["run_local_adapter_staged_count"] == 1, signed
        assert staged_path.exists(), signed
        staged = json.loads(staged_path.read_text(encoding="utf-8"))
        assert staged["product_training_route_binding_config_patch_ready"] is True, staged
        assert staged["training_loop_kwargs_patch"]["turbocore_native_update_training_path_enabled"] is True, staged
        _assert_default_off(signed)

        unsafe = build_optimizer_product_training_route_binding_run_local_staging(
            run_dir=run_dir / "unsafe",
            config_adapter_report={**_signed_adapter_report(), "request_fields_emitted": True},
            write_artifact=False,
        )
        assert unsafe["run_local_adapter_staged"] is False, unsafe
        assert "route_binding_request_fields_not_closed" in unsafe["blocked_reasons"], unsafe

    _assert_product_launch_staging_wired()
    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_product_training_route_binding_run_local_staging_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "summary": current["summary"],
        "product_launch_staging_wired_count": 2,
        "synthetic_signed_run_local_adapter_staged": True,
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


def _assert_product_launch_staging_wired() -> None:
    targets = (
        REPO_ROOT / "backend" / "routers" / "training.py",
        REPO_ROOT / "backend" / "core" / "services" / "training_queue_support.py",
    )
    for target in targets:
        text = target.read_text(encoding="utf-8")
        assert "build_optimizer_product_training_route_binding_run_local_staging(" in text, target
        call_start = text.index("build_optimizer_product_training_route_binding_run_local_staging(")
        call = text[call_start: text.index(")", call_start)]
        assert "run_dir=run_dir" in call, target
        assert "refresh_config_adapter_artifact=False" in call, target
        assert "write_artifact=False" in call, target


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
