"""Run-local staging contract for TurboCore optimizer route-binding artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_optimizer_product_training_route_binding_config_adapter import (
    ARTIFACT as CONFIG_ADAPTER_ARTIFACT,
    build_optimizer_product_training_route_binding_config_adapter,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_run_local_staging.json"
RUN_LOCAL_ADAPTER_NAME = "turbocore_optimizer_product_training_route_binding_config_adapter.json"
RUN_LOCAL_STAGING_REPORT_NAME = "turbocore_optimizer_route_binding_staging.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
PRODUCT_LAUNCH_STAGING_TARGETS = (
    REPO_ROOT / "backend" / "routers" / "training.py",
    REPO_ROOT / "backend" / "core" / "services" / "training_queue_support.py",
)


def build_optimizer_product_training_route_binding_run_local_staging(
    *,
    run_dir: str | Path | None = None,
    config_adapter_report: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
    write_run_local_adapter: bool = True,
    write_run_local_report: bool = True,
    refresh_config_adapter_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    run_directory = Path(run_dir) if run_dir is not None else directory / "run_local_route_binding_staging"
    adapter = _config_adapter_report(
        config_adapter_report=config_adapter_report,
        artifact_dir=directory,
        refresh_config_adapter_artifact=refresh_config_adapter_artifact,
    )
    ready = _adapter_ready(adapter)
    staged = bool(ready and write_run_local_adapter)
    blockers = [] if ready else _blockers(adapter)
    adapter_summary = _as_dict(adapter.get("summary"))
    payload = {
        "schema_version": 1,
        "staging": "turbocore_optimizer_product_training_route_binding_run_local_staging_v0",
        "gate": "optimizer_product_training_route_binding_run_local_staging",
        "ok": True,
        "roadmap": ROADMAP,
        "artifact_first": True,
        "run_local_adapter_staged": staged,
        "run_local_adapter_path": str(run_directory / RUN_LOCAL_ADAPTER_NAME) if staged else "",
        "product_training_route_bound": False,
        "training_path_enabled": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "post_training_route_request_fields": {},
        "summary": {
            "owner_release_direction_recorded_count": int(
                adapter_summary.get("owner_release_direction_recorded_count", 0) or 0
            ),
            "owner_release_direction_approval_recorded_count": int(
                adapter_summary.get("owner_release_direction_approval_recorded_count", 0) or 0
            ),
            "run_local_adapter_staged_count": 1 if staged else 0,
            "runtime_config_patch_applied_count": 0,
            "training_path_enabled_count": 0,
            "request_fields_emitted_count": 0,
            "schema_exposure_allowed_count": 0,
            "ui_exposure_allowed_count": 0,
            "product_launch_staging_wired_count": product_launch_staging_wired_count(),
        },
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run-local route-binding adapter staged for entry_train worker consumption"
            if staged
            else "do not stage run-local route-binding adapter until signed owner/product decisions are recorded"
        ),
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if staged:
        run_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / RUN_LOCAL_ADAPTER_NAME).write_text(
            json.dumps(adapter, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if write_run_local_report or staged:
        run_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / RUN_LOCAL_STAGING_REPORT_NAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _config_adapter_report(
    *,
    config_adapter_report: Mapping[str, Any] | None,
    artifact_dir: Path,
    refresh_config_adapter_artifact: bool,
) -> dict[str, Any]:
    if isinstance(config_adapter_report, Mapping):
        return dict(config_adapter_report)
    if refresh_config_adapter_artifact:
        return build_optimizer_product_training_route_binding_config_adapter(
            artifact_dir=artifact_dir,
            write_artifact=True,
        )
    source = artifact_dir / CONFIG_ADAPTER_ARTIFACT.name
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def product_launch_staging_wired_count() -> int:
    return sum(1 for target in PRODUCT_LAUNCH_STAGING_TARGETS if _has_readonly_staging_call(target))


def _has_readonly_staging_call(target: Path) -> bool:
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return False
    marker = "build_optimizer_product_training_route_binding_run_local_staging("
    if marker not in text:
        return False
    call_start = text.index(marker)
    try:
        call = text[call_start: text.index(")", call_start)]
    except ValueError:
        return False
    return all(
        required in call
        for required in ("run_dir=run_dir", "refresh_config_adapter_artifact=False", "write_artifact=False")
    )


def _adapter_ready(adapter: Mapping[str, Any]) -> bool:
    return bool(
        adapter.get("product_training_route_binding_config_patch_ready") is True
        and isinstance(adapter.get("training_loop_kwargs_patch"), Mapping)
        and adapter.get("request_fields_emitted") is False
        and adapter.get("schema_exposure_allowed") is False
        and adapter.get("ui_exposure_allowed") is False
        and adapter.get("post_training_route_request_fields") == {}
    )


def _blockers(adapter: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not adapter:
        blockers.append("route_binding_config_adapter_artifact_missing")
        return blockers
    if adapter.get("product_training_route_binding_config_patch_ready") is not True:
        blockers.append("route_binding_config_patch_not_ready")
    if not isinstance(adapter.get("training_loop_kwargs_patch"), Mapping):
        blockers.append("route_binding_config_patch_missing")
    if adapter.get("request_fields_emitted") is True or adapter.get("post_training_route_request_fields"):
        blockers.append("route_binding_request_fields_not_closed")
    if adapter.get("schema_exposure_allowed") is True or adapter.get("ui_exposure_allowed") is True:
        blockers.append("route_binding_product_exposure_not_closed")
    return blockers


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "RUN_LOCAL_ADAPTER_NAME",
    "RUN_LOCAL_STAGING_REPORT_NAME",
    "build_optimizer_product_training_route_binding_run_local_staging",
    "product_launch_staging_wired_count",
]
