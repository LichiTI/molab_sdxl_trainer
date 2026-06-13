"""Runtime config applier for post-approval TurboCore optimizer route binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from core.turbocore_optimizer_product_training_route_binding_config_adapter import (
    ROUTE_BINDING_MODE_FIELD,
    ROUTE_BINDING_SWITCHES,
    build_optimizer_product_training_route_binding_config_adapter,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_runtime_applier.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
PATCH_FIELDS = (ROUTE_BINDING_MODE_FIELD, *ROUTE_BINDING_SWITCHES)


def apply_optimizer_product_training_route_binding_runtime_patch(
    config: MutableMapping[str, Any],
    *,
    config_adapter_report: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    refresh_config_adapter_artifact: bool = True,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    if isinstance(config_adapter_report, Mapping):
        adapter = dict(config_adapter_report)
        adapter_source = "supplied"
    elif refresh_config_adapter_artifact:
        adapter = build_optimizer_product_training_route_binding_config_adapter(
            artifact_dir=directory,
            write_artifact=True,
        )
        adapter_source = "refreshed"
    else:
        adapter = _read_adapter_artifact(directory)
        adapter_source = "existing_artifact" if adapter else "missing_existing_artifact"
    before = {field: config.get(field) for field in PATCH_FIELDS if field in config}
    patch = _patch(adapter)
    blockers = _blockers(adapter, patch)
    applied = bool(patch and not blockers)
    if applied:
        config.update(patch)
    after = {field: config.get(field) for field in PATCH_FIELDS if field in config}
    adapter_summary = _as_dict(adapter.get("summary"))
    payload = {
        "schema_version": 1,
        "applier": "turbocore_optimizer_product_training_route_binding_runtime_applier_v0",
        "gate": "optimizer_product_training_route_binding_runtime_applier",
        "ok": bool(not blockers or not patch),
        "roadmap": ROADMAP,
        "artifact_first": True,
        "config_adapter_source": adapter_source,
        "config_adapter_artifact_refreshed": bool(adapter_source == "refreshed"),
        "runtime_config_patch_applied": applied,
        "product_training_route_bound": applied,
        "training_path_enabled": bool(applied and config.get("turbocore_native_update_training_path_enabled") is True),
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "post_training_route_request_fields": {},
        "training_loop_kwargs_patch": patch if applied else {},
        "config_fields_before": before,
        "config_fields_after": after,
        "summary": {
            "owner_release_direction_recorded_count": int(
                adapter_summary.get("owner_release_direction_recorded_count", 0) or 0
            ),
            "owner_release_direction_approval_recorded_count": int(
                adapter_summary.get("owner_release_direction_approval_recorded_count", 0) or 0
            ),
            "runtime_config_patch_applied_count": 1 if applied else 0,
            "runtime_config_patch_field_count": len(patch) if applied else 0,
            "training_path_enabled_count": 1 if applied else 0,
            "request_fields_emitted_count": 0,
            "schema_exposure_allowed_count": 0,
            "ui_exposure_allowed_count": 0,
        },
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "native optimizer route-binding patch applied to worker config"
            if applied
            else "worker config remains unchanged until signed owner/product decisions are recorded"
        ),
    }
    if write_artifact:
        target = Path(report_path) if report_path is not None else directory / ARTIFACT.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _patch(adapter: Mapping[str, Any]) -> dict[str, Any]:
    value = adapter.get("training_loop_kwargs_patch")
    if not isinstance(value, Mapping):
        return {}
    return {field: value[field] for field in PATCH_FIELDS if field in value}


def _read_adapter_artifact(directory: Path) -> dict[str, Any]:
    source = directory / "turbocore_optimizer_product_training_route_binding_config_adapter.json"
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _blockers(adapter: Mapping[str, Any], patch: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if adapter.get("product_training_route_binding_config_patch_ready") is not True:
        blockers.append("route_binding_config_patch_not_ready")
    if not patch:
        blockers.append("runtime_config_patch_empty")
    if set(patch) != set(PATCH_FIELDS):
        blockers.append("runtime_config_patch_field_set_mismatch")
    if patch.get(ROUTE_BINDING_MODE_FIELD) != "native_experimental":
        blockers.append("runtime_config_patch_mode_not_native_experimental")
    for field in ROUTE_BINDING_SWITCHES:
        if patch.get(field) is not True:
            blockers.append(f"runtime_config_patch_switch_not_true:{field}")
    if adapter.get("request_fields_emitted") is True or adapter.get("post_training_route_request_fields"):
        blockers.append("route_binding_adapter_request_fields_not_closed")
    if adapter.get("schema_exposure_allowed") is True or adapter.get("ui_exposure_allowed") is True:
        blockers.append("route_binding_adapter_product_exposure_not_closed")
    return _dedupe(blockers)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["apply_optimizer_product_training_route_binding_runtime_patch"]
