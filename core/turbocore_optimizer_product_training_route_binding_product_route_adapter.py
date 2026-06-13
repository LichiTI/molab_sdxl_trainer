"""Product trainer route adapter coverage for TurboCore optimizer route binding."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from core.configs import UnifiedTrainingConfig
from core.lulynx_trainer.training_loop import TrainingLoop
from core.lulynx_trainer.turbocore_native_update_route_binding import (
    ROUTE_BINDING_KWARGS,
    build_turbocore_native_update_training_loop_kwargs,
)
from core.turbocore_optimizer_product_training_route_binding_config_adapter import (
    build_optimizer_product_training_route_binding_config_adapter,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_product_route_adapter.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"

ROUTES = {
    "lulynx_lora": ("backend/core/lulynx_trainer/trainer.py", "TrainingLoop", "explicit_kwargs"),
    "controlnet": ("backend/core/lulynx_trainer/controlnet_trainer.py", "ControlNetTrainingLoop", "helper_kwargs"),
    "ip_adapter": ("backend/core/lulynx_trainer/ip_adapter_trainer.py", "IPAdapterTrainingLoop", "helper_kwargs"),
    "lllite": ("backend/core/lulynx_trainer/lllite_trainer.py", "LLLiteTrainingLoop", "helper_kwargs"),
}


def build_optimizer_product_training_route_binding_product_route_adapter(
    *,
    config_adapter_report: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    refresh_config_adapter_artifact: bool = True,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    adapter, adapter_source = _config_adapter_report(
        config_adapter_report=config_adapter_report,
        directory=directory,
        refresh_config_adapter_artifact=refresh_config_adapter_artifact,
    )
    adapter_summary = _as_dict(adapter.get("summary"))
    signature_fields = set(inspect.signature(TrainingLoop.__init__).parameters)
    config_fields = set(getattr(UnifiedTrainingConfig, "model_fields", {}))
    route_reports = [
        _route_report(route_id, path, class_name, mode)
        for route_id, (path, class_name, mode) in ROUTES.items()
    ]
    default_patch = build_turbocore_native_update_training_loop_kwargs(SimpleNamespace())
    signed_patch = build_turbocore_native_update_training_loop_kwargs(
        SimpleNamespace(
            turbocore_native_update_mode="native_experimental",
            turbocore_native_update_dispatch_enabled=True,
            turbocore_native_update_training_path_enabled=True,
            turbocore_native_update_require_native_cuda=True,
        )
    )
    ready_routes = [item for item in route_reports if item["route_binding_kwargs_wired"] is True]
    payload = {
        "schema_version": 1,
        "adapter": "turbocore_optimizer_product_training_route_binding_product_route_adapter_v0",
        "gate": "optimizer_product_training_route_binding_product_route_adapter",
        "ok": bool(len(ready_routes) == len(ROUTES) and _all_present(signature_fields) and _all_present(config_fields)),
        "roadmap": ROADMAP,
        "artifact_first": True,
        "config_adapter_source": adapter_source,
        "config_adapter_artifact_refreshed": bool(adapter_source == "refreshed"),
        "product_training_route_binding_kwargs_wired": len(ready_routes) == len(ROUTES),
        "product_training_route_bound": False,
        "training_path_enabled": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "post_training_route_request_fields": {},
        "default_training_loop_kwargs_patch": default_patch,
        "synthetic_signed_training_loop_kwargs_patch": signed_patch,
        "route_reports": route_reports,
        "summary": {
            "product_training_route_count": len(ROUTES),
            "product_training_route_binding_kwargs_wired_count": len(ready_routes),
            "training_loop_constructor_switch_field_count": sum(
                1 for field in ROUTE_BINDING_KWARGS if field in signature_fields
            ),
            "unified_config_switch_field_count": sum(1 for field in ROUTE_BINDING_KWARGS if field in config_fields),
            "default_training_loop_kwargs_patch_field_count": len(default_patch),
            "synthetic_signed_training_loop_kwargs_patch_field_count": len(signed_patch),
            "owner_release_direction_recorded_count": int(
                adapter_summary.get("owner_release_direction_recorded_count", 0) or 0
            ),
            "owner_release_direction_approval_recorded_count": int(
                adapter_summary.get("owner_release_direction_approval_recorded_count", 0) or 0
            ),
            "product_training_route_binding_config_patch_ready_count": int(
                adapter_summary.get("product_training_route_binding_config_patch_ready_count", 0) or 0
            ),
            "training_path_enabled_count": 0,
            "request_fields_emitted_count": 0,
            "schema_exposure_allowed_count": 0,
            "ui_exposure_allowed_count": 0,
        },
        "blocked_reasons": []
        if len(ready_routes) == len(ROUTES)
        else ["product_training_route_binding_kwargs_missing"],
        "recommended_next_step": "keep route kwargs default-off until signed owner/product decisions are recorded",
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _route_report(route_id: str, relative_path: str, class_name: str, mode: str) -> dict[str, Any]:
    source = REPO_ROOT / relative_path
    call = _first_call(source, class_name)
    explicit = {keyword.arg for keyword in call.keywords if keyword.arg} if call is not None else set()
    has_helper = (
        any(keyword.arg is None and _uses_helper(keyword.value) for keyword in call.keywords)
        if call
        else False
    )
    wired = _all_present(explicit) if mode == "explicit_kwargs" else has_helper
    return {
        "route_id": route_id,
        "source": relative_path,
        "constructor": class_name,
        "binding_mode": mode,
        "route_binding_kwargs_wired": bool(wired),
        "explicit_route_binding_kwarg_count": sum(1 for field in ROUTE_BINDING_KWARGS if field in explicit),
        "helper_kwargs_present": bool(has_helper),
    }


def _first_call(source: Path, class_name: str) -> ast.Call | None:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == class_name:
            return node
    return None


def _uses_helper(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_turbocore_native_update_training_loop_kwargs"
    )


def _all_present(values: set[str]) -> bool:
    return all(field in values for field in ROUTE_BINDING_KWARGS)


def _config_adapter_report(
    *,
    config_adapter_report: Mapping[str, Any] | None,
    directory: Path,
    refresh_config_adapter_artifact: bool,
) -> tuple[dict[str, Any], str]:
    if isinstance(config_adapter_report, Mapping):
        return dict(config_adapter_report), "supplied"
    if refresh_config_adapter_artifact:
        return (
            build_optimizer_product_training_route_binding_config_adapter(
                artifact_dir=directory,
                write_artifact=True,
            ),
            "refreshed",
        )
    report = _read_config_adapter_artifact(directory)
    return report, "existing_artifact" if report else "missing_existing_artifact"


def _read_config_adapter_artifact(directory: Path) -> dict[str, Any]:
    source = directory / "turbocore_optimizer_product_training_route_binding_config_adapter.json"
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["build_optimizer_product_training_route_binding_product_route_adapter"]
