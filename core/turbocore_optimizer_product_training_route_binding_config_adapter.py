"""Config adapter contract for post-approval TurboCore optimizer route binding."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_optimizer_product_training_route_binding_preflight import (
    build_optimizer_product_training_route_binding_preflight,
)
from core.turbocore_optimizer_product_training_route_binding_training_loop_contract import (
    build_optimizer_product_training_route_binding_training_loop_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_product_training_route_binding_config_adapter.json"
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"

ROUTE_BINDING_MODE_FIELD = "turbocore_native_update_mode"
ROUTE_BINDING_SWITCHES = (
    "turbocore_native_update_dispatch_enabled",
    "turbocore_native_update_training_path_enabled",
    "turbocore_native_update_require_native_cuda",
)


def build_optimizer_product_training_route_binding_config_adapter(
    *,
    preflight_report: Mapping[str, Any] | None = None,
    training_loop_contract: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    write_artifact: bool = True,
) -> dict[str, Any]:
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    preflight = _as_dict(preflight_report) or build_optimizer_product_training_route_binding_preflight(
        artifact_dir=directory,
        write_artifact=True,
    )
    contract = _as_dict(training_loop_contract) or build_optimizer_product_training_route_binding_training_loop_contract(
        artifact_dir=directory,
        write_artifact=True,
    )
    signature_fields = set(inspect.signature(TrainingLoop.__init__).parameters)
    preflight_summary = _as_dict(preflight.get("summary"))
    candidate = _as_dict(preflight.get("post_approval_training_route_binding_candidate"))
    switches = _as_dict(candidate.get("existing_training_loop_switches"))
    patch_ready = bool(
        preflight.get("product_training_route_binding_preflight_ready") is True
        and contract.get("post_approval_training_loop_context_ready") is True
        and all(signature_field in signature_fields for signature_field in ROUTE_BINDING_SWITCHES)
    )
    patch = _kwargs_patch(switches) if patch_ready else {}
    payload = {
        "schema_version": 1,
        "adapter": "turbocore_optimizer_product_training_route_binding_config_adapter_v0",
        "gate": "optimizer_product_training_route_binding_config_adapter",
        "ok": bool(contract.get("ok") is True and _signature_ready(signature_fields)),
        "roadmap": ROADMAP,
        "artifact_first": True,
        "product_training_route_binding_config_patch_ready": bool(patch_ready),
        "product_training_route_bound": False,
        "training_path_enabled": False,
        "request_fields_emitted": False,
        "schema_exposure_allowed": False,
        "ui_exposure_allowed": False,
        "backend_router_registered": False,
        "post_training_route_request_fields": {},
        "training_loop_kwargs_patch": patch,
        "summary": {
            "training_loop_constructor_switch_field_count": sum(
                1 for field in ROUTE_BINDING_SWITCHES if field in signature_fields
            ),
            "training_loop_constructor_mode_field_present_count": 1
            if ROUTE_BINDING_MODE_FIELD in signature_fields
            else 0,
            "owner_release_direction_recorded_count": int(
                preflight_summary.get("owner_release_direction_recorded_count", 0) or 0
            ),
            "owner_release_direction_approval_recorded_count": int(
                preflight_summary.get("owner_release_direction_approval_recorded_count", 0) or 0
            ),
            "product_training_route_binding_config_patch_ready_count": 1 if patch_ready else 0,
            "training_loop_kwargs_patch_field_count": len(patch),
            "request_fields_emitted_count": 0,
            "schema_exposure_allowed_count": 0,
            "ui_exposure_allowed_count": 0,
        },
        "blocked_reasons": _blocked_reasons(preflight, contract, signature_fields, patch_ready),
        "recommended_next_step": (
            "after real owner approval, apply this kwargs patch through the existing trainer config path"
            if patch_ready
            else "keep route-binding kwargs patch empty until signed owner/product decisions are recorded"
        ),
    }
    if write_artifact:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def _kwargs_patch(switches: Mapping[str, Any]) -> dict[str, Any]:
    if not all(switches.get(field) is True for field in ROUTE_BINDING_SWITCHES):
        return {}
    return {
        ROUTE_BINDING_MODE_FIELD: "native_experimental",
        **{field: True for field in ROUTE_BINDING_SWITCHES},
    }


def _signature_ready(signature_fields: set[str]) -> bool:
    return ROUTE_BINDING_MODE_FIELD in signature_fields and all(
        field in signature_fields for field in ROUTE_BINDING_SWITCHES
    )


def _blocked_reasons(
    preflight: Mapping[str, Any],
    contract: Mapping[str, Any],
    signature_fields: set[str],
    patch_ready: bool,
) -> list[str]:
    reasons: list[str] = []
    if preflight.get("product_training_route_binding_preflight_ready") is not True:
        reasons.append("product_training_route_binding_preflight_not_ready")
    if contract.get("post_approval_training_loop_context_ready") is not True:
        reasons.append("training_loop_route_binding_contract_not_ready")
    if not _signature_ready(signature_fields):
        reasons.append("training_loop_constructor_route_binding_fields_missing")
    if not patch_ready:
        reasons.append("training_loop_kwargs_patch_not_emitted")
    return reasons


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["build_optimizer_product_training_route_binding_config_adapter"]
