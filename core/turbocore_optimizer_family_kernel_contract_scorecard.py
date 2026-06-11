"""Native entrypoint scorecard for shared TurboCore optimizer family kernels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.services.native_module_loader import ensure_lulynx_native_artifact_path, load_lulynx_native


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    REPO_ROOT
    / "temp"
    / "turbocore_optimizer"
    / "turbocore_optimizer_family_kernel_contract_scorecard.json"
)
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ENTRYPOINT = "get_optimizer_family_kernel_contracts"
KERNEL_SOURCE = "optimizer_family_precondition_contract_cuda_v0"
REQUIRED_FAMILIES = (
    "adam_like_formula",
    "adaptive_lr_state_machine",
    "custom_formula",
    "closure_or_second_order",
    "factored_memory_layout",
    "model_or_shape_aware",
    "schedule_free_state_machine",
    "simple_formula",
    "state_adapter_special",
    "fused_backward",
)


def build_optimizer_family_kernel_contract_scorecard(*, write_artifact: bool = True) -> dict[str, Any]:
    loader = ensure_lulynx_native_artifact_path()
    native = load_lulynx_native()
    entrypoint_present = bool(native is not None and hasattr(native, ENTRYPOINT))
    native_payload: Mapping[str, Any] = {}
    error = ""
    if entrypoint_present:
        try:
            candidate = getattr(native, ENTRYPOINT)()
            native_payload = candidate if isinstance(candidate, Mapping) else {}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    contracts = [dict(item) for item in native_payload.get("contracts", []) if isinstance(item, Mapping)]
    validations = [dict(item) for item in native_payload.get("validations", []) if isinstance(item, Mapping)]
    families = {str(item.get("native_route_family", "") or "") for item in contracts}
    missing_families = [family for family in REQUIRED_FAMILIES if family not in families]
    validation_ok_count = sum(1 for item in validations if item.get("ok") is True)
    summary = {
        "required_family_count": len(REQUIRED_FAMILIES),
        "required_family_present_count": len(REQUIRED_FAMILIES) - len(missing_families),
        "optimizer_family_contract_count": len(contracts),
        "native_payload_contract_count": int(native_payload.get("optimizer_family_contract_count", 0) or 0),
        "validation_count": len(validations),
        "validation_ok_count": validation_ok_count,
        "entrypoint_present_count": 1 if entrypoint_present else 0,
        "kernel_source_ready_count": sum(1 for item in contracts if item.get("kernel_source") == KERNEL_SOURCE),
        "native_kernel_present_count": sum(1 for item in contracts if item.get("native_kernel_present") is True),
        "runtime_dispatch_ready_count": sum(1 for item in contracts if item.get("runtime_dispatch_ready") is True),
        "native_dispatch_allowed_count": sum(1 for item in contracts if item.get("native_dispatch_allowed") is True),
        "training_path_enabled_count": sum(1 for item in contracts if item.get("training_path_enabled") is True),
        "kernel_executed_count": sum(1 for item in contracts if item.get("kernel_executed") is True),
        "product_native_ready_count": sum(1 for item in contracts if item.get("product_native_ready") is True),
    }
    top_level_closed = all(
        not bool(native_payload.get(key, False))
        for key in (
            "runtime_dispatch_ready",
            "native_dispatch_allowed",
            "training_path_enabled",
            "kernel_executed",
            "product_native_ready",
        )
    )
    ok = bool(
        native is not None
        and entrypoint_present
        and not error
        and not missing_families
        and summary["optimizer_family_contract_count"] == len(REQUIRED_FAMILIES)
        and summary["native_payload_contract_count"] == len(REQUIRED_FAMILIES)
        and summary["validation_count"] == len(REQUIRED_FAMILIES)
        and summary["validation_ok_count"] == len(REQUIRED_FAMILIES)
        and summary["kernel_source_ready_count"] == len(REQUIRED_FAMILIES)
        and summary["native_kernel_present_count"] == len(REQUIRED_FAMILIES)
        and summary["runtime_dispatch_ready_count"] == 0
        and summary["native_dispatch_allowed_count"] == 0
        and summary["training_path_enabled_count"] == 0
        and summary["kernel_executed_count"] == 0
        and summary["product_native_ready_count"] == 0
        and top_level_closed
    )
    payload = {
        "schema_version": 1,
        "scorecard": "turbocore_optimizer_family_kernel_contract_scorecard_v0",
        "gate": "optimizer_family_kernel_contract_entrypoint",
        "ok": ok,
        "promotion_ready": False,
        "roadmap": ROADMAP,
        "entrypoint": ENTRYPOINT,
        "native_importable": native is not None,
        "entrypoint_present": entrypoint_present,
        "loader": loader,
        "error": error,
        "required_families": list(REQUIRED_FAMILIES),
        "missing_families": missing_families,
        "kernel_source": KERNEL_SOURCE,
        "shared_by_family": True,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
        "kernel_executed": False,
        "product_native_ready": False,
        "default_behavior_changed": False,
        "summary": summary,
        "contracts": contracts,
        "validations": validations,
        "promotion_blockers": [
            "contract_entrypoint_is_precondition_only",
            "family_specific_runtime_launch_not_supported",
            "training_path_dispatch_not_enabled",
            "owner_release_approval_missing",
        ],
        "recommended_next_step": "keep this proof in the quick suite; use family runtime smokes only for failure localization",
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = ["build_optimizer_family_kernel_contract_scorecard"]
