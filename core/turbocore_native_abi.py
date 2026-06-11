"""Validation helpers for future TurboCore native capability reports."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from core.turbocore_workspace_pipeline import (
    build_turbocore_native_training_capability_stub,
    build_workspace_pipeline_native_capability_stub,
)
from core.turbocore_optimizer_abi import NATIVE_OPTIMIZER_STATEFUL_ENTRYPOINTS
from core.turbocore_optimizer_abi import NATIVE_FLAT_ADAMW_OWNER_ENTRYPOINTS


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _required_entrypoints_for(feature: str) -> list[str]:
    stub = build_workspace_pipeline_native_capability_stub()
    features = _as_dict(stub.get("features"))
    feature_report = _as_dict(features.get(feature))
    return [str(item) for item in feature_report.get("required_entrypoints", [])]


def validate_native_feature_entrypoints(
    feature: str,
    report: Dict[str, Any],
    *,
    required_entrypoints: Iterable[str] | None = None,
) -> Dict[str, Any]:
    required = list(required_entrypoints) if required_entrypoints is not None else _required_entrypoints_for(feature)
    entrypoints = [str(item) for item in _as_dict(report).get("entrypoints", [])]
    missing = [name for name in required if name not in entrypoints]
    return {
        "feature": str(feature),
        "ok": not missing,
        "required_entrypoints": required,
        "reported_entrypoints": entrypoints,
        "missing_entrypoints": missing,
        "available": bool(_as_dict(report).get("available", False)),
        "status": str(_as_dict(report).get("status", "unknown") or "unknown"),
    }


def validate_workspace_pipeline_native_capabilities(report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the workspace/data-pipeline subset of a native report."""

    features = _as_dict(_as_dict(report).get("features"))
    workspace = validate_native_feature_entrypoints("workspace_pool", _as_dict(features.get("workspace_pool")))
    data_pipeline = validate_native_feature_entrypoints("data_pipeline", _as_dict(features.get("data_pipeline")))
    return {
        "schema_version": 1,
        "validator": "turbocore_workspace_pipeline_native_capabilities",
        "ok": bool(workspace["ok"] and data_pipeline["ok"]),
        "training_path_enabled": False,
        "features": {
            "workspace_pool": workspace,
            "data_pipeline": data_pipeline,
        },
    }


def validate_native_optimizer_stateful_capability(report: Dict[str, Any]) -> Dict[str, Any]:
    features = _as_dict(_as_dict(report).get("features"))
    optimizer = validate_native_feature_entrypoints(
        "native_optimizer",
        _as_dict(features.get("native_optimizer")),
        required_entrypoints=NATIVE_OPTIMIZER_STATEFUL_ENTRYPOINTS,
    )
    native_feature = _as_dict(features.get("native_optimizer"))
    flat_owner = validate_flat_adamw_owner_capability(native_feature)
    return {
        "schema_version": 1,
        "validator": "turbocore_native_optimizer_stateful_capability",
        "ok": bool(optimizer["ok"] and flat_owner["ok"]),
        "training_path_enabled": False,
        "features": {
            "native_optimizer": {
                **optimizer,
                "stateful": bool(native_feature.get("stateful", False)),
                "supported_optimizers": list(native_feature.get("supported_optimizers", []))
                if isinstance(native_feature.get("supported_optimizers"), list)
                else [],
                "flat_owner": flat_owner,
            },
        },
    }


def validate_flat_adamw_owner_capability(native_optimizer: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the nested persistent flat AdamW owner contract."""

    flat_owner = _as_dict(_as_dict(native_optimizer).get("flat_owner"))
    entrypoints = [str(item) for item in flat_owner.get("entrypoints", [])]
    missing = [name for name in NATIVE_FLAT_ADAMW_OWNER_ENTRYPOINTS if name not in entrypoints]
    buffers = [str(item) for item in flat_owner.get("required_buffers", [])]
    required_buffers = ["param_flat", "grad_flat", "exp_avg", "exp_avg_sq"]
    missing_buffers = [name for name in required_buffers if name not in buffers]
    layout = str(flat_owner.get("layout", "") or "")
    descriptor_schema = str(flat_owner.get("descriptor_schema", "") or "")
    descriptor_required = flat_owner.get("descriptor_required_fields") if isinstance(flat_owner.get("descriptor_required_fields"), list) else []
    missing_descriptor_fields = [name for name in ["layout", "buffers", "stream"] if name not in descriptor_required]
    binding_request = validate_flat_adamw_owner_binding_request_capability(flat_owner)
    ok = bool(flat_owner) and not missing and not missing_buffers and layout == "flat_contiguous_fp32_buffers" and descriptor_schema == "flat_adamw_owner_descriptor_v1" and not missing_descriptor_fields and binding_request["ok"]
    return {
        "schema_version": 1,
        "validator": "turbocore_flat_adamw_owner_capability",
        "ok": ok,
        "available": bool(flat_owner.get("available", False)),
        "status": str(flat_owner.get("status", "unknown") or "unknown"),
        "reason": str(flat_owner.get("reason", "") or ""),
        "training_path_enabled": bool(flat_owner.get("training_path_enabled", False)),
        "native_kernel_present": bool(flat_owner.get("native_kernel_present", False)),
        "layout": layout,
        "required_entrypoints": list(NATIVE_FLAT_ADAMW_OWNER_ENTRYPOINTS),
        "reported_entrypoints": entrypoints,
        "missing_entrypoints": missing,
        "required_buffers": required_buffers,
        "reported_buffers": buffers,
        "missing_buffers": missing_buffers,
        "supports_direct_gradient_write": bool(flat_owner.get("supports_direct_gradient_write", False)),
        "supports_external_tensor_handles": bool(flat_owner.get("supports_external_tensor_handles", False)),
        "supports_stream_descriptor": bool(flat_owner.get("supports_stream_descriptor", False)),
        "descriptor_schema": descriptor_schema,
        "descriptor_required_fields": list(descriptor_required),
        "missing_descriptor_fields": missing_descriptor_fields,
        "binding_request": binding_request,
    }


def validate_flat_adamw_owner_binding_request_capability(flat_owner: Dict[str, Any]) -> Dict[str, Any]:
    """Validate optional native tensor binding request metadata."""

    binding = _as_dict(_as_dict(flat_owner).get("binding_request"))
    if not binding:
        return {
            "schema_version": 1,
            "validator": "turbocore_flat_adamw_binding_request_capability",
            "ok": True,
            "present": False,
            "available": False,
            "binding_request_schema": "",
            "entrypoints": [],
            "missing_entrypoints": [],
            "native_kernel_present": False,
            "training_path_enabled": False,
        }
    entrypoints = [str(item) for item in binding.get("entrypoints", [])] if isinstance(binding.get("entrypoints"), list) else []
    required = [
        "validate_flat_adamw_tensor_binding_request",
        "probe_flat_adamw_tensor_object_binding",
        "create_flat_adamw_tensor_binding_session",
        "tensor_binding_session_snapshot",
        "tensor_binding_session_validate",
        "tensor_binding_session_launch_plan",
        "tensor_binding_session_stream_guard_probe",
        "tensor_binding_session_noop_launch",
        "tensor_binding_session_cpu_reference_guard",
        "tensor_binding_session_cuda_stub_launch",
        "tensor_binding_session_cuda_adamw_tensor_probe",
        "tensor_binding_session_cuda_adamw_runtime_probe",
        "get_adamw_cuda_kernel_contract",
        "destroy_tensor_binding_session",
    ]
    missing = [name for name in required if name not in entrypoints]
    schema = str(binding.get("binding_request_schema", "") or "")
    ok = schema == "turbocore_native_tensor_binding_request_v1" and not missing
    return {
        "schema_version": 1,
        "validator": "turbocore_flat_adamw_binding_request_capability",
        "ok": ok,
        "present": True,
        "available": bool(binding.get("available", False)),
        "binding_request_schema": schema,
        "entrypoints": entrypoints,
        "missing_entrypoints": missing,
        "supports_external_tensor_handles": bool(binding.get("supports_external_tensor_handles", False)),
        "supports_tensor_object_sessions": bool(binding.get("supports_tensor_object_sessions", False)),
        "supports_launch_plan": bool(binding.get("supports_launch_plan", False)),
        "kernel_registry": _validate_tensor_binding_kernel_registry(binding),
        "native_kernel_present": bool(binding.get("native_kernel_present", False)),
        "training_path_enabled": bool(binding.get("training_path_enabled", False)),
    }


def _validate_tensor_binding_kernel_registry(binding: Dict[str, Any]) -> Dict[str, Any]:
    registry = _as_dict(_as_dict(binding).get("kernel_registry"))
    if not registry:
        return {
            "schema_version": 1,
            "ok": False,
            "present": False,
            "entrypoints": [],
            "missing_entrypoints": ["tensor_binding_session_noop_launch"],
            "dry_run_launch_supported": False,
            "native_kernel_present": False,
            "training_path_enabled": False,
        }
    entrypoints = [str(item) for item in registry.get("entrypoints", [])] if isinstance(registry.get("entrypoints"), list) else []
    required = [
        "tensor_binding_session_noop_launch",
        "tensor_binding_session_cpu_reference_guard",
        "tensor_binding_session_cuda_stub_launch",
    ]
    missing = [name for name in required if name not in entrypoints]
    supported_plans = [str(item) for item in registry.get("supported_plans", [])] if isinstance(registry.get("supported_plans"), list) else []
    ok = not missing and "adamw_flat_fp32_launch_plan_v0" in supported_plans
    return {
        "schema_version": 1,
        "ok": ok,
        "present": True,
        "available": bool(registry.get("available", False)),
        "status": str(registry.get("status", "unknown") or "unknown"),
        "reason": str(registry.get("reason", "") or ""),
        "entrypoints": entrypoints,
        "missing_entrypoints": missing,
        "supported_plans": supported_plans,
        "dry_run_launch_supported": bool(registry.get("dry_run_launch_supported", False)),
        "cpu_reference_guard_supported": bool(registry.get("cpu_reference_guard_supported", False)),
        "cuda_stub_launch_supported": bool(registry.get("cuda_stub_launch_supported", False)),
        "kernel_contract": _validate_adamw_kernel_contract(registry.get("kernel_contract")),
        "native_kernel_present": bool(registry.get("native_kernel_present", False)),
        "training_path_enabled": bool(registry.get("training_path_enabled", False)),
    }


def _validate_adamw_kernel_contract(value: Any) -> Dict[str, Any]:
    contract = _as_dict(value)
    if not contract:
        return {
            "schema_version": 1,
            "ok": False,
            "present": False,
            "contract": "",
            "native_kernel_present": False,
            "training_path_enabled": False,
        }
    name = str(contract.get("contract", "") or "")
    launch_plan = str(contract.get("launch_plan", "") or "")
    ok = name == "adamw_flat_fp32_cuda_kernel_v0" and launch_plan == "adamw_flat_fp32_launch_plan_v0"
    return {
        "schema_version": 1,
        "ok": ok,
        "present": True,
        "contract": name,
        "launch_plan": launch_plan,
        "available": bool(contract.get("available", False)),
        "native_kernel_present": bool(contract.get("native_kernel_present", False)),
        "training_path_enabled": bool(contract.get("training_path_enabled", False)),
    }


def validate_inactive_native_training_capability_stub() -> Dict[str, Any]:
    """Validate the canonical inactive native training capability stub."""

    return validate_workspace_pipeline_native_capabilities(
        build_turbocore_native_training_capability_stub()
    )


__all__ = [
    "validate_inactive_native_training_capability_stub",
    "validate_flat_adamw_owner_binding_request_capability",
    "validate_flat_adamw_owner_capability",
    "validate_native_optimizer_stateful_capability",
    "validate_native_feature_entrypoints",
    "validate_workspace_pipeline_native_capabilities",
]
