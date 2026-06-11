"""Runtime loadability contract for image GGUF descriptors.

This module is report-only. It does not load models or enable dispatch; it gives
future native/runtime loaders a stable ABI to satisfy before `runtime_loadable`
can become true.
"""

from __future__ import annotations

from typing import Any

try:
    from core.contracts.image_gguf_runtime import build_image_gguf_runtime_loader_abi
except ImportError:
    from backend.core.contracts.image_gguf_runtime import build_image_gguf_runtime_loader_abi


RUNTIME_CONTRACT_SCHEMA_VERSION = 1
LOADABILITY = "shape_only_reference"
SUPPORTED_COMPONENTS = {"vae", "clip", "t5", "sd15_unet", "sdxl_unet", "anima_dit", "newbie_dit"}
SUPPORTED_TENSOR_TYPES = {"f16", "f32"}
EXPERIMENTAL_TENSOR_TYPES = {"bf16"}


def build_image_gguf_runtime_contract(
    *,
    component: str,
    container_contract: dict[str, Any],
    shape_contract: dict[str, Any],
    issues: list[str],
    tensor_type_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tensor_policy = _tensor_type_policy(tensor_type_counts or shape_contract.get("tensor_type_counts") or {})
    blockers: list[str] = []
    if issues:
        blockers.append("container validation must pass before runtime loading can be considered")
    if not container_contract.get("ok"):
        blockers.append("tensor namespace contract is incomplete")
    if not shape_contract.get("ok"):
        blockers.append("shape-only reference contract is incomplete")
    if component not in SUPPORTED_COMPONENTS:
        blockers.append(f"unsupported image GGUF component for runtime contract: {component or '<missing>'}")
    if not tensor_policy["ok"]:
        blockers.extend(str(item) for item in tensor_policy["blockers"])
    blockers.append(f"runtime model loader is not implemented for component: {component or '<missing>'}")
    blockers = list(dict.fromkeys(blockers))
    runtime_features = _required_runtime_features(component)
    quality_gates = _quality_gates(component)
    loader_abi = build_image_gguf_runtime_loader_abi(
        component=component,
        tensor_type_policy=tensor_policy,
        required_runtime_features=runtime_features,
        quality_gates=quality_gates,
        blockers=blockers,
    )
    return {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "runtime_loadable": False,
        "loadability": LOADABILITY,
        "component": component,
        "supported_components": sorted(SUPPORTED_COMPONENTS),
        "container_contract_ok": bool(container_contract.get("ok")),
        "shape_contract_ok": bool(shape_contract.get("ok")),
        "tensor_type_policy": tensor_policy,
        "required_runtime_features": runtime_features,
        "runtime_entrypoint": "",
        "runtime_loader": {
            "implemented": False,
            "provider": "lulynx_image_runtime_loader",
            "abi": "image_gguf_runtime_loader_v1",
        },
        "runtime_loader_abi": loader_abi,
        "quality_gates": quality_gates,
        "blockers": blockers,
    }


def _tensor_type_policy(counts: dict[str, Any]) -> dict[str, Any]:
    normalized = {str(key).lower(): int(value or 0) for key, value in counts.items()}
    unsupported = sorted(key for key in normalized if key not in SUPPORTED_TENSOR_TYPES and key not in EXPERIMENTAL_TENSOR_TYPES)
    experimental = sorted(key for key in normalized if key in EXPERIMENTAL_TENSOR_TYPES)
    blockers = []
    if unsupported:
        blockers.append(f"unsupported tensor types for image runtime loader contract: {unsupported}")
    return {
        "ok": not unsupported,
        "supported": sorted(SUPPORTED_TENSOR_TYPES),
        "experimental": sorted(EXPERIMENTAL_TENSOR_TYPES),
        "observed": normalized,
        "unsupported": unsupported,
        "requires_explicit_opt_in": experimental,
        "blockers": blockers,
    }


def _required_runtime_features(component: str) -> list[str]:
    common = ["gguf_tensor_descriptor_reader", "tensor_name_mapper", "dtype_cast_policy"]
    if component == "vae":
        return common + ["vae_module_builder", "vae_reconstruction_quality_gate"]
    if component == "clip":
        return common + ["clip_text_encoder_builder", "hidden_state_quality_gate"]
    if component == "t5":
        return common + ["t5_encoder_builder", "sharded_text_encoder_policy", "hidden_state_quality_gate"]
    if component in {"sd15_unet", "sdxl_unet"}:
        return common + ["unet_module_builder", "conditioning_metadata_mapper", "single_step_drift_quality_gate"]
    if component in {"anima_dit", "newbie_dit"}:
        return common + ["dit_module_builder", "conditioning_metadata_mapper", "single_step_drift_quality_gate"]
    return common + ["component_runtime_adapter"]


def _quality_gates(component: str) -> list[str]:
    if component == "vae":
        return ["vae_reconstruction_error", "latent_roundtrip_shape"]
    if component in {"clip", "t5"}:
        return ["hidden_state_shape", "hidden_state_similarity"]
    if component in {"sd15_unet", "sdxl_unet", "anima_dit", "newbie_dit"}:
        return ["single_step_forward_shape", "single_step_forward_drift", "sample_smoke_optional"]
    return ["component_specific_quality_gate"]


__all__ = ["build_image_gguf_runtime_contract"]
