"""Smoke tests for image GGUF runtime-loader ABI contracts."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.contracts.image_gguf_runtime import (  # noqa: E402
    IMAGE_GGUF_RUNTIME_LOADER_ABI,
    ImageGGUFRuntimeLoadRequest,
    ImageGGUFRuntimeLoadResult,
    build_image_gguf_runtime_loader_abi,
)


def test_runtime_load_request_contract() -> None:
    request = ImageGGUFRuntimeLoadRequest(
        gguf_path="H:/models/anima.gguf",
        component="Anima_DiT",
        family="anima",
        required_runtime_features=["dit_module_builder", "dit_module_builder", "single_step_drift_quality_gate"],
        quality_gates="single_step_forward_shape",
    )
    assert request.abi == IMAGE_GGUF_RUNTIME_LOADER_ABI, request
    assert request.component == "anima_dit", request
    assert request.load_mode == "descriptor_only", request
    assert request.required_runtime_features == ["dit_module_builder", "single_step_drift_quality_gate"], request
    assert request.quality_gates == ["single_step_forward_shape"], request
    assert request.dry_run is True, request
    print("PASS: image GGUF runtime load request contract normalizes ABI fields")


def test_runtime_loader_abi_is_report_only() -> None:
    abi = build_image_gguf_runtime_loader_abi(
        component="vae",
        tensor_type_policy={"ok": True, "observed": {"f16": 7}},
        required_runtime_features=["vae_module_builder"],
        quality_gates=["vae_reconstruction_error"],
        blockers=["runtime model loader is not implemented for component: vae"],
    )
    assert abi["abi"] == IMAGE_GGUF_RUNTIME_LOADER_ABI, abi
    assert abi["implemented"] is False, abi
    assert abi["report_only"] is True, abi
    assert abi["supported_load_modes"] == ["descriptor_only"], abi
    assert abi["reads_tensor_payloads"] is False, abi
    assert abi["builds_model_modules"] is False, abi
    assert abi["training_path_enabled"] is False, abi
    print("PASS: image GGUF runtime loader ABI stays report-only")


def test_runtime_load_result_report_only() -> None:
    request = ImageGGUFRuntimeLoadRequest(gguf_path="H:/models/vae.gguf", component="vae", family="diffusers_vae")
    runtime_contract = {
        "component": "vae",
        "loadability": "shape_only_reference",
        "blockers": ["runtime model loader is not implemented for component: vae"],
    }
    result = ImageGGUFRuntimeLoadResult.report_only(request=request, runtime_contract=runtime_contract)
    payload = result.model_dump(mode="json")
    assert payload["status"] == "skipped", payload
    assert payload["runtime_loadable"] is False, payload
    assert payload["loader_implemented"] is False, payload
    assert payload["data"]["reads_tensor_payloads"] is False, payload
    assert payload["data"]["builds_model_modules"] is False, payload
    assert payload["data"]["training_path_enabled"] is False, payload
    assert payload["issues"][0]["code"] == "image_gguf.runtime_loader_blocked", payload
    print("PASS: image GGUF runtime load result reports skipped loader state")


if __name__ == "__main__":
    test_runtime_load_request_contract()
    test_runtime_loader_abi_is_report_only()
    test_runtime_load_result_report_only()
    print("\nAll image GGUF runtime contract smoke tests passed!")
