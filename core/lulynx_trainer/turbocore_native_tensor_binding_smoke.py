"""Smoke checks for TurboCore native tensor binding request contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_native_tensor_binding import (  # noqa: E402
    NATIVE_TENSOR_BINDING_REQUEST_SCHEMA,
    build_flat_adamw_native_binding_request,
    evaluate_flat_adamw_native_binding_request,
)
from core.turbocore_optimizer_abi import build_flat_adamw_owner_capability_stub  # noqa: E402
from core.turbocore_tensor_handle_registry import TurboCoreTensorHandleRegistry  # noqa: E402


def _make_handles() -> tuple[TurboCoreTensorHandleRegistry, dict[str, str]]:
    registry = TurboCoreTensorHandleRegistry(namespace="native_binding_smoke")
    buffers = {
        "param_flat": torch.arange(32, dtype=torch.float32).contiguous(),
        "grad_flat": torch.full((32,), 0.01, dtype=torch.float32).contiguous(),
        "exp_avg": torch.zeros(32, dtype=torch.float32).contiguous(),
        "exp_avg_sq": torch.zeros(32, dtype=torch.float32).contiguous(),
    }
    return registry, registry.register_flat_adamw_buffers(**buffers)


def test_binding_request_shape_ready_but_not_performance_ready() -> None:
    registry, handles = _make_handles()
    capability = build_flat_adamw_owner_capability_stub()
    request = build_flat_adamw_native_binding_request(
        registry,
        handles,
        native_flat_owner_capability=capability,
    )
    readiness = request["readiness"]
    assert request["schema"] == NATIVE_TENSOR_BINDING_REQUEST_SCHEMA, request
    assert request["training_path_enabled"] is False, request
    assert readiness["ok"] is True, readiness
    assert readiness["request_shape_ready"] is True, readiness
    assert readiness["native_binding_ready"] is False, readiness
    assert readiness["performance_test_ready"] is False, readiness
    assert "native_external_tensor_handles_unsupported" in readiness["blocked_reasons"], readiness
    assert "native_kernel_missing" in readiness["blocked_reasons"], readiness


def test_binding_request_can_report_future_native_readiness() -> None:
    registry, handles = _make_handles()
    capability = build_flat_adamw_owner_capability_stub()
    capability["available"] = True
    capability["supports_external_tensor_handles"] = True
    capability["native_kernel_present"] = True
    request = build_flat_adamw_native_binding_request(
        registry,
        handles,
        native_flat_owner_capability=capability,
    )
    readiness = request["readiness"]
    assert readiness["native_binding_ready"] is True, readiness
    assert readiness["performance_test_ready"] is True, readiness
    assert readiness["blocked_reasons"] == [], readiness


def test_pointer_export_is_rejected() -> None:
    registry, handles = _make_handles()
    request = build_flat_adamw_native_binding_request(registry, handles)
    request["bindings"][0]["pointer_exported"] = True
    readiness = evaluate_flat_adamw_native_binding_request(request)
    assert readiness["ok"] is False, readiness
    assert readiness["pointer_exported"] is True, readiness
    assert readiness["invalid_bindings"], readiness


def main() -> int:
    test_binding_request_shape_ready_but_not_performance_ready()
    test_binding_request_can_report_future_native_readiness()
    test_pointer_export_is_rejected()
    print("turbocore_native_tensor_binding_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
