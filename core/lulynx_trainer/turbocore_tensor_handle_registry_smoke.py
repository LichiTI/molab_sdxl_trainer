"""Smoke checks for TurboCore tensor handle registry contracts."""

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

from core.turbocore_flat_adamw_state import PersistentFlatAdamW  # noqa: E402
from core.turbocore_flat_buffer_descriptor import (  # noqa: E402
    ADAMW_FLAT_BUFFER_ROLES,
    validate_flat_adamw_owner_descriptor,
)
from core.turbocore_tensor_handle_registry import (  # noqa: E402
    TORCH_TENSOR_HANDLE_KIND,
    TurboCoreTensorHandleRegistry,
    build_flat_adamw_owner_descriptor_from_handles,
    build_tensor_object_map_for_handles,
    register_persistent_flat_adamw_buffers,
)


def _make_buffers(numel: int = 32) -> dict[str, torch.Tensor]:
    base = torch.linspace(-0.1, 0.1, steps=numel, dtype=torch.float32).contiguous()
    return {
        "param_flat": base.clone().contiguous(),
        "grad_flat": torch.full((numel,), 0.001, dtype=torch.float32).contiguous(),
        "exp_avg": torch.zeros(numel, dtype=torch.float32).contiguous(),
        "exp_avg_sq": torch.zeros(numel, dtype=torch.float32).contiguous(),
    }


def test_register_flat_adamw_buffers_and_descriptor() -> None:
    registry = TurboCoreTensorHandleRegistry(namespace="smoke")
    buffers = _make_buffers(48)
    handles = registry.register_flat_adamw_buffers(**buffers)
    assert sorted(handles) == sorted(ADAMW_FLAT_BUFFER_ROLES), handles

    descriptor = build_flat_adamw_owner_descriptor_from_handles(registry, handles)
    validation = validate_flat_adamw_owner_descriptor(descriptor)
    assert validation["ok"] is True, validation
    assert descriptor["training_path_enabled"] is False, descriptor
    for buffer in descriptor["buffers"]:
        assert buffer["handle_kind"] == TORCH_TENSOR_HANDLE_KIND, buffer
        assert buffer["handle_id"] == handles[buffer["role"]], buffer
        assert registry.resolve(buffer["handle_id"]).data_ptr() == buffers[buffer["role"]].data_ptr()

    snapshot = registry.snapshot()
    assert snapshot["handle_count"] == 4, snapshot
    assert snapshot["pointer_exported"] is False, snapshot
    assert snapshot["training_path_enabled"] is False, snapshot

    tensor_map = build_tensor_object_map_for_handles(registry, handles)
    assert sorted(tensor_map) == sorted(handles.values()), tensor_map
    assert tensor_map[handles["grad_flat"]].data_ptr() == buffers["grad_flat"].data_ptr()

    first_handle = handles["param_flat"]
    assert registry.release(first_handle) is True
    try:
        registry.resolve(first_handle)
    except KeyError:
        pass
    else:  # pragma: no cover - assertion guard
        raise AssertionError("released handle still resolves")
    registry.clear()
    assert registry.snapshot()["handle_count"] == 0


def test_reject_bad_tensor_shapes_and_dtypes() -> None:
    registry = TurboCoreTensorHandleRegistry(namespace="bad_inputs")
    non_contiguous = torch.arange(12, dtype=torch.float32).view(3, 4).t()
    try:
        registry.register(non_contiguous, role="param_flat")
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-contiguous tensor was accepted")

    try:
        registry.register(torch.zeros(4, dtype=torch.float16), role="grad_flat")
    except ValueError as exc:
        assert "dtype" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("float16 tensor was accepted")

    buffers = _make_buffers(16)
    buffers["exp_avg_sq"] = torch.zeros(17, dtype=torch.float32)
    try:
        registry.register_flat_adamw_buffers(**buffers)
    except ValueError as exc:
        assert "same numel" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched AdamW flat buffers were accepted")


def test_persistent_flat_owner_helper() -> None:
    owner = PersistentFlatAdamW([torch.randn(2, 3), torch.randn(5)])
    registry, handles, descriptor = register_persistent_flat_adamw_buffers(owner)
    validation = validate_flat_adamw_owner_descriptor(descriptor)
    assert validation["ok"] is True, validation
    assert validation["training_path_enabled"] is False, validation
    for role in ADAMW_FLAT_BUFFER_ROLES:
        assert registry.record(handles[role]).numel == int(owner.param_flat.numel())
    registry.clear()


def main() -> int:
    test_register_flat_adamw_buffers_and_descriptor()
    test_reject_bad_tensor_shapes_and_dtypes()
    test_persistent_flat_owner_helper()
    print("turbocore_tensor_handle_registry_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
