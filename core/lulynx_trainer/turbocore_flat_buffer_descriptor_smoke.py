"""Smoke checks for TurboCore flat buffer descriptor contracts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_flat_buffer_descriptor import (  # noqa: E402
    ADAMW_FLAT_BUFFER_ROLES,
    build_reference_flat_adamw_owner_descriptor,
    validate_flat_adamw_owner_descriptor,
)


def test_reference_descriptor_passes() -> None:
    descriptor = build_reference_flat_adamw_owner_descriptor(numel=128, device_type="cuda", device_index=0)
    payload = validate_flat_adamw_owner_descriptor(descriptor)
    assert payload["ok"] is True, payload
    assert payload["reported_roles"] == sorted(ADAMW_FLAT_BUFFER_ROLES), payload
    assert payload["numel_mismatch"] is False, payload


def test_missing_role_fails() -> None:
    descriptor = build_reference_flat_adamw_owner_descriptor(numel=16)
    descriptor["buffers"] = descriptor["buffers"][:-1]
    payload = validate_flat_adamw_owner_descriptor(descriptor)
    assert payload["ok"] is False, payload
    assert "exp_avg_sq" in payload["missing_roles"], payload


def test_bad_dtype_and_numel_fail() -> None:
    descriptor = build_reference_flat_adamw_owner_descriptor(numel=16)
    descriptor["buffers"][0]["dtype"] = "float16"
    descriptor["buffers"][1]["numel"] = 17
    payload = validate_flat_adamw_owner_descriptor(descriptor)
    assert payload["ok"] is False, payload
    assert payload["invalid_buffers"], payload
    assert payload["numel_mismatch"] is True, payload


def test_torch_tensor_handle_requires_id() -> None:
    descriptor = build_reference_flat_adamw_owner_descriptor(numel=16, handle_kind="torch_tensor")
    descriptor["buffers"][0]["handle_id"] = ""
    payload = validate_flat_adamw_owner_descriptor(descriptor)
    assert payload["ok"] is False, payload
    first = payload["invalid_buffers"][0]
    assert first["role"] == "param_flat", payload
    assert "missing_handle_id" in first["reasons"], payload


def main() -> int:
    test_reference_descriptor_passes()
    test_missing_role_fails()
    test_bad_dtype_and_numel_fail()
    test_torch_tensor_handle_requires_id()
    print("turbocore_flat_buffer_descriptor_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
