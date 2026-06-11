"""Torch tensor handoff guards for native cache reader payload shadows."""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict

import numpy as np
import torch


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _float_close(left: float, right: float, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= max(atol, rtol * max(abs(float(left)), abs(float(right)), 1.0))


def _tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    tensor = value.detach().cpu()
    flat = tensor.reshape(-1).to(dtype=torch.float64)
    element_count = int(flat.numel())
    finite_count = int(torch.isfinite(flat).sum().item()) if element_count else 0
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "canonical_dtype": str(tensor.dtype).replace("torch.", ""),
        "element_count": element_count,
        "decoded_element_count": element_count,
        "decoded_finite_count": finite_count,
        "decoded_sum": float(flat.sum().item()) if element_count else 0.0,
        "decoded_min": float(flat.min().item()) if element_count else 0.0,
        "decoded_max": float(flat.max().item()) if element_count else 0.0,
        "sample_values": [float(item) for item in flat[:4].tolist()],
    }


def _compare_tensor_summaries(
    native_summary: Dict[str, Any],
    python_summary: Dict[str, Any],
    *,
    max_mismatches: int,
) -> Dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    match_count = 0
    for field in ("shape", "canonical_dtype", "element_count", "decoded_element_count", "decoded_finite_count"):
        if native_summary.get(field) == python_summary.get(field):
            match_count += 1
        elif len(mismatches) < max_mismatches:
            mismatches.append({"field": field, "native": native_summary.get(field), "python": python_summary.get(field)})
    for field in ("decoded_sum", "decoded_min", "decoded_max"):
        if _float_close(float(native_summary.get(field, 0.0) or 0.0), float(python_summary.get(field, 0.0) or 0.0)):
            match_count += 1
        elif len(mismatches) < max_mismatches:
            mismatches.append({"field": field, "native": native_summary.get(field), "python": python_summary.get(field)})
    native_values = list(native_summary.get("sample_values", []) or [])
    python_values = list(python_summary.get("sample_values", []) or [])
    if len(native_values) == len(python_values) and all(
        _float_close(float(native), float(python)) for native, python in zip(native_values, python_values)
    ):
        match_count += 1
    elif len(mismatches) < max_mismatches:
        mismatches.append({"field": "sample_values", "native": native_values, "python": python_values})
    return {
        "batch_parity_field_count": 9,
        "batch_parity_field_matches": match_count,
        "batch_parity_guard_passed": match_count == 9 and not mismatches,
        "batch_mismatch_count": len(mismatches),
        "batch_mismatches": mismatches,
    }


def _payload_contract(native_batch_payload_shadow: Dict[str, Any]) -> tuple[Any, torch.dtype, list[int], int, list[str]]:
    if not native_batch_payload_shadow:
        return None, torch.float32, [], 0, []
    if not bool(native_batch_payload_shadow.get("batch_cpu_payload_ready", False)):
        return None, torch.float32, [], 0, ["batch_cpu_payload_shadow_not_ready"]
    payload = native_batch_payload_shadow.get("batch_cpu_payload_bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        return None, torch.float32, [], 0, ["torch_handoff_payload_bytes_missing"]
    canonical_dtype = str(native_batch_payload_shadow.get("canonical_dtype") or "")
    dtype = _TORCH_DTYPES.get(canonical_dtype)
    if dtype is None:
        return None, torch.float32, [], 0, ["torch_handoff_dtype_not_supported"]
    shape = [int(dim) for dim in list(native_batch_payload_shadow.get("shape", []) or [])]
    if not shape:
        return None, dtype, [], 0, ["torch_handoff_shape_missing"]
    expected_elements = int(np.prod(shape, dtype=np.int64))
    return payload, dtype, shape, expected_elements, []


def _torch_handoff_summary(native_batch_payload_shadow: Dict[str, Any]) -> tuple[Dict[str, Any] | None, list[str]]:
    payload, dtype, shape, expected_elements, blockers = _payload_contract(native_batch_payload_shadow)
    if blockers or payload is None:
        return None, blockers
    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", UserWarning)
            tensor = torch.frombuffer(memoryview(payload), dtype=dtype).reshape(shape)
    except Exception:
        return None, ["torch_handoff_frombuffer_failed"]
    if int(tensor.numel()) != expected_elements:
        return None, ["torch_handoff_element_count_mismatch"]
    summary = _tensor_summary(tensor)
    summary.update(
        {
            "provider": "torch_frombuffer_batch_payload_shadow",
            "torch_tensor_handoff_shadow": True,
            "torch_tensor_handoff_ready": True,
            "storage_aliases_payload": True,
            "source_buffer_read_only": isinstance(payload, bytes),
            "torch_write_protection_enforced": False,
            "torch_frombuffer_warning_count": len(captured_warnings),
            "device": str(tensor.device),
            "requires_grad": bool(tensor.requires_grad),
            "is_contiguous": bool(tensor.is_contiguous()),
            "payload_byte_count": int(memoryview(payload).nbytes),
            "returns_tensor_payloads": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "training_path_enabled": False,
        }
    )
    return summary, []


def _pin_and_transfer_shadow(staging_tensor: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
    pin_memory_attempted = bool(torch.cuda.is_available())
    pin_memory_ready = False
    pin_memory_blocked_reasons: list[str] = []
    handoff_tensor = staging_tensor
    if pin_memory_attempted:
        try:
            pinned_tensor = staging_tensor.pin_memory()
            pin_memory_ready = bool(pinned_tensor.is_pinned())
            if pin_memory_ready:
                handoff_tensor = pinned_tensor
            else:
                pin_memory_blocked_reasons.append("pin_memory_not_ready")
        except Exception as exc:
            pin_memory_blocked_reasons.append(f"pin_memory_failed:{type(exc).__name__}")
    else:
        pin_memory_blocked_reasons.append("cuda_not_available")

    device_transfer_probe_ran = False
    device_transfer_ms = 0.0
    device_transfer_target = ""
    device_transfer_blocked_reasons: list[str] = []
    if pin_memory_ready and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            device_tensor = handoff_tensor.to(device="cuda", non_blocking=True)
            torch.cuda.synchronize()
            device_transfer_ms = (time.perf_counter() - start) * 1000.0
            device_transfer_target = str(device_tensor.device)
            device_transfer_probe_ran = True
        except Exception as exc:
            device_transfer_blocked_reasons.append(f"device_transfer_failed:{type(exc).__name__}")
    elif not torch.cuda.is_available():
        device_transfer_blocked_reasons.append("cuda_not_available")
    else:
        device_transfer_blocked_reasons.extend(pin_memory_blocked_reasons or ["pin_memory_not_ready"])

    return handoff_tensor, {
        "pin_memory_attempted": pin_memory_attempted,
        "pin_memory_ready": pin_memory_ready,
        "pin_memory_blocked_reasons": pin_memory_blocked_reasons,
        "device_transfer_probe_ran": device_transfer_probe_ran,
        "device_transfer_target": device_transfer_target,
        "device_transfer_ms": float(device_transfer_ms),
        "device_transfer_blocked_reasons": device_transfer_blocked_reasons,
    }


def _torch_owned_handoff_summary(native_batch_payload_shadow: Dict[str, Any]) -> tuple[Dict[str, Any] | None, list[str]]:
    payload, dtype, shape, expected_elements, blockers = _payload_contract(native_batch_payload_shadow)
    if blockers or payload is None:
        return None, [reason.replace("torch_handoff", "torch_owned_handoff") for reason in blockers]
    payload_view = memoryview(payload)
    expected_bytes = expected_elements * max(int(torch.empty((), dtype=dtype).element_size()), 1)
    if int(payload_view.nbytes) != expected_bytes:
        return None, ["torch_owned_handoff_byte_length_mismatch"]
    owned_payload = bytearray(payload_view)
    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", UserWarning)
            owned_view = torch.frombuffer(owned_payload, dtype=dtype).reshape(shape)
        staging_tensor = owned_view.clone(memory_format=torch.contiguous_format)
    except Exception:
        return None, ["torch_owned_handoff_frombuffer_failed"]
    if int(staging_tensor.numel()) != expected_elements:
        return None, ["torch_owned_handoff_element_count_mismatch"]

    lifetime_guard_checked = bool(owned_payload)
    lifetime_guard_passed = True
    if lifetime_guard_checked:
        snapshot = staging_tensor.clone(memory_format=torch.contiguous_format)
        owned_payload[0] ^= 0xFF
        lifetime_guard_passed = bool(torch.equal(staging_tensor, snapshot))
        owned_payload[0] ^= 0xFF

    handoff_tensor, transfer_summary = _pin_and_transfer_shadow(staging_tensor)
    summary = _tensor_summary(handoff_tensor)
    summary.update(
        {
            "provider": "torch_owned_copy_batch_payload_shadow",
            "torch_owned_tensor_handoff_shadow": True,
            "torch_owned_tensor_handoff_ready": True,
            "storage_aliases_payload": False,
            "storage_aliases_source_payload": False,
            "storage_aliases_owned_payload": False,
            "source_buffer_read_only": isinstance(payload, bytes),
            "owned_copy_byte_count": int(len(owned_payload)),
            "owned_copy_writeable": True,
            "detached_tensor_storage": True,
            "tensor_lifetime_guard_checked": lifetime_guard_checked,
            "tensor_lifetime_guard_passed": lifetime_guard_passed,
            "write_protection_strategy": "private_owned_copy_detached_tensor",
            "torch_write_protection_enforced": bool(lifetime_guard_passed),
            "torch_frombuffer_warning_count": len(captured_warnings),
            "device": str(handoff_tensor.device),
            "requires_grad": bool(handoff_tensor.requires_grad),
            "is_contiguous": bool(handoff_tensor.is_contiguous()),
            "is_pinned": bool(handoff_tensor.is_pinned()) if hasattr(handoff_tensor, "is_pinned") else False,
            "payload_byte_count": int(payload_view.nbytes),
            "returns_tensor_payloads": False,
            "cache_reader_path_enabled": False,
            "prefetch_queue_training_path_enabled": False,
            "training_path_enabled": False,
        }
    )
    summary.update(transfer_summary)
    return summary, []


def _compare_torch_handoff(torch_summary: Dict[str, Any], python_latents: torch.Tensor, *, max_mismatches: int) -> Dict[str, Any]:
    comparison = _compare_tensor_summaries(torch_summary, _tensor_summary(python_latents), max_mismatches=max_mismatches)
    checks = {
        "ready": bool(torch_summary.get("torch_tensor_handoff_ready", False)),
        "device": str(torch_summary.get("device") or "") == "cpu",
        "requires_grad": bool(torch_summary.get("requires_grad", True)) is False,
        "is_contiguous": bool(torch_summary.get("is_contiguous", False)) is True,
    }
    field_count = int(comparison.get("batch_parity_field_count", 0) or 0) + len(checks)
    field_matches = int(comparison.get("batch_parity_field_matches", 0) or 0)
    mismatches = list(comparison.get("batch_mismatches", []) or [])
    for field, matched in checks.items():
        if matched:
            field_matches += 1
        elif len(mismatches) < max_mismatches:
            mismatches.append({"field": f"torch_handoff_{field}", "native": torch_summary.get(field), "python": True})
    return {
        "torch_tensor_handoff_guard_ran": True,
        "torch_tensor_handoff_guard_passed": field_matches == field_count and not mismatches,
        "torch_tensor_handoff_field_count": field_count,
        "torch_tensor_handoff_field_matches": field_matches,
        "torch_tensor_handoff_mismatch_count": len(mismatches),
        "torch_tensor_handoff_mismatches": mismatches,
        "native_torch_tensor_handoff_reference": torch_summary,
        "torch_tensor_handoff_safety_note": "torch_frombuffer_over_read_only_bytes_is_shadow_only_write_protection_not_enforced",
    }


def _compare_torch_owned_handoff(torch_summary: Dict[str, Any], python_latents: torch.Tensor, *, max_mismatches: int) -> Dict[str, Any]:
    comparison = _compare_tensor_summaries(torch_summary, _tensor_summary(python_latents), max_mismatches=max_mismatches)
    expected_bytes = int(python_latents.detach().cpu().contiguous().numel() * python_latents.element_size())
    checks = {
        "ready": (bool(torch_summary.get("torch_owned_tensor_handoff_ready", False)), torch_summary.get("torch_owned_tensor_handoff_ready")),
        "device": (str(torch_summary.get("device") or "") == "cpu", torch_summary.get("device")),
        "requires_grad": (bool(torch_summary.get("requires_grad", True)) is False, torch_summary.get("requires_grad")),
        "is_contiguous": (bool(torch_summary.get("is_contiguous", False)) is True, torch_summary.get("is_contiguous")),
        "warning_free": (int(torch_summary.get("torch_frombuffer_warning_count", 0) or 0) == 0, torch_summary.get("torch_frombuffer_warning_count")),
        "source_payload_alias_free": (bool(torch_summary.get("storage_aliases_source_payload", True)) is False, torch_summary.get("storage_aliases_source_payload")),
        "owned_payload_alias_free": (bool(torch_summary.get("storage_aliases_owned_payload", True)) is False, torch_summary.get("storage_aliases_owned_payload")),
        "lifetime_guard": (bool(torch_summary.get("tensor_lifetime_guard_passed", False)) is True, torch_summary.get("tensor_lifetime_guard_passed")),
        "write_protection_guard": (bool(torch_summary.get("torch_write_protection_enforced", False)) is True, torch_summary.get("torch_write_protection_enforced")),
        "payload_byte_count": (int(torch_summary.get("payload_byte_count", 0) or 0) == expected_bytes, torch_summary.get("payload_byte_count")),
    }
    field_count = int(comparison.get("batch_parity_field_count", 0) or 0) + len(checks)
    field_matches = int(comparison.get("batch_parity_field_matches", 0) or 0)
    mismatches = list(comparison.get("batch_mismatches", []) or [])
    for field, (matched, native_value) in checks.items():
        if matched:
            field_matches += 1
        elif len(mismatches) < max_mismatches:
            mismatches.append({"field": f"torch_owned_handoff_{field}", "native": native_value, "python": True})
    return {
        "torch_owned_tensor_handoff_guard_ran": True,
        "torch_owned_tensor_handoff_guard_passed": field_matches == field_count and not mismatches,
        "torch_owned_tensor_handoff_field_count": field_count,
        "torch_owned_tensor_handoff_field_matches": field_matches,
        "torch_owned_tensor_handoff_mismatch_count": len(mismatches),
        "torch_owned_tensor_handoff_mismatches": mismatches,
        "native_torch_owned_tensor_handoff_reference": torch_summary,
        "torch_owned_tensor_handoff_safety_note": "private_owned_copy_detached_tensor_shadow_only_no_training_dispatch",
    }


def run_torch_tensor_handoff_guards(
    native_batch_payload_shadow: Dict[str, Any],
    python_latents: torch.Tensor,
    *,
    max_mismatches: int = 8,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "torch_tensor_handoff_guard_ran": False,
        "torch_tensor_handoff_guard_passed": False,
        "torch_owned_tensor_handoff_guard_ran": False,
        "torch_owned_tensor_handoff_guard_passed": False,
    }
    torch_summary, torch_blockers = _torch_handoff_summary(native_batch_payload_shadow)
    if torch_blockers:
        report["torch_tensor_handoff_blocked_reasons"] = torch_blockers
    elif torch_summary is not None:
        report.update(_compare_torch_handoff(torch_summary, python_latents, max_mismatches=max_mismatches))

    owned_summary, owned_blockers = _torch_owned_handoff_summary(native_batch_payload_shadow)
    if owned_blockers:
        report["torch_owned_tensor_handoff_blocked_reasons"] = owned_blockers
    elif owned_summary is not None:
        report.update(_compare_torch_owned_handoff(owned_summary, python_latents, max_mismatches=max_mismatches))
    return report


__all__ = ["run_torch_tensor_handoff_guards"]
