"""CUDA stream descriptor helpers for TurboCore report-only probes."""

from __future__ import annotations

from typing import Any

import torch


def current_torch_stream_descriptor(
    device: torch.device,
    *,
    capture_stage: str,
    request_event_chain: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe descriptor for the current PyTorch CUDA stream."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "descriptor": "turbocore_borrowed_cuda_stream_descriptor_v0",
        "device_type": "cuda" if device.type == "cuda" else str(device.type),
        "device_index": device.index,
        "stream_kind": "torch_current",
        "stream_id": "",
        "stream_source": "torch.cuda.current_stream",
        "stream_capture_stage": str(capture_stage or "unknown"),
        "python_stream_object_alive": True,
        "python_stream_lifetime_scope": "descriptor_only",
        "cuda_stream_handle": 0,
        "stream_handle_reported": False,
        "stream_handle_nonzero": False,
        "event_chain_probe_requested": bool(request_event_chain),
        "training_path_enabled": False,
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        payload["stream_source"] = "non_cuda_owner"
        payload["blocked_reasons"] = ["cuda_stream_unavailable"]
        return payload
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = torch.cuda.current_stream(index)
    handle = int(getattr(stream, "cuda_stream", 0) or 0)
    payload.update(
        {
            "device_index": int(index),
            "stream_id": str(handle) if handle else "",
            "cuda_stream_handle": handle,
            "stream_handle_reported": True,
            "stream_handle_nonzero": bool(handle),
            "blocked_reasons": ["external_stream_borrow_not_verified", "stream_lifetime_not_bound"],
        }
    )
    return payload


__all__ = ["current_torch_stream_descriptor"]
