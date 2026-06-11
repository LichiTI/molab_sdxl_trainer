# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Probe-only runtime contract for lossless compute decode.

The module intentionally exposes a closed runtime contract only. It does not
import the native extension, allocate CUDA/D3D12 resources, bind torch tensors,
or participate in trainer dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LosslessComputeDecodeRuntimeRequest:
    enabled: bool = False
    runtime_flag: str = "lossless_compute_decode_runtime"
    source: str = "probe_only_contract"


def build_lossless_compute_decode_runtime_contract(
    request: LosslessComputeDecodeRuntimeRequest | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if request is None:
        enabled = False
        source = "probe_only_contract"
    elif isinstance(request, LosslessComputeDecodeRuntimeRequest):
        enabled = bool(request.enabled)
        source = request.source
    else:
        enabled = bool(request.get("enabled", False))
        source = str(request.get("source") or "mapping")

    blockers = [
        "native_d3d12_cuda_external_memory_functional_probe_missing",
        "native_d3d12_cuda_fence_bridge_functional_probe_missing",
        "real_cache_sparse_candidate_missing",
        "guarded_trainer_runtime_ab_missing",
    ]
    return {
        "schema": "lulynx.lossless-compute-decode-runtime-contract.v1",
        "available": True,
        "source": source,
        "requested_enabled": enabled,
        "effective_enabled": False,
        "probe_only": True,
        "imports_native_extension": False,
        "runs_cuda": False,
        "runs_d3d12": False,
        "binds_torch_tensor": False,
        "training_path_enabled": False,
        "resource_center_allowed": False,
        "default_enabled": False,
        "safe_to_auto_execute": False,
        "runtime_ab_ready": False,
        "product_ready": False,
        "blockers": blockers,
        "next_recommended": (
            "keep runtime adapter closed until native external-memory, fence, "
            "real-cache candidate, and guarded trainer A/B evidence are present"
        ),
    }


__all__ = [
    "LosslessComputeDecodeRuntimeRequest",
    "build_lossless_compute_decode_runtime_contract",
]
