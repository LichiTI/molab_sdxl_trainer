"""TurboCore phase-1 contracts and math-equivalent reference helpers.

This module is intentionally conservative.  It does not enable native kernels
in normal training yet; it gives the four phase-1 targets a small, testable
Python contract so future Rust/CUDA implementations have clear parity anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


PHASE1_FEATURE_ORDER: Tuple[str, ...] = (
    "lora_fused",
    "native_optimizer",
    "static_route_step",
    "workspace_pool",
    "data_pipeline",
)


@dataclass(frozen=True)
class TurboCorePhase1Feature:
    name: str
    priority: int
    contract: str
    fallback: str = "standard"


PHASE1_FEATURES: Tuple[TurboCorePhase1Feature, ...] = (
    TurboCorePhase1Feature(
        name="lora_fused",
        priority=1,
        contract="delta=(xA)B with optional scale/add epilogue",
    ),
    TurboCorePhase1Feature(
        name="native_optimizer",
        priority=2,
        contract="LoRA AdamW update path after autograd gradients exist",
    ),
    TurboCorePhase1Feature(
        name="static_route_step",
        priority=3,
        contract="route/cache-specific train_step boundary with stable inputs",
    ),
    TurboCorePhase1Feature(
        name="workspace_pool",
        priority=4,
        contract="TurboCore-owned temporary buffer reuse by device/dtype/shape",
    ),
    TurboCorePhase1Feature(
        name="data_pipeline",
        priority=5,
        contract="bounded host staging queue with explicit workspace lease/release",
    ),
)


def phase1_capability_stub() -> Dict[str, Any]:
    return {
        "order": list(PHASE1_FEATURE_ORDER),
        "features": [
            {
                "name": feature.name,
                "priority": feature.priority,
                "contract": feature.contract,
                "fallback": feature.fallback,
                "available": False,
                "status": "scaffold",
            }
            for feature in PHASE1_FEATURES
        ],
    }


def lora_delta_reference(
    x: Any,
    down_weight: Any,
    up_weight: Any,
    *,
    scale: float = 1.0,
    base_output: Any = None,
) -> Any:
    """Math-equivalent LoRA delta reference for future fused kernels.

    The target native kernel should compute the same value as:

    ``base_output + scale * linear(linear(x, down_weight), up_weight)``

    when ``base_output`` is provided, otherwise it returns only the scaled
    LoRA delta.  The helper keeps shape semantics broad by using
    ``torch.nn.functional.linear`` on the last input dimension.
    """
    import torch.nn.functional as F

    delta = F.linear(F.linear(x, down_weight), up_weight)
    if scale != 1.0:
        delta = delta * scale
    if base_output is None:
        return delta
    return base_output + delta


def is_lora_parameter_name(name: str) -> bool:
    lowered = str(name or "").lower()
    return any(
        token in lowered
        for token in (
            "lora_down",
            "lora_up",
            "lora_a",
            "lora_b",
            "hada_w",
            "lokr_w",
            "adapter",
        )
    )


def collect_lora_optimizer_params(named_parameters: Iterable[Tuple[str, Any]]) -> List[Any]:
    """Return trainable LoRA/adapter params for native-optimizer candidates."""
    selected: List[Any] = []
    for name, param in named_parameters:
        if not is_lora_parameter_name(name):
            continue
        if bool(getattr(param, "requires_grad", False)):
            selected.append(param)
    return selected


def static_route_step_key(
    *,
    model_type: str,
    training_type: str = "lora",
    cached: bool = False,
    live_text_encoder: bool = False,
) -> str:
    """Return the planned route-specific train_step boundary key."""
    model = str(model_type or "unknown").strip().lower()
    train = str(training_type or "lora").strip().lower()
    cache_part = "cached" if cached else "live"
    if live_text_encoder:
        cache_part = "live_text"
    return f"{model}_{train}_{cache_part}"


class NativeWorkspacePool:
    """Small Python model of the future TurboCore-owned workspace pool.

    This pool is deliberately limited to buffers it allocates itself.  It never
    assumes ownership of normal PyTorch tensors passed in from the training
    graph.
    """

    def __init__(self, max_cached_buffers: int = 32) -> None:
        self.max_cached_buffers = max(1, int(max_cached_buffers))
        self._buffers: Dict[Tuple[str, str, Tuple[int, ...]], List[Any]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(shape: Iterable[int], *, dtype: Any, device: Any) -> Tuple[str, str, Tuple[int, ...]]:
        shape_tuple = tuple(int(dim) for dim in shape)
        return (str(device), str(dtype), shape_tuple)

    def acquire(self, shape: Iterable[int], *, dtype: Any, device: Any) -> Any:
        import torch

        key = self._key(shape, dtype=dtype, device=device)
        bucket = self._buffers.get(key) or []
        if bucket:
            self.hits += 1
            return bucket.pop()
        self.misses += 1
        return torch.empty(tuple(int(dim) for dim in shape), dtype=dtype, device=device)

    def release(self, tensor: Any) -> None:
        key = self._key(tuple(tensor.shape), dtype=tensor.dtype, device=tensor.device)
        bucket = self._buffers.setdefault(key, [])
        cached_count = sum(len(values) for values in self._buffers.values())
        if cached_count < self.max_cached_buffers:
            bucket.append(tensor)

    def stats(self) -> Dict[str, int]:
        cached_bytes = 0
        for values in self._buffers.values():
            for tensor in values:
                cached_bytes += int(tensor.numel()) * int(tensor.element_size())
        return {
            "hits": self.hits,
            "misses": self.misses,
            "cached_buffers": sum(len(values) for values in self._buffers.values()),
            "cached_bytes": cached_bytes,
        }

    def clear(self) -> None:
        self._buffers.clear()


__all__ = [
    "PHASE1_FEATURE_ORDER",
    "PHASE1_FEATURES",
    "NativeWorkspacePool",
    "collect_lora_optimizer_params",
    "is_lora_parameter_name",
    "lora_delta_reference",
    "phase1_capability_stub",
    "static_route_step_key",
]
