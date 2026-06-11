"""Newbie TensorRT transformer forward adapter.

This is an inactive adapter that mirrors the Newbie sampler's DiT call shape.
It is meant for LAB wiring and future architecture review; importing it or
constructing it does not patch `sample_newbie` or enable product generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

from backend.core.contracts import GenerationRequest

from .generation_preflight import TensorRtGenerationPreflight, preflight_newbie_tensorrt_generation_request
from .runtime_adapter import StaticTransformerRuntimeSpec, StaticTransformerTensorRtRuntime


@dataclass(frozen=True)
class NewbieTensorRtForwardPlan:
    preflight: TensorRtGenerationPreflight
    enabled: bool = False
    generation_path_enabled: bool = False
    training_path_enabled: bool = False

    @property
    def usable(self) -> bool:
        return bool(self.enabled and self.preflight.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "enabled": self.enabled,
            "generation_path_enabled": False,
            "training_path_enabled": False,
            "preflight": self.preflight.to_dict(),
        }


class NewbieTensorRtTransformerAdapter:
    """Callable DiT-style wrapper around a static transformer TensorRT runtime."""

    def __init__(self, runtime: StaticTransformerTensorRtRuntime) -> None:
        self.runtime = runtime

    @classmethod
    def from_spec(cls, spec: StaticTransformerRuntimeSpec) -> "NewbieTensorRtTransformerAdapter":
        return cls(StaticTransformerTensorRtRuntime(spec))

    def __call__(
        self,
        sample: Any,
        timestep: Any,
        encoder_hidden_states: Any,
        *args: Any,
        pooled_emb: Any | None = None,
        added_cond_kwargs: Mapping[str, Any] | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        text_embeds = pooled_emb
        if text_embeds is None and isinstance(added_cond_kwargs, Mapping):
            text_embeds = added_cond_kwargs.get("text_embeds")
        if text_embeds is None:
            text_embeds = kwargs.get("text_embeds")
        if text_embeds is None:
            raise ValueError("Newbie TensorRT transformer adapter requires pooled text_embeds")
        spec = getattr(self.runtime, "spec", None)
        output = self.runtime.infer(
            {
                "sample": sample,
                "timestep": _normalize_timestep(timestep, sample, spec),
                "encoder_hidden_states": _fit_sequence_to_static_tokens(encoder_hidden_states, spec),
                "text_embeds": _fit_last_dim(text_embeds, spec, "pooled_dim"),
            }
        )
        if not return_dict:
            return (output,)
        return SimpleNamespace(sample=output)


def plan_newbie_tensorrt_forward_adapter(
    request: GenerationRequest | Mapping[str, Any],
    spec: StaticTransformerRuntimeSpec,
    *,
    enabled: bool = False,
    vae_scale_factor: int = 16,
    positive_tokens: int = 512,
    negative_tokens: int = 0,
    cfg_strategy: str = "separate_calls",
) -> NewbieTensorRtForwardPlan:
    preflight = preflight_newbie_tensorrt_generation_request(
        request,
        spec,
        vae_scale_factor=vae_scale_factor,
        positive_tokens=positive_tokens,
        negative_tokens=negative_tokens,
        cfg_strategy=cfg_strategy,
    )
    return NewbieTensorRtForwardPlan(preflight=preflight, enabled=bool(enabled))


def _normalize_timestep(timestep: Any, sample: Any, spec: Any) -> Any:
    expected_batch = _expected_shape_value(spec, "batch") or _shape_dim(sample, 0) or 1
    shape = tuple(getattr(timestep, "shape", ()) or ())
    if shape == (expected_batch,):
        return timestep
    if hasattr(timestep, "reshape") and _numel(timestep) == 1:
        return timestep.reshape(1).expand(expected_batch) if hasattr(timestep.reshape(1), "expand") else timestep.reshape(1)
    return timestep


def _fit_sequence_to_static_tokens(value: Any, spec: Any) -> Any:
    tokens = _expected_shape_value(spec, "tokens")
    hidden_dim = _expected_shape_value(spec, "hidden_dim")
    fitted = _fit_dim(value, dim=1, target=tokens)
    return _fit_dim(fitted, dim=2, target=hidden_dim)


def _fit_last_dim(value: Any, spec: Any, key: str) -> Any:
    return _fit_dim(value, dim=-1, target=_expected_shape_value(spec, key))


def _fit_dim(value: Any, *, dim: int, target: int) -> Any:
    if not target or not hasattr(value, "shape"):
        return value
    shape = tuple(getattr(value, "shape", ()) or ())
    if not shape:
        return value
    index = dim if dim >= 0 else len(shape) + dim
    if index < 0 or index >= len(shape):
        return value
    current = int(shape[index])
    if current == target:
        return value
    if current > target and hasattr(value, "narrow"):
        return value.narrow(index, 0, target)
    if current < target:
        try:
            import torch

            pad_shape = list(shape)
            pad_shape[index] = target - current
            padding = torch.zeros(tuple(pad_shape), dtype=value.dtype, device=value.device)
            return torch.cat([value, padding], dim=index)
        except Exception:
            return value
    return value


def _expected_shape_value(spec: Any, key: str) -> int:
    shape = getattr(spec, "shape", None)
    if isinstance(shape, Mapping):
        try:
            return int(shape.get(key) or 0)
        except Exception:
            return 0
    return 0


def _shape_dim(value: Any, index: int) -> int:
    shape = tuple(getattr(value, "shape", ()) or ())
    try:
        return int(shape[index])
    except Exception:
        return 0


def _numel(value: Any) -> int:
    attr = getattr(value, "numel", None)
    if callable(attr):
        try:
            return int(attr())
        except Exception:
            return 0
    shape = tuple(getattr(value, "shape", ()) or ())
    total = 1
    for item in shape:
        total *= int(item)
    return total if shape else 0


__all__ = [
    "NewbieTensorRtForwardPlan",
    "NewbieTensorRtTransformerAdapter",
    "plan_newbie_tensorrt_forward_adapter",
]
