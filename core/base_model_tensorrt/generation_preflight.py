"""Preflight checks for optional TensorRT transformer generation wiring.

This module reports whether a generation request can use an already validated
static transformer TensorRT engine.  It does not load models, run samplers, or
enable generation by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:  # Launcher imports run with backend/ on sys.path; core tests may use project root.
    from backend.core.contracts import GenerationRequest
except ImportError:  # pragma: no cover - exercised by launcher package imports
    from core.contracts import GenerationRequest

from .runtime_adapter import StaticTransformerRuntimeSpec


@dataclass(frozen=True)
class TensorRtGenerationPreflight:
    ok: bool
    family: str
    component: str
    cfg_strategy: str
    engine_calls_per_step: int
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    request_shape: Mapping[str, int]
    engine_shape: Mapping[str, int]
    generation_path_enabled: bool = False
    training_path_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "family": self.family,
            "component": self.component,
            "cfg_strategy": self.cfg_strategy,
            "engine_calls_per_step": self.engine_calls_per_step,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "request_shape": dict(self.request_shape),
            "engine_shape": dict(self.engine_shape),
            "generation_path_enabled": False,
            "training_path_enabled": False,
        }


def preflight_newbie_tensorrt_generation_request(
    request: GenerationRequest | Mapping[str, Any],
    spec: StaticTransformerRuntimeSpec,
    *,
    vae_scale_factor: int = 16,
    positive_tokens: int = 333,
    negative_tokens: int = 0,
    pooled_dim: int = 1024,
    cfg_strategy: str = "separate_calls",
) -> TensorRtGenerationPreflight:
    req = request if isinstance(request, GenerationRequest) else GenerationRequest.model_validate(dict(request or {}))
    family = str(spec.family or "newbie").strip().lower()
    component = str(spec.component or "transformer").strip().lower()
    engine_shape = dict(spec.shape or {})
    request_shape = _request_shape(
        req,
        vae_scale_factor=vae_scale_factor,
        positive_tokens=positive_tokens,
        pooled_dim=pooled_dim,
    )
    blockers: list[str] = []
    warnings: list[str] = []
    normalized_cfg_strategy = _normalize_cfg_strategy(cfg_strategy)
    negative_cfg = _has_negative_cfg(req, negative_tokens=negative_tokens)

    if family != "newbie":
        blockers.append("tensorrt_generation_preflight_only_validated_for_newbie")
    if component != "transformer":
        blockers.append("tensorrt_generation_preflight_requires_transformer_component")
    if str(spec.precision or "fp32").lower() != "fp32":
        blockers.append("newbie_generation_tensorrt_requires_fp32_engine")
    if not str(spec.engine_path or "").strip():
        blockers.append("engine_path_required")
    if int(req.batch_size or 1) != int(engine_shape.get("batch") or 1):
        blockers.append("batch_size_mismatch")
    if negative_cfg and normalized_cfg_strategy == "blocked":
        blockers.append("cfg_negative_prompt_requires_second_static_engine_call_or_concat_contract")
    elif negative_cfg and normalized_cfg_strategy == "concat_batch":
        blockers.append("cfg_concat_batch_contract_not_validated_for_static_engine")
    elif negative_cfg and normalized_cfg_strategy == "separate_calls":
        warnings.append("cfg_negative_prompt_uses_separate_engine_calls")

    for key in ("latent_height", "latent_width", "tokens", "hidden_dim", "pooled_dim", "latent_channels"):
        if int(request_shape.get(key) or 0) != int(engine_shape.get(key) or 0):
            blockers.append(f"static_shape_mismatch:{key}")

    if int(req.steps or 0) > 1:
        warnings.append("static_transformer_smoke_is_single_step_only_generation_loop_not_wired")
    if str(req.sampler or "auto").strip().lower() not in {"auto", "euler"}:
        warnings.append("sampler_not_validated_for_static_tensorrt_transformer")

    return TensorRtGenerationPreflight(
        ok=not blockers,
        family=family,
        component=component,
        cfg_strategy=normalized_cfg_strategy,
        engine_calls_per_step=2 if negative_cfg and normalized_cfg_strategy == "separate_calls" else 1,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        request_shape=request_shape,
        engine_shape=engine_shape,
    )


def _request_shape(
    request: GenerationRequest,
    *,
    vae_scale_factor: int,
    positive_tokens: int,
    pooled_dim: int,
) -> dict[str, int]:
    scale = max(int(vae_scale_factor or 16), 1)
    return {
        "batch": int(request.batch_size or 1),
        "latent_channels": 16,
        "latent_height": int(request.height) // scale,
        "latent_width": int(request.width) // scale,
        "tokens": int(positive_tokens or 0),
        "hidden_dim": 2304,
        "pooled_dim": int(pooled_dim or 1024),
        "patch_size": 2,
    }


def _has_negative_cfg(request: GenerationRequest, *, negative_tokens: int) -> bool:
    if float(request.guidance_scale or 0.0) <= 1.0:
        return False
    return bool(str(request.negative_prompt or "").strip() or int(negative_tokens or 0) > 0)


def _normalize_cfg_strategy(value: str | None) -> str:
    key = str(value or "separate_calls").strip().lower().replace("-", "_")
    aliases = {
        "separate": "separate_calls",
        "two_calls": "separate_calls",
        "second_call": "separate_calls",
        "none": "blocked",
        "off": "blocked",
        "disabled": "blocked",
        "concat": "concat_batch",
        "batch": "concat_batch",
    }
    key = aliases.get(key, key)
    if key not in {"separate_calls", "concat_batch", "blocked"}:
        return "separate_calls"
    return key


__all__ = ["TensorRtGenerationPreflight", "preflight_newbie_tensorrt_generation_request"]
