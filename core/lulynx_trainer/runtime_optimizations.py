"""
Shared runtime optimization helpers for native Lulynx trainers.

The goal is to keep attention backend selection / torch.compile policy in one
place so SDXL, Anima and Newbie can reuse the same runtime decisions.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
import os
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


_ATTENTION_ALIASES = {
    "": "auto",
    "auto": "auto",
    "default": "auto",
    "torch": "torch",
    "native": "torch",
    "xformers": "xformers",
    "sdpa": "sdpa",
    "flash": "flash2",
    "flash2": "flash2",
    "flashattn": "flash2",
    "flashattention": "flash2",
    "flashattention2": "flash2",
    "fa2": "flash2",
    "sage": "sageattn",
    "sageattn": "sageattn",
    "sageattention": "sageattn",
    "flex": "flexattn",
    "flexattn": "flexattn",
    "flexattention": "flexattn",
    "sparge": "spargeattn2",
    "spargeattn": "spargeattn2",
    "spargeattn2": "spargeattn2",
}

_SUPPORTED_ATTENTION_BACKENDS = {"auto", "torch", "xformers", "sdpa", "flash2", "sageattn", "flexattn", "spargeattn2"}
_SUPPORTED_SDPA_BACKEND_POLICIES = {"auto", "cutlass", "flash", "cudnn", "math"}
_DIFFUSERS_UNET_MODEL_ARCHS = {"sdxl", "sd15"}
_DIT_MODEL_ARCHS = {"anima", "newbie"}
_DIFFUSERS_PROCESSOR_BACKENDS = {"flash2", "xformers", "sageattn", "spargeattn2"}
_DIT_PATCHED_BACKENDS = {"flash2", "xformers", "sageattn", "spargeattn2", "flexattn"}


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _normalize_attention_backend(value: Any) -> str:
    normalized = _ATTENTION_ALIASES.get(str(value or "").strip().lower(), None)
    if normalized is None:
        return "auto"
    return normalized


def _normalize_sdpa_backend_policy(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "cutlass"
    return normalized if normalized in _SUPPORTED_SDPA_BACKEND_POLICIES else "cutlass"


def _normalize_compile_shape_strategy(value: Any) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "default": "auto",
        "pad": "fixed_pad",
        "static_pad": "fixed_pad",
        "flatten": "token_flatten",
        "tokenflatten": "token_flatten",
        "no_pad": "native",
        "native_no_pad": "native",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"auto", "fixed_pad", "token_flatten", "native"} else "auto"


def _normalize_compile_target_strategy(value: Any) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "default": "auto",
        "per_block": "block",
        "forward_impl": "inner_forward",
        "inner": "inner_forward",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"auto", "block", "inner_forward"} else "auto"


def _runtime_route(config: Any) -> str:
    for key in ("schema_id", "model_type", "model_arch", "training_type"):
        raw = str(getattr(config, key, "") or "").strip().lower().replace("-", "_")
        if raw.startswith("anima"):
            return "anima"
        if raw.startswith("newbie"):
            return "newbie"
        if raw.startswith("flux"):
            return "flux"
        if raw.startswith("sdxl"):
            return "sdxl"
        if raw.startswith("sd15") or raw.startswith("sd_lora") or raw.startswith("sd_1"):
            return "sd15"
    return "sdxl"


def _resolve_compile_shape_strategy(config: Any, *, route: str, compile_requested: bool) -> tuple[str, str]:
    requested = _normalize_compile_shape_strategy(getattr(config, "compile_shape_strategy", "auto"))
    route_name = str(route or "").strip().lower()
    if not compile_requested:
        return requested, "compile_not_requested"
    if route_name not in {"anima", "newbie"}:
        if requested in {"token_flatten", "native"}:
            return "fixed_pad", f"{route_name or 'route'} does not support token-flatten compile shape yet"
        return ("fixed_pad" if requested == "auto" else requested), "non_native_route_defaults_to_fixed_pad"

    fixed_visual = int(getattr(config, f"{route_name}_fixed_visual_tokens", 0) or 0)
    cache_first = (
        _boolish(getattr(config, "use_cache", False), default=False)
        if route_name == "newbie"
        else _boolish(getattr(config, "anima_cached_training", True), default=True)
    )
    native_bucket_compile = _boolish(getattr(config, "native_token_bucket_compile", True), default=True) and cache_first
    if requested == "auto":
        if fixed_visual <= 0 and native_bucket_compile:
            return "token_flatten", "auto shape strategy resolved to token_flatten from native token buckets"
        return "fixed_pad", "auto shape strategy resolved to fixed_pad"
    if requested in {"token_flatten", "native"} and not native_bucket_compile and fixed_visual <= 0:
        return "fixed_pad", "token-flatten shape strategy fell back to fixed_pad because native token buckets are unavailable"
    return requested, "explicit shape strategy kept"


def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _enable_diffusers_attention_backend(unet: Any, plan: "RuntimeOptimizationPlan") -> int:
    from .diffusers_attention import (
        build_diffusers_attention_kernel,
        install_diffusers_attention_processor,
    )

    kernel = build_diffusers_attention_kernel(plan.attention_backend)
    patched = install_diffusers_attention_processor(unet, kernel)
    plan.reasons.append(
        f"attention backend {kernel.backend_id} installed generic diffusers processors "
        f"on {patched} U-Net attention modules"
    )
    return patched


def _fallback_to_sdpa_attention(model: Any, plan: "RuntimeOptimizationPlan", reason: str) -> bool:
    """Best-effort final attention fallback when a requested backend fails late."""

    if plan.attention_backend == "sdpa":
        return False
    previous_backend = plan.attention_backend
    plan.warnings.append(
        f"attention backend {previous_backend} failed at apply time; falling back to sdpa: {reason}"
    )
    plan.reasons.append(f"attention backend {previous_backend} runtime fallback to sdpa")
    plan.attention_backend = "sdpa"
    try:
        _apply_attention_backend_once(model, plan)
        return True
    except Exception as fallback_exc:
        plan.warnings.append(f"sdpa fallback failed after {previous_backend}: {fallback_exc}")
        plan.attention_backend = previous_backend
        return False


def _flex_attention_available() -> bool:
    try:
        from torch.nn.attention.flex_attention import flex_attention

        return callable(flex_attention)
    except Exception:
        return False


def _flexattention_runtime_active(config: Any) -> bool:
    runtime_id = str(
        getattr(config, "runtime_id", "")
        or getattr(config, "execution_profile_id", "")
        or ""
    ).strip().lower()
    if runtime_id == "flexattention":
        return True
    return _boolish(os.environ.get("LULYNX_FLEXATTENTION_STARTUP", ""), default=False)


def _swap_config_enabled(config: Any) -> bool:
    granularity = str(getattr(config, "swap_granularity", "off") or "off").strip().lower().replace("-", "_")
    try:
        swap_ratio = float(getattr(config, "swap_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        swap_ratio = 0.0
    try:
        swap_count = int(getattr(config, "swap_count", 0) or 0)
    except (TypeError, ValueError):
        swap_count = 0
    try:
        legacy_blocks = int(getattr(config, "blocks_to_swap", 0) or 0)
    except (TypeError, ValueError):
        legacy_blocks = 0
    return (granularity != "off" and (swap_ratio > 0.0 or swap_count > 0 or granularity == "auto")) or legacy_blocks > 0


def _resolve_split_chunks(config: Any) -> int:
    """Read ``anima_split_attn`` (or numeric override) from config.

    The UI exposes ``split_attn`` as a boolean; route_service maps it to
    ``anima_split_attn``. ``True`` means "split attention to save VRAM" — the
    default chunk count is 2, which halves the attention-matrix peak. A
    numeric override (``anima_split_attn_chunks=N``) is honoured if present.
    """
    explicit = getattr(config, "anima_split_attn_chunks", None)
    if explicit is not None:
        try:
            n = int(explicit)
            if n > 1:
                return n
        except (TypeError, ValueError):
            pass

    enabled = _boolish(
        getattr(config, "anima_split_attn", False)
        or getattr(config, "split_attn", False),
        default=False,
    )
    return 2 if enabled else 0


@dataclass
class RuntimeOptimizationPlan:
    attention_backend: str
    requested_attention_backend: str
    sdpa_backend_policy: str = "cutlass"
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    torch_compile_dynamic: bool = False
    torch_compile_fullgraph: bool = False
    torch_compile_scope: str = ""  # "", "per_block", "full"
    torch_compile_allow_full_with_per_block: bool = False
    anima_compile_scope: str = ""  # "", "per_block", "full", "full_cudagraph"
    compile_shape_strategy: str = "auto"
    compile_target_strategy: str = "auto"
    attention_split_chunks: int = 0  # 0/1 = disabled, >1 = head-group split
    attention_early_deletion: bool = False  # del Q/K/V immediately after attention
    amd_sdpa_slice_trigger_gb: float = 0.0
    amd_sdpa_slice_target_gb: float = 0.0
    dynamo_recompile_limit: int = 0  # 0 = leave torch default
    activation_memory_budget: float = 0.0  # 0 = off; (0,1] caps AOT partitioner saved set
    gradient_checkpointing: bool = False  # mirrored for the budget mutual-exclusion guard
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def log_lines(self) -> Iterable[str]:
        yield (
            f"Runtime optimization: attention={self.attention_backend} "
            f"(requested={self.requested_attention_backend}), "
            f"torch_compile={'on' if self.torch_compile else 'off'}"
            + (f", sdpa_backend={self.sdpa_backend_policy}" if self.attention_backend == "sdpa" else "")
            + (f", anima_compile_scope={self.anima_compile_scope}" if self.anima_compile_scope else "")
            + (f", compile_shape={self.compile_shape_strategy}" if self.compile_shape_strategy != "auto" else "")
            + (f", compile_target={self.compile_target_strategy}" if self.compile_target_strategy != "auto" else "")
            + (f", split_chunks={self.attention_split_chunks}" if self.attention_split_chunks > 1 else "")
            + (", early_deletion=on" if self.attention_early_deletion else "")
            + (
                f", amd_sdpa_slice={self.amd_sdpa_slice_trigger_gb:.2f}->{self.amd_sdpa_slice_target_gb:.2f}GB"
                if self.amd_sdpa_slice_trigger_gb > 0 and self.amd_sdpa_slice_target_gb > 0
                else ""
            )
        )
        for reason in self.reasons:
            yield f"[runtime-opt] {reason}"
        for warning in self.warnings:
            yield f"[runtime-opt][warn] {warning}"


def build_runtime_optimization_plan(config: Any) -> RuntimeOptimizationPlan:
    requested_attention = _normalize_attention_backend(getattr(config, "attention_backend", None) or getattr(config, "attn_mode", None))
    requested_compile = _boolish(getattr(config, "torch_compile", False), default=False)
    route = _runtime_route(config)
    requested_device = str(getattr(config, "device", "") or "").strip().lower()

    plan = RuntimeOptimizationPlan(
        attention_backend=requested_attention,
        requested_attention_backend=requested_attention,
        sdpa_backend_policy=_normalize_sdpa_backend_policy(getattr(config, "sdpa_backend_policy", "cutlass")),
        torch_compile=requested_compile,
        torch_compile_backend=str(getattr(config, "torch_compile_backend", "inductor") or "inductor"),
        torch_compile_mode=str(getattr(config, "torch_compile_mode", "default") or "default"),
        torch_compile_dynamic=_boolish(getattr(config, "torch_compile_dynamic", False), default=False),
        torch_compile_fullgraph=_boolish(getattr(config, "torch_compile_fullgraph", False), default=False),
        torch_compile_scope=str(getattr(config, "torch_compile_scope", "") or ""),
        torch_compile_allow_full_with_per_block=_boolish(
            getattr(config, "torch_compile_allow_full_with_per_block", False),
            default=False,
        ),
        anima_compile_scope=str(getattr(config, "anima_compile_scope", "") or ""),
        compile_shape_strategy=_normalize_compile_shape_strategy(getattr(config, "compile_shape_strategy", "auto")),
        compile_target_strategy=_normalize_compile_target_strategy(getattr(config, "compile_target_strategy", "auto")),
        attention_split_chunks=_resolve_split_chunks(config),
        attention_early_deletion=_boolish(getattr(config, "attention_early_deletion", False), default=False),
        amd_sdpa_slice_trigger_gb=float(getattr(config, "amd_sdpa_slice_trigger_gb", 0.0) or 0.0),
        amd_sdpa_slice_target_gb=float(getattr(config, "amd_sdpa_slice_target_gb", 0.0) or 0.0),
        dynamo_recompile_limit=int(getattr(config, "dynamo_recompile_limit", 0) or 0),
        activation_memory_budget=float(getattr(config, "activation_memory_budget", 0.0) or 0.0),
        gradient_checkpointing=_boolish(getattr(config, "gradient_checkpointing", False), default=False),
    )

    resolved_shape_strategy, shape_reason = _resolve_compile_shape_strategy(
        config,
        route=route,
        compile_requested=requested_compile,
    )
    if resolved_shape_strategy != plan.compile_shape_strategy:
        plan.reasons.append(
            f"compile_shape_strategy={plan.compile_shape_strategy} resolved to {resolved_shape_strategy}: {shape_reason}"
        )
    elif shape_reason and shape_reason != "explicit shape strategy kept":
        plan.reasons.append(f"compile_shape_strategy={resolved_shape_strategy}: {shape_reason}")
    plan.compile_shape_strategy = resolved_shape_strategy

    if requested_attention == "auto":
        explicit_xformers = _boolish(getattr(config, "xformers", False), default=False)
        explicit_sdpa = _boolish(getattr(config, "sdpa", False), default=False) or _boolish(
            getattr(config, "use_sdpa", False), default=False
        )

        if route in {"anima", "newbie"}:
            # DiT routes can use the native attention patcher; prefer FA2,
            # then SageAttention, then SDPA.  xformers is a legacy UNet knob,
            # so only let it win when the user requested attention_backend
            # explicitly instead of leaving the backend on auto.
            if explicit_sdpa:
                plan.attention_backend = "sdpa"
                plan.reasons.append(f"auto attention selected sdpa for {route} because sdpa/use_sdpa=true")
            elif _importable("flash_attn"):
                plan.attention_backend = "flash2"
                plan.reasons.append(f"auto attention selected flash2 for {route} (DiT FA2-first policy)")
            elif _importable("sageattention"):
                plan.attention_backend = "sageattn"
                plan.reasons.append(f"auto attention selected sageattn for {route} (FA2 unavailable)")
            else:
                plan.attention_backend = "sdpa"
                plan.reasons.append(f"auto attention selected sdpa for {route} (FA2/sageattn unavailable)")
        elif explicit_xformers:
            plan.attention_backend = "xformers"
            plan.reasons.append("auto attention selected xformers because xformers=true")
        elif explicit_sdpa:
            plan.attention_backend = "sdpa"
            plan.reasons.append("auto attention selected sdpa because sdpa/use_sdpa=true")
        else:
            plan.attention_backend = "sdpa"
            plan.reasons.append("auto attention selected sdpa for U-Net/SDXL route")

    if plan.attention_backend == "xformers" and not _importable("xformers"):
        plan.warnings.append("xformers requested but package is unavailable, falling back to sdpa")
        plan.attention_backend = "sdpa"

    if plan.attention_backend in {"flash2", "sageattn", "spargeattn2"}:
        _pkg_map = {"flash2": "flash_attn", "sageattn": "sageattention", "spargeattn2": "spas_sage_attn"}
        mod = _pkg_map.get(plan.attention_backend, "")
        if not _importable(mod):
            plan.warnings.append(
                f"{plan.attention_backend} requested but package '{mod}' is not installed. "
                f"Falling back to sdpa."
            )
            plan.reasons.append(f"fa2_unavailable_reason=package_{mod}_not_installed")
            plan.attention_backend = "sdpa"

    if plan.attention_backend == "flexattn":
        if not _flexattention_runtime_active(config):
            plan.warnings.append(
                "flexattn requested outside the FlexAttention runtime; falling back to sdpa."
            )
            plan.attention_backend = "sdpa"
        elif not _flex_attention_available():
            plan.warnings.append(
                "flexattn requested but this PyTorch build does not provide "
                "torch.nn.attention.flex_attention.flex_attention. Falling back to sdpa."
            )
            plan.reasons.append("flexattn_unavailable_reason=torch_flex_attention_missing")
            plan.attention_backend = "sdpa"

    if plan.attention_backend not in _SUPPORTED_ATTENTION_BACKENDS:
        plan.warnings.append(f"unknown attention backend '{plan.attention_backend}', using torch fallback")
        plan.attention_backend = "torch"

    if requested_device == "mps":
        if plan.attention_backend != "sdpa":
            plan.reasons.append("Apple MPS runtime forces PyTorch SDPA attention.")
        plan.attention_backend = "sdpa"
        plan.sdpa_backend_policy = "math"
        if plan.torch_compile:
            plan.warnings.append("torch.compile disabled on Apple MPS runtime")
            plan.torch_compile = False

    if plan.torch_compile and not hasattr(torch, "compile"):
        plan.warnings.append("torch.compile requested but this PyTorch build does not provide torch.compile")
        plan.torch_compile = False

    if plan.torch_compile and _swap_config_enabled(config):
        plan.warnings.append(
            "torch.compile is incompatible with memory swap / blocks_to_swap; "
            "disabling swap to avoid compilation failures."
        )
        if hasattr(config, "blocks_to_swap"):
            config.blocks_to_swap = 0
        if hasattr(config, "swap_granularity"):
            config.swap_granularity = "off"
        if hasattr(config, "swap_count"):
            config.swap_count = 0
        if hasattr(config, "swap_ratio"):
            config.swap_ratio = 0.0

    # Promote anima_compile_scope to torch_compile when torch_compile_scope is empty
    if plan.anima_compile_scope and not plan.torch_compile_scope:
        if plan.anima_compile_scope == "per_block":
            plan.torch_compile = True
            plan.torch_compile_scope = "per_block"
            plan.reasons.append("anima_compile_scope=per_block promoted to torch_compile_scope=per_block")
        elif plan.anima_compile_scope == "full_cudagraph":
            plan.reasons.append("anima_compile_scope=full_cudagraph handled by training loop CUDAGraph path")

    if plan.torch_compile and route == "anima" and plan.torch_compile_scope == "per_block":
        fixed_text = int(getattr(config, "anima_fixed_text_tokens", 0) or 0)
        fixed_visual = int(getattr(config, "anima_fixed_visual_tokens", 0) or 0)
        native_bucket_compile = _boolish(
            getattr(config, "native_token_bucket_compile", True),
            default=True,
        ) and _boolish(getattr(config, "anima_cached_training", True), default=True)
        token_bucket_shape = plan.compile_shape_strategy in {"token_flatten", "native"}
        if fixed_text == 0 or (fixed_visual == 0 and not native_bucket_compile):
            plan.warnings.append(
                "Anima per_block compile requires fixed token budgets "
                "(anima_fixed_text_tokens plus anima_fixed_visual_tokens or no-pad token buckets) for static shapes; "
                "strict compile contract will disable Anima compile without them."
            )
        elif fixed_visual == 0 and native_bucket_compile:
            if token_bucket_shape:
                plan.reasons.append("Anima per_block compile will use token-count static shapes from no-pad cached visual token buckets")
            else:
                plan.reasons.append("Anima per_block compile will use no-pad cached visual token buckets")

    return plan


def build_sdpa_backend_context(plan: RuntimeOptimizationPlan):
    if getattr(plan, "attention_backend", "") != "sdpa":
        return nullcontext()

    policy = _normalize_sdpa_backend_policy(getattr(plan, "sdpa_backend_policy", "cutlass"))
    plan.sdpa_backend_policy = policy
    if policy == "auto":
        plan.reasons.append("sdpa backend policy=auto (PyTorch default dispatch)")
        return nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        plan.warnings.append(f"SDPA backend policy '{policy}' requested but torch.nn.attention is unavailable: {exc}")
        return nullcontext()

    mapping = {
        "cutlass": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
        "flash": [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH],
        "cudnn": [SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH],
        "math": [SDPBackend.MATH],
    }
    backends = mapping.get(policy)
    if not backends:
        plan.warnings.append(f"unknown SDPA backend policy '{policy}', using PyTorch default dispatch")
        return nullcontext()

    backend_names = ", ".join(str(item).split(".")[-1].lower() for item in backends)
    plan.reasons.append(f"sdpa backend policy={policy} (priority={backend_names})")
    try:
        return sdpa_kernel(backends, set_priority=True)
    except TypeError:
        try:
            return sdpa_kernel(backends)
        except Exception as exc:
            plan.warnings.append(
                f"failed to apply SDPA backend policy '{policy}', using PyTorch default dispatch: {exc}"
            )
            return nullcontext()
    except Exception as exc:
        plan.warnings.append(f"failed to apply SDPA backend policy '{policy}', using PyTorch default dispatch: {exc}")
        return nullcontext()


def _apply_attention_backend_once(model: Any, plan: RuntimeOptimizationPlan) -> None:
    unet = getattr(model, "unet", None)
    vae = getattr(model, "vae", None)

    if unet is None:
        return

    backend = plan.attention_backend
    split_chunks = int(plan.attention_split_chunks or 0)
    model_arch = str(getattr(model, "model_arch", "") or "").strip().lower()
    native_status = getattr(model, "native_unet_status", None) or getattr(unet, "native_unet_status", None)
    native_backend = ""
    if isinstance(native_status, dict):
        native_backend = str(native_status.get("backend") or "").strip().lower()
    elif native_status is not None:
        native_backend = str(getattr(native_status, "backend", "") or "").strip().lower()
    if model_arch == "sdxl" and native_backend == "lulynx_native":
        patched = 0
        for module in unet.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "attention_backend"):
                try:
                    object.__setattr__(config, "attention_backend", backend)
                    patched += 1
                except Exception:
                    pass
        plan.reasons.append(
            f"attention backend {backend} assigned to native SDXL attention modules ({patched} modules)"
        )
        logger.info("Enabled %s attention for native SDXL U-Net (%s modules patched)", backend, patched)
        return

    if backend in _DIFFUSERS_PROCESSOR_BACKENDS:
        if model_arch in _DIFFUSERS_UNET_MODEL_ARCHS:
            patched = _enable_diffusers_attention_backend(unet, plan)
            logger.info("Enabled %s attention for %s U-Net (%s modules patched)", backend, model_arch, patched)
        elif model_arch in _DIT_MODEL_ARCHS:
            from .anima_attention import patch_anima_attention

            patched = patch_anima_attention(
                unet,
                backend=backend,
                split_chunks=split_chunks,
                amd_sdpa_slice_trigger_gb=plan.amd_sdpa_slice_trigger_gb,
                amd_sdpa_slice_target_gb=plan.amd_sdpa_slice_target_gb,
                early_deletion=plan.attention_early_deletion,
            )
            if patched > 0:
                logger.info(f"Enabled {backend} attention for DiT ({patched} modules patched)")
                plan.reasons.append(f"attention backend {backend} patched {patched} DiT modules")
            else:
                raise RuntimeError(
                    f"{backend} requested, but no compatible DiT attention modules were patched "
                    f"for model_arch={model_arch or 'unknown'}"
                )
        else:
            raise RuntimeError(
                f"{backend} requested, but model_arch={model_arch or 'unknown'} is not wired "
                "for the native attention adapter"
            )
    elif backend in _DIT_PATCHED_BACKENDS:
        if model_arch not in _DIT_MODEL_ARCHS:
            raise RuntimeError(
                f"{backend} requested, but model_arch={model_arch or 'unknown'} is not a DiT route"
            )
        from .anima_attention import patch_anima_attention

        patched = patch_anima_attention(
            unet,
            backend=backend,
            split_chunks=split_chunks,
            amd_sdpa_slice_trigger_gb=plan.amd_sdpa_slice_trigger_gb,
            amd_sdpa_slice_target_gb=plan.amd_sdpa_slice_target_gb,
            early_deletion=plan.attention_early_deletion,
        )
        if patched > 0:
            logger.info(f"Enabled {backend} attention for DiT ({patched} modules patched)")
            plan.reasons.append(f"attention backend {backend} patched {patched} DiT modules")
        else:
            raise RuntimeError(
                f"{backend} requested, but no compatible attention modules were patched "
                f"for model_arch={model_arch or 'unknown'}"
            )
    elif backend == "sdpa":
        if model_arch in _DIFFUSERS_UNET_MODEL_ARCHS:
            patched = _enable_diffusers_attention_backend(unet, plan)
            logger.info("Enabled SDPA attention for %s U-Net via generic processor (%s modules patched)", model_arch, patched)
            return
        # If split_chunks>1, patch the DiT attention modules so the chunked
        # path runs even though the dispatch backend is sdpa. Falls through
        # cleanly for non-DiT models (no matching modules → no-op).
        if split_chunks > 1 or plan.attention_early_deletion:
            from .anima_attention import patch_anima_attention

            patched = patch_anima_attention(
                unet,
                backend="sdpa",
                split_chunks=split_chunks,
                amd_sdpa_slice_trigger_gb=plan.amd_sdpa_slice_trigger_gb,
                amd_sdpa_slice_target_gb=plan.amd_sdpa_slice_target_gb,
                early_deletion=plan.attention_early_deletion,
            )
            if patched > 0:
                reasons = []
                if split_chunks > 1:
                    reasons.append(f"split_chunks={split_chunks}")
                if plan.attention_early_deletion:
                    reasons.append("early_deletion=on")
                logger.info(
                    f"SDPA attention with {', '.join(reasons)} enabled "
                    f"({patched} DiT modules patched)"
                )
                plan.reasons.append(f"sdpa attention patched {patched} DiT modules ({', '.join(reasons)})")
        if hasattr(unet, "set_use_sdpa"):
            unet.set_use_sdpa(True)
            logger.info("Enabled SDPA attention for U-Net")
            plan.reasons.append("sdpa enabled via U-Net set_use_sdpa")
        else:
            if split_chunks <= 1:
                plan.warnings.append("UNet does not expose set_use_sdpa, keeping default torch attention")
                plan.attention_backend = "torch"
    else:
        logger.info("Using native torch attention backend")

    if vae is not None and plan.attention_backend == "xformers":
        try:
            encoder_mid = getattr(getattr(vae, "encoder", None), "mid_block", None)
            decoder_mid = getattr(getattr(vae, "decoder", None), "mid_block", None)
            if encoder_mid and getattr(encoder_mid, "attentions", None):
                encoder_mid.attentions[0].set_use_memory_efficient_attention_xformers(True)
            if decoder_mid and getattr(decoder_mid, "attentions", None):
                decoder_mid.attentions[0].set_use_memory_efficient_attention_xformers(True)
        except Exception as exc:
            plan.warnings.append(f"failed to enable xformers on VAE attention: {exc}")


def apply_attention_backend(model: Any, plan: RuntimeOptimizationPlan) -> None:
    try:
        _apply_attention_backend_once(model, plan)
    except Exception as exc:
        if _fallback_to_sdpa_attention(model, plan, str(exc)):
            logger.warning("Attention backend fallback applied: %s", exc)
            return
        raise


def apply_dynamo_budgets_if_requested(plan: RuntimeOptimizationPlan) -> None:
    """Apply recompile-limit pin and activation-memory budget before compiling.

    Idempotent: the pin only ever raises (max of current and requested) and the
    budget assignment is a plain module attr. Both knobs default to 0 = leave
    torch untouched. Must run BEFORE the torch.compile calls so the budgets are
    in place when the first graph (and its backward) is traced.
    """
    if plan.dynamo_recompile_limit > 0:
        try:
            from .dynamo_budget import pin_recompile_limit

            pin_recompile_limit(plan.dynamo_recompile_limit, log=plan.reasons.append)
        except Exception as exc:  # noqa: BLE001 - never block compile on a budget knob
            plan.warnings.append(f"dynamo recompile-limit pin failed: {exc}")
    if plan.activation_memory_budget > 0.0:
        try:
            from .dynamo_budget import apply_activation_memory_budget

            apply_activation_memory_budget(
                plan.activation_memory_budget,
                gradient_checkpointing=plan.gradient_checkpointing,
                log=plan.reasons.append,
            )
        except Exception as exc:  # noqa: BLE001
            plan.warnings.append(f"activation_memory_budget apply failed: {exc}")


def apply_torch_compile_if_requested(module: Any, plan: RuntimeOptimizationPlan, *, label: str) -> Any:
    if module is None or not plan.torch_compile:
        return module
    if plan.torch_compile_scope == "full_core":
        plan.reasons.append(f"skipped full torch.compile for {label} because torch_compile_scope=full_core")
        return module
    if plan.torch_compile_scope == "per_block" and not plan.torch_compile_allow_full_with_per_block:
        plan.reasons.append(f"skipped full torch.compile for {label} because torch_compile_scope=per_block")
        return module

    apply_dynamo_budgets_if_requested(plan)
    try:
        compiled = torch.compile(
            module,
            backend=plan.torch_compile_backend,
            mode=plan.torch_compile_mode,
            dynamic=plan.torch_compile_dynamic,
            fullgraph=plan.torch_compile_fullgraph,
        )
        logger.info(
            "torch.compile enabled for %s (backend=%s, mode=%s, dynamic=%s, fullgraph=%s)",
            label,
            plan.torch_compile_backend,
            plan.torch_compile_mode,
            plan.torch_compile_dynamic,
            plan.torch_compile_fullgraph,
        )
        return compiled
    except Exception as exc:
        plan.warnings.append(f"failed to compile {label}: {exc}")
        logger.warning("torch.compile failed for %s: %s", label, exc)
        return module


def apply_per_block_compile(
    model: Any,
    plan: RuntimeOptimizationPlan,
    *,
    route: str | None = None,
) -> None:
    """Compile each UNet/DiT block individually instead of the full model.

    Per-block compilation reduces recompilation overhead when block shapes
    vary (e.g., due to gradient checkpointing) and allows torch.compile
    to optimize each block's graph independently.

    Only takes effect when ``plan.torch_compile`` is True and
    ``plan.torch_compile_scope`` is ``"per_block"``.  When scope is ``""``
    or ``"full"``, the caller should use ``apply_torch_compile_if_requested``
    on the whole model instead.
    """
    if not plan.torch_compile or plan.torch_compile_scope != "per_block":
        return

    apply_dynamo_budgets_if_requested(plan)
    unet = getattr(model, "unet", None) or model
    compiled_count = 0
    route_name = str(route or _infer_compile_route(model)).strip().lower()
    _candidate_type, detect_compile_targets = _load_compile_target_detector()
    candidates = [
        candidate
        for candidate in detect_compile_targets(
            model,
            route=route_name,
            target_strategy=plan.compile_target_strategy,
        )
        if candidate.target_type != "full_core"
    ]
    if candidates:
        for candidate in candidates:
            if not candidate.eligible:
                plan.warnings.append(
                    f"per_block_compile: skipped {candidate.path}: {candidate.reason}"
                )
                logger.info("per_block_compile skipped %s: %s", candidate.path, candidate.reason)
                continue
            if _compile_detected_candidate(model, candidate, plan):
                compiled_count += 1

        if compiled_count > 0:
            plan.reasons.append(
                f"per_block_compile: detector compiled {compiled_count} targets route={route_name}"
            )
            logger.info("Per-block compile via detector: %d targets compiled", compiled_count)
        else:
            plan.warnings.append("per_block_compile: detector found candidates but no targets were compiled")
        return

    if route_name == "anima":
        plan.warnings.append("per_block_compile: no stable Anima block targets found by detector")
        return

    # Collect block-like submodules
    block_collections = []
    if hasattr(unet, "down_blocks"):
        block_collections.append(("down_blocks", unet.down_blocks))
    if hasattr(unet, "mid_block"):
        block_collections.append(("mid_block", [unet.mid_block]))
    if hasattr(unet, "up_blocks"):
        block_collections.append(("up_blocks", unet.up_blocks))
    # DiT architectures: net.blocks
    if hasattr(unet, "net") and hasattr(unet.net, "blocks"):
        block_collections.append(("net.blocks", unet.net.blocks))
    # Generic: _block_modules
    if not block_collections and hasattr(unet, "_block_modules"):
        fallback_source = unet._block_modules
        block_collections.append(
            ("_block_modules", fallback_source() if callable(fallback_source) else fallback_source)
        )

    if not block_collections:
        plan.warnings.append("per_block_compile: no block collections found in model")
        return

    for collection_name, blocks in block_collections:
        for i, block in enumerate(blocks):
            try:
                compiled = torch.compile(
                    block,
                    backend=plan.torch_compile_backend,
                    mode=plan.torch_compile_mode,
                    dynamic=plan.torch_compile_dynamic,
                    fullgraph=plan.torch_compile_fullgraph,
                )
                # Replace the block in the collection
                if isinstance(blocks, (list,)):
                    blocks[i] = compiled
                elif isinstance(blocks, nn.ModuleList):
                    blocks[i] = compiled
                else:
                    # ModuleList supports item assignment
                    try:
                        blocks[i] = compiled
                    except Exception:
                        plan.warnings.append(
                            f"per_block_compile: cannot replace {collection_name}[{i}]"
                        )
                        continue
                compiled_count += 1
            except Exception as exc:
                plan.warnings.append(
                    f"per_block_compile: failed to compile {collection_name}[{i}]: {exc}"
                )
                logger.warning("per_block_compile failed for %s[%d]: %s", collection_name, i, exc)

    if compiled_count > 0:
        plan.reasons.append(f"per_block_compile: compiled {compiled_count} blocks")
        logger.info("Per-block compile: %d blocks compiled", compiled_count)
    else:
        plan.warnings.append("per_block_compile: no blocks were compiled")


def _count_compiled_target_messages(reasons: Iterable[str]) -> int:
    count = 0
    prefix = "per_block_compile: compiled "
    for raw in reasons:
        reason = str(raw)
        if not reason.startswith(prefix):
            continue
        suffix = reason[len(prefix):]
        parts = suffix.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1] == "blocks":
            count += int(parts[0])
        else:
            count += 1
    return count


def build_compile_target_profile(
    model: Any,
    plan: RuntimeOptimizationPlan,
    *,
    route: str | None = None,
    reasons_start: int = 0,
    warnings_start: int = 0,
) -> dict[str, Any]:
    """Return a read-only profile for route-specific per-block compile."""

    route_name = str(route or _infer_compile_route(model)).strip().lower()
    _candidate_type, detect_compile_targets = _load_compile_target_detector()
    candidates = [
        candidate
        for candidate in detect_compile_targets(
            model,
            route=route_name,
            target_strategy=plan.compile_target_strategy,
        )
        if candidate.target_type != "full_core"
    ]
    start_reasons = max(int(reasons_start or 0), 0)
    start_warnings = max(int(warnings_start or 0), 0)
    new_reasons = list(getattr(plan, "reasons", []) or [])[start_reasons:]
    new_warnings = list(getattr(plan, "warnings", []) or [])[start_warnings:]
    compiled_targets = _count_compiled_target_messages(new_reasons)
    return {
        "route": route_name,
        "torch_compile": bool(getattr(plan, "torch_compile", False)),
        "torch_compile_scope": str(getattr(plan, "torch_compile_scope", "") or ""),
        "torch_compile_dynamic": bool(getattr(plan, "torch_compile_dynamic", False)),
        "compile_shape_strategy": str(getattr(plan, "compile_shape_strategy", "auto") or "auto"),
        "compile_target_strategy": str(getattr(plan, "compile_target_strategy", "auto") or "auto"),
        "candidate_targets": [candidate.path for candidate in candidates],
        "eligible_targets": sum(1 for candidate in candidates if candidate.eligible),
        "ineligible_targets": [
            {"path": candidate.path, "reason": candidate.reason}
            for candidate in candidates
            if not candidate.eligible
        ],
        "compiled_targets": compiled_targets,
        "applied": compiled_targets > 0,
        "reasons": new_reasons,
        "warnings": new_warnings,
    }


def apply_full_core_compile(
    model: Any,
    plan: RuntimeOptimizationPlan,
    *,
    route: str,
) -> bool:
    """Compile the route's stable core stack, leaving dynamic pre/post eager."""

    if not plan.torch_compile or plan.torch_compile_scope not in {"full", "full_core"}:
        return False

    apply_dynamo_budgets_if_requested(plan)
    route_name = str(route or "").strip().lower()
    _candidate_type, detect_compile_targets = _load_compile_target_detector()
    candidates = [
        candidate
        for candidate in detect_compile_targets(
            model,
            route=route_name,
            target_strategy=plan.compile_target_strategy,
        )
        if candidate.target_type == "full_core"
    ]
    if not candidates:
        plan.warnings.append(f"full_core_compile: no full-core target found for route={route_name}")
        return False

    candidate = candidates[0]
    if not candidate.eligible:
        plan.warnings.append(
            f"full_core_compile: skipped {candidate.path}: {candidate.reason}"
        )
        return False

    if _compile_detected_candidate(model, candidate, plan):
        plan.reasons.append(f"full_core_compile: compiled {candidate.path} route={route_name}")
        logger.info("Full-core compile enabled for %s target=%s", route_name, candidate.path)
        return True
    return False


def _infer_compile_route(model: Any) -> str:
    unet = getattr(model, "unet", None) or model
    if hasattr(unet, "down_blocks") or hasattr(unet, "up_blocks") or hasattr(unet, "mid_block"):
        return "sdxl"
    if hasattr(unet, "single_transformer_blocks"):
        return "flux"
    if hasattr(unet, "transformer_blocks"):
        return "newbie"
    if hasattr(unet, "blocks") or (hasattr(unet, "net") and hasattr(unet.net, "blocks")):
        return "anima"
    return "sdxl"


def _resolve_candidate_parent(root: Any, path: str) -> tuple[Any, str | int, Any]:
    parts = path.split(".")
    if not parts:
        raise AttributeError("empty compile target path")

    current = getattr(root, "unet", None) or root
    if parts[0] == "_run_blocks":
        return current, "_run_blocks", getattr(current, "_run_blocks")

    index_part = None
    for pos, part in enumerate(parts):
        if part.isdigit():
            index_part = pos
            break
        current = getattr(current, part)

    if index_part is None:
        parent = getattr(root, "unet", None) or root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        key = parts[-1]
        return parent, key, getattr(parent, key)

    index = int(parts[index_part])
    block = current[index]
    if index_part == len(parts) - 1:
        return current, index, block

    attr = parts[index_part + 1]
    return block, attr, getattr(block, attr)


def _compile_detected_candidate(
    model: Any,
    candidate: Any,
    plan: RuntimeOptimizationPlan,
) -> bool:
    try:
        parent, key, target = _resolve_candidate_parent(model, candidate.path)
        compiled = torch.compile(
            target,
            backend=plan.torch_compile_backend,
            mode=plan.torch_compile_mode,
            dynamic=plan.torch_compile_dynamic,
            fullgraph=plan.torch_compile_fullgraph,
        )
        if isinstance(target, nn.Module) and not isinstance(compiled, nn.Module):
            target.forward = compiled
            plan.reasons.append(f"per_block_compile: compiled {candidate.path}.forward")
            return True
        if isinstance(key, int):
            parent[key] = compiled
        else:
            setattr(parent, key, compiled)
        plan.reasons.append(f"per_block_compile: compiled {candidate.path}")
        return True
    except Exception as exc:
        plan.warnings.append(f"per_block_compile: failed to compile {candidate.path}: {exc}")
        logger.warning("per_block_compile failed for %s: %s", candidate.path, exc)
        return False


def _load_compile_target_detector() -> tuple[Any, Any]:
    try:
        from .compile_target_detector import CompileTargetCandidate, detect_compile_targets

        return CompileTargetCandidate, detect_compile_targets
    except Exception:
        # Some smoke tests load this file directly by path under a synthetic
        # module name.  Avoid importing package __init__ in that mode.
        import importlib.util
        import sys
        from pathlib import Path

        module_name = "_lulynx_compile_target_detector_standalone"
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing.CompileTargetCandidate, existing.detect_compile_targets
        path = Path(__file__).with_name("compile_target_detector.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.CompileTargetCandidate, module.detect_compile_targets


# ═══════════════════════════════════════════════════════════════════════════
# Experimental Attention Profile — sliding-window / chunked attention
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AttentionProfile:
    """Configuration for experimental attention profiles."""
    enabled: bool = False
    window_size: int = 0  # 0 = full attention, >0 = sliding window
    backend: str = "auto"
    torch_fallback_max_tokens: int = 2048
    launcher_attention_backend: str = "auto"
    flex_runtime_active: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "AttentionProfile":
        return cls(
            enabled=bool(getattr(config, "experimental_attention_profile_enabled", False)),
            window_size=int(getattr(config, "experimental_attention_profile_window", 0) or 0),
            backend=_normalize_sliding_window_backend(
                getattr(config, "experimental_attention_profile_backend", "auto")
            ),
            torch_fallback_max_tokens=max(
                int(getattr(config, "experimental_attention_profile_torch_max_tokens", 2048) or 2048),
                1,
            ),
            launcher_attention_backend=_normalize_attention_backend(
                getattr(config, "attention_backend", "auto")
            ),
            flex_runtime_active=_flexattention_runtime_active(config),
        )

    @property
    def is_active(self) -> bool:
        return self.enabled and self.window_size > 0


_SLIDING_WINDOW_BACKEND_ALIASES = {
    "": "auto",
    "auto": "auto",
    "default": "auto",
    "flex": "flex",
    "flexattn": "flex",
    "flexattention": "flex",
    "sdpa": "sdpa_masked",
    "sdpa_masked": "sdpa_masked",
    "torch": "torch_fallback",
    "torch_fallback": "torch_fallback",
    "legacy": "torch_fallback",
}
_SUPPORTED_SLIDING_WINDOW_BACKENDS = {"auto", "flex", "sdpa_masked", "torch_fallback"}


def _normalize_sliding_window_backend(value: Any) -> str:
    normalized = _SLIDING_WINDOW_BACKEND_ALIASES.get(str(value or "").strip().lower(), None)
    if normalized is None:
        return "auto"
    return normalized if normalized in _SUPPORTED_SLIDING_WINDOW_BACKENDS else "auto"


def _sliding_window_backend_from_attention_backend(value: Any) -> Optional[str]:
    attention_backend = _normalize_attention_backend(value)
    if attention_backend == "flexattn":
        return "flex"
    if attention_backend == "sdpa":
        return "sdpa_masked"
    if attention_backend == "torch":
        return "torch_fallback"
    return None


def resolve_sliding_window_backend(
    query: torch.Tensor,
    requested: Any = "auto",
    *,
    launcher_attention_backend: Any = "auto",
    flex_runtime_active: bool = False,
) -> str:
    """Resolve the sliding-window attention backend for the current tensor/device."""
    backend = _normalize_sliding_window_backend(requested)
    if backend != "auto":
        return backend

    launcher_backend = _sliding_window_backend_from_attention_backend(launcher_attention_backend)
    if launcher_backend is not None:
        return launcher_backend

    if query.is_cuda and flex_runtime_active and _flex_attention_available():
        return "flex"
    return "sdpa_masked"


def _sliding_window_mask(
    query_len: int,
    key_len: int,
    window_size: int,
    device: torch.device,
    *,
    causal: bool = True,
) -> torch.Tensor:
    query_positions = torch.arange(query_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    distance = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
    if not causal:
        return distance.abs() < window_size
    return (distance >= 0) & (distance < window_size)


def _sliding_window_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    scale: Optional[float],
    *,
    causal: bool = True,
) -> torch.Tensor:
    batch, heads, seq_len, head_dim = query.shape
    if scale is None:
        scale = head_dim ** -0.5
    mask = _sliding_window_mask(seq_len, int(key.shape[-2]), window_size, query.device, causal=causal)
    attn_bias = torch.where(mask, 0.0, float("-inf")).to(dtype=query.dtype, device=query.device)
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale + attn_bias
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, value)


def _sliding_window_attention_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    scale: Optional[float],
    *,
    causal: bool = True,
) -> torch.Tensor:
    seq_len = int(query.shape[-2])
    mask = _sliding_window_mask(seq_len, int(key.shape[-2]), window_size, query.device, causal=causal)
    try:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            scale=scale,
        )
    except TypeError:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
        )


def _sliding_window_attention_flex(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    scale: Optional[float],
    *,
    causal: bool = True,
) -> torch.Tensor:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def sliding_mask(_batch, _head, q_idx, kv_idx):
        distance = q_idx - kv_idx
        if not causal:
            return distance.abs() < window_size
        return (distance >= 0) & (distance < window_size)

    block_mask = create_block_mask(
        sliding_mask,
        B=None,
        H=None,
        Q_LEN=int(query.shape[-2]),
        KV_LEN=int(key.shape[-2]),
        device=query.device,
    )
    out = flex_attention(query, key, value, block_mask=block_mask, scale=scale)
    return out[0] if isinstance(out, tuple) else out


def sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    scale: Optional[float] = None,
    backend: str = "auto",
    torch_fallback_max_tokens: int = 2048,
    launcher_attention_backend: str = "auto",
    flex_runtime_active: bool = False,
    causal: bool = True,
) -> torch.Tensor:
    """Compute causal sliding-window attention through a selectable backend.

    ``flex`` is the preferred long-sequence route. ``sdpa_masked`` is the
    compatibility route. ``torch_fallback`` materializes an ``n x n`` mask and
    is guarded so it cannot silently handle large sequences.

    Args:
        query: (batch, heads, seq_len, head_dim)
        key:   (batch, heads, seq_len, head_dim)
        value: (batch, heads, seq_len, head_dim)
        window_size: number of tokens each query attends to

    Returns:
        Attention output with same shape as query.
    """
    seq_len = int(query.shape[-2])
    resolved = resolve_sliding_window_backend(
        query,
        backend,
        launcher_attention_backend=launcher_attention_backend,
        flex_runtime_active=flex_runtime_active,
    )
    if resolved == "flex":
        if not _flex_attention_available():
            logger.warning("sliding_window_attention flex backend unavailable; falling back to sdpa_masked")
            resolved = "sdpa_masked"
        else:
            try:
                return _sliding_window_attention_flex(query, key, value, window_size, scale, causal=causal)
            except Exception as exc:
                logger.warning("sliding_window_attention flex backend failed; falling back to sdpa_masked: %s", exc)
                resolved = "sdpa_masked"

    if resolved == "sdpa_masked":
        return _sliding_window_attention_sdpa(query, key, value, window_size, scale, causal=causal)

    limit = max(int(torch_fallback_max_tokens or 2048), 1)
    if seq_len > limit:
        raise RuntimeError(
            "sliding_window_attention torch_fallback would materialize an O(n^2) mask "
            f"for seq_len={seq_len}; choose backend='flex'/'sdpa_masked' or raise "
            f"experimental_attention_profile_torch_max_tokens above {limit}."
        )
    return _sliding_window_attention_torch(query, key, value, window_size, scale, causal=causal)


def apply_attention_profile(
    config: Any,
    model: Any,
    plan: RuntimeOptimizationPlan,
) -> None:
    """Apply experimental attention profile settings to the model.

    If ``experimental_attention_profile_enabled`` is True and
    ``experimental_attention_profile_window`` > 0, attaches the profile
    metadata to the model so the training loop can use it.
    """
    profile = AttentionProfile.from_config(config)
    if not profile.is_active:
        return

    # Attach profile metadata to the model for the training loop to consume
    profile.launcher_attention_backend = getattr(plan, "attention_backend", profile.launcher_attention_backend)
    if profile.backend == "auto":
        mapped = _sliding_window_backend_from_attention_backend(profile.launcher_attention_backend)
        if mapped is None and profile.launcher_attention_backend not in {"auto", "sdpa", "torch", "flexattn"}:
            plan.warnings.append(
                "experimental_attention_profile: resolved attention backend "
                f"'{profile.launcher_attention_backend}' does not expose a sliding-window route here; "
                "falling back through flex/sdpa for the windowed profile."
            )
    model._attention_profile = profile
    plan.reasons.append(
        f"experimental_attention_profile: window_size={profile.window_size}, backend={profile.backend}"
    )
    logger.info(
        "Attention profile: sliding window enabled, window_size=%d, backend=%s",
        profile.window_size,
        profile.backend,
    )


def apply_sdxl_attention_profile(
    config: Any,
    model: Any,
    plan: RuntimeOptimizationPlan,
    profile: Optional[AttentionProfile] = None,
) -> int:
    """Install Diffusers U-Net self-attention processors for the attention profile."""
    profile = profile or AttentionProfile.from_config(config)
    if not profile.is_active:
        return 0

    route = _runtime_route(config)
    model_route = str(getattr(model, "model_arch", "") or "").strip().lower()
    if model_route in _DIFFUSERS_UNET_MODEL_ARCHS:
        route = model_route
    route_label = route.upper() if route in _DIFFUSERS_UNET_MODEL_ARCHS else "Diffusers"

    unet = getattr(model, "unet", None) or model
    if unet is None:
        plan.warnings.append(f"experimental_attention_profile: {route_label} U-Net is unavailable")
        return 0

    native_status = getattr(model, "native_unet_status", None) or getattr(unet, "native_unet_status", None)
    native_backend = ""
    if isinstance(native_status, dict):
        native_backend = str(native_status.get("backend") or "").strip().lower()
    elif native_status is not None:
        native_backend = str(getattr(native_status, "backend", "") or "").strip().lower()
    if native_backend == "lulynx_native":
        plan.warnings.append(
            f"experimental_attention_profile: native {route_label} U-Net does not expose diffusers attention processors yet"
        )
        return 0

    from .diffusers_attention import (
        SlidingWindowDiffusersAttentionKernel,
        build_diffusers_attention_kernel,
        install_diffusers_attention_processor,
    )

    profile.launcher_attention_backend = str(
        getattr(plan, "attention_backend", profile.launcher_attention_backend) or profile.launcher_attention_backend
    )
    fallback_backend = profile.launcher_attention_backend
    if fallback_backend in {"", "auto", "flex", "flexattn"}:
        fallback_backend = "sdpa"
    try:
        fallback_kernel = build_diffusers_attention_kernel(fallback_backend)
    except Exception as exc:
        plan.warnings.append(
            "experimental_attention_profile: failed to build cross-attention fallback "
            f"backend '{fallback_backend}', using sdpa: {exc}"
        )
        fallback_kernel = build_diffusers_attention_kernel("sdpa")

    kernel = SlidingWindowDiffusersAttentionKernel(profile, fallback_kernel=fallback_kernel)
    patched = install_diffusers_attention_processor(unet, kernel)
    plan.reasons.append(
        f"experimental_attention_profile: installed {route_label} sliding-window self-attention "
        f"processor on {patched} Diffusers U-Net attention modules "
        f"(window_size={profile.window_size}, backend={profile.backend}, fallback={fallback_kernel.backend_id})"
    )
    logger.info(
        "%s attention profile: installed sliding-window processor on %d Diffusers U-Net modules, window_size=%d, backend=%s",
        route_label,
        patched,
        profile.window_size,
        profile.backend,
    )
    return patched


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Attention Fused KV — fuse K/V projections to save memory
# ═══════════════════════════════════════════════════════════════════════════

class FusedKVProjection(nn.Module):
    """Fused key-value projection that shares a single linear layer for K and V.

    Instead of separate K and V projections, this module computes a single
    projection and splits the output.  This reduces parameter count and
    memory bandwidth for cross-attention layers.
    """

    def __init__(self, embed_dim: int, kv_dim: int, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        # Single projection for both K and V
        self.kv_proj = nn.Linear(embed_dim, kv_dim * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        return k, v


_FUSED_PROJECTION_MEMORY_MODES = {"keep_original", "drop_original", "materialize_on_save"}


def normalize_fused_projection_memory_mode(raw: Any) -> str:
    value = str(raw or "keep_original").strip().lower().replace("-", "_")
    aliases = {
        "": "keep_original",
        "auto": "keep_original",
        "keep": "keep_original",
        "compat": "keep_original",
        "compatible": "keep_original",
        "drop": "drop_original",
        "delete": "drop_original",
        "save": "materialize_on_save",
        "materialize": "materialize_on_save",
        "materialize_save": "materialize_on_save",
    }
    value = aliases.get(value, value)
    return value if value in _FUSED_PROJECTION_MEMORY_MODES else "keep_original"


def _register_fused_projection_state_dict_hook(
    module: nn.Module,
    fused_attr: str,
    original_attrs: tuple[str, ...],
    fused_weight_name: str,
    chunk_count: int,
) -> None:
    if getattr(module, "_fused_projection_state_dict_hook", None) is not None:
        return

    def _hook(mod: nn.Module, state_dict: dict[str, torch.Tensor], prefix: str, _local_metadata: dict[str, Any]) -> None:
        fused = getattr(mod, fused_attr, None)
        if fused is None:
            return
        fused_linear = getattr(fused, fused_weight_name, None)
        if fused_linear is None or not hasattr(fused_linear, "weight"):
            return
        weight_chunks = fused_linear.weight.detach().chunk(chunk_count, dim=0)
        bias_chunks = (
            fused_linear.bias.detach().chunk(chunk_count, dim=0)
            if getattr(fused_linear, "bias", None) is not None
            else None
        )
        for index, attr in enumerate(original_attrs):
            state_dict.setdefault(prefix + attr + ".weight", weight_chunks[index].clone())
            if bias_chunks is not None:
                state_dict.setdefault(prefix + attr + ".bias", bias_chunks[index].clone())

    module._fused_projection_state_dict_hook = module.register_state_dict_post_hook(_hook)


def _apply_fused_projection_memory_mode(
    module: nn.Module,
    *,
    mode: str,
    fused_attr: str,
    original_attrs: tuple[str, ...],
    fused_weight_name: str,
    chunk_count: int,
) -> str:
    resolved = normalize_fused_projection_memory_mode(mode)
    if resolved == "keep_original":
        module._fused_projection_memory_mode = resolved
        return resolved
    if resolved == "materialize_on_save":
        _register_fused_projection_state_dict_hook(
            module,
            fused_attr,
            original_attrs,
            fused_weight_name,
            chunk_count,
        )
    for attr in original_attrs:
        if hasattr(module, attr):
            setattr(module, attr, None)
    module._fused_projection_memory_mode = resolved
    return resolved


def apply_cross_attn_fused_kv(
    config: Any,
    model: Any,
    plan: RuntimeOptimizationPlan,
) -> None:
    """Apply fused KV projection to cross-attention layers if requested.

    When ``cross_attn_fused_kv`` is True, replaces separate K and V linear
    projections in cross-attention layers with a single fused projection.

    This is a Warehouse implementation that works with standard PyTorch
    attention patterns (no custom CUDA kernels required).
    """
    if not bool(getattr(config, "cross_attn_fused_kv", False)):
        return

    unet = getattr(model, "unet", None) or model
    fused_count = 0
    memory_mode = normalize_fused_projection_memory_mode(
        getattr(config, "fused_projection_memory_mode", "keep_original")
    )
    dropped_count = 0

    # Walk the model looking for cross-attention modules with separate K/V
    for name, module in unet.named_modules():
        if not hasattr(module, "attn2"):
            continue
        attn = module.attn2
        # Check for separate K and V projections (standard diffusers pattern)
        if not (hasattr(attn, "to_k") and hasattr(attn, "to_v")):
            continue

        to_k = attn.to_k
        to_v = attn.to_v

        if not isinstance(to_k, nn.Linear) or not isinstance(to_v, nn.Linear):
            continue
        if to_k.in_features != to_v.in_features:
            continue

        # Create fused projection
        fused = FusedKVProjection(
            embed_dim=to_k.in_features,
            kv_dim=to_k.out_features,
            bias=to_k.bias is not None,
        )

        # Copy weights: [K; V] → single projection
        with torch.no_grad():
            fused.kv_proj.weight.copy_(torch.cat([to_k.weight, to_v.weight], dim=0))
            if to_k.bias is not None and to_v.bias is not None:
                fused.kv_proj.bias.copy_(torch.cat([to_k.bias, to_v.bias], dim=0))

        fused = fused.to(device=to_k.weight.device, dtype=to_k.weight.dtype)

        # Replace the module
        attn._fused_kv = fused
        resolved_mode = _apply_fused_projection_memory_mode(
            attn,
            mode=memory_mode,
            fused_attr="_fused_kv",
            original_attrs=("to_k", "to_v"),
            fused_weight_name="kv_proj",
            chunk_count=2,
        )
        if resolved_mode != "keep_original":
            dropped_count += 1
        fused_count += 1

    if fused_count > 0:
        plan.reasons.append(
            f"cross_attn_fused_kv: fused {fused_count} cross-attention K/V pairs "
            f"(memory_mode={memory_mode}, dropped_original={dropped_count})"
        )
        logger.info("Cross-attention fused KV: %d layer pairs fused", fused_count)
    else:
        plan.warnings.append("cross_attn_fused_kv requested but no fusable cross-attention layers found")


class FusedQKVProjection(nn.Module):
    """Fused query-key-value projection for self-attention.

    Merges three separate Q/K/V linear projections into a single
    ``nn.Linear(in, 3*out)`` and splits the result.  Reduces kernel
    launch overhead for self-attention layers where Q/K/V share the
    same input.
    """

    def __init__(self, in_dim: int, proj_dim: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.qkv_proj = nn.Linear(in_dim, proj_dim * 3, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v


def _is_lora_wrapped(module: nn.Module) -> bool:
    """Check if a Linear has been wrapped by LoRA injection."""
    cls_name = type(module).__name__
    return cls_name in (
        "LoRALinear", "DoRALinear", "LoRAFALinear", "VeRALinear",
        "TLoRALinear", "LoConModule", "LoHaModule", "LoKrModule",
        "FeRALinear",
    ) or hasattr(module, "lora_down")


def apply_anima_fused_kv(
    model: Any,
    plan: RuntimeOptimizationPlan,
    memory_mode: str = "keep_original",
) -> int:
    """Fuse K/V projections in Anima DiT cross-attention layers.

    Only applies to modules whose name contains ``cross_attn`` and
    whose ``k_proj`` / ``v_proj`` are plain ``nn.Linear`` (not LoRA-wrapped).
    """
    target = getattr(model, "unet", None) or getattr(model, "dit", None) or model
    fused_count = 0
    skipped_lora = 0
    memory_mode = normalize_fused_projection_memory_mode(memory_mode)
    dropped_count = 0

    for name, module in target.named_modules():
        if "cross_attn" not in name:
            continue
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        if k_proj is None or v_proj is None:
            continue
        if not isinstance(k_proj, nn.Linear) or not isinstance(v_proj, nn.Linear):
            if _is_lora_wrapped(k_proj) or _is_lora_wrapped(v_proj):
                skipped_lora += 1
            continue
        if _is_lora_wrapped(k_proj) or _is_lora_wrapped(v_proj):
            skipped_lora += 1
            continue
        if k_proj.in_features != v_proj.in_features:
            continue

        fused = FusedKVProjection(
            embed_dim=k_proj.in_features,
            kv_dim=k_proj.out_features,
            bias=k_proj.bias is not None,
        )
        with torch.no_grad():
            fused.kv_proj.weight.copy_(torch.cat([k_proj.weight, v_proj.weight], dim=0))
            if k_proj.bias is not None and v_proj.bias is not None:
                fused.kv_proj.bias.copy_(torch.cat([k_proj.bias, v_proj.bias], dim=0))

        fused = fused.to(device=k_proj.weight.device, dtype=k_proj.weight.dtype)
        module._fused_kv = fused
        resolved_mode = _apply_fused_projection_memory_mode(
            module,
            mode=memory_mode,
            fused_attr="_fused_kv",
            original_attrs=("k_proj", "v_proj"),
            fused_weight_name="kv_proj",
            chunk_count=2,
        )
        if resolved_mode != "keep_original":
            dropped_count += 1
        fused_count += 1

    if fused_count > 0:
        plan.reasons.append(
            f"anima_fused_kv: fused {fused_count} cross-attention K/V pairs "
            f"(memory_mode={memory_mode}, dropped_original={dropped_count})"
        )
        logger.info("Anima fused KV: %d cross-attention pairs fused", fused_count)
    if skipped_lora > 0:
        plan.warnings.append(f"anima_fused_kv: skipped {skipped_lora} LoRA-wrapped K/V pairs")
        logger.info("Anima fused KV: skipped %d LoRA-wrapped pairs", skipped_lora)
    return fused_count


def apply_anima_fused_qkv(
    model: Any,
    plan: RuntimeOptimizationPlan,
    memory_mode: str = "keep_original",
) -> int:
    """Fuse Q/K/V projections in Anima DiT self-attention layers.

    Only applies to modules whose name contains ``self_attn`` and
    whose ``q_proj``, ``k_proj``, ``v_proj`` are plain ``nn.Linear``
    with matching ``in_features``.
    """
    target = getattr(model, "unet", None) or getattr(model, "dit", None) or model
    fused_count = 0
    skipped_lora = 0
    memory_mode = normalize_fused_projection_memory_mode(memory_mode)
    dropped_count = 0

    for name, module in target.named_modules():
        if "self_attn" not in name:
            continue
        q_proj = getattr(module, "q_proj", None)
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        if q_proj is None or k_proj is None or v_proj is None:
            continue
        if not (isinstance(q_proj, nn.Linear) and isinstance(k_proj, nn.Linear) and isinstance(v_proj, nn.Linear)):
            if _is_lora_wrapped(q_proj) or _is_lora_wrapped(k_proj) or _is_lora_wrapped(v_proj):
                skipped_lora += 1
            continue
        if _is_lora_wrapped(q_proj) or _is_lora_wrapped(k_proj) or _is_lora_wrapped(v_proj):
            skipped_lora += 1
            continue
        if not (q_proj.in_features == k_proj.in_features == v_proj.in_features):
            continue

        fused = FusedQKVProjection(
            in_dim=q_proj.in_features,
            proj_dim=q_proj.out_features,
            bias=q_proj.bias is not None,
        )
        with torch.no_grad():
            fused.qkv_proj.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
            if q_proj.bias is not None and k_proj.bias is not None and v_proj.bias is not None:
                fused.qkv_proj.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))

        fused = fused.to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)
        module._fused_qkv = fused
        resolved_mode = _apply_fused_projection_memory_mode(
            module,
            mode=memory_mode,
            fused_attr="_fused_qkv",
            original_attrs=("q_proj", "k_proj", "v_proj"),
            fused_weight_name="qkv_proj",
            chunk_count=3,
        )
        if resolved_mode != "keep_original":
            dropped_count += 1
        fused_count += 1

    if fused_count > 0:
        plan.reasons.append(
            f"anima_fused_qkv: fused {fused_count} self-attention Q/K/V triples "
            f"(memory_mode={memory_mode}, dropped_original={dropped_count})"
        )
        logger.info("Anima fused QKV: %d self-attention triples fused", fused_count)
    if skipped_lora > 0:
        plan.warnings.append(f"anima_fused_qkv: skipped {skipped_lora} LoRA-wrapped Q/K/V triples")
        logger.info("Anima fused QKV: skipped %d LoRA-wrapped triples", skipped_lora)
    return fused_count


@dataclass
class FusedOptimizerConfig:
    """Configuration for blockwise fused optimizer updates."""
    enabled: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "FusedOptimizerConfig":
        return cls(
            enabled=bool(getattr(config, "blockwise_fused_optimizers", False)),
        )


def apply_blockwise_fused_optimizers(
    config: Any,
    optimizer: Any,
    plan: RuntimeOptimizationPlan,
) -> bool:
    """Apply blockwise fused optimizer optimization if requested.

    When ``blockwise_fused_optimizers`` is True, groups optimizer parameter
    groups by block to improve memory locality during optimizer.step().

    Returns True if optimization was applied.
    """
    fuse_cfg = FusedOptimizerConfig.from_config(config)
    if not fuse_cfg.enabled:
        return False

    if optimizer is None:
        return False

    # Ensure optimizer state is contiguous per block
    # This is a no-op for most optimizers but signals to the training loop
    # that blockwise update ordering is preferred.
    optimizer._blockwise_fused = True
    plan.reasons.append("blockwise_fused_optimizers: optimizer marked for blockwise updates")
    logger.info("Blockwise fused optimizers enabled")
    return True


