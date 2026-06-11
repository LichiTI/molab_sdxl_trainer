from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import math
import os
import re
from typing import Any, Dict

import torch


_AMD_RUNTIME_IDS = {"rocm-amd", "rocm-amd-sage2"}
_AMD_SAGE2_RUNTIME_IDS = {"rocm-amd-sage2"}
_AMD_SAGE2_GFX_PREFIXES = ("gfx11", "gfx12")
_AMD_UNSAFE_OPTIMIZERS = {
    "adamw8bit",
    "lion8bit",
    "sgdnesterov8bit",
    "pagedadamw",
    "pagedadamw8bit",
    "pagedadamw32bit",
    "pagedlion8bit",
    "pagedlion32bit",
}

_AMD_UNSUPPORTED_SCHEMA_PREFIXES = ("flux-", "lumina-")


@dataclass(frozen=True)
class AmdRuntimeProbe:
    runtime_id: str
    hip_version: str = ""
    bf16_supported: bool = False
    selected_gpu_name: str = ""
    selected_gpu_arch: str = ""
    selected_gpu_memory_mb: int = 0
    gpu_count: int = 0
    gpu_summary: str = ""
    runtime_experimental: bool = False


@dataclass(frozen=True)
class AmdRuntimeProfile:
    runtime_profile_name: str
    empty_cache_interval: int
    sdpa_slice_trigger_gb: float
    sdpa_slice_target_gb: float
    dataloader_num_workers: int
    persistent_data_loader_workers: bool


@dataclass
class AmdRuntimeGuardResult:
    runtime_id: str
    hip_version: str = ""
    bf16_supported: bool = False
    selected_gpu_name: str = ""
    selected_gpu_arch: str = ""
    selected_gpu_memory_mb: int = 0
    gpu_summary: str = ""
    runtime_profile_name: str = ""
    empty_cache_interval: int = 0
    sdpa_slice_trigger_gb: float = 0.0
    sdpa_slice_target_gb: float = 0.0
    forced_overrides: Dict[str, Any] = field(default_factory=dict)
    disabled_features: Dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    route_supported: bool = True
    route_reason: str = ""

    @property
    def is_amd(self) -> bool:
        return self.runtime_id in _AMD_RUNTIME_IDS

    @property
    def is_amd_sage2(self) -> bool:
        return self.runtime_id in _AMD_SAGE2_RUNTIME_IDS


def detect_runtime_id(config: Any) -> str:
    return str(
        getattr(config, "runtime_id", "")
        or getattr(config, "execution_profile_id", "")
        or ""
    ).strip().lower()


def is_amd_runtime(config: Any) -> bool:
    return detect_runtime_id(config) in _AMD_RUNTIME_IDS


def _normalize_gfx_arch(value: Any) -> str:
    raw = str(value or "").strip().lower()
    match = re.search(r"gfx\d+", raw)
    return match.group(0) if match else raw


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _is_sage2_gfx_supported(gfx_arch: str) -> bool:
    normalized = _normalize_gfx_arch(gfx_arch)
    return any(normalized.startswith(prefix) for prefix in _AMD_SAGE2_GFX_PREFIXES)


def build_amd_runtime_probe(config: Any) -> AmdRuntimeProbe:
    runtime_id = detect_runtime_id(config)
    hip_version = str(getattr(getattr(torch, "version", None), "hip", "") or "")
    gpu_name = ""
    gpu_arch = ""
    gpu_memory_mb = 0
    gpu_count = 0
    bf16_supported = False
    if torch.cuda.is_available():
        try:
            gpu_count = int(torch.cuda.device_count())
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(getattr(props, "name", "") or "")
            gpu_arch = _normalize_gfx_arch(
                getattr(props, "gcnArchName", "")
                or getattr(props, "gfx_arch", "")
                or getattr(props, "arch", "")
            )
            gpu_memory_mb = int(getattr(props, "total_memory", 0) // (1024 * 1024))
        except Exception:
            gpu_name = ""
            gpu_memory_mb = 0
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
    if not gpu_arch:
        gpu_arch = _normalize_gfx_arch(
            getattr(config, "amd_gfx_arch", "")
            or os.environ.get("LULYNX_AMD_GFX_ARCH", "")
            or os.environ.get("AMD_GFX_ARCH", "")
        )
    gpu_summary = gpu_name or ("CUDA/HIP device" if torch.cuda.is_available() else "No GPU detected")
    if gpu_arch:
        gpu_summary = f"{gpu_summary} [{gpu_arch}]"
    if gpu_memory_mb > 0:
        gpu_summary = f"{gpu_summary} ({gpu_memory_mb} MiB)"
    return AmdRuntimeProbe(
        runtime_id=runtime_id,
        hip_version=hip_version,
        bf16_supported=bf16_supported,
        selected_gpu_name=gpu_name,
        selected_gpu_arch=gpu_arch,
        selected_gpu_memory_mb=gpu_memory_mb,
        gpu_count=gpu_count,
        gpu_summary=gpu_summary,
        runtime_experimental=(runtime_id in _AMD_RUNTIME_IDS),
    )


def _profile_from_memory(memory_mb: int) -> AmdRuntimeProfile:
    memory_gb = memory_mb / 1024 if memory_mb > 0 else 0.0
    if memory_gb >= 20:
        return AmdRuntimeProfile("amd-rocm-large", 50, 2.5, 1.25, 2, False)
    if memory_gb >= 12:
        return AmdRuntimeProfile("amd-rocm-balanced", 25, 1.5, 0.75, 1, False)
    return AmdRuntimeProfile("amd-rocm-safe", 10, 0.75, 0.35, 0, False)


def _route_supported(schema_id: str) -> tuple[bool, str]:
    normalized = str(schema_id or "").strip().lower()
    for prefix in _AMD_UNSUPPORTED_SCHEMA_PREFIXES:
        if normalized.startswith(prefix):
            return False, (
                "AMD ROCm experimental runtime does not currently expose this training route. "
                "Use Anima / SDXL / Newbie / SD routes instead."
            )
    return True, ""


def build_amd_runtime_guard(config: Any) -> AmdRuntimeGuardResult:
    probe = build_amd_runtime_probe(config)
    profile = _profile_from_memory(probe.selected_gpu_memory_mb)
    schema_id = str(getattr(config, "schema_id", "") or "")
    route_supported, route_reason = _route_supported(schema_id)
    result = AmdRuntimeGuardResult(
        runtime_id=probe.runtime_id,
        hip_version=probe.hip_version,
        bf16_supported=probe.bf16_supported,
        selected_gpu_name=probe.selected_gpu_name,
        selected_gpu_arch=probe.selected_gpu_arch,
        selected_gpu_memory_mb=probe.selected_gpu_memory_mb,
        gpu_summary=probe.gpu_summary,
        runtime_profile_name=profile.runtime_profile_name,
        empty_cache_interval=profile.empty_cache_interval,
        sdpa_slice_trigger_gb=profile.sdpa_slice_trigger_gb,
        sdpa_slice_target_gb=profile.sdpa_slice_target_gb,
        route_supported=route_supported,
        route_reason=route_reason,
    )
    if not result.is_amd:
        return result

    result.notes.append("AMD ROCm runtime is treated as experimental.")
    if probe.hip_version:
        result.notes.append(f"HIP runtime detected: {probe.hip_version}")
    else:
        result.warnings.append("ROCm runtime selected but torch.version.hip is empty; verify the active runtime.")

    requested_precision = str(getattr(config, "mixed_precision", "bf16") or "bf16").lower()
    if requested_precision == "bf16" and not probe.bf16_supported:
        result.forced_overrides["mixed_precision"] = "fp16"
        result.warnings.append("BF16 is unavailable on this AMD runtime; mixed_precision was downgraded to fp16.")

    optimizer_name = str(getattr(getattr(config, "optimizer_type", ""), "value", getattr(config, "optimizer_type", "")) or "").lower()
    if optimizer_name in _AMD_UNSAFE_OPTIMIZERS or optimizer_name.startswith("paged"):
        result.forced_overrides["optimizer_type"] = "AdamW"
        result.disabled_features["optimizer_type"] = "AMD ROCm hides 8bit / paged optimizer routes and falls back to AdamW."
    optimizer_args = str(getattr(config, "optimizer_args", "") or "")
    if "pytorch_optimizer" in optimizer_args.lower():
        result.forced_overrides["optimizer_type"] = "AdamW"
        result.disabled_features["pytorch_optimizer"] = "pytorch_optimizer.* is not part of the AMD ROCm baseline."

    requested_attention = str(
        getattr(config, "attention_backend", "")
        or getattr(config, "anima_attn_mode", "")
        or getattr(config, "attn_mode", "")
        or "sdpa"
    ).strip().lower()
    if result.is_amd_sage2 and requested_attention in {"auto", "sageattn", "sageattention"}:
        sage2_available = _module_available("sageattention")
        sage2_gfx_supported = _is_sage2_gfx_supported(probe.selected_gpu_arch)
        if sage2_available and sage2_gfx_supported:
            result.forced_overrides["attention_backend"] = "sageattn"
            result.forced_overrides["anima_attn_mode"] = "sageattn"
            result.notes.append(
                "AMD ROCm SageAttention 2 experimental path enabled for "
                f"{probe.selected_gpu_arch or 'unknown gfx'}."
            )
        else:
            result.forced_overrides["attention_backend"] = "sdpa"
            result.forced_overrides["anima_attn_mode"] = "sdpa"
            reason = "sageattention package is unavailable"
            if sage2_available and not sage2_gfx_supported:
                reason = f"GPU arch {probe.selected_gpu_arch or 'unknown'} is not in gfx11/gfx12"
            result.disabled_features["attention_backend"] = (
                "AMD ROCm SageAttention 2 experimental path fell back to SDPA: "
                f"{reason}."
            )
            result.warnings.append(result.disabled_features["attention_backend"])
    elif requested_attention in {
        "xformers",
        "flash2",
        "sageattn",
        "sageattention",
        "flex",
        "flexattn",
        "flexattention",
        "spargeattn2",
        "auto",
        "torch",
    }:
        result.forced_overrides["attention_backend"] = "sdpa"
        result.forced_overrides["anima_attn_mode"] = "sdpa"
        result.disabled_features["attention_backend"] = "AMD ROCm runtime currently forces SDPA."

    if bool(getattr(config, "xformers", False)):
        result.forced_overrides["xformers"] = False
    if bool(getattr(config, "sageattn", False)) and result.forced_overrides.get("attention_backend") != "sageattn":
        result.forced_overrides["sageattn"] = False
    if bool(getattr(config, "flashattn", False)):
        result.forced_overrides["flashattn"] = False
    if bool(getattr(config, "torch_compile", False)):
        result.forced_overrides["torch_compile"] = False
        result.forced_overrides["anima_compile_scope"] = ""
        result.disabled_features["torch_compile"] = "torch.compile is automatically disabled on the AMD ROCm experimental path."

    result.forced_overrides.setdefault("dataloader_num_workers", profile.dataloader_num_workers)
    result.forced_overrides.setdefault("persistent_data_loader_workers", profile.persistent_data_loader_workers)
    result.forced_overrides.setdefault(
        "amd_empty_cache_interval",
        max(0, int(getattr(config, "amd_empty_cache_interval", 0) or 0)) or profile.empty_cache_interval,
    )
    result.forced_overrides.setdefault(
        "amd_sdpa_slice_trigger_gb",
        float(getattr(config, "amd_sdpa_slice_trigger_gb", 0.0) or 0.0) or profile.sdpa_slice_trigger_gb,
    )
    result.forced_overrides.setdefault(
        "amd_sdpa_slice_target_gb",
        float(getattr(config, "amd_sdpa_slice_target_gb", 0.0) or 0.0) or profile.sdpa_slice_target_gb,
    )
    result.empty_cache_interval = int(result.forced_overrides["amd_empty_cache_interval"])
    result.sdpa_slice_trigger_gb = float(result.forced_overrides["amd_sdpa_slice_trigger_gb"])
    result.sdpa_slice_target_gb = float(result.forced_overrides["amd_sdpa_slice_target_gb"])
    return result


def apply_amd_runtime_guard(config: Any, guard: AmdRuntimeGuardResult) -> None:
    if not guard.is_amd:
        return
    if not guard.route_supported:
        raise RuntimeError(guard.route_reason)
    for key, value in guard.forced_overrides.items():
        try:
            setattr(config, key, value)
        except Exception:
            pass


def estimate_sdpa_chunk_count(
    q: torch.Tensor,
    *,
    trigger_gb: float,
    target_gb: float,
) -> int:
    if trigger_gb <= 0.0 or target_gb <= 0.0:
        return 0
    if q.dim() != 4:
        return 0
    batch, heads, tokens, _ = q.shape
    if heads <= 1 or tokens <= 1:
        return 0
    bytes_per_element = max(int(q.element_size()), 1)
    estimated_bytes = batch * heads * tokens * tokens * bytes_per_element
    trigger_bytes = trigger_gb * (1024 ** 3)
    if estimated_bytes <= trigger_bytes:
        return 0
    target_bytes = max(target_gb * (1024 ** 3), 1.0)
    chunks = int(math.ceil(estimated_bytes / target_bytes))
    return max(2, min(heads, chunks))


def build_amd_banner_lines(guard: AmdRuntimeGuardResult) -> list[str]:
    if not guard.is_amd:
        return []
    lines = [
        "[amd-rocm] Experimental AMD runtime guard active",
        (
            "[amd-rocm] "
            f"gpu={guard.gpu_summary or 'unknown'} "
            f"hip={guard.hip_version or 'unknown'} "
            f"bf16={'yes' if guard.bf16_supported else 'no'} "
            f"profile={guard.runtime_profile_name}"
        ),
        (
            "[amd-rocm] "
            f"empty_cache_interval={guard.empty_cache_interval} "
            f"sdpa_slice_trigger_gb={guard.sdpa_slice_trigger_gb:.2f} "
            f"sdpa_slice_target_gb={guard.sdpa_slice_target_gb:.2f}"
        ),
    ]
    for key, value in guard.forced_overrides.items():
        lines.append(f"[amd-rocm] forced {key}={value}")
    for warning in guard.warnings:
        lines.append(f"[amd-rocm][warn] {warning}")
    for note in guard.notes:
        lines.append(f"[amd-rocm] {note}")
    return lines
