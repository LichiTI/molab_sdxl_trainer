# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Apple Silicon / MPS experimental runtime guard.

The guard is intentionally conservative. It keeps the training core on pure
PyTorch paths and disables CUDA-oriented acceleration stacks that are not
available on Apple M-series GPUs.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Dict

import torch


_MPS_RUNTIME_IDS = {"apple-mps", "mps", "mac-mps", "darwin-mps"}
_MPS_UNSAFE_OPTIMIZERS = {
    "adamw8bit",
    "lion8bit",
    "sgdnesterov8bit",
    "pagedadamw",
    "pagedadamw8bit",
    "pagedadamw32bit",
    "pagedlion8bit",
    "pytorchoptimizer",
    "prodigyplus.prodigyplusschedulefree",
}
_MPS_UNSUPPORTED_SCHEMA_PREFIXES = ("flux-", "lumina-")


@dataclass(frozen=True)
class MpsRuntimeProbe:
    runtime_id: str
    is_darwin: bool = False
    mps_built: bool = False
    mps_available: bool = False
    mps_summary: str = ""


@dataclass
class MpsRuntimeGuardResult:
    runtime_id: str
    mps_built: bool = False
    mps_available: bool = False
    forced_overrides: Dict[str, Any] = field(default_factory=dict)
    disabled_features: Dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    route_supported: bool = True
    route_reason: str = ""

    @property
    def is_mps(self) -> bool:
        return self.runtime_id in _MPS_RUNTIME_IDS


def detect_runtime_id(config: Any) -> str:
    return str(
        getattr(config, "runtime_id", "")
        or getattr(config, "execution_profile_id", "")
        or ""
    ).strip().lower()


def is_mps_runtime(config: Any) -> bool:
    return detect_runtime_id(config) in _MPS_RUNTIME_IDS


def mps_backend_built() -> bool:
    try:
        return bool(torch.backends.mps.is_built())
    except Exception:
        return False


def mps_backend_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def build_mps_runtime_probe(config: Any) -> MpsRuntimeProbe:
    runtime_id = detect_runtime_id(config)
    built = mps_backend_built()
    available = mps_backend_available()
    is_darwin = platform.system().lower() == "darwin"
    if available:
        summary = "Apple Metal Performance Shaders available"
    elif built:
        summary = "PyTorch MPS backend built but not available"
    else:
        summary = "PyTorch MPS backend unavailable"
    return MpsRuntimeProbe(
        runtime_id=runtime_id,
        is_darwin=is_darwin,
        mps_built=built,
        mps_available=available,
        mps_summary=summary,
    )


def _route_supported(schema_id: str) -> tuple[bool, str]:
    normalized = str(schema_id or "").strip().lower()
    for prefix in _MPS_UNSUPPORTED_SCHEMA_PREFIXES:
        if normalized.startswith(prefix):
            return False, (
                "Apple MPS experimental runtime does not currently expose this training route. "
                "Use SD / SDXL / Anima / Newbie LoRA routes first."
            )
    return True, ""


def build_mps_runtime_guard(config: Any) -> MpsRuntimeGuardResult:
    probe = build_mps_runtime_probe(config)
    route_supported, route_reason = _route_supported(str(getattr(config, "schema_id", "") or ""))
    result = MpsRuntimeGuardResult(
        runtime_id=probe.runtime_id,
        mps_built=probe.mps_built,
        mps_available=probe.mps_available,
        route_supported=route_supported,
        route_reason=route_reason,
    )
    if not result.is_mps:
        return result

    result.notes.append("Apple MPS runtime is experimental and uses conservative pure-PyTorch settings.")
    if not probe.is_darwin:
        result.warnings.append("Apple MPS runtime was selected on a non-macOS host.")
    if not probe.mps_available:
        result.warnings.append(f"{probe.mps_summary}; training will not be able to use an Apple GPU here.")

    result.forced_overrides["device"] = "mps" if probe.mps_available else "cpu"

    requested_precision = str(getattr(config, "mixed_precision", "no") or "no").lower()
    if requested_precision in {"bf16", "fp16"}:
        result.forced_overrides["mixed_precision"] = "no"
        result.disabled_features["mixed_precision"] = "Apple MPS runtime defaults to fp32 for training stability."

    optimizer_name = str(getattr(getattr(config, "optimizer_type", ""), "value", getattr(config, "optimizer_type", "")) or "").lower()
    if optimizer_name in _MPS_UNSAFE_OPTIMIZERS or optimizer_name.startswith("paged"):
        result.forced_overrides["optimizer_type"] = "AdamW"
        result.disabled_features["optimizer_type"] = "Apple MPS hides 8bit / paged / plugin optimizers and falls back to AdamW."

    requested_attention = str(
        getattr(config, "attention_backend", "")
        or getattr(config, "anima_attn_mode", "")
        or getattr(config, "attn_mode", "")
        or "sdpa"
    ).strip().lower()
    if requested_attention in {
        "xformers",
        "flash2",
        "sageattn",
        "flex",
        "flexattn",
        "flexattention",
        "spargeattn2",
        "auto",
        "torch",
    }:
        result.forced_overrides["attention_backend"] = "sdpa"
        result.forced_overrides["anima_attn_mode"] = "sdpa"
        result.disabled_features["attention_backend"] = "Apple MPS runtime forces PyTorch SDPA."

    for flag in ("xformers", "sageattn", "flashattn", "flash_attention", "flexattn", "spargeattn2"):
        if bool(getattr(config, flag, False)):
            result.forced_overrides[flag] = False

    if bool(getattr(config, "torch_compile", False)):
        result.forced_overrides["torch_compile"] = False
        result.forced_overrides["anima_compile_scope"] = ""
        result.disabled_features["torch_compile"] = "torch.compile is disabled on the Apple MPS experimental path."

    result.forced_overrides.setdefault("dataloader_num_workers", 0)
    result.forced_overrides.setdefault("persistent_data_loader_workers", False)
    result.forced_overrides.setdefault("mem_eff_attn", False)
    result.forced_overrides.setdefault("gradient_checkpointing", True)
    return result


def apply_mps_runtime_guard(config: Any, guard: MpsRuntimeGuardResult) -> None:
    if not guard.is_mps:
        return
    if not guard.route_supported:
        raise RuntimeError(guard.route_reason)
    for key, value in guard.forced_overrides.items():
        try:
            setattr(config, key, value)
        except Exception:
            pass


def build_mps_banner_lines(guard: MpsRuntimeGuardResult) -> list[str]:
    if not guard.is_mps:
        return []
    lines = [
        "[Apple MPS] Experimental runtime active.",
        f"[Apple MPS] backend_available={guard.mps_available}, backend_built={guard.mps_built}",
    ]
    for warning in guard.warnings:
        lines.append(f"[Apple MPS][warn] {warning}")
    for key, value in guard.forced_overrides.items():
        lines.append(f"[Apple MPS] override {key}={value}")
    return lines
