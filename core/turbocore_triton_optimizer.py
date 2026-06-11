"""Triton research kernels for TurboCore optimizer experiments.

The v0 AdamW path intentionally targets flat fp32 CUDA buffers. It is a
developer benchmark candidate, not a training dispatcher.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
from typing import Any

import torch


def _append_env_path(name: str, path: Path) -> None:
    if not path.exists():
        return
    value = str(path)
    current = os.environ.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if value not in parts:
        os.environ[name] = os.pathsep.join([value, *parts]) if parts else value


def _seed_windows_python_build_env() -> None:
    """Help Triton/TCC find Python headers and libs in portable dev envs."""

    if os.name != "nt":
        return
    version_tag = f"Python{sys.version_info.major}{sys.version_info.minor}"
    local_programs = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / version_tag
    prefixes = [Path(sys.prefix), Path(sys.base_prefix), Path(sys.exec_prefix), local_programs]
    include_candidates = [Path(path) for path in {sysconfig.get_paths().get("include", ""), sysconfig.get_paths().get("platinclude", "")} if path]
    lib_candidates = [Path(path) for path in {sysconfig.get_config_var("LIBDIR") or "", sysconfig.get_config_var("LIBPL") or ""} if path]
    for prefix in prefixes:
        include_candidates.append(prefix / "include")
        lib_candidates.append(prefix / "libs")
    for include_dir in include_candidates:
        if (include_dir / "Python.h").exists():
            _append_env_path("CPATH", include_dir)
            _append_env_path("INCLUDE", include_dir)
            break
    for lib_dir in lib_candidates:
        if any(lib_dir.glob(f"python{sys.version_info.major}{sys.version_info.minor}*.lib")):
            _append_env_path("LIBRARY_PATH", lib_dir)
            _append_env_path("LIB", lib_dir)
            break


_seed_windows_python_build_env()


try:  # pragma: no cover - host-specific import availability
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


def triton_adamw_flat_available() -> bool:
    return triton is not None and tl is not None and bool(torch.cuda.is_available())


def triton_adamw_flat_unavailable_reason() -> str:
    if triton is None or tl is None:
        return "triton_unavailable"
    if not bool(torch.cuda.is_available()):
        return "cuda_unavailable"
    return "available"


if triton is not None and tl is not None:  # pragma: no branch - definition guard

    @triton.jit
    def _adamw_flat_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        m_new = m * beta1 + g * (1.0 - beta1)
        v_new = v * beta2 + g * g * (1.0 - beta2)
        p_decayed = p * (1.0 - lr * weight_decay)
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2
        p_new = p_decayed - lr * m_hat / (tl.sqrt(v_hat) + eps)

        tl.store(exp_avg_ptr + offsets, m_new, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, v_new, mask=mask)
        tl.store(param_ptr + offsets, p_new, mask=mask)


def triton_adamw_flat_v0_step_(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    step: int,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    block_size: int = 1024,
) -> dict[str, Any]:
    """Run one in-place flat fp32 AdamW step on CUDA tensors."""

    _validate_flat_step_inputs(param, grad, exp_avg, exp_avg_sq)
    if not triton_adamw_flat_available():
        raise RuntimeError(f"Triton AdamW flat v0 unavailable: {triton_adamw_flat_unavailable_reason()}")
    if int(step) <= 0:
        raise ValueError("AdamW step index must be >= 1")
    n_elements = int(param.numel())
    bias_correction1 = 1.0 - float(beta1) ** int(step)
    bias_correction2 = 1.0 - float(beta2) ** int(step)
    grid = (triton.cdiv(n_elements, int(block_size)),)
    _adamw_flat_kernel[grid](
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        n_elements,
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
        float(bias_correction1),
        float(bias_correction2),
        BLOCK_SIZE=int(block_size),
        num_warps=4,
    )
    return {
        "candidate": "triton_adamw_flat_v0",
        "native_kernel_present": True,
        "training_path_enabled": False,
        "step": int(step),
        "numel": n_elements,
        "block_size": int(block_size),
        "dtype": str(param.dtype).replace("torch.", ""),
        "device": str(param.device),
    }


def _validate_flat_step_inputs(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
) -> None:
    tensors = (param, grad, exp_avg, exp_avg_sq)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise RuntimeError("Triton AdamW flat v0 requires CUDA tensors")
    if any(tensor.dtype is not torch.float32 for tensor in tensors):
        raise RuntimeError("Triton AdamW flat v0 currently supports fp32 flat buffers only")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("Triton AdamW flat v0 requires contiguous flat buffers")
    shape = tuple(param.shape)
    if any(tuple(tensor.shape) != shape for tensor in tensors):
        raise ValueError("AdamW flat buffers must have identical shapes")
    if param.dim() != 1:
        raise ValueError("Triton AdamW flat v0 requires 1D flattened buffers")


def triton_adamw_flat_metadata() -> dict[str, Any]:
    return {
        "name": "triton_adamw_flat_v0",
        "available": triton_adamw_flat_available(),
        "reason": triton_adamw_flat_unavailable_reason(),
        "fusion_scope": "flat_fp32_adamw_param_moment_update",
        "native_kernel_present": bool(triton_adamw_flat_available()),
        "training_path_enabled": False,
        "layout": "flat_contiguous_buffers_only",
    }


__all__ = [
    "triton_adamw_flat_available",
    "triton_adamw_flat_metadata",
    "triton_adamw_flat_unavailable_reason",
    "triton_adamw_flat_v0_step_",
]
