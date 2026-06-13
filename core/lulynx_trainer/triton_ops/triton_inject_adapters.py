"""Inject Lulynx fused adapter-variant kernels (ChimeraHydra, DoRA) — default-off.

A companion to :mod:`triton_inject` for the adapter *variants* whose wrappers
are distinct classes from the standard ``LoRALinear``:

* ``ChimeraHydraLinear`` — a dual-pool gated mixture-of-experts LoRA. The expert
  projections are a dense batched low-rank op (``chimera_hydra._mixed_delta``);
  only the gate weights are sparse. We patch the block ``forward`` so routing
  (gate Linear, softmax, top-k) stays eager but each pool's projection +
  gated-sum runs through :func:`chimera_hydra_lora.fused`, keeping the
  ``[*, experts, out]`` deltas in SRAM.
* ``DoRALinear`` — weight-decomposed LoRA. The ``O(out*in)`` row-norm is
  unavoidable, so :func:`dora_lora.fused` keeps it in torch (autograd routes its
  gradient back to magnitude/down/up) and fuses only the base GEMM + low-rank
  LoRA + row_scale/bias epilogue. QDoRA (quantised base) keeps the eager path;
  the mode (full/style/structure) only sets ``requires_grad``, which the
  autograd backward already honours, so every mode is fusable.

Every patch mirrors :mod:`triton_inject`: an instance attribute marks "patched"
(single source of truth), per-call guards keep the eager path whenever the
fused kernel would diverge or be unsafe, any kernel failure falls back instead
of crashing, and :func:`revert_chimera` / :func:`revert_dora` undo it. This is a
clean-room Lulynx module; it shares no source with any reference implementation.
"""

from __future__ import annotations

import types

import torch
from torch import nn

from .lora import chimera_hydra_lora, dora_lora


def _is_bf16(t: object) -> bool:
    return isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Fused ChimeraHydra dual-pool LoRA injection
# ---------------------------------------------------------------------------

# Stashes the pre-patch bound ``forward``; presence marks a patched block.
_CHIMERA_ORIG = "_lulynx_triton_chimera_orig"


def _chimera_linear_cls() -> type:
    """Lazy import to avoid an import cycle."""
    from ..chimera_hydra import ChimeraHydraLinear

    return ChimeraHydraLinear


def _chimera_eligible(module: nn.Module) -> bool:
    """Both pools' factors and the frozen base must be bf16."""
    cd = getattr(module, "content_down", None)
    cu = getattr(module, "content_up", None)
    fd = getattr(module, "frequency_down", None)
    fu = getattr(module, "frequency_up", None)
    if cd is None or cu is None or fd is None or fu is None:
        return False
    if not (_is_bf16(cd) and _is_bf16(cu) and _is_bf16(fd) and _is_bf16(fu)):
        return False
    original = getattr(module, "original", None)
    if original is None or not _is_bf16(getattr(original, "weight", None)):
        return False
    if getattr(original, "bias", None) is not None and not _is_bf16(original.bias):
        return False
    return True


def _chimera_fused_forward(self: nn.Module, x: torch.Tensor, *, frequency_features=None):
    """Patched ``ChimeraHydraLinear.forward`` — fused pools, eager routing."""
    orig = getattr(self, _CHIMERA_ORIG)
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or x.device.type != "cuda"
    ):
        return orig(x, frequency_features=frequency_features)
    try:
        base = self.original(x)
        content_x = self.dropout(x)
        freq_x = self.dropout(frequency_features if frequency_features is not None else x)
        if freq_x.shape != x.shape:
            return orig(x, frequency_features=frequency_features)
        c_w = self._pool_weights(self.content_gate(x), self.config.content_top_k)
        f_w = self._pool_weights(self.frequency_gate(freq_x), self.config.frequency_top_k)
        fp32 = getattr(self, "_lulynx_triton_fp32_bwd", False)
        c_delta = chimera_hydra_lora.fused(
            content_x, self.content_down, self.content_up, c_w, fp32_backward=fp32
        )
        f_delta = chimera_hydra_lora.fused(
            freq_x, self.frequency_down, self.frequency_up, f_w, fp32_backward=fp32
        )
        return base + (c_delta + f_delta) * self.scaling
    except Exception:
        return orig(x, frequency_features=frequency_features)


def apply_chimera(*roots: nn.Module, fp32_backward: bool = False) -> int:
    """Patch every eligible ``ChimeraHydraLinear`` forward in place. Idempotent.

    Returns the number of modules newly patched.
    """
    chimera_cls = _chimera_linear_cls()
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if not isinstance(module, chimera_cls):
                continue
            if hasattr(module, _CHIMERA_ORIG):
                continue
            if not _chimera_eligible(module):
                continue
            setattr(module, _CHIMERA_ORIG, module.forward)
            module._lulynx_triton_fp32_bwd = bool(fp32_backward)
            module.forward = types.MethodType(_chimera_fused_forward, module)
            count += 1
    return count


def revert_chimera(*roots: nn.Module) -> int:
    """Undo :func:`apply_chimera`. Returns the number of modules reverted."""
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            orig = getattr(module, _CHIMERA_ORIG, None)
            if orig is None:
                continue
            module.forward = orig
            delattr(module, _CHIMERA_ORIG)
            if hasattr(module, "_lulynx_triton_fp32_bwd"):
                delattr(module, "_lulynx_triton_fp32_bwd")
            count += 1
    return count


def count_chimera_eligible(*roots: nn.Module) -> int:
    """Report how many ChimeraHydra blocks *would* be patched, without mutating."""
    chimera_cls = _chimera_linear_cls()
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if isinstance(module, chimera_cls) and _chimera_eligible(module):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Fused DoRA injection
# ---------------------------------------------------------------------------

# Stashes the pre-patch bound ``forward``; presence marks a patched wrapper.
_DORA_ORIG = "_lulynx_triton_dora_orig"


def _dora_linear_cls() -> type:
    """Lazy import to avoid an import cycle."""
    from ...lulynx.dora_layer import DoRALinear

    return DoRALinear


def _dora_eligible(module: nn.Module) -> bool:
    """The frozen base, both factors and the magnitude must be bf16; not QDoRA."""
    if getattr(module, "is_quantized", False):
        return False
    bw = getattr(module, "base_weight", None)
    a = getattr(module, "lora_A", None)
    b = getattr(module, "lora_B", None)
    m = getattr(module, "m", None)
    if bw is None or a is None or b is None or m is None:
        return False
    if not (_is_bf16(bw) and _is_bf16(a) and _is_bf16(b) and _is_bf16(m)):
        return False
    bias = getattr(module, "base_bias", None)
    if bias is not None and not _is_bf16(bias):
        return False
    return True


def _dora_fused_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Patched ``DoRALinear.forward`` — fused when safe, eager otherwise."""
    orig = getattr(self, _DORA_ORIG)
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or x.device.type != "cuda"
        or getattr(self, "is_quantized", False)
    ):
        return orig(x)
    try:
        return dora_lora.fused(
            x,
            self.base_weight,
            self.base_bias,
            self.lora_A,
            self.lora_B,
            float(self.scaling),
            self.m,
            fp32_backward=getattr(self, "_lulynx_triton_fp32_bwd", False),
        )
    except Exception:
        return orig(x)


def apply_dora(*roots: nn.Module, fp32_backward: bool = False) -> int:
    """Patch every eligible ``DoRALinear`` forward in place. Idempotent.

    Returns the number of modules newly patched.
    """
    dora_cls = _dora_linear_cls()
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if not isinstance(module, dora_cls):
                continue
            if hasattr(module, _DORA_ORIG):
                continue
            if not _dora_eligible(module):
                continue
            setattr(module, _DORA_ORIG, module.forward)
            module._lulynx_triton_fp32_bwd = bool(fp32_backward)
            module.forward = types.MethodType(_dora_fused_forward, module)
            count += 1
    return count


def revert_dora(*roots: nn.Module) -> int:
    """Undo :func:`apply_dora`. Returns the number of modules reverted."""
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            orig = getattr(module, _DORA_ORIG, None)
            if orig is None:
                continue
            module.forward = orig
            delattr(module, _DORA_ORIG)
            if hasattr(module, "_lulynx_triton_fp32_bwd"):
                delattr(module, "_lulynx_triton_fp32_bwd")
            count += 1
    return count


def count_dora_eligible(*roots: nn.Module) -> int:
    """Report how many DoRALinear wrappers *would* be patched, without mutating."""
    dora_cls = _dora_linear_cls()
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if isinstance(module, dora_cls) and _dora_eligible(module):
                count += 1
    return count
