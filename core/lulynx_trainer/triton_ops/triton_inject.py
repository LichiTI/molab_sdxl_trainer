"""Inject Lulynx fused Triton kernels into a built LoRA network (default-off).

The injector walks a module tree whose ``LoRALinear`` wrappers have already
been applied and swaps the ``forward`` of every *eligible* wrapper for a fused
Triton path (:func:`triton_ops.lora.base_lora.fused`). The fused path keeps the
low-rank LoRA hidden state in SRAM, so a layer's down-projection, up-projection,
scale multiply and residual add collapse into one kernel launch on top of the
cuDNN base GEMM.

Patching is intentionally conservative. A wrapper is patched only when the
fused kernel is *numerically identical* to the eager path; anything that would
diverge keeps the original forward:

* DoRA wrappers (magnitude renormalisation — handled by a separate kernel),
* active LoRA dropout (the fused kernel has no dropout),
* Vortex-managed base weights (custom paged-weight GEMM),
* non-bf16 parameters (the kernel is a bf16 specialisation).

Per-call guards add the runtime conditions (CUDA bf16 input, positive preview
scale) and wrap the kernel in a ``try`` so any unexpected Triton failure falls
back to the eager forward instead of crashing training.

Every patch is reversible with :func:`revert`. This is a clean-room Lulynx
module; it shares no source with any reference implementation.
"""

from __future__ import annotations

import types

import torch
from torch import nn

from .blocks import adaln_norm
from .lora import base_lora, qkv_lora

# Instance attribute that stashes the pre-patch bound forward. Its presence on
# a module is the single source of truth for "this module is patched".
_ORIG_ATTR = "_lulynx_triton_orig_forward"


def _lora_linear_cls() -> type:
    """Lazy import to avoid an import cycle (lora_injector never imports us)."""
    from ..lora_injector import LoRALinear

    return LoRALinear


def _is_bf16(t: object) -> bool:
    return isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16


def _eligible(module: nn.Module) -> bool:
    """Structural (run-stable) checks deciding whether a wrapper may be fused.

    Runtime-varying conditions (input dtype/device, preview scale) are checked
    per call inside :func:`_fused_forward`, not here.
    """
    if getattr(module, "use_dora", False):
        return False
    if getattr(module, "vortex_enabled", False):
        return False
    original = getattr(module, "original", None)
    lora = getattr(module, "lora", None)
    if original is None or lora is None:
        return False
    if getattr(original, "_vortex_managed", False):
        return False
    down = getattr(lora, "lora_down", None)
    up = getattr(lora, "lora_up", None)
    if down is None or up is None:
        return False
    # The fused kernel has no dropout: only fuse when dropout is a no-op.
    if not isinstance(getattr(lora, "dropout", None), nn.Identity):
        return False
    if getattr(lora, "scaling", None) is None:
        return False
    if not (_is_bf16(original.weight) and _is_bf16(down.weight) and _is_bf16(up.weight)):
        return False
    if original.bias is not None and not _is_bf16(original.bias):
        return False
    return True


def _fused_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Patched ``LoRALinear.forward`` — fused when safe, eager otherwise."""
    orig = getattr(self, _ORIG_ATTR)
    preview_scale = float(getattr(self, "_preview_lora_scale", 1.0))
    if (
        preview_scale <= 0.0
        or not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or x.device.type != "cuda"
        or getattr(self.original, "_vortex_managed", False)
    ):
        return orig(x)
    lora = self.lora
    try:
        return base_lora.fused(
            x,
            self.original.weight,
            self.original.bias,
            lora.lora_down.weight,
            lora.lora_up.weight,
            float(lora.scaling) * preview_scale,
            getattr(self, "_lulynx_triton_fp32_bwd", False),
        )
    except Exception:
        return orig(x)


def apply(*roots: nn.Module, fp32_backward: bool = False) -> int:
    """Patch every eligible ``LoRALinear`` reachable from ``roots`` in place.

    Idempotent and overlap-safe: already-patched modules are skipped, so passing
    both the UNet and the LoRA network (which may share wrappers) is fine.

    ``fp32_backward`` selects the high-accuracy (slower) LoRA backward; the
    default bf16 backward matches eager accuracy and keeps training competitive.

    Returns the number of modules newly patched.
    """
    lora_linear = _lora_linear_cls()
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if not isinstance(module, lora_linear):
                continue
            if hasattr(module, _ORIG_ATTR):
                continue
            if not _eligible(module):
                continue
            setattr(module, _ORIG_ATTR, module.forward)
            module._lulynx_triton_fp32_bwd = bool(fp32_backward)
            module.forward = types.MethodType(_fused_forward, module)
            count += 1
    return count


def revert(*roots: nn.Module) -> int:
    """Undo :func:`apply` for every patched module reachable from ``roots``.

    Returns the number of modules reverted.
    """
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            orig = getattr(module, _ORIG_ATTR, None)
            if orig is None:
                continue
            module.forward = orig
            delattr(module, _ORIG_ATTR)
            if hasattr(module, "_lulynx_triton_fp32_bwd"):
                delattr(module, "_lulynx_triton_fp32_bwd")
            count += 1
    return count


def count_eligible(*roots: nn.Module) -> int:
    """Report how many wrappers *would* be patched, without mutating anything."""
    lora_linear = _lora_linear_cls()
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if isinstance(module, lora_linear) and _eligible(module):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Fused Q/K/V LoRA injection (self-attention only)
# ---------------------------------------------------------------------------
#
# A self-attention projection runs three independent LoRALinear chains
# (q/k/v) over the *same* activation x. The patched Anima attention forward
# (anima_attention._dispatched_forward) consults ``module._fused_qkv`` and, for
# self-attention (context is None), calls ``q_raw, k_raw, v_raw = fused_qkv(x)``
# before applying q_norm/_split_heads itself. We install a callable there that
# fuses the three LoRA paths into one kernel (qkv_lora.fused) while keeping the
# raw-projection contract. The existing weight-concat fused_qkv skips LoRA
# wrappers, so this targets a disjoint set of modules.

# Stashes the pre-injection ``_fused_qkv`` (usually None); its presence marks a
# module as qkv-injected and lets :func:`revert_qkv` restore the prior value.
_QKV_MARKER = "_lulynx_triton_qkv_prev"


def _qkv_eligible(qp: nn.Module, kp: nn.Module, vp: nn.Module) -> bool:
    """All three projections must be independently fusable and share the input
    contract the fused kernel assumes: one ``x``, one ``rank`` (single R_PAD).

    Per-head output widths and scales may differ (GQA-friendly); only
    in_features and rank must match across q/k/v.
    """
    if not (_eligible(qp) and _eligible(kp) and _eligible(vp)):
        return False
    in_q = qp.original.weight.shape[1]
    if kp.original.weight.shape[1] != in_q or vp.original.weight.shape[1] != in_q:
        return False
    rank = qp.lora.lora_down.weight.shape[0]
    if kp.lora.lora_down.weight.shape[0] != rank or vp.lora.lora_down.weight.shape[0] != rank:
        return False
    return True


class _QKVFused:
    """Callable installed as ``module._fused_qkv``; returns raw ``(q, k, v)``.

    Equivalent to ``(q_proj(x), k_proj(x), v_proj(x))`` but fused: ``x`` is read
    once and the three low-rank hidden states share SRAM. Per-call guards keep
    the eager path whenever the fused kernel would diverge or be unsafe, and any
    unexpected kernel failure falls back instead of crashing training.
    """

    __slots__ = ("qp", "kp", "vp", "fp32_backward")

    def __init__(self, qp: nn.Module, kp: nn.Module, vp: nn.Module, fp32_backward: bool) -> None:
        self.qp, self.kp, self.vp = qp, kp, vp
        self.fp32_backward = bool(fp32_backward)

    def _eager(self, x: torch.Tensor):
        return self.qp(x), self.kp(x), self.vp(x)

    def __call__(self, x: torch.Tensor):
        qp, kp, vp = self.qp, self.kp, self.vp
        ps_q = float(getattr(qp, "_preview_lora_scale", 1.0))
        ps_k = float(getattr(kp, "_preview_lora_scale", 1.0))
        ps_v = float(getattr(vp, "_preview_lora_scale", 1.0))
        if (
            not isinstance(x, torch.Tensor)
            or x.dtype != torch.bfloat16
            or x.device.type != "cuda"
            or min(ps_q, ps_k, ps_v) <= 0.0
            or getattr(qp.original, "_vortex_managed", False)
            or getattr(kp.original, "_vortex_managed", False)
            or getattr(vp.original, "_vortex_managed", False)
        ):
            return self._eager(x)
        try:
            return qkv_lora.fused(
                x,
                qp.original.weight, kp.original.weight, vp.original.weight,
                qp.original.bias, kp.original.bias, vp.original.bias,
                qp.lora.lora_down.weight, kp.lora.lora_down.weight, vp.lora.lora_down.weight,
                qp.lora.lora_up.weight, kp.lora.lora_up.weight, vp.lora.lora_up.weight,
                float(qp.lora.scaling) * ps_q,
                float(kp.lora.scaling) * ps_k,
                float(vp.lora.scaling) * ps_v,
                self.fp32_backward,
            )
        except Exception:
            return self._eager(x)


def apply_qkv(*roots: nn.Module, fp32_backward: bool = False) -> int:
    """Install the fused Q/K/V LoRA path on every eligible self-attention.

    A module qualifies when its name contains ``self_attn`` and its
    ``q_proj``/``k_proj``/``v_proj`` are all eligible bf16 ``LoRALinear``
    wrappers sharing in_features and rank. We set ``module._fused_qkv``; the
    patched attention forward reads it only for self-attention, so installing it
    on a module never patched by :func:`patch_anima_attention` is a safe no-op.

    Idempotent. Returns the number of attention modules newly fused.
    """
    lora_linear = _lora_linear_cls()
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for name, module in root.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if "self_attn" not in name:
                continue
            if hasattr(module, _QKV_MARKER):
                continue
            qp = getattr(module, "q_proj", None)
            kp = getattr(module, "k_proj", None)
            vp = getattr(module, "v_proj", None)
            if not (
                isinstance(qp, lora_linear)
                and isinstance(kp, lora_linear)
                and isinstance(vp, lora_linear)
            ):
                continue
            if not _qkv_eligible(qp, kp, vp):
                continue
            setattr(module, _QKV_MARKER, getattr(module, "_fused_qkv", None))
            module._fused_qkv = _QKVFused(qp, kp, vp, fp32_backward)
            count += 1
    return count


def revert_qkv(*roots: nn.Module) -> int:
    """Undo :func:`apply_qkv`, restoring any prior ``_fused_qkv`` value.

    Returns the number of modules reverted.
    """
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for _name, module in root.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if not hasattr(module, _QKV_MARKER):
                continue
            module._fused_qkv = getattr(module, _QKV_MARKER)
            delattr(module, _QKV_MARKER)
            count += 1
    return count


def count_qkv_eligible(*roots: nn.Module) -> int:
    """Report how many self-attention modules *would* be qkv-fused."""
    lora_linear = _lora_linear_cls()
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for name, module in root.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if "self_attn" not in name:
                continue
            qp = getattr(module, "q_proj", None)
            kp = getattr(module, "k_proj", None)
            vp = getattr(module, "v_proj", None)
            if (
                isinstance(qp, lora_linear)
                and isinstance(kp, lora_linear)
                and isinstance(vp, lora_linear)
                and _qkv_eligible(qp, kp, vp)
            ):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Fused AdaLN modulation injection (DiT block ``_apply_adaln``)
# ---------------------------------------------------------------------------
#
# An Anima DiT block normalises+modulates its activation before each sub-layer
# via ``_apply_adaln(x, shift, scale)`` (LayerNorm then ``* (1 + scale) +
# shift``). That is bandwidth-bound — two kernels and an HBM round-trip of the
# normalised tensor — so we swap the bound method for one that calls the fused
# adaln_norm kernel (forward + Triton backward). Unlike the LoRA kernels this
# speeds up training too. Patching the instance method covers all three
# self/cross/mlp modulations in the block.

# Stashes the pre-patch bound ``_apply_adaln``; presence marks a patched block.
_ADALN_ORIG = "_lulynx_triton_adaln_orig"


def _is_adaln_block(module: nn.Module) -> bool:
    """A DiT block exposes a callable ``_apply_adaln`` and adaln modulations."""
    return callable(getattr(module, "_apply_adaln", None)) and hasattr(
        module, "adaln_modulation_mlp"
    )


def _fused_apply_adaln(self: nn.Module, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """Patched ``_apply_adaln`` — fused when safe, eager otherwise."""
    orig = getattr(self, _ADALN_ORIG)
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or x.device.type != "cuda"
    ):
        return orig(x, shift, scale)
    try:
        return adaln_norm.fused(x, shift, scale)
    except Exception:
        return orig(x, shift, scale)


def apply_adaln(*roots: nn.Module) -> int:
    """Patch every DiT block's ``_apply_adaln`` to the fused kernel in place.

    Idempotent and overlap-safe. Returns the number of blocks newly patched.
    """
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if hasattr(module, _ADALN_ORIG):
                continue
            if not _is_adaln_block(module):
                continue
            setattr(module, _ADALN_ORIG, module._apply_adaln)
            module._apply_adaln = types.MethodType(_fused_apply_adaln, module)
            count += 1
    return count


def revert_adaln(*roots: nn.Module) -> int:
    """Undo :func:`apply_adaln` for every patched block. Returns count reverted."""
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            orig = getattr(module, _ADALN_ORIG, None)
            if orig is None:
                continue
            module._apply_adaln = orig
            delattr(module, _ADALN_ORIG)
            count += 1
    return count


def count_adaln_eligible(*roots: nn.Module) -> int:
    """Report how many DiT blocks *would* be adaln-patched."""
    seen: set[int] = set()
    count = 0
    for root in roots:
        if root is None:
            continue
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if _is_adaln_block(module):
                count += 1
    return count
