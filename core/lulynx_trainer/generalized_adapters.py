# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Generalized adapter family (lulynx clean-room).

This module owns the project-local Generalized adapter skeleton

    ΔW = W·A + B            (weight-side, two paths)
    Δbias = c1·c2 (opt.)    (bias-side, two paths, when bias is present)

where W is the frozen original weight, A is a multiplicative re-shaping path,
and B is an additive low-rank path.  Concrete subclasses choose how A and B
are parameterized:

- :class:`GLoRALinearLayer` / :class:`GLoRAConv2dLayer`: low-rank
  ``A = a1·a2`` and ``B = b1·b2``.  Phase 2 augments these with optional
  rank-dropout, module-dropout, bias adaptation (only when the base module
  has a bias), no-materialize fast path, and tucker conv variant.
- (future) ``GLoKr`` will switch to Kronecker parameterizations.

All Phase 2 augmentations default to OFF: with defaults the layer is
numerically identical to the Phase 1 standard tier and produces a
bit-compatible state-dict.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class _GeneralizedDeltaBase(nn.Module):
    """Skeleton for ΔW = W·A + B style adapters.

    Subclasses implement :meth:`_compute_A` / :meth:`_compute_B`.  The base
    class handles W reference, scaling, dropout, ΔW assembly, rank/module
    dropout, optional bias-side delta, no-materialize fast path, and
    :func:`F.linear` forward.

    The frozen base weight ``W`` is kept as a non-persistent buffer reference
    (no clone) so that the layer adds zero extra residency on top of the base
    model.  dtype/device are re-synchronized at every forward call to remain
    safe under mixed-precision wrappers.
    """

    is_generalized: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        org_weight: torch.Tensor,
        *,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        no_materialize_forward: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, int(rank))
        self.alpha = float(alpha) if alpha not in (None, 0) else float(self.rank)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rank_dropout = float(rank_dropout)
        self.module_dropout = float(module_dropout)
        self.no_materialize_forward = bool(no_materialize_forward)
        # Non-persistent reference to the frozen base weight; never modified.
        self.register_buffer("_org_weight_ref", org_weight.detach(), persistent=False)

    # -- subclass hooks --------------------------------------------------

    def _compute_A(self) -> torch.Tensor:
        raise NotImplementedError

    def _compute_B(self) -> torch.Tensor:
        raise NotImplementedError

    def _has_bias_delta(self) -> bool:
        return False

    def _compute_bias_delta(self) -> Optional[torch.Tensor]:
        return None

    # -- shared maths ----------------------------------------------------

    def _resolved_base_weight(self) -> torch.Tensor:
        # Linear stores weight as [out, in]; subclasses that flatten conv
        # weights override to reshape here.
        return self._org_weight_ref

    def get_delta_weight(self) -> torch.Tensor:
        """Return ΔW = (W·A + B) * scaling, shaped [out, in]."""
        a = self._compute_A()
        b = self._compute_B()
        w = self._resolved_base_weight().to(dtype=a.dtype, device=a.device)
        # W is [out, in], A is [in, in] → W·A is [out, in]; B is [out, in].
        delta = w @ a + b
        return delta * self.scaling

    def get_delta_bias(self) -> Optional[torch.Tensor]:
        """Return Δbias = (c1·c2) * scaling, shaped [out], or None."""
        delta = self._compute_bias_delta()
        return delta * self.scaling if delta is not None else None

    def _module_dropout_drops(self) -> bool:
        if self.module_dropout <= 0.0 or not self.training:
            return False
        return torch.rand((), device="cpu").item() < self.module_dropout

    def _apply_rank_dropout(self, delta_w: torch.Tensor) -> torch.Tensor:
        if self.rank_dropout <= 0.0 or not self.training:
            return delta_w
        out_dim = delta_w.shape[0]
        drop_mask = torch.ones(out_dim, device=delta_w.device, dtype=delta_w.dtype)
        n_drop = max(1, int(out_dim * self.rank_dropout))
        drop_idx = torch.randperm(out_dim, device=delta_w.device)[:n_drop]
        drop_mask[drop_idx] = 0.0
        delta_w = delta_w * drop_mask.unsqueeze(1)
        return delta_w / max(1e-6, 1.0 - float(self.rank_dropout))

    def _make_output_zeros(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(*x.shape[:-1], self.out_features)

    def _can_use_no_materialize_forward(self, x: torch.Tensor) -> bool:
        if not self.no_materialize_forward:
            return False
        if self.rank_dropout > 0.0 and self.training:
            return False
        if self._has_bias_delta():
            return False  # bias path is cheap enough; keep materialized to stay simple
        return x.ndim >= 1 and x.shape[-1] == self.in_features

    def _forward_no_materialize(self, x: torch.Tensor) -> torch.Tensor:
        # ΔW·x = W·A·x + B·x.  Each path becomes three matmuls (no [in,in]
        # or [out,in] materialization of A, B), then we add and scale.
        w = self._resolved_base_weight().to(dtype=x.dtype, device=x.device)
        out = self._matmul_A_path(x, w) + self._matmul_B_path(x)
        return out * self.scaling

    def _matmul_A_path(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _matmul_B_path(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._module_dropout_drops():
            return self._make_output_zeros(x)
        x = self.dropout(x)
        if self._can_use_no_materialize_forward(x):
            return self._forward_no_materialize(x)
        delta_w = self.get_delta_weight()
        delta_w = self._apply_rank_dropout(delta_w)
        out = F.linear(x, delta_w.to(dtype=x.dtype))
        bias_delta = self.get_delta_bias()
        if bias_delta is not None:
            out = out + bias_delta.to(dtype=out.dtype)
        return out


# ---------------------------------------------------------------------------
# GLoRA — Phase 1 standard + Phase 2 extras (all defaults off)
# ---------------------------------------------------------------------------


class GLoRALinearLayer(_GeneralizedDeltaBase):
    """GLoRA for ``nn.Linear`` with optional Phase 2 extras.

    Parameterization (matches the LyCORIS ``a1/a2/b1/b2`` checkpoint
    convention so exports remain readable by downstream tooling):

    - ``a2 ∈ [r, in]``,  ``a1 ∈ [in, r]``  →  ``A = a1·a2 ∈ [in, in]``
    - ``b2 ∈ [r, in]``,  ``b1 ∈ [out, r]`` →  ``B = b1·b2 ∈ [out, in]``

    Optional bias path (only active when the base ``nn.Linear`` has a bias):

    - ``c2 ∈ [r]``,  ``c1 ∈ [out, r]`` →  ``Δbias = c1·c2``

    Initialization: ``a1, b1, c1`` use Kaiming; ``a2, b2, c2`` are zero so
    ΔW = 0 and Δbias = 0 at step 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        *,
        org_weight: torch.Tensor,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        no_materialize_forward: bool = False,
        org_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            rank,
            alpha,
            dropout,
            org_weight,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            no_materialize_forward=no_materialize_forward,
        )
        self.a1 = nn.Parameter(torch.empty(self.in_features, self.rank))
        self.a2 = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.b1 = nn.Parameter(torch.empty(self.out_features, self.rank))
        self.b2 = nn.Parameter(torch.empty(self.rank, self.in_features))
        self._bias_enabled = org_bias is not None
        if self._bias_enabled:
            self.c1 = nn.Parameter(torch.empty(self.out_features, self.rank))
            self.c2 = nn.Parameter(torch.empty(self.rank))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.a1, a=math.sqrt(5))
        nn.init.zeros_(self.a2)
        nn.init.kaiming_uniform_(self.b1, a=math.sqrt(5))
        nn.init.zeros_(self.b2)
        if self._bias_enabled:
            nn.init.kaiming_uniform_(self.c1, a=math.sqrt(5))
            nn.init.zeros_(self.c2)

    def _compute_A(self) -> torch.Tensor:
        return self.a1 @ self.a2  # [in, in]

    def _compute_B(self) -> torch.Tensor:
        return self.b1 @ self.b2  # [out, in]

    def _has_bias_delta(self) -> bool:
        return self._bias_enabled

    def _compute_bias_delta(self) -> Optional[torch.Tensor]:
        if not self._bias_enabled:
            return None
        return self.c1 @ self.c2  # [out]

    # no-materialize: x·a2ᵀ → ·a1ᵀ → ·Wᵀ , plus x·b2ᵀ → ·b1ᵀ
    def _matmul_A_path(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        a2 = self.a2.to(dtype=x.dtype, device=x.device)
        a1 = self.a1.to(dtype=x.dtype, device=x.device)
        h = F.linear(x, a2)        # [..., r]   x · a2ᵀ
        h = F.linear(h, a1)        # [..., in]  · a1ᵀ
        return F.linear(h, w)      # [..., out] · Wᵀ

    def _matmul_B_path(self, x: torch.Tensor) -> torch.Tensor:
        b2 = self.b2.to(dtype=x.dtype, device=x.device)
        b1 = self.b1.to(dtype=x.dtype, device=x.device)
        h = F.linear(x, b2)        # [..., r]
        return F.linear(h, b1)     # [..., out]


class GLoRAConv2dLayer(_GeneralizedDeltaBase):
    """GLoRA for ``nn.Conv2d`` with optional Phase 2 extras.

    Conv2d weights are flattened along ``[out, in_per_group * kH * kW]`` to
    reuse the same ΔW = W·A + B skeleton.  ``A`` reshapes the flattened input
    space; ``B`` is a low-rank delta in that same flattened space.

    When ``use_tucker=True`` and the kernel is not 1×1, the B path uses a
    three-segment tucker parameterization ``B = b1 · bm · b2`` where ``bm``
    is shaped along the kernel axes — this matches the LyCORIS LoKr tucker
    convention and reduces the parameter count for wide-kernel convs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        *,
        org_weight: torch.Tensor,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_tucker: bool = False,
        org_bias: Optional[torch.Tensor] = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(int(d) for d in kernel_size)
        self.stride = tuple(int(d) for d in stride)
        self.padding = tuple(int(d) for d in padding)
        self.dilation = tuple(int(d) for d in dilation)
        self.groups = max(int(groups), 1)
        in_per_group = self.in_channels // self.groups
        flat_in = in_per_group * self.kernel_size[0] * self.kernel_size[1]
        super().__init__(
            in_features=flat_in,
            out_features=self.out_channels,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            org_weight=org_weight,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            no_materialize_forward=False,  # conv path always materializes
        )
        self.a1 = nn.Parameter(torch.empty(self.in_features, self.rank))
        self.a2 = nn.Parameter(torch.empty(self.rank, self.in_features))
        kernel_is_unit = all(k == 1 for k in self.kernel_size)
        self._tucker_enabled = bool(use_tucker) and not kernel_is_unit
        if self._tucker_enabled:
            # B = b1 [out, r, 1, 1] · bm [r, r, kH, kW] · b2 [r, in_per_group]
            self.b1 = nn.Parameter(torch.empty(self.out_features, self.rank))
            self.bm = nn.Parameter(
                torch.empty(self.rank, self.rank, self.kernel_size[0], self.kernel_size[1])
            )
            self.b2 = nn.Parameter(torch.empty(self.rank, in_per_group))
        else:
            self.b1 = nn.Parameter(torch.empty(self.out_features, self.rank))
            self.b2 = nn.Parameter(torch.empty(self.rank, self.in_features))
        self._bias_enabled = org_bias is not None
        if self._bias_enabled:
            self.c1 = nn.Parameter(torch.empty(self.out_features, self.rank))
            self.c2 = nn.Parameter(torch.empty(self.rank))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.a1, a=math.sqrt(5))
        nn.init.zeros_(self.a2)
        nn.init.kaiming_uniform_(self.b1, a=math.sqrt(5))
        if self._tucker_enabled:
            nn.init.kaiming_uniform_(self.bm, a=math.sqrt(5))
        nn.init.zeros_(self.b2)
        if self._bias_enabled:
            nn.init.kaiming_uniform_(self.c1, a=math.sqrt(5))
            nn.init.zeros_(self.c2)

    def _resolved_base_weight(self) -> torch.Tensor:
        # Conv2d weight is [out, in/groups, kH, kW] → flatten to [out, flat_in]
        return self._org_weight_ref.reshape(self.out_channels, -1)

    def _compute_A(self) -> torch.Tensor:
        return self.a1 @ self.a2

    def _compute_B(self) -> torch.Tensor:
        if self._tucker_enabled:
            # Materialize tucker B back to [out, in_per_group, kH, kW] → [out, flat_in]
            # B = b1 @ bm @ b2, with bm carrying the kernel axes.
            # b1 [out, r], bm [r, r, kH, kW], b2 [r, in_per_group]
            # Step 1: w_tmp = einsum("or, rskl -> oskl", b1, bm) → [out, r, kH, kW]
            w_tmp = torch.einsum("or,rskl->oskl", self.b1, self.bm)
            # Step 2: w_full = einsum("oskl, si -> oikl", w_tmp, b2) → [out, in_per_group, kH, kW]
            w_full = torch.einsum("oskl,si->oikl", w_tmp, self.b2)
            return w_full.reshape(self.out_features, -1)
        return self.b1 @ self.b2

    def _has_bias_delta(self) -> bool:
        return self._bias_enabled

    def _compute_bias_delta(self) -> Optional[torch.Tensor]:
        if not self._bias_enabled:
            return None
        return self.c1 @ self.c2

    def get_delta_weight_matrix(self) -> torch.Tensor:
        return _GeneralizedDeltaBase.get_delta_weight(self)

    def get_delta_weight(self) -> torch.Tensor:
        return self.get_delta_weight_matrix().reshape(
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._module_dropout_drops():
            height = x.shape[-2]
            width = x.shape[-1]
            out_h = ((height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
            out_w = ((width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
            return x.new_zeros(x.shape[0], self.out_channels, out_h, out_w)

        delta_w = self.get_delta_weight()
        if self.rank_dropout > 0.0 and self.training:
            out_dim = delta_w.shape[0]
            drop_mask = torch.ones(out_dim, device=delta_w.device, dtype=delta_w.dtype)
            n_drop = max(1, int(out_dim * self.rank_dropout))
            drop_idx = torch.randperm(out_dim, device=delta_w.device)[:n_drop]
            drop_mask[drop_idx] = 0.0
            delta_w = delta_w * drop_mask.view(-1, 1, 1, 1)
            delta_w = delta_w / max(1e-6, 1.0 - float(self.rank_dropout))

        bias_delta = self.get_delta_bias()
        return F.conv2d(
            self.dropout(x),
            delta_w.to(dtype=x.dtype),
            bias=bias_delta.to(dtype=x.dtype) if bias_delta is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


# ---------------------------------------------------------------------------
# State-dict helpers (used by LyCORISInjector dispatch)
# ---------------------------------------------------------------------------

_GLORA_CORE_ATTRS: Tuple[str, ...] = ("a1", "a2", "b1", "b2")
_GLORA_OPTIONAL_ATTRS: Tuple[str, ...] = ("bm", "c1", "c2")


def _layer_has_attr(layer: nn.Module, attr: str) -> bool:
    return getattr(layer, attr, None) is not None


def collect_glora_layer_state(layer: nn.Module, base_name: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for attr in _GLORA_CORE_ATTRS:
        tensor = getattr(layer, attr, None)
        if tensor is not None:
            out[f"{base_name}.{attr}.weight"] = tensor.data
    for attr in _GLORA_OPTIONAL_ATTRS:
        if _layer_has_attr(layer, attr):
            tensor = getattr(layer, attr)
            out[f"{base_name}.{attr}.weight"] = tensor.data
    out[f"{base_name}.alpha"] = torch.tensor(float(layer.alpha))
    return out


def load_glora_layer_state(
    layer: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    base_name: str,
) -> Tuple[int, int]:
    loaded = 0
    total = 0
    expected_attrs = list(_GLORA_CORE_ATTRS) + [
        attr for attr in _GLORA_OPTIONAL_ATTRS if _layer_has_attr(layer, attr)
    ]
    for attr in expected_attrs:
        param = getattr(layer, attr, None)
        if param is None:
            continue
        total += 1
        key = f"{base_name}.{attr}.weight"
        if key not in state_dict:
            continue
        value = state_dict[key].to(device=param.device, dtype=param.dtype)
        if tuple(value.shape) != tuple(param.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: checkpoint {tuple(value.shape)} != layer {tuple(param.shape)}"
            )
        param.data.copy_(value)
        loaded += 1
    return loaded, total


def glora_state_dict_keys(base_name: str) -> Dict[str, str]:
    """Legacy helper for Phase 1 callers; returns the core tensor key map."""
    return {attr: f"{base_name}.{attr}.weight" for attr in _GLORA_CORE_ATTRS}


def is_glora_layer(layer: nn.Module) -> bool:
    return isinstance(layer, (GLoRALinearLayer, GLoRAConv2dLayer))


__all__ = [
    "_GeneralizedDeltaBase",
    "GLoRALinearLayer",
    "GLoRAConv2dLayer",
    "collect_glora_layer_state",
    "load_glora_layer_state",
    "glora_state_dict_keys",
    "is_glora_layer",
]
