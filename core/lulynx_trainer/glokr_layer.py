# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""GLoKr — Kronecker-parameterized Generalized adapter (lulynx clean-room research).

GLoKr keeps the Generalized two-path skeleton

    ΔW = W·A + B

and parameterizes both A and B as Kronecker products instead of low-rank
factorizations.  This is a project-original design: LyCORIS lists GLoKr as
``TODO`` with no implementation, so the maths below are derived from scratch
against the Generalized skeleton plus the standard Kronecker factorization
contract.

Math (Linear, ``in = in_a × in_b``, ``out = out_a × out_b``):

    A1 ∈ [in_a, in_a],  A2 ∈ [in_b, in_b]   →   A = A1 ⊗ A2 ∈ [in, in]
    B1 ∈ [out_a, in_a], B2 ∈ [out_b, in_b]  →   B = B1 ⊗ B2 ∈ [out, in]

The B path mirrors the standard LoKr delta layout so callers familiar with
``factor``-based factorization see the same factor semantics; only the keys
(``glokr_a1/a2/b1/b2``) differ to mark the contract as GLoKr.

Initialization (ΔW = 0 at step 0):

    A1 ~ Kaiming, A2 = 0  →  A = 0
    B1 ~ Kaiming, B2 = 0  →  B = 0

Default policy:

- The shared base class' rank/module-dropout knobs still apply.
- ``no_materialize_forward`` is supported via Kronecker-aware chained
  matmuls; parity against the materialized delta is asserted in the smoke.
- Bias adaptation reuses the GLoRA-style ``c1·c2`` low-rank bias delta when
  the base module has a bias.  Bias is bounded by ``out`` so a low-rank
  approximation is enough — there is no benefit to a Kronecker-shaped bias.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generalized_adapters import _GeneralizedDeltaBase


# ---------------------------------------------------------------------------
# Factorization helper (clean-room — picks an integer factor that divides
# both ``in`` and ``out``, biased toward roughly balanced splits).
# ---------------------------------------------------------------------------


def _select_glokr_factor(in_dim: int, out_dim: int, requested: int) -> int:
    """Choose a factor ``f`` so that ``in_dim % f == 0`` and ``out_dim % f == 0``.

    When the requested factor does not divide both dimensions, fall back to
    the closest balanced candidate (search ``√max(dim)`` downward, then
    common small factors).  Returning ``1`` is the safe fallback and yields a
    degenerate Kronecker (``A2/B2`` become 1×1 scalars).
    """
    candidates = []
    if requested and requested > 0:
        candidates.append(int(requested))
    # Search around √max(dim) — this maximizes Kronecker parameter savings.
    target = max(in_dim, out_dim)
    sqrt_hi = max(2, int(math.isqrt(target)))
    candidates.extend(range(sqrt_hi, 1, -1))
    candidates.extend([16, 12, 8, 6, 4, 3, 2, 1])
    seen: set[int] = set()
    for candidate in candidates:
        if candidate in seen or candidate <= 0:
            continue
        seen.add(candidate)
        if in_dim % candidate == 0 and out_dim % candidate == 0:
            return candidate
    return 1


def _split_dim(dim: int, factor: int) -> Tuple[int, int]:
    """Return ``(left, right)`` with ``left * right == dim`` and ``right == factor``."""
    if factor <= 0 or dim % factor != 0:
        return dim, 1
    return dim // factor, factor


# ---------------------------------------------------------------------------
# GLoKr Linear
# ---------------------------------------------------------------------------


class GLoKrLinearLayer(_GeneralizedDeltaBase):
    """Kronecker-parameterized GLoRA for ``nn.Linear``.

    Tensor naming intentionally diverges from LoKr (``glokr_*``) so loaders
    do not confuse the two contracts: GLoKr depends on the frozen base
    weight; LoKr does not.
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
        factor: int = -1,
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
        # Choose a single shared factor so A and B agree on the split.
        resolved_factor = _select_glokr_factor(self.in_features, self.out_features, int(factor))
        self.factor = resolved_factor
        self.in_a, self.in_b = _split_dim(self.in_features, resolved_factor)
        self.out_a, self.out_b = _split_dim(self.out_features, resolved_factor)

        # A = A1 ⊗ A2 ∈ [in, in]
        self.glokr_a1 = nn.Parameter(torch.empty(self.in_a, self.in_a))
        self.glokr_a2 = nn.Parameter(torch.empty(self.in_b, self.in_b))
        # B = B1 ⊗ B2 ∈ [out, in]
        self.glokr_b1 = nn.Parameter(torch.empty(self.out_a, self.in_a))
        self.glokr_b2 = nn.Parameter(torch.empty(self.out_b, self.in_b))

        # Optional GLoRA-style bias delta — bounded by ``out`` so a low-rank
        # factorization suffices; no benefit to Kronecker-shaped bias.
        self._bias_enabled = org_bias is not None
        if self._bias_enabled:
            self.c1 = nn.Parameter(torch.empty(self.out_features, self.rank))
            self.c2 = nn.Parameter(torch.empty(self.rank))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.glokr_a1, a=math.sqrt(5))
        nn.init.zeros_(self.glokr_a2)
        nn.init.kaiming_uniform_(self.glokr_b1, a=math.sqrt(5))
        nn.init.zeros_(self.glokr_b2)
        if self._bias_enabled:
            nn.init.kaiming_uniform_(self.c1, a=math.sqrt(5))
            nn.init.zeros_(self.c2)

    # -- maths -----------------------------------------------------------

    def _compute_A(self) -> torch.Tensor:
        return torch.kron(self.glokr_a1, self.glokr_a2)

    def _compute_B(self) -> torch.Tensor:
        return torch.kron(self.glokr_b1, self.glokr_b2)

    def _has_bias_delta(self) -> bool:
        return self._bias_enabled

    def _compute_bias_delta(self) -> Optional[torch.Tensor]:
        if not self._bias_enabled:
            return None
        return self.c1 @ self.c2  # [out]

    # -- no-materialize fast path ---------------------------------------
    #
    # Identity (derived against torch.kron's index convention
    # ``kron(M1,M2)[i1·k2+i2, j1·k4+j2] = M1[i1,j1]·M2[i2,j2]``):
    #
    #     y = (M1 ⊗ M2) · x   ⇔   Y = M1 · X · M2ᵀ
    #
    # where ``X = x.reshape(in_a, in_b)`` (PyTorch row-major) and
    # ``Y = y.reshape(out_a, out_b)`` for the B path (out_a = in_a for A).
    # This lets us avoid materializing the [in,in] / [out,in] matrix.

    def _matmul_A_path(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        a1 = self.glokr_a1.to(dtype=x.dtype, device=x.device)
        a2 = self.glokr_a2.to(dtype=x.dtype, device=x.device)
        x_view = x.reshape(*x.shape[:-1], self.in_a, self.in_b)   # [..., in_a, in_b]
        h = a1 @ x_view                                            # [..., in_a, in_b]
        h = h @ a2.transpose(-1, -2)                               # [..., in_a, in_b]
        h = h.reshape(*x.shape[:-1], self.in_features)             # [..., in]
        return F.linear(h, w)                                      # [..., out]

    def _matmul_B_path(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.glokr_b1.to(dtype=x.dtype, device=x.device)
        b2 = self.glokr_b2.to(dtype=x.dtype, device=x.device)
        x_view = x.reshape(*x.shape[:-1], self.in_a, self.in_b)   # [..., in_a, in_b]
        h = b1 @ x_view                                            # [..., out_a, in_b]
        h = h @ b2.transpose(-1, -2)                               # [..., out_a, out_b]
        return h.reshape(*x.shape[:-1], self.out_features)         # [..., out]


# ---------------------------------------------------------------------------
# State-dict helpers (used by LyCORISInjector dispatch)
# ---------------------------------------------------------------------------

_GLOKR_CORE_ATTRS: Tuple[str, ...] = ("glokr_a1", "glokr_a2", "glokr_b1", "glokr_b2")
_GLOKR_OPTIONAL_ATTRS: Tuple[str, ...] = ("c1", "c2")


def collect_glokr_layer_state(layer: nn.Module, base_name: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for attr in _GLOKR_CORE_ATTRS:
        tensor = getattr(layer, attr, None)
        if tensor is not None:
            out[f"{base_name}.{attr}"] = tensor.data
    for attr in _GLOKR_OPTIONAL_ATTRS:
        tensor = getattr(layer, attr, None)
        if tensor is not None:
            out[f"{base_name}.{attr}.weight"] = tensor.data
    out[f"{base_name}.alpha"] = torch.tensor(float(layer.alpha))
    return out


def load_glokr_layer_state(
    layer: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    base_name: str,
) -> Tuple[int, int]:
    loaded = 0
    total = 0
    for attr in _GLOKR_CORE_ATTRS:
        param = getattr(layer, attr, None)
        if param is None:
            continue
        total += 1
        key = f"{base_name}.{attr}"
        if key not in state_dict:
            continue
        value = state_dict[key].to(device=param.device, dtype=param.dtype)
        if tuple(value.shape) != tuple(param.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: checkpoint {tuple(value.shape)} != layer {tuple(param.shape)}"
            )
        param.data.copy_(value)
        loaded += 1
    for attr in _GLOKR_OPTIONAL_ATTRS:
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


def is_glokr_layer(layer: nn.Module) -> bool:
    return isinstance(layer, GLoKrLinearLayer)


__all__ = [
    "GLoKrLinearLayer",
    "collect_glokr_layer_state",
    "load_glokr_layer_state",
    "is_glokr_layer",
    "_select_glokr_factor",
]
