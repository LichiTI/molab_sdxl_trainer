# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""OrthoLoRA — orthogonality-constrained LoRA (Phase 8.1 / #108).

Standard LoRA learns rank-r updates without any constraint on the
relationship between rows of ``lora_down`` or columns of ``lora_up``.
OrthoLoRA enforces orthogonality on these matrices, which has been
shown to:

  * Reduce interference between LoRA directions (cleaner low-rank basis).
  * Provide a smoother optimisation landscape.

Two projection strategies are supported:

  * ``"cayley"``  — keeps an orthogonal matrix on the Stiefel manifold via
    a Cayley transform.  Requires square sub-matrices.
  * ``"gram_schmidt"`` — applies Gram-Schmidt re-orthogonalisation after
    each optimizer step.  Works for any rectangular shape.

The module exposes a callable ``OrthoLoRAProjector`` that the trainer
calls once per optimizer step (typically right before zero_grad).
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orthogonalisation primitives
# ---------------------------------------------------------------------------

def gram_schmidt(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise Gram-Schmidt orthogonalisation.

    Returns a matrix of the same shape whose rows are orthonormal.
    Works for rectangular ``[r, d]`` matrices when ``r <= d``.
    """
    if matrix.dim() != 2:
        return matrix
    rows, _ = matrix.shape
    out = matrix.clone().to(torch.float32)
    for i in range(rows):
        v = out[i]
        for j in range(i):
            v = v - torch.dot(out[j], v) * out[j]
        norm = torch.linalg.norm(v) + eps
        out[i] = v / norm
    return out.to(matrix.dtype)


def cayley_orthogonalise(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cayley transform projection onto the orthogonal group.

    For an arbitrary square matrix M, returns ``Q = (I - A)(I + A)^-1``
    where ``A = (M - M^T) / 2`` is the skew-symmetric part.  Only valid
    for square matrices.
    """
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        return matrix

    M = matrix.to(torch.float32)
    A = 0.5 * (M - M.t())
    n = A.shape[0]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    try:
        Q = torch.linalg.solve(I + A, I - A)
    except RuntimeError:
        # Singular matrix — fall back to gram-schmidt
        return gram_schmidt(matrix)
    return Q.to(matrix.dtype)


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class OrthoLoRAProjector:
    """Apply an orthogonality projection to LoRA weights once per step.

    The projector inspects each registered LoRA layer, locates its
    ``lora_down`` and ``lora_up`` parameters, and re-projects them onto
    the orthogonal manifold using the chosen strategy.

    Parameters
    ----------
    method : str
        ``"gram_schmidt"`` (default) or ``"cayley"``.
    interval : int
        Apply the projection every ``interval`` calls.  Default 1 = every step.
    target_layers : list of str, optional
        Only project layers whose registered name matches one of these.
        ``None`` means project all layers.
    """

    def __init__(
        self,
        *,
        method: str = "gram_schmidt",
        interval: int = 1,
        target_layers: Optional[List[str]] = None,
    ) -> None:
        if method not in {"gram_schmidt", "cayley"}:
            raise ValueError(f"Unknown ortho method: {method}")
        self.method = method
        self.interval = max(1, int(interval))
        self.target_layers = set(target_layers) if target_layers else None
        self._step_counter = 0
        self._layers: List[tuple] = []  # list of (name, lora_layer)

    def register_layer(self, name: str, layer: nn.Module) -> None:
        """Register a LoRA wrapper / lycoris layer for projection."""
        if self.target_layers and name not in self.target_layers:
            return
        self._layers.append((name, layer))

    def register_from_injector(self, injector: object) -> int:
        """Register every layer present on the injector's ``injected_layers`` map."""
        layers = getattr(injector, "injected_layers", None)
        if not isinstance(layers, dict):
            return 0
        added = 0
        for name, layer in layers.items():
            self.register_layer(name, layer)
            added += 1
        return added

    @torch.no_grad()
    def step(self) -> int:
        """Run one projection sweep.  Returns the count of projected matrices."""
        self._step_counter += 1
        if self._step_counter % self.interval != 0:
            return 0

        projected = 0
        for _, layer in self._layers:
            for matrix in self._collect_matrices(layer):
                if matrix is None:
                    continue
                self._project_inplace(matrix)
                projected += 1
        return projected

    def reset(self) -> None:
        self._step_counter = 0
        self._layers.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_matrices(self, layer: nn.Module) -> List[Optional[torch.Tensor]]:
        """Find the lora_down / lora_up weight tensors on ``layer``.

        Supports the standard LoRALinear layout (``lora.lora_down.weight`` /
        ``lora.lora_up.weight``) and the alternate ``lora_A`` / ``lora_B``
        layout used by DoRA-style wrappers.
        """
        matrices: List[Optional[torch.Tensor]] = []
        adapter = getattr(layer, "lora", layer)

        for attr in ("lora_down", "lora_A"):
            sub = getattr(adapter, attr, None)
            weight = getattr(sub, "weight", sub) if sub is not None else None
            if isinstance(weight, torch.Tensor):
                matrices.append(weight.data)
                break
        for attr in ("lora_up", "lora_B"):
            sub = getattr(adapter, attr, None)
            weight = getattr(sub, "weight", sub) if sub is not None else None
            if isinstance(weight, torch.Tensor):
                matrices.append(weight.data)
                break
        return matrices

    def _project_inplace(self, matrix: torch.Tensor) -> None:
        if self.method == "cayley" and matrix.shape[0] == matrix.shape[1]:
            new_w = cayley_orthogonalise(matrix)
        else:
            new_w = gram_schmidt(matrix)
        matrix.copy_(new_w)
