# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Hutchinson trace estimator for the Hessian, restricted to LoRA parameters.

This is a **deep diagnostic** feature — each call performs one additional
backward pass (Hessian-vector product), so it is expensive and should only
be enabled when explicitly requested by the user.

Algorithm:
    Tr(H) ≈ E[v^T H v]  where v ~ Rademacher(±1)

For each random vector v:
    1. g  = autograd.grad(loss, params, create_graph=True)
    2. gv = g · v
    3. Hv = autograd.grad(gv, params)
    4. trace_sample = v · Hv
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class HessianSnapshot:
    trace: float
    num_params: int
    num_vectors: int

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class LayerwiseHessianSnapshot:
    layer_traces: Dict[str, float]
    total_trace: float
    num_vectors: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "layer_traces": dict(self.layer_traces),
            "total_trace": self.total_trace,
            "num_vectors": self.num_vectors,
        }


class HessianTraceEstimator:
    """Hutchinson trace estimator restricted to trainable (LoRA) parameters."""

    def __init__(self, num_vectors: int = 1) -> None:
        self._num_vectors = max(num_vectors, 1)
        self._last_snapshot: Optional[HessianSnapshot] = None
        self._last_layerwise: Optional[LayerwiseHessianSnapshot] = None

    def estimate(
        self,
        loss: torch.Tensor,
        params: List[nn.Parameter],
    ) -> HessianSnapshot:
        """Estimate Tr(H) via Hutchinson with Rademacher vectors.

        Must be called BEFORE ``loss.backward()`` because we need
        ``create_graph=True`` on the first-order gradients.
        """
        params = [p for p in params if p.requires_grad]
        if not params:
            snap = HessianSnapshot(trace=0.0, num_params=0, num_vectors=0)
            self._last_snapshot = snap
            return snap

        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True,
        )
        flat_grad = torch.cat([g.contiguous().flatten() for g in grads])
        num_params = flat_grad.numel()

        trace_sum = 0.0
        for _ in range(self._num_vectors):
            v = (
                torch.randint(0, 2, flat_grad.shape, device=flat_grad.device)
                .float()
                .mul_(2)
                .sub_(1)
            )

            gv = (flat_grad * v).sum()

            hvp_grads = torch.autograd.grad(
                gv, params, retain_graph=True,
            )
            flat_hvp = torch.cat(
                [h.contiguous().flatten() for h in hvp_grads]
            )

            trace_sum += float((v * flat_hvp).sum())

        trace_est = trace_sum / self._num_vectors

        snap = HessianSnapshot(
            trace=trace_est,
            num_params=num_params,
            num_vectors=self._num_vectors,
        )
        self._last_snapshot = snap
        return snap

    @property
    def last_snapshot(self) -> Optional[HessianSnapshot]:
        return self._last_snapshot

    def estimate_per_layer(
        self,
        loss: torch.Tensor,
        named_params: List[Tuple[str, nn.Parameter]],
    ) -> LayerwiseHessianSnapshot:
        """Per-layer Hessian trace via a single HVP — same cost as aggregate.

        ``named_params`` should be a list of (name, param) tuples, e.g. from
        ``[(n, p) for n, p in model.named_parameters() if p.requires_grad]``.
        """
        filtered = [(n, p) for n, p in named_params if p.requires_grad]
        if not filtered:
            snap = LayerwiseHessianSnapshot(layer_traces={}, total_trace=0.0, num_vectors=0)
            self._last_layerwise = snap
            return snap

        names = [n for n, _ in filtered]
        params = [p for _, p in filtered]
        sizes = [p.numel() for p in params]

        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True,
        )
        flat_grad = torch.cat([g.contiguous().flatten() for g in grads])

        layer_trace_sums: Dict[str, float] = {n: 0.0 for n in names}
        total_sum = 0.0

        for _ in range(self._num_vectors):
            v = (
                torch.randint(0, 2, flat_grad.shape, device=flat_grad.device)
                .float()
                .mul_(2)
                .sub_(1)
            )
            gv = (flat_grad * v).sum()
            hvp_grads = torch.autograd.grad(gv, params, retain_graph=True)

            offset = 0
            for i, name in enumerate(names):
                n = sizes[i]
                v_slice = v[offset : offset + n]
                h_slice = hvp_grads[i].contiguous().flatten()
                trace_contrib = float((v_slice * h_slice).sum())
                layer_trace_sums[name] += trace_contrib
                total_sum += trace_contrib
                offset += n

        nv = self._num_vectors
        layer_traces = {n: layer_trace_sums[n] / nv for n in names}
        total_trace = total_sum / nv

        snap = LayerwiseHessianSnapshot(
            layer_traces=layer_traces,
            total_trace=total_trace,
            num_vectors=nv,
        )
        self._last_layerwise = snap
        return snap

    @property
    def last_layerwise(self) -> Optional[LayerwiseHessianSnapshot]:
        return self._last_layerwise
