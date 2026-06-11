# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""ReFT — Representation Finetuning (Phase 8.4 / #112).

ReFT trains low-rank interventions on hidden states at selected layers
while leaving the base model weights frozen.  At each intervention
point a forward hook applies::

    h' = h + W2 @ (W1 @ h + b)

where W1: (rank, hidden) and W2: (hidden, rank).  The rank is small
(typically 4-16) so the parameter cost is negligible.

This is a generic, model-agnostic implementation that hooks onto any
``nn.Module`` whose forward returns a tensor or a tuple where the first
element is a tensor.

Typical wiring::

    handles = install_reft(model, target_modules=["block_0", "block_5"], rank=8)
    optimizer.add_param_group({"params": get_reft_params(model)})
    # ... train ...
    remove_reft(model)
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intervention module
# ---------------------------------------------------------------------------

class ReFTIntervention(nn.Module):
    """Low-rank residual intervention applied to hidden states.

    Parameters
    ----------
    hidden_size : int
        Last-dim size of the hidden state we intervene on.
    rank : int
        Bottleneck rank.  ``rank << hidden_size`` keeps params small.
    init_scale : float
        Magnitude for ``W2`` initialisation.  ``0.0`` makes the
        intervention a no-op at the start (LoRA-style).
    """

    def __init__(self, hidden_size: int, rank: int = 8, init_scale: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.W1 = nn.Linear(hidden_size, rank, bias=True)
        self.W2 = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.W1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W1.bias)
        if init_scale > 0:
            nn.init.normal_(self.W2.weight, mean=0.0, std=init_scale)
        else:
            nn.init.zeros_(self.W2.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta = self.W2(self.W1(hidden))
        return hidden + delta


# ---------------------------------------------------------------------------
# Installation helpers
# ---------------------------------------------------------------------------

def _resolve_module(root: nn.Module, name: str) -> Optional[nn.Module]:
    """Return the sub-module at the dotted path, or None if not found."""
    if not name:
        return None
    parts = name.split(".")
    obj: nn.Module = root
    for p in parts:
        if hasattr(obj, p):
            child = getattr(obj, p)
            if isinstance(child, nn.Module):
                obj = child
                continue
        if isinstance(obj, nn.ModuleList) and p.isdigit():
            obj = obj[int(p)]
            continue
        if isinstance(obj, nn.ModuleDict) and p in obj:
            obj = obj[p]
            continue
        return None
    return obj


def _infer_hidden_size(module: nn.Module) -> Optional[int]:
    """Best-effort hidden-size inference from a module's weights."""
    for name in ("hidden_size", "embed_dim", "dim"):
        if hasattr(module, name):
            try:
                return int(getattr(module, name))
            except Exception:
                pass
    # Fallback: walk children for a Linear/LayerNorm
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child.out_features
        if isinstance(child, (nn.LayerNorm,)):
            normalized_shape = child.normalized_shape
            if isinstance(normalized_shape, (tuple, list)) and normalized_shape:
                return int(normalized_shape[-1])
    return None


def install_reft(
    model: nn.Module,
    target_modules: Sequence[str],
    *,
    rank: int = 8,
    init_scale: float = 0.0,
    hidden_size: Optional[int] = None,
) -> List[nn.Module]:
    """Install ReFT interventions at each ``target_modules`` site.

    Returns the list of created intervention modules.  Repeated calls on
    the same site are no-ops.
    """
    if model is None:
        return []

    interventions: List[nn.Module] = []
    for path in target_modules:
        module = _resolve_module(model, path)
        if module is None:
            logger.warning("install_reft: target '%s' not found, skipping", path)
            continue
        if hasattr(module, "_reft_intervention"):
            interventions.append(module._reft_intervention)
            continue

        dim = hidden_size or _infer_hidden_size(module)
        if dim is None:
            logger.warning("install_reft: cannot infer hidden_size for '%s'", path)
            continue

        intervention = ReFTIntervention(dim, rank=rank, init_scale=init_scale)
        try:
            ref_param = next(p for p in module.parameters() if p.is_floating_point())
            intervention.to(device=ref_param.device, dtype=ref_param.dtype)
        except StopIteration:
            pass

        module._reft_intervention = intervention  # type: ignore[attr-defined]
        module.add_module("_reft_intervention_module", intervention)

        handle = module.register_forward_hook(_make_reft_hook(intervention))
        module._reft_hook_handle = handle  # type: ignore[attr-defined]

        interventions.append(intervention)

    if interventions:
        logger.info("install_reft: installed %d interventions (rank=%d)", len(interventions), rank)
    return interventions


def _make_reft_hook(intervention: nn.Module):
    def _hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            if output.shape[-1] == intervention.hidden_size:
                return intervention(output)
            return output
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            primary = output[0]
            if primary.shape[-1] == intervention.hidden_size:
                new_primary = intervention(primary)
                return (new_primary,) + output[1:]
        return output
    return _hook


def remove_reft(model: nn.Module) -> int:
    """Remove all ReFT hooks and intervention modules. Returns count removed."""
    if model is None:
        return 0
    removed = 0
    for _, module in model.named_modules():
        handle = getattr(module, "_reft_hook_handle", None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
            try:
                delattr(module, "_reft_hook_handle")
            except Exception:
                pass
        if hasattr(module, "_reft_intervention"):
            try:
                delattr(module, "_reft_intervention")
            except Exception:
                pass
            if "_reft_intervention_module" in module._modules:
                try:
                    del module._modules["_reft_intervention_module"]
                except Exception:
                    pass
            removed += 1
    if removed:
        logger.info("remove_reft: removed %d interventions", removed)
    return removed


def get_reft_params(model: nn.Module) -> List[nn.Parameter]:
    """Collect all trainable ReFT parameters for optimizer registration."""
    params: List[nn.Parameter] = []
    if model is None:
        return params
    for _, module in model.named_modules():
        intervention = getattr(module, "_reft_intervention", None)
        if isinstance(intervention, ReFTIntervention):
            params.extend(p for p in intervention.parameters() if p.requires_grad)
    return params


def get_reft_state_dict(model: nn.Module, prefix: str = "reft") -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    if model is None:
        return state
    for name, module in model.named_modules():
        intervention = getattr(module, "_reft_intervention", None)
        if isinstance(intervention, ReFTIntervention):
            safe_name = name.replace(".", "_") if name else "root"
            for key, value in intervention.state_dict().items():
                state[f"{prefix}.{safe_name}.{key}"] = value
    return state
