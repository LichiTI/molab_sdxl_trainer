# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""AdaLN Guidance — learnable scale/shift bias injected into DiT AdaLN
modulation paths to steer generation (#114).

DiT-style models (Anima, Newbie) use Adaptive LayerNorm where each
transformer block computes ``modulate(norm(x), shift, scale, gate)``.
AdaLN Guidance trains a small per-layer bias (``shift_bias``,
``scale_bias``) that is added to the standard modulation parameters at
each block.  This gives a low-parameter steering knob that complements
LoRA-style adapters.

Typical wiring::

    from .adaln_guidance import install_adaln_guidance, get_adaln_guidance_params

    handles = install_adaln_guidance(
        dit_model,
        num_blocks=24,
        modulation_dim=1152,
        scale=0.0,           # start as a no-op
    )
    optimizer.add_param_group({"params": get_adaln_guidance_params(dit_model)})
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AdaLNGuidanceBias(nn.Module):
    """Learnable per-block shift / scale bias added to AdaLN modulation outputs.

    The expected modulation tensor is ``[batch, modulation_dim * 6]`` (the
    standard DiT layout: 6 chunks for shift_msa, scale_msa, gate_msa,
    shift_mlp, scale_mlp, gate_mlp).  This module adds a learnable bias to
    the ``shift_*`` and ``scale_*`` chunks only, leaving the gates untouched.
    """

    def __init__(
        self,
        modulation_dim: int,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.modulation_dim = modulation_dim
        # Bias for shift_msa, scale_msa, shift_mlp, scale_mlp (4 chunks of dim)
        self.shift_msa_bias = nn.Parameter(torch.full((modulation_dim,), init_scale))
        self.scale_msa_bias = nn.Parameter(torch.full((modulation_dim,), init_scale))
        self.shift_mlp_bias = nn.Parameter(torch.full((modulation_dim,), init_scale))
        self.scale_mlp_bias = nn.Parameter(torch.full((modulation_dim,), init_scale))

    def forward(self, modulation: torch.Tensor) -> torch.Tensor:
        """Add bias to the modulation tensor.

        Expected shape: ``[batch, 6 * modulation_dim]`` ordered as
        ``[shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp]``.
        Falls back to a no-op for unexpected shapes.
        """
        if modulation.dim() < 2:
            return modulation
        last = modulation.shape[-1]
        if last != 6 * self.modulation_dim:
            # Not the standard DiT layout — skip silently
            return modulation

        chunks = modulation.chunk(6, dim=-1)
        shift_msa = chunks[0] + self.shift_msa_bias
        scale_msa = chunks[1] + self.scale_msa_bias
        shift_mlp = chunks[3] + self.shift_mlp_bias
        scale_mlp = chunks[4] + self.scale_mlp_bias
        return torch.cat([shift_msa, scale_msa, chunks[2], shift_mlp, scale_mlp, chunks[5]], dim=-1)


# ---------------------------------------------------------------------------
# Installation helpers
# ---------------------------------------------------------------------------

def _find_modulation_modules(model: nn.Module) -> List[tuple]:
    """Locate modules likely to be the AdaLN modulation projection in DiT.

    Heuristic: any nn.Linear named like ``adaLN_modulation`` or
    ``modulation`` whose out_features is divisible by 6.
    """
    targets: List[tuple] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        ident = name.lower()
        if "modulation" not in ident and "adaln" not in ident:
            continue
        if module.out_features % 6 != 0:
            continue
        targets.append((name, module))
    return targets


def install_adaln_guidance(
    model: nn.Module,
    *,
    init_scale: float = 0.0,
    modulation_dim: Optional[int] = None,
) -> List[nn.Module]:
    """Install AdaLN guidance biases on every modulation projection in *model*.

    Returns the list of installed :class:`AdaLNGuidanceBias` modules.
    Calling this twice is a no-op (already-installed modules are skipped).
    """
    if model is None:
        return []

    targets = _find_modulation_modules(model)
    if not targets:
        logger.info("install_adaln_guidance: no AdaLN modulation modules found")
        return []

    installed: List[nn.Module] = []
    for name, linear in targets:
        if hasattr(linear, "_adaln_guidance"):
            installed.append(linear._adaln_guidance)
            continue

        dim = modulation_dim or (linear.out_features // 6)
        bias = AdaLNGuidanceBias(dim, init_scale=init_scale)
        bias.to(device=linear.weight.device, dtype=linear.weight.dtype)
        linear._adaln_guidance = bias  # type: ignore[attr-defined]

        # Register as a child so state_dict / .to() pick it up
        linear.add_module("_adaln_guidance_module", bias)

        # Patch forward to apply the bias post-projection
        original_forward = linear.forward

        def _patched(self_module, input_tensor, _orig=original_forward, _bias_mod=bias):
            out = _orig(input_tensor)
            return _bias_mod(out)

        # Bind as a method on the linear instance
        import types
        linear.forward = types.MethodType(_patched, linear)
        linear._adaln_original_forward = original_forward  # type: ignore[attr-defined]

        installed.append(bias)

    logger.info("install_adaln_guidance: installed %d guidance bias modules", len(installed))
    return installed


def remove_adaln_guidance(model: nn.Module) -> int:
    """Remove all installed AdaLN guidance hooks. Returns count removed."""
    if model is None:
        return 0
    removed = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "_adaln_guidance"):
            if hasattr(module, "_adaln_original_forward"):
                module.forward = module._adaln_original_forward  # type: ignore[assignment]
                delattr(module, "_adaln_original_forward")
            delattr(module, "_adaln_guidance")
            if hasattr(module, "_adaln_guidance_module"):
                # Drop child registration
                try:
                    del module._modules["_adaln_guidance_module"]
                except Exception:
                    pass
            removed += 1
    if removed:
        logger.info("remove_adaln_guidance: removed %d biases", removed)
    return removed


def get_adaln_guidance_params(model: nn.Module) -> List[nn.Parameter]:
    """Collect all guidance bias parameters for optimizer registration."""
    params: List[nn.Parameter] = []
    if model is None:
        return params
    for _, module in model.named_modules():
        bias = getattr(module, "_adaln_guidance", None)
        if isinstance(bias, AdaLNGuidanceBias):
            params.extend(p for p in bias.parameters() if p.requires_grad)
    return params
