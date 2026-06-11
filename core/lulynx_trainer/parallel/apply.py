# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Declarative tensor-parallel application (cleanroom Lulynx).

:class:`ParallelSpec` maps module-name substrings to a parallel style
(``"column"`` / ``"row"``).  :func:`apply_tensor_parallel` walks the model and
swaps each matched ``nn.Linear`` for the corresponding parallel layer built from
its weights, so an existing module tree is sharded in place.

Typical attention/MLP spec::

    spec = ParallelSpec({
        "q_proj": "column", "k_proj": "column", "v_proj": "column",
        "output_proj": "row",          # row consumes the column-sharded heads
        "layer1": "column", "layer2": "row",   # MLP up / down
    })

With ``tp_size == 1`` the swap is a no-op in behavior (the parallel layers hold
full weights and act like the originals).

Clean-room Lulynx module; references no external parallelism source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from .layers import ColumnParallelLinear, RowParallelLinear
from .process_groups import ProcessGroups


@dataclass
class ParallelSpec:
    """Substring-pattern → parallel style (``"column"`` | ``"row"``)."""

    entries: dict[str, str]

    def match(self, module_path: str) -> Optional[str]:
        for pattern, style in self.entries.items():
            if pattern in module_path:
                return style
        return None


def _resolve_parent(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]


def apply_tensor_parallel(
    model: nn.Module,
    spec: ParallelSpec,
    groups: Optional[ProcessGroups] = None,
    *,
    gather_column_output: bool = False,
) -> int:
    """Swap matched ``nn.Linear`` modules in-place for parallel layers.

    Returns the number of layers replaced.  ``column`` layers keep their output
    sharded by default (``gather_column_output=False``) so a following ``row``
    layer can consume it; set ``gather_column_output=True`` for a standalone
    column projection that must return a full tensor.
    """
    replacements: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        style = spec.match(name)
        if style == "column":
            replacements.append((name, ColumnParallelLinear.from_linear(
                module, groups=groups, gather_output=gather_column_output)))
        elif style == "row":
            replacements.append((name, RowParallelLinear.from_linear(
                module, groups=groups, input_is_parallel=True)))

    for name, new_module in replacements:
        parent, attr = _resolve_parent(model, name)
        setattr(parent, attr, new_module)
    return len(replacements)


__all__ = ["ParallelSpec", "apply_tensor_parallel"]
