# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
# Warehouse implementation.

import torch
from typing import Callable


@torch.no_grad()
def select_optimal_noise(
    latents: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], float],
    n_candidates: int = 4,
) -> torch.Tensor:
    if n_candidates <= 1:
        return torch.randn_like(latents)

    best_noise = None
    best_loss = float("inf")

    for _ in range(n_candidates):
        candidate = torch.randn_like(latents)
        loss_val = loss_fn(candidate)
        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_noise = candidate

    return best_noise

