# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Immiscible diffusion: minibatch noise↔data assignment by squared L2 distance.

Standard diffusion draws an independent Gaussian noise for each data point, so
within a minibatch the noise-to-data mapping crosses heavily — every data point
can be pushed toward (almost) any noise, which makes the denoiser's target
direction high-variance.  *Immiscible diffusion* (Li et al., 2024) reassigns the
sampled noises to data points so that each data point is paired with a *nearby*
noise (minimizing total squared L2 over the minibatch), making the noise/data
layers "immiscible" — the assignment no longer crosses — which empirically
speeds up convergence.  Crucially it helps **standard** diffusion (DDPM/EDM),
not only flow matching.

This is the L2 sibling of :func:`cosine_ot.minibatch_ot_cosine` (which pairs by
direction for rectified flow).  Same conventions: no-grad, ``B<=1`` passthrough,
greedy O(B²) assignment (optimal enough for the B = 1-16 typical of diffusion
training), shape preserved.

Clean-room Lulynx implementation; shares no source with any reference.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def minibatch_immiscible_l2(latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Reorder ``noise`` so each latent is paired with a near (low-L2) noise.

    Greedy assignment over the minibatch: process latents 0..B-1, each grabbing
    the unused noise with the smallest squared-L2 distance.  For the small
    batches typical in diffusion training this closely tracks the optimal
    (Hungarian) assignment at a fraction of the cost.

    Args:
        latents: ``(B, ...)`` encoded image latents.
        noise:   ``(B, ...)`` Gaussian noise, same shape as ``latents``.

    Returns:
        ``noise[perm]`` — noise reordered to its assigned latents; shape
        identical to ``noise``.  ``B<=1`` returns ``noise`` unchanged.
    """
    B = latents.shape[0]
    if B <= 1:
        return noise

    lat_flat = latents.reshape(B, -1).float()
    noi_flat = noise.reshape(B, -1).float()

    # cost[i, j] = ||latents[i] - noise[j]||^2 (monotone in L2, cheaper to rank)
    cost = torch.cdist(lat_flat, noi_flat, p=2)  # (B, B)

    perm = torch.zeros(B, dtype=torch.long, device=latents.device)
    used = torch.zeros(B, dtype=torch.bool, device=latents.device)
    for i in range(B):
        row = cost[i].clone()
        row[used] = float("inf")  # mask already-assigned noise columns
        j = int(row.argmin())
        perm[i] = j
        used[j] = True

    return noise[perm]


__all__ = ["minibatch_immiscible_l2"]
