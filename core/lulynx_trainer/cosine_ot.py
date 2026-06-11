# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""
Minibatch optimal transport noise pairing using cosine similarity.

Warehouse implementation for rectified flow training.
Pairs latents with noise samples that are most similar in direction,
which reduces transport cost and produces straighter flow trajectories.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def minibatch_ot_cosine(latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Pair noise samples to latents using greedy cosine-similarity optimal transport.

    For each latent, assigns the unused noise sample with the highest cosine
    similarity (closest direction in latent space). Rows are processed in
    index order 0..B-1; this O(B^2) greedy is effective for the small batch
    sizes typical in diffusion training (B = 1-16).

    Args:
        latents: Tensor of shape (B, C, H, W) or (B, D) representing encoded
                 image latents.
        noise:   Tensor of the same shape as `latents` containing noise samples
                 to be reordered.

    Returns:
        noise[perm] — noise reordered so that noise[perm[i]] is the best
        cosine-similar match for latents[i].  Shape is identical to `noise`.
    """
    B = latents.shape[0]
    if B <= 1:
        return noise

    # Flatten to (B, D) and cast to float32 for stable norm computation.
    lat_flat = latents.reshape(B, -1).float()
    noi_flat = noise.reshape(B, -1).float()

    # Normalise to unit vectors so dot product equals cosine similarity.
    lat_norm = F.normalize(lat_flat, dim=1)
    noi_norm = F.normalize(noi_flat, dim=1)

    # Cosine similarity matrix: sim[i, j] = cos(latents[i], noise[j])
    sim = lat_norm @ noi_norm.T  # (B, B)

    # Greedy assignment: process latent rows in order 0..B-1.
    # For each row, pick the highest-similarity unused noise column.
    perm = torch.zeros(B, dtype=torch.long, device=latents.device)
    used = torch.zeros(B, dtype=torch.bool, device=latents.device)

    for i in range(B):
        available = sim[i].clone()
        available[used] = -2.0  # mask already-assigned columns (below any valid cosine value)
        j = available.argmax()
        perm[i] = j
        used[j] = True

    return noise[perm]

