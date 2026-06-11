# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""
2D Perlin noise generation for spatially-correlated noise offsets in diffusion training.

Warehouse implementation — no AGPL code.
"""

import math
import torch


def generate_perlin_2d(
    shape: tuple,
    scale: float = 4.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a batch of 2D Perlin noise fields.

    Args:
        shape: (B, C, H, W) — batch, channels, height, width.
        scale: Controls the frequency of features; larger values produce
               broader, smoother features.
        device: Target device for the output tensor.
        dtype: Floating-point dtype for the output tensor.

    Returns:
        Tensor of shape (B, C, H, W) with values roughly in [-1, 1].
    """
    B, C, H, W = shape

    # Number of gradient-vector grid cells needed to cover the image.
    gh = int(math.ceil(H / scale)) + 1
    gw = int(math.ceil(W / scale)) + 1

    # Random gradient angles at every grid vertex — one set per (batch, channel).
    angles = torch.rand(B, C, gh, gw, device=device, dtype=dtype) * (2.0 * math.pi)
    grad_x = torch.cos(angles)  # (B, C, gh, gw)
    grad_y = torch.sin(angles)

    # Pixel positions in grid-cell space.
    y_coords = torch.arange(H, device=device, dtype=dtype) / scale  # (H,)
    x_coords = torch.arange(W, device=device, dtype=dtype) / scale  # (W,)

    # Integer grid-cell indices for the top-left corner of each pixel's cell.
    y0 = y_coords.long().clamp(0, gh - 2)  # (H,)
    x0 = x_coords.long().clamp(0, gw - 2)  # (W,)
    y1 = y0 + 1
    x1 = x0 + 1

    # Fractional positions inside the cell, shape (H,) / (W,).
    fy = (y_coords - y0.float())  # in [0, 1]
    fx = (x_coords - x0.float())

    # Quintic smoothstep: 6t^5 - 15t^4 + 10t^3  (Perlin's improved formula).
    wy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)  # (H,)
    wx = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)  # (W,)

    # Build full 2-D index arrays for the four corners.
    # y0_2d/x0_2d shape: (H, W) — every (row, col) combination.
    y0_2d = y0.unsqueeze(1).expand(H, W)   # (H, W)
    y1_2d = y1.unsqueeze(1).expand(H, W)
    x0_2d = x0.unsqueeze(0).expand(H, W)   # (H, W)
    x1_2d = x1.unsqueeze(0).expand(H, W)

    # Flatten spatial dims for a single gather call per gradient component.
    # grad_x: (B, C, gh, gw) -> gather at flat (H*W) spatial positions.
    def _gather_2d(g: torch.Tensor, yi: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Gather gradient values at integer grid positions (yi, xi).

        Args:
            g:  (B, C, gh, gw)
            yi: (H, W)  long
            xi: (H, W)  long

        Returns:
            (B, C, H, W)
        """
        flat_idx = (yi * gw + xi).view(-1)  # (H*W,)
        g_flat = g.view(B, C, -1)            # (B, C, gh*gw)
        out = g_flat[:, :, flat_idx]          # (B, C, H*W)
        return out.view(B, C, H, W)

    gx00 = _gather_2d(grad_x, y0_2d, x0_2d)  # (B, C, H, W)
    gy00 = _gather_2d(grad_y, y0_2d, x0_2d)
    gx01 = _gather_2d(grad_x, y0_2d, x1_2d)
    gy01 = _gather_2d(grad_y, y0_2d, x1_2d)
    gx10 = _gather_2d(grad_x, y1_2d, x0_2d)
    gy10 = _gather_2d(grad_y, y1_2d, x0_2d)
    gx11 = _gather_2d(grad_x, y1_2d, x1_2d)
    gy11 = _gather_2d(grad_y, y1_2d, x1_2d)

    # Offset vectors from each corner to the pixel (broadcast over B, C).
    # fy/fx are (H,)/(W,) — reshape for (B, C, H, W) arithmetic.
    fy_ = fy.view(1, 1, H, 1)        # fractional y offset from top edge
    fx_ = fx.view(1, 1, 1, W)        # fractional x offset from left edge
    fy1 = (fy - 1.0).view(1, 1, H, 1)  # y offset from bottom edge
    fx1 = (fx - 1.0).view(1, 1, 1, W)  # x offset from right edge

    # Dot products: gradient · offset vector at each corner.
    n00 = gx00 * fx_  + gy00 * fy_   # top-left
    n01 = gx01 * fx1  + gy01 * fy_   # top-right
    n10 = gx10 * fx_  + gy10 * fy1   # bottom-left
    n11 = gx11 * fx1  + gy11 * fy1   # bottom-right

    # Bilinear interpolation with smoothstep weights.
    wx_ = wx.view(1, 1, 1, W)
    wy_ = wy.view(1, 1, H, 1)

    nx0 = n00 + wx_ * (n01 - n00)  # interpolate along x at top row
    nx1 = n10 + wx_ * (n11 - n10)  # interpolate along x at bottom row
    result = nx0 + wy_ * (nx1 - nx0)  # interpolate along y

    return result


def apply_perlin_noise_offset(
    noise: torch.Tensor,
    strength: float = 0.1,
    scale: float = 4.0,
) -> torch.Tensor:
    """Add a spatially-correlated Perlin noise offset to an existing noise tensor.

    Args:
        noise:    (B, C, H, W) noise tensor — the base noise to modify.
        strength: Magnitude of the Perlin offset.  Set to 0 to pass through unchanged.
        scale:    Perlin grid scale forwarded to :func:`generate_perlin_2d`.

    Returns:
        Noise tensor of the same shape and dtype as the input, with the Perlin
        offset added.
    """
    if strength <= 0.0:
        return noise

    perlin = generate_perlin_2d(
        noise.shape,
        scale=scale,
        device=noise.device,
        dtype=noise.dtype,
    )
    return noise + strength * perlin


# ---------------------------------------------------------------------------
# Self-test (run with  python -m backend.core.lulynx_trainer.perlin_noise)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    def _run_tests() -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running Perlin noise tests on {device} …")

        # --- shape preservation ---
        for shape in [(1, 1, 64, 64), (2, 4, 32, 32), (1, 3, 128, 64)]:
            out = generate_perlin_2d(shape, scale=8.0, device=device)
            assert out.shape == torch.Size(shape), f"shape mismatch: {out.shape} vs {shape}"
        print("  [PASS] shape preservation")

        # --- value range (should be close to [-1, 1]) ---
        out = generate_perlin_2d((4, 4, 64, 64), scale=8.0, device=device)
        mn, mx = out.min().item(), out.max().item()
        assert -2.0 < mn and mx < 2.0, f"values out of expected range: [{mn}, {mx}]"
        print(f"  [PASS] value range  min={mn:.4f}  max={mx:.4f}")

        # --- spatial correlation: neighbouring pixels should be similar ---
        out = generate_perlin_2d((1, 1, 64, 64), scale=16.0, device=device)
        diff = (out[:, :, 1:, :] - out[:, :, :-1, :]).abs().mean().item()
        assert diff < 0.15, f"output not smooth enough (mean pixel diff {diff:.4f})"
        print(f"  [PASS] spatial smoothness  mean_diff={diff:.4f}")

        # --- apply_perlin_noise_offset ---
        base = torch.randn(2, 4, 32, 32, device=device)
        modified = apply_perlin_noise_offset(base, strength=0.5, scale=4.0)
        assert modified.shape == base.shape
        assert not torch.equal(modified, base)
        print("  [PASS] apply_perlin_noise_offset modifies input")

        # --- strength=0 is a no-op ---
        noop = apply_perlin_noise_offset(base, strength=0.0)
        assert torch.equal(noop, base)
        print("  [PASS] strength=0 is a no-op")

        # --- dtype forwarding ---
        for dt in (torch.float16, torch.float32):
            out = generate_perlin_2d((1, 1, 32, 32), dtype=dt, device=device)
            assert out.dtype == dt, f"dtype mismatch: {out.dtype}"
        print("  [PASS] dtype forwarding")

        print("All tests passed.")

    _run_tests()
    sys.exit(0)

