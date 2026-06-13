"""Regression smoke locking the native-Anima DiT block MLP activation = GELU.

Root cause of the #133/#164 render-gray bug (fixed 2026-06-09): the executable
subset's per-block MLP used ``F.silu`` while the reference ``GPT2FeedForward``
uses ``nn.GELU()``. The ~6% per-block error (MLP cosine 0.9397) compounded over
28 blocks x 20+ sampling steps and contracted the latent to a constant -> flat
gray. The fix is one line in ``anima_native_dit.py`` (``F.silu`` -> ``F.gelu``).

This smoke locks that fix WITHOUT depending on the AGPL reference tree: it loads
the real Anima checkpoint into our own executable subset, takes a real block's
MLP, and asserts its forward equals ``layer2(gelu(layer1(x)))`` and is *not*
equal to the ``silu`` variant. The bit-exact ref-parity validator
(``.runs/anima_133_render/ref_compare.py``, backbone cos = 1.0000) is dev-only
and read-only against ``/ref`` -- it is not committed as a test to avoid a hard
reference dependency.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/anima_native_mlp_gelu_regression_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.anima_native_dit import load_anima_native_executable_subset


def _resolve_checkpoint() -> Path | None:
    """Find a local Anima DiT checkpoint; skip-clean when none is present."""
    base = Path(__file__).resolve().parents[3] / "models" / "anima" / "diffusion_models"
    for name in (
        "anima-base-v1.0.safetensors",
        "anima-preview2.safetensors",
    ):
        candidate = base / name
        if candidate.exists():
            return candidate
    matches = sorted(base.glob("*.safetensors")) if base.exists() else []
    return matches[0] if matches else None


def main() -> int:
    checkpoint = _resolve_checkpoint()
    if checkpoint is None:
        print("SKIP: no local Anima checkpoint under models/anima/diffusion_models")
        return 0

    # One block is enough -- the MLP activation is identical across all blocks.
    model, _report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=(0,),
        device="cpu",
        dtype=torch.float32,
    )
    mlp = model.net.blocks[0].mlp
    assert hasattr(mlp, "layer1") and hasattr(mlp, "layer2"), "unexpected MLP structure"

    torch.manual_seed(0)
    in_features = mlp.layer1.in_features
    x = torch.randn(1, 8, in_features)

    with torch.no_grad():
        out = mlp(x)
        gelu_ref = mlp.layer2(F.gelu(mlp.layer1(x)))
        silu_ref = mlp.layer2(F.silu(mlp.layer1(x)))

    # The block MLP must match the GELU reference bit-for-bit...
    assert torch.allclose(out, gelu_ref, atol=1e-6, rtol=0), (
        "native-Anima block MLP no longer matches GELU -- the SiLU render-gray "
        "regression may have returned (anima_native_dit.py _Mlp.forward)"
    )
    # ...and must be clearly distinguishable from the buggy SiLU variant
    # (guards against an accidental silent revert).
    silu_gap = (out - silu_ref).abs().max().item()
    assert silu_gap > 1e-4, (
        f"block MLP output is indistinguishable from SiLU (max gap {silu_gap:.2e}); "
        "the activation may have regressed to F.silu"
    )

    print(
        "Anima native MLP GELU regression smoke passed: "
        f"checkpoint={checkpoint.name}, in_features={in_features}, "
        f"gelu_match=True, silu_gap={silu_gap:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
