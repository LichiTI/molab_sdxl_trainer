# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke: EasyControl v2 two-stream *training-step consumption* contract.

The patch smoke (``easycontrol_v2_anima_patch_smoke``) proves the patch mechanism
in isolation.  This smoke proves the **consumption contract the trainer/loop now
rely on** (#184-A wiring): the per-step condition source resolution and the
default-off parity the loop guarantees.

Mirrors exactly what ``run_lulynx_forward_input_stage_handler`` does per step:

* **Default-off parity (no adapter).** With ``easycontrol_v2_enabled=False`` the
  trainer never builds/installs an adapter; the loop passes ``adapter=None`` and
  the forward is byte-identical to today. (Covered structurally: nothing touched.)
* **No-condition parity (adapter installed, no source).** When neither paired
  ``cond_latents`` nor a clean latent is available the handler calls
  ``clear_cond()`` and the patched forward is bitwise-original.
* **Derived-reference consumption (cache-first fallback).** With no sidecar
  ``cond_latents`` the handler derives a coarse reference by
  ``adaptive_avg_pool2d(clean_latent, (16,16))`` and ``set_cond``s it; this smoke
  reproduces that derivation and asserts the patched forward runs and a loss
  backprops gradient into the adapter (cond_proj + b_cond) -- i.e. the two-stream
  is really used in the training step. (This leaks the target -> a usable signal
  that proves wiring; it is NOT a control-quality claim.)
* **Production-shape source.** A paired ``[B,C,H,W]`` cond_latent also encodes and
  backprops, proving the sidecar path is plumbed.
* **Clear restores parity.** After ``clear_cond()`` the forward is bitwise-original
  again (the loop clears after every forward).

Run directly:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/easycontrol_v2_train_consumption_smoke.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch
import torch.nn.functional as F

from core.lulynx_trainer.anima_native_dit import AnimaNativeExecutableSubset
from core.lulynx_trainer.easycontrol_v2_adapter import (
    EasyControlV2Adapter,
    EasyControlV2AdapterConfig,
)
from core.lulynx_trainer.easycontrol_v2_anima_patch import (
    install_easycontrol_v2_anima_executable_subset_patch,
)

HIDDEN = 8
HEAD_DIM = 4
COND_CHANNELS = 16


def _subset_shapes(n_blocks: int) -> "dict[str, tuple[int, ...]]":
    shapes: dict[str, tuple[int, ...]] = {
        "net.x_embedder.proj.1.weight": (HIDDEN, 64),
        "net.t_embedding_norm.weight": (HIDDEN,),
        "net.t_embedder.1.linear_1.weight": (HIDDEN, HIDDEN),
        "net.t_embedder.1.linear_2.weight": (3 * HIDDEN, HIDDEN),
        "net.final_layer.linear.weight": (64, HIDDEN),
        "net.final_layer.adaln_modulation.1.weight": (HIDDEN, HIDDEN),
        "net.final_layer.adaln_modulation.2.weight": (2 * HIDDEN, HIDDEN),
    }
    for index in range(n_blocks):
        prefix = f"net.blocks.{index}"
        for attn in ("self_attn", "cross_attn"):
            shapes[f"{prefix}.{attn}.q_proj.weight"] = (HIDDEN, HIDDEN)
            shapes[f"{prefix}.{attn}.k_proj.weight"] = (HIDDEN, HIDDEN)
            shapes[f"{prefix}.{attn}.v_proj.weight"] = (HIDDEN, HIDDEN)
            shapes[f"{prefix}.{attn}.output_proj.weight"] = (HIDDEN, HIDDEN)
            shapes[f"{prefix}.{attn}.q_norm.weight"] = (HEAD_DIM,)
            shapes[f"{prefix}.{attn}.k_norm.weight"] = (HEAD_DIM,)
        shapes[f"{prefix}.mlp.layer1.weight"] = (2 * HIDDEN, HIDDEN)
        shapes[f"{prefix}.mlp.layer2.weight"] = (HIDDEN, 2 * HIDDEN)
        for branch in ("self_attn", "cross_attn", "mlp"):
            shapes[f"{prefix}.adaln_modulation_{branch}.1.weight"] = (HIDDEN, HIDDEN)
            shapes[f"{prefix}.adaln_modulation_{branch}.2.weight"] = (3 * HIDDEN, HIDDEN)
    return shapes


def _build_subset(n_blocks: int) -> AnimaNativeExecutableSubset:
    return AnimaNativeExecutableSubset(
        _subset_shapes(n_blocks),
        block_indices=tuple(range(n_blocks)),
        device="cpu",
        dtype=torch.float32,
    )


def _adapter(n_blocks: int, *, b_cond: float, init_zero_out: bool) -> EasyControlV2Adapter:
    return EasyControlV2Adapter(
        EasyControlV2AdapterConfig(
            hidden_size=HIDDEN,
            cond_channels=COND_CHANNELS,
            cond_lora_rank=2,
            num_blocks=n_blocks,
            b_cond_init=b_cond,
            init_zero_out=init_zero_out,
        )
    )


def _inputs():
    sample = torch.randn(1, 15, 4, 4)  # 15 latent + 1 mask channel -> 64 patch features
    timestep = torch.ones(1)
    text = torch.randn(1, 6, HIDDEN)
    return sample, timestep, text


def _derive_reference(clean_latents: torch.Tensor) -> torch.Tensor:
    """Exactly the handler's cache-first fallback: pool clean latent to 16x16."""
    return F.adaptive_avg_pool2d(clean_latents, (16, 16))


def test_no_condition_parity() -> None:
    torch.manual_seed(0)
    model = _build_subset(2)
    adapter = _adapter(2, b_cond=-4.0, init_zero_out=True)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample
    handle = install_easycontrol_v2_anima_executable_subset_patch(model, adapter)
    try:
        adapter.clear_cond()  # loop's "no source this step" branch
        with torch.no_grad():
            out = model(sample, ts, text).sample
        assert torch.equal(baseline, out), "installed-but-uncondiitoned patch broke bitwise parity"
    finally:
        handle.remove()
    print("PASS: installed adapter with no condition is bitwise parity (loop no-source branch)")


def test_derived_reference_consumption_backprops() -> None:
    torch.manual_seed(1)
    model = _build_subset(2)
    # Gentle-active start mirrors the trainer's enabled default (b_cond_init=-4).
    adapter = _adapter(2, b_cond=-4.0, init_zero_out=True)
    sample, ts, text = _inputs()

    # The cache-first batch has a clean target latent but NO sidecar cond_latents,
    # so the handler derives a coarse reference. cond_channels==latent channels==16.
    clean_latents = torch.randn(1, COND_CHANNELS, 4, 4)
    ref = _derive_reference(clean_latents)
    assert ref.shape == (1, COND_CHANNELS, 16, 16), ref.shape

    handle = install_easycontrol_v2_anima_executable_subset_patch(model, adapter)
    try:
        adapter.set_cond(ref)
        assert adapter.current_cond_tokens is not None
        # 16x16 grid -> 256 condition tokens, hidden=HIDDEN.
        assert tuple(adapter.current_cond_tokens.shape) == (1, 256, HIDDEN), adapter.current_cond_tokens.shape
        out = model(sample, ts, text).sample
        out.square().mean().backward()
    finally:
        handle.remove()

    g_proj = adapter.cond_proj.weight.grad
    g_bcond = adapter.blocks[0].b_cond.grad
    assert g_proj is not None and float(g_proj.abs().sum()) > 0.0, "cond_proj got no gradient"
    assert g_bcond is not None, "b_cond got no gradient (gate not on the graph)"
    print("PASS: derived-reference condition is consumed and backprops into the adapter")


def test_production_shape_source_encodes() -> None:
    torch.manual_seed(2)
    model = _build_subset(1)
    adapter = _adapter(1, b_cond=4.0, init_zero_out=False)
    sample, ts, text = _inputs()
    # Production sidecar path: paired [B,C,H,W] cond_latent fed straight in.
    cond_latents = torch.randn(1, COND_CHANNELS, 3, 3)
    handle = install_easycontrol_v2_anima_executable_subset_patch(model, adapter)
    try:
        adapter.set_cond(cond_latents)
        assert tuple(adapter.current_cond_tokens.shape) == (1, 9, HIDDEN), adapter.current_cond_tokens.shape
        out = model(sample, ts, text).sample
        out.square().mean().backward()
    finally:
        handle.remove()
    assert adapter.cond_proj.weight.grad is not None and float(adapter.cond_proj.weight.grad.abs().sum()) > 0.0
    print("PASS: production-shape [B,C,H,W] cond source encodes and backprops")


def test_clear_restores_parity() -> None:
    torch.manual_seed(3)
    model = _build_subset(2)
    adapter = _adapter(2, b_cond=4.0, init_zero_out=False)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample
    handle = install_easycontrol_v2_anima_executable_subset_patch(model, adapter)
    try:
        adapter.set_cond(_derive_reference(torch.randn(1, COND_CHANNELS, 4, 4)))
        with torch.no_grad():
            conditioned = model(sample, ts, text).sample
        assert not torch.equal(baseline, conditioned), "open condition should change the output"
        adapter.clear_cond()  # loop clears after every forward
        with torch.no_grad():
            restored = model(sample, ts, text).sample
        assert torch.equal(baseline, restored), "clear_cond did not restore bitwise parity"
    finally:
        handle.remove()
    print("PASS: clear_cond after a conditioned step restores bitwise parity (loop post-forward clear)")


def main() -> int:
    test_no_condition_parity()
    test_derived_reference_consumption_backprops()
    test_production_shape_source_encodes()
    test_clear_restores_parity()
    print("\n[easycontrol_v2_train_consumption_smoke] 4/4 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
