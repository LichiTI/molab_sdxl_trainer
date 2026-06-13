# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test: EasyControl v2 two-stream patch on the REAL Anima executable subset.

Builds a synthetic-shape ``AnimaNativeExecutableSubset`` (CPU, no real weights)
and drives a full model forward through the patched ``_Block.forward`` to prove:

* **No-condition bitwise parity** -- an installed patch with no active condition
  tokens reproduces the unpatched output exactly, and ``handle.remove()``
  restores the original forward.
* **Zero-init near-identity** -- a freshly built (zero-init, b_cond=-10) adapter
  with a condition set perturbs the output only negligibly.
* **b_cond gates condition mass** -- very negative b_cond -> ~no drift; positive
  b_cond + non-zero LoRA -> large drift (the condition stream is really used).
* **Condition-path gradients flow** -- cond / cond_proj / b_cond all receive
  gradient through the patched two-stream attention.
* **Readiness flips** -- ``patch_supported`` is now True and the guard passes.

Run directly:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/easycontrol_v2_anima_patch_smoke.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch

from core.lulynx_trainer.anima_native_dit import AnimaNativeExecutableSubset
from core.lulynx_trainer.easycontrol_v2_adapter import (
    EasyControlV2Adapter,
    EasyControlV2AdapterConfig,
    easycontrol_v2_anima_executable_subset_readiness,
    require_easycontrol_v2_anima_executable_subset_patch_ready,
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


def _adapter(n_blocks: int, *, b_cond: float = -10.0, init_zero_out: bool = True) -> EasyControlV2Adapter:
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


def _rel_drift(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (b.norm() + 1e-8))


def test_no_condition_is_bitwise_parity() -> None:
    torch.manual_seed(0)
    model = _build_subset(2)
    adapter = _adapter(2)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample

    handle = install_easycontrol_v2_anima_executable_subset_patch(model, adapter)
    try:
        assert handle.block_count == 2, handle.block_count
        with torch.no_grad():
            patched = model(sample, ts, text).sample  # adapter has no cond -> original path
        assert torch.equal(baseline, patched), "no-condition patch broke bitwise parity"
    finally:
        handle.remove()
    assert not handle.active
    with torch.no_grad():
        restored = model(sample, ts, text).sample
    assert torch.equal(baseline, restored), "handle.remove did not restore the original forward"
    print("PASS: no-condition patch is bitwise parity; remove() restores original forward")


def test_zero_init_condition_is_near_identity() -> None:
    torch.manual_seed(1)
    model = _build_subset(2)
    adapter = _adapter(2, b_cond=-10.0, init_zero_out=True)
    sample, ts, text = _inputs()
    with torch.no_grad():
        baseline = model(sample, ts, text).sample

    handle = install_easycontrol_v2_anima_executable_subset_patch(model, adapter)
    try:
        adapter.set_cond(torch.randn(1, COND_CHANNELS, 4, 4))
        with torch.no_grad():
            conditioned = model(sample, ts, text).sample
    finally:
        handle.remove()
    drift = _rel_drift(conditioned, baseline)
    assert drift < 0.05, f"zero-init condition drifted too far from identity: {drift}"
    assert drift > 0.0, "condition path had literally zero effect (b_cond gate not wired)"
    print(f"PASS: zero-init condition is near-identity (relative drift={drift:.2e})")


def test_b_cond_gates_condition_mass() -> None:
    torch.manual_seed(2)
    sample, ts, text = _inputs()

    # Very negative b_cond -> condition columns masked -> ~no drift.
    model = _build_subset(1)
    closed = _adapter(1, b_cond=-30.0, init_zero_out=True)
    with torch.no_grad():
        baseline = model(sample, ts, text).sample
    h = install_easycontrol_v2_anima_executable_subset_patch(model, closed)
    try:
        closed.set_cond(torch.randn(1, COND_CHANNELS, 4, 4))
        with torch.no_grad():
            out_closed = model(sample, ts, text).sample
    finally:
        h.remove()
    assert torch.allclose(out_closed, baseline, atol=1e-4), "very negative b_cond should mask the condition"

    # Positive b_cond + non-zero LoRA -> condition is really used -> large drift.
    model2 = _build_subset(1)
    open_gate = _adapter(1, b_cond=15.0, init_zero_out=False)
    with torch.no_grad():
        baseline2 = model2(sample, ts, text).sample
    h2 = install_easycontrol_v2_anima_executable_subset_patch(model2, open_gate)
    try:
        open_gate.set_cond(torch.randn(1, COND_CHANNELS, 4, 4))
        with torch.no_grad():
            out_open = model2(sample, ts, text).sample
    finally:
        h2.remove()
    drift = _rel_drift(out_open, baseline2)
    assert drift > 1e-2, f"open b_cond should visibly use the condition, drift={drift}"
    print("PASS: b_cond gates condition mass (closed~identity, open->visible drift)")


def test_condition_path_backpropagates() -> None:
    torch.manual_seed(3)
    model = _build_subset(2)
    adapter = _adapter(2, b_cond=10.0, init_zero_out=False)
    sample, ts, text = _inputs()
    cond = torch.randn(1, COND_CHANNELS, 4, 4, requires_grad=True)

    handle = install_easycontrol_v2_anima_executable_subset_patch(model, adapter)
    try:
        adapter.set_cond(cond)
        output = model(sample, ts, text).sample
        output.square().mean().backward()
    finally:
        handle.remove()

    assert cond.grad is not None and float(cond.grad.abs().sum()) > 0.0, "no gradient to condition input"
    assert adapter.cond_proj.weight.grad is not None and float(adapter.cond_proj.weight.grad.abs().sum()) > 0.0
    assert adapter.blocks[0].b_cond.grad is not None, "b_cond received no gradient"
    print("PASS: condition path backpropagates (cond / cond_proj / b_cond all get gradient)")


def test_readiness_now_supports_patch() -> None:
    model = _build_subset(2)
    report = easycontrol_v2_anima_executable_subset_readiness(model)
    assert report["ready"] is True, report
    assert report["real_executable_subset"] is True, report
    assert report["patch_supported"] is True, "patch_supported did not flip after implementing the patch"
    assert report["training_step_consumption"] is True, "training-loop consumption is now wired (opt-in via easycontrol_v2_enabled)"
    assert report["blocked_reason"] is None, report
    # Guard must no longer raise now that the patch exists.
    ok = require_easycontrol_v2_anima_executable_subset_patch_ready(model)
    assert ok["patch_supported"] is True
    print("PASS: readiness reports patch_supported=True and the guard passes (consumption wired, opt-in)")


def main() -> int:
    test_no_condition_is_bitwise_parity()
    test_zero_init_condition_is_near_identity()
    test_b_cond_gates_condition_mass()
    test_condition_path_backpropagates()
    test_readiness_now_supports_patch()
    print("\n[easycontrol_v2_anima_patch_smoke] 5/5 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
