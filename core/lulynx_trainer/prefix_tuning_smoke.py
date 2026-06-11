# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for prefix_tuning.py (Phase 9.4 / #113).

Verifies:
  1. PrefixTuningModule prepends learned vectors and extends attention mask
  2. PostfixTuningModule appends learned vectors and extends attention mask
  3. Initialisation strategies (normal / uniform / zeros) produce expected stats
  4. Modules are differentiable end-to-end
"""

from __future__ import annotations

import sys
import os
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.prefix_tuning",
    os.path.join(_HERE, "prefix_tuning.py"),
)
_pt = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.prefix_tuning"] = _pt
_spec.loader.exec_module(_pt)


def test_prefix_module_prepends_tokens():
    mod = _pt.PrefixTuningModule(hidden_size=16, prefix_length=4, init="normal")
    h = torch.randn(2, 8, 16)
    out_h, out_mask = mod(h, attention_mask=None)
    assert out_h.shape == (2, 12, 16), f"shape mismatch {out_h.shape}"
    assert out_mask is None
    print("PASS: PrefixTuningModule prepends tokens")


def test_prefix_module_extends_attention_mask():
    mod = _pt.PrefixTuningModule(hidden_size=16, prefix_length=4)
    h = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[1, 6:] = 0  # batch 1 has padding
    _, out_mask = mod(h, attention_mask=mask)
    assert out_mask.shape == (2, 12)
    # Prefix tokens always valid
    assert (out_mask[:, :4] == 1).all()
    # Padding preserved at the end
    assert (out_mask[1, 4 + 6:] == 0).all()
    print("PASS: PrefixTuningModule extends attention mask")


def test_postfix_module_appends_tokens():
    mod = _pt.PostfixTuningModule(hidden_size=16, postfix_length=3, init="zeros")
    h = torch.randn(2, 8, 16)
    out_h, _ = mod(h, attention_mask=None)
    assert out_h.shape == (2, 11, 16)
    # zeros init: tail tokens should be zero
    assert torch.allclose(out_h[:, 8:, :], torch.zeros(2, 3, 16))
    print("PASS: PostfixTuningModule appends zero-init tokens")


def test_postfix_module_extends_attention_mask():
    mod = _pt.PostfixTuningModule(hidden_size=16, postfix_length=3)
    h = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.long)
    _, out_mask = mod(h, attention_mask=mask)
    assert out_mask.shape == (2, 11)
    assert (out_mask[:, 8:] == 1).all()
    print("PASS: PostfixTuningModule extends attention mask")


def test_init_strategies_change_distribution():
    normal_mod = _pt.PrefixTuningModule(64, 32, init="normal")
    zeros_mod = _pt.PrefixTuningModule(64, 32, init="zeros")
    uniform_mod = _pt.PrefixTuningModule(64, 32, init="uniform")

    assert normal_mod.prefix_embedding.std().item() > 0.005
    assert torch.allclose(zeros_mod.prefix_embedding, torch.zeros_like(zeros_mod.prefix_embedding))
    # uniform: max should hover near 0.02
    assert uniform_mod.prefix_embedding.abs().max().item() <= 0.025
    print("PASS: init strategies produce expected distributions")


def test_prefix_module_is_differentiable():
    mod = _pt.PrefixTuningModule(hidden_size=8, prefix_length=2)
    h = torch.randn(1, 4, 8, requires_grad=True)
    out, _ = mod(h)
    out.sum().backward()
    assert mod.prefix_embedding.grad is not None
    assert mod.prefix_embedding.grad.abs().sum().item() > 0
    print("PASS: PrefixTuningModule supports backprop")


if __name__ == "__main__":
    test_prefix_module_prepends_tokens()
    test_prefix_module_extends_attention_mask()
    test_postfix_module_appends_tokens()
    test_postfix_module_extends_attention_mask()
    test_init_strategies_change_distribution()
    test_prefix_module_is_differentiable()
    print("\nAll prefix_tuning smoke tests passed!")
