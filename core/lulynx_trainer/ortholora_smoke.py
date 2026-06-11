# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for ortholora.py (Phase 8.1 / #108)."""

from __future__ import annotations

import os
import sys
import importlib.util

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.ortholora",
    os.path.join(_HERE, "ortholora.py"),
)
_ol = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.ortholora"] = _ol
_spec.loader.exec_module(_ol)


def _is_orthonormal(matrix: torch.Tensor, atol: float = 1e-3) -> bool:
    """Check that rows are orthonormal: M @ M^T ≈ I_r."""
    rows = matrix.shape[0]
    eye = torch.eye(rows, dtype=matrix.dtype)
    gram = (matrix.float() @ matrix.float().t())
    return torch.allclose(gram, eye, atol=atol)


def test_gram_schmidt_produces_orthonormal_rows():
    torch.manual_seed(0)
    M = torch.randn(4, 16)
    O = _ol.gram_schmidt(M)
    assert _is_orthonormal(O), f"rows not orthonormal: {(O @ O.t())}"
    print("PASS: gram_schmidt produces orthonormal rows")


def test_gram_schmidt_handles_dependent_rows():
    M = torch.zeros(3, 8)
    M[0] = torch.ones(8)
    M[1] = torch.ones(8)  # duplicate
    M[2, 0] = 1.0
    O = _ol.gram_schmidt(M)
    # Result must not be NaN
    assert not torch.isnan(O).any()
    print("PASS: gram_schmidt handles linearly dependent rows without NaN")


def test_cayley_returns_orthogonal_matrix_for_square():
    torch.manual_seed(0)
    M = torch.randn(8, 8)
    Q = _ol.cayley_orthogonalise(M)
    eye = torch.eye(8)
    gram = (Q.float() @ Q.float().t())
    assert torch.allclose(gram, eye, atol=1e-3)
    print("PASS: cayley_orthogonalise produces orthogonal square matrix")


def test_cayley_passthrough_for_non_square():
    M = torch.randn(4, 16)
    out = _ol.cayley_orthogonalise(M)
    # Non-square: returned unchanged (caller should fall back)
    assert torch.allclose(out, M)
    print("PASS: cayley returns input unchanged for non-square matrix")


class _FakeLora(nn.Module):
    def __init__(self, rank=4, dim=16):
        super().__init__()
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)


class _FakeWrapper(nn.Module):
    def __init__(self, rank=4, dim=16):
        super().__init__()
        self.lora = _FakeLora(rank=rank, dim=dim)


class _FakeInjector:
    def __init__(self):
        self.injected_layers = {}


def test_projector_orthogonalises_lora_weights():
    torch.manual_seed(0)
    inj = _FakeInjector()
    inj.injected_layers["block_0"] = _FakeWrapper(rank=4, dim=16)
    inj.injected_layers["block_1"] = _FakeWrapper(rank=4, dim=16)

    proj = _ol.OrthoLoRAProjector(method="gram_schmidt", interval=1)
    n = proj.register_from_injector(inj)
    assert n == 2

    count = proj.step()
    # Two layers × (down + up) = 4 matrices
    assert count == 4

    for name, wrapper in inj.injected_layers.items():
        down = wrapper.lora.lora_down.weight.data  # [4, 16] — orthonormal rows
        assert _is_orthonormal(down), f"down for {name} not orthonormal"
    print("PASS: projector orthogonalises lora_down rows")


def test_projector_respects_interval():
    inj = _FakeInjector()
    inj.injected_layers["block_0"] = _FakeWrapper(rank=4, dim=16)

    proj = _ol.OrthoLoRAProjector(method="gram_schmidt", interval=3)
    proj.register_from_injector(inj)

    # Steps 1, 2 should not project; step 3 should
    assert proj.step() == 0
    assert proj.step() == 0
    assert proj.step() == 2  # down + up
    print("PASS: projector respects interval setting")


def test_projector_target_layers_filter():
    inj = _FakeInjector()
    inj.injected_layers["block_0"] = _FakeWrapper()
    inj.injected_layers["block_1"] = _FakeWrapper()
    inj.injected_layers["block_2"] = _FakeWrapper()

    proj = _ol.OrthoLoRAProjector(target_layers=["block_0", "block_2"])
    proj.register_from_injector(inj)
    # Only the two whitelisted layers actually get registered
    assert len(proj._layers) == 2
    count = proj.step()
    assert count == 4  # 2 layers × 2 matrices
    print("PASS: target_layers filter limits projection to whitelist")


def test_unknown_method_raises():
    try:
        _ol.OrthoLoRAProjector(method="garbage")
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("PASS: unknown method raises ValueError")


def test_reset_clears_layers_and_counter():
    inj = _FakeInjector()
    inj.injected_layers["x"] = _FakeWrapper()
    proj = _ol.OrthoLoRAProjector()
    proj.register_from_injector(inj)
    proj.step()
    proj.reset()
    assert proj._layers == []
    assert proj._step_counter == 0
    print("PASS: reset clears registered layers and counter")


if __name__ == "__main__":
    test_gram_schmidt_produces_orthonormal_rows()
    test_gram_schmidt_handles_dependent_rows()
    test_cayley_returns_orthogonal_matrix_for_square()
    test_cayley_passthrough_for_non_square()
    test_projector_orthogonalises_lora_weights()
    test_projector_respects_interval()
    test_projector_target_layers_filter()
    test_unknown_method_raises()
    test_reset_clears_layers_and_counter()
    print("\nAll OrthoLoRA smoke tests passed!")
