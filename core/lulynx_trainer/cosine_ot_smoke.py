# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for cosine-similarity optimal transport noise matching."""

from __future__ import annotations
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_ot = _import_module("cosine_ot", os.path.join(_HERE, "cosine_ot.py"))
minibatch_ot_cosine = _ot.minibatch_ot_cosine

import torch


def _make_pair(B, C=4, H=8, W=8):
    latents = torch.randn(B, C, H, W)
    noise = torch.randn(B, C, H, W)
    return latents, noise


def test_valid_permutation():
    B = 4
    latents, noise = _make_pair(B)
    result = minibatch_ot_cosine(latents, noise)
    # Result must be composed entirely of rows from the original noise batch
    assert result.shape[0] == B
    # Each output row must match exactly one input row
    matched = 0
    for i in range(B):
        for j in range(B):
            if torch.allclose(result[i], noise[j], atol=1e-5):
                matched += 1
                break
    assert matched == B, (
        f"Output must be a valid permutation of noise rows; matched {matched}/{B}"
    )


def test_batch_one_noop():
    latents, noise = _make_pair(1)
    result = minibatch_ot_cosine(latents, noise)
    assert torch.allclose(result, noise, atol=1e-5), (
        "With B=1, minibatch_ot_cosine should return noise unchanged"
    )


def test_improves_similarity():
    torch.manual_seed(42)
    B = 8
    latents, noise = _make_pair(B, C=4, H=16, W=16)
    result = minibatch_ot_cosine(latents, noise)

    def batch_cosine_sim(a, b):
        a_flat = a.view(a.shape[0], -1)
        b_flat = b.view(b.shape[0], -1)
        a_norm = a_flat / (a_flat.norm(dim=1, keepdim=True) + 1e-8)
        b_norm = b_flat / (b_flat.norm(dim=1, keepdim=True) + 1e-8)
        return (a_norm * b_norm).sum(dim=1).mean().item()

    ot_sim = batch_cosine_sim(latents, result)
    # Random baseline: average over several shuffles
    random_sims = []
    for _ in range(10):
        perm = torch.randperm(B)
        random_sims.append(batch_cosine_sim(latents, noise[perm]))
    random_sim = sum(random_sims) / len(random_sims)

    assert ot_sim >= random_sim - 1e-3, (
        f"OT pairing (cosine sim={ot_sim:.4f}) should be >= random pairing ({random_sim:.4f})"
    )


def test_output_shape():
    B = 4
    latents, noise = _make_pair(B, C=8, H=16, W=16)
    result = minibatch_ot_cosine(latents, noise)
    assert result.shape == noise.shape, (
        f"Output shape {result.shape} must match input noise shape {noise.shape}"
    )


def test_no_grad():
    latents, noise = _make_pair(4)
    result = minibatch_ot_cosine(latents, noise)
    assert not result.requires_grad, (
        "Output of minibatch_ot_cosine should not require gradients"
    )


if __name__ == "__main__":
    test_valid_permutation()
    test_batch_one_noop()
    test_improves_similarity()
    test_output_shape()
    test_no_grad()
    print("cosine_ot_smoke: all tests passed")
