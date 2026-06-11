# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for concept direction training."""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_cd = _import_module(
    "concept_direction",
    os.path.join(_HERE, "concept_direction.py"),
)
ConceptDirectionTrainer = _cd.ConceptDirectionTrainer
ConceptDirectionPair = _cd.ConceptDirectionPair

import torch
import torch.nn as nn


def test_parse_valid_pairs():
    json_str = '[{"positive":"a smiling person","negative":"a person"}]'
    pairs = ConceptDirectionTrainer.parse_pairs(json_str)
    assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
    assert isinstance(pairs[0], ConceptDirectionPair), "Expected ConceptDirectionPair instance"
    assert pairs[0].positive == "a smiling person", f"Unexpected positive: {pairs[0].positive}"
    assert pairs[0].negative == "a person", f"Unexpected negative: {pairs[0].negative}"
    print("PASS: parse_valid_pairs")


def test_parse_invalid_json():
    result = ConceptDirectionTrainer.parse_pairs("not valid json {{{{")
    assert isinstance(result, list), "Expected list return type"
    assert len(result) == 0, f"Expected empty list for invalid JSON, got {len(result)} items"
    print("PASS: parse_invalid_json")


def test_sample_pair_uniformity():
    pairs = [
        ConceptDirectionPair(positive="smiling", negative="neutral"),
        ConceptDirectionPair(positive="angry", negative="neutral"),
        ConceptDirectionPair(positive="surprised", negative="neutral"),
    ]
    trainer = ConceptDirectionTrainer(pairs=pairs)
    seen = set()
    for _ in range(100):
        p = trainer.sample_pair()
        seen.add(p.positive)
    assert len(seen) == 3, f"Expected all 3 pairs to appear in 100 samples, only saw: {seen}"
    print("PASS: sample_pair_uniformity")


def test_sample_timesteps_range():
    pairs = [ConceptDirectionPair(positive="a cat", negative="an animal")]
    trainer = ConceptDirectionTrainer(pairs=pairs)
    device = torch.device("cpu")
    ts = trainer.sample_timesteps(batch_size=100, num_train_timesteps=1000, device=device)
    assert ts.shape == (100,), f"Expected shape (100,), got {ts.shape}"
    assert int(ts.min()) >= 0, f"Timestep below 0: {int(ts.min())}"
    assert int(ts.max()) <= 999, f"Timestep above 999: {int(ts.max())}"
    print("PASS: sample_timesteps_range")


def test_direction_loss_finite():
    pairs = [ConceptDirectionPair(positive="a bright room", negative="a dark room")]
    trainer = ConceptDirectionTrainer(pairs=pairs, weight=1.0, guidance_scale=1.0)

    model = nn.Linear(4, 4, bias=False)

    x_pos = torch.randn(2, 4)
    x_neg = torch.randn(2, 4)

    with torch.no_grad():
        base_pos = model(x_pos)
        base_neg = model(x_neg)

    pred_pos = model(x_pos)
    pred_neg = model(x_neg)

    loss = trainer.compute_direction_loss(
        pred_positive=pred_pos,
        pred_negative=pred_neg,
        base_positive=base_pos,
        base_negative=base_neg,
    )

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print("PASS: direction_loss_finite")


if __name__ == "__main__":
    test_parse_valid_pairs()
    test_parse_invalid_json()
    test_sample_pair_uniformity()
    test_sample_timesteps_range()
    test_direction_loss_finite()
    print("\nAll concept direction smoke tests passed!")
