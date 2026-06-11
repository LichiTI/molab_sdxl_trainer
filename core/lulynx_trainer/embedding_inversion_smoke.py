# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for embedding_inversion.py (Phase 8.8 / #116)."""

from __future__ import annotations

import os
import sys
import importlib.util
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.embedding_inversion",
    os.path.join(_HERE, "embedding_inversion.py"),
)
_ti = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.embedding_inversion"] = _ti
_spec.loader.exec_module(_ti)


class _StubTokenizer:
    """Minimal tokenizer matching the HF interface used by the handler."""

    def __init__(self):
        self._vocab = {f"tok_{i}": i for i in range(100)}

    def __len__(self):
        return len(self._vocab)

    def add_tokens(self, tokens):
        added = 0
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = len(self._vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        if isinstance(token, str):
            return self._vocab.get(token)
        return [self._vocab.get(t) for t in token]


class _StubTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 100, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))


def test_add_single_placeholder_token():
    tok = _StubTokenizer()
    enc = _StubTextEncoder()
    handler = _ti.TextualInversionHandler(enc, tok)
    out = handler.add_placeholder_tokens(["<concept>"], num_vectors=1)
    assert "<concept>" in out
    assert len(out["<concept>"]) == 1
    assert handler._embedding_table().num_embeddings == 101
    print("PASS: single placeholder token added and embedding resized")


def test_add_multi_vector_placeholder():
    tok = _StubTokenizer()
    enc = _StubTextEncoder()
    handler = _ti.TextualInversionHandler(enc, tok)
    out = handler.add_placeholder_tokens(["<concept>"], num_vectors=4)
    ids = out["<concept>"]
    assert len(ids) == 4
    assert all(i is not None for i in ids)
    assert handler._embedding_table().num_embeddings == 104
    print("PASS: multi-vector placeholder allocates 4 slots")


def test_freeze_only_placeholder_rows():
    tok = _StubTokenizer()
    enc = _StubTextEncoder()
    handler = _ti.TextualInversionHandler(enc, tok)
    handler.add_placeholder_tokens(["<a>"], num_vectors=2)
    handler.freeze_all_but_placeholders()

    emb = handler._embedding_table()
    # Only the embedding weight should be trainable
    assert emb.weight.requires_grad
    other = [p for n, p in enc.named_parameters() if "embed.weight" not in n]
    assert all(not p.requires_grad for p in other)
    print("PASS: freeze_all_but_placeholders freezes everything except embedding")


def test_grad_hook_zeros_non_placeholder_rows():
    tok = _StubTokenizer()
    enc = _StubTextEncoder(vocab_size=20, dim=8)
    handler = _ti.TextualInversionHandler(enc, tok)
    ids = handler.add_placeholder_tokens(["<concept>"], num_vectors=2)["<concept>"]
    handler.freeze_all_but_placeholders()

    emb = handler._embedding_table()
    # Backward through ALL rows
    out = emb.weight.sum()
    out.backward()
    grad = emb.weight.grad
    assert grad is not None

    # Placeholder rows must have non-zero gradients
    for tid in ids:
        assert grad[tid].abs().sum().item() > 0

    # Non-placeholder rows must have zero gradients
    placeholder_set = set(ids)
    for r in range(grad.shape[0]):
        if r not in placeholder_set:
            assert torch.allclose(grad[r], torch.zeros_like(grad[r])), f"row {r} got non-zero grad"
    print("PASS: gradient hook zeros non-placeholder rows")


def test_save_and_load_round_trip():
    tok = _StubTokenizer()
    enc = _StubTextEncoder(vocab_size=20, dim=8)
    handler = _ti.TextualInversionHandler(enc, tok)
    ids = handler.add_placeholder_tokens(["<concept>"], num_vectors=2)["<concept>"]

    # Set distinctive values
    emb = handler._embedding_table()
    with torch.no_grad():
        emb.weight.data[ids[0]] = torch.full((8,), 7.0)
        emb.weight.data[ids[1]] = torch.full((8,), 3.0)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "ti.pt"
        handler.save(str(out_path))

        # Reset embedding to zeros
        with torch.no_grad():
            emb.weight.data[ids[0]].zero_()
            emb.weight.data[ids[1]].zero_()

        handler.load(str(out_path))

        assert torch.allclose(emb.weight.data[ids[0]], torch.full((8,), 7.0))
        assert torch.allclose(emb.weight.data[ids[1]], torch.full((8,), 3.0))
    print("PASS: save -> reset -> load round-trip preserves learned rows")


def test_init_strategy_zeros():
    tok = _StubTokenizer()
    enc = _StubTextEncoder()
    handler = _ti.TextualInversionHandler(enc, tok, init_strategy="zeros")
    ids = handler.add_placeholder_tokens(["<a>"], num_vectors=1)["<a>"]

    emb = handler._embedding_table()
    assert torch.allclose(emb.weight.data[ids[0]], torch.zeros_like(emb.weight.data[ids[0]]))
    print("PASS: init_strategy='zeros' fills placeholder rows with zeros")


def test_init_strategy_copy_token():
    tok = _StubTokenizer()
    enc = _StubTextEncoder()
    handler = _ti.TextualInversionHandler(enc, tok, init_strategy="copy_token")
    # Pre-set a known row to copy from
    src_id = tok.convert_tokens_to_ids("tok_5")
    emb = handler._embedding_table()
    with torch.no_grad():
        emb.weight.data[src_id] = torch.full((emb.embedding_dim,), 0.42)

    ids = handler.add_placeholder_tokens(["<a>"], num_vectors=2, init_token="tok_5")["<a>"]
    for tid in ids:
        assert torch.allclose(emb.weight.data[tid], emb.weight.data[src_id])
    print("PASS: init_strategy='copy_token' copies from source token")


def test_export_learned_rows_returns_per_token_tensors():
    tok = _StubTokenizer()
    enc = _StubTextEncoder(vocab_size=20, dim=8)
    handler = _ti.TextualInversionHandler(enc, tok)
    handler.add_placeholder_tokens(["<a>", "<b>"], num_vectors=3)
    rows = handler.export_learned_rows()
    assert set(rows.keys()) == {"<a>", "<b>"}
    assert rows["<a>"].shape == (3, 8)
    assert rows["<b>"].shape == (3, 8)
    print("PASS: export_learned_rows returns one tensor per base token")


if __name__ == "__main__":
    test_add_single_placeholder_token()
    test_add_multi_vector_placeholder()
    test_freeze_only_placeholder_rows()
    test_grad_hook_zeros_non_placeholder_rows()
    test_save_and_load_round_trip()
    test_init_strategy_zeros()
    test_init_strategy_copy_token()
    test_export_learned_rows_returns_per_token_tensors()
    print("\nAll embedding_inversion smoke tests passed!")
