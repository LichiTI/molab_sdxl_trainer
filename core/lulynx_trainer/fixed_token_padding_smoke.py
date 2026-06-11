# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for fixed_token_padding.py (#101)."""

from __future__ import annotations

import os
import sys
import importlib.util

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.fixed_token_padding",
    os.path.join(_HERE, "fixed_token_padding.py"),
)
_ftp = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.fixed_token_padding"] = _ftp
_spec.loader.exec_module(_ftp)


class _StubTokenizer:
    """Mock tokenizer matching the HF-style call interface."""

    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, captions, *, padding=None, max_length=None, truncation=None, return_tensors="pt"):
        # Encode each caption as len(caption) ids in [3, ..., 3 + len)
        encoded_ids = []
        encoded_masks = []
        for cap in captions:
            ids = list(range(3, 3 + len(cap)))
            mask = [1] * len(ids)
            if max_length and truncation and len(ids) > max_length:
                ids = ids[:max_length]
                mask = mask[:max_length]
            if padding == "max_length" and max_length:
                while len(ids) < max_length:
                    ids.append(self.pad_token_id)
                    mask.append(0)
            encoded_ids.append(ids)
            encoded_masks.append(mask)
        # Pad to longest if no max_length given
        target = max((len(x) for x in encoded_ids), default=0)
        for ids, mask in zip(encoded_ids, encoded_masks):
            while len(ids) < target:
                ids.append(self.pad_token_id)
                mask.append(0)
        return {
            "input_ids": torch.tensor(encoded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoded_masks, dtype=torch.long),
        }

    def encode(self, caption, add_special_tokens=True):
        return list(range(3, 3 + len(caption)))


def test_fixed_length_pads_short_inputs():
    tok = _StubTokenizer()
    padder = _ftp.FixedTokenPadder(tok, fixed_length=32)
    encoded = padder.encode_batch(["abc", "ab"])
    assert encoded["input_ids"].shape == (2, 32)
    assert encoded["attention_mask"].shape == (2, 32)
    # First sample has 3 valid tokens
    assert encoded["attention_mask"][0, :3].sum() == 3
    assert encoded["attention_mask"][0, 3:].sum() == 0
    print("PASS: short captions are padded to fixed_length=32")


def test_fixed_length_truncates_long_inputs():
    tok = _StubTokenizer()
    padder = _ftp.FixedTokenPadder(tok, fixed_length=8)
    long_cap = "a" * 100
    encoded = padder.encode_batch([long_cap])
    assert encoded["input_ids"].shape == (1, 8)
    assert encoded["attention_mask"].sum() == 8
    print("PASS: long captions are truncated to fixed_length=8")


def test_consistent_shape_across_batches():
    tok = _StubTokenizer()
    padder = _ftp.FixedTokenPadder(tok, fixed_length=64)
    # Variety of caption lengths
    batches = [
        padder.encode_batch(["x"]),
        padder.encode_batch(["a longer caption", "y"]),
        padder.encode_batch(["aaa", "bbbbbb", "cccccccccc"]),
    ]
    for b in batches:
        assert b["input_ids"].shape[-1] == 64
    assert _ftp.verify_static_shape(batches) is True
    print("PASS: all batches have identical token shape (compile-friendly)")


def test_verify_static_shape_detects_drift():
    a = {"input_ids": torch.zeros((2, 32), dtype=torch.long)}
    b = {"input_ids": torch.zeros((2, 64), dtype=torch.long)}
    assert _ftp.verify_static_shape([a, b]) is False
    print("PASS: verify_static_shape flags shape drift")


def test_encode_one_returns_single_row():
    tok = _StubTokenizer()
    padder = _ftp.FixedTokenPadder(tok, fixed_length=16)
    encoded = padder.encode_one("hello")
    assert encoded["input_ids"].shape == (1, 16)
    print("PASS: encode_one wraps single caption correctly")


def test_invalid_fixed_length_raises():
    try:
        _ftp.FixedTokenPadder(_StubTokenizer(), fixed_length=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("PASS: fixed_length<1 raises ValueError")


def test_pad_token_id_inferred_from_tokenizer():
    class _NoPadTokenizer:
        eos_token_id = 99

        def __call__(self, captions, **_kwargs):
            ids = [[3, 4]] * len(captions)
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), 2, dtype=torch.long),
            }

        def encode(self, *args, **kwargs):
            return [3, 4]

    tok = _NoPadTokenizer()
    padder = _ftp.FixedTokenPadder(tok, fixed_length=8)
    encoded = padder.encode_batch(["x"])
    assert encoded["input_ids"].shape == (1, 8)
    # Padding should use eos_token_id when pad is missing
    assert encoded["input_ids"][0, -1].item() == 99
    print("PASS: pad token falls back to eos_token_id when pad_token_id missing")


if __name__ == "__main__":
    test_fixed_length_pads_short_inputs()
    test_fixed_length_truncates_long_inputs()
    test_consistent_shape_across_batches()
    test_verify_static_shape_detects_drift()
    test_encode_one_returns_single_row()
    test_invalid_fixed_length_raises()
    test_pad_token_id_inferred_from_tokenizer()
    print("\nAll fixed_token_padding smoke tests passed!")
