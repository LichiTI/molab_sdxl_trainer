# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for caption_augment.py (Phase 8.13 / #100)."""

from __future__ import annotations

import os
import sys
import importlib.util
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.caption_augment",
    os.path.join(_HERE, "caption_augment.py"),
)
_ca = importlib.util.module_from_spec(_spec)
sys.modules["core.lulynx_trainer.caption_augment"] = _ca
_spec.loader.exec_module(_ca)


def test_shuffle_respects_keep_tokens():
    rng = random.Random(0)
    tags = ["girl", "1girl", "blue eyes", "blonde hair", "smile"]
    out = _ca.shuffle_tags(tags, keep_tokens=2, rng=rng)
    assert out[:2] == tags[:2], "first keep_tokens must be preserved"
    assert sorted(out) == sorted(tags)
    print("PASS: shuffle_tags preserves keep_tokens prefix")


def test_swap_adjacent_pairs_identity_at_zero_rate():
    tags = ["a", "b", "c", "d"]
    out = _ca.swap_adjacent_pairs(tags, rate=0.0)
    assert out == tags
    print("PASS: swap_adjacent_pairs is identity at rate=0")


def test_swap_adjacent_pairs_full_rate_swaps_pairs():
    tags = ["a", "b", "c", "d"]
    out = _ca.swap_adjacent_pairs(tags, rate=1.0, rng=random.Random(0))
    # rate=1.0 always swaps each adjacent pair, in stride-2 fashion
    assert out == ["b", "a", "d", "c"]
    print("PASS: swap_adjacent_pairs swaps each pair at rate=1.0")


def test_shuffle_groups_preserves_separator():
    rng = random.Random(0)
    tags = ["a", "b", "|||", "c", "d", "|||", "e", "f"]
    out = _ca.shuffle_tags_groups(tags, separator="|||", rng=rng)
    # separator markers preserved at original group positions
    assert out.count("|||") == 2
    # all original tokens present
    assert sorted(out) == sorted(tags)
    print("PASS: shuffle_tags_groups preserves separators and content")


def test_drop_tags_zero_rate_returns_copy():
    tags = ["a", "b", "c"]
    out = _ca.drop_tags(tags, rate=0.0)
    assert out == tags
    assert out is not tags
    print("PASS: drop_tags(rate=0) returns a copy")


def test_drop_targeted_tags_substring_match():
    tags = ["1girl", "girl", "blue eyes", "smile"]
    out = _ca.drop_targeted_tags(tags, targets=("girl",))
    assert "1girl" not in out and "girl" not in out
    assert "blue eyes" in out and "smile" in out
    print("PASS: drop_targeted_tags removes substring matches")


def test_replace_synonyms_picks_from_list():
    rng = random.Random(0)
    tags = ["smile"]
    out = _ca.replace_synonyms(tags, {"smile": ["grin", "smirk"]}, rng=rng)
    assert out[0] in {"grin", "smirk"}
    print("PASS: replace_synonyms picks from a list")


def test_apply_caption_augments_combines_strategies():
    cfg = _ca.CaptionAugmentConfig(
        shuffle=True,
        keep_tokens=1,
        tag_dropout_rate=0.0,
        tag_swap_rate=0.0,
        targeted_tag_drops=("nsfw",),
    )
    rng = random.Random(0)
    tags = ["1girl", "smile", "blue eyes", "nsfw_tag", "blonde"]
    out = _ca.apply_caption_augments(tags, cfg, rng=rng)
    # First token preserved, nsfw filtered, others shuffled
    assert out[0] == "1girl"
    assert "nsfw_tag" not in out
    assert sorted(out[1:]) == sorted(["smile", "blue eyes", "blonde"])
    print("PASS: apply_caption_augments combines drop + shuffle correctly")


def test_apply_caption_augments_deterministic_with_seed():
    cfg = _ca.CaptionAugmentConfig(shuffle=True, keep_tokens=0)
    tags = ["a", "b", "c", "d", "e", "f"]
    out_a = _ca.apply_caption_augments(tags, cfg, rng=random.Random(42))
    out_b = _ca.apply_caption_augments(tags, cfg, rng=random.Random(42))
    assert out_a == out_b
    print("PASS: apply_caption_augments is deterministic per RNG seed")


if __name__ == "__main__":
    test_shuffle_respects_keep_tokens()
    test_swap_adjacent_pairs_identity_at_zero_rate()
    test_swap_adjacent_pairs_full_rate_swaps_pairs()
    test_shuffle_groups_preserves_separator()
    test_drop_tags_zero_rate_returns_copy()
    test_drop_targeted_tags_substring_match()
    test_replace_synonyms_picks_from_list()
    test_apply_caption_augments_combines_strategies()
    test_apply_caption_augments_deterministic_with_seed()
    print("\nAll caption_augment smoke tests passed!")
