# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Caption augmentation variants (Phase 8.13 / #100).

A small library of caption augmentation strategies that operate on a
list of tag strings and return a possibly-mutated list.  All functions
are pure in-place operations on a copy and accept a ``random.Random``
instance for deterministic seeding.

Strategies
----------
- ``shuffle_tags`` — full random reorder
- ``shuffle_tags_groups`` — split by separator, shuffle within each group
- ``swap_adjacent_pairs`` — swap random adjacent tag pairs with probability ``rate``
- ``drop_tags`` — drop tags independently with probability ``rate``
- ``drop_targeted_tags`` — drop tags whose substring matches any target
- ``replace_synonyms`` — substitute tags by entries from a synonym dict

Usage::

    from .caption_augment import CaptionAugmentConfig, apply_caption_augments

    cfg = CaptionAugmentConfig(
        shuffle=True,
        keep_tokens=2,
        tag_swap_rate=0.1,
        tag_group_shuffle=True,
        tag_group_separator="|||",
    )
    new_tags = apply_caption_augments(tags, cfg, rng=random.Random(42))
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class CaptionAugmentConfig:
    """Configuration for caption-level augmentation."""

    shuffle: bool = False
    keep_tokens: int = 0
    tag_swap_rate: float = 0.0          # probability of swapping each adjacent pair
    tag_group_shuffle: bool = False     # shuffle tags within groups split by separator
    tag_group_separator: str = "|||"
    tag_dropout_rate: float = 0.0
    targeted_tag_drops: tuple = field(default_factory=tuple)
    synonyms: dict = field(default_factory=dict)


def shuffle_tags(tags: List[str], keep_tokens: int = 0, *, rng: Optional[random.Random] = None) -> List[str]:
    """Full random shuffle of ``tags``, optionally keeping the first
    ``keep_tokens`` fixed."""
    rng = rng or random
    if keep_tokens >= len(tags):
        return list(tags)
    head = list(tags[:keep_tokens])
    tail = list(tags[keep_tokens:])
    rng.shuffle(tail)
    return head + tail


def shuffle_tags_groups(
    tags: List[str],
    separator: str = "|||",
    *,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Split tags into groups by ``separator`` markers and shuffle within each group.

    A tag exactly equal to ``separator`` acts as a group boundary and is preserved.
    """
    rng = rng or random
    if not tags:
        return list(tags)

    groups: List[List[str]] = [[]]
    boundaries: List[int] = []  # group indices that are pure separators
    for i, t in enumerate(tags):
        if t == separator:
            boundaries.append(len(groups))
            groups.append([separator])
            groups.append([])
        else:
            groups[-1].append(t)

    for i, g in enumerate(groups):
        if i in boundaries:
            continue
        rng.shuffle(g)

    out: List[str] = []
    for g in groups:
        out.extend(g)
    return out


def swap_adjacent_pairs(
    tags: List[str],
    rate: float,
    keep_tokens: int = 0,
    *,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """For each adjacent pair after the first ``keep_tokens``, swap with probability ``rate``."""
    if rate <= 0.0 or len(tags) < keep_tokens + 2:
        return list(tags)
    rng = rng or random
    out = list(tags)
    i = keep_tokens
    while i < len(out) - 1:
        if rng.random() < rate:
            out[i], out[i + 1] = out[i + 1], out[i]
            i += 2
        else:
            i += 1
    return out


def drop_tags(
    tags: List[str],
    rate: float,
    keep_tokens: int = 0,
    *,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Independently drop each tag with probability ``rate`` (after keep_tokens)."""
    if rate <= 0.0:
        return list(tags)
    rng = rng or random
    head = list(tags[:keep_tokens])
    tail = [t for t in tags[keep_tokens:] if rng.random() >= rate]
    return head + tail


def drop_targeted_tags(tags: List[str], targets: Iterable[str]) -> List[str]:
    """Drop any tag that contains one of ``targets`` as a substring."""
    targets = tuple(t.strip().lower() for t in targets if t.strip())
    if not targets:
        return list(tags)
    return [t for t in tags if not any(target in t.lower() for target in targets)]


def replace_synonyms(tags: List[str], synonyms: dict, *, rng: Optional[random.Random] = None) -> List[str]:
    """Replace tags by random choice from their synonym list, if present."""
    if not synonyms:
        return list(tags)
    rng = rng or random
    out: List[str] = []
    for t in tags:
        candidates = synonyms.get(t)
        if candidates:
            choice = rng.choice(list(candidates)) if isinstance(candidates, (list, tuple)) else candidates
            out.append(str(choice))
        else:
            out.append(t)
    return out


def apply_caption_augments(
    tags: List[str],
    config: CaptionAugmentConfig,
    *,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Apply all configured augmentations in a deterministic order."""
    out = list(tags)
    if config.targeted_tag_drops:
        out = drop_targeted_tags(out, config.targeted_tag_drops)
    if config.tag_dropout_rate > 0.0:
        out = drop_tags(out, config.tag_dropout_rate, config.keep_tokens, rng=rng)
    if config.synonyms:
        out = replace_synonyms(out, config.synonyms, rng=rng)
    if config.tag_group_shuffle:
        out = shuffle_tags_groups(out, config.tag_group_separator, rng=rng)
    elif config.shuffle:
        out = shuffle_tags(out, config.keep_tokens, rng=rng)
    if config.tag_swap_rate > 0.0:
        out = swap_adjacent_pairs(out, config.tag_swap_rate, config.keep_tokens, rng=rng)
    return out
