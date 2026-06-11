# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke checks for eval_dataset helper contract.

This stays intentionally lightweight: the helper ultimately depends on the
full dataset loader stack, which can make a tiny smoke flaky on Windows.
Here we verify the key invariants directly from source.
"""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    source = (Path(__file__).resolve().parent / "eval_dataset.py").read_text(encoding="utf-8")

    required_snippets = (
        "shuffle_caption=False",
        "flip_augment=False",
        "color_augment=False",
        "caption_dropout_rate=0.0",
        "tag_dropout_rate=0.0",
        "shuffle=False",
    )
    missing = [snippet for snippet in required_snippets if snippet not in source]
    if missing:
        raise AssertionError(f"eval_dataset.py is missing required deterministic eval snippets: {missing}")

    print("PASS: eval dataset smoke", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
