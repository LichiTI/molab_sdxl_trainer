"""Smoke tests for attention-driven execution profile selection."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
for path in (REPO_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from backend.core.execution_manifest import get_profile_entry  # noqa: E402
from backend.core.execution_resolver import TrainingExecutionResolver  # noqa: E402


def test_default_profile_can_follow_dedicated_attention() -> None:
    resolver = TrainingExecutionResolver(BACKEND_ROOT)

    entry = resolver._resolve_profile_entry("standard", "flash2")
    assert entry is not None
    assert entry.id == "flash2"

    entry = resolver._resolve_profile_entry("", "sparge")
    assert entry is not None
    assert entry.id == "spargeattn2"

    entry = resolver._resolve_profile_entry("default", "flexattention")
    assert entry is not None
    assert entry.id == "flexattention"


def test_explicit_non_default_profile_stays_strict() -> None:
    resolver = TrainingExecutionResolver(BACKEND_ROOT)

    entry = resolver._resolve_profile_entry("apple-mps", "flash2")
    assert entry is not None
    assert entry.id == "apple-mps"


def test_attention_normalization_before_auto_default() -> None:
    resolver = TrainingExecutionResolver(BACKEND_ROOT)
    entry = get_profile_entry("standard")
    assert entry is not None

    resolved, fallback_reason = resolver._resolve_attention(
        entry=entry,
        requested=" Auto ",
        allow_fallback=False,
        model_type="sdxl",
        training_type="lora",
        available=["sdpa", "torch"],
    )
    assert resolved == "sdpa"
    assert fallback_reason == ""


def main() -> None:
    test_default_profile_can_follow_dedicated_attention()
    test_explicit_non_default_profile_stays_strict()
    test_attention_normalization_before_auto_default()
    print("execution resolver profile auto smoke: ok")


if __name__ == "__main__":
    main()
