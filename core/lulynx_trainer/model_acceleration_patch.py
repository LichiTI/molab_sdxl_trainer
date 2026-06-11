"""Small patch helpers for model acceleration policies."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def apply_typed_patch_map(
    *,
    patch: Mapping[str, Any],
    patch_bool: Callable[[str, bool], None],
    patch_int: Callable[[str, int], None],
    patch_text: Callable[[str, str], None],
) -> None:
    """Apply a policy patch while preserving bool/int/text semantics."""

    for key, value in patch.items():
        if isinstance(value, bool):
            patch_bool(key, value)
        elif isinstance(value, int):
            patch_int(key, value)
        else:
            patch_text(key, str(value))


__all__ = ["apply_typed_patch_map"]
