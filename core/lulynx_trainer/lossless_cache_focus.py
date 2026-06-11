"""Diagnostic sample-order helpers for lossless cache probes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def parse_focus_sample_ids(value: str | Iterable[Any] | None) -> tuple[str, ...]:
    """Parse a comma/space separated focus sample-id list, preserving order."""

    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.replace(";", ",").replace("\n", ",").split(",")
        expanded: list[str] = []
        for item in raw_items:
            expanded.extend(part for part in item.strip().split() if part)
    else:
        expanded = [str(item).strip() for item in value if str(item).strip()]

    seen: set[str] = set()
    output: list[str] = []
    for item in expanded:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return tuple(output)


def sample_id_of(sample: Any) -> str:
    value = getattr(sample, "sample_id", None)
    if value is not None and str(value):
        return str(value)
    stem = getattr(sample, "stem", None)
    return str(stem) if stem is not None and str(stem) else ""


def focus_sample_sequence(samples: Sequence[Any], focus_ids: Iterable[Any]) -> tuple[list[Any], dict[str, Any]]:
    """Move requested sample ids to the front for focused diagnostics.

    The helper does not drop samples. It only changes order, so the same dataset
    can still run normal short-window A/B while making one suspicious sample
    appear in the first few steps.
    """

    requested = parse_focus_sample_ids(focus_ids)
    ordered = list(samples)
    if not requested or not ordered:
        return ordered, {
            "enabled": False,
            "requested_sample_ids": list(requested),
            "matched_sample_ids": [],
            "missing_sample_ids": list(requested),
            "focused_count": 0,
            "total_count": len(ordered),
        }

    rank = {sample_id: index for index, sample_id in enumerate(requested)}
    focused = [sample for sample in ordered if sample_id_of(sample) in rank]
    focused.sort(key=lambda sample: rank.get(sample_id_of(sample), len(rank)))
    focused_ids = {sample_id_of(sample) for sample in focused}
    rest = [sample for sample in ordered if sample_id_of(sample) not in focused_ids]
    matched = [sample_id for sample_id in requested if sample_id in focused_ids]
    missing = [sample_id for sample_id in requested if sample_id not in focused_ids]
    return focused + rest, {
        "enabled": bool(focused),
        "requested_sample_ids": list(requested),
        "matched_sample_ids": matched,
        "missing_sample_ids": missing,
        "focused_count": len(focused),
        "total_count": len(ordered),
    }


__all__ = [
    "focus_sample_sequence",
    "parse_focus_sample_ids",
    "sample_id_of",
]
