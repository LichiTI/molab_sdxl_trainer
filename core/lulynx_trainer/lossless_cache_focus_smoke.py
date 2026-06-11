"""Smoke checks for lossless focused sample-order diagnostics."""

from __future__ import annotations

from pathlib import Path
import random
import sys
from types import SimpleNamespace


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_cache_focus import (  # type: ignore[no-redef]
        focus_sample_sequence,
        parse_focus_sample_ids,
        sample_id_of,
    )
else:
    from .lossless_cache_focus import focus_sample_sequence, parse_focus_sample_ids, sample_id_of


def _samples(ids: list[str]):
    return [SimpleNamespace(sample_id=str(item), stem=f"sample_{item}") for item in ids]


def _ids(samples: list[object]) -> list[str]:
    return [sample_id_of(sample) for sample in samples]


def test_parse_focus_sample_ids() -> None:
    assert parse_focus_sample_ids("8, 32 8;59\n32") == ("8", "32", "59")
    assert parse_focus_sample_ids([8, "32", "  ", 8, "59"]) == ("8", "32", "59")
    assert parse_focus_sample_ids(None) == ()


def test_focus_sequence_reports_missing_without_dropping() -> None:
    ordered, report = focus_sample_sequence(_samples(["0", "1", "2", "3"]), ("3", "1", "9"))
    assert _ids(ordered) == ["3", "1", "0", "2"]
    assert report["enabled"] is True
    assert report["matched_sample_ids"] == ["3", "1"]
    assert report["missing_sample_ids"] == ["9"]
    assert report["focused_count"] == 2
    assert report["total_count"] == 4


def test_focus_after_shuffle() -> None:
    samples = _samples(["0", "1", "8", "32", "59", "70"])
    random.Random(1337).shuffle(samples)
    ordered, report = focus_sample_sequence(samples, ("8", "32", "59"))
    assert _ids(ordered)[:3] == ["8", "32", "59"]
    assert report["missing_sample_ids"] == []


def main() -> int:
    test_parse_focus_sample_ids()
    test_focus_sequence_reports_missing_without_dropping()
    test_focus_after_shuffle()
    print("lossless_cache_focus_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
