"""Smoke checks for Anima staged-resolution cache-first planning."""

from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.lulynx_trainer.anima_cache_builder import _resize_for_target_resolution
from core.lulynx_trainer.staged_resolution import (
    build_staged_resolution_plan,
    parse_stage_batch_sizes,
)


def main() -> None:
    batches = parse_stage_batch_sizes("512:2, 768:1, 1024:1")
    assert batches == {512: 2, 768: 1, 1024: 1}, batches

    stages = build_staged_resolution_plan(
        enabled=True,
        final_resolution="1024,1024",
        max_epochs=10,
        ratios={512: 20, 768: 30, 1024: 50, 1536: 10},
        stage_batch_sizes="512:2,768:1,1024:1",
        data_dir="H:/tmp/anima_staged_smoke",
    )
    assert [stage.resolution for stage in stages] == [512, 768, 1024], stages
    assert [stage.start_epoch for stage in stages] == [0, 2, 5], stages
    assert [stage.batch_size for stage in stages] == [2, 1, 1], stages
    assert all("anima_staged" in stage.cache_dir for stage in stages), stages

    resized = _resize_for_target_resolution(Image.new("RGB", (1200, 800)), 768)
    assert max(resized.size) == 768, resized.size
    assert resized.size[0] % 16 == 0 and resized.size[1] % 16 == 0, resized.size

    print("staged_resolution_smoke: ok")


if __name__ == "__main__":
    main()
