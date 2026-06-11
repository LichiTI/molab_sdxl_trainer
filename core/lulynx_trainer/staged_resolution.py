"""Helpers for staged/progressive resolution training.

The first supported production path is Anima cache-first training: build one
cache directory per stage before training, then switch cached datasets at epoch
boundaries without resetting optimizer or scheduler state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StagedResolutionStage:
    resolution: int
    ratio: float
    start_epoch: int
    batch_size: int | None = None
    cache_dir: str = ""


def parse_resolution_value(value: Any) -> int:
    if isinstance(value, int):
        return max(int(value), 1)
    raw = str(value or "").strip()
    if not raw:
        return 1024
    parts = [part.strip() for part in raw.replace("x", ",").split(",") if part.strip()]
    if not parts:
        return 1024
    return max(int(float(max(parts, key=lambda item: float(item)))), 1)


def parse_stage_batch_sizes(value: Any) -> dict[int, int]:
    raw = str(value or "").strip()
    if not raw:
        return {}
    result: dict[int, int] = {}
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if ":" not in token:
            continue
        left, right = token.split(":", 1)
        try:
            resolution = int(float(left.strip()))
            batch_size = int(float(right.strip().replace("bs", "")))
        except ValueError:
            continue
        if resolution > 0 and batch_size > 0:
            result[resolution] = batch_size
    return result


def build_staged_resolution_plan(
    *,
    enabled: bool,
    final_resolution: Any,
    max_epochs: int,
    ratios: dict[int, Any],
    stage_batch_sizes: Any = "",
    data_dir: str = "",
) -> list[StagedResolutionStage]:
    if not enabled:
        return []
    final_res = parse_resolution_value(final_resolution)
    normalized: list[tuple[int, float]] = []
    for resolution in sorted(ratios):
        if resolution > final_res:
            continue
        try:
            ratio = float(ratios.get(resolution) or 0.0)
        except (TypeError, ValueError):
            ratio = 0.0
        if ratio > 0:
            normalized.append((int(resolution), ratio))
    if not normalized:
        normalized = [(final_res, 100.0)]
    if normalized[-1][0] != final_res:
        normalized.append((final_res, 1.0))

    total_ratio = sum(ratio for _, ratio in normalized) or 1.0
    total_epochs = max(int(max_epochs or 1), 1)
    batch_sizes = parse_stage_batch_sizes(stage_batch_sizes)
    cache_root = Path(data_dir) / ".lulynx_cache" / "anima_staged" if data_dir else Path("")

    stages: list[StagedResolutionStage] = []
    cumulative = 0.0
    for idx, (resolution, ratio) in enumerate(normalized):
        start_epoch = 0 if idx == 0 else min(max(int(round(cumulative / total_ratio * total_epochs)), 0), total_epochs - 1)
        cache_dir = str(cache_root / f"res_{resolution}") if data_dir else ""
        stages.append(
            StagedResolutionStage(
                resolution=resolution,
                ratio=ratio,
                start_epoch=start_epoch,
                batch_size=batch_sizes.get(resolution),
                cache_dir=cache_dir,
            )
        )
        cumulative += ratio

    deduped: list[StagedResolutionStage] = []
    for stage in stages:
        if deduped and deduped[-1].resolution == stage.resolution:
            continue
        deduped.append(stage)
    return deduped


def stages_to_config_strings(stages: list[StagedResolutionStage]) -> tuple[str, str]:
    if not stages:
        return "", ""
    steps = ",".join(str(stage.start_epoch) for stage in stages[1:])
    values = ",".join(str(stage.resolution) for stage in stages)
    return steps, values


def stages_to_summary(stages: list[StagedResolutionStage]) -> list[dict[str, Any]]:
    return [
        {
            "resolution": stage.resolution,
            "ratio": stage.ratio,
            "start_epoch": stage.start_epoch,
            "batch_size": stage.batch_size,
            "cache_dir": stage.cache_dir,
        }
        for stage in stages
    ]
