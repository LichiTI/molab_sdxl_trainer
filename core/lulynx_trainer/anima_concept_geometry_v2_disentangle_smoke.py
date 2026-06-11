"""Synthetic multi-concept geometry smoke for Concept Geometry v2 disentanglement.

This does not train a model. It fabricates a small Anima cache-like dataset
with three identities, shared outfits, and shared backgrounds, then verifies
that Concept Geometry v2 geometry keeps identity groups coherent while still recording
local neighbors and bounded sibling candidates.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _load_local_module(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_cached = _load_local_module("anima_cached_dataset_disentangle", "anima_cached_dataset.py")
_prep = _load_local_module("concept_geometry_prep_disentangle", "concept_geometry_prep.py")
AnimaCachedDataset = _cached.AnimaCachedDataset
_ConceptGeometryCurriculumBatchSampler = _cached._ConceptGeometryCurriculumBatchSampler
build_concept_geometry = _prep.build_concept_geometry


def _unit(values: list[float], dim: int = 16) -> np.ndarray:
    out = np.zeros((dim,), dtype=np.float32)
    out[: len(values)] = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(out))
    return out / max(norm, 1e-6)


IDENTITIES = {
    "lulu": _unit([1.0, 0.0, 0.0, 0.2]),
    "momo": _unit([0.0, 1.0, 0.0, 0.2]),
    "raven": _unit([0.0, 0.0, 1.0, 0.2]),
}
OUTFITS = {
    "red hoodie": _unit([0.0, 0.4, 0.6, 1.0]),
    "blue armor": _unit([0.6, 0.0, 0.4, 1.0]),
    "white dress": _unit([0.4, 0.6, 0.0, 1.0]),
    "black cape": _unit([0.5, 0.5, 0.5, 1.0]),
}
BACKGROUNDS = {
    "neon city background": _unit([0.2, 0.0, 0.2, 0.0, 1.0]),
    "forest background": _unit([0.0, 0.2, 0.2, 0.0, 1.0]),
}


def _write_synthetic_sample(root: Path, stem: str, identity: str, outfit: str, background: str, pose: str) -> None:
    vector = 1.00 * IDENTITIES[identity] + 0.45 * OUTFITS[outfit] + 0.20 * BACKGROUNDS[background]
    vector = vector / max(float(np.linalg.norm(vector)), 1e-6)
    latent = np.tile(vector.reshape(16, 1, 1), (1, 8, 8)).astype(np.float32)
    latent += np.linspace(0.0, 0.03, 8, dtype=np.float32).reshape(1, 1, 8)
    np.savez(root / f"{stem}_64x64_anima.npz", latents_64=latent)
    np.savez(
        root / f"{stem}_anima_te.npz",
        prompt_embeds=np.zeros((4, 32), dtype=np.float32),
        attn_mask=np.ones((4,), dtype=np.bool_),
    )
    caption = f"{identity}, original character, {outfit}, {background}, {pose}, sharp eyes"
    (root / f"{stem}.txt").write_text(caption, encoding="utf-8")


def _make_dataset(root: Path) -> None:
    outfits = list(OUTFITS.keys())
    backgrounds = list(BACKGROUNDS.keys())
    poses = ["standing pose", "sitting pose", "dynamic angle", "looking back"]
    for identity_index, identity in enumerate(IDENTITIES):
        for variant in range(4):
            stem = f"{identity}_{variant}"
            _write_synthetic_sample(
                root,
                stem,
                identity,
                outfits[(identity_index + variant) % len(outfits)],
                backgrounds[variant % len(backgrounds)],
                poses[variant],
            )


def _same_group(samples: dict[str, dict[str, Any]], stem: str, other: str) -> bool:
    return samples[stem].get("concept_group") == samples[other].get("concept_group")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_dataset(root)
        geometry_path = root / "concept_geometry.json"
        payload = build_concept_geometry(
            root,
            output_path=geometry_path,
            backend="latent_tags",
            feature_dim=96,
            neighbors=3,
            concept_depth=4,
        )
        samples = payload["samples"]
        group_counts = Counter(sample["concept_group"] for sample in samples.values())
        assert group_counts == {"lulu": 4, "momo": 4, "raven": 4}, group_counts

        sibling_lengths = []
        neighbor_same = 0
        neighbor_total = 0
        for stem, sample in samples.items():
            assert sample["concept_group_source"] in {"identity", "tag", "folder"}
            assert sample["concept_group"] in IDENTITIES
            siblings = sample.get("sibling_ids", [])
            sibling_lengths.append(len(siblings))
            assert all(_same_group(samples, stem, sibling) for sibling in siblings), (stem, siblings)
            assert len(siblings) <= 9
            for neighbor in sample.get("neighbor_ids", []):
                neighbor_total += 1
                if _same_group(samples, stem, neighbor):
                    neighbor_same += 1

        same_ratio = neighbor_same / max(neighbor_total, 1)
        assert same_ratio >= 0.60, same_ratio

        dataset = AnimaCachedDataset(
            data_dir=root,
            concept_geometry_enabled=True,
            concept_geometry_path=str(geometry_path),
            concept_geometry_sampler_mode="concept_batch",
            concept_geometry_seed=123,
        )
        batch = next(iter(_ConceptGeometryCurriculumBatchSampler(dataset, batch_size=3, shuffle=True)))
        batch_stems = [dataset.samples[index].stem for index in batch]
        assert len(batch_stems) == len(set(batch_stems)) == 3

        print(
            "Concept Geometry v2 synthetic disentangle smoke passed: "
            f"groups={dict(group_counts)} neighbor_same_ratio={same_ratio:.3f} "
            f"sibling_len_range=({min(sibling_lengths)}, {max(sibling_lengths)}) "
            f"concept_batch={batch_stems}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


