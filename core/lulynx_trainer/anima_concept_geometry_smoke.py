"""Smoke test for Concept Geometry Sampling on the Anima cached route."""

from __future__ import annotations

import json
import sys
import tempfile
import importlib.util
from pathlib import Path

import numpy as np
import torch

def _load_local_module(name: str, filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_cached = _load_local_module("anima_cached_dataset_local", "anima_cached_dataset.py")
_prep = _load_local_module("concept_geometry_prep_local", "concept_geometry_prep.py")
_inspector = _load_local_module("dataset_inspector_local", "dataset_inspector.py")
AnimaCachedDataset = _cached.AnimaCachedDataset
_ConceptGeometryCurriculumBatchSampler = _cached._ConceptGeometryCurriculumBatchSampler
anima_cached_collate = _cached.anima_cached_collate
build_concept_geometry = _prep.build_concept_geometry
InspectorOptions = _inspector.InspectorOptions
inspect_dataset = _inspector.inspect_dataset


def _write_sample(root: Path, stem: str, caption: str) -> None:
    np.savez(
        root / f"{stem}_64x64_anima.npz",
        latents_64=np.zeros((16, 8, 8), dtype=np.float32),
    )
    np.savez(
        root / f"{stem}_anima_te.npz",
        prompt_embeds=np.zeros((4, 32), dtype=np.float32),
        attn_mask=np.ones((4,), dtype=np.bool_),
    )
    (root / f"{stem}.txt").write_text(caption, encoding="utf-8")


def _write_manual_geometry(root: Path) -> Path:
    geometry_path = root / "concept_geometry_manual.json"
    payload = {
        "meta": {"backend_resolved": "manual"},
        "samples": {
            "sample_a": {
                "stage": "core",
                "density": 0.95,
                "radius": 0.10,
                "curriculum_score": 0.10,
                "loss_weight": 1.00,
                "concept_group": "animal",
                "concept_path": ["animal", "fox"],
                "path_depth": 2,
            },
            "sample_b": {
                "stage": "mid",
                "density": 0.60,
                "radius": 0.45,
                "curriculum_score": 0.45,
                "loss_weight": 0.95,
                "concept_group": "animal",
                "concept_path": ["animal", "mechanical fox"],
                "path_depth": 2,
            },
            "sample_c": {
                "stage": "edge",
                "density": 0.15,
                "radius": 0.90,
                "curriculum_score": 0.90,
                "loss_weight": 0.85,
                "concept_group": "animal",
                "concept_path": ["animal", "cyberpunk fox"],
                "path_depth": 2,
            },
        },
    }
    geometry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return geometry_path


def _write_manual_v2_geometry(root: Path) -> Path:
    geometry_path = root / "concept_geometry_v2_manual.json"
    payload = {
        "meta": {
            "geometry_version": 2,
            "backend_requested": "hybrid",
            "backend_resolved": "latent+tags",
            "feature_sources": ["latent", "tags"],
            "fallback_reasons": ["dino: missing local path"],
        },
        "samples": {
            "sample_a": {
                "stage": "core",
                "density": 0.95,
                "radius": 0.10,
                "curriculum_score": 0.10,
                "loss_weight": 1.00,
                "concept_group": "animal",
                "concept_path": ["animal", "fox"],
                "path_depth": 2,
                "geometry_version": 2,
                "backend_requested": "hybrid",
                "backend_resolved": "latent+tags",
                "feature_sources": {"latent": {}, "tags": {}},
                "fallback_reasons": ["dino: missing local path"],
                "tag_buckets": {"identity": ["fox"]},
                "source_density": {"latent": 0.9, "tags": 0.8},
                "neighbor_ids": ["sample_b", "sample_c"],
                "sibling_ids": ["sample_b"],
                "conflict_score": 0.5,
            },
            "sample_b": {
                "stage": "mid",
                "density": 0.60,
                "radius": 0.45,
                "curriculum_score": 0.45,
                "loss_weight": 0.95,
                "concept_group": "animal",
                "concept_path": ["animal", "mechanical fox"],
                "path_depth": 2,
                "geometry_version": 2,
                "backend_requested": "hybrid",
                "backend_resolved": "latent+tags",
                "feature_sources": {"latent": {}, "tags": {}},
                "tag_buckets": {"identity": ["mechanical fox"]},
                "source_density": {"latent": 0.6, "tags": 0.5},
                "neighbor_ids": ["sample_a"],
                "sibling_ids": ["sample_a"],
                "conflict_score": 0.0,
            },
            "sample_c": {
                "stage": "edge",
                "density": 0.15,
                "radius": 0.90,
                "curriculum_score": 0.90,
                "loss_weight": 0.85,
                "concept_group": "weather",
                "concept_path": ["weather", "rain"],
                "path_depth": 2,
                "geometry_version": 2,
                "backend_requested": "hybrid",
                "backend_resolved": "latent+tags",
                "feature_sources": {"latent": {}, "tags": {}},
                "tag_buckets": {"setting": ["rain"]},
                "source_density": {"latent": 0.2, "tags": 0.1},
                "neighbor_ids": ["sample_a"],
                "sibling_ids": [],
                "conflict_score": 1.0,
            },
        },
    }
    geometry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return geometry_path


def _test_prep_script(root: Path) -> None:
    payload = build_concept_geometry(
        root,
        backend="latent_tags",
        caption_extension=".txt",
        concept_depth=3,
        feature_dim=128,
        neighbors=2,
    )
    assert payload["meta"]["geometry_version"] == 2
    assert set(payload["meta"]["feature_sources"]) == {"latent", "tags"}
    samples = payload.get("samples", {})
    assert len(samples) == 3
    for stem in ("sample_a", "sample_b", "sample_c"):
        assert stem in samples
        assert samples[stem]["stage"] in {"core", "mid", "edge"}
        assert "density" in samples[stem]
        assert "concept_path" in samples[stem]
        assert "neighbor_ids" in samples[stem]
        assert "tag_buckets" in samples[stem]


def _test_optional_sources(root: Path) -> None:
    clip_features = root / "clip_features.npz"
    dino_features = root / "dino_features.npz"
    np.savez(
        clip_features,
        sample_a=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        sample_b=np.array([0.9, 0.1, 0.0], dtype=np.float32),
        sample_c=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    np.savez(
        dino_features,
        sample_a=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        sample_b=np.array([0.1, 0.9, 0.0], dtype=np.float32),
        sample_c=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    clip_payload = build_concept_geometry(root, backend="clip", clip_model_path=str(clip_features), feature_dim=64, neighbors=1)
    assert clip_payload["meta"]["backend_resolved"] == "clip"
    dino_payload = build_concept_geometry(root, backend="dino", dino_model_path=str(dino_features), feature_dim=64, neighbors=1)
    assert dino_payload["meta"]["backend_resolved"] == "dino"
    hybrid_payload = build_concept_geometry(root, backend="hybrid", clip_model_path="", dino_model_path="", feature_dim=64, neighbors=1)
    assert "dino:" in " ".join(hybrid_payload["meta"]["fallback_reasons"])
    assert set(hybrid_payload["meta"]["feature_sources"]) == {"clip", "latent", "tags"}


def _test_dataset_curriculum_and_weighting(root: Path, geometry_path: Path) -> None:
    dataset = AnimaCachedDataset(
        data_dir=root,
        caption_extension=".txt",
        concept_geometry_enabled=True,
        concept_geometry_path=str(geometry_path),
        concept_geometry_sampler_mode="density_curriculum",
        concept_geometry_loss_weighting=True,
        concept_geometry_density_power=1.0,
        concept_geometry_total_epochs=4,
    )
    assert dataset.is_concept_geometry_enabled()

    stem_to_index = {sample.stem: idx for idx, sample in enumerate(dataset.samples)}
    batch = anima_cached_collate([dataset[0], dataset[1], dataset[2]])
    assert "geometry_weights" in batch
    assert "geometry_stages" in batch
    assert batch["geometry_weights"].shape[0] == 3

    dataset.set_current_epoch(0)
    early_weights = dataset.get_concept_geometry_sampling_weights()
    dataset.set_current_epoch(3)
    late_weights = dataset.get_concept_geometry_sampling_weights()

    early_core = early_weights[stem_to_index["sample_a"]]
    early_edge = early_weights[stem_to_index["sample_c"]]
    late_core = late_weights[stem_to_index["sample_a"]]
    late_edge = late_weights[stem_to_index["sample_c"]]

    assert early_core > early_edge
    assert late_edge > late_core

    weights = batch["geometry_weights"]
    assert isinstance(weights, torch.Tensor)
    assert not torch.allclose(weights, torch.ones_like(weights))


def _test_concept_batch_sampler(root: Path, geometry_path: Path) -> None:
    dataset = AnimaCachedDataset(
        data_dir=root,
        caption_extension=".txt",
        concept_geometry_enabled=True,
        concept_geometry_path=str(geometry_path),
        concept_geometry_sampler_mode="concept_batch",
        concept_geometry_seed=7,
    )
    sampler = _ConceptGeometryCurriculumBatchSampler(dataset, batch_size=3, shuffle=True, drop_last=False)
    first = next(iter(sampler))
    stems = [dataset.samples[index].stem for index in first]
    assert len(first) == 3
    assert any(stem in stems for stem in ("sample_a", "sample_b"))

    singleton = _ConceptGeometryCurriculumBatchSampler(dataset, batch_size=1, shuffle=True, drop_last=False)
    assert all(len(batch) == 1 for batch in singleton)

    v1_path = _write_manual_geometry(root)
    v1_dataset = AnimaCachedDataset(
        data_dir=root,
        caption_extension=".txt",
        concept_geometry_enabled=True,
        concept_geometry_path=str(v1_path),
        concept_geometry_sampler_mode="concept_batch",
        concept_geometry_seed=7,
    )
    v1_sampler = _ConceptGeometryCurriculumBatchSampler(v1_dataset, batch_size=2, shuffle=True, drop_last=False)
    assert all(len(batch) <= 2 for batch in v1_sampler)


def _test_smart_sd_subset_discovery() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        lulu = parent / "6_lulu"
        momo = parent / "8_momo"
        lulu.mkdir()
        momo.mkdir()
        _write_sample(lulu, "0", "lulu, red dress, studio")
        _write_sample(momo, "0", "momo, red dress, studio")

        parent_payload = build_concept_geometry(
            parent,
            backend="latent_tags",
            caption_extension=".txt",
            concept_depth=3,
            feature_dim=64,
            neighbors=1,
        )
        samples = parent_payload.get("samples", {})
        assert set(samples.keys()) == {"6_lulu/0", "8_momo/0"}
        assert samples["6_lulu/0"]["concept_group"] == "lulu"
        assert samples["8_momo/0"]["concept_group"] == "momo"

        parent_geometry = parent / "concept_geometry.json"
        parent_geometry.write_text(json.dumps(parent_payload, indent=2), encoding="utf-8")
        parent_dataset = AnimaCachedDataset(
            data_dir=parent,
            caption_extension=".txt",
            concept_geometry_enabled=True,
            concept_geometry_path=str(parent_geometry),
            concept_geometry_sampler_mode="concept_batch",
        )
        assert [sample.sample_id for sample in parent_dataset.samples] == ["6_lulu/0", "8_momo/0"]
        assert parent_dataset.is_concept_geometry_enabled()

        direct_payload = build_concept_geometry(
            lulu,
            backend="latent_tags",
            caption_extension=".txt",
            concept_depth=3,
            feature_dim=64,
            neighbors=1,
        )
        assert set(direct_payload.get("samples", {}).keys()) == {"0"}
        direct_sample = next(iter(direct_payload["samples"].values()))
        assert direct_sample["concept_group"] == "lulu"
        direct_dataset = AnimaCachedDataset(data_dir=lulu, caption_extension=".txt")
        assert [sample.sample_id for sample in direct_dataset.samples] == ["0"]


def _test_caption_parsing_structured_and_nl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_sample(
            root,
            "structured",
            "concept: lulu, clothes: red dress, white shirt, pose: standing, setting: garden",
        )
        _write_sample(
            root,
            "natural",
            "Lulu is wearing a blue dress in a moonlit garden while standing.",
        )
        _write_sample(
            root,
            "zh_natural",
            "露露穿着红色连衣裙站在月光花园里。",
        )
        _write_sample(
            root,
            "english_v2",
            "Portrait of Lulu with blue eyes wearing a red dress, holding a sword near a window, anime style, without hat.",
        )
        _write_sample(
            root,
            "co_concept",
            "Lulu with Momo standing in a garden.",
        )
        _write_sample(root, "json_sidecar", "")
        (root / "json_sidecar.json").write_text(
            json.dumps(
                {
                    "concept": "Lulu Chan",
                    "tags": ["red dress", "blue eyes"],
                    "nl": "Portrait of Lulu Chan holding a sword near a window, anime style, without hat.",
                    "categories": {
                        "clothes": ["red dress"],
                        "appearance": ["blue eyes"],
                        "setting": ["window"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        payload = build_concept_geometry(
            root,
            backend="latent_tags",
            caption_extension=".txt",
            concept_depth=4,
            feature_dim=64,
            neighbors=1,
        )
        structured = payload["samples"]["structured"]
        assert structured["concept_group"] == "lulu"
        assert structured["concept_group_source"] == "concept"
        assert "red dress" in structured["tag_buckets"].get("appearance", [])
        assert "white shirt" in structured["tag_buckets"].get("appearance", [])
        assert "standing" in structured["tag_buckets"].get("pose", [])
        assert "garden" in structured["tag_buckets"].get("setting", [])

        natural = payload["samples"]["natural"]
        assert natural["concept_group"] == "lulu"
        assert "blue dress" in natural["tag_buckets"].get("appearance", [])
        assert "moonlit garden" in natural["tag_buckets"].get("setting", [])
        assert "standing" in natural["tag_buckets"].get("pose", [])

        zh_natural = payload["samples"]["zh_natural"]
        assert zh_natural["concept_group"] == "露露"
        assert "红色连衣裙" in zh_natural["tag_buckets"].get("appearance", [])
        assert "月光花园" in zh_natural["tag_buckets"].get("setting", [])
        assert "站在" in zh_natural["tag_buckets"].get("pose", [])

        english_v2 = payload["samples"]["english_v2"]
        assert english_v2["concept_group"] == "lulu"
        assert "blue eyes" in english_v2["tag_buckets"].get("appearance", [])
        assert "red dress" in english_v2["tag_buckets"].get("appearance", [])
        assert "holding sword" in english_v2["tag_buckets"].get("pose", [])
        assert "window" in english_v2["tag_buckets"].get("setting", [])
        assert "anime" in english_v2["tag_buckets"].get("style", [])
        assert "hat" not in english_v2["tag_buckets"].get("appearance", [])
        assert english_v2["parse_confidence"] >= 0.75

        co_concept = payload["samples"]["co_concept"]
        assert co_concept["concept_group"] == "lulu"
        assert co_concept["co_concepts"] == ["momo"]
        assert "multiple identity concepts detected" in co_concept["parse_warnings"]
        assert payload["meta"]["co_concept_count"] >= 1

        json_sidecar = payload["samples"]["json_sidecar"]
        assert json_sidecar["concept_group"] == "lulu chan"
        assert "red dress" in json_sidecar["tag_buckets"].get("appearance", [])
        assert "blue eyes" in json_sidecar["tag_buckets"].get("appearance", [])
        assert "holding sword" in json_sidecar["tag_buckets"].get("pose", [])
        assert "window" in json_sidecar["tag_buckets"].get("setting", [])
        assert "anime" in json_sidecar["tag_buckets"].get("style", [])
        assert "hat" not in json_sidecar["tag_buckets"].get("appearance", [])

        dataset = AnimaCachedDataset(data_dir=root, caption_extension=".txt")
        json_sample = next(sample for sample in dataset.samples if sample.sample_id == "json_sidecar")
        training_caption, _ = dataset._load_caption(json_sample)
        assert "red dress" in training_caption
        assert "Portrait of Lulu Chan" in training_caption
        assert "{" not in training_caption

        alias_payload = build_concept_geometry(
            root,
            backend="latent_tags",
            caption_extension=".txt",
            concept_depth=4,
            feature_dim=64,
            neighbors=1,
            alias_map={"露露": "lulu", "lulu chan": "lulu", "momo chan": "momo"},
        )
        assert alias_payload["samples"]["zh_natural"]["concept_group"] == "lulu"
        assert alias_payload["meta"]["alias_count"] == 3

        priority_root = root / "folder_priority"
        folder = priority_root / "6_lulu"
        folder.mkdir(parents=True)
        _write_sample(folder, "priority", "concept: momo, clothes: blue jacket")
        explicit_payload = build_concept_geometry(
            priority_root,
            backend="latent_tags",
            caption_extension=".txt",
            feature_dim=64,
            neighbors=1,
            concept_source_priority="explicit,folder,tag,stem",
        )
        folder_payload = build_concept_geometry(
            priority_root,
            backend="latent_tags",
            caption_extension=".txt",
            feature_dim=64,
            neighbors=1,
            concept_source_priority="folder,explicit,tag,stem",
        )
        assert explicit_payload["samples"]["priority"]["concept_group"] == "momo"
        assert folder_payload["samples"]["priority"]["concept_group"] == "lulu"


def _test_semantic_embedding_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_sample(root, "sample_a", "concept: lulu, clothes: red dress, setting: garden")
        _write_sample(root, "sample_b", "concept: momo, clothes: blue jacket, setting: city")
        _write_sample(root, "sample_c", "露露穿着红色连衣裙站在花园里。")
        embedding_features = root / "text_embedding_features.npz"
        np.savez(
            embedding_features,
            sample_a=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            sample_b=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            sample_c=np.array([0.5, 0.5, 0.0], dtype=np.float32),
        )
        payload = build_concept_geometry(
            root,
            backend="latent_tags",
            caption_extension=".txt",
            concept_depth=3,
            feature_dim=64,
            neighbors=1,
            semantic_enhance=True,
            embedding_provider="local_path",
            embedding_model_path=str(embedding_features),
        )
        meta = payload["meta"]
        assert meta["semantic_enhanced"] is True
        assert "text_embedding" in meta["feature_sources"]
        assert "text_embedding" in payload["samples"]["sample_a"]["feature_sources"]

        fallback_payload = build_concept_geometry(
            root,
            backend="latent_tags",
            caption_extension=".txt",
            feature_dim=64,
            neighbors=1,
            semantic_enhance=True,
            embedding_provider="auto_download",
            embedding_allow_download=False,
        )
        assert fallback_payload["meta"]["semantic_enhanced"] is False
        assert any("auto_download provider requires explicit download approval" in item for item in fallback_payload["meta"]["fallback_reasons"])

        translation_fallback_payload = build_concept_geometry(
            root,
            backend="latent_tags",
            caption_extension=".txt",
            feature_dim=64,
            neighbors=1,
            semantic_enhance=True,
            embedding_provider="local_path",
            embedding_model_path=str(embedding_features),
            translation_enabled=True,
            translation_provider="local_path",
            translation_model_path=str(root / "missing_translation_model"),
        )
        assert translation_fallback_payload["meta"]["semantic_enhanced"] is True
        assert any(item.startswith("translation:") for item in translation_fallback_payload["meta"]["fallback_reasons"])


def _test_dataset_inspector_preflight() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        lulu = parent / "6_lulu"
        momo = parent / "8_momo"
        lulu.mkdir()
        momo.mkdir()
        _write_sample(lulu, "0", "lulu, red dress, studio")
        _write_sample(lulu, "1", "lulu, blue dress, garden")
        _write_sample(momo, "0", "momo, red dress, studio")
        (momo / "1_anima_te.npz").write_bytes((momo / "0_anima_te.npz").read_bytes())
        (momo / "1.txt").write_text("momo, blue dress, garden", encoding="utf-8")

        geometry_payload = build_concept_geometry(parent, backend="latent_tags", caption_extension=".txt", neighbors=1)
        geometry_path = parent / "concept_geometry.json"
        geometry_path.write_text(json.dumps(geometry_payload, indent=2), encoding="utf-8")

        report = inspect_dataset(
            InspectorOptions(
                data_dir=parent,
                caption_extension=".txt",
                geometry_path=geometry_path,
                concept_geometry_sampler_mode="concept_batch",
                batch_size=2,
            )
        )
        assert report["scan_mode"] == "subset_parent"
        assert report["dataset"]["sample_count"] == 4
        assert report["dataset"]["concept_group_count"] == 2
        assert report["cache"]["cache_pair_count"] == 3
        assert report["cache"]["missing_latent_count"] == 1
        assert report["concept_geometry"]["geometry_attach_rate"] == 1.0
        assert report["concept_geometry"]["concept_group_count"] == 2
        assert "8_momo/0" in {item["sample_id"] for item in report["sample_preview"]}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_sample(root, "sample_a", "animal, fox, portrait")
        _write_sample(root, "sample_b", "animal, mechanical fox, neon city")
        _write_sample(root, "sample_c", "animal, cyberpunk mechanical fox, extreme angle, rain")

        _test_prep_script(root)
        _test_optional_sources(root)
        geometry_path = _write_manual_geometry(root)
        _test_dataset_curriculum_and_weighting(root, geometry_path)
        v2_geometry_path = _write_manual_v2_geometry(root)
        _test_concept_batch_sampler(root, v2_geometry_path)
        _test_smart_sd_subset_discovery()
        _test_caption_parsing_structured_and_nl()
        _test_semantic_embedding_source()
        _test_dataset_inspector_preflight()

    print("Anima Concept Geometry smoke passed: v2 geometry, categorized/NL captions, semantic embedding source/fallback, optional sources, smart subset discovery, inspector preflight, concept batches, v1 fallback, and loss weighting are wired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


