"""Dataset discovery helpers shared by trainer data paths.

The smart subset mode is intentionally compatible with the useful part of
sd-scripts/kohya DreamBooth discovery without inheriting its awkward edge case:
pointing directly at ``6_lulu`` is treated as one valid subset, not as a broken
dataset root that needs to be reorganized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SD_SUBSET_RE = re.compile(r"^(?P<repeats>\d+)_(?P<name>.+)$")


@dataclass(frozen=True)
class DatasetSubset:
    root: Path
    rel_parts: tuple[str, ...]
    repeats: int = 1
    concept: str = ""
    source: str = "root"


@dataclass(frozen=True)
class DatasetSampleRecord:
    sample_id: str
    stem: str
    subset_root: Path
    subset_name: str
    concept: str
    repeats: int
    rel_parts: tuple[str, ...]
    image_path: Path | None = None
    caption_path: Path | None = None
    latent_path: Path | None = None
    text_cache_path: Path | None = None


@dataclass(frozen=True)
class DatasetDiscoveryResult:
    data_dir: Path
    scan_mode: str
    subsets: tuple[DatasetSubset, ...]
    samples: tuple[DatasetSampleRecord, ...]
    warnings: tuple[str, ...] = ()


def parse_sd_subset_name(name: str) -> tuple[int, str] | None:
    match = SD_SUBSET_RE.match(str(name or "").strip())
    if match is None:
        return None
    repeats = max(int(match.group("repeats") or 1), 1)
    concept = match.group("name").strip("_ ")
    if not concept:
        return None
    return repeats, concept


def normalize_caption_extension(caption_extension: str = ".txt") -> str:
    value = str(caption_extension or ".txt").strip()
    return value if value.startswith(".") else f".{value}"


def discover_smart_subsets(data_dir: str | Path) -> list[DatasetSubset]:
    """Discover sd-scripts style subsets with a trainer-friendly fallback.

    Rules:
    - If ``data_dir`` itself is named like ``6_lulu``, use it as one subset.
    - Otherwise, use immediate children named like ``N_concept``.
    - If neither applies, return ``data_dir`` as a native trainer root.
    """

    root = Path(data_dir)
    parsed_self = parse_sd_subset_name(root.name)
    if parsed_self is not None:
        repeats, concept = parsed_self
        return [DatasetSubset(root=root, rel_parts=(concept,), repeats=repeats, concept=concept, source="self")]

    subsets: list[DatasetSubset] = []
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_dir():
                continue
            parsed = parse_sd_subset_name(child.name)
            if parsed is None:
                continue
            repeats, concept = parsed
            subsets.append(
                DatasetSubset(root=child, rel_parts=(concept,), repeats=repeats, concept=concept, source="child")
            )
    if subsets:
        return subsets
    return [DatasetSubset(root=root, rel_parts=(), repeats=1, concept="", source="root")]


def discover_dataset_samples(data_dir: str | Path, caption_extension: str = ".txt") -> DatasetDiscoveryResult:
    root = Path(data_dir)
    warnings: list[str] = []
    if not root.exists():
        return DatasetDiscoveryResult(root, "invalid", (), (), (f"data_dir does not exist: {root}",))
    if not root.is_dir():
        return DatasetDiscoveryResult(root, "invalid", (), (), (f"data_dir is not a directory: {root}",))

    subsets = discover_smart_subsets(root)
    scan_mode = "native_recursive"
    if len(subsets) == 1 and subsets[0].source == "self":
        scan_mode = "single_subset"
    elif any(subset.source == "child" for subset in subsets):
        scan_mode = "subset_parent"

    raw: dict[tuple[str, Path], dict[str, object]] = {}

    def ensure(stem: str, subset: DatasetSubset) -> dict[str, object]:
        key = (stem, subset.root)
        row = raw.get(key)
        if row is not None:
            return row
        row = {
            "stem": stem,
            "subset": subset,
            "image_path": None,
            "caption_path": None,
            "latent_path": None,
            "text_cache_path": None,
        }
        raw[key] = row
        return row

    for subset in subsets:
        for image_path in iter_images_for_subset(subset):
            if any(image_path.stem.endswith(sidecar) for sidecar in ("_mask", "_alpha", "_control")):
                continue
            row = ensure(image_path.stem, subset)
            row["image_path"] = image_path
            row["caption_path"] = resolve_caption_path(subset.root, image_path.stem, caption_extension)

        for text_path in iter_anima_text_cache_paths(subset.root):
            stem = anima_text_cache_stem(text_path)
            if stem:
                ensure(stem, subset)["text_cache_path"] = text_path

        for latent_path in iter_anima_latent_cache_paths(subset.root):
            stem = anima_latent_cache_stem(latent_path)
            if not stem:
                continue
            row = ensure(stem, subset)
            existing = row.get("latent_path")
            if existing is None or latent_path.stat().st_mtime < Path(existing).stat().st_mtime:
                row["latent_path"] = latent_path
            if row.get("caption_path") is None:
                row["caption_path"] = resolve_caption_path(subset.root, stem, caption_extension)

    id_map = assign_stable_sample_ids(list(raw.keys()), root)
    samples: list[DatasetSampleRecord] = []
    for key, row in sorted(raw.items(), key=lambda item: (str(item[0][1]), item[0][0])):
        stem, subset_root = key
        subset = row["subset"]
        assert isinstance(subset, DatasetSubset)
        samples.append(
            DatasetSampleRecord(
                sample_id=id_map[key],
                stem=stem,
                subset_root=subset_root,
                subset_name=subset.root.name,
                concept=subset.concept,
                repeats=subset.repeats,
                rel_parts=subset.rel_parts,
                image_path=_optional_path(row.get("image_path")),
                caption_path=_optional_path(row.get("caption_path")),
                latent_path=_optional_path(row.get("latent_path")),
                text_cache_path=_optional_path(row.get("text_cache_path")),
            )
        )

    if not samples:
        scan_mode = "empty"
        warnings.append("No supported images or Anima cache pairs were discovered.")
    return DatasetDiscoveryResult(root, scan_mode, tuple(subsets), tuple(samples), tuple(warnings))


def iter_images_for_subset(subset: DatasetSubset) -> Iterable[Path]:
    """Iterate images using sd-style non-recursive subset scanning when possible."""

    iterator = subset.root.glob("*") if subset.source in {"self", "child"} else subset.root.rglob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def iter_anima_text_cache_paths(root: Path) -> Iterable[Path]:
    for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
        for path in sorted(root.glob(f"*{suffix}")):
            if path.is_file():
                yield path


def iter_anima_latent_cache_paths(root: Path) -> Iterable[Path]:
    for suffix in ("_anima.npz", "_anima.safetensors", "_anima.pt"):
        for path in sorted(root.glob(f"*{suffix}")):
            if path.is_file() and "_anima_te" not in path.stem:
                yield path


def anima_text_cache_stem(path: Path) -> str:
    for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return ""


def anima_latent_cache_stem(path: Path) -> str:
    match = re.match(r"^(?P<stem>.+)_\d+x\d+_anima(?:\.(?:npz|safetensors|pt))$", path.name)
    return match.group("stem") if match is not None else ""


def caption_candidates_for_stem(root: Path, stem: str, caption_extension: str = ".txt") -> list[Path]:
    suffix = normalize_caption_extension(caption_extension)
    candidates = []
    if suffix.lower() != ".json":
        candidates.append(root / f"{stem}.json")
    candidates.append(root / f"{stem}{suffix}")
    if suffix.lower() != ".txt":
        candidates.append(root / f"{stem}.txt")
    if suffix.lower() != ".caption":
        candidates.append(root / f"{stem}.caption")
    for image_suffix in sorted(IMAGE_SUFFIXES):
        candidates.append(root / f"{stem}{image_suffix}.txt")
    return _dedupe_paths(candidates)


def resolve_caption_path(root: Path, stem: str, caption_extension: str = ".txt") -> Path | None:
    for candidate in caption_candidates_for_stem(root, stem, caption_extension):
        if candidate.is_file():
            return candidate
    return None


def relative_sample_id(data_dir: Path, sample_root: Path, stem: str) -> str:
    try:
        rel_parent = sample_root.relative_to(data_dir)
    except ValueError:
        rel_parent = Path()
    if str(rel_parent) in {"", "."}:
        return stem
    return (rel_parent / stem).as_posix()


def assign_stable_sample_ids(
    rows: Sequence[tuple[str, Path]],
    data_dir: str | Path,
) -> dict[tuple[str, Path], str]:
    """Assign old stem ids when unique, relative ids only when needed."""

    data_root = Path(data_dir)
    counts: dict[str, int] = {}
    for stem, _root in rows:
        counts[stem] = counts.get(stem, 0) + 1
    ids: dict[tuple[str, Path], str] = {}
    used: set[str] = set()
    for stem, root in rows:
        sample_id = stem if counts.get(stem, 0) == 1 else relative_sample_id(data_root, root, stem)
        if sample_id in used:
            sample_id = relative_sample_id(data_root, root, stem)
        ids[(stem, root)] = sample_id
        used.add(sample_id)
    return ids


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _optional_path(value: object) -> Path | None:
    return value if isinstance(value, Path) else None
