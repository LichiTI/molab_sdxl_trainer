"""Keymap manifest inspection for Warehouse native model backends.

The inspector validates checkpoint key coverage without loading tensors.  It is
intended to keep early native backend work honest while the actual block
implementations are still being built.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from ..safetensors_loader import open_safetensors
except ImportError:
    from safetensors import safe_open as _safe_open

    def open_safetensors(
        path: str | Path,
        *,
        framework: str = "pt",
        device: str = "cpu",
        disable_mmap: bool = False,
    ):
        del disable_mmap
        return _safe_open(str(path), framework=framework, device=device)


@dataclass(frozen=True)
class KeymapResolvedEntry:
    source_key: str
    local_source_key: str
    target_key: str
    rule_name: str
    shape: list[int]
    dtype: str

    @property
    def rank(self) -> int:
        return len(self.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "local_source_key": self.local_source_key,
            "target_key": self.target_key,
            "rule_name": self.rule_name,
            "shape": self.shape,
            "rank": self.rank,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class KeymapInspectionReport:
    manifest_path: str
    model_path: str
    family: str
    component: str
    total_keys: int
    component_keys: int
    matched_keys: int
    unmatched_keys: int
    required_prefixes_missing: list[str]
    rule_counts: dict[str, int]
    rank_counts: dict[str, int]
    dtype_counts: dict[str, int]
    duplicate_targets: int
    duplicate_targets_sample: list[str]
    mappings_sample: list[dict[str, Any]]
    unmatched_keys_sample: list[str]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "model_path": self.model_path,
            "family": self.family,
            "component": self.component,
            "total_keys": self.total_keys,
            "component_keys": self.component_keys,
            "matched_keys": self.matched_keys,
            "unmatched_keys": self.unmatched_keys,
            "required_prefixes_missing": self.required_prefixes_missing,
            "rule_counts": self.rule_counts,
            "rank_counts": self.rank_counts,
            "dtype_counts": self.dtype_counts,
            "duplicate_targets": self.duplicate_targets,
            "duplicate_targets_sample": self.duplicate_targets_sample,
            "mappings_sample": self.mappings_sample,
            "unmatched_keys_sample": self.unmatched_keys_sample,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class StateTensorPlan:
    source_key: str
    target_key: str
    shape: list[int]
    dtype: str
    rule_name: str

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "target_key": self.target_key,
            "shape": self.shape,
            "dtype": self.dtype,
            "rule_name": self.rule_name,
            "numel": self.numel,
        }


@dataclass(frozen=True)
class StateShapeMismatch:
    target_key: str
    source_shape: list[int]
    target_shape: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_key": self.target_key,
            "source_shape": self.source_shape,
            "target_shape": self.target_shape,
        }


@dataclass(frozen=True)
class StateMappingPlan:
    manifest_path: str
    model_path: str
    family: str
    component: str
    source_format: str
    target_format: str
    dry_run: bool
    tensors: int
    total_parameters: int
    dtype_counts: dict[str, int]
    rank_counts: dict[str, int]
    rule_counts: dict[str, int]
    duplicate_targets: int
    missing_targets: int
    unexpected_targets: int
    shape_mismatches: int
    duplicate_targets_sample: list[str]
    missing_targets_sample: list[str]
    unexpected_targets_sample: list[str]
    shape_mismatches_sample: list[dict[str, Any]]
    tensor_plan_sample: list[dict[str, Any]]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "model_path": self.model_path,
            "family": self.family,
            "component": self.component,
            "source_format": self.source_format,
            "target_format": self.target_format,
            "dry_run": self.dry_run,
            "tensors": self.tensors,
            "total_parameters": self.total_parameters,
            "dtype_counts": self.dtype_counts,
            "rank_counts": self.rank_counts,
            "rule_counts": self.rule_counts,
            "duplicate_targets": self.duplicate_targets,
            "missing_targets": self.missing_targets,
            "unexpected_targets": self.unexpected_targets,
            "shape_mismatches": self.shape_mismatches,
            "duplicate_targets_sample": self.duplicate_targets_sample,
            "missing_targets_sample": self.missing_targets_sample,
            "unexpected_targets_sample": self.unexpected_targets_sample,
            "shape_mismatches_sample": self.shape_mismatches_sample,
            "tensor_plan_sample": self.tensor_plan_sample,
            "ok": self.ok,
        }


def _path_cache_key(path: str | Path) -> tuple[str, int, int]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=32)
def _load_manifest_cached(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    with Path(path).open("r", encoding="utf-8") as handle:
        return handle.read()


@lru_cache(maxsize=16)
def _read_safetensors_key_metadata_cached(path: str, mtime_ns: int, size: int) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    del mtime_ns, size
    with open_safetensors(str(path), framework="pt", device="cpu") as handle:
        return tuple(
            (
                key,
                tuple(int(dim) for dim in handle.get_slice(key).get_shape()),
                str(handle.get_slice(key).get_dtype()),
            )
            for key in handle.keys()
        )


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    cache_key = _path_cache_key(manifest_path)
    return json.loads(_load_manifest_cached(*cache_key))


def read_safetensors_keys(path: str | Path) -> list[str]:
    # safe_open exposes metadata and keys without materializing tensor payloads.
    with open_safetensors(str(path), framework="pt", device="cpu") as handle:
        return list(handle.keys())


def read_safetensors_key_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    cache_key = _path_cache_key(path)
    return {
        key: {
            "shape": list(shape),
            "dtype": dtype,
        }
        for key, shape, dtype in _read_safetensors_key_metadata_cached(*cache_key)
    }


def _replace_leaf_prefix(leaf: str, replacements: dict[str, str]) -> str:
    for source_prefix, target_prefix in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if leaf == source_prefix:
            return target_prefix
        if leaf.startswith(source_prefix + "."):
            return target_prefix + leaf[len(source_prefix) :]
    return leaf


def _render_regex_target(rule: dict[str, Any], match: re.Match[str]) -> str:
    values: dict[str, Any] = dict(match.groupdict())
    if "leaf" in values:
        values["leaf"] = _replace_leaf_prefix(str(values["leaf"]), dict(rule.get("leaf_replacements", {})))

    if "input_index" in values:
        input_index = int(values["input_index"])
        values["down_block"] = (input_index - 1) // 3
        values["down_resnet_index"] = (input_index - 1) % 3
        values.setdefault("resnet_index", values["down_resnet_index"])
    if "output_index" in values:
        output_index = int(values["output_index"])
        values["up_block"] = output_index // 3
        values["up_resnet_index"] = output_index % 3
        values.setdefault("resnet_index", values["up_resnet_index"])

    return str(rule["target_template"]).format(**values)


def build_resolved_keymap_entries(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[KeymapResolvedEntry], list[str], dict[str, dict[str, Any]]]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    if model_path is None:
        model_path = manifest.get("expected_local_model")
    if not model_path:
        raise ValueError(f"{manifest_path}: model_path is required")

    metadata = read_safetensors_key_metadata(model_path)
    checkpoint_prefix = str(manifest.get("checkpoint_prefix", ""))
    target_prefix = str(manifest.get("target_prefix", ""))

    if checkpoint_prefix:
        component_pairs = [
            (key, key[len(checkpoint_prefix) :])
            for key in metadata
            if key.startswith(checkpoint_prefix)
        ]
    else:
        component_pairs = [(key, key) for key in metadata]

    exact_rules = {
        str(rule["source"]): rule
        for rule in manifest.get("top_level_rules", [])
    }
    regex_rules = [
        (rule, re.compile(str(rule["source_regex"])))
        for rule in manifest.get("regex_rules", [])
    ]

    entries: list[KeymapResolvedEntry] = []
    unmatched: list[str] = []
    for full_key, local_key in component_pairs:
        exact_rule = exact_rules.get(local_key)
        if exact_rule is not None:
            rule_name = str(exact_rule.get("name") or exact_rule["source"])
            target_key = str(exact_rule["target"])
        else:
            rule_name = ""
            target_key = ""
            for regex_rule, pattern in regex_rules:
                match = pattern.match(local_key)
                if match is None:
                    continue
                rule_name = str(regex_rule.get("name") or regex_rule["source_regex"])
                target_key = _render_regex_target(regex_rule, match)
                break
            if not rule_name:
                unmatched.append(full_key)
                continue

        tensor_meta = metadata[full_key]
        entries.append(
            KeymapResolvedEntry(
                source_key=full_key,
                local_source_key=local_key,
                target_key=target_prefix + target_key,
                rule_name=rule_name,
                shape=list(tensor_meta["shape"]),
                dtype=str(tensor_meta["dtype"]),
            )
        )

    return manifest, entries, unmatched, metadata


def inspect_keymap_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
) -> KeymapInspectionReport:
    manifest_path = Path(manifest_path)
    manifest, entries, unmatched, metadata = build_resolved_keymap_entries(manifest_path, model_path)
    if model_path is None:
        model_path = manifest.get("expected_local_model")
    if not model_path:
        raise ValueError(f"{manifest_path}: model_path is required")

    model_path = Path(model_path)
    all_keys = list(metadata)
    checkpoint_prefix = str(manifest.get("checkpoint_prefix", ""))
    if checkpoint_prefix:
        component_keys = [key for key in all_keys if key.startswith(checkpoint_prefix)]
    else:
        component_keys = all_keys
    full_key_set = set(all_keys)

    missing_required = [
        prefix
        for prefix in manifest.get("required_source_prefixes", [])
        if not any(key.startswith(prefix) for key in full_key_set)
    ]

    rule_counts: Counter[str] = Counter()
    rank_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for entry in entries:
        rule_counts[entry.rule_name] += 1
        rank_counts[str(entry.rank)] += 1
        dtype_counts[entry.dtype] += 1
        target_counts[entry.target_key] += 1

    duplicate_targets = sorted(target for target, count in target_counts.items() if count > 1)

    expectations = manifest.get("coverage_expectations", {})
    min_component_keys = int(expectations.get("min_component_keys", 0))
    min_rule_matches = int(expectations.get("min_rule_matches", 0))
    max_unmatched_keys = expectations.get("max_unmatched_keys")
    if max_unmatched_keys is not None:
        max_unmatched_keys = int(max_unmatched_keys)

    ok = (
        not missing_required
        and len(component_keys) >= min_component_keys
        and len(entries) >= min_rule_matches
        and (max_unmatched_keys is None or len(unmatched) <= max_unmatched_keys)
        and not duplicate_targets
    )

    return KeymapInspectionReport(
        manifest_path=str(manifest_path),
        model_path=str(model_path),
        family=str(manifest.get("family", "")),
        component=str(manifest.get("component", "")),
        total_keys=len(all_keys),
        component_keys=len(component_keys),
        matched_keys=len(entries),
        unmatched_keys=len(unmatched),
        required_prefixes_missing=missing_required,
        rule_counts=dict(sorted(rule_counts.items())),
        rank_counts=dict(sorted(rank_counts.items())),
        dtype_counts=dict(sorted(dtype_counts.items())),
        duplicate_targets=len(duplicate_targets),
        duplicate_targets_sample=duplicate_targets[:40],
        mappings_sample=[entry.to_dict() for entry in entries[:20]],
        unmatched_keys_sample=unmatched[:40],
        ok=ok,
    )


def _normalize_target_metadata(target_metadata: Any) -> dict[str, list[int]]:
    if target_metadata is None:
        return {}
    normalized: dict[str, list[int]] = {}
    for key, value in dict(target_metadata).items():
        if isinstance(value, dict) and "shape" in value:
            normalized[str(key)] = list(value["shape"])
        elif hasattr(value, "shape"):
            normalized[str(key)] = [int(dim) for dim in value.shape]
        else:
            normalized[str(key)] = [int(dim) for dim in value]
    return normalized


def build_state_mapping_plan(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
    target_metadata: dict[str, Any] | None = None,
) -> StateMappingPlan:
    manifest_path = Path(manifest_path)
    manifest, entries, unmatched, _metadata = build_resolved_keymap_entries(manifest_path, model_path)
    if model_path is None:
        model_path = manifest.get("expected_local_model")
    if not model_path:
        raise ValueError(f"{manifest_path}: model_path is required")

    tensor_plans = [
        StateTensorPlan(
            source_key=entry.source_key,
            target_key=entry.target_key,
            shape=entry.shape,
            dtype=entry.dtype,
            rule_name=entry.rule_name,
        )
        for entry in entries
    ]

    target_counts: Counter[str] = Counter(plan.target_key for plan in tensor_plans)
    duplicate_targets = sorted(target for target, count in target_counts.items() if count > 1)
    source_target_shapes: dict[str, list[int]] = {
        plan.target_key: plan.shape
        for plan in tensor_plans
        if target_counts[plan.target_key] == 1
    }
    target_shapes = _normalize_target_metadata(target_metadata)
    dry_run = target_metadata is None

    if dry_run:
        missing_targets: list[str] = []
        unexpected_targets: list[str] = []
        shape_mismatches: list[StateShapeMismatch] = []
    else:
        expected_targets = set(source_target_shapes)
        actual_targets = set(target_shapes)
        missing_targets = sorted(expected_targets - actual_targets)
        unexpected_targets = sorted(actual_targets - expected_targets)
        shape_mismatches = [
            StateShapeMismatch(
                target_key=target_key,
                source_shape=source_shape,
                target_shape=target_shapes[target_key],
            )
            for target_key, source_shape in sorted(source_target_shapes.items())
            if target_key in target_shapes and list(source_shape) != list(target_shapes[target_key])
        ]

    dtype_counts: Counter[str] = Counter()
    rank_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    rule_numel: defaultdict[str, int] = defaultdict(int)
    total_parameters = 0
    for plan in tensor_plans:
        dtype_counts[plan.dtype] += 1
        rank_counts[str(len(plan.shape))] += 1
        rule_counts[plan.rule_name] += 1
        rule_numel[plan.rule_name] += plan.numel
        total_parameters += plan.numel

    ok = (
        not unmatched
        and not duplicate_targets
        and (dry_run or (not missing_targets and not unexpected_targets and not shape_mismatches))
    )

    return StateMappingPlan(
        manifest_path=str(manifest_path),
        model_path=str(model_path),
        family=str(manifest.get("family", "")),
        component=str(manifest.get("component", "")),
        source_format=str(manifest.get("source_format", "")),
        target_format=str(manifest.get("target_format", "")),
        dry_run=dry_run,
        tensors=len(tensor_plans),
        total_parameters=total_parameters,
        dtype_counts=dict(sorted(dtype_counts.items())),
        rank_counts=dict(sorted(rank_counts.items())),
        rule_counts=dict(sorted(rule_counts.items())),
        duplicate_targets=len(duplicate_targets),
        missing_targets=len(missing_targets),
        unexpected_targets=len(unexpected_targets),
        shape_mismatches=len(shape_mismatches),
        duplicate_targets_sample=duplicate_targets[:40],
        missing_targets_sample=missing_targets[:40],
        unexpected_targets_sample=unexpected_targets[:40],
        shape_mismatches_sample=[item.to_dict() for item in shape_mismatches[:40]],
        tensor_plan_sample=[plan.to_dict() for plan in tensor_plans[:20]],
        ok=ok,
    )


def inspect_many(manifest_paths: list[str | Path]) -> list[KeymapInspectionReport]:
    return [inspect_keymap_manifest(path) for path in manifest_paths]


def build_many_state_mapping_plans(manifest_paths: list[str | Path]) -> list[StateMappingPlan]:
    return [build_state_mapping_plan(path) for path in manifest_paths]

