"""JSON serialization helpers for TrainerRegistry and RouteContractSet.

Provides ``to_dict`` / ``from_dict`` converters for individual entries and
full-collection ``to_json`` / ``from_json`` round-trip helpers.  All
conversions are stdlib-only (``json``, ``pathlib``) and preserve enum types,
frozensets, and nested frozen dataclasses.

Example::

    from core.warehouse.training_features import (
        TrainerRegistry, load_registry_json, save_registry_json,
    )

    registry = load_registry_json("trainers.json")
    save_registry_json(registry, "trainers_out.json", indent=2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .route_contract import (
    ParamChoice,
    ParamRange,
    RouteContract,
    RouteContractSet,
    VramHint,
)
from .trainer_registry import TrainerEntry, TrainerRegistry
from .types import Capability, ModelArchitecture, RouteFamily


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dictify(obj: Any) -> Any:
    """Recursively convert *obj* into a JSON-serializable structure.

    Handles enums (via ``.value``), frozensets/tuples (→ sorted lists),
    and nested dicts/lists.
    """
    if hasattr(obj, "value"):          # str-enum
        return obj.value
    if isinstance(obj, frozenset):
        return sorted(_dictify(v) for v in obj)
    if isinstance(obj, tuple):
        return [_dictify(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dictify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dictify(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# ParamRange / ParamChoice / VramHint  (individual)
# ---------------------------------------------------------------------------

def param_range_to_dict(pr: ParamRange) -> dict[str, Any]:
    return {
        "name": pr.name,
        "min_value": pr.min_value,
        "max_value": pr.max_value,
        "default": pr.default,
        "step": pr.step,
        "label": pr.label,
    }


def param_range_from_dict(data: dict[str, Any]) -> ParamRange:
    return ParamRange(
        name=data["name"],
        min_value=float(data["min_value"]),
        max_value=float(data["max_value"]),
        default=float(data["default"]),
        step=float(data.get("step", 1.0)),
        label=data.get("label", ""),
    )


def param_choice_to_dict(pc: ParamChoice) -> dict[str, Any]:
    return {
        "name": pc.name,
        "choices": sorted(pc.choices),
        "default": pc.default,
        "label": pc.label,
    }


def param_choice_from_dict(data: dict[str, Any]) -> ParamChoice:
    return ParamChoice(
        name=data["name"],
        choices=frozenset(data["choices"]),
        default=data["default"],
        label=data.get("label", ""),
    )


def vram_hint_to_dict(vh: VramHint) -> dict[str, Any]:
    return {
        "min_mb": vh.min_mb,
        "recommended_mb": vh.recommended_mb,
        "resolution": vh.resolution,
    }


def vram_hint_from_dict(data: dict[str, Any]) -> VramHint:
    return VramHint(
        min_mb=int(data["min_mb"]),
        recommended_mb=int(data["recommended_mb"]),
        resolution=data.get("resolution", "512x512"),
    )


# ---------------------------------------------------------------------------
# TrainerEntry / TrainerRegistry
# ---------------------------------------------------------------------------

def trainer_entry_to_dict(entry: TrainerEntry) -> dict[str, Any]:
    return {
        "key": entry.key,
        "display_name": entry.display_name,
        "architectures": _dictify(entry.architectures),
        "capabilities": _dictify(entry.capabilities),
        "description": entry.description,
        "extra": _dictify(entry.extra),
    }


def trainer_entry_from_dict(data: dict[str, Any]) -> TrainerEntry:
    return TrainerEntry(
        key=data["key"],
        display_name=data["display_name"],
        architectures=frozenset(ModelArchitecture(a) for a in data.get("architectures", [])),
        capabilities=frozenset(Capability(c) for c in data.get("capabilities", [])),
        description=data.get("description", ""),
        extra=data.get("extra", {}),
    )


def registry_to_dict(registry: TrainerRegistry) -> dict[str, Any]:
    return {"entries": [trainer_entry_to_dict(e) for e in registry]}


def registry_from_dict(data: dict[str, Any]) -> TrainerRegistry:
    reg = TrainerRegistry()
    for item in data.get("entries", []):
        reg.register(trainer_entry_from_dict(item))
    return reg


def registry_to_json(registry: TrainerRegistry, indent: int | None = 2) -> str:
    return json.dumps(registry_to_dict(registry), indent=indent, ensure_ascii=False)


def registry_from_json(text: str) -> TrainerRegistry:
    return registry_from_dict(json.loads(text))


def save_registry_json(
    registry: TrainerRegistry,
    path: str | Path,
    indent: int = 2,
) -> None:
    Path(path).write_text(registry_to_json(registry, indent=indent), encoding="utf-8")


def load_registry_json(path: str | Path) -> TrainerRegistry:
    return registry_from_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# RouteContract / RouteContractSet
# ---------------------------------------------------------------------------

def route_contract_to_dict(contract: RouteContract) -> dict[str, Any]:
    return {
        "route_id": contract.route_id,
        "display_name": contract.display_name,
        "family": contract.family.value,
        "architectures": _dictify(contract.architectures),
        "capabilities": _dictify(contract.capabilities),
        "description": contract.description,
        "param_ranges": [param_range_to_dict(pr) for pr in contract.param_ranges],
        "param_choices": [param_choice_to_dict(pc) for pc in contract.param_choices],
        "vram_hints": [vram_hint_to_dict(vh) for vh in contract.vram_hints],
        "tags": _dictify(contract.tags),
        "extra": _dictify(contract.extra),
    }


def route_contract_from_dict(data: dict[str, Any]) -> RouteContract:
    return RouteContract(
        route_id=data["route_id"],
        display_name=data["display_name"],
        family=RouteFamily(data["family"]),
        architectures=frozenset(ModelArchitecture(a) for a in data.get("architectures", [])),
        capabilities=frozenset(Capability(c) for c in data.get("capabilities", [])),
        description=data.get("description", ""),
        param_ranges=tuple(param_range_from_dict(pr) for pr in data.get("param_ranges", [])),
        param_choices=tuple(param_choice_from_dict(pc) for pc in data.get("param_choices", [])),
        vram_hints=tuple(vram_hint_from_dict(vh) for vh in data.get("vram_hints", [])),
        tags=frozenset(data.get("tags", [])),
        extra=data.get("extra", {}),
    )


def contract_set_to_dict(contract_set: RouteContractSet) -> dict[str, Any]:
    return {"contracts": [route_contract_to_dict(c) for c in contract_set]}


def contract_set_from_dict(data: dict[str, Any]) -> RouteContractSet:
    cs = RouteContractSet()
    for item in data.get("contracts", []):
        cs.add(route_contract_from_dict(item))
    return cs


def contract_set_to_json(contract_set: RouteContractSet, indent: int | None = 2) -> str:
    return json.dumps(contract_set_to_dict(contract_set), indent=indent, ensure_ascii=False)


def contract_set_from_json(text: str) -> RouteContractSet:
    return contract_set_from_dict(json.loads(text))


def save_contract_set_json(
    contract_set: RouteContractSet,
    path: str | Path,
    indent: int = 2,
) -> None:
    Path(path).write_text(
        contract_set_to_json(contract_set, indent=indent), encoding="utf-8"
    )


def load_contract_set_json(path: str | Path) -> RouteContractSet:
    return contract_set_from_json(Path(path).read_text(encoding="utf-8"))
