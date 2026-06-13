"""
Training route contracts — declarative binding between trainer types and
their required inputs, outputs, and capabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class TrainingRouteContract:
    """Declares the contract for a training route.

    A *route* is a named configuration profile that maps a training type
    (e.g. LoRA, DreamBooth, full fine-tune) to the inputs it expects,
    outputs it produces, and host capabilities it requires.

    Attributes:
        trainer_type: Identifier for the training strategy.
        display_name: Human-readable label.
        description: What this route trains.
        required_capabilities: Capability names the host must provide.
        required_hooks: Hook names that must be callable.
        input_schema: JSON-Schema dict describing expected input parameters.
        output_schema: JSON-Schema dict describing the training artefacts.
        default_config: Key-value defaults applied when the user omits values.
        metadata: Arbitrary route metadata.
    """

    trainer_type: str
    display_name: str = ""
    description: str = ""
    required_capabilities: Sequence[str] = field(default_factory=tuple)
    required_hooks: Sequence[str] = field(default_factory=tuple)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    default_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "trainer_type": self.trainer_type,
            "display_name": self.display_name,
            "description": self.description,
            "required_capabilities": list(self.required_capabilities),
            "required_hooks": list(self.required_hooks),
            "default_config": self.default_config,
            "metadata": self.metadata,
        }
        if self.input_schema is not None:
            d["input_schema"] = self.input_schema
        if self.output_schema is not None:
            d["output_schema"] = self.output_schema
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingRouteContract:
        return cls(
            trainer_type=data["trainer_type"],
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            required_capabilities=data.get("required_capabilities", []),
            required_hooks=data.get("required_hooks", []),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            default_config=data.get("default_config", {}),
            metadata=data.get("metadata", {}),
        )


class _RouteRegistry:
    """Internal singleton that stores resolved route contracts."""

    def __init__(self) -> None:
        self._routes: Dict[str, TrainingRouteContract] = {}

    def register(self, contract: TrainingRouteContract) -> None:
        if contract.trainer_type in self._routes:
            raise ValueError(
                f"Route contract for '{contract.trainer_type}' already registered"
            )
        self._routes[contract.trainer_type] = contract

    def get(self, trainer_type: str) -> Optional[TrainingRouteContract]:
        return self._routes.get(trainer_type)

    def all(self) -> List[TrainingRouteContract]:
        return list(self._routes.values())

    def clear(self) -> None:
        self._routes.clear()


_global_registry = _RouteRegistry()


def resolve_route_contract(
    trainer_type: str,
    *,
    from_dict: Optional[Dict[str, Any]] = None,
    from_file: Optional[Path | str] = None,
) -> TrainingRouteContract:
    """Look up or lazily register a route contract for *trainer_type*.

    Resolution order:
    1. Already-registered contract (in-memory cache).
    2. If *from_dict* is supplied, register and return it.
    3. If *from_file* is supplied, load JSON, register, and return it.
    4. ``ValueError`` if the contract cannot be resolved.
    """
    existing = _global_registry.get(trainer_type)
    if existing is not None:
        return existing

    if from_dict is not None:
        contract = TrainingRouteContract.from_dict(from_dict)
        _global_registry.register(contract)
        return contract

    if from_file is not None:
        data = json.loads(Path(from_file).read_text(encoding="utf-8"))
        contract = TrainingRouteContract.from_dict(data)
        _global_registry.register(contract)
        return contract

    raise ValueError(
        f"No route contract registered for trainer type '{trainer_type}'"
    )


def clear_route_registry() -> None:
    """Reset the global route registry (useful in tests)."""
    _global_registry.clear()
