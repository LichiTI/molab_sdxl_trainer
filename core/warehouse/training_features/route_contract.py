"""Route Contract — metadata-oriented contracts for training routes.

A route contract describes *what* a training route supports: which model
architectures, which capabilities, which parameter ranges, and what
hardware constraints apply.  Contracts are pure data — they do not
reference file paths, scripts, or execution details.  Pure-stdlib,
no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import Capability, ModelArchitecture, RouteFamily


# ---------------------------------------------------------------------------
# Parameter constraints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamRange:
    """Acceptable range for a single numeric training parameter."""

    name: str
    min_value: float
    max_value: float
    default: float
    step: float = 1.0
    label: str = ""

    def validate(self, value: float) -> bool:
        """Return True if *value* falls within [min_value, max_value]."""
        return self.min_value <= value <= self.max_value

    def clamp(self, value: float) -> float:
        """Clamp *value* to the valid range."""
        return max(self.min_value, min(self.max_value, value))


@dataclass(frozen=True)
class ParamChoice:
    """A parameter constrained to a set of allowed values."""

    name: str
    choices: frozenset[str]
    default: str
    label: str = ""

    def validate(self, value: str) -> bool:
        return value in self.choices


# ---------------------------------------------------------------------------
# Hardware hints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VramHint:
    """Approximate VRAM requirements for a route at a given resolution."""

    min_mb: int
    recommended_mb: int
    resolution: str = "512x512"


# ---------------------------------------------------------------------------
# Route contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteContract:
    """Immutable description of a training route's metadata and constraints.

    A contract answers: "what does this route support?" without answering
    "how does it run?"  This separation lets UI and validation layers
    reason about route suitability without importing trainer code.
    """

    route_id: str
    display_name: str
    family: RouteFamily
    architectures: frozenset[ModelArchitecture]
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    description: str = ""
    param_ranges: tuple[ParamRange, ...] = ()
    param_choices: tuple[ParamChoice, ...] = ()
    vram_hints: tuple[VramHint, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)
    extra: dict[str, Any] = field(default_factory=dict)

    # -- convenience --------------------------------------------------------

    def supports_architecture(self, arch: ModelArchitecture) -> bool:
        return arch in self.architectures

    def supports_capability(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def has_all_capabilities(self, caps: set[Capability]) -> bool:
        return caps <= self.capabilities

    def param_range(self, name: str) -> ParamRange | None:
        """Look up a :class:`ParamRange` by parameter name."""
        for pr in self.param_ranges:
            if pr.name == name:
                return pr
        return None

    def param_choice(self, name: str) -> ParamChoice | None:
        """Look up a :class:`ParamChoice` by parameter name."""
        for pc in self.param_choices:
            if pc.name == name:
                return pc
        return None

    def vram_for_resolution(self, resolution: str) -> VramHint | None:
        """Return the VRAM hint for a specific resolution, or None."""
        for vh in self.vram_hints:
            if vh.resolution == resolution:
                return vh
        return None

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate a dict of parameter values against this contract.

        Returns a list of human-readable error messages (empty = valid).
        """
        errors: list[str] = []
        for pr in self.param_ranges:
            if pr.name in params:
                val = params[pr.name]
                if not isinstance(val, (int, float)):
                    errors.append(f"{pr.name}: expected numeric, got {type(val).__name__}")
                elif not pr.validate(val):
                    errors.append(
                        f"{pr.name}: {val} out of range [{pr.min_value}, {pr.max_value}]"
                    )
        for pc in self.param_choices:
            if pc.name in params:
                val = params[pc.name]
                if not isinstance(val, str):
                    errors.append(f"{pc.name}: expected string, got {type(val).__name__}")
                elif not pc.validate(val):
                    errors.append(f"{pc.name}: '{val}' not in {sorted(pc.choices)}")
        return errors


# ---------------------------------------------------------------------------
# Contract collection
# ---------------------------------------------------------------------------

class RouteContractSet:
    """An ordered, queryable collection of :class:`RouteContract` objects."""

    def __init__(self) -> None:
        self._contracts: dict[str, RouteContract] = {}

    def add(self, contract: RouteContract) -> None:
        self._contracts[contract.route_id] = contract

    def remove(self, route_id: str) -> bool:
        return self._contracts.pop(route_id, None) is not None

    def get(self, route_id: str) -> RouteContract | None:
        return self._contracts.get(route_id)

    def __contains__(self, route_id: str) -> bool:
        return route_id in self._contracts

    def __len__(self) -> int:
        return len(self._contracts)

    def __iter__(self):
        return iter(self._contracts.values())

    @property
    def route_ids(self) -> list[str]:
        return list(self._contracts.keys())

    def query(
        self,
        *,
        family: RouteFamily | None = None,
        architecture: ModelArchitecture | None = None,
        capability: Capability | None = None,
        all_capabilities: set[Capability] | None = None,
    ) -> list[RouteContract]:
        """Return contracts matching all provided filters (AND semantics)."""
        result: list[RouteContract] = []
        for c in self._contracts.values():
            if family is not None and c.family != family:
                continue
            if architecture is not None and not c.supports_architecture(architecture):
                continue
            if capability is not None and not c.supports_capability(capability):
                continue
            if all_capabilities is not None and not c.has_all_capabilities(all_capabilities):
                continue
            result.append(c)
        return result
