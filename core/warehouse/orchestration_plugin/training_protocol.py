"""
Training protocol — declarative definition of a training workflow's
fields, events, and validation rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class ProtocolField:
    """A single field within a training protocol.

    Attributes:
        name: Field identifier (e.g. ``"learning_rate"``).
        field_type: Expected type label (``"float"``, ``"int"``, ``"str"``, ...).
        description: Human-readable explanation.
        required: Whether the field must be provided.
        default: Default value when omitted (``None`` means no default).
        min_value: Optional numeric lower bound.
        max_value: Optional numeric upper bound.
        allowed_values: Optional set of permitted values.
        group: Logical grouping label for UI rendering.
    """

    name: str
    field_type: str = "str"
    description: str = ""
    required: bool = True
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Sequence[Any]] = None
    group: str = ""

    def validate(self, value: Any) -> List[str]:
        """Validate a single value against this field's constraints."""
        errors: List[str] = []
        if value is None:
            if self.required and self.default is None:
                errors.append(f"'{self.name}' is required")
            return errors

        if self.allowed_values is not None and value not in self.allowed_values:
            errors.append(
                f"'{self.name}' must be one of {list(self.allowed_values)}, got {value!r}"
            )

        if self.field_type in ("float", "int"):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{self.name}' must be numeric, got {type(value).__name__}")
                return errors
            if self.min_value is not None and numeric < self.min_value:
                errors.append(f"'{self.name}' must be >= {self.min_value}")
            if self.max_value is not None and numeric > self.max_value:
                errors.append(f"'{self.name}' must be <= {self.max_value}")

        return errors


@dataclass(frozen=True)
class ProtocolEventDefinition:
    """Declares an event that occurs during training execution.

    Attributes:
        name: Event identifier (e.g. ``"epoch_start"``).
        description: What triggers this event.
        payload_fields: Names of fields present in the event payload.
        category: Logical grouping (``"lifecycle"``, ``"metric"``, ...).
    """

    name: str
    description: str = ""
    payload_fields: Sequence[str] = field(default_factory=tuple)
    category: str = "lifecycle"


@dataclass
class TrainingProtocol:
    """A complete training protocol definition.

    A protocol describes the parameter space and event lifecycle for a
    category of training workflows.  It acts as a contract: consumers
    validate configurations against its fields, and hosts emit its events
    during training execution.

    Attributes:
        name: Protocol identifier.
        version: Protocol version string.
        description: What this protocol trains.
        fields: Declared configuration fields.
        events: Declared lifecycle events.
        metadata: Arbitrary protocol metadata.
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    fields: List[ProtocolField] = field(default_factory=list)
    events: List[ProtocolEventDefinition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- Field management ------------------------------------------------------

    def add_field(self, field_def: ProtocolField) -> None:
        self.fields.append(field_def)

    def get_field(self, name: str) -> Optional[ProtocolField]:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def required_fields(self) -> List[ProtocolField]:
        return [f for f in self.fields if f.required]

    def fields_by_group(self, group: str) -> List[ProtocolField]:
        return [f for f in self.fields if f.group == group]

    @property
    def groups(self) -> List[str]:
        seen: Set[str] = set()
        result: List[str] = []
        for f in self.fields:
            if f.group and f.group not in seen:
                seen.add(f.group)
                result.append(f.group)
        return result

    # -- Event management ------------------------------------------------------

    def add_event(self, event_def: ProtocolEventDefinition) -> None:
        self.events.append(event_def)

    def get_event(self, name: str) -> Optional[ProtocolEventDefinition]:
        for e in self.events:
            if e.name == name:
                return e
        return None

    def event_names(self) -> List[str]:
        return [e.name for e in self.events]

    # -- Validation ------------------------------------------------------------

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate a configuration dict against this protocol's fields.

        Returns a list of error strings (empty means valid).
        """
        errors: List[str] = []
        seen: Set[str] = set()

        for field_def in self.fields:
            value = config.get(field_def.name)
            seen.add(field_def.name)
            errors.extend(field_def.validate(value))

        unknown = set(config.keys()) - seen
        for key in sorted(unknown):
            errors.append(f"unknown field '{key}'")

        return errors

    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of *config* with missing fields filled from defaults."""
        result = dict(config)
        for field_def in self.fields:
            if field_def.name not in result and field_def.default is not None:
                result[field_def.name] = field_def.default
        return result
