"""
Y-1 - Plugin Event Bus & Hook System - Interface Contract

Defines the protocol surface for a capability-gated, tiered event bus that
plugins can subscribe to for training lifecycle events.

This module contains NO behavioral implementation.  It specifies:
- What hook events exist and what tier/capability they require.
- What payloads events carry (field definitions only).
- What the event bus and plugin runtime must look like structurally.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# Enums - Shared vocabulary
# ---------------------------------------------------------------------------

class CapabilityTier(enum.IntEnum):
    """Safety tier for plugin capabilities.

    Tier 1 (OBSERVE): Read-only observation (metrics, config snapshots).
    Tier 2 (HOOK):    Observe training loop internals (tensor hooks).
    Tier 3 (MUTATE):  Mutate training state (loss, scheduler, optimizer).
    """
    OBSERVE = 1
    HOOK = 2
    MUTATE = 3


class PayloadMutability(enum.Enum):
    """Whether an event payload may be mutated by a handler."""
    READ_ONLY = "read_only"
    MUTABLE = "mutable"
    EXCLUSIVE_MUTABLE = "exclusive_mutable"


# ---------------------------------------------------------------------------
# Data Models - Hook & Capability definitions (frozen, no behavior)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HookDefinition:
    """Static description of a subscribable training-lifecycle event."""
    event: str
    required_capability: str
    tier: CapabilityTier
    payload_mutability: PayloadMutability
    exclusive: bool
    default_priority: int
    description: str


@dataclass(frozen=True)
class CapabilityDefinition:
    """A named permission that a plugin may be granted."""
    name: str
    tier: CapabilityTier
    description: str


@dataclass(frozen=True)
class FieldDefinition:
    """Schema-level description of one field in a training event payload."""
    name: str
    field_type: str  # "string", "integer", "number", "boolean", "object", "number[]"
    required: bool
    description: str


@dataclass(frozen=True)
class EventProtocolSpec:
    """Full schema for a training event protocol (version + field list)."""
    event: str
    protocol_version: str
    description: str
    fields: tuple[FieldDefinition, ...]
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Dispatch result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandlerReport:
    """Per-handler result within a dispatch cycle."""
    plugin_id: str
    handler_name: str
    priority: int
    mutable: bool
    status: str  # "ok", "skipped", "error"
    duration_ms: float
    slow: bool
    error: str | None = None


@dataclass(frozen=True)
class DispatchReport:
    """Summary of one event dispatch across all registered handlers."""
    event: str
    handled: int
    errors: tuple[HandlerReport, ...]
    skipped: tuple[HandlerReport, ...]
    exclusive_conflict: bool
    mutated: bool
    elapsed_ms: float
    slow_handlers: int
    handlers: tuple[HandlerReport, ...]


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandlerRegistration:
    """A single handler registration request."""
    plugin_id: str
    event: str
    handler_name: str
    handler: Callable[[Any], Any]
    priority: int = 0
    mutable: bool = False
    predicate: Callable[[Any], bool] | None = None
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Protocols - Structural interfaces for implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class HookCatalogProtocol(Protocol):
    """Queryable catalog of all known hook event definitions."""

    def get_hook(self, event: str) -> HookDefinition | None: ...

    def list_hooks(self) -> Sequence[HookDefinition]: ...


@runtime_checkable
class CapabilityCatalogProtocol(Protocol):
    """Queryable catalog of all known capability definitions."""

    def get_capability(self, name: str) -> CapabilityDefinition | None: ...

    def list_capabilities(self) -> Sequence[CapabilityDefinition]: ...


@runtime_checkable
class EventBusProtocol(Protocol):
    """Thread-safe event dispatch bus with tiered capability gating.

    Implementations MUST:
    - Be safe for concurrent register/emit from multiple threads.
    - Sort handlers by priority (descending) before dispatch.
    - Freeze read-only payloads before passing to non-mutable handlers.
    - Respect exclusive flag (only first handler runs for exclusive hooks).
    - Return a DispatchReport summarizing what happened.
    """

    def register_handler(self, registration: HandlerRegistration) -> None:
        """Register a handler for an event. Thread-safe."""
        ...

    def has_handlers(self, event: str) -> bool:
        """Return True if any handler is registered for the event."""
        ...

    def emit(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        slow_handler_threshold_ms: float = 25.0,
    ) -> DispatchReport:
        """Dispatch an event to all registered handlers. Thread-safe."""
        ...

    def clear(self) -> None:
        """Remove all handler registrations. Thread-safe."""
        ...


@runtime_checkable
class PluginRuntimeProtocol(Protocol):
    """High-level runtime that manages plugin lifecycle and event dispatch.

    Implementations combine a HookCatalog, CapabilityCatalog, and EventBus
    with plugin manifest loading, approval/enabled state, and audit logging.
    """

    @property
    def hook_catalog(self) -> HookCatalogProtocol: ...

    @property
    def capability_catalog(self) -> CapabilityCatalogProtocol: ...

    @property
    def event_bus(self) -> EventBusProtocol: ...

    def has_handlers(self, event: str) -> bool: ...

    def emit_event(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        source: str = "",
    ) -> DispatchReport: ...

    def emit_mutation_event(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        source: str = "",
    ) -> DispatchReport:
        """Emit a mutation-allowed event. Returns report with result_payload."""
        ...


# ---------------------------------------------------------------------------
# Standard hook event names (convention constants)
# ---------------------------------------------------------------------------

HOOK_EVENT_APP_START = "on_app_start"
HOOK_EVENT_CONFIG_LOADED = "on_config_loaded"
HOOK_EVENT_DATASET_PREPARED = "on_dataset_prepared"
HOOK_EVENT_TRAIN_LAUNCH = "on_train_launch"
HOOK_EVENT_TRAIN_COMPLETE = "on_train_complete"
HOOK_EVENT_BEFORE_FORWARD = "before_forward"
HOOK_EVENT_AFTER_LOSS = "after_loss"
HOOK_EVENT_AFTER_BACKWARD = "after_backward"
HOOK_EVENT_BEFORE_OPTIMIZER_STEP = "before_optimizer_step"
HOOK_EVENT_AFTER_OPTIMIZER_STEP = "after_optimizer_step"
HOOK_EVENT_MODIFY_LOSS = "modify_loss"
HOOK_EVENT_MODIFY_SCHEDULER_STEP = "modify_scheduler_step"
HOOK_EVENT_MODIFY_OPTIMIZER_STEP = "modify_optimizer_step"

# Well-known capability names
CAP_READ_RUNTIME_STATS = "read_runtime_stats"
CAP_READ_STEP_METRICS = "read_step_metrics"
CAP_READ_DATASET_META = "read_dataset_meta"
CAP_READ_TRAIN_CONFIG = "read_train_config"
CAP_HOOK_BEFORE_FORWARD = "hook_before_forward"
CAP_HOOK_AFTER_LOSS = "hook_after_loss"
CAP_HOOK_AFTER_BACKWARD = "hook_after_backward"
CAP_HOOK_BEFORE_OPTIMIZER_STEP = "hook_before_optimizer_step"
CAP_HOOK_AFTER_OPTIMIZER_STEP = "hook_after_optimizer_step"
CAP_WRITE_AUX_LOGS = "write_aux_logs"
CAP_MODIFY_LOSS = "modify_loss"
CAP_MODIFY_SCHEDULER_STEP = "modify_scheduler_step"
CAP_MODIFY_OPTIMIZER_STEP = "modify_optimizer_step"
CAP_REPLACE_TRAINING_COMPONENT = "replace_training_component"
CAP_WRITE_CHECKPOINT = "write_checkpoint"
CAP_NETWORK_ACCESS = "network_access"
