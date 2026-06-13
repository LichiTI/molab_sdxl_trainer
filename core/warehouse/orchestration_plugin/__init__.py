"""
Warehouse orchestration / plugin component package.

Reusable, dependency-light building blocks for plugin systems,
event dispatch, training protocol, and schema management.
"""

from .event_bus import (
    DispatchReport,
    EventBus,
    HandlerConstraints,
    HandlerRegistration,
)
from .hook_contract import (
    CapabilityDefinition,
    CapabilityRegistry,
    HookCatalog,
    HookDefinition,
)
from .plugin_descriptor import (
    PluginDescriptor,
    PluginManifest,
    PluginSignature,
)
from .route_contract import (
    TrainingRouteContract,
    resolve_route_contract,
)
from .training_protocol import (
    ProtocolEventDefinition,
    ProtocolField,
    TrainingProtocol,
)
from .schema_registry import (
    DirectorySchemaSource,
    SchemaEntry,
    SchemaRegistry,
    SchemaSource,
    SchemaTransformer,
)

__all__ = [
    # Event bus
    "DispatchReport",
    "EventBus",
    "HandlerConstraints",
    "HandlerRegistration",
    # Hook / capability contracts
    "CapabilityDefinition",
    "CapabilityRegistry",
    "HookCatalog",
    "HookDefinition",
    # Plugin descriptor
    "PluginDescriptor",
    "PluginManifest",
    "PluginSignature",
    # Route contract
    "TrainingRouteContract",
    "resolve_route_contract",
    # Training protocol
    "ProtocolEventDefinition",
    "ProtocolField",
    "TrainingProtocol",
    # Schema registry
    "DirectorySchemaSource",
    "SchemaEntry",
    "SchemaRegistry",
    "SchemaSource",
    "SchemaTransformer",
]

