"""Plugin manifest, event bus, capability tiers, and policy engine.

Provides a tiered plugin system for extensible training pipelines:
- Manifest schema and JSON loader
- Hook catalog with tier-based capability requirements
- Thread-safe event bus with mutation tracking
- Approval store, enabled-state store, and trust store (JSON-file backed)
- Signature verification (SHA-256 attestation)
- Tier-based policy evaluation engine
- Training event protocol definitions
"""

from lulynx_plugin_core.manifest import PluginManifest, PluginHookBinding, load_manifest
from lulynx_plugin_core.hooks import HookDefinition, get_hook, list_hooks
from lulynx_plugin_core.capabilities import CapabilityDef, get_capability, list_capabilities
from lulynx_plugin_core.event_bus import EventBus, HandlerRegistration
from lulynx_plugin_core.approval import ApprovalStore
from lulynx_plugin_core.enabled import EnabledStore
from lulynx_plugin_core.trust import TrustStore, compute_package_hash
from lulynx_plugin_core.audit import AuditLog
from lulynx_plugin_core.policy import PolicyDecision, evaluate_policy
from lulynx_plugin_core.dispatch_protocol import (
    LossMutationOutcome,
    RouteInfo,
    TrainingSnapshotDescriptor,
    TrainingTypeNormalizer,
    apply_loss_mutation,
    build_training_snapshot,
    describe_training_event,
    list_training_events,
)
from lulynx_plugin_core.orchestrator import (
    PluginDescriptor,
    PluginOrchestrator,
    PluginState,
)
from lulynx_plugin_core.diagnostics import (
    DiagnosticsCollector,
    EventMetrics,
    HandlerMetrics,
)

__all__ = [
    "ApprovalStore",
    "AuditLog",
    "CapabilityDef",
    "EnabledStore",
    "EventBus",
    "HandlerRegistration",
    "HookDefinition",
    "PluginHookBinding",
    "PluginManifest",
    "PolicyDecision",
    "TrustStore",
    "compute_package_hash",
    "evaluate_policy",
    "get_capability",
    "get_hook",
    "list_capabilities",
    "list_hooks",
    "load_manifest",
    # dispatch_protocol
    "LossMutationOutcome",
    "RouteInfo",
    "TrainingSnapshotDescriptor",
    "TrainingTypeNormalizer",
    "apply_loss_mutation",
    "build_training_snapshot",
    "describe_training_event",
    "list_training_events",
    # orchestrator
    "PluginDescriptor",
    "PluginOrchestrator",
    "PluginState",
    # diagnostics
    "DiagnosticsCollector",
    "EventMetrics",
    "HandlerMetrics",
]
