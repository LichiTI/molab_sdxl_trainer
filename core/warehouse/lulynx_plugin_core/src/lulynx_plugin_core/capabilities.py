"""Capability tier definitions.

Each capability maps to a security tier (1–3) that controls whether
user approval or trust verification is required before a plugin can
exercise it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityDef:
    """Descriptor for a single capability name and its tier."""

    name: str
    tier: int
    description: str


_CAPABILITIES: tuple[CapabilityDef, ...] = (
    # Tier 1: safe read-only
    CapabilityDef("read_runtime_stats",  1, "Read runtime summary and backend status."),
    CapabilityDef("read_step_metrics",   1, "Read train-step level metrics snapshots."),
    CapabilityDef("read_dataset_meta",   1, "Read dataset metadata and counts."),
    CapabilityDef("read_train_config",   1, "Read normalized training config snapshots."),
    CapabilityDef("optimizer_provider:pytorch_optimizer", 1, "Expose the local pytorch-optimizer package as an optimizer provider."),
    # Tier 2: observe + write aux
    CapabilityDef("hook_before_forward",        2, "Observe before forward pass hook."),
    CapabilityDef("hook_after_loss",            2, "Observe after loss computation hook."),
    CapabilityDef("hook_after_backward",        2, "Observe after backward pass hook."),
    CapabilityDef("hook_before_optimizer_step", 2, "Observe before optimizer step hook."),
    CapabilityDef("hook_after_optimizer_step",  2, "Observe after optimizer step hook."),
    CapabilityDef("write_aux_logs",             2, "Write plugin-owned logs and metrics artifacts."),
    # Tier 3: mutation / high-risk
    CapabilityDef("modify_loss",                 3, "Modify final loss tensor before backward."),
    CapabilityDef("modify_scheduler_step",       3, "Modify scheduler stepping behavior."),
    CapabilityDef("modify_optimizer_step",       3, "Modify optimizer stepping behavior."),
    CapabilityDef("replace_training_component",  3, "Replace model/scheduler/optimizer components."),
    CapabilityDef("write_checkpoint",            3, "Write or mutate checkpoint artifacts."),
    CapabilityDef("network_access",              3, "Open outbound network requests during training."),
)

_CAP_INDEX: dict[str, CapabilityDef] = {c.name: c for c in _CAPABILITIES}


def get_capability(name: str) -> CapabilityDef | None:
    """Look up a capability definition by name."""
    return _CAP_INDEX.get(str(name or "").strip())


def list_capabilities() -> list[dict]:
    """Return all capability definitions as plain dicts."""
    return [
        {"name": c.name, "tier": c.tier, "description": c.description}
        for c in _CAPABILITIES
    ]
