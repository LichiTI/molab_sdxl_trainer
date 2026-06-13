"""Hook event catalog.

Defines the set of recognized hook events, their tier requirements,
mutation semantics, and exclusivity rules.  The catalog is a static
frozen table — no runtime registration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HookDefinition:
    """Descriptor for a single hook event type."""

    event: str
    required_capability: str
    tier: int
    read_only: bool
    allows_mutation: bool
    exclusive: bool
    default_priority: int
    description: str


# Catalog of recognized hooks, ordered by tier then event name.
_HOOKS: tuple[HookDefinition, ...] = (
    # --- Tier 1: observe-only lifecycle events ---
    HookDefinition("on_app_start",          "read_runtime_stats",  1, True,  False, False, 0,    "Application startup completed."),
    HookDefinition("on_config_loaded",      "read_train_config",   1, True,  False, False, 0,    "Training config parsed and normalized."),
    HookDefinition("on_dataset_prepared",   "read_dataset_meta",   1, True,  False, False, 0,    "Dataset preflight completed."),
    HookDefinition("on_train_launch",       "read_train_config",   1, True,  False, False, 0,    "Training process launched."),
    HookDefinition("on_train_complete",     "read_step_metrics",   1, True,  False, False, 0,    "Training completed with final status."),
    # --- Tier 2: per-step observe hooks ---
    HookDefinition("before_forward",        "hook_before_forward",        2, True,  False, False, 100, "Observe before model forward."),
    HookDefinition("after_loss",            "hook_after_loss",            2, True,  False, False, 100, "Observe after loss computed."),
    HookDefinition("after_backward",        "hook_after_backward",        2, True,  False, False, 100, "Observe after backward."),
    HookDefinition("before_optimizer_step", "hook_before_optimizer_step", 2, True,  False, False, 100, "Observe before optimizer step."),
    HookDefinition("after_optimizer_step",  "hook_after_optimizer_step",  2, True,  False, False, 100, "Observe after optimizer step."),
    # --- Tier 3: mutation hooks ---
    HookDefinition("modify_loss",            "modify_loss",            3, False, True, True, 1000, "Mutate final loss before backward."),
    HookDefinition("modify_scheduler_step",  "modify_scheduler_step",  3, False, True, True, 1000, "Replace scheduler stepping behavior."),
    HookDefinition("modify_optimizer_step",  "modify_optimizer_step",  3, False, True, True, 1000, "Replace optimizer stepping behavior."),
)

_HOOK_INDEX: dict[str, HookDefinition] = {h.event: h for h in _HOOKS}


def get_hook(event: str) -> HookDefinition | None:
    """Look up a hook definition by event name."""
    return _HOOK_INDEX.get(str(event or "").strip())


def list_hooks() -> list[dict]:
    """Return all hook definitions as plain dicts."""
    return [
        {
            "event": h.event,
            "required_capability": h.required_capability,
            "tier": h.tier,
            "read_only": h.read_only,
            "allows_mutation": h.allows_mutation,
            "exclusive": h.exclusive,
            "default_priority": h.default_priority,
            "description": h.description,
        }
        for h in _HOOKS
    ]
