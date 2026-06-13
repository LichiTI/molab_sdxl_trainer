"""Tier-based policy evaluation engine.

Combines capability inference, approval state, trust verification,
and developer-mode bypass into a single policy decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from lulynx_plugin_core.capabilities import get_capability
from lulynx_plugin_core.hooks import get_hook
from lulynx_plugin_core.manifest import PluginManifest


@dataclass(frozen=True)
class PolicyDecision:
    """Immutable result of a policy evaluation."""

    enabled: bool
    required_tier: int
    requires_approval: bool
    requires_trust: bool
    approved: bool
    trust_ok: bool
    reasons: tuple[str, ...]
    unknown_capabilities: tuple[str, ...]
    unknown_hooks: tuple[str, ...]


def infer_tier(manifest: PluginManifest) -> tuple[int, list[str], list[str]]:
    """Infer the maximum required tier from a manifest's capabilities and hooks."""
    tier = 1
    unknown_caps: list[str] = []
    unknown_hooks: list[str] = []
    for cap_name in manifest.capabilities:
        cap = get_capability(cap_name)
        if cap is None:
            unknown_caps.append(cap_name)
        else:
            tier = max(tier, cap.tier)
    for binding in manifest.hooks:
        hook = get_hook(binding.event)
        if hook is None:
            unknown_hooks.append(binding.event)
        else:
            tier = max(tier, hook.tier)
    return tier, sorted(set(unknown_caps)), sorted(set(unknown_hooks))


def evaluate_policy(
    *,
    manifest: PluginManifest,
    approval_result: dict,
    trust_result: dict,
    developer_mode: bool = False,
    activation_enabled: bool = True,
) -> PolicyDecision:
    """Evaluate whether a plugin should be activated.

    Parameters
    ----------
    manifest:
        The plugin's parsed manifest.
    approval_result:
        Output of ``ApprovalStore.check()``.
    trust_result:
        Output of ``TrustStore.evaluate()``.
    developer_mode:
        If True, bypasses approval/trust failures (but still records reasons).
    activation_enabled:
        If False, the plugin is deactivated regardless of other checks.
    """
    tier, unknown_caps, unknown_hooks = infer_tier(manifest)
    reasons: list[str] = []
    if unknown_caps:
        reasons.append("unknown_capabilities")
    if unknown_hooks:
        reasons.append("unknown_hooks")

    requires_approval = tier >= 2
    requires_trust = tier >= 3
    approved = bool(approval_result.get("approved"))
    trust_ok = bool(trust_result.get("ok"))

    if requires_approval and not approved:
        reasons.append(approval_result.get("reason", "approval_missing"))
    if requires_trust and not trust_ok:
        reasons.append(trust_result.get("reason", "trust_failed"))

    if developer_mode:
        if reasons:
            reasons.append("developer_mode_bypass")
        final_enabled = True
    else:
        final_enabled = not reasons

    if not activation_enabled:
        reasons.append("activation_disabled")
        final_enabled = False

    return PolicyDecision(
        enabled=final_enabled,
        required_tier=tier,
        requires_approval=requires_approval,
        requires_trust=requires_trust,
        approved=approved,
        trust_ok=trust_ok,
        reasons=tuple(reasons),
        unknown_capabilities=tuple(unknown_caps),
        unknown_hooks=tuple(unknown_hooks),
    )
