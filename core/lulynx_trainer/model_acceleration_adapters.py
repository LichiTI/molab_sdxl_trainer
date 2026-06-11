"""Adapter-specific acceleration recommendations.

This module stays stdlib-only so policy/preflight code can import it without
touching torch or route runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AdapterAccelerationPatch:
    supported: bool
    patch: dict[str, Any]
    reason: str = ""
    notes: tuple[str, ...] = ()


def lora_fa_policy_patch(model_family: str, config: Mapping[str, Any]) -> AdapterAccelerationPatch:
    """Return route-aware fields for explicit LoRA-FA acceleration opt-in."""

    family = str(model_family or "").strip().lower()
    if family in {"sdxl", "sd15", "anima"}:
        return AdapterAccelerationPatch(
            supported=True,
            patch={"lora_type": "lora_fa", "network_module": "networks.lora_fa", "lora_fa_enabled": True},
            notes=("LoRA-FA uses the networks.lora_fa adapter path on SDXL/SD15/Anima routes.",),
        )
    if family == "newbie":
        return AdapterAccelerationPatch(
            supported=True,
            patch={"newbie_adapter_type": "lora_fa", "network_module": "networks.lora", "lora_fa_enabled": True},
            notes=("Newbie consumes LoRA-FA through newbie_adapter_type plus the LoRA injector marker.",),
        )
    if family == "flux":
        return AdapterAccelerationPatch(
            supported=False,
            patch={},
            reason="Flux LoRA preview currently supports network_module=networks.lora only.",
        )
    return AdapterAccelerationPatch(
        supported=False,
        patch={},
        reason="LoRA-FA route support is not known for this model family.",
    )


__all__ = ["AdapterAccelerationPatch", "lora_fa_policy_patch"]
