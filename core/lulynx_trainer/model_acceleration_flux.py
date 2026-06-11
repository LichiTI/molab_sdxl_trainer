"""Flux-specific acceleration policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FluxLowVramRecommendation:
    patch: dict[str, Any]
    notes: tuple[str, ...] = ()


def flux_low_vram_patch() -> FluxLowVramRecommendation:
    """Return low-VRAM fields consumed by the Flux LoRA preview trainer."""

    return FluxLowVramRecommendation(
        patch={
            "te_vae_offload_strategy": "aggressive",
            "enable_sequential_cpu_offload": True,
            "flux_transformer_offload": "aggressive",
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
        },
        notes=(
            "Flux low-VRAM uses component offload plus transformer streaming offload in the preview trainer.",
            "Generic module_offload remains disabled for Flux because the preview trainer does not consume it.",
        ),
    )


__all__ = ["FluxLowVramRecommendation", "flux_low_vram_patch"]
