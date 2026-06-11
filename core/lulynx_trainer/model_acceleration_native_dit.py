"""Native DiT acceleration policy helpers for Anima/Newbie."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeDitLowVramRecommendation:
    patch: dict[str, Any]
    notes: tuple[str, ...] = ()


def native_dit_low_vram_patch(model_family: str) -> NativeDitLowVramRecommendation:
    """Return low-VRAM fields consumed by native Anima/Newbie DiT trainers."""

    family = str(model_family or "").strip().lower()
    if family == "anima":
        prefix = "anima"
        runtime_profile = "anima_low_vram"
    elif family == "newbie":
        prefix = "newbie"
        runtime_profile = "standard"
    else:
        return NativeDitLowVramRecommendation(patch={})

    return NativeDitLowVramRecommendation(
        patch={
            "native_runtime_profile": runtime_profile,
            f"{prefix}_block_residency": "streaming_offload",
            f"{prefix}_block_checkpointing": True,
            f"{prefix}_block_checkpointing_mode": "block",
            "sparse_swap_enabled": True,
            "sparse_swap_warm_fraction": 0.25,
            "pcie_transfer_format": "raw_bf16",
            "pcie_delta_cache_enabled": True,
            "pcie_delta_cache_mode": "observe",
        },
        notes=(
            f"{family.title()} low-VRAM uses native DiT streaming residency with block checkpointing.",
            "Block prefetch is left off in the low-VRAM profile; sparse swap is the safer default.",
            "PCIe transfer/cache knobs remain conservative and are consumed by the existing residency runtime.",
        ),
    )


__all__ = ["NativeDitLowVramRecommendation", "native_dit_low_vram_patch"]
