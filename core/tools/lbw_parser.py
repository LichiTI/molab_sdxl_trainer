# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Per-block weight (LBW) parser for LoRA merge operations."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# SDXL block layout: 26 positions
SDXL_BLOCKS: List[str] = [
    "BASE",
    "IN00", "IN01", "IN02", "IN03", "IN04", "IN05",
    "IN06", "IN07", "IN08", "IN09", "IN10", "IN11",
    "M00",
    "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05",
    "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11",
]

# SD1.5 block layout: 17 positions
SD15_BLOCKS: List[str] = [
    "BASE",
    "IN01", "IN02", "IN04", "IN05", "IN07", "IN08",
    "M00",
    "OUT03", "OUT04", "OUT05", "OUT06", "OUT07",
    "OUT08", "OUT09", "OUT10", "OUT11",
]

def parse_lbw_weights(
    weight_string: str,
    model_type: str = "sdxl",
) -> Dict[str, float]:
    """Parse a comma-separated LBW weight string into block→weight mapping.

    Args:
        weight_string: Comma-separated float values, e.g. "1,0.5,0.5,..."
        model_type: "sdxl" (26 weights) or "sd15" (17 weights)

    Returns:
        Dict mapping block ID (e.g. "IN00", "M00", "OUT05") to weight float.

    Raises:
        ValueError: If weight count doesn't match expected block count.
    """
    blocks = SDXL_BLOCKS if model_type.lower() in ("sdxl", "xl") else SD15_BLOCKS

    parts = [p.strip() for p in weight_string.split(",") if p.strip()]
    if len(parts) != len(blocks):
        raise ValueError(
            f"Expected {len(blocks)} weights for {model_type}, got {len(parts)}. "
            f"Format: {','.join(blocks)}"
        )

    result = {}
    for block_id, val_str in zip(blocks, parts):
        try:
            result[block_id] = float(val_str)
        except ValueError:
            raise ValueError(f"Invalid weight value '{val_str}' for block {block_id}")

    return result


def apply_lbw_to_layer_key(
    key: str,
    lbw_weights: Dict[str, float],
    default_weight: float = 1.0,
) -> float:
    """Given a LoRA layer key, return the LBW weight for its block.

    Uses block ID detection from the key name (e.g. "lora_unet_input_blocks_3_" → IN03).
    """
    block_id = _detect_block_id(key)
    if block_id and block_id in lbw_weights:
        return lbw_weights[block_id]
    # TE layers map to BASE
    if "text" in key.lower() or "te_" in key.lower():
        return lbw_weights.get("BASE", default_weight)
    return default_weight


def _detect_block_id(key: str) -> Optional[str]:
    """Detect the block ID from a LoRA state dict key."""
    key_lower = key.lower()

    # UNet input blocks: "input_blocks_N" or "down_blocks_N"
    import re

    # Diffusers-style: "down_blocks.N" or "up_blocks.N" or "mid_block"
    m = re.search(r"down_blocks?[._](\d+)", key_lower)
    if m:
        idx = int(m.group(1))
        return f"IN{idx:02d}" if idx < 12 else None

    m = re.search(r"up_blocks?[._](\d+)", key_lower)
    if m:
        idx = int(m.group(1))
        return f"OUT{idx:02d}" if idx < 12 else None

    if "mid_block" in key_lower or "middle_block" in key_lower:
        return "M00"

    # LDM-style: "input_blocks.N" or "output_blocks.N"
    m = re.search(r"input_blocks?[._](\d+)", key_lower)
    if m:
        idx = int(m.group(1))
        return f"IN{idx:02d}" if idx < 12 else None

    m = re.search(r"output_blocks?[._](\d+)", key_lower)
    if m:
        idx = int(m.group(1))
        return f"OUT{idx:02d}" if idx < 12 else None

    if "middle_block" in key_lower or "mid_block" in key_lower:
        return "M00"

    return None
