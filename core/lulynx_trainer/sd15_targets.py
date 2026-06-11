"""
SD1.5 LoRA target module definitions.

Provides structured, block-aware target resolution for SD1.5 UNet and
text encoder LoRA injection.  The canonical flat target lists live in
``model_family.py``; this module adds block-granularity for features
like per-block learning-rate scaling and selective layer freezing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


# ── UNet block taxonomy ────────────────────────────────────────────────
# SD1.5 UNet2DConditionModel layout (diffusers naming):
#
#   down_blocks.{0..3}.resnets.{0..1}       -- ResNet convolutions
#   down_blocks.{0..3}.attentions.{0..1}    -- Transformer2DModel blocks
#   mid_block.resnets.{0,1}                 -- Mid ResNet
#   mid_block.attentions.0                  -- Mid Transformer
#   up_blocks.{0..3}.resnets.{0..2}         -- Up ResNet (block 0 has 3)
#   up_blocks.{0..3}.attentions.{0..2}      -- Up Transformer
#   down_blocks.{0..3}.downsamplers.0       -- Downsample conv (blocks 1,2,3)
#   up_blocks.{0..3}.upsamplers.0           -- Upsample conv (blocks 1,2,3)
#
# Each Transformer2DModel contains attention blocks with linear projections:
#   to_q, to_k, to_v, to_out.0, proj_in, proj_out, ff.net.0.proj, ff.net.2

# Target module names inside each Transformer2DModel / attention block
_TRANSFORMER_LINEAR_TARGETS: List[str] = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
]

_TRANSFORMER_FF_TARGETS: List[str] = [
    "ff.net.0.proj",
    "ff.net.2",
]

_TRANSFORMER_PROJ_TARGETS: List[str] = [
    "proj_in",
    "proj_out",
]

# All linear targets within a transformer block
ALL_TRANSFORMER_TARGETS: List[str] = (
    _TRANSFORMER_LINEAR_TARGETS + _TRANSFORMER_FF_TARGETS + _TRANSFORMER_PROJ_TARGETS
)

# Text encoder (CLIP-L) target names
TE_TARGETS: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
]

# Flat list for ``model_family.py`` compatibility (mirrors _SD15_UNET_TARGETS)
UNET_TARGET_MODULES: List[str] = list(ALL_TRANSFORMER_TARGETS)
TEXT_ENCODER_TARGET_MODULES: List[str] = list(TE_TARGETS)


# ── block grouping ─────────────────────────────────────────────────────

def _down_block_prefix(idx: int) -> str:
    return f"down_blocks.{idx}"


def _up_block_prefix(idx: int) -> str:
    return f"up_blocks.{idx}"


@dataclass(frozen=True)
class SD15BlockTargets:
    """Groups UNet target module paths by structural block.

    Each set contains full dotted module paths (relative to the UNet root)
    whose leaf name matches a transformer linear target.  These are useful
    for per-block learning-rate scaling or selective freezing.
    """
    down_blocks: Dict[int, Set[str]] = field(default_factory=dict)
    mid_block: Set[str] = field(default_factory=set)
    up_blocks: Dict[int, Set[str]] = field(default_factory=dict)

    def all_paths(self) -> Set[str]:
        """Return the union of all block-level target paths."""
        paths: Set[str] = set()
        for s in self.down_blocks.values():
            paths |= s
        paths |= self.mid_block
        for s in self.up_blocks.values():
            paths |= s
        return paths


def enumerate_attention_module_names(model) -> SD15BlockTargets:
    """Walk a UNet module and classify attention layers by block.

    Returns an ``SD15BlockTargets`` whose sets contain the *module names*
    (as used by ``model.named_modules()``) of every linear layer that
    matches the SD1.5 transformer target list.

    This is the authoritative way to get per-block targets when the flat
    target list from ``model_family.py`` is insufficient.
    """
    targets_set = set(ALL_TRANSFORMER_TARGETS)
    result = SD15BlockTargets()

    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if leaf not in targets_set:
            continue
        # Not a linear leaf -- check the parent name for the actual linear
        # name.  named_modules yields e.g. "down_blocks.0.attentions.0.transformer_blocks.0.to_q"
        # We want to classify by the block prefix.
        if name.startswith("down_blocks."):
            parts = name.split(".")
            try:
                idx = int(parts[1])
            except (ValueError, IndexError):
                continue
            result.down_blocks.setdefault(idx, set()).add(name)
        elif name.startswith("mid_block."):
            result.mid_block.add(name)
        elif name.startswith("up_blocks."):
            parts = name.split(".")
            try:
                idx = int(parts[1])
            except (ValueError, IndexError):
                continue
            result.up_blocks.setdefault(idx, set()).add(name)

    return result


# ── per-block LR scale presets ─────────────────────────────────────────

@dataclass(frozen=True)
class SD15BlockLRScale:
    """Per-block learning-rate multiplier for SD1.5 UNet.

    Keys: ``down_0`` .. ``down_3``, ``mid``, ``up_0`` .. ``up_3``.
    Values are floats; 1.0 = default LR, 0.0 = frozen.
    """
    down_0: float = 1.0
    down_1: float = 1.0
    down_2: float = 1.0
    down_3: float = 1.0
    mid: float = 1.0
    up_0: float = 1.0
    up_1: float = 1.0
    up_2: float = 1.0
    up_3: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "down_blocks.0": self.down_0,
            "down_blocks.1": self.down_1,
            "down_blocks.2": self.down_2,
            "down_blocks.3": self.down_3,
            "mid_block": self.mid,
            "up_blocks.0": self.up_0,
            "up_blocks.1": self.up_1,
            "up_blocks.2": self.up_2,
            "up_blocks.3": self.up_3,
        }


# Common presets
DEFAULT_LR_SCALE = SD15BlockLRScale()

DOWN_HEAVY_LR_SCALE = SD15BlockLRScale(
    down_0=1.5, down_1=1.5, down_2=1.2, down_3=1.0,
    mid=0.8,
    up_0=0.5, up_1=0.5, up_2=0.5, up_3=0.5,
)

UP_HEAVY_LR_SCALE = SD15BlockLRScale(
    down_0=0.5, down_1=0.5, down_2=0.5, down_3=0.5,
    mid=0.8,
    up_0=1.2, up_1=1.2, up_2=1.5, up_3=1.5,
)
