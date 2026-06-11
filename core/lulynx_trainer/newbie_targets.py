"""
Newbie LoRA target module presets.

Maps the ``NewbieLoraTarget`` enum (minimal / balanced / full) to concrete
UNet and text-encoder module name lists that the LoRA injector uses.

These presets are *additive* on top of the SDXL-compatible base targets
already registered in ``model_family.py``.  The idea is:

- **minimal**  – attention projections only (smallest adapter footprint)
- **balanced** – attention projections + feed-forward
- **full**     – attention + feed-forward + input/output projections

Native Newbie checkpoints use a NextDiT-style module tree.  Target names
therefore include the observed transformer leaves (``attention.qkv``,
``attention.out``, ``feed_forward.w*``) rather than only SDXL UNet names.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ── UNet target presets ───────────────────────────────────────────────

_NEWBIE_UNET_MINIMAL: List[str] = [
    "attention.qkv",
    "attention.out",
]

_NEWBIE_UNET_BALANCED: List[str] = [
    "attention.qkv",
    "attention.out",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
]

_NEWBIE_UNET_FULL: List[str] = [
    "attention.qkv",
    "attention.out",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "adaLN_modulation.1",
    "final_layer.linear",
    "time_text_embed.1",
    "t_embedder.mlp.0",
    "t_embedder.mlp.2",
]


# ── Text-encoder target presets ───────────────────────────────────────

_NEWBIE_TE_MINIMAL: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
]

_NEWBIE_TE_BALANCED: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
]

_NEWBIE_TE_FULL: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
    "layer_norm1",
    "layer_norm2",
    "embeddings.position_embedding",
    "embeddings.token_embedding",
]


# ── Preset registry ───────────────────────────────────────────────────

_NEWBIE_PRESETS: Dict[str, Tuple[List[str], List[str]]] = {
    "minimal": (_NEWBIE_UNET_MINIMAL, _NEWBIE_TE_MINIMAL),
    "balanced": (_NEWBIE_UNET_BALANCED, _NEWBIE_TE_BALANCED),
    "full": (_NEWBIE_UNET_FULL, _NEWBIE_TE_FULL),
}


def get_newbie_targets(preset: str = "minimal") -> Tuple[List[str], List[str]]:
    """Return ``(unet_targets, text_encoder_targets)`` for the given preset.

    Falls back to ``"minimal"`` if *preset* is not recognised.
    """
    key = str(preset or "minimal").strip().lower()
    if key not in _NEWBIE_PRESETS:
        key = "minimal"
    return _NEWBIE_PRESETS[key]


def list_newbie_presets() -> List[str]:
    """Return available preset names."""
    return list(_NEWBIE_PRESETS.keys())
