"""Anima native DiT LoRA target definitions.

Anima preview2 checkpoints use a ``net.*`` DiT backbone, not an SDXL
UNet.  These target names intentionally follow the observed native
checkpoint layout:

``net.blocks.N.self_attn.{q,k,v}_proj/output_proj``,
``net.blocks.N.cross_attn.*``, ``net.blocks.N.mlp.layer*``,
``net.blocks.N.adaln_modulation_*``, and ``net.llm_adapter.*``.
"""

from __future__ import annotations

from typing import Dict, List


# Module suffixes inside each ``net.blocks.N`` DiT block.
ANIMA_DIT_SELF_ATTN_TARGETS: List[str] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
]

ANIMA_DIT_CROSS_ATTN_TARGETS: List[str] = [
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.output_proj",
]

ANIMA_DIT_MLP_TARGETS: List[str] = [
    "mlp.layer1",
    "mlp.layer2",
]

ANIMA_DIT_ADALN_TARGETS: List[str] = [
    "adaln_modulation_self_attn.1",
    "adaln_modulation_self_attn.2",
    "adaln_modulation_cross_attn.1",
    "adaln_modulation_cross_attn.2",
    "adaln_modulation_mlp.1",
    "adaln_modulation_mlp.2",
    "final_layer.adaln_modulation.1",
    "final_layer.adaln_modulation.2",
]


# LLM adapter target suffixes.  The adapter sits under ``net.llm_adapter`` and
# should be treated separately from the DiT block projections.
ANIMA_LLM_ADAPTER_TARGETS: List[str] = [
    "llm_adapter",
    "llm_adapter.blocks",
    "llm_adapter.proj",
    "llm_adapter.norm",
]


ANIMA_DIT_BLOCK_TARGETS: List[str] = (
    ANIMA_DIT_SELF_ATTN_TARGETS
    + ANIMA_DIT_CROSS_ATTN_TARGETS
    + ANIMA_DIT_MLP_TARGETS
    + ANIMA_DIT_ADALN_TARGETS
)


ANIMA_DIT_TARGET_GROUPS: Dict[str, List[str]] = {
    "self_attn": ANIMA_DIT_SELF_ATTN_TARGETS,
    "cross_attn": ANIMA_DIT_CROSS_ATTN_TARGETS,
    "mlp": ANIMA_DIT_MLP_TARGETS,
    "adaln_modulation": ANIMA_DIT_ADALN_TARGETS,
    "llm_adapter": ANIMA_LLM_ADAPTER_TARGETS,
}


ANIMA_DIT_TARGET_PATTERNS: List[str] = [
    "net.blocks.*.self_attn.q_proj",
    "net.blocks.*.self_attn.k_proj",
    "net.blocks.*.self_attn.v_proj",
    "net.blocks.*.self_attn.output_proj",
    "net.blocks.*.cross_attn.q_proj",
    "net.blocks.*.cross_attn.k_proj",
    "net.blocks.*.cross_attn.v_proj",
    "net.blocks.*.cross_attn.output_proj",
    "net.blocks.*.mlp.layer1",
    "net.blocks.*.mlp.layer2",
    "net.blocks.*.adaln_modulation_self_attn",
    "net.blocks.*.adaln_modulation_cross_attn",
    "net.blocks.*.adaln_modulation_mlp",
    "net.llm_adapter.*",
]


# Compatibility aliases for callers that have not yet been renamed.
# They now point at the native DiT target contract, not SDXL UNet targets.
ANIMA_TRANSFORMER_TARGETS: List[str] = ANIMA_DIT_BLOCK_TARGETS + ANIMA_LLM_ADAPTER_TARGETS
ANIMA_UNET_TARGETS: List[str] = ANIMA_TRANSFORMER_TARGETS


# Prompt encoders remain separate from the native DiT backbone.  These lists are
# only for future text-side adapter injection and do not imply training readiness.
ANIMA_QWEN3_TE_TARGETS: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

ANIMA_T5_TE_TARGETS: List[str] = [
    "q",
    "k",
    "v",
    "o",
    "wi_0",
    "wi_1",
    "wo",
]

ANIMA_CLIP_TE_TARGETS: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
]


def get_anima_dit_targets(include_llm_adapter: bool = True) -> List[str]:
    """Return native Anima DiT target suffixes."""
    targets = list(ANIMA_DIT_BLOCK_TARGETS)
    if include_llm_adapter:
        targets.extend(ANIMA_LLM_ADAPTER_TARGETS)
    return targets


def get_anima_target_groups(include_llm_adapter: bool = True) -> Dict[str, List[str]]:
    """Return grouped native Anima target suffixes."""
    groups = {
        name: list(targets)
        for name, targets in ANIMA_DIT_TARGET_GROUPS.items()
        if include_llm_adapter or name != "llm_adapter"
    }
    return groups


def get_anima_text_encoder_targets(encoder_type: str = "qwen3") -> List[str]:
    """Return LoRA targets for a future Anima prompt encoder route."""
    mapping = {
        "clip": ANIMA_CLIP_TE_TARGETS,
        "qwen3": ANIMA_QWEN3_TE_TARGETS,
        "t5": ANIMA_T5_TE_TARGETS,
    }
    targets = mapping.get(encoder_type)
    if targets is None:
        raise ValueError(
            f"Unknown Anima encoder type {encoder_type!r}. "
            f"Expected one of: {', '.join(mapping)}"
        )
    return list(targets)
