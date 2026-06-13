"""Route family and label lookup tables.

Maps normalized training-type strings to human-readable labels and
structural family identifiers.  Tables are ordered so that more-specific
prefixes are checked first.
"""

from __future__ import annotations

from typing import Mapping

# Ordered prefix→family mapping.  First match wins.
FAMILY_TABLE: tuple[tuple[str, str], ...] = (
    ("newbie",        "newbie"),
    ("anima",         "anima"),
    ("sdxl",          "sdxl"),
    ("flux",          "flux"),
    ("lumina2",       "lumina2"),
    ("lumina",        "lumina"),
    ("qwen-image",    "qwen-image"),
    ("hunyuan-dit",   "hunyuan-dit"),
    ("hunyuan-image", "hunyuan-image"),
    ("sd",            "stable"),
)

# Exact-match label table.  Falls back to title-cased normalization.
LABEL_TABLE: Mapping[str, str] = {
    "newbie-lora":       "Newbie LoRA",
    "anima-lora":        "Anima LoRA",
    "anima-finetune":    "Anima finetune",
    "sdxl-lora":         "SDXL LoRA",
    "sdxl-finetune":     "SDXL finetune",
    "sd-lora":           "Stable LoRA",
    "sd-dreambooth":     "Stable DreamBooth",
    "flux-lora":         "Flux LoRA",
    "flux-finetune":     "Flux finetune",
    "lumina-lora":       "Lumina LoRA",
    "lumina2-lora":      "Lumina2 LoRA",
    "qwen-image-lora":   "Qwen Image LoRA",
    "hunyuan-dit-lora":  "HunyuanDiT LoRA",
    "hunyuan-image-lora": "Hunyuan Image LoRA",
}


def _normalize(raw: str | None) -> str:
    return str(raw or "").strip().lower()


def resolve_family(training_type: str | None) -> str:
    """Return the structural route family for a training type string.

    Walks the FAMILY_TABLE in order and returns the first prefix match.
    Returns ``"generic"`` when nothing matches.
    """
    key = _normalize(training_type)
    for prefix, family in FAMILY_TABLE:
        if key.startswith(prefix):
            return family
    return "generic"


def resolve_label(training_type: str | None) -> str:
    """Return a human-readable label for a training type string.

    Checks the exact-match LABEL_TABLE first, then falls back to a
    title-cased version of the normalized key with hyphens replaced
    by spaces.  Returns ``"Generic Training"`` when the input is empty.
    """
    key = _normalize(training_type)
    if not key:
        return "Generic Training"
    exact = LABEL_TABLE.get(key)
    if exact:
        return exact
    return key.replace("-", " ").title()
