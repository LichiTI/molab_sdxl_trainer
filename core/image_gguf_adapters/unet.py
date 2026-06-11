"""UNet-family image GGUF probe adapters."""

from __future__ import annotations

from pathlib import Path

from ..image_gguf_contracts import ImageGGUFComponent, ImageGGUFManifest, TensorInfo
from .common import build_prefix_probe_manifest


COMMON_UNET_REQUIRED_TENSORS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.time_embed.0.weight",
    "model.diffusion_model.middle_block.0.in_layers.0.weight",
    "model.diffusion_model.output_blocks.0.0.in_layers.0.weight",
    "model.diffusion_model.out.2.weight",
]
COMMON_UNET_REQUIRED_PREFIXES = ["model.diffusion_model."]
COMMON_UNET_MARKERS = ["model.diffusion_model.input_blocks.", "model.diffusion_model.output_blocks."]
SD15_HINTS = {"sd15", "sd1_5", "sd_1_5", "sd15_unet", "stable_diffusion_v1", "stable_diffusion_1_5"}
SDXL_HINTS = {"sdxl", "sdxl_unet", "stable_diffusion_xl", "stable_diffusion_xl_base"}


def _hint(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(".", "_")


def _has_prefix(tensors: dict[str, TensorInfo], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in tensors)


def _unet_score(tensors: dict[str, TensorInfo], *, hint_score: int = 0, marker_score: int = 0) -> int:
    score = hint_score + marker_score
    score += sum(8 for marker in COMMON_UNET_MARKERS if _has_prefix(tensors, marker))
    score += sum(4 for key in COMMON_UNET_REQUIRED_TENSORS if key in tensors)
    return score


class SD15UNetProbeAdapter:
    adapter_id = "sd15_unet_probe_v1"
    component = ImageGGUFComponent.SD15_UNET
    family = "sd15_unet"

    required_tensors = COMMON_UNET_REQUIRED_TENSORS
    required_prefixes = COMMON_UNET_REQUIRED_PREFIXES
    optional_prefixes = [
        "cond_stage_model.transformer.",
        "first_stage_model.encoder.",
        "first_stage_model.decoder.",
        "first_stage_model.quant_conv.",
        "first_stage_model.post_quant_conv.",
    ]

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = _hint(family_hint)
        has_sd15_marker = _has_prefix(tensors, "cond_stage_model.")
        has_sdxl_marker = _has_prefix(tensors, "conditioner.embedders.")
        if has_sdxl_marker:
            return 0
        if hint not in SD15_HINTS and not has_sd15_marker:
            return 0
        return _unet_score(
            tensors,
            hint_score=70 if hint in SD15_HINTS else 0,
            marker_score=50 if has_sd15_marker else 0,
        )

    def build_manifest(self, source_path: Path, tensors: dict[str, TensorInfo]) -> ImageGGUFManifest:
        return build_prefix_probe_manifest(
            source_path=source_path,
            adapter_id=self.adapter_id,
            component=self.component.value,
            family=self.family,
            tensors=tensors,
            required_tensors=list(self.required_tensors),
            required_prefixes=list(self.required_prefixes),
            optional_prefixes=list(self.optional_prefixes),
            notes=[
                "Phase 2 probe only; no GGUF file is written.",
                "SD1.5 UNet probe accepts original checkpoint bundles with cond_stage_model/first_stage_model prefixes.",
            ],
        )


class SDXLUNetProbeAdapter:
    adapter_id = "sdxl_unet_probe_v1"
    component = ImageGGUFComponent.SDXL_UNET
    family = "sdxl_unet"

    required_tensors = COMMON_UNET_REQUIRED_TENSORS
    required_prefixes = COMMON_UNET_REQUIRED_PREFIXES
    optional_prefixes = [
        "conditioner.embedders.",
        "first_stage_model.encoder.",
        "first_stage_model.decoder.",
        "first_stage_model.quant_conv.",
        "first_stage_model.post_quant_conv.",
    ]
    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = _hint(family_hint)
        has_sd15_marker = _has_prefix(tensors, "cond_stage_model.")
        has_sdxl_marker = _has_prefix(tensors, "conditioner.embedders.")
        if has_sd15_marker:
            return 0
        if hint not in SDXL_HINTS and not has_sdxl_marker:
            return 0
        return _unet_score(
            tensors,
            hint_score=70 if hint in SDXL_HINTS else 0,
            marker_score=55 if has_sdxl_marker else 0,
        )

    def build_manifest(self, source_path: Path, tensors: dict[str, TensorInfo]) -> ImageGGUFManifest:
        return build_prefix_probe_manifest(
            source_path=source_path,
            adapter_id=self.adapter_id,
            component=self.component.value,
            family=self.family,
            tensors=tensors,
            required_tensors=list(self.required_tensors),
            required_prefixes=list(self.required_prefixes),
            optional_prefixes=list(self.optional_prefixes),
            notes=[
                "Phase 2 probe only; no GGUF file is written.",
                "SDXL UNet probe accepts full checkpoint bundles and counts known conditioner/VAE prefixes.",
            ],
        )


__all__ = ["SD15UNetProbeAdapter", "SDXLUNetProbeAdapter"]
