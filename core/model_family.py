# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Lightweight model family capability registry for launcher contracts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelFamily:
    unet_target_modules: List[str]
    text_encoder_target_modules: List[str]
    has_dual_text_encoders: bool = False
    uses_pooled_prompt_embeds: bool = False
    uses_time_ids: bool = False
    uses_clip_pooled_features: bool = False
    default_sampler_pipeline: Optional[str] = None
    is_stub: bool = False
    latent_channels: int = 4
    vae_scaling_factor: float = 0.13025
    supported_training_types: tuple[str, ...] = ("lora",)
    supported_adapters: tuple[str, ...] = ("lora", "dora", "lokr", "loha")
    default_profiles: tuple[str, ...] = ("default",)
    resolution_constraints: Dict[str, int] = field(default_factory=dict)
    export_compatibility: tuple[str, ...] = ("kohya-safetensors",)
    runtime_requirements: tuple[str, ...] = ()
    unsupported_features: tuple[str, ...] = ()


_SDXL_UNET_TARGETS: List[str] = [
    "to_q", "to_k", "to_v", "to_out.0",
    "proj_in", "proj_out",
    "ff.net.0.proj", "ff.net.2",
]

_SDXL_TE_TARGETS: List[str] = [
    "q_proj", "k_proj", "v_proj", "out_proj",
    "fc1", "fc2",
]

_SD15_UNET_TARGETS: List[str] = list(_SDXL_UNET_TARGETS)
_SD15_TE_TARGETS: List[str] = list(_SDXL_TE_TARGETS)
_FLUX_UNET_TARGETS: List[str] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "proj_mlp",
    "proj_out",
]
_FLUX_TE_TARGETS: List[str] = list(_SDXL_TE_TARGETS)
_ANIMA_DIT_TARGETS: List[str] = list(_SDXL_UNET_TARGETS)
_ANIMA_TE_TARGETS: List[str] = list(_SDXL_TE_TARGETS)
_NEWBIE_UNET_TARGETS: List[str] = ["attention.qkv", "attention.out"]


_MODEL_FAMILIES: Dict[str, ModelFamily] = {
    "sdxl": ModelFamily(
        unet_target_modules=_SDXL_UNET_TARGETS,
        text_encoder_target_modules=_SDXL_TE_TARGETS,
        has_dual_text_encoders=True,
        uses_pooled_prompt_embeds=True,
        uses_time_ids=True,
        default_sampler_pipeline="sdxl",
        latent_channels=4,
        vae_scaling_factor=0.13025,
        supported_training_types=("lora", "full_finetune", "dreambooth", "controlnet", "lllite", "textual_inversion", "ip-adapter", "ileco", "addift"),
        default_profiles=("default", "sdxl_lora_balanced", "sdxl_lora_low_vram"),
        resolution_constraints={"default_resolution": 1024, "multiple_of": 64},
        export_compatibility=("kohya-safetensors", "diffusers-lora"),
    ),
    "sd15": ModelFamily(
        unet_target_modules=_SD15_UNET_TARGETS,
        text_encoder_target_modules=_SD15_TE_TARGETS,
        default_sampler_pipeline="sd15",
        latent_channels=4,
        vae_scaling_factor=0.18215,
        supported_training_types=("lora", "dreambooth", "controlnet", "textual_inversion", "ip-adapter", "ileco", "addift"),
        resolution_constraints={"default_resolution": 512, "multiple_of": 64},
        export_compatibility=("kohya-safetensors", "diffusers-lora"),
    ),
    "anima": ModelFamily(
        unet_target_modules=_ANIMA_DIT_TARGETS,
        text_encoder_target_modules=_ANIMA_TE_TARGETS,
        default_sampler_pipeline="anima",
        latent_channels=16,
        vae_scaling_factor=1.0,
        supported_training_types=("lora", "full_finetune", "ileco", "addift"),
        resolution_constraints={"default_resolution": 1024, "multiple_of": 64},
        export_compatibility=("kohya-safetensors",),
        runtime_requirements=("qwen_image_stack",),
    ),
    "newbie": ModelFamily(
        unet_target_modules=_NEWBIE_UNET_TARGETS,
        text_encoder_target_modules=_SDXL_TE_TARGETS,
        has_dual_text_encoders=True,
        uses_clip_pooled_features=True,
        default_sampler_pipeline="newbie",
        latent_channels=16,
        vae_scaling_factor=0.3611,
        supported_training_types=("lora",),
        resolution_constraints={"default_resolution": 1024, "multiple_of": 64},
        export_compatibility=("kohya-safetensors",),
        runtime_requirements=("newbie_model_stack",),
    ),
    "flux": ModelFamily(
        unet_target_modules=_FLUX_UNET_TARGETS,
        text_encoder_target_modules=_FLUX_TE_TARGETS,
        default_sampler_pipeline=None,
        is_stub=False,
        latent_channels=16,
        vae_scaling_factor=0.3611,
        supported_training_types=("lora",),
        resolution_constraints={"default_resolution": 1024, "multiple_of": 64},
        export_compatibility=("kohya-safetensors", "diffusers-lora"),
        runtime_requirements=("flux_model_stack",),
    ),
}


def get_model_family(arch: str) -> ModelFamily:
    arch = getattr(arch, "value", arch)
    key = str(arch or "sdxl").strip().lower()
    family = _MODEL_FAMILIES.get(key)
    if family is None:
        _logger.warning("Unknown model architecture %r; falling back to SDXL family.", key)
        return _MODEL_FAMILIES["sdxl"]
    if family.is_stub:
        _logger.info("Model family %r is a stub (loader uses temporary scaffold).", key)
    return family


def list_families() -> List[str]:
    return list(_MODEL_FAMILIES.keys())
