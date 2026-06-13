"""Static optimization capability hints for model-family preflight.

The data in this module is deliberately descriptive. It lets request and
preflight layers explain which optimization tracks are ready, gated, or still
stubbed without importing trainer implementations or reference project code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OptimizationCapability:
    family: str
    display_name: str
    backbone_kind: str
    native_route_status: str
    latent_channels: int
    default_resolution: int
    supports_latent_cache: bool
    supports_text_cache: bool
    supports_transformer_weight_compression: bool
    supports_block_residency: bool
    supports_flow_shift: bool
    recommended_attention: tuple[str, ...]
    optimization_tracks: tuple[str, ...]
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unsupported_training_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, tuple):
                data[key] = list(value)
        return data


_CAPABILITIES: dict[str, OptimizationCapability] = {
    "sdxl": OptimizationCapability(
        family="sdxl",
        display_name="SDXL",
        backbone_kind="unet",
        native_route_status="ready",
        latent_channels=4,
        default_resolution=1024,
        supports_latent_cache=True,
        supports_text_cache=True,
        supports_transformer_weight_compression=False,
        supports_block_residency=True,
        supports_flow_shift=False,
        recommended_attention=("sdpa", "xformers", "flash2"),
        optimization_tracks=("latent_cache", "text_cache", "sdxl_low_vram", "native_unet"),
    ),
    "anima": OptimizationCapability(
        family="anima",
        display_name="Anima",
        backbone_kind="dit_transformer",
        native_route_status="ready",
        latent_channels=16,
        default_resolution=1024,
        supports_latent_cache=True,
        supports_text_cache=True,
        supports_transformer_weight_compression=True,
        supports_block_residency=True,
        supports_flow_shift=True,
        recommended_attention=("sdpa", "flash2", "sageattn"),
        optimization_tracks=("online_cache", "transformer_weight_compression", "block_residency", "flow_shift"),
        notes=("Anima uses the native DiT path and can share most transformer-side optimization policies.",),
    ),
    "newbie": OptimizationCapability(
        family="newbie",
        display_name="Newbie",
        backbone_kind="dit_transformer",
        native_route_status="ready",
        latent_channels=16,
        default_resolution=1024,
        supports_latent_cache=True,
        supports_text_cache=True,
        supports_transformer_weight_compression=True,
        supports_block_residency=True,
        supports_flow_shift=True,
        recommended_attention=("sdpa", "flash2"),
        optimization_tracks=("materialized_cache", "transformer_weight_compression", "streaming_block_residency", "flow_shift"),
    ),
    "flux": OptimizationCapability(
        family="flux",
        display_name="FLUX",
        backbone_kind="dit_transformer",
        native_route_status="lora_preview",
        latent_channels=16,
        default_resolution=1024,
        supports_latent_cache=True,
        supports_text_cache=True,
        supports_transformer_weight_compression=True,
        supports_block_residency=False,
        supports_flow_shift=True,
        recommended_attention=("sdpa",),
        optimization_tracks=("latent_cache", "text_cache", "transformer_weight_compression", "flow_shift"),
        notes=(
            "FLUX should enter through the native DiT/request path, not a separate diffusion-pipe runner.",
            "FLUX LoRA uses the native preview trainer with frozen VAE/text encoders and transformer LoRA adapters.",
            "The first safe optimization track is cache-first conditioning plus frozen transformer compression.",
            "channels_last is expected to have limited benefit because the trainable backbone is transformer-heavy.",
        ),
        warnings=(
            "FLUX LoRA support is preview-level; full finetune and ControlNet schemas remain disabled.",
        ),
        unsupported_training_types=("full_finetune", "controlnet"),
    ),
    "lumina": OptimizationCapability(
        family="lumina",
        display_name="Lumina",
        backbone_kind="dit_transformer",
        native_route_status="placeholder",
        latent_channels=16,
        default_resolution=1024,
        supports_latent_cache=True,
        supports_text_cache=True,
        supports_transformer_weight_compression=True,
        supports_block_residency=False,
        supports_flow_shift=True,
        recommended_attention=("sdpa",),
        optimization_tracks=("latent_cache", "text_cache", "transformer_weight_compression"),
        warnings=("Lumina is selectable as a placeholder only; the trainer core is not wired yet.",),
    ),
    "lumina2": OptimizationCapability(
        family="lumina2",
        display_name="Lumina2",
        backbone_kind="dit_transformer",
        native_route_status="placeholder",
        latent_channels=16,
        default_resolution=1024,
        supports_latent_cache=True,
        supports_text_cache=True,
        supports_transformer_weight_compression=True,
        supports_block_residency=False,
        supports_flow_shift=True,
        recommended_attention=("sdpa",),
        optimization_tracks=("latent_cache", "text_cache", "transformer_weight_compression"),
        warnings=("Lumina2 is selectable as a placeholder only; the trainer core is not wired yet.",),
    ),
    "qwen-image": OptimizationCapability(
        family="qwen-image",
        display_name="Qwen Image",
        backbone_kind="dit_transformer",
        native_route_status="placeholder",
        latent_channels=16,
        default_resolution=1024,
        supports_latent_cache=True,
        supports_text_cache=True,
        supports_transformer_weight_compression=True,
        supports_block_residency=False,
        supports_flow_shift=True,
        recommended_attention=("sdpa",),
        optimization_tracks=("latent_cache", "text_cache", "transformer_weight_compression"),
        warnings=("Qwen Image is selectable as a placeholder only; the trainer core is not wired yet.",),
    ),
    "hunyuan-dit": OptimizationCapability(
        family="hunyuan-dit",
        display_name="HunyuanDiT",
        backbone_kind="dit_transformer",
        native_route_status="placeholder",
        latent_channels=4,
        default_resolution=1024,
        supports_latent_cache=True,
        supports_text_cache=True,
        supports_transformer_weight_compression=True,
        supports_block_residency=False,
        supports_flow_shift=True,
        recommended_attention=("sdpa",),
        optimization_tracks=("latent_cache", "text_cache", "transformer_weight_compression"),
        warnings=("HunyuanDiT is selectable as a placeholder only; the trainer core is not wired yet.",),
    ),
}

_SCHEMA_FAMILY: dict[str, str] = {
    "sdxl-lora": "sdxl",
    "sdxl-finetune": "sdxl",
    "sdxl-dreambooth": "sdxl",
    "sdxl-controlnet": "sdxl",
    "anima-lora": "anima",
    "anima-finetune": "anima",
    "newbie-lora": "newbie",
    "flux-lora": "flux",
    "flux-finetune": "flux",
    "flux-controlnet": "flux",
    "lumina-lora": "lumina",
    "lumina2-lora": "lumina2",
    "lumina-finetune": "lumina",
    "qwen-image-lora": "qwen-image",
    "hunyuan-dit-lora": "hunyuan-dit",
    "hunyuan-image-lora": "hunyuan-dit",
}


def family_from_schema(schema_id: str, model_type: str = "") -> str:
    schema = str(schema_id or "").strip().lower().replace("_", "-")
    model = str(model_type or "").strip().lower().replace("_", "-")
    return _SCHEMA_FAMILY.get(schema, model)


def get_model_optimization_capability(family: str) -> OptimizationCapability | None:
    return _CAPABILITIES.get(str(family or "").strip().lower().replace("_", "-"))


def capability_dict_for_schema(schema_id: str, model_type: str = "") -> dict[str, Any] | None:
    capability = get_model_optimization_capability(family_from_schema(schema_id, model_type))
    return capability.to_dict() if capability is not None else None


__all__ = [
    "OptimizationCapability",
    "capability_dict_for_schema",
    "family_from_schema",
    "get_model_optimization_capability",
]
