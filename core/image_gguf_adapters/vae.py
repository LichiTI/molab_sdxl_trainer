"""VAE-family image GGUF probe adapters."""

from __future__ import annotations

from pathlib import Path

from ..image_gguf_contracts import ImageGGUFComponent, ImageGGUFManifest, TensorInfo
from .common import build_prefix_probe_manifest


class DiffusersVAEProbeAdapter:
    adapter_id = "diffusers_vae_probe_v1"
    component = ImageGGUFComponent.VAE
    family = "diffusers_vae"

    required_tensors = [
        "encoder.conv_in.weight",
        "encoder.conv_out.weight",
        "decoder.conv_in.weight",
        "decoder.conv_out.weight",
    ]
    required_prefixes = ["encoder.", "decoder."]
    optional_prefixes = ["quant_conv.", "post_quant_conv."]
    family_markers = ["encoder.down_blocks.", "decoder.up_blocks.", "decoder.mid_block."]

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = str(family_hint or "").strip().lower().replace("-", "_")
        score = 60 if hint in {"vae", "diffusers_vae"} else 0
        score += sum(12 for marker in self.family_markers if any(key.startswith(marker) for key in tensors))
        score += sum(4 for key in self.required_tensors if key in tensors)
        return score

    def build_manifest(self, source_path: Path, tensors: dict[str, TensorInfo]) -> ImageGGUFManifest:
        return build_prefix_probe_manifest(
            source_path=source_path,
            tensors=tensors,
            adapter_id=self.adapter_id,
            component=self.component.value,
            family=self.family,
            required_tensors=self.required_tensors,
            required_prefixes=self.required_prefixes,
            optional_prefixes=self.optional_prefixes,
            notes=[
                "Phase 2 probe only; no GGUF file is written.",
                "Diffusers VAE probe follows encoder./decoder. safetensors namespaces.",
            ],
        )


class QwenImageVAEProbeAdapter:
    adapter_id = "qwen_image_vae_probe_v1"
    component = ImageGGUFComponent.VAE
    family = "qwen_image_vae"

    required_tensors = [
        "conv1.weight",
        "conv2.weight",
        "encoder.downsamples.0.residual.2.weight",
        "decoder.upsamples.0.residual.2.weight",
    ]
    required_prefixes = ["conv1.", "conv2.", "encoder.", "decoder."]
    optional_prefixes: list[str] = []
    family_markers = ["encoder.downsamples.", "decoder.upsamples.", "decoder.middle.1.to_qkv."]

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = str(family_hint or "").strip().lower().replace("-", "_")
        score = 60 if hint in {"vae", "qwen_image_vae", "anima_vae"} else 0
        score += sum(12 for marker in self.family_markers if any(key.startswith(marker) for key in tensors))
        score += sum(4 for key in self.required_tensors if key in tensors)
        return score

    def build_manifest(self, source_path: Path, tensors: dict[str, TensorInfo]) -> ImageGGUFManifest:
        return build_prefix_probe_manifest(
            source_path=source_path,
            tensors=tensors,
            adapter_id=self.adapter_id,
            component=self.component.value,
            family=self.family,
            required_tensors=self.required_tensors,
            required_prefixes=self.required_prefixes,
            optional_prefixes=self.optional_prefixes,
            notes=[
                "Phase 2 probe only; no GGUF file is written.",
                "Qwen Image VAE probe follows the conv1/conv2 plus encoder./decoder. namespace.",
            ],
        )

__all__ = ["DiffusersVAEProbeAdapter", "QwenImageVAEProbeAdapter"]
