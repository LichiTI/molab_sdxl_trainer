"""Text-encoder image GGUF probe adapters."""

from __future__ import annotations

from pathlib import Path

from ..image_gguf_contracts import ImageGGUFComponent, ImageGGUFManifest, TensorInfo
from .common import build_prefix_probe_manifest


class CLIPTextProbeAdapter:
    adapter_id = "clip_text_probe_v1"
    component = ImageGGUFComponent.CLIP
    family = "clip_text"

    required_tensors = [
        "text_model.embeddings.token_embedding.weight",
        "text_model.embeddings.position_embedding.weight",
        "text_model.encoder.layers.0.self_attn.q_proj.weight",
        "text_model.encoder.layers.0.self_attn.k_proj.weight",
        "text_model.encoder.layers.0.self_attn.v_proj.weight",
        "text_model.encoder.layers.0.mlp.fc1.weight",
    ]
    required_prefixes = ["text_model."]
    optional_prefixes = ["text_projection.", "logit_scale"]
    family_markers = ["text_model.encoder.layers.", "text_model.final_layer_norm."]

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = str(family_hint or "").strip().lower().replace("-", "_")
        score = 60 if hint in {"clip", "clip_text", "openclip"} else 0
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
                "CLIP text probe follows Hugging Face text_model.* namespaces.",
            ],
        )


class JinaCLIPTextProbeAdapter:
    adapter_id = "jina_clip_text_probe_v1"
    component = ImageGGUFComponent.CLIP
    family = "jina_clip_text"

    required_tensors = [
        "model.embeddings.word_embeddings.weight",
        "model.embeddings.token_type_embeddings.weight",
        "model.encoder.layers.0.mixer.Wqkv.weight",
        "model.encoder.layers.0.mixer.out_proj.weight",
        "model.encoder.layers.0.mlp.fc1.weight",
        "model.emb_ln.weight",
    ]
    required_prefixes = ["model."]
    optional_prefixes = ["spiece_model"]
    family_markers = ["model.encoder.layers.", "model.emb_ln.", "model.embeddings.word_embeddings."]

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = str(family_hint or "").strip().lower().replace("-", "_")
        score = 60 if hint in {"clip", "jina_clip", "jina_clip_text"} else 0
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
                "Jina CLIP probe follows model.encoder.layers.* mixer namespaces.",
            ],
        )


class T5EncoderProbeAdapter:
    adapter_id = "t5_encoder_probe_v1"
    component = ImageGGUFComponent.T5
    family = "t5_encoder"

    required_tensors = [
        "encoder.block.0.layer.0.SelfAttention.q.weight",
        "encoder.block.0.layer.0.SelfAttention.k.weight",
        "encoder.block.0.layer.0.SelfAttention.v.weight",
        "encoder.block.0.layer.1.DenseReluDense.wo.weight",
    ]
    required_prefixes = ["encoder.block."]
    optional_prefixes = ["shared.", "encoder.final_layer_norm."]
    family_markers = ["encoder.block."]

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = str(family_hint or "").strip().lower().replace("-", "_")
        score = 60 if hint in {"t5", "t5_encoder", "text_encoder_2"} else 0
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
                "T5 encoder probe may describe one safetensors shard; shard merging is a later export concern.",
            ],
        )

__all__ = ["CLIPTextProbeAdapter", "JinaCLIPTextProbeAdapter", "T5EncoderProbeAdapter"]
