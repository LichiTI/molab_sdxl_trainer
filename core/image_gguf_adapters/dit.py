"""DiT-family image GGUF probe adapters for Anima and Newbie."""

from __future__ import annotations

from pathlib import Path

from ..image_gguf_contracts import ImageGGUFComponent, ImageGGUFManifest, TensorInfo
from .common import build_probe_manifest, count_prefix_hits


class AnimaDiTProbeAdapter:
    adapter_id = "anima_dit_probe_v1"
    component = ImageGGUFComponent.ANIMA_DIT
    family = "anima"

    required_tensors = [
        "net.x_embedder.proj.1.weight",
        "net.final_layer.linear.weight",
        "net.blocks.0.self_attn.q_proj.weight",
        "net.blocks.0.self_attn.k_proj.weight",
        "net.blocks.0.self_attn.v_proj.weight",
        "net.blocks.0.self_attn.output_proj.weight",
        "net.blocks.0.cross_attn.q_proj.weight",
        "net.blocks.0.mlp.layer1.weight",
        "net.blocks.0.mlp.layer2.weight",
    ]
    required_prefixes = [
        "net.blocks.",
        "net.final_layer.",
        "net.t_embedder.",
        "net.x_embedder.",
    ]
    optional_prefixes = ["net.llm_adapter.", "net.t_embedding_norm."]
    family_markers = ["net.blocks.", "net.x_embedder.", "net.final_layer.adaln_modulation"]

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = str(family_hint or "").strip().lower().replace("-", "_")
        score = 60 if hint in {"anima", "anima_dit"} else 0
        score += sum(12 for marker in self.family_markers if any(key.startswith(marker) for key in tensors))
        score += sum(4 for key in self.required_tensors if key in tensors)
        return score

    def build_manifest(self, source_path: Path, tensors: dict[str, TensorInfo]) -> ImageGGUFManifest:
        prefix_hits = count_prefix_hits(tensors, self.required_prefixes + self.optional_prefixes)
        matched = sum(prefix_hits[prefix] for prefix in self.required_prefixes + self.optional_prefixes)
        expected_prefixes = tuple(self.required_prefixes + self.optional_prefixes)
        unexpected = [key for key in sorted(tensors) if not key.startswith(expected_prefixes)]
        warnings = []
        for prefix in self.optional_prefixes:
            if prefix_hits.get(prefix, 0) == 0:
                warnings.append(f"optional prefix not present: {prefix}")
        notes = [
            "Phase 1 manifest probe only; no GGUF file is written.",
            "Anima DiT probe follows the net.* native training/keymap namespace.",
        ]
        return build_probe_manifest(
            source_path=source_path,
            adapter_id=self.adapter_id,
            component=self.component.value,
            family=self.family,
            tensors=tensors,
            required_tensors=list(self.required_tensors),
            required_prefixes=list(self.required_prefixes),
            matched_tensors=matched,
            unexpected_tensors_sample=unexpected,
            notes=notes,
            warnings=warnings,
        )


class NewbieDiTProbeAdapter:
    adapter_id = "newbie_dit_probe_v1"
    component = ImageGGUFComponent.NEWBIE_DIT
    family = "newbie"

    required_tensors = [
        "x_embedder.weight",
        "final_layer.linear.weight",
        "layers.0.attention.qkv.weight",
        "layers.0.attention.out.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.feed_forward.w3.weight",
    ]
    required_prefixes = [
        "layers.",
        "final_layer.",
        "t_embedder.",
        "x_embedder.",
    ]
    optional_prefixes = [
        "context_refiner.",
        "noise_refiner.",
        "cap_embedder.",
        "clip_text_pooled_proj.",
        "time_text_embed.",
        "norm_final.",
    ]
    family_markers = ["layers.", "noise_refiner.", "context_refiner.", "final_layer.adaLN_modulation"]

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = str(family_hint or "").strip().lower().replace("-", "_")
        score = 60 if hint in {"newbie", "newbie_dit", "native_newbie"} else 0
        score += sum(12 for marker in self.family_markers if any(key.startswith(marker) for key in tensors))
        score += sum(4 for key in self.required_tensors if key in tensors)
        return score

    def build_manifest(self, source_path: Path, tensors: dict[str, TensorInfo]) -> ImageGGUFManifest:
        prefix_hits = count_prefix_hits(tensors, self.required_prefixes + self.optional_prefixes)
        matched = sum(prefix_hits[prefix] for prefix in self.required_prefixes + self.optional_prefixes)
        expected_prefixes = tuple(self.required_prefixes + self.optional_prefixes)
        unexpected = [key for key in sorted(tensors) if not key.startswith(expected_prefixes)]
        warnings = []
        for prefix in self.optional_prefixes:
            if prefix_hits.get(prefix, 0) == 0:
                warnings.append(f"optional prefix not present: {prefix}")
        notes = [
            "Phase 1 manifest probe only; no GGUF file is written.",
            "Newbie DiT probe follows the native NextDiT-style training namespace.",
        ]
        return build_probe_manifest(
            source_path=source_path,
            adapter_id=self.adapter_id,
            component=self.component.value,
            family=self.family,
            tensors=tensors,
            required_tensors=list(self.required_tensors),
            required_prefixes=list(self.required_prefixes),
            matched_tensors=matched,
            unexpected_tensors_sample=unexpected,
            notes=notes,
            warnings=warnings,
        )


__all__ = ["AnimaDiTProbeAdapter", "NewbieDiTProbeAdapter"]
