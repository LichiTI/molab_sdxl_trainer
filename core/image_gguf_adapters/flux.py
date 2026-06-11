"""FLUX-family probe adapters.

FLUX transformer weights are recognized deliberately but kept as a generic
tensor bundle until the image GGUF runtime/export contract supports that model
family. This prevents weak DiT namespace overlap from selecting Anima/Newbie.
"""

from __future__ import annotations

from pathlib import Path

from ..image_gguf_contracts import ImageGGUFComponent, ImageGGUFCompatibility, ImageGGUFManifest, TensorInfo
from .common import build_probe_manifest, count_prefix_hits


class FluxTransformerProbeAdapter:
    adapter_id = "flux_transformer_probe_v1"
    component = ImageGGUFComponent.GENERIC_TENSOR_BUNDLE
    family = "flux_transformer"

    required_prefixes = [
        "transformer_blocks.",
        "single_transformer_blocks.",
    ]
    optional_prefixes = [
        "context_embedder.",
        "time_text_embed.",
        "norm_out.",
        "proj_out.",
    ]

    def score(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> int:
        hint = str(family_hint or "").strip().lower().replace("-", "_")
        prefix_hits = count_prefix_hits(tensors, self.required_prefixes + self.optional_prefixes)
        score = 80 if hint in {"flux", "flux_transformer", "flux_dit"} else 0
        score += sum(24 for prefix in self.required_prefixes if prefix_hits.get(prefix, 0) > 0)
        score += sum(8 for prefix in self.optional_prefixes if prefix_hits.get(prefix, 0) > 0)
        return score

    def build_manifest(self, source_path: Path, tensors: dict[str, TensorInfo]) -> ImageGGUFManifest:
        prefixes = self.required_prefixes + self.optional_prefixes
        prefix_hits = count_prefix_hits(tensors, prefixes)
        matched = sum(prefix_hits[prefix] for prefix in prefixes)
        expected_prefixes = tuple(prefixes)
        unexpected = [key for key in sorted(tensors) if not key.startswith(expected_prefixes)]
        manifest = build_probe_manifest(
            source_path=source_path,
            adapter_id=self.adapter_id,
            component=self.component.value,
            family=self.family,
            tensors=tensors,
            required_tensors=[],
            required_prefixes=[],
            matched_tensors=matched,
            unexpected_tensors_sample=unexpected,
            notes=[
                "FLUX transformer namespace recognized by shadow probe.",
                "FLUX image GGUF export/runtime loader is not implemented in the current target set.",
            ],
            warnings=["flux transformer is recognized but not export/runtime compatible yet"],
        )
        return ImageGGUFManifest(
            **{**manifest.to_dict(), "compatibility": ImageGGUFCompatibility.PROBE_ONLY.value, "ok": True}
        )


__all__ = ["FluxTransformerProbeAdapter"]
