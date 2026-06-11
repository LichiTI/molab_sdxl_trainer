"""Read-only image-model GGUF compatibility probe.

This module implements Phase 1 from ``backend/docs/image_model_gguf_roadmap.md``:
adapter contract, registry, and manifest reports. It never writes GGUF files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - surfaced by caller in support dependency checks
    safe_open = None  # type: ignore[assignment]

from .image_gguf_adapters import (
    AnimaDiTProbeAdapter,
    CLIPTextProbeAdapter,
    DiffusersVAEProbeAdapter,
    FluxTransformerProbeAdapter,
    JinaCLIPTextProbeAdapter,
    NewbieDiTProbeAdapter,
    QwenImageVAEProbeAdapter,
    SD15UNetProbeAdapter,
    SDXLUNetProbeAdapter,
    T5EncoderProbeAdapter,
)
from .image_gguf_adapters.common import build_probe_manifest
from .image_gguf_contracts import (
    ImageGGUFAdapter,
    ImageGGUFComponent,
    ImageGGUFCompatibility,
    ImageGGUFManifest,
    TensorInfo,
)


class ImageGGUFAdapterRegistry:
    def __init__(self, adapters: Iterable[ImageGGUFAdapter] | None = None) -> None:
        self._adapters = list(adapters) if adapters is not None else _default_adapters()

    def list_adapters(self) -> list[dict[str, str]]:
        return [
            {
                "adapter_id": adapter.adapter_id,
                "component": adapter.component.value,
                "family": adapter.family,
            }
            for adapter in self._adapters
        ]

    def select(self, tensors: dict[str, TensorInfo], *, family_hint: str = "") -> ImageGGUFAdapter | None:
        ranked = sorted(
            ((adapter.score(tensors, family_hint=family_hint), adapter) for adapter in self._adapters),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] <= 0:
            return None
        return ranked[0][1]


def _default_adapters() -> list[ImageGGUFAdapter]:
    return [
        FluxTransformerProbeAdapter(),
        AnimaDiTProbeAdapter(),
        NewbieDiTProbeAdapter(),
        SD15UNetProbeAdapter(),
        SDXLUNetProbeAdapter(),
        QwenImageVAEProbeAdapter(),
        DiffusersVAEProbeAdapter(),
        JinaCLIPTextProbeAdapter(),
        CLIPTextProbeAdapter(),
        T5EncoderProbeAdapter(),
    ]


def probe_image_gguf_manifest(
    model_path: str | Path,
    *,
    family_hint: str = "",
    registry: ImageGGUFAdapterRegistry | None = None,
) -> ImageGGUFManifest:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Phase 1 image GGUF probe currently accepts .safetensors files only")

    tensors = read_safetensors_tensor_info(path)
    selected = (registry or ImageGGUFAdapterRegistry()).select(tensors, family_hint=family_hint)
    if selected is None:
        return _unknown_manifest(path, tensors)
    return selected.build_manifest(path, tensors)


def read_safetensors_tensor_info(path: str | Path) -> dict[str, TensorInfo]:
    if safe_open is None:
        raise RuntimeError("image GGUF probe requires safetensors")
    tensors: dict[str, TensorInfo] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_slice = handle.get_slice(key)
            tensors[str(key)] = TensorInfo(
                key=str(key),
                shape=[int(dim) for dim in tensor_slice.get_shape()],
                dtype=str(tensor_slice.get_dtype()),
            )
    return tensors


def _unknown_manifest(path: Path, tensors: dict[str, TensorInfo]) -> ImageGGUFManifest:
    compatibility = ImageGGUFCompatibility.UNKNOWN.value
    component = ImageGGUFComponent.UNKNOWN.value
    family = "unknown"
    if tensors:
        compatibility = ImageGGUFCompatibility.PROBE_ONLY.value
        component = ImageGGUFComponent.GENERIC_TENSOR_BUNDLE.value
        family = "generic"
    return build_probe_manifest(
        source_path=path,
        adapter_id="generic_tensor_bundle_probe_v1",
        component=component,
        family=family,
        tensors=tensors,
        required_tensors=[],
        required_prefixes=[],
        matched_tensors=len(tensors),
        unexpected_tensors_sample=[] if component == ImageGGUFComponent.GENERIC_TENSOR_BUNDLE.value else list(tensors)[:40],
        notes=["No image GGUF family adapter matched this tensor namespace."],
        warnings=["generic tensor bundle is not a runtime-compatible image GGUF contract"],
    )


__all__ = [
    "ImageGGUFAdapterRegistry",
    "probe_image_gguf_manifest",
    "read_safetensors_tensor_info",
]
