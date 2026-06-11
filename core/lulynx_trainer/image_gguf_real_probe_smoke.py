"""Read-only image GGUF probe against local real image-model weights.

This smoke test is intentionally report-only. It validates that image GGUF probe
contracts recognize local image-model checkpoints without loading tensor data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.image_gguf_probe import probe_image_gguf_manifest  # noqa: E402


REAL_PROBE_TARGETS = [
    {
        "family_hint": "anima",
        "component": "anima_dit",
        "family": "anima",
        "path": ROOT / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors",
    },
    {
        "family_hint": "newbie",
        "component": "newbie_dit",
        "family": "newbie",
        "path": ROOT / "models" / "newbie" / "diffusion_pytorch_model.safetensors",
    },
    {
        "family_hint": "qwen_image_vae",
        "component": "vae",
        "family": "qwen_image_vae",
        "path": ROOT / "models" / "anima" / "vae" / "qwen_image_vae.safetensors",
    },
    {
        "family_hint": "vae",
        "component": "vae",
        "family": "diffusers_vae",
        "path": ROOT / "models" / "newbie" / "vae" / "diffusion_pytorch_model.safetensors",
    },
    {
        "family_hint": "vae",
        "component": "vae",
        "family": "diffusers_vae",
        "path": ROOT / "models" / "flux" / "FLUX.1-schnell" / "vae" / "diffusion_pytorch_model.safetensors",
    },
    {
        "family_hint": "clip",
        "component": "clip",
        "family": "jina_clip_text",
        "path": ROOT / "models" / "newbie" / "clip_model" / "jina-clip-v2.safetensors",
    },
    {
        "family_hint": "clip",
        "component": "clip",
        "family": "clip_text",
        "path": ROOT / "models" / "flux" / "FLUX.1-schnell" / "text_encoder" / "model.safetensors",
    },
    {
        "family_hint": "t5",
        "component": "t5",
        "family": "t5_encoder",
        "path": ROOT
        / "models"
        / "flux"
        / "FLUX.1-schnell"
        / "text_encoder_2"
        / "model-00001-of-00002.safetensors",
    },
    {
        "family_hint": "t5",
        "component": "t5",
        "family": "t5_encoder",
        "path": ROOT
        / "models"
        / "flux"
        / "FLUX.1-schnell"
        / "text_encoder_2"
        / "model-00002-of-00002.safetensors",
        "allow_missing_required_tensors": True,
    },
    {
        "family_hint": "sdxl",
        "component": "sdxl_unet",
        "family": "sdxl_unet",
        "path": ROOT / "models" / "sdxl" / "silentEraFurrymixNAIXL_v10.safetensors",
    },
]


def _summary(manifest: dict[str, object]) -> dict[str, object]:
    shape_summary = manifest.get("shape_summary", {})
    total_numel = shape_summary.get("total_numel", 0) if isinstance(shape_summary, dict) else 0
    return {
        "source_path": manifest["source_path"],
        "component": manifest["component"],
        "family": manifest["family"],
        "ok": manifest["ok"],
        "tensor_count": manifest["tensor_count"],
        "matched_tensors": manifest["matched_tensors"],
        "missing_required_tensors": manifest["missing_required_tensors"],
        "missing_required_prefixes": manifest["missing_required_prefixes"],
        "unexpected_tensors_sample": manifest["unexpected_tensors_sample"],
        "dtype_counts": manifest["dtype_counts"],
        "rank_counts": manifest["rank_counts"],
        "total_numel": total_numel,
    }


def _probe_target(target: dict[str, object]) -> dict[str, object] | None:
    path = Path(target["path"])
    if not path.is_file():
        print(f"SKIP: local real weight not found: {path}")
        return None

    manifest = probe_image_gguf_manifest(path, family_hint=str(target["family_hint"])).to_dict()
    assert manifest["component"] == target["component"], manifest
    assert manifest["family"] == target["family"], manifest
    allow_missing = bool(target.get("allow_missing_required_tensors"))
    if allow_missing:
        assert manifest["tensor_count"] > 0, manifest
        assert manifest["matched_tensors"] > 0, manifest
    else:
        assert manifest["ok"] is True, manifest
    assert manifest["tensor_count"] > 0, manifest
    min_matched = int(target.get("min_matched_tensors", manifest["tensor_count"]))
    assert manifest["matched_tensors"] >= min_matched, manifest
    if not allow_missing:
        assert manifest["missing_required_tensors"] == [], manifest
    assert manifest["missing_required_prefixes"] == [], manifest
    assert manifest["unexpected_tensors_sample"] == [], manifest
    return _summary(manifest)


def main() -> int:
    reports = []
    for target in REAL_PROBE_TARGETS:
        report = _probe_target(target)
        if report is not None:
            reports.append(report)

    if not reports:
        print("SKIP: no local real image-model weights found for image GGUF probe")
        return 0

    print(json.dumps(reports, indent=2, ensure_ascii=False))
    print("\nAll available real image GGUF probe smoke checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
