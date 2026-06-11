"""Light real-weight smoke for image GGUF component exports.

The targets are intentionally small enough for a local smoke run. Large DiT,
UNet, and T5 shards remain covered by read-only probes until runtime/export
quality gates are stronger.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.tools.image_gguf_exporter import export_image_gguf_component  # noqa: E402


REAL_EXPORT_TARGETS = [
    {
        "family_hint": "vae",
        "component": "vae",
        "family": "diffusers_vae",
        "path": ROOT / "models" / "newbie" / "vae" / "diffusion_pytorch_model.safetensors",
        "name": "newbie-vae-real-smoke",
    },
    {
        "family_hint": "clip",
        "component": "clip",
        "family": "clip_text",
        "path": ROOT / "models" / "flux" / "FLUX.1-schnell" / "text_encoder" / "model.safetensors",
        "name": "flux-clip-real-smoke",
    },
]


def _run_target(target: dict[str, object], output_dir: Path) -> dict[str, object] | None:
    path = Path(target["path"])
    if not path.is_file():
        print(f"SKIP: local real weight not found: {path}")
        return None

    dst = output_dir / f"{target['name']}.gguf"
    result = export_image_gguf_component(
        path,
        dst,
        family_hint=str(target["family_hint"]),
        name=str(target["name"]),
        file_type="f16",
        overwrite=True,
    )
    assert result.ok is True, result
    assert result.component == target["component"], result
    assert result.family == target["family"], result
    assert result.tensor_count > 0, result
    assert result.converted_tensors > 0, result
    assert Path(result.output_path).is_file(), result
    assert Path(result.sidecar_path).is_file(), result

    import gguf

    reader = gguf.GGUFReader(result.output_path)
    assert len(reader.tensors) == result.tensor_count, result
    sidecar = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert sidecar["compatibility"] == "container_compatible", sidecar
    return {
        "source_path": str(path),
        "output_path": result.output_path,
        "sidecar_path": result.sidecar_path,
        "component": result.component,
        "family": result.family,
        "tensor_count": result.tensor_count,
        "converted_tensors": result.converted_tensors,
        "output_size_bytes": result.output_size_bytes,
    }


def main() -> int:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("SKIP: real image GGUF export requires gguf")
        return 0

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        output_dir = Path(tmp)
        reports = []
        for target in REAL_EXPORT_TARGETS:
            report = _run_target(target, output_dir)
            if report is not None:
                reports.append(report)

        if not reports:
            print("SKIP: no local real image weights found for image GGUF export")
            return 0
        print(json.dumps(reports, indent=2, ensure_ascii=False))
    print("\nAll available real image GGUF export smoke checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
