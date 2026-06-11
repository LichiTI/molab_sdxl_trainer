"""Read-only real-weight smoke for image GGUF export plans."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.tools.image_gguf_exporter import plan_image_gguf_export  # noqa: E402


REAL_PLAN_TARGETS = [
    {
        "family_hint": "vae",
        "paths": [ROOT / "models" / "newbie" / "vae" / "diffusion_pytorch_model.safetensors"],
        "component": "vae",
        "ok": True,
    },
    {
        "family_hint": "clip",
        "paths": [ROOT / "models" / "flux" / "FLUX.1-schnell" / "text_encoder" / "model.safetensors"],
        "component": "clip",
        "ok": True,
    },
    {
        "family_hint": "t5",
        "paths": [
            ROOT / "models" / "flux" / "FLUX.1-schnell" / "text_encoder_2" / "model-00001-of-00002.safetensors",
            ROOT / "models" / "flux" / "FLUX.1-schnell" / "text_encoder_2" / "model-00002-of-00002.safetensors",
        ],
        "component": "t5",
        "ok": True,
    },
    {
        "family_hint": "anima",
        "paths": [ROOT / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"],
        "component": "anima_dit",
        "ok": True,
        "requires_container_only_warning": True,
    },
    {
        "family_hint": "newbie",
        "paths": [ROOT / "models" / "newbie" / "diffusion_pytorch_model.safetensors"],
        "component": "newbie_dit",
        "ok": True,
        "requires_container_only_warning": True,
    },
    {
        "family_hint": "sdxl",
        "paths": [ROOT / "models" / "sdxl" / "silentEraFurrymixNAIXL_v10.safetensors"],
        "component": "sdxl_unet",
        "ok": True,
        "requires_container_only_warning": True,
    },
]


def _run_target(target: dict[str, object]) -> dict[str, object] | None:
    paths = [Path(path) for path in target["paths"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        print(f"SKIP: local real weight not found: {missing[0]}")
        return None
    plan = plan_image_gguf_export(paths, family_hint=str(target["family_hint"]), file_type="f16").to_dict()
    assert plan["component"] == target["component"], plan
    assert plan["ok"] is target["ok"], plan
    assert plan["unique_tensor_count"] > 0, plan
    assert plan["estimated_output_size_bytes"] > 0, plan
    if plan["ok"]:
        assert plan["compatibility"] == "container_candidate", plan
        assert not plan["errors"], plan
        if target.get("requires_container_only_warning"):
            assert any("container-compatible only" in item for item in plan["warnings"]), plan
    else:
        assert plan["errors"], plan
    return {
        "component": plan["component"],
        "family": plan["family"],
        "ok": plan["ok"],
        "unique_tensor_count": plan["unique_tensor_count"],
        "estimated_output_size_bytes": plan["estimated_output_size_bytes"],
        "errors": plan["errors"],
        "warnings": plan["warnings"][:2],
    }


def main() -> int:
    reports = []
    for target in REAL_PLAN_TARGETS:
        report = _run_target(target)
        if report is not None:
            reports.append(report)
    if not reports:
        print("SKIP: no local real image weights found for image GGUF export plans")
        return 0
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    print("\nAll available real image GGUF export plan smoke checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
