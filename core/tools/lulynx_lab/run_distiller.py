"""CLI runner for the experimental Lulynx LAB LoRA distiller.

The launcher calls this file in a separate runtime Python process so the UI can
track the job without importing torch-heavy modules in the pywebview process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Distiller config must be a JSON object.")
    return data


def _parse_learning_rate(value: Any) -> float:
    try:
        parsed = float(str(value or "1e-5").strip())
    except (TypeError, ValueError):
        parsed = 1e-5
    if parsed <= 0:
        parsed = 1e-5
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lulynx LAB LoRA distillation.")
    parser.add_argument("--config", required=True, help="Path to launcher-generated distiller config JSON.")
    args = parser.parse_args()

    backend_root = _backend_root()
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    project_root = backend_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    dry_run = bool(config.get("dry_run", False))

    print(f"[LAB Distiller] Config: {config_path}", flush=True)
    print(f"[LAB Distiller] Dry run: {dry_run}", flush=True)
    print(f"[LAB Distiller] Teacher LoRA: {config.get('lora_path')}", flush=True)
    print(f"[LAB Distiller] Output: {config.get('output_path')}", flush=True)
    print(f"[LAB Distiller] Dtype: {config.get('dtype') or 'bf16'}", flush=True)
    print(f"[LAB Distiller] Learning rate: {_parse_learning_rate(config.get('learning_rate'))}", flush=True)

    required = ("unet_path", "lora_path", "output_path")
    missing = [key for key in required if not str(config.get(key) or "").strip()]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    if dry_run:
        print("[LAB Distiller] Contract validation completed; real distillation was not started.", flush=True)
        return 0

    import torch
    from core.tools.lulynx_lab.distiller import LoRADistiller

    def _parse_dtype(value: Any, device: str) -> torch.dtype:
        key = str(value or "bf16").strip().lower()
        if key == "auto":
            return torch.bfloat16 if str(device).startswith("cuda") and torch.cuda.is_available() else torch.float32
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16", "half"}:
            return torch.float16
        if key in {"fp32", "float32", "float"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {value}")

    device = str(config.get("device") or "cuda")
    dtype = _parse_dtype(config.get("dtype") or "bf16", device)
    learning_rate = _parse_learning_rate(config.get("learning_rate"))

    distiller = LoRADistiller(
        unet_path=str(config["unet_path"]),
        lora_path=str(config["lora_path"]),
        llm_path=str(config.get("llm_path") or "Qwen/Qwen2.5-0.5B"),
        projector_path=str(config.get("projector_path") or "") or None,
        teacher_path=str(config.get("teacher_path") or "") or None,
        allow_tokenizer_only_clip=bool(config.get("allow_tokenizer_only_clip", False)),
        device=device,
        dtype=dtype,
        learning_rate=learning_rate,
    )
    distiller.distill(
        steps=int(config.get("steps") or 1000),
        batch_size=int(config.get("batch_size") or 4),
    )
    output_path = Path(str(config["output_path"])).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distiller.save(str(output_path))
    print(f"[LAB Distiller] Saved: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
