"""Smoke probe for V2 LoRA native mixed precision training dispatch."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_lora_forward_preflight import build_lora_forward_dispatch_preflight  # noqa: E402
from core.turbocore_lora_native_abi import probe_lora_fused_native_abi  # noqa: E402
from core.turbocore_lora_native_runtime import probe_lora_native_training_dispatch  # noqa: E402


def _ensure_native_artifact_dir() -> dict[str, str | None]:
    old = {"LULYNX_NATIVE_ARTIFACT_DIR": os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR")}
    if not os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR"):
        artifact_dir = REPO_ROOT / "backend" / "native" / "target" / "release"
        if artifact_dir.exists():
            os.environ["LULYNX_NATIVE_ARTIFACT_DIR"] = str(artifact_dir)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "schema_version": 1,
            "probe": "turbocore_lora_native_mixed_precision_smoke",
            "ok": True,
            "skipped": True,
            "reason": "cuda_not_available",
            "training_path_enabled": False,
        }
    old = _ensure_native_artifact_dir()
    try:
        fp16 = probe_lora_native_training_dispatch(
            x_shape=(2, 64, 320),
            rank=4,
            out_features=320,
            dtype="float16",
            device=torch.device("cuda"),
        )
        abi = probe_lora_fused_native_abi(
            x_shape=(2, 64, 320),
            rank=4,
            out_features=320,
            dtype="float16",
        )
        preflight = build_lora_forward_dispatch_preflight(
            x_shape=(2, 64, 320),
            dtype="float16",
            rank=4,
            native_abi_report=abi,
            native_training_report=fp16,
            request_training_dispatch=True,
            allow_experimental_native=True,
        )
    finally:
        _restore_env(old)
    assert fp16["ok"] is True, fp16
    assert fp16["dtype"] == "float16", fp16
    assert fp16["kernel_executed"] is True, fp16
    assert fp16["training_tensor_binding"] is True, fp16
    assert fp16["fallback_to_pytorch_lora"] is False, fp16
    assert abi["ok"] is True, abi
    assert preflight["native_dispatch_allowed"] is True, preflight
    return {
        "schema_version": 1,
        "probe": "turbocore_lora_native_mixed_precision_smoke",
        "ok": True,
        "fp16": fp16,
        "abi": abi,
        "preflight": preflight,
        "training_path_enabled": True,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
