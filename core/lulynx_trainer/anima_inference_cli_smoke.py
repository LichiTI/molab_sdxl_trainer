# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for anima_inference_cli.py (Phase 9.4 / #120).

Verifies pure helpers without loading real models:
  1. _resolve_dtype maps strings to torch.dtype
  2. _detect_arch picks newbie vs anima from directory layout
  3. argparse exposes all required flags
"""

from __future__ import annotations

import sys
import os
import json
import importlib.util
import tempfile
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_BACKEND_ROOT = os.path.join(_REPO_ROOT, "backend")
for _path in (_REPO_ROOT, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_cli():
    """Import the CLI module without triggering its argparse main()."""
    spec = importlib.util.spec_from_file_location(
        "core.lulynx_trainer.anima_inference_cli",
        os.path.join(_HERE, "anima_inference_cli.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["core.lulynx_trainer.anima_inference_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_resolve_dtype_known_aliases():
    cli = _load_cli()
    assert cli._resolve_dtype("fp16") is torch.float16
    assert cli._resolve_dtype("BF16") is torch.bfloat16
    assert cli._resolve_dtype("float32") is torch.float32
    print("PASS: _resolve_dtype maps known aliases")


def test_resolve_dtype_unknown_raises():
    cli = _load_cli()
    try:
        cli._resolve_dtype("garbage")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "garbage" in str(e).lower() or "unknown" in str(e).lower()
    print("PASS: _resolve_dtype rejects unknown dtypes")


def test_detect_arch_newbie_layout():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "transformer").mkdir()
        (root / "clip_model").mkdir()
        arch = cli._detect_arch(str(root))
        assert arch == "newbie", f"expected newbie, got {arch}"
    print("PASS: _detect_arch identifies newbie from directory layout")


def test_detect_arch_anima_default():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tmp:
        # bare dir with no transformer/clip_model -> defaults to anima
        arch = cli._detect_arch(tmp)
        assert arch == "anima"
    print("PASS: _detect_arch defaults to anima when layout uncertain")


def test_detect_arch_reads_config_model_arch():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text(json.dumps({"model_arch": "newbie"}))
        arch = cli._detect_arch(str(root))
        assert arch == "newbie"
    print("PASS: _detect_arch honors model_arch from config.json")


def test_main_function_exists():
    cli = _load_cli()
    assert hasattr(cli, "main")
    assert callable(cli.main)
    print("PASS: CLI exposes main() entry point")


def test_cli_args_build_generation_request():
    cli = _load_cli()
    args = cli.build_parser().parse_args([
        "--model_path", "models/anima",
        "--adapter_path", "models/lora/demo.safetensors",
        "--output_dir", "output/inference",
        "--prompt", "a cinematic cat",
        "--negative_prompt", "low quality",
        "--width", "768",
        "--height", "512",
        "--steps", "12",
        "--guidance_scale", "4.5",
        "--seed", "1234",
        "--sampler_name", "dpm_solver",
        "--dtype", "fp16",
        "--device", "cuda:0",
        "--batch_size", "2",
        "--arch", "anima",
    ])
    request = cli.build_generation_request_from_args(args)

    assert request.schema_id == "generation.image"
    assert request.compat_mode is True
    assert request.metadata.source == "cli"
    assert request.model_path == "models/anima"
    assert request.adapter_path == "models/lora/demo.safetensors"
    assert request.output_dir == "output/inference"
    assert request.prompt == "a cinematic cat"
    assert request.negative_prompt == "low quality"
    assert request.width == 768
    assert request.height == 512
    assert request.steps == 12
    assert request.guidance_scale == 4.5
    assert request.seed == 1234
    assert request.sampler == "dpm_solver"
    assert request.dtype == "fp16"
    assert request.device == "cuda:0"
    assert request.batch_size == 2
    assert request.arch == "anima"
    print("PASS: CLI args normalize into GenerationRequest")


def test_request_json_dry_run_uses_generation_runner():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        request_path = root / "generation_request.json"
        result_path = root / "result.json"
        request_path.write_text(json.dumps({
            "schema_id": "generation.image",
            "prompt": "dry request",
            "output_dir": "output/inference",
            "output_name": "preview",
            "width": 512,
            "height": 512,
            "dry_run": True,
        }), encoding="utf-8")

        args = cli.build_parser().parse_args([
            "--request_json", str(request_path),
            "--dry_run",
            "--run_result_json", str(result_path),
        ])
        request = cli.build_generation_request_from_args(args)
        result = cli.run_generation_request_dry_run(request)
        cli.write_generation_run_result(result, result_path)
        saved = json.loads(result_path.read_text(encoding="utf-8"))

        assert result.status == "succeeded"
        assert saved["data"]["runner_id"] == "generation.dry-run"
        assert saved["artifacts"][0]["files"][0]["path"].endswith("preview.png")
    print("PASS: request JSON can dry-run through GenerationRequest runner")


def test_newbie_tensorrt_opt_in_replaces_unet_with_adapter():
    cli = _load_cli()
    request = cli.GenerationRequest(
        arch="newbie",
        prompt="test",
        negative_prompt="bad",
        width=1024,
        height=1024,
        steps=2,
        guidance_scale=5.0,
        newbie_tensorrt_enabled=True,
        newbie_tensorrt_engine_path="H:/tmp/newbie.engine",
        newbie_tensorrt_vae_scale_factor=16,
        newbie_tensorrt_tokens=512,
    )
    model = type("Model", (), {"unet": object()})()

    updated, report = cli._maybe_apply_newbie_tensorrt_transformer(model, request, "newbie")

    assert updated is model
    assert report["enabled"] is True
    assert report["latent_scale_factor"] == 16
    assert report["preflight"]["ok"] is True
    assert report["preflight"]["engine_calls_per_step"] == 2
    assert report["preflight"]["generation_path_enabled"] is False
    assert type(model.unet).__name__ == "NewbieTensorRtTransformerAdapter"
    print("PASS: Newbie TensorRT opt-in replaces unet with adapter")


def test_newbie_tensorrt_opt_in_requires_engine_path():
    cli = _load_cli()
    request = cli.GenerationRequest(
        arch="newbie",
        prompt="test",
        width=1024,
        height=1024,
        newbie_tensorrt_enabled=True,
    )
    model = type("Model", (), {"unet": object()})()

    try:
        cli._maybe_apply_newbie_tensorrt_transformer(model, request, "newbie")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "engine_path" in str(exc)
    print("PASS: Newbie TensorRT opt-in requires engine path")


def test_newbie_tensorrt_opt_in_blocks_shape_mismatch():
    cli = _load_cli()
    request = cli.GenerationRequest(
        arch="newbie",
        prompt="test",
        width=512,
        height=512,
        guidance_scale=1.0,
        newbie_tensorrt_engine_path="H:/tmp/newbie.engine",
        newbie_tensorrt_vae_scale_factor=16,
        newbie_tensorrt_tokens=512,
    )
    model = type("Model", (), {"unet": object()})()

    try:
        cli._maybe_apply_newbie_tensorrt_transformer(model, request, "newbie")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "static_shape_mismatch" in str(exc)
    print("PASS: Newbie TensorRT opt-in blocks shape mismatch")


if __name__ == "__main__":
    test_resolve_dtype_known_aliases()
    test_resolve_dtype_unknown_raises()
    test_detect_arch_newbie_layout()
    test_detect_arch_anima_default()
    test_detect_arch_reads_config_model_arch()
    test_main_function_exists()
    test_cli_args_build_generation_request()
    test_request_json_dry_run_uses_generation_runner()
    test_newbie_tensorrt_opt_in_replaces_unet_with_adapter()
    test_newbie_tensorrt_opt_in_requires_engine_path()
    test_newbie_tensorrt_opt_in_blocks_shape_mismatch()
    print("\nAll anima_inference_cli smoke tests passed!")
