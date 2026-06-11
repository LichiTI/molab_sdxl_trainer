"""LAB CLI for Newbie transformer TensorRT spike steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _prepare_path() -> None:
    backend_root = Path(__file__).resolve().parents[3]
    project_root = backend_root.parent
    for item in (str(project_root), str(backend_root)):
        if item not in sys.path:
            sys.path.insert(0, item)


_prepare_path()

from core.base_model_tensorrt.newbie_export import (  # noqa: E402
    NewbieStaticShape,
    default_newbie_checkpoint,
    default_newbie_config_path,
    inspect_newbie_safetensors,
    run_newbie_synthetic_forward,
)
from core.base_model_tensorrt.newbie_engine import (  # noqa: E402
    build_newbie_tensorrt_engine,
    compare_newbie_tensorrt_parity,
    export_newbie_static_onnx,
)
from core.base_model_tensorrt.newbie_diagnostics import diagnose_newbie_tensorrt_windows  # noqa: E402
from core.base_model_tensorrt.newbie_tap_diagnostics import compare_newbie_tensorrt_tap_parity  # noqa: E402
from core.base_model_tensorrt.newbie_offline_parity import (  # noqa: E402
    compare_newbie_output_artifacts,
    write_newbie_tensorrt_output_artifact,
    write_newbie_torch_output_artifact,
)
from core.base_model_tensorrt.generation_preflight import preflight_newbie_tensorrt_generation_request  # noqa: E402
from core.base_model_tensorrt.runtime_gate import build_static_transformer_runtime_gate  # noqa: E402
from core.base_model_tensorrt.runtime_smoke import run_newbie_static_tensorrt_runtime_smoke  # noqa: E402
from core.base_model_tensorrt.runtime_adapter import StaticTransformerRuntimeSpec  # noqa: E402
from backend.core.contracts import GenerationRequest  # noqa: E402


def _shape_from_args(args: argparse.Namespace) -> NewbieStaticShape:
    return NewbieStaticShape(
        batch=args.batch,
        latent_channels=args.latent_channels,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        tokens=args.tokens,
        hidden_dim=args.hidden_dim,
        pooled_dim=args.pooled_dim,
        patch_size=args.patch_size,
    )


def _write_or_print(result: dict[str, object], output: str) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text)


def _generation_request_from_args(args: argparse.Namespace) -> GenerationRequest:
    if args.generation_request_json:
        data = json.loads(Path(args.generation_request_json).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit("--generation-request-json must contain a JSON object")
        return GenerationRequest.model_validate(data)
    return GenerationRequest(
        arch="newbie",
        prompt=args.prompt or "test",
        negative_prompt=args.negative_prompt or "",
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        sampler=args.sampler,
        batch_size=args.generation_batch_size,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Newbie transformer TensorRT spike steps.")
    parser.add_argument("action", choices=(
        "inspect",
        "forward",
        "export-onnx",
        "build-engine",
        "parity",
        "tap-parity",
        "diagnose-windows",
        "torch-output",
        "trt-output",
        "compare-outputs",
        "runtime-gate",
        "runtime-smoke",
        "generation-preflight",
    ))
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--onnx-path", default="")
    parser.add_argument("--engine-path", default="")
    parser.add_argument("--torch-artifact-path", default="")
    parser.add_argument("--tensorrt-artifact-path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--layers", default="0", help="Layer list or range, for example 0 or 0-1")
    parser.add_argument("--tap-layer", type=int, default=-1, help="Layer index to expose debug tap outputs for export-onnx/tap-parity.")
    parser.add_argument("--windows", default="", help="Semicolon-separated layer windows for diagnose-windows, for example 0-7;0-15;0-35")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "fp32", "float16", "fp16", "bfloat16", "bf16"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=2304)
    parser.add_argument("--pooled-dim", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--fp32-layer-policy", default="none", choices=("none", "all", "sensitive", "sensitive-projections", "sensitive_projections", "sensitive-block-matmul", "sensitive_block_matmul", "norm", "non-matmul", "non_matmul"))
    parser.add_argument("--fp32-layer-name-filter", action="append", default=[], help="Additional TensorRT layer-name substring to force to FP32; repeatable LAB-only diagnostic option.")
    parser.add_argument("--external-data", action="store_true", help="Write ONNX weights as external data for very large exports.")
    parser.add_argument("--no-reuse-existing", action="store_true", help="Force diagnose-windows to rebuild ONNX/engine artifacts.")
    parser.add_argument("--cleanup-artifacts", action="store_true", help="Remove each diagnose-windows output subdirectory after its report is captured.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop diagnose-windows at the first failed window.")
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--include-context-refiner", action="store_true")
    parser.add_argument("--include-noise-refiner", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--generation-request-json", default="", help="GenerationRequest JSON for generation-preflight.")
    parser.add_argument("--prompt", default="test", help="Prompt used by generation-preflight when no request JSON is supplied.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt used by generation-preflight.")
    parser.add_argument("--width", type=int, default=1024, help="Generation width for generation-preflight.")
    parser.add_argument("--height", type=int, default=1024, help="Generation height for generation-preflight.")
    parser.add_argument("--steps", type=int, default=20, help="Generation steps for generation-preflight.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Guidance scale for generation-preflight.")
    parser.add_argument("--sampler", default="euler", help="Sampler name for generation-preflight.")
    parser.add_argument("--generation-batch-size", type=int, default=1, help="Generation batch size for generation-preflight.")
    parser.add_argument("--vae-scale-factor", type=int, default=16, help="Latent scale factor assumed by generation-preflight.")
    parser.add_argument("--positive-tokens", type=int, default=512, help="Positive conditioning token count assumed by generation-preflight.")
    parser.add_argument("--negative-tokens", type=int, default=0, help="Negative conditioning token count assumed by generation-preflight.")
    parser.add_argument("--cfg-strategy", default="separate-calls", choices=("separate-calls", "separate_calls", "blocked", "concat-batch", "concat_batch"), help="CFG strategy for generation-preflight; route 2 uses separate-calls.")
    args = parser.parse_args()

    checkpoint = args.checkpoint_path or str(default_newbie_checkpoint(args.model_root))
    config_path = args.config_path or str(default_newbie_config_path(args.model_root))
    shape = _shape_from_args(args)
    if args.action == "inspect":
        result = inspect_newbie_safetensors(checkpoint, config_path=config_path)
    elif args.action == "forward":
        result = run_newbie_synthetic_forward(
            checkpoint_path=checkpoint,
            model_root=args.model_root,
            config_path=config_path,
            layer_indices=args.layers,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            include_context_refiner=args.include_context_refiner,
            include_noise_refiner=args.include_noise_refiner,
            strict=args.strict,
        )
    elif args.action == "export-onnx":
        result = export_newbie_static_onnx(
            checkpoint_path=checkpoint,
            model_root=args.model_root,
            config_path=config_path,
            output_path=args.output_path,
            output_dir=args.output_dir,
            layer_indices=args.layers,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            opset=args.opset,
            external_data=args.external_data,
            tap_layer_index=args.tap_layer if args.tap_layer >= 0 else None,
            strict=args.strict,
        )
    elif args.action == "build-engine":
        result = build_newbie_tensorrt_engine(
            onnx_path=args.onnx_path,
            output_path=args.engine_path or args.output_path,
            output_dir=args.output_dir,
            layer_indices=args.layers,
            shape=shape,
            opset=args.opset,
            precision=args.precision,
            workspace_mb=args.workspace_mb,
            fp32_layer_policy=args.fp32_layer_policy,
            fp32_layer_name_filters=args.fp32_layer_name_filter,
        )
    elif args.action == "diagnose-windows":
        result = diagnose_newbie_tensorrt_windows(
            checkpoint_path=checkpoint,
            model_root=args.model_root,
            config_path=config_path,
            output_dir=args.output_dir,
            windows=args.windows,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            opset=args.opset,
            precision=args.precision,
            fp32_layer_policy=args.fp32_layer_policy,
            workspace_mb=args.workspace_mb,
            external_data=args.external_data,
            reuse_existing=not args.no_reuse_existing,
            cleanup_artifacts=args.cleanup_artifacts,
            stop_on_failure=args.stop_on_failure,
        )
    elif args.action == "parity":
        result = compare_newbie_tensorrt_parity(
            checkpoint_path=checkpoint,
            model_root=args.model_root,
            config_path=config_path,
            engine_path=args.engine_path,
            output_dir=args.output_dir,
            layer_indices=args.layers,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            opset=args.opset,
            precision=args.precision,
        )
    elif args.action == "tap-parity":
        if args.tap_layer < 0:
            raise SystemExit("tap-parity requires --tap-layer")
        result = compare_newbie_tensorrt_tap_parity(
            checkpoint_path=checkpoint,
            model_root=args.model_root,
            config_path=config_path,
            engine_path=args.engine_path,
            output_dir=args.output_dir,
            layer_indices=args.layers,
            shape=shape,
            tap_layer_index=args.tap_layer,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            opset=args.opset,
            precision=args.precision,
        )
    elif args.action == "torch-output":
        artifact_path = args.output_path or args.torch_artifact_path
        if not artifact_path:
            raise SystemExit("torch-output requires --output-path or --torch-artifact-path")
        result = write_newbie_torch_output_artifact(
            artifact_path=artifact_path,
            checkpoint_path=checkpoint,
            model_root=args.model_root,
            config_path=config_path,
            layer_indices=args.layers,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
        )
    elif args.action == "trt-output":
        artifact_path = args.output_path or args.tensorrt_artifact_path
        if not artifact_path:
            raise SystemExit("trt-output requires --output-path or --tensorrt-artifact-path")
        result = write_newbie_tensorrt_output_artifact(
            artifact_path=artifact_path,
            engine_path=args.engine_path,
            output_dir=args.output_dir,
            layer_indices=args.layers,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            opset=args.opset,
            precision=args.precision,
        )
    elif args.action == "runtime-gate":
        result = build_static_transformer_runtime_gate(
            family="newbie",
            engine_path=args.engine_path or args.output_path,
            output_dir=args.output_dir,
            layer_indices=args.layers,
            shape=shape,
            precision=args.precision,
            opset=args.opset,
        )
    elif args.action == "runtime-smoke":
        result = run_newbie_static_tensorrt_runtime_smoke(
            engine_path=args.engine_path,
            output_dir=args.output_dir,
            layer_indices=args.layers,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            precision=args.precision,
            opset=args.opset,
            artifact_path=args.output_path,
        )
    elif args.action == "generation-preflight":
        spec = StaticTransformerRuntimeSpec.from_newbie_shape(
            engine_path=args.engine_path or args.output_path,
            layer_indices=args.layers,
            shape=shape,
            precision=args.precision,
        )
        request = _generation_request_from_args(args)
        result = preflight_newbie_tensorrt_generation_request(
            request,
            spec,
            vae_scale_factor=args.vae_scale_factor,
            positive_tokens=args.positive_tokens,
            negative_tokens=args.negative_tokens,
            cfg_strategy=args.cfg_strategy,
        ).to_dict()
        result.update({"success": True, "kind": "newbie_tensorrt_generation_preflight"})
    else:
        if not args.torch_artifact_path or not args.tensorrt_artifact_path:
            raise SystemExit("compare-outputs requires --torch-artifact-path and --tensorrt-artifact-path")
        result = compare_newbie_output_artifacts(
            torch_artifact_path=args.torch_artifact_path,
            tensorrt_artifact_path=args.tensorrt_artifact_path,
        )
    _write_or_print(result, args.output)
    return 0 if bool(result.get("success", result.get("available", False))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
