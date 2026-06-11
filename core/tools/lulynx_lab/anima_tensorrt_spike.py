"""LAB CLI for Anima transformer TensorRT export experiments."""

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

from core.base_model_tensorrt.anima_export import (  # noqa: E402
    AnimaStaticShape,
    export_anima_static_onnx,
    run_anima_synthetic_forward,
)
from core.base_model_tensorrt.anima_engine import (  # noqa: E402
    build_anima_tensorrt_engine,
    compare_anima_tensorrt_parity,
)


def _shape_from_args(args: argparse.Namespace) -> AnimaStaticShape:
    return AnimaStaticShape(
        batch=args.batch,
        latent_channels=args.latent_channels,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        tokens=args.tokens,
        context_dim=args.context_dim,
    )


def _write_or_print(result: dict[str, object], output: str) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Anima transformer TensorRT spike steps.")
    parser.add_argument("action", choices=("forward", "export-onnx", "build-engine", "parity"))
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--onnx-path", default="")
    parser.add_argument("--engine-path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--blocks", default="0", help="Block list or range, for example 0 or 0-1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "fp32", "float16", "fp16", "bfloat16", "bf16"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--context-dim", type=int, default=1024)
    parser.add_argument("--disable-mmap", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    shape = _shape_from_args(args)
    if args.action == "forward":
        result = run_anima_synthetic_forward(
            checkpoint_path=args.checkpoint_path,
            model_root=args.model_root,
            block_indices=args.blocks,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            disable_mmap=args.disable_mmap,
        )
    elif args.action == "export-onnx":
        result = export_anima_static_onnx(
            checkpoint_path=args.checkpoint_path,
            model_root=args.model_root,
            output_path=args.output_path,
            output_dir=args.output_dir,
            block_indices=args.blocks,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            opset=args.opset,
            disable_mmap=args.disable_mmap,
            strict=args.strict,
        )
    elif args.action == "build-engine":
        result = build_anima_tensorrt_engine(
            onnx_path=args.onnx_path,
            output_path=args.engine_path or args.output_path,
            output_dir=args.output_dir,
            block_indices=args.blocks,
            shape=shape,
            opset=args.opset,
            precision=args.precision,
            workspace_mb=args.workspace_mb,
        )
    else:
        result = compare_anima_tensorrt_parity(
            checkpoint_path=args.checkpoint_path,
            model_root=args.model_root,
            engine_path=args.engine_path,
            output_dir=args.output_dir,
            block_indices=args.blocks,
            shape=shape,
            device=args.device,
            dtype_name=args.dtype,
            seed=args.seed,
            opset=args.opset,
            precision=args.precision,
            disable_mmap=args.disable_mmap,
        )
    _write_or_print(result, args.output)
    return 0 if bool(result.get("success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
