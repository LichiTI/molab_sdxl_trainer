# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""TensorRT runtime inference for the image upscaler.

TensorRT's Python package does not allocate CUDA buffers for us.  The heavy
runtime already carries PyTorch, so this module uses torch CUDA tensors as the
input/output buffers and passes their device pointers to TensorRT.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _torch_dtype_for_trt(trt: Any, dtype: Any) -> Any:
    import torch

    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT8:
        return torch.int8
    if hasattr(trt.DataType, "BOOL") and dtype == trt.DataType.BOOL:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT tensor dtype: {dtype}")


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tuple(shape))


class TensorRtUpscalerEngine:
    """Run a TensorRT upscaler engine against RGB image tiles."""

    def __init__(self, engine_path: str, *, scale: int = 4) -> None:
        import tensorrt as trt  # type: ignore
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT upscaling requires CUDA, but torch.cuda is not available")

        src = Path(engine_path)
        if not src.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.trt = trt
        self.torch = torch
        self.scale = max(1, int(scale or 4))
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(src.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.input_name, self.output_name = self._find_io_names()
        self.input_dtype = _torch_dtype_for_trt(trt, self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = _torch_dtype_for_trt(trt, self.engine.get_tensor_dtype(self.output_name))
        self.input_shape = _shape_tuple(self.engine.get_tensor_shape(self.input_name))
        self.stream = torch.cuda.Stream()

    def _find_io_names(self) -> tuple[str, str]:
        trt = self.trt
        inputs: list[str] = []
        outputs: list[str] = []
        for index in range(int(self.engine.num_io_tensors)):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"Expected one TensorRT input and one output, got {inputs} -> {outputs}")
        return inputs[0], outputs[0]

    def _static_hw(self) -> tuple[int | None, int | None]:
        if len(self.input_shape) != 4:
            return None, None
        h = self.input_shape[2]
        w = self.input_shape[3]
        return (h if h > 0 else None), (w if w > 0 else None)

    def _effective_tile_size(self, requested_tile_size: int) -> int:
        static_h, static_w = self._static_hw()
        limit = min(value for value in (static_h, static_w) if value) if (static_h or static_w) else None
        requested = max(16, int(requested_tile_size or 256))
        return min(requested, limit) if limit else requested

    def _infer_tile(self, tile: Any) -> Any:
        torch = self.torch
        tile = tile.contiguous().to(device="cuda", dtype=self.input_dtype)
        actual_h = int(tile.shape[2])
        actual_w = int(tile.shape[3])
        run_tile = tile
        static_h, static_w = self._static_hw()
        if static_h or static_w:
            target_h = static_h or actual_h
            target_w = static_w or actual_w
            if actual_h > target_h or actual_w > target_w:
                raise RuntimeError(
                    f"Tile {actual_w}x{actual_h} exceeds static TensorRT input {target_w}x{target_h}"
                )
            pad_h = target_h - actual_h
            pad_w = target_w - actual_w
            if pad_h or pad_w:
                run_tile = torch.nn.functional.pad(run_tile, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            self.context.set_input_shape(self.input_name, tuple(run_tile.shape))
            unresolved = self.context.infer_shapes()
            if unresolved:
                raise RuntimeError(f"TensorRT could not infer shapes for: {unresolved}")

        output_shape = _shape_tuple(self.context.get_tensor_shape(self.output_name))
        if any(dim <= 0 for dim in output_shape):
            output_shape = (1, int(run_tile.shape[1]), int(run_tile.shape[2]) * self.scale, int(run_tile.shape[3]) * self.scale)
        output = torch.empty(output_shape, device="cuda", dtype=self.output_dtype)

        self.context.set_tensor_address(self.input_name, int(run_tile.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(int(self.stream.cuda_stream))
            if not ok:
                raise RuntimeError("TensorRT execution failed")
        current_stream.wait_stream(self.stream)
        return output[:, :, : actual_h * self.scale, : actual_w * self.scale]

    def upscale_image(
        self,
        img_path: str,
        output_path: str,
        *,
        fmt: str = "png",
        tile_size: int = 256,
        tile_pad: int = 0,
    ) -> str:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        torch = self.torch
        with Image.open(img_path) as src_img:
            img = src_img.convert("RGB")
            img_np = np.array(img)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to("cuda")
        output_t = self.tile_process(img_t, tile_size=tile_size, tile_pad=tile_pad)
        output = output_t.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output).save(str(dst), format=fmt)
        return str(dst)

    def tile_process(self, img: Any, *, tile_size: int, tile_pad: int) -> Any:
        torch = self.torch
        batch, channel, height, width = img.shape
        if int(batch) != 1:
            raise RuntimeError("TensorRT upscaler only supports batch size 1")
        tile_size = self._effective_tile_size(tile_size)
        static_h, static_w = self._static_hw()
        if static_h or static_w:
            tile_pad = 0
        else:
            tile_pad = max(0, int(tile_pad or 0))

        output = torch.empty((batch, channel, height * self.scale, width * self.scale), device="cuda", dtype=torch.float32)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                input_start_y = y * tile_size
                input_end_y = min(input_start_y + tile_size, height)
                input_start_x = x * tile_size
                input_end_x = min(input_start_x + tile_size, width)

                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)

                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                output_tile = self._infer_tile(input_tile).float()

                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale

                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * self.scale
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * self.scale

                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                    :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
                ]

        return output


def upscale_image_with_tensorrt(
    *,
    engine_path: str,
    input_path: str,
    output_path: str,
    scale: int = 4,
    fmt: str = "png",
    tile_size: int = 256,
    tile_pad: int = 0,
) -> dict[str, Any]:
    import time

    started = time.perf_counter()
    engine = TensorRtUpscalerEngine(engine_path, scale=scale)
    result_path = engine.upscale_image(input_path, output_path, fmt=fmt, tile_size=tile_size, tile_pad=tile_pad)
    dst = Path(result_path)
    return {
        "schema_version": 1,
        "kind": "upscaler_tensorrt_inference",
        "success": True,
        "engine_path": str(Path(engine_path)),
        "input_path": str(Path(input_path)),
        "output_path": str(dst),
        "scale": int(scale),
        "format": fmt,
        "tile_size": engine._effective_tile_size(tile_size),
        "tile_pad": int(tile_pad or 0),
        "bytes": dst.stat().st_size if dst.exists() else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
