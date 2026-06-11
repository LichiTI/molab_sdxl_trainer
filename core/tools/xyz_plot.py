"""Manual XYZ plot generator for the toolbox API.

The generator is intentionally explicit and synchronous. It only runs when the
user calls the toolbox endpoint and never participates in training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _parse_axis(axis: dict[str, Any] | None) -> tuple[str | None, list[Any]]:
    if not axis:
        return None, [None]
    name = axis.get("name") or axis.get("id")
    values = axis.get("values") or []
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    if not name or not values:
        return None, [None]
    return str(name), list(values)


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_pipeline(model_path: str, model_type: str, dtype: torch.dtype):
    model_type = model_type.lower().strip()
    if model_type in {"sdxl", "xl"}:
        from diffusers import StableDiffusionXLPipeline as Pipe
    else:
        from diffusers import StableDiffusionPipeline as Pipe

    path = Path(model_path)
    if path.is_file():
        return Pipe.from_single_file(str(path), torch_dtype=dtype)
    return Pipe.from_pretrained(str(path), torch_dtype=dtype)


def generate_xyz_plot(
    model_path: str,
    output_path: str,
    base_params: dict[str, Any],
    x_axis: dict[str, Any],
    y_axis: dict[str, Any] | None = None,
    z_axis: dict[str, Any] | None = None,
    model_type: str = "sdxl",
    vae_path: str | None = None,
    device: str = "cuda",
    dtype_name: str = "fp16",
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dtype_name == "fp16" and device == "cuda" else torch.float32
    pipe = _load_pipeline(model_path, model_type, dtype)
    if vae_path:
        from diffusers import AutoencoderKL

        vae_src = Path(vae_path)
        pipe.vae = AutoencoderKL.from_single_file(str(vae_src), torch_dtype=dtype) if vae_src.is_file() else AutoencoderKL.from_pretrained(str(vae_src), torch_dtype=dtype)
    pipe.to(device)

    x_name, x_values = _parse_axis(x_axis)
    y_name, y_values = _parse_axis(y_axis)
    z_name, z_values = _parse_axis(z_axis)
    if not x_name:
        raise ValueError("x_axis requires name/id and values")

    files: list[str] = []
    for z_index, z_value in enumerate(z_values):
        rows = []
        for y_value in y_values:
            images = []
            for x_value in x_values:
                params = dict(base_params)
                params[x_name] = _coerce_value(x_value)
                if y_name:
                    params[y_name] = _coerce_value(y_value)
                if z_name:
                    params[z_name] = _coerce_value(z_value)
                generator = torch.manual_seed(int(params.get("seed", 42)))
                images.append(pipe(
                    prompt=params.get("prompt", ""),
                    negative_prompt=params.get("negative_prompt", ""),
                    num_inference_steps=int(params.get("steps", 20)),
                    guidance_scale=float(params.get("cfg", 7.0)),
                    width=int(params.get("width", 1024)),
                    height=int(params.get("height", 1024)),
                    generator=generator,
                ).images[0])
            rows.append(images)

        grid = _stitch(rows, x_values, y_values if y_name else [""], x_name, y_name or "")
        save_path = out if not z_name else out.with_name(f"{out.stem}_z{z_index}{out.suffix or '.png'}")
        grid.save(str(save_path))
        files.append(str(save_path))

    return {"success": True, "output_paths": files, "x_axis": x_name, "y_axis": y_name, "z_axis": z_name}


def _stitch(rows, x_labels, y_labels, x_title, y_title):
    from PIL import Image, ImageDraw, ImageFont

    image_w, image_h = rows[0][0].size
    label_w = 180 if y_title else 24
    label_h = 48
    canvas = Image.new("RGB", (label_w + image_w * len(rows[0]), label_h + image_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for col, label in enumerate(x_labels):
        draw.text((label_w + col * image_w + 8, 14), f"{x_title}: {label}", fill="black", font=font)
    for row_idx, row in enumerate(rows):
        if y_title:
            draw.text((8, label_h + row_idx * image_h + 12), f"{y_title}: {y_labels[row_idx]}", fill="black", font=font)
        for col_idx, img in enumerate(row):
            canvas.paste(img, (label_w + col_idx * image_w, label_h + row_idx * image_h))
    return canvas

