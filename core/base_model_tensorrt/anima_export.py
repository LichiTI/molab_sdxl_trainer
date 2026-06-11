from __future__ import annotations

import importlib.util
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .family_adapters import ANIMA_ADAPTER, MODELS_ROOT, resolve_family_components


@dataclass(frozen=True)
class AnimaStaticShape:
    batch: int = 1
    latent_channels: int = 16
    latent_height: int = 4
    latent_width: int = 4
    tokens: int = 4
    context_dim: int = 1024
    patch_size: int = 2

    @property
    def visual_tokens(self) -> int:
        return (self.latent_height // self.patch_size) * (self.latent_width // self.patch_size)

    def validate(self) -> None:
        values = {
            "batch": self.batch,
            "latent_channels": self.latent_channels,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "tokens": self.tokens,
            "context_dim": self.context_dim,
            "patch_size": self.patch_size,
        }
        for key, value in values.items():
            if int(value) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.latent_channels != 16:
            raise ValueError("Anima native DiT expects 16 latent channels")
        if self.latent_height % self.patch_size or self.latent_width % self.patch_size:
            raise ValueError("Anima latent height/width must be divisible by patch_size")

    def input_signature(self) -> dict[str, Any]:
        self.validate()
        return {
            "sample": [self.batch, self.latent_channels, self.latent_height, self.latent_width],
            "timestep": [self.batch],
            "encoder_hidden_states": [self.batch, self.tokens, self.context_dim],
            "padding_mask": [self.batch, 1, self.latent_height, self.latent_width],
            "visual_tokens": self.visual_tokens,
            "patch_size": self.patch_size,
        }

    def to_dict(self) -> dict[str, int]:
        return {
            "batch": self.batch,
            "latent_channels": self.latent_channels,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "tokens": self.tokens,
            "context_dim": self.context_dim,
            "patch_size": self.patch_size,
        }


def parse_block_indices(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None or value == "":
        return (0,)
    if isinstance(value, str):
        items: list[int] = []
        for part in value.replace(";", ",").split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                if end < start:
                    raise ValueError(f"Invalid descending block range: {token}")
                items.extend(range(start, end + 1))
            else:
                items.append(int(token))
    else:
        items = [int(item) for item in value]
    if not items:
        return (0,)
    deduped: list[int] = []
    seen: set[int] = set()
    for item in items:
        if item < 0:
            raise ValueError("Anima block indices must be non-negative")
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return tuple(deduped)


def default_anima_checkpoint(model_root: str | Path = "") -> Path:
    resolved = resolve_family_components(ANIMA_ADAPTER, model_root)
    return Path(str(resolved["export_target_path"]))


def default_anima_onnx_path(
    *,
    output_dir: str | Path = "",
    shape: AnimaStaticShape | None = None,
    block_indices: Sequence[int] = (0,),
    opset: int = 18,
) -> Path:
    shape = shape or AnimaStaticShape()
    blocks = parse_block_indices(block_indices)
    block_label = _block_label(blocks)
    root = Path(output_dir) if str(output_dir or "").strip() else MODELS_ROOT / "anima" / "tensorrt_spike"
    name = (
        f"anima_transformer_{block_label}_"
        f"{shape.latent_height}x{shape.latent_width}_tok{shape.tokens}_op{int(opset)}.onnx"
    )
    return root / name


def default_anima_engine_path(
    *,
    output_dir: str | Path = "",
    shape: AnimaStaticShape | None = None,
    block_indices: Sequence[int] = (0,),
    opset: int = 18,
    precision: str = "fp16",
) -> Path:
    onnx_path = default_anima_onnx_path(
        output_dir=output_dir,
        shape=shape,
        block_indices=block_indices,
        opset=opset,
    )
    suffix = _precision_suffix(precision, default="fp16")
    return onnx_path.with_name(f"{onnx_path.stem}_{suffix}.engine")


def _precision_suffix(value: str | None, *, default: str) -> str:
    key = str(value or default).strip().lower().replace("-", "_")
    aliases = {"float32": "fp32", "float16": "fp16", "half": "fp16", "bfloat16": "bf16"}
    return aliases.get(key, key)


def create_anima_export_wrapper(model: Any) -> Any:
    import torch

    class _AnimaTensorRtExportWrapper(torch.nn.Module):
        def __init__(self, inner: Any):
            super().__init__()
            self.inner = inner

        def forward(
            self,
            sample: Any,
            timestep: Any,
            encoder_hidden_states: Any,
            padding_mask: Any,
        ) -> Any:
            result = self.inner(sample, timestep, encoder_hidden_states, padding_mask)
            return result.sample

    return _AnimaTensorRtExportWrapper(model)


def create_anima_static_export_wrapper(model: Any, shape: AnimaStaticShape) -> Any:
    import torch

    shape.validate()

    class _AnimaStaticTensorRtExportWrapper(torch.nn.Module):
        def __init__(self, inner: Any):
            super().__init__()
            self.inner = inner

        def _patchify(self, sample: Any, padding_mask: Any) -> Any:
            mask = padding_mask.to(device=sample.device, dtype=sample.dtype)
            merged = torch.cat([sample, mask], dim=1)
            batch = shape.batch
            patch_h = shape.latent_height // shape.patch_size
            patch_w = shape.latent_width // shape.patch_size
            channels = shape.latent_channels + 1
            patches = merged.reshape(
                batch,
                channels,
                patch_h,
                shape.patch_size,
                patch_w,
                shape.patch_size,
            )
            patches = patches.permute(0, 2, 4, 1, 3, 5)
            return patches.reshape(batch, patch_h * patch_w, channels * shape.patch_size * shape.patch_size)

        def _unpatchify(self, patch_values: Any) -> Any:
            batch = shape.batch
            patch_h = shape.latent_height // shape.patch_size
            patch_w = shape.latent_width // shape.patch_size
            patches = patch_values.reshape(
                batch,
                patch_h,
                patch_w,
                shape.latent_channels,
                shape.patch_size,
                shape.patch_size,
            )
            patches = patches.permute(0, 3, 1, 4, 2, 5)
            return patches.reshape(batch, shape.latent_channels, shape.latent_height, shape.latent_width)

        def forward(
            self,
            sample: Any,
            timestep: Any,
            encoder_hidden_states: Any,
            padding_mask: Any,
        ) -> Any:
            patches = self._patchify(sample, padding_mask)
            x = self.inner.net.x_embedder.proj(patches)
            context = encoder_hidden_states.to(device=x.device, dtype=x.dtype)
            timesteps = timestep.to(device=x.device, dtype=x.dtype)
            emb, adaln_lora = self.inner.net.t_embedder(timesteps)
            emb = self.inner.net.t_embedding_norm(emb)
            x = self.inner._run_blocks(x, emb, context, adaln_lora)
            patch_values = self.inner.net.final_layer(x, emb, adaln_lora)
            return self._unpatchify(patch_values)

    return _AnimaStaticTensorRtExportWrapper(model)


def create_anima_synthetic_inputs(
    *,
    shape: AnimaStaticShape | None = None,
    device: str = "cpu",
    dtype_name: str = "float32",
    seed: int = 1337,
) -> tuple[Any, Any, Any, Any]:
    import torch

    shape = shape or AnimaStaticShape()
    shape.validate()
    dtype = _torch_dtype(torch, dtype_name)
    target_device = _target_device(torch, device)
    generator_device = "cuda" if str(target_device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    sample = torch.randn(
        shape.batch,
        shape.latent_channels,
        shape.latent_height,
        shape.latent_width,
        device=target_device,
        dtype=dtype,
        generator=generator,
    )
    timestep = torch.full((shape.batch,), 250.0, device=target_device, dtype=dtype)
    context = torch.randn(
        shape.batch,
        shape.tokens,
        shape.context_dim,
        device=target_device,
        dtype=dtype,
        generator=generator,
    )
    padding_mask = torch.zeros(
        shape.batch,
        1,
        shape.latent_height,
        shape.latent_width,
        device=target_device,
        dtype=dtype,
    )
    return sample, timestep, context, padding_mask


def run_anima_synthetic_forward(
    *,
    checkpoint_path: str | Path = "",
    model_root: str | Path = "",
    block_indices: str | Sequence[int] = (0,),
    shape: AnimaStaticShape | None = None,
    device: str = "cpu",
    dtype_name: str = "float32",
    seed: int = 1337,
    disable_mmap: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = shape or AnimaStaticShape()
    blocks = parse_block_indices(block_indices)
    checkpoint = Path(checkpoint_path) if str(checkpoint_path or "").strip() else default_anima_checkpoint(model_root)
    model, weight_report, target_device, normalized_dtype = _load_anima_subset(
        checkpoint,
        block_indices=blocks,
        device=device,
        dtype_name=dtype_name,
        disable_mmap=disable_mmap,
    )
    wrapper = create_anima_export_wrapper(model).eval()
    inputs = create_anima_synthetic_inputs(
        shape=shape,
        device=target_device,
        dtype_name=normalized_dtype,
        seed=seed,
    )

    import torch

    with torch.no_grad():
        output = wrapper(*inputs)

    return {
        "schema_version": 1,
        "kind": "anima_tensorrt_synthetic_forward",
        "success": True,
        "checkpoint_path": str(checkpoint),
        "block_indices": list(blocks),
        "device": str(target_device),
        "dtype": normalized_dtype,
        "shape": shape.to_dict(),
        "input_signature": shape.input_signature(),
        "weight_load_report": _report_to_dict(weight_report),
        "output": _tensor_summary(output),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def export_anima_static_onnx(
    *,
    checkpoint_path: str | Path = "",
    model_root: str | Path = "",
    output_path: str | Path = "",
    output_dir: str | Path = "",
    block_indices: str | Sequence[int] = (0,),
    shape: AnimaStaticShape | None = None,
    device: str = "cpu",
    dtype_name: str = "float32",
    seed: int = 1337,
    opset: int = 18,
    disable_mmap: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    phase = "prepare"
    shape = shape or AnimaStaticShape()
    blocks = parse_block_indices(block_indices)
    checkpoint = Path(checkpoint_path) if str(checkpoint_path or "").strip() else default_anima_checkpoint(model_root)
    dst = Path(output_path) if str(output_path or "").strip() else default_anima_onnx_path(
        output_dir=output_dir,
        shape=shape,
        block_indices=blocks,
        opset=opset,
    )
    try:
        model, weight_report, target_device, normalized_dtype = _load_anima_subset(
            checkpoint,
            block_indices=blocks,
            device=device,
            dtype_name=dtype_name,
            disable_mmap=disable_mmap,
        )
        wrapper = create_anima_static_export_wrapper(model, shape).eval()
        inputs = create_anima_synthetic_inputs(
            shape=shape,
            device=target_device,
            dtype_name=normalized_dtype,
            seed=seed,
        )
        phase = "torch_forward"

        import torch

        with torch.no_grad():
            output = wrapper(*inputs)
        output_summary = _tensor_summary(output)

        phase = "onnx_export"
        dst.parent.mkdir(parents=True, exist_ok=True)
        export_kwargs: dict[str, Any] = {
            "model": wrapper,
            "args": inputs,
            "f": str(dst),
            "input_names": ["sample", "timestep", "encoder_hidden_states", "padding_mask"],
            "output_names": ["sample_out"],
            "opset_version": max(13, int(opset or 18)),
            "do_constant_folding": True,
        }
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
        with torch.no_grad():
            torch.onnx.export(**export_kwargs)

        phase = "onnx_check"
        check = _onnx_check(dst)
        success = bool(check.get("ok", True)) if check.get("available") else True
        return {
            "schema_version": 1,
            "kind": "anima_tensorrt_static_onnx_export",
            "success": success,
            "checkpoint_path": str(checkpoint),
            "onnx_path": str(dst),
            "bytes": dst.stat().st_size if dst.exists() else 0,
            "block_indices": list(blocks),
            "device": str(target_device),
            "dtype": normalized_dtype,
            "shape": shape.to_dict(),
            "input_signature": shape.input_signature(),
            "weight_load_report": _report_to_dict(weight_report),
            "torch_output": output_summary,
            "onnx_check": check,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        if strict:
            raise
        return {
            "schema_version": 1,
            "kind": "anima_tensorrt_static_onnx_export",
            "success": False,
            "phase": phase,
            "checkpoint_path": str(checkpoint),
            "onnx_path": str(dst),
            "block_indices": list(blocks),
            "shape": shape.to_dict(),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _load_anima_subset(
    checkpoint: Path,
    *,
    block_indices: Sequence[int],
    device: str,
    dtype_name: str,
    disable_mmap: bool,
) -> tuple[Any, Any, str, str]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Anima checkpoint not found: {checkpoint}")

    import torch
    from core.lulynx_trainer.anima_native_dit import load_anima_native_executable_subset

    target_device = _target_device(torch, device)
    normalized_dtype = _normalize_dtype_name(dtype_name)
    dtype = _torch_dtype(torch, normalized_dtype)
    model, report = load_anima_native_executable_subset(
        checkpoint,
        block_indices=tuple(block_indices),
        device=target_device,
        dtype=dtype,
        disable_mmap=disable_mmap,
    )
    model.to(device=target_device, dtype=dtype)
    model.eval()
    return model, report, target_device, normalized_dtype


def load_anima_export_subset(
    checkpoint: str | Path,
    *,
    block_indices: Sequence[int],
    device: str,
    dtype_name: str,
    disable_mmap: bool = False,
) -> tuple[Any, Any, str, str]:
    return _load_anima_subset(
        Path(checkpoint),
        block_indices=block_indices,
        device=device,
        dtype_name=dtype_name,
        disable_mmap=disable_mmap,
    )


def summarize_tensor(tensor: Any) -> dict[str, Any]:
    return _tensor_summary(tensor)


def _normalize_dtype_name(value: str) -> str:
    name = str(value or "float32").strip().lower()
    aliases = {
        "fp32": "float32",
        "float": "float32",
        "fp16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
    }
    normalized = aliases.get(name, name)
    if normalized not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported dtype for Anima spike: {value}")
    return normalized


def _torch_dtype(torch_module: Any, value: str) -> Any:
    normalized = _normalize_dtype_name(value)
    if normalized == "float16":
        return torch_module.float16
    if normalized == "bfloat16":
        return torch_module.bfloat16
    return torch_module.float32


def _target_device(torch_module: Any, value: str) -> str:
    requested = str(value or "cpu").strip().lower()
    if requested.startswith("cuda") and torch_module.cuda.is_available():
        return requested
    return "cpu"


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total = int(detached.numel())
    stats = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "all_finite": bool(finite.all().item()) if total else True,
        "finite_ratio": finite_count / total if total else 1.0,
        "mean": float(stats.mean().item()) if total else 0.0,
        "std": float(stats.std(unbiased=False).item()) if total else 0.0,
        "min": float(stats.min().item()) if total else 0.0,
        "max": float(stats.max().item()) if total else 0.0,
    }


def _onnx_check(path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("onnx") is None:
        return {"available": False, "ok": None}
    try:
        import onnx  # type: ignore

        exported = onnx.load(str(path))
        onnx.checker.check_model(exported)
        return {"available": True, "ok": True}
    except Exception as exc:
        return {"available": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _report_to_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        return dict(report.to_dict())
    if hasattr(report, "__dict__"):
        return dict(report.__dict__)
    return {"repr": repr(report)}


def _block_label(block_indices: Sequence[int]) -> str:
    blocks = parse_block_indices(block_indices)
    if len(blocks) == 1:
        return f"b{blocks[0]}"
    if blocks == tuple(range(blocks[0], blocks[-1] + 1)):
        return f"b{blocks[0]}-{blocks[-1]}"
    return "b" + "_".join(str(item) for item in blocks)
