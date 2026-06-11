from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .family_adapters import NEWBIE_ADAPTER, resolve_family_components


_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_CONTEXT_RE = re.compile(r"^context_refiner\.(\d+)\.")
_NOISE_RE = re.compile(r"^noise_refiner\.(\d+)\.")


@dataclass(frozen=True)
class NewbieStaticShape:
    batch: int = 1
    latent_channels: int = 16
    latent_height: int = 4
    latent_width: int = 4
    tokens: int = 4
    hidden_dim: int = 2304
    pooled_dim: int = 1024
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
            "hidden_dim": self.hidden_dim,
            "pooled_dim": self.pooled_dim,
            "patch_size": self.patch_size,
        }
        for key, value in values.items():
            if int(value) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.latent_height % self.patch_size or self.latent_width % self.patch_size:
            raise ValueError("Newbie latent height/width must be divisible by patch_size")

    def input_signature(self) -> dict[str, Any]:
        self.validate()
        return {
            "sample": [self.batch, self.latent_channels, self.latent_height, self.latent_width],
            "timestep": [self.batch],
            "encoder_hidden_states": [self.batch, self.tokens, self.hidden_dim],
            "text_embeds": [self.batch, self.pooled_dim],
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
            "hidden_dim": self.hidden_dim,
            "pooled_dim": self.pooled_dim,
            "patch_size": self.patch_size,
        }


def default_newbie_checkpoint(model_root: str | Path = "") -> Path:
    resolved = resolve_family_components(NEWBIE_ADAPTER, model_root)
    return Path(str(resolved["export_target_path"]))


def default_newbie_config_path(model_root: str | Path = "") -> Path:
    resolved = resolve_family_components(NEWBIE_ADAPTER, model_root)
    config = resolved.get("components", {}).get("transformer_config", {})
    return Path(str(config.get("path") or ""))


def parse_layer_indices(value: str | Sequence[int] | None) -> tuple[int, ...]:
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
                    raise ValueError(f"Invalid descending layer range: {token}")
                items.extend(range(start, end + 1))
            else:
                items.append(int(token))
    else:
        items = [int(item) for item in value]
    if not items:
        return (0,)
    result: list[int] = []
    seen: set[int] = set()
    for item in items:
        if item < 0:
            raise ValueError("Newbie layer indices must be non-negative")
        if item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def inspect_newbie_safetensors(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path = "",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        return {"available": False, "reason": "export_target_missing", "path": str(checkpoint)}
    records = _read_safetensors_shape_records(checkpoint)
    config = _read_config(config_path)
    if not config and checkpoint.parent.joinpath("transformer", "config.json").is_file():
        config = _read_config(checkpoint.parent / "transformer" / "config.json")
    return analyze_newbie_shape_records(records, config=config, checkpoint_path=checkpoint, config_path=config_path)


def analyze_newbie_shape_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path = "",
    config_path: str | Path = "",
) -> dict[str, Any]:
    config = dict(config or {})
    keys = tuple(records.keys())
    layer_indices = _collect_indices(keys, _LAYER_RE)
    context_indices = _collect_indices(keys, _CONTEXT_RE)
    noise_indices = _collect_indices(keys, _NOISE_RE)
    x_shape = _shape(records, "x_embedder.weight")
    final_shape = _shape(records, "final_layer.linear.weight")
    qkv_shape = _shape(records, "layers.0.attention.qkv.weight")
    hidden_dim = int(config.get("dim") or (x_shape[0] if x_shape else 0) or (final_shape[1] if final_shape else 0))
    latent_channels = int(config.get("in_channels") or 16)
    patch_size = int(config.get("patch_size") or _infer_patch_size(x_shape, latent_channels) or 2)
    pooled_dim = int(config.get("clip_text_dim") or _shape(records, "clip_text_pooled_proj.1.weight", [1024, 1024])[1])
    required = _required_selective_keys((0,))
    missing = [key for key in required if key not in records]
    return {
        "available": True,
        "kind": "newbie_nextdit_header_introspection",
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "tensor_count": len(keys),
        "bytes": Path(checkpoint_path).stat().st_size if str(checkpoint_path) and Path(checkpoint_path).is_file() else 0,
        "model_type": config.get("model_type", ""),
        "class_name": config.get("_class_name", ""),
        "layer_count": len(layer_indices),
        "layer_indices": list(layer_indices),
        "context_refiner_count": len(context_indices),
        "noise_refiner_count": len(noise_indices),
        "hidden_dim": hidden_dim,
        "latent_channels_hint": latent_channels,
        "patch_size": patch_size,
        "pooled_dim": pooled_dim,
        "qkv_shape": qkv_shape,
        "final_linear_shape": final_shape,
        "x_embedder_shape": x_shape,
        "selective_forward": {
            "ready": not missing,
            "default_layers": [0],
            "required_key_count": len(required),
            "missing_keys": missing,
            "estimated_selected_keys": len(select_newbie_keys(keys, layer_indices=(0,))),
        },
        "static_tensorrt_export": {
            "selected_layer_wrapper_ready": not missing,
            "validated_static_shape": {"batch": 1, "latent_height": 4, "latent_width": 4, "tokens": 4},
            "validated_production_static_shape": {"batch": 1, "latent_height": 64, "latent_width": 64, "tokens": 512},
            "validated_layer_ranges": ["0", "0-1", "0-3", "0-7", "0-15", "0-35"],
            "validated_precision": "fp32",
            "validated_fp32_production_shape": True,
            "validated_fp32_production_shape_parity": "offline",
            "validated_fp32_production_mean_abs": 0.006306241266429424,
            "validated_fp32_production_max_abs": 0.021442890167236328,
            "validated_fp16_mixed_policy": "sensitive",
            "validated_fp16_mixed_layer_ranges": ["0", "0-1", "0-3", "0-7"],
            "validated_fp16_block_matmul_policy": "sensitive_block_matmul",
            "validated_fp16_block_matmul_layer_ranges": ["0-15"],
            "rejected_fp16_mixed_full36_policies": ["sensitive", "sensitive_projections", "sensitive_block_matmul"],
            "validated_bf16_engine_build_layer_ranges": ["0-35"],
            "validated_bf16_sensitive_layer_ranges": ["0-7", "0-9", "0-11"],
            "bf16_sensitive_first_failed_layer_range": "0-12",
            "bf16_sensitive_first_failed_tap": "tap_layer_input",
            "rejected_bf16_local_fp32_layer_ranges": ["12", "8-12", "0-12"],
            "rejected_bf16_local_fp32_policies": ["layer12_block_matmul", "layers8_12_block_matmul", "sensitive_block_matmul", "sensitive_block_matmul_adaln_modulation"],
            "rejected_bf16_full36_policies": ["none", "sensitive", "sensitive_block_matmul"],
            "pending_low_precision_followups": ["keep_fp32_default_until_low_precision_hidden_state_drift_is_solved"],
            "cautions": ["newbie_fp16_mixed_full36_parity_not_acceptable", "newbie_same_process_production_parity_requires_more_vram"],
        },
        "export_blockers": ["newbie_fp16_mixed_full36_parity_not_acceptable"],
        "notes": [
            "Header-only inspection does not load the 6GB+ transformer tensors.",
            "Full 36-layer static ONNX/TensorRT export is proven at 4x4/tok4 in FP32.",
            "Full 36-layer production static shape 64x64/tok512 is proven in FP32 with split/offline parity on RTX 4070 Ti SUPER 16 GB.",
            "FP16 sensitive mixed precision is validated through layers 0-7; sensitive_block_matmul is validated through layers 0-15 but is close to FP32 size.",
            "BF16 sensitive parity is validated through layers 0-11; the first tested failing window is 0-12 and tap diagnostics show hidden-state drift before layer 12.",
            "Local FP32 preservation around layer 12, layers 8-12, all block MatMuls, and adaLN modulation did not restore BF16 parity; keep FP32 as the safe default.",
            "Same-process production parity still requires more than 16 GB VRAM; use torch-output/trt-output/compare-outputs for local validation.",
        ],
    }


def select_newbie_keys(
    keys: Sequence[str],
    *,
    layer_indices: str | Sequence[int] = (0,),
    include_context_refiner: bool = False,
    include_noise_refiner: bool = False,
) -> tuple[str, ...]:
    layers = parse_layer_indices(layer_indices)
    prefixes = [
        "x_embedder.",
        "t_embedder.",
        "time_text_embed.",
        "clip_text_pooled_proj.",
        "final_layer.",
        "norm_final.",
    ]
    prefixes.extend(f"layers.{index}." for index in layers)
    if include_context_refiner:
        prefixes.append("context_refiner.")
    if include_noise_refiner:
        prefixes.append("noise_refiner.")
    return tuple(key for key in keys if key.startswith(tuple(prefixes)))


def run_newbie_synthetic_forward(
    *,
    checkpoint_path: str | Path = "",
    model_root: str | Path = "",
    config_path: str | Path = "",
    layer_indices: str | Sequence[int] = (0,),
    shape: NewbieStaticShape | None = None,
    device: str = "cpu",
    dtype_name: str = "float32",
    seed: int = 1337,
    include_context_refiner: bool = False,
    include_noise_refiner: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    phase = "prepare"
    shape = shape or NewbieStaticShape()
    checkpoint = Path(checkpoint_path) if str(checkpoint_path or "").strip() else default_newbie_checkpoint(model_root)
    cfg_path = Path(config_path) if str(config_path or "").strip() else default_newbie_config_path(model_root)
    layers = parse_layer_indices(layer_indices)
    try:
        phase = "load_selected_weights"
        model, selected_keys, target_device, normalized_dtype = _load_newbie_selective_wrapper(
            checkpoint,
            config_path=cfg_path,
            layer_indices=layers,
            device=device,
            dtype_name=dtype_name,
            include_context_refiner=include_context_refiner,
            include_noise_refiner=include_noise_refiner,
        )
        phase = "synthetic_forward"
        inputs = create_newbie_synthetic_inputs(
            shape=shape,
            device=target_device,
            dtype_name=normalized_dtype,
            seed=seed,
        )
        import torch

        with torch.no_grad():
            output_obj = model(
                sample=inputs["sample"],
                timestep=inputs["timestep"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                added_cond_kwargs={"text_embeds": inputs["text_embeds"]},
            )
        output = getattr(output_obj, "sample", output_obj)
        return {
            "schema_version": 1,
            "kind": "newbie_tensorrt_synthetic_forward",
            "success": True,
            "checkpoint_path": str(checkpoint),
            "config_path": str(cfg_path),
            "layer_indices": list(layers),
            "selected_key_count": len(selected_keys),
            "device": str(target_device),
            "dtype": normalized_dtype,
            "shape": shape.to_dict(),
            "input_signature": shape.input_signature(),
            "output": _tensor_summary(output),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        if strict:
            raise
        return {
            "schema_version": 1,
            "kind": "newbie_tensorrt_synthetic_forward",
            "success": False,
            "phase": phase,
            "checkpoint_path": str(checkpoint),
            "config_path": str(cfg_path),
            "layer_indices": list(layers),
            "shape": shape.to_dict(),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def create_newbie_synthetic_inputs(
    *,
    shape: NewbieStaticShape | None = None,
    device: str = "cpu",
    dtype_name: str = "float32",
    seed: int = 1337,
) -> dict[str, Any]:
    import torch

    shape = shape or NewbieStaticShape()
    shape.validate()
    dtype = _torch_dtype(torch, dtype_name)
    target_device = _target_device(torch, device)
    generator_device = "cuda" if str(target_device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    return {
        "sample": torch.randn(
            shape.batch,
            shape.latent_channels,
            shape.latent_height,
            shape.latent_width,
            device=target_device,
            dtype=dtype,
            generator=generator,
        ),
        "timestep": torch.full((shape.batch,), 250.0, device=target_device, dtype=torch.float32),
        "encoder_hidden_states": torch.randn(
            shape.batch,
            shape.tokens,
            shape.hidden_dim,
            device=target_device,
            dtype=dtype,
            generator=generator,
        ) * 0.01,
        "text_embeds": torch.randn(shape.batch, shape.pooled_dim, device=target_device, dtype=dtype, generator=generator) * 0.01,
    }


def _load_newbie_selective_wrapper(
    checkpoint: Path,
    *,
    config_path: Path,
    layer_indices: Sequence[int],
    device: str,
    dtype_name: str,
    include_context_refiner: bool,
    include_noise_refiner: bool,
) -> tuple[Any, tuple[str, ...], str, str]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Newbie checkpoint not found: {checkpoint}")
    import torch
    from core.lulynx_trainer.newbie_loader import _NextDiTWrapper
    from core.lulynx_trainer.safetensors_loader import open_safetensors

    target_device = _target_device(torch, device)
    normalized_dtype = _normalize_dtype_name(dtype_name)
    dtype = _torch_dtype(torch, normalized_dtype)
    with open_safetensors(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        selected_keys = select_newbie_keys(
            keys,
            layer_indices=layer_indices,
            include_context_refiner=include_context_refiner,
            include_noise_refiner=include_noise_refiner,
        )
        if not selected_keys:
            raise RuntimeError("No Newbie tensors selected for synthetic forward")
        state = {key: handle.get_tensor(key) for key in selected_keys}
    model = _NextDiTWrapper(state)
    config = _read_config(config_path)
    if config:
        model.patch_size = int(config.get("patch_size", getattr(model, "patch_size", 1)) or 1)
        model.in_channels = int(config.get("in_channels", getattr(model, "in_channels", 16)) or 16)
        model.config = type("NextDiTConfig", (), dict(config))()
    model.to(device=target_device, dtype=dtype)
    model.eval()
    return model, selected_keys, target_device, normalized_dtype


def load_newbie_selective_wrapper(
    checkpoint: str | Path,
    *,
    config_path: str | Path,
    layer_indices: Sequence[int],
    device: str,
    dtype_name: str,
    include_context_refiner: bool = False,
    include_noise_refiner: bool = False,
) -> tuple[Any, tuple[str, ...], str, str]:
    return _load_newbie_selective_wrapper(
        Path(checkpoint),
        config_path=Path(config_path),
        layer_indices=layer_indices,
        device=device,
        dtype_name=dtype_name,
        include_context_refiner=include_context_refiner,
        include_noise_refiner=include_noise_refiner,
    )


def summarize_tensor(tensor: Any) -> dict[str, Any]:
    return _tensor_summary(tensor)


def _read_safetensors_shape_records(path: Path) -> dict[str, dict[str, Any]]:
    from safetensors import safe_open

    records: dict[str, dict[str, Any]] = {}
    with safe_open(str(path), framework="numpy") as handle:
        for key in handle.keys():
            view = handle.get_slice(key)
            records[str(key)] = {"shape": [int(dim) for dim in view.get_shape()], "dtype": str(view.get_dtype())}
    return records


def _read_config(path: str | Path) -> dict[str, Any]:
    item = Path(path) if str(path or "").strip() else Path()
    if not item.is_file():
        return {}
    try:
        return json.loads(item.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_indices(keys: Sequence[str], pattern: re.Pattern[str]) -> tuple[int, ...]:
    values = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    return tuple(sorted(values))


def _shape(records: Mapping[str, Mapping[str, Any]], key: str, fallback: Sequence[int] | None = None) -> list[int]:
    value = records.get(key, {})
    shape = value.get("shape", fallback or [])
    return [int(dim) for dim in shape]


def _infer_patch_size(x_shape: Sequence[int], latent_channels: int) -> int:
    if len(x_shape) != 2 or latent_channels <= 0:
        return 0
    features = int(x_shape[1])
    if features % latent_channels:
        return 0
    patch_area = features // latent_channels
    root = int(patch_area ** 0.5)
    return root if root * root == patch_area else 0


def _required_selective_keys(layer_indices: Sequence[int]) -> tuple[str, ...]:
    required = [
        "x_embedder.weight",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.weight",
        "final_layer.linear.weight",
    ]
    for index in layer_indices:
        required.extend([
            f"layers.{index}.attention.qkv.weight",
            f"layers.{index}.attention.out.weight",
            f"layers.{index}.feed_forward.w1.weight",
            f"layers.{index}.feed_forward.w2.weight",
            f"layers.{index}.feed_forward.w3.weight",
        ])
    return tuple(required)


def _normalize_dtype_name(value: str) -> str:
    name = str(value or "float32").strip().lower()
    aliases = {"fp32": "float32", "float": "float32", "fp16": "float16", "half": "float16", "bf16": "bfloat16"}
    normalized = aliases.get(name, name)
    if normalized not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported dtype for Newbie spike: {value}")
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
