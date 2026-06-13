"""Warehouse weight compression helpers for frozen model weights.

The stable native path casts frozen parameters to ``torch.float8_e4m3fn``.
Optional torchao / optimum.quanto paths are discovered at runtime and are kept
behind the same contract so missing dependencies fail clearly instead of
changing training behavior silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import importlib
import importlib.util
import logging
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from .fp8_quantize import quantize_base_weights_fp8

logger = logging.getLogger(__name__)

NATIVE_FP8_FORMAT = "fp8_e4m3"
TORCHAO_FORMATS = {"torchao_int8", "torchao_uint4", "torchao_float8"}
QUANTO_FORMATS = {"quanto_int8", "quanto_float8"}
SUPPORTED_COMPRESSION_FORMATS = {NATIVE_FP8_FORMAT, *TORCHAO_FORMATS, *QUANTO_FORMATS}
COMPRESSION_PRESET_MAP: dict[str, dict[str, Any]] = {
    "off": {"enabled": False, "target": "none", "format": NATIVE_FP8_FORMAT},
    "stable_backbone_int8": {"enabled": True, "target": "backbone", "format": "torchao_int8"},
    "aggressive_backbone_uint4": {"enabled": True, "target": "backbone", "format": "torchao_uint4"},
    "text_encoder_int8": {"enabled": True, "target": "text_encoder", "format": "torchao_int8"},
    "both_int8": {"enabled": True, "target": "both", "format": "torchao_int8"},
    "experimental_float8": {"enabled": True, "target": "backbone", "format": "torchao_float8"},
}


def normalize_weight_compression_preset(value: Any) -> str:
    preset = str(value or "off").strip().lower().replace("-", "_")
    aliases = {
        "": "off",
        "none": "off",
        "disabled": "off",
        "safe": "stable_backbone_int8",
        "stable": "stable_backbone_int8",
        "backbone_int8": "stable_backbone_int8",
        "int8": "stable_backbone_int8",
        "aggressive": "aggressive_backbone_uint4",
        "backbone_uint4": "aggressive_backbone_uint4",
        "uint4": "aggressive_backbone_uint4",
        "te_int8": "text_encoder_int8",
        "text_int8": "text_encoder_int8",
        "text_encoder": "text_encoder_int8",
        "all_int8": "both_int8",
        "both": "both_int8",
        "float8": "experimental_float8",
        "fp8_experimental": "experimental_float8",
    }
    normalized = aliases.get(preset, preset)
    return normalized if normalized in COMPRESSION_PRESET_MAP else "off"


def resolve_weight_compression_preset(value: Any) -> dict[str, Any]:
    preset = normalize_weight_compression_preset(value)
    resolved = dict(COMPRESSION_PRESET_MAP[preset])
    resolved["preset"] = preset
    return resolved

_BACKBONE_TARGETS = {"backbone", "both"}
_TEXT_ENCODER_TARGETS = {"text_encoder", "text_encoders", "te", "both"}
_LINEAR_ONLY_FORMATS = TORCHAO_FORMATS | QUANTO_FORMATS


@dataclass(frozen=True)
class CompressionFormatInfo:
    name: str
    backend: str
    target_bits: int
    linear_only: bool
    available: bool
    unavailable_reason: str = ""


@dataclass
class WeightCompressionComponentResult:
    name: str
    parameter_count: int = 0
    compressed_count: int = 0
    skipped_trainable_count: int = 0
    skipped_adapter_count: int = 0
    skipped_pattern_count: int = 0
    bytes_saved: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def estimated_saved_mb(self) -> float:
        return self.bytes_saved / (1024 * 1024)


@dataclass
class WeightCompressionResult:
    enabled: bool
    target: str
    format: str
    backend: str = "native"
    components: list[WeightCompressionComponentResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def estimated_saved_mb(self) -> float:
        return sum(component.estimated_saved_mb for component in self.components)

    @property
    def compressed_count(self) -> int:
        return sum(component.compressed_count for component in self.components)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "target": self.target,
            "format": self.format,
            "backend": self.backend,
            "estimated_saved_mb": self.estimated_saved_mb,
            "compressed_count": self.compressed_count,
            "components": [
                {
                    "name": c.name,
                    "parameter_count": c.parameter_count,
                    "compressed_count": c.compressed_count,
                    "skipped_trainable_count": c.skipped_trainable_count,
                    "skipped_adapter_count": c.skipped_adapter_count,
                    "skipped_pattern_count": c.skipped_pattern_count,
                    "estimated_saved_mb": c.estimated_saved_mb,
                    "warnings": list(c.warnings),
                }
                for c in self.components
            ],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class WeightCompressionRuntimeConfig:
    requested: bool
    enabled: bool
    target: str
    format: str
    preset: str
    legacy_fp8_base: bool = False
    preset_applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "enabled": self.enabled,
            "target": self.target,
            "format": self.format,
            "preset": self.preset,
            "legacy_fp8_base": self.legacy_fp8_base,
            "preset_applied": self.preset_applied,
        }


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _boolish(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def get_weight_compression_format_info(format_name: Any) -> CompressionFormatInfo:
    fmt = normalize_weight_compression_format(format_name)
    if fmt == NATIVE_FP8_FORMAT:
        return CompressionFormatInfo(fmt, "native", 8, False, hasattr(torch, "float8_e4m3fn"))
    if fmt in TORCHAO_FORMATS:
        available = _has_module("torchao")
        reason = "" if available else "torchao is not installed"
        bits = 4 if fmt == "torchao_uint4" else 8
        return CompressionFormatInfo(fmt, "torchao", bits, True, available, reason)
    if fmt in QUANTO_FORMATS:
        available = _has_module("optimum.quanto")
        reason = "" if available else "optimum.quanto is not installed"
        return CompressionFormatInfo(fmt, "quanto", 8, True, available, reason)
    return CompressionFormatInfo(fmt, "unknown", 0, True, False, f"unsupported format: {fmt}")


def available_weight_compression_formats() -> dict[str, dict[str, Any]]:
    return {
        fmt: {
            "backend": (info := get_weight_compression_format_info(fmt)).backend,
            "available": info.available,
            "linear_only": info.linear_only,
            "unavailable_reason": info.unavailable_reason,
        }
        for fmt in sorted(SUPPORTED_COMPRESSION_FORMATS)
    }


def _split_patterns(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.replace(";", ",").split(",")
    else:
        raw = [str(item) for item in value]
    return tuple(item.strip().lower() for item in raw if item and item.strip())


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    text = name.lower()
    return any(fnmatch.fnmatch(text, pattern) or pattern in text for pattern in patterns)


def normalize_weight_compression_target(target: Any, *, legacy_fp8_base: bool = False) -> str:
    value = str(target or "").strip().lower()
    aliases = {
        "": "none",
        "off": "none",
        "disabled": "none",
        "unet": "backbone",
        "dit": "backbone",
        "transformer": "backbone",
        "text": "text_encoder",
        "text_encoder_only": "text_encoder",
        "text_encoders": "text_encoder",
        "te": "text_encoder",
        "all": "both",
    }
    value = aliases.get(value, value)
    if value not in {"none", "text_encoder", "backbone", "both"}:
        value = "none"
    if value == "none" and legacy_fp8_base:
        value = "backbone"
    return value


def normalize_weight_compression_format(value: Any) -> str:
    fmt = str(value or NATIVE_FP8_FORMAT).strip().lower().replace("-", "_")
    aliases = {
        "fp8": NATIVE_FP8_FORMAT,
        "native_fp8": NATIVE_FP8_FORMAT,
        "float8": NATIVE_FP8_FORMAT,
        "float8_e4m3": NATIVE_FP8_FORMAT,
        "e4m3": NATIVE_FP8_FORMAT,
        "int8": "torchao_int8",
        "uint4": "torchao_uint4",
        "int4": "torchao_uint4",
        "torchao_int4": "torchao_uint4",
        "torchao_fp8": "torchao_float8",
        "quanto_qint8": "quanto_int8",
        "quanto_qfloat8": "quanto_float8",
    }
    return aliases.get(fmt, fmt)


def resolve_weight_compression_runtime_config(config: Any) -> WeightCompressionRuntimeConfig:
    """Resolve trainer-facing weight compression fields, including presets."""
    legacy_fp8 = _boolish(_get_config_value(config, "fp8_base", False))
    preset = normalize_weight_compression_preset(_get_config_value(config, "weight_compression_preset", "off"))
    preset_spec = resolve_weight_compression_preset(preset)
    preset_requested = preset != "off" and bool(preset_spec.get("enabled"))

    raw_target = str(_get_config_value(config, "weight_compression_target", "none") or "none").strip().lower()
    target_is_open = raw_target in {"", "auto", "default", "off", "none", "disabled"}
    target_value = raw_target
    preset_applied = False
    if target_is_open and preset_requested:
        target_value = str(preset_spec.get("target") or "none")
        preset_applied = True
    if target_is_open and legacy_fp8 and target_value in {"", "auto", "default", "off", "none", "disabled"}:
        target_value = "backbone"
    target = normalize_weight_compression_target(target_value, legacy_fp8_base=legacy_fp8)

    raw_format = str(_get_config_value(config, "weight_compression_format", "") or "").strip().lower()
    format_is_open = raw_format in {"", "auto", "default"} or (
        preset_requested and normalize_weight_compression_format(raw_format) == NATIVE_FP8_FORMAT
    )
    format_value = raw_format
    if format_is_open and preset_requested:
        format_value = str(preset_spec.get("format") or NATIVE_FP8_FORMAT)
        preset_applied = True
    if format_is_open and not preset_requested:
        format_value = NATIVE_FP8_FORMAT
    fmt = normalize_weight_compression_format(format_value)

    enabled = _boolish(_get_config_value(config, "weight_compression_enabled", False)) or legacy_fp8 or preset_requested
    return WeightCompressionRuntimeConfig(
        requested=enabled,
        enabled=enabled and target != "none",
        target=target,
        format=fmt,
        preset=preset,
        legacy_fp8_base=legacy_fp8,
        preset_applied=preset_applied,
    )



def _collect_adapter_param_ids(lora_injector: Any) -> set[int]:
    """Collect adapter-owned parameter ids while leaving wrapped base weights eligible.

    ``LoRAInjector.injected_layers`` contains wrapper modules whose parameters
    include both the adapter branch and the frozen original Linear.  For weight
    compression we must skip only the adapter branch, not ``original``.
    """
    ids: set[int] = set()
    if lora_injector is None:
        return ids
    if hasattr(lora_injector, "get_trainable_params"):
        for param in lora_injector.get_trainable_params():
            ids.add(id(param))
    injected = getattr(lora_injector, "injected_layers", None)
    if isinstance(injected, dict):
        for layer in injected.values():
            base_ids: set[int] = set()
            for attr_name in ("original", "original_layer", "base_layer"):
                base = getattr(layer, attr_name, None)
                if isinstance(base, nn.Module):
                    base_ids.update(id(param) for param in base.parameters())
            for param in layer.parameters():
                if id(param) not in base_ids:
                    ids.add(id(param))
    return ids


def _param_itemsize(param: torch.nn.Parameter) -> int:
    try:
        return param.dtype.itemsize  # type: ignore[return-value]
    except AttributeError:
        return param.element_size()


def compress_frozen_weights_fp8(
    module: nn.Module,
    *,
    component_name: str,
    lora_injector: Any = None,
    include_patterns: str | Sequence[str] | None = None,
    exclude_patterns: str | Sequence[str] | None = None,
) -> WeightCompressionComponentResult:
    result = WeightCompressionComponentResult(name=component_name)
    lora_ids = _collect_adapter_param_ids(lora_injector)
    includes = _split_patterns(include_patterns)
    excludes = _split_patterns(exclude_patterns)
    dtype = torch.float8_e4m3fn

    # Native fp8 direct-cast is only forward-safe for ``nn.Linear`` weights:
    # RMSNorm/LayerNorm scales and biases are consumed by elementwise ops
    # (``x * rsqrt(var)``) that cannot promote fp8, and nn.Embedding lookups on
    # fp8 are unsupported. So we walk modules and cast *only* the 2-D weight of
    # each frozen Linear (the matmul path), leaving norms/embeddings/biases in
    # their original dtype. The Linear forward is made fp8-safe separately
    # (LoRALinear._base_forward for wrapped layers; a dequant shim for the rest).
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        result.parameter_count += 1
        qualified_name = f"{component_name}.{name}" if name else component_name
        if includes and not _matches_any(qualified_name, includes):
            result.skipped_pattern_count += 1
            continue
        if excludes and _matches_any(qualified_name, excludes):
            result.skipped_pattern_count += 1
            continue
        eligible, reason = _linear_module_is_eligible(child, lora_ids)
        if not eligible:
            if reason == "adapter":
                result.skipped_adapter_count += 1
            else:
                result.skipped_trainable_count += 1
            continue

        orig_bytes = _param_itemsize(weight)
        try:
            weight.data = weight.data.to(dtype)
        except RuntimeError as exc:
            result.warnings.append(f"{qualified_name}: {exc}")
            continue
        new_bytes = dtype.itemsize if hasattr(dtype, "itemsize") else weight.element_size()
        result.bytes_saved += weight.numel() * max(orig_bytes - new_bytes, 0)
        result.compressed_count += 1

    logger.info(
        "Weight compression %s: %d/%d frozen Linear weights compressed, estimated savings %.1f MB",
        component_name,
        result.compressed_count,
        result.parameter_count,
        result.estimated_saved_mb,
    )
    return result


def _linear_module_is_eligible(module: nn.Linear, lora_ids: set[int]) -> tuple[bool, str]:
    for param in module.parameters(recurse=False):
        if id(param) in lora_ids or getattr(param, "_lora_leaf", False):
            return False, "adapter"
        if param.requires_grad:
            return False, "trainable"
    return True, ""


def _estimate_module_savings_bytes(module: nn.Linear, target_bits: int) -> int:
    saved = 0
    target_bytes = max(target_bits / 8.0, 0.5)
    for param in module.parameters(recurse=False):
        current = _param_itemsize(param)
        saved += int(param.numel() * max(current - target_bytes, 0))
    return saved


def _torchao_config(format_name: str) -> Any:
    quant_mod = importlib.import_module("torchao.quantization")
    if format_name == "torchao_int8":
        config_cls = getattr(quant_mod, "Int8WeightOnlyConfig", None)
        if config_cls is not None:
            return config_cls()
        return getattr(quant_mod, "int8_weight_only")()
    if format_name == "torchao_uint4":
        config_cls = getattr(quant_mod, "Int4WeightOnlyConfig", None) or getattr(quant_mod, "UIntXWeightOnlyConfig", None)
        if config_cls is not None:
            return config_cls()
        factory = getattr(quant_mod, "int4_weight_only", None) or getattr(quant_mod, "uint4_weight_only")
        return factory()
    if format_name == "torchao_float8":
        config_cls = getattr(quant_mod, "Float8WeightOnlyConfig", None)
        if config_cls is not None:
            return config_cls()
        return getattr(quant_mod, "float8_weight_only")()
    raise ValueError(f"unsupported torchao format: {format_name}")


def _torchao_apply_fn(format_name: str) -> Callable[[nn.Linear], None]:
    quant_mod = importlib.import_module("torchao.quantization")
    quantize_ = getattr(quant_mod, "quantize_")
    config = _torchao_config(format_name)

    def apply(module: nn.Linear) -> None:
        quantize_(module, config)

    return apply


def _quanto_apply_fn(format_name: str) -> Callable[[nn.Linear], None]:
    quanto = importlib.import_module("optimum.quanto")
    quantize = getattr(quanto, "quantize")
    freeze = getattr(quanto, "freeze", None)
    qtype_name = "qint8" if format_name == "quanto_int8" else "qfloat8"
    qtype = getattr(quanto, qtype_name)

    def apply(module: nn.Linear) -> None:
        quantize(module, weights=qtype, activations=None)
        if freeze is not None:
            freeze(module)

    return apply


def compress_frozen_linear_weights_backend(
    module: nn.Module,
    *,
    component_name: str,
    format_name: str,
    lora_injector: Any = None,
    include_patterns: str | Sequence[str] | None = None,
    exclude_patterns: str | Sequence[str] | None = None,
) -> WeightCompressionComponentResult:
    info = get_weight_compression_format_info(format_name)
    result = WeightCompressionComponentResult(name=component_name)
    if not info.available:
        result.warnings.append(info.unavailable_reason)
        return result

    if info.backend == "torchao":
        apply_fn = _torchao_apply_fn(info.name)
    elif info.backend == "quanto":
        apply_fn = _quanto_apply_fn(info.name)
    else:
        result.warnings.append(f"unsupported backend: {info.backend}")
        return result

    lora_ids = _collect_adapter_param_ids(lora_injector)
    includes = _split_patterns(include_patterns)
    excludes = _split_patterns(exclude_patterns)

    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        result.parameter_count += sum(1 for _ in child.parameters(recurse=False))
        qualified_name = f"{component_name}.{name}" if name else component_name
        if includes and not _matches_any(qualified_name, includes):
            result.skipped_pattern_count += 1
            continue
        if excludes and _matches_any(qualified_name, excludes):
            result.skipped_pattern_count += 1
            continue
        eligible, reason = _linear_module_is_eligible(child, lora_ids)
        if not eligible:
            if reason == "adapter":
                result.skipped_adapter_count += 1
            elif reason == "trainable":
                result.skipped_trainable_count += 1
            continue

        estimated = _estimate_module_savings_bytes(child, info.target_bits)
        try:
            apply_fn(child)
        except Exception as exc:  # optional backends vary across versions
            result.warnings.append(f"{qualified_name}: {exc}")
            continue
        result.bytes_saved += estimated
        result.compressed_count += 1

    logger.info(
        "Weight compression %s via %s/%s: %d Linear modules compressed, estimated savings %.1f MB",
        component_name,
        info.backend,
        info.name,
        result.compressed_count,
        result.estimated_saved_mb,
    )
    return result



def probe_weight_compression_format(
    format_name: Any,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[bool, str]:
    """Run a tiny Linear forward probe for a compression format."""
    fmt = normalize_weight_compression_format(format_name)
    info = get_weight_compression_format_info(fmt)
    if not info.available:
        return False, info.unavailable_reason
    if fmt == NATIVE_FP8_FORMAT:
        return False, "native fp8_e4m3 direct-cast is storage-only and not training-forward safe"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        module = nn.Linear(16, 16, bias=False).to(device=device, dtype=dtype).eval()
        input_tensor = torch.randn(2, 16, device=device, dtype=dtype)
        if info.backend == "torchao":
            _torchao_apply_fn(fmt)(module)
        elif info.backend == "quanto":
            _quanto_apply_fn(fmt)(module)
        else:
            return False, f"unsupported backend: {info.backend}"
        output = module(input_tensor)
        if not torch.isfinite(output.float()).all():
            return False, "probe output contains non-finite values"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
def iter_text_encoder_components(model_container: Any) -> list[tuple[str, nn.Module]]:
    components: list[tuple[str, nn.Module]] = []
    for name in ("text_encoder", "text_encoder_1", "text_encoder_2", "text_encoder_3"):
        module = getattr(model_container, name, None)
        if isinstance(module, nn.Module) and all(module is not seen for _, seen in components):
            components.append((name, module))
    return components


def _compress_component(
    module: nn.Module,
    *,
    component_name: str,
    format_name: str,
    lora_injector: Any,
    include_patterns: str,
    exclude_patterns: str,
) -> WeightCompressionComponentResult:
    if format_name == NATIVE_FP8_FORMAT:
        return compress_frozen_weights_fp8(
            module,
            component_name=component_name,
            lora_injector=lora_injector,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    return compress_frozen_linear_weights_backend(
        module,
        component_name=component_name,
        format_name=format_name,
        lora_injector=lora_injector,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )


def apply_weight_compression(
    model_container: Any,
    *,
    enabled: bool = False,
    target: str = "none",
    format: str = NATIVE_FP8_FORMAT,
    lora_injector: Any = None,
    train_text_encoder: bool = False,
    include_patterns: str = "",
    exclude_patterns: str = "",
    legacy_fp8_base: bool = False,
) -> WeightCompressionResult:
    effective_target = normalize_weight_compression_target(target, legacy_fp8_base=legacy_fp8_base)
    effective_format = normalize_weight_compression_format(format)
    info = get_weight_compression_format_info(effective_format)
    requested = bool(enabled) or bool(legacy_fp8_base)
    result = WeightCompressionResult(
        enabled=requested and effective_target != "none",
        target=effective_target,
        format=effective_format,
        backend=info.backend,
    )
    if not result.enabled:
        return result
    if effective_format not in SUPPORTED_COMPRESSION_FORMATS:
        result.enabled = False
        result.warnings.append(f"Unsupported weight compression format: {effective_format}")
        return result
    if not info.available:
        result.enabled = False
        result.warnings.append(info.unavailable_reason)
        return result

    if effective_target in _BACKBONE_TARGETS:
        backbone = getattr(model_container, "unet", None)
        if isinstance(backbone, nn.Module):
            result.components.append(
                _compress_component(
                    backbone,
                    component_name="backbone",
                    format_name=effective_format,
                    lora_injector=lora_injector,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                )
            )
        else:
            result.warnings.append("Backbone compression skipped: no UNet/DiT module found")

    if effective_target in _TEXT_ENCODER_TARGETS:
        text_encoders = iter_text_encoder_components(model_container)
        if train_text_encoder:
            result.warnings.append("Text encoder compression skipped: text encoder training is enabled")
        elif not text_encoders:
            result.warnings.append("Text encoder compression skipped: no text encoder modules found")
        else:
            for name, text_encoder in text_encoders:
                result.components.append(
                    _compress_component(
                        text_encoder,
                        component_name=name,
                        format_name=effective_format,
                        lora_injector=lora_injector,
                        include_patterns=include_patterns,
                        exclude_patterns=exclude_patterns,
                    )
                )

    return result


__all__ = [
    "NATIVE_FP8_FORMAT",
    "QUANTO_FORMATS",
    "SUPPORTED_COMPRESSION_FORMATS",
    "TORCHAO_FORMATS",
    "CompressionFormatInfo",
    "WeightCompressionComponentResult",
    "WeightCompressionResult",
    "WeightCompressionRuntimeConfig",
    "apply_weight_compression",
    "available_weight_compression_formats",
    "compress_frozen_linear_weights_backend",
    "compress_frozen_weights_fp8",
    "get_weight_compression_format_info",
    "normalize_weight_compression_preset",
    "resolve_weight_compression_preset",
    "iter_text_encoder_components",
    "normalize_weight_compression_format",
    "resolve_weight_compression_runtime_config",
    "probe_weight_compression_format",
    "normalize_weight_compression_target",
    "quantize_base_weights_fp8",
]





