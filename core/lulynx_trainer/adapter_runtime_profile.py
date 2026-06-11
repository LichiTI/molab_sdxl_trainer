"""Runtime adapter profile helpers.

The acceleration policy can select adapter variants such as LoRA-FA or
RS-LoRA, but the useful contract is the layer that was actually injected.
This module keeps that evidence small and JSON-friendly for manifests and
runtime state.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _field(config: Any, name: str, default: Any = "") -> Any:
    try:
        return getattr(config, name, default)
    except Exception:
        return default


def _safe_numel(param: Any) -> int:
    try:
        return int(param.numel())
    except Exception:
        return 0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _iter_parameters(module: Any):
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return
    try:
        yield from parameters()
    except Exception:
        return


def _iter_adapter_parameters(layer: Any):
    inner = getattr(layer, "lora", None)
    if inner is not None:
        yield from _iter_adapter_parameters(inner)
        return
    has_down_up = hasattr(layer, "lora_down") and hasattr(layer, "lora_up")
    if has_down_up:
        yield from _iter_parameters(getattr(layer, "lora_down")) or ()
        yield from _iter_parameters(getattr(layer, "lora_up")) or ()
        return
    get_trainable = getattr(layer, "get_trainable_params", None)
    if callable(get_trainable):
        try:
            yield from get_trainable()
            return
        except Exception:
            pass
    yield from _iter_parameters(layer) or ()


def _layer_type(layer: Any) -> str:
    return type(layer).__name__


def _effective_layer_type(layer: Any) -> str:
    inner = getattr(layer, "lora", None)
    if inner is not None:
        return type(inner).__name__
    return _layer_type(layer)


def _has_flag(layer: Any, name: str) -> bool:
    inner = getattr(layer, "lora", None)
    return _boolish(getattr(layer, name, False)) or (inner is not None and _boolish(getattr(inner, name, False)))


def _infer_adapter_method(layer_types: Mapping[str, int], effective_types: Mapping[str, int], config: Any) -> str:
    if layer_types.get("LoRAFALinear", 0) > 0:
        return "lora_fa"
    if effective_types.get("DoRALinear", 0) > 0:
        return "dora"
    for method, type_name in (
        ("vera", "VeRALinear"),
        ("tlora", "TLoRALinear"),
        ("hydralora", "HydraLoRALinear"),
        ("fera", "FeRALinear"),
        ("flexrank_lora", "FlexRankLoRALinear"),
    ):
        if layer_types.get(type_name, 0) > 0 or effective_types.get(type_name, 0) > 0:
            return method
    if _boolish(_field(config, "rs_lora_enabled", False)):
        return "rs_lora"
    network_module = _value(_field(config, "network_module", "")).strip().lower()
    if "lycoris" in network_module:
        return "lycoris"
    return "lora"


def build_adapter_runtime_profile(config: Any, injector: Any, *, model_arch: str = "") -> dict[str, Any]:
    injected = getattr(injector, "injected_layers", {}) if injector is not None else {}
    if not isinstance(injected, Mapping):
        injected = {}

    layer_types: Counter[str] = Counter()
    effective_types: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    total_params = 0
    trainable_params = 0
    rs_lora_realized = False
    activation_recompute_realized = False

    for name, layer in injected.items():
        layer_types[_layer_type(layer)] += 1
        effective_types[_effective_layer_type(layer)] += 1
        prefix = str(name).split(".", 1)[0] if str(name) else ""
        if prefix:
            prefix_counts[prefix] += 1
        rs_lora_realized = rs_lora_realized or _has_flag(layer, "rs_lora_enabled")
        activation_recompute_realized = activation_recompute_realized or _has_flag(layer, "activation_recompute")
        for param in _iter_adapter_parameters(layer) or ():
            count = _safe_numel(param)
            total_params += count
            if bool(getattr(param, "requires_grad", False)):
                trainable_params += count

    model_arch_value = str(model_arch or _value(_field(config, "model_type", ""))).strip().lower()
    rank = _safe_int(_field(config, "network_dim", getattr(injector, "rank", 0) if injector is not None else 0), 0)
    alpha = _safe_float(_field(config, "network_alpha", getattr(injector, "alpha", 0.0) if injector is not None else 0.0), 0.0)
    dropout = _safe_float(_field(config, "network_dropout", getattr(injector, "dropout", 0.0) if injector is not None else 0.0), 0.0)
    layer_type_dict = dict(sorted(layer_types.items()))
    effective_type_dict = dict(sorted(effective_types.items()))
    return {
        "enabled": bool(injected),
        "source": "runtime_injector",
        "model_arch": model_arch_value,
        "network_module": _value(_field(config, "network_module", "")),
        "newbie_adapter_type": str(_field(config, "newbie_adapter_type", "") or "").strip().lower().replace("-", "_"),
        "adapter_method": _infer_adapter_method(layer_type_dict, effective_type_dict, config),
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "lora_fa_enabled": _boolish(_field(config, "lora_fa_enabled", False)),
        "rs_lora_enabled": _boolish(_field(config, "rs_lora_enabled", False)),
        "dora_enabled": _boolish(_field(config, "dora_enabled", False)) or _boolish(_field(config, "use_dora", False)),
        "vera_enabled": _boolish(_field(config, "vera_enabled", False)),
        "injected_layer_count": len(injected),
        "layer_types": layer_type_dict,
        "effective_layer_types": effective_type_dict,
        "prefix_counts": dict(sorted(prefix_counts.items())),
        "sample_layers": [str(name) for name in list(injected.keys())[:8]],
        "total_adapter_parameter_count": int(total_params),
        "trainable_adapter_parameter_count": int(trainable_params),
        "lora_fa_realized": layer_types.get("LoRAFALinear", 0) > 0,
        "rs_lora_realized": bool(rs_lora_realized),
        "activation_recompute_realized": bool(activation_recompute_realized),
    }


__all__ = ["build_adapter_runtime_profile"]
