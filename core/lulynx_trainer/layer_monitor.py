"""Low-frequency per-layer LoRA monitor statistics.

The collector intentionally runs only on sampled optimizer steps. Most metrics
need tensor reductions and scalar reads, so collecting them every step would add
avoidable synchronization overhead on CUDA.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch


@dataclass
class LayerMonitorResult:
    layers: List[Dict[str, Any]]
    elapsed_seconds: float
    sampled_layers: int
    total_layers: int
    mode: str = "sampled"
    sample_size: int = 4096


def should_collect_layer_monitor(
    *,
    enabled: bool,
    step: int,
    interval: int,
) -> bool:
    if not enabled:
        return False
    interval = max(int(interval or 1), 1)
    return step <= 1 or step % interval == 0


def collect_lora_layer_stats(
    lora_injector: Any,
    optimizer: Any = None,
    *,
    max_layers: int = 10,
    sparsity_epsilon: float = 1e-8,
    mode: str = "sampled",
    sample_size: int = 4096,
) -> LayerMonitorResult:
    started = time.perf_counter()
    mode = str(mode or "sampled").lower()
    if mode not in {"sampled", "exact"}:
        mode = "sampled"
    sample_size = max(int(sample_size or 4096), 128)
    injected = getattr(lora_injector, "injected_layers", None) if lora_injector is not None else None
    if not isinstance(injected, Mapping) or not injected:
        return LayerMonitorResult([], time.perf_counter() - started, 0, 0, mode=mode, sample_size=sample_size)

    param_lr = _optimizer_param_lrs(optimizer)
    items = list(injected.items())
    total_layers = len(items)
    max_layers = max(int(max_layers or 0), 0)
    if max_layers > 0:
        items = _select_layers(items, max_layers)

    layers: List[Dict[str, Any]] = []
    with torch.no_grad():
        for index, (name, module) in enumerate(items):
            params = [p for p in _iter_trainable_params(module)]
            if not params:
                continue
            lr = _layer_lr(params, param_lr)
            if mode == "exact":
                stats = _reduce_params(params, sparsity_epsilon=float(sparsity_epsilon or 1e-8))
            else:
                stats = _reduce_params_sampled(
                    params,
                    sparsity_epsilon=float(sparsity_epsilon or 1e-8),
                    sample_size=sample_size,
                )
            layers.append(
                {
                    "name": str(name),
                    "active": index == 0,
                    "state": "training" if any(p.grad is not None for p in params) else "prep",
                    "lr": lr,
                    "grad_norm": stats["grad_norm"],
                    "weight_norm": stats["weight_norm"],
                    "l2_norm": stats["l2_norm"],
                    "rms": stats["rms"],
                    "std": stats["std"],
                    "sparsity": stats["sparsity"],
                    "approximate": int(stats.get("sampled_elements", 0)) < int(stats.get("total_elements", 0)),
                    "monitor_mode": mode,
                    "sample_size": sample_size,
                    "sampled_elements": stats.get("sampled_elements", 0),
                    "total_elements": stats.get("total_elements", 0),
                }
            )

    return LayerMonitorResult(
        layers=layers,
        elapsed_seconds=time.perf_counter() - started,
        sampled_layers=len(layers),
        total_layers=total_layers,
        mode=mode,
        sample_size=sample_size,
    )


def _iter_trainable_params(module: Any) -> Iterable[torch.Tensor]:
    if not hasattr(module, "parameters"):
        return []
    return (p for p in module.parameters() if isinstance(p, torch.Tensor) and p.requires_grad)


def _optimizer_param_lrs(optimizer: Any) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for group in getattr(optimizer, "param_groups", []) or []:
        try:
            lr = float(group.get("lr", 0.0) or 0.0)
        except Exception:
            lr = 0.0
        for param in group.get("params", []) or []:
            result[id(param)] = lr
    return result


def _layer_lr(params: List[torch.Tensor], param_lr: Dict[int, float]) -> float:
    values = [param_lr.get(id(p)) for p in params if id(p) in param_lr]
    values = [float(v) for v in values if v is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _select_layers(items: List[Tuple[str, Any]], max_layers: int) -> List[Tuple[str, Any]]:
    if len(items) <= max_layers:
        return items
    if max_layers == 1:
        return [items[0]]
    last = len(items) - 1
    selected = []
    seen = set()
    for i in range(max_layers):
        idx = round(i * last / (max_layers - 1))
        if idx not in seen:
            selected.append(items[idx])
            seen.add(idx)
    return selected


def _reduce_params(params: List[torch.Tensor], sparsity_epsilon: float) -> Dict[str, float]:
    weight_sq_sum = None
    grad_sq_sum = None
    weight_sum = None
    weight_count = 0
    sparse_count = None

    for param in params:
        data = param.detach().float()
        sq = data.square().sum()
        weight_sq_sum = sq if weight_sq_sum is None else weight_sq_sum + sq
        weight_sum = data.sum() if weight_sum is None else weight_sum + data.sum()
        weight_count += int(data.numel())
        sparse = data.abs().le(sparsity_epsilon).sum()
        sparse_count = sparse if sparse_count is None else sparse_count + sparse

        if param.grad is not None:
            grad = param.grad.detach().float()
            grad_sq = grad.square().sum()
            grad_sq_sum = grad_sq if grad_sq_sum is None else grad_sq_sum + grad_sq

    if weight_count <= 0 or weight_sq_sum is None:
        return _empty_stats()

    mean = weight_sum / weight_count
    variance = torch.clamp(weight_sq_sum / weight_count - mean.square(), min=0.0)
    weight_norm = torch.sqrt(weight_sq_sum)
    grad_norm = torch.sqrt(grad_sq_sum) if grad_sq_sum is not None else weight_norm.new_tensor(0.0)
    rms = torch.sqrt(weight_sq_sum / weight_count)
    std = torch.sqrt(variance)
    sparsity = (sparse_count.float() / weight_count) if sparse_count is not None else weight_norm.new_tensor(0.0)

    return {
        "grad_norm": _finite_float(grad_norm),
        "weight_norm": _finite_float(weight_norm),
        "l2_norm": _finite_float(weight_norm),
        "rms": _finite_float(rms),
        "std": _finite_float(std),
        "sparsity": max(0.0, min(_finite_float(sparsity), 1.0)),
        "sampled_elements": weight_count,
        "total_elements": weight_count,
    }


def _reduce_params_sampled(params: List[torch.Tensor], sparsity_epsilon: float, sample_size: int) -> Dict[str, float]:
    total_elements = sum(int(p.numel()) for p in params)
    if total_elements <= 0:
        return _empty_stats()
    if total_elements <= sample_size * 16:
        return _reduce_params(params, sparsity_epsilon)

    weight_samples: List[torch.Tensor] = []
    grad_samples: List[torch.Tensor] = []
    for param in params:
        data = param.detach().float().reshape(-1)
        numel = int(data.numel())
        if numel <= 0:
            continue
        sample_count = min(numel, max(1, round(sample_size * (numel / total_elements))))
        weight_samples.append(_sample_flat_tensor(data, sample_count))
        if param.grad is not None:
            grad = param.grad.detach().float().reshape(-1)
            grad_samples.append(_sample_flat_tensor(grad, min(int(grad.numel()), sample_count)))

    if not weight_samples:
        return _empty_stats()
    sample = torch.cat(weight_samples) if len(weight_samples) > 1 else weight_samples[0]
    sampled_elements = int(sample.numel())
    scale = float(total_elements) / max(sampled_elements, 1)

    weight_sq_est = sample.square().sum() * scale
    weight_sum_est = sample.sum() * scale
    sparse_est = sample.abs().le(sparsity_epsilon).sum().float() * scale
    grad_sq_est = None
    if grad_samples:
        grad_sample = torch.cat(grad_samples) if len(grad_samples) > 1 else grad_samples[0]
        grad_sq_est = grad_sample.square().sum() * (float(total_elements) / max(int(grad_sample.numel()), 1))

    count_tensor = weight_sq_est.new_tensor(float(total_elements))
    mean = weight_sum_est / count_tensor
    variance = torch.clamp(weight_sq_est / count_tensor - mean.square(), min=0.0)
    weight_norm = torch.sqrt(weight_sq_est)
    grad_norm = torch.sqrt(grad_sq_est) if grad_sq_est is not None else weight_norm.new_tensor(0.0)
    rms = torch.sqrt(weight_sq_est / count_tensor)
    std = torch.sqrt(variance)
    sparsity = (sparse_est / count_tensor) if sparse_est is not None else weight_norm.new_tensor(0.0)

    return {
        "grad_norm": _finite_float(grad_norm),
        "weight_norm": _finite_float(weight_norm),
        "l2_norm": _finite_float(weight_norm),
        "rms": _finite_float(rms),
        "std": _finite_float(std),
        "sparsity": max(0.0, min(_finite_float(sparsity), 1.0)),
        "sampled_elements": sampled_elements,
        "total_elements": total_elements,
    }


def _sample_flat_tensor(data: torch.Tensor, sample_count: int) -> torch.Tensor:
    numel = int(data.numel())
    sample_count = min(max(int(sample_count or 1), 1), numel)
    if sample_count >= numel:
        return data
    return data[:sample_count]


def _empty_stats() -> Dict[str, float]:
    return {
        "grad_norm": 0.0,
        "weight_norm": 0.0,
        "l2_norm": 0.0,
        "rms": 0.0,
        "std": 0.0,
        "sparsity": 0.0,
        "sampled_elements": 0,
        "total_elements": 0,
    }


def _finite_float(value: torch.Tensor) -> float:
    result = float(value.detach().cpu().item())
    return result if math.isfinite(result) else 0.0
