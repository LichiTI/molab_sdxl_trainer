"""CDM-QTA inspired quantized LoRA training probe.

This is a report-only primitive for studying quantized LoRA adapter training.
It does not enable a trainer route. The tiny module uses straight-through
fake-quantized LoRA weights during forward while gradients still update the
fp32 LoRA parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_QTA_BITS = {4, 8}


@dataclass(frozen=True)
class CDMQTALoraProbeConfig:
    enabled: bool = False
    in_features: int = 8
    out_features: int = 8
    rank: int = 4
    alpha: float = 4.0
    quant_bits: int = 8
    quantize_down: bool = True
    quantize_up: bool = True
    bias: bool = False

    def normalized(self) -> "CDMQTALoraProbeConfig":
        bits = int(self.quant_bits if self.quant_bits is not None else 8)
        if bits not in SUPPORTED_QTA_BITS:
            bits = 8
        in_features = max(int(self.in_features or 1), 1)
        out_features = max(int(self.out_features or 1), 1)
        rank = min(max(int(self.rank or 1), 1), min(in_features, out_features))
        alpha = 1.0 if self.alpha is None else float(self.alpha)
        return CDMQTALoraProbeConfig(
            enabled=bool(self.enabled),
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=max(alpha, 0.0),
            quant_bits=bits,
            quantize_down=bool(self.quantize_down),
            quantize_up=bool(self.quantize_up),
            bias=bool(self.bias),
        )


@dataclass(frozen=True)
class CDMQTAQuantizedTensor:
    dequantized: torch.Tensor
    scale: torch.Tensor
    qmin: int
    qmax: int
    bits: int


def fake_quantize_symmetric_ste(weight: torch.Tensor, *, bits: int = 8) -> CDMQTAQuantizedTensor:
    """Return a fake-quantized tensor with straight-through gradients."""
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    bits = int(bits)
    if bits not in SUPPORTED_QTA_BITS:
        raise ValueError(f"CDM-QTA probe supports bits {sorted(SUPPORTED_QTA_BITS)}, got {bits}")
    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax
    max_abs = weight.detach().abs().amax()
    scale = torch.clamp(max_abs / float(qmax), min=torch.finfo(weight.float().dtype).eps).to(device=weight.device)
    quantized = torch.clamp(torch.round(weight.detach() / scale), qmin, qmax)
    dequantized = (quantized * scale).to(dtype=weight.dtype)
    ste = weight + (dequantized - weight).detach()
    return CDMQTAQuantizedTensor(dequantized=ste, scale=scale, qmin=qmin, qmax=qmax, bits=bits)


class CDMQTALoraLinearProbe(nn.Module):
    """Tiny Linear+LoRA probe with optional fake-quantized LoRA branches."""

    def __init__(self, config: CDMQTALoraProbeConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if isinstance(config, Mapping):
            config = CDMQTALoraProbeConfig(**config)
        self.config = (config or CDMQTALoraProbeConfig()).normalized()
        self.base = nn.Linear(self.config.in_features, self.config.out_features, bias=self.config.bias)
        self.lora_down = nn.Linear(self.config.in_features, self.config.rank, bias=False)
        self.lora_up = nn.Linear(self.config.rank, self.config.out_features, bias=False)
        self.scale = self.config.alpha / max(float(self.config.rank), 1.0)
        self.reset_parameters()
        for param in self.base.parameters():
            param.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base.weight, a=5 ** 0.5)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_weight = self._maybe_quantize(self.lora_down.weight, self.config.quantize_down)
        up_weight = self._maybe_quantize(self.lora_up.weight, self.config.quantize_up)
        down = F.linear(x, down_weight)
        delta = F.linear(down, up_weight) * self.scale
        return self.base(x) + delta

    def _maybe_quantize(self, weight: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not self.config.enabled or not enabled:
            return weight
        return fake_quantize_symmetric_ste(weight, bits=self.config.quant_bits).dequantized

    def trainable_lora_param_count(self) -> int:
        return int(self.lora_down.weight.numel() + self.lora_up.weight.numel())


def estimate_cdm_qta_lora_memory(config: CDMQTALoraProbeConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(config, Mapping):
        config = CDMQTALoraProbeConfig(**config)
    cfg = (config or CDMQTALoraProbeConfig()).normalized()
    params = cfg.in_features * cfg.rank + cfg.rank * cfg.out_features
    fp32_param_bytes = params * 4
    quantized_weight_bytes = _ceil_div(params * cfg.quant_bits, 8)
    scale_bytes = 8  # two fp32 scales: down and up
    adamw_state_bytes = params * 8
    fp32_master_bytes = params * 4
    qta_training_bytes = fp32_master_bytes + quantized_weight_bytes + scale_bytes + adamw_state_bytes
    baseline_training_bytes = fp32_param_bytes + adamw_state_bytes
    return {
        "schema_version": 1,
        "adapter_trainable_params": int(params),
        "quant_bits": int(cfg.quant_bits),
        "baseline_training_bytes": int(baseline_training_bytes),
        "qta_training_bytes": int(qta_training_bytes),
        "forward_quantized_weight_bytes": int(quantized_weight_bytes + scale_bytes),
        "fp32_master_bytes": int(fp32_master_bytes),
        "adamw_state_bytes": int(adamw_state_bytes),
        "estimated_training_memory_ratio": float(qta_training_bytes / max(baseline_training_bytes, 1)),
        "estimated_forward_weight_ratio": float((quantized_weight_bytes + scale_bytes) / max(fp32_param_bytes, 1)),
    }


def build_cdm_qta_lora_quant_train_scorecard(
    config: CDMQTALoraProbeConfig | Mapping[str, Any] | None = None,
    *,
    reference_reports: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(config, Mapping):
        config = CDMQTALoraProbeConfig(**config)
    cfg = (config or CDMQTALoraProbeConfig()).normalized()
    memory = estimate_cdm_qta_lora_memory(cfg)
    refs = dict(reference_reports or {})
    blockers = [
        "real_anima_newbie_training_ab_missing",
        "optimizer_state_parity_missing",
        "quality_drift_gate_missing",
        "energy_metering_missing",
    ]
    comparisons = {
        "turbocore": _reference_state(refs.get("turbocore")),
        "fp8": _reference_state(refs.get("fp8")),
        "fused_adamw": _reference_state(refs.get("fused_adamw")),
        "weight_compression": _reference_state(refs.get("weight_compression")),
    }
    return {
        "schema_version": 1,
        "scorecard": "cdm_qta_lora_quant_train_probe_v0",
        "ok": True,
        "probe_ready": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "route": "report_only_quantized_lora_training_probe",
        "config": {
            "enabled": bool(cfg.enabled),
            "in_features": int(cfg.in_features),
            "out_features": int(cfg.out_features),
            "rank": int(cfg.rank),
            "alpha": float(cfg.alpha),
            "quant_bits": int(cfg.quant_bits),
            "quantize_down": bool(cfg.quantize_down),
            "quantize_up": bool(cfg.quantize_up),
        },
        "memory_estimate": memory,
        "comparisons_required": comparisons,
        "promotion_blockers": blockers,
        "recommended_next_step": "run tiny train-step parity and timed A/B before any trainer wiring",
        "notes": [
            "The probe quantizes LoRA branch weights in forward only; fp32 LoRA parameters remain the trainable source of truth.",
            "This is distinct from frozen backbone weight compression and quantized optimizer-state work.",
        ],
    }


def _reference_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"available": False, "source": "", "ready": False}
    ready = bool(value.get("promotion_ready") or value.get("ok") or value.get("ready"))
    return {
        "available": True,
        "source": str(value.get("scorecard") or value.get("source") or ""),
        "ready": ready,
    }


def _ceil_div(numerator: int, denominator: int) -> int:
    return int((int(numerator) + int(denominator) - 1) // int(denominator))


__all__ = [
    "CDMQTALoraLinearProbe",
    "CDMQTALoraProbeConfig",
    "CDMQTAQuantizedTensor",
    "build_cdm_qta_lora_quant_train_scorecard",
    "estimate_cdm_qta_lora_memory",
    "fake_quantize_symmetric_ste",
]
