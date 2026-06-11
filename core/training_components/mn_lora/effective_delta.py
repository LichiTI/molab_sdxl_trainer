"""Effective ΔW controls for MN-LoRA.

Unlike per-parameter guards, this module works on paired LoRA factors:

    ΔW_eff = scaling * (lora_up.weight @ lora_down.weight)

That makes the telemetry and optional clipping describe the actual adapter
weight injected into the frozen base layer.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch


class MNLoRAEffectiveDeltaController:
    """Monitor and optionally clip effective LoRA ΔW against base weights."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_norm_ratio: float = 0.25,
        max_rms_ratio: float = 0.05,
        clip_enabled: bool = True,
        fisher_weighted: bool = False,
        fisher_beta: float = 0.95,
        fisher_strength: float = 1.0,
        fisher_min_weight: float = 0.5,
        fisher_max_weight: float = 4.0,
        min_base_norm: float = 1e-8,
        min_base_rms: float = 1e-8,
        modules: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_norm_ratio = max(0.0, float(max_norm_ratio))
        self.max_rms_ratio = max(0.0, float(max_rms_ratio))
        self.clip_enabled = bool(clip_enabled)
        self.fisher_weighted = bool(fisher_weighted)
        self.fisher_beta = max(0.0, min(0.9999, float(fisher_beta)))
        self.fisher_strength = max(0.0, float(fisher_strength))
        self.fisher_min_weight = max(0.0, float(fisher_min_weight))
        self.fisher_max_weight = max(self.fisher_min_weight, float(fisher_max_weight))
        self.min_base_norm = max(0.0, float(min_base_norm))
        self.min_base_rms = max(0.0, float(min_base_rms))
        self.modules: Dict[str, Any] = {}
        if modules:
            self.register_modules(modules)

        self._calls = 0
        self._pairs_seen = 0
        self._pairs_clipped = 0
        self._norm_ratio_sum = 0.0
        self._norm_ratio_max = 0.0
        self._rms_ratio_sum = 0.0
        self._rms_ratio_max = 0.0
        self._scale_min = 1.0
        self._scale_last = 1.0
        self._fisher_ema: Dict[str, float] = {}
        self._fisher_weight_sum = 0.0
        self._fisher_weight_max = 0.0
        self._fisher_updates = 0

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / max(tensor.numel(), 1) ** 0.5

    def register_modules(self, modules: Mapping[str, Any]) -> None:
        for name, module in modules.items():
            if self._resolve_pair(module) is not None:
                self.modules[str(name)] = module

    def _resolve_pair(self, module: Any) -> Optional[tuple[Any, torch.Tensor, torch.Tensor, float]]:
        adapter = getattr(module, "lora", module)
        down = getattr(adapter, "lora_down", None)
        up = getattr(adapter, "lora_up", None)
        if down is None or up is None or not hasattr(down, "weight") or not hasattr(up, "weight"):
            return None
        base = getattr(module, "original", None)
        if base is None:
            base = getattr(adapter, "base_layer", None)
        base_weight = getattr(base, "weight", None)
        if base_weight is None:
            return None
        scaling = float(getattr(adapter, "scaling", 1.0))
        return adapter, down.weight, up.weight, scaling

    def _effective_delta(self, down_weight: torch.Tensor, up_weight: torch.Tensor, scaling: float) -> Optional[torch.Tensor]:
        if down_weight.ndim != 2 or up_weight.ndim != 2:
            return None
        if up_weight.shape[1] != down_weight.shape[0]:
            return None
        return (up_weight.float() @ down_weight.float()) * float(scaling)

    def _gradient_energy(self, down_weight: torch.Tensor, up_weight: torch.Tensor, scaling: float) -> float:
        values = []
        if down_weight.grad is not None:
            values.append(down_weight.grad.detach().float().pow(2).mean())
        if up_weight.grad is not None:
            values.append(up_weight.grad.detach().float().pow(2).mean())
        if not values:
            return 0.0
        energy = torch.stack(values).mean() * (float(scaling) ** 2)
        return float(energy.detach().cpu())

    def _update_fisher_weight(self, name: str, down_weight: torch.Tensor, up_weight: torch.Tensor, scaling: float) -> float:
        if not self.fisher_weighted or self.fisher_strength <= 0:
            return 1.0
        energy = self._gradient_energy(down_weight, up_weight, scaling)
        prev = self._fisher_ema.get(name)
        if prev is None:
            ema = energy
        else:
            ema = self.fisher_beta * prev + (1.0 - self.fisher_beta) * energy
        self._fisher_ema[name] = float(ema)
        weight = 1.0 + self.fisher_strength * (ema ** 0.5)
        weight = max(self.fisher_min_weight, min(self.fisher_max_weight, weight))
        self._fisher_updates += 1
        self._fisher_weight_sum += float(weight)
        self._fisher_weight_max = max(self._fisher_weight_max, float(weight))
        return float(weight)

    @torch.no_grad()
    def apply_all(self) -> None:
        if not self.enabled or not self.modules:
            return
        self._calls += 1

        for name, module in self.modules.items():
            resolved = self._resolve_pair(module)
            if resolved is None:
                continue
            _adapter, down_weight, up_weight, scaling = resolved
            delta = self._effective_delta(down_weight.data, up_weight.data, scaling)
            if delta is None:
                continue

            base_weight = getattr(getattr(module, "original", None), "weight", None)
            if base_weight is None:
                adapter = getattr(module, "lora", module)
                base_layer = getattr(adapter, "base_layer", None)
                base_weight = getattr(base_layer, "weight", None)
            if base_weight is None:
                continue

            base = base_weight.detach().float()
            base_norm = base.norm(2)
            base_rms = self._rms(base)
            delta_norm = delta.norm(2)
            delta_rms = self._rms(delta)
            fisher_weight = self._update_fisher_weight(name, down_weight, up_weight, scaling)

            scale = 1.0
            norm_ratio = 0.0
            rms_ratio = 0.0
            if float(base_norm.detach().cpu()) > self.min_base_norm:
                raw_norm_ratio = float((delta_norm / base_norm.clamp_min(1e-16)).detach().cpu())
                norm_ratio = raw_norm_ratio * fisher_weight
                self._norm_ratio_sum += norm_ratio
                self._norm_ratio_max = max(self._norm_ratio_max, norm_ratio)
                if self.max_norm_ratio > 0 and norm_ratio > self.max_norm_ratio:
                    scale = min(scale, self.max_norm_ratio / max(norm_ratio, 1e-16))

            if float(base_rms.detach().cpu()) > self.min_base_rms:
                raw_rms_ratio = float((delta_rms / base_rms.clamp_min(1e-16)).detach().cpu())
                rms_ratio = raw_rms_ratio * fisher_weight
                self._rms_ratio_sum += rms_ratio
                self._rms_ratio_max = max(self._rms_ratio_max, rms_ratio)
                if self.max_rms_ratio > 0 and rms_ratio > self.max_rms_ratio:
                    scale = min(scale, self.max_rms_ratio / max(rms_ratio, 1e-16))

            self._pairs_seen += 1
            self._scale_last = float(scale)
            self._scale_min = min(self._scale_min, float(scale))
            if self.clip_enabled and scale < 0.999999:
                self._pairs_clipped += 1
                up_weight.data.mul_(float(scale))

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "clip_enabled": bool(self.clip_enabled),
            "registered_pairs": int(len(self.modules)),
            "calls": int(self._calls),
            "pairs_seen": int(self._pairs_seen),
            "pairs_clipped": int(self._pairs_clipped),
            "clip_rate": float(self._pairs_clipped / self._pairs_seen) if self._pairs_seen else 0.0,
            "norm_ratio_avg": float(self._norm_ratio_sum / self._pairs_seen) if self._pairs_seen else 0.0,
            "norm_ratio_max": float(self._norm_ratio_max),
            "rms_ratio_avg": float(self._rms_ratio_sum / self._pairs_seen) if self._pairs_seen else 0.0,
            "rms_ratio_max": float(self._rms_ratio_max),
            "scale_min": float(self._scale_min if self._pairs_seen else 1.0),
            "scale_last": float(self._scale_last),
            "max_norm_ratio": float(self.max_norm_ratio),
            "max_rms_ratio": float(self.max_rms_ratio),
            "fisher_weighted": bool(self.fisher_weighted),
            "fisher_pairs": int(len(self._fisher_ema)),
            "fisher_updates": int(self._fisher_updates),
            "fisher_weight_avg": float(self._fisher_weight_sum / self._fisher_updates) if self._fisher_updates else 1.0,
            "fisher_weight_max": float(self._fisher_weight_max if self._fisher_updates else 1.0),
            "fisher_beta": float(self.fisher_beta),
            "fisher_strength": float(self.fisher_strength),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_norm_ratio": self.max_norm_ratio,
            "max_rms_ratio": self.max_rms_ratio,
            "clip_enabled": self.clip_enabled,
            "fisher_weighted": self.fisher_weighted,
            "fisher_beta": self.fisher_beta,
            "fisher_strength": self.fisher_strength,
            "fisher_min_weight": self.fisher_min_weight,
            "fisher_max_weight": self.fisher_max_weight,
            "fisher_ema": dict(self._fisher_ema),
            "min_base_norm": self.min_base_norm,
            "min_base_rms": self.min_base_rms,
            "telemetry": self.get_telemetry_snapshot(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.enabled = bool(state_dict.get("enabled", self.enabled))
        self.max_norm_ratio = float(state_dict.get("max_norm_ratio", self.max_norm_ratio))
        self.max_rms_ratio = float(state_dict.get("max_rms_ratio", self.max_rms_ratio))
        self.clip_enabled = bool(state_dict.get("clip_enabled", self.clip_enabled))
        self.fisher_weighted = bool(state_dict.get("fisher_weighted", self.fisher_weighted))
        self.fisher_beta = float(state_dict.get("fisher_beta", self.fisher_beta))
        self.fisher_strength = float(state_dict.get("fisher_strength", self.fisher_strength))
        self.fisher_min_weight = float(state_dict.get("fisher_min_weight", self.fisher_min_weight))
        self.fisher_max_weight = float(state_dict.get("fisher_max_weight", self.fisher_max_weight))
        self._fisher_ema = {str(k): float(v) for k, v in dict(state_dict.get("fisher_ema", {})).items()}
        self.min_base_norm = float(state_dict.get("min_base_norm", self.min_base_norm))
        self.min_base_rms = float(state_dict.get("min_base_rms", self.min_base_rms))
