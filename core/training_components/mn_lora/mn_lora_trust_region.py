"""Trust-region update clipping for MN-LoRA.

This controller is deliberately post-optimizer: it observes the actual tensor
delta that AdamW/GSP/TG-WD/Pilot/PlusPlus produced, then clamps only updates
that exceed a conservative relative budget.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class MNLoRATrustRegionController:
    """Clamp per-parameter updates to a small relative trust region."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_update_rms_ratio: float = 0.01,
        max_update_norm_ratio: float = 0.10,
        min_base_rms: float = 1e-8,
        min_base_norm: float = 1e-8,
        hotspot_only: bool = False,
        param_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_update_rms_ratio = max(0.0, float(max_update_rms_ratio))
        self.max_update_norm_ratio = max(0.0, float(max_update_norm_ratio))
        self.min_base_rms = max(0.0, float(min_base_rms))
        self.min_base_norm = max(0.0, float(min_base_norm))
        self.hotspot_only = bool(hotspot_only)
        self.param_names = dict(param_names or {})
        self._calls = 0
        self._skipped = 0
        self._clipped = 0
        self._scale_sum = 0.0
        self._scale_min = 1.0
        self._scale_last = 1.0

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / max(tensor.numel(), 1) ** 0.5

    def _param_name(self, param: torch.nn.Parameter) -> str:
        return self.param_names.get(id(param), "")

    def _should_guard(self, param: torch.nn.Parameter) -> bool:
        name = self._param_name(param).lower()
        if not name:
            return True
        return any(token in name for token in ("lora", "hada", "lokr", "dora", "adapter"))

    def should_prepare(
        self,
        param: torch.nn.Parameter,
        *,
        sparse_tier: str = "hot",
    ) -> bool:
        if not self.enabled or not self._should_guard(param):
            return False
        if self.hotspot_only and sparse_tier != "hot":
            return False
        return True

    @torch.no_grad()
    def apply(self, param: torch.nn.Parameter, old_weight: torch.Tensor, *, sparse_tier: str = "hot") -> None:
        if not self.enabled or not self._should_guard(param):
            return
        if self.hotspot_only and sparse_tier != "hot":
            self._skipped += 1
            return
        delta = param.data - old_weight
        if delta.numel() == 0:
            return

        self._calls += 1
        delta_f = delta.detach().to(dtype=torch.float32)
        old_f = old_weight.detach().to(dtype=torch.float32)
        scale = torch.ones((), device=delta.device, dtype=torch.float32)

        if self.max_update_rms_ratio > 0:
            base_rms_raw = self._rms(old_f)
            if float(base_rms_raw.detach().cpu()) > self.min_base_rms:
                delta_rms = self._rms(delta_f).clamp_min(1e-16)
                rms_cap = base_rms_raw * self.max_update_rms_ratio
                scale = torch.minimum(scale, (rms_cap / delta_rms).to(scale.device))

        if self.max_update_norm_ratio > 0:
            base_norm_raw = old_f.norm(2)
            if float(base_norm_raw.detach().cpu()) > self.min_base_norm:
                delta_norm = delta_f.norm(2).clamp_min(1e-16)
                norm_cap = base_norm_raw * self.max_update_norm_ratio
                scale = torch.minimum(scale, (norm_cap / delta_norm).to(scale.device))

        scale = scale.clamp(max=1.0)
        scale_value = float(scale.detach().cpu())
        self._scale_last = scale_value
        self._scale_sum += scale_value
        self._scale_min = min(self._scale_min, scale_value)
        if scale_value < 0.999999:
            self._clipped += 1
            param.data.copy_(old_weight + delta * scale.to(dtype=delta.dtype))

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "calls": int(self._calls),
            "skipped": int(self._skipped),
            "clipped": int(self._clipped),
            "clip_rate": float(self._clipped / self._calls) if self._calls else 0.0,
            "scale_avg": float(self._scale_sum / self._calls) if self._calls else 1.0,
            "scale_min": float(self._scale_min if self._calls else 1.0),
            "scale_last": float(self._scale_last),
            "max_update_rms_ratio": float(self.max_update_rms_ratio),
            "max_update_norm_ratio": float(self.max_update_norm_ratio),
            "hotspot_only": bool(self.hotspot_only),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_update_rms_ratio": self.max_update_rms_ratio,
            "max_update_norm_ratio": self.max_update_norm_ratio,
            "min_base_rms": self.min_base_rms,
            "min_base_norm": self.min_base_norm,
            "hotspot_only": self.hotspot_only,
            "telemetry": self.get_telemetry_snapshot(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.enabled = bool(state_dict.get("enabled", self.enabled))
        self.max_update_rms_ratio = float(state_dict.get("max_update_rms_ratio", self.max_update_rms_ratio))
        self.max_update_norm_ratio = float(state_dict.get("max_update_norm_ratio", self.max_update_norm_ratio))
        self.min_base_rms = float(state_dict.get("min_base_rms", self.min_base_rms))
        self.min_base_norm = float(state_dict.get("min_base_norm", self.min_base_norm))
        self.hotspot_only = bool(state_dict.get("hotspot_only", self.hotspot_only))
