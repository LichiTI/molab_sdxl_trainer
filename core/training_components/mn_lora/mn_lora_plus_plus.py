"""Lightweight adaptive update scaling for MN-LoRA.

MN-LoRA++ is intentionally implemented as a wrapper-side controller, not as a
new optimizer.  It observes post-optimizer deltas at tensor and LoRA-rank
granularity, then scales the just-applied update conservatively.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class MNLoRAPlusPlusController:
    """Tensor/rank adaptive multiplier controller for LoRA parameters."""

    def __init__(
        self,
        *,
        rank_adapt: bool = True,
        module_adapt: bool = True,
        lr_up: float = 1.01,
        lr_down: float = 0.95,
        min_mult: float = 0.25,
        max_mult: float = 2.0,
        lora_up_max_mult: float = 1.25,
        protected_max_mult: float = 1.0,
        update_rms_cap: float = 0.01,
        param_names: Optional[Dict[int, str]] = None,
    ) -> None:
        if min_mult <= 0 or max_mult < min_mult:
            raise ValueError("MN-LoRA++ requires 0 < min_mult <= max_mult.")
        self.rank_adapt = bool(rank_adapt)
        self.module_adapt = bool(module_adapt)
        self.lr_up = float(lr_up)
        self.lr_down = float(lr_down)
        self.min_mult = float(min_mult)
        self.max_mult = float(max_mult)
        self.lora_up_max_mult = float(lora_up_max_mult)
        self.protected_max_mult = float(protected_max_mult)
        self.update_rms_cap = max(0.0, float(update_rms_cap))
        self.param_names = dict(param_names or {})
        self._state: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / max(tensor.numel(), 1) ** 0.5

    def _param_name(self, param: torch.nn.Parameter) -> str:
        return self.param_names.get(id(param), "")

    def _role(self, param: torch.nn.Parameter) -> str:
        lowered = self._param_name(param).lower()
        if any(token in lowered for token in ("magnitude", "dora", "norm")):
            return "protected"
        if any(token in lowered for token in ("lora_up", "lora_b", "hada_w2", "lokr_w1_b")):
            return "up"
        if any(token in lowered for token in ("lora_down", "lora_a", "hada_w1", "lokr_w1_a")):
            return "down"
        return "generic"

    def _max_for_role(self, role: str) -> float:
        if role == "protected":
            return min(self.protected_max_mult, self.max_mult)
        if role == "up":
            return min(self.lora_up_max_mult, self.max_mult)
        return self.max_mult

    def _rank_axis(self, param: torch.nn.Parameter, role: str) -> Optional[int]:
        if param.ndim < 2:
            return None
        if role == "up":
            return 1
        return 0

    def _state_for(self, param_id: str, delta: torch.Tensor, role: str, rank_axis: Optional[int]) -> Dict[str, Any]:
        state = self._state.setdefault(param_id, {})
        if "module_mult" not in state:
            state["module_mult"] = 1.0
            state["prev_delta"] = None
        if self.rank_adapt and rank_axis is not None:
            rank_count = int(delta.shape[rank_axis])
            rank_mult = state.get("rank_mult")
            if not torch.is_tensor(rank_mult) or int(rank_mult.numel()) != rank_count:
                state["rank_mult"] = torch.ones(rank_count, device=delta.device, dtype=torch.float32)
                state["prev_rank_delta"] = None
        state["role"] = role
        return state

    def _trend(self, current: torch.Tensor, previous: Optional[torch.Tensor]) -> float:
        if previous is None:
            return 0.0
        prev = previous.to(device=current.device, dtype=torch.float32)
        cur = current.to(dtype=torch.float32)
        denom = cur.norm().mul(prev.norm()).clamp_min(1e-16)
        return float(torch.dot(cur.flatten(), prev.flatten()).div(denom).detach().cpu())

    def _adapt_scalar(self, value: float, trend: float, role: str) -> float:
        if trend > 0.25:
            value *= self.lr_up
        elif trend < -0.05:
            value *= self.lr_down
        return max(self.min_mult, min(self._max_for_role(role), value))

    def _adapt_rank(self, state: Dict[str, Any], delta: torch.Tensor, rank_axis: int, role: str) -> torch.Tensor:
        rank_mult = state["rank_mult"].to(device=delta.device, dtype=torch.float32)
        moved = delta.detach().to(dtype=torch.float32).movedim(rank_axis, 0).flatten(1)
        previous = state.get("prev_rank_delta")
        if torch.is_tensor(previous) and previous.shape == moved.shape:
            prev = previous.to(device=moved.device, dtype=torch.float32)
            denom = moved.norm(dim=1).mul(prev.norm(dim=1)).clamp_min(1e-16)
            trend = (moved * prev).sum(dim=1).div(denom)
            rank_mult = torch.where(trend > 0.25, rank_mult * self.lr_up, rank_mult)
            rank_mult = torch.where(trend < -0.05, rank_mult * self.lr_down, rank_mult)
            rank_mult = rank_mult.clamp_(self.min_mult, self._max_for_role(role))
        state["prev_rank_delta"] = moved.detach().cpu()
        state["rank_mult"] = rank_mult.detach().cpu()
        shape = [1] * delta.ndim
        shape[rank_axis] = int(rank_mult.numel())
        return rank_mult.reshape(shape).to(device=delta.device, dtype=delta.dtype)

    @torch.no_grad()
    def apply(self, param: torch.nn.Parameter, old_weight: torch.Tensor) -> None:
        delta = param.data - old_weight
        if delta.numel() == 0:
            return
        param_id = str(id(param))
        role = self._role(param)
        rank_axis = self._rank_axis(param, role)
        state = self._state_for(param_id, delta, role, rank_axis)

        scale: Any = 1.0
        if self.module_adapt:
            trend = self._trend(delta, state.get("prev_delta"))
            state["module_mult"] = self._adapt_scalar(float(state["module_mult"]), trend, role)
            scale = float(state["module_mult"])
        state["prev_delta"] = delta.detach().to(dtype=torch.float32).cpu()

        if self.rank_adapt and rank_axis is not None:
            rank_scale = self._adapt_rank(state, delta, rank_axis, role)
            scale = rank_scale * float(scale)

        scaled_delta = delta * scale
        if self.update_rms_cap > 0:
            base_rms = self._rms(old_weight.detach().to(dtype=torch.float32))
            if float(base_rms.detach().cpu()) > 1e-6:
                cap = base_rms * self.update_rms_cap
                scaled_rms = self._rms(scaled_delta.detach().to(dtype=torch.float32)).clamp_min(1e-16)
                scaled_delta = scaled_delta * (cap / scaled_rms).clamp_max(1.0).to(dtype=scaled_delta.dtype)
        param.data.copy_(old_weight + scaled_delta)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rank_adapt": self.rank_adapt,
            "module_adapt": self.module_adapt,
            "lr_up": self.lr_up,
            "lr_down": self.lr_down,
            "min_mult": self.min_mult,
            "max_mult": self.max_mult,
            "lora_up_max_mult": self.lora_up_max_mult,
            "protected_max_mult": self.protected_max_mult,
            "update_rms_cap": self.update_rms_cap,
            "state": self._state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.rank_adapt = bool(state_dict.get("rank_adapt", self.rank_adapt))
        self.module_adapt = bool(state_dict.get("module_adapt", self.module_adapt))
        self.lr_up = float(state_dict.get("lr_up", self.lr_up))
        self.lr_down = float(state_dict.get("lr_down", self.lr_down))
        self.min_mult = float(state_dict.get("min_mult", self.min_mult))
        self.max_mult = float(state_dict.get("max_mult", self.max_mult))
        self.lora_up_max_mult = float(state_dict.get("lora_up_max_mult", self.lora_up_max_mult))
        self.protected_max_mult = float(state_dict.get("protected_max_mult", self.protected_max_mult))
        self.update_rms_cap = float(state_dict.get("update_rms_cap", self.update_rms_cap))
        self._state = dict(state_dict.get("state", {}))
