"""LoRA-KFAC-Lite preconditioner for MN-LoRA P4.

This is intentionally a conservative KFAC approximation for Linear LoRA pairs.
It uses hooks to collect:

    x  -> lora_down input
    h  -> lora_up input
    dy -> lora_up output gradient

Then it preconditions A/down and B/up gradients with small-rank/full +
large-dimension/diagonal factors. Conv and exotic adapter variants are skipped.
The project LoRA layer can bypass lora_down/lora_up modules with an F.linear
fast path, so we also hook the adapter module itself and reconstruct h there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch


class LoRAKFACLiteController:
    """Lightweight KFAC-style gradient preconditioner for LoRA Linear pairs."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        modules: Optional[Mapping[str, Any]] = None,
        ema_decay: float = 0.95,
        damping: float = 1e-3,
        update_interval: int = 1,
        precondition_interval: int = 1,
        max_samples: int = 2048,
        grad_clip: float = 3.0,
        stacked_grad_clip: float = 2.0,
        active_ratio: float = 0.40,
        warmup_steps: int = 10,
        refresh_interval: int = 10,
        min_active_modules: int = 16,
        grad_beta: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        self.enabled = bool(enabled)
        self.ema_decay = max(0.0, min(0.9999, float(ema_decay)))
        self.damping = max(0.0, float(damping))
        self.update_interval = max(1, int(update_interval))
        self.precondition_interval = max(1, int(precondition_interval))
        self.max_samples = max(1, int(max_samples))
        self.grad_clip = max(0.0, float(grad_clip))
        self.stacked_grad_clip = max(0.0, float(stacked_grad_clip))
        self.active_ratio = max(0.0, min(1.0, float(active_ratio)))
        self.warmup_steps = max(0, int(warmup_steps))
        self.refresh_interval = max(1, int(refresh_interval))
        self.min_active_modules = max(0, int(min_active_modules))
        self.grad_beta = max(0.0, min(0.9999, float(grad_beta)))
        self.eps = max(1e-12, float(eps))
        self.stacked_with_gsp = False

        self.modules: Dict[str, Any] = {}
        self.active_modules: set[str] = set()
        self._handles: List[Any] = []
        self._module_handles: Dict[str, List[Any]] = {}
        self._activations: Dict[str, Dict[str, torch.Tensor]] = {}
        self._factors: Dict[str, Dict[str, torch.Tensor]] = {}
        self._module_grad_ema: Dict[str, float] = {}

        self._calls = 0
        self._updates = 0
        self._preconditioned = 0
        self._preconditioned_params = 0
        self._skipped = 0
        self._skip_reasons: Dict[str, int] = {}
        self._registered = 0
        self._grad_norm_ratio_sum = 0.0
        self._grad_norm_ratio_min = 1.0
        self._grad_norm_ratio_max = 0.0
        self._last_grad_norm_ratio = 1.0
        self._last_step = 0
        self._last_updates = 0
        self._last_preconditioned = 0
        self._last_preconditioned_params = 0
        self._last_active_count = 0
        self._last_refresh_step = -1
        self._inactive_hook_skips = 0
        self._inactive_step_skips = 0

        if modules:
            self.register_modules(modules)

    def set_stacked_with_gsp(self, enabled: bool) -> None:
        self.stacked_with_gsp = bool(enabled)

    def _record_skip(self, reason: str) -> None:
        self._skipped += 1
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        for handles in self._module_handles.values():
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    pass
        self._module_handles.clear()

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _adapter(module: Any) -> Any:
        return getattr(module, "lora", module)

    def _resolve_pair_or_reason(self, module: Any) -> Tuple[Optional[Tuple[Any, Any]], str]:
        adapter = self._adapter(module)
        down = getattr(adapter, "lora_down", None)
        up = getattr(adapter, "lora_up", None)
        if down is None or up is None:
            return None, "missing_lora_pair"
        if not hasattr(down, "weight") or not hasattr(up, "weight"):
            return None, "missing_weight"
        if down.weight.ndim != 2 or up.weight.ndim != 2:
            return None, "non_linear_weight"
        if up.weight.shape[1] != down.weight.shape[0]:
            return None, "shape_mismatch"
        return (down, up), "ok"

    def _resolve_pair(self, module: Any) -> Optional[Tuple[Any, Any]]:
        pair, _reason = self._resolve_pair_or_reason(module)
        return pair

    def register_modules(self, modules: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        for name, module in modules.items():
            key = str(name)
            if key in self.modules:
                continue
            pair, reason = self._resolve_pair_or_reason(module)
            if pair is None:
                self._record_skip(f"register_{reason}")
                continue
            down, up = pair
            self.modules[key] = module
            self._registered += 1
            self._activations.setdefault(key, {})
            if not self._hotspot_enabled():
                self._install_module_hooks(key, module, pair)

    def _install_module_hooks(self, name: str, module: Any, pair: Tuple[Any, Any]) -> None:
        if name in self._module_handles:
            return
        down, up = pair
        adapter = self._adapter(module)
        self._module_handles[name] = [
            down.register_forward_hook(self._make_down_forward_hook(name)),
            up.register_forward_hook(self._make_up_forward_hook(name)),
            up.register_full_backward_hook(self._make_up_backward_hook(name)),
            adapter.register_forward_hook(self._make_adapter_forward_hook(name)),
            adapter.register_full_backward_hook(self._make_adapter_backward_hook(name)),
        ]

    def _remove_module_hooks(self, name: str) -> None:
        handles = self._module_handles.pop(name, [])
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    def _hotspot_enabled(self) -> bool:
        return self.active_ratio < 0.999

    def _is_active(self, name: str) -> bool:
        if not self._hotspot_enabled():
            return True
        return name in self.active_modules

    def _should_capture(self, name: str) -> bool:
        if self._is_active(name):
            return True
        self._inactive_hook_skips += 1
        return False

    def _update_module_grad_signal(self, name: str, down: Any, up: Any) -> None:
        values = []
        if getattr(down, "weight", None) is not None and down.weight.grad is not None:
            values.append(down.weight.grad.detach().float().norm())
        if getattr(up, "weight", None) is not None and up.weight.grad is not None:
            values.append(up.weight.grad.detach().float().norm())
        if not values:
            return
        score = float(torch.stack(values).mean().detach().cpu())
        prev = self._module_grad_ema.get(name)
        if prev is None:
            ema = score
        else:
            ema = self.grad_beta * prev + (1.0 - self.grad_beta) * score
        self._module_grad_ema[name] = float(ema)

    def _refresh_active_modules(self, step: int, force: bool = False) -> None:
        if not self._hotspot_enabled():
            self.active_modules = set(self.modules.keys())
            self._last_active_count = len(self.active_modules)
            return
        if int(step) <= self.warmup_steps and not force:
            return
        if not force and self._last_refresh_step >= 0 and int(step) - self._last_refresh_step < self.refresh_interval:
            return
        if not self.modules:
            return
        scored = [(self._module_grad_ema.get(name, 0.0), name) for name in self.modules.keys()]
        scored.sort(key=lambda item: item[0], reverse=True)
        active_count = max(self.min_active_modules, int(len(scored) * self.active_ratio))
        active_count = min(max(active_count, 0), len(scored))
        self.active_modules = {name for _score, name in scored[:active_count]}
        self._last_active_count = len(self.active_modules)
        self._last_refresh_step = int(step)
        for name in list(self._module_handles.keys()):
            if name not in self.active_modules:
                self._remove_module_hooks(name)
        for name in self.active_modules:
            module = self.modules.get(name)
            if module is None:
                continue
            pair = self._resolve_pair(module)
            if pair is not None:
                self._install_module_hooks(name, module, pair)
        for name in list(self._activations.keys()):
            if name not in self.active_modules:
                self._activations.pop(name, None)
        for name in list(self._factors.keys()):
            if name not in self.active_modules:
                self._factors.pop(name, None)

    def _make_down_forward_hook(self, name: str):
        def hook(_module: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
            if not self._should_capture(name):
                return
            if not inputs:
                return
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                self._activations.setdefault(name, {})["x"] = self._flatten_2d(x).detach()

        return hook

    def _make_up_forward_hook(self, name: str):
        def hook(_module: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
            if not self._should_capture(name):
                return
            if not inputs:
                return
            h = inputs[0]
            if isinstance(h, torch.Tensor):
                self._activations.setdefault(name, {})["h"] = self._flatten_2d(h).detach()

        return hook

    def _make_up_backward_hook(self, name: str):
        def hook(_module: Any, _grad_input: Tuple[Any, ...], grad_output: Tuple[Any, ...]) -> None:
            if not self._should_capture(name):
                return
            if not grad_output:
                return
            dy = grad_output[0]
            if isinstance(dy, torch.Tensor):
                self._activations.setdefault(name, {})["dy"] = self._flatten_2d(dy).detach()

        return hook

    def _make_adapter_forward_hook(self, name: str):
        def hook(module: Any, inputs: Tuple[Any, ...], _output: Any) -> None:
            if not self._should_capture(name):
                return
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            pair = self._resolve_pair(module)
            if pair is None:
                return
            down, _up = pair
            try:
                dropped = module.dropout(x) if hasattr(module, "dropout") else x
                hidden = torch.nn.functional.linear(dropped, down.weight)
                entry = self._activations.setdefault(name, {})
                entry["x"] = self._flatten_2d(dropped).detach()
                entry["h"] = self._flatten_2d(hidden).detach()
            except Exception:
                self._record_skip("adapter_forward_capture_error")

        return hook

    def _make_adapter_backward_hook(self, name: str):
        def hook(module: Any, _grad_input: Tuple[Any, ...], grad_output: Tuple[Any, ...]) -> None:
            if not self._should_capture(name):
                return
            if not grad_output:
                return
            dy = grad_output[0]
            if not isinstance(dy, torch.Tensor):
                return
            try:
                scaling = float(getattr(module, "scaling", 1.0) or 1.0)
                self._activations.setdefault(name, {})["dy"] = self._flatten_2d(dy * scaling).detach()
            except Exception:
                self._record_skip("adapter_backward_capture_error")

        return hook

    def _flatten_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.reshape(-1, tensor.shape[-1])
        if flat.shape[0] > self.max_samples:
            index = torch.randperm(flat.shape[0], device=flat.device)[: self.max_samples]
            flat = flat.index_select(0, index)
        return flat.float()

    def _ema_store(self, name: str, key: str, value: torch.Tensor) -> None:
        state = self._factors.setdefault(name, {})
        value = value.detach().float().cpu()
        old = state.get(key)
        if old is None or old.shape != value.shape:
            state[key] = value
        else:
            state[key] = old.mul(self.ema_decay).add(value, alpha=1.0 - self.ema_decay)

    def _inv_spd(self, matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.float()
        eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        damped = matrix + eye * self.damping
        evals, evecs = torch.linalg.eigh(damped)
        evals = evals.clamp_min(self.eps)
        return evecs @ torch.diag(1.0 / evals) @ evecs.T

    def _update_factors(self, name: str, down: Any, up: Any) -> bool:
        entry = self._activations.get(name, {})
        x = entry.get("x")
        h = entry.get("h")
        dy = entry.get("dy")
        if x is None or h is None or dy is None:
            self._record_skip("missing_activation")
            return False
        if down.weight.grad is None and up.weight.grad is None:
            self._record_skip("missing_grad")
            return False
        try:
            x = x.to(device=down.weight.device)
            h = h.to(device=up.weight.device)
            dy = dy.to(device=up.weight.device)
            rows = max(1, min(x.shape[0], h.shape[0], dy.shape[0]))
            x = x[:rows]
            h = h[:rows]
            dy = dy[:rows]
            down_w = down.weight.detach().float()
            up_w = up.weight.detach().float()
            if down_w.ndim != 2 or up_w.ndim != 2:
                self._record_skip("non_linear_weight")
                return False
            d_h = dy @ up_w
            a_in_diag = x.pow(2).mean(dim=0)
            a_out = d_h.T @ d_h / max(1, d_h.shape[0])
            b_in = h.T @ h / max(1, h.shape[0])
            b_out_diag = dy.pow(2).mean(dim=0)
            self._ema_store(name, "a_in_diag", a_in_diag)
            self._ema_store(name, "a_out", a_out)
            self._ema_store(name, "b_in", b_in)
            self._ema_store(name, "b_out_diag", b_out_diag)
            return True
        except Exception:
            self._record_skip("factor_update_error")
            return False

    def _copy_preconditioned(self, grad: torch.Tensor, new_grad: torch.Tensor) -> bool:
        before = grad.detach().float().norm()
        after = new_grad.detach().float().norm()
        changed = False
        if float(before.cpu()) > self.eps and float(after.cpu()) > self.eps:
            ratio = float((after / before.clamp_min(self.eps)).detach().cpu())
            clip_limit = self.grad_clip
            if self.stacked_with_gsp and self.stacked_grad_clip > 0:
                clip_limit = min(clip_limit, self.stacked_grad_clip) if clip_limit > 0 else self.stacked_grad_clip
            if clip_limit > 0 and ratio > clip_limit:
                new_grad = new_grad * (clip_limit / max(ratio, self.eps))
                ratio = clip_limit
            self._grad_norm_ratio_sum += ratio
            self._grad_norm_ratio_min = min(self._grad_norm_ratio_min, ratio)
            self._grad_norm_ratio_max = max(self._grad_norm_ratio_max, ratio)
            self._last_grad_norm_ratio = ratio
            changed = True
        grad.copy_(new_grad.to(dtype=grad.dtype))
        if changed:
            self._preconditioned_params += 1
            self._last_preconditioned_params += 1
        return changed

    def _precondition_grads(self, name: str, down: Any, up: Any) -> bool:
        state = self._factors.get(name)
        if not state:
            self._record_skip("missing_factors")
            return False
        changed = False
        try:
            if down.weight.grad is not None and "a_out" in state and "a_in_diag" in state:
                g = down.weight.grad.data.float()
                a_out_inv = self._inv_spd(state["a_out"].to(g.device))
                a_in_inv_diag = 1.0 / (state["a_in_diag"].to(g.device).sqrt() + self.damping)
                new_g = (a_out_inv @ g) * a_in_inv_diag.view(1, -1)
                changed = self._copy_preconditioned(down.weight.grad.data, new_g) or changed
            elif down.weight.grad is not None:
                self._record_skip("missing_down_factors")
            if up.weight.grad is not None and "b_in" in state and "b_out_diag" in state:
                g = up.weight.grad.data.float()
                b_in_inv = self._inv_spd(state["b_in"].to(g.device))
                b_out_inv_diag = 1.0 / (state["b_out_diag"].to(g.device).sqrt() + self.damping)
                new_g = (g * b_out_inv_diag.view(-1, 1)) @ b_in_inv
                changed = self._copy_preconditioned(up.weight.grad.data, new_g) or changed
            elif up.weight.grad is not None:
                self._record_skip("missing_up_factors")
            return changed
        except Exception:
            self._record_skip("precondition_error")
            return False

    @torch.no_grad()
    def pre_step(self, step: int) -> None:
        if not self.enabled or not self.modules:
            return
        self._calls += 1
        self._last_step = int(step)
        self._last_updates = 0
        self._last_preconditioned = 0
        self._last_preconditioned_params = 0
        for name, module in self.modules.items():
            pair, reason = self._resolve_pair_or_reason(module)
            if pair is None:
                self._record_skip(f"step_{reason}")
                continue
            down, up = pair
            self._update_module_grad_signal(name, down, up)
        self._refresh_active_modules(step)

        for name, module in self.modules.items():
            if not self._is_active(name):
                self._inactive_step_skips += 1
                continue
            pair, reason = self._resolve_pair_or_reason(module)
            if pair is None:
                self._record_skip(f"step_{reason}")
                continue
            down, up = pair
            if step % self.update_interval == 0:
                if self._update_factors(name, down, up):
                    self._updates += 1
                    self._last_updates += 1
            if step % self.precondition_interval == 0:
                if self._precondition_grads(name, down, up):
                    self._preconditioned += 1
                    self._last_preconditioned += 1

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "registered_modules": int(len(self.modules)),
            "registered_total": int(self._registered),
            "calls": int(self._calls),
            "factor_layers": int(len(self._factors)),
            "updates": int(self._updates),
            "preconditioned": int(self._preconditioned),
            "preconditioned_params": int(self._preconditioned_params),
            "skipped": int(self._skipped),
            "skip_reasons": dict(self._skip_reasons),
            "last_step": int(self._last_step),
            "last_updates": int(self._last_updates),
            "last_preconditioned": int(self._last_preconditioned),
            "last_preconditioned_params": int(self._last_preconditioned_params),
            "update_hit_rate": float(self._updates / max(1, self._calls * max(1, len(self.modules)))),
            "precondition_hit_rate": float(self._preconditioned / max(1, self._calls * max(1, len(self.modules)))),
            "ema_decay": float(self.ema_decay),
            "damping": float(self.damping),
            "update_interval": int(self.update_interval),
            "precondition_interval": int(self.precondition_interval),
            "max_samples": int(self.max_samples),
            "active_ratio": float(self.active_ratio),
            "warmup_steps": int(self.warmup_steps),
            "refresh_interval": int(self.refresh_interval),
            "active_modules": int(len(self.active_modules)),
            "hooked_modules": int(len(self._module_handles)),
            "last_active_count": int(self._last_active_count),
            "inactive_hook_skips": int(self._inactive_hook_skips),
            "inactive_step_skips": int(self._inactive_step_skips),
            "grad_clip": float(self.grad_clip),
            "stacked_with_gsp": bool(self.stacked_with_gsp),
            "stacked_grad_clip": float(self.stacked_grad_clip),
            "grad_norm_ratio_avg": float(self._grad_norm_ratio_sum / self._preconditioned_params) if self._preconditioned_params else 1.0,
            "grad_norm_ratio_min": float(self._grad_norm_ratio_min if self._preconditioned_params else 1.0),
            "grad_norm_ratio_max": float(self._grad_norm_ratio_max if self._preconditioned_params else 1.0),
            "grad_norm_ratio_last": float(self._last_grad_norm_ratio),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ema_decay": self.ema_decay,
            "damping": self.damping,
            "update_interval": self.update_interval,
            "precondition_interval": self.precondition_interval,
            "max_samples": self.max_samples,
            "grad_clip": self.grad_clip,
            "stacked_grad_clip": self.stacked_grad_clip,
            "active_ratio": self.active_ratio,
            "warmup_steps": self.warmup_steps,
            "refresh_interval": self.refresh_interval,
            "min_active_modules": self.min_active_modules,
            "stacked_with_gsp": self.stacked_with_gsp,
            "factors": self._factors,
            "module_grad_ema": dict(self._module_grad_ema),
            "telemetry": self.get_telemetry_snapshot(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.enabled = bool(state_dict.get("enabled", self.enabled))
        self.ema_decay = float(state_dict.get("ema_decay", self.ema_decay))
        self.damping = float(state_dict.get("damping", self.damping))
        self.update_interval = int(state_dict.get("update_interval", self.update_interval))
        self.precondition_interval = int(state_dict.get("precondition_interval", self.precondition_interval))
        self.max_samples = int(state_dict.get("max_samples", self.max_samples))
        self.grad_clip = float(state_dict.get("grad_clip", self.grad_clip))
        self.stacked_grad_clip = float(state_dict.get("stacked_grad_clip", self.stacked_grad_clip))
        self.active_ratio = float(state_dict.get("active_ratio", self.active_ratio))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.refresh_interval = int(state_dict.get("refresh_interval", self.refresh_interval))
        self.min_active_modules = int(state_dict.get("min_active_modules", self.min_active_modules))
        self.stacked_with_gsp = bool(state_dict.get("stacked_with_gsp", self.stacked_with_gsp))
        factors = state_dict.get("factors", {})
        if isinstance(factors, dict):
            self._factors = factors
        module_grad_ema = state_dict.get("module_grad_ema", {})
        if isinstance(module_grad_ema, dict):
            self._module_grad_ema = {str(k): float(v) for k, v in module_grad_ema.items()}
