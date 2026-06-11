import torch
import dataclasses
from collections import OrderedDict
from .trace_guided_wd import TraceGuidedWeightDecay
from .gradient_subspace import GradientSubspaceProjection
from .mn_lora_plus_plus import MNLoRAPlusPlusController
from .mn_lora_trust_region import MNLoRATrustRegionController
from .effective_delta import MNLoRAEffectiveDeltaController
from .lora_kfac_lite import LoRAKFACLiteController
from .fisher_ewc import MNLoRAFisherEWCController
from .gradient_conflict import MNLoRAGradientConflictController
from core.training_pilot import TrainingPilot
from typing import Dict, Any, Optional

class MNLoRAOptimizer(torch.optim.Optimizer):
    """
    Manifold-Aware Natural LoRA Optimizer Wrapper
    
    Wraps a base optimizer (AdamW, etc.) and injects:
    1. TG-WD: Trace-Guided Weight Decay (via Gradient History)
    2. GSP: Gradient Subspace Projection (via SVD)
    3. TrainingPilot: Automated Layer-wise LR Adjustment
    """
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        enable_tgwd: bool = True,
        enable_gsp: bool = True,
        enable_pilot: bool = True,      # 默认开启自动驾驶
        tgwd_config: Dict = None,
        gsp_config: Dict = None,
        pilot_config: Dict = None,
        plus_plus_config: Dict = None,
        kfac_lite_config: Dict = None,
        trust_region_config: Dict = None,
        effective_delta_config: Dict = None,
        fisher_ewc_config: Dict = None,
        gradient_conflict_config: Dict = None,
        lora_modules: Optional[Dict[str, Any]] = None,
        param_names: Optional[Dict[int, str]] = None,
        on_pilot_decision: Optional[Any] = None,  # 新增: 决策回调
        max_cache_entries: int = 500,    # V3.1: 可配置的梯度缓存上限
    ):
        self.base_optimizer = base_optimizer
        self.enable_tgwd = enable_tgwd
        self.enable_gsp = enable_gsp
        self.enable_pilot = enable_pilot
        self.on_pilot_decision = on_pilot_decision
        
        self.tgwd = TraceGuidedWeightDecay(**(tgwd_config or {})) if enable_tgwd else None
        self.gsp = GradientSubspaceProjection(**(gsp_config or {})) if enable_gsp else None
        self.pilot = TrainingPilot(**(pilot_config or {})) if enable_pilot else None
        plus_plus_config = dict(plus_plus_config or {})
        self.plus_plus = None
        if plus_plus_config.pop("enabled", False):
            plus_plus_config.setdefault("param_names", param_names or {})
            self.plus_plus = MNLoRAPlusPlusController(**plus_plus_config)
        self.param_names = dict(param_names or {})
        kfac_lite_config = dict(kfac_lite_config or {})
        kfac_lite_config.setdefault("modules", lora_modules or {})
        self.kfac_lite = LoRAKFACLiteController(**kfac_lite_config)
        self.kfac_lite.set_stacked_with_gsp(self.gsp is not None)
        trust_region_config = dict(trust_region_config or {})
        trust_region_config.setdefault("param_names", self.param_names)
        self.trust_region = MNLoRATrustRegionController(**trust_region_config)
        effective_delta_config = dict(effective_delta_config or {})
        effective_delta_config.setdefault("modules", lora_modules or {})
        self.effective_delta = MNLoRAEffectiveDeltaController(**effective_delta_config)
        all_params = [p for group in self.base_optimizer.param_groups for p in group.get("params", [])]
        fisher_ewc_config = dict(fisher_ewc_config or {})
        fisher_ewc_config.setdefault("params", all_params)
        fisher_ewc_config.setdefault("param_names", self.param_names)
        self.fisher_ewc = MNLoRAFisherEWCController(**fisher_ewc_config)
        gradient_conflict_config = dict(gradient_conflict_config or {})
        self.gradient_conflict = MNLoRAGradientConflictController(**gradient_conflict_config)
        
        self.pilot_interval = 50
        self._step_count = 0
        
        # LRU Cache for Gradient History (optimized for consumer VRAM)
        # V3.1 Fix: max_cache_entries is now configurable via constructor.
        # For large models (Flux/SD3), tune this down to avoid OOM.
        # Each entry holds a full gradient clone, so memory = entries * avg_grad_size.
        self._prev_grads = OrderedDict()
        self.max_cache_entries = max_cache_entries
        
        # Proxy param_groups
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults

    def _param_key(self, param: torch.nn.Parameter) -> str:
        return self.param_names.get(id(param), str(id(param)))

    def _gsp_sparse_tier(self, param_id: str) -> str:
        if self.gsp is None:
            return "hot"
        return self.gsp.layer_sparse_tier.get(param_id, "hot")

    def state_dict(self):
        state = self.base_optimizer.state_dict()
        if self.plus_plus is not None:
            state = dict(state)
            state["mn_lora_plus_plus"] = self.plus_plus.state_dict()
        if self.trust_region is not None and self.trust_region.enabled:
            state = dict(state)
            state["mn_lora_trust_region"] = self.trust_region.state_dict()
        if self.effective_delta is not None and self.effective_delta.enabled:
            state = dict(state)
            state["mn_lora_effective_delta"] = self.effective_delta.state_dict()
        if self.kfac_lite is not None and self.kfac_lite.enabled:
            state = dict(state)
            state["mn_lora_kfac_lite"] = self.kfac_lite.state_dict()
        if self.fisher_ewc is not None and self.fisher_ewc.enabled:
            state = dict(state)
            state["mn_lora_fisher_ewc"] = self.fisher_ewc.state_dict()
        if self.gradient_conflict is not None and self.gradient_conflict.enabled:
            state = dict(state)
            state["mn_lora_gradient_conflict"] = self.gradient_conflict.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        plus_plus_state = None
        if isinstance(state_dict, dict) and "mn_lora_plus_plus" in state_dict:
            state_dict = dict(state_dict)
            plus_plus_state = state_dict.pop("mn_lora_plus_plus")
        trust_region_state = None
        if isinstance(state_dict, dict) and "mn_lora_trust_region" in state_dict:
            state_dict = dict(state_dict)
            trust_region_state = state_dict.pop("mn_lora_trust_region")
        effective_delta_state = None
        if isinstance(state_dict, dict) and "mn_lora_effective_delta" in state_dict:
            state_dict = dict(state_dict)
            effective_delta_state = state_dict.pop("mn_lora_effective_delta")
        kfac_lite_state = None
        if isinstance(state_dict, dict) and "mn_lora_kfac_lite" in state_dict:
            state_dict = dict(state_dict)
            kfac_lite_state = state_dict.pop("mn_lora_kfac_lite")
        fisher_ewc_state = None
        if isinstance(state_dict, dict) and "mn_lora_fisher_ewc" in state_dict:
            state_dict = dict(state_dict)
            fisher_ewc_state = state_dict.pop("mn_lora_fisher_ewc")
        gradient_conflict_state = None
        if isinstance(state_dict, dict) and "mn_lora_gradient_conflict" in state_dict:
            state_dict = dict(state_dict)
            gradient_conflict_state = state_dict.pop("mn_lora_gradient_conflict")
        self.base_optimizer.load_state_dict(state_dict)
        if self.plus_plus is not None and plus_plus_state is not None:
            self.plus_plus.load_state_dict(plus_plus_state)
        if self.trust_region is not None and trust_region_state is not None:
            self.trust_region.load_state_dict(trust_region_state)
        if self.effective_delta is not None and effective_delta_state is not None:
            self.effective_delta.load_state_dict(effective_delta_state)
        if self.kfac_lite is not None and kfac_lite_state is not None:
            self.kfac_lite.load_state_dict(kfac_lite_state)
        if self.fisher_ewc is not None and fisher_ewc_state is not None:
            self.fisher_ewc.load_state_dict(fisher_ewc_state)
        if self.gradient_conflict is not None and gradient_conflict_state is not None:
            self.gradient_conflict.load_state_dict(gradient_conflict_state)

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe MN-LoRA runtime snapshot for manifests/reports."""
        gsp_snapshot = self.gsp.get_telemetry_snapshot() if self.gsp is not None else {}
        tgwd_stats: Dict[str, Any] = {}
        if self.tgwd is not None:
            try:
                tgwd_stats = self.tgwd.get_stats()
            except Exception:
                tgwd_stats = {}
        return {
            "type": type(self).__name__,
            "base_optimizer_type": type(self.base_optimizer).__name__,
            "step_count": int(self._step_count),
            "gsp_enabled": self.gsp is not None,
            "tgwd_enabled": self.tgwd is not None,
            "pilot_enabled": self.pilot is not None,
            "plus_plus_enabled": self.plus_plus is not None,
            "kfac_lite_enabled": bool(self.kfac_lite is not None and self.kfac_lite.enabled),
            "trust_region_enabled": bool(self.trust_region is not None and self.trust_region.enabled),
            "effective_delta_enabled": bool(self.effective_delta is not None and self.effective_delta.enabled),
            "fisher_ewc_enabled": bool(self.fisher_ewc is not None and self.fisher_ewc.enabled),
            "gradient_conflict_enabled": bool(self.gradient_conflict is not None and self.gradient_conflict.enabled),
            "prev_grad_cache_entries": len(self._prev_grads),
            "gsp": gsp_snapshot,
            "tgwd": tgwd_stats,
            "kfac_lite": self.kfac_lite.get_telemetry_snapshot() if self.kfac_lite is not None else {},
            "trust_region": self.trust_region.get_telemetry_snapshot() if self.trust_region is not None else {},
            "effective_delta": self.effective_delta.get_telemetry_snapshot() if self.effective_delta is not None else {},
            "fisher_ewc": self.fisher_ewc.get_telemetry_snapshot() if self.fisher_ewc is not None else {},
            "gradient_conflict": self.gradient_conflict.get_telemetry_snapshot() if self.gradient_conflict is not None else {},
        }
        
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def step(self, closure=None):
        self._step_count += 1
        loss = None
        if closure is not None:
            loss = closure()

        if self.gsp:
            phase = "warmup" if self._step_count <= self.gsp.warmup_steps else "steady"
            self.gsp.step(self._step_count, phase)

        all_params = [p for group in self.param_groups for p in group.get("params", [])]
        if self.fisher_ewc and self.fisher_ewc.enabled:
            main_grads = {}
            for p in all_params:
                if p.grad is not None:
                    main_grads[self._param_key(p)] = p.grad.detach().clone()
            ewc_grads, _ewc_stats = self.fisher_ewc.build_penalty_grads(
                all_params,
                step=self._step_count,
                update_fisher=True,
            )
            if ewc_grads:
                if self.gradient_conflict and self.gradient_conflict.enabled and main_grads:
                    resolved_grads, _conflict_stats = self.gradient_conflict.resolve([main_grads, ewc_grads])
                    for p in all_params:
                        resolved = resolved_grads.get(self._param_key(p))
                        if resolved is None:
                            continue
                        if p.grad is None:
                            p.grad = resolved.detach().to(device=p.device, dtype=p.dtype)
                        else:
                            p.grad.copy_(resolved.detach().to(device=p.grad.device, dtype=p.grad.dtype))
                else:
                    for p in all_params:
                        penalty = ewc_grads.get(self._param_key(p))
                        if penalty is None:
                            continue
                        if p.grad is None:
                            p.grad = penalty.detach().to(device=p.device, dtype=p.dtype)
                        else:
                            p.grad.add_(penalty.detach().to(device=p.grad.device, dtype=p.grad.dtype))

        # --- TrainingPilot: Decision checkpoint ---
        if self.pilot and self._step_count % self.pilot_interval == 0 and self.gsp:
            decisions = self.pilot.analyze_and_decide(self.gsp.get_stats())
            if decisions and self.on_pilot_decision:
                dict_decisions = [dataclasses.asdict(d) for d in decisions]
                for d in dict_decisions:
                    d['step'] = self._step_count
                self.on_pilot_decision(dict_decisions)
        
        # --- Phase 1: Pre-step gradient manipulation (GSP only) ---
        if self.kfac_lite and self.kfac_lite.enabled:
            self.kfac_lite.pre_step(self._step_count)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = self._param_key(p)
                grad = p.grad.data
                
                # --- TG-WD: Update trace cache (does NOT modify grad) ---
                if self.tgwd and self.tgwd.should_update(self._step_count):
                    if param_id in self._prev_grads:
                        # Safety check for KeyError race condition
                        if param_id in self._prev_grads:
                            self._prev_grads.move_to_end(param_id)
                        self.tgwd.update_trace_for_layer(param_id, grad, self._prev_grads[param_id])
                
                # Cache gradient for next TG-WD update
                if self.tgwd:
                    next_step = self._step_count + 1
                    if self.tgwd.should_update(next_step):
                        if len(self._prev_grads) >= self.max_cache_entries and param_id not in self._prev_grads:
                            self._prev_grads.popitem(last=False)
                        self._prev_grads[param_id] = grad.detach().clone()
                        self._prev_grads.move_to_end(param_id)
                
                # --- GSP: Project gradient to subspace ---
                if self.gsp:
                    self.gsp.observe_gradient(param_id, grad)
                    need_init = param_id not in self.gsp.V_cache
                    need_periodic = self._step_count % self.gsp.update_interval == 0
                    need_lazy = self.gsp.should_trigger_svd_update(param_id)
                    should_update_cache = self.gsp.should_update_subspace_cache(param_id, has_cache=not need_init)
                    if should_update_cache and (need_init or need_periodic or need_lazy):
                        self.gsp.update_subspace(param_id, p.data, force=need_init)
                    projected_grad = self.gsp.project_gradient(param_id, grad)
                    p.grad.data.copy_(projected_grad)
        
        # --- Phase 2: Cache old weights for Post-Step Scaling ---
        old_weights = {}
        if self.pilot or self.plus_plus or (self.trust_region is not None and self.trust_region.enabled):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        param_id = self._param_key(p)
                        needs_old_weight = bool(self.pilot or self.plus_plus)
                        if self.trust_region is not None and self.trust_region.enabled:
                            needs_old_weight = needs_old_weight or self.trust_region.should_prepare(
                                p,
                                sparse_tier=self._gsp_sparse_tier(param_id),
                            )
                        if needs_old_weight:
                            old_weights[param_id] = p.data.clone()
        
        # --- Phase 3: Base optimizer step ---
        result = self.base_optimizer.step(closure)
        
        # --- Phase 4: Post-Step Adjustments ---
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = self._param_key(p)
                
                # --- TrainingPilot: Post-Step Scaling for AdamW compatibility ---
                # w_final = w_old + (w_new - w_old) * multiplier
                if self.pilot and param_id in old_weights:
                    lr_mult = self.pilot.get_multiplier(param_id)
                    if lr_mult != 1.0:
                        w_old = old_weights[param_id]
                        delta = p.data - w_old
                        p.data.copy_(w_old + delta * lr_mult)

                # --- MN-LoRA++: rank/module-wise delta scaling ---
                if self.plus_plus and param_id in old_weights:
                    self.plus_plus.apply(p, old_weights[param_id])

                # --- P3 Trust Region: final post-step update boundary ---
                if self.trust_region and self.trust_region.enabled and param_id in old_weights:
                    self.trust_region.apply(p, old_weights[param_id], sparse_tier=self._gsp_sparse_tier(param_id))
                 
                # --- TG-WD: Decoupled Weight Decay (Post-Step) ---
                # w = w * (1 - lr * lambda_tgwd)
                if self.tgwd:
                    # V3.1 Fix: Pass device to avoid cuda/cpu mismatch
                    decay_factor = self.tgwd.get_decay_factor(param_id, device=str(p.device))
                    p.data.mul_(1.0 - lr * decay_factor)

        # --- P3.1 Effective ΔW: final paired LoRA weight boundary ---
        if self.effective_delta and self.effective_delta.enabled:
            self.effective_delta.apply_all()
         
        return result

