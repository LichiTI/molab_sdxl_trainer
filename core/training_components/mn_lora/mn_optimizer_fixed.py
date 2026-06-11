import torch
import dataclasses
from collections import OrderedDict
from .trace_guided_wd import TraceGuidedWeightDecay
from .gradient_subspace import GradientSubspaceProjection
from core.training_pilot import TrainingPilot
from typing import Dict, Any, Optional, List

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
        enable_pilot: bool = True,
        tgwd_config: Dict = None,
        gsp_config: Dict = None,
        pilot_config: Dict = None,
        on_pilot_decision: Optional[Any] = None
    ):
        self.base_optimizer = base_optimizer
        self.enable_tgwd = enable_tgwd
        self.enable_gsp = enable_gsp
        self.enable_pilot = enable_pilot
        self.on_pilot_decision = on_pilot_decision
        
        # Initialize modules
        self.tgwd = TraceGuidedWeightDecay(**(tgwd_config or {})) if enable_tgwd else None
        self.gsp = GradientSubspaceProjection(**(gsp_config or {})) if enable_gsp else None
        self.pilot = TrainingPilot(**(pilot_config or {})) if enable_pilot else None
        
        self.pilot_interval = 50
        self._step_count = 0
        
        # LRU Cache for Gradient History (optimized for consumer VRAM)
        self._prev_grads = OrderedDict() 
        self.max_cache_entries = 500  # TODO: Make dynamic based on model size
        
        # Proxy param_groups
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults

    def state_dict(self):
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def step(self, closure=None):
        self._step_count += 1
        loss = None
        if closure is not None:
            loss = closure()

        # --- Phase 0: TrainingPilot Decision & Data Collection ---
        # 收集统计信息并做出决策
        pilot_multipliers = {}
        if self.pilot and self._step_count % self.pilot_interval == 0:
            # 1. 收集当前层的统计信息 (norm, grad_norm)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_id = str(id(p))
                    self.pilot.update_stats(
                        param_id,
                        norm=p.data.norm().item(),
                        grad_norm=p.grad.norm().item(),
                    )
            
            # 2. 让 Pilot 做出决策
            pilot_multipliers = self.pilot.decide_lr_multipliers()
            
            # 3. 触发回调 (如果需要)
            if self.on_pilot_decision:
                # 将决策转换为字典列表以便序列化
                decisions = [{"layer": k, "multiplier": v} for k, v in pilot_multipliers.items()]
                self.on_pilot_decision(decisions)

        # --- Phase 1: Pre-step gradient manipulation (GSP & TG-WD) ---
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = str(id(p))
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
                        # V2.6: Keep tensor on GPU, avoid .item() sync
                        self._prev_grads[param_id] = grad.detach().clone()
                        self._prev_grads.move_to_end(param_id)
                
                # --- GSP: Project gradient to subspace ---
                if self.gsp:
                    # Update subspace if needed
                    if self._step_count % self.gsp.update_interval == 0:
                        self.gsp.update_subspace(param_id, p.data)
                    
                    # Project gradient
                    projected_grad = self.gsp.project_gradient(param_id, grad)
                    p.grad.data.copy_(projected_grad)

        # --- Phase 2: Cache old weights for Post-Step Scaling ---
        old_weights = {}
        if self.pilot and pilot_multipliers:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        param_id = str(id(p))
                        # Only cache if we have a non-1.0 multiplier for this layer
                        if pilot_multipliers.get(param_id, 1.0) != 1.0:
                            old_weights[param_id] = p.data.clone()

        # --- Phase 3: Base optimizer step ---
        result = self.base_optimizer.step(closure)
        
        # --- Phase 4: Post-Step Adjustments ---
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = str(id(p))
                
                # --- TrainingPilot: Post-Step Scaling for AdamW compatibility ---
                # w_final = w_old + (w_new - w_old) * multiplier
                if self.pilot and param_id in old_weights:
                    lr_mult = pilot_multipliers.get(param_id, 1.0)
                    if lr_mult != 1.0:
                        w_old = old_weights[param_id]
                        delta = p.data - w_old
                        p.data.copy_(w_old + delta * lr_mult)
                
                # --- TG-WD: Decoupled Weight Decay (Post-Step) ---
                # w = w * (1 - lr * lambda_tgwd)
                if self.tgwd:
                    decay_factor = self.tgwd.get_decay_factor(param_id)
                    # decay_factor is a Tensor, ensure it's on the same device
                    if isinstance(decay_factor, torch.Tensor):
                        decay_factor = decay_factor.to(p.data.device)
                    p.data.mul_(1.0 - lr * decay_factor)
        
        return result
