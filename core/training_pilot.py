from typing import Dict, Any, List, Optional
import dataclasses
from .constants import PILOT_EMA_ALPHA, PILOT_EMA_BASELINE_SPIKE, PILOT_EMA_BASELINE_DROP

@dataclasses.dataclass
class PilotDecision:
    layer_name: str
    multiplier: float
    reason: str  # snake_case key for i18n
    metric_value: float
    k_ratio_mod: float = 0.0  # V2.1: Allow Pilot to adjust subspace dimension

class TrainingPilot:
    """
    TrainingPilot: Lulynx 训练器的"自动驾驶"模块。
    
    设计理念：
    1. 动态诊断：基于层级的 Norm、梯度和残差数据，判断模型生长状态。
    2. 精准干预：针对性调整不同层的学习率（Layer-wise Adaptive LR）。
    3. 稳定优先：调整幅度受限，并设有冷却时间，防止震荡。
    """
    
    def __init__(
        self,
        strategy: str = "population",    # "heuristic", "population", "ema", "pid"
        target_norm_high: float = 1.5,
        target_norm_low: float = 0.05,
        suppress_factor: float = 0.5,
        boost_factor: float = 1.2,
        min_multiplier: float = 0.1,
        max_multiplier: float = 3.0,
        # PID params
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.05
    ):
        self.strategy = strategy
        self.target_norm_high = target_norm_high
        self.target_norm_low = target_norm_low
        self.suppress_factor = suppress_factor
        self.boost_factor = boost_factor
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        
        # PID Params
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        
        # State Tracking
        self.multipliers: Dict[str, float] = {}
        self.ema_history: Dict[str, float] = {}
        self.pid_integral: Dict[str, float] = {}
        self.pid_prev_error: Dict[str, float] = {}
        
    def analyze_and_decide(self, layer_stats: Dict[str, Any]) -> List[PilotDecision]:
        decisions = []
        
        # Pre-calculation for Population Strategy
        pop_mean = 0.0
        pop_std = 0.0
        if self.strategy == "population" and layer_stats:
            norms = []
            for _, stats in layer_stats.items():
                n = getattr(stats, 'projected_norm', stats.get('projected_norm', 1.0) if isinstance(stats, dict) else 1.0)
                norms.append(n)
            import numpy as np
            if len(norms) > 1:
                pop_mean = float(np.mean(norms))
                pop_std = float(np.std(norms))
            else:
                pop_mean = norms[0] if norms else 1.0
                pop_std = 0.0
        
        for name, stats in layer_stats.items():
            current_mult = self.multipliers.get(name, 1.0)
            new_mult = current_mult
            reason = "normal"
            
            # Extract Norm & Residual
            norm = getattr(stats, 'projected_norm', 1.0)
            residual = getattr(stats, 'residual_ratio', 0.0)
            if isinstance(stats, dict):
                norm = stats.get('projected_norm', 1.0)
                residual = stats.get('residual_ratio', 0.0)
            metric_val = norm
            
            # V2.1: k_ratio modification logic
            k_mod = 0.0
            
            # --- Strategy Dispatch ---
            
            if self.strategy == "population":
                # Strategy 1: Population Statistics (Relative)
                limit_high = pop_mean + 2 * pop_std
                limit_low = max(0.01, pop_mean - 1.5 * pop_std) # Prevent negative threshold
                
                # Check for zero variance
                if pop_std == 0:
                    continue  # Population strategy requires variance to find outliers
                
                if norm > limit_high:
                    new_mult = current_mult * self.suppress_factor
                    reason = "overfit_pop"
                elif norm < limit_low:
                    new_mult = current_mult * self.boost_factor
                    reason = "underfit_pop"
                    
            elif self.strategy == "ema":
                # Strategy 2: EMA Trend
                baseline = self.ema_history.get(name, norm) # First time init with current
                # Update EMA
                self.ema_history[name] = (1 - PILOT_EMA_ALPHA) * baseline + PILOT_EMA_ALPHA * norm
                
                # Compare against baseline
                # If current spike is huge vs historical baseline
                if norm > baseline * PILOT_EMA_BASELINE_SPIKE:
                    new_mult = current_mult * self.suppress_factor
                    reason = "spike_ema"
                elif norm < baseline * PILOT_EMA_BASELINE_DROP:
                    new_mult = current_mult * self.boost_factor
                    reason = "drop_ema"
                    
            elif self.strategy == "pid":
                # Strategy 3: PID Control
                target = 1.0 # Ideal norm setpoint
                error = norm - target
                
                integral = self.pid_integral.get(name, 0.0) + error
                prev_error = self.pid_prev_error.get(name, error)
                derivative = error - prev_error
                
                # Update state
                self.pid_integral[name] = integral
                self.pid_prev_error[name] = error
                
                # PID Output (Adjustment to multiplier)
                # We want negative feedback: error > 0 (norm too big) -> reduce LR
                control_signal = self.pid_kp * error + self.pid_ki * integral + self.pid_kd * derivative
                
                # Apply control to multiplier (inverse relationship)
                # If control_signal is positive (norm > target), we want to reduce multiplier
                # new_mult = current_mult * (1.0 - control_signal * scaling)
                # Heuristic mapping for stability
                adjustment = 1.0 - (control_signal * 0.1) 
                new_mult = current_mult * adjustment
                
                reason = "pid_adjust"
                
            else:
                # Default / Legacy Heuristic
                if norm > self.target_norm_high:
                    new_mult = current_mult * self.suppress_factor
                    reason = "overfit_static"
                elif norm < self.target_norm_low:
                    new_mult = current_mult * self.boost_factor
                    reason = "underfit_static"

            # Limit Range for LR
            new_mult = max(self.min_multiplier, min(self.max_multiplier, new_mult))
            
            # V2.1: Secondary Logic - Architecture Drift Check
            # If residual is high but norm is low/normal, it means the subspace is outdated
            # We should increase k_ratio to capture the new direction
            if residual > 0.4 and norm < self.target_norm_high:
                k_mod = 0.05
                if reason == "normal":
                    reason = "drift_high_residual"
                else:
                    reason += "_and_drift"

            if abs(new_mult - current_mult) > 1e-5 or k_mod != 0.0:
                self.multipliers[name] = new_mult
                decisions.append(PilotDecision(
                    layer_name=name,
                    multiplier=new_mult,
                    reason=reason,
                    metric_value=metric_val,
                    k_ratio_mod=k_mod
                ))
                
        return decisions

    def get_multiplier(self, layer_name: str) -> float:
        """获取指定层的当前学习率乘数"""
        return self.multipliers.get(layer_name, 1.0)
    
    def reset(self):
        """重置所有决策和状态"""
        self.multipliers.clear()
        self.ema_history.clear()
        self.pid_integral.clear()
        self.pid_prev_error.clear()
