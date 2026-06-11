"""
TG-WD: Trace-Guided Weight Decay (V2.6 Optimized)

Uses a Trajectory Curvature Proxy to dynamically adjust Weight Decay.
This is NOT exact Hessian trace (which requires expensive Hutchinson sampling),
but a lightweight approximation suitable for consumer GPUs (RTX 3060-4090).

Principle:
- Curvature Proxy = 1.0 - CosineSimilarity(g_t, g_{t-1})
- High curvature (gradient oscillation) -> stronger decay -> seek flat minima
- Low curvature (stable gradients) -> normal decay

V2.6 Optimizations:
- No .item() sync locks, keep Tensor on GPU
- Batch update support for reduced CPU-GPU sync
"""

import torch
import numpy as np
from typing import Dict, Optional, List


class TraceGuidedWeightDecay:
    """
    TG-WD: Trace-Guided Weight Decay (V2.6 Optimized)
    """
    
    def __init__(
        self,
        base_lambda: float = 0.01,
        alpha: float = 1.0,         # 调节因子: lambda = base * (1 + alpha * trace)
        probe_interval: int = 50,
        n_probes: int = 1,
        finite_diff_eps: float = 1e-3,
        debug_mode: bool = False,   # V2.6: 调试模式才同步取值
    ):
        self.base_lambda = base_lambda
        self.alpha = alpha
        self.probe_interval = probe_interval
        self.n_probes = n_probes
        self.finite_diff_eps = finite_diff_eps
        self.debug_mode = debug_mode
        
        # V2.6: 使用 Tensor 缓存代替 float 缓存
        self._trace_tensor_cache: Dict[str, torch.Tensor] = {}
        self.trace_cache: Dict[str, float] = {}  # 保留兼容性
        
    def should_update(self, step: int) -> bool:
        return step > 0 and step % self.probe_interval == 0
    
    def update_trace_for_layer(
        self,
        layer_name: str,
        grad: torch.Tensor,
        prev_grad: Optional[torch.Tensor]
    ):
        """
        使用 Trajectory Curvature Approximation (TCA) 估计曲率
        
        V2.6: 保持 Tensor 计算，避免 .item() 同步锁
        
        原理:
        Trace_Proxy = 1.0 - CosineSimilarity(g_t, g_{t-1})
        - If grad aligns (1.0) -> Trace ~ 0 (Flat)
        - If grad flips (-1.0) -> Trace ~ 2 (Sharp Oscillation)
        """
        if prev_grad is None:
            return

        # Flatten
        g_t = grad.view(-1)
        g_prev = prev_grad.view(-1)
        
        # Cosine Similarity (保持在 GPU 上)
        norm_t = g_t.norm() + 1e-8
        norm_prev = g_prev.norm() + 1e-8
        
        cosine = torch.dot(g_t, g_prev) / (norm_t * norm_prev)
        
        # V2.6: 保持为 Tensor，不调用 .item()
        raw_curvature = 1.0 - cosine
        
        # Normalize: [0, 2] -> [0, 1]
        normalized = torch.clamp(raw_curvature / 2.0, 0.0, 1.0)
        
        # 存储 Tensor 版本
        self._trace_tensor_cache[layer_name] = normalized
        
        # 仅调试模式同步 (用于 logging)
        if self.debug_mode:
            self.trace_cache[layer_name] = normalized.item()
    
    def update_traces_batch(
        self,
        layer_grads: Dict[str, torch.Tensor],
        prev_grads: Dict[str, torch.Tensor]
    ):
        """
        V2.6: 批量更新所有层的 trace (更高效)
        
        一次性处理所有层，最后统一同步
        """
        for layer_name, grad in layer_grads.items():
            prev_grad = prev_grads.get(layer_name)
            if prev_grad is not None:
                self.update_trace_for_layer(layer_name, grad, prev_grad)
        
        # 批量同步 (仅在需要 logging 时)
        if self.debug_mode:
            self._sync_cache()
    
    def _sync_cache(self):
        """同步 Tensor 缓存到 float 缓存 (用于 logging)"""
        for name, tensor in self._trace_tensor_cache.items():
            self.trace_cache[name] = tensor.item()
    
    def get_decay_factor(self, layer_name: str, device: str = None) -> float:
        """获取 weight decay 因子
        
        Args:
            layer_name: 层名称
            device: 目标设备 (V3.1 Fix: 避免硬编码 'cuda')
        """
        # 优先从 Tensor 缓存获取
        if layer_name in self._trace_tensor_cache:
            # V2.6 Optimization: 返回 Tensor 标量，避免 CPU 同步 (.item())
            # 这样可以在 GPU 上直接进行后续计算
            trace_val = self._trace_tensor_cache[layer_name]
        else:
            # V3.1 Fix: Use provided device or detect available device
            # instead of hardcoding 'cuda' which crashes on CPU-only setups
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            trace_val = torch.tensor(self.trace_cache.get(layer_name, 0.0), device=device)
        
        return self.base_lambda * (1.0 + self.alpha * trace_val)
    
    def get_decay_factors_batch(self, layer_names: List[str], device: Optional[str] = None) -> Dict[str, float]:
        """
        V2.6: 批量获取 decay factors (减少同步次数)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 收集所有需要的 tensors
        result = {}
        for name in layer_names:
            if name in self._trace_tensor_cache:
                trace_val = self._trace_tensor_cache[name]
                result[name] = self.base_lambda * (1.0 + self.alpha * trace_val)
            else:
                result[name] = torch.tensor(self.base_lambda, device=device)

        return result
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息 (用于 logging)"""
        if not self._trace_tensor_cache:
            return {"mean_trace": 0.0, "max_trace": 0.0, "min_trace": 0.0}
        
        tensors = list(self._trace_tensor_cache.values())
        stacked = torch.stack(tensors)
        
        return {
            "mean_trace": stacked.mean().item(),
            "max_trace": stacked.max().item(),
            "min_trace": stacked.min().item(),
        }
