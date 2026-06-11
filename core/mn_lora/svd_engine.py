"""
SVD 引擎 (V2.6)

提供多种 SVD 计算策略：
- Full: 完整 SVD (精确但慢)
- Incremental: 增量更新 (快速，略有误差)
- Randomized: 随机化 SVD (最快，有损)
"""

import torch
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("SVDEngine")


class SVDMode(Enum):
    """SVD 模式"""
    FULL = "full"
    INCREMENTAL = "incremental"
    RANDOMIZED = "randomized"


@dataclass
class SVDResult:
    """SVD 结果"""
    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor
    rank: int
    is_approximate: bool = False


class SVDEngine(ABC):
    """SVD 引擎抽象基类"""
    
    @abstractmethod
    def compute(self, weight: torch.Tensor, k: Optional[int] = None) -> SVDResult:
        """计算 SVD"""
        pass
    
    @abstractmethod
    def update(self, layer_name: str, weight: torch.Tensor) -> SVDResult:
        """更新 SVD (可能使用缓存)"""
        pass


class FullSVDEngine(SVDEngine):
    """
    完整 SVD 引擎
    
    精确计算，无缓存
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._cache: Dict[str, SVDResult] = {}
    
    def compute(self, weight: torch.Tensor, k: Optional[int] = None) -> SVDResult:
        """完整 SVD 分解"""
        if not self.use_gpu:
            weight = weight.cpu()
            
        # V2.6 Fix: Handle 4D Tensors (Conv2d)
        if weight.dim() > 2:
            weight = weight.view(weight.size(0), -1)
        
        # V3.1 Fix: FP16 Safety - PyTorch LAPACK backend does not support FP16 SVD
        orig_dtype = weight.dtype
        if weight.dtype != torch.float32 and weight.dtype != torch.float64:
            weight = weight.float()
        
        try:
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        except RuntimeError as e: # Torch usually raises RuntimeError for lapack
            logger.error(f"SVD computation failed: {e}")
            # Return dummy result to prevent crash
            device = weight.device
            return SVDResult(
                U=torch.eye(weight.shape[0], device=device),
                S=torch.zeros(min(weight.shape), device=device),
                Vh=torch.eye(weight.shape[1], device=device),
                rank=0,
                is_approximate=True
            )
        
        if k is not None and k < S.shape[0]:
            U = U[:, :k]
            S = S[:k]
            Vh = Vh[:k, :]
        
        # V3.1 Fix: Cast back to original dtype
        if U.dtype != orig_dtype:
            U = U.to(orig_dtype)
            S = S.to(orig_dtype)
            Vh = Vh.to(orig_dtype)
        
        return SVDResult(
            U=U,
            S=S,
            Vh=Vh,
            rank=S.shape[0],
            is_approximate=False,
        )
    
    def update(self, layer_name: str, weight: torch.Tensor) -> SVDResult:
        """每次都重新计算"""
        result = self.compute(weight)
        self._cache[layer_name] = result
        return result

    def clear_cache(self):
        self._cache.clear()


class IncrementalSVDEngine(SVDEngine):
    """
    增量 SVD 引擎
    
    利用权重变化小的特性，增量更新
    """
    
    def __init__(
        self,
        change_threshold: float = 0.05,    # 变化 <5% 跳过
        error_threshold: float = 0.1,      # 误差 >10% 强制重算
        force_recompute_interval: int = 500,  # 每 500 步强制重算
    ):
        self.change_threshold = change_threshold
        self.error_threshold = error_threshold
        self.force_recompute_interval = force_recompute_interval
        
        self._cache: Dict[str, SVDResult] = {}
        self._weight_cache: Dict[str, torch.Tensor] = {}
        self._update_count: Dict[str, int] = {}
    
    def compute(self, weight: torch.Tensor, k: Optional[int] = None) -> SVDResult:
        """完整 SVD (用于初始化或强制重算)"""
        # V2.6 Fix: Handle 4D Tensors
        if weight.dim() > 2:
            weight = weight.view(weight.size(0), -1)
        
        # V3.1 Fix: FP16 Safety - PyTorch LAPACK backend does not support FP16 SVD
        orig_dtype = weight.dtype
        if weight.dtype != torch.float32 and weight.dtype != torch.float64:
            weight = weight.float()
            
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        if k is not None and k < S.shape[0]:
            U = U[:, :k]
            S = S[:k]
            Vh = Vh[:k, :]
        
        # V3.1 Fix: Cast back to original dtype
        if U.dtype != orig_dtype:
            U = U.to(orig_dtype)
            S = S.to(orig_dtype)
            Vh = Vh.to(orig_dtype)
        
        return SVDResult(U=U, S=S, Vh=Vh, rank=S.shape[0], is_approximate=False)
    
    def update(self, layer_name: str, weight: torch.Tensor) -> SVDResult:
        """增量更新"""
        self._update_count[layer_name] = self._update_count.get(layer_name, 0) + 1
        
        # 首次计算
        if layer_name not in self._cache:
            result = self.compute(weight)
            self._cache[layer_name] = result
            self._weight_cache[layer_name] = weight.detach().clone()
            return result
        
        # 强制重算周期
        if self._update_count[layer_name] % self.force_recompute_interval == 0:
            logger.debug(f"[IncrementalSVD] Force recompute for {layer_name}")
            result = self.compute(weight)
            self._cache[layer_name] = result
            self._weight_cache[layer_name] = weight.detach().clone()
            return result
        
        # 检查变化量
        old_weight = self._weight_cache[layer_name]
        delta = weight - old_weight
        change_ratio = delta.norm() / (old_weight.norm() + 1e-8)
        
        if change_ratio < self.change_threshold:
            # 变化太小，复用缓存
            return self._cache[layer_name]
        
        # 增量更新 (简化版：使用低秩近似)
        # 对于小变化，使用 rank-1 更新
        result = self._incremental_update(layer_name, weight, delta)
        
        return result
    
    def _incremental_update(
        self, 
        layer_name: str, 
        weight: torch.Tensor,
        delta: torch.Tensor,
    ) -> SVDResult:
        """
        增量更新 SVD
        
        使用 Brand's algorithm 的简化版本
        """
        old_result = self._cache[layer_name]
        
        # 简化策略：如果变化较大，直接重算
        # 完整的 Brand's algorithm 实现较复杂
        delta_norm = delta.norm()
        weight_norm = weight.norm()
        
        if delta_norm / weight_norm > 0.1:
            # 变化超过 10%，重新计算
            result = self.compute(weight)
            self._cache[layer_name] = result
            self._weight_cache[layer_name] = weight.detach().clone()
            return result
        
        # 对于小变化，使用一阶近似
        # 更新 S 值（最重要的变化）
        # S_new ≈ S_old + U^T @ ΔW @ V
        U = old_result.U
        Vh = old_result.Vh
        S = old_result.S
        
        # 计算 S 的修正量
        delta_S = torch.diag(U.T @ delta @ Vh.T)
        S_new = S + delta_S
        
        # 确保 S 非负
        S_new = torch.clamp(S_new, min=0)
        
        result = SVDResult(
            U=U,  # U 和 V 的变化通常较小，保持不变
            S=S_new,
            Vh=Vh,
            rank=S.shape[0],
            is_approximate=True,
        )
        
        self._cache[layer_name] = result
        self._weight_cache[layer_name] = weight.detach().clone()
        
        return result


class RandomizedSVDEngine(SVDEngine):
    """
    随机化 SVD 引擎
    
    使用随机投影加速，只计算前 k 个奇异值
    """
    
    def __init__(
        self,
        default_rank: int = 64,
        oversampling: int = 10,
        n_power_iterations: int = 2,
    ):
        self.default_rank = default_rank
        self.oversampling = oversampling
        self.n_power_iterations = n_power_iterations
        self._cache: Dict[str, SVDResult] = {}
    
    def compute(self, weight: torch.Tensor, k: Optional[int] = None) -> SVDResult:
        """随机化 SVD"""
        k = k or self.default_rank
        # V2.6 Fix: Handle 4D Tensors first
        if weight.dim() > 2:
            weight = weight.view(weight.size(0), -1)
            
        k = min(k, min(weight.shape))
        
        # V3.1 Fix: FP16 Safety - QR/SVD operations require FP32
        orig_dtype = weight.dtype
        if weight.dtype != torch.float32 and weight.dtype != torch.float64:
            weight = weight.float()
        
        m, n = weight.shape
        
        # 随机投影矩阵
        P = torch.randn(n, k + self.oversampling, device=weight.device, dtype=weight.dtype)
        
        # Power iteration (提高精度)
        Y = weight @ P
        for _ in range(self.n_power_iterations):
            Y = weight @ (weight.T @ Y)
        
        # QR 分解得到正交基
        Q, _ = torch.linalg.qr(Y)
        
        # 投影到低维空间
        B = Q.T @ weight
        
        # 小矩阵 SVD
        U_small, S, Vh = torch.linalg.svd(B, full_matrices=False)
        
        # 恢复 U
        U = Q @ U_small
        
        # 截断到 k
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
        
        # V3.1 Fix: Cast back to original dtype
        if U.dtype != orig_dtype:
            U = U.to(orig_dtype)
            S = S.to(orig_dtype)
            Vh = Vh.to(orig_dtype)
        
        return SVDResult(
            U=U,
            S=S,
            Vh=Vh,
            rank=k,
            is_approximate=True,
        )
    
    def update(self, layer_name: str, weight: torch.Tensor) -> SVDResult:
        """更新（每次都重新计算，因为已经很快）"""
        result = self.compute(weight)
        self._cache[layer_name] = result
        return result


# ========== 工厂函数 ==========

def create_svd_engine(mode: str = "incremental", **kwargs) -> SVDEngine:
    """
    创建 SVD 引擎
    
    Args:
        mode: "full" | "incremental" | "randomized"
        **kwargs: 引擎特定参数
    """
    if mode == "full":
        return FullSVDEngine(**kwargs)
    elif mode == "incremental":
        return IncrementalSVDEngine(**kwargs)
    elif mode == "randomized":
        return RandomizedSVDEngine(**kwargs)
    else:
        logger.warning(f"Unknown SVD mode: {mode}, using incremental")
        return IncrementalSVDEngine(**kwargs)


# ========== 全局引擎 ==========

_global_engine: Optional[SVDEngine] = None


def get_svd_engine() -> SVDEngine:
    """获取全局 SVD 引擎"""
    global _global_engine
    if _global_engine is None:
        _global_engine = IncrementalSVDEngine()
    return _global_engine


def set_svd_engine(engine: SVDEngine):
    """设置全局 SVD 引擎"""
    global _global_engine
    _global_engine = engine
