"""
V 矩阵分级缓存 (V2.6)

根据矩阵大小选择缓存位置：
- 小矩阵 (rank < threshold): 常驻 GPU
- 大矩阵 (rank >= threshold): CPU 交换
"""

import torch
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CacheMode(Enum):
    """缓存模式"""
    GPU = "gpu"          # 全部 GPU
    CPU = "cpu"          # 全部 CPU
    TIERED = "tiered"    # 分级缓存


@dataclass
class CacheConfig:
    """缓存配置"""
    mode: CacheMode = CacheMode.TIERED
    rank_threshold: int = 64          # 分级阈值
    max_gpu_cache_mb: float = 500.0   # GPU 缓存上限 (MB)
    pin_memory: bool = True           # 固定内存 (加速传输)


class VMatrixCache:
    """
    V 矩阵分级缓存
    
    用途:
    - 缓存 Gradient Subspace Projection 的 V 矩阵
    - 根据矩阵大小自动选择存储位置
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # GPU 缓存
        self._gpu_cache: Dict[str, torch.Tensor] = {}
        
        # CPU 缓存 (pinned memory)
        self._cpu_cache: Dict[str, torch.Tensor] = {}
        
        # 统计
        self._gpu_bytes: int = 0
        self._cpu_bytes: int = 0
        self._hit_count: int = 0
        self._miss_count: int = 0
    
    def put(self, layer_name: str, V: torch.Tensor):
        """
        存储 V 矩阵
        
        根据配置和矩阵大小选择存储位置
        """
        rank = V.shape[1] if V.dim() >= 2 else V.shape[0]
        size_bytes = V.numel() * V.element_size()
        
        if self.config.mode == CacheMode.GPU:
            self._put_gpu(layer_name, V)
        
        elif self.config.mode == CacheMode.CPU:
            self._put_cpu(layer_name, V)
        
        elif self.config.mode == CacheMode.TIERED:
            # 分级策略
            if rank < self.config.rank_threshold:
                # 小矩阵放 GPU
                self._put_gpu(layer_name, V)
            else:
                # 大矩阵放 CPU
                self._put_cpu(layer_name, V)
    
    def _put_gpu(self, layer_name: str, V: torch.Tensor):
        """存入 GPU"""
        # 检查是否超出限制
        size_bytes = V.numel() * V.element_size()
        max_bytes = self.config.max_gpu_cache_mb * 1024 * 1024
        
        if self._gpu_bytes + size_bytes > max_bytes:
            # 超出限制，降级到 CPU
            self._put_cpu(layer_name, V)
            return
        
        # 移除旧缓存
        if layer_name in self._gpu_cache:
            old = self._gpu_cache[layer_name]
            self._gpu_bytes -= old.numel() * old.element_size()
        if layer_name in self._cpu_cache:
            old = self._cpu_cache.pop(layer_name)
            self._cpu_bytes -= old.numel() * old.element_size()
        
        # 存入 GPU
        self._gpu_cache[layer_name] = V.detach()
        self._gpu_bytes += size_bytes
    
    def _put_cpu(self, layer_name: str, V: torch.Tensor):
        """存入 CPU (pinned memory)"""
        size_bytes = V.numel() * V.element_size()
        
        # 移除旧缓存
        if layer_name in self._cpu_cache:
            old = self._cpu_cache[layer_name]
            self._cpu_bytes -= old.numel() * old.element_size()
        if layer_name in self._gpu_cache:
            old = self._gpu_cache.pop(layer_name)
            self._gpu_bytes -= old.numel() * old.element_size()
        
        # 转移到 CPU (可选 pinned memory)
        if self.config.pin_memory and V.is_cuda:
            cpu_tensor = V.detach().cpu().pin_memory()
        else:
            cpu_tensor = V.detach().cpu()
        
        self._cpu_cache[layer_name] = cpu_tensor
        self._cpu_bytes += size_bytes
    
    def get(self, layer_name: str, device: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        获取 V 矩阵
        
        Args:
            layer_name: 层名
            device: 目标设备 (None = 返回原始位置)
        
        Returns:
            V 矩阵，如果不存在返回 None
        """
        # GPU 缓存查找
        if layer_name in self._gpu_cache:
            self._hit_count += 1
            V = self._gpu_cache[layer_name]
            if device and V.device != torch.device(device):
                return V.to(device)
            return V
        
        # CPU 缓存查找
        if layer_name in self._cpu_cache:
            self._hit_count += 1
            V = self._cpu_cache[layer_name]
            if device:
                return V.to(device)
            return V
        
        self._miss_count += 1
        return None
    
    def get_to_gpu(self, layer_name: str) -> Optional[torch.Tensor]:
        """获取并传输到 GPU (用于计算)"""
        return self.get(layer_name, device="cuda")
    
    def contains(self, layer_name: str) -> bool:
        """检查是否存在"""
        return layer_name in self._gpu_cache or layer_name in self._cpu_cache
    
    def remove(self, layer_name: str):
        """移除缓存"""
        if layer_name in self._gpu_cache:
            old = self._gpu_cache.pop(layer_name)
            self._gpu_bytes -= old.numel() * old.element_size()
        if layer_name in self._cpu_cache:
            old = self._cpu_cache.pop(layer_name)
            self._cpu_bytes -= old.numel() * old.element_size()
    
    def clear(self):
        """清空所有缓存"""
        self._gpu_cache.clear()
        self._cpu_cache.clear()
        self._gpu_bytes = 0
        self._cpu_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "mode": self.config.mode.value,
            "gpu_layers": len(self._gpu_cache),
            "cpu_layers": len(self._cpu_cache),
            "gpu_mb": self._gpu_bytes / (1024 * 1024),
            "cpu_mb": self._cpu_bytes / (1024 * 1024),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / max(1, self._hit_count + self._miss_count),
        }


# ========== 便捷函数 ==========

_global_cache: Optional[VMatrixCache] = None


def get_v_cache() -> VMatrixCache:
    """获取全局缓存"""
    global _global_cache
    if _global_cache is None:
        _global_cache = VMatrixCache()
    return _global_cache


def set_cache_mode(mode: str, rank_threshold: int = 64):
    """设置缓存模式"""
    global _global_cache
    config = CacheConfig(
        mode=CacheMode(mode),
        rank_threshold=rank_threshold,
    )
    _global_cache = VMatrixCache(config)


def auto_detect_cache_mode() -> str:
    """
    自动检测最佳缓存模式
    
    Returns:
        "gpu" | "cpu" | "tiered"
    """
    if not torch.cuda.is_available():
        return "cpu"
    
    # 获取可用显存
    try:
        # 使用 mem_get_info 获取真实可用显存 (free, total)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024 ** 3)
        
        if free_gb > 16:
            return "gpu"  # 显存充足
        elif free_gb > 8:
            return "tiered"  # 中等显存
        else:
            return "cpu"  # 显存紧张
    except Exception as e:
        logger.warning(f"[VMatrixCache] Failed to detect VRAM: {e}, falling back to tiered mode")
        return "tiered"
