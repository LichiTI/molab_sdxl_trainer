"""
动态资源管理器

OOM 保护和自适应 Batch Size
"""

import torch
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class ResourceConfig:
    """资源配置"""
    # VRAM 阈值
    vram_warning_threshold: float = 0.85  # 85% 警告
    vram_critical_threshold: float = 0.92  # 92% 临界
    vram_emergency_threshold: float = 0.97  # 97% 紧急

    # 自适应
    enable_adaptive_batch: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 8

    # 梯度累积
    enable_adaptive_accumulation: bool = True
    min_accumulation: int = 1
    max_accumulation: int = 16

    # 缓存清理
    cache_clear_interval: int = 100  # 每 N 步清理

    # 自适应梯度检查点
    enable_adaptive_checkpointing: bool = False
    # VRAM ratio above which to enable gradient checkpointing
    checkpointing_vram_threshold: float = 0.90

    # 自适应 CPU offload
    enable_adaptive_cpu_offload: bool = False
    # VRAM ratio above which to enable CPU offload for text encoders
    cpu_offload_vram_threshold: float = 0.95

    # Peak VRAM tracking
    enable_peak_tracking: bool = True
    peak_tracking_window: int = 50  # track peak over last N steps


class DynamicResourceManager:
    """
    动态资源管理器
    
    功能:
    - VRAM 监控
    - OOM 保护
    - 自适应 Batch Size
    - 自适应梯度累积
    """
    
    def __init__(self, config: Optional[ResourceConfig] = None):
        self.config = config or ResourceConfig()

        # 状态
        self.current_batch_size: int = 1
        self.current_accumulation: int = 4
        self._step_count: int = 0
        self._oom_count: int = 0
        self._last_vram_usage: float = 0.0

        # Peak VRAM tracking
        self._peak_vram_history: List[float] = []
        self._peak_vram_gb: float = 0.0
        self._checkpointing_enabled_by_rm: bool = False
        self._cpu_offload_enabled_by_rm: bool = False

        # 回调
        self.on_batch_change: Optional[Callable[[int, int], None]] = None
        self.on_oom_warning: Optional[Callable[[float], None]] = None
        self.on_checkpointing_change: Optional[Callable[[bool], None]] = None
        self.on_cpu_offload_change: Optional[Callable[[bool], None]] = None
    
    def get_vram_usage(self) -> Dict[str, float]:
        """获取 VRAM 使用情况"""
        if not torch.cuda.is_available():
            return {"used_gb": 0, "total_gb": 0, "usage_ratio": 0}
        
        try:
            used = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
            return {
                "used_gb": round(used, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "usage_ratio": round(reserved / total, 3) if total > 0 else 0,
            }
        except Exception:
            return {"used_gb": 0, "total_gb": 0, "usage_ratio": 0}
    
    def check_vram(self) -> str:
        """
        检查 VRAM 状态
        
        Returns:
            "ok" | "warning" | "critical" | "emergency"
        """
        vram = self.get_vram_usage()
        ratio = vram["usage_ratio"]
        self._last_vram_usage = ratio
        
        if ratio >= self.config.vram_emergency_threshold:
            return "emergency"
        elif ratio >= self.config.vram_critical_threshold:
            return "critical"
        elif ratio >= self.config.vram_warning_threshold:
            return "warning"
        return "ok"
    
    def step(self) -> Dict[str, Any]:
        """
        每步调用，检查资源并调整

        Returns:
            {
                "vram_status": str,
                "batch_adjusted": bool,
                "new_batch_size": int,
                "new_accumulation": int,
                "checkpointing_enabled": bool,
                "cpu_offload_enabled": bool,
            }
        """
        self._step_count += 1
        result = {
            "vram_status": "ok",
            "batch_adjusted": False,
            "new_batch_size": self.current_batch_size,
            "new_accumulation": self.current_accumulation,
            "checkpointing_enabled": self._checkpointing_enabled_by_rm,
            "cpu_offload_enabled": self._cpu_offload_enabled_by_rm,
        }

        # 定期清理缓存
        if self._step_count % self.config.cache_clear_interval == 0:
            self._clear_cache()

        # Track peak VRAM
        if self.config.enable_peak_tracking and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self._peak_vram_history.append(peak)
            if len(self._peak_vram_history) > self.config.peak_tracking_window:
                self._peak_vram_history.pop(0)
            self._peak_vram_gb = max(self._peak_vram_history) if self._peak_vram_history else peak
            torch.cuda.reset_peak_memory_stats()

        # 检查 VRAM
        status = self.check_vram()
        result["vram_status"] = status

        # Adaptive gradient checkpointing
        if self.config.enable_adaptive_checkpointing:
            should_checkpoint = self._last_vram_usage >= self.config.checkpointing_vram_threshold
            if should_checkpoint and not self._checkpointing_enabled_by_rm:
                self._checkpointing_enabled_by_rm = True
                logger.info("[ResourceManager] Enabling gradient checkpointing (VRAM at %.0f%%)",
                            self._last_vram_usage * 100)
                if self.on_checkpointing_change:
                    self.on_checkpointing_change(True)
                result["checkpointing_enabled"] = True
            elif not should_checkpoint and self._checkpointing_enabled_by_rm and self._step_count > 200:
                # Only disable after sustained low usage
                recent = self._peak_vram_history[-20:] if len(self._peak_vram_history) >= 20 else []
                if recent and max(recent) / self._get_total_vram_gb() < self.config.checkpointing_vram_threshold * 0.85:
                    self._checkpointing_enabled_by_rm = False
                    logger.info("[ResourceManager] Disabling gradient checkpointing (VRAM stabilized)")
                    if self.on_checkpointing_change:
                        self.on_checkpointing_change(False)
                    result["checkpointing_enabled"] = False

        # Adaptive CPU offload for text encoders
        if self.config.enable_adaptive_cpu_offload:
            should_offload = self._last_vram_usage >= self.config.cpu_offload_vram_threshold
            if should_offload and not self._cpu_offload_enabled_by_rm:
                self._cpu_offload_enabled_by_rm = True
                logger.info("[ResourceManager] Enabling CPU offload for text encoders (VRAM at %.0f%%)",
                            self._last_vram_usage * 100)
                if self.on_cpu_offload_change:
                    self.on_cpu_offload_change(True)
                result["cpu_offload_enabled"] = True

        if status == "emergency":
            # 紧急：立即减小 batch
            if self.config.enable_adaptive_batch:
                self._reduce_batch()
            elif self.config.enable_adaptive_accumulation:
                self._increase_accumulation()
            result["batch_adjusted"] = True
            result["new_batch_size"] = self.current_batch_size
            result["new_accumulation"] = self.current_accumulation

            if self.on_oom_warning:
                self.on_oom_warning(self._last_vram_usage)

        elif status == "critical":
            # 临界：增加梯度累积
            if self.config.enable_adaptive_accumulation:
                self._increase_accumulation()
                result["batch_adjusted"] = True
                result["new_accumulation"] = self.current_accumulation

        elif status == "ok" and self._step_count > 100:
            # 稳定后尝试增加效率
            if self._last_vram_usage < 0.7:
                self._try_increase_batch()
                result["batch_adjusted"] = True
                result["new_batch_size"] = self.current_batch_size

        return result

    def _get_total_vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return 0.0

    def get_peak_vram_gb(self) -> float:
        """Return the peak VRAM usage in GB over the tracking window."""
        return self._peak_vram_gb
    
    def _reduce_batch(self):
        """减小 batch size"""
        if self.current_batch_size > self.config.min_batch_size:
            old = self.current_batch_size
            self.current_batch_size = max(
                self.config.min_batch_size,
                self.current_batch_size // 2
            )
            logger.warning(f"[OOM Protection] Batch size: {old} -> {self.current_batch_size}")
            
            # 同时增加梯度累积保持总 batch 不变
            if self.config.enable_adaptive_accumulation:
                self._increase_accumulation()
            
            if self.on_batch_change:
                self.on_batch_change(self.current_batch_size, self.current_accumulation)
    
    def _increase_accumulation(self):
        """增加梯度累积"""
        if self.current_accumulation < self.config.max_accumulation:
            old = self.current_accumulation
            self.current_accumulation = min(
                self.config.max_accumulation,
                self.current_accumulation * 2
            )
            logger.info(f"[Adaptive] Gradient accumulation: {old} -> {self.current_accumulation}")
    
    def _try_increase_batch(self):
        """尝试增加 batch size"""
        if not self.config.enable_adaptive_batch:
            return
        
        if self.current_batch_size < self.config.max_batch_size:
            # 只在显存充足时增加
            if self._last_vram_usage < 0.65:
                old = self.current_batch_size
                self.current_batch_size = min(
                    self.config.max_batch_size,
                    self.current_batch_size + 1
                )
                logger.info(f"[Adaptive] Batch size: {old} -> {self.current_batch_size}")
                
                if self.on_batch_change:
                    self.on_batch_change(self.current_batch_size, self.current_accumulation)
    
    def _clear_cache(self):
        """清理 CUDA 缓存"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to empty CUDA cache: {e}")
    
    def handle_oom(self) -> bool:
        """
        处理 OOM 错误
        
        Returns:
            是否可以恢复
        """
        self._oom_count += 1
        
        # 清理缓存
        self._clear_cache()
        
        # 减小 batch
        if self.current_batch_size > self.config.min_batch_size:
            self._reduce_batch()
            return True
        
        # 已经是最小 batch，增加梯度累积
        if self.current_accumulation < self.config.max_accumulation:
            self._increase_accumulation()
            return True
        
        # 无法恢复
        logger.error("[OOM] Cannot recover: already at minimum batch and maximum accumulation")
        return False
    
    def get_optimal_settings(self, dataset_size: int, target_steps: int) -> Dict[str, int]:
        """
        计算最优 batch/accumulation 设置
        
        Args:
            dataset_size: 数据集大小
            target_steps: 目标训练步数
        """
        # 计算每步实际处理的样本数
        effective_batch = self.current_batch_size * self.current_accumulation
        
        # 计算需要的 epochs
        steps_per_epoch = dataset_size // effective_batch
        needed_epochs = (target_steps + steps_per_epoch - 1) // steps_per_epoch
        
        return {
            "batch_size": self.current_batch_size,
            "accumulation": self.current_accumulation,
            "effective_batch": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "needed_epochs": needed_epochs,
        }


# ========== OOM 安全执行 ==========

def oom_safe_execute(func, *args, max_retries: int = 3, **kwargs):
    """
    OOM 安全执行
    
    遇到 OOM 时自动清理并重试
    """
    manager = DynamicResourceManager()
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"[OOM] Attempt {attempt + 1}/{max_retries}: {e}")
            
            if not manager.handle_oom():
                raise
            
            # 等待一会儿让 GPU 稳定
            time.sleep(1.0)
    
    raise RuntimeError("Failed after max retries due to OOM")
