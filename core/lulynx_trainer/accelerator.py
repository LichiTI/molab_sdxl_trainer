"""
多 GPU 加速器

使用 Hugging Face Accelerate 实现数据并行
"""

import torch
import logging
from typing import Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AcceleratorConfig:
    """加速器配置"""
    mixed_precision: str = "bf16"  # no, fp16, bf16
    gradient_accumulation_steps: int = 4
    split_batches: bool = False
    log_with: Optional[str] = None  # tensorboard, wandb


class MultiGPUAccelerator:
    """
    多 GPU 加速器
    
    封装 Hugging Face Accelerate 库的核心功能
    
    用法:
        accelerator = MultiGPUAccelerator()
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        
        for batch in dataloader:
            loss = model(batch)
            accelerator.backward(loss)
            optimizer.step()
    """
    
    def __init__(self, config: Optional[AcceleratorConfig] = None):
        self.config = config or AcceleratorConfig()
        self._accelerator = None
        self._is_initialized = False
        
    def _lazy_init(self):
        """延迟初始化 Accelerator"""
        if self._is_initialized:
            return
        
        try:
            from accelerate import Accelerator
            
            self._accelerator = Accelerator(
                mixed_precision=self.config.mixed_precision,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                split_batches=self.config.split_batches,
                log_with=self.config.log_with,
            )
            
            self._is_initialized = True
            logger.info(f"Accelerator initialized: {self._accelerator.device}")
            logger.info(f"  Num processes: {self._accelerator.num_processes}")
            logger.info(f"  Mixed precision: {self.config.mixed_precision}")
            
        except ImportError:
            logger.warning("accelerate not installed, multi-GPU disabled")
            self._accelerator = None
            self._is_initialized = True
    
    @property
    def accelerator(self):
        """获取 Accelerator 实例"""
        self._lazy_init()
        return self._accelerator
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        self._lazy_init()
        return self._accelerator is not None
    
    @property
    def device(self) -> str:
        """获取当前设备"""
        if self.is_available:
            return self._accelerator.device
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def is_main_process(self) -> bool:
        """是否为主进程"""
        if self.is_available:
            return self._accelerator.is_main_process
        return True
    
    @property
    def num_processes(self) -> int:
        """进程数量"""
        if self.is_available:
            return self._accelerator.num_processes
        return 1
    
    def prepare(self, *args):
        """
        准备模型、优化器、数据加载器
        
        Args:
            *args: 模型、优化器、数据加载器等
            
        Returns:
            准备好的对象元组
        """
        if self.is_available:
            return self._accelerator.prepare(*args)
        return args
    
    def backward(self, loss: torch.Tensor):
        """反向传播"""
        if self.is_available:
            self._accelerator.backward(loss)
        else:
            loss.backward()
    
    def clip_grad_norm(self, params, max_norm: float):
        """梯度裁剪"""
        if self.is_available:
            self._accelerator.clip_grad_norm_(params, max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(params, max_norm)
    
    def wait_for_everyone(self):
        """等待所有进程"""
        if self.is_available:
            self._accelerator.wait_for_everyone()
    
    def save(self, obj, path: str):
        """保存对象 (仅主进程)"""
        if self.is_main_process:
            torch.save(obj, path)
    
    def print(self, *args, **kwargs):
        """打印 (仅主进程)"""
        if self.is_main_process:
            print(*args, **kwargs)
    
    def log(self, values: dict, step: Optional[int] = None):
        """记录日志"""
        if self.is_available and self.config.log_with:
            self._accelerator.log(values, step=step)
    
    def end_training(self):
        """结束训练"""
        if self.is_available:
            self._accelerator.end_training()
    
    def unwrap_model(self, model):
        """解包模型 (获取原始模型)"""
        if self.is_available:
            return self._accelerator.unwrap_model(model)
        return model
    
    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """收集所有进程的张量"""
        if self.is_available:
            return self._accelerator.gather(tensor)
        return tensor


# ========== 便捷函数 ==========

_global_accelerator: Optional[MultiGPUAccelerator] = None


def get_accelerator() -> MultiGPUAccelerator:
    """获取全局加速器"""
    global _global_accelerator
    if _global_accelerator is None:
        _global_accelerator = MultiGPUAccelerator()
    return _global_accelerator


def init_accelerator(
    mixed_precision: str = "bf16",
    gradient_accumulation_steps: int = 4,
) -> MultiGPUAccelerator:
    """初始化全局加速器"""
    global _global_accelerator
    
    config = AcceleratorConfig(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    _global_accelerator = MultiGPUAccelerator(config)
    return _global_accelerator


def is_main_process() -> bool:
    """是否为主进程"""
    return get_accelerator().is_main_process


def print_main(*args, **kwargs):
    """仅主进程打印"""
    if is_main_process():
        print(*args, **kwargs)
