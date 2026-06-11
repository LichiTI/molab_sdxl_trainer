"""
REapp Accelerator | [v1.4.2]
Hardware abstraction gateway for unified resource orchestration.
"""

import sys
import logging
import platform
import shutil
from typing import Optional, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass

# Lazy import handling
torch = None
HAS_TORCH = False

def _ensure_torch():
    global torch, HAS_TORCH
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False

from .gpu_resource_manager import gpu_resource_manager
from .log_sanitizer import sanitize_log, add_sensitive_path

logger = logging.getLogger(__name__)

@dataclass
class HardwareInfo:
    """标准化硬件遥测快照"""
    platform: str
    python_version: str
    has_cuda: bool
    cuda_version: Optional[str]
    device_name: Optional[str]
    vram_total_gb: float
    vram_free_gb: float
    ram_total_gb: float
    ram_available_gb: float

class Accelerator:
    """
    硬件感知操作的统一入口，封装了 GPU 锁控制与系统安全特性。
    """
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Accelerator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.gpu_manager = gpu_resource_manager

    @property
    def has_cuda(self) -> bool:
        """检测 CUDA 环境"""
        _ensure_torch()
        return HAS_TORCH and torch.cuda.is_available()

    @property
    def has_gpu(self) -> bool:
        return self.has_cuda

    @property
    def device(self) -> str:
        """返回当前最优计算设备 (cuda/cpu)"""
        _ensure_torch()
        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def warmup(self):
        """显式预热加载 Torch (应在后台线程调用)"""
        _ensure_torch()

    def get_hardware_info(self) -> HardwareInfo:
        """
        采集实时系统统计数据。
        即便在驱动异常或硬件挂起时仍能安全返回结果。
        注意：此方法不再触发 torch 的阻塞式加载。
        """
        import psutil
        
        info = HardwareInfo(
            platform=platform.system(),
            python_version=sys.version.split()[0],
            has_cuda=False,
            cuda_version=None,
            device_name=None,
            vram_total_gb=0,
            vram_free_gb=0,
            ram_total_gb=0,
            ram_available_gb=0
        )

        vm = psutil.virtual_memory()
        info.ram_total_gb = round(vm.total / (1024**3), 1)
        info.ram_available_gb = round(vm.available / (1024**3), 1)

        # Sync state if torch usage is detected externally
        if not HAS_TORCH and 'torch' in sys.modules:
            _ensure_torch()

        # Non-blocking check: Only use torch if already loaded
        if HAS_TORCH and torch is not None and torch.cuda.is_available():
            try:
                info.has_cuda = True
                info.cuda_version = torch.version.cuda
                info.device_name = torch.cuda.get_device_name(0)
                
                props = torch.cuda.get_device_properties(0)
                info.vram_total_gb = round(props.total_memory / (1024**3), 1)
                
                free, total = torch.cuda.mem_get_info()
                info.vram_free_gb = round(free / (1024**3), 1)
            except Exception as e:
                logger.warning(f"Failed to query extended GPU info: {e}")

        return info

    def estimate_vram_requirement(self, resolution: int, batch_size: int, lora_rank: int, gradient_checkpointing: bool = True) -> float:
        """
        V_total = V_base + (V_batch * B * R_f) * gamma_{cg} + (Rank * delta)
        """
        base_gb = 8.0 # SDXL 基础开销
        
        res_factor = (resolution * resolution) / (1024 * 1024)
        batch_gb = 1.5 * batch_size * res_factor
        
        if not gradient_checkpointing:
            batch_gb *= 3.0
            
        return round(base_gb + batch_gb + (lora_rank * 0.01), 1)

    def sanitize(self, text: str) -> str:
        """日志敏感信息脱敏"""
        return sanitize_log(text)

    def register_sensitive_path(self, path: str):
        """路径动态屏蔽注册"""
        add_sensitive_path(path)

    def lock_gpu(self, holder_id: str, description: str = ""):
        """GPU 互斥锁获取"""
        return self.gpu_manager.acquire_gpu(holder_id, description)

    def is_gpu_busy(self) -> bool:
        """探测 GPU 锁定状态"""
        return self.gpu_manager.is_locked

accelerator = Accelerator()
