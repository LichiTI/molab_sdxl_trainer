import torch
from typing import Dict, Optional
from collections import deque
from .types import AuditMode
from ..constants import DEFAULT_VRAM_MARGIN_MB, VRAM_THRESHOLD_STOP, VRAM_THRESHOLD_LITE

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class HardwareWatchdog:
    """
    硬件监控器 - V10.0 动态阈值算法
    
    改进:
    - 动态学习显存波动底噪 (前 50 步)
    - 自适应调整熔断阈值
    """
    
    def __init__(self, vram_total_mb: float = 0, learning_steps: int = 50, mode_override: str = ""):
        self.vram_total_mb = vram_total_mb
        self._last_mode = AuditMode.PRO
        self._mode_override = mode_override.upper().strip()

        # V10.0: 动态阈值学习
        self._learning_steps = learning_steps
        self._vram_history: deque = deque(maxlen=learning_steps)
        self._noise_floor: Optional[float] = None  # 学习到的底噪 (MB)
        self._step_count = 0
        self._lock = torch.multiprocessing.Lock() if HAS_TORCH else None
        import threading
        self._thread_lock = threading.Lock()
        
    def get_vram_info(self) -> Dict[str, float]:
        """获取显存信息"""
        if not HAS_TORCH or not torch.cuda.is_available():
            return {"used": 0, "total": 0, "free": 0}
        
        try:
            used = torch.cuda.memory_allocated() / 1024 / 1024
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            free = total - used
            return {"used": used, "total": total, "free": free}
        except Exception:
            return {"used": 0, "total": 0, "free": 0}
    
    def _learn_noise_floor(self, used_mb: float):
        """V10.0: 学习显存波动底噪"""
        with self._thread_lock:
             self._vram_history.append(used_mb)
             self._step_count += 1
             
             if self._step_count >= self._learning_steps and len(self._vram_history) > 1:
                 # 计算标准差作为底噪估计 (Sample Variance)
                 # Direct calculation to avoid list copy overhead (Issue 275)
                 vram_list = list(self._vram_history) 
                 n = len(vram_list)
                 mean_vram = sum(vram_list) / n
                 variance = sum((x - mean_vram) ** 2 for x in vram_list) / (n - 1)
                 self._noise_floor = (variance ** 0.5) * 2  # 2σ 作为安全边界
    
    def check_policy(self, is_sampling: bool = False) -> AuditMode:
        """
        V10.0 动态阈值决策
        
        数学定义:
        Threshold_STOP = min(0.98 × V_total, V_total - max(200MB, noise_floor))
        """
        if is_sampling:
            return AuditMode.SUSPEND

        if self._mode_override in ("PRO", "LITE", "STOP"):
            return AuditMode[self._mode_override]
        
        vram = self.get_vram_info()
        if vram["total"] == 0:
            return AuditMode.PRO
        
        used = vram["used"]
        total = vram["total"]
        ratio = used / total
        
        # V10.0: 学习底噪
        self._learn_noise_floor(used)
        
        # V10.0: 动态阈值 (使用学习到的底噪或默认值)
        safety_margin = max(DEFAULT_VRAM_MARGIN_MB, self._noise_floor or DEFAULT_VRAM_MARGIN_MB)
        threshold_stop = min(VRAM_THRESHOLD_STOP * total, total - safety_margin)
        threshold_lite = VRAM_THRESHOLD_LITE
        
        if used > threshold_stop:
            return AuditMode.STOP
        elif ratio > threshold_lite:
            return AuditMode.LITE
        else:
            return AuditMode.PRO
