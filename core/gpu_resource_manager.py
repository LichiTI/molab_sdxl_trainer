"""
REapp GPU Resource Manager | [v1.5.0]
Mutex-based hardware scheduler for VRAM access orchestration.
"""

import threading
from typing import Optional
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class GPULockInfo:
    """硬件锁定遥测载荷"""
    holder_id: str
    acquired_at: datetime
    description: str = ""

class GPUResourceManager:
    """
    单点 GPU 互斥锁管理器。
    防止并发训练与审计任务导致的 CUDA OOM 指数级风险。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global GPU lock."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._gpu_lock = threading.RLock()
        self._current_holder: Optional[GPULockInfo] = None
        self._initialized = True
    
    @property
    def current_holder(self) -> Optional[str]:
        return self._current_holder.holder_id if self._current_holder else None
    
    @property
    def is_locked(self) -> bool:
        return self._current_holder is not None
    
    def get_lock_info(self) -> Optional[GPULockInfo]:
        return self._current_holder
    
    def try_acquire_gpu(self, holder_id: str, description: str = "") -> bool:
        """
        试探性获取锁 (Non-blocking)
        
        State Transition:
            S_{Idle} -> S_{Busy} (if try_lock=T)
        """
        acquired = self._gpu_lock.acquire(blocking=False)
        if acquired:
            self._current_holder = GPULockInfo(
                holder_id=holder_id,
                acquired_at=datetime.now(),
                description=description
            )
            # 控制台反馈流
            logger.info(f"[GPU] 🔒 GPU 已锁定: {holder_id} ({description})")
        return acquired
    
    def release_gpu(self, holder_id: str) -> bool:
        """释放锁资源"""
        if self._current_holder and self._current_holder.holder_id == holder_id:
            logger.info(f"[GPU] 🔓 GPU 已释放: {holder_id}")
            self._current_holder = None
            self._gpu_lock.release()
            return True
        return False
    
    @contextmanager
    def acquire_gpu(self, holder_id: str, description: str = "", timeout: float = None):
        """
        阻塞式锁原语 (Mutex Primitive)
        
        Locking Logic:
            acquire(RLock, timeout) -> yield Holder
        """
        acquired = self._gpu_lock.acquire(blocking=True, timeout=timeout if timeout else -1)
        if not acquired:
            current = self._current_holder
            raise TimeoutError(
                f"无法获取 GPU 资源，当前被 {current.holder_id if current else '未知'} 占用"
            )
        
        self._current_holder = GPULockInfo(
            holder_id=holder_id,
            acquired_at=datetime.now(),
            description=description
        )
        logger.info(f"[GPU] 🔒 GPU 已锁定: {holder_id} ({description})")
        
        try:
            yield self._current_holder
        finally:
            # 安全释放：无论状态如何，只要我们持有锁，就尝试清理
            # 只有当 current_holder 依然是我们时，才执行应用层释放逻辑
            if self._current_holder and self._current_holder.holder_id == holder_id:
                logger.info(f"[GPU] 🔓 GPU 已释放: {holder_id}")
                self._current_holder = None
            
            # 底层锁必须释放
            try:
                self._gpu_lock.release()
            except RuntimeError:
                # 可能是锁已经被释放（防御性编程）
                pass
    
    def get_status_message(self) -> str:
        """返回状态遥测摘要"""
        if not self._current_holder:
            return "GPU 空闲"
        
        elapsed = (datetime.now() - self._current_holder.acquired_at).total_seconds()
        return (
            f"GPU 正在被 {self._current_holder.holder_id} 使用 "
            f"({self._current_holder.description}, 已运行 {int(elapsed)}秒)"
        )

gpu_resource_manager = GPUResourceManager()
