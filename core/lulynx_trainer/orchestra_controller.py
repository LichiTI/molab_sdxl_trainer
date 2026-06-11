"""
Orchestra Controller
MN-LoRA 系统中枢控制器

职责：
1. 协调各子系统的作动频率 (Time-Scale Decoupling)
2. 管理冷却机制 (Cooldown System) 以防止系统共振
3. 作为单一事实来源 (Single Source of Truth) 管理全局状态

架构设计:
Level 4: TE Removal (一次性)
Level 3: Rank Pruning (低频)
Level 2: Dual LR Adjustment (中频)
Level 1: LISA / Gradient Stats (高频)
"""

import logging
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class CooldownState:
    """冷却状态"""
    active: bool = False
    details: str = ""
    remaining_steps: int = 0

class OrchestraController:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OrchestraController, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # 频率配置 (Time-Scale Decoupling)
        self.intervals = {
            "lisa": 1,           # Level 1: 每步
            "lr_check": 10,      # Level 2: 每 10 步检查 LR
            "rank_check": 100,   # Level 3: 每 100 步检查 Rank
            "te_check": 500,     # Level 4: 每 500 步检查 TE 稳态
        }
        
        # 冷却配置 (Cooldown)
        self.cooldown_durations = {
            "rank_pruned": 100,  # 剪枝后锁定 100 步 (等待权重重新适应)
            "te_removed": 50,    # TE 剔除后锁定 50 步 (等待 Batch Size 适应)
        }
        
        # 全局状态
        self.step_count = 0
        self.is_te_removed = False
        
        # 冷却状态
        self.lr_lock = CooldownState()
        self.rank_lock = CooldownState()
        
        logger.info("[Orchestra] Controller initialized. System ready.")

    def reset(self):
        """Reset singleton state between training sessions."""
        self.step_count = 0
        self.is_te_removed = False
        self.lr_lock = CooldownState()
        self.rank_lock = CooldownState()

    def step(self):
        """每步调用，更新状态"""
        self.step_count += 1
        
        # 更新冷却状态
        self._update_cooldown(self.lr_lock)
        self._update_cooldown(self.rank_lock)

    def _update_cooldown(self, state: CooldownState):
        if state.active:
            state.remaining_steps -= 1
            if state.remaining_steps <= 0:
                logger.info(f"[Orchestra] Cooldown released: {state.details}")
                state.active = False
                state.remaining_steps = 0
                state.details = ""

    # ========== 权限查询 (Should I run?) ==========

    def should_run_lisa(self) -> bool:
        """Level 1: LISA (Always run unless specifically blocked)"""
        return True

    def should_run_lr_adjustment(self) -> bool:
        """Level 2: Dual LR Adjustment"""
        # 1. 检查频率
        if self.step_count % self.intervals["lr_check"] != 0:
            return False
            
        # 2. 检查 LR 锁 (High Priority 动作会锁定 LR)
        if self.lr_lock.active:
            # logger.debug(f"[Orchestra] LR adjustment locked by: {self.lr_lock.details}")
            return False
            
        return True

    def should_run_rank_pruning(self) -> bool:
        """Level 3: Rank Pruning"""
        # 1. 检查频率
        if self.step_count % self.intervals["rank_check"] != 0:
            return False
            
        # 2. 检查 Rank 锁 (如果有更高优先级事件)
        if self.rank_lock.active:
            return False
            
        return True

    def should_run_te_check(self) -> bool:
        """Level 4: TE Removal Check"""
        if self.is_te_removed:
            return False
            
        if self.step_count % self.intervals["te_check"] != 0:
            return False
            
        return True

    # ========== 事件通知 (I just did something!) ==========

    def notify_rank_pruned(self, layer_count: int):
        """通知：刚刚执行了 Rank 剪枝"""
        logger.info(f"[Orchestra] Event: {layer_count} layers pruned. Triggering system cooldown.")
        
        # 触发冷却：锁定 LR 调整
        self._trigger_cooldown(
            self.lr_lock, 
            duration=self.cooldown_durations["rank_pruned"],
            reason="Post-Pruning Adaptation"
        )

    def notify_te_removed(self):
        """通知：刚刚剔除了 TE"""
        logger.info(f"[Orchestra] Event: TE removed. Triggering system cooldown.")
        self.is_te_removed = True
        
        # 触发冷却
        self._trigger_cooldown(
            self.lr_lock,
            duration=self.cooldown_durations["te_removed"],
            reason="Post-TE-Removal Adaptation"
        )

    def _trigger_cooldown(self, state: CooldownState, duration: int, reason: str):
        state.active = True
        state.remaining_steps = duration
        state.details = reason
        logger.info(f"[Orchestra] 🔒 Locking system for {duration} steps ({reason})")

# 全局单例访问
def get_orchestra() -> OrchestraController:
    return OrchestraController()
