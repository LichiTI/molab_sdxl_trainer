"""
Auto Controller: 自动训练控制器

扩展 TrainingPilot，添加高级自动化功能:
1. Auto-Freeze TE (自动冻结 Text Encoder)
2. Smart Early Stop (智能早停)
3. Smart LR Decay (智能学习率衰减)
4. Dynamic Batch Size (动态批次大小)
"""

import torch
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ========== 事件类型 ==========

class AutoEvent(Enum):
    """自动控制事件类型"""
    TE_FROZEN = "te_frozen"            # Text Encoder 已冻结
    EARLY_STOP = "early_stop"          # 触发早停
    LR_DECAY = "lr_decay"              # LR 衰减
    BATCH_SIZE_UP = "batch_size_up"    # Batch Size 增加
    CHECKPOINT_SAVE = "checkpoint_save"  # 建议保存 Checkpoint


# ========== 指标追踪器 ==========

@dataclass
class MetricsTracker:
    """
    指标追踪器
    
    追踪训练过程中的关键指标，用于自动控制决策
    """
    # 语义指标
    clip_drift: float = 0.0
    clip_drift_history: List[float] = field(default_factory=list)
    
    # 权重指标
    stable_rank: float = 0.0
    stable_rank_history: List[float] = field(default_factory=list)
    effective_rank: float = 0.0
    
    # 梯度指标
    gradient_rank: float = 0.0
    gradient_rank_history: List[float] = field(default_factory=list)
    gsnr: float = 0.0
    gsnr_history: List[float] = field(default_factory=list)
    
    # 损失指标
    loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    
    # 死神经元
    dead_neuron_rate: float = 0.0
    
    # 历史长度
    history_length: int = 50
    
    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
                # 更新历史
                history_key = f"{key}_history"
                if hasattr(self, history_key):
                    history = getattr(self, history_key)
                    history.append(value)
                    if len(history) > self.history_length:
                        history.pop(0)
    
    def get_trend(self, key: str, window: int = 10) -> Optional[float]:
        """计算趋势 (正=上升, 负=下降)"""
        history_key = f"{key}_history"
        if not hasattr(self, history_key):
            return None
        
        history = getattr(self, history_key)
        if len(history) < window:
            return None
        
        recent = history[-window:]
        if len(recent) < 2:
            return 0.0
        
        # 线性回归斜率
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def is_plateau(self, key: str, window: int = 20, threshold: float = 0.001) -> bool:
        """检测是否进入平台期"""
        trend = self.get_trend(key, window)
        if trend is None:
            return False
        return abs(trend) < threshold


# ========== 自动控制配置 ==========

@dataclass
class AutoControlConfig:
    """自动控制配置"""
    
    # === Auto-Freeze TE ===
    auto_freeze_te: bool = True
    clip_drift_warning: float = 0.03     # 警告线
    clip_drift_danger: float = 0.05      # 危险线 -> 自动冻结
    clip_drift_consecutive: int = 5      # 连续 N 次超过阈值才触发
    
    # === Smart Early Stop ===
    smart_early_stop: bool = True
    stable_rank_collapse_threshold: float = 0.3  # 相对于初始值的比例
    stable_rank_consecutive: int = 10    # 连续 N 次低于阈值
    loss_plateau_window: int = 50        # 检测平台期的窗口
    
    # === Smart LR Decay ===
    smart_lr_decay: bool = True
    lr_decay_factor: float = 0.5         # 衰减因子
    gradient_rank_plateau_window: int = 30  # Gradient Rank 平台期窗口
    max_decays: int = 3                  # 最大衰减次数
    
    # === Dynamic Batch Size ===
    dynamic_batch_size: bool = False     # 默认关闭
    target_gsnr: float = 5.0             # 目标 GSNR
    batch_size_step: int = 1             # 每次调整步长
    
    # === Warmup ===
    warmup_steps: int = 100              # Warmup 期间不触发任何自动控制


# ========== 自动控制器 ==========

class AutoController:
    """
    自动训练控制器
    
    监听训练指标，自动执行控制动作
    """
    
    def __init__(self, config: Optional[AutoControlConfig] = None):
        self.config = config or AutoControlConfig()
        self.metrics = MetricsTracker()
        
        # 状态
        self._step = 0
        self._te_frozen = False
        self._should_stop = False
        self._lr_decay_count = 0
        self._current_batch_size = 1
        self._initial_stable_rank: Optional[float] = None
        
        # 连续计数器
        self._clip_drift_exceed_count = 0
        self._rank_collapse_count = 0
        
        # 事件回调
        self._event_callbacks: List[Callable[[AutoEvent, Dict[str, Any]], None]] = []
        
        # 事件日志
        self._events: List[Tuple[int, AutoEvent, Dict[str, Any]]] = []
    
    def register_callback(self, callback: Callable[[AutoEvent, Dict[str, Any]], None]):
        """注册事件回调"""
        self._event_callbacks.append(callback)
    
    def _emit_event(self, event: AutoEvent, data: Dict[str, Any] = None):
        """发送事件"""
        data = data or {}
        self._events.append((self._step, event, data))
        
        logger.info(f"[AutoController] Event: {event.value} at step {self._step}")
        
        for callback in self._event_callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"[AutoController] Callback error: {e}")
    
    def step(
        self,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        每步调用
        
        Args:
            step: 当前步数
            metrics: 指标字典 {clip_drift, stable_rank, gsnr, loss, ...}
            model: 模型 (用于冻结 TE)
            optimizer: 优化器 (用于调整 LR)
        
        Returns:
            动作结果 {should_stop, te_frozen, lr_decayed, ...}
        """
        self._step = step
        
        # 更新指标
        if metrics:
            self.metrics.update(**metrics)
            
            # 记录初始 Stable Rank
            if self._initial_stable_rank is None and "stable_rank" in metrics:
                self._initial_stable_rank = metrics["stable_rank"]
        
        # Warmup 期间不触发
        if step < self.config.warmup_steps:
            return {"warmup": True}
        
        result = {}
        
        # === Auto-Freeze TE ===
        if self.config.auto_freeze_te and not self._te_frozen:
            freeze_result = self._check_auto_freeze_te(model)
            result.update(freeze_result)
        
        # === Smart Early Stop ===
        if self.config.smart_early_stop and not self._should_stop:
            stop_result = self._check_early_stop()
            result.update(stop_result)
        
        # === Smart LR Decay ===
        if self.config.smart_lr_decay and optimizer:
            decay_result = self._check_lr_decay(optimizer)
            result.update(decay_result)
        
        return result
    
    def _check_auto_freeze_te(self, model: Optional[torch.nn.Module]) -> Dict[str, Any]:
        """检查是否需要冻结 TE"""
        clip_drift = self.metrics.clip_drift
        
        if clip_drift > self.config.clip_drift_danger:
            self._clip_drift_exceed_count += 1
            
            if self._clip_drift_exceed_count >= self.config.clip_drift_consecutive:
                # 触发冻结
                if model is not None:
                    self._freeze_text_encoder(model)
                    self._te_frozen = True
                    self._emit_event(AutoEvent.TE_FROZEN, {
                        "clip_drift": clip_drift,
                        "threshold": self.config.clip_drift_danger,
                    })
                    return {"te_frozen": True, "clip_drift": clip_drift}
        else:
            self._clip_drift_exceed_count = 0
        
        return {"te_warning": clip_drift > self.config.clip_drift_warning}
    
    def _freeze_text_encoder(self, model: torch.nn.Module):
        """冻结 Text Encoder"""
        frozen_count = 0
        
        for name, param in model.named_parameters():
            # 识别 TE 参数 (根据常见命名)
            if any(te_name in name.lower() for te_name in [
                "text_encoder", "te_", "clip_", "encoder.layer"
            ]):
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
        
        logger.info(f"[AutoController] Froze {frozen_count} Text Encoder parameters")
    
    def _check_early_stop(self) -> Dict[str, Any]:
        """检查是否需要早停"""
        stable_rank = self.metrics.stable_rank
        
        # Rank Collapse 检测
        if self._initial_stable_rank and self._initial_stable_rank > 0:
            ratio = stable_rank / self._initial_stable_rank
            
            if ratio < self.config.stable_rank_collapse_threshold:
                self._rank_collapse_count += 1
                
                if self._rank_collapse_count >= self.config.stable_rank_consecutive:
                    self._should_stop = True
                    self._emit_event(AutoEvent.EARLY_STOP, {
                        "reason": "rank_collapse",
                        "stable_rank": stable_rank,
                        "initial_rank": self._initial_stable_rank,
                        "ratio": ratio,
                    })
                    return {"should_stop": True, "reason": "rank_collapse"}
            else:
                self._rank_collapse_count = 0
        
        # Loss 平台期检测
        if self.metrics.is_plateau("loss", self.config.loss_plateau_window):
            # 不直接停止，但发出建议保存的事件
            self._emit_event(AutoEvent.CHECKPOINT_SAVE, {
                "reason": "loss_plateau",
            })
        
        return {"should_stop": False}
    
    def _check_lr_decay(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """检查是否需要 LR 衰减"""
        if self._lr_decay_count >= self.config.max_decays:
            return {}
        
        metric_key = "gradient_rank" if self.metrics.gradient_rank_history else "gsnr"
        if self.metrics.is_plateau(metric_key, self.config.gradient_rank_plateau_window):
            # 衰减 LR
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.config.lr_decay_factor
            
            self._lr_decay_count += 1
            
            self._emit_event(AutoEvent.LR_DECAY, {
                "decay_factor": self.config.lr_decay_factor,
                "decay_count": self._lr_decay_count,
            })
            
            return {"lr_decayed": True, "decay_count": self._lr_decay_count}
        
        return {}
    
    # ========== 状态查询 ==========
    
    @property
    def should_stop(self) -> bool:
        return self._should_stop
    
    @property
    def te_frozen(self) -> bool:
        return self._te_frozen
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            "step": self._step,
            "te_frozen": self._te_frozen,
            "should_stop": self._should_stop,
            "lr_decay_count": self._lr_decay_count,
            "clip_drift": self.metrics.clip_drift,
            "stable_rank": self.metrics.stable_rank,
            "gsnr": self.metrics.gsnr,
            "events_count": len(self._events),
        }
    
    def get_events(self) -> List[Tuple[int, str, Dict[str, Any]]]:
        """获取事件日志"""
        return [(step, event.value, data) for step, event, data in self._events]


# ========== 集成回调 ==========

class AutoControllerCallback:
    """
    训练回调
    
    集成到训练循环中
    """
    
    def __init__(
        self,
        controller: AutoController,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics_provider: Optional[Callable[[], Dict[str, float]]] = None,
    ):
        self.controller = controller
        self.model = model
        self.optimizer = optimizer
        self.metrics_provider = metrics_provider
    
    def on_step_end(self, step: int, loss: float = None, **extra_metrics):
        """步结束回调"""
        # 收集指标
        metrics = {"loss": loss} if loss is not None else {}
        metrics.update(extra_metrics)
        
        if self.metrics_provider:
            provider_metrics = self.metrics_provider()
            if provider_metrics:
                metrics.update(provider_metrics)
        
        # 调用控制器
        result = self.controller.step(
            step=step,
            metrics=metrics,
            model=self.model,
            optimizer=self.optimizer,
        )
        
        return result


# ========== 便捷函数 ==========

def create_auto_controller(
    auto_freeze_te: bool = True,
    smart_early_stop: bool = True,
    smart_lr_decay: bool = True,
    **kwargs
) -> AutoController:
    """创建自动控制器"""
    config = AutoControlConfig(
        auto_freeze_te=auto_freeze_te,
        smart_early_stop=smart_early_stop,
        smart_lr_decay=smart_lr_decay,
        **kwargs
    )
    return AutoController(config)


def create_auto_controller_callback(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics_provider: Optional[Callable[[], Dict[str, float]]] = None,
    **config_kwargs
) -> AutoControllerCallback:
    """创建自动控制器回调"""
    controller = create_auto_controller(**config_kwargs)
    return AutoControllerCallback(
        controller=controller,
        model=model,
        optimizer=optimizer,
        metrics_provider=metrics_provider,
    )
