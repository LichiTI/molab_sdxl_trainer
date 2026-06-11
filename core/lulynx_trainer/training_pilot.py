"""
Training Pilot: 自适应学习率 (ORPHANED — not imported by any module)

This is a second copy of the training pilot that was intended for lulynx_trainer
integration but is dead code. The active version lives at core/training_pilot.py
and is only consumed by MNLoRAOptimizer. Do not import from this file.

基于实时 Layer Stats 自动调整各层学习率
"""

import torch
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PilotAggressiveness(Enum):
    """调整激进程度"""
    CONSERVATIVE = "conservative"  # 保守：小幅调整
    BALANCED = "balanced"          # 平衡：默认
    AGGRESSIVE = "aggressive"      # 激进：大幅调整


@dataclass
class LayerStats:
    """层统计信息"""
    name: str
    norm: float = 0.0           # 权重范数
    grad_norm: float = 0.0      # 梯度范数
    sparsity: float = 0.0       # 稀疏度
    mean: float = 0.0
    std: float = 0.0
    
    # 历史数据
    norm_history: List[float] = field(default_factory=list)
    grad_history: List[float] = field(default_factory=list)


@dataclass
class PilotConfig:
    """Pilot 配置"""
    enabled: bool = True
    
    # 阈值
    threshold_high_norm: float = 2.0      # Norm 过高阈值
    threshold_low_grad: float = 1e-6      # 梯度过低阈值 (死神经元)
    threshold_high_grad: float = 1.0      # 梯度过高阈值 (爆炸)
    
    # LR 调整因子
    lr_multiplier_cool: float = 0.5       # 降温因子
    lr_multiplier_boost: float = 2.0      # 激活因子
    lr_multiplier_normal: float = 1.0     # 正常因子
    
    # 变化率
    max_lr_change_per_step: float = 0.1   # 每步最大变化率
    
    # 激进程度
    aggressiveness: PilotAggressiveness = PilotAggressiveness.BALANCED
    
    # 历史
    history_length: int = 10


class TrainingPilot:
    """
    自动驾驶训练引擎
    
    功能:
    1. 抑制过热 (Overfit Suppression) - 降低高 Norm 层的 LR
    2. 唤醒静默 (Dead Neuron Awakening) - 提高低梯度层的 LR
    3. 梯度爆炸防护 - 紧急降低爆炸层的 LR
    """
    
    def __init__(self, config: Optional[PilotConfig] = None):
        self.config = config or PilotConfig()
        self._layer_stats: Dict[str, LayerStats] = {}
        self._current_multipliers: Dict[str, float] = {}
        self._callbacks: List[Callable] = []
        self._step_count = 0
    
    def update_stats(self, layer_name: str, **kwargs):
        """更新层统计"""
        if layer_name not in self._layer_stats:
            self._layer_stats[layer_name] = LayerStats(name=layer_name)
        
        stats = self._layer_stats[layer_name]
        
        for key, value in kwargs.items():
            if hasattr(stats, key):
                setattr(stats, key, value)
        
        # 更新历史
        if "norm" in kwargs:
            stats.norm_history.append(kwargs["norm"])
            if len(stats.norm_history) > self.config.history_length:
                stats.norm_history.pop(0)
        
        if "grad_norm" in kwargs:
            stats.grad_history.append(kwargs["grad_norm"])
            if len(stats.grad_history) > self.config.history_length:
                stats.grad_history.pop(0)
    
    def decide_lr_multipliers(self) -> Dict[str, float]:
        """
        决定各层的 LR 乘数
        
        Returns:
            层名 -> LR 乘数
        """
        multipliers = {}
        
        for name, stats in self._layer_stats.items():
            multiplier = self._decide_single_layer(stats)
            
            # 平滑变化
            if name in self._current_multipliers:
                old = self._current_multipliers[name]
                change = multiplier - old
                max_change = self.config.max_lr_change_per_step
                change = max(-max_change, min(max_change, change))
                multiplier = old + change
            
            multipliers[name] = multiplier
        
        self._current_multipliers = multipliers
        self._step_count += 1
        
        # 触发回调
        for callback in self._callbacks:
            callback(multipliers)
        
        return multipliers
    
    def _decide_single_layer(self, stats: LayerStats) -> float:
        """决定单层的 LR 乘数"""
        cfg = self.config
        
        # 策略 1: 梯度爆炸防护 (最高优先级)
        if stats.grad_norm > cfg.threshold_high_grad:
            logger.warning(f"[Pilot] Gradient explosion detected in {stats.name}, cooling down")
            return cfg.lr_multiplier_cool * 0.5  # 紧急降温
        
        # 策略 2: 抑制过热
        if stats.norm > cfg.threshold_high_norm:
            # 检查趋势
            if len(stats.norm_history) >= 3:
                trend = stats.norm_history[-1] - stats.norm_history[-3]
                if trend > 0:  # 仍在上升
                    return self._apply_aggressiveness(cfg.lr_multiplier_cool)
            return cfg.lr_multiplier_cool
        
        # 策略 3: 唤醒静默神经元
        if stats.grad_norm < cfg.threshold_low_grad:
            # 检查是否持续静默
            if len(stats.grad_history) >= 3:
                avg = sum(stats.grad_history[-3:]) / 3
                if avg < cfg.threshold_low_grad:
                    return self._apply_aggressiveness(cfg.lr_multiplier_boost)
            return cfg.lr_multiplier_boost
        
        # 正常
        return cfg.lr_multiplier_normal
    
    def _apply_aggressiveness(self, multiplier: float) -> float:
        """根据激进程度调整乘数"""
        if self.config.aggressiveness == PilotAggressiveness.CONSERVATIVE:
            # 保守：减半调整幅度
            if multiplier < 1.0:
                return 1.0 - (1.0 - multiplier) * 0.5
            else:
                return 1.0 + (multiplier - 1.0) * 0.5
        
        elif self.config.aggressiveness == PilotAggressiveness.AGGRESSIVE:
            # 激进：加倍调整幅度
            if multiplier < 1.0:
                return max(0.1, 1.0 - (1.0 - multiplier) * 2.0)
            else:
                return min(4.0, 1.0 + (multiplier - 1.0) * 2.0)
        
        return multiplier
    
    def apply_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        param_name_map: Dict[str, int],  # 参数名 -> param_group 索引
    ):
        """
        应用 LR 乘数到优化器
        
        Args:
            optimizer: PyTorch 优化器
            base_lr: 基础学习率
            param_name_map: 参数名到 param_group 索引的映射
        """
        multipliers = self.decide_lr_multipliers()
        
        for name, multiplier in multipliers.items():
            if name in param_name_map:
                idx = param_name_map[name]
                if idx < len(optimizer.param_groups):
                    new_lr = base_lr * multiplier
                    optimizer.param_groups[idx]["lr"] = new_lr
    
    def register_callback(self, callback: Callable):
        """注册 LR 更新回调"""
        self._callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "step": self._step_count,
            "multipliers": dict(self._current_multipliers),
            "layer_stats": {
                name: {
                    "norm": stats.norm,
                    "grad_norm": stats.grad_norm,
                    "sparsity": stats.sparsity,
                }
                for name, stats in self._layer_stats.items()
            },
        }
    
    def get_lr_heatmap_data(self) -> List[Dict[str, Any]]:
        """
        获取 LR 热力图数据 (供 Neuralverse 可视化)
        
        Returns:
            层列表，每个包含 name, multiplier, color
        """
        data = []
        
        for name, multiplier in self._current_multipliers.items():
            # 计算颜色
            if multiplier < 0.8:
                color = "blue"   # 降温
            elif multiplier > 1.2:
                color = "red"    # 激活
            else:
                color = "green"  # 正常
            
            data.append({
                "name": name,
                "multiplier": multiplier,
                "color": color,
                "norm": self._layer_stats.get(name, LayerStats(name)).norm,
                "grad_norm": self._layer_stats.get(name, LayerStats(name)).grad_norm,
            })
        
        return data


class PilotCallback:
    """
    训练回调
    
    集成到训练循环中
    """
    
    def __init__(
        self,
        pilot: TrainingPilot,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        update_interval: int = 10,
    ):
        self.pilot = pilot
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.update_interval = update_interval
        self._step = 0
        self._param_name_map: Dict[str, int] = {}
    
    def register_params(self, model: torch.nn.Module):
        """注册模型参数"""
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad:
                self._param_name_map[name] = min(i, len(self.optimizer.param_groups) - 1)
    
    def on_step_end(self, model: torch.nn.Module):
        """训练步结束回调"""
        self._step += 1
        
        if self._step % self.update_interval != 0:
            return
        
        # 收集统计
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.pilot.update_stats(
                    name,
                    norm=param.data.norm().item(),
                    grad_norm=param.grad.norm().item(),
                )
        
        # 应用 LR
        self.pilot.apply_to_optimizer(
            self.optimizer,
            self.base_lr,
            self._param_name_map,
        )


# ========== 便捷函数 ==========

def create_training_pilot(
    aggressiveness: str = "balanced",
) -> TrainingPilot:
    """创建训练 Pilot"""
    config = PilotConfig(
        aggressiveness=PilotAggressiveness(aggressiveness),
    )
    return TrainingPilot(config)


def create_pilot_callback(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    aggressiveness: str = "balanced",
    broadcast: bool = True,
) -> PilotCallback:
    """创建 Pilot 回调"""
    pilot = create_training_pilot(aggressiveness)
    
    # 注册 WebSocket 广播
    if broadcast:
        try:
            from routers.websocket import sync_broadcast_lr_update
            
            def on_lr_update(multipliers):
                data = pilot.get_lr_heatmap_data()
                sync_broadcast_lr_update(data)
            
            pilot.register_callback(on_lr_update)
        except ImportError:
            pass  # WebSocket 不可用
    
    callback = PilotCallback(pilot, optimizer, base_lr)
    callback.register_params(model)
    return callback
