"""
训练安全守卫

功能:
- NaN 自动熔断
- Prodigy 安全预设
- 自动回滚和恢复
"""

import torch
import math
import logging
import json
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


from ..constants import DEFAULT_QUARANTINE_DIR

class SafeGuardAction(Enum):
    """安全守卫动作"""
    CONTINUE = "continue"        # 继续训练
    REDUCE_LR = "reduce_lr"      # 降低学习率
    ROLLBACK = "rollback"        # 回滚到检查点
    STOP = "stop"                # 停止训练


@dataclass
class SafeGuardConfig:
    """安全守卫配置"""
    # NaN 检测
    enable_nan_detection: bool = True
    nan_check_interval: int = 10  # 每 N 步检查一次
    gradient_check_interval: int = 10  # 每 N 步扫描一次梯度；loss NaN 仍按 nan_check_interval 检查
    gradient_scan_mode: str = "batched"  # legacy | batched | foreach | off
    max_nan_count: int = 3        # 最大 NaN 次数后停止
    
    # Loss 异常检测
    enable_loss_spike_detection: bool = True
    loss_spike_threshold: float = 10.0  # Loss 突增阈值
    loss_window_size: int = 50          # 移动平均窗口
    
    # LR 死锁检测 (针对 Prodigy)
    enable_lr_deadlock_detection: bool = True
    lr_deadlock_threshold: float = 1e-8  # LR 低于此值视为死锁
    lr_deadlock_steps: int = 200         # 连续 N 步 LR 不变
    
    # 自动恢复
    enable_auto_recovery: bool = True
    lr_reduction_factor: float = 0.5    # 回滚后 LR 降低因子
    
    # 坏样本剔除 (Bad Sample Culling)
    # report: only append JSONL events; move: move files into quarantine_dir.
    enable_bad_sample_culling: bool = False
    bad_sample_mode: str = "report"
    quarantine_dir: str = DEFAULT_QUARANTINE_DIR # 隔离目录
    bad_sample_report_path: str = ""
    max_reported_samples: int = 32
    
    # 回调
    on_nan_detected: Optional[Callable[[int, float], None]] = None
    on_rollback: Optional[Callable[[int, str], None]] = None
    on_cull_samples: Optional[Callable[[List[str]], None]] = None


@dataclass
class ProdigySafeGuardPreset:
    """
    Prodigy 安全预设
    
    基于社区最佳实践的推荐参数组合
    """
    # 必须开启的安全选项
    decouple: bool = True           # 解耦权重衰减
    safeguard_warmup: bool = True   # 安全预热
    use_bias_correction: bool = True
    
    # 推荐值
    d_coef: float = 1.0             # 保守起始值
    weight_decay: float = 0.0       # Prodigy 不建议用 weight_decay
    growth_rate: float = float("inf")  # 默认
    
    # 学习率边界
    lr_lower_bound: float = 1e-8
    lr_upper_bound: float = 1.0
    
    @classmethod
    def conservative(cls) -> "ProdigySafeGuardPreset":
        """保守预设 - 最稳定"""
        return cls(
            d_coef=0.5,
            safeguard_warmup=True,
            weight_decay=0.0,
        )
    
    @classmethod
    def balanced(cls) -> "ProdigySafeGuardPreset":
        """平衡预设 - 推荐"""
        return cls(
            d_coef=1.0,
            safeguard_warmup=True,
            weight_decay=0.0,
        )
    
    @classmethod
    def aggressive(cls) -> "ProdigySafeGuardPreset":
        """激进预设 - 快速收敛但有风险"""
        return cls(
            d_coef=2.0,
            safeguard_warmup=True,
            weight_decay=0.01,
        )
    
    def to_optimizer_kwargs(self) -> Dict[str, Any]:
        """转换为优化器参数"""
        return {
            "d_coef": self.d_coef,
            "decouple": self.decouple,
            "safeguard_warmup": self.safeguard_warmup,
            "use_bias_correction": self.use_bias_correction,
            "weight_decay": self.weight_decay,
            "growth_rate": self.growth_rate,
        }


class TrainingSafeGuard:
    """
    训练安全守卫
    
    监控训练过程，自动处理异常情况
    """
    
    def __init__(self, config: Optional[SafeGuardConfig] = None):
        self.config = config or SafeGuardConfig()
        
        # 历史记录
        self._loss_history: List[float] = []
        self._lr_history: List[float] = []
        self._nan_count: int = 0
        self._lr_deadlock_count: int = 0
        
        # 检查点
        self._last_safe_state: Optional[Dict] = None
        self._last_safe_step: int = 0
        self._bad_sample_events: List[Dict[str, Any]] = []
    
    def check(
        self,
        step: int,
        loss: float,
        lr: float,
        gradients: Optional[torch.Tensor] = None,
        filenames: Optional[List[str]] = None,
    ) -> SafeGuardAction:
        """
        检查训练状态
        
        Returns:
            建议的动作
        """
        # NaN 检测
        if self.config.enable_nan_detection:
            if step % self.config.nan_check_interval == 0:
                action = self._check_nan(step, loss, gradients)
                if action != SafeGuardAction.CONTINUE:
                    return action
        
        # Loss 异常检测
        if self.config.enable_loss_spike_detection:
            action = self._check_loss_spike(step, loss, filenames)
            if action != SafeGuardAction.CONTINUE:
                return action
        
        # LR 死锁检测
        if self.config.enable_lr_deadlock_detection:
            action = self._check_lr_deadlock(step, lr)
            if action != SafeGuardAction.CONTINUE:
                return action
        
        # 记录历史
        self._loss_history.append(loss)
        if len(self._loss_history) > self.config.loss_window_size:
            self._loss_history.pop(0)
        
        self._lr_history.append(lr)
        if len(self._lr_history) > self.config.lr_deadlock_steps:
            self._lr_history.pop(0)
        
        return SafeGuardAction.CONTINUE
    
    def _check_nan(
        self,
        step: int,
        loss: float,
        gradients: Optional[torch.Tensor] = None,
    ) -> SafeGuardAction:
        """检查 NaN"""
        is_nan = False
        
        # 检查 Loss
        if math.isnan(loss) or math.isinf(loss):
            is_nan = True
            logger.warning(f"[SafeGuard] NaN/Inf loss detected at step {step}")
        
        # 检查梯度
        if gradients is not None:
            if torch.isnan(gradients).any() or torch.isinf(gradients).any():
                is_nan = True
                logger.warning(f"[SafeGuard] NaN/Inf gradients detected at step {step}")
        
        if is_nan:
            self._nan_count += 1
            
            if self.config.on_nan_detected:
                self.config.on_nan_detected(step, loss)
            
            if self._nan_count >= self.config.max_nan_count:
                logger.error(f"[SafeGuard] Max NaN count reached ({self._nan_count}), stopping")
                return SafeGuardAction.STOP
            
            if self.config.enable_auto_recovery and self._last_safe_state:
                logger.info(f"[SafeGuard] Triggering rollback to step {self._last_safe_step}")
                return SafeGuardAction.ROLLBACK
            
            return SafeGuardAction.REDUCE_LR
        
        return SafeGuardAction.CONTINUE
    
    def _check_loss_spike(self, step: int, loss: float, filenames: Optional[List[str]] = None) -> SafeGuardAction:
        """检查 Loss 突增"""
        if len(self._loss_history) < 10:
            return SafeGuardAction.CONTINUE
        
        # 计算移动平均
        avg_loss = sum(self._loss_history[-10:]) / 10
        
        # 增加最小 Loss 检查，防止在 Loss 很小时误触
        if loss > avg_loss * self.config.loss_spike_threshold and avg_loss > 0.01:
            logger.warning(f"[SafeGuard] Loss spike detected: {loss:.4f} vs avg {avg_loss:.4f}")
            
            if self.config.enable_bad_sample_culling and filenames:
                self._handle_bad_samples(
                    step=step,
                    loss=loss,
                    avg_loss=avg_loss,
                    reason="loss_spike",
                    filenames=filenames,
                )
            
            return SafeGuardAction.REDUCE_LR
        
        return SafeGuardAction.CONTINUE


    def _normalize_filenames(self, filenames: Optional[List[str]]) -> List[str]:
        if not filenames:
            return []
        if isinstance(filenames, (str, Path)):
            items = [str(filenames)]
        else:
            items = [str(item) for item in filenames if item]
        seen = set()
        normalized = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        return normalized

    def _handle_bad_samples(
        self,
        *,
        step: int,
        loss: float,
        avg_loss: float,
        reason: str,
        filenames: Optional[List[str]],
    ) -> None:
        samples = self._normalize_filenames(filenames)
        if not samples:
            return
        limit = max(int(getattr(self.config, "max_reported_samples", 32) or 32), 1)
        reported = samples[:limit]
        event = {
            "timestamp": time.time(),
            "step": int(step),
            "reason": reason,
            "loss": float(loss),
            "avg_loss": float(avg_loss),
            "sample_count": len(samples),
            "samples": reported,
            "truncated": len(samples) > len(reported),
            "mode": str(getattr(self.config, "bad_sample_mode", "report") or "report").lower(),
        }
        self._bad_sample_events.append(event)
        if len(self._bad_sample_events) > 200:
            self._bad_sample_events.pop(0)
        self._write_bad_sample_event(event)

        mode = str(getattr(self.config, "bad_sample_mode", "report") or "report").strip().lower()
        if mode in {"move", "quarantine"}:
            self._quarantine_samples(reported)
            if self.config.on_cull_samples:
                self.config.on_cull_samples(reported)
        else:
            logger.warning(
                "[SafeGuard] Bad sample report recorded for %d sample(s); mode=report so files were not moved",
                len(reported),
            )

    def _write_bad_sample_event(self, event: Dict[str, Any]) -> None:
        report_path = str(getattr(self.config, "bad_sample_report_path", "") or "").strip()
        if not report_path:
            return
        try:
            target = Path(report_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[SafeGuard] Failed to write bad sample report: %s", e)
    def _quarantine_samples(self, filenames: List[str]):
        """隔离坏样本"""
        import shutil
        
        q_dir = Path(self.config.quarantine_dir)
        q_dir.mkdir(parents=True, exist_ok=True)
        
        for fname in filenames:
            try:
                src = Path(fname)
                if src.exists():
                    dst = q_dir / src.name
                    # Avoid overwrite
                    counter = 1
                    while dst.exists():
                        dst = q_dir / f"{src.stem}_{counter}{src.suffix}"
                        counter += 1
                        
                    shutil.move(str(src), str(dst))
                    logger.info(f"[SafeGuard] Moved bad sample {src.name} to {dst}")
                    
                    # Also move/delete caption file if exists
                    for ext in ['.txt', '.caption', '.json']:
                        sidecar = src.with_suffix(ext)
                        if sidecar.exists():
                            shutil.move(str(sidecar), str(q_dir / sidecar.name))
            except Exception as e:
                logger.error(f"[SafeGuard] Failed to quarantine {fname}: {e}")
    
    def _check_lr_deadlock(self, step: int, lr: float) -> SafeGuardAction:
        """检查 LR 死锁 (Prodigy 特有问题)"""
        if lr < self.config.lr_deadlock_threshold:
            self._lr_deadlock_count += 1
            
            if self._lr_deadlock_count >= self.config.lr_deadlock_steps:
                logger.warning(f"[SafeGuard] LR deadlock detected: lr={lr:.2e} for {self._lr_deadlock_count} steps")
                # LR 死锁时，尝试重置 Prodigy 状态
                return SafeGuardAction.REDUCE_LR  # 触发回调处理
        else:
            self._lr_deadlock_count = 0
        
        return SafeGuardAction.CONTINUE
    
    def save_safe_state(self, step: int, state: Dict):
        """保存安全状态 (用于回滚)"""
        self._last_safe_state = state
        self._last_safe_step = step
        self._nan_count = 0  # 重置计数
    
    def get_rollback_state(self) -> Optional[Dict]:
        """获取回滚状态"""
        return self._last_safe_state
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "nan_count": self._nan_count,
            "lr_deadlock_count": self._lr_deadlock_count,
            "loss_history_len": len(self._loss_history),
            "last_safe_step": self._last_safe_step,
            "avg_recent_loss": sum(self._loss_history[-10:]) / 10 if len(self._loss_history) >= 10 else None,
            "bad_sample_event_count": len(self._bad_sample_events),
            "last_bad_sample_event": self._bad_sample_events[-1] if self._bad_sample_events else None,
        }


# ========== Prodigy 安全包装 ==========

def create_safe_prodigy_optimizer(
    params,
    preset: str = "balanced",
    base_lr: float = 1.0,
) -> torch.optim.Optimizer:
    """
    创建带安全预设的 Prodigy 优化器
    
    Args:
        params: 模型参数
        preset: "conservative" | "balanced" | "aggressive"
        base_lr: 基础学习率 (Prodigy 自动调整)
    """
    try:
        from prodigyopt import Prodigy
    except ImportError:
        logger.warning("prodigyopt not installed, falling back to AdamW")
        return torch.optim.AdamW(params, lr=1e-4)
    
    # 获取预设
    presets = {
        "conservative": ProdigySafeGuardPreset.conservative(),
        "balanced": ProdigySafeGuardPreset.balanced(),
        "aggressive": ProdigySafeGuardPreset.aggressive(),
    }
    config = presets.get(preset, ProdigySafeGuardPreset.balanced())
    
    logger.info(f"[Prodigy] Using {preset} preset: d_coef={config.d_coef}")
    
    return Prodigy(
        params,
        lr=base_lr,
        **config.to_optimizer_kwargs()
    )


def reset_prodigy_state(optimizer) -> bool:
    """
    重置 Prodigy 优化器状态
    
    用于 LR 死锁恢复
    """
    try:
        for group in optimizer.param_groups:
            if "d" in group:
                group["d"] = group.get("d_coef", 1.0)
            if "k" in group:
                group["k"] = 0
        
        # 清除动量
        for state in optimizer.state.values():
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
        
        logger.info("[Prodigy] State reset for LR recovery")
        return True
        
    except Exception as e:
        logger.error(f"[Prodigy] Failed to reset state: {e}")
        return False
