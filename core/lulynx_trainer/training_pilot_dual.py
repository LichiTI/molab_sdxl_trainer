"""
Dual Validated Dynamic LR Pilot
双重验证动态学习率控制器

策略：
1. 方法 A：基于梯度统计 (Gradient Stats)
2. 方法 B：基于损失趋势 (Loss Trend)
3. 双重验证：两种方法差异 < 5% 才调整
4. Orchestra 集成：服从全局冷却指令
"""

import torch
import numpy as np
import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class DualValidatedDynamicLR:
    """
    双重验证的动态 LR 调整
    """
    
    def __init__(
        self,
        # 方法 A：梯度统计
        grad_norm_window: int = 20,  # 滑动窗口
        grad_norm_threshold: float = 0.001,  # 梯度范数阈值
        
        # 方法 B：损失趋势
        loss_window: int = 50,  # 滑动窗口
        loss_trend_threshold: float = 0.001,  # 损失趋势阈值
        
        # 双重验证
        validation_threshold: float = 0.05,  # 两种方法差异 < 5%
        
        # LR 调整
        lr_min_multiplier: float = 0.1,
        lr_max_multiplier: float = 3.0,
        
        # Orchestra 频率由 OrchestraChecker 控制，这里只需要知道是否被允许
    ):
        self.grad_norm_window = grad_norm_window
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_window = loss_window
        self.loss_trend_threshold = loss_trend_threshold
        self.validation_threshold = validation_threshold
        
        self.lr_min_multiplier = lr_min_multiplier
        self.lr_max_multiplier = lr_max_multiplier
        
        # 状态
        self._step = 0
        self._grad_norm_history = []
        self._loss_history = []
        
        # 验证结果记录
        self._last_adjustment_valid = False
        self._last_validation_diff = 0.0
    
    def validate_and_adjust(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: float,
    ) -> bool:
        """
        验证并调整 LR
        
        Returns:
            True: LR 已调整
            False: LR 未调整 (被 Orchestra 阻止或验证失败)
        """
        self._step += 1
        
        # 0. 检查 Orchestra 控制器准入
        try:
            from core.lulynx_trainer.orchestra_controller import get_orchestra
            orchestra = get_orchestra()
            if not orchestra.should_run_lr_adjustment():
                # 被冷却或频率限制阻止
                # 依然更新历史数据，保持统计连续性
                self._update_history(model, loss)
                return False
        except ImportError:
            pass
            
        # 1. 更新历史并计算方法 A (梯度统计)
        # 注意：这里我们只计算 multipliers，历史更新在 _update_history 中统一处理
        # 但为了 logic 简单，我们在这里计算
        self._update_history(model, loss)
        
        lr_multiplier_a = self._method_a_gradient_stats()
        
        # 2. 计算方法 B (损失趋势)
        lr_multiplier_b = self._method_b_loss_trend()
        
        # 3. 双重验证
        is_valid, validation_diff = self._dual_validation(
            lr_multiplier_a, lr_multiplier_b
        )
        
        # 4. 调整 LR
        if is_valid:
            # 取平均
            lr_multiplier = (lr_multiplier_a + lr_multiplier_b) / 2
            
            # 如果乘数接近 1.0 (例如 0.99-1.01)，则不调整，减少震荡
            if 0.99 < lr_multiplier < 1.01:
                return False

            # 调整 LR
            self._adjust_lr(optimizer, lr_multiplier)
            
            self._last_adjustment_valid = True
            self._last_validation_diff = validation_diff
            
            logger.info(
                f"[DualValidatedLR] Step {self._step}: "
                f"Method A={lr_multiplier_a:.3f}, "
                f"Method B={lr_multiplier_b:.3f}, "
                f"Diff={validation_diff:.3f} < {self.validation_threshold:.3f} ✅"
            )
            
            return True
        else:
            self._last_adjustment_valid = False
            self._last_validation_diff = validation_diff
            
            # 只有差异特别大才 log warning，否则 debug
            if validation_diff > 0.2:
                logger.debug(
                    f"[DualValidatedLR] Validation failed: A={lr_multiplier_a:.2f}, B={lr_multiplier_b:.2f}, Diff={validation_diff:.2f}"
                )
            
            return False

    def _update_history(self, model: torch.nn.Module, loss: float):
        """更新统计历史"""
        # 更新 Loss
        self._loss_history.append(loss)
        if len(self._loss_history) > self.loss_window:
            self._loss_history.pop(0)

        # 更新 Gradient Norm
        total_grad_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
        avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0.0
        
        self._grad_norm_history.append(avg_grad_norm)
        if len(self._grad_norm_history) > self.grad_norm_window:
            self._grad_norm_history.pop(0)

    def _method_a_gradient_stats(self) -> float:
        """方法 A：基于梯度统计"""
        if len(self._grad_norm_history) < self.grad_norm_window:
            return 1.0
        
        x = np.arange(len(self._grad_norm_history))
        y = np.array(self._grad_norm_history)
        
        if len(x) < 2: return 1.0
        
        # 线性回归斜率
        slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        
        # 梯度下降 -> 降低 LR (假设收敛)
        # 梯度上升 -> 保持或微增 (假设需要跳出鞍点? 或者震荡?)
        # 用户策略: "梯度范数小 -> 降低 LR" (接近收敛); "梯度范数大 -> 保持 LR"
        # 这里的 slope 是变化率。
        # 如果 slope < 0 (梯度在减小): 说明正在收敛 -> 降低 LR 以精细收敛
        # 如果 slope > 0 (梯度在增大): 说明可能在震荡或跳出 -> 保持 LR
        
        if slope < -self.grad_norm_threshold:
            # 梯度减小，降低 LR
            # slope 是负数，例如 -0.001
            # multiplier = 1.0 + slope * 100 = 0.9
            return max(self.lr_min_multiplier, 0.5 + slope * 100)
        elif slope > self.grad_norm_threshold:
            # 梯度增大
            return min(self.lr_max_multiplier, 1.0 + slope * 100)
        else:
            return 1.0

    def _method_b_loss_trend(self) -> float:
        """方法 B：基于损失趋势"""
        if len(self._loss_history) < self.loss_window:
            return 1.0
        
        x = np.arange(len(self._loss_history))
        y = np.array(self._loss_history)
        
        if len(x) < 2: return 1.0
        
        slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        
        # Loss 上升 (slope > 0) -> 降低 LR (震荡)
        # Loss 下降 (slope < 0) -> 保持 LR (正常学习)
        
        if slope > self.loss_trend_threshold:
            # Loss 上升/震荡 -> 降低 LR
            # slope 正数
            # multiplier = 1.0 - slope * ... ?
            # 用户逻辑: max(min, 0.5 + slope * 100) ?? 
            # 如果 slope 是正的，0.5 + 正数 > 0.5。
            # 实际上震荡应该减小 LR。
            # 假设我们用 inverse logic:
            # 强震荡 (slope 大) -> LR 变小
            return max(self.lr_min_multiplier, 1.0 / (1.0 + slope * 100))
        else:
            # Loss 下降
            return min(self.lr_max_multiplier, 1.0 - slope * 50) # 下降越快越可以胆大一点

    def _dual_validation(self, ma: float, mb: float) -> Tuple[bool, float]:
        diff = abs(ma - mb) / max(abs(ma), abs(mb), 1e-8)
        return diff < self.validation_threshold, diff

    def _adjust_lr(self, optimizer: torch.optim.Optimizer, multiplier: float):
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * multiplier
            new_lr = max(old_lr * self.lr_min_multiplier, min(old_lr * self.lr_max_multiplier, new_lr))
            param_group['lr'] = new_lr
