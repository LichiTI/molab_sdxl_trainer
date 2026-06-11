"""
LN Guard (LayerNorm 卫士) - 防炸色系统

针对 V-Prediction 模型（如 NoobAI, Illusion）的色彩稳定性模块。
通过弹性正则化控制 LayerNorm 的 gamma/beta 漂移。

技术原理:
1. 训练开始时捕获所有 LN 层的 γ 和 β 基线值  
2. 使用 L2 正则惩罚偏离基线的参数
3. Anti-Fry 模式：监控高频梯度，动态调整 λ
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("LNGuard")


class LNGuard:
    """
    LayerNorm 卫士 - 防止 V-Pred 模型炸色
    
    使用方式:
        guard = LNGuard(lambda_scale=0.01)
        guard.capture_baseline(model)
        loss = guard.compute_loss(model)
    """
    
    def __init__(
        self,
        lambda_scale: float = 0.01,
        lambda_shift: float = 0.005,
        anti_fry: bool = False,
        anti_fry_threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Args:
            lambda_scale: γ (scale) 惩罚强度
            lambda_shift: β (shift) 惩罚强度 (通常小于 scale)
            anti_fry: 是否启用 Anti-Fry 模式 (针对 NoobAI/Illusion)
            anti_fry_threshold: 高频梯度阈值
            device: 计算设备
        """
        self.lambda_scale = lambda_scale
        self.lambda_shift = lambda_shift
        self.anti_fry = anti_fry
        self.anti_fry_threshold = anti_fry_threshold
        self.device = device
        
        # 基线值缓存 {layer_name: (gamma_baseline, beta_baseline)}
        self._baseline_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # 动态 lambda 调整 (Anti-Fry)
        self._dynamic_lambda: Dict[str, float] = {}
        
        # 梯度监控 (Anti-Fry)
        self._grad_history: Dict[str, List[float]] = {}
    
    def _find_ln_layers(self, model: nn.Module) -> Dict[str, nn.LayerNorm]:
        """查找模型中所有 LayerNorm 层"""
        ln_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                ln_layers[name] = module
        return ln_layers
    
    def capture_baseline(self, model: nn.Module) -> int:
        """
        捕获所有 LN 层的基线参数
        
        Args:
            model: 目标模型
            
        Returns:
            num_layers: 捕获的 LN 层数量
        """
        self._baseline_params.clear()
        self._dynamic_lambda.clear()
        self._grad_history.clear()
        
        ln_layers = self._find_ln_layers(model)
        
        for name, layer in ln_layers.items():
            if layer.weight is not None and layer.bias is not None:
                # 克隆并分离基线值
                gamma_baseline = layer.weight.data.clone().detach()
                beta_baseline = layer.bias.data.clone().detach()
                
                self._baseline_params[name] = (gamma_baseline, beta_baseline)
                self._dynamic_lambda[name] = 1.0  # 初始动态系数
                self._grad_history[name] = []
        
        return len(self._baseline_params)
    
    def compute_loss(self, model: nn.Module, weight: float = 1.0) -> torch.Tensor:
        """
        计算 LN 弹性正则化损失
        
        Loss = λ_scale * Σ||γ - γ_base||² + λ_shift * Σ||β - β_base||²
        
        Args:
            model: 当前模型
            weight: 全局损失权重
            
        Returns:
            loss: LN 正则化损失
        """
        if not self._baseline_params:
            return torch.tensor(0.0, device=self.device)
        
        ln_layers = self._find_ln_layers(model)
        
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        for name, layer in ln_layers.items():
            if name not in self._baseline_params:
                continue
            
            gamma_baseline, beta_baseline = self._baseline_params[name]
            dynamic_weight = self._dynamic_lambda.get(name, 1.0)
            
            # γ (scale) 损失 - L2 范数
            if layer.weight is not None:
                gamma_diff = layer.weight - gamma_baseline.to(layer.weight.device)
                gamma_loss = (gamma_diff ** 2).sum()
                total_loss = total_loss + self.lambda_scale * dynamic_weight * gamma_loss
            
            # β (shift) 损失 - L2 范数 (权重较小)
            if layer.bias is not None:
                beta_diff = layer.bias - beta_baseline.to(layer.bias.device)
                beta_loss = (beta_diff ** 2).sum()
                total_loss = total_loss + self.lambda_shift * dynamic_weight * beta_loss
            
            count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return weight * total_loss
    
    def update_anti_fry(self, model: nn.Module) -> Dict[str, float]:
        """
        Anti-Fry 模式：监控高频梯度并动态调整 λ
        
        当某层的梯度方差异常高时（炸色前兆），增加该层的惩罚
        
        Returns:
            adjustments: 各层的动态系数 {layer_name: multiplier}
        """
        if not self.anti_fry:
            return {}
        
        adjustments = {}
        ln_layers = self._find_ln_layers(model)
        
        for name, layer in ln_layers.items():
            if name not in self._baseline_params:
                continue
            
            if layer.weight is not None and layer.weight.grad is not None:
                # 计算梯度方差 (高频活动指标)
                grad_var = layer.weight.grad.var().item()
                
                # 记录历史
                if name not in self._grad_history:
                    self._grad_history[name] = []
                self._grad_history[name].append(grad_var)
                
                # 保持最近 100 步
                if len(self._grad_history[name]) > 100:
                    self._grad_history[name] = self._grad_history[name][-100:]
                
                # 计算相对于历史的异常程度
                if len(self._grad_history[name]) > 10:
                    mean_var = sum(self._grad_history[name]) / len(self._grad_history[name])
                    if grad_var > mean_var * (1 + self.anti_fry_threshold):
                        # 梯度异常高 -> 增加惩罚
                        boost = min(3.0, grad_var / (mean_var + 1e-8))
                        self._dynamic_lambda[name] = boost
                        adjustments[name] = boost
                    else:
                        # 恢复正常
                        self._dynamic_lambda[name] = max(1.0, self._dynamic_lambda.get(name, 1.0) * 0.95)
        
        return adjustments
    
    def get_stats(self) -> Dict[str, Dict]:
        """获取各 LN 层的统计信息"""
        stats = {}
        for name, (gamma, beta) in self._baseline_params.items():
            stats[name] = {
                "gamma_baseline_mean": gamma.mean().item(),
                "gamma_baseline_std": gamma.std().item(),
                "beta_baseline_mean": beta.mean().item(),
                "beta_baseline_std": beta.std().item(),
                "dynamic_lambda": self._dynamic_lambda.get(name, 1.0),
            }
        return stats
