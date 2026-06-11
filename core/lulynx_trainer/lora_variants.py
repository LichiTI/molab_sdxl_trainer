"""
高级 LoRA 变体

支持:
- DoRA (Weight-Decomposed Low-Rank Adaptation)
- LoRA+ (Asymmetric Learning Rate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ==================== DoRA ====================

class DoRALinear(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation)
    
    论文: https://arxiv.org/abs/2402.09353
    
    核心思想:
    W' = m * (W + ΔW) / ||W + ΔW||
    
    其中:
    - m 是可学习的幅度参数
    - ΔW = BA (标准 LoRA)
    - 对权重进行方向/幅度分解
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        
        # DoRA 幅度参数
        self.magnitude = nn.Parameter(torch.ones(out_features, dtype=dtype))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor, original_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量
            original_weight: 原始权重矩阵
        """
        # 合并权重并计算方向 (归一化)
        merged_weight = torch.addmm(
            original_weight.to(dtype=self.lora_A.dtype),
            self.lora_B,
            self.lora_A,
            beta=1.0,
            alpha=self.scaling,
        )
        weight_norm = merged_weight.norm(dim=1, keepdim=True)
        final_weight = merged_weight * (self.magnitude.unsqueeze(1) / (weight_norm + 1e-8))
        
        # 前向传播
        return F.linear(self.dropout(x), final_weight)
    
    def get_delta_weight(self) -> torch.Tensor:
        """获取 LoRA 增量"""
        return self.lora_B @ self.lora_A * self.scaling


class DoRAInjector:
    """DoRA 注入器"""
    
    def __init__(self, rank: int = 4, alpha: float = 1.0):
        self.rank = rank
        self.alpha = alpha
        self._dora_layers: Dict[str, DoRALinear] = {}
        self._original_weights: Dict[str, torch.Tensor] = {}
    
    def inject(self, module: nn.Module, name: str = "") -> int:
        """
        注入 DoRA 层
        
        Returns:
            注入的层数
        """
        count = 0
        
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, nn.Linear):
                # 创建 DoRA 层
                dora = DoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=self.rank,
                    alpha=self.alpha,
                    dtype=child.weight.dtype,
                )
                dora.to(child.weight.device)
                
                # 保存原始权重引用
                self._original_weights[full_name] = child.weight
                self._dora_layers[full_name] = dora
                
                count += 1
            else:
                count += self.inject(child, full_name)
        
        return count
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        params = []
        for dora in self._dora_layers.values():
            params.extend([dora.lora_A, dora.lora_B, dora.magnitude])
        return params


# ==================== LoRA+ ====================

@dataclass
class LoraPlusConfig:
    """LoRA+ 配置"""
    # 学习率比例
    lora_A_lr_ratio: float = 1.0    # A 矩阵学习率倍数
    lora_B_lr_ratio: float = 16.0   # B 矩阵学习率倍数 (论文推荐 16x)
    
    # 基础学习率
    base_lr: float = 1e-4


def create_lora_plus_param_groups(
    lora_layers: Dict[str, nn.Module],
    config: LoraPlusConfig,
) -> List[Dict[str, Any]]:
    """
    创建 LoRA+ 参数组
    
    LoRA+ 核心思想: B 矩阵使用更高学习率
    
    论文: https://arxiv.org/abs/2402.12354
    
    Returns:
        可传递给优化器的参数组列表
    """
    lora_A_params = []
    lora_B_params = []
    other_params = []
    
    for name, layer in lora_layers.items():
        for param_name, param in layer.named_parameters():
            if "lora_A" in param_name or "lora_down" in param_name:
                lora_A_params.append(param)
            elif "lora_B" in param_name or "lora_up" in param_name:
                lora_B_params.append(param)
            else:
                other_params.append(param)
    
    param_groups = [
        {
            "params": lora_A_params,
            "lr": config.base_lr * config.lora_A_lr_ratio,
            "name": "lora_A",
        },
        {
            "params": lora_B_params,
            "lr": config.base_lr * config.lora_B_lr_ratio,
            "name": "lora_B",
        },
    ]
    
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": config.base_lr,
            "name": "other",
        })
    
    logger.info(f"[LoRA+] Created param groups: "
                f"A={len(lora_A_params)} (lr={config.base_lr * config.lora_A_lr_ratio:.2e}), "
                f"B={len(lora_B_params)} (lr={config.base_lr * config.lora_B_lr_ratio:.2e})")
    
    return param_groups


def create_lora_plus_optimizer(
    lora_layers: Dict[str, nn.Module],
    base_lr: float = 1e-4,
    lora_B_ratio: float = 16.0,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    创建 LoRA+ 优化器
    
    Args:
        lora_layers: LoRA 层字典
        base_lr: 基础学习率
        lora_B_ratio: B 矩阵学习率倍数 (默认 16x)
        weight_decay: 权重衰减
    
    Returns:
        配置好的优化器
    """
    config = LoraPlusConfig(
        base_lr=base_lr,
        lora_B_lr_ratio=lora_B_ratio,
    )
    
    param_groups = create_lora_plus_param_groups(lora_layers, config)
    
    # 添加 weight_decay
    for group in param_groups:
        group["weight_decay"] = weight_decay
    
    return torch.optim.AdamW(param_groups)


# ==================== 统一接口 ====================

class AdvancedLoRAType:
    """高级 LoRA 类型"""
    STANDARD = "lora"
    DORA = "dora"
    LORA_PLUS = "lora+"


def get_optimizer_for_lora_variant(
    variant: str,
    lora_layers: Dict[str, nn.Module],
    base_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    根据 LoRA 变体获取最优优化器
    
    Args:
        variant: "lora" | "dora" | "lora+"
        lora_layers: LoRA 层
        base_lr: 基础学习率
        weight_decay: 权重衰减
    """
    all_params = []
    for layer in lora_layers.values():
        all_params.extend(layer.parameters())
    
    if variant == AdvancedLoRAType.LORA_PLUS:
        return create_lora_plus_optimizer(
            lora_layers, base_lr, lora_B_ratio=16.0, weight_decay=weight_decay
        )
    else:
        # 标准 LoRA 和 DoRA 使用统一学习率
        return torch.optim.AdamW(
            all_params,
            lr=base_lr,
            weight_decay=weight_decay,
        )
