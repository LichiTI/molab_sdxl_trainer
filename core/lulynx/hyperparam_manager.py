"""
Lulynx Hyperparam Manager
MN-LoRA 超参数管理器

职责：
1. 统一管理分散在各个模块的超参数 (Dual LR, TE Removal, Smart Rank, etc.)
2. 提供简单易用的预设 (Presets): FAST, BALANCED, QUALITY
3. 作为单一事实来源，防止参数冲突
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MNPreset(Enum):
    SLIM = "slim"           # 默认实用档：保留约束收益，避免 KFAC 等重型路径
    FAST = "fast"           # 极速模式：激进剪枝，快速收敛
    BALANCED = "balanced"   # 平衡模式：默认推荐
    QUALITY = "quality"     # 画质模式：保守剪枝，高精度

@dataclass
class MNLoraConfig:
    # 基础配置
    preset: str = "slim"
    
    # Dynamic Rank Pruner
    rank_pruning_enabled: bool = True
    prune_threshold: float = 0.05
    min_rank: int = 8
    
    # Dual Validated LR
    pilot_enabled: bool = True
    pilot_strategy: str = "dual" # dual | standard | off
    lr_validation_threshold: float = 0.05
    
    # TE Manager
    te_management_enabled: bool = True
    te_removal_strategy: str = "to_cpu"
    te_consecutive_steps: int = 10
    
    # Manifold Constraint (GSP)
    manifold_enabled: bool = True
    manifold_strength: float = 0.1
    manifold_proj_dim: int = 128
    manifold_sparse_freq: int = 1

class LulynxHyperparamManager:
    """
    超参数管理器单例
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LulynxHyperparamManager, cls).__new__(cls)
            cls._instance._config = MNLoraConfig()
        return cls._instance
    
    @classmethod
    def get_config(cls) -> MNLoraConfig:
        return cls()._config
    
    @classmethod
    def apply_preset(cls, preset_name: str, model_arch: str = "sdxl") -> MNLoraConfig:
        """
        应用预设并注入架构特定参数
        """
        manager = cls()
        config = MNLoraConfig()
        config.preset = preset_name
        
        # 1. 架构特定基础设置 (SDXL vs Flux)
        is_flux = model_arch.lower() == "flux"
        if is_flux:
            config.manifold_proj_dim = 256
            config.manifold_strength = 0.05 # Flux 空间更大，约束需稍微放宽
            config.manifold_sparse_freq = 4  # Flux 计算重，默认开启稀疏
        else:
            config.manifold_proj_dim = 128
            config.manifold_strength = 0.1
            config.manifold_sparse_freq = 1
        
        # 2. 应用模式预设覆盖
        if preset_name == MNPreset.FAST.value:
            # FAST: 激进剪枝, 尽早卸载 TE
            config.rank_pruning_enabled = True
            config.prune_threshold = 0.10      # 容忍 10% 的低利用率
            config.min_rank = 4                # 允许压得更低
            config.te_management_enabled = True
            config.te_consecutive_steps = 5    # 更快卸载 TE
            config.manifold_enabled = False    # 关闭 GSP 以换取速度
            
        elif preset_name == MNPreset.QUALITY.value:
            # QUALITY: 保守剪枝, 强约束
            config.rank_pruning_enabled = True
            config.prune_threshold = 0.01      # 仅剔除极低利用率 (<1%)
            config.min_rank = 16               # 保留更多 Rank
            config.te_management_enabled = False # 不卸载 TE (防止微小精度损失)
            config.manifold_enabled = True
            config.manifold_strength = 0.2 if not is_flux else 0.1
            
        else: # SLIM/BALANCED
            # SLIM/BALANCED: 默认设置
            config.rank_pruning_enabled = True
            config.prune_threshold = 0.05
            config.min_rank = 8
            config.te_management_enabled = True
            config.te_consecutive_steps = 10
            config.manifold_enabled = True
            
        manager._config = config
        logger.info(f"[HyperparamManager] Applied {preset_name.upper()} preset for architecture: {model_arch.upper()}")
        return config

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """导出为字典供前端或 Config 使用"""
        return asdict(cls()._config)

# 全局快捷访问
def get_mn_config() -> MNLoraConfig:
    return LulynxHyperparamManager.get_config()
