"""
Diffusion Transformer (DiT) 通用处理模块

支持架构:
- Flux (Black Forest Labs)
- SD3 / MMDiT (Stability AI) - 预留
- PixArt-α - 预留
- Hunyuan-DiT - 预留

设计原则:
- 架构无关的抽象接口
- 适配器模式支持不同模型
- 向前兼容新模型
"""

from .base import DiTBase, DiTConfig, DiTType
from .analyzer import DiTAnalyzer, DiTLayerInfo
from .flux_adapter import FluxAdapter

__all__ = [
    # 核心
    "DiTBase",
    "DiTConfig",
    "DiTType",
    
    # 分析
    "DiTAnalyzer",
    "DiTLayerInfo",
    
    # 适配器
    "FluxAdapter",
]
