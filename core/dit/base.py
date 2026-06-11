"""
DiT 基础抽象类

定义 DiT 架构的通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path


class DiTType(Enum):
    """DiT 架构类型"""
    FLUX_DEV = "flux_dev"
    FLUX_SCHNELL = "flux_schnell"
    SD3 = "sd3"
    SD3_MEDIUM = "sd3_medium"
    LUMINA = "lumina"
    PIXART_ALPHA = "pixart_alpha"
    HUNYUAN = "hunyuan_dit"
    UNKNOWN = "unknown"


@dataclass
class DiTConfig:
    """DiT 配置"""
    dit_type: DiTType = DiTType.FLUX_DEV
    
    # 层配置
    num_transformer_blocks: int = 19  # Flux: 19 double blocks
    num_single_blocks: int = 38       # Flux: 38 single blocks
    hidden_size: int = 3072
    num_attention_heads: int = 24
    
    # LoRA 相关
    lora_targets: Set[str] = field(default_factory=lambda: {
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
        "ff.net.0.proj", "ff.net.2"
    })


@dataclass
class DiTLayerInfo:
    """DiT 层信息"""
    name: str
    block_type: str  # "double" | "single" | "other"
    block_index: int
    layer_type: str  # "attn" | "ff" | "norm"
    component: str   # "to_q", "to_k", "to_v", etc.
    shape: tuple
    dtype: str
    
    # LoRA 相关
    is_lora: bool = False
    lora_rank: int = 0
    lora_alpha: float = 0.0


class DiTBase(ABC):
    """
    DiT 处理基类
    
    所有 DiT 适配器必须实现这些方法
    """
    
    def __init__(self, config: Optional[DiTConfig] = None):
        self.config = config or DiTConfig()
        self._loaded = False
        self._state_dict: Dict[str, Any] = {}
    
    @property
    @abstractmethod
    def dit_type(self) -> DiTType:
        """返回 DiT 类型"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """
        加载模型权重
        
        Args:
            path: 模型路径 (safetensors 或目录)
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def get_layer_info(self) -> List[DiTLayerInfo]:
        """
        获取所有层信息
        
        Returns:
            层信息列表
        """
        pass
    
    @abstractmethod
    def get_lora_targets(self) -> Set[str]:
        """
        获取可注入 LoRA 的目标层
        
        Returns:
            层名模式集合
        """
        pass
    
    @abstractmethod
    def parse_lora(self, lora_path: str) -> Dict[str, Any]:
        """
        解析 LoRA 文件
        
        Args:
            lora_path: LoRA safetensors 路径
            
        Returns:
            解析后的层信息
        """
        pass
    
    def detect_lora_rank(self, lora_weights: Dict) -> int:
        """
        检测 LoRA rank
        
        Args:
            lora_weights: LoRA 权重字典
        """
        for key, tensor in lora_weights.items():
            if "lora_A" in key or "lora_down" in key:
                return tensor.shape[0]
        return 0
    
    def is_loaded(self) -> bool:
        """是否已加载"""
        return self._loaded


def detect_dit_type(path: str) -> DiTType:
    """
    自动检测 DiT 模型类型
    
    Args:
        path: 模型路径
        
    Returns:
        检测到的 DiT 类型
    """
    path_lower = str(path).lower()

    # 从路径名猜测
    if "flux" in path_lower:
        if "schnell" in path_lower:
            return DiTType.FLUX_SCHNELL
        return DiTType.FLUX_DEV

    if "sd3" in path_lower:
        if "medium" in path_lower:
            return DiTType.SD3_MEDIUM
        return DiTType.SD3

    if "lumina" in path_lower:
        return DiTType.LUMINA

    if "pixart" in path_lower:
        return DiTType.PIXART_ALPHA

    if "hunyuan" in path_lower:
        return DiTType.HUNYUAN

    # 尝试从权重结构检测
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())

            # Flux 特征
            if any("transformer_blocks" in k and "single_transformer_blocks" in k for k in keys):
                return DiTType.FLUX_DEV

            # SD3 特征
            if any("joint_transformer_blocks" in k for k in keys):
                return DiTType.SD3

            # Lumina 特征 (single-stream transformer blocks, no single_transformer_blocks)
            if any("transformer_blocks" in k for k in keys) and not any("single_transformer_blocks" in k for k in keys):
                return DiTType.LUMINA

    except Exception:
        pass

    return DiTType.UNKNOWN


def create_adapter(dit_type: DiTType):
    """
    创建 DiT 适配器

    Args:
        dit_type: DiT 类型

    Returns:
        对应的适配器实例
    """
    if dit_type in [DiTType.FLUX_DEV, DiTType.FLUX_SCHNELL]:
        from .flux_adapter import FluxAdapter
        return FluxAdapter()

    if dit_type in [DiTType.SD3, DiTType.SD3_MEDIUM]:
        from .sd3_adapter import SD3Adapter
        return SD3Adapter()

    if dit_type == DiTType.LUMINA:
        from .lumina_adapter import LuminaAdapter
        return LuminaAdapter()

    raise ValueError(f"Unsupported DiT type: {dit_type}")
