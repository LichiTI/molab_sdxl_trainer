"""
Flux DiT 适配器

支持 Flux Dev 和 Flux Schnell 模型
"""

import logging
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .base import DiTBase, DiTConfig, DiTType, DiTLayerInfo

logger = logging.getLogger(__name__)


@dataclass
class FluxConfig(DiTConfig):
    """Flux 特定配置"""
    dit_type: DiTType = DiTType.FLUX_DEV
    
    # Flux 架构参数
    num_transformer_blocks: int = 19      # Double stream blocks
    num_single_blocks: int = 38           # Single stream blocks
    hidden_size: int = 3072
    num_attention_heads: int = 24
    mlp_ratio: float = 4.0
    
    # 文本编码器
    text_encoder: str = "t5"  # T5-XXL
    use_clip: bool = True     # 同时使用 CLIP


class FluxAdapter(DiTBase):
    """
    Flux 模型适配器
    
    支持:
    - Flux.1 Dev
    - Flux.1 Schnell
    - Flux LoRA 分析
    """
    
    # Flux LoRA 层名模式
    LORA_PATTERNS = {
        # Double stream blocks
        "transformer_blocks.{}.attn.to_q",
        "transformer_blocks.{}.attn.to_k", 
        "transformer_blocks.{}.attn.to_v",
        "transformer_blocks.{}.attn.to_out.0",
        "transformer_blocks.{}.ff.net.0.proj",
        "transformer_blocks.{}.ff.net.2",
        "transformer_blocks.{}.ff_context.net.0.proj",
        "transformer_blocks.{}.ff_context.net.2",
        
        # Single stream blocks
        "single_transformer_blocks.{}.attn.to_q",
        "single_transformer_blocks.{}.attn.to_k",
        "single_transformer_blocks.{}.attn.to_v",
        "single_transformer_blocks.{}.attn.to_out.0",
        "single_transformer_blocks.{}.proj_mlp",
        "single_transformer_blocks.{}.proj_out",
    }
    
    def __init__(self, config: Optional[FluxConfig] = None):
        super().__init__(config or FluxConfig())
        self._layers: List[DiTLayerInfo] = []
        self._lora_data: Dict[str, Any] = {}
    
    @property
    def dit_type(self) -> DiTType:
        return self.config.dit_type
    
    def load(self, path: str) -> bool:
        """加载 Flux 模型或 LoRA"""
        try:
            from safetensors import safe_open
            
            path = Path(path)
            if not path.exists():
                logger.error(f"Path not found: {path}")
                return False
            
            # 单文件
            if path.suffix == ".safetensors":
                with safe_open(str(path), framework="pt", device="cpu") as f:
                    self._state_dict = {k: f.get_tensor(k) for k in f.keys()}
            
            # 目录 (多文件模型)
            elif path.is_dir():
                safetensor_files = list(path.glob("*.safetensors"))
                for sf in safetensor_files:
                    with safe_open(str(sf), framework="pt", device="cpu") as f:
                        for k in f.keys():
                            self._state_dict[k] = f.get_tensor(k)
            
            self._loaded = True
            self._parse_layers()
            
            logger.info(f"Loaded Flux model: {len(self._state_dict)} tensors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Flux model: {e}")
            return False
    
    def _parse_layers(self):
        """解析层信息"""
        self._layers = []
        
        for key, tensor in self._state_dict.items():
            info = self._parse_layer_name(key, tensor)
            if info:
                self._layers.append(info)
    
    def _parse_layer_name(self, name: str, tensor) -> Optional[DiTLayerInfo]:
        """解析单个层名"""
        # 检测是否是 LoRA 权重
        is_lora = "lora_" in name or "lora." in name
        
        # 提取块类型和索引
        block_type = "other"
        block_index = -1
        
        if "transformer_blocks." in name:
            block_type = "double"
            parts = name.split("transformer_blocks.")[1].split(".")
            try:
                block_index = int(parts[0])
            except ValueError:
                pass
        elif "single_transformer_blocks." in name:
            block_type = "single"
            parts = name.split("single_transformer_blocks.")[1].split(".")
            try:
                block_index = int(parts[0])
            except ValueError:
                pass
        
        # 提取层类型
        if "attn" in name:
            layer_type = "attn"
        elif "ff" in name or "mlp" in name:
            layer_type = "ff"
        elif "norm" in name:
            layer_type = "norm"
        else:
            layer_type = "other"
        
        # 提取组件名
        component = name.split(".")[-1]
        if is_lora:
            # lora_A.weight -> 去掉 lora 部分
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p in ["to_q", "to_k", "to_v", "to_out", "proj", "proj_mlp", "proj_out"]:
                    component = p
                    break
        
        # LoRA rank 检测
        lora_rank = 0
        if is_lora and ("lora_A" in name or "lora_down" in name):
            lora_rank = tensor.shape[0] if len(tensor.shape) >= 1 else 0
        
        return DiTLayerInfo(
            name=name,
            block_type=block_type,
            block_index=block_index,
            layer_type=layer_type,
            component=component,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            is_lora=is_lora,
            lora_rank=lora_rank,
        )
    
    def get_layer_info(self) -> List[DiTLayerInfo]:
        """获取所有层信息"""
        return self._layers
    
    def get_lora_targets(self) -> Set[str]:
        """获取 LoRA 目标层"""
        targets = set()
        
        # Double blocks
        for i in range(self.config.num_transformer_blocks):
            for pattern in self.LORA_PATTERNS:
                if "{}" in pattern:
                    targets.add(pattern.format(i))
        
        # Single blocks
        for i in range(self.config.num_single_blocks):
            for pattern in self.LORA_PATTERNS:
                if "single_" in pattern and "{}" in pattern:
                    targets.add(pattern.format(i))
        
        return targets
    
    def parse_lora(self, lora_path: str) -> Dict[str, Any]:
        """
        解析 Flux LoRA 文件
        
        Returns:
            {
                "rank": int,
                "alpha": float,
                "layers": {...},
                "double_blocks": [...],
                "single_blocks": [...],
                "stats": {...}
            }
        """
        if not self.load(lora_path):
            return {}
        
        result = {
            "rank": 0,
            "alpha": 1.0,
            "layers": {},
            "double_blocks": [],
            "single_blocks": [],
            "stats": {
                "total_params": 0,
                "lora_params": 0,
                "double_block_count": 0,
                "single_block_count": 0,
            }
        }
        
        # 提取 LoRA 层
        double_indices = set()
        single_indices = set()
        
        for layer in self._layers:
            if layer.is_lora:
                result["layers"][layer.name] = {
                    "shape": layer.shape,
                    "block_type": layer.block_type,
                    "block_index": layer.block_index,
                    "component": layer.component,
                }
                
                if layer.lora_rank > 0:
                    result["rank"] = max(result["rank"], layer.lora_rank)
                
                if layer.block_type == "double":
                    double_indices.add(layer.block_index)
                elif layer.block_type == "single":
                    single_indices.add(layer.block_index)
                
                # 统计参数
                param_count = 1
                for dim in layer.shape:
                    param_count *= dim
                result["stats"]["lora_params"] += param_count
        
        result["double_blocks"] = sorted(double_indices)
        result["single_blocks"] = sorted(single_indices)
        result["stats"]["double_block_count"] = len(double_indices)
        result["stats"]["single_block_count"] = len(single_indices)
        result["stats"]["total_params"] = sum(
            1 for dim in t.shape for t in self._state_dict.values()
        ) if self._state_dict else 0
        
        return result
    
    def get_block_summary(self) -> Dict[str, Any]:
        """
        获取块摘要
        
        Returns:
            {
                "double_blocks": {0: {...}, 1: {...}, ...},
                "single_blocks": {0: {...}, 1: {...}, ...}
            }
        """
        summary = {
            "double_blocks": {},
            "single_blocks": {},
        }
        
        for layer in self._layers:
            if layer.block_index < 0:
                continue
            
            if layer.block_type == "double":
                if layer.block_index not in summary["double_blocks"]:
                    summary["double_blocks"][layer.block_index] = {
                        "layers": [],
                        "has_lora": False,
                    }
                summary["double_blocks"][layer.block_index]["layers"].append(layer.name)
                if layer.is_lora:
                    summary["double_blocks"][layer.block_index]["has_lora"] = True
            
            elif layer.block_type == "single":
                if layer.block_index not in summary["single_blocks"]:
                    summary["single_blocks"][layer.block_index] = {
                        "layers": [],
                        "has_lora": False,
                    }
                summary["single_blocks"][layer.block_index]["layers"].append(layer.name)
                if layer.is_lora:
                    summary["single_blocks"][layer.block_index]["has_lora"] = True
        
        return summary
