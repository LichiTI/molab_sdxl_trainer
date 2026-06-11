"""
SD3 MMDiT 适配器

支持 SD3 / SD3.5 Medium 架构
"""

import torch
import logging
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass

from .base import DiTBase, DiTConfig, DiTLayerInfo, DiTType

logger = logging.getLogger("SD3Adapter")


@dataclass
class SD3Config(DiTConfig):
    """SD3 配置"""
    dit_type: DiTType = DiTType.SD3
    
    # SD3 MMDiT 配置
    num_joint_blocks: int = 24      # SD3: 24 joint transformer blocks
    hidden_size: int = 1536         # SD3 Medium: 1536
    num_attention_heads: int = 24
    
    # 三路文本编码器
    text_encoder_dims: tuple = (768, 1280, 4096)  # CLIP-L, CLIP-G, T5-XXL
    
    model_file_patterns: tuple = ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors")
    
    # LoRA 目标
    lora_targets: Optional[Set[str]] = None
    
    def __post_init__(self):
        if self.lora_targets is None:
            self.lora_targets = {
                "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
                "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
                "ff.net.0.proj", "ff.net.2",
            }


class SD3Adapter(DiTBase):
    """
    SD3 MMDiT 适配器
    
    SD3 使用 MMDiT (Multimodal Diffusion Transformer)：
    - Joint attention blocks 处理图像和文本
    - 三路文本编码器 (CLIP-L, CLIP-G, T5-XXL)
    """
    
    # SD3 层名模式
    JOINT_BLOCK_PATTERN = "joint_transformer_blocks"
    CONTEXT_BLOCK_PATTERN = "context_embedder"
    
    # 层类型映射
    LAYER_TYPE_MAP = {
        "to_q": "attn",
        "to_k": "attn", 
        "to_v": "attn",
        "to_out": "attn",
        "add_q_proj": "attn",
        "add_k_proj": "attn",
        "add_v_proj": "attn",
        "add_out_proj": "attn",
        "net.0.proj": "ff",
        "net.2": "ff",
        "norm": "norm",
    }
    
    def __init__(self, config: Optional[SD3Config] = None):
        super().__init__(config or SD3Config())
        self._layers: List[DiTLayerInfo] = []
    
    @property
    def dit_type(self) -> DiTType:
        return DiTType.SD3
    
    def load(self, path: str) -> bool:
        """加载 SD3 模型"""
        try:
            from safetensors import safe_open
            
            path = Path(path)
            
            if path.is_dir():
                # Diffusers 格式
                transformer_path = path / "transformer" / self.config.model_file_patterns[0]
                # 尝试其他模式
                if not transformer_path.exists():
                    for pattern in self.config.model_file_patterns[1:]:
                        p = path / "transformer" / pattern
                        if p.exists():
                            transformer_path = p
                            break
            else:
                transformer_path = path

            if not transformer_path.exists():
                logger.error(f"[SD3Adapter] Transformer model file not found: {transformer_path}")
                return False
            
            with safe_open(str(transformer_path), framework="pt", device="cpu") as f:
                self._state_dict = {k: f.get_tensor(k) for k in f.keys()}
            
            self._parse_layers()
            self._loaded = True
            
            logger.info(f"[SD3Adapter] Loaded {len(self._layers)} layers from {path}")
            return True
            
        except Exception as e:
            logger.error(f"[SD3Adapter] Load failed: {e}")
            return False
    
    def _parse_layers(self):
        """解析层结构"""
        self._layers = []
        
        for key, tensor in self._state_dict.items():
            layer_info = self._parse_layer_key(key, tensor)
            if layer_info:
                self._layers.append(layer_info)
    
    def _parse_layer_key(self, key: str, tensor: torch.Tensor) -> Optional[DiTLayerInfo]:
        """解析层键名"""
        # 跳过非权重
        if not key.endswith(".weight"):
            return None
        
        # Joint transformer blocks
        if self.JOINT_BLOCK_PATTERN in key:
            parts = key.split(".")
            
            # 提取 block index
            block_idx = -1
            for i, part in enumerate(parts):
                if part == "joint_transformer_blocks" and i + 1 < len(parts):
                    try:
                        block_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                    break
            
            # 确定层类型
            layer_type = "other"
            component = ""
            for pattern, lt in self.LAYER_TYPE_MAP.items():
                if pattern in key:
                    layer_type = lt
                    component = pattern
                    break
            
            return DiTLayerInfo(
                name=key,
                block_type="joint",
                block_index=block_idx,
                layer_type=layer_type,
                component=component,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
            )
        
        return None
    
    def get_layer_info(self) -> List[DiTLayerInfo]:
        """获取所有层信息"""
        return self._layers
    
    def get_lora_targets(self) -> Set[str]:
        """获取 LoRA 目标层"""
        return self.config.lora_targets
    
    def parse_lora(self, lora_path: str) -> Dict[str, Any]:
        """解析 SD3 LoRA"""
        try:
            from safetensors import safe_open
            
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                lora_weights = {k: f.get_tensor(k) for k in f.keys()}
            
            # 分析结构
            layers = {}
            rank = self.detect_lora_rank(lora_weights)
            
            for key in lora_weights.keys():
                if "lora_" not in key.lower():
                    continue
                
                # 提取基础层名
                base_name = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
                base_name = base_name.replace(".lora_down.weight", "").replace(".lora_up.weight", "")
                
                if base_name not in layers:
                    layers[base_name] = {
                        "rank": rank,
                        "has_A": False,
                        "has_B": False,
                    }
                
                if "lora_A" in key or "lora_down" in key:
                    layers[base_name]["has_A"] = True
                if "lora_B" in key or "lora_up" in key:
                    layers[base_name]["has_B"] = True
            
            return {
                "type": "sd3_lora",
                "rank": rank,
                "layers": layers,
                "total_params": sum(t.numel() for t in lora_weights.values()),
            }
            
        except Exception as e:
            logger.error(f"[SD3Adapter] Parse LoRA failed: {e}")
            return {}
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练推荐配置"""
        return {
            "recommended_rank": 16,
            "recommended_alpha": 16,
            "learning_rate": 1e-4,
            "text_encoder_lr": 1e-5,
            "train_text_encoder": True,
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "notes": "SD3 建议使用较低 rank (8-16)，训练 T5 需要更多显存",
        }
