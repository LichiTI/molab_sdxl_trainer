"""
DiT LoRA 分析器

提供统一的 DiT LoRA 分析接口
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .base import DiTType, DiTLayerInfo, detect_dit_type, create_adapter

logger = logging.getLogger("DiTAnalyzer")


@dataclass
class DiTLoRAAnalysis:
    """DiT LoRA 分析结果"""
    # 基本信息
    path: str = ""
    dit_type: DiTType = DiTType.UNKNOWN
    rank: int = 0
    alpha: float = 1.0
    
    # 块统计
    num_double_blocks: int = 0
    num_single_blocks: int = 0
    double_block_indices: List[int] = field(default_factory=list)
    single_block_indices: List[int] = field(default_factory=list)
    
    # 层统计
    total_layers: int = 0
    attn_layers: int = 0
    ff_layers: int = 0
    
    # 参数统计
    total_params: int = 0
    trainable_params: int = 0
    
    # 健康度指标
    coverage_ratio: float = 0.0  # LoRA 覆盖率
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "dit_type": self.dit_type.value,
            "rank": self.rank,
            "alpha": self.alpha,
            "num_double_blocks": self.num_double_blocks,
            "num_single_blocks": self.num_single_blocks,
            "double_block_indices": self.double_block_indices,
            "single_block_indices": self.single_block_indices,
            "total_layers": self.total_layers,
            "attn_layers": self.attn_layers,
            "ff_layers": self.ff_layers,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "coverage_ratio": self.coverage_ratio,
        }


class DiTAnalyzer:
    """
    DiT LoRA 分析器
    
    用法:
        analyzer = DiTAnalyzer()
        analyzer = DiTAnalyzer()
        # result = analyzer.analyze("path/to/flux_lora.safetensors")
    """
    
    def __init__(self):
        self._cache: Dict[str, DiTLoRAAnalysis] = {}
    
    def analyze(self, path: str, use_cache: bool = True) -> DiTLoRAAnalysis:
        """
        分析 DiT LoRA 文件
        
        Args:
            path: LoRA 文件路径
            use_cache: 是否使用缓存
            
        Returns:
            分析结果
        """
        path = str(Path(path).resolve())
        
        # 检查缓存
        if use_cache and path in self._cache:
            return self._cache[path]
        
        # 检测类型
        dit_type = detect_dit_type(path)
        
        if dit_type == DiTType.UNKNOWN:
            logger.warning(f"Unknown DiT type for: {path}")
            return DiTLoRAAnalysis(path=path, dit_type=dit_type)
        
        try:
            # 创建适配器
            adapter = create_adapter(dit_type)
            
            # 解析 LoRA
            try:
                lora_data = adapter.parse_lora(path)
            except Exception as e:
                logger.error(f"Failed to parse LoRA data: {e}")
                lora_data = None
            
            if not lora_data:
                return DiTLoRAAnalysis(path=path, dit_type=dit_type)
            
            # 构建分析结果
            layers = adapter.get_layer_info()
            
            result = DiTLoRAAnalysis(
                path=path,
                dit_type=dit_type,
                rank=lora_data.get("rank", 0),
                alpha=lora_data.get("alpha", 1.0),
                num_double_blocks=len(lora_data.get("double_blocks", [])),
                num_single_blocks=len(lora_data.get("single_blocks", [])),
                double_block_indices=lora_data.get("double_blocks", []),
                single_block_indices=lora_data.get("single_blocks", []),
                total_layers=len([l for l in layers if l.is_lora]),
                attn_layers=len([l for l in layers if l.is_lora and l.layer_type == "attn"]),
                ff_layers=len([l for l in layers if l.is_lora and l.layer_type == "ff"]),
                trainable_params=lora_data.get("stats", {}).get("lora_params", 0),
            )
            
            # 计算覆盖率
            expected_blocks = adapter.config.num_transformer_blocks + adapter.config.num_single_blocks
            actual_blocks = result.num_double_blocks + result.num_single_blocks
            result.coverage_ratio = actual_blocks / max(expected_blocks, 1)
            
            # 缓存结果
            self._cache[path] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze DiT LoRA: {e}")
            return DiTLoRAAnalysis(path=path, dit_type=dit_type)
    
    def compare(self, path1: str, path2: str) -> Dict[str, Any]:
        """
        比较两个 DiT LoRA
        
        Returns:
            比较结果
        """
        analysis1 = self.analyze(path1)
        analysis2 = self.analyze(path2)
        
        return {
            "lora1": analysis1.to_dict(),
            "lora2": analysis2.to_dict(),
            "comparison": {
                "same_type": analysis1.dit_type == analysis2.dit_type,
                "rank_diff": analysis2.rank - analysis1.rank,
                "coverage_diff": analysis2.coverage_ratio - analysis1.coverage_ratio,
                "param_diff": analysis2.trainable_params - analysis1.trainable_params,
                "double_block_overlap": len(
                    set(analysis1.double_block_indices) & set(analysis2.double_block_indices)
                ),
                "single_block_overlap": len(
                    set(analysis1.single_block_indices) & set(analysis2.single_block_indices)
                ),
            }
        }
    
    def get_layer_weights(self, path: str, layer_name: str) -> Optional[Any]:
        """
        获取指定层的权重
        
        用于详细分析或 SVD 计算
        """
        dit_type = detect_dit_type(path)
        if dit_type == DiTType.UNKNOWN:
            return None
        
        try:
            adapter = create_adapter(dit_type)
            if adapter.load(path):
                return adapter._state_dict.get(layer_name)
        except Exception as e:
            logger.error(f"Failed to get layer weights: {e}")
        
        return None
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()


# ========== 便捷函数 ==========

def analyze_dit_lora(path: str) -> DiTLoRAAnalysis:
    """分析 DiT LoRA (便捷函数)"""
    analyzer = DiTAnalyzer()
    return analyzer.analyze(path)


def is_dit_lora(path: str) -> bool:
    """检测是否是 DiT LoRA"""
    dit_type = detect_dit_type(path)
    return dit_type != DiTType.UNKNOWN


def get_dit_type(path: str) -> str:
    """获取 DiT 类型名称"""
    return detect_dit_type(path).value
