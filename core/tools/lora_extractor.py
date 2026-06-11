"""
LoRA 提取器

从微调模型中提取 LoRA
"""

import torch
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractConfig:
    """提取配置"""
    rank: int = 32
    clamp_quantile: float = 0.99
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    
    # 目标层
    include_patterns: List[str] = None
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        if self.include_patterns is None:
            self.include_patterns = [
                "attn", "to_q", "to_k", "to_v", "to_out",
                "ff", "proj", "linear",
            ]
        if self.exclude_patterns is None:
            self.exclude_patterns = ["norm", "bias", "embedding"]


class LoRAExtractor:
    """
    从微调模型中提取 LoRA
    
    算法:
    1. 计算权重差异: ΔW = W_finetune - W_base
    2. SVD 分解: ΔW ≈ U @ S @ V^T
    3. 低秩近似: A = U[:, :r] @ diag(sqrt(S[:r])), B = diag(sqrt(S[:r])) @ V[:r, :]
    """
    
    def __init__(self, config: Optional[ExtractConfig] = None):
        self.config = config or ExtractConfig()
    
    def extract(
        self,
        base_model_path: str,
        finetuned_model_path: str,
        output_path: str,
    ) -> Dict[str, any]:
        """
        提取 LoRA
        
        Args:
            base_model_path: 基础模型路径
            finetuned_model_path: 微调模型路径
            output_path: 输出 LoRA 路径
        
        Returns:
            提取统计信息
        """
        logger.info(f"[LoRAExtractor] Loading base model from {base_model_path}")
        base_weights = self._load_weights(base_model_path)
        
        logger.info(f"[LoRAExtractor] Loading finetuned model from {finetuned_model_path}")
        finetuned_weights = self._load_weights(finetuned_model_path)
        
        # 提取差异
        lora_weights = {}
        stats = {
            "layers_extracted": 0,
            "layers_skipped": 0,
            "total_params": 0,
        }
        
        for key in base_weights.keys():
            if key not in finetuned_weights:
                continue
            
            # 检查是否应该提取
            if not self._should_extract(key):
                stats["layers_skipped"] += 1
                continue
            
            base_w = base_weights[key].to(self.config.device, dtype=self.config.dtype)
            fine_w = finetuned_weights[key].to(self.config.device, dtype=self.config.dtype)
            
            # 跳过相同的权重
            if torch.allclose(base_w, fine_w, atol=1e-6):
                stats["layers_skipped"] += 1
                continue
            
            # 计算差异并提取 LoRA
            lora_a, lora_b = self._extract_layer(base_w, fine_w)
            
            if lora_a is not None and lora_b is not None:
                base_name = self._normalize_key(key)
                lora_weights[f"{base_name}.lora_down.weight"] = lora_a.cpu()
                lora_weights[f"{base_name}.lora_up.weight"] = lora_b.cpu()
                lora_weights[f"{base_name}.alpha"] = torch.tensor(float(self.config.rank))
                
                stats["layers_extracted"] += 1
                stats["total_params"] += lora_a.numel() + lora_b.numel()
        
        # 保存
        if lora_weights:
            from safetensors.torch import save_file
            save_file(lora_weights, output_path)
            logger.info(f"[LoRAExtractor] Saved LoRA to {output_path}")
        else:
            logger.warning("[LoRAExtractor] No layers extracted!")
        
        return stats
    
    def _load_weights(self, path: str) -> Dict[str, torch.Tensor]:
        """加载模型权重"""
        path = Path(path)
        
        if path.suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(path), framework="pt", device="cpu") as f:
                return {k: f.get_tensor(k) for k in f.keys()}
        
        elif path.suffix in [".ckpt", ".pt", ".pth"]:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        
        elif path.is_dir():
            # Diffusers 格式
            weights = {}
            for sf_path in path.rglob("*.safetensors"):
                from safetensors import safe_open
                with safe_open(str(sf_path), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        weights[k] = f.get_tensor(k)
            return weights
        
        else:
            raise ValueError(f"Unsupported model format: {path}")
    
    def _should_extract(self, key: str) -> bool:
        """判断是否应该提取"""
        key_lower = key.lower()
        
        # 排除模式
        for pattern in self.config.exclude_patterns:
            if pattern in key_lower:
                return False
        
        # 包含模式
        for pattern in self.config.include_patterns:
            if pattern in key_lower:
                return True
        
        return False
    
    def _normalize_key(self, key: str) -> str:
        """规范化键名"""
        # 移除 .weight 后缀
        if key.endswith(".weight"):
            key = key[:-7]
        # 替换点号
        return key.replace(".", "_")
    
    def _extract_layer(
        self,
        base_w: torch.Tensor,
        fine_w: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        从单层提取 LoRA
        
        Returns:
            (lora_A, lora_B) 或 (None, None)
        """
        # 计算差异
        delta_w = fine_w - base_w
        
        # 只处理 2D 权重
        if delta_w.ndim != 2:
            return None, None
        
        # Clamp 异常值
        if self.config.clamp_quantile < 1.0:
            threshold = torch.quantile(delta_w.abs().flatten(), self.config.clamp_quantile)
            delta_w = delta_w.clamp(-threshold, threshold)
        
        # SVD 分解
        try:
            U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        except Exception as e:
            logger.warning(f"SVD failed: {e}")
            return None, None
        
        # 低秩近似
        rank = min(self.config.rank, S.shape[0])
        
        # A = V[:r, :]^T @ diag(sqrt(S[:r]))
        # B = diag(sqrt(S[:r])) @ U[:, :r]^T
        sqrt_s = torch.sqrt(S[:rank])
        
        lora_a = (Vh[:rank, :].T * sqrt_s).T  # [rank, in_features]
        lora_b = (U[:, :rank] * sqrt_s)       # [out_features, rank]
        
        # 转换为标准 LoRA 格式
        # lora_down: [rank, in_features]
        # lora_up: [out_features, rank]
        return lora_a, lora_b
    
    def estimate_quality(
        self,
        base_w: torch.Tensor,
        fine_w: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ) -> float:
        """估算重建质量"""
        delta_w = fine_w - base_w
        reconstructed = lora_b @ lora_a
        
        error = (delta_w - reconstructed).norm()
        original = delta_w.norm()
        
        return 1.0 - (error / original).item()


def extract_lora(
    base_model: str,
    finetuned_model: str,
    output_path: str,
    rank: int = 32,
) -> Dict[str, any]:
    """
    便捷函数：提取 LoRA
    
    Args:
        base_model: 基础模型路径
        finetuned_model: 微调模型路径
        output_path: 输出路径
        rank: LoRA rank
    
    Returns:
        提取统计
    """
    config = ExtractConfig(rank=rank)
    extractor = LoRAExtractor(config)
    return extractor.extract(base_model, finetuned_model, output_path)
