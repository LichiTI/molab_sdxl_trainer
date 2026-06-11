"""
Masked Training

支持根据 mask 图像进行加权损失训练
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class MaskedLoss:
    """
    带 Mask 的损失计算
    
    用途:
    - 角色训练：只训练人物区域
    - 背景保护：降低背景权重
    - 局部修复：强化特定区域
    """
    
    def __init__(
        self,
        mask_weight: float = 10.0,
        background_weight: float = 1.0,
        blur_kernel_size: int = 5,
        normalize_mask: bool = True,
    ):
        """
        Args:
            mask_weight: mask 区域的权重 (白色区域)
            background_weight: 非 mask 区域的权重 (黑色区域)
            blur_kernel_size: mask 边缘模糊核大小
            normalize_mask: 是否归一化 mask 权重
        """
        self.mask_weight = mask_weight
        self.background_weight = background_weight
        self.blur_kernel_size = blur_kernel_size
        self.normalize_mask = normalize_mask
    
    def prepare_mask(
        self,
        mask: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        准备 mask 张量
        
        Args:
            mask: 原始 mask [B, 1, H, W] 或 [B, 3, H, W]
            target_size: 目标尺寸 (H, W) - 通常是 latent 尺寸
        
        Returns:
            处理后的 mask [B, 1, H, W]
        """
        # 转换为单通道
        if mask.shape[1] == 3:
            mask = mask.mean(dim=1, keepdim=True)
        
        # 归一化到 [0, 1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # 缩放到 latent 尺寸
        if mask.shape[-2:] != target_size:
            mask = F.interpolate(
                mask,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        
        # 边缘模糊 (平滑过渡)
        if self.blur_kernel_size > 1:
            padding = self.blur_kernel_size // 2
            mask = F.avg_pool2d(
                mask,
                kernel_size=self.blur_kernel_size,
                stride=1,
                padding=padding,
            )
        
        return mask
    
    def compute_weights(self, mask: torch.Tensor) -> torch.Tensor:
        """
        计算像素权重
        
        Args:
            mask: 归一化后的 mask [B, 1, H, W]
        
        Returns:
            权重张量 [B, 1, H, W]
        """
        # 权重 = background_weight + (mask_weight - background_weight) * mask
        weights = self.background_weight + (self.mask_weight - self.background_weight) * mask
        
        # 归一化 (可选)
        if self.normalize_mask:
            # 保持平均权重为 1
            weights = weights / weights.mean()
        
        return weights
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算加权 MSE 损失
        
        Args:
            pred: 预测值 [B, C, H, W]
            target: 目标值 [B, C, H, W]
            mask: mask 张量 [B, 1, H, W]，None 则使用普通 MSE
        
        Returns:
            损失标量
        """
        if mask is None:
            return F.mse_loss(pred, target)
        
        # 准备 mask
        mask = self.prepare_mask(mask, pred.shape[-2:])
        
        # 计算权重
        weights = self.compute_weights(mask)
        
        # 扩展到所有通道
        weights = weights.expand_as(pred)
        
        # 加权 MSE
        loss = (F.mse_loss(pred, target, reduction="none") * weights).mean()
        
        return loss
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(pred, target, mask)


class MaskedDatasetMixin:
    """
    数据集 Mask 支持 Mixin
    
    用法:
    ```python
    class MyDataset(MaskedDatasetMixin, BaseDataset):
        pass
    ```
    """
    
    MASK_SUFFIX = "_mask"
    MASK_EXTENSIONS = [".png", ".jpg", ".webp"]
    
    def find_mask_path(self, image_path: str) -> Optional[str]:
        """
        查找对应的 mask 文件
        
        支持格式:
        - image.png -> image_mask.png
        - image.jpg -> image_mask.jpg
        """
        path = Path(image_path)
        stem = path.stem
        parent = path.parent
        
        for ext in self.MASK_EXTENSIONS:
            mask_path = parent / f"{stem}{self.MASK_SUFFIX}{ext}"
            if mask_path.exists():
                return str(mask_path)
        
        return None
    
    def load_mask(self, mask_path: str) -> Optional[torch.Tensor]:
        """加载 mask 图像"""
        try:
            from PIL import Image
            import torchvision.transforms as T
            
            mask = Image.open(mask_path).convert("L")
            mask_tensor = T.ToTensor()(mask)
            
            return mask_tensor.unsqueeze(0)  # [1, 1, H, W]
            
        except Exception as e:
            logger.warning(f"Failed to load mask {mask_path}: {e}")
            return None


def create_masked_loss(
    mask_weight: float = 10.0,
    background_weight: float = 1.0,
) -> MaskedLoss:
    """创建 Masked Loss 实例"""
    return MaskedLoss(
        mask_weight=mask_weight,
        background_weight=background_weight,
    )
