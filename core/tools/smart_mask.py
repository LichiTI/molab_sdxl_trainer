"""
Smart Mask Selector: 智能 Mask 选择器

基于 Segment Anything Model (SAM) 实现一键分割功能。

特性:
- 点击选择: 点击图片任意位置，自动分割区域
- 多点选择: 正点 (选中) + 负点 (排除)
- 边缘羽化: 可配置的边缘平滑
- 轻量模型: 使用 MobileSAM (~40MB)

使用方法:
1. 加载图片
2. 点击要选择的区域
3. 获取 mask
4. 保存为 _mask.png
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import io
import base64

logger = logging.getLogger("SmartMaskSelector")

# SAM 延迟导入
SAM_AVAILABLE = False
_sam_model = None
_sam_predictor = None


def _load_sam_model():
    """延迟加载 SAM 模型"""
    global SAM_AVAILABLE, _sam_model, _sam_predictor
    
    if _sam_predictor is not None:
        return True
    
    try:
        # 尝试 MobileSAM
        from mobile_sam import sam_model_registry, SamPredictor
        
        logger.info("[SAM] Loading MobileSAM model...")
        
        # 模型会自动下载
        model_type = "vit_t"  # tiny model
        _sam_model = sam_model_registry[model_type]()
        _sam_predictor = SamPredictor(_sam_model)
        
        SAM_AVAILABLE = True
        logger.info("[SAM] MobileSAM loaded successfully")
        return True
        
    except ImportError:
        logger.warning("[SAM] MobileSAM not installed. Run: pip install mobile-sam")
        SAM_AVAILABLE = False
        return False
    except Exception as e:
        logger.error(f"[SAM] Failed to load model: {e}")
        SAM_AVAILABLE = False
        return False


@dataclass
class PointPrompt:
    """点击提示"""
    x: int
    y: int
    label: int = 1  # 1=选中, 0=排除


@dataclass
class SegmentResult:
    """分割结果"""
    mask: np.ndarray  # H x W, bool
    score: float
    area: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class SmartMaskSelector:
    """
    智能 Mask 选择器
    
    提供基于 SAM 的图像分割功能
    """
    
    def __init__(self, feather_radius: int = 3):
        self.feather_radius = feather_radius
        self._image: Optional[np.ndarray] = None
        self._image_path: Optional[str] = None
        self._masks: List[np.ndarray] = []
        self._points: List[PointPrompt] = []
    
    def load_image(self, image_path: str) -> bool:
        """加载图片"""
        if not _load_sam_model():
            return False
        
        try:
            with Image.open(image_path) as img_src:
                image = img_src.convert("RGB")
                self._image = np.array(image)
            self._image_path = image_path
            
            # 设置 SAM 图片
            _sam_predictor.set_image(self._image)
            
            # 清空之前的 mask
            self._masks.clear()
            self._points.clear()
            
            logger.info(f"[SAM] Loaded image: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"[SAM] Failed to load image: {e}")
            return False
    
    def load_image_bytes(self, image_bytes: bytes) -> bool:
        """从字节加载图片"""
        if not _load_sam_model():
            return False
        
        try:
            with Image.open(io.BytesIO(image_bytes)) as img_src:
                image = img_src.convert("RGB")
                self._image = np.array(image)
            self._image_path = None
            
            _sam_predictor.set_image(self._image)
            self._masks.clear()
            self._points.clear()
            
            return True
        except Exception as e:
            logger.error(f"[SAM] Failed to load image bytes: {e}")
            return False
    
    def add_point(self, x: int, y: int, label: int = 1) -> Optional[SegmentResult]:
        """
        添加点击提示并获取 mask
        
        Args:
            x, y: 点击坐标
            label: 1=选中, 0=排除
        
        Returns:
            分割结果
        """
        if _sam_predictor is None or self._image is None:
            logger.error("[SAM] No image loaded")
            return None
        
        self._points.append(PointPrompt(x=x, y=y, label=label))
        
        # 准备输入
        input_points = np.array([[p.x, p.y] for p in self._points])
        input_labels = np.array([p.label for p in self._points])
        
        # 预测
        masks, scores, _ = _sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # 选择最佳 mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        # 计算统计
        area = int(np.sum(best_mask))
        y_indices, x_indices = np.where(best_mask)
        if len(y_indices) > 0:
            bbox = (
                int(x_indices.min()),
                int(y_indices.min()),
                int(x_indices.max()),
                int(y_indices.max()),
            )
        else:
            bbox = (0, 0, 0, 0)
        
        # 保存 mask
        self._masks = [best_mask]
        
        return SegmentResult(
            mask=best_mask,
            score=float(best_score),
            area=area,
            bbox=bbox,
        )
    
    def clear_points(self):
        """清空点击提示"""
        self._points.clear()
        self._masks.clear()
    
    def get_combined_mask(self) -> Optional[np.ndarray]:
        """获取合并后的 mask"""
        if not self._masks:
            return None
        
        # 合并所有 mask
        combined = np.zeros_like(self._masks[0], dtype=bool)
        for mask in self._masks:
            combined = combined | mask
        
        return combined
    
    def apply_feather(self, mask: np.ndarray) -> np.ndarray:
        """应用边缘羽化"""
        if self.feather_radius <= 0:
            return mask.astype(np.float32)
        
        try:
            from scipy.ndimage import gaussian_filter
            
            # 转为 float 并模糊
            mask_float = mask.astype(np.float32)
            feathered = gaussian_filter(mask_float, sigma=self.feather_radius)
            
            # 归一化到 0-1
            if feathered.max() > 0:
                feathered = feathered / feathered.max()
            
            return feathered
            
        except ImportError:
            logger.warning("[SAM] scipy not available, skipping feather")
            return mask.astype(np.float32)
    
    def save_mask(
        self,
        output_path: Optional[str] = None,
        feather: bool = True,
        invert: bool = False,
    ) -> Optional[str]:
        """
        保存 mask 为图片
        
        Args:
            output_path: 输出路径，默认为 原文件名_mask.png
            feather: 是否应用边缘羽化
            invert: 是否反转 mask
        
        Returns:
            保存的文件路径
        """
        mask = self.get_combined_mask()
        if mask is None:
            logger.error("[SAM] No mask to save")
            return None
        
        # 处理 mask
        if feather:
            mask_float = self.apply_feather(mask)
        else:
            mask_float = mask.astype(np.float32)
        
        if invert:
            mask_float = 1.0 - mask_float
        
        # 转为 8-bit
        mask_uint8 = (mask_float * 255).astype(np.uint8)
        
        # 生成输出路径
        if output_path is None:
            if self._image_path:
                base = Path(self._image_path)
                output_path = str(base.parent / f"{base.stem}_mask.png")
            else:
                output_path = "mask.png"
        
        # 保存
        Image.fromarray(mask_uint8).save(output_path)
        logger.info(f"[SAM] Saved mask to {output_path}")
        
        return output_path
    
    def get_mask_base64(self, feather: bool = True) -> Optional[str]:
        """获取 mask 的 Base64 编码"""
        mask = self.get_combined_mask()
        if mask is None:
            return None
        
        if feather:
            mask_float = self.apply_feather(mask)
        else:
            mask_float = mask.astype(np.float32)
        
        mask_uint8 = (mask_float * 255).astype(np.uint8)
        
        # 转为 PNG bytes
        img = Image.fromarray(mask_uint8)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def get_overlay_image(self, alpha: float = 0.5, color: Tuple[int, int, int] = (255, 0, 0)) -> Optional[np.ndarray]:
        """
        获取叠加了 mask 的预览图
        
        Args:
            alpha: 叠加透明度
            color: mask 颜色 (R, G, B)
        
        Returns:
            叠加后的 RGB 图像
        """
        if self._image is None:
            return None
        
        mask = self.get_combined_mask()
        if mask is None:
            return self._image.copy()
        
        # 创建叠加图
        overlay = self._image.copy()
        mask_3d = np.stack([mask] * 3, axis=-1)
        
        # 应用颜色
        colored_mask = np.zeros_like(overlay)
        colored_mask[:, :] = color
        
        # 混合
        overlay = np.where(
            mask_3d,
            (overlay * (1 - alpha) + colored_mask * alpha).astype(np.uint8),
            overlay,
        )
        
        return overlay


# ========== 便捷函数 ==========

def check_sam_available() -> Dict[str, Any]:
    """检查 SAM 是否可用"""
    available = _load_sam_model()
    return {
        "available": available,
        "model": "MobileSAM" if available else None,
        "install_command": "pip install mobile-sam" if not available else None,
    }


def segment_image(
    image_path: str,
    points: List[Tuple[int, int, int]],  # [(x, y, label), ...]
    feather_radius: int = 3,
) -> Optional[str]:
    """
    一键分割图片
    
    Args:
        image_path: 图片路径
        points: 点击列表 [(x, y, label), ...]
        feather_radius: 羽化半径
    
    Returns:
        生成的 mask 文件路径
    """
    selector = SmartMaskSelector(feather_radius=feather_radius)
    
    if not selector.load_image(image_path):
        return None
    
    for x, y, label in points:
        selector.add_point(x, y, label)
    
    return selector.save_mask()


def batch_segment(
    image_dir: str,
    default_points: List[Tuple[int, int, int]] = None,
    output_dir: str = None,
) -> Dict[str, str]:
    """
    批量分割目录中的图片
    
    使用默认点击位置 (如中心点)
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) if output_dir else image_dir
    
    results = {}
    
    for img_path in image_dir.glob("*.png"):
        if "_mask" in img_path.stem:
            continue
        
        # 使用中心点
        img = Image.open(img_path)
        w, h = img.size
        center = (w // 2, h // 2, 1)
        
        points = default_points or [center]
        
        selector = SmartMaskSelector()
        if selector.load_image(str(img_path)):
            for x, y, label in points:
                selector.add_point(x, y, label)
            
            mask_path = output_dir / f"{img_path.stem}_mask.png"
            selector.save_mask(str(mask_path))
            results[str(img_path)] = str(mask_path)
    
    return results
