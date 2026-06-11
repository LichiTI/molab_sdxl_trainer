"""
分层 Caption Dropout

支持对不同类型的 Tag 应用不同的 Dropout 策略
"""

import random
import re
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TagCategory(Enum):
    """Tag 类别"""
    TRIGGER = "trigger"       # 触发词 (如 "1girl", 角色名)
    STYLE = "style"           # 风格词 (如 "anime style", "oil painting")
    QUALITY = "quality"       # 质量词 (如 "masterpiece", "best quality")
    CONTENT = "content"       # 内容词 (如 "looking at viewer", "sitting")
    MODIFIER = "modifier"     # 修饰词 (如 "long hair", "blue eyes")
    UNKNOWN = "unknown"


@dataclass
class DropoutConfig:
    """Dropout 配置"""
    # 全局设置
    enabled: bool = True
    global_dropout_rate: float = 0.1
    shuffle_tags: bool = True
    
    # 分层 Dropout 率
    trigger_dropout: float = 0.0      # 触发词几乎不 dropout
    style_dropout: float = 0.05       # 风格词低 dropout
    quality_dropout: float = 0.3      # 质量词高 dropout
    content_dropout: float = 0.15     # 内容词中等 dropout
    modifier_dropout: float = 0.2     # 修饰词中等 dropout
    
    # 锁定词 (永不 dropout)
    locked_tags: Set[str] = field(default_factory=set)
    
    # 保留词数
    keep_tokens: int = 0  # 前 N 个词不 dropout


@dataclass
class TagInfo:
    """Tag 信息"""
    text: str
    category: TagCategory
    is_locked: bool = False
    dropout_rate: float = 0.1


class SmartCaptionProcessor:
    """
    智能 Caption 处理器
    
    功能:
    - 分类 Tags
    - 分层 Dropout
    - 智能 Shuffle
    """
    
    # 预定义关键词分类
    TRIGGER_PATTERNS = [
        r"^\d+girl$", r"^\d+boy$", r"^\d+other$",
        r"^solo$", r"^duo$", r"^trio$",
    ]
    
    STYLE_KEYWORDS = {
        "anime", "manga", "realistic", "photorealistic", "oil painting",
        "watercolor", "sketch", "line art", "cel shading", "3d render",
        "pixel art", "digital art", "illustration", "concept art",
    }
    
    QUALITY_KEYWORDS = {
        "masterpiece", "best quality", "high quality", "ultra detailed",
        "extremely detailed", "highres", "absurdres", "4k", "8k",
        "hd", "uhd", "professional", "award winning",
    }
    
    def __init__(self, config: Optional[DropoutConfig] = None):
        self.config = config or DropoutConfig()
        self._trigger_patterns = [re.compile(p) for p in self.TRIGGER_PATTERNS]
        self._custom_triggers: Set[str] = set()
    
    def add_custom_triggers(self, triggers: List[str]):
        """添加自定义触发词"""
        self._custom_triggers.update(t.lower().strip() for t in triggers)
    
    def categorize_tag(self, tag: str) -> TagCategory:
        """分类单个 Tag"""
        tag_lower = tag.lower().strip()
        
        # 检查锁定词
        if tag_lower in self.config.locked_tags:
            return TagCategory.TRIGGER  # 锁定词视为触发词
        
        # 检查自定义触发词
        if tag_lower in self._custom_triggers:
            return TagCategory.TRIGGER
        
        # 检查触发词模式
        for pattern in self._trigger_patterns:
            if pattern.match(tag_lower):
                return TagCategory.TRIGGER
        
        # 检查风格词
        if any(style in tag_lower for style in self.STYLE_KEYWORDS):
            return TagCategory.STYLE
        
        # 检查质量词
        if any(quality in tag_lower for quality in self.QUALITY_KEYWORDS):
            return TagCategory.QUALITY
        
        # 检查内容词 (动作相关)
        action_indicators = ["ing ", " at ", "sitting", "standing", "looking", "holding"]
        if any(ind in tag_lower for ind in action_indicators):
            return TagCategory.CONTENT
        
        # 默认为修饰词
        return TagCategory.MODIFIER
    
    def get_dropout_rate(self, category: TagCategory) -> float:
        """获取分类的 Dropout 率"""
        rates = {
            TagCategory.TRIGGER: self.config.trigger_dropout,
            TagCategory.STYLE: self.config.style_dropout,
            TagCategory.QUALITY: self.config.quality_dropout,
            TagCategory.CONTENT: self.config.content_dropout,
            TagCategory.MODIFIER: self.config.modifier_dropout,
            TagCategory.UNKNOWN: self.config.global_dropout_rate,
        }
        return rates.get(category, self.config.global_dropout_rate)
    
    def process(self, caption: str, separator: str = ", ") -> str:
        """
        处理 Caption
        
        Args:
            caption: 原始 caption
            separator: 分隔符
            
        Returns:
            处理后的 caption
        """
        if not self.config.enabled:
            return caption
        
        # 分割 tags
        tags = [t.strip() for t in caption.split(separator) if t.strip()]
        
        if not tags:
            return caption
        
        # 保留前 N 个
        keep_tokens = self.config.keep_tokens
        kept_tags = tags[:keep_tokens]
        process_tags = tags[keep_tokens:]
        
        # 分类和处理
        result_tags = list(kept_tags)  # 保留词直接加入
        
        for tag in process_tags:
            category = self.categorize_tag(tag)
            dropout_rate = self.get_dropout_rate(category)
            
            # 锁定词永不 dropout
            if tag.lower().strip() in self.config.locked_tags:
                result_tags.append(tag)
                continue
            
            # 随机决定是否 dropout
            if random.random() > dropout_rate:
                result_tags.append(tag)
        
        # Shuffle (保留词不参与)
        if self.config.shuffle_tags and len(result_tags) > keep_tokens:
            shuffleable = result_tags[keep_tokens:]
            random.shuffle(shuffleable)
            result_tags = result_tags[:keep_tokens] + shuffleable
        
        return separator.join(result_tags)
    
    def analyze_caption(self, caption: str, separator: str = ", ") -> Dict[str, List[str]]:
        """
        分析 Caption 结构
        
        Returns:
            按类别分组的 tags
        """
        tags = [t.strip() for t in caption.split(separator) if t.strip()]
        
        result = {cat.value: [] for cat in TagCategory}
        
        for tag in tags:
            category = self.categorize_tag(tag)
            result[category.value].append(tag)
        
        return result
    
    def get_statistics(self, captions: List[str]) -> Dict[str, Any]:
        """
        获取多个 Caption 的统计信息
        """
        total_tags = 0
        category_counts = {cat.value: 0 for cat in TagCategory}
        
        for caption in captions:
            analysis = self.analyze_caption(caption)
            for cat, tags in analysis.items():
                category_counts[cat] += len(tags)
                total_tags += len(tags)
        
        return {
            "total_captions": len(captions),
            "total_tags": total_tags,
            "category_counts": category_counts,
            "category_ratios": {
                cat: count / total_tags if total_tags > 0 else 0
                for cat, count in category_counts.items()
            },
        }


# ========== 便捷函数 ==========

def create_smart_processor(
    locked_tags: List[str] = None,
    custom_triggers: List[str] = None,
    keep_tokens: int = 0,
) -> SmartCaptionProcessor:
    """创建智能处理器"""
    config = DropoutConfig(
        locked_tags=set(locked_tags or []),
        keep_tokens=keep_tokens,
    )
    processor = SmartCaptionProcessor(config)
    
    if custom_triggers:
        processor.add_custom_triggers(custom_triggers)
    
    return processor


def apply_smart_dropout(
    caption: str,
    locked_tags: List[str] = None,
    dropout_rate: float = 0.1,
) -> str:
    """快速应用智能 Dropout"""
    config = DropoutConfig(
        global_dropout_rate=dropout_rate,
        locked_tags=set(locked_tags or []),
    )
    processor = SmartCaptionProcessor(config)
    return processor.process(caption)
