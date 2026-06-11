"""
Block Weight Training

分层训练：冻结/解冻特定 UNet 层
"""

import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class BlockWeightPreset(Enum):
    """Block Weight 预设"""
    FULL = "full"                    # 全部训练
    CHARACTER = "character"          # 角色预设 (构图层)
    STYLE = "style"                  # 风格预设 (细节层)
    LIGHT_STYLE = "light_style"      # 轻量风格
    MIDJOURNEY = "midjourney"        # MidJourney 风格
    CUSTOM = "custom"


@dataclass
class BlockWeightConfig:
    """Block Weight 配置"""
    preset: BlockWeightPreset = BlockWeightPreset.FULL
    
    # UNet 层权重 (0 = 冻结, 1 = 完全训练)
    # IN 块 (0-11): 控制构图和布局
    in_weights: List[float] = field(default_factory=lambda: [1.0] * 12)
    
    # MID 块: 控制整体语义
    mid_weight: float = 1.0
    
    # OUT 块 (0-11): 控制细节和风格
    out_weights: List[float] = field(default_factory=lambda: [1.0] * 12)
    
    # Text Encoder
    te_weight: float = 1.0
    te2_weight: float = 1.0  # SDXL
    zero_threshold: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset.value,
            "in_weights": self.in_weights,
            "mid_weight": self.mid_weight,
            "out_weights": self.out_weights,
            "te_weight": self.te_weight,
            "te2_weight": self.te2_weight,
            "zero_threshold": self.zero_threshold,
        }
    
    @classmethod
    def from_preset(cls, preset: BlockWeightPreset) -> "BlockWeightConfig":
        """从预设创建配置"""
        if preset == BlockWeightPreset.FULL:
            return cls(preset=preset)
        
        elif preset == BlockWeightPreset.CHARACTER:
            # 角色预设：主要训练 IN 层 (构图) + MID
            return cls(
                preset=preset,
                in_weights=[1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2],
                mid_weight=1.0,
                out_weights=[0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0],
                te_weight=0.5,
            )
        
        elif preset == BlockWeightPreset.STYLE:
            # 风格预设：主要训练 OUT 层 (细节)
            return cls(
                preset=preset,
                in_weights=[0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0],
                mid_weight=1.0,
                out_weights=[1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2],
                te_weight=1.0,
            )
        
        elif preset == BlockWeightPreset.LIGHT_STYLE:
            # 轻量风格：只训练最外层
            return cls(
                preset=preset,
                in_weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
                mid_weight=0.5,
                out_weights=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                te_weight=0.0,
            )
        
        elif preset == BlockWeightPreset.MIDJOURNEY:
            # MidJourney 风格：平衡
            return cls(
                preset=preset,
                in_weights=[0.5, 0.5, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.5, 0.5],
                mid_weight=1.0,
                out_weights=[0.5, 0.5, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.5, 0.5],
                te_weight=0.7,
            )
        
        return cls(preset=preset)

    @classmethod
    def from_settings(
        cls,
        preset: str = "",
        in_weights: Any = None,
        mid_weight: Any = None,
        out_weights: Any = None,
        te_weight: Any = 1.0,
        te2_weight: Any = 1.0,
        zero_threshold: Any = 0.0,
    ) -> "BlockWeightConfig":
        preset_ids = {item.value for item in BlockWeightPreset}
        preset_enum = BlockWeightPreset(preset) if preset in preset_ids else BlockWeightPreset.FULL
        base = cls.from_preset(preset_enum)

        return cls(
            preset=preset_enum,
            in_weights=cls._parse_weight_list(in_weights, base.in_weights, expected_len=12),
            mid_weight=cls._parse_scalar(mid_weight, base.mid_weight),
            out_weights=cls._parse_weight_list(out_weights, base.out_weights, expected_len=12),
            te_weight=cls._parse_scalar(te_weight, base.te_weight),
            te2_weight=cls._parse_scalar(te2_weight, base.te2_weight),
            zero_threshold=max(cls._parse_scalar(zero_threshold, 0.0), 0.0),
        )

    @staticmethod
    def _parse_scalar(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        if isinstance(value, str) and not value.strip():
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning(f"[BlockWeight] Invalid scalar value {value!r}, fallback to {default}")
            return float(default)

    @classmethod
    def _parse_weight_list(
        cls,
        value: Any,
        default: List[float],
        expected_len: int,
    ) -> List[float]:
        if value is None:
            return list(default)

        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return list(default)
            parts = [part.strip() for part in re.split(r"[,\s|/]+", raw) if part.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            logger.warning(f"[BlockWeight] Invalid weight list {value!r}, fallback to default")
            return list(default)

        parsed: List[float] = []
        for part in parts:
            try:
                parsed.append(float(part))
            except (TypeError, ValueError):
                logger.warning(f"[BlockWeight] Invalid weight item {part!r}, using default slice")
                return list(default)

        if not parsed:
            return list(default)

        if len(parsed) < expected_len:
            parsed.extend(default[len(parsed):expected_len])
        elif len(parsed) > expected_len:
            parsed = parsed[:expected_len]

        return parsed


class BlockWeightManager:
    """
    Block Weight 管理器
    
    应用分层训练权重
    """
    
    # SDXL UNet 层名映射
    SDXL_LAYER_MAP = {
        "down_blocks": {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
        },
        "mid_block": ["mid"],
        "up_blocks": {
            0: [6, 7],
            1: [8, 9],
            2: [10, 11],
        },
    }
    
    def __init__(self, config: Optional[BlockWeightConfig] = None):
        self.config = config or BlockWeightConfig()
        self._frozen_layers: Set[str] = set()
        self._layer_weights: Dict[str, float] = {}
    
    def apply_to_lora_injector(self, lora_injector) -> Dict[str, float]:
        """
        应用权重到 LoRA 注入器
        
        Returns:
            层名 -> 权重映射
        """
        layer_weights = {}
        self._frozen_layers.clear()

        layer_names = lora_injector.get_layer_names()
        grouped_layers = self._build_grouped_layers(layer_names)
        grouped_weights = self._build_group_weight_map(grouped_layers)

        for layer_name in layer_names:
            group_key = self._normalize_group_key(layer_name)
            weight = grouped_weights.get(group_key, self._get_weight_for_layer(layer_name))
            weight = self._apply_zero_threshold(weight)
            layer_weights[layer_name] = weight

            if weight <= 0:
                self._frozen_layers.add(layer_name)
                lora_injector.freeze_layer(layer_name)
            else:
                lora_injector.set_layer_lr_scale(layer_name, weight)
        
        self._layer_weights = layer_weights
        
        logger.info(f"[BlockWeight] Applied weights: {len(self._frozen_layers)} layers frozen")
        return layer_weights

    def _apply_zero_threshold(self, weight: float) -> float:
        threshold = max(float(getattr(self.config, "zero_threshold", 0.0) or 0.0), 0.0)
        weight = float(weight)
        if threshold > 0.0 and weight <= threshold:
            return 0.0
        return weight

    def _build_grouped_layers(self, layer_names: List[str]) -> Dict[str, List[str]]:
        grouped = {
            "down": [],
            "mid": [],
            "up": [],
            "te1": [],
            "te2": [],
        }
        seen = {key: set() for key in grouped}

        for layer_name in layer_names:
            category = self._categorize_layer(layer_name)
            if category not in grouped:
                continue

            group_key = self._normalize_group_key(layer_name)
            if group_key not in seen[category]:
                grouped[category].append(group_key)
                seen[category].add(group_key)

        return grouped

    def _build_group_weight_map(self, grouped_layers: Dict[str, List[str]]) -> Dict[str, float]:
        weight_map: Dict[str, float] = {}
        weight_map.update(self._spread_bucket_weights(grouped_layers.get("down", []), self.config.in_weights))
        weight_map.update(self._spread_bucket_weights(grouped_layers.get("up", []), self.config.out_weights))

        for group_key in grouped_layers.get("mid", []):
            weight_map[group_key] = self.config.mid_weight
        for group_key in grouped_layers.get("te1", []):
            weight_map[group_key] = self.config.te_weight
        for group_key in grouped_layers.get("te2", []):
            weight_map[group_key] = self.config.te2_weight

        return weight_map

    def _spread_bucket_weights(self, group_keys: List[str], bucket_weights: List[float]) -> Dict[str, float]:
        if not group_keys:
            return {}

        if len(group_keys) == 1:
            return {group_keys[0]: float(bucket_weights[0])}

        total_groups = len(group_keys)
        total_buckets = max(len(bucket_weights), 1)
        result: Dict[str, float] = {}

        for index, group_key in enumerate(group_keys):
            bucket_index = round(index * (total_buckets - 1) / max(total_groups - 1, 1))
            bucket_index = max(0, min(bucket_index, total_buckets - 1))
            result[group_key] = float(bucket_weights[bucket_index])

        return result

    def _categorize_layer(self, layer_name: str) -> str:
        name_lower = layer_name.lower()

        if name_lower.startswith("te2.") or ".te2." in name_lower or "text_encoder_2" in name_lower:
            return "te2"
        if name_lower.startswith("te1.") or ".te1." in name_lower or name_lower.startswith("te.") or ".te." in name_lower:
            return "te1"
        if "text_encoder" in name_lower:
            return "te1"
        if "mid_block" in name_lower or "middle_block" in name_lower:
            return "mid"
        if "down_blocks" in name_lower or "input_blocks" in name_lower:
            return "down"
        if "up_blocks" in name_lower or "output_blocks" in name_lower:
            return "up"
        return "other"

    def _normalize_group_key(self, layer_name: str) -> str:
        name_lower = layer_name.lower()
        suffixes = [
            ".ff.net.0.proj",
            ".to_out.0",
            ".proj_out",
            ".proj_in",
            ".out_proj",
            ".q_proj",
            ".k_proj",
            ".v_proj",
            ".fc1",
            ".fc2",
            ".to_q",
            ".to_k",
            ".to_v",
            ".ff.net.2",
        ]

        for suffix in suffixes:
            if suffix in name_lower:
                return name_lower.split(suffix, 1)[0]

        return name_lower.rsplit(".", 1)[0]
    
    def _get_weight_for_layer(self, layer_name: str) -> float:
        """获取层权重"""
        name_lower = layer_name.lower()
        
        # Text Encoder
        if "text_encoder" in name_lower or "_te_" in name_lower:
            if "text_encoder_2" in name_lower or "_te2_" in name_lower:
                return self.config.te2_weight
            return self.config.te_weight
        
        # UNet Down blocks
        if "down_blocks" in name_lower or "input_blocks" in name_lower:
            block_idx = self._extract_block_index(name_lower)
            if 0 <= block_idx < len(self.config.in_weights):
                return self.config.in_weights[block_idx]
        
        # UNet Mid block
        if "mid_block" in name_lower or "middle_block" in name_lower:
            return self.config.mid_weight
        
        # UNet Up blocks
        if "up_blocks" in name_lower or "output_blocks" in name_lower:
            block_idx = self._extract_block_index(name_lower)
            if 0 <= block_idx < len(self.config.out_weights):
                return self.config.out_weights[block_idx]
        
        # 默认完全训练
        return 1.0
    
    def _extract_block_index(self, layer_name: str) -> int:
        """从层名提取块索引"""
        # 匹配 "down_blocks.0" 或 "input_blocks[0]" 等
        patterns = [
            r"(?:down|up|input|output)_blocks\.(\d+)",
            r"(?:down|up|input|output)_blocks\[(\d+)\]",
            r"block_(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, layer_name)
            if match:
                return int(match.group(1))
        
        return -1
    
    def get_frozen_layers(self) -> Set[str]:
        """获取被冻结的层"""
        return self._frozen_layers
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        # 统计各类层的权重
        in_active = sum(1 for w in self.config.in_weights if w > 0)
        out_active = sum(1 for w in self.config.out_weights if w > 0)
        
        return {
            "preset": self.config.preset.value,
            "in_layers_active": f"{in_active}/12",
            "mid_active": self.config.mid_weight > 0,
            "out_layers_active": f"{out_active}/12",
            "te_active": self.config.te_weight > 0,
            "total_frozen": len(self._frozen_layers),
        }
    
    def visualize(self) -> str:
        """
        生成可视化文本
        """
        lines = ["Block Weight Configuration:"]
        lines.append("")
        
        # IN 块
        lines.append("IN Blocks (Structure/Composition):")
        in_bar = "".join(
            "█" if w >= 0.8 else "▓" if w >= 0.5 else "░" if w > 0 else " "
            for w in self.config.in_weights
        )
        lines.append(f"  [{in_bar}]")
        
        # MID 块
        mid_char = "█" if self.config.mid_weight >= 0.8 else "▓" if self.config.mid_weight >= 0.5 else "░" if self.config.mid_weight > 0 else " "
        lines.append(f"MID Block: [{mid_char}]")
        
        # OUT 块
        lines.append("OUT Blocks (Style/Details):")
        out_bar = "".join(
            "█" if w >= 0.8 else "▓" if w >= 0.5 else "░" if w > 0 else " "
            for w in self.config.out_weights
        )
        lines.append(f"  [{out_bar}]")
        
        # Text Encoder
        te_status = f"TE1: {self.config.te_weight:.1f}, TE2: {self.config.te2_weight:.1f}"
        lines.append(f"Text Encoders: {te_status}")
        
        return "\n".join(lines)


# ========== 便捷函数 ==========

def get_preset_list() -> List[Dict[str, str]]:
    """获取预设列表"""
    return [
        {"id": "full", "name": "完整训练", "description": "训练所有层"},
        {"id": "character", "name": "角色预设", "description": "侧重构图和形体，适合角色 LoRA"},
        {"id": "style", "name": "风格预设", "description": "侧重细节和画风，适合风格 LoRA"},
        {"id": "light_style", "name": "轻量风格", "description": "只训练最外层，快速画风微调"},
        {"id": "midjourney", "name": "MidJourney", "description": "平衡预设，模仿 MJ 风格"},
        {"id": "custom", "name": "自定义", "description": "手动设置每层权重"},
    ]


def create_block_weight_manager(preset: str = "full") -> BlockWeightManager:
    """创建 Block Weight 管理器"""
    preset_enum = BlockWeightPreset(preset) if preset in [p.value for p in BlockWeightPreset] else BlockWeightPreset.FULL
    config = BlockWeightConfig.from_preset(preset_enum)
    return BlockWeightManager(config)


def create_block_weight_manager_from_settings(
    preset: str = "",
    in_weights: Any = None,
    mid_weight: Any = None,
    out_weights: Any = None,
    te_weight: Any = 1.0,
    te2_weight: Any = 1.0,
    zero_threshold: Any = 0.0,
) -> BlockWeightManager:
    return BlockWeightManager(
        BlockWeightConfig.from_settings(
            preset=preset,
            in_weights=in_weights,
            mid_weight=mid_weight,
            out_weights=out_weights,
            te_weight=te_weight,
            te2_weight=te2_weight,
            zero_threshold=zero_threshold,
        )
    )
