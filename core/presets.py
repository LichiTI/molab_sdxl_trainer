from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SmartPresetManager:
    """
    Omni-Tuner 智能预设管理器
    
    自动配置训练参数以适应不同的硬件和目标
    """
    
    PRESETS = {
        "survival": {
            "name": "Survival Mode (生存模式)",
            "description": "8GB VRAM 极致优化，牺牲速度换取运行",
            "config": {
                "optimizer_type": "adafactor",
                "learning_rate": 1e-4, # Adafactor 通常需要稍高的 LR
                "network_dim": 8,
                "network_alpha": 4,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "mixed_precision": "fp16",
                "lisa_enabled": True,
                "lisa_active_ratio": 2, # 激进的 LISA
                "pissa_enabled": False, # 节省初始化显存
                "dora_enabled": False,
                "gradient_checkpointing": True,
                "cache_latents": True,
                "cache_text_encoder_outputs": True,
            }
        },
        "balanced": {
            "name": "Balanced Mode (平衡模式)",
            "description": "标准配置，速度与质量的最佳平衡 (推荐 12-16GB)",
            "config": {
                "optimizer_type": "adamw8bit",
                "learning_rate": 1e-4,
                "network_dim": 32,
                "network_alpha": 16,
                "mixed_precision": "bf16", # 优先尝试 bf16
                "lisa_enabled": False,
                "pissa_enabled": True, # 加速收敛
                "pissa_svd_algo": "rsvd",
                "dora_enabled": False,
                "gradient_checkpointing": True,
            }
        },
        "artist": {
            "name": "Artist Mode (画师模式)",
            "description": "追求极致画质与细节表现 (24GB+)",
            "config": {
                "optimizer_type": "prodigy", # 自适应，高质量
                "learning_rate": 1.0, # Prodigy 默认 LR
                "network_dim": 128,
                "network_alpha": 128, # Alpha=Dim for stabilizing high rank
                "lisa_enabled": False,
                "pissa_enabled": False, # Prodigy 自带适配，或者配合 DoRA
                "dora_enabled": True, # 核心：权重分解
                "smart_rank_enabled": True, # 动态剪枝
                "smart_rank_min": 16,
                "monitor_svd_algo": "full", # 精确监控
                "min_snr_gamma": 5.0, # 细节增强
            }
        },
        "speedster": {
            "name": "Speedster Mode (极速模式)",
            "description": "最大化吞吐量，适合快速验证",
            "config": {
                "optimizer_type": "adamw8bit",
                "network_dim": 16,
                "network_alpha": 8,
                "lisa_enabled": False,
                "pissa_enabled": True,
                "pissa_svd_algo": "rsvd",
                "dora_enabled": False,
                "gradient_accumulation_steps": 1,
                # Batch size 应该由外部自动探测决定，这里设为较激进的默认值
                "train_batch_size": 4, 
            }
        }
    }

    @classmethod
    def get_preset(cls, preset_name: str) -> Optional[Dict[str, Any]]:
        return cls.PRESETS.get(preset_name.lower())

    @classmethod
    def apply_preset(cls, current_config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
        """
        将预设应用到当前配置
        """
        preset = cls.get_preset(preset_name)
        if not preset:
            logger.warning(f"Preset {preset_name} not found, skipping.")
            return current_config
            
        new_config = current_config.copy()
        preset_config = preset["config"]
        
        logger.info(f"Applying preset: {preset.get('name', preset_name)}")
        
        for key, value in preset_config.items():
            # Validate key existence and type if possible
            if key in new_config:
                 # Optional: Check type match?
                 # if not isinstance(value, type(new_config[key])) and new_config[key] is not None:
                 #    logger.warning(f"Type mismatch for {key}: {type(value)} vs {type(new_config[key])}")
                 pass
            
            new_config[key] = value
            logger.debug(f"  Override {key} -> {value}")
            
        return new_config
