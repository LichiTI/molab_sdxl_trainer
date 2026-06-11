"""
MN-LoRA Smart Presets | V2.6
针对不同模型架构的预置配置
"""

from typing import Dict, Any

# Slim 默认档：实测 100 步里比 full/practical 更均衡，KFAC 仍作为实验选项。
SLIM_PRESET: Dict[str, Any] = {
    "k_ratio": 0.5,
    "update_interval": 20,
    "residual_threshold": 0.3,
    "lazy_update": True,
    "lazy_threshold": 0.5,
    "enable_sparse_sampling": True,
    "sparsity_ratio": 0.5,
    # Pilot
    "pilot_strategy": "population",
    "pilot_aggressiveness": 0.5,
    # TG-WD
    "tgwd_base_decay": 0.01,
    "tgwd_update_interval": 50,
}

# SDXL 预设：更大的模型需要更保守的投影
SDXL_PRESET: Dict[str, Any] = {
    "k_ratio": 0.6,              # 保留更多主成分
    "update_interval": 150,       # 大模型更新不宜太频繁
    "residual_threshold": 0.25,   # 更严格的残差阈值
    "lazy_update": True,
    "lazy_threshold": 0.4,
    "enable_sparse_sampling": True,  # 开启稀疏采样节省计算
    "sparsity_ratio": 0.6,
    # Pilot
    "pilot_strategy": "ema",      # SDXL 适合 EMA 趋势检测
    "pilot_aggressiveness": 0.4,  # 略保守
    # TG-WD
    "tgwd_base_decay": 0.01,
    "tgwd_update_interval": 100,
}

# Flux 预设：DiT 架构需要特殊处理
FLUX_PRESET: Dict[str, Any] = {
    "k_ratio": 0.5,
    "update_interval": 100,
    "residual_threshold": 0.35,   # DiT 允许更高残差
    "lazy_update": True,
    "lazy_threshold": 0.5,
    "enable_sparse_sampling": True,
    "sparsity_ratio": 0.5,
    # Pilot
    "pilot_strategy": "population",  # DiT 适合群体统计
    "pilot_aggressiveness": 0.5,
    # TG-WD
    "tgwd_base_decay": 0.015,
    "tgwd_update_interval": 80,
}

# SD 1.5 预设：小模型可以更激进
SD15_PRESET: Dict[str, Any] = {
    "k_ratio": 0.4,               # 可以更激进地压缩
    "update_interval": 80,        # 更频繁更新
    "residual_threshold": 0.4,
    "lazy_update": False,         # 小模型不需要懒惰更新
    "lazy_threshold": 0.5,
    "enable_sparse_sampling": False,  # 小模型不需要稀疏采样
    "sparsity_ratio": 1.0,
    # Pilot
    "pilot_strategy": "heuristic",
    "pilot_aggressiveness": 0.6,
    # TG-WD
    "tgwd_base_decay": 0.02,
    "tgwd_update_interval": 50,
}

# V2.7: Dataset-Specific Presets
# 动漫专攻：高秩保留，弱正则化，保留线条细节
ANIME_PRESET: Dict[str, Any] = {
    "k_ratio": 0.75,              # 极高的子空间保留率
    "update_interval": 200,       # 很少更新 SVD，防止画风漂移
    "residual_threshold": 0.2,    # 严格的残差控制
    "lazy_update": True,
    "lazy_threshold": 0.3,
    "enable_sparse_sampling": False, # 关闭稀疏采样，追求全量精度
    "sparsity_ratio": 1.0,
    # Pilot
    "pilot_strategy": "pid",      # PID 精确控制
    "pilot_aggressiveness": 0.3,  # 低攻击性
    # TG-WD
    "tgwd_base_decay": 0.005,     # 极低的权重衰减
    "tgwd_update_interval": 200,
}

# 写实专攻：低秩泛化，强正则化，防止过拟合
REALISM_PRESET: Dict[str, Any] = {
    "k_ratio": 0.35,              # 激进压缩，强迫模型学习通用特征
    "update_interval": 50,        # 频繁更新，捕捉光影变化
    "residual_threshold": 0.45,   # 宽松
    "lazy_update": False,
    "lazy_threshold": 0.6,
    "enable_sparse_sampling": True,
    "sparsity_ratio": 0.7,
    # Pilot
    "pilot_strategy": "population",
    "pilot_aggressiveness": 0.8,  # 高攻击性，压制过拟合
    # TG-WD
    "tgwd_base_decay": 0.03,      # 强权重衰减
    "tgwd_update_interval": 25,
}

# 预设映射
PRESET_MAP: Dict[str, Dict[str, Any]] = {
    "slim": SLIM_PRESET,
    "sdxl": SDXL_PRESET,
    "flux": FLUX_PRESET,
    "sd15": SD15_PRESET,
    "sd3": FLUX_PRESET,
    # Dataset specific
    "anime": ANIME_PRESET,
    "realism": REALISM_PRESET,
}

def get_preset_for_model(model_type: str) -> Dict[str, Any]:
    """获取指定模型类型的预设配置"""
    model_type_lower = model_type.lower()
    return PRESET_MAP.get(model_type_lower, SD15_PRESET)  # 默认 SD15


def select_mnlora_preset(model_type: str, preset_name: str = "") -> Dict[str, Any]:
    """Select an optimizer preset from either explicit preset name or model type.

    Public UI presets such as fast/balanced/quality are handled by the broader
    hyperparameter manager. This selector only consumes architecture/dataset
    presets that exist in PRESET_MAP, then falls back to the model family.
    """
    explicit = str(preset_name or "").strip().lower()
    if explicit in PRESET_MAP:
        return PRESET_MAP[explicit]
    return get_preset_for_model(model_type)


def split_mnlora_preset(preset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Split a flat MN-LoRA preset into optimizer component configs.

    The runtime optimizer expects separate configuration dictionaries for GSP,
    TG-WD, and TrainingPilot. Keeping the public preset flat is convenient for
    UI/metadata, but it must be normalized before constructing MNLoRAOptimizer.
    """
    return {
        "gsp_config": {
            "k_ratio": float(preset.get("k_ratio", 0.5)),
            "update_interval": int(preset.get("update_interval", 100)),
            "residual_threshold": float(preset.get("residual_threshold", 0.3)),
            "lazy_update": bool(preset.get("lazy_update", True)),
            "lazy_threshold": float(preset.get("lazy_threshold", 0.5)),
            "enable_sparse_sampling": bool(preset.get("enable_sparse_sampling", False)),
            "sparsity_ratio": float(preset.get("sparsity_ratio", 0.5)),
        },
        "tgwd_config": {
            "base_lambda": float(preset.get("tgwd_base_decay", 0.01)),
            "probe_interval": int(preset.get("tgwd_update_interval", 50)),
        },
        "pilot_config": {
            "strategy": str(preset.get("pilot_strategy", "population")),
        },
    }
