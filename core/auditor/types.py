from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

class AuditMode(Enum):
    """审计模式"""
    SUSPEND = "SUSPEND"  # 采样中，暂停审计
    STOP = "STOP"        # 显存红线，熔断
    LITE = "LITE"        # 显存 95-98%，降级模式
    PRO = "PRO"          #显存 < 95%，全速模式

class SVDAlgorithm(Enum):
    """SVD 算法选择"""
    STANDARD = "standard"  # 标准 SVD (完整矩阵, 精度最高)
    RSVD = "rsvd"          # 随机化 SVD (最快速度, 适合大矩阵)
    BRANDS = "brands"      # Brand's SVD (增量更新, 保持细节)

@dataclass
class AuditConfig:
    """审计配置"""
    # 算法选择
    svd_algorithm: SVDAlgorithm = SVDAlgorithm.RSVD
    
    # rSVD 参数
    rsvd_k_pro: int = 50   # PRO 模式下的投影维度
    rsvd_k_lite: int = 10  # LITE 模式下的投影维度
    
    # 采样参数
    sample_interval_pro: int = 50    # PRO 模式采样间隔
    sample_interval_lite: int = 100  # LITE 模式采样间隔
    
    # Dead Neuron 阈值
    dead_neuron_epsilon: float = 1e-5
    dead_neuron_stride_pro: int = 10   # 10% 采样
    dead_neuron_stride_lite: int = 100 # 1% 采样
    
    # Gradient Coherence 投影维度
    grad_proj_dim_pro: int = 8
    grad_proj_dim_lite: int = 1
    
    # 高级模式
    advanced_stats_enabled: bool = False

@dataclass
class AuditMetrics:
    """审计指标结果"""
    # 权重拓扑
    stable_rank: Optional[float] = None
    svd_entropy: Optional[float] = None
    spectral_smoothness: Optional[float] = None
    dead_neuron_rate: Optional[float] = None
    rms: Optional[float] = None
    rms_scaled: Optional[float] = None
    
    # 训练动力学
    update_ratio: Optional[float] = None
    grad_coherence: Optional[float] = None
    gsnr: Optional[float] = None
    hessian_trace: Optional[float] = None
    
    # 语义与画质
    clip_drift: Optional[float] = None
    attn_entropy: Optional[float] = None
    noise_pred_std: Optional[float] = None
    
    # 几何与安全
    act_drift: Optional[float] = None
    forgetting_probe: Optional[float] = None

    # ICU 健康聚合
    icu_score: Optional[int] = None

    # 元信息
    mode: str = "PRO"
    svd_algorithm: str = "rsvd"
