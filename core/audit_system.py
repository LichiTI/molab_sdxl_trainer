"""
REapp Audit System | [v1.5.2]
Unified gateway for weight telemetry, spectral analysis, and dynamic pruning.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Default values for Audit System hardware and pruning thresholds
DEFAULT_SAMPLE_INTERVAL = 50
DEFAULT_PRUNE_THRESHOLD = 0.05
DEFAULT_MIN_RANK = 8
DEFAULT_WARMUP_RATIO = 0.15

@dataclass
class AuditSystemConfig:
    """审计系统全局遥测与剪枝配置载荷"""
    # 审计 (Telemetry)
    enable_auditor: bool = True
    auditor_output: str = "./logs/audit.jsonl"
    svd_algorithm: str = "rsvd"  # "rsvd" or "standard"
    advanced_stats: bool = False
    sample_interval: int = DEFAULT_SAMPLE_INTERVAL
    
    # 动态剪枝 (Pruning)
    enable_dynamic_pruning: bool = False
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD
    min_rank: int = DEFAULT_MIN_RANK
    warmup_ratio: float = DEFAULT_WARMUP_RATIO


class AuditSystem:
    """
    审计与动态剪枝集成总线。
    负责协调 LoRAAuditor 的权重遥测与 DynamicRankPruner 的在线参数压缩。
    """
    
    def __init__(
        self,
        config: Optional[AuditSystemConfig] = None,
        output_dir: str = "./logs",
        enable_auditor: bool = True,
        enable_dynamic_pruning: bool = False,
        on_audit_data: Optional[Callable[[Dict], None]] = None,
        on_prune: Optional[Callable[[Dict], None]] = None,
    ):
        self.config = config or AuditSystemConfig(
            auditor_output=str(Path(output_dir) / "audit.jsonl"),
            enable_auditor=enable_auditor,
            enable_dynamic_pruning=enable_dynamic_pruning,
        )
        
        self._auditor = None
        self._pruner = None
        self._on_audit_data = on_audit_data
        self._on_prune = on_prune
        
        # 实例化下层组件
        self._init_components()
    
    def _init_components(self):
        """同步初始化遥测与剪枝子系统"""
        # LoRAAuditor 模块加载
        if self.config.enable_auditor:
            try:
                from .lora_auditor import LoRAAuditor, AuditConfig as AuditorConfig, SVDAlgorithm
                
                auditor_config = AuditorConfig(
                    svd_algorithm=SVDAlgorithm.RSVD if self.config.svd_algorithm == "rsvd" else SVDAlgorithm.STANDARD,
                    sample_interval_pro=self.config.sample_interval,
                    advanced_stats_enabled=self.config.advanced_stats,
                )
                
                self._auditor = LoRAAuditor(
                    output_file=self.config.auditor_output,
                    config=auditor_config,
                    on_data=self._on_audit_data,
                )
                logger.info(f"[AuditSystem] LoRAAuditor initialized -> {self.config.auditor_output}")
            except ImportError as e:
                logger.warning(f"[AuditSystem] Warning: Could not load LoRAAuditor: {e}")
        
        # DynamicRankPruner 模块加载
        if self.config.enable_dynamic_pruning:
            try:
                from .svd_callback import DynamicRankPruner
                
                self._pruner = DynamicRankPruner(
                    prune_threshold=self.config.prune_threshold,
                    min_rank=self.config.min_rank,
                    warmup_ratio=self.config.warmup_ratio,
                    on_prune=self._on_prune,
                )
                logger.info(f"[AuditSystem] DynamicRankPruner initialized (threshold={self.config.prune_threshold})")
            except ImportError as e:
                logger.warning(f"[AuditSystem] Warning: Could not load DynamicRankPruner: {e}")
    
    def step(
        self,
        step: int,
        total_steps: int,
        epoch: int,
        loss: float,
        lr_unet: float = 0.0,
        lr_te: float = 0.0,
        network=None,
        optimizer=None,
        noise_pred=None,
        vram_gb: float = 0.0,
        power_w: float = 0.0,
        throughput: float = 0.0,
    ) -> Dict[str, Any]:
        """执行单步审计积分与剪枝决策"""
        result = {}
        
        # 1. 权重频谱遥测
        if self._auditor:
            audit_result = self._auditor.step(
                step=step,
                total_steps=total_steps,
                epoch=epoch,
                loss=loss,
                lr_unet=lr_unet,
                lr_te=lr_te,
                optimizer=optimizer,
                network=network,
                noise_pred=noise_pred,
                vram_gb=vram_gb,
                power_w=power_w,
                throughput=throughput,
            )
            result["audit"] = audit_result
        
        # 2. 在线特征反馈剪枝
        if self._pruner and network is not None:
            prune_result = self._pruner.step(
                step=step,
                total_steps=total_steps,
                network=network,
                optimizer=optimizer,
            )
            if prune_result:
                result["pruned"] = prune_result
        
        return result
    
    def set_sampling(self, is_sampling: bool):
        """采样状态同步：用于在图像预览期间挂起审计采样"""
        if self._auditor:
            self._auditor.set_sampling(is_sampling)
    
    def get_diagnosis_report(self):
        """生成系统层级的诊断报告摘要"""
        if self._auditor:
            try:
                from .lora_auditor import AuditInterpreter
                interpreter = AuditInterpreter()
                return interpreter.generate_report()
            except Exception:
                pass
        return None
    
    def get_prune_summary(self) -> Optional[Dict]:
        """获取当前剪枝操作的统计快照"""
        if self._pruner:
            return self._pruner.get_summary()
        return None
    
    def close(self):
        """安全销毁审计网关注册"""
        if self._auditor:
            self._auditor.close()
        logger.info("[AuditSystem] Closed")


# ========== 便捷工厂函数 ==========

def create_audit_system(
    output_dir: str = "./logs",
    enable_auditor: bool = True,
    enable_pruning: bool = False,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    min_rank: int = DEFAULT_MIN_RANK,
) -> AuditSystem:
    """快速实例化审计系统"""
    config = AuditSystemConfig(
        auditor_output=str(Path(output_dir) / "audit.jsonl"),
        enable_auditor=enable_auditor,
        enable_dynamic_pruning=enable_pruning,
        prune_threshold=prune_threshold,
        min_rank=min_rank,
    )
    return AuditSystem(config=config)


def auto_inject_if_enabled() -> Optional[AuditSystem]:
    """根据环境变量配置执行零入侵自动注入"""
    enabled = os.environ.get('LULYNX_AUDIT_ENABLED', '').lower() in ('1', 'true', 'yes')
    if not enabled:
        return None
    
    output_dir = os.environ.get('LULYNX_AUDIT_OUTPUT', './logs')
    pruning_enabled = os.environ.get('LULYNX_DYNAMIC_PRUNING', '').lower() in ('1', 'true', 'yes')
    prune_threshold = float(os.environ.get('LULYNX_PRUNE_THRESHOLD', str(DEFAULT_PRUNE_THRESHOLD)))
    min_rank = int(os.environ.get('LULYNX_MIN_RANK', str(DEFAULT_MIN_RANK)))
    
    logger.info(f"[AuditSystem] Auto-inject enabled")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Dynamic Pruning: {pruning_enabled}")
    
    return create_audit_system(
        output_dir=output_dir,
        enable_auditor=True,
        enable_pruning=pruning_enabled,
        prune_threshold=prune_threshold,
        min_rank=min_rank,
    )
