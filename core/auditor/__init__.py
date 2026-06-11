from .types import AuditConfig, AuditMode, SVDAlgorithm, AuditMetrics
from .watchdog import HardwareWatchdog
from .engine import MetricEngine
from .interpreter import DiagnosisItem, DiagnosisReport, AuditInterpreter
from .auditor import LoRAAuditor, stateless_projection
import os
from typing import Optional, Callable

def create_auditor(
    output_file: str = "./logs/audit.jsonl",
    svd_algorithm: str = "rsvd",
    advanced_stats: bool = False,
    on_data: Optional[Callable] = None
) -> LoRAAuditor:
    """
    创建审计器的便捷函数
    """
    algo_map = {
        "rsvd": SVDAlgorithm.RSVD,
        "standard": SVDAlgorithm.STANDARD,
        "brands": SVDAlgorithm.BRANDS,
    }
    config = AuditConfig(
        svd_algorithm=algo_map.get(svd_algorithm, SVDAlgorithm.RSVD),
        advanced_stats_enabled=advanced_stats
    )
    return LoRAAuditor(output_file=output_file, config=config, on_data=on_data)

def auto_inject_if_enabled() -> Optional[LoRAAuditor]:
    """
    自动注入审计器 (通过环境变量)
    """
    if os.environ.get('LULYNX_AUDIT_ENABLED', '').lower() in ('1', 'true', 'yes'):
        output_file = os.environ.get('LULYNX_AUDIT_OUTPUT', './logs/audit.jsonl')
        svd_algo = os.environ.get('LULYNX_SVD_ALGORITHM', 'rsvd')
        advanced = os.environ.get('LULYNX_ADVANCED_STATS', '').lower() in ('1', 'true', 'yes')
        
        print(f'[LoRAAuditor] Auto-inject enabled')
        print(f'  Output: {output_file}')
        print(f'  SVD Algorithm: {svd_algo}')
        
        return create_auditor(
            output_file=output_file,
            svd_algorithm=svd_algo,
            advanced_stats=advanced
        )
    return None
