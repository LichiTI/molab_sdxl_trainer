"""
Lulynx Core - LoRA Training Toolkit

核心模块：
- lora_auditor: LoRA 审计系统 V10.0 (rSVD, 13 项指标, 语义解释器)
- svd_callback: SVD 回调监控
- gpu_resource_manager: GPU 资源管理
"""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "SVDCallback": (".svd_callback", "SVDCallback"),
    "auto_inject_if_enabled": (".svd_callback", "auto_inject_if_enabled"),
    "LoRAAuditor": (".lora_auditor", "LoRAAuditor"),
    "AuditConfig": (".lora_auditor", "AuditConfig"),
    "AuditMetrics": (".lora_auditor", "AuditMetrics"),
    "AuditMode": (".lora_auditor", "AuditMode"),
    "SVDAlgorithm": (".lora_auditor", "SVDAlgorithm"),
    "HardwareWatchdog": (".lora_auditor", "HardwareWatchdog"),
    "MetricEngine": (".lora_auditor", "MetricEngine"),
    "create_auditor": (".lora_auditor", "create_auditor"),
    "AuditInterpreter": (".lora_auditor", "AuditInterpreter"),
    "DiagnosisReport": (".lora_auditor", "DiagnosisReport"),
    "DiagnosisItem": (".lora_auditor", "DiagnosisItem"),
    "stateless_projection": (".lora_auditor", "stateless_projection"),
    "AuditLauncher": (".audit_launcher", "AuditLauncher"),
    "LogParser": (".audit_launcher", "LogParser"),
    "TrainingMetrics": (".audit_launcher", "TrainingMetrics"),
    "create_launcher": (".audit_launcher", "create_launcher"),
    "wrap_training_output": (".audit_launcher", "wrap_training_output"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    try:
        value = getattr(import_module(module_name, __name__), attr_name)
    except Exception:
        value = None
    globals()[name] = value
    return value

__all__ = [
    # SVD Callback (legacy compatibility)
    'SVDCallback',
    'auto_inject_if_enabled',
    
    # LoRA Auditor V10.0
    'LoRAAuditor',
    'AuditConfig',
    'AuditMetrics', 
    'AuditMode',
    'SVDAlgorithm',
    'HardwareWatchdog',
    'MetricEngine',
    'create_auditor',
    
    # V10.0 Interpreter
    'AuditInterpreter',
    'DiagnosisReport',
    'DiagnosisItem',
    'stateless_projection',
    
    # Audit Launcher (optional legacy)
    'AuditLauncher',
    'LogParser',
    'TrainingMetrics',
    'create_launcher',
    'wrap_training_output',
]
