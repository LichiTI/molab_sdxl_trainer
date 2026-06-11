# MN-LoRA Optimizer Wrapper
# Explicit API — no monkey-patching, no auto-injection.

"""
MN-LoRA 优化器显式包装 API

设计理念：
1. **显式调用** — 不修改 torch.optim，不劫持 accelerate.prepare
2. **无副作用** — 模块加载时不执行任何注入
3. **Fallback** — 包装失败时返回原始优化器

使用方法：
    from training_components.mn_lora.hijacker import wrap_optimizer

    optimizer = AdamW(params, lr=1e-4)
    optimizer = wrap_optimizer(optimizer, enable_gsp=True, enable_tgwd=True)
"""

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def wrap_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    enable_gsp: bool = True,
    enable_tgwd: bool = True,
    enable_pilot: bool = True,
    gsp_config: Optional[Dict] = None,
    tgwd_config: Optional[Dict] = None,
    pilot_config: Optional[Dict] = None,
    plus_plus_config: Optional[Dict] = None,
    kfac_lite_config: Optional[Dict] = None,
    trust_region_config: Optional[Dict] = None,
    effective_delta_config: Optional[Dict] = None,
    fisher_ewc_config: Optional[Dict] = None,
    gradient_conflict_config: Optional[Dict] = None,
    lora_modules: Optional[Dict[str, object]] = None,
    param_names: Optional[Dict[int, str]] = None,
) -> torch.optim.Optimizer:
    """
    Wrap an optimizer with MN-LoRA enhancements (GSP, TG-WD, TrainingPilot).

    Returns the original optimizer unchanged if MN-LoRA components
    cannot be imported or if wrapping fails.

    Args:
        optimizer: The base optimizer to wrap.
        enable_gsp: Enable Gradient Subspace Projection.
        enable_tgwd: Enable Trace-Guided Weight Decay.
        enable_pilot: Enable TrainingPilot.
        gsp_config: GSP configuration dict.
        tgwd_config: TG-WD configuration dict.
        pilot_config: Pilot configuration dict.

    Returns:
        Wrapped optimizer, or the original optimizer on failure.
    """
    try:
        from .mn_optimizer import MNLoRAOptimizer
    except ImportError as e:
        logger.warning(f"[MN-LoRA] MNLoRAOptimizer not available: {e}")
        return optimizer

    try:
        wrapped = MNLoRAOptimizer(
            base_optimizer=optimizer,
            enable_tgwd=enable_tgwd,
            enable_gsp=enable_gsp,
            enable_pilot=enable_pilot,
            tgwd_config=tgwd_config or {},
            gsp_config=gsp_config or {},
            pilot_config=pilot_config or {},
            plus_plus_config=plus_plus_config or {},
            kfac_lite_config=kfac_lite_config or {},
            trust_region_config=trust_region_config or {},
            effective_delta_config=effective_delta_config or {},
            fisher_ewc_config=fisher_ewc_config or {},
            gradient_conflict_config=gradient_conflict_config or {},
            lora_modules=lora_modules or {},
            param_names=param_names or {},
        )
        logger.info(
            f"[MN-LoRA] Wrapped {type(optimizer).__name__} -> MNLoRAOptimizer "
            f"(GSP={enable_gsp}, TG-WD={enable_tgwd}, Pilot={enable_pilot}, "
            f"PlusPlus={bool((plus_plus_config or {}).get('enabled', False))})"
        )
        return wrapped
    except Exception as e:
        logger.error(f"[MN-LoRA] Wrap failed, returning original optimizer: {e}")
        return optimizer
