"""
SmartRank: 动态秩调整控制器
核心逻辑：基于 Stable Rank 和 SVD Entropy 动态优化 LoRA 层秩
"""

import torch
import torch.nn as nn
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math

logger = logging.getLogger(__name__)

class SmartRankController:
    """SmartRank 控制器"""
    
    def __init__(
        self,
        lora_injector: Any,
        min_rank: int = 4,
        max_rank: int = 128,
        interval: int = 50,
        energy_threshold: float = 0.9,  # 能量保留阈值 $(0.9 = 90\%)$
    ):
        self.injector = lora_injector
        self.min_rank = min_rank
        # max_rank 初始通常受限于硬件，这里主要是上限保护
        self.max_rank = max_rank
        self.interval = interval
        self.energy_threshold = energy_threshold
        
        self.step_count = 0
        self.history: Dict[str, List[float]] = {}
        
    def step(self, step: int, metrics: Dict[str, Dict[str, float]]):
        """执行一步 SmartRank 检查"""
        self.step_count = step
        
        if step == 0 or step % self.interval != 0:
            return
        
        logger.info(f"SmartRank analyzing at step {step}...")
        
        # 遍历所有注入的层
        for name, layer_metrics in metrics.items():
            if "stable_rank" not in layer_metrics:
                continue
            
            stable_rank = layer_metrics["stable_rank"]
            lora_linear = self.injector.injected_layers.get(name)
            if not lora_linear:
                continue
                
            current_rank = lora_linear.lora.rank
            
            # 策略 1: 过剩剪枝 (Pruning)
            # 如果 stable_rank 远小于 current_rank，说明维度利用不足
            # 注意: stable_rank 是信息理论上的有效 rank，通常小于物理 rank
            # 我们根据 stable_rank 的分布来确定新的物理 rank
            
            if stable_rank < current_rank * 0.7:
                self._apply_pruning(name, lora_linear, stable_rank)
            
    def _apply_pruning(self, name: str, lora_linear: Any, suggested_rank: float):
        """对指定层执行 SVD 剪枝"""
        # 计算新秩 (向上取整并对齐 4 的倍数以便于张量对齐)
        new_rank = max(int(math.ceil(suggested_rank / 4) * 4), self.min_rank)
        
        if new_rank >= lora_linear.lora.rank:
            return
            
        logger.info(f"SmartRank Pruning [{name}]: {lora_linear.lora.rank} -> {new_rank} (StableRank: {suggested_rank})")
        
        with torch.no_grad():
            # 1. 提取当前 LoRA 权重矩阵 W_lora = BA
            # 考虑 alpha/rank 缩放
            scaling = lora_linear.lora.scaling
            down_w = lora_linear.lora.lora_down.weight.data
            up_w = lora_linear.lora.lora_up.weight.data
            
            # W_lora = (up_w @ down_w) * scaling
            W_lora = (up_w @ down_w) * scaling
            
            # 2. 对 W_lora 执行 SVD 分解
            # W_lora ~= U S V^T
            try:
                # 转为 float32 执行 SVD 保证精度
                U, S, V = torch.linalg.svd(W_lora.float(), full_matrices=False)
                
                # 3. 截断到新秩
                U_new = U[:, :new_rank]
                S_new = S[:new_rank]
                V_new = V[:new_rank, :]
                
                # 4. 重新初始化 LoRA 层参数
                # 新的 BA * new_scaling 应等于 U_new S_new V_new
                # 我们倾向于保持 alpha 不变，更新 alpha/rank 缩放
                alpha = lora_linear.lora.alpha
                new_scaling = alpha / new_rank
                
                # B = U_new * sqrt(S_new) / new_scaling
                # A = sqrt(S_new) * V_new
                S_sqrt = torch.sqrt(S_new)
                new_up_w = (U_new * S_sqrt.unsqueeze(0)) / new_scaling
                new_down_w = S_sqrt.unsqueeze(1) * V_new
                
                # 5. 更新模型参数
                # 注意: 动态改变参数形状会导致优化器失效，这里需要处理优化器状态
                # 在本 Demo/Prototype 中，我们暂时假设用户手动重启或处理了优化器。
                # 完整实现需要调用 optimizer.reinitialize_param(p_old, p_new)
                
                # 此处仅演示参数替换 (静态逻辑)
                # 真正的动态 SmartRank 需要层重定向
                lora_linear.lora.rank = new_rank
                lora_linear.lora.scaling = new_scaling
                
                # 修改层结构 (重构 nn.Linear)
                device = down_w.device
                dtype = down_w.dtype
                
                new_down_layer = nn.Linear(lora_linear.lora.lora_down.in_features, new_rank, bias=False).to(device, dtype)
                new_up_layer = nn.Linear(new_rank, lora_linear.lora.lora_up.out_features, bias=False).to(device, dtype)
                
                new_down_layer.weight.data.copy_(new_down_w.to(dtype))
                new_up_layer.weight.data.copy_(new_up_w.to(dtype))
                
                lora_linear.lora.lora_down = new_down_layer
                lora_linear.lora.lora_up = new_up_layer
                
                logger.debug(f"SmartRank: Layer {name} updated to rank {new_rank}")
                
            except Exception as e:
                logger.error(f"SmartRank SVD failed for {name}: {e}")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取更新后的所有可训练参数 (供优化器更新使用)"""
        params = []
        for lora_linear in self.injector.injected_layers.values():
            params.extend(lora_linear.lora.parameters())
        return params


def infer_rank_from_svd(
    weight: torch.Tensor,
    min_rank: int = 4,
    max_rank: int = 128,
    energy_threshold: float = 0.9,
) -> int:
    """Infer a suitable LoRA rank from the SVD energy distribution of a weight matrix.

    Computes the singular value decomposition of ``weight`` and returns the
    smallest rank that retains at least ``energy_threshold`` fraction of the
    total singular-value energy (sum of squares), clamped to
    ``[min_rank, max_rank]``.

    Args:
        weight: 2-D weight matrix (out_features, in_features).
        min_rank: Lower bound for the returned rank.
        max_rank: Upper bound for the returned rank.
        energy_threshold: Fraction of total energy to preserve (0 < t <= 1).

    Returns:
        Inferred rank as an integer, aligned to a multiple of 4.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2-D weight, got shape {weight.shape}")

    S = torch.linalg.svdvals(weight.float())
    total_energy = (S ** 2).sum()
    if total_energy < 1e-12:
        return min_rank

    cumulative = torch.cumsum(S ** 2, dim=0)
    # First index where cumulative/total >= threshold
    ratios = cumulative / total_energy
    mask = ratios >= energy_threshold
    if mask.any():
        idx = mask.nonzero(as_tuple=True)[0][0].item()
    else:
        idx = len(S) - 1
    rank = idx + 1  # singular values are 1-indexed for rank

    # Align to multiple of 4
    rank = max(min_rank, int(math.ceil(rank / 4) * 4))
    rank = min(max_rank, rank)
    return rank

@dataclass(frozen=True)
class RankAdvice:
    """Report-only SmartRank recommendation.

    This object never mutates model layers or optimizer state. It is intended for
    UI/advisor output before enabling dynamic SmartRank pruning.
    """

    current_rank: int
    suggested_rank: int
    min_rank: int
    max_rank: int
    severity: str
    reason: str
    source: str = "heuristic"
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_rank": int(self.current_rank),
            "suggested_rank": int(self.suggested_rank),
            "min_rank": int(self.min_rank),
            "max_rank": int(self.max_rank),
            "severity": self.severity,
            "reason": self.reason,
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "notes": list(self.notes),
        }


def _align_rank(rank: int, min_rank: int, max_rank: int, multiple: int = 4) -> int:
    rank = int(math.ceil(max(rank, min_rank) / multiple) * multiple)
    return max(min_rank, min(rank, max_rank))


def advise_rank(
    current_rank: int,
    inferred_rank: Optional[int] = None,
    stable_rank: Optional[float] = None,
    min_rank: int = 4,
    max_rank: int = 128,
    prune_ratio: float = 0.7,
    grow_ratio: float = 0.92,
) -> RankAdvice:
    """Create a report-only rank recommendation.

    Args:
        current_rank: Configured physical LoRA/LyCORIS rank.
        inferred_rank: Optional rank inferred from a weight SVD scan.
        stable_rank: Optional runtime stable-rank metric from auditor output.
        min_rank: Lower bound for suggestions.
        max_rank: Upper bound for suggestions.
        prune_ratio: Stable-rank ratio below which the rank looks over-provisioned.
        grow_ratio: Ratio above which the rank looks saturated.

    Returns:
        RankAdvice. The caller may show this in UI/report, but should not mutate
        training config automatically.
    """
    current = max(int(current_rank or min_rank), 1)
    min_rank = max(int(min_rank or 4), 1)
    max_rank = max(int(max_rank or current), min_rank)
    notes = ["Report-only advice; dynamic rank pruning is not enabled by this function."]

    candidates: List[int] = []
    source_bits: List[str] = []
    confidence = 0.35

    if inferred_rank is not None and inferred_rank > 0:
        candidates.append(int(inferred_rank))
        source_bits.append("svd")
        confidence = max(confidence, 0.72)
    if stable_rank is not None and stable_rank > 0:
        candidates.append(int(math.ceil(float(stable_rank) / 4.0) * 4))
        source_bits.append("stable_rank")
        confidence = max(confidence, 0.65)

    if candidates:
        raw_suggestion = max(candidates)
    elif current >= 96:
        raw_suggestion = max(current // 2, min_rank)
        notes.append("No SVD/runtime metrics supplied; high rank heuristic used.")
    elif current >= 64:
        raw_suggestion = max(int(current * 0.75), min_rank)
        notes.append("No SVD/runtime metrics supplied; medium-high rank heuristic used.")
    else:
        raw_suggestion = current
        notes.append("Current rank is modest; keep it unless quality or speed metrics say otherwise.")

    suggested = _align_rank(raw_suggestion, min_rank, max_rank)
    source = "+".join(source_bits) if source_bits else "heuristic"

    if stable_rank is not None and stable_rank > 0:
        ratio = float(stable_rank) / max(float(current), 1.0)
        if ratio < prune_ratio and suggested < current:
            return RankAdvice(current, suggested, min_rank, max_rank, "watch", "rank appears over-provisioned", source, confidence, notes)
        if ratio > grow_ratio and current < max_rank:
            grown = _align_rank(max(current + 4, int(current * 1.25)), min_rank, max_rank)
            return RankAdvice(current, grown, min_rank, max_rank, "info", "rank appears saturated", source, confidence, notes)

    if suggested < current:
        drop_ratio = 1.0 - (suggested / max(current, 1))
        severity = "watch" if drop_ratio >= 0.25 else "info"
        return RankAdvice(current, suggested, min_rank, max_rank, severity, "lower rank may be enough", source, confidence, notes)
    if suggested > current:
        return RankAdvice(current, suggested, min_rank, max_rank, "info", "higher rank may improve capacity", source, confidence, notes)
    return RankAdvice(current, current, min_rank, max_rank, "ok", "current rank looks reasonable", source, confidence, notes)


def advise_rank_from_weight(
    weight: torch.Tensor,
    current_rank: int,
    min_rank: int = 4,
    max_rank: int = 128,
    energy_threshold: float = 0.9,
) -> RankAdvice:
    """Infer SVD rank from a weight tensor and return report-only advice."""
    inferred = infer_rank_from_svd(weight, min_rank=min_rank, max_rank=max_rank, energy_threshold=energy_threshold)
    return advise_rank(
        current_rank=current_rank,
        inferred_rank=inferred,
        min_rank=min_rank,
        max_rank=max_rank,
    )
