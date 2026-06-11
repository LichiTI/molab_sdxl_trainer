"""
PCGrad: Projected Conflicting Gradients (概念卫士)

解决多概念训练时的梯度冲突问题。

问题场景:
    同时训练 "White Fur" 和 "Dark Background" 时，
    某些图片可能导致这两个概念的梯度方向冲突 (夹角 > 90°)，
    简单的 Batch Mean 会相互抵消。

解决方案:
    检测冲突 → 投影到法平面 → 消除破坏性干涉

参考:
    Gradient Surgery for Multi-Task Learning (arXiv:2001.06782)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GradientConflictInfo:
    """梯度冲突信息"""
    sample_i: int
    sample_j: int
    cosine_similarity: float
    is_conflict: bool


@dataclass
class PCGradStats:
    """PCGrad 统计"""
    total_pairs: int = 0
    conflict_pairs: int = 0
    projections_applied: int = 0
    avg_conflict_angle: float = 0.0


def _normalize_reduction(reduction: str) -> str:
    normalized = str(reduction or "mean").strip().lower()
    if normalized not in {"mean", "sum"}:
        return "mean"
    return normalized


def _ordered_param_names(per_sample_grads: List[Dict[str, torch.Tensor]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for grad_map in per_sample_grads:
        for name in grad_map.keys():
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return ordered


def _template_tensors(
    per_sample_grads: List[Dict[str, torch.Tensor]],
    param_names: List[str],
) -> Dict[str, torch.Tensor]:
    templates: Dict[str, torch.Tensor] = {}
    for name in param_names:
        for grad_map in per_sample_grads:
            tensor = grad_map.get(name)
            if isinstance(tensor, torch.Tensor):
                templates[name] = tensor
                break
    return templates


def _clone_grad_map(
    grad_map: Dict[str, torch.Tensor],
    param_names: List[str],
    templates: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for name in param_names:
        tensor = grad_map.get(name)
        if isinstance(tensor, torch.Tensor):
            cloned[name] = tensor.detach().clone()
            continue
        template = templates.get(name)
        if template is not None:
            cloned[name] = torch.zeros_like(template)
    return cloned


def _flatten_named_grads(
    grad_map: Dict[str, torch.Tensor],
    param_names: List[str],
) -> Optional[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    for name in param_names:
        tensor = grad_map.get(name)
        if isinstance(tensor, torch.Tensor):
            chunks.append(tensor.detach().float().reshape(-1))
    if not chunks:
        return None
    return torch.cat(chunks)


def _check_conflict(
    grad_i: Dict[str, torch.Tensor],
    grad_j: Dict[str, torch.Tensor],
    param_names: List[str],
    conflict_threshold: float,
) -> Tuple[bool, float]:
    flat_i = _flatten_named_grads(grad_i, param_names)
    flat_j = _flatten_named_grads(grad_j, param_names)
    if flat_i is None or flat_j is None:
        return False, 1.0
    norm_i = torch.linalg.vector_norm(flat_i)
    norm_j = torch.linalg.vector_norm(flat_j)
    if norm_i <= 1e-12 or norm_j <= 1e-12:
        return False, 1.0
    cos_sim = torch.dot(flat_i, flat_j).div(norm_i * norm_j).clamp(-1.0, 1.0).item()
    return cos_sim < float(conflict_threshold), cos_sim


def _project_gradient(
    grad_target: Dict[str, torch.Tensor],
    grad_reference: Dict[str, torch.Tensor],
    param_names: List[str],
) -> None:
    for name in param_names:
        g_t = grad_target.get(name)
        g_r = grad_reference.get(name)
        if g_t is None or g_r is None:
            continue
        dot = torch.dot(g_t.reshape(-1), g_r.reshape(-1))
        ref_norm_sq = torch.dot(g_r.reshape(-1), g_r.reshape(-1))
        if ref_norm_sq <= 1e-12:
            continue
        grad_target[name] = g_t - (dot / ref_norm_sq) * g_r


def _reduce_gradient_samples(
    gradient_samples: List[Dict[str, torch.Tensor]],
    param_names: List[str],
    reduction: str,
) -> Dict[str, torch.Tensor]:
    reduced: Dict[str, torch.Tensor] = {}
    for name in param_names:
        grads = [grad_map[name] for grad_map in gradient_samples if name in grad_map]
        if not grads:
            continue
        stacked = torch.stack(grads)
        reduced[name] = stacked.mean(dim=0) if reduction == "mean" else stacked.sum(dim=0)
    return reduced


def resolve_pcgrad_gradients(
    per_sample_grads: List[Dict[str, torch.Tensor]],
    conflict_threshold: float = 0.0,
    reduction: str = "mean",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    reduction = _normalize_reduction(reduction)
    input_count = len(per_sample_grads)
    stats: Dict[str, Any] = {
        "input_count": input_count,
        "param_count": 0,
        "total_pairs": 0,
        "conflict_pairs": 0,
        "conflict_rate": 0.0,
        "projections": 0,
        "avg_conflict_angle": 0.0,
        "reduction": reduction,
        "conflict_threshold": float(conflict_threshold),
    }
    if input_count == 0:
        return {}, stats

    param_names = _ordered_param_names(per_sample_grads)
    stats["param_count"] = len(param_names)
    if not param_names:
        return {}, stats

    templates = _template_tensors(per_sample_grads, param_names)
    base_grads = [
        _clone_grad_map(grad_map, param_names, templates)
        for grad_map in per_sample_grads
    ]
    projected_grads = [
        _clone_grad_map(grad_map, param_names, templates)
        for grad_map in per_sample_grads
    ]

    conflict_angles: List[float] = []
    for i in range(input_count):
        for j in range(i + 1, input_count):
            stats["total_pairs"] += 1
            is_conflict, cos_sim = _check_conflict(
                base_grads[i],
                base_grads[j],
                param_names,
                conflict_threshold,
            )
            if not is_conflict:
                continue
            stats["conflict_pairs"] += 1
            conflict_angles.append(math.degrees(math.acos(max(min(cos_sim, 1.0), -1.0))))

    for i in range(input_count):
        for j in range(input_count):
            if i == j:
                continue
            is_conflict, _ = _check_conflict(
                projected_grads[i],
                base_grads[j],
                param_names,
                conflict_threshold,
            )
            if not is_conflict:
                continue
            _project_gradient(projected_grads[i], base_grads[j], param_names)
            stats["projections"] += 1

    if stats["conflict_pairs"] > 0:
        stats["conflict_rate"] = stats["conflict_pairs"] / max(stats["total_pairs"], 1)
        stats["avg_conflict_angle"] = sum(conflict_angles) / len(conflict_angles)
        logger.debug(
            "[PCGrad] Resolved %s pair conflicts with %s projections",
            stats["conflict_pairs"],
            stats["projections"],
        )

    return _reduce_gradient_samples(projected_grads, param_names, reduction), stats


class PCGradOptimizer:
    """
    PCGrad 梯度优化器
    
    包装标准优化器，在更新前处理梯度冲突
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        conflict_threshold: float = 0.0,  # cos < 0 表示冲突
        reduction: str = "mean",  # mean, sum
    ):
        self.optimizer = optimizer
        self.conflict_threshold = conflict_threshold
        self.reduction = reduction
        
        # 统计
        self.stats = PCGradStats()
        self._step_count = 0
    
    def step(self, per_sample_grads: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        执行优化步骤
        
        Args:
            per_sample_grads: 每个样本的梯度列表 [{param_name: grad}, ...]
                              如果为 None，使用标准 backward 梯度
        """
        if per_sample_grads is not None and len(per_sample_grads) > 1:
            # 应用 PCGrad
            resolved_grads = self._resolve_conflicts(per_sample_grads)
            self._apply_grads(resolved_grads)
        
        self.optimizer.step()
        self._step_count += 1
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()
    
    def _resolve_conflicts(
        self,
        per_sample_grads: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        解决梯度冲突
        
        使用 PCGrad 算法投影冲突梯度
        """
        resolved, stats = resolve_pcgrad_gradients(
            per_sample_grads,
            conflict_threshold=self.conflict_threshold,
            reduction=self.reduction,
        )
        self.stats.total_pairs += int(stats.get("total_pairs", 0) or 0)
        self.stats.conflict_pairs += int(stats.get("conflict_pairs", 0) or 0)
        self.stats.projections_applied += int(stats.get("projections", 0) or 0)
        if stats.get("avg_conflict_angle", 0.0):
            self.stats.avg_conflict_angle = float(stats["avg_conflict_angle"])
        return resolved
    
    def _check_conflict(
        self,
        grad_i: Dict[str, torch.Tensor],
        grad_j: Dict[str, torch.Tensor],
        param_names: List[str],
    ) -> Tuple[bool, float]:
        """
        检查两个梯度是否冲突
        
        Returns:
            (is_conflict, cosine_similarity)
        """
        # 将所有参数的梯度展平并拼接
        flat_i = torch.cat([grad_i[n].flatten() for n in param_names])
        flat_j = torch.cat([grad_j[n].flatten() for n in param_names])
        
        # 计算余弦相似度
        cos_sim = torch.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0)).item()
        
        return cos_sim < self.conflict_threshold, cos_sim
    
    def _project_gradient(
        self,
        grad_target: Dict[str, torch.Tensor],
        grad_reference: Dict[str, torch.Tensor],
        param_names: List[str],
    ):
        """
        将 target 梯度投影到 reference 的法平面
        
        公式: g_target' = g_target - (g_target · g_ref / ||g_ref||²) * g_ref
        """
        for name in param_names:
            g_t = grad_target[name]
            g_r = grad_reference[name]
            
            # 计算投影
            dot = torch.dot(g_t.flatten(), g_r.flatten())
            ref_norm_sq = torch.dot(g_r.flatten(), g_r.flatten())
            
            if ref_norm_sq > 1e-8:
                proj = (dot / ref_norm_sq) * g_r
                grad_target[name] = g_t - proj
    
    def _apply_grads(self, grads: Dict[str, torch.Tensor]):
        """应用解决后的梯度到模型"""
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    # 查找对应的解决后梯度
                    for name, resolved_grad in grads.items():
                        if param.shape == resolved_grad.shape:
                            param.grad.copy_(resolved_grad)
                            break
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        conflict_rate = 0.0
        if self.stats.total_pairs > 0:
            conflict_rate = self.stats.conflict_pairs / self.stats.total_pairs
        
        return {
            "step": self._step_count,
            "total_pairs": self.stats.total_pairs,
            "conflict_pairs": self.stats.conflict_pairs,
            "conflict_rate": conflict_rate,
            "projections": self.stats.projections_applied,
        }


class GradientConflictDetector:
    """
    梯度冲突检测器
    
    用于监控 (不修改梯度)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self._gradient_cache: Dict[str, torch.Tensor] = {}
        self._conflicts: List[GradientConflictInfo] = []
    
    def cache_gradient(self, sample_id: int):
        """缓存当前梯度"""
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
        
        self._gradient_cache[sample_id] = grads

        # Auto cleanup cache to prevent memory leak (Cap at 100 samples)
        if len(self._gradient_cache) > 100:
            # Drop the oldest sample (assuming insertion order)
            oldest_key = next(iter(self._gradient_cache))
            del self._gradient_cache[oldest_key]
    
    def detect_conflicts(self, threshold: float = 0.0) -> List[GradientConflictInfo]:
        """检测所有缓存梯度之间的冲突"""
        self._conflicts.clear()
        
        sample_ids = list(self._gradient_cache.keys())
        
        for i, id_i in enumerate(sample_ids):
            for j, id_j in enumerate(sample_ids):
                if i >= j:
                    continue
                
                # 计算余弦相似度
                cos_sim = self._compute_cosine(
                    self._gradient_cache[id_i],
                    self._gradient_cache[id_j],
                )
                
                is_conflict = cos_sim < threshold
                
                self._conflicts.append(GradientConflictInfo(
                    sample_i=id_i,
                    sample_j=id_j,
                    cosine_similarity=cos_sim,
                    is_conflict=is_conflict,
                ))
        
        return self._conflicts
    
    def _compute_cosine(
        self,
        grads_a: Dict[str, torch.Tensor],
        grads_b: Dict[str, torch.Tensor],
    ) -> float:
        """计算两个梯度集合的整体余弦相似度"""
        common_keys = set(grads_a.keys()) & set(grads_b.keys())
        
        if not common_keys:
            return 0.0
        
        flat_a = torch.cat([grads_a[k].flatten() for k in common_keys])
        flat_b = torch.cat([grads_b[k].flatten() for k in common_keys])
        
        return torch.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
    
    def clear_cache(self):
        """清空缓存"""
        self._gradient_cache.clear()
        self._conflicts.clear()
    
    def get_conflict_rate(self) -> float:
        """获取冲突率"""
        if not self._conflicts:
            return 0.0
        
        conflict_count = sum(1 for c in self._conflicts if c.is_conflict)
        return conflict_count / len(self._conflicts)


# ========== 便捷函数 ==========

def wrap_optimizer_with_pcgrad(
    optimizer: torch.optim.Optimizer,
    conflict_threshold: float = 0.0,
) -> PCGradOptimizer:
    """包装优化器为 PCGrad 版本"""
    return PCGradOptimizer(
        optimizer=optimizer,
        conflict_threshold=conflict_threshold,
    )


def compute_per_sample_gradients(
    model: nn.Module,
    batch_inputs: Dict[str, torch.Tensor],
    loss_fn,
) -> List[Dict[str, torch.Tensor]]:
    """
    计算每个样本的独立梯度
    
    注意: 这需要逐样本前向传播，会增加计算成本
    """
    batch_size = next(iter(batch_inputs.values())).shape[0]
    
    per_sample_grads = []
    
    for i in range(batch_size):
        # 提取单个样本
        single_input = {k: v[i:i+1] for k, v in batch_inputs.items()}
        
        # 前向传播
        output = model(**single_input)
        loss = loss_fn(output)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 收集梯度
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
        
        per_sample_grads.append(grads)
    
    return per_sample_grads
