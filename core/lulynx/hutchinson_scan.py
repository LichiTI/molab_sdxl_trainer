"""
Hutchinson 迹估算器 (模型 X 光机)

在 30 秒内扫描整个模型，估算各层的信息熵/秩。
用于自动推荐冻结层和训练层策略。

技术原理:
Tr(W^T W) ≈ (1/m) Σ ||W v_i||²
其中 v_i 是 Rademacher 分布 (±1) 的随机探测向量
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("HutchinsonScan")


@dataclass
class LayerScanResult:
    """单层扫描结果"""
    name: str
    trace: float              # Tr(W^T W) 估算值
    entropy: float            # 归一化信息熵 (0~1)
    param_count: int          # 参数数量
    recommendation: str       # 推荐策略: "freeze", "train", "constrain"
    

class HutchinsonScanner:
    """
    Hutchinson 迹估算器 - 快速模型 X 光扫描
    
    使用方式:
        scanner = HutchinsonScanner(num_probes=30)
        results = scanner.scan(model)
        heatmap = scanner.generate_heatmap()
    """
    
    def __init__(
        self,
        num_probes: int = 30,
        device: str = "cuda"
    ):
        """
        Args:
            num_probes: 探测向量数量 (越多越准确，默认 30)
            device: 计算设备
        """
        self.num_probes = num_probes
        self.device = device
        
        # 扫描结果
        self._results: List[LayerScanResult] = []
        self._max_trace: float = 0
    
    def _rademacher_vector(self, size: int) -> torch.Tensor:
        """生成 Rademacher 分布随机向量 (±1)"""
        return torch.randint(0, 2, (size,), device=self.device).float() * 2 - 1
    
    def _estimate_trace(self, weight: torch.Tensor) -> float:
        """
        使用 Hutchinson 估算器计算 Tr(W^T W)
        
        Tr(A) ≈ (1/m) Σ v_i^T A v_i = (1/m) Σ ||W v_i||²
        """
        weight = weight.to(dtype=torch.float32)

        # Flatten to 2D: [out_features, in_features] for any dimension
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)  # [1, n]
        elif weight.dim() > 2:
            weight = weight.view(weight.size(0), -1)  # [out, in*k*k...]
        # weight.dim() == 2 is already correct
        
        in_features = weight.size(1)
        
        trace_sum = 0.0
        for _ in range(self.num_probes):
            v = self._rademacher_vector(in_features)
            Wv = torch.mv(weight, v)
            trace_sum += (Wv ** 2).sum().item()
        
        return trace_sum / self.num_probes
    
    def scan(self, model: nn.Module) -> List[LayerScanResult]:
        """
        扫描模型所有可训练层
        
        Args:
            model: 目标模型
        
        Returns:
            results: 各层扫描结果列表
        """
        self._results.clear()
        self._max_trace = 0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 1:
                continue
            
            # 估算迹
            trace = self._estimate_trace(param.data)
            self._max_trace = max(self._max_trace, trace)
            
            self._results.append(LayerScanResult(
                name=name,
                trace=trace,
                entropy=0,  # 稍后归一化
                param_count=param.numel(),
                recommendation="",  # 稍后决策
            ))
        
        # 归一化并生成推荐
        if self._max_trace > 0:
            for result in self._results:
                # 归一化到 0~1
                result.entropy = result.trace / (self._max_trace + 1e-8)
                
                # 推荐策略
                if result.entropy < 0.2:
                    result.recommendation = "freeze"     # 低熵 = 骨架层
                elif result.entropy > 0.7:
                    result.recommendation = "train"      # 高熵 = 皮肉层
                else:
                    result.recommendation = "constrain"  # 中等 = 需要约束
        
        return self._results
    
    def generate_heatmap(self) -> Dict[str, Dict]:
        """
        生成热力图数据 (供前端可视化)
        
        Returns:
            heatmap: {
                "layers": [{"name": ..., "entropy": ..., "recommendation": ...}],
                "summary": {"freeze": N, "train": M, "constrain": K}
            }
        """
        summary = {"freeze": 0, "train": 0, "constrain": 0}
        layers = []
        
        for result in self._results:
            layers.append({
                "name": result.name,
                "entropy": round(result.entropy, 4),
                "trace": round(result.trace, 2),
                "params": result.param_count,
                "recommendation": result.recommendation,
                "color": self._entropy_to_color(result.entropy),
            })
            summary[result.recommendation] = summary.get(result.recommendation, 0) + 1
        
        return {
            "layers": layers,
            "summary": summary,
            "total_layers": len(layers),
        }
    
    def _entropy_to_color(self, entropy: float) -> str:
        """将熵值转换为颜色代码"""
        if entropy < 0.2:
            return "#ef4444"  # 红色 = 冻结
        elif entropy < 0.4:
            return "#f97316"  # 橙色
        elif entropy < 0.6:
            return "#eab308"  # 黄色
        elif entropy < 0.8:
            return "#84cc16"  # 黄绿
        else:
            return "#22c55e"  # 绿色 = 训练
    
    def recommend_freeze_layers(self) -> List[str]:
        """返回推荐冻结的层名称列表"""
        return [r.name for r in self._results if r.recommendation == "freeze"]
    
    def recommend_train_layers(self) -> List[str]:
        """返回推荐训练的层名称列表"""
        return [r.name for r in self._results if r.recommendation == "train"]
    
    def recommend_constrain_layers(self) -> List[str]:
        """返回推荐约束的层名称列表"""
        return [r.name for r in self._results if r.recommendation == "constrain"]
    
    def get_stats(self) -> Dict:
        """获取扫描统计信息"""
        if not self._results:
            return {}
        
        traces = [r.trace for r in self._results]
        entropies = [r.entropy for r in self._results]
        
        return {
            "num_layers": len(self._results),
            "trace_mean": np.mean(traces),
            "trace_std": np.std(traces),
            "trace_max": max(traces),
            "entropy_mean": np.mean(entropies),
            "freeze_count": len(self.recommend_freeze_layers()),
            "train_count": len(self.recommend_train_layers()),
            "constrain_count": len(self.recommend_constrain_layers()),
        }
