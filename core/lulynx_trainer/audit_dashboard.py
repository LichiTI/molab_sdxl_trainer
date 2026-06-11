"""
Auditor 仪表盘数据

将 LoRAAuditor 的输出转换为可视化友好的格式
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class HealthLevel(Enum):
    """健康等级"""
    EXCELLENT = "excellent"   # 极好
    GOOD = "good"             # 良好
    WARNING = "warning"       # 警告
    CRITICAL = "critical"     # 危险


@dataclass
class AuditScore:
    """审计评分"""
    # 核心评分 (0-100)
    generalization: float = 0.0     # 泛化性评分
    fidelity: float = 0.0           # 保真度评分
    stability: float = 0.0          # 稳定性评分
    efficiency: float = 0.0         # 效率评分
    
    # 综合评分
    overall: float = 0.0
    health_level: HealthLevel = HealthLevel.GOOD
    
    # 建议
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["health_level"] = self.health_level.value
        return result


@dataclass
class LayerDiagnostic:
    """层级诊断"""
    layer_name: str
    layer_type: str  # "unet_down" | "unet_mid" | "unet_up" | "te"
    
    # 指标
    weight_norm: float = 0.0
    gradient_norm: float = 0.0
    svd_entropy: float = 0.0
    rank_usage: float = 0.0
    
    # 状态
    status: str = "normal"  # "normal" | "overfitting" | "underfitting" | "inactive"
    
    # 建议
    suggestion: str = ""


class AuditDashboard:
    """
    审计仪表盘
    
    将 LoRAAuditor 数据转换为前端友好格式
    """
    
    def __init__(self):
        self._history: List[Dict] = []
        self._layer_stats: Dict[str, List[float]] = {}
    
    def process_audit_data(self, audit_result: Dict) -> AuditScore:
        """
        处理审计数据，生成评分
        
        Args:
            audit_result: LoRAAuditor.step() 的返回值
        """
        score = AuditScore()
        
        metrics = audit_result.get("metrics", {})
        svd_data = audit_result.get("svd", {})
        
        # 1. 泛化性评分 (基于 SVD 熵)
        svd_entropy = svd_data.get("mean_entropy", 0.5)
        # 熵越高 = 信息越分散 = 泛化性越好
        # 但极高熵可能意味着没学到东西
        if 0.3 <= svd_entropy <= 0.7:
            score.generalization = 80 + (0.5 - abs(svd_entropy - 0.5)) * 40
        elif svd_entropy < 0.3:
            score.generalization = svd_entropy / 0.3 * 60
            score.suggestions.append("SVD 熵过低，可能过拟合。建议增加 Dropout 或减少训练步数")
        else:
            score.generalization = max(40, 100 - (svd_entropy - 0.7) * 100)
            score.suggestions.append("SVD 熵过高，模型可能欠拟合。建议增加训练步数或提高学习率")
        
        # 2. 保真度评分 (基于 Loss 和收敛速度)
        loss = metrics.get("loss", 0.1)
        loss_trend = metrics.get("loss_trend", 0)  # 负数 = 下降
        
        # Loss 越低越好，但要结合趋势
        if loss < 0.05:
            score.fidelity = min(95, 90 + (0.05 - loss) * 100)
            if loss < 0.01:
                score.suggestions.append("Loss 极低，可能过拟合。建议使用 sample 检查输出质量")
        elif loss < 0.1:
            score.fidelity = 70 + (0.1 - loss) * 400
        else:
            score.fidelity = max(30, 70 - (loss - 0.1) * 200)
        
        # 3. 稳定性评分 (基于 Loss 方差和梯度范数)
        loss_variance = metrics.get("loss_variance", 0.01)
        grad_norm = metrics.get("grad_norm", 1.0)
        
        # 方差越小越稳定
        stability = 100 - min(50, loss_variance * 1000)
        
        # 梯度范数也影响稳定性
        if grad_norm > 10:
            stability -= 20
            score.suggestions.append("梯度范数过大，建议降低学习率或增加梯度裁剪")
        elif grad_norm < 0.01:
            stability -= 10
            score.suggestions.append("梯度范数过小，模型可能已收敛或学习率过低")
        
        score.stability = max(0, min(100, stability))
        
        # 4. 效率评分 (基于 Rank 利用率)
        rank_usage = svd_data.get("rank_usage", 0.5)
        
        # Rank 利用率 50-80% 最佳
        if 0.5 <= rank_usage <= 0.8:
            score.efficiency = 80 + (rank_usage - 0.5) * 66
        elif rank_usage < 0.5:
            score.efficiency = rank_usage * 160
            score.suggestions.append(f"Rank 利用率仅 {rank_usage*100:.0f}%，考虑使用动态剪枝")
        else:
            score.efficiency = max(60, 100 - (rank_usage - 0.8) * 200)
        
        # 5. 综合评分
        score.overall = (
            score.generalization * 0.3 +
            score.fidelity * 0.3 +
            score.stability * 0.2 +
            score.efficiency * 0.2
        )
        
        # 确定健康等级
        if score.overall >= 80:
            score.health_level = HealthLevel.EXCELLENT
        elif score.overall >= 60:
            score.health_level = HealthLevel.GOOD
        elif score.overall >= 40:
            score.health_level = HealthLevel.WARNING
        else:
            score.health_level = HealthLevel.CRITICAL
        
        # 记录历史
        self._history.append(score.to_dict())
        
        return score
    
    def diagnose_layers(self, layer_data: Dict[str, Dict]) -> List[LayerDiagnostic]:
        """
        诊断各层状态
        
        Args:
            layer_data: 层级统计数据
        """
        diagnostics = []
        
        for name, stats in layer_data.items():
            diag = LayerDiagnostic(
                layer_name=name,
                layer_type=self._classify_layer(name),
                weight_norm=stats.get("weight_norm", 0),
                gradient_norm=stats.get("grad_norm", 0),
                svd_entropy=stats.get("svd_entropy", 0.5),
                rank_usage=stats.get("rank_usage", 0.5),
            )
            
            # 判断状态
            if diag.svd_entropy < 0.2:
                diag.status = "overfitting"
                diag.suggestion = "该层可能过拟合，考虑冻结或降低学习率"
            elif diag.svd_entropy > 0.8:
                diag.status = "underfitting"
                diag.suggestion = "该层学习不足，考虑增加学习率"
            elif diag.gradient_norm < 0.001:
                diag.status = "inactive"
                diag.suggestion = "该层几乎无梯度，可能已收敛或被冻结"
            else:
                diag.status = "normal"
            
            diagnostics.append(diag)
        
        return diagnostics
    
    def _classify_layer(self, name: str) -> str:
        """分类层类型"""
        if "down_blocks" in name or "input" in name:
            return "unet_down"
        elif "mid_block" in name:
            return "unet_mid"
        elif "up_blocks" in name or "output" in name:
            return "unet_up"
        elif "text_encoder" in name or "te" in name:
            return "te"
        return "other"
    
    def get_radar_chart_data(self, score: AuditScore) -> Dict[str, Any]:
        """
        获取雷达图数据
        """
        return {
            "labels": ["泛化性", "保真度", "稳定性", "效率"],
            "values": [
                score.generalization,
                score.fidelity,
                score.stability,
                score.efficiency,
            ],
            "max": 100,
        }
    
    def get_trend_data(self) -> Dict[str, List]:
        """
        获取趋势数据
        """
        if len(self._history) < 2:
            return {"steps": [], "overall": [], "generalization": [], "fidelity": []}
        
        return {
            "steps": list(range(len(self._history))),
            "overall": [h["overall"] for h in self._history],
            "generalization": [h["generalization"] for h in self._history],
            "fidelity": [h["fidelity"] for h in self._history],
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        if not self._history:
            return {"status": "no_data"}
        
        latest = self._history[-1]
        return {
            "status": "ok",
            "latest_score": latest,
            "trend": "improving" if len(self._history) > 1 and self._history[-1]["overall"] > self._history[-2]["overall"] else "stable",
            "total_checks": len(self._history),
        }
