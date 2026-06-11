
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    CONSERVATIVE = "conservative" # 稳健下潜
    BREAKOUT = "breakout"         # 流形突破
    PRECISION = "precision"       # 精度雕刻

@dataclass
class StrategyCard:
    """策略卡片数据结构"""
    id: str
    type: StrategyType
    title: str
    description: str
    
    # Visual Indicators
    predicted_loss_trend: str # "↓↓", "→", "↑"
    risk_level: str           # "Low", "Medium", "High"
    
    # Parameter Adjustments (Relative or Absolute)
    params: Dict[str, Any]
    
    # Internal Score (for ranking)
    confidence_score: float

class BayesianAdvisor:
    """
    MN-LoRA Bayesian Advisor (Pit Stop Engine)
    
    不负责实时控制，只负责在 'Pit Stop' (暂停) 时提供建议。
    输入：历史 Loss 曲线, Manifold Telemetry (Stable Rank, Drift)
    输出：3 张策略卡片
    """
    
    def __init__(self, history_window: int = 50):
        self.history_window = history_window
        # Simple buffer for recent stats
        self.loss_history: List[float] = []
        self.drift_history: List[float] = []
        self.rank_history: List[float] = []
        
    def push_telemetry(self, loss: float, drift: float, stable_rank: float):
        """接收遥测数据"""
        self.loss_history.append(loss)
        self.drift_history.append(drift)
        self.rank_history.append(stable_rank)
        
        # Keep window
        if len(self.loss_history) > self.history_window:
            self.loss_history.pop(0)
            self.drift_history.pop(0)
            self.rank_history.pop(0)
            
    def analyze_state(self) -> Dict[str, Any]:
        """分析当前训练状态"""
        if len(self.loss_history) < 10:
            return {"status": "insufficient_data"}
            
        # 1. Trend Analysis (Linear Regression on Loss)
        y = np.array(self.loss_history)
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        
        # 2. Stability Analysis (Variance)
        loss_var = np.var(y)
        
        # 3. Manifold Stability
        drift_avg = np.mean(self.drift_history[-10:]) if self.drift_history else 0.0
        rank_avg = np.mean(self.rank_history[-10:]) if self.rank_history else 0.0
        
        state = "stable"
        if slope > 0:
            state = "diverging"
        elif slope > -0.0001 and loss_var < 0.001:
            state = "plateau" # 平台期
            
        return {
            "slope": slope,
            "variance": loss_var,
            "drift_avg": drift_avg,
            "rank_avg": rank_avg,
            "state": state
        }
        
    def generate_strategies(self, current_params: Dict[str, Any]) -> List[StrategyCard]:
        """生成 3 张策略卡片"""
        state = self.analyze_state()
        
        is_plateau = state.get("state") == "plateau"
        drift_high = state.get("drift_avg", 0) > 0.2
        
        cards = []
        
        # 1. Conservative (Always available)
        cards.append(StrategyCard(
            id="strat_conservative",
            type=StrategyType.CONSERVATIVE,
            title="稳健下潜 (Conservative)",
            description="当前趋势稳定。微调学习率衰减，进一步巩固收敛成果。",
            predicted_loss_trend="↓",
            risk_level="Low",
            params={
                "learning_rate": current_params.get("learning_rate", 1e-4) * 0.9,
                "k_ratio": current_params.get("k_ratio", 0.5), # Keep
                "weight_decay": current_params.get("weight_decay", 0.1) * 1.1 # Slightly stronger reg
            },
            confidence_score=0.9
        ))
        
        # 2. Breakout (If plateau or high drift needed)
        # If plateau, we recommend HIGH jump.
        breakout_lr_scale = 1.5 if is_plateau else 1.2
        cards.append(StrategyCard(
            id="strat_breakout",
            type=StrategyType.BREAKOUT,
            title="流形突破 (Breakout)",
            description="检测到 Loss 平台期。" if is_plateau else "尝试突破当前局部最优。" + " 大幅提升 k_ratio 和 LR，寻找新极小值。",
            predicted_loss_trend="↕",
            risk_level="High",
            params={
                "learning_rate": current_params.get("learning_rate", 1e-4) * breakout_lr_scale,
                "k_ratio": min(1.0, current_params.get("k_ratio", 0.5) + 0.2), # Increase subspace
                "weight_decay": current_params.get("weight_decay", 0.1) * 0.8 # Loosen reg
            },
            confidence_score=0.7 if is_plateau else 0.4
        ))
        
        # 3. Precision (For high rank/drift control)
        cards.append(StrategyCard(
            id="strat_precision",
            type=StrategyType.PRECISION,
            title="精度雕刻 (Precision)",
            description="收紧子空间投影，专注于优化主成分纹理细节。适合收尾。",
            predicted_loss_trend="→",
            risk_level="Very Low",
            params={
                "learning_rate": current_params.get("learning_rate", 1e-4) * 0.8,
                "k_ratio": max(0.1, current_params.get("k_ratio", 0.5) - 0.15), # Tighten subspace
                "update_interval": 200 # Update less frequently
            },
            confidence_score=0.8
        ))
        
        return cards
