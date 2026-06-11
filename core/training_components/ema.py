"""
EMA (Exponential Moving Average) 训练

平滑权重更新，提高泛化能力
"""

import torch
import torch.nn as nn
import copy
import logging
from typing import Optional, Dict, Any
from core.safe_pickle import safe_torch_load

logger = logging.getLogger(__name__)


class EMAModel:
    """
    EMA 权重平均
    
    公式: shadow = decay * shadow + (1 - decay) * model
    
    优点:
    - 平滑训练过程
    - 减少过拟合
    - 提高最终模型质量
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        device: Optional[str] = None,
        use_ema_warmup: bool = True,
        inv_gamma: float = 1.0,
        power: float = 2/3,
    ):
        """
        Args:
            model: 源模型
            decay: EMA 衰减率
            update_after_step: 多少步后开始 EMA
            update_every: 每 N 步更新一次
            device: 存储设备
            use_ema_warmup: 是否使用 warmup
            inv_gamma, power: warmup 参数
        """
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        
        self.optimization_step = 0
        self.cur_decay_value = 0.0
        
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"Decay must be between 0 and 1, got {decay}")

        # 创建影子模型
        try:
            self.shadow = copy.deepcopy(model)
        except Exception as e:
            logger.error(f"Failed to deepcopy model for EMA: {e}")
            raise
        self.shadow.requires_grad_(False)
        
        if device is not None:
            self.shadow.to(device)
        
        logger.info(f"[EMAModel] Initialized with decay={decay}")
    
    def get_decay(self, optimization_step: int) -> float:
        """获取当前衰减率 (支持 warmup)"""
        step = max(0, optimization_step - self.update_after_step)
        
        if step <= 0:
            return 0.0
        
        if self.use_ema_warmup:
            # Warmup 公式
            cur_decay = 1 - (1 + step / self.inv_gamma) ** (-self.power)
            cur_decay = min(cur_decay, self.decay)
        else:
            cur_decay = self.decay
        
        return cur_decay
    
    @torch.no_grad()
    def step(self, model: nn.Module):
        """更新 EMA"""
        self.optimization_step += 1
        
        if self.optimization_step % self.update_every != 0:
            return
        
        self.cur_decay_value = self.get_decay(self.optimization_step)
        
        if self.cur_decay_value == 0.0:
            # 还没开始 EMA
            self._copy_weights(model)
            return
        
        # EMA 更新
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.cur_decay_value)
            s_param.data.add_(m_param.data, alpha=1 - self.cur_decay_value)
    
    @torch.no_grad()
    def _copy_weights(self, model: nn.Module):
        """复制权重"""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.copy_(m_param.data)
    
    def copy_to(self, model: nn.Module):
        """将 EMA 权重复制到模型"""
        for m_param, s_param in zip(model.parameters(), self.shadow.parameters()):
            m_param.data.copy_(s_param.data)
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "decay": self.decay,
            "optimization_step": self.optimization_step,
            "shadow": self.shadow.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态"""
        self.decay = state_dict["decay"]
        self.optimization_step = state_dict["optimization_step"]
        self.shadow.load_state_dict(state_dict["shadow"])
    
    def eval(self):
        """设置为评估模式"""
        self.shadow.eval()
    
    def train(self):
        """设置为训练模式 (影子模型通常不需要)"""
        pass
    
    @property
    def module(self) -> nn.Module:
        """获取影子模型"""
        return self.shadow


class EMACallback:
    """
    EMA 训练回调
    
    用法:
    ```python
    ema_callback = EMACallback(model, decay=0.9999)
    
    for batch in dataloader:
        loss = train_step(batch)
        ema_callback.on_step_end()
    
    # 使用 EMA 模型进行推理
    ema_model = ema_callback.get_ema_model()
    ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_every: int = 1,
    ):
        self.ema = EMAModel(model, decay=decay, update_every=update_every)
        self.model = model
    
    def on_step_end(self):
        """训练步结束时调用"""
        self.ema.step(self.model)
    
    def get_ema_model(self) -> nn.Module:
        """获取 EMA 模型"""
        return self.ema.module
    
    def save_ema(self, path: str):
        """保存 EMA 权重"""
        torch.save(self.ema.state_dict(), path)
    
    def load_ema(self, path: str):
        """加载 EMA 权重"""
        state_dict = safe_torch_load(path, map_location="cpu")
        self.ema.load_state_dict(state_dict)


# ========== 便捷函数 ==========

def create_ema(
    model: nn.Module,
    decay: float = 0.9999,
) -> EMAModel:
    """创建 EMA 模型"""
    return EMAModel(model, decay=decay)


def create_ema_callback(
    model: nn.Module,
    decay: float = 0.9999,
) -> EMACallback:
    """创建 EMA 回调"""
    return EMACallback(model, decay=decay)


class EMAStateTracker:
    """针对 LoRA / LyCORIS 状态字典的轻量 EMA 跟踪器。"""

    def __init__(
        self,
        initial_state: Dict[str, torch.Tensor],
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        device: str = "cpu",
        use_ema_warmup: bool = True,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
    ):
        if not initial_state:
            raise ValueError("EMAStateTracker requires a non-empty initial_state")
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"Decay must be between 0 and 1, got {decay}")

        self.decay = float(decay)
        self.update_after_step = max(int(update_after_step), 0)
        self.update_every = max(int(update_every), 1)
        self.device = device
        self.use_ema_warmup = bool(use_ema_warmup)
        self.inv_gamma = max(float(inv_gamma), 1e-6)
        self.power = float(power)
        self.optimization_step = 0
        self.cur_decay_value = 0.0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.param_dtypes: Dict[str, torch.dtype] = {}
        self._copy_from_state(initial_state, clone=True)

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step)
        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay = 1 - (1 + step / self.inv_gamma) ** (-self.power)
            cur_decay = min(cur_decay, self.decay)
        else:
            cur_decay = self.decay

        return cur_decay

    @torch.no_grad()
    def _copy_from_state(self, state_dict: Dict[str, torch.Tensor], clone: bool = False):
        for name, tensor in state_dict.items():
            detached = tensor.detach()
            self.param_dtypes[name] = detached.dtype
            prepared = detached.to(self.device, dtype=torch.float32)
            self.shadow[name] = prepared.clone() if clone else prepared

    @torch.no_grad()
    def step(self, state_dict: Dict[str, torch.Tensor]):
        if not state_dict:
            return

        self.optimization_step += 1
        if self.optimization_step % self.update_every != 0:
            return

        self.cur_decay_value = self.get_decay(self.optimization_step)

        for name, tensor in state_dict.items():
            current = tensor.detach().to(self.device, dtype=torch.float32)
            shadow = self.shadow.get(name)

            if shadow is None or shadow.shape != current.shape:
                self.shadow[name] = current.clone()
                self.param_dtypes[name] = tensor.dtype
                continue

            if self.cur_decay_value == 0.0:
                shadow.copy_(current)
            else:
                shadow.mul_(self.cur_decay_value)
                shadow.add_(current, alpha=1 - self.cur_decay_value)

            self.param_dtypes[name] = tensor.dtype

    def get_ema_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: tensor.detach().clone().to(dtype=self.param_dtypes.get(name, tensor.dtype), device="cpu")
            for name, tensor in self.shadow.items()
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "device": self.device,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "optimization_step": self.optimization_step,
            "cur_decay_value": self.cur_decay_value,
            "shadow": self.get_ema_state_dict(),
            "param_dtypes": {name: str(dtype) for name, dtype in self.param_dtypes.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.decay = float(state_dict.get("decay", self.decay))
        self.update_after_step = int(state_dict.get("update_after_step", self.update_after_step))
        self.update_every = max(int(state_dict.get("update_every", self.update_every)), 1)
        self.device = state_dict.get("device", self.device)
        self.use_ema_warmup = bool(state_dict.get("use_ema_warmup", self.use_ema_warmup))
        self.inv_gamma = max(float(state_dict.get("inv_gamma", self.inv_gamma)), 1e-6)
        self.power = float(state_dict.get("power", self.power))
        self.optimization_step = int(state_dict.get("optimization_step", 0))
        self.cur_decay_value = float(state_dict.get("cur_decay_value", 0.0))
        self._copy_from_state(state_dict.get("shadow", {}), clone=True)
