import torch
import random
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class LISAScheduler:
    """
    LISA (Layerwise Importance Sampling) 调度器
    
    原理: 在训练过程中定期随机选择一部分层进行更新，冻结其余层。
    这可以显著减少反向传播所需的显存，因为不需要为冻结层存储激活值。
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        active_ratio: float = 0.2, 
        interval: int = 1
    ):
        self.model = model
        self.active_ratio = active_ratio
        self.interval = interval
        self._step_count = 0
        
        # 获取所有可训练的适配器层。
        # Native trainer 中，LoRA/DoRA 都挂在 LoRALinear.lora 上。
        # 直接抓 adapter 可以避免：
        # 1) 标准 LoRA 同时命中父层和 lora_down/lora_up 子层，导致重复/覆盖切换
        # 2) DoRA 只有直接参数 (lora_A/lora_B/m)，按旧规则完全识别不到
        self.layers: List[torch.nn.Module] = []
        seen = set()
        for module in model.modules():
            adapter = getattr(module, "lora", None)
            if adapter is None or not isinstance(adapter, torch.nn.Module):
                continue

            adapter_id = id(adapter)
            if adapter_id in seen:
                continue

            if not any(param.requires_grad for param in adapter.parameters()):
                continue

            self.layers.append(adapter)
            seen.add(adapter_id)
        
        if not self.layers:
            logger.warning("[LISA] 未在模型中找到可训练的层，LISA 将不生效")
        else:
            logger.info(f"[LISA] 已初始化，总计 {len(self.layers)} 层，激活比例 {active_ratio}")

    def step(self):
        """每步调用，根据 interval 切换激活层"""
        if not self.layers:
            return
            
        if self._step_count % self.interval == 0:
            self._update_active_layers()
        
        self._step_count += 1

    def _update_active_layers(self):
        """随机选择并激活层"""
        num_active = max(1, int(len(self.layers) * self.active_ratio))
        active_indices = set(random.sample(range(len(self.layers)), num_active))
        
        for i, layer in enumerate(self.layers):
            is_active = i in active_indices
            # 设置 requires_grad
            for param in layer.parameters():
                param.requires_grad = is_active
                
        # logger.debug(f"[LISA] Step {self._step_count}: 激活了 {num_active} 个层")

    def reset(self):
        """重置所有层为激活状态（例如在保存模型前调用）"""
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
