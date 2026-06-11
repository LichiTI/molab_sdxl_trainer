# Proxy Regularizer
# 基于模型浓缩代理的抗遗忘正则化

import torch
import json
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ProxyRegularizer:
    """
    Proxy Regularizer: 基于模型统计指纹的抗遗忘正则化
    
    原理:
    在训练过程中，强制模型权重的统计分布（均值、范数、奇异值谱）
    与原始基座模型保持相似，从而防止灾难性遗忘。
    
    用途:
    1. 防止 LoRA 训练时"忘记画手"等问题
    2. 在分块训练中维持全局一致性
    3. 轻量级 EWC (Elastic Weight Consolidation) 替代方案
    
    集成方式:
    1. 训练前：加载 .proxy.json 文件
    2. 训练中：每 N 步计算正则化损失
    3. 在总损失中添加 loss_reg = lambda * proxy_regularizer.compute_loss(model)
    """
    
    def __init__(
        self,
        proxy_path: str,
        lambda_norm: float = 0.01,      # 范数正则化强度
        lambda_mean: float = 0.001,     # 均值正则化强度
        lambda_spectrum: float = 0.005, # 奇异值谱正则化强度
        enabled_layers: Optional[list] = None,  # 只对指定层应用 (None = 全部)
        update_interval: int = 10       # 每 N 步计算一次
    ):
        """
        Args:
            proxy_path: .proxy.json 文件路径
            lambda_norm: 范数正则化强度
            lambda_mean: 均值正则化强度
            lambda_spectrum: 奇异值谱正则化强度 (计算昂贵，默认较低)
            enabled_layers: 只对指定层应用正则化
            update_interval: 更新间隔
        """
        self.lambda_norm = lambda_norm
        self.lambda_mean = lambda_mean
        self.lambda_spectrum = lambda_spectrum
        self.enabled_layers = set(enabled_layers) if enabled_layers else None
        self.update_interval = update_interval
        
        # 加载 proxy
        self.proxy = self._load_proxy(proxy_path)
        self._step = 0
        self._cached_loss = torch.tensor(0.0)
        self._last_matched_layers = 0
        self._last_missing_layers = 0
        self._last_loss_value = 0.0
        self._last_computed = False
        
    def _load_proxy(self, path: str) -> Dict[str, Any]:
        """加载 proxy 文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"[ProxyRegularizer] Checkpoint load failed: {e}")
            return {}
    
    def should_compute(self, step: int) -> bool:
        """是否应该在这一步计算正则化损失"""
        return step % self.update_interval == 0

    def _zero_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Return a graph-safe zero tensor on the model's active device."""
        for param in model.parameters():
            if param.requires_grad:
                return param.new_zeros(())
        try:
            return next(model.parameters()).new_zeros(())
        except StopIteration:
            return torch.tensor(0.0)
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        step: int = None
    ) -> torch.Tensor:
        """
        计算正则化损失
        
        Args:
            model: 当前模型 (含 LoRA)
            step: 当前训练步数 (用于间隔控制)
        
        Returns:
            正则化损失张量
        """
        if step is not None:
            self._step = step
            if not self.should_compute(step):
                self._last_computed = False
                return self._zero_loss(model)
        
        total_loss = self._zero_loss(model)
        layer_count = 0
        self._last_missing_layers = 0
        
        layers_dict = self.proxy.get('layers', {})
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 检查是否在启用列表中
            if self.enabled_layers and name not in self.enabled_layers:
                continue
            
            # 查找对应的 proxy 统计
            proxy_stats = layers_dict.get(name)
            if proxy_stats is None:
                self._last_missing_layers += 1
                continue
            
            layer_count += 1
            
            # --- 自动适配统计粒度 ---
            
            # 1. 专家级：逐通道 MSE 约束 (Dimension-level Pro)
            if 'per_channel_mean' in proxy_stats:
                try:
                    t_flat = param.view(param.size(0), -1)
                    current_per_channel_mean = t_flat.mean(dim=1)
                    target_per_channel_mean = torch.tensor(
                        proxy_stats['per_channel_mean'], 
                        device=param.device, 
                        dtype=param.dtype
                    )
                    channel_loss = (current_per_channel_mean - target_per_channel_mean).pow(2).mean()
                    total_loss = total_loss + self.lambda_mean * 10.0 * channel_loss
                except Exception as e:
                    logger.debug(f"ProxyReg per_channel failed for {name}: {e}")

            # 2. 均衡级：直方图分布快照约束 (Distribution Sketch)
            elif 'dist_sketch' in proxy_stats:
                try:
                    # 计算当前权重的直方图分布 (归一化)
                    # 我们遵循 model_condenser 的 16-bins 和固定范围
                    # torch.histc is not differentiable, so this sketch remains
                    # a monitoring-only proxy. Differentiable mean/norm losses
                    # below still provide training signal when present.
                    hist = torch.histc(param.detach().float(), bins=16, min=-0.1, max=0.1)
                    current_sketch = hist / (hist.sum() + 1e-10)
                    target_sketch = torch.tensor(
                        proxy_stats['dist_sketch'], 
                        device=param.device, 
                        dtype=param.dtype
                    )
                    sketch_loss = (current_sketch - target_sketch).pow(2).mean()
                    self._cached_loss = sketch_loss.detach()
                except Exception as e:
                    logger.debug(f"ProxyReg dist_sketch failed for {name}: {e}")

            # 3. 基础级：全局均值/范数约束 (Always Fallback)
            # 范数正则化: ||W||_F 不应偏离太多
            target_norm_value = proxy_stats.get('l2_norm', proxy_stats.get('frobenius'))
            if self.lambda_norm > 0 and target_norm_value is not None:
                current_norm = param.norm()
                target_norm = param.new_tensor(target_norm_value)
                norm_loss = (current_norm - target_norm).pow(2)
                total_loss = total_loss + self.lambda_norm * norm_loss
            
            # 均值正则化: mean(W) 不应偏移
            if self.lambda_mean > 0 and 'mean' in proxy_stats:
                current_mean = param.mean()
                target_mean = param.new_tensor(proxy_stats['mean'])
                mean_loss = (current_mean - target_mean).pow(2)
                total_loss = total_loss + self.lambda_mean * mean_loss
            
            # 4. 结构级：奇异值谱正则化 (如果不分档位，独立存在)
            if self.lambda_spectrum > 0 and 'spectrum' in proxy_stats:
                if param.dim() >= 2:
                    try:
                        # 仅对足够大的层计算（防止由于小层导致的 SVD 性能损耗）
                        if param.size(0) >= 128:
                            p = param.view(param.size(0), -1)
                            # 使用 svdvals 进行快速计算
                            current_s = torch.linalg.svdvals(p.float())
                            target_s = torch.tensor(
                                proxy_stats['spectrum'],
                                device=param.device,
                                dtype=torch.float32
                            )
                            
                            min_len = min(len(current_s), len(target_s))
                            if min_len > 0:
                                current_s = current_s[:min_len] / (current_s[:min_len].sum() + 1e-8)
                                target_s = target_s[:min_len] / (target_s[:min_len].sum() + 1e-8)
                                spectrum_loss = (current_s - target_s).pow(2).sum()
                                total_loss = total_loss + self.lambda_spectrum * spectrum_loss
                    except Exception as e:
                        logger.debug(f"ProxyReg spectrum failed for {name}: {e}")
        
        # 按层数归一化
        if layer_count > 0:
            total_loss = total_loss / layer_count
        
        self._cached_loss = total_loss.detach()
        self._last_matched_layers = layer_count
        self._last_loss_value = float(self._cached_loss.detach().cpu())
        self._last_computed = True
        return total_loss
    
    def get_layer_deviations(self, model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
        """
        获取每层的偏离程度 (用于监控)
        
        Returns:
            {layer_name: {norm_dev, mean_dev, spectrum_dev}}
        """
        deviations = {}
        layers_dict = self.proxy.get('layers', {})
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            proxy_stats = layers_dict.get(name)
            if proxy_stats is None:
                continue
            
            dev = {}
            
            target_norm = proxy_stats.get('frobenius', proxy_stats.get('l2_norm'))
            if target_norm is not None:
                current = param.detach().norm().item()
                dev['norm_dev'] = abs(current - target_norm) / (target_norm + 1e-8)
            
            if 'mean' in proxy_stats:
                current = param.detach().mean().item()
                target = proxy_stats['mean']
                dev['mean_dev'] = abs(current - target)
            
            deviations[name] = dev
        
        return deviations

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        return {
            "step": int(self._step),
            "computed": bool(self._last_computed),
            "matched_layers": int(self._last_matched_layers),
            "missing_layers": int(self._last_missing_layers),
            "last_loss": float(self._last_loss_value),
            "update_interval": int(self.update_interval),
            "lambda_norm": float(self.lambda_norm),
            "lambda_mean": float(self.lambda_mean),
            "lambda_spectrum": float(self.lambda_spectrum),
            "proxy_layers": int(len(self.proxy.get("layers", {}))),
        }


class AdaptiveProxyRegularizer(ProxyRegularizer):
    """
    自适应 Proxy Regularizer
    
    根据层的"重要性"动态调整正则化强度：
    - 方差小的层 = 更重要 = 更强正则化
    - 方差大的层 = 更灵活 = 更弱正则化
    """
    
    def __init__(
        self,
        proxy_path: str,
        base_lambda: float = 0.01,
        importance_scale: float = 2.0,
        **kwargs
    ):
        super().__init__(proxy_path, **kwargs)
        self.base_lambda = base_lambda
        self.importance_scale = importance_scale
        
        # 预计算每层的重要性
        self._importance = self._compute_importance()
    
    def _compute_importance(self) -> Dict[str, float]:
        """根据 std 计算层重要性 (std 越小越重要)"""
        importance = {}
        layers_dict = self.proxy.get('layers', {})
        
        # 收集所有 std
        stds = []
        for name, stats in layers_dict.items():
            if 'std' in stats:
                stds.append(stats['std'])
        
        if not stds:
            return importance
        
        # 归一化
        mean_std = sum(stds) / len(stds)
        
        for name, stats in layers_dict.items():
            if 'std' in stats:
                # std 越小，重要性越高
                # importance = mean_std / (std + eps)
                importance[name] = mean_std / (stats['std'] + 1e-8)
        
        return importance
    
    def compute_loss(self, model: torch.nn.Module, step: int = None) -> torch.Tensor:
        """使用自适应权重计算正则化损失"""
        if step is not None:
            self._step = step
            if not self.should_compute(step):
                return self._zero_loss(model)
        
        total_loss = self._zero_loss(model)
        layer_count = 0
        
        layers_dict = self.proxy.get('layers', {})
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            proxy_stats = layers_dict.get(name)
            if proxy_stats is None:
                continue
            
            # 获取该层的重要性系数
            importance = self._importance.get(name, 1.0)
            layer_lambda = self.base_lambda * (1 + self.importance_scale * (importance - 1))
            
            layer_count += 1
            
            # 范数正则化
            if 'frobenius' in proxy_stats:
                current_norm = param.norm()
                target_norm = param.new_tensor(proxy_stats['frobenius'])
                norm_loss = (current_norm - target_norm).pow(2)
                total_loss = total_loss + layer_lambda * norm_loss
        
        if layer_count > 0:
            total_loss = total_loss / layer_count
        
        self._cached_loss = total_loss.detach()
        return total_loss


# 工厂函数
def create_proxy_regularizer(
    proxy_path: str,
    adaptive: bool = False,
    **kwargs
) -> ProxyRegularizer:
    """
    创建 Proxy Regularizer
    
    Args:
        proxy_path: .proxy.json 文件路径
        adaptive: 是否使用自适应版本
        **kwargs: 传递给构造函数的参数
    """
    if adaptive:
        return AdaptiveProxyRegularizer(proxy_path, **kwargs)
    return ProxyRegularizer(proxy_path, **kwargs)
