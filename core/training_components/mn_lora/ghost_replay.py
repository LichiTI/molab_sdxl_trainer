# Ghost Replay Extension
# 输出空间一致性约束：防止微调偏离原始模型行为

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from core.safe_pickle import safe_torch_load


import logging

logger = logging.getLogger(__name__)


def _timestep_value(timestep: Any) -> float:
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return 0.0
        return float(timestep.detach().float().flatten()[0].cpu())
    try:
        return float(timestep)
    except Exception:
        return 0.0


def _timestep_bucket(timestep: Any, num_train_timesteps: int = 1000) -> str:
    max_step = max(1.0, float(num_train_timesteps or 1000))
    ratio = max(0.0, min(1.0, _timestep_value(timestep) / max_step))
    if ratio < 1.0 / 3.0:
        return "low"
    if ratio < 2.0 / 3.0:
        return "mid"
    return "high"


class ReferenceOutputCache:
    """
    参考输出缓存
    
    支持多模态输入（字典形式），适配 Diffusion UNet 等复杂接口。
    """
    
    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self.samples: List[Dict[str, Any]] = [] # 存储包含 inputs, refs, timesteps 的字典
        
    def add_sample(
        self,
        inputs: Dict[str, torch.Tensor],
        reference_output: torch.Tensor,
        timestep: int = 0
    ):
        """添加一个参考样本"""
        if len(self.samples) >= self.cache_size:
            self.samples.pop(0)
        
        # 深度脱离并移动到 CPU
        detached_inputs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        sample = {
            'inputs': detached_inputs,
            'refs': reference_output.detach().cpu(),
            'timestep': timestep
        }
        self.samples.append(sample)
    
    def get_timestep_bucket_counts(self, num_train_timesteps: int = 1000) -> Dict[str, int]:
        counts = {"low": 0, "mid": 0, "high": 0}
        for sample in self.samples:
            bucket = _timestep_bucket(sample.get("timestep", 0), num_train_timesteps)
            counts[bucket] += 1
        return counts

    def _balanced_indices(self, batch_size: int, num_train_timesteps: int) -> List[int]:
        buckets: Dict[str, List[int]] = {"low": [], "mid": [], "high": []}
        for index, sample in enumerate(self.samples):
            bucket = _timestep_bucket(sample.get("timestep", 0), num_train_timesteps)
            buckets[bucket].append(index)
        for values in buckets.values():
            if values:
                perm = torch.randperm(len(values)).tolist()
                values[:] = [values[i] for i in perm]

        selected: List[int] = []
        bucket_order = ["low", "mid", "high"]
        cursor = 0
        while len(selected) < min(batch_size, len(self.samples)):
            progressed = False
            for _ in bucket_order:
                bucket = bucket_order[cursor % len(bucket_order)]
                cursor += 1
                if buckets[bucket]:
                    selected.append(buckets[bucket].pop(0))
                    progressed = True
                    break
            if not progressed:
                break
        return selected

    def get_batch(
        self,
        batch_size: int = 8,
        device: str = 'cuda',
        timestep_strategy: str = "random",
        num_train_timesteps: int = 1000,
    ) -> List[Dict[str, Any]]:
        """获取一批参考样本"""
        if len(self.samples) == 0:
            return []

        if str(timestep_strategy or "random").lower() == "balanced":
            indices = self._balanced_indices(batch_size, num_train_timesteps)
        else:
            indices = torch.randperm(len(self.samples))[:batch_size].tolist()
        batch = []
        for i in indices:
            s = self.samples[i]
            # 移动回目标设备
            batch.append({
                'inputs': {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in s['inputs'].items()},
                'refs': s['refs'].to(device),
                'timestep': s['timestep']
            })
        return batch
    
    def __len__(self):
        return len(self.samples)
    
    def save(self, path: str):
        try:
            torch.save({
                'samples': self.samples,
                'cache_size': self.cache_size,
            }, path)
        except Exception as e:
            logger.error(f"[GhostReplay] Failed to save cache to {path}: {e}")
    
    @classmethod
    def load(cls, path: str) -> 'ReferenceOutputCache':
        # NOTE: This cache contains a list of dicts (tensors + basic types),
        # so `weights_only=True` may fail on newer Torch versions.
        data = safe_torch_load(path, map_location="cpu")
        cache = cls(cache_size=data['cache_size'])
        cache.samples = data['samples']
        return cache


class GhostReplayRegularizer:
    """
    Ghost Replay 正则化器
    
    核心修复：
    1. 恢复梯度流 (移除 torch.no_grad)
    2. 支持复杂模型调用协议 (UNet 签名)
    """
    
    def __init__(
        self,
        reference_cache: ReferenceOutputCache,
        lambda_replay: float = 0.1,
        replay_interval: int = 50,
        loss_type: str = 'mse',
        max_deviation: float = 0.5,
        timestep_strategy: str = "random",
        num_train_timesteps: int = 1000,
    ):
        self.cache = reference_cache
        self.lambda_replay = lambda_replay
        self.replay_interval = replay_interval
        self.loss_type = loss_type
        self.max_deviation = max_deviation
        self.timestep_strategy = str(timestep_strategy or "random").lower()
        self.num_train_timesteps = int(num_train_timesteps or 1000)
        
        self._step = 0
        self._last_deviation = 0.0
        self._last_scaled_deviation = 0.0
        self._last_batch_size = 0
        self._last_success_count = 0
        self._last_failure_count = 0
        self._last_replayed = False
        self._last_timestep_buckets: Dict[str, int] = {"low": 0, "mid": 0, "high": 0}
    
    def should_replay(self, step: int) -> bool:
        self._step = step
        return step > 0 and step % self.replay_interval == 0
    
    def compute_loss(
        self,
        model,
        step: int = None,
        batch_size: int = 4,
        device: str = 'cuda'
    ) -> torch.Tensor:
        self._last_replayed = False
        self._last_batch_size = 0
        self._last_success_count = 0
        self._last_failure_count = 0
        self._last_timestep_buckets = {"low": 0, "mid": 0, "high": 0}
        if step is not None and not self.should_replay(step):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        batch = self.cache.get_batch(
            batch_size,
            device,
            timestep_strategy=self.timestep_strategy,
            num_train_timesteps=self.num_train_timesteps,
        )
        self._last_batch_size = len(batch)
        for sample in batch:
            bucket = _timestep_bucket(sample.get("timestep", 0), self.num_train_timesteps)
            self._last_timestep_buckets[bucket] = self._last_timestep_buckets.get(bucket, 0) + 1
        if not batch:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        total_loss = None
        success_count = 0
        failure_count = 0
        
        # 批量处理以提高效率 (假定输入 shape 一致)
        # 注意：此处移除了 torch.no_grad()，允许梯度传导回 model
        for sample in batch:
            inputs = sample['inputs']
            refs = sample['refs']
            
            # 模型调用适配
            try:
                # 尝试解包字典作为关键字参数 (适配 Diffusers 等)
                if isinstance(inputs, dict):
                    output = model(**inputs)
                    # 处理 Diffusers 的 BaseOutput
                    if hasattr(output, 'sample'):
                        output = output.sample
                else:
                    output = model(inputs)
            except Exception as e:
                # 只在 Debug 模式打印
                logger.debug(f"[GhostReplay] Forward error: {e}")
                failure_count += 1
                continue

            # 计算 Loss (refs 已 detach，梯度只会流向 output)
            # V3.1 Fix: Ensure shape compatibility to avoid broadcasting warnings
            # output may be 2D [batch, features] while refs may be 1D [features]
            if output.shape != refs.shape:
                try:
                    refs = refs.view_as(output)
                except RuntimeError:
                    # If view_as fails, try to match dimensions
                    if refs.dim() < output.dim():
                        refs = refs.unsqueeze(0).expand_as(output)
                    elif refs.dim() > output.dim():
                        refs = refs.squeeze()
                        if refs.shape != output.shape:
                            try:
                                refs = refs.view_as(output)
                            except RuntimeError as e:
                                logger.debug(f"[GhostReplay] Shape adapt failed: {e}")
                                failure_count += 1
                                continue
            
            if self.loss_type == 'mse':
                sample_loss = F.mse_loss(output, refs)
            elif self.loss_type == 'cosine':
                cos_sim = F.cosine_similarity(output.view(-1), refs.view(-1), dim=0)
                sample_loss = 1.0 - cos_sim
            else:
                sample_loss = F.mse_loss(output, refs)
                
            total_loss = sample_loss if total_loss is None else total_loss + sample_loss
            success_count += 1

        self._last_success_count = success_count
        self._last_failure_count = failure_count
        if success_count <= 0 or total_loss is None:
            self._last_deviation = 0.0
            self._last_scaled_deviation = 0.0
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = total_loss / success_count
        self._last_replayed = True
        self._last_deviation = float(loss.detach().cpu())
        
        # 平滑阈值逻辑：超过 max_deviation 后线性增加惩罚
        # 如果偏差在允许范围内，降低惩罚权重
        if self._last_deviation < self.max_deviation:
            loss = loss * 0.2
        self._last_scaled_deviation = float(loss.detach().cpu())
        return self.lambda_replay * loss
    
    @property
    def last_deviation(self) -> float:
        """获取上次的偏差值"""
        return self._last_deviation

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        return {
            "step": int(self._step),
            "replayed": bool(self._last_replayed),
            "batch_size": int(self._last_batch_size),
            "success_count": int(self._last_success_count),
            "failure_count": int(self._last_failure_count),
            "last_deviation": float(self._last_deviation),
            "last_scaled_deviation": float(self._last_scaled_deviation),
            "lambda_replay": float(self.lambda_replay),
            "replay_interval": int(self.replay_interval),
            "loss_type": str(self.loss_type),
            "max_deviation": float(self.max_deviation),
            "timestep_strategy": str(self.timestep_strategy),
            "num_train_timesteps": int(self.num_train_timesteps),
            "selected_timestep_buckets": dict(self._last_timestep_buckets),
            "cache_timestep_buckets": self.cache.get_timestep_bucket_counts(self.num_train_timesteps),
        }


class GhostReplayCallback:
    """
    Ghost Replay 训练回调
    
    集成到训练循环中使用
    """
    
    def __init__(
        self,
        cache_path: str = None,
        lambda_replay: float = 0.1,
        replay_interval: int = 50,
        cache_size: int = 100,
        timestep_strategy: str = "random",
        num_train_timesteps: int = 1000,
    ):
        self.cache_path = cache_path
        self.lambda_replay = lambda_replay
        self.replay_interval = replay_interval
        self.cache_size = cache_size
        self.timestep_strategy = str(timestep_strategy or "random").lower()
        self.num_train_timesteps = int(num_train_timesteps or 1000)
        
        self.cache = None
        self.regularizer = None
        self._enabled = False
    
    def initialize(self, reference_model = None, sample_inputs: List[Dict[str, Any]] = None):
        """
        初始化缓存
        
        Args:
            reference_model: 原始模型（用于生成参考输出）
            sample_inputs: 样本输入列表 (List of Dicts)
        """
        if self.cache_path and Path(self.cache_path).exists():
            # 加载已有缓存
            self.cache = ReferenceOutputCache.load(self.cache_path)
            logger.info(f"[GhostReplay] Loaded {len(self.cache)} cached samples")
        else:
            # 创建新缓存
            self.cache = ReferenceOutputCache(cache_size=self.cache_size)
            
            if reference_model is not None and sample_inputs is not None:
                logger.info(f"[GhostReplay] Generating reference outputs...")
                reference_model.eval()
                # 显存优化：不再把整个模型丢进 cache，而是只存输出
                with torch.no_grad():
                    for inputs in sample_inputs[:self.cache_size]:
                        # 适配 dict 或 tensor 输入
                        if isinstance(inputs, dict):
                            out = reference_model(**inputs)
                            if hasattr(out, 'sample'): # 适配 Diffusers
                                out = out.sample
                            self.cache.add_sample(inputs, out.squeeze(0))
                        else:
                            out = reference_model(inputs.unsqueeze(0))
                            self.cache.add_sample({'input': inputs}, out.squeeze(0))
                
                logger.info(f"[GhostReplay] Cached {len(self.cache)} samples")
                
                if self.cache_path:
                    try:
                        self.cache.save(self.cache_path)
                    except Exception as e:
                        logger.error(f"[GhostReplay] Cache save failed: {e}")
        
        self.regularizer = GhostReplayRegularizer(
            self.cache,
            lambda_replay=self.lambda_replay,
            replay_interval=self.replay_interval,
            timestep_strategy=self.timestep_strategy,
            num_train_timesteps=self.num_train_timesteps,
        )
        self._enabled = True
    
    def on_step(self, model, step: int, device: str = 'cuda') -> torch.Tensor:
        """
        训练步骤回调
        
        Returns:
            正则化损失（添加到总损失）
        """
        if not self._enabled or self.regularizer is None:
            return torch.tensor(0.0, device=device)
        
        return self.regularizer.compute_loss(model, step, device=device)
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @property
    def last_deviation(self) -> float:
        if self.regularizer:
            return self.regularizer.last_deviation
        return 0.0

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        if self.regularizer:
            snapshot = self.regularizer.get_telemetry_snapshot()
        else:
            snapshot = {}
        snapshot.update({
            "enabled": bool(self._enabled),
            "cache_size": int(len(self.cache)) if self.cache is not None else 0,
        })
        return snapshot


# 工厂函数
def create_ghost_replay(
    cache_path: str = None,
    lambda_replay: float = 0.1,
    replay_interval: int = 50,
    timestep_strategy: str = "random",
    num_train_timesteps: int = 1000,
) -> GhostReplayCallback:
    """创建 Ghost Replay 回调"""
    return GhostReplayCallback(
        cache_path=cache_path,
        lambda_replay=lambda_replay,
        replay_interval=replay_interval,
        timestep_strategy=timestep_strategy,
        num_train_timesteps=num_train_timesteps,
    )
