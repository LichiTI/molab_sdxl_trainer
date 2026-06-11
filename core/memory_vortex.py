"""
Vortex Memory Manager | [R&D]
负责管理模型权重在 CPU Pinned Memory 与 GPU 显存之间的动态流转。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import threading
from contextlib import contextmanager
from collections import OrderedDict


# Configure debug logging
logger = logging.getLogger("VortexManager")

@dataclass
class OffloadConfig:
    """卸载配置"""
    enable_vortex: bool = True
    pin_memory: bool = True
    chunk_size: int = 1  # 每次搬运的层数 (未启用)
    stream_priority: int = 0  # 0: High, 1: Low (CUDA Stream priority)
    cache_limit_gb: float = 2.0 # 显存保留给 Cache 的最大值
    eviction_threshold: float = 0.9 # 到达 Limit 的 90% 时开始清理
    
    # [V2] Strategy Configuration
    # 'standard': V1 Behavior (Safe for 8GB+)
    # 'active': V2 Active Block Management (Aggressive, for 4GB)
    strategy: str = "standard"
    
    # [V2.1] Memory Profile
    # 'standard': 8GB+ (cache 2GB, strategy standard)
    # 'low_vram': 4-6GB (cache 1GB, strategy active)
    # 'extreme': 4GB (cache 0.5GB, strategy active)
    profile: str = "standard"
    
    def apply_profile(self):
        """根据 profile 自动设置 cache_limit 和 strategy"""
        if self.profile == "low_vram":
            self.cache_limit_gb = 1.0
            self.strategy = "active"
        elif self.profile == "extreme":
            self.cache_limit_gb = 0.5
            self.strategy = "active"
        else:  # standard
            self.cache_limit_gb = 2.0
            self.strategy = "standard"


class VortexManager:
    """
    Vortex 核心管理器
    实现权重的同步/异步弹跳机制
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VortexManager, cls).__new__(cls)
        return cls._instance


    def __init__(self):
        self.config = OffloadConfig()
        self.compute_stream = None
        self.transfer_stream = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        
        # 拓扑追踪 (Linked List)
        self._last_accessed_layer = None
        
        # LRU Cache (Phase 3)
        self.cache_pool: OrderedDict[nn.Module, torch.Tensor] = OrderedDict()
        self.current_cache_bytes: int = 0
        
        # 统计信息
        self.stats = {
            "swaps": 0,
            "bytes_transferred": 0,
            "prefetches": 0,
            "cache_hits": 0,
            "evictions": 0
        }

    def initialize(self, high_priority: bool = True):
        """初始化 CUDA Streams"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, Vortex disabled.")
            self.config.enable_vortex = False
            return

        with self._lock:
            # 创建高优先级流用于计算，低优先级流用于搬运
            # Ensure we wrap stream creation to avoid None issues in emulation/CPU modes
            try:
                self.compute_stream = torch.cuda.current_stream()
                self.transfer_stream = torch.cuda.Stream()
            except Exception as e:
                logger.warning(f"CUDA Stream init failed (likely CPU mode): {e}")
                self.config.enable_vortex = False
                return
            
            logger.info("Vortex Memory Manager initialized with dual streams & LRU Cache.")

    def register_layer(self, layer: nn.Module):
        """
        注册层到 Vortex 管理
        将权重移动到 CPU Pinned Memory
        """
        if not self.config.enable_vortex:
            return

        with torch.no_grad():
            if hasattr(layer, "weight") and layer.weight is not None:
                # 移动到 CPU
                cpu_weight = layer.weight.data.cpu()
                if self.config.pin_memory:
                    cpu_weight = cpu_weight.pin_memory()
                
                # 替换原始权重为 CPU引用，并清除 GPU 占用
                layer.weight.data = cpu_weight
                
                # 初始化 Vortex 状态元数据
                setattr(layer, "_vortex_managed", True)
                setattr(layer, "_vortex_next_layer", None)  # 指向下一层
                setattr(layer, "_vortex_gpu_cache", None)   # GPU Tensor 缓存 (Transitive)
                setattr(layer, "_vortex_event", None)       # 传输完成事件
                setattr(layer, "_vortex_keep_alive", False) # 审计联动: 强制保留
                setattr(layer, "_vortex_status", "CPU")     # CPU | TRANSFERRING | GPU

    def evict_until_safe(self, required_bytes: int = 0):
        """
        [LRU Policy] 清理显存直到满足要求
        """
        limit_bytes = int(self.config.cache_limit_gb * 1024**3)
        threshold_bytes = int(limit_bytes * self.config.eviction_threshold)
        
        # 简单策略: 如果 (当前 + 需要) > 限额，则清理
        max_evictions = len(self.cache_pool) * 2  # Safety brake for infinite loops
        eviction_ops = 0
        
        while (self.current_cache_bytes + required_bytes) > threshold_bytes:
            if eviction_ops > max_evictions:
                logger.warning("Vortex: Eviction infinite loop detected. Breaking.")
                break
            eviction_ops += 1
            
            if not self.cache_pool:
                break # Cache empty, nothing to evict
            
            # Pop first item (LRU)
            # 注意: popitem(last=False) returns first item
            layer, gpu_tensor = self.cache_pool.popitem(last=False)
            
            # 跳过 Keep Alive 的层 (比如 Auditor 认为很重要的)
            if getattr(layer, "_vortex_keep_alive", False):
                # 重新塞回尾部 (最近使用)，避免死循环
                # 但这会导致无法清理。为了防止死循环，如果全是 keep_alive，则 break
                self.cache_pool[layer] = gpu_tensor
                # 简单处理: 我们假设 Cache 够大，不会全是 keep_alive
                break 

            # 释放显存
            tensor_bytes = gpu_tensor.element_size() * gpu_tensor.numel()
            self.current_cache_bytes -= tensor_bytes
            
            # 更新状态
            layer._vortex_gpu_cache = None
            layer._vortex_status = "CPU"
            del gpu_tensor
            self.stats["evictions"] += 1
            
            # torch.cuda.empty_cache() # Expensive

    def cache_layer(self, layer: nn.Module):
        """
        将层加入 LRU Cache
        """
        if layer._vortex_gpu_cache is None:
            return

        # 估算大小
        tensor = layer._vortex_gpu_cache
        size_bytes = tensor.element_size() * tensor.numel()
        
        # 尝试清理空间
        self.evict_until_safe(size_bytes)
        
        # 加入 Cache
        if layer in self.cache_pool:
            # 已经在 Cache 中，移动到末尾 (Recently Used)
            self.cache_pool.move_to_end(layer)
        else:
            self.cache_pool[layer] = tensor
            self.current_cache_bytes += size_bytes
            
        layer._vortex_status = "GPU"

    def _should_evict_aggressively(self) -> bool:
        """[V2] Check if active strategy is enabled"""
        return self.config.strategy == "active"

    def mark_as_done(self, layer: nn.Module):
        """
        [Backward Completed]
        Aggressive cleanup: Try to cache, but if Limit is reached, Evict immediately.
        """
        if layer._vortex_gpu_cache is None:
            return

        # 尝试放入 Cache
        self.cache_layer(layer)
        
        # 强制检查 Limit (Double Check)
        # 如果当前 Cache 依然爆满 (evict_until_safe 没清掉足够的?)
        # 这里的策略是: Backward 刚结束的层是 "Most Recently Used"，理论上应该保留
        # 但是如果我们需要腾空间给 *Backward Previous Layer* (Gradient Flow)
        # 这是一个动态对决。
        # 简单策略: 只要放进 LRU 就行，下次 alloc 会自动踢掉 "Oldest" (即 Forward 最早的层)
        
        # [V2] Active Strategy: Immediate Eviction
        if self._should_evict_aggressively():
            # In Active mode, we assume the layer won't be reused immediately.
            # We enforce strict limit based on configuration.
            # evict_until_safe uses cache_limit_gb.
            active_limit = self.config.cache_limit_gb * 1024**3
            
            if self.current_cache_bytes > active_limit:
                 self.evict_until_safe(0) # try to clear ONLY if over active limit
        pass

    def prefetch_layer(self, layer: nn.Module):
        """
        [异步] 在 transfer_stream 上发起搬运
        """
        # 1. Cache Hit Check
        if layer._vortex_status == "GPU" or layer in self.cache_pool:
            if layer in self.cache_pool:
                self.cache_pool.move_to_end(layer)
            self.stats["cache_hits"] += 1
            return 
            
        if layer._vortex_status != "CPU":
            return

        # 2. Evict if needed (Sync operation on CPU logic, fast)
        # 4GB Protocol: Must ensure space BEFORE allocation
        # Calculate size needed
        required_bytes = layer.weight.element_size() * layer.weight.numel()
        self.evict_until_safe(required_bytes)
        
        # 3. 异步拷贝
        # Phase 2 逻辑: 直接分配
        gpu_weight = torch.empty_like(layer.weight.data, device=self.device)
        
        with torch.cuda.stream(self.transfer_stream):
            gpu_weight.copy_(layer.weight.data, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            
        layer._vortex_gpu_cache = gpu_weight
        layer._vortex_event = event
        layer._vortex_status = "TRANSFERRING"
        self.stats["prefetches"] += 1
        
        # [Fix] Add to LRU Cache
        self.cache_pool[layer] = gpu_weight
        with torch.no_grad():
             self.current_cache_bytes += gpu_weight.numel() * gpu_weight.element_size()

    def ensure_layer_on_gpu(self, layer: nn.Module) -> torch.Tensor:
        """
        [同步/等待] 确保层在 GPU 上可用于计算
        """
        if layer in self.cache_pool:
            self.cache_pool.move_to_end(layer)
            layer._vortex_status = "GPU"   # Ensure status sync
            return self.cache_pool[layer]

        if layer._vortex_status == "GPU":
            return layer._vortex_gpu_cache

        elif layer._vortex_status == "TRANSFERRING":
            # 等待搬运完成
            self.compute_stream.wait_event(layer._vortex_event)
            layer._vortex_status = "GPU"
            return layer._vortex_gpu_cache

        else: # CPU Fallback (未预取到，强制同步搬运)
            # logger.debug("Cache miss, sync transfer")
            gpu_weight = layer.weight.data.to(self.device, non_blocking=True)
            layer._vortex_gpu_cache = gpu_weight
            layer._vortex_status = "GPU"
            self.stats["swaps"] += 1
            
            # [Fix] Add to LRU Cache
            self.cache_pool[layer] = gpu_weight
            with torch.no_grad():
                 self.current_cache_bytes += gpu_weight.numel() * gpu_weight.element_size()
            
            return gpu_weight

    class VortexFunction(torch.autograd.Function):
        """
        自定义 Autograd Function
        在前向传播时搬运权重，计算后立即释放
        """
        @staticmethod
        def forward(ctx, input_tensor, weight_cpu, bias, manager: 'VortexManager', layer: nn.Module):
            ctx.manager = manager
            ctx.layer = layer
            
            # 1. 获取 GPU 权重 (预取感知)
            gpu_weight = manager.ensure_layer_on_gpu(layer)
            
            # [V2 Strategy Logic]
            # Active: Rematerialization (Forward 后立刻释放引用，Backward 时重载)
            # Standard: LRU Caching (Forward 后保留引用，由 Cache 容量决定驱逐)
            saved_weight = gpu_weight
            if manager.config.strategy == 'active':
                saved_weight = None # 不在 Autograd 上下文中持有 GPU 引用
            
            # 记录用于 backward
            ctx.save_for_backward(input_tensor, saved_weight, bias)
            
            # 2. 计算
            output = torch.nn.functional.linear(input_tensor, gpu_weight, bias)
            
            # 3. 显存回收逻辑 (Transient)
            if manager.config.strategy == 'active':
                # Active 模式: Forward 结束即视为该层"本次使用结束"
                # 显式触发 mark_as_done，允许 Cache 立即驱逐
                manager.mark_as_done(layer)
            
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input_tensor, saved_weight, bias = ctx.saved_tensors
            manager = ctx.manager
            layer = ctx.layer
            
            # Rematerialization (重物化)
            gpu_weight = saved_weight
            if gpu_weight is None and layer:
                # Active 模式下，权重已被释放，需重新从 CPU/Cache 加载
                gpu_weight = manager.ensure_layer_on_gpu(layer)

            grad_input = grad_weight = grad_bias = None

            # 计算梯度
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(gpu_weight)
            if ctx.needs_input_grad[1]:
                grad_weight_gpu = grad_output.t().mm(input_tensor)
                grad_weight = grad_weight_gpu
                
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0)
            
            # Backward 结束，该层的 GPU 权重彻底失去利用价值
            if ctx.layer:
                manager.mark_as_done(ctx.layer)
                
            return grad_input, grad_weight, grad_bias, None, None

    def apply_linear(self, layer: nn.Module, input_tensor: torch.Tensor):
        """
        替代 nn.Linear 的调用
        """
        if not getattr(layer, "_vortex_managed", False):
            # Fallback to standard execution
            # Note: layer is nn.Linear
            return F.linear(input_tensor, layer.weight, layer.bias)

        # ---------------------------
        # 1. 拓扑学习 (Topology Learning)
        # ---------------------------
        if self._last_accessed_layer is not None and self._last_accessed_layer is not layer:
            # 记录链表关系: Last -> Current
            self._last_accessed_layer._vortex_next_layer = layer
        
        self._last_accessed_layer = layer

        # ---------------------------
        # 2. 预测性预取 (Predictive Pre-fetching)
        # ---------------------------
        if layer._vortex_next_layer:
            self.prefetch_layer(layer._vortex_next_layer)

        # ---------------------------
        # 3. 执行计算
        # ---------------------------
        return self.VortexFunction.apply(
            input_tensor, 
            layer.weight, # unused in forward actually, but good for signature
            layer.bias, 
            self,
            layer
        )

# 全局单例
vortex_manager = VortexManager()
