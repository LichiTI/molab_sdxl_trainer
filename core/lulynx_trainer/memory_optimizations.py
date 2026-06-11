"""
显存优化工具集

Warehouse 实现的显存优化功能，基于 PyTorch 公开 API。
- BlockSwapOffloader: UNet block 级别 CPU↔GPU 异步搬运
- apply_channels_last: channels_last 内存格式优化
- PipelineSlicer: VAE slicing/tiling + attention slicing
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# BlockSwapOffloader — UNet block 级别 CPU↔GPU 异步搬运
# ═══════════════════════════════════════════════════════════════════════════

def _clean_device_memory(
    device: torch.device,
    *,
    collect_gc: bool = True,
    release_cache: bool = True,
) -> None:
    """清理指定设备的缓存"""
    if collect_gc:
        gc.collect()
    if release_cache and device.type == "cuda":
        torch.cuda.empty_cache()
    elif release_cache and device.type == "xpu":
        torch.xpu.empty_cache()


def _synchronize_device(device: torch.device) -> None:
    """同步指定设备"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def _swap_weights_cuda(
    device: torch.device,
    src_module: nn.Module,
    dst_module: nn.Module,
    should_swap: Optional[Callable[[nn.Module, str], bool]] = None,
) -> None:
    """在 CUDA stream 上异步交换两个同构模块的权重"""
    pairs: List[Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]] = []

    dst_named = {k: v for k, v in dst_module.named_modules()}
    for name, src_mod in src_module.named_modules():
        if not hasattr(src_mod, "weight") or src_mod.weight is None:
            continue
        if getattr(src_mod, "lulynx_weight_residency_active", False):
            continue
        if should_swap is not None and not should_swap(src_mod, name):
            continue
        dst_mod = dst_named.get(name)
        if dst_mod is not None and dst_mod.weight.shape == src_mod.weight.shape:
            pairs.append((src_mod, dst_mod, src_mod.weight.data, dst_mod.weight.data))
        elif dst_mod is not None and dst_mod.weight.data.device.type != device.type:
            dst_mod.weight.data = dst_mod.weight.data.to(device)

    torch.cuda.current_stream().synchronize()

    stream = torch.Stream(device="cuda")
    with torch.cuda.stream(stream):
        # src (GPU) → CPU
        for src_mod, dst_mod, gpu_view, cpu_view in pairs:
            gpu_view.record_stream(stream)
            src_mod.weight.data = gpu_view.data.to("cpu", non_blocking=True)
        stream.synchronize()

        # dst (CPU) → GPU
        for src_mod, dst_mod, gpu_view, cpu_view in pairs:
            gpu_view.copy_(dst_mod.weight.data, non_blocking=True)
            dst_mod.weight.data = gpu_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()


def _move_weights_to_device(
    module: nn.Module,
    device: torch.device,
    should_swap: Optional[Callable[[nn.Module, str], bool]] = None,
    exclude_lora_leaves: bool = False,
) -> None:
    """将模块的所有权重移动到指定设备。

    Args:
        module: 要移动的模块
        device: 目标设备
        should_swap: 可选的过滤回调，返回 False 的子模块不被移动
        exclude_lora_leaves: 如果为 True，标记了 _lora_leaf=True 的参数不会被移动
                           （用于 block swap 时保留 LoRA 参数在 GPU）
    """
    for name, mod in module.named_modules():
        if getattr(mod, "lulynx_weight_residency_active", False):
            continue
        if should_swap is not None and not should_swap(mod, name):
            continue
        if exclude_lora_leaves and getattr(mod, "_lora_leaf", False):
            continue
        if hasattr(mod, "weight") and mod.weight is not None:
            mod.weight.data = mod.weight.data.to(device, non_blocking=True)
        if hasattr(mod, "bias") and mod.bias is not None:
            mod.bias.data = mod.bias.data.to(device, non_blocking=True)


@dataclass
class SwapStats:
    """搬运统计"""
    swap_count: int = 0
    prefetch_count: int = 0
    direct_move_count: int = 0
    wait_count: int = 0
    total_swap_ms: float = 0.0
    total_prefetch_ms: float = 0.0
    total_direct_move_ms: float = 0.0
    total_wait_ms: float = 0.0
    prepare_count: int = 0
    total_prepare_ms: float = 0.0
    pipeline_enqueue_count: int = 0
    total_pipeline_enqueue_ms: float = 0.0
    pipeline_event_wait_count: int = 0
    total_pipeline_event_wait_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        def _avg(total_ms: float, count: int) -> float:
            return round(float(total_ms or 0.0) / max(int(count or 0), 1), 2)

        return {
            "swap_count": int(self.swap_count),
            "prefetch_count": int(self.prefetch_count),
            "direct_move_count": int(self.direct_move_count),
            "wait_count": int(self.wait_count),
            "prepare_count": int(self.prepare_count),
            "pipeline_enqueue_count": int(self.pipeline_enqueue_count),
            "pipeline_event_wait_count": int(self.pipeline_event_wait_count),
            "total_swap_ms": round(float(self.total_swap_ms), 2),
            "total_prefetch_ms": round(float(self.total_prefetch_ms), 2),
            "total_direct_move_ms": round(float(self.total_direct_move_ms), 2),
            "total_wait_ms": round(float(self.total_wait_ms), 2),
            "total_prepare_ms": round(float(self.total_prepare_ms), 2),
            "total_pipeline_enqueue_ms": round(float(self.total_pipeline_enqueue_ms), 2),
            "total_pipeline_event_wait_ms": round(float(self.total_pipeline_event_wait_ms), 2),
            "avg_swap_ms": _avg(self.total_swap_ms, self.swap_count),
            "avg_prefetch_ms": _avg(self.total_prefetch_ms, self.prefetch_count),
            "avg_direct_move_ms": _avg(self.total_direct_move_ms, self.direct_move_count),
            "avg_wait_ms": _avg(self.total_wait_ms, self.wait_count),
            "avg_prepare_ms": _avg(self.total_prepare_ms, self.prepare_count),
            "avg_pipeline_enqueue_ms": _avg(self.total_pipeline_enqueue_ms, self.pipeline_enqueue_count),
            "avg_pipeline_event_wait_ms": _avg(self.total_pipeline_event_wait_ms, self.pipeline_event_wait_count),
        }


def normalize_block_swap_strategy(value: Any) -> str:
    strategy = str(value or "auto").strip().lower().replace("-", "_")
    return strategy if strategy in {"auto", "sync", "async", "pipeline"} else "auto"


@dataclass
class SwapPlan:
    enabled: bool
    requested_granularity: str = "off"
    effective_granularity: str = "off"
    swap_ratio: float = 0.0
    swap_count: int = 0
    units_total: int = 0
    units_swapped: int = 0
    block_merge_size: int = 2
    source: str = "manual"
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": "swap" if self.enabled else "none",
            "requested_granularity": self.requested_granularity,
            "effective_granularity": self.effective_granularity,
            "swap_ratio": self.swap_ratio,
            "swap_count": self.swap_count,
            "units_total": self.units_total,
            "units_swapped": self.units_swapped,
            "block_merge_size": self.block_merge_size,
            "source": self.source,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


def normalize_swap_granularity(value: Any) -> str:
    granularity = str(value or "off").strip().lower().replace("-", "_")
    return granularity if granularity in {"off", "auto", "block", "merged_block", "layer"} else "off"


def resolve_swap_count(total_units: int, swap_ratio: float = 0.0, swap_count: int = 0) -> int:
    if total_units <= 1:
        return 0
    try:
        count = int(swap_count or 0)
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        try:
            ratio = max(0.0, min(1.0, float(swap_ratio or 0.0)))
        except (TypeError, ValueError):
            ratio = 0.0
        count = int(round(total_units * ratio)) if ratio > 0 else 0
    return max(0, min(count, total_units - 1))


def _detect_low_bandwidth_pcie() -> Tuple[bool, str]:
    return False, "unknown"


def build_swap_units(stage_metadata: Optional[List[str]], total_blocks: int, merge_size: int) -> List[List[int]]:
    if total_blocks <= 0:
        return []
    merge_size = max(2, int(merge_size or 2))
    if not stage_metadata or len(stage_metadata) != total_blocks:
        return [list(range(i, min(i + merge_size, total_blocks))) for i in range(0, total_blocks, merge_size)]
    units: List[List[int]] = []
    start = 0
    while start < total_blocks:
        stage = stage_metadata[start]
        end = start + 1
        while end < total_blocks and stage_metadata[end] == stage:
            end += 1
        if stage == "mid":
            for idx in range(start, end):
                units.append([idx])
        else:
            for idx in range(start, end, merge_size):
                units.append(list(range(idx, min(idx + merge_size, end))))
        start = end
    return units


def build_swap_plan(config: Any, total_units: int, stage_metadata: Optional[List[str]] = None) -> SwapPlan:
    requested = normalize_swap_granularity(getattr(config, "swap_granularity", "off"))
    source = "manual"
    legacy_count = int(getattr(config, "blocks_to_swap", 0) or 0)
    swap_ratio = max(0.0, min(1.0, float(getattr(config, "swap_ratio", 0.0) or 0.0)))
    swap_count = int(getattr(config, "swap_count", 0) or 0)
    merge_size = max(2, int(getattr(config, "block_merge_size", 2) or 2))
    if requested == "off" and legacy_count > 0:
        requested = "block"
        swap_count = legacy_count
        source = "legacy"
    effective = requested
    reason = "manual selection"
    if requested == "auto":
        low_bandwidth, pcie = _detect_low_bandwidth_pcie()
        effective = "merged_block" if low_bandwidth else "block"
        source = "auto" if pcie != "unknown" else "fallback"
        reason = f"PCIe {pcie}; selected {effective}"
    units_total = total_units
    if effective == "merged_block":
        units_total = len(build_swap_units(stage_metadata, total_units, merge_size))
    units_swapped = resolve_swap_count(units_total, swap_ratio, swap_count)
    enabled = effective != "off" and units_swapped > 0
    return SwapPlan(enabled, requested, effective, swap_ratio, swap_count, units_total, units_swapped, merge_size, source, reason)


class BlockSwapOffloader:
    """
    UNet block 级别 CPU↔GPU 异步搬运器

    原理：
    - 将 UNet 的 down_blocks/mid_block/up_blocks 分为 GPU 常驻区和 CPU 交换区
    - 前向传播时，当前 block 到达前将下一个需要的 block 异步搬到 GPU
    - 反向传播时通过 full_backward_hook 实现反向搬运
    - 使用 ThreadPoolExecutor + CUDA Stream 实现零等待搬运
    """

    def __init__(
        self,
        blocks: Union[List[nn.Module], nn.ModuleList],
        blocks_to_swap: int,
        device: torch.device,
        enable_backward: bool = True,
        should_swap: Optional[Callable[[nn.Module, str], bool]] = None,
        units: Optional[List[List[int]]] = None,
        selected_unit_indices: Optional[List[int]] = None,
        strategy: str = "auto",
        release_cache_on_prepare: bool = False,
    ):
        self.blocks = list(blocks)
        self.num_blocks = len(self.blocks)
        self.units = units or [[idx] for idx in range(self.num_blocks)]
        self.num_units = len(self.units)
        self._block_to_unit: Dict[int, int] = {}
        for unit_idx, unit in enumerate(self.units):
            for block_idx in unit:
                self._block_to_unit[block_idx] = unit_idx
        explicit_indices = []
        if selected_unit_indices:
            seen = set()
            for value in selected_unit_indices:
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < self.num_units and idx not in seen:
                    seen.add(idx)
                    explicit_indices.append(idx)
        max_swappable = max(self.num_units - 1, 0)  # 至少保留 1 个 unit 在 GPU
        if explicit_indices and len(explicit_indices) >= self.num_units:
            explicit_indices = explicit_indices[-max_swappable:]
        self._selected_unit_indices = sorted(explicit_indices)
        self._selected_unit_set = set(self._selected_unit_indices)
        self._explicit_selection = bool(self._selected_unit_indices)
        self.blocks_to_swap = (
            len(self._selected_unit_indices)
            if self._explicit_selection
            else min(blocks_to_swap, max_swappable)
        )
        self.device = device
        self.enable_backward = enable_backward and units is None and not self._explicit_selection
        self.needs_step_prepare = self._explicit_selection
        self.stats = SwapStats()
        self._stats_lock = threading.Lock()
        self._should_swap = should_swap
        self._prepare_log_count = 0
        self.requested_strategy = normalize_block_swap_strategy(strategy)
        self._pipeline_stream_lock = threading.Lock()
        self._pipeline_stream_index = 0
        self._pipeline_streams: List[Any] = []
        self._pipeline_stream_capability = self._probe_pipeline_stream_capability(self.requested_strategy)
        self.effective_strategy, self.strategy_fallback_reason = self._resolve_strategy(self.requested_strategy)
        self.release_cache_on_prepare = bool(release_cache_on_prepare)

        max_workers = 2 if self.effective_strategy == "pipeline" else 1
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[int, Any] = {}
        self._ready_units: set[int] = set()
        self._handles: List[Any] = []

        if self.enable_backward and self.blocks_to_swap > 0:
            self._register_backward_hooks()

    def _record_stats(self, **updates: float) -> None:
        with self._stats_lock:
            for name, delta in updates.items():
                current = getattr(self.stats, name, 0)
                setattr(self.stats, name, current + delta)

    def _probe_pipeline_stream_capability(self, requested: str) -> Dict[str, Any]:
        capability: Dict[str, Any] = {
            "requested": requested == "pipeline",
            "available": False,
            "reason": "pipeline strategy was not requested",
            "stream_count": 0,
            "device": str(self.device),
        }
        if requested != "pipeline":
            return capability
        if self.device.type != "cuda":
            capability["reason"] = "pipeline stream experiment requires CUDA"
            return capability
        if not torch.cuda.is_available():
            capability["reason"] = "CUDA is not available"
            return capability
        try:
            with torch.cuda.device(self.device):
                self._pipeline_streams = [torch.cuda.Stream(device=self.device) for _ in range(2)]
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.device))
            capability.update(
                {
                    "available": True,
                    "reason": "CUDA stream/event pipeline experiment is available",
                    "stream_count": len(self._pipeline_streams),
                }
            )
        except Exception as exc:
            self._pipeline_streams = []
            capability["reason"] = f"failed to create CUDA stream/event: {type(exc).__name__}: {exc}"
        return capability

    def _resolve_strategy(self, requested: str) -> Tuple[str, str]:
        if requested == "sync":
            return "sync", ""
        if requested == "auto":
            if self.device.type in {"cuda", "xpu"}:
                return "async", "auto selected async for accelerator block swap"
            return "sync", "auto selected sync because the target device is not an accelerator"
        if requested == "async":
            if self.device.type in {"cuda", "xpu"}:
                return "async", ""
            return "sync", "async block swap requires an accelerator target; using sync"
        if requested == "pipeline":
            if self.device.type != "cuda":
                return "sync", "pipeline block swap currently requires CUDA; using sync"
            if self._explicit_selection:
                return "async", "pipeline is not enabled for explicit precision-swap selections; using async"
            if not self._pipeline_stream_capability.get("available"):
                reason = self._pipeline_stream_capability.get("reason") or "pipeline stream experiment is unavailable"
                return "async", f"{reason}; using async"
            return "pipeline", ""
        return "async", "unknown block swap strategy; using async"

    def _register_backward_hooks(self) -> None:
        """为 CPU 交换区的 block 注册反向传播 hook。

        反向传播按 forward 的逆序遍历 blocks，因此：
        - forward 顺序: [0, 1, 2, ..., N-1]
        - backward 顺序: [N-1, N-2, ..., 1, 0]
        - GPU 常驻区: [0, ..., N-1-swap)  (前 N-swap 个)
        - CPU 交换区: [N-swap, ..., N-1]  (后 swap 个)

        当 backward 到达 block_idx 时：
        - 如果 block_idx 在 GPU 常驻区的后部分（但靠近 CPU 交换区），
          需要将一个 GPU block 换出到 CPU，同时将下一个 CPU block 换入 GPU
          （因为 backward 接下来需要它）。
        - 如果 block_idx 在 CPU 交换区，需要等待该 block 换入 GPU 完成。
        """
        for i, block in enumerate(self.blocks):
            hook = self._make_backward_hook(i)
            if hook is not None:
                handle = block.register_full_backward_hook(hook)
                self._handles.append(handle)

    def _make_backward_hook(self, block_idx: int):
        """创建反向传播 hook，实现反向搬运。

        Forward 顺序: block 0, 1, 2, ..., N-1
        Backward 顺序: block N-1, N-2, ..., 1, 0

        GPU 常驻区: blocks [0, N-swap)
        CPU 交换区: blocks [N-swap, N)

        Backward 阶段:
        - 当 backward 到达一个 GPU 常驻区的 block（靠近 CPU 交换区边界）时，
          将其换出到 CPU（因为 forward 不再需要它），同时预取 backward 下一步
          需要的 CPU block 到 GPU。
        - 当 backward 到达一个刚换入 GPU 的 block 时，等待换入完成。
        """
        if self.blocks_to_swap == 0:
            return None

        gpu_end = self.num_blocks - self.blocks_to_swap  # GPU 常驻区结束索引

        # backward 遍历顺序: N-1, N-2, ..., 0
        # 对于 block_idx，backward 中下一个需要的 block 是 block_idx - 1
        # （如果 block_idx > 0）

        should_swap = False
        should_wait = False
        swap_gpu_idx = -1  # 换出的 GPU block 索引
        swap_cpu_idx = -1  # 换入的 CPU block 索引

        # 如果当前 block 在 CPU 交换区（forward 阶段被换入 GPU 用于 backward）
        # backward 完成后需要将其换出到 CPU，并预取 backward 下一个需要的 block
        if block_idx >= gpu_end:
            should_wait = True
            # backward 下一个需要的 block 是 block_idx - 1
            # 如果 block_idx - 1 也在 CPU 交换区，需要预取它
            # 同时可以换出当前 block（backward 已完成）到 CPU，
            # 换入一个 GPU 常驻区后部的 block 到 CPU（因为它不再需要了）
            if block_idx > gpu_end:
                # 可以换出当前 block（已在 GPU），换入 block_idx - 1（在 CPU）
                swap_cpu_idx = block_idx - 1  # 需要从 CPU 换入的 block
                swap_gpu_idx = block_idx       # 换出到 CPU 的 block
                should_swap = True
        elif block_idx >= gpu_end - self.blocks_to_swap and block_idx < gpu_end:
            # 当前 block 在 GPU 常驻区后部，backward 完成后可以换出
            # 需要预取 CPU 交换区中 backward 下一个需要的 block
            # backward 下一步需要 block_idx - 1
            # 如果 block_idx - 1 在 CPU 交换区，需要换入
            if block_idx > 0 and block_idx - 1 >= gpu_end:
                swap_cpu_idx = block_idx - 1  # CPU block 需要换入
                swap_gpu_idx = block_idx       # GPU block 换出到 CPU
                should_swap = True

        if not should_swap and not should_wait:
            return None

        offloader = self

        def hook_fn(module, grad_input, grad_output):
            if should_wait:
                # 等待当前 block 换入 GPU 完成（如果有的话）
                offloader._wait_swap(block_idx)
            if should_swap:
                offloader._submit_swap(swap_cpu_idx, swap_gpu_idx)
            return None

        return hook_fn

    def _move_unit_to_device(self, unit_idx: int, device: torch.device) -> None:
        for block_idx in self.units[unit_idx]:
            block = self.blocks[block_idx]
            _move_weights_to_device(
                block,
                device,
                self._should_swap,
                exclude_lora_leaves=(device.type == "cpu"),
            )

    def _next_pipeline_stream(self) -> Any:
        if not self._pipeline_streams:
            return None
        with self._pipeline_stream_lock:
            stream = self._pipeline_streams[self._pipeline_stream_index % len(self._pipeline_streams)]
            self._pipeline_stream_index += 1
            return stream

    def _wait_pipeline_event(self, event: Any) -> None:
        if event is None or self.device.type != "cuda":
            return
        started = time.perf_counter()
        try:
            torch.cuda.current_stream(self.device).wait_event(event)
        except Exception:
            event.synchronize()
        self._record_stats(
            pipeline_event_wait_count=1,
            total_pipeline_event_wait_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _submit_swap(self, cpu_idx: int, gpu_idx: int) -> None:
        """异步提交 block 搬运任务"""
        def do_swap():
            started = time.perf_counter()
            _swap_weights_cuda(self.device, self.blocks[gpu_idx], self.blocks[cpu_idx], self._should_swap)
            self._record_stats(total_swap_ms=(time.perf_counter() - started) * 1000.0)
            return cpu_idx, gpu_idx

        if self.effective_strategy == "sync":
            do_swap()
            self._record_stats(swap_count=1)
            return
        self._futures[cpu_idx] = self._pool.submit(do_swap)
        self._record_stats(swap_count=1)

    def _submit_unit_prefetch(self, unit_idx: int) -> None:
        def do_move():
            started = time.perf_counter()
            self._move_unit_to_device(unit_idx, self.device)
            _synchronize_device(self.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._record_stats(total_swap_ms=elapsed_ms, total_prefetch_ms=elapsed_ms)
            return unit_idx

        if self.effective_strategy == "sync":
            do_move()
            self._ready_units.add(unit_idx)
            self._record_stats(swap_count=1, prefetch_count=1)
            return
        if self.effective_strategy == "pipeline":
            def do_pipeline_move():
                started = time.perf_counter()
                stream = self._next_pipeline_stream()
                if stream is None:
                    self._move_unit_to_device(unit_idx, self.device)
                    _synchronize_device(self.device)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    self._record_stats(total_swap_ms=elapsed_ms, total_prefetch_ms=elapsed_ms)
                    return {"unit_idx": unit_idx, "event": None, "mode": "sync_fallback"}
                with torch.cuda.device(self.device), torch.cuda.stream(stream):
                    self._move_unit_to_device(unit_idx, self.device)
                    event = torch.cuda.Event()
                    event.record(stream)
                enqueue_ms = (time.perf_counter() - started) * 1000.0
                self._record_stats(
                    pipeline_enqueue_count=1,
                    total_pipeline_enqueue_ms=enqueue_ms,
                    total_swap_ms=enqueue_ms,
                    total_prefetch_ms=enqueue_ms,
                )
                return {"unit_idx": unit_idx, "event": event, "mode": "stream_event"}

            self._futures[unit_idx] = self._pool.submit(do_pipeline_move)
            self._record_stats(swap_count=1, prefetch_count=1)
            return
        self._futures[unit_idx] = self._pool.submit(do_move)
        self._record_stats(swap_count=1, prefetch_count=1)

    def _wait_swap(self, block_idx: int) -> None:
        """等待指定 block 的搬运完成"""
        if block_idx not in self._futures:
            return
        future = self._futures.pop(block_idx)
        started = time.perf_counter()
        future.result()
        self._record_stats(wait_count=1, total_wait_ms=(time.perf_counter() - started) * 1000.0)

    def _wait_unit(self, unit_idx: int) -> None:
        if unit_idx not in self._futures:
            return
        future = self._futures.pop(unit_idx)
        started = time.perf_counter()
        result = future.result()
        if isinstance(result, dict):
            self._wait_pipeline_event(result.get("event"))
        self._record_stats(wait_count=1, total_wait_ms=(time.perf_counter() - started) * 1000.0)

    def prepare_before_forward(self) -> None:
        """
        前向传播前的准备工作：
        1. 等待所有待定搬运
        2. 将 GPU 常驻区 block 移到 GPU
        3. 将 CPU 交换区 block 的权重移到 CPU
        """
        if self.blocks_to_swap == 0:
            return

        started = time.perf_counter()
        # 等待所有待定搬运
        for idx in list(self._futures.keys()):
            self._wait_swap(idx)
        self._ready_units.clear()

        if self._explicit_selection:
            for unit_idx in range(self.num_units):
                target = torch.device("cpu") if unit_idx in self._selected_unit_set else self.device
                self._move_unit_to_device(unit_idx, target)
        else:
            gpu_units_end = self.num_units - self.blocks_to_swap
            for unit_idx in range(gpu_units_end):
                self._move_unit_to_device(unit_idx, self.device)

            for unit_idx in range(gpu_units_end, self.num_units):
                self._move_unit_to_device(unit_idx, torch.device("cpu"))

        _synchronize_device(self.device)
        _clean_device_memory(
            self.device,
            collect_gc=self.release_cache_on_prepare,
            release_cache=self.release_cache_on_prepare,
        )
        self._record_stats(
            prepare_count=1,
            total_prepare_ms=(time.perf_counter() - started) * 1000.0,
        )

        self._prepare_log_count += 1
        if self._prepare_log_count == 1:
            logger.info(
                f"BlockSwap: {self.blocks_to_swap}/{self.num_units} swap units on CPU, "
                f"{self.num_units - self.blocks_to_swap} units on GPU ({self.num_blocks} blocks total), "
                f"strategy={self.effective_strategy}"
                + (f", selected={self._selected_unit_indices}" if self._explicit_selection else "")
            )

    def offload_selected_after_step(self) -> Dict[str, Any]:
        """Return explicit precision-swap units to CPU after optimizer step.

        Explicit unit selection intentionally keeps units on GPU through
        backward for correctness.  Once optimizer.step() and zero_grad() are
        complete, frozen base block weights are cold again and can be returned
        to CPU before the next step.
        """

        if self.blocks_to_swap == 0 or not self._explicit_selection:
            return {}
        started = time.perf_counter()
        moved_units = 0
        for unit_idx in self._selected_unit_indices:
            if unit_idx in self._futures:
                self._wait_unit(unit_idx)
            self._ready_units.discard(unit_idx)
            self._move_unit_to_device(unit_idx, torch.device("cpu"))
            moved_units += 1
        _synchronize_device(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._record_stats(prepare_count=1, total_prepare_ms=elapsed_ms)
        return {
            "mode": "explicit_selected_after_step",
            "units": moved_units,
            "selected_unit_indices": list(self._selected_unit_indices),
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def ensure_block_on_device(self, block_idx: int) -> None:
        """保证 block_idx 的权重已在 GPU 上，必要时阻塞等待异步拷贝完成。

        用于 forward_pre_hook：在 block 真正执行前调用，确保权重就绪。
        """
        if self.blocks_to_swap == 0:
            return
        if block_idx < 0 or block_idx >= self.num_blocks:
            return
        unit_idx = self._block_to_unit.get(block_idx, block_idx)
        if self._explicit_selection:
            if unit_idx not in self._selected_unit_set:
                return
        else:
            gpu_units_end = self.num_units - self.blocks_to_swap
            if unit_idx < gpu_units_end:
                return
        if unit_idx in self._ready_units:
            self._ready_units.discard(unit_idx)
            return
        if unit_idx not in self._futures and self._explicit_selection:
            started = time.perf_counter()
            self._move_unit_to_device(unit_idx, self.device)
            _synchronize_device(self.device)
            self._record_stats(
                direct_move_count=1,
                total_direct_move_ms=(time.perf_counter() - started) * 1000.0,
            )
            return
        if unit_idx in self._futures:
            self._wait_unit(unit_idx)
            return
        started = time.perf_counter()
        self._move_unit_to_device(unit_idx, self.device)
        _synchronize_device(self.device)
        self._record_stats(
            direct_move_count=1,
            total_direct_move_ms=(time.perf_counter() - started) * 1000.0,
        )

    def prefetch_next(self, current_idx: int) -> None:
        """在当前 block 执行后，调度下一个需要搬运的 block 异步预取到 GPU。

        用于 forward_hook：在 block 执行后调用，隐藏搬运延迟。
        当前 block 的非驻留权重不会立即换出到 CPU（由 prepare_before_forward
        和 backward hook 管理换出时机）。
        """
        if self.blocks_to_swap == 0:
            return
        next_idx = current_idx + 1
        if next_idx >= self.num_blocks:
            return
        unit_idx = self._block_to_unit.get(next_idx, next_idx)
        if self._explicit_selection:
            if unit_idx not in self._selected_unit_set:
                return
        else:
            gpu_units_end = self.num_units - self.blocks_to_swap
            if unit_idx < gpu_units_end:
                return
        if unit_idx in self._futures:
            return
        if not self._explicit_selection:
            self._submit_unit_prefetch(unit_idx)
            return
        self._submit_unit_prefetch(unit_idx)

    def install_forward_hooks(self, model: nn.Module) -> None:
        """将 block swap hook 注册到模型各 block 的 forward 路径上。

        对每个 block 注册：
        - forward_pre_hook: 调用 ensure_block_on_device(i)，保证权重在 GPU
        - forward_hook: 调用 prefetch_next(i)，调度下一个 block 的预取
        """
        if self.blocks_to_swap == 0:
            return
        setter = getattr(model, "set_block_swap_offloader", None)
        if callable(setter):
            setter(self)
            self._native_observer_model = model
            logger.info(
                f"BlockSwap: installed native block observer on {len(self.blocks)} blocks "
                f"({self.blocks_to_swap} swap units on CPU, {self.num_units - self.blocks_to_swap} units on GPU)"
            )
            return
        self._forward_hook_handles = []
        for idx, block in enumerate(self.blocks):
            offloader = self  # capture in closure

            def make_pre_hook(block_idx):
                def pre_hook(module, args):
                    offloader.ensure_block_on_device(block_idx)
                return pre_hook

            def make_post_hook(block_idx):
                def post_hook(module, args, output):
                    offloader.prefetch_next(block_idx)
                return post_hook

            pre_handle = block.register_forward_pre_hook(make_pre_hook(idx))
            post_handle = block.register_forward_hook(make_post_hook(idx))
            self._forward_hook_handles.append(pre_handle)
            self._forward_hook_handles.append(post_handle)

        logger.info(
            f"BlockSwap: installed forward hooks on {len(self.blocks)} blocks "
            f"({self.blocks_to_swap} swap units on CPU, {self.num_units - self.blocks_to_swap} units on GPU)"
        )

    def remove_forward_hooks(self) -> None:
        """移除 install_forward_hooks 注册的所有 hook"""
        for handle in getattr(self, "_forward_hook_handles", []):
            handle.remove()
        self._forward_hook_handles = []

    def _validate_device_placement(self, label: str = "") -> Dict[str, List[int]]:
        """验证所有 block 参数的设备位置是否正确（仅在 debug/测试路径使用）。

        Returns:
            dict with keys 'cpu_blocks' (indices of blocks with weights on CPU)
            and 'gpu_blocks' (indices of blocks with weights on GPU).
        """
        gpu_end = self.num_blocks - self.blocks_to_swap
        misplaced: Dict[str, List[int]] = {"cpu_blocks": [], "gpu_blocks": []}
        for idx, block in enumerate(self.blocks):
            expected_device = self.device if idx < gpu_end else torch.device("cpu")
            for name, param in block.named_parameters():
                # Compare device types, not exact device objects (cuda != cuda:0 but both are CUDA)
                param_is_cpu = param.device.type == "cpu"
                expected_is_cpu = expected_device.type == "cpu"
                if param_is_cpu != expected_is_cpu:
                    if param_is_cpu:
                        misplaced["cpu_blocks"].append(idx)
                    else:
                        misplaced["gpu_blocks"].append(idx)
                    break
        if misplaced["cpu_blocks"] or misplaced["gpu_blocks"]:
            logger.warning(
                f"BlockSwap device validation ({label}): "
                f"blocks with CPU params (expected GPU): {misplaced['cpu_blocks']}, "
                f"blocks with GPU params (expected CPU): {misplaced['gpu_blocks']}"
            )
        return misplaced

    def on_forward_block(self, block_idx: int) -> None:
        """前向传播到某个 block 时调用，触发下一个 block 的异步搬运。

        注意：当使用 install_forward_hooks 时，此方法不再需要手动调用，
        hook 会自动触发 prefetch_next。此方法保留以兼容旧调用路径。
        """
        if self.blocks_to_swap == 0:
            return
        if not self.enable_backward and block_idx >= self.blocks_to_swap:
            return

        cpu_idx = block_idx
        gpu_idx = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
        self._submit_swap(cpu_idx, gpu_idx)

    def wait_block(self, block_idx: int) -> None:
        """等待指定 block 的搬运完成"""
        if self.blocks_to_swap == 0:
            return
        self._wait_swap(block_idx)

    def cleanup(self) -> None:
        """清理资源"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.remove_forward_hooks()
        setter = getattr(getattr(self, "_native_observer_model", None), "set_block_swap_offloader", None)
        if callable(setter):
            setter(None)
        self._native_observer_model = None
        self._pool.shutdown(wait=False)
        self._futures.clear()
        self._ready_units.clear()

    def strategy_state(self) -> Dict[str, str]:
        return {
            "block_swap_strategy": self.effective_strategy,
            "requested_block_swap_strategy": self.requested_strategy,
            "block_swap_strategy_fallback_reason": self.strategy_fallback_reason,
        }

    def profile_state(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = self.stats.as_dict()
        return {
            "requested_strategy": self.requested_strategy,
            "resolved_strategy": self.effective_strategy,
            "fallback_reason": self.strategy_fallback_reason,
            "prefetch_enabled": self.effective_strategy in {"async", "pipeline"},
            "pipeline_requested": self.requested_strategy == "pipeline",
            "pipeline_active": self.effective_strategy == "pipeline",
            "pipeline_stream_capability": dict(self._pipeline_stream_capability),
            "max_workers": 2 if self.effective_strategy == "pipeline" else 1,
            "blocks_total": int(self.num_blocks),
            "units_total": int(self.num_units),
            "units_swapped": int(self.blocks_to_swap),
            "explicit_selection": bool(self._explicit_selection),
            "enable_backward": bool(self.enable_backward),
            "pending_futures": int(len(self._futures)),
            "ready_units": int(len(self._ready_units)),
            "stats": stats,
        }

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class LayerSwapOffloader(BlockSwapOffloader):
    """子层级 CPU↔GPU 搬运器，仅面向高级显存调优场景。"""

    @staticmethod
    def collect_layers(blocks: Union[List[nn.Module], nn.ModuleList]) -> List[nn.Module]:
        layers: List[nn.Module] = []
        seen: set[int] = set()
        for block in blocks:
            for module in block.modules():
                if id(module) in seen:
                    continue
                if getattr(module, "_lora_leaf", False):
                    continue
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if any(param.requires_grad or param.numel() > 0 for param in module.parameters(recurse=False)):
                        layers.append(module)
                        seen.add(id(module))
        return layers

    def __init__(
        self,
        blocks: Union[List[nn.Module], nn.ModuleList],
        layers_to_swap: int,
        device: torch.device,
        should_swap: Optional[Callable[[nn.Module, str], bool]] = None,
    ):
        layers = self.collect_layers(blocks)
        super().__init__(
            blocks=layers,
            blocks_to_swap=layers_to_swap,
            device=device,
            enable_backward=False,
            should_swap=should_swap,
        )


# ═══════════════════════════════════════════════════════════════════════════
# channels_last 内存格式优化
# ═══════════════════════════════════════════════════════════════════════════

def apply_channels_last(
    *models: nn.Module,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    将模型的 4D/5D 浮点张量转为 channels_last 内存布局。

    channels_last 布局将通道维度放在最后，匹配 GPU 的 NHWC 计算模式，
    可以显著提升 L2 cache 命中率，尤其在 Conv2d 密集的 UNet 中效果明显。

    Returns:
        每个模型转换的张量数统计
    """
    stats: Dict[str, int] = {}

    for model in models:
        if model is None:
            continue

        name = type(model).__name__
        converted_4d = 0
        converted_5d = 0

        for param in model.parameters():
            if not param.is_floating_point():
                continue

            if param.dim() == 4:
                if not param.is_contiguous(memory_format=torch.channels_last):
                    param.data = param.data.contiguous(memory_format=torch.channels_last)
                    converted_4d += 1
            elif param.dim() == 5:
                if not param.is_contiguous(memory_format=torch.channels_last_3d):
                    param.data = param.data.contiguous(memory_format=torch.channels_last_3d)
                    converted_5d += 1

        for buf in model.buffers():
            if not buf.is_floating_point():
                continue

            if buf.dim() == 4:
                if not buf.is_contiguous(memory_format=torch.channels_last):
                    buf.data = buf.data.contiguous(memory_format=torch.channels_last)
                    converted_4d += 1
            elif buf.dim() == 5:
                if not buf.is_contiguous(memory_format=torch.channels_last_3d):
                    buf.data = buf.data.contiguous(memory_format=torch.channels_last_3d)
                    converted_5d += 1

        total = converted_4d + converted_5d
        stats[name] = total

        if verbose and total > 0:
            logger.info(f"channels_last applied to {name}: 4D={converted_4d}, 5D={converted_5d}")

    return stats


def to_channels_last_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """将单个 4D 张量转为 channels_last 格式"""
    if tensor.dim() == 4 and tensor.is_floating_point():
        if not tensor.is_contiguous(memory_format=torch.channels_last):
            return tensor.contiguous(memory_format=torch.channels_last)
    elif tensor.dim() == 5 and tensor.is_floating_point():
        if not tensor.is_contiguous(memory_format=torch.channels_last_3d):
            return tensor.contiguous(memory_format=torch.channels_last_3d)
    return tensor


# ═══════════════════════════════════════════════════════════════════════════
# PipelineSlicer — VAE slicing/tiling + attention slicing
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SlicingConfig:
    """Pipeline slicing 配置"""
    vae_slicing: bool = True
    vae_tiling: bool = False
    attention_slicing: bool = True
    vae_tile_sample_min_width: int = 512
    vae_tile_sample_min_height: int = 512


class PipelineSlicer:
    """
    Pipeline 推理显存优化管理器

    - VAE Slicing: 将 VAE 解码分片处理，降低峰值显存
    - VAE Tiling: 将大图分块解码，适用于超分辨率场景
    - Attention Slicing: UNet attention 分片计算
    """

    def __init__(self, config: Optional[SlicingConfig] = None):
        self.config = config or SlicingConfig()

    def apply_to_pipeline(self, pipeline: Any) -> None:
        """将 slicing 优化应用到 diffusers pipeline"""
        if pipeline is None:
            return

        # VAE Slicing
        if self.config.vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
            try:
                pipeline.enable_vae_slicing()
                logger.info("VAE slicing enabled")
            except Exception as e:
                logger.debug(f"VAE slicing not available: {e}")

        # VAE Tiling
        if self.config.vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
            try:
                pipeline.enable_vae_tiling()
                logger.info("VAE tiling enabled")
            except Exception as e:
                logger.debug(f"VAE tiling not available: {e}")

        # Attention Slicing
        if self.config.attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            try:
                pipeline.enable_attention_slicing("auto")
                logger.info("Attention slicing enabled")
            except Exception as e:
                logger.debug(f"Attention slicing not available: {e}")

    def apply_to_unet(self, unet: Any) -> None:
        """直接对 UNet 应用 attention slicing"""
        if unet is None:
            return

        if self.config.attention_slicing:
            if hasattr(unet, "enable_attention_slicing"):
                try:
                    unet.enable_attention_slicing("auto")
                    logger.info("UNet attention slicing enabled")
                except Exception as e:
                    logger.debug(f"UNet attention slicing failed: {e}")

    def apply_to_vae(self, vae: Any) -> None:
        """直接对 VAE 应用 slicing/tiling"""
        if vae is None:
            return

        if self.config.vae_slicing and hasattr(vae, "enable_slicing"):
            try:
                vae.enable_slicing()
                logger.info("VAE slicing enabled")
            except Exception as e:
                logger.debug(f"VAE slicing failed: {e}")

        if self.config.vae_tiling and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
                logger.info("VAE tiling enabled")
            except Exception as e:
                logger.debug(f"VAE tiling failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 统一 Attention 后端分发器
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AttentionBackend:
    """Attention 后端信息"""
    name: str
    available: bool
    module: Any = None


def probe_attention_backends() -> Dict[str, AttentionBackend]:
    """
    探测系统可用的 attention 后端

    Returns:
        后端名称 → AttentionBackend 映射
    """
    backends: Dict[str, AttentionBackend] = {}

    # FlashAttention 2
    try:
        import flash_attn
        backends["flash2"] = AttentionBackend("flash2", True, flash_attn)
    except ImportError:
        backends["flash2"] = AttentionBackend("flash2", False)
    backends["flash"] = backends["flash2"]

    # SageAttention
    try:
        import sageattention
        backends["sageattn"] = AttentionBackend("sageattn", True, sageattention)
    except ImportError:
        backends["sageattn"] = AttentionBackend("sageattn", False)

    # xformers
    try:
        import xformers.ops
        backends["xformers"] = AttentionBackend("xformers", True, xformers.ops)
    except ImportError:
        backends["xformers"] = AttentionBackend("xformers", False)

    # SDPA (PyTorch native)
    try:
        from torch.nn.functional import scaled_dot_product_attention
        backends["sdpa"] = AttentionBackend("sdpa", True)
    except Exception:
        backends["sdpa"] = AttentionBackend("sdpa", False)

    # torch native (always available)
    backends["torch"] = AttentionBackend("torch", True)

    return backends


def select_attention_backend(
    preferred: str = "auto",
    backends: Optional[Dict[str, AttentionBackend]] = None,
) -> str:
    """
    选择最佳可用的 attention 后端

    优先级链: preferred → sdpa → torch
    """
    if backends is None:
        backends = probe_attention_backends()

    preferred = preferred.lower().strip()

    if preferred in ("auto", ""):
        # 自动选择: flash2 > sageattn > xformers > sdpa > torch
        for name in ("flash2", "sageattn", "xformers", "sdpa", "torch"):
            if backends.get(name, AttentionBackend("", False)).available:
                return name
        return "torch"

    if preferred in {"flash", "flashattn", "flashattention", "flashattention2", "fa2"}:
        preferred = "flash2"

    if preferred in backends and backends[preferred].available:
        return preferred

    # 降级链
    fallback_chain = {
        "flash": ["sdpa", "torch"],
        "flash2": ["sdpa", "torch"],
        "sageattn": ["sdpa", "torch"],
        "xformers": ["sdpa", "torch"],
        "sdpa": ["torch"],
    }
    for fallback in fallback_chain.get(preferred, ["torch"]):
        if backends.get(fallback, AttentionBackend("", False)).available:
            logger.warning(f"Attention backend '{preferred}' not available, using '{fallback}'")
            return fallback

    return "torch"


def apply_attention_to_unet(unet: Any, backend_name: str) -> str:
    """
    将选定的 attention 后端应用到 UNet

    Returns:
        实际应用的后端名称
    """
    if unet is None:
        return backend_name

    if backend_name == "xformers":
        if hasattr(unet, "enable_xformers_memory_efficient_attention"):
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory efficient attention enabled for UNet")
                return "xformers"
            except Exception as e:
                logger.warning(f"xformers failed: {e}, falling back to sdpa")
                backend_name = "sdpa"

    if backend_name == "sdpa":
        if hasattr(unet, "set_use_sdpa"):
            unet.set_use_sdpa(True)
            logger.info("SDPA attention enabled for UNet")
            return "sdpa"
        elif hasattr(unet, "enable_sdpa"):
            try:
                unet.enable_sdpa()
                logger.info("SDPA attention enabled for UNet")
                return "sdpa"
            except Exception:
                pass

    # torch fallback — 默认行为，无需操作
    logger.info("Using native torch attention for UNet")
    return "torch"


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════

def cpu_offload_checkpoint(
    run_function: Callable,
    *args,
    preserve_rng_state: bool = True,
    **kwargs,
) -> Any:
    """Run ``run_function`` with CPU-offloaded activation checkpointing.

    Uses PyTorch's public ``torch.autograd.graph.save_on_cpu`` context so saved
    tensors are stored on CPU RAM during autograd, reducing VRAM pressure while
    keeping gradient flow correct for both positional and kwargs-only call sites.
    """
    save_on_cpu = getattr(torch.autograd.graph, "save_on_cpu", None)
    if save_on_cpu is None:
        return run_function(*args, **kwargs)

    with save_on_cpu(pin_memory=torch.cuda.is_available(), device_type="cpu"):
        return run_function(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter CPU Residency — swap adapter weights to CPU RAM between steps
# ═══════════════════════════════════════════════════════════════════════════

class AdapterCPUResidency:
    """Keep adapter (LoRA) weights on CPU RAM and move to GPU only during training.

    This is the ``vram_swap_to_ram`` contract.  Between optimizer steps, adapter
    weights are held on CPU to free VRAM for the base model.  At the start of
    each training step, weights are moved to GPU; after the optimizer step they
    are moved back to CPU.

    Usage::

        residency = AdapterCPUResidency(device="cuda")
        # After LoRA injection:
        residency.register_parameters(lora_injector.get_trainable_params())

        # In training loop:
        with residency.step_context():
            loss = training_step(batch)
            loss.backward()
            optimizer.step()
        # Weights are back on CPU after context exit.
    """

    def __init__(self, device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self._params: list[torch.nn.Parameter] = []
        self._param_original_devices: dict[int, torch.device] = {}
        self._active = False

    def register_parameters(self, params: Iterable[torch.nn.Parameter]) -> int:
        """Register parameters for CPU residency management."""
        count = 0
        for p in params:
            if isinstance(p, torch.nn.Parameter):
                pid = id(p)
                if pid not in self._param_original_devices:
                    self._param_original_devices[pid] = p.device
                    self._params.append(p)
                    count += 1
        return count

    def move_to_device(self, device: torch.device) -> None:
        """Move all registered parameters to *device*."""
        for p in self._params:
            if p.device != device:
                p.data = p.data.to(device, non_blocking=True)
                if p.grad is not None:
                    p.grad = p.grad.to(device, non_blocking=True)

    def to_gpu(self) -> None:
        """Move adapter weights to GPU."""
        self.move_to_device(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def to_cpu(self) -> None:
        """Move adapter weights to CPU RAM."""
        self.move_to_device(torch.device("cpu"))

    @contextmanager
    def step_context(self):
        """Context manager that moves weights to GPU for a training step, then back to CPU."""
        self._active = True
        self.to_gpu()
        try:
            yield
        finally:
            # Move optimizer state to CPU too
            self.to_cpu()
            self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def managed_param_count(self) -> int:
        return len(self._params)

    def estimate_vram_savings_mb(self) -> float:
        """Estimate VRAM savings in MB when adapters are on CPU."""
        total_bytes = 0
        for p in self._params:
            total_bytes += p.data.nelement() * p.data.element_size()
            if p.grad is not None:
                total_bytes += p.grad.nelement() * p.grad.element_size()
        return total_bytes / (1024 * 1024)

    def cleanup(self) -> None:
        """Clear all registered parameters."""
        self._params.clear()
        self._param_original_devices.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer state device migration (for SafeFallback step-level CPU mode)
# ═══════════════════════════════════════════════════════════════════════════

def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """将优化器状态中的所有张量移动到指定设备。

    用于 SafeFallback CPU 模式：OOM 时将模型和优化器状态移到 CPU 完成一整步训练，
    完成后再移回 CUDA。
    """
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param.grad is not None and param.grad.device != device:
                param.grad = param.grad.to(device, non_blocking=True)
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state.get(param)
            if state is None:
                continue
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device != device:
                    state[key] = value.to(device, non_blocking=True)


def estimate_vram_for_config(
    resolution: int = 1024,
    batch_size: int = 1,
    lora_rank: int = 32,
    gradient_checkpointing: bool = True,
    blocks_to_swap: int = 0,
    precision: str = "bf16",
) -> float:
    """
    估算训练所需 VRAM (GB)

    公式: V = V_base + V_batch * B * R_res * gamma_ckpt + V_rank * rank - V_swap * swap
    """
    base_gb = 8.0  # SDXL 基础开销

    res_factor = (resolution ** 2) / (1024 ** 2)
    batch_gb = 1.5 * batch_size * res_factor

    if not gradient_checkpointing:
        batch_gb *= 3.0

    precision_factor = 0.5 if precision in ("fp16", "bf16") else 1.0
    batch_gb *= precision_factor

    rank_gb = lora_rank * 0.01

    # blocks_to_swap 节省的显存
    swap_gb = blocks_to_swap * 0.3  # 每个 block 约 0.3GB

    return round(max(base_gb + batch_gb + rank_gb - swap_gb, 2.0), 1)


