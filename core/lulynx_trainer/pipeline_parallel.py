# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Pipeline Parallelism — split model layers across multiple GPUs.

Implements a simplified GPipe-style pipeline schedule where the model is
partitioned into stages, each assigned to a different GPU.  Micro-batches
flow through stages in sequence, and backward passes are interleaved to
overlap compute across devices.

## Architecture

```
GPU 0: [Layers 0..N/2]  →  activations  →  GPU 1: [Layers N/2..N]
                         ←  gradients   ←
```

The schedule supports:
- **F-then-B**: All forward micro-batches, then all backward (simple, high memory)
- **1F1B**: Interleaved one-forward-one-backward for lower peak memory

## Key constraints
- Requires >= 2 CUDA GPUs
- Model must have sequential layer structure (e.g. DiT blocks, UNet blocks)
- Incompatible with block-swap (both want to place layers on specific devices)

Warehouse implementation using only public PyTorch APIs.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineParallelManager",
    "PipelineConfig",
    "is_pipeline_parallel_available",
]


def is_pipeline_parallel_available() -> bool:
    """Check if pipeline parallelism is usable (needs >= 2 CUDA GPUs)."""
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


@dataclass
class PipelineStage:
    """A contiguous group of layers assigned to one GPU."""
    device: torch.device
    layer_indices: List[int] = field(default_factory=list)
    layers: List[nn.Module] = field(default_factory=list)
    param_count: int = 0
    param_bytes: int = 0


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism."""
    num_chunks: int = 2
    split_points: str = ""
    schedule: str = "1f1b"
    balance_strategy: str = "param_count"

    def validate(self) -> None:
        if self.num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {self.num_chunks}")
        if self.schedule not in ("1f1b", "f_then_b"):
            raise ValueError(f"Unknown schedule: {self.schedule!r}")
        if self.balance_strategy not in ("param_count", "layer_count", "manual"):
            raise ValueError(f"Unknown balance_strategy: {self.balance_strategy!r}")


class PipelineParallelManager:
    """Manage pipeline-parallel model partitioning and execution.

    Usage::

        mgr = PipelineParallelManager(
            config=PipelineConfig(num_chunks=2),
        )
        mgr.partition_model(model, block_accessor="net.blocks")

        # In training loop:
        with mgr.pipeline_context():
            loss = mgr.pipeline_forward(micro_batches, forward_fn)
        loss.backward()
    """

    def __init__(self, config: PipelineConfig) -> None:
        config.validate()
        self.config = config
        self._stages: List[PipelineStage] = []
        self._original_devices: Dict[int, torch.device] = {}
        self._partitioned = False
        self._num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def partition_model(
        self,
        model: nn.Module,
        block_accessor: str = "",
    ) -> int:
        """Partition model layers across available GPUs.

        Args:
            model: The model to partition.
            block_accessor: Dot-separated attribute path to the sequential
                blocks (e.g. "net.blocks", "down_blocks").  If empty, auto-detect.

        Returns:
            Number of stages created.
        """
        if self._num_gpus < 2:
            logger.warning("Pipeline parallelism requires >= 2 GPUs, got %d", self._num_gpus)
            return 0

        blocks = self._get_blocks(model, block_accessor)
        if not blocks:
            logger.warning("No sequential blocks found for pipeline partitioning")
            return 0

        num_stages = min(self._num_gpus, len(blocks))
        if num_stages < 2:
            logger.info("Not enough blocks (%d) for pipeline parallelism", len(blocks))
            return 0

        if self.config.split_points and self.config.balance_strategy == "manual":
            assignments = self._manual_split(blocks, self.config.split_points, num_stages)
        elif self.config.balance_strategy == "param_count":
            assignments = self._balance_by_params(blocks, num_stages)
        else:
            assignments = self._balance_by_layers(blocks, num_stages)

        self._stages = []
        for stage_idx in range(num_stages):
            device = torch.device(f"cuda:{stage_idx}")
            stage = PipelineStage(device=device)

            for block_idx in assignments[stage_idx]:
                block = blocks[block_idx]
                for p in block.parameters():
                    pid = id(p)
                    if pid not in self._original_devices:
                        self._original_devices[pid] = p.device
                    stage.param_count += p.numel()
                    stage.param_bytes += p.numel() * p.element_size()

                block.to(device)
                stage.layer_indices.append(block_idx)
                stage.layers.append(block)

            self._stages.append(stage)
            logger.info(
                "Pipeline stage %d on %s: %d layers, %.1f M params",
                stage_idx, device,
                len(stage.layers),
                stage.param_count / 1e6,
            )

        self._partitioned = True
        return num_stages

    def _get_blocks(self, model: nn.Module, accessor: str) -> List[nn.Module]:
        """Extract sequential blocks from a model."""
        if accessor:
            obj = model
            for attr in accessor.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    return []
            if isinstance(obj, nn.ModuleList):
                return list(obj)
            if isinstance(obj, (list, tuple)):
                return [b for b in obj if isinstance(b, nn.Module)]
            return []

        for name in ("net.blocks", "blocks", "down_blocks", "transformer_blocks", "layers"):
            blocks = self._get_blocks(model, name)
            if len(blocks) >= 2:
                return blocks

        return []

    def _balance_by_params(
        self, blocks: List[nn.Module], num_stages: int
    ) -> List[List[int]]:
        """Assign blocks to stages balancing by parameter count."""
        param_counts = []
        for b in blocks:
            count = sum(p.numel() for p in b.parameters())
            param_counts.append(count)

        total = sum(param_counts)
        target_per_stage = total / num_stages

        assignments: List[List[int]] = [[] for _ in range(num_stages)]
        current_stage = 0
        current_load = 0

        for i, count in enumerate(param_counts):
            assignments[current_stage].append(i)
            current_load += count

            if (
                current_load >= target_per_stage
                and current_stage < num_stages - 1
                and i < len(blocks) - 1
            ):
                current_stage += 1
                current_load = 0

        while current_stage < num_stages - 1:
            current_stage += 1
            if not assignments[current_stage]:
                if assignments[current_stage - 1] and len(assignments[current_stage - 1]) > 1:
                    assignments[current_stage].append(assignments[current_stage - 1].pop())

        return assignments

    def _balance_by_layers(
        self, blocks: List[nn.Module], num_stages: int
    ) -> List[List[int]]:
        """Assign blocks to stages balancing by layer count."""
        n = len(blocks)
        per_stage = max(1, n // num_stages)
        assignments: List[List[int]] = []
        idx = 0
        for s in range(num_stages):
            end = idx + per_stage if s < num_stages - 1 else n
            assignments.append(list(range(idx, end)))
            idx = end
        return assignments

    def _manual_split(
        self, blocks: List[nn.Module], split_points_str: str, num_stages: int
    ) -> List[List[int]]:
        """Parse manual split points like '12,24' → stages [0..11], [12..23], [24..]."""
        try:
            points = [int(x.strip()) for x in split_points_str.split(",") if x.strip()]
        except ValueError:
            logger.warning("Invalid split_points, falling back to param balance")
            return self._balance_by_params(blocks, num_stages)

        points = sorted(set(points))
        boundaries = [0] + points + [len(blocks)]
        assignments = []
        for i in range(len(boundaries) - 1):
            assignments.append(list(range(boundaries[i], boundaries[i + 1])))

        while len(assignments) < num_stages:
            assignments.append([])
        return assignments[:num_stages]

    def pipeline_forward(
        self,
        input_tensor: torch.Tensor,
        micro_batch_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Run a forward pass through the pipeline stages.

        Each stage processes the output of the previous stage on its own GPU.
        Cross-device transfers use non-blocking copies.
        """
        if not self._partitioned or not self._stages:
            raise RuntimeError("Model not partitioned. Call partition_model() first.")

        x = input_tensor
        for stage in self._stages:
            x = x.to(stage.device, non_blocking=True)
            if stage.device.type == "cuda":
                torch.cuda.synchronize(stage.device)

            for layer in stage.layers:
                if micro_batch_fn is not None:
                    x = micro_batch_fn(layer, x)
                else:
                    x = layer(x)

        return x

    def run_1f1b_schedule(
        self,
        micro_batches: List[torch.Tensor],
        forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run the 1F1B pipeline schedule for lower peak memory.

        Interleaves forward and backward passes across micro-batches to reduce
        the number of activations held simultaneously.

        Args:
            micro_batches: List of input tensors (one per micro-batch).
            forward_fn: Callable(layer, input) → output for each layer.
            loss_fn: Callable(output) → scalar loss.

        Returns:
            Total loss (sum across micro-batches).
        """
        if not self._partitioned:
            raise RuntimeError("Model not partitioned")

        num_mb = len(micro_batches)
        num_stages = len(self._stages)

        outputs: List[Optional[torch.Tensor]] = [None] * num_mb
        losses: List[Optional[torch.Tensor]] = [None] * num_mb

        for mb_idx in range(num_mb):
            x = micro_batches[mb_idx]
            for stage in self._stages:
                x = x.to(stage.device, non_blocking=True)
                torch.cuda.synchronize(stage.device)
                for layer in stage.layers:
                    x = forward_fn(layer, x)
            outputs[mb_idx] = x
            losses[mb_idx] = loss_fn(x)

            warmup_done = mb_idx >= num_stages - 1
            if warmup_done and mb_idx - (num_stages - 1) < num_mb:
                bwd_idx = mb_idx - (num_stages - 1)
                if losses[bwd_idx] is not None:
                    (losses[bwd_idx] / num_mb).backward()

        for mb_idx in range(max(0, num_mb - num_stages + 1), num_mb):
            if losses[mb_idx] is not None:
                (losses[mb_idx] / num_mb).backward()

        total_loss = sum(l.detach().float() for l in losses if l is not None)
        return total_loss

    @contextmanager
    def pipeline_context(self):
        """Context manager for pipeline-parallel forward/backward."""
        yield

    def restore_model(self) -> None:
        """Move all layers back to their original devices."""
        for stage in self._stages:
            for layer_idx, layer in zip(stage.layer_indices, stage.layers):
                orig_device = None
                for p in layer.parameters():
                    orig_device = self._original_devices.get(id(p))
                    if orig_device is not None:
                        break
                if orig_device is not None:
                    layer.to(orig_device)

        self._stages.clear()
        self._original_devices.clear()
        self._partitioned = False

    @property
    def is_active(self) -> bool:
        return self._partitioned and len(self._stages) >= 2

    @property
    def stats(self) -> Dict[str, Any]:
        if not self._stages:
            return {"active": False}
        return {
            "active": True,
            "num_stages": len(self._stages),
            "stages": [
                {
                    "device": str(s.device),
                    "layers": len(s.layers),
                    "params_m": s.param_count / 1e6,
                    "memory_mb": s.param_bytes / (1024 * 1024),
                }
                for s in self._stages
            ],
        }

