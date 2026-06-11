"""Low-VRAM helpers for Flux LoRA training."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


def normalize_component_offload_strategy(
    value: Any,
    *,
    cuda_available: bool,
    total_vram_gb: float = 0.0,
    sequential_cpu_offload: bool = False,
    module_offload: bool = False,
) -> str:
    raw = str(value or "phase").strip().lower().replace("-", "_")
    if raw in {"resident", "none", "off", "disabled"}:
        return "resident"
    if raw in {"aggressive", "sequential", "sequential_cpu", "cpu", "low_vram"}:
        return "aggressive"
    if not cuda_available:
        return "resident"
    if sequential_cpu_offload or module_offload:
        return "aggressive"
    if total_vram_gb and total_vram_gb <= 20.0:
        return "aggressive"
    return "phase"


def move_trainable_parameters(module: nn.Module, device: torch.device, dtype: torch.dtype | None = None) -> int:
    moved = 0
    for param in module.parameters(recurse=True):
        if not param.requires_grad:
            continue
        target_dtype = dtype if dtype is not None and param.is_floating_point() else param.dtype
        if param.device != device or param.dtype != target_dtype:
            param.data = param.data.to(device=device, dtype=target_dtype, non_blocking=True)
            moved += 1
        if param.grad is not None and param.grad.device != device:
            param.grad.data = param.grad.data.to(device=device, non_blocking=True)
    return moved


def move_frozen_tensors(module: nn.Module, device: torch.device) -> int:
    moved = 0
    for param in module.parameters(recurse=True):
        if param.requires_grad:
            continue
        if param.device != device:
            param.data = param.data.to(device=device, non_blocking=True)
            moved += 1
    for buffer in module.buffers(recurse=True):
        if buffer.device != device:
            buffer.data = buffer.data.to(device=device, non_blocking=True)
            moved += 1
    return moved


class FluxTransformerStreamingOffloader:
    """Stream frozen Flux transformer blocks while keeping LoRA params resident."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.transformer = transformer
        self.device = device
        self.dtype = dtype
        self.log = log or (lambda _message: None)
        self.blocks: List[nn.Module] = list(getattr(transformer, "transformer_blocks", []) or [])
        self.blocks.extend(list(getattr(transformer, "single_transformer_blocks", []) or []))
        self._handles: List[Any] = []
        self.onload_count = 0
        self.offload_count = 0

    @property
    def enabled(self) -> bool:
        return bool(self.blocks) and self.device.type == "cuda"

    def install(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "reason": "Flux transformer streaming offload requires CUDA blocks"}
        move_trainable_parameters(self.transformer, self.device, self.dtype)
        self._move_static_modules_to_device()
        for block in self.blocks:
            self.offload_block(block)
            self._handles.append(block.register_full_backward_hook(self._make_backward_hook()))
        self.log(f"Flux transformer streaming offload enabled for {len(self.blocks)} blocks.")
        return {"enabled": True, "blocks": len(self.blocks), "mode": "streaming_checkpoint"}

    def _move_static_modules_to_device(self) -> None:
        for name, child in self.transformer.named_children():
            if name in {"transformer_blocks", "single_transformer_blocks"}:
                continue
            move_frozen_tensors(child, self.device)
            move_trainable_parameters(child, self.device, self.dtype)

    def prepare_for_forward(self) -> None:
        move_trainable_parameters(self.transformer, self.device, self.dtype)
        self._move_static_modules_to_device()

    def onload_block(self, block: nn.Module) -> None:
        self.onload_count += move_frozen_tensors(block, self.device)

    def offload_block(self, block: nn.Module) -> None:
        self.offload_count += move_frozen_tensors(block, torch.device("cpu"))

    def checkpoint(self, module: nn.Module, *args: Any) -> Any:
        def run(*inner_args: Any) -> Any:
            self.onload_block(module)
            return module(*inner_args)

        result = checkpoint.checkpoint(run, *args, use_reentrant=False)
        self.offload_block(module)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def _make_backward_hook(self) -> Callable[[nn.Module, Any, Any], None]:
        def hook(module: nn.Module, _grad_input: Any, _grad_output: Any) -> None:
            self.offload_block(module)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return hook

    def offload_all_blocks(self) -> None:
        for block in self.blocks:
            self.offload_block(block)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": "streaming_checkpoint" if self.enabled else "none",
            "blocks": len(self.blocks),
            "onload_count": int(self.onload_count),
            "offload_count": int(self.offload_count),
        }

    def cleanup(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


__all__ = [
    "FluxTransformerStreamingOffloader",
    "move_frozen_tensors",
    "move_trainable_parameters",
    "normalize_component_offload_strategy",
]
