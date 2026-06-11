"""CUDAGraph capture for static-shape training loops.

CUDAGraphs record GPU operations into a reusable graph, eliminating
CPU-side kernel launch overhead.  This is most effective when:
  - Model and batch shapes are fixed across iterations
  - No dynamic control flow (no conditional branching on data)
  - No Python-side data-dependent operations

Typical usage for DiT/UNet training:

    capture = CUDAGraphCapture(model, sample_input, device="cuda")
    capture.warmup(num_steps=3)
    capture.capture()
    output = capture.replay(input_tensor)

Limitations:
  - All inputs must have the same shape as the warmup sample
  - Changing batch size or resolution requires re-capture
  - Only works on CUDA devices
  - Incompatible with gradient checkpointing and block swap
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CUDAGraphCapture:
    """Capture and replay a model's forward pass using CUDA graphs.

    Parameters
    ----------
    model : nn.Module
        The model to capture.  Must be on CUDA and in eval or train mode.
    sample_inputs : dict or tuple or torch.Tensor
        Sample inputs matching the model's forward signature.
        Used for warmup and to determine tensor shapes for the graph.
    device : str or torch.device
        CUDA device for graph capture.
    """

    def __init__(
        self,
        model: nn.Module,
        sample_inputs: Any,
        device: str = "cuda",
    ):
        self.model = model
        self.device = torch.device(device)
        self.sample_inputs = sample_inputs
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: Optional[Any] = None
        self._static_output: Optional[Any] = None
        self._captured = False

    def warmup(self, num_steps: int = 3) -> None:
        """Run a few forward passes to warm up CUDA kernels and memory pools.

        This is required before capture — CUDA needs to allocate all
        memory that the graph will use.
        """
        self.model.train()
        with torch.no_grad():
            for _ in range(num_steps):
                if isinstance(self.sample_inputs, dict):
                    self._forward_dict(self.sample_inputs)
                elif isinstance(self.sample_inputs, (tuple, list)):
                    self.model(*self.sample_inputs)
                else:
                    self.model(self.sample_inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        logger.info("CUDAGraph warmup complete (%d steps)", num_steps)

    def capture(self) -> None:
        """Capture the forward pass into a CUDA graph.

        After capture, call ``replay()`` instead of ``model()`` for
        the recorded forward pass.
        """
        if self._captured:
            self._free_graph()

        self.graph = torch.cuda.CUDAGraph()

        # Allocate static input tensors (required for graph replay)
        self._static_inputs = self._make_static(self.sample_inputs)

        # Capture
        with torch.cuda.graph(self.graph):
            if isinstance(self._static_inputs, dict):
                self._static_output = self._forward_dict(self._static_inputs)
            elif isinstance(self._static_inputs, (tuple, list)):
                self._static_output = self.model(*self._static_inputs)
            else:
                self._static_output = self.model(self._static_inputs)

        self._captured = True
        logger.info("CUDAGraph captured successfully")

    def replay(self, inputs: Any) -> Any:
        """Replay the captured graph with new inputs.

        The new inputs must have the same shape and dtype as the
        warmup sample.  Data is copied into the static input tensors
        before replaying.

        Parameters
        ----------
        inputs : dict or tuple or torch.Tensor
            New inputs with the same shape as the warmup sample.

        Returns
        -------
        The model's output from the graph replay.
        """
        if not self._captured:
            raise RuntimeError("CUDAGraph not captured. Call capture() first.")

        self._copy_to_static(inputs)
        self.graph.replay()
        return self._static_output

    def _forward_dict(self, inputs: dict) -> Any:
        """Forward pass with dict inputs (for UNet-style models)."""
        return self.model(**inputs)

    def _make_static(self, inputs: Any) -> Any:
        """Create static tensors matching *inputs* shapes for graph capture."""
        if isinstance(inputs, torch.Tensor):
            return torch.zeros_like(inputs, device=self.device)
        if isinstance(inputs, dict):
            return {k: self._make_static(v) for k, v in inputs.items()}
        if isinstance(inputs, (tuple, list)):
            return type(inputs)(self._make_static(v) for v in inputs)
        # Non-tensor inputs (ints, strings, None) pass through as-is
        return inputs

    def _copy_to_static(self, inputs: Any) -> None:
        """Copy data from *inputs* into the pre-allocated static tensors."""
        if isinstance(inputs, torch.Tensor):
            self._static_inputs.copy_(inputs)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    self._static_inputs[k].copy_(v)
        elif isinstance(inputs, (tuple, list)):
            for i, v in enumerate(inputs):
                if isinstance(v, torch.Tensor):
                    self._static_inputs[i].copy_(v)

    def _free_graph(self) -> None:
        """Release the captured graph and static tensors."""
        if self.graph is not None:
            del self.graph
            self.graph = None
        self._static_inputs = None
        self._static_output = None
        self._captured = False

    def __del__(self):
        self._free_graph()

    @property
    def is_captured(self) -> bool:
        return self._captured


def cudagraph_available() -> bool:
    """Check if CUDA graphs are available on this system."""
    return torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph")
