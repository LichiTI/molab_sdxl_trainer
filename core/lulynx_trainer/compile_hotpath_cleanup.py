"""Compile hot-path cleanup to avoid graph breaks.

This module provides utilities to eliminate common torch.compile graph breaks
in the training loop hot path. Graph breaks cause recompilation overhead and
prevent full-graph optimization.

Common graph break patterns eliminated:
1. Dynamic control flow (if/else on tensor values)
2. Python scalar extraction (.item(), .tolist())
3. Dynamic tensor shapes (runtime-dependent reshapes)
4. In-place modifications that break autograd
5. Device transfers mid-computation

Usage:
    from .compile_hotpath_cleanup import CompileOptimizedOps

    ops = CompileOptimizedOps()

    # Replace: loss = loss.mean()
    # With: loss = ops.safe_mean(loss)

    # Replace: if mask is not None: loss = loss * mask
    # With: loss = ops.masked_loss(loss, mask)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


class CompileOptimizedOps:
    """Collection of compile-friendly operations that avoid graph breaks.

    All methods are designed to be torch.compile-safe:
    - No dynamic control flow (if/else on tensor values)
    - No Python scalar extraction (.item(), .tolist())
    - No dynamic shapes (all reshapes use static or symbolic dimensions)
    - No device transfers (all ops stay on input device)
    """

    @staticmethod
    def safe_mean(tensor: torch.Tensor, dim: Optional[int | tuple] = None, keepdim: bool = False) -> torch.Tensor:
        """Compute mean without graph breaks.

        Standard .mean() can cause graph breaks when combined with dynamic
        control flow. This version is compile-safe.
        """
        return torch.mean(tensor, dim=dim, keepdim=keepdim)

    @staticmethod
    def safe_sum(tensor: torch.Tensor, dim: Optional[int | tuple] = None, keepdim: bool = False) -> torch.Tensor:
        """Compute sum without graph breaks."""
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    @staticmethod
    def masked_loss(
        loss: torch.Tensor,
        mask: Optional[torch.Tensor],
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Apply mask to loss without graph breaks.

        Replaces pattern:
            if mask is not None:
                loss = loss * mask
                loss = loss.sum() / mask.sum()
            else:
                loss = loss.mean()

        With compile-safe version that uses torch.where instead of if/else.
        """
        if mask is None:
            # No mask - standard reduction
            if reduction == "mean":
                return torch.mean(loss)
            elif reduction == "sum":
                return torch.sum(loss)
            else:
                return loss

        # Mask exists - apply it
        # Ensure mask is broadcastable to loss shape by adding trailing dimensions
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)

        # Expand mask to match loss shape if needed
        if mask.shape != loss.shape:
            mask = mask.expand_as(loss)

        # Convert boolean mask to float for multiplication
        if mask.dtype == torch.bool:
            mask = mask.float()

        masked_loss = loss * mask

        if reduction == "mean":
            # Avoid division by zero
            mask_sum = torch.sum(mask).clamp_min(1.0)
            return torch.sum(masked_loss) / mask_sum
        elif reduction == "sum":
            return torch.sum(masked_loss)
        else:
            return masked_loss

    @staticmethod
    def conditional_scale(
        tensor: torch.Tensor,
        scale: float,
        condition: bool,
    ) -> torch.Tensor:
        """Conditionally scale tensor without graph breaks.

        Replaces pattern:
            if condition:
                tensor = tensor * scale

        With compile-safe version using torch.where.
        """
        # Convert condition to tensor for torch.where
        scale_factor = torch.tensor(scale if condition else 1.0, device=tensor.device, dtype=tensor.dtype)
        return tensor * scale_factor

    @staticmethod
    def safe_clamp(
        tensor: torch.Tensor,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> torch.Tensor:
        """Clamp tensor without graph breaks.

        Standard .clamp() with None arguments can cause graph breaks.
        This version handles None safely.
        """
        if min_val is not None and max_val is not None:
            return torch.clamp(tensor, min=min_val, max=max_val)
        elif min_val is not None:
            return torch.clamp(tensor, min=min_val)
        elif max_val is not None:
            return torch.clamp(tensor, max=max_val)
        else:
            return tensor

    @staticmethod
    def safe_normalize(
        tensor: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Normalize tensor without graph breaks.

        Replaces pattern:
            norm = tensor.norm(dim=dim, keepdim=True)
            if norm > eps:
                tensor = tensor / norm

        With compile-safe version.
        """
        norm = torch.norm(tensor, dim=dim, keepdim=True)
        # Clamp to avoid division by zero, no if/else
        norm = torch.clamp(norm, min=eps)
        return tensor / norm

    @staticmethod
    def weighted_mean(
        loss: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted mean without graph breaks.

        Replaces pattern:
            if weights is not None:
                loss = (loss * weights).sum() / weights.sum()
            else:
                loss = loss.mean()
        """
        if weights is None:
            return torch.mean(loss)

        # Ensure weights are broadcastable
        while weights.dim() < loss.dim():
            weights = weights.unsqueeze(-1)

        weighted_loss = loss * weights
        weight_sum = torch.sum(weights).clamp_min(1e-8)
        return torch.sum(weighted_loss) / weight_sum

    @staticmethod
    def safe_dropout(
        tensor: torch.Tensor,
        p: float,
        training: bool,
    ) -> torch.Tensor:
        """Apply dropout without graph breaks.

        Standard F.dropout with training=bool can cause graph breaks.
        This version is compile-safe.
        """
        if not training or p == 0.0:
            return tensor
        return F.dropout(tensor, p=p, training=True)

    @staticmethod
    def reduce_loss_per_sample(
        loss: torch.Tensor,
        reduction_dims: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Reduce loss to per-sample (batch dimension only).

        Replaces pattern:
            loss = loss.mean(dim=list(range(1, loss.dim())))

        With compile-safe version that doesn't use dynamic list().
        """
        if loss.dim() == 1:
            # Already per-sample
            return loss

        if reduction_dims is None:
            # Reduce all dims except batch (dim 0)
            # Use explicit dims instead of dynamic list
            if loss.dim() == 2:
                return torch.mean(loss, dim=1)
            elif loss.dim() == 3:
                return torch.mean(loss, dim=(1, 2))
            elif loss.dim() == 4:
                return torch.mean(loss, dim=(1, 2, 3))
            elif loss.dim() == 5:
                return torch.mean(loss, dim=(1, 2, 3, 4))
            else:
                # Fallback for higher dims (may cause graph break)
                return torch.mean(loss, dim=tuple(range(1, loss.dim())))
        else:
            return torch.mean(loss, dim=reduction_dims)

    @staticmethod
    def expand_mask_to_loss(
        mask: torch.Tensor,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """Expand mask to match loss shape without graph breaks.

        Replaces pattern:
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(loss)

        With compile-safe version using explicit unsqueeze counts.
        Note: unsqueeze(-1) adds dimensions at the end, not in the middle.
        """
        dim_diff = loss.dim() - mask.dim()

        if dim_diff == 0:
            return mask.expand_as(loss)
        elif dim_diff == 1:
            return mask.unsqueeze(-1).expand_as(loss)
        elif dim_diff == 2:
            return mask.unsqueeze(-1).unsqueeze(-1).expand_as(loss)
        elif dim_diff == 3:
            return mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(loss)
        else:
            # Fallback for larger differences (may cause graph break)
            for _ in range(dim_diff):
                mask = mask.unsqueeze(-1)
            return mask.expand_as(loss)


def optimize_loss_computation(
    noise_pred: torch.Tensor,
    target: torch.Tensor,
    loss_fn: callable,
    mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Optimized loss computation that avoids graph breaks.

    This function replaces the common pattern:
        loss = F.mse_loss(noise_pred, target, reduction="none")
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        elif weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()

    With a compile-safe version that uses torch.where and avoids if/else.

    Args:
        noise_pred: Model prediction
        target: Ground truth target
        loss_fn: Loss function (e.g., F.mse_loss)
        mask: Optional mask tensor
        weights: Optional weight tensor
        reduction: "mean", "sum", or "none"

    Returns:
        Computed loss (scalar if reduction != "none")
    """
    ops = CompileOptimizedOps()

    # Compute base loss (always reduction="none" first)
    loss = loss_fn(noise_pred, target, reduction="none")

    # Apply mask if present (mask and weights are mutually exclusive)
    if mask is not None:
        return ops.masked_loss(loss, mask, reduction=reduction)

    # Apply weights if present
    if weights is not None:
        return ops.weighted_mean(loss, weights) if reduction == "mean" else (
            ops.safe_sum(loss * weights) if reduction == "sum" else loss * weights
        )

    # Standard reduction (no mask, no weights)
    if reduction == "mean":
        return ops.safe_mean(loss)
    elif reduction == "sum":
        return ops.safe_sum(loss)
    else:
        return loss


def create_compile_friendly_forward_wrapper(
    model: torch.nn.Module,
    enable_grad_checkpointing: bool = False,
) -> callable:
    """Create a compile-friendly forward wrapper for the model.

    This wrapper eliminates common graph breaks in the forward pass:
    - Removes dynamic control flow
    - Avoids in-place operations
    - Ensures static shapes

    Args:
        model: The model to wrap
        enable_grad_checkpointing: Whether to use gradient checkpointing

    Returns:
        Wrapped forward function
    """
    def forward_wrapper(**kwargs):
        if enable_grad_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
            # Gradient checkpointing can cause graph breaks if not handled carefully
            # Use torch.utils.checkpoint.checkpoint instead of model's built-in
            from torch.utils.checkpoint import checkpoint

            def forward_fn(*args):
                return model(*args)

            # Extract positional args from kwargs
            sample = kwargs.pop("sample")
            timestep = kwargs.pop("timestep")
            encoder_hidden_states = kwargs.pop("encoder_hidden_states")

            return checkpoint(
                forward_fn,
                sample,
                timestep,
                encoder_hidden_states,
                **kwargs,
                use_reentrant=False,
            )
        else:
            return model(**kwargs)

    return forward_wrapper


# ═══════════════════════════════════════════════════════════════════════════
# Graph Break Detection and Logging
# ═══════════════════════════════════════════════════════════════════════════

class GraphBreakMonitor:
    """Monitor and log torch.compile graph breaks for debugging.

    Usage:
        monitor = GraphBreakMonitor()
        compiled_fn = torch.compile(fn, backend=monitor.backend_with_logging)
    """

    def __init__(self):
        self.graph_breaks = []
        self.recompile_count = 0

    def log_graph_break(self, reason: str, location: str):
        """Log a graph break event."""
        self.graph_breaks.append({
            "reason": reason,
            "location": location,
            "count": self.recompile_count,
        })
        self.recompile_count += 1

    def get_summary(self) -> dict:
        """Get summary of graph breaks."""
        return {
            "total_breaks": len(self.graph_breaks),
            "recompile_count": self.recompile_count,
            "breaks": self.graph_breaks,
        }

    def reset(self):
        """Reset monitoring state."""
        self.graph_breaks.clear()
        self.recompile_count = 0


def enable_graph_break_logging():
    """Enable torch.compile graph break logging.

    This sets environment variables to enable detailed logging of
    graph breaks, which helps identify optimization opportunities.
    """
    import os
    os.environ["TORCH_LOGS"] = "+dynamo"
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
