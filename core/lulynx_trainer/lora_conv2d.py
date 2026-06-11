# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Native Conv2d LoRA wrapper used by PiSSA conv initialization."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _normalize_svd_algo(value: object) -> str:
    normalized = str(value or "rsvd").strip().lower().replace("-", "_")
    aliases = {"svd": "full", "full_svd": "full", "lowrank": "rsvd", "randomized": "rsvd"}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"rsvd", "full"} else "rsvd"


class LoRAConv2dLayer(nn.Module):
    """LoRA branch for Conv2d using down-kernel + 1x1 up projection."""

    def __init__(
        self,
        original_layer: nn.Conv2d,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        rs_lora_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.rank = max(int(rank), 1)
        self.alpha = float(alpha)
        self.rs_lora_enabled = bool(rs_lora_enabled)
        self.scaling_strategy = "alpha_over_sqrt_rank" if self.rs_lora_enabled else "alpha_over_rank"
        self.scaling = self.alpha / math.sqrt(self.rank) if self.rs_lora_enabled else self.alpha / self.rank
        self.lora_down = nn.Conv2d(
            in_channels=original_layer.in_channels,
            out_channels=self.rank,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            dilation=original_layer.dilation,
            groups=1,
            bias=False,
            padding_mode=original_layer.padding_mode,
        )
        self.lora_up = nn.Conv2d(self.rank, original_layer.out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling

    def get_weight_matrix(self) -> torch.Tensor:
        up = self.lora_up.weight[:, :, 0, 0]
        return torch.einsum("or,rihw->oihw", up, self.lora_down.weight) * self.scaling


class LoRAConv2d(nn.Module):
    """Wrap ``nn.Conv2d`` with native LoRA and optional PiSSA initialization."""

    def __init__(
        self,
        original_layer: nn.Conv2d,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        adapter_init_strategy: str = "default",
        pissa_niter: int = 1,
        svd_algo: str = "rsvd",
        pissa_oversample: int = 8,
        rs_lora_enabled: bool = False,
    ) -> None:
        super().__init__()
        if original_layer.groups != 1:
            raise ValueError("Native Conv2d LoRA currently supports groups=1 only")

        self.original = original_layer
        self.use_dora = False
        self.adapter_init_strategy = str(adapter_init_strategy or "default").strip().lower().replace("-", "_")
        self.pissa_niter = int(pissa_niter or 0)
        self.svd_algo = _normalize_svd_algo(svd_algo)
        self.pissa_oversample = max(0, int(pissa_oversample or 0))
        self.rs_lora_enabled = bool(rs_lora_enabled)
        self.applied_adapter_init_strategy = "default"
        self.lora = LoRAConv2dLayer(
            original_layer=original_layer,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            rs_lora_enabled=self.rs_lora_enabled,
        )
        self._mark_lora_leaves()

        requested_init = str(getattr(original_layer, "_adapter_init_strategy", self.adapter_init_strategy) or "default")
        requested_init = requested_init.strip().lower().replace("-", "_")
        if requested_init == "default" and getattr(original_layer, "_pissa_init", False):
            requested_init = "pissa"
        if requested_init == "pissa":
            self._apply_pissa_init(original_layer, rank)

        for param in self.original.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def _mark_lora_leaves(self) -> None:
        self.lora.lora_down._lora_leaf = True
        self.lora.lora_up._lora_leaf = True

    def _flatten_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.reshape(weight.shape[0], -1)

    def _effective_rank(self, weight: torch.Tensor, rank: int) -> int:
        return max(1, min(int(rank), int(weight.shape[0]), int(weight.shape[1])))

    def _write_low_rank_init(self, original_layer: nn.Conv2d, lora_up: torch.Tensor, lora_down: torch.Tensor) -> None:
        scaling = float(self.lora.scaling)
        up_weight = self.lora.lora_up.weight.data
        down_weight = self.lora.lora_down.weight.data
        up_weight.zero_()
        down_weight.zero_()
        rank = min(lora_up.shape[1], lora_down.shape[0], up_weight.shape[1], down_weight.shape[0])
        kernel_shape = down_weight.shape[1:]
        up_weight[:, :rank, 0, 0].copy_((lora_up[:, :rank] / scaling).to(device=up_weight.device, dtype=up_weight.dtype))
        down_weight[:rank].copy_(
            lora_down[:rank].reshape(rank, *kernel_shape).to(device=down_weight.device, dtype=down_weight.dtype)
        )
        init_up = up_weight.detach().clone()
        init_down = down_weight.detach().clone()
        if "adapter_init_lora_up" in self._buffers:
            self._buffers["adapter_init_lora_up"] = init_up
        else:
            self.register_buffer("adapter_init_lora_up", init_up, persistent=False)
        if "adapter_init_lora_down" in self._buffers:
            self._buffers["adapter_init_lora_down"] = init_down
        else:
            self.register_buffer("adapter_init_lora_down", init_down, persistent=False)

        delta = (lora_up[:, :rank] @ lora_down[:rank, :]).reshape_as(original_layer.weight.data)
        original_layer.weight.data.copy_((original_layer.weight.data.float() - delta).to(original_layer.weight.dtype))

    def _apply_pissa_init(self, original_layer: nn.Conv2d, rank: int) -> None:
        with torch.no_grad():
            weight = self._flatten_weight(original_layer.weight.data.float())
            effective_rank = self._effective_rank(weight, rank)
            if self.svd_algo == "full":
                u, s, vh = torch.linalg.svd(weight, full_matrices=False)
                u = u[:, :effective_rank]
                s = s[:effective_rank]
                v = vh[:effective_rank, :]
            else:
                q = min(min(weight.shape), effective_rank + self.pissa_oversample)
                u, s, v_lowrank = torch.svd_lowrank(weight, q=q, niter=max(self.pissa_niter, 0))
                u = u[:, :effective_rank]
                s = s[:effective_rank]
                v = v_lowrank[:, :effective_rank].T

            s_sqrt = torch.sqrt(s.clamp_min(0))
            self._write_low_rank_init(original_layer, u * s_sqrt, s_sqrt.unsqueeze(1) * v)
            self.applied_adapter_init_strategy = "pissa"
            logger.debug("PiSSA init applied to Conv2d layer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        preview_scale = float(getattr(self, "_preview_lora_scale", 1.0))
        if preview_scale <= 0:
            return original_out
        if x.device != self.lora.lora_down.weight.device or x.dtype != self.lora.lora_down.weight.dtype:
            self.lora.to(device=x.device, dtype=x.dtype)
        return original_out + self.lora(x) * preview_scale
