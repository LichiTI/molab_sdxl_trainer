"""Small native SDXL module islands used for parity-first development."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .keymap_inspector import build_resolved_keymap_entries
from .state_loader import load_mapped_state_dict, load_mapped_state_dict_into_module
from .weight_residency import LulynxManagedConv2d, LulynxManagedLinear


SDXL_SHELL_TARGET_PREFIXES = (
    "conv_in.",
    "time_embedding.",
    "add_embedding.",
    "conv_norm_out.",
    "conv_out.",
)


class _CompatConfig(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _should_gradient_checkpoint(module: nn.Module) -> bool:
    return bool(
        module.training
        and getattr(module, "gradient_checkpointing", False)
        and torch.is_grad_enabled()
    )


def _checkpoint(function: Any, *args: torch.Tensor) -> torch.Tensor:
    return torch_checkpoint(function, *args, use_reentrant=False)


def _set_gradient_checkpointing(root: nn.Module, enabled: bool) -> None:
    for module in root.modules():
        if hasattr(module, "gradient_checkpointing"):
            setattr(module, "gradient_checkpointing", enabled)


@dataclass(frozen=True)
class _TensorShape:
    shape: tuple[int, ...]


@dataclass(frozen=True)
class NativeSDXLResnetBlockConfig:
    in_channels: int
    out_channels: int
    time_embed_dim: int
    norm_num_groups: int = 32
    conv_kernel_size: int = 3
    eps: float = 1e-5
    dropout: float = 0.0
    use_conv_shortcut: bool = False


@dataclass(frozen=True)
class NativeSDXLDownsamplerConfig:
    channels: int
    out_channels: int | None = None
    conv_kernel_size: int = 3
    padding: int = 1


@dataclass(frozen=True)
class NativeSDXLUpsamplerConfig:
    channels: int
    out_channels: int | None = None
    conv_kernel_size: int = 3
    padding: int = 1


@dataclass(frozen=True)
class NativeSDXLDownBlockConfig:
    resnets: tuple[NativeSDXLResnetBlockConfig, ...]
    downsampler: NativeSDXLDownsamplerConfig | None = None


@dataclass(frozen=True)
class NativeSDXLUpBlockConfig:
    resnets: tuple[NativeSDXLResnetBlockConfig, ...]
    upsampler: NativeSDXLUpsamplerConfig | None = None


@dataclass(frozen=True)
class NativeSDXLAttentionConfig:
    query_dim: int
    cross_attention_dim: int | None = None
    heads: int = 8
    dim_head: int = 64
    attention_backend: str = "sdpa"


@dataclass(frozen=True)
class NativeSDXLTransformerBlockConfig:
    dim: int
    cross_attention_dim: int
    heads: int
    dim_head: int
    ff_inner_dim: int
    attention_backend: str = "sdpa"


@dataclass(frozen=True)
class NativeSDXLTransformer2DConfig:
    channels: int
    transformer_blocks: tuple[NativeSDXLTransformerBlockConfig, ...]
    norm_num_groups: int = 32


@dataclass(frozen=True)
class NativeSDXLCrossAttnDownBlockConfig:
    resnets: tuple[NativeSDXLResnetBlockConfig, ...]
    attentions: tuple[NativeSDXLTransformer2DConfig, ...]
    downsampler: NativeSDXLDownsamplerConfig | None = None


@dataclass(frozen=True)
class NativeSDXLCrossAttnUpBlockConfig:
    resnets: tuple[NativeSDXLResnetBlockConfig, ...]
    attentions: tuple[NativeSDXLTransformer2DConfig, ...]
    upsampler: NativeSDXLUpsamplerConfig | None = None


@dataclass(frozen=True)
class NativeSDXLMidBlockConfig:
    resnets: tuple[NativeSDXLResnetBlockConfig, ...]
    attention: NativeSDXLTransformer2DConfig


@dataclass(frozen=True)
class NativeSDXLShellConfig:
    in_channels: int = 4
    base_channels: int = 320
    time_embed_dim: int = 1280
    add_time_embed_dim: int = 256
    add_embed_in_dim: int = 2816
    add_embed_dim: int = 1280
    out_channels: int = 4
    norm_num_groups: int = 32
    conv_kernel_size: int = 3


@dataclass(frozen=True)
class NativeSDXLUNetSkeletonConfig:
    shell: NativeSDXLShellConfig
    down_blocks: tuple[NativeSDXLDownBlockConfig | NativeSDXLCrossAttnDownBlockConfig, ...]
    mid_block: NativeSDXLMidBlockConfig
    up_blocks: tuple[NativeSDXLUpBlockConfig | NativeSDXLCrossAttnUpBlockConfig, ...]


class NativeSDXLTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear_1 = LulynxManagedLinear(in_features, out_features)
        self.linear_2 = LulynxManagedLinear(out_features, out_features)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class NativeSDXLAddEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear_1 = LulynxManagedLinear(in_features, out_features)
        self.linear_2 = LulynxManagedLinear(out_features, out_features)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class NativeSDXLResnetBlock2D(nn.Module):
    """Warehouse SDXL ResNet block matching the mapped LDM weight contract."""

    def __init__(self, config: NativeSDXLResnetBlockConfig) -> None:
        super().__init__()
        self.config = config
        padding = config.conv_kernel_size // 2
        self.norm1 = nn.GroupNorm(config.norm_num_groups, config.in_channels, eps=config.eps)
        self.conv1 = LulynxManagedConv2d(
            config.in_channels,
            config.out_channels,
            kernel_size=config.conv_kernel_size,
            padding=padding,
        )
        self.time_emb_proj = LulynxManagedLinear(config.time_embed_dim, config.out_channels)
        self.norm2 = nn.GroupNorm(config.norm_num_groups, config.out_channels, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = LulynxManagedConv2d(
            config.out_channels,
            config.out_channels,
            kernel_size=config.conv_kernel_size,
            padding=padding,
        )
        self.conv_shortcut = (
            LulynxManagedConv2d(config.in_channels, config.out_channels, kernel_size=1)
            if config.use_conv_shortcut or config.in_channels != config.out_channels
            else None
        )
        self.gradient_checkpointing = False

    def _forward_body(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        time_states = self.time_emb_proj(F.silu(temb))
        hidden_states = hidden_states + time_states[:, :, None, None]
        hidden_states = self.conv2(self.dropout(F.silu(self.norm2(hidden_states))))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return hidden_states + residual

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        if _should_gradient_checkpoint(self):
            return _checkpoint(self._forward_body, hidden_states, temb)
        return self._forward_body(hidden_states, temb)

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLDownsample2D(nn.Module):
    def __init__(self, config: NativeSDXLDownsamplerConfig) -> None:
        super().__init__()
        self.config = config
        out_channels = config.out_channels or config.channels
        self.conv = LulynxManagedConv2d(
            config.channels,
            out_channels,
            kernel_size=config.conv_kernel_size,
            stride=2,
            padding=config.padding,
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if _should_gradient_checkpoint(self):
            return _checkpoint(self.conv, hidden_states)
        return self.conv(hidden_states)

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLUpsample2D(nn.Module):
    def __init__(self, config: NativeSDXLUpsamplerConfig) -> None:
        super().__init__()
        self.config = config
        out_channels = config.out_channels or config.channels
        self.conv = LulynxManagedConv2d(
            config.channels,
            out_channels,
            kernel_size=config.conv_kernel_size,
            padding=config.padding,
        )
        self.gradient_checkpointing = False

    def _forward_body(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if _should_gradient_checkpoint(self):
            return _checkpoint(self._forward_body, hidden_states)
        return self._forward_body(hidden_states)

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLDownBlock2D(nn.Module):
    """ResNet-only SDXL down block with optional downsampler.

    Cross-attention down blocks will layer attention modules on top of this
    skeleton once their parity contract is ready.
    """

    def __init__(self, config: NativeSDXLDownBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.resnets = nn.ModuleList(
            NativeSDXLResnetBlock2D(resnet_config)
            for resnet_config in config.resnets
        )
        self.downsamplers = nn.ModuleList()
        if config.downsampler is not None:
            self.downsamplers.append(NativeSDXLDownsample2D(config.downsampler))

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
        return hidden_states

    def forward_with_skips(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            skips.append(hidden_states)
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
            skips.append(hidden_states)
        return hidden_states, skips

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLUpBlock2D(nn.Module):
    def __init__(self, config: NativeSDXLUpBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.resnets = nn.ModuleList(
            NativeSDXLResnetBlock2D(resnet_config)
            for resnet_config in config.resnets
        )
        self.upsamplers = nn.ModuleList()
        if config.upsampler is not None:
            self.upsamplers.append(NativeSDXLUpsample2D(config.upsampler))

    def forward(self, hidden_states: torch.Tensor, res_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...], temb: torch.Tensor) -> torch.Tensor:
        skips = list(res_hidden_states)
        for resnet in self.resnets:
            if not skips:
                raise ValueError("not enough skip tensors for SDXL up block")
            hidden_states = torch.cat([hidden_states, skips.pop()], dim=1)
            hidden_states = resnet(hidden_states, temb)
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
        return hidden_states

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLAttention(nn.Module):
    def __init__(self, config: NativeSDXLAttentionConfig) -> None:
        super().__init__()
        self.config = config
        inner_dim = config.heads * config.dim_head
        cross_dim = config.cross_attention_dim or config.query_dim
        self.to_q = LulynxManagedLinear(config.query_dim, inner_dim, bias=False)
        self.to_k = LulynxManagedLinear(cross_dim, inner_dim, bias=False)
        self.to_v = LulynxManagedLinear(cross_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([LulynxManagedLinear(inner_dim, config.query_dim)])

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        backend = str(self.config.attention_backend or "sdpa").lower()
        if backend == "flash2" and query.is_cuda:
            try:
                from flash_attn import flash_attn_func

                return flash_attn_func(query, key, value)
            except Exception:
                pass
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        return hidden_states.transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, tokens, _channels = hidden_states.shape
        heads = self.config.heads
        dim_head = self.config.dim_head
        query = self.to_q(hidden_states).view(batch, tokens, heads, dim_head)
        key = self.to_k(context).view(batch, context.shape[1], heads, dim_head)
        value = self.to_v(context).view(batch, context.shape[1], heads, dim_head)
        hidden_states = self._attention(query, key, value).reshape(batch, tokens, heads * dim_head)
        return self.to_out[0](hidden_states)


class NativeSDXLGEGLU(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = LulynxManagedLinear(in_features, out_features * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class NativeSDXLFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [
                NativeSDXLGEGLU(dim, inner_dim),
                nn.Identity(),
                LulynxManagedLinear(inner_dim, dim),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class NativeSDXLBasicTransformerBlock(nn.Module):
    def __init__(self, config: NativeSDXLTransformerBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn1 = NativeSDXLAttention(
            NativeSDXLAttentionConfig(
                query_dim=config.dim,
                cross_attention_dim=None,
                heads=config.heads,
                dim_head=config.dim_head,
                attention_backend=config.attention_backend,
            )
        )
        self.norm2 = nn.LayerNorm(config.dim)
        self.attn2 = NativeSDXLAttention(
            NativeSDXLAttentionConfig(
                query_dim=config.dim,
                cross_attention_dim=config.cross_attention_dim,
                heads=config.heads,
                dim_head=config.dim_head,
                attention_backend=config.attention_backend,
            )
        )
        self.norm3 = nn.LayerNorm(config.dim)
        self.ff = NativeSDXLFeedForward(config.dim, config.ff_inner_dim)
        self.gradient_checkpointing = False

    def _forward_body(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn1(self.norm1(hidden_states))
        hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states)
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        if _should_gradient_checkpoint(self):
            return _checkpoint(self._forward_body, hidden_states, encoder_hidden_states)
        return self._forward_body(hidden_states, encoder_hidden_states)


class NativeSDXLTransformer2DModel(nn.Module):
    def __init__(self, config: NativeSDXLTransformer2DConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.GroupNorm(config.norm_num_groups, config.channels, eps=1e-6)
        self.proj_in = LulynxManagedLinear(config.channels, config.channels)
        self.transformer_blocks = nn.ModuleList(
            NativeSDXLBasicTransformerBlock(block_config)
            for block_config in config.transformer_blocks
        )
        self.proj_out = LulynxManagedLinear(config.channels, config.channels)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch, channels, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return hidden_states + residual

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLCrossAttnDownBlock2D(nn.Module):
    def __init__(self, config: NativeSDXLCrossAttnDownBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.resnets = nn.ModuleList(
            NativeSDXLResnetBlock2D(resnet_config)
            for resnet_config in config.resnets
        )
        self.attentions = nn.ModuleList(
            NativeSDXLTransformer2DModel(attn_config)
            for attn_config in config.attentions
        )
        self.downsamplers = nn.ModuleList()
        if config.downsampler is not None:
            self.downsamplers.append(NativeSDXLDownsample2D(config.downsampler))

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet, attention in zip(self.resnets, self.attentions, strict=True):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attention(hidden_states, encoder_hidden_states)
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
        return hidden_states

    def forward_with_skips(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        for resnet, attention in zip(self.resnets, self.attentions, strict=True):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attention(hidden_states, encoder_hidden_states)
            skips.append(hidden_states)
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
            skips.append(hidden_states)
        return hidden_states, skips

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLCrossAttnUpBlock2D(nn.Module):
    def __init__(self, config: NativeSDXLCrossAttnUpBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.resnets = nn.ModuleList(
            NativeSDXLResnetBlock2D(resnet_config)
            for resnet_config in config.resnets
        )
        self.attentions = nn.ModuleList(
            NativeSDXLTransformer2DModel(attn_config)
            for attn_config in config.attentions
        )
        self.upsamplers = nn.ModuleList()
        if config.upsampler is not None:
            self.upsamplers.append(NativeSDXLUpsample2D(config.upsampler))

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        skips = list(res_hidden_states)
        for resnet, attention in zip(self.resnets, self.attentions, strict=True):
            if not skips:
                raise ValueError("not enough skip tensors for SDXL cross-attn up block")
            hidden_states = torch.cat([hidden_states, skips.pop()], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attention(hidden_states, encoder_hidden_states)
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
        return hidden_states

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLMidBlock2D(nn.Module):
    def __init__(self, config: NativeSDXLMidBlockConfig) -> None:
        super().__init__()
        self.config = config
        if len(config.resnets) != 2:
            raise ValueError("SDXL mid block expects exactly two ResNet blocks")
        self.resnets = nn.ModuleList(
            NativeSDXLResnetBlock2D(resnet_config)
            for resnet_config in config.resnets
        )
        self.attentions = nn.ModuleList([NativeSDXLTransformer2DModel(config.attention)])

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLUNetSkeleton(nn.Module):
    """Composable SDXL UNet skeleton built from native module islands."""

    def __init__(self, config: NativeSDXLUNetSkeletonConfig) -> None:
        super().__init__()
        self.config = config
        self.shell = NativeSDXLShellModules(config.shell)
        down_modules: list[nn.Module] = []
        for block_config in config.down_blocks:
            if isinstance(block_config, NativeSDXLCrossAttnDownBlockConfig):
                down_modules.append(NativeSDXLCrossAttnDownBlock2D(block_config))
            else:
                down_modules.append(NativeSDXLDownBlock2D(block_config))
        self.down_blocks = nn.ModuleList(down_modules)
        self.mid_block = NativeSDXLMidBlock2D(config.mid_block)
        up_modules: list[nn.Module] = []
        for block_config in config.up_blocks:
            if isinstance(block_config, NativeSDXLCrossAttnUpBlockConfig):
                up_modules.append(NativeSDXLCrossAttnUpBlock2D(block_config))
            else:
                up_modules.append(NativeSDXLUpBlock2D(block_config))
        self.up_blocks = nn.ModuleList(up_modules)
        self._block_swap_offloader: Any | None = None

    def set_block_swap_offloader(self, offloader: Any | None) -> None:
        self._block_swap_offloader = offloader

    def _enter_block(self, index: int) -> None:
        offloader = self._block_swap_offloader
        if offloader is not None:
            offloader.ensure_block_on_device(index)

    def _leave_block(self, index: int) -> None:
        offloader = self._block_swap_offloader
        if offloader is not None:
            offloader.prefetch_next(index)

    def forward(
        self,
        sample: torch.Tensor,
        timestep_embedding: torch.Tensor,
        add_embedding: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        temb = self.shell.forward_time_embedding(timestep_embedding)
        temb = temb + self.shell.forward_add_embedding(add_embedding)
        hidden_states = self.shell.forward_input(sample)
        skips: list[torch.Tensor] = [hidden_states]
        block_index = 0
        for block in self.down_blocks:
            self._enter_block(block_index)
            if isinstance(block, NativeSDXLCrossAttnDownBlock2D):
                hidden_states, block_skips = block.forward_with_skips(hidden_states, temb, encoder_hidden_states)
            else:
                hidden_states, block_skips = block.forward_with_skips(hidden_states, temb)
            self._leave_block(block_index)
            block_index += 1
            skips.extend(block_skips)
        self._enter_block(block_index)
        hidden_states = self.mid_block(hidden_states, temb, encoder_hidden_states)
        self._leave_block(block_index)
        block_index += 1
        for block in self.up_blocks:
            self._enter_block(block_index)
            resnets_count = len(block.resnets)
            block_skips = [skips.pop() for _ in range(resnets_count)]
            block_skips.reverse()
            if isinstance(block, NativeSDXLCrossAttnUpBlock2D):
                hidden_states = block(hidden_states, block_skips, temb, encoder_hidden_states)
            else:
                hidden_states = block(hidden_states, block_skips, temb)
            self._leave_block(block_index)
            block_index += 1
        return self.shell.forward_output(hidden_states)

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


class NativeSDXLUNetSkeletonCompat(nn.Module):
    """Diffusers-style call wrapper around `NativeSDXLUNetSkeleton`.

    The wrapper accepts the broad UNet2DConditionModel call shape while keeping
    the native skeleton explicit internally.  Callers may pass precomputed
    `timestep_embedding` and `add_embedding` for deterministic tests; otherwise
    SDXL-style `text_embeds` and `time_ids` are projected into `add_embedding`.
    """

    def __init__(self, skeleton: NativeSDXLUNetSkeleton) -> None:
        super().__init__()
        self.skeleton = skeleton
        self.config = _CompatConfig(
            {
                "sample_size": None,
                "in_channels": skeleton.config.shell.in_channels,
                "out_channels": skeleton.config.shell.out_channels,
                "addition_time_embed_dim": skeleton.config.shell.add_time_embed_dim,
                "time_cond_proj_dim": None,
            }
        )
        self._gradient_checkpointing = False

    @property
    def down_blocks(self) -> nn.ModuleList:
        return self.skeleton.down_blocks

    @property
    def mid_block(self) -> nn.Module:
        return self.skeleton.mid_block

    @property
    def up_blocks(self) -> nn.ModuleList:
        return self.skeleton.up_blocks

    @property
    def add_embedding(self) -> nn.Module:
        return self.skeleton.shell.add_embedding

    @property
    def time_embedding(self) -> nn.Module:
        return self.skeleton.shell.time_embedding

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def set_block_swap_offloader(self, offloader: Any | None) -> None:
        self.skeleton.set_block_swap_offloader(offloader)

    def enable_gradient_checkpointing(self, *args: Any, **kwargs: Any) -> None:
        self._gradient_checkpointing = True
        _set_gradient_checkpointing(self.skeleton, True)

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False
        _set_gradient_checkpointing(self.skeleton, False)

    def set_attn_processor(self, *args: Any, **kwargs: Any) -> None:
        # Native attention chooses its runtime backend from config today.  This
        # method keeps the diffusers training surface callable while native
        # attention processors are still converging.
        return None

    def enable_attention_slicing(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _project_timestep(self, timestep: torch.Tensor | int | float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(timestep):
            values = timestep.to(device=device)
        else:
            values = torch.tensor([timestep], device=device)
        values = values.flatten()
        if values.numel() == 1:
            values = values.repeat(batch_size)
        return timestep_embedding(
            values,
            self.skeleton.config.shell.base_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        ).to(device=device, dtype=dtype)

    def _project_time_ids(self, time_ids: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values = time_ids.to(device=device)
        if values.ndim == 1:
            values = values.reshape(batch_size, -1)
        if values.shape[0] != batch_size:
            raise ValueError(f"SDXL time_ids batch mismatch: got {values.shape[0]}, expected {batch_size}")
        time_embeds = timestep_embedding(
            values.flatten(),
            self.skeleton.config.shell.add_time_embed_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        return time_embeds.reshape(batch_size, -1).to(device=device, dtype=dtype)

    def _resolve_add_embedding(
        self,
        added_cond_kwargs: dict[str, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        added_cond_kwargs = added_cond_kwargs or {}
        if "add_embedding" in added_cond_kwargs:
            return added_cond_kwargs["add_embedding"].to(device=device, dtype=dtype)
        if "text_embeds" in added_cond_kwargs and "time_ids" in added_cond_kwargs:
            text_embeds = added_cond_kwargs["text_embeds"].to(device=device, dtype=dtype)
            if text_embeds.ndim != 2:
                raise ValueError(f"SDXL text_embeds must be rank 2, got shape {tuple(text_embeds.shape)}")
            if text_embeds.shape[0] != batch_size:
                raise ValueError(f"SDXL text_embeds batch mismatch: got {text_embeds.shape[0]}, expected {batch_size}")
            time_embeds = self._project_time_ids(added_cond_kwargs["time_ids"], batch_size, device, dtype)
            add_embedding = torch.cat([text_embeds, time_embeds], dim=-1)
            expected = self.skeleton.config.shell.add_embed_in_dim
            if add_embedding.shape[1] != expected:
                raise ValueError(
                    f"SDXL add embedding width mismatch: got {add_embedding.shape[1]}, expected {expected}"
                )
            return add_embedding
        # Keep a deterministic compatibility fallback for shape-only probes that
        # do not have tokenizer metadata yet.
        return torch.zeros(
            batch_size,
            self.skeleton.config.shell.add_embed_in_dim,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int | float,
        encoder_hidden_states: torch.Tensor,
        *,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        timestep_embedding_value = None
        if added_cond_kwargs and "timestep_embedding" in added_cond_kwargs:
            timestep_embedding_value = added_cond_kwargs["timestep_embedding"].to(device=sample.device, dtype=sample.dtype)
        else:
            timestep_embedding_value = self._project_timestep(timestep, sample.shape[0], sample.device, sample.dtype)
        add_embedding = self._resolve_add_embedding(added_cond_kwargs, sample.shape[0], sample.device, sample.dtype)
        output = self.skeleton(sample, timestep_embedding_value, add_embedding, encoder_hidden_states)
        if return_dict:
            return SimpleNamespace(sample=output)
        return (output,)


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
) -> torch.Tensor:
    half = dim // 2
    exponent = -torch.log(torch.tensor(float(max_period), device=timesteps.device))
    exponent = exponent * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / max(half - downscale_freq_shift, 1)
    args = scale * timesteps.float()[:, None] * torch.exp(exponent)[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half:], embedding[:, :half]], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class NativeSDXLShellModules(nn.Module):
    """Native SDXL shell blocks that are independent from ResNet/attention.

    This is not a full UNet.  It lets us validate mapped weights and parity for
    the low-risk outer modules before replacing the hot down/mid/up blocks.
    """

    def __init__(self, config: NativeSDXLShellConfig | None = None) -> None:
        super().__init__()
        self.config = config or NativeSDXLShellConfig()
        padding = self.config.conv_kernel_size // 2
        self.conv_in = LulynxManagedConv2d(
            self.config.in_channels,
            self.config.base_channels,
            kernel_size=self.config.conv_kernel_size,
            padding=padding,
        )
        self.time_embedding = NativeSDXLTimestepEmbedding(
            self.config.base_channels,
            self.config.time_embed_dim,
        )
        self.add_embedding = NativeSDXLAddEmbedding(
            self.config.add_embed_in_dim,
            self.config.add_embed_dim,
        )
        self.conv_norm_out = nn.GroupNorm(
            num_groups=self.config.norm_num_groups,
            num_channels=self.config.base_channels,
            eps=1e-5,
            affine=True,
        )
        self.conv_out = LulynxManagedConv2d(
            self.config.base_channels,
            self.config.out_channels,
            kernel_size=self.config.conv_kernel_size,
            padding=padding,
        )

    @property
    def native_ready_targets(self) -> tuple[str, ...]:
        return SDXL_SHELL_TARGET_PREFIXES

    def forward_input(self, sample: torch.Tensor) -> torch.Tensor:
        return self.conv_in(sample)

    def forward_time_embedding(self, timestep_embedding: torch.Tensor) -> torch.Tensor:
        return self.time_embedding(timestep_embedding)

    def forward_add_embedding(self, add_embedding: torch.Tensor) -> torch.Tensor:
        return self.add_embedding(add_embedding)

    def forward_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        return self.conv_out(hidden_states)

    def shape_metadata(self) -> dict[str, list[int]]:
        return {
            key: [int(dim) for dim in value.shape]
            for key, value in self.state_dict().items()
        }


def build_sdxl_shell_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLShellConfig:
    conv_in = state_dict["conv_in.weight"]
    time_1 = state_dict["time_embedding.linear_1.weight"]
    add_1 = state_dict["add_embedding.linear_1.weight"]
    conv_out = state_dict["conv_out.weight"]
    return NativeSDXLShellConfig(
        in_channels=int(conv_in.shape[1]),
        base_channels=int(conv_in.shape[0]),
        time_embed_dim=int(time_1.shape[0]),
        add_embed_in_dim=int(add_1.shape[1]),
        add_embed_dim=int(add_1.shape[0]),
        out_channels=int(conv_out.shape[0]),
        conv_kernel_size=int(conv_in.shape[2]),
    )


def build_sdxl_resnet_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLResnetBlockConfig:
    conv1 = state_dict["conv1.weight"]
    conv2 = state_dict["conv2.weight"]
    time_proj = state_dict["time_emb_proj.weight"]
    norm1 = state_dict["norm1.weight"]
    use_conv_shortcut = "conv_shortcut.weight" in state_dict
    return NativeSDXLResnetBlockConfig(
        in_channels=int(conv1.shape[1]),
        out_channels=int(conv2.shape[0]),
        time_embed_dim=int(time_proj.shape[1]),
        norm_num_groups=min(32, int(norm1.shape[0])),
        conv_kernel_size=int(conv1.shape[2]),
        use_conv_shortcut=use_conv_shortcut,
    )


def build_sdxl_downsampler_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLDownsamplerConfig:
    conv = state_dict["conv.weight"]
    return NativeSDXLDownsamplerConfig(
        channels=int(conv.shape[1]),
        out_channels=int(conv.shape[0]),
        conv_kernel_size=int(conv.shape[2]),
        padding=int(conv.shape[2]) // 2,
    )


def build_sdxl_upsampler_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLUpsamplerConfig:
    conv = state_dict["conv.weight"]
    return NativeSDXLUpsamplerConfig(
        channels=int(conv.shape[1]),
        out_channels=int(conv.shape[0]),
        conv_kernel_size=int(conv.shape[2]),
        padding=int(conv.shape[2]) // 2,
    )


def build_sdxl_down_block_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLDownBlockConfig:
    resnet_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("resnets.")
        }
    )
    resnet_configs = []
    for index in resnet_indices:
        prefix = f"resnets.{index}."
        resnet_state = _strip_target_prefix(state_dict, prefix)
        resnet_configs.append(build_sdxl_resnet_config_from_state(resnet_state))

    downsampler_state = _strip_target_prefix(state_dict, "downsamplers.0")
    downsampler = build_sdxl_downsampler_config_from_state(downsampler_state) if downsampler_state else None
    return NativeSDXLDownBlockConfig(
        resnets=tuple(resnet_configs),
        downsampler=downsampler,
    )


def build_sdxl_attention_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLTransformer2DConfig:
    proj_in = state_dict["proj_in.weight"]
    channels = int(proj_in.shape[0])
    block_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("transformer_blocks.")
        }
    )
    block_configs = []
    for index in block_indices:
        prefix = f"transformer_blocks.{index}."
        block_state = _strip_target_prefix(state_dict, prefix)
        query_dim = int(block_state["attn1.to_q.weight"].shape[1])
        inner_dim = int(block_state["attn1.to_q.weight"].shape[0])
        cross_attention_dim = int(block_state["attn2.to_k.weight"].shape[1])
        dim_head = 64 if inner_dim % 64 == 0 else inner_dim
        heads = max(1, inner_dim // dim_head)
        ff_inner_dim = int(block_state["ff.net.2.weight"].shape[1])
        block_configs.append(
            NativeSDXLTransformerBlockConfig(
                dim=query_dim,
                cross_attention_dim=cross_attention_dim,
                heads=heads,
                dim_head=dim_head,
                ff_inner_dim=ff_inner_dim,
            )
        )
    return NativeSDXLTransformer2DConfig(
        channels=channels,
        transformer_blocks=tuple(block_configs),
        norm_num_groups=min(32, channels),
    )


def build_sdxl_cross_attn_down_block_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLCrossAttnDownBlockConfig:
    base_config = build_sdxl_down_block_config_from_state(state_dict)
    attention_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("attentions.")
        }
    )
    attention_configs = []
    for index in attention_indices:
        prefix = f"attentions.{index}."
        attention_configs.append(build_sdxl_attention_config_from_state(_strip_target_prefix(state_dict, prefix)))
    return NativeSDXLCrossAttnDownBlockConfig(
        resnets=base_config.resnets,
        attentions=tuple(attention_configs),
        downsampler=base_config.downsampler,
    )


def build_sdxl_up_block_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLUpBlockConfig:
    resnet_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("resnets.")
        }
    )
    resnet_configs = []
    for index in resnet_indices:
        prefix = f"resnets.{index}."
        resnet_configs.append(build_sdxl_resnet_config_from_state(_strip_target_prefix(state_dict, prefix)))
    upsampler_state = _strip_target_prefix(state_dict, "upsamplers.0")
    upsampler = build_sdxl_upsampler_config_from_state(upsampler_state) if upsampler_state else None
    return NativeSDXLUpBlockConfig(resnets=tuple(resnet_configs), upsampler=upsampler)


def build_sdxl_cross_attn_up_block_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLCrossAttnUpBlockConfig:
    base_config = build_sdxl_up_block_config_from_state(state_dict)
    attention_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("attentions.")
        }
    )
    attention_configs = []
    for index in attention_indices:
        prefix = f"attentions.{index}."
        attention_configs.append(build_sdxl_attention_config_from_state(_strip_target_prefix(state_dict, prefix)))
    return NativeSDXLCrossAttnUpBlockConfig(
        resnets=base_config.resnets,
        attentions=tuple(attention_configs),
        upsampler=base_config.upsampler,
    )


def build_sdxl_mid_block_config_from_state(state_dict: dict[str, torch.Tensor]) -> NativeSDXLMidBlockConfig:
    resnet_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("resnets.")
        }
    )
    resnet_configs = []
    for index in resnet_indices:
        prefix = f"resnets.{index}."
        resnet_configs.append(build_sdxl_resnet_config_from_state(_strip_target_prefix(state_dict, prefix)))
    attention_state = _strip_target_prefix(state_dict, "attentions.0")
    return NativeSDXLMidBlockConfig(
        resnets=tuple(resnet_configs),
        attention=build_sdxl_attention_config_from_state(attention_state),
    )


def _strip_target_prefix(state_dict: dict[str, torch.Tensor], target_prefix: str) -> dict[str, torch.Tensor]:
    prefix = target_prefix.rstrip(".") + "."
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _strip_metadata_prefix(state_dict: dict[str, _TensorShape], target_prefix: str) -> dict[str, _TensorShape]:
    prefix = target_prefix.rstrip(".") + "."
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _metadata_state_from_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
) -> dict[str, _TensorShape]:
    _manifest, entries, unmatched, _metadata = build_resolved_keymap_entries(manifest_path, model_path)
    if unmatched:
        raise ValueError(f"unmatched SDXL keymap entries: {unmatched[:5]}")
    return {
        entry.target_key: _TensorShape(tuple(int(dim) for dim in entry.shape))
        for entry in entries
    }


def load_sdxl_shell_state_from_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    return load_mapped_state_dict(
        manifest_path,
        model_path,
        target_prefixes=SDXL_SHELL_TARGET_PREFIXES,
    )


def load_sdxl_resnet_state_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    state = load_mapped_state_dict(
        manifest_path,
        model_path,
        target_prefixes=(target_prefix.rstrip(".") + ".",),
    )
    return _strip_target_prefix(state, target_prefix)


def load_sdxl_down_block_state_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    state = load_mapped_state_dict(
        manifest_path,
        model_path,
        target_prefixes=(target_prefix.rstrip(".") + ".",),
    )
    return _strip_target_prefix(state, target_prefix)


def load_sdxl_attention_state_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    state = load_mapped_state_dict(
        manifest_path,
        model_path,
        target_prefixes=(target_prefix.rstrip(".") + ".",),
    )
    return _strip_target_prefix(state, target_prefix)


def load_sdxl_mid_block_state_from_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    state = load_mapped_state_dict(
        manifest_path,
        model_path,
        target_prefixes=("mid_block.",),
    )
    return _strip_target_prefix(state, "mid_block")


def load_sdxl_up_block_state_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    state = load_mapped_state_dict(
        manifest_path,
        model_path,
        target_prefixes=(target_prefix.rstrip(".") + ".",),
    )
    return _strip_target_prefix(state, target_prefix)


def build_sdxl_shell_from_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLShellModules:
    state_dict = load_sdxl_shell_state_from_manifest(manifest_path, model_path)
    shell = NativeSDXLShellModules(build_sdxl_shell_config_from_state(state_dict))
    if dtype is not None or device is not None:
        shell = shell.to(device=device, dtype=dtype)
        state_dict = {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            for key, value in state_dict.items()
        }
    missing, unexpected = shell.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"SDXL shell state mismatch: missing={missing}, unexpected={unexpected}")
    return shell


def build_sdxl_resnet_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLResnetBlock2D:
    state_dict = load_sdxl_resnet_state_from_manifest(manifest_path, target_prefix, model_path)
    block = NativeSDXLResnetBlock2D(build_sdxl_resnet_config_from_state(state_dict))
    if dtype is not None or device is not None:
        block = block.to(device=device, dtype=dtype)
        state_dict = {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            for key, value in state_dict.items()
        }
    missing, unexpected = block.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"SDXL ResNet state mismatch: missing={missing}, unexpected={unexpected}")
    return block


def build_sdxl_down_block_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLDownBlock2D:
    state_dict = load_sdxl_down_block_state_from_manifest(manifest_path, target_prefix, model_path)
    block = NativeSDXLDownBlock2D(build_sdxl_down_block_config_from_state(state_dict))
    if dtype is not None or device is not None:
        block = block.to(device=device, dtype=dtype)
        state_dict = {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            for key, value in state_dict.items()
        }
    missing, unexpected = block.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"SDXL down block state mismatch: missing={missing}, unexpected={unexpected}")
    return block


def build_sdxl_attention_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLTransformer2DModel:
    state_dict = load_sdxl_attention_state_from_manifest(manifest_path, target_prefix, model_path)
    attention = NativeSDXLTransformer2DModel(build_sdxl_attention_config_from_state(state_dict))
    if dtype is not None or device is not None:
        attention = attention.to(device=device, dtype=dtype)
        state_dict = {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            for key, value in state_dict.items()
        }
    missing, unexpected = attention.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"SDXL attention state mismatch: missing={missing}, unexpected={unexpected}")
    return attention


def build_sdxl_cross_attn_down_block_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLCrossAttnDownBlock2D:
    state_dict = load_sdxl_down_block_state_from_manifest(manifest_path, target_prefix, model_path)
    block = NativeSDXLCrossAttnDownBlock2D(build_sdxl_cross_attn_down_block_config_from_state(state_dict))
    if dtype is not None or device is not None:
        block = block.to(device=device, dtype=dtype)
        state_dict = {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            for key, value in state_dict.items()
        }
    missing, unexpected = block.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"SDXL cross-attn down block state mismatch: missing={missing}, unexpected={unexpected}")
    return block


def build_sdxl_mid_block_from_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLMidBlock2D:
    state_dict = load_sdxl_mid_block_state_from_manifest(manifest_path, model_path)
    block = NativeSDXLMidBlock2D(build_sdxl_mid_block_config_from_state(state_dict))
    if dtype is not None or device is not None:
        block = block.to(device=device, dtype=dtype)
        state_dict = {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            for key, value in state_dict.items()
        }
    missing, unexpected = block.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"SDXL mid block state mismatch: missing={missing}, unexpected={unexpected}")
    return block


def build_sdxl_up_block_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLUpBlock2D:
    state_dict = load_sdxl_up_block_state_from_manifest(manifest_path, target_prefix, model_path)
    block = NativeSDXLUpBlock2D(build_sdxl_up_block_config_from_state(state_dict))
    if dtype is not None or device is not None:
        block = block.to(device=device, dtype=dtype)
        state_dict = {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            for key, value in state_dict.items()
        }
    missing, unexpected = block.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"SDXL up block state mismatch: missing={missing}, unexpected={unexpected}")
    return block


def build_sdxl_unet_skeleton_config_from_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
) -> NativeSDXLUNetSkeletonConfig:
    metadata_state = _metadata_state_from_manifest(manifest_path, model_path)
    shell_state = {
        key: value
        for key, value in metadata_state.items()
        if key.startswith(SDXL_SHELL_TARGET_PREFIXES)
    }

    down_configs: list[NativeSDXLDownBlockConfig | NativeSDXLCrossAttnDownBlockConfig] = []
    down_indices = sorted(
        {
            int(key.split(".")[1])
            for key in metadata_state
            if key.startswith("down_blocks.")
        }
    )
    for index in down_indices:
        block_state = _strip_metadata_prefix(metadata_state, f"down_blocks.{index}")
        has_attention = any(key.startswith("attentions.") for key in block_state)
        if has_attention:
            down_configs.append(build_sdxl_cross_attn_down_block_config_from_state(block_state))  # type: ignore[arg-type]
        else:
            down_configs.append(build_sdxl_down_block_config_from_state(block_state))  # type: ignore[arg-type]

    up_configs: list[NativeSDXLUpBlockConfig | NativeSDXLCrossAttnUpBlockConfig] = []
    up_indices = sorted(
        {
            int(key.split(".")[1])
            for key in metadata_state
            if key.startswith("up_blocks.")
        }
    )
    for index in up_indices:
        block_state = _strip_metadata_prefix(metadata_state, f"up_blocks.{index}")
        has_attention = any(key.startswith("attentions.") for key in block_state)
        if has_attention:
            up_configs.append(build_sdxl_cross_attn_up_block_config_from_state(block_state))  # type: ignore[arg-type]
        else:
            up_configs.append(build_sdxl_up_block_config_from_state(block_state))  # type: ignore[arg-type]

    return NativeSDXLUNetSkeletonConfig(
        shell=build_sdxl_shell_config_from_state(shell_state),  # type: ignore[arg-type]
        down_blocks=tuple(down_configs),
        mid_block=build_sdxl_mid_block_config_from_state(_strip_metadata_prefix(metadata_state, "mid_block")),  # type: ignore[arg-type]
        up_blocks=tuple(up_configs),
    )


def build_sdxl_unet_skeleton_from_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLUNetSkeleton:
    skeleton = NativeSDXLUNetSkeleton(build_sdxl_unet_skeleton_config_from_manifest(manifest_path, model_path))
    if dtype is not None or device is not None:
        skeleton = skeleton.to(device=device, dtype=dtype)

    def _transform_key(key: str) -> str:
        return f"shell.{key}" if key.startswith(SDXL_SHELL_TARGET_PREFIXES) else key

    load_mapped_state_dict_into_module(
        skeleton,
        manifest_path,
        model_path,
        key_transform=_transform_key,
        strict=True,
    )
    return skeleton


def build_sdxl_unet_compat_from_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLUNetSkeletonCompat:
    return NativeSDXLUNetSkeletonCompat(
        build_sdxl_unet_skeleton_from_manifest(
            manifest_path,
            model_path,
            device=device,
            dtype=dtype,
        )
    )


def build_sdxl_cross_attn_up_block_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> NativeSDXLCrossAttnUpBlock2D:
    state_dict = load_sdxl_up_block_state_from_manifest(manifest_path, target_prefix, model_path)
    block = NativeSDXLCrossAttnUpBlock2D(build_sdxl_cross_attn_up_block_config_from_state(state_dict))
    if dtype is not None or device is not None:
        block = block.to(device=device, dtype=dtype)
        state_dict = {
            key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            for key, value in state_dict.items()
        }
    missing, unexpected = block.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"SDXL cross-attn up block state mismatch: missing={missing}, unexpected={unexpected}")
    return block


def build_sdxl_native_module_from_manifest(
    manifest_path: str | Path,
    target_prefix: str,
    model_path: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    normalized = target_prefix.rstrip(".")
    if normalized == "mid_block":
        return build_sdxl_mid_block_from_manifest(manifest_path, model_path, device=device, dtype=dtype)
    if normalized.startswith("down_blocks."):
        state_dict = load_sdxl_down_block_state_from_manifest(manifest_path, normalized, model_path)
        has_attention = any(key.startswith("attentions.") for key in state_dict)
        if has_attention:
            return build_sdxl_cross_attn_down_block_from_manifest(
                manifest_path,
                normalized,
                model_path,
                device=device,
                dtype=dtype,
            )
        return build_sdxl_down_block_from_manifest(
            manifest_path,
            normalized,
            model_path,
            device=device,
            dtype=dtype,
        )
    if normalized.startswith("up_blocks."):
        state_dict = load_sdxl_up_block_state_from_manifest(manifest_path, normalized, model_path)
        has_attention = any(key.startswith("attentions.") for key in state_dict)
        if has_attention:
            return build_sdxl_cross_attn_up_block_from_manifest(
                manifest_path,
                normalized,
                model_path,
                device=device,
                dtype=dtype,
            )
        return build_sdxl_up_block_from_manifest(
            manifest_path,
            normalized,
            model_path,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported SDXL native module prefix: {target_prefix}")


def compare_shape_metadata(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, list[str]]:
    expected_shapes = {str(key): list(value) for key, value in expected.items()}
    actual_shapes = {str(key): list(value) for key, value in actual.items()}
    expected_keys = set(expected_shapes)
    actual_keys = set(actual_shapes)
    mismatched = [
        key
        for key in sorted(expected_keys & actual_keys)
        if expected_shapes[key] != actual_shapes[key]
    ]
    return {
        "missing": sorted(expected_keys - actual_keys),
        "unexpected": sorted(actual_keys - expected_keys),
        "mismatched": mismatched,
    }

