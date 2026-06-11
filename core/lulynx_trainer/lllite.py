"""ControlNet-LLLite: lightweight LoRA-like ControlNet adapter for SDXL.

Warehouse implementation — NOT derived from any AGPL-licensed code.
Architecture based on the LLLite paper/design: small conditioning encoder +
per-layer LoRA-like adapters injected into the frozen UNet's attention blocks.

Each LLLiteModule wraps a Linear or Conv2d layer and adds:
  1. A conditioning path that embeds the control image once per step.
  2. A low-rank down→mid→up path whose mid layer fuses the conditioning embedding.
  3. Zero-initialised output so the module starts as identity.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditioning encoder — processes control image to per-resolution embeddings
# ---------------------------------------------------------------------------

class ConditioningEncoder(nn.Module):
    """Small conv stack that maps a 3-channel control image to latent-space
    conditioning features at multiple resolution depths.

    For SDXL the latent is 1/8 of the input resolution.  We produce embeddings
    at three depth levels matching the UNet's spatial resolutions:
      depth 1: 1/8  (input_blocks early,  same as latent)
      depth 2: 1/16 (input_blocks mid)
      depth 3: 1/32 (input_blocks late + middle + output_blocks)
    """

    def __init__(self, cond_emb_dim: int = 32):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        half = max(cond_emb_dim // 2, 8)

        # 1/8  → depth-1 features
        self.to_depth1 = nn.Sequential(
            nn.Conv2d(3, half, kernel_size=4, stride=4, padding=0),
            nn.SiLU(),
            nn.Conv2d(half, cond_emb_dim, kernel_size=2, stride=2, padding=0),
            nn.SiLU(),
        )
        # 1/16 → depth-2 features
        self.to_depth2 = nn.Sequential(
            nn.Conv2d(3, half, kernel_size=4, stride=4, padding=0),
            nn.SiLU(),
            nn.Conv2d(half, cond_emb_dim, kernel_size=4, stride=4, padding=0),
            nn.SiLU(),
        )
        # 1/32 → depth-3 features
        self.to_depth3 = nn.Sequential(
            nn.Conv2d(3, half, kernel_size=4, stride=4, padding=0),
            nn.SiLU(),
            nn.Conv2d(half, half, kernel_size=4, stride=4, padding=0),
            nn.SiLU(),
            nn.Conv2d(half, cond_emb_dim, kernel_size=2, stride=2, padding=0),
            nn.SiLU(),
        )

    def forward(self, control_image: torch.Tensor) -> dict[int, torch.Tensor]:
        """Return {1: emb1, 2: emb2, 3: emb3} conditioned on *control_image*."""
        return {
            1: self.to_depth1(control_image),
            2: self.to_depth2(control_image),
            3: self.to_depth3(control_image),
        }


# ---------------------------------------------------------------------------
# Per-layer LLLite modules
# ---------------------------------------------------------------------------

class LLLiteLinear(nn.Module):
    """LLLite adapter for nn.Linear layers.

    Parameters
    ----------
    in_dim : int
        Input dimension of the wrapped Linear.
    cond_emb_dim : int
        Channel dimension of the conditioning embedding.
    mlp_dim : int
        Bottleneck dimension for the low-rank path.
    dropout : float
        Optional dropout in the adapter path.
    multiplier : float
        Output scaling (1.0 = normal).
    """

    def __init__(
        self,
        in_dim: int,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        dropout: float = 0.0,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.multiplier = multiplier

        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)
        self.up = nn.Linear(mlp_dim, in_dim)

        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()

        # Zero-init the up layer → module starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self._cond_emb: Optional[torch.Tensor] = None

    def set_conditioning(self, cond_emb: torch.Tensor) -> None:
        """Cache the conditioning embedding for this step.

        *cond_emb* shape: (B, cond_emb_dim, H, W) from the ConditioningEncoder.
        For Linear targets we reshape to (B, H*W, cond_emb_dim).
        """
        B, C, H, W = cond_emb.shape
        self._cond_emb = cond_emb.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the adapter residual to be added to the original output."""
        if self._cond_emb is None:
            return torch.zeros_like(x)

        cx = self._cond_emb
        # Match batch/spatial: if cond has fewer tokens, truncate or tile
        if cx.shape[1] != x.shape[1]:
            cx = cx[:, : x.shape[1], :]
            if cx.shape[1] < x.shape[1]:
                repeat = (x.shape[1] + cx.shape[1] - 1) // cx.shape[1]
                cx = cx.repeat(1, repeat, 1)[:, : x.shape[1], :]

        h = self.down(x)
        h = F.silu(h)
        h_mid = torch.cat([h, cx], dim=-1)
        h = self.mid(h_mid)
        h = F.silu(h)
        h = self.drop(h)
        h = self.up(h)
        return h * self.multiplier


class LLLiteConv2d(nn.Module):
    """LLLite adapter for nn.Conv2d layers."""

    def __init__(
        self,
        in_channels: int,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        dropout: float = 0.0,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.multiplier = multiplier

        self.down = nn.Conv2d(in_channels, mlp_dim, kernel_size=1, stride=1, padding=0)
        self.mid = nn.Conv2d(mlp_dim + cond_emb_dim, mlp_dim, kernel_size=1, stride=1, padding=0)
        self.up = nn.Conv2d(mlp_dim, in_channels, kernel_size=1, stride=1, padding=0)

        if dropout > 0:
            self.drop = nn.Dropout2d(dropout)
        else:
            self.drop = nn.Identity()

        # Zero-init → identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self._cond_emb: Optional[torch.Tensor] = None

    def set_conditioning(self, cond_emb: torch.Tensor) -> None:
        """Cache the conditioning embedding (B, C, H, W)."""
        self._cond_emb = cond_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the adapter residual."""
        if self._cond_emb is None:
            return torch.zeros_like(x)

        cx = self._cond_emb
        # Spatial matching
        if cx.shape[2:] != x.shape[2:]:
            cx = F.interpolate(cx, size=x.shape[2:], mode="bilinear", align_corners=False)

        h = self.down(x)
        h = F.silu(h)
        h_mid = torch.cat([h, cx], dim=1)
        h = self.mid(h_mid)
        h = F.silu(h)
        h = self.drop(h)
        h = self.up(h)
        return h * self.multiplier


# ---------------------------------------------------------------------------
# Injection / removal
# ---------------------------------------------------------------------------

# Target module patterns for SDXL UNet
_DEFAULT_ATTN_TARGETS = {
    # attn1 (self-attention): Q, K, V
    "to_q",
    "to_k",
    "to_v",
    # attn2 (cross-attention): Q only (K/V receive CLIP conditioning — skip)
    # to_out is skipped by default (large dim, minimal control benefit)
}

_DEPTH_RANGES = {
    # input_blocks index → depth level
    "input": [(0, 2, 1), (3, 5, 2), (6, 99, 3)],
    # middle_block is always depth 3
    "middle": [(0, 0, 3)],
    # output_blocks index → depth level (reverse of input)
    "output": [(0, 2, 3), (3, 5, 2), (6, 99, 1)],
}


def _block_depth(block_group: str, block_idx: int) -> int:
    for start, end, depth in _DEPTH_RANGES.get(block_group, []):
        if start <= block_idx <= end:
            return depth
    return 3  # default to deepest


def inject_lllite(
    unet: nn.Module,
    cond_emb_dim: int = 32,
    mlp_dim: int = 64,
    dropout: float = 0.0,
    multiplier: float = 1.0,
    skip_input_blocks: bool = False,
    skip_output_blocks: bool = True,
    transformer_only: bool = True,
    attn_qkv_only: bool = True,
    transformer_max_block_index: Optional[int] = None,
) -> Tuple[ConditioningEncoder, List[str]]:
    """Inject LLLite adapter modules into the UNet.

    Returns
    -------
    (encoder, injected_names) : tuple
        *encoder* is the ConditioningEncoder (to be trained alongside adapters).
        *injected_names* is the list of fully-qualified module names that got
        an LLLite adapter attached.
    """
    encoder = ConditioningEncoder(cond_emb_dim=cond_emb_dim)
    injected_names: List[str] = []

    def _should_target(name: str, module: nn.Module, block_group: str) -> bool:
        if skip_input_blocks and "input_blocks" in block_group:
            return False
        if skip_output_blocks and "output_blocks" in block_group:
            return False
        # Only target attention linear layers
        if transformer_only and not isinstance(module, (nn.Linear, nn.Conv2d)):
            return False
        # Skip time/embed layers
        for skip in ("time_embed", "label_emb", "emb_layers", "norm", "proj_in", "proj_out"):
            if skip in name:
                return False
        # attn_qkv_only: only to_q, to_k, to_v in attn1, to_q in attn2
        if attn_qkv_only:
            leaf = name.rsplit(".", 1)[-1] if "." in name else name
            if leaf not in _DEFAULT_ATTN_TARGETS:
                return False
            # Skip attn2.to_k and attn2.to_v (CLIP conditioning)
            if "attn2" in name and leaf in ("to_k", "to_v"):
                return False
        return True

    # Collect targets first (cannot modify during iteration)
    targets = []
    for name, module in unet.named_modules():
        if not _should_target(name, module, name):
            continue
        if isinstance(module, nn.Linear):
            targets.append((name, module, "linear"))
        elif isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            targets.append((name, module, "conv2d"))

    for name, module, kind in targets:
        if kind == "linear":
            adapter = LLLiteLinear(
                in_dim=module.in_features,
                cond_emb_dim=cond_emb_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                multiplier=multiplier,
            )
        elif kind == "conv2d":
            adapter = LLLiteConv2d(
                in_channels=module.in_channels,
                cond_emb_dim=cond_emb_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                multiplier=multiplier,
            )
        else:
            continue

        # Determine depth from block index
        depth = 3  # default
        for group_key in ("input_blocks", "middle_block", "output_blocks"):
            if group_key in name:
                block_group = "input" if "input_blocks" in name else (
                    "middle" if "middle_block" in name else "output"
                )
                # Extract block index
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p in ("input_blocks", "output_blocks"):
                        try:
                            idx = int(parts[i + 1])
                            depth = _block_depth(block_group, idx)
                        except (ValueError, IndexError):
                            pass
                        break
                break

        adapter._depth = depth  # type: ignore[attr-defined]
        adapter._full_name = name  # type: ignore[attr-defined]
        # Attach to the parent module
        parent_name, leaf_name = name.rsplit(".", 1)
        parent = unet.get_submodule(parent_name)
        setattr(parent, leaf_name + "_lllite", adapter)
        # Store reference to original for forward hook
        adapter._original = module  # type: ignore[attr-defined]
        injected_names.append(name)

    # Register forward hooks to add LLLite residual
    _hooks: List[torch.utils.hooks.RemovableHook] = []

    def _make_hook(adapter: nn.Module):
        def hook(module: nn.Module, input: tuple, output: torch.Tensor):
            return output + adapter(input[0])
        return hook

    for name in injected_names:
        adapter = _get_adapter(unet, name)
        original = unet.get_submodule(name)
        hook = original.register_forward_hook(_make_hook(adapter))
        _hooks.append(hook)

    # Store hooks and encoder on the UNet for cleanup
    unet._lllite_hooks = _hooks  # type: ignore[attr-defined]
    unet._lllite_encoder = encoder  # type: ignore[attr-defined]
    unet._lllite_adapters = injected_names  # type: ignore[attr-defined]
    unet._lllite_injected = True  # type: ignore[attr-defined]

    logger.info(
        f"LLLite injected {len(injected_names)} adapter modules "
        f"(cond_emb_dim={cond_emb_dim}, mlp_dim={mlp_dim})"
    )
    return encoder, injected_names


def _get_adapter(unet: nn.Module, original_name: str) -> nn.Module:
    """Retrieve the LLLite adapter for *original_name*."""
    parent_name, leaf_name = original_name.rsplit(".", 1)
    parent = unet.get_submodule(parent_name)
    return getattr(parent, leaf_name + "_lllite")


def set_lllite_conditioning(
    unet: nn.Module,
    cond_embeddings: dict[int, torch.Tensor],
) -> None:
    """Push conditioning embeddings from the encoder to all adapters."""
    if not getattr(unet, "_lllite_injected", False):
        return
    for name in unet._lllite_adapters:  # type: ignore[attr-defined]
        adapter = _get_adapter(unet, name)
        depth = getattr(adapter, "_depth", 3)
        emb = cond_embeddings.get(depth)
        if emb is not None:
            adapter.set_conditioning(emb)


def remove_lllite(unet: nn.Module) -> None:
    """Remove all LLLite adapters and hooks from the UNet."""
    # Remove hooks
    for hook in getattr(unet, "_lllite_hooks", []):
        hook.remove()

    # Remove adapter modules
    for name in getattr(unet, "_lllite_adapters", []):
        parent_name, leaf_name = name.rsplit(".", 1)
        parent = unet.get_submodule(parent_name)
        attr = leaf_name + "_lllite"
        if hasattr(parent, attr):
            delattr(parent, attr)

    # Clean up attributes
    for attr in ("_lllite_hooks", "_lllite_encoder", "_lllite_adapters", "_lllite_injected"):
        if hasattr(unet, attr):
            delattr(unet, attr)


def get_lllite_state_dict(unet: nn.Module) -> dict[str, torch.Tensor]:
    """Collect all LLLite adapter weights into a state dict."""
    state = {}
    encoder = getattr(unet, "_lllite_encoder", None)
    if encoder is not None:
        for k, v in encoder.state_dict().items():
            state[f"lllite_encoder.{k}"] = v

    for name in getattr(unet, "_lllite_adapters", []):
        adapter = _get_adapter(unet, name)
        for k, v in adapter.state_dict().items():
            state[f"lllite_adapters.{name}.{k}"] = v

    return state


def load_lllite_state_dict(unet: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load LLLite adapter weights from a state dict."""
    encoder = getattr(unet, "_lllite_encoder", None)
    if encoder is not None:
        enc_state = {k.removeprefix("lllite_encoder."): v
                     for k, v in state_dict.items() if k.startswith("lllite_encoder.")}
        if enc_state:
            encoder.load_state_dict(enc_state, strict=False)

    for name in getattr(unet, "_lllite_adapters", []):
        adapter = _get_adapter(unet, name)
        prefix = f"lllite_adapters.{name}."
        ad_state = {k.removeprefix(prefix): v
                    for k, v in state_dict.items() if k.startswith(prefix)}
        if ad_state:
            adapter.load_state_dict(ad_state, strict=False)

