"""
LyCORIS 层实现

支持:
- LoHa (Hadamard Low-Rank)
- LoKr (Kronecker Low-Rank)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from core.safe_pickle import safe_torch_load
from .anima_train_norm_compat import resolve_anima_train_norm_state_for_layer
from .lokr_load_resolver import resolve_lokr_state_for_layer
from .generalized_adapters import (
    GLoRALinearLayer,
    GLoRAConv2dLayer,
    collect_glora_layer_state,
    load_glora_layer_state,
)
from .glokr_layer import (
    GLoKrLinearLayer,
    collect_glokr_layer_state,
    load_glokr_layer_state,
)

logger = logging.getLogger(__name__)


class LyCORISType(Enum):
    """LyCORIS 类型"""
    LOHA = "loha"
    LOKR = "lokr"
    LOCON = "locon"
    IA3 = "ia3"
    FULL = "full"
    DIAG_OFT = "diag-oft"
    GLORA = "glora"
    GLOKR = "glokr"


@dataclass
class LyCORISConfig:
    """LyCORIS 配置"""
    lycoris_type: LyCORISType = LyCORISType.LOHA
    rank: int = 8
    alpha: float = 1.0
    dropout: float = 0.0

    # LoHa 特定
    loha_use_effective: bool = True

    # LoKr 特定
    lokr_factor: int = -1  # -1 = 自动
    lokr_rank_dropout: float = 0.0   # drop entire rank dimensions
    lokr_module_dropout: float = 0.0  # drop entire LoKr layer output
    lokr_full_matrix: bool = False
    lokr_decompose_both: bool = False
    lokr_unbalanced_factorization: bool = False
    lokr_no_materialize_forward: bool = False
    lokr_no_materialize_strategy: str = "legacy"

    # Norm targeting — inject into LayerNorm/RMSNorm layers as well
    train_norm: bool = False

    # Conv2d dimensions
    conv_dim: int = 0  # 0 = same as rank
    conv_alpha: float = 0.0  # 0 = same as alpha

    # GLoRA-specific (Phase 1 standard + Phase 2 extras, all defaults off)
    glora_rank_dropout: float = 0.0
    glora_module_dropout: float = 0.0
    glora_no_materialize_forward: bool = False  # ΔW=W·A+B fast path without materializing A,B
    glora_use_tucker: bool = False              # tucker B path for Conv2d kernels >1
    glora_train_bias: bool = True               # adapt bias when base module has bias

    # GLoKr (Kronecker-parameterized Generalized adapter, research-grade)
    glokr_factor: int = -1                  # -1 = auto-pick balanced factor
    glokr_rank_dropout: float = 0.0
    glokr_module_dropout: float = 0.0
    glokr_no_materialize_forward: bool = False
    glokr_train_bias: bool = True

    # Preset target modules
    presets: str = ""  # "full", "attn-only", "attn-mlp", or custom comma-separated module list


class LoHaLayer(nn.Module):
    """
    LoHa (Hadamard Low-Rank Adaptation)
    
    公式: ΔW = (A1 ⊙ B1) @ (A2 ⊙ B2)^T
    
    相比 LoRA:
    - 更低的 rank 即可达到相似效果
    - 参数量更少
    - 训练更稳定
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        use_effective_conv: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Hadamard 分解矩阵
        # W1 = A1 ⊙ B1, W2 = A2 ⊙ B2
        # ΔW = W1 @ W2^T
        
        self.hada_w1_a = nn.Parameter(torch.empty(rank, in_features))
        self.hada_w1_b = nn.Parameter(torch.empty(rank, in_features))
        self.hada_w2_a = nn.Parameter(torch.empty(rank, out_features))
        self.hada_w2_b = nn.Parameter(torch.empty(rank, out_features))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.hada_w1_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.hada_w1_b, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.hada_w2_a, a=math.sqrt(5))
        nn.init.zeros_(self.hada_w2_b)  # B2 初始化为 0，保证初始输出为 0
    
    def get_delta_weight(self) -> torch.Tensor:
        """计算 ΔW"""
        # W1 = A1 ⊙ B1 [rank, in_features]
        w1 = self.hada_w1_a * self.hada_w1_b
        
        # W2 = A2 ⊙ B2 [rank, out_features]
        w2 = self.hada_w2_a * self.hada_w2_b
        
        # ΔW = W1^T @ W2 [in_features, out_features]
        # 转置后: [out_features, in_features]
        delta_w = w2.T @ w1
        
        return delta_w * self.scaling

    def _can_use_no_materialize_forward(self, x: torch.Tensor) -> bool:
        return x.ndim >= 1 and x.shape[-1] == self.in_features

    def _forward_no_materialize(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self.hada_w1_a * self.hada_w1_b
        w2 = self.hada_w2_a * self.hada_w2_b
        hidden = F.linear(x, w1)
        out = F.linear(hidden, w2.T)
        if self.scaling != 1.0:
            out = out * self.scaling
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.dropout(x)
        if self._can_use_no_materialize_forward(x):
            return self._forward_no_materialize(x)

        delta_w = self.get_delta_weight()
        return F.linear(x, delta_w)


class LoKrLayer(nn.Module):
    """LoKr (Kronecker Low-Rank Adaptation)."""

    FULL_MATRIX_DIM_SENTINEL = 100000
    VALID_NO_MATERIALIZE_STRATEGIES = {"auto", "legacy", "matmul"}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        factor: int = -1,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        full_matrix: bool = False,
        decompose_both: bool = False,
        unbalanced_factorization: bool = False,
        no_materialize_forward: bool = False,
        no_materialize_strategy: str = "legacy",
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, int(rank))
        self.full_matrix = bool(full_matrix) or self.rank >= self.FULL_MATRIX_DIM_SENTINEL
        self.decompose_both = bool(decompose_both)
        self.unbalanced_factorization = bool(unbalanced_factorization)
        self.no_materialize_forward = bool(no_materialize_forward)
        self.no_materialize_strategy = self._normalize_no_materialize_strategy(no_materialize_strategy)
        self.alpha = float(alpha) if alpha not in (None, 0) else float(self.rank)

        requested_factor = int(factor)
        factor = self._select_factor(self.in_features, self.out_features, requested_factor)
        self.factor = factor

        out_left, out_right = self._split_dimension(self.out_features, factor)
        in_left, in_right = self._split_dimension(self.in_features, factor)
        if self.unbalanced_factorization:
            out_left, out_right = out_right, out_left

        self.out_a = out_left
        self.out_b = out_right
        self.in_a = in_left
        self.in_b = in_right

        w1_rank_threshold = min(self.out_a, self.in_a)
        w2_rank_threshold = min(self.out_b, self.in_b)
        self.w1_decomposed = self.decompose_both and not self.full_matrix and self.rank < w1_rank_threshold
        self.w2_decomposed = not self.full_matrix and self.rank < w2_rank_threshold

        if self.w1_decomposed:
            self.lokr_w1_a = nn.Parameter(torch.empty(self.out_a, self.rank))
            self.lokr_w1_b = nn.Parameter(torch.empty(self.rank, self.in_a))
        else:
            self.lokr_w1 = nn.Parameter(torch.empty(self.out_a, self.in_a))

        if self.w2_decomposed:
            self.lokr_w2_a = nn.Parameter(torch.empty(self.out_b, self.rank))
            self.lokr_w2_b = nn.Parameter(torch.empty(self.rank, self.in_b))
        else:
            self.lokr_w2 = nn.Parameter(torch.empty(self.out_b, self.in_b))

        self.scaling = 1.0 if (not self.w1_decomposed and not self.w2_decomposed and self.full_matrix) else self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self._init_weights()

    def _make_output_zeros(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(*x.shape[:-1], self.out_features)

    def _split_dimension(self, dim: int, factor: int) -> tuple[int, int]:
        if factor <= 0 or dim % factor != 0:
            return dim, 1
        return dim // factor, factor

    def _select_factor(self, in_f: int, out_f: int, requested: int) -> int:
        candidates = []
        if requested > 0:
            candidates.append(requested)
        candidates.extend([16, 12, 8, 6, 4, 3, 2, 1])
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate > 0 and in_f % candidate == 0 and out_f % candidate == 0:
                if requested > 0 and candidate != requested:
                    logger.warning(
                        "LoKr factor=%s is incompatible with Linear(%s, %s); using factor=%s instead.",
                        requested,
                        in_f,
                        out_f,
                        candidate,
                    )
                return candidate
        return 1

    def _materialize_w1(self) -> torch.Tensor:
        if self.w1_decomposed:
            return self.lokr_w1_a @ self.lokr_w1_b
        return self.lokr_w1

    def _materialize_w2(self) -> torch.Tensor:
        if self.w2_decomposed:
            return self.lokr_w2_a @ self.lokr_w2_b
        return self.lokr_w2

    def _normalize_no_materialize_strategy(self, strategy: str) -> str:
        normalized = str(strategy or "legacy").strip().lower()
        if normalized in self.VALID_NO_MATERIALIZE_STRATEGIES:
            return normalized
        logger.warning(
            "Unsupported LoKr no-materialize strategy=%s; falling back to legacy.",
            strategy,
        )
        return "legacy"

    def _init_weights(self):
        if self.w1_decomposed:
            nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            nn.init.zeros_(self.lokr_w1_b)
        else:
            nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))

        if self.w2_decomposed:
            nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            nn.init.zeros_(self.lokr_w2_b)
        else:
            nn.init.zeros_(self.lokr_w2)

    def get_delta_weight(self) -> torch.Tensor:
        w1 = self._materialize_w1()
        w2 = self._materialize_w2()
        delta_w = torch.kron(w1, w2).reshape(self.out_features, self.in_features)
        return delta_w * self.scaling

    def _can_use_no_materialize_forward(self, x: torch.Tensor) -> bool:
        if not self.no_materialize_forward:
            return False
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            return False
        if self.rank_dropout > 0 and self.training:
            return False
        if self.factor <= 1 or (self.in_b == 1 and self.out_b == 1):
            return False
        return (
            self.in_a * self.in_b == self.in_features
            and self.out_a * self.out_b == self.out_features
        )

    def get_resolved_no_materialize_strategy(self) -> str:
        strategy = self.no_materialize_strategy
        if strategy != "auto":
            return strategy

        # Benchmarks on the FlashAttention2-targeted training stack consistently
        # favored the matmul formulation only for larger 2k-class feature widths
        # paired with wider Kronecker factors. Elsewhere the legacy F.linear path
        # stayed more stable.
        larger_dim = max(self.in_features, self.out_features)
        if larger_dim >= 2048 and self.factor >= 16:
            return "matmul"
        return "legacy"

    def _forward_no_materialize_legacy(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self._materialize_w1()
        w2 = self._materialize_w2()
        x_view = x.reshape(*x.shape[:-1], self.in_a, self.in_b)
        x_w2 = F.linear(x_view, w2)
        w1_in = x_w2.transpose(-1, -2)
        out = F.linear(w1_in, w1).transpose(-1, -2)
        out = out.reshape(*x.shape[:-1], self.out_features)
        if self.scaling != 1.0:
            out = out * self.scaling
        return out

    def _forward_no_materialize_matmul(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self._materialize_w1()
        w2 = self._materialize_w2()
        flat_x = x.reshape(-1, self.in_a, self.in_b)
        tmp = torch.matmul(flat_x, w2.t())
        out = torch.matmul(tmp.transpose(-1, -2), w1.t()).transpose(-1, -2)
        out = out.reshape(*x.shape[:-1], self.out_features)
        if self.scaling != 1.0:
            out = out * self.scaling
        return out

    def _forward_no_materialize(self, x: torch.Tensor) -> torch.Tensor:
        strategy = self.get_resolved_no_materialize_strategy()
        if strategy == "matmul":
            return self._forward_no_materialize_matmul(x)
        return self._forward_no_materialize_legacy(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.module_dropout > 0 and self.training:
            if torch.rand((), device="cpu").item() < self.module_dropout:
                return self._make_output_zeros(x)

        x = self.dropout(x)
        if self._can_use_no_materialize_forward(x):
            return self._forward_no_materialize(x)

        delta_w = self.get_delta_weight()

        if self.rank_dropout > 0 and self.training:
            drop_mask = torch.ones(self.out_features, device=delta_w.device, dtype=delta_w.dtype)
            n_drop = max(1, int(self.out_features * self.rank_dropout))
            drop_idx = torch.randperm(self.out_features, device=delta_w.device)[:n_drop]
            drop_mask[drop_idx] = 0.0
            delta_w = delta_w * drop_mask.unsqueeze(1)
            delta_w = delta_w / max(1e-6, 1.0 - float(self.rank_dropout))

        return F.linear(x, delta_w)


class LoKrConv2dLayer(LoKrLayer):
    """Kronecker-style low-rank delta for Conv2d weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        factor: int = -1,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        full_matrix: bool = False,
        decompose_both: bool = False,
        unbalanced_factorization: bool = False,
        no_materialize_forward: bool = False,
        no_materialize_strategy: str = "legacy",
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(int(dim) for dim in kernel_size)
        self.stride = tuple(int(dim) for dim in stride)
        self.padding = tuple(int(dim) for dim in padding)
        self.dilation = tuple(int(dim) for dim in dilation)
        self.groups = max(int(groups), 1)
        flat_in = (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
        super().__init__(
            in_features=flat_in,
            out_features=out_channels,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            factor=factor,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            full_matrix=full_matrix,
            decompose_both=decompose_both,
            unbalanced_factorization=unbalanced_factorization,
            no_materialize_forward=no_materialize_forward,
            no_materialize_strategy=no_materialize_strategy,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def _can_use_no_materialize_forward(self, x: torch.Tensor) -> bool:
        return False

    def _make_output_zeros(self, x: torch.Tensor) -> torch.Tensor:
        height = x.shape[-2]
        width = x.shape[-1]
        out_h = ((height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        out_w = ((width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1
        return x.new_zeros(x.shape[0], self.out_channels, out_h, out_w)

    def get_delta_weight_matrix(self) -> torch.Tensor:
        return super().get_delta_weight()

    def get_delta_weight(self) -> torch.Tensor:
        matrix = self.get_delta_weight_matrix()
        return matrix.reshape(
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.module_dropout > 0 and self.training:
            if torch.rand((), device="cpu").item() < self.module_dropout:
                return self._make_output_zeros(x)

        delta_w = self.get_delta_weight()
        if self.rank_dropout > 0 and self.training:
            drop_mask = torch.ones(self.out_channels, device=delta_w.device, dtype=delta_w.dtype)
            n_drop = max(1, int(self.out_channels * self.rank_dropout))
            drop_idx = torch.randperm(self.out_channels, device=delta_w.device)[:n_drop]
            drop_mask[drop_idx] = 0.0
            delta_w = delta_w * drop_mask.view(-1, 1, 1, 1)
            delta_w = delta_w / max(1e-6, 1.0 - float(self.rank_dropout))

        return F.conv2d(
            self.dropout(x),
            delta_w,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class LoConLayer(nn.Module):
    """
    LoCon (Convolutional LoRA)
    
    专门用于 nn.Conv2d 层的 LoRA 实现
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoCon 结构: Conv2d(kernel_size) -> Conv2d(1x1)
        self.lora_down = nn.Conv2d(
            in_channels, 
            rank, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        self.lora_up = nn.Conv2d(
            rank, 
            out_channels, 
            (1, 1), 
            bias=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播计算增量"""
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


class FullRankAdapter(nn.Module):
    """Full-rank trainable delta for a Linear layer.

    This is intentionally simple and memory-heavy, but it gives Anima a real
    fallback for LyCORIS ``full`` style experiments where low-rank structure is
    not desired.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.scaling = alpha
        self.full_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.dropout(x), self.full_weight) * self.scaling


class IA3Adapter(nn.Module):
    """IA3-style activation scaling adapter.

    The adapter keeps the original module intact and applies a learned
    per-output multiplicative scale. It is initialized as identity.
    """

    def __init__(self, out_features: int, alpha: float = 1.0):
        super().__init__()
        self.out_features = out_features
        self.alpha = alpha
        self.scaling = alpha
        self.ia3_scale = nn.Parameter(torch.zeros(out_features))

    def apply_to_output(self, output: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + torch.tanh(self.ia3_scale).to(device=output.device, dtype=output.dtype) * self.scaling
        if output.dim() >= 2 and output.shape[-1] == self.out_features:
            return output * scale
        if output.dim() == 4 and output.shape[1] == self.out_features:
            return output * scale.view(1, -1, 1, 1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DiagOFTAdapter(nn.Module):
    """Diagonal OFT-style output transform initialized as identity.

    This keeps the base projection untouched and learns a small per-channel
    diagonal transform on the projection output. It is not a full block-OFT
    implementation, but it is shape-safe for Anima DiT linear projections and
    works as an OFT-family adapter surface until a richer block version lands.
    """

    def __init__(self, out_features: int, alpha: float = 1.0):
        super().__init__()
        self.out_features = out_features
        self.alpha = alpha
        self.scaling = alpha
        self.diag_oft = nn.Parameter(torch.zeros(out_features))

    def apply_to_output(self, output: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.diag_oft).to(device=output.device, dtype=output.dtype) * self.scaling
        if output.dim() >= 2 and output.shape[-1] == self.out_features:
            return output + output * delta
        if output.dim() == 4 and output.shape[1] == self.out_features:
            return output + output * delta.view(1, -1, 1, 1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _NormAdapter(nn.Module):
    """Lightweight adapter for LayerNorm / RMSNorm layers.

    Applies a per-channel scaling + bias residual on top of the norm output:
        output = x + scale * x + bias
    where scale and bias are low-rank (rank-1) trainable parameters.
    """

    def __init__(
        self,
        norm_dim: int,
        alpha: float = 1.0,
        *,
        base_weight: Optional[torch.Tensor] = None,
        base_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.norm_dim = norm_dim
        self.alpha = alpha
        self.scaling = alpha / 1.0  # rank-1 so scaling = alpha
        self.scale = nn.Parameter(torch.ones(norm_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(norm_dim))
        if base_weight is None:
            base_weight = torch.ones(norm_dim, dtype=torch.float32)
        self.register_buffer("base_weight", base_weight.detach().float().clone(), persistent=False)
        if base_bias is None:
            self.base_bias = None
        else:
            self.register_buffer("base_bias", base_bias.detach().float().clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.scale * self.scaling) + self.bias * self.scaling


class LyCORISInjector:
    """
    LyCORIS 注入器
    
    将 LoCon/LoHa/LoKr 层注入到模型中
    支持 nn.Linear 和 nn.Conv2d
    """
    
    def __init__(self, config: LyCORISConfig):
        self.config = config
        self._injected_layers: Dict[str, nn.Module] = {}

    @property
    def injected_layers(self) -> Dict[str, nn.Module]:
        return self._injected_layers
    
    # -- preset target module resolution --------------------------------

    PRESET_TARGETS = {
        "full": ["Linear", "Conv2d", "LayerNorm", "RMSNorm"],
        "attn-only": ["to_q", "to_k", "to_v", "to_out", "q_proj", "k_proj", "v_proj", "out_proj"],
        "attn-mlp": ["to_q", "to_k", "to_v", "to_out", "q_proj", "k_proj", "v_proj", "out_proj",
                      "ff", "mlp", "fc1", "fc2", "net"],
    }

    def _configured_presets(self) -> List[str]:
        raw = str(getattr(self.config, "presets", "") or "").strip()
        if not raw:
            return []
        return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]

    def _matches_target(self, name: str, module: nn.Module, target_modules: List[str]) -> bool:
        class_name = type(module).__name__
        for target in target_modules:
            normalized = str(target or "").strip()
            if not normalized:
                continue
            if normalized in {"Linear", "nn.Linear"} and isinstance(module, nn.Linear):
                return True
            if normalized in {"Conv2d", "nn.Conv2d"} and isinstance(module, nn.Conv2d):
                return True
            if normalized in {class_name, f"nn.{class_name}"}:
                return True
            if normalized in name:
                return True
        return False

    def resolve_target_modules(self, target_modules: List[str]) -> List[str]:
        """Resolve preset names in target_modules to concrete module name patterns."""
        resolved = []
        for t in target_modules:
            preset = self.PRESET_TARGETS.get(t.lower().strip())
            if preset:
                resolved.extend(preset)
                if t.lower().strip() == "full":
                    self.config.train_norm = True
            else:
                resolved.append(t)
        return resolved

    def inject(self, model: nn.Module, target_modules: List[str], prefix: str = "") -> Dict[str, nn.Module]:
        """注入 LyCORIS 层"""
        injected_now: Dict[str, nn.Module] = {}

        # Resolve explicit target list plus config-level LyCORIS presets.
        target_modules = self.resolve_target_modules([*target_modules, *self._configured_presets()])

        for name, module in model.named_modules():
            # Standard targets: nn.Linear and nn.Conv2d
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if not self._matches_target(name, module, target_modules):
                    continue
                lycoris_layer = self._create_layer(module)
                if lycoris_layer is None:
                    continue
                lycoris_layer._block_weight_lr_scale = 1.0
                lycoris_layer._block_weight_frozen = False
                self._inject_layer(name, module, lycoris_layer)
                full_name = f"{prefix}.{name}" if prefix else name
                self._injected_layers[full_name] = lycoris_layer
                injected_now[full_name] = lycoris_layer

            # Norm targets: LayerNorm and RMSNorm when train_norm is enabled
            elif self.config.train_norm and isinstance(module, (nn.LayerNorm,)):
                # Also match RMSNorm (common in DiT models like Anima/Newbie)
                lycoris_layer = self._create_norm_layer(module)
                if lycoris_layer is not None:
                    lycoris_layer._block_weight_lr_scale = 1.0
                    lycoris_layer._block_weight_frozen = False
                    self._inject_layer(name, module, lycoris_layer)
                    full_name = f"{prefix}.{name}" if prefix else name
                    self._injected_layers[full_name] = lycoris_layer
                    injected_now[full_name] = lycoris_layer

        # Second pass: catch RMSNorm modules that aren't nn.LayerNorm subclasses
        if self.config.train_norm:
            for name, module in model.named_modules():
                if name in [n for n, _ in injected_now.items() if not prefix]:
                    continue
                if name in [n.lstrip(prefix + ".") if prefix else n for n in self._injected_layers]:
                    continue
                # Detect RMSNorm by class name (diffusers/transformers use custom classes)
                class_name = type(module).__name__
                if class_name == "RMSNorm":
                    lycoris_layer = self._create_norm_layer(module)
                    if lycoris_layer is not None:
                        lycoris_layer._block_weight_lr_scale = 1.0
                        lycoris_layer._block_weight_frozen = False
                        self._inject_layer(name, module, lycoris_layer)
                        full_name = f"{prefix}.{name}" if prefix else name
                        self._injected_layers[full_name] = lycoris_layer
                        injected_now[full_name] = lycoris_layer

        logger.info(f"[LyCORISInjector] Injected {len(injected_now)} {self.config.lycoris_type.value} layers"
                     f"{' (including norm)' if self.config.train_norm else ''}")
        return injected_now
    
    def _create_layer(self, module: nn.Module) -> Optional[nn.Module]:
        """创建 LyCORIS 层"""
        if self.config.lycoris_type == LyCORISType.LOCON:
            if isinstance(module, nn.Conv2d):
                rank = int(self.config.conv_dim or self.config.rank)
                alpha = float(self.config.conv_alpha or self.config.alpha)
                return LoConLayer(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    rank=rank,
                    alpha=alpha,
                    dropout=self.config.dropout,
                ).to(module.weight.device, dtype=module.weight.dtype)
            elif isinstance(module, nn.Linear):
                # LoCon for Linear is just LoRA
                from .lora_injector import LoRALayer
                return LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.config.rank,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout,
                ).to(module.weight.device, dtype=module.weight.dtype)
            return None

        elif self.config.lycoris_type == LyCORISType.IA3:
            if isinstance(module, nn.Linear):
                return IA3Adapter(module.out_features, alpha=self.config.alpha).to(
                    module.weight.device, dtype=module.weight.dtype
                )
            if isinstance(module, nn.Conv2d):
                return IA3Adapter(module.out_channels, alpha=self.config.alpha).to(
                    module.weight.device, dtype=module.weight.dtype
                )
            return None

        elif self.config.lycoris_type == LyCORISType.FULL:
            if not isinstance(module, nn.Linear):
                return None
            return FullRankAdapter(
                in_features=module.in_features,
                out_features=module.out_features,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
            ).to(module.weight.device, dtype=module.weight.dtype)

        elif self.config.lycoris_type == LyCORISType.DIAG_OFT:
            if isinstance(module, nn.Linear):
                return DiagOFTAdapter(module.out_features, alpha=self.config.alpha).to(
                    module.weight.device, dtype=module.weight.dtype
                )
            if isinstance(module, nn.Conv2d):
                return DiagOFTAdapter(module.out_channels, alpha=self.config.alpha).to(
                    module.weight.device, dtype=module.weight.dtype
                )
            return None

        elif self.config.lycoris_type == LyCORISType.LOHA:
            if not isinstance(module, nn.Linear):
                return None
            return LoHaLayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
            ).to(module.weight.device, dtype=module.weight.dtype)
        
        elif self.config.lycoris_type == LyCORISType.GLORA:
            train_bias = bool(getattr(self.config, "glora_train_bias", True))
            if isinstance(module, nn.Conv2d):
                rank = int(self.config.conv_dim or self.config.rank)
                alpha = float(self.config.conv_alpha or self.config.alpha)
                org_bias = module.bias if (train_bias and module.bias is not None) else None
                return GLoRAConv2dLayer(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    rank=rank,
                    alpha=alpha,
                    dropout=self.config.dropout,
                    org_weight=module.weight,
                    org_bias=org_bias,
                    rank_dropout=float(getattr(self.config, "glora_rank_dropout", 0.0)),
                    module_dropout=float(getattr(self.config, "glora_module_dropout", 0.0)),
                    use_tucker=bool(getattr(self.config, "glora_use_tucker", False)),
                ).to(module.weight.device, dtype=module.weight.dtype)
            if not isinstance(module, nn.Linear):
                return None
            org_bias = module.bias if (train_bias and module.bias is not None) else None
            return GLoRALinearLayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                org_weight=module.weight,
                org_bias=org_bias,
                rank_dropout=float(getattr(self.config, "glora_rank_dropout", 0.0)),
                module_dropout=float(getattr(self.config, "glora_module_dropout", 0.0)),
                no_materialize_forward=bool(getattr(self.config, "glora_no_materialize_forward", False)),
            ).to(module.weight.device, dtype=module.weight.dtype)

        elif self.config.lycoris_type == LyCORISType.GLOKR:
            # GLoKr is a project-original research adapter; Linear only in the
            # first cut so we don't ship un-validated Conv2d Kronecker code.
            if not isinstance(module, nn.Linear):
                return None
            train_bias = bool(getattr(self.config, "glokr_train_bias", True))
            org_bias = module.bias if (train_bias and module.bias is not None) else None
            return GLoKrLinearLayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                org_weight=module.weight,
                org_bias=org_bias,
                factor=int(getattr(self.config, "glokr_factor", -1)),
                rank_dropout=float(getattr(self.config, "glokr_rank_dropout", 0.0)),
                module_dropout=float(getattr(self.config, "glokr_module_dropout", 0.0)),
                no_materialize_forward=bool(getattr(self.config, "glokr_no_materialize_forward", False)),
            ).to(module.weight.device, dtype=module.weight.dtype)

        elif self.config.lycoris_type == LyCORISType.LOKR:
            if isinstance(module, nn.Conv2d):
                rank = int(self.config.conv_dim or self.config.rank)
                alpha = float(self.config.conv_alpha or self.config.alpha)
                return LoKrConv2dLayer(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    rank=rank,
                    alpha=alpha,
                    dropout=self.config.dropout,
                    factor=self.config.lokr_factor,
                    rank_dropout=self.config.lokr_rank_dropout,
                    module_dropout=self.config.lokr_module_dropout,
                    full_matrix=self.config.lokr_full_matrix,
                    decompose_both=self.config.lokr_decompose_both,
                    unbalanced_factorization=self.config.lokr_unbalanced_factorization,
                    no_materialize_forward=self.config.lokr_no_materialize_forward,
                    no_materialize_strategy=self.config.lokr_no_materialize_strategy,
                ).to(module.weight.device, dtype=module.weight.dtype)
            if not isinstance(module, nn.Linear):
                return None
            return LoKrLayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                factor=self.config.lokr_factor,
                rank_dropout=self.config.lokr_rank_dropout,
                module_dropout=self.config.lokr_module_dropout,
                full_matrix=self.config.lokr_full_matrix,
                decompose_both=self.config.lokr_decompose_both,
                unbalanced_factorization=self.config.lokr_unbalanced_factorization,
                no_materialize_forward=self.config.lokr_no_materialize_forward,
                no_materialize_strategy=self.config.lokr_no_materialize_strategy,
            ).to(module.weight.device, dtype=module.weight.dtype)
        
        else:
            raise ValueError(f"Unsupported LyCORIS type: {self.config.lycoris_type}")
    
    def _create_norm_layer(self, module: nn.Module) -> Optional[nn.Module]:
        """Create a LyCORIS adapter for a LayerNorm/RMSNorm module.

        For norm layers we use a rank-1 LoCon (essentially a scalar + bias
        adjustment) rather than a full LoHa/LoKr decomposition, since norm
        parameters are typically small.
        """
        if isinstance(module, nn.LayerNorm):
            norm_dim = module.normalized_shape[0]
            return _NormAdapter(
                norm_dim,
                alpha=self.config.alpha,
                base_weight=module.weight.detach(),
                base_bias=(module.bias.detach() if module.bias is not None else None),
            ).to(
                module.weight.device, dtype=module.weight.dtype
            )
        # RMSNorm (detected by class name since there's no stdlib base class)
        class_name = type(module).__name__
        if class_name == "RMSNorm":
            # RMSNorm has a .weight parameter of shape [hidden_size]
            if hasattr(module, "weight") and module.weight is not None:
                norm_dim = module.weight.shape[0]
                return _NormAdapter(
                    norm_dim,
                    alpha=self.config.alpha,
                    base_weight=module.weight.detach(),
                    base_bias=(module.bias.detach() if hasattr(module, "bias") and module.bias is not None else None),
                ).to(
                    module.weight.device, dtype=module.weight.dtype
                )
        return None

    def _inject_layer(self, name: str, module: nn.Module, lycoris_layer: nn.Module):
        """注入单个层"""
        original_forward = module.forward

        # Store the original forward so merge_export can restore it later
        module._original_forward = original_forward  # type: ignore[attr-defined]

        if isinstance(lycoris_layer, (IA3Adapter, DiagOFTAdapter)):
            def new_forward(x):
                return lycoris_layer.apply_to_output(original_forward(x))
        else:
            def new_forward(x):
                return original_forward(x) + lycoris_layer(x)

        module.forward = new_forward
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        params = []
        for layer in self._injected_layers.values():
            params.extend(param for param in layer.parameters() if param.requires_grad)
        return params

    def get_trainable_params(self) -> List[nn.Parameter]:
        return self.get_trainable_parameters()

    def get_residency_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: set[int] = set()
        for layer in self._injected_layers.values():
            for param in layer.parameters():
                if not isinstance(param, nn.Parameter):
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        return params

    def get_layer_names(self) -> List[str]:
        return list(self._injected_layers.keys())

    def freeze_layer(self, layer_name: str):
        layer = self._injected_layers.get(layer_name)
        if not layer:
            return

        layer._block_weight_lr_scale = 0.0
        layer._block_weight_frozen = True
        for param in layer.parameters():
            param.requires_grad = False

    def set_layer_lr_scale(self, layer_name: str, scale: float):
        layer = self._injected_layers.get(layer_name)
        if not layer:
            return

        scale = max(float(scale), 0.0)
        layer._block_weight_lr_scale = scale
        layer._block_weight_frozen = scale <= 0.0
        for param in layer.parameters():
            param.requires_grad = scale > 0.0

    def get_param_groups(self, base_lr: float, weight_decay: float = 0.0) -> List[Dict]:
        grouped: Dict[float, Dict[str, object]] = {}

        for layer_name, layer in self._injected_layers.items():
            params = [param for param in layer.parameters() if param.requires_grad]
            if not params:
                continue

            lr_scale = max(float(getattr(layer, "_block_weight_lr_scale", 1.0) or 0.0), 0.0)
            if lr_scale <= 0.0:
                continue

            effective_lr = float(base_lr) * lr_scale
            group_key = round(effective_lr, 12)
            group = grouped.setdefault(
                group_key,
                {
                    "params": [],
                    "lr": effective_lr,
                    "weight_decay": weight_decay,
                },
            )
            group["params"].extend(params)

        return list(grouped.values())

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = {}

        for name, layer in self._injected_layers.items():
            base_name = name.replace(".", "_")

            if isinstance(layer, LoHaLayer):
                state_dict[f"{base_name}.hada_w1_a"] = layer.hada_w1_a.data
                state_dict[f"{base_name}.hada_w1_b"] = layer.hada_w1_b.data
                state_dict[f"{base_name}.hada_w2_a"] = layer.hada_w2_a.data
                state_dict[f"{base_name}.hada_w2_b"] = layer.hada_w2_b.data
            elif isinstance(layer, LoKrLayer):
                for attr in ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b"):
                    if hasattr(layer, attr):
                        state_dict[f"{base_name}.{attr}"] = getattr(layer, attr).data
            elif isinstance(layer, (GLoRALinearLayer, GLoRAConv2dLayer)):
                state_dict.update(collect_glora_layer_state(layer, base_name))
                continue  # alpha already included
            elif isinstance(layer, GLoKrLinearLayer):
                state_dict.update(collect_glokr_layer_state(layer, base_name))
                continue
            elif isinstance(layer, LoConLayer):
                state_dict[f"{base_name}.lora_down.weight"] = layer.lora_down.weight.data
                state_dict[f"{base_name}.lora_up.weight"] = layer.lora_up.weight.data
            elif isinstance(layer, FullRankAdapter):
                state_dict[f"{base_name}.full_weight"] = layer.full_weight.data
            elif isinstance(layer, IA3Adapter):
                state_dict[f"{base_name}.ia3_scale"] = layer.ia3_scale.data
            elif isinstance(layer, DiagOFTAdapter):
                state_dict[f"{base_name}.diag_oft"] = layer.diag_oft.data
            elif isinstance(layer, _NormAdapter):
                state_dict[f"{base_name}.norm_scale"] = layer.scale.data
                state_dict[f"{base_name}.norm_bias"] = layer.bias.data
            elif hasattr(layer, 'lora_down') and hasattr(layer, 'lora_up'):
                state_dict[f"{base_name}.lora_down.weight"] = layer.lora_down.weight.data
                state_dict[f"{base_name}.lora_up.weight"] = layer.lora_up.weight.data

            state_dict[f"{base_name}.alpha"] = torch.tensor(layer.alpha)

        return state_dict
    
    def save(self, path: str):
        """保存权重"""
        from safetensors.torch import save_file
        save_file(self.get_lora_state_dict(), path)
        
        logger.info(f"[LyCORISInjector] Saved to {path}")

    def save_lora(self, path: str, metadata: Optional[Dict] = None):
        self.save(path)

    def load_lora(self, path: str, *, disable_mmap: bool = False):
        metadata: Optional[Dict[str, str]] = None
        try:
            from .safetensors_loader import load_safetensors
            state_dict = load_safetensors(path, disable_mmap=disable_mmap)
            try:
                from safetensors import safe_open

                metadata = {}
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for key, value in (handle.metadata() or {}).items():
                        metadata[str(key)] = str(value)
            except Exception:
                metadata = None
        except (ImportError, Exception):
            state_dict = safe_torch_load(path, map_location="cpu")

        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        loaded_keys, total_expected = self.load_lora_state_dict(state_dict, metadata=metadata)

        logger.info(
            "[LyCORISInjector] Loaded weights from %s: %s/%s tensors matched.",
            path,
            loaded_keys,
            total_expected,
        )

    def load_lora_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, str]] = None,
    ) -> tuple[int, int]:
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        loaded_keys = 0
        total_expected = 0

        for name, layer in self._injected_layers.items():
            base_name = name.replace(".", "_")

            if isinstance(layer, LoHaLayer):
                key_map = {
                    f"{base_name}.hada_w1_a": layer.hada_w1_a,
                    f"{base_name}.hada_w1_b": layer.hada_w1_b,
                    f"{base_name}.hada_w2_a": layer.hada_w2_a,
                    f"{base_name}.hada_w2_b": layer.hada_w2_b,
                }
            elif isinstance(layer, LoKrLayer):
                key_map = {
                    attr: getattr(layer, attr)
                    for attr in ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b")
                    if hasattr(layer, attr)
                }
            elif isinstance(layer, (GLoRALinearLayer, GLoRAConv2dLayer)):
                glora_loaded, glora_total = load_glora_layer_state(layer, state_dict, base_name)
                loaded_keys += glora_loaded
                total_expected += glora_total
                continue
            elif isinstance(layer, GLoKrLinearLayer):
                glokr_loaded, glokr_total = load_glokr_layer_state(layer, state_dict, base_name)
                loaded_keys += glokr_loaded
                total_expected += glokr_total
                continue
            elif isinstance(layer, _NormAdapter):
                key_map = {
                    f"{base_name}.norm_scale": layer.scale,
                    f"{base_name}.norm_bias": layer.bias,
                }
            elif isinstance(layer, FullRankAdapter):
                key_map = {
                    f"{base_name}.full_weight": layer.full_weight,
                }
            elif isinstance(layer, IA3Adapter):
                key_map = {
                    f"{base_name}.ia3_scale": layer.ia3_scale,
                }
            elif isinstance(layer, DiagOFTAdapter):
                key_map = {
                    f"{base_name}.diag_oft": layer.diag_oft,
                }
            elif hasattr(layer, "lora_down") and hasattr(layer, "lora_up"):
                key_map = {
                    f"{base_name}.lora_down.weight": layer.lora_down.weight,
                    f"{base_name}.lora_up.weight": layer.lora_up.weight,
                }
            else:
                key_map = {}

            total_expected += len(key_map)
            if isinstance(layer, LoKrLayer):
                resolved = resolve_lokr_state_for_layer(
                    state_dict,
                    layer=layer,
                    layer_base_name=base_name,
                    metadata=metadata,
                )
                if resolved is None:
                    continue
                for attr, param in key_map.items():
                    value = resolved.direct_assignments.get(attr)
                    if value is None:
                        continue
                    cast_value = value.to(device=param.device, dtype=param.dtype)
                    if tuple(cast_value.shape) != tuple(param.shape):
                        raise RuntimeError(
                            f"Shape mismatch for {base_name}.{attr}: checkpoint {tuple(cast_value.shape)} != layer {tuple(param.shape)}"
                        )
                    param.data.copy_(cast_value)
                    loaded_keys += 1
                continue

            if isinstance(layer, _NormAdapter):
                resolved_norm = resolve_anima_train_norm_state_for_layer(
                    state_dict,
                    layer=layer,
                    layer_base_name=base_name,
                )
                if resolved_norm is None:
                    continue
                for key, param in key_map.items():
                    attr = "norm_scale" if key.endswith(".norm_scale") else "norm_bias"
                    value = resolved_norm.get(attr)
                    if value is None:
                        continue
                    cast_value = value.to(device=param.device, dtype=param.dtype)
                    if tuple(cast_value.shape) != tuple(param.shape):
                        raise RuntimeError(
                            f"Shape mismatch for {base_name}.{attr}: checkpoint {tuple(cast_value.shape)} != layer {tuple(param.shape)}"
                        )
                    param.data.copy_(cast_value)
                    loaded_keys += 1
                continue

            for key, param in key_map.items():
                source_key = key
                if source_key in state_dict:
                    value = state_dict[source_key].to(device=param.device, dtype=param.dtype)
                    if tuple(value.shape) != tuple(param.shape):
                        raise RuntimeError(
                            f"Shape mismatch for {source_key}: checkpoint {tuple(value.shape)} != layer {tuple(param.shape)}"
                        )
                    param.data.copy_(value)
                    loaded_keys += 1

        return loaded_keys, total_expected


# ========== 便捷函数 ==========

def create_loha_layer(
    in_features: int,
    out_features: int,
    rank: int = 8,
    alpha: float = 1.0,
) -> LoHaLayer:
    """创建 LoHa 层"""
    return LoHaLayer(in_features, out_features, rank, alpha)


def create_lokr_layer(
    in_features: int,
    out_features: int,
    rank: int = 8,
    alpha: float = 1.0,
) -> LoKrLayer:
    """创建 LoKr 层"""
    return LoKrLayer(in_features, out_features, rank, alpha)


def create_lycoris_injector(
    lycoris_type: str = "loha",
    rank: int = 8,
    alpha: float = 1.0,
) -> LyCORISInjector:
    """创建 LyCORIS 注入器"""
    config = LyCORISConfig(
        lycoris_type=LyCORISType(lycoris_type),
        rank=rank,
        alpha=alpha,
    )
    return LyCORISInjector(config)
