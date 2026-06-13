"""
LoRA 注入器

在模型的注意力层注入低秩矩阵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set, Iterable
import logging
import math
from ..lulynx.dora_layer import DoRALinear
from .model_family import get_model_family, ModelFamily
from .tlora import TLoRALinear

logger = logging.getLogger(__name__)


def _normalize_adapter_init_strategy(value: object) -> str:
    normalized = str(value or "default").strip().lower().replace("-", "_")
    aliases = {
        "": "default",
        "none": "default",
        "off": "default",
        "disabled": "default",
        "standard": "default",
        "kaiming": "default",
        "pissa_init": "pissa",
        "o_lora": "olora",
        "orthogonal_lora": "olora",
        "loft_q": "loftq",
        "loftq_init": "loftq",
    }
    normalized = aliases.get(normalized.replace(" ", ""), normalized)
    return normalized if normalized in {"default", "pissa", "olora", "loftq"} else "default"


def _normalize_loftq_quant_type(value: object) -> str:
    normalized = str(value or "rowwise").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "rowwise",
        "default": "rowwise",
        "uniform": "rowwise",
        "symmetric": "rowwise",
        "per_channel": "rowwise",
        "per_output": "rowwise",
        "global": "tensorwise",
        "per_tensor": "tensorwise",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"rowwise", "tensorwise"} else "rowwise"


def _normalize_svd_algo(value: object) -> str:
    normalized = str(value or "rsvd").strip().lower().replace("-", "_")
    aliases = {"svd": "full", "full_svd": "full", "lowrank": "rsvd", "randomized": "rsvd"}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"rsvd", "full"} else "rsvd"


def _normalize_adapter_init_export_mode(value: object) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "auto",
        "default": "auto",
        "none": "raw",
        "off": "raw",
        "native": "raw",
        "training": "raw",
        "lora无损兼容导出": "lora_compatible",
        "compatible": "lora_compatible",
        "standard": "lora_compatible",
        "standard_lora": "lora_compatible",
        "lora_compatible_export": "lora_compatible",
        "lora快速近似导出": "approximate",
        "fast": "approximate",
        "quick": "approximate",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"auto", "raw", "lora_compatible", "approximate"} else "auto"


class _LoRABranchRecomputeFn(torch.autograd.Function):
    """LoRA branch with a smaller autograd tape.

    PyTorch's default ``linear(linear(x))`` path keeps the intermediate rank
    projection for backward.  This function keeps the original input and
    adapter weights only, then recomputes the down projection in backward.
    It also avoids saving any temporary casted copy of ``x``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, down_weight: torch.Tensor, up_weight: torch.Tensor, scaling: float) -> torch.Tensor:
        input_shape = tuple(x.shape)
        x_2d = x.reshape(-1, input_shape[-1])
        compute_dtype = down_weight.dtype
        x_compute = x_2d if x_2d.dtype == compute_dtype else x_2d.to(dtype=compute_dtype)
        hidden = F.linear(x_compute, down_weight)
        if hidden.dtype != up_weight.dtype:
            hidden = hidden.to(dtype=up_weight.dtype)
        out = F.linear(hidden, up_weight)
        if scaling != 1.0:
            out = out * scaling
        ctx.save_for_backward(x, down_weight, up_weight)
        ctx.scaling = float(scaling)
        ctx.input_shape = input_shape
        return out.reshape(*input_shape[:-1], up_weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, down_weight, up_weight = ctx.saved_tensors
        input_shape = ctx.input_shape
        scaling = ctx.scaling

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        compute_dtype = down_weight.dtype
        grad_compute = grad_2d if grad_2d.dtype == compute_dtype else grad_2d.to(dtype=compute_dtype)
        if scaling != 1.0:
            grad_compute = grad_compute * scaling

        x_2d = x.reshape(-1, input_shape[-1])
        x_compute = x_2d if x_2d.dtype == compute_dtype else x_2d.to(dtype=compute_dtype)
        hidden = F.linear(x_compute, down_weight)
        if hidden.dtype != up_weight.dtype:
            hidden_for_up = hidden.to(dtype=up_weight.dtype)
            grad_for_up = grad_compute.to(dtype=up_weight.dtype)
        else:
            hidden_for_up = hidden
            grad_for_up = grad_compute

        grad_x = grad_down = grad_up = None
        if ctx.needs_input_grad[2]:
            grad_up = grad_for_up.t().matmul(hidden_for_up)
            if grad_up.dtype != up_weight.dtype:
                grad_up = grad_up.to(dtype=up_weight.dtype)

        grad_hidden = grad_compute.matmul(up_weight.to(dtype=compute_dtype))
        if ctx.needs_input_grad[1]:
            grad_down = grad_hidden.t().matmul(x_compute)
            if grad_down.dtype != down_weight.dtype:
                grad_down = grad_down.to(dtype=down_weight.dtype)
        if ctx.needs_input_grad[0]:
            grad_x_2d = grad_hidden.matmul(down_weight.to(dtype=compute_dtype))
            grad_x = grad_x_2d.reshape(input_shape)
            if grad_x.dtype != x.dtype:
                grad_x = grad_x.to(dtype=x.dtype)

        return grad_x, grad_down, grad_up, None


def lora_branch_recompute(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    return _LoRABranchRecomputeFn.apply(x, down_weight, up_weight, float(scaling))


class LoRALayer(nn.Module):
    """单个 LoRA 层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        activation_recompute: bool = False,
        rs_lora_enabled: bool = False,
        adapter_init_export_mode: str = "raw",
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.rs_lora_enabled = bool(rs_lora_enabled)
        self.scaling_strategy = "alpha_over_sqrt_rank" if self.rs_lora_enabled else "alpha_over_rank"
        self.scaling = alpha / math.sqrt(rank) if self.rs_lora_enabled else alpha / rank
        self.activation_recompute = bool(activation_recompute)
        
        # 低秩分解: W' = W + BA * scaling
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)  # 初始输出为 0，不影响原模型

    def _can_use_fast_path(self, x: torch.Tensor) -> bool:
        return x.ndim >= 1 and isinstance(self.dropout, nn.Identity)

    def _forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_recompute and torch.is_grad_enabled():
            return lora_branch_recompute(x, self.lora_down.weight, self.lora_up.weight, self.scaling)
        hidden = F.linear(x, self.lora_down.weight)
        out = F.linear(hidden, self.lora_up.weight)
        if self.scaling != 1.0:
            out = out * self.scaling
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算 LoRA 增量"""
        if self._can_use_fast_path(x):
            return self._forward_fast(x)
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
    
    def get_weight_matrix(self) -> torch.Tensor:
        """获取合并后的权重矩阵 (用于分析)"""
        return (self.lora_up.weight @ self.lora_down.weight) * self.scaling


# Internal DoRALayer removed in favor of backend.core.lulynx.dora_layer.DoRALinear


from ..memory_vortex_v2 import vortex_manager_v2 as vortex_manager

class LoRALinear(nn.Module):
    """包装原始 Linear 层并添加 LoRA"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        use_dora: bool = False,
        dora_mode: str = "full",
        adapter_init_strategy: str = "default",
        pissa_niter: int = 1,
        svd_algo: str = "rsvd",
        pissa_oversample: int = 8,
        pissa_apply_conv2d: bool = False,
        loftq_bits: int = 4,
        loftq_quant_type: str = "rowwise",
        vortex_enabled: bool = False,
        activation_recompute: bool = False,
        rs_lora_enabled: bool = False,
    ):
        super().__init__()
        self.original = original_layer
        self.use_dora = use_dora
        self.dora_mode = dora_mode
        self.adapter_init_strategy = _normalize_adapter_init_strategy(adapter_init_strategy)
        self.pissa_niter = pissa_niter
        self.svd_algo = _normalize_svd_algo(svd_algo)
        self.pissa_oversample = max(0, int(pissa_oversample or 0))
        self.pissa_apply_conv2d = bool(pissa_apply_conv2d)
        self.loftq_bits = min(max(int(loftq_bits or 4), 2), 8)
        self.loftq_quant_type = _normalize_loftq_quant_type(loftq_quant_type)
        self.vortex_enabled = vortex_enabled
        self.activation_recompute = bool(activation_recompute)
        self.rs_lora_enabled = bool(rs_lora_enabled)
        self.applied_adapter_init_strategy = "default"
        
        if use_dora:
            # Map original_layer to base_layer argument for DoRALinear
            self.lora = DoRALinear(
                base_layer=original_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                mode=dora_mode,
                rs_lora_enabled=self.rs_lora_enabled,
            )
        else:
            self.lora = LoRALayer(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                activation_recompute=self.activation_recompute,
                rs_lora_enabled=self.rs_lora_enabled,
            )

        # 标记 adapter 叶子模块，让 BlockSwap 跳过它们
        self._mark_lora_leaves()
        
        requested_init = _normalize_adapter_init_strategy(getattr(original_layer, "_adapter_init_strategy", ""))
        if requested_init == "default" and getattr(original_layer, "_pissa_init", False):
            requested_init = "pissa"
        if requested_init == "default":
            requested_init = self.adapter_init_strategy

        # Adapter initialization is applied after leaf marking and before the
        # base layer is frozen so decomposition-style init can rewrite W.
        if requested_init == "pissa":
            self._apply_pissa_init(original_layer, rank)
        elif requested_init == "olora":
            self._apply_olora_init(original_layer, rank)
        elif requested_init == "loftq":
            self._apply_loftq_init(original_layer, rank)
        
        # 冻结原始权重
        for param in self.original.parameters():
            param.requires_grad = False

        # Vortex 注册
        if self.vortex_enabled:
            vortex_manager.register_layer(self.original)

    def _mark_lora_leaves(self):
        """标记 adapter 叶子模块，让 BlockSwap 等搬运逻辑跳过它们"""
        lora = self.lora
        if hasattr(lora, "lora_down"):
            lora.lora_down._lora_leaf = True
        if hasattr(lora, "lora_up"):
            lora.lora_up._lora_leaf = True
        # DoRA parameters (lora_A, lora_B are nn.Parameter, not modules,
        # so they won't be caught by .modules() iteration — no marking needed).
        # But mark the DoRA module itself to prevent BlockSwap from
        # traversing into it and desyncing base_weight reference.
        if hasattr(lora, "lora_A") and hasattr(lora, "base_layer"):
            lora._lora_leaf = True

    def _effective_rank(self, weight: torch.Tensor, rank: int) -> int:
        return max(1, min(int(rank), int(weight.shape[0]), int(weight.shape[1])))

    def _quantize_dequantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        levels = float((1 << (self.loftq_bits - 1)) - 1)
        if levels < 1:
            return weight
        if self.loftq_quant_type == "tensorwise":
            scale = weight.abs().amax().clamp_min(1e-8) / levels
            return torch.round(weight / scale).clamp(-levels, levels) * scale

        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / levels
        return torch.round(weight / scale).clamp(-levels, levels) * scale

    def _write_low_rank_init(self, original_layer: nn.Linear, lora_up: torch.Tensor, lora_down: torch.Tensor) -> None:
        scaling = float(self.lora.scaling)
        up_weight = self.lora.lora_up.weight.data
        down_weight = self.lora.lora_down.weight.data
        up_weight.zero_()
        down_weight.zero_()
        rank = min(lora_up.shape[1], lora_down.shape[0], up_weight.shape[1], down_weight.shape[0])
        up_weight[:, :rank].copy_((lora_up[:, :rank] / scaling).to(device=up_weight.device, dtype=up_weight.dtype))
        down_weight[:rank, :].copy_(lora_down[:rank, :].to(device=down_weight.device, dtype=down_weight.dtype))
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
        delta = lora_up[:, :rank] @ lora_down[:rank, :]
        original_layer.weight.data.copy_((original_layer.weight.data.float() - delta).to(original_layer.weight.dtype))

    def _apply_pissa_init(self, original_layer, rank):
        """执行 PiSSA SVD 初始化"""
        if self.use_dora:
            logger.warning("PiSSA initialization is skipped for DoRA layers to avoid incompatible weight layouts.")
            return

        with torch.no_grad():
            W = original_layer.weight.data.float()
            effective_rank = self._effective_rank(W, rank)
            
            if self.svd_algo == "full":
                # 完整 SVD: W = U S V^T
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                # 截断
                U = U[:, :effective_rank]
                S = S[:effective_rank]
                V = Vh[:effective_rank, :]
                logger.debug(f"PiSSA full SVD applied")
            else:
                # 默认 rSVD (speed)
                # W ~= U S V^T
                q = min(min(W.shape), effective_rank + self.pissa_oversample)
                U, S, V_lowrank = torch.svd_lowrank(W, q=q, niter=max(int(self.pissa_niter or 0), 0))
                U = U[:, :effective_rank]
                S = S[:effective_rank]
                V = V_lowrank[:, :effective_rank].T
                logger.debug(f"PiSSA rSVD applied")

            # LoRA B = U * sqrt(S), A = sqrt(S) * V^T
            # 注意: W = W_resid + B @ A
            # 所以 W_resid = W - B @ A
            S_sqrt = torch.sqrt(S)
            lora_up = U * S_sqrt
            lora_down = S_sqrt.unsqueeze(1) * V
            
            # 更新原始权重为残差 W_resid = W - B @ A * scaling
            # 这样 original(x) + lora(x) 初始时正好等于原模型的输出
            self._write_low_rank_init(original_layer, lora_up, lora_down)
            self.applied_adapter_init_strategy = "pissa"
            logger.debug(f"PiSSA init applied to layer")

    def _apply_olora_init(self, original_layer, rank):
        """Apply OLoRA-style QR initialization without changing checkpoint format."""
        if self.use_dora:
            logger.warning("OLoRA initialization is skipped for DoRA layers to avoid incompatible weight layouts.")
            return

        with torch.no_grad():
            W = original_layer.weight.data.float()
            effective_rank = self._effective_rank(W, rank)
            # QR on W.T gives W ~= R.T @ Q.T, matching LoRA up/down layout.
            Q, R = torch.linalg.qr(W.T, mode="reduced")
            q_rank = min(effective_rank, Q.shape[1], R.shape[0])
            lora_down = Q[:, :q_rank].T.contiguous()
            lora_up = R[:q_rank, :].T.contiguous()
            self._write_low_rank_init(original_layer, lora_up, lora_down)
            self.applied_adapter_init_strategy = "olora"
            logger.debug("OLoRA QR init applied to layer")

    def _apply_loftq_init(self, original_layer, rank):
        """Apply LoftQ-style fake-quant residual initialization for native LoRA."""
        if self.use_dora:
            logger.warning("LoftQ initialization is skipped for DoRA layers to avoid incompatible weight layouts.")
            return

        with torch.no_grad():
            W = original_layer.weight.data.float()
            quantized = self._quantize_dequantize_weight(W)
            residual = W - quantized
            effective_rank = self._effective_rank(residual, rank)
            U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
            U = U[:, :effective_rank]
            S = S[:effective_rank]
            V = Vh[:effective_rank, :]
            S_sqrt = torch.sqrt(S.clamp_min(0))
            lora_up = U * S_sqrt
            lora_down = S_sqrt.unsqueeze(1) * V
            self._write_low_rank_init(original_layer, lora_up, lora_down)
            original_layer.weight.data.copy_(quantized.to(device=original_layer.weight.device, dtype=original_layer.weight.dtype))
            self.applied_adapter_init_strategy = "loftq"
            logger.debug("LoftQ fake-quant residual init applied to layer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preview_scale = float(getattr(self, "_preview_lora_scale", 1.0))
        if self.use_dora:
            if preview_scale <= 0:
                return self.original(x)
            if getattr(self.original, "_vortex_managed", False):
                raise RuntimeError(
                    "DoRA + Vortex combination is not supported. "
                    "Please disable either Vortex memory optimization or DoRA training mode."
                )
            
            if x.device != self.lora.lora_A.device or x.dtype != self.lora.lora_A.dtype:
                self.lora.to(device=x.device, dtype=x.dtype)
            # Unified DoRA uses integrated base_weight, just forward(x)
            return self.lora(x)

        # 统一处理原始层输出 (Vortex aware)
        if getattr(self.original, "_vortex_managed", False):
            original_out = vortex_manager.apply_linear(self.original, x)
        else:
            original_out = self._base_forward(x)

        # 标准 LoRA: 原始输出 + LoRA 增量
        if x.device != self.lora.lora_down.weight.device or x.dtype != self.lora.lora_down.weight.dtype:
            self.lora.to(device=x.device, dtype=x.dtype)
        if preview_scale <= 0:
            return original_out
        return original_out + self.lora(x) * preview_scale

    def _base_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen base GEMM, FP8-aware.

        Default (non-fp8 weight): plain ``original(x)`` — bit-identical to legacy.
        When the base weight is fp8 (``fp8_base`` storage), a normal GEMM would
        error, so we route through ``fp8_base_linear_forward`` which runs on Ada
        FP8 tensor cores if ``_fp8_base_compute`` is set, otherwise dequantizes to
        bf16 — either way making the previously storage-only fp8 path forward-safe.
        """
        original = self.original
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        weight = getattr(original, "weight", None)
        if fp8_dtype is not None and getattr(weight, "dtype", None) == fp8_dtype:
            from .fp8_quantize import fp8_base_linear_forward

            if getattr(original, "_fp8_base_compute", False):
                return fp8_base_linear_forward(original, x)
            bias = getattr(original, "bias", None)
            b = bias.to(x.dtype) if bias is not None else None
            return F.linear(x, weight.to(x.dtype), b)
        return original(x)

    
    @property
    def weight(self):
        """兼容性：返回原始权重"""
        return self.original.weight
    
    @property
    def bias(self):
        """兼容性：返回原始偏置"""
        return self.original.bias


class LoRAInjector:
    """LoRA 注入器"""

    # Backward-compatible aliases – still referenced by external callers
    # (e.g. LyCORIS path in trainer.py).  Values come from the registry
    # so they stay in sync with model_family.py.
    SDXL_UNET_TARGETS: List[str] = list(get_model_family("sdxl").unet_target_modules)
    SDXL_TE_TARGETS: List[str] = list(get_model_family("sdxl").text_encoder_target_modules)

    def __init__(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        adapter_init_strategy: str = "default",
        pissa_enabled: bool = False,
        pissa_niter: int = 1,
        svd_algo: str = "rsvd",
        pissa_oversample: int = 8,
        pissa_apply_conv2d: bool = False,
        loftq_bits: int = 4,
        loftq_quant_type: str = "rowwise",
        adapter_init_export_mode: str = "auto",
        dora_enabled: bool = False,
        dora_mode: str = "full",
        vortex_enabled: bool = False,
        model_arch: Optional[str] = None,
        tlora_enabled: bool = False,
        tlora_min_rank: int = 1,
        tlora_rank_schedule: str = "constant",
        tlora_orthogonal_init: bool = False,
        tlora_total_steps: int = 1000,
        vera_enabled: bool = False,
        vera_d_initial: float = 0.1,
        vera_prng_key: int = 0,
        lora_fa_enabled: bool = False,
        hydralora_enabled: bool = False,
        hydralora_num_experts: int = 4,
        hydralora_routing: str = "top_k",
        hydralora_top_k: int = 2,
        hydralora_sparse_top_k: bool = False,
        fera_enabled: bool = False,
        fera_gate_init: float = 0.0,
        flexrank_enabled: bool = False,
        flexrank_rank_range_min: int = 1,
        activation_recompute: bool = False,
        rs_lora_enabled: bool = False,
        adapter_target_policy: str = "all",
        adapter_target_selected: Optional[Iterable[str]] = None,
        adapter_target_rank_map: Optional[Dict[str, int]] = None,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.adapter_init_strategy = _normalize_adapter_init_strategy(adapter_init_strategy)
        if self.adapter_init_strategy == "default" and pissa_enabled:
            self.adapter_init_strategy = "pissa"
        self.pissa_enabled = self.adapter_init_strategy == "pissa"
        self.pissa_niter = pissa_niter
        self.svd_algo = _normalize_svd_algo(svd_algo)
        self.pissa_oversample = max(0, int(pissa_oversample or 0))
        self.pissa_apply_conv2d = bool(pissa_apply_conv2d)
        self.loftq_bits = min(max(int(loftq_bits or 4), 2), 8)
        self.loftq_quant_type = _normalize_loftq_quant_type(loftq_quant_type)
        self.dora_enabled = dora_enabled
        self.dora_mode = dora_mode
        self.vortex_enabled = vortex_enabled
        self.tlora_enabled = tlora_enabled
        self.tlora_min_rank = tlora_min_rank
        self.tlora_rank_schedule = tlora_rank_schedule
        self.tlora_orthogonal_init = tlora_orthogonal_init
        self.tlora_total_steps = tlora_total_steps
        self.vera_enabled = vera_enabled
        self.vera_d_initial = vera_d_initial
        self.vera_prng_key = vera_prng_key
        self.lora_fa_enabled = lora_fa_enabled
        self.hydralora_enabled = hydralora_enabled
        self.hydralora_num_experts = hydralora_num_experts
        self.hydralora_routing = hydralora_routing
        self.hydralora_top_k = hydralora_top_k
        self.hydralora_sparse_top_k = hydralora_sparse_top_k
        self.fera_enabled = fera_enabled
        self.fera_gate_init = fera_gate_init
        self.flexrank_enabled = flexrank_enabled
        self.flexrank_rank_range_min = flexrank_rank_range_min
        self.activation_recompute = bool(activation_recompute)
        self.rs_lora_enabled = bool(rs_lora_enabled)
        self.adapter_init_export_mode = _normalize_adapter_init_export_mode(adapter_init_export_mode)
        self.injected_layers: Dict[str, LoRALinear] = {}

        # Adapter target policy (FG-LoRA style selective injection). Default "all"
        # (or no selection/rank supplied) keeps every matched module at self.rank,
        # so the policy is a strict no-op and injection stays bitwise-identical.
        self.adapter_target_policy = str(adapter_target_policy or "all").strip().lower() or "all"
        self._adapter_target_selected = (
            {str(name) for name in adapter_target_selected} if adapter_target_selected else None
        )
        self._adapter_target_rank_map = (
            {str(key): max(int(value), 1) for key, value in dict(adapter_target_rank_map).items()}
            if adapter_target_rank_map
            else None
        )
        self._adapter_target_policy_active = bool(
            self.adapter_target_policy != "all"
            and (self._adapter_target_selected or self._adapter_target_rank_map)
        )

        # VeRA shared buffers (created on first injection if vera_enabled)
        self._vera_buffers: Optional["VeRASharedBuffers"] = None

        # Resolve family once; defaults to SDXL for unknown/None arch
        self._family: ModelFamily = get_model_family(model_arch)

    def inject_unet(self, unet: nn.Module) -> Dict[str, LoRALinear]:
        """注入 UNet"""
        targets = self.target_modules or self._family.unet_target_modules
        return self._inject_model(unet, targets, prefix="unet", apply_policy=True)

    def inject_text_encoder(self, text_encoder: nn.Module, name: str = "te") -> Dict[str, LoRALinear]:
        """注入 Text Encoder"""
        targets = self._family.text_encoder_target_modules
        return self._inject_model(text_encoder, targets, prefix=name)

    def inject(
        self,
        model: nn.Module,
        target_modules: List[str],
        prefix: str = "",
        apply_policy: bool = False,
        exclude_name_substrings: Optional[List[str]] = None,
    ) -> Dict[str, LoRALinear]:
        """Backward-compatible explicit target injection entrypoint.

        ``apply_policy`` defaults to ``False`` so legacy callers keep injecting
        every supplied target unchanged. Native-family unet injection opts in
        (``apply_policy=True``) so an active adapter target policy also governs
        the explicit-target path, matching ``inject_unet``. When the policy is
        inactive (default ``"all"``) this stays a strict no-op.

        ``exclude_name_substrings`` skips any module whose qualified name contains
        one of the substrings, regardless of target match. Needed because target
        suffixes like ``self_attn.q_proj`` substring-match a frozen sibling subtree
        (e.g. ``anima_llm_adapter.blocks.N.self_attn.q_proj``) that must not receive
        an adapter. Defaults to None (no exclusion) so all legacy paths are unchanged.
        """
        return self._inject_model(
            model, target_modules, prefix=prefix, apply_policy=apply_policy,
            exclude_name_substrings=exclude_name_substrings,
        )

    def _inject_model(
        self,
        model: nn.Module,
        target_names: List[str],
        prefix: str = "",
        apply_policy: bool = False,
        exclude_name_substrings: Optional[List[str]] = None,
    ) -> Dict[str, LoRALinear]:
        """向模型注入 LoRA 层"""
        injected = {}
        policy_active = bool(self._adapter_target_policy_active and apply_policy)
        excludes = tuple(s for s in (exclude_name_substrings or ()) if s)

        for name, module in model.named_modules():
            # Skip frozen sibling subtrees whose names substring-collide with a
            # target suffix (e.g. anima_llm_adapter when llm_adapter is not trained).
            if excludes and any(s in name for s in excludes):
                continue
            # Native Conv2d LoRA is intentionally limited to PiSSA conv init.
            is_linear_target = isinstance(module, nn.Linear)
            is_pissa_conv_target = (
                isinstance(module, nn.Conv2d)
                and self.pissa_enabled
                and self.pissa_apply_conv2d
            )
            if not is_linear_target and not is_pissa_conv_target:
                continue
                
            # 检查名称是否匹配，并记下命中的 target 名（leaf 段或 dotted 全名），
            # 供 policy 过滤与 per-type rank 解析使用。dotted target
            # ("cross_attn.v_proj"、"final_layer.adaln_modulation.2") 必须按全名匹配，
            # 否则会被 leaf 段 ("v_proj"、"2") 误判而漏掉。命中条件与旧逻辑等价。
            layer_name = name.split(".")[-1]
            if layer_name in target_names:
                matched_target = layer_name
            else:
                matched_target = next(
                    (target for target in target_names if target in name), None
                )
            if matched_target is None:
                continue

            # Adapter target policy: optionally restrict which target types receive
            # an adapter and assign a per-type rank, keyed by the matched target
            # name. Inactive policy (policy_active=False) keeps layer_rank ==
            # self.rank for every matched module, so construction stays
            # bitwise-identical to the legacy path.
            if (
                policy_active
                and self._adapter_target_selected is not None
                and matched_target not in self._adapter_target_selected
            ):
                continue
            # Rank lookup prefers the full module path (true per-layer rank, e.g.
            # FG-LoRA orthogonal redistribution), then the matched target type
            # (legacy per-type rank), then the uniform rank. Backward compatible:
            # a per-type map carries no full-path keys, so it resolves exactly as
            # before; only a full-path map activates per-layer rank.
            layer_rank = (
                max(int(self._adapter_target_rank_map.get(
                    name, self._adapter_target_rank_map.get(matched_target, self.rank))), 1)
                if (policy_active and self._adapter_target_rank_map)
                else self.rank
            )

            # Mark decomposition-style initializers for the wrapper constructor.
            if self.adapter_init_strategy != "default":
                module._adapter_init_strategy = self.adapter_init_strategy
            if self.pissa_enabled:
                module._pissa_init = True
                
            # 创建 LoRA 包装
            if is_pissa_conv_target:
                if module.groups != 1:
                    logger.warning("Skipping grouped Conv2d PiSSA LoRA target %s; groups=%s is not supported.", name, module.groups)
                    continue
                from .lora_conv2d import LoRAConv2d

                lora_linear = LoRAConv2d(
                    original_layer=module,
                    rank=layer_rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    adapter_init_strategy=self.adapter_init_strategy,
                    pissa_niter=self.pissa_niter,
                    svd_algo=self.svd_algo,
                    pissa_oversample=self.pissa_oversample,
                    rs_lora_enabled=self.rs_lora_enabled,
                )
                lora_linear._block_weight_lr_scale = 1.0
                lora_linear._block_weight_frozen = False
            elif self.fera_enabled:
                from .fera import FeRALinear
                lora_linear = FeRALinear(
                    original_layer=module,
                    rank=layer_rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    gate_init=self.fera_gate_init,
                )
            elif self.hydralora_enabled:
                from .hydralora import HydraLoRAConfig, HydraLoRALinear
                lora_linear = HydraLoRALinear(
                    original=module,
                    config=HydraLoRAConfig(
                        num_experts=max(int(self.hydralora_num_experts or 1), 1),
                        rank=layer_rank,
                        alpha=self.alpha,
                        routing=str(self.hydralora_routing or "top_k"),
                        top_k=max(int(self.hydralora_top_k or 1), 1),
                        sparse_top_k=bool(self.hydralora_sparse_top_k),
                        dropout=self.dropout,
                    ),
                )
            elif self.vera_enabled:
                from .vera_layer import VeRALinear, VeRASharedBuffers
                if self._vera_buffers is None:
                    self._vera_buffers = VeRASharedBuffers(
                        rank=self.rank,
                        prng_key=self.vera_prng_key,
                    )
                lora_linear = VeRALinear(
                    original_layer=module,
                    shared_buffers=self._vera_buffers,
                    d_initial=self.vera_d_initial,
                    alpha=self.alpha,
                )
            elif self.lora_fa_enabled:
                from .lora_fa_layer import LoRAFALinear
                lora_linear = LoRAFALinear(
                    original_layer=module,
                    rank=layer_rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
            elif self.tlora_enabled:
                from .tlora import TLoRALinear
                lora_linear = TLoRALinear(
                    original_layer=module,
                    max_rank=layer_rank,
                    min_rank=self.tlora_min_rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    schedule=self.tlora_rank_schedule,
                    total_steps=self.tlora_total_steps,
                    orthogonal_init=self.tlora_orthogonal_init,
                )
                # TLoRALinear has _block_weight attributes
                lora_linear._block_weight_lr_scale = 1.0
                lora_linear._block_weight_frozen = False
            elif self.flexrank_enabled:
                from .flexrank_lora import FlexRankLoRALinear
                lora_linear = FlexRankLoRALinear(
                    original_layer=module,
                    max_rank=layer_rank,
                    min_rank=self.flexrank_rank_range_min,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                lora_linear._block_weight_lr_scale = 1.0
                lora_linear._block_weight_frozen = False
            else:
                lora_linear = LoRALinear(
                    original_layer=module,
                    rank=layer_rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    use_dora=self.dora_enabled,
                    dora_mode=self.dora_mode,
                    adapter_init_strategy=self.adapter_init_strategy,
                    pissa_niter=self.pissa_niter,
                    svd_algo=self.svd_algo,
                    pissa_oversample=self.pissa_oversample,
                    pissa_apply_conv2d=self.pissa_apply_conv2d,
                    loftq_bits=self.loftq_bits,
                    loftq_quant_type=self.loftq_quant_type,
                    vortex_enabled=self.vortex_enabled,
                    activation_recompute=self.activation_recompute,
                    rs_lora_enabled=self.rs_lora_enabled,
                )
                lora_linear._block_weight_lr_scale = 1.0
                lora_linear._block_weight_frozen = False
            
            # 替换原始层
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                
            setattr(parent, child_name, lora_linear)
            
            full_name = f"{prefix}.{name}" if prefix else name
            injected[full_name] = lora_linear
            self.injected_layers[full_name] = lora_linear
            
        logger.info(f"Injected {len(injected)} LoRA layers into {prefix}")
        return injected

    def get_layer_names(self) -> List[str]:
        return list(self.injected_layers.keys())

    def set_global_step(self, step: int) -> None:
        """Push the current training step to T-LoRA layers for rank scheduling."""
        from .tlora import TLoRALinear
        for layer in self.injected_layers.values():
            if isinstance(layer, TLoRALinear):
                layer.set_global_step(step)

    def _iter_layer_trainable_params(self, layer: LoRALinear) -> Iterable[nn.Parameter]:
        from .vera_layer import VeRALinear
        from .lora_fa_layer import LoRAFALinear
        from .tlora import TLoRALinear
        from .hydralora import HydraLoRALinear
        from .fera import FeRALinear
        if isinstance(layer, FeRALinear):
            return layer.get_trainable_params()
        if isinstance(layer, HydraLoRALinear):
            return layer.get_trainable_params()
        if isinstance(layer, VeRALinear):
            return [layer.vera_lambda_d, layer.vera_lambda_b]
        if isinstance(layer, LoRAFALinear):
            return layer.get_trainable_params()
        if isinstance(layer, TLoRALinear):
            return list(layer.lora_down.parameters()) + list(layer.lora_up.parameters())
        if (
            hasattr(layer, "lora_down")
            and hasattr(layer.lora_down, "weight")
            and hasattr(layer, "lora_up")
            and hasattr(layer.lora_up, "weight")
        ):
            return list(layer.lora_down.parameters()) + list(layer.lora_up.parameters())
        return layer.lora.parameters()

    def _iter_layer_residency_params(self, layer: LoRALinear) -> Iterable[nn.Parameter]:
        from .vera_layer import VeRALinear
        from .lora_fa_layer import LoRAFALinear
        from .tlora import TLoRALinear
        from .hydralora import HydraLoRALinear
        from .fera import FeRALinear

        if isinstance(layer, FeRALinear):
            return layer.get_trainable_params()
        if isinstance(layer, HydraLoRALinear):
            return layer.get_trainable_params()
        if isinstance(layer, VeRALinear):
            params: List[nn.Parameter] = [layer.vera_lambda_d, layer.vera_lambda_b]
            buffers_ref = getattr(layer, "_buffers_ref", None)
            if buffers_ref is not None:
                shared_a = getattr(buffers_ref, "_A", None)
                shared_b = getattr(buffers_ref, "_B", None)
                if isinstance(shared_a, nn.Parameter):
                    params.append(shared_a)
                if isinstance(shared_b, nn.Parameter):
                    params.append(shared_b)
            return params
        if isinstance(layer, LoRAFALinear):
            return [layer.lora_down.weight, layer.lora_up.weight]
        if isinstance(layer, TLoRALinear):
            return list(layer.lora_down.parameters()) + list(layer.lora_up.parameters())
        if (
            hasattr(layer, "lora_down")
            and hasattr(layer.lora_down, "weight")
            and hasattr(layer, "lora_up")
            and hasattr(layer.lora_up, "weight")
        ):
            return list(layer.lora_down.parameters()) + list(layer.lora_up.parameters())

        adapter = getattr(layer, "lora", None)
        if adapter is not None and hasattr(adapter, "lora_A") and hasattr(adapter, "lora_B") and hasattr(adapter, "m"):
            params = [adapter.lora_A, adapter.lora_B, adapter.m]
            return [param for param in params if isinstance(param, nn.Parameter)]
        if adapter is not None and hasattr(adapter, "lora_down") and hasattr(adapter, "lora_up"):
            return list(adapter.lora_down.parameters()) + list(adapter.lora_up.parameters())
        return [param for param in layer.parameters() if isinstance(param, nn.Parameter)]

    def freeze_layer(self, layer_name: str):
        layer = self.injected_layers.get(layer_name)
        if not layer:
            return

        layer._block_weight_lr_scale = 0.0
        layer._block_weight_frozen = True
        for param in self._iter_layer_trainable_params(layer):
            param.requires_grad = False

    def set_layer_lr_scale(self, layer_name: str, scale: float):
        layer = self.injected_layers.get(layer_name)
        if not layer:
            return

        scale = max(float(scale), 0.0)
        layer._block_weight_lr_scale = scale
        layer._block_weight_frozen = scale <= 0.0
        for param in self._iter_layer_trainable_params(layer):
            param.requires_grad = scale > 0.0

    def get_param_groups(self, base_lr: float, weight_decay: float = 0.0) -> List[Dict]:
        grouped: Dict[float, Dict[str, object]] = {}

        for layer_name, layer in self.injected_layers.items():
            params = [param for param in self._iter_layer_trainable_params(layer) if param.requires_grad]
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
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取所有可训练参数"""
        params = []
        for lora_linear in self.injected_layers.values():
            params.extend(param for param in self._iter_layer_trainable_params(lora_linear) if param.requires_grad)
        return params

    def get_residency_params(self) -> List[nn.Parameter]:
        """Return adapter-owned parameters that should follow CPU residency.

        This includes trainable weights plus frozen adapter-side parameters
        that are still required during forward passes, such as LoRA-FA's
        frozen down projection and VeRA shared matrices.
        """
        params: List[nn.Parameter] = []
        seen: Set[int] = set()
        for lora_linear in self.injected_layers.values():
            for param in self._iter_layer_residency_params(lora_linear):
                if not isinstance(param, nn.Parameter):
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        return params

    def _standard_lora_weights(
        self,
        layer: nn.Module,
        export_mode: str = "raw",
        down: Optional[torch.Tensor] = None,
        up: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        adapter = getattr(layer, "lora", None)
        if adapter is None or not hasattr(adapter, "lora_down") or not hasattr(adapter, "lora_up"):
            return None
        export_mode = _normalize_adapter_init_export_mode(export_mode)
        down = adapter.lora_down.weight.data if down is None else down
        up = adapter.lora_up.weight.data if up is None else up
        init_down = getattr(layer, "adapter_init_lora_down", None)
        init_up = getattr(layer, "adapter_init_lora_up", None)
        strategy = _normalize_adapter_init_strategy(getattr(layer, "applied_adapter_init_strategy", "default"))
        if strategy == "default" or export_mode == "raw" or init_down is None or init_up is None:
            return down, up

        init_down = init_down.detach().to(device=down.device, dtype=down.dtype)
        init_up = init_up.detach().to(device=up.device, dtype=up.dtype)
        if export_mode == "lora_compatible":
            export_down = torch.cat([down.detach(), init_down], dim=0)
            export_up = torch.cat([up.detach(), -init_up], dim=1)
            return export_down, export_up

        export_down = down.detach().float() - init_down.float()
        export_up = up.detach().float()
        return export_down.to(device=down.device, dtype=down.dtype), export_up.to(device=up.device, dtype=up.dtype)

    def export_adapter_init_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        adapter_init_export_mode: str = "raw",
    ) -> Dict[str, torch.Tensor]:
        export_mode = _normalize_adapter_init_export_mode(adapter_init_export_mode)
        if export_mode == "auto":
            export_mode = "raw"
        if export_mode == "raw":
            return state_dict

        if export_mode == "lora_compatible":
            logger.warning(
                "adapter_init_export_mode=lora_compatible doubles rank for initialized layers to preserve the "
                "trained delta on a standard base model. Use raw for exact resume checkpoints."
            )

        exported = dict(state_dict)
        for name, layer in self.injected_layers.items():
            if _normalize_adapter_init_strategy(getattr(layer, "applied_adapter_init_strategy", "default")) == "default":
                continue
            base_name = name.replace(".", "_")
            down_key = f"{base_name}.lora_down.weight"
            up_key = f"{base_name}.lora_up.weight"
            if down_key not in exported or up_key not in exported:
                continue
            weights = self._standard_lora_weights(layer, export_mode, exported[down_key], exported[up_key])
            if weights is None:
                continue
            exported[down_key], exported[up_key] = weights
        return exported
    
    def get_lora_state_dict(self, adapter_init_export_mode: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """获取 LoRA 权重状态字典"""
        from .vera_layer import VeRALinear
        from .lora_fa_layer import LoRAFALinear
        from .hydralora import HydraLoRALinear
        from .fera import FeRALinear
        export_mode = _normalize_adapter_init_export_mode(adapter_init_export_mode or "raw")
        state_dict = {}
        for name, lora_linear in self.injected_layers.items():
            base_name = name.replace(".", "_")
            adapter = lora_linear.lora if hasattr(lora_linear, "lora") and not isinstance(lora_linear, (VeRALinear, LoRAFALinear)) else None

            if isinstance(lora_linear, FeRALinear):
                state_dict[f"{base_name}.fera_lora_down.weight"] = lora_linear.lora_down.weight.data
                state_dict[f"{base_name}.fera_lora_up.weight"] = lora_linear.lora_up.weight.data
                state_dict[f"{base_name}.fera_residual_gate"] = lora_linear.residual_gate.data
            elif isinstance(lora_linear, HydraLoRALinear):
                state_dict[f"{base_name}.hydralora_lora_down"] = lora_linear.lora_down.data
                state_dict[f"{base_name}.hydralora_lora_up"] = lora_linear.lora_up.data
                state_dict[f"{base_name}.hydralora_gate.weight"] = lora_linear.gate.weight.data
                state_dict[f"{base_name}.hydralora_num_experts"] = torch.tensor(lora_linear.config.num_experts)
                state_dict[f"{base_name}.hydralora_top_k"] = torch.tensor(lora_linear.config.top_k)
            elif isinstance(lora_linear, VeRALinear):
                # Export as standard LoRA format
                weights = lora_linear.export_standard_lora_weights()
                state_dict[f"{base_name}.lora_down.weight"] = weights["lora_down.weight"]
                state_dict[f"{base_name}.lora_up.weight"] = weights["lora_up.weight"]
            elif isinstance(lora_linear, LoRAFALinear):
                # Same format as standard LoRA
                state_dict[f"{base_name}.lora_down.weight"] = lora_linear.lora_down.weight.data
                state_dict[f"{base_name}.lora_up.weight"] = lora_linear.lora_up.weight.data
            elif isinstance(lora_linear, TLoRALinear):
                state_dict[f"{base_name}.lora_down.weight"] = lora_linear.lora_down.weight.data
                state_dict[f"{base_name}.lora_up.weight"] = lora_linear.lora_up.weight.data
                state_dict[f"{base_name}.tlora_current_rank"] = torch.tensor(lora_linear.current_rank)
                state_dict[f"{base_name}.tlora_min_rank"] = torch.tensor(lora_linear.min_rank)
                state_dict[f"{base_name}.tlora_max_rank"] = torch.tensor(lora_linear.max_rank)
                state_dict[f"{base_name}.tlora_alpha"] = torch.tensor(lora_linear.alpha)
            elif (
                hasattr(lora_linear, "lora_down")
                and hasattr(lora_linear.lora_down, "weight")
                and hasattr(lora_linear, "lora_up")
                and hasattr(lora_linear.lora_up, "weight")
            ):
                state_dict[f"{base_name}.lora_down.weight"] = lora_linear.lora_down.weight.data
                state_dict[f"{base_name}.lora_up.weight"] = lora_linear.lora_up.weight.data
                if hasattr(lora_linear, "min_rank"):
                    state_dict[f"{base_name}.flexrank_min_rank"] = torch.tensor(int(getattr(lora_linear, "min_rank")))
                if hasattr(lora_linear, "max_rank"):
                    state_dict[f"{base_name}.flexrank_max_rank"] = torch.tensor(int(getattr(lora_linear, "max_rank")))
                if hasattr(lora_linear, "alpha"):
                    state_dict[f"{base_name}.flexrank_alpha"] = torch.tensor(float(getattr(lora_linear, "alpha")))
            elif adapter is not None and hasattr(adapter, "lora_down") and hasattr(adapter, "lora_up"):
                standard_weights = self._standard_lora_weights(lora_linear, export_mode)
                if standard_weights is None:
                    continue
                down_weight, up_weight = standard_weights
                state_dict[f"{base_name}.lora_down.weight"] = down_weight
                state_dict[f"{base_name}.lora_up.weight"] = up_weight
            elif adapter is not None:
                state_dict[f"{base_name}.lora_A"] = adapter.lora_A.data
                state_dict[f"{base_name}.lora_B"] = adapter.lora_B.data
                state_dict[f"{base_name}.m"] = adapter.m.data

        # Include VeRA shared buffers for resume (not for export)
        if self._vera_buffers is not None and self._vera_buffers._A is not None:
            state_dict["vera_shared_A"] = self._vera_buffers.shared_A.data
            state_dict["vera_shared_B"] = self._vera_buffers.shared_B.data

        return state_dict
    
    def save_lora(self, path: str, metadata: Optional[Dict] = None):
        """保存 LoRA 权重为 safetensors"""
        try:
            from safetensors.torch import save_file
        except ImportError:
            # Fallback to torch.save
            torch.save(self.get_lora_state_dict(), path)
            logger.warning("safetensors not available, saved as .pt")
            return
            
        state_dict = self.get_lora_state_dict()
        
        # 添加元数据
        if metadata is None:
            metadata = {}
        metadata.setdefault("ss_network_dim", str(self.rank))
        metadata.setdefault("ss_network_alpha", str(self.alpha))
        metadata.setdefault("ss_output_name", "lulynx_lora")
        if self.rs_lora_enabled:
            metadata.update({
                "ss_rs_lora": "true",
                "ss_scaling_strategy": "alpha_over_sqrt_rank",
            })
        
        save_file(state_dict, path, metadata=metadata)
        logger.info(f"Saved LoRA to {path}")

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        """Load LoRA weights from an in-memory state dict."""
        from .vera_layer import VeRALinear
        from .lora_fa_layer import LoRAFALinear
        from .tlora import TLoRALinear
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        loaded_keys = 0
        total_expected = 0

        # Restore VeRA shared buffers if present
        if self.vera_enabled and self._vera_buffers is not None:
            if "vera_shared_A" in state_dict:
                self._vera_buffers._A = nn.Parameter(
                    state_dict["vera_shared_A"].to(device=self._vera_buffers.device),
                    requires_grad=False,
                )
                loaded_keys += 1
            if "vera_shared_B" in state_dict:
                self._vera_buffers._B = nn.Parameter(
                    state_dict["vera_shared_B"].to(device=self._vera_buffers.device),
                    requires_grad=False,
                )
                loaded_keys += 1

        for name, lora_linear in self.injected_layers.items():
            base_name = name.replace(".", "_")

            if isinstance(lora_linear, VeRALinear):
                total_expected += 2
                d_key = f"{base_name}.vera_lambda_d"
                b_key = f"{base_name}.vera_lambda_b"
                if d_key in state_dict:
                    lora_linear.vera_lambda_d.data.copy_(
                        state_dict[d_key].to(device=lora_linear.vera_lambda_d.device, dtype=lora_linear.vera_lambda_d.dtype)
                    )
                    loaded_keys += 1
                if b_key in state_dict:
                    lora_linear.vera_lambda_b.data.copy_(
                        state_dict[b_key].to(device=lora_linear.vera_lambda_b.device, dtype=lora_linear.vera_lambda_b.dtype)
                    )
                    loaded_keys += 1
                continue

            if isinstance(lora_linear, LoRAFALinear):
                total_expected += 2
                down_key = f"{base_name}.lora_down.weight"
                up_key = f"{base_name}.lora_up.weight"
                # lora_down is frozen — still load for resume consistency
                if down_key in state_dict:
                    lora_linear.lora_down.weight.data.copy_(
                        state_dict[down_key].to(device=lora_linear.lora_down.weight.device, dtype=lora_linear.lora_down.weight.dtype)
                    )
                    loaded_keys += 1
                if up_key in state_dict:
                    lora_linear.lora_up.weight.data.copy_(
                        state_dict[up_key].to(device=lora_linear.lora_up.weight.device, dtype=lora_linear.lora_up.weight.dtype)
                    )
                    loaded_keys += 1
                continue

            if isinstance(lora_linear, TLoRALinear):
                total_expected += 2
                down_key = f"{base_name}.lora_down.weight"
                up_key = f"{base_name}.lora_up.weight"
                if down_key in state_dict:
                    src = state_dict[down_key]
                    dst_shape = lora_linear.lora_down.weight.shape
                    if src.shape == dst_shape:
                        lora_linear.lora_down.weight.data.copy_(
                            src.to(device=lora_linear.lora_down.weight.device, dtype=lora_linear.lora_down.weight.dtype)
                        )
                    else:
                        # Rank mismatch: copy what fits
                        min_r = min(src.shape[0], dst_shape[0])
                        lora_linear.lora_down.weight.data[:min_r].copy_(
                            src[:min_r].to(device=lora_linear.lora_down.weight.device, dtype=lora_linear.lora_down.weight.dtype)
                        )
                    loaded_keys += 1
                if up_key in state_dict:
                    src = state_dict[up_key]
                    dst_shape = lora_linear.lora_up.weight.shape
                    if src.shape == dst_shape:
                        lora_linear.lora_up.weight.data.copy_(
                            src.to(device=lora_linear.lora_up.weight.device, dtype=lora_linear.lora_up.weight.dtype)
                        )
                    else:
                        min_r = min(src.shape[1], dst_shape[1])
                        lora_linear.lora_up.weight.data[:, :min_r].copy_(
                            src[:, :min_r].to(device=lora_linear.lora_up.weight.device, dtype=lora_linear.lora_up.weight.dtype)
                        )
                    loaded_keys += 1
                # Restore rank metadata if available
                rank_key = f"{base_name}.tlora_current_rank"
                if rank_key in state_dict:
                    restored_rank = int(state_dict[rank_key].item())
                    lora_linear.set_global_step(
                        int(restored_rank * lora_linear.total_steps / max(lora_linear.max_rank - lora_linear.min_rank, 1))
                    )
                continue

            if (
                hasattr(lora_linear, "lora_down")
                and hasattr(lora_linear.lora_down, "weight")
                and hasattr(lora_linear, "lora_up")
                and hasattr(lora_linear.lora_up, "weight")
            ):
                total_expected += 2
                down_key = f"{base_name}.lora_down.weight"
                up_key = f"{base_name}.lora_up.weight"
                if down_key in state_dict:
                    src = state_dict[down_key]
                    dst_shape = lora_linear.lora_down.weight.shape
                    if src.shape == dst_shape:
                        lora_linear.lora_down.weight.data.copy_(
                            src.to(device=lora_linear.lora_down.weight.device, dtype=lora_linear.lora_down.weight.dtype)
                        )
                    else:
                        min_r = min(src.shape[0], dst_shape[0])
                        lora_linear.lora_down.weight.data[:min_r].copy_(
                            src[:min_r].to(device=lora_linear.lora_down.weight.device, dtype=lora_linear.lora_down.weight.dtype)
                        )
                    loaded_keys += 1
                if up_key in state_dict:
                    src = state_dict[up_key]
                    dst_shape = lora_linear.lora_up.weight.shape
                    if src.shape == dst_shape:
                        lora_linear.lora_up.weight.data.copy_(
                            src.to(device=lora_linear.lora_up.weight.device, dtype=lora_linear.lora_up.weight.dtype)
                        )
                    else:
                        min_r = min(src.shape[1], dst_shape[1])
                        lora_linear.lora_up.weight.data[:, :min_r].copy_(
                            src[:, :min_r].to(device=lora_linear.lora_up.weight.device, dtype=lora_linear.lora_up.weight.dtype)
                        )
                    loaded_keys += 1
                continue

            adapter = getattr(lora_linear, "lora", None)
            if hasattr(adapter, "lora_down") and hasattr(adapter, "lora_up"):
                total_expected += 2
                down_key = f"{base_name}.lora_down.weight"
                up_key = f"{base_name}.lora_up.weight"

                if down_key in state_dict:
                    adapter.lora_down.weight.data.copy_(
                        state_dict[down_key].to(
                            device=adapter.lora_down.weight.device,
                            dtype=adapter.lora_down.weight.dtype,
                        )
                    )
                    loaded_keys += 1

                if up_key in state_dict:
                    adapter.lora_up.weight.data.copy_(
                        state_dict[up_key].to(
                            device=adapter.lora_up.weight.device,
                            dtype=adapter.lora_up.weight.dtype,
                        )
                    )
                    loaded_keys += 1
            else:
                total_expected += 3
                key_map = {
                    f"{base_name}.lora_A": adapter.lora_A,
                    f"{base_name}.lora_B": adapter.lora_B,
                    f"{base_name}.m": adapter.m,
                }
                for key, param in key_map.items():
                    if key in state_dict:
                        param.data.copy_(state_dict[key].to(device=param.device, dtype=param.dtype))
                        loaded_keys += 1

        return loaded_keys, total_expected

    def load_lora(self, path: str, *, disable_mmap: bool = False):
        """加载 LoRA 权重 (Resume)"""
        try:
            from .safetensors_loader import load_safetensors
            state_dict = load_safetensors(path, disable_mmap=disable_mmap)
        except (ImportError, Exception):
            # Fallback
            try:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return

        loaded_keys, total_expected = self.load_lora_state_dict(state_dict)
        logger.info(f"Loaded LoRA weights from {path}: {loaded_keys}/{total_expected} tensors matched.")

    
    def get_layer_stats(self) -> Dict[str, Dict]:
        """获取各层统计信息 (用于审计)"""
        stats = {}
        for name, lora_linear in self.injected_layers.items():
            with torch.no_grad():
                weight = lora_linear.lora.get_weight_matrix()
                stats[name] = {
                    "rank": self.rank,
                    "norm": weight.norm().item(),
                    "mean": weight.mean().item(),
                    "std": weight.std().item(),
                    "max": weight.abs().max().item(),
                }
        return stats


def infer_rank_from_weights(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer the LoRA rank from a checkpoint's state dict.

    Inspects lora_down.weight tensors to determine the rank. Works with
    standard LoRA, LoRA-FA, T-LoRA, and kohya-format checkpoints.

    Returns None if no lora_down weights are found.
    """
    ranks = set()
    for key, tensor in state_dict.items():
        # Standard format: prefix.lora_down.weight
        if key.endswith(".lora_down.weight") and tensor.ndim >= 2:
            ranks.add(tensor.shape[0])
        # kohya format: prefix.lora_down.weight (same shape convention)
        # Also handle keys without the leading module name
        if "lora_down" in key and "weight" in key and tensor.ndim >= 2:
            # lora_down shape is (rank, in_features)
            ranks.add(tensor.shape[0])
        # T-LoRA metadata
        if key.endswith(".tlora_max_rank") and tensor.numel() == 1:
            ranks.add(int(tensor.item()))
    if len(ranks) == 1:
        return ranks.pop()
    if len(ranks) > 1:
        logger.warning(
            "Multiple ranks detected in checkpoint (%s). "
            "Using the most common rank.",
            sorted(ranks),
        )
        # Return most common rank
        rank_counts: Dict[int, int] = {}
        for key, tensor in state_dict.items():
            if key.endswith(".lora_down.weight") and tensor.ndim >= 2:
                r = tensor.shape[0]
                rank_counts[r] = rank_counts.get(r, 0) + 1
        if rank_counts:
            return max(rank_counts, key=rank_counts.get)  # type: ignore[arg-type]
    return None
