"""EasyControl v2 adapter primitives.

This module intentionally does not patch real DiT blocks yet.  It models the
trainable EasyControl v2 pieces that need stable save/load and shape contracts:
condition-token projection, per-block LoRA deltas, and a learned condition-key
attention bias.  The real Anima/Newbie block integration can build on this
without changing the existing EasyControl v1 latent-residual path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EasyControlV2AdapterConfig:
    hidden_size: int = 2048
    cond_channels: int = 16
    cond_lora_rank: int = 16
    cond_lora_alpha: float = 16.0
    num_blocks: int = 28
    b_cond_init: float = -10.0
    cond_scale: float = 1.0
    apply_ffn_lora: bool = True
    init_zero_out: bool = True

    def normalized(self) -> "EasyControlV2AdapterConfig":
        hidden_size = max(int(self.hidden_size or 0), 1)
        cond_channels = max(int(self.cond_channels or 0), 1)
        rank = max(int(self.cond_lora_rank or 0), 1)
        num_blocks = max(int(self.num_blocks or 0), 1)
        alpha = float(self.cond_lora_alpha if self.cond_lora_alpha is not None else rank)
        if not math.isfinite(alpha) or alpha <= 0:
            alpha = float(rank)
        cond_scale = float(self.cond_scale if self.cond_scale is not None else 1.0)
        if not math.isfinite(cond_scale):
            cond_scale = 1.0
        return EasyControlV2AdapterConfig(
            hidden_size=hidden_size,
            cond_channels=cond_channels,
            cond_lora_rank=rank,
            cond_lora_alpha=alpha,
            num_blocks=num_blocks,
            b_cond_init=float(self.b_cond_init if self.b_cond_init is not None else -10.0),
            cond_scale=max(cond_scale, 0.0),
            apply_ffn_lora=bool(self.apply_ffn_lora),
            init_zero_out=bool(self.init_zero_out),
        )

    def to_metadata(self) -> Dict[str, str]:
        cfg = self.normalized()
        return {f"ss_easycontrol_v2_{key}": str(value) for key, value in asdict(cfg).items()}


class EasyControlV2LoRAProjection(nn.Module):
    """Standalone LoRA delta projection used by the v2 condition stream."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        init_zero_out: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(int(rank), 1)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.down = nn.Linear(self.in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        if init_zero_out:
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source_dtype = x.dtype
        y = F.linear(x.float(), self.down.weight.float())
        y = F.linear(y, self.up.weight.float())
        return (y * self.scale).to(source_dtype)


class EasyControlV2BlockAdapter(nn.Module):
    """Per-block condition-stream trainables for the v2 two-stream route."""

    def __init__(self, config: EasyControlV2AdapterConfig) -> None:
        super().__init__()
        cfg = config.normalized()
        self.config = cfg
        d = cfg.hidden_size
        r = cfg.cond_lora_rank
        a = cfg.cond_lora_alpha
        self.qkv = EasyControlV2LoRAProjection(
            d, 3 * d, rank=r, alpha=a, init_zero_out=cfg.init_zero_out
        )
        self.out = EasyControlV2LoRAProjection(
            d, d, rank=r, alpha=a, init_zero_out=cfg.init_zero_out
        )
        if cfg.apply_ffn_lora:
            self.ffn1 = EasyControlV2LoRAProjection(
                d, 4 * d, rank=r, alpha=a, init_zero_out=cfg.init_zero_out
            )
            self.ffn2 = EasyControlV2LoRAProjection(
                4 * d, d, rank=r, alpha=a, init_zero_out=cfg.init_zero_out
            )
        else:
            self.ffn1 = None
            self.ffn2 = None
        self.b_cond = nn.Parameter(torch.tensor(cfg.b_cond_init, dtype=torch.float32))

    def forward(
        self,
        cond_tokens: torch.Tensor,
        *,
        block_index: int = 0,
    ) -> Dict[str, torch.Tensor]:
        del block_index
        if cond_tokens.dim() != 3:
            raise ValueError(f"cond_tokens must be [B, S, D], got {tuple(cond_tokens.shape)}")
        if cond_tokens.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"cond_tokens hidden dim {cond_tokens.shape[-1]} != {self.config.hidden_size}"
            )
        qkv_delta = self.qkv(cond_tokens) * self.config.cond_scale
        out_delta = self.out(cond_tokens) * self.config.cond_scale
        result = {
            "qkv_delta": qkv_delta,
            "out_delta": out_delta,
            "b_cond": self.b_cond.to(dtype=cond_tokens.dtype),
        }
        if self.ffn1 is not None and self.ffn2 is not None:
            ffn_hidden = self.ffn1(cond_tokens) * self.config.cond_scale
            result["ffn1_delta"] = ffn_hidden
            result["ffn2_delta"] = self.ffn2(torch.nn.functional.silu(ffn_hidden)) * self.config.cond_scale
        return result


class EasyControlV2Adapter(nn.Module):
    """Trainable v2 adapter skeleton for condition-token DiT integration."""

    network_spec = "easycontrol_v2"

    def __init__(self, config: EasyControlV2AdapterConfig) -> None:
        super().__init__()
        cfg = config.normalized()
        self.config = cfg
        self.cond_proj = nn.Linear(cfg.cond_channels, cfg.hidden_size, bias=False)
        if cfg.init_zero_out:
            nn.init.zeros_(self.cond_proj.weight)
        else:
            nn.init.kaiming_uniform_(self.cond_proj.weight, a=math.sqrt(5))
        self.blocks = nn.ModuleList([EasyControlV2BlockAdapter(cfg) for _ in range(cfg.num_blocks)])
        self._cond_tokens: Optional[torch.Tensor] = None

    def encode_cond_latents(self, cond_latents: torch.Tensor) -> torch.Tensor:
        if cond_latents.dim() == 4:
            tokens = cond_latents.flatten(2).transpose(1, 2).contiguous()
        elif cond_latents.dim() == 3:
            tokens = cond_latents
        else:
            raise ValueError(f"cond_latents must be [B,C,H,W] or [B,S,C], got {tuple(cond_latents.shape)}")
        if tokens.shape[-1] != self.config.cond_channels:
            raise ValueError(
                f"cond channel dim {tokens.shape[-1]} != {self.config.cond_channels}"
            )
        return self.cond_proj(tokens)

    def set_cond(self, cond_latents: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        self._cond_tokens = None if cond_latents is None else self.encode_cond_latents(cond_latents)
        return self._cond_tokens

    def clear_cond(self) -> None:
        self._cond_tokens = None

    def has_cond(self) -> bool:
        return self._cond_tokens is not None

    @property
    def current_cond_tokens(self) -> Optional[torch.Tensor]:
        return self._cond_tokens

    def block_deltas(
        self,
        block_index: int,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        tokens = cond_tokens if cond_tokens is not None else self._cond_tokens
        if tokens is None:
            return None
        idx = int(block_index)
        if idx < 0 or idx >= len(self.blocks):
            raise IndexError(f"EasyControl v2 block index out of range: {block_index}")
        return self.blocks[idx](tokens, block_index=idx)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def adapter_metadata(self) -> Dict[str, str]:
        metadata = self.config.to_metadata()
        metadata["ss_network_spec"] = self.network_spec
        metadata["ss_easycontrol_v2_mergeable"] = "false"
        metadata["ss_easycontrol_v2_training_step_wired"] = "false"
        return metadata


def build_easycontrol_v2_adapter_config(payload: Mapping[str, Any] | Any = None, **overrides: Any) -> EasyControlV2AdapterConfig:
    values: Dict[str, Any] = {}
    if isinstance(payload, Mapping):
        values.update(payload)
    elif payload is not None:
        for field in EasyControlV2AdapterConfig.__dataclass_fields__:
            if hasattr(payload, field):
                values[field] = getattr(payload, field)
    values.update(overrides)
    return EasyControlV2AdapterConfig(
        hidden_size=int(values.get("hidden_size", values.get("easycontrol_v2_hidden_size", 2048)) or 2048),
        cond_channels=int(values.get("cond_channels", values.get("easycontrol_v2_cond_channels", 16)) or 16),
        cond_lora_rank=int(values.get("cond_lora_rank", values.get("easycontrol_v2_cond_lora_rank", 16)) or 16),
        cond_lora_alpha=float(values.get("cond_lora_alpha", values.get("easycontrol_v2_cond_lora_alpha", 16.0)) or 16.0),
        num_blocks=int(values.get("num_blocks", values.get("easycontrol_v2_num_blocks", 28)) or 28),
        b_cond_init=float(values.get("b_cond_init", values.get("easycontrol_v2_b_cond_init", -10.0)) or -10.0),
        cond_scale=float(values.get("cond_scale", values.get("easycontrol_v2_scale", 1.0)) or 0.0),
        apply_ffn_lora=bool(values.get("apply_ffn_lora", values.get("easycontrol_v2_apply_ffn_lora", True))),
        init_zero_out=bool(values.get("init_zero_out", values.get("easycontrol_v2_init_zero_out", True))),
    ).normalized()


def build_easycontrol_v2_adapter_config_from_metadata(
    metadata: Mapping[str, Any],
    **overrides: Any,
) -> EasyControlV2AdapterConfig:
    values: Dict[str, Any] = {}
    for field in EasyControlV2AdapterConfig.__dataclass_fields__:
        key = f"ss_easycontrol_v2_{field}"
        if key in metadata:
            values[field] = metadata[key]
    values.update(overrides)
    return EasyControlV2AdapterConfig(
        hidden_size=_metadata_int(values.get("hidden_size"), 2048),
        cond_channels=_metadata_int(values.get("cond_channels"), 16),
        cond_lora_rank=_metadata_int(values.get("cond_lora_rank"), 16),
        cond_lora_alpha=_metadata_float(values.get("cond_lora_alpha"), 16.0),
        num_blocks=_metadata_int(values.get("num_blocks"), 28),
        b_cond_init=_metadata_float(values.get("b_cond_init"), -10.0),
        cond_scale=_metadata_float(values.get("cond_scale"), 1.0),
        apply_ffn_lora=_metadata_bool(values.get("apply_ffn_lora"), True),
        init_zero_out=_metadata_bool(values.get("init_zero_out"), True),
    ).normalized()


def _metadata_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _metadata_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _metadata_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def easycontrol_v2_trainable_param_count(module: EasyControlV2Adapter) -> int:
    return sum(param.numel() for param in module.get_trainable_params())


def easycontrol_v2_state_keys(module: EasyControlV2Adapter) -> Tuple[str, ...]:
    return tuple(sorted(module.state_dict().keys()))


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"attention tokens must be [B, S, D], got {tuple(x.shape)}")
    b, s, d = x.shape
    heads = int(num_heads)
    if heads <= 0 or d % heads != 0:
        raise ValueError(f"hidden dim {d} must be divisible by num_heads={num_heads}")
    return x.reshape(b, s, heads, d // heads).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"attention heads must be [B, heads, S, D], got {tuple(x.shape)}")
    b, h, s, d = x.shape
    return x.transpose(1, 2).reshape(b, s, h * d)


def easycontrol_v2_two_stream_attention_step(
    target_tokens: torch.Tensor,
    cond_tokens: Optional[torch.Tensor],
    block_deltas: Optional[Mapping[str, torch.Tensor]],
    *,
    qkv_proj: nn.Module,
    out_proj: nn.Module,
    num_heads: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the v2 attention math for a synthetic DiT-style block.

    This is a patchable primitive, not the production Anima/Newbie integration.
    It assumes a fused `qkv_proj: D -> 3D` and `out_proj: D -> D` pair so tests
    can prove baseline equivalence and condition-path gradients before touching
    real block classes.
    """

    target_q, target_k, target_v = qkv_proj(target_tokens).chunk(3, dim=-1)
    target_out = F.scaled_dot_product_attention(
        _split_heads(target_q, num_heads),
        _split_heads(target_k, num_heads),
        _split_heads(target_v, num_heads),
        dropout_p=0.0,
    )
    target_out = out_proj(_merge_heads(target_out))
    if cond_tokens is None or block_deltas is None:
        return target_out, cond_tokens

    cond_q, cond_k, cond_v = qkv_proj(cond_tokens).chunk(3, dim=-1)
    qkv_delta = block_deltas.get("qkv_delta")
    if qkv_delta is not None:
        dq, dk, dv = qkv_delta.chunk(3, dim=-1)
        cond_q = cond_q + dq
        cond_k = cond_k + dk
        cond_v = cond_v + dv

    target_out = easycontrol_v2_extended_attention(
        _split_heads(target_q, num_heads),
        _split_heads(target_k, num_heads),
        _split_heads(target_v, num_heads),
        _split_heads(cond_k, num_heads),
        _split_heads(cond_v, num_heads),
        block_deltas.get("b_cond", -10.0),
    )
    target_out = out_proj(_merge_heads(target_out))

    cond_out = F.scaled_dot_product_attention(
        _split_heads(cond_q, num_heads),
        _split_heads(cond_k, num_heads),
        _split_heads(cond_v, num_heads),
        dropout_p=0.0,
    )
    cond_out = out_proj(_merge_heads(cond_out))
    out_delta = block_deltas.get("out_delta")
    if out_delta is not None:
        cond_out = cond_out + out_delta
    return target_out, cond_tokens + cond_out


class EasyControlV2SyntheticPatchHandle:
    """Restores a synthetic block patched by EasyControl v2 smoke helpers."""

    def __init__(self, block: nn.Module, original_forward: Any) -> None:
        self.block = block
        self.original_forward = original_forward
        self.active = True

    def remove(self) -> None:
        if self.active:
            self.block.forward = self.original_forward
            self.active = False


def easycontrol_v2_projection_attention_step(
    target_tokens: torch.Tensor,
    cond_tokens: Optional[torch.Tensor],
    block_deltas: Optional[Mapping[str, torch.Tensor]],
    *,
    attention: nn.Module,
    num_heads: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """v2 attention step for Anima-style separate q/k/v/output projections."""

    q_proj = getattr(attention, "q_proj")
    k_proj = getattr(attention, "k_proj")
    v_proj = getattr(attention, "v_proj")
    out_proj = getattr(attention, "output_proj")
    target_q = q_proj(target_tokens)
    target_k = k_proj(target_tokens)
    target_v = v_proj(target_tokens)
    if cond_tokens is None or block_deltas is None:
        target_out = F.scaled_dot_product_attention(
            _split_heads(target_q, num_heads),
            _split_heads(target_k, num_heads),
            _split_heads(target_v, num_heads),
            dropout_p=0.0,
        )
        return out_proj(_merge_heads(target_out)), cond_tokens

    cond_q = q_proj(cond_tokens)
    cond_k = k_proj(cond_tokens)
    cond_v = v_proj(cond_tokens)
    qkv_delta = block_deltas.get("qkv_delta")
    if qkv_delta is not None:
        dq, dk, dv = qkv_delta.chunk(3, dim=-1)
        cond_q = cond_q + dq
        cond_k = cond_k + dk
        cond_v = cond_v + dv

    target_out = easycontrol_v2_extended_attention(
        _split_heads(target_q, num_heads),
        _split_heads(target_k, num_heads),
        _split_heads(target_v, num_heads),
        _split_heads(cond_k, num_heads),
        _split_heads(cond_v, num_heads),
        block_deltas.get("b_cond", -10.0),
    )
    target_out = out_proj(_merge_heads(target_out))

    cond_out = F.scaled_dot_product_attention(
        _split_heads(cond_q, num_heads),
        _split_heads(cond_k, num_heads),
        _split_heads(cond_v, num_heads),
        dropout_p=0.0,
    )
    cond_out = out_proj(_merge_heads(cond_out))
    out_delta = block_deltas.get("out_delta")
    if out_delta is not None:
        cond_out = cond_out + out_delta
    return target_out, cond_tokens + cond_out


class EasyControlV2AnimaTinyPatchHandle(EasyControlV2SyntheticPatchHandle):
    """Restores an Anima tiny block patched by the v2 readiness helper."""


def install_easycontrol_v2_anima_tiny_block_patch(
    block: nn.Module,
    adapter: EasyControlV2Adapter,
    *,
    block_index: int,
    num_heads: int,
) -> EasyControlV2AnimaTinyPatchHandle:
    """Patch Lulynx's `AnimaNativeDiTTinyTrainable` block for readiness smoke.

    This helper intentionally targets only the local tiny trainable block shape:
    `forward(x, context, cond)`, `self_attn.{q,k,v,output}_proj`,
    `cross_attn`, `mlp`, and `adaln_modulation`.  It is not a production
    real-weight Anima patcher.
    """

    for attr in ("self_attn", "cross_attn", "mlp", "adaln_modulation"):
        if not hasattr(block, attr):
            raise ValueError(f"not an Anima tiny block: missing {attr}")
    for attr in ("q_proj", "k_proj", "v_proj", "output_proj"):
        if not hasattr(block.self_attn, attr):
            raise ValueError(f"not an Anima tiny self_attn: missing {attr}")

    original_forward = block.forward

    def patched_forward(x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_tokens = adapter.current_cond_tokens
        if cond_tokens is None:
            return original_forward(x, context, cond)
        deltas = adapter.block_deltas(block_index, cond_tokens)
        self_out, next_cond = easycontrol_v2_projection_attention_step(
            x,
            cond_tokens,
            deltas,
            attention=block.self_attn,
            num_heads=num_heads,
        )
        adapter._cond_tokens = next_cond
        gate = block.adaln_modulation(cond).chunk(6, dim=-1)[-1].unsqueeze(1)
        x_next = x + self_out
        x_next = x_next + block.cross_attn(x_next, context)
        x_next = x_next + block.mlp(x_next) * torch.tanh(gate)
        return x_next

    block.forward = patched_forward
    return EasyControlV2AnimaTinyPatchHandle(block, original_forward)


def easycontrol_v2_anima_tiny_patch_readiness(block: nn.Module) -> Dict[str, Any]:
    """Report whether a module matches the local Anima tiny patch contract."""

    required = {
        "self_attn": hasattr(block, "self_attn"),
        "cross_attn": hasattr(block, "cross_attn"),
        "mlp": hasattr(block, "mlp"),
        "adaln_modulation": hasattr(block, "adaln_modulation"),
    }
    self_attn = getattr(block, "self_attn", None)
    for attr in ("q_proj", "k_proj", "v_proj", "output_proj"):
        required[f"self_attn.{attr}"] = hasattr(self_attn, attr)
    missing = tuple(name for name, present in required.items() if not present)
    return {
        "family": "anima",
        "scope": "tiny_readiness_only",
        "ready": not missing,
        "missing": missing,
        "real_executable_subset": False,
        "training_step_consumption": False,
    }


def easycontrol_v2_anima_executable_subset_readiness(module_or_block: nn.Module) -> Dict[str, Any]:
    """Read-only readiness report for real-weight Anima executable subset blocks.

    This deliberately does not install a patch.  It only distinguishes the
    real/executable-subset structure from the tiny readiness helper and reports
    the current block-patching gate as closed.
    """

    is_subset = bool(getattr(module_or_block, "is_anima_executable_subset", False))
    blocks = []
    if is_subset:
        blocks = list(getattr(getattr(module_or_block, "net", None), "blocks", []) or [])
    else:
        blocks = [module_or_block]

    missing: List[str] = []
    block_count = len(blocks)
    for index, block in enumerate(blocks):
        prefix = f"blocks.{index}"
        for attr in (
            "self_attn",
            "cross_attn",
            "mlp",
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
        ):
            if not hasattr(block, attr):
                missing.append(f"{prefix}.{attr}")
        self_attn = getattr(block, "self_attn", None)
        for attr in ("q_proj", "k_proj", "v_proj", "output_proj", "q_norm", "k_norm"):
            if not hasattr(self_attn, attr):
                missing.append(f"{prefix}.self_attn.{attr}")
    structural_ready = block_count > 0 and not missing
    return {
        "family": "anima",
        "scope": "executable_subset_readiness_only",
        "ready": structural_ready,
        "block_count": block_count,
        "missing": tuple(missing),
        "real_executable_subset": is_subset,
        "patch_supported": False,
        "training_step_consumption": False,
        "blocked_reason": (
            "real executable subset EasyControl v2 patch is not implemented"
            if structural_ready
            else "module does not match Anima executable subset block contract"
        ),
    }


def require_easycontrol_v2_anima_executable_subset_patch_ready(module_or_block: nn.Module) -> Dict[str, Any]:
    """Guard against accidentally treating the real executable subset as wired."""

    report = easycontrol_v2_anima_executable_subset_readiness(module_or_block)
    if not bool(report.get("patch_supported", False)):
        raise RuntimeError(
            "EasyControl v2 real Anima executable-subset patch is blocked: "
            f"{report.get('blocked_reason')}"
        )
    return report


def build_easycontrol_v2_adapter_profile(
    adapter: EasyControlV2Adapter,
    *,
    target: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Small profile object for preflight/trainer reporting."""

    report: Dict[str, Any] = {
        "network_spec": adapter.network_spec,
        "trainable_param_count": easycontrol_v2_trainable_param_count(adapter),
        "mergeable": False,
        "training_step_consumption": False,
        "metadata": adapter.adapter_metadata(),
    }
    if target is not None:
        if bool(getattr(target, "is_anima_executable_subset", False)):
            report["target_readiness"] = easycontrol_v2_anima_executable_subset_readiness(target)
        elif bool(getattr(target, "is_tiny_anima_trainable_smoke", False)):
            blocks = list(getattr(getattr(target, "net", None), "blocks", []) or [])
            report["target_readiness"] = (
                easycontrol_v2_anima_tiny_patch_readiness(blocks[0])
                if blocks
                else {
                    "family": "anima",
                    "scope": "tiny_readiness_only",
                    "ready": False,
                    "missing": ("net.blocks.0",),
                    "real_executable_subset": False,
                    "training_step_consumption": False,
                }
            )
        else:
            report["target_readiness"] = {
                "family": "unknown",
                "scope": "unsupported_target",
                "ready": False,
                "patch_supported": False,
                "training_step_consumption": False,
            }
    return report


def install_easycontrol_v2_synthetic_attention_patch(
    block: nn.Module,
    adapter: EasyControlV2Adapter,
    *,
    block_index: int,
    num_heads: int,
    qkv_attr: str = "qkv",
    out_attr: str = "out",
) -> EasyControlV2SyntheticPatchHandle:
    """Patch a tiny DiT-like block for smoke tests only.

    The target block must expose `qkv` and `out` linear-like modules and accept
    a single `[B, S, D]` tensor in `forward`.  Real model integration should use
    a family-specific patcher instead of this synthetic helper.
    """

    qkv_proj = getattr(block, qkv_attr)
    out_proj = getattr(block, out_attr)
    original_forward = block.forward

    def patched_forward(target_tokens: torch.Tensor) -> torch.Tensor:
        cond_tokens = adapter.current_cond_tokens
        deltas = adapter.block_deltas(block_index, cond_tokens) if cond_tokens is not None else None
        output, next_cond = easycontrol_v2_two_stream_attention_step(
            target_tokens,
            cond_tokens,
            deltas,
            qkv_proj=qkv_proj,
            out_proj=out_proj,
            num_heads=num_heads,
        )
        adapter._cond_tokens = next_cond
        return output

    block.forward = patched_forward
    return EasyControlV2SyntheticPatchHandle(block, original_forward)


def easycontrol_v2_extended_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cond_k: Optional[torch.Tensor],
    cond_v: Optional[torch.Tensor],
    b_cond: torch.Tensor | float,
    *,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Attention where target queries can see target keys plus condition keys.

    Shapes are `[B, heads, tokens, dim]`.  When `cond_k`/`cond_v` are absent,
    this is exactly PyTorch SDPA on the target tensors.  When present, a scalar
    `b_cond` bias is added only to the condition-key columns before softmax.
    """

    if cond_k is None or cond_v is None or cond_k.numel() == 0 or cond_v.numel() == 0:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or cond_k.dim() != 4 or cond_v.dim() != 4:
        raise ValueError("EasyControl v2 extended attention expects [B, heads, tokens, dim] tensors")
    if k.shape[:-2] != cond_k.shape[:-2] or v.shape[:-2] != cond_v.shape[:-2]:
        raise ValueError("target and condition attention tensors must share batch/head dimensions")
    if k.shape[-1] != cond_k.shape[-1] or v.shape[-1] != cond_v.shape[-1]:
        raise ValueError("target and condition attention tensors must share head dimension")
    merged_k = torch.cat([k, cond_k], dim=-2)
    merged_v = torch.cat([v, cond_v], dim=-2)
    target_tokens = k.shape[-2]
    cond_tokens = cond_k.shape[-2]
    attn_bias = q.new_zeros((q.shape[-2], target_tokens + cond_tokens))
    attn_bias[:, target_tokens:] = torch.as_tensor(b_cond, dtype=q.dtype, device=q.device)
    return F.scaled_dot_product_attention(
        q,
        merged_k,
        merged_v,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
    )
