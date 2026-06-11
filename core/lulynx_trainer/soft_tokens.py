"""Default-off Soft Tokens primitive for SoftREPA-style text conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SoftTokenConfig:
    num_tokens: int = 4
    hidden_size: int = 8
    layer_ids: tuple[int, ...] = (0,)
    timestep_boundaries: tuple[float, ...] = (0.5,)
    init_std: float = 0.02
    adapter_scope: str = "softrepa_style"
    token_mode: str = "softrepa"
    agsm_enabled: bool = False
    dispersive_enabled: bool = False
    metadata_version: int = 1

    def validate(self) -> None:
        if self.num_tokens < 1:
            raise ValueError("num_tokens must be >= 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if not self.layer_ids:
            raise ValueError("layer_ids must not be empty")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("layer_ids must be unique")
        if any(int(item) < 0 for item in self.layer_ids):
            raise ValueError("layer_ids must be non-negative")
        last = 0.0
        for value in self.timestep_boundaries:
            value = float(value)
            if not 0.0 < value < 1.0:
                raise ValueError("timestep_boundaries must be inside (0, 1)")
            if value <= last:
                raise ValueError("timestep_boundaries must be strictly increasing")
            last = value
        if self.adapter_scope != "softrepa_style":
            raise ValueError("adapter_scope must stay softrepa_style")
        if self.token_mode != "softrepa":
            raise ValueError("only current softrepa token_mode is supported")
        if self.agsm_enabled:
            raise ValueError("Soft Tokens AGSM is archived and not enabled")
        if self.dispersive_enabled:
            raise ValueError("Soft Tokens dispersive mode is archived and not enabled")


@dataclass(frozen=True)
class SoftTokenPrependResult:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor | None
    prepended_tokens: int
    layer_index: int | None
    bucket_index: int | torch.Tensor | None
    default_behavior_changed: bool = False


class SoftTokenBank(nn.Module):
    """Layer/timestep-aware learnable text tokens.

    This module is intentionally a primitive, not trainer wiring. It gives a
    future route a stable tensor surface: select tokens by DiT layer and
    timestep bucket, then prepend them to text conditioning.
    """

    def __init__(self, config: SoftTokenConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        cfg = _coerce_config(config)
        cfg.validate()
        self.config = cfg
        bucket_count = len(cfg.timestep_boundaries) + 1
        self.tokens = nn.Parameter(torch.empty(len(cfg.layer_ids), bucket_count, cfg.num_tokens, cfg.hidden_size))
        nn.init.normal_(self.tokens, mean=0.0, std=cfg.init_std)

    def tokens_for(
        self,
        layer_index: int,
        timestep: Any,
        *,
        total_steps: int | None = None,
    ) -> tuple[torch.Tensor, int | torch.Tensor]:
        layer_slot = _layer_slot(layer_index, self.config.layer_ids)
        bucket = select_soft_token_bucket(
            timestep,
            total_steps=total_steps,
            boundaries=self.config.timestep_boundaries,
        )
        layer_tokens = self.tokens[layer_slot]
        if isinstance(bucket, int):
            return layer_tokens[bucket], bucket
        idx = bucket.to(device=layer_tokens.device, dtype=torch.long).reshape(-1)
        return layer_tokens.index_select(0, idx), bucket

    def prepend(
        self,
        text_embeds: torch.Tensor,
        *,
        layer_index: int,
        timestep: Any,
        total_steps: int | None = None,
        attention_mask: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> SoftTokenPrependResult:
        if not enabled:
            return SoftTokenPrependResult(text_embeds, attention_mask, 0, None, None)
        soft_tokens, bucket = self.tokens_for(layer_index, timestep, total_steps=total_steps)
        return prepend_soft_tokens(
            text_embeds,
            soft_tokens,
            attention_mask=attention_mask,
            layer_index=layer_index,
            bucket_index=bucket,
        )

    def metadata(self) -> dict[str, str]:
        return build_soft_tokens_metadata(self.config)

    def merge_decision(self) -> dict[str, Any]:
        return build_soft_tokens_merge_decision(self.metadata())


def prepend_soft_tokens(
    text_embeds: torch.Tensor,
    soft_tokens: torch.Tensor | None,
    *,
    attention_mask: torch.Tensor | None = None,
    layer_index: int | None = None,
    bucket_index: int | torch.Tensor | None = None,
) -> SoftTokenPrependResult:
    if soft_tokens is None or soft_tokens.numel() == 0:
        return SoftTokenPrependResult(text_embeds, attention_mask, 0, layer_index, bucket_index)
    if text_embeds.dim() != 3:
        raise ValueError("text_embeds must be [batch, tokens, hidden]")
    batch, _, hidden = text_embeds.shape
    tokens = _batch_tokens(soft_tokens, batch, hidden, text_embeds.device, text_embeds.dtype)
    merged = torch.cat([tokens, text_embeds], dim=1)
    mask = None
    if attention_mask is not None:
        if attention_mask.shape[0] != batch:
            raise ValueError("attention_mask batch size must match text_embeds")
        prefix = torch.ones(batch, tokens.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
        mask = torch.cat([prefix, attention_mask], dim=1)
    return SoftTokenPrependResult(
        embeddings=merged,
        attention_mask=mask,
        prepended_tokens=int(tokens.shape[1]),
        layer_index=layer_index,
        bucket_index=bucket_index,
        default_behavior_changed=False,
    )


def select_soft_token_bucket(
    timestep: Any,
    *,
    total_steps: int | None = None,
    boundaries: Sequence[float] = (0.5,),
) -> int | torch.Tensor:
    boundary_values = tuple(float(item) for item in boundaries)
    if isinstance(timestep, torch.Tensor):
        fraction = _tensor_fraction(timestep, total_steps)
        boundary_tensor = torch.tensor(boundary_values, device=fraction.device, dtype=fraction.dtype)
        return torch.bucketize(fraction.contiguous(), boundary_tensor)
    fraction = _scalar_fraction(timestep, total_steps)
    for idx, boundary in enumerate(boundary_values):
        if fraction < boundary:
            return idx
    return len(boundary_values)


def build_soft_tokens_metadata(config: SoftTokenConfig | Mapping[str, Any] | None = None) -> dict[str, str]:
    cfg = _coerce_config(config)
    cfg.validate()
    return {
        "ss_adapter_type": "soft_tokens",
        "ss_soft_tokens_version": str(cfg.metadata_version),
        "ss_soft_tokens_scope": cfg.adapter_scope,
        "ss_soft_tokens_style": cfg.token_mode,
        "ss_soft_tokens_num_tokens": str(cfg.num_tokens),
        "ss_soft_tokens_hidden_size": str(cfg.hidden_size),
        "ss_soft_tokens_layers": ",".join(str(item) for item in cfg.layer_ids),
        "ss_soft_tokens_timestep_boundaries": ",".join(_fmt_float(item) for item in cfg.timestep_boundaries),
        "ss_soft_tokens_non_mergeable": "true",
        "ss_soft_tokens_requires_live_text_conditioning": "true",
        "ss_soft_tokens_agsm_enabled": "false",
        "ss_soft_tokens_dispersive_enabled": "false",
        "ss_training_path_enabled": "false",
        "ss_default_behavior_changed": "false",
    }


def build_soft_tokens_merge_decision(metadata: Mapping[str, Any], *, requested_merge: bool = True) -> dict[str, Any]:
    adapter_type = str(metadata.get("ss_adapter_type") or "").strip()
    non_mergeable = str(metadata.get("ss_soft_tokens_non_mergeable") or "").strip().lower() == "true"
    live_text = str(metadata.get("ss_soft_tokens_requires_live_text_conditioning") or "").strip().lower() == "true"
    blockers: list[str] = []
    if adapter_type != "soft_tokens":
        blockers.append("unexpected_adapter_type")
    if requested_merge and non_mergeable:
        blockers.append("soft_tokens_are_prompt_conditioning")
    if requested_merge and live_text:
        blockers.append("live_text_conditioning_injection_required")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "soft_tokens_merge_refusal_v0",
        "ok": ready,
        "merge_allowed": ready,
        "requested_merge": bool(requested_merge),
        "non_mergeable": bool(non_mergeable),
        "requires_live_text_conditioning": bool(live_text),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "merge may proceed"
            if ready
            else "keep Soft Tokens as live text-conditioning state; do not bake into base weights"
        ),
    }


def build_soft_tokens_scorecard(
    *,
    config: SoftTokenConfig | Mapping[str, Any] | None = None,
    disabled_parity_ok: bool = False,
    prepend_shape_ok: bool = False,
    layer_timestep_selection_ok: bool = False,
    gradient_flow_ok: bool = False,
    metadata_roundtrip_ok: bool = False,
    merge_refusal_ok: bool = False,
    archived_modes_blocked: bool = False,
) -> dict[str, Any]:
    cfg = _coerce_config(config)
    blockers: list[str] = []
    try:
        cfg.validate()
    except ValueError as exc:
        blockers.append(f"invalid_config:{exc}")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_missing")
    if not prepend_shape_ok:
        blockers.append("prepend_shape_missing")
    if not layer_timestep_selection_ok:
        blockers.append("layer_timestep_selection_missing")
    if not gradient_flow_ok:
        blockers.append("gradient_flow_missing")
    if not metadata_roundtrip_ok:
        blockers.append("metadata_roundtrip_missing")
    if not merge_refusal_ok:
        blockers.append("merge_refusal_missing")
    if not archived_modes_blocked:
        blockers.append("archived_agsm_dispersive_block_missing")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "soft_tokens_adapter_primitive_v0",
        "ok": ready,
        "primitive_ready": ready,
        "adapter_type": "soft_tokens",
        "adapter_scope": cfg.adapter_scope,
        "token_mode": cfg.token_mode,
        "num_tokens": cfg.num_tokens,
        "hidden_size": cfg.hidden_size,
        "layer_ids": list(cfg.layer_ids),
        "timestep_boundaries": list(cfg.timestep_boundaries),
        "non_mergeable": True,
        "requires_live_text_conditioning": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Soft Tokens trainer/checkpoint preflight before adapter registry wiring"
            if ready
            else "complete Soft Tokens parity, shape, routing, gradient, metadata, and archived-mode checks"
        ),
    }


def _coerce_config(config: SoftTokenConfig | Mapping[str, Any] | None) -> SoftTokenConfig:
    if isinstance(config, SoftTokenConfig):
        return config
    values = dict(config or {})
    layer_ids = _coerce_int_tuple(values.get("layer_ids", SoftTokenConfig.layer_ids))
    boundaries = _coerce_float_tuple(values.get("timestep_boundaries", SoftTokenConfig.timestep_boundaries))
    return SoftTokenConfig(
        num_tokens=int(values.get("num_tokens", SoftTokenConfig.num_tokens)),
        hidden_size=int(values.get("hidden_size", SoftTokenConfig.hidden_size)),
        layer_ids=layer_ids,
        timestep_boundaries=boundaries,
        init_std=float(values.get("init_std", SoftTokenConfig.init_std)),
        adapter_scope=str(values.get("adapter_scope", SoftTokenConfig.adapter_scope)),
        token_mode=str(values.get("token_mode", SoftTokenConfig.token_mode)),
        agsm_enabled=_boolish(values.get("agsm_enabled", SoftTokenConfig.agsm_enabled)),
        dispersive_enabled=_boolish(values.get("dispersive_enabled", SoftTokenConfig.dispersive_enabled)),
        metadata_version=int(values.get("metadata_version", SoftTokenConfig.metadata_version)),
    )


def _coerce_int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def _coerce_float_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return tuple(float(item) for item in value)


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _tensor_fraction(timestep: torch.Tensor, total_steps: int | None) -> torch.Tensor:
    value = timestep.to(dtype=torch.float32)
    if total_steps is not None:
        value = value / float(max(int(total_steps) - 1, 1))
    return value.clamp(0.0, 1.0)


def _scalar_fraction(timestep: Any, total_steps: int | None) -> float:
    value = float(timestep)
    if total_steps is not None:
        value = value / float(max(int(total_steps) - 1, 1))
    return min(max(value, 0.0), 1.0)


def _layer_slot(layer_index: int, layer_ids: Sequence[int]) -> int:
    try:
        return tuple(int(item) for item in layer_ids).index(int(layer_index))
    except ValueError as exc:
        raise ValueError(f"layer_index {layer_index} is not configured for Soft Tokens") from exc


def _batch_tokens(
    soft_tokens: torch.Tensor,
    batch: int,
    hidden: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tokens = soft_tokens.to(device=device, dtype=dtype)
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0).expand(batch, -1, -1)
    elif tokens.dim() == 3:
        if tokens.shape[0] == 1 and batch != 1:
            tokens = tokens.expand(batch, -1, -1)
        elif tokens.shape[0] != batch:
            raise ValueError("batched soft_tokens must match text_embeds batch")
    else:
        raise ValueError("soft_tokens must be [tokens, hidden] or [batch, tokens, hidden]")
    if tokens.shape[-1] != hidden:
        raise ValueError("soft_tokens hidden size must match text_embeds")
    return tokens


def _fmt_float(value: float) -> str:
    return ("%0.6f" % float(value)).rstrip("0").rstrip(".")


__all__ = [
    "SoftTokenBank",
    "SoftTokenConfig",
    "SoftTokenPrependResult",
    "build_soft_tokens_merge_decision",
    "build_soft_tokens_metadata",
    "build_soft_tokens_scorecard",
    "prepend_soft_tokens",
    "select_soft_token_bucket",
]
