"""Default-off ChimeraHydra dual-pool MoE LoRA primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ChimeraHydraConfig:
    content_experts: int = 4
    frequency_experts: int = 2
    rank: int = 4
    alpha: float = 4.0
    routing: str = "top_k"
    content_top_k: int = 2
    frequency_top_k: int = 1
    gate_init_std: float = 0.01
    dropout: float = 0.0
    adapter_scope: str = "chimera_hydra"
    metadata_version: int = 1

    def validate(self) -> None:
        if self.content_experts < 1:
            raise ValueError("content_experts must be >= 1")
        if self.frequency_experts < 1:
            raise ValueError("frequency_experts must be >= 1")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.routing not in {"dense", "top_k"}:
            raise ValueError("routing must be dense or top_k")
        if self.content_top_k < 1 or self.content_top_k > self.content_experts:
            raise ValueError("content_top_k must be within content expert count")
        if self.frequency_top_k < 1 or self.frequency_top_k > self.frequency_experts:
            raise ValueError("frequency_top_k must be within frequency expert count")
        if self.adapter_scope != "chimera_hydra":
            raise ValueError("adapter_scope must stay chimera_hydra")


class ChimeraHydraLinear(nn.Module):
    """Dual-pool additive MoE LoRA layer.

    Existing HydraLoRA is a single-pool expert router. ChimeraHydra keeps a
    separate content pool and frequency/FEI pool, then adds both LoRA deltas.
    """

    def __init__(self, original: nn.Linear, config: ChimeraHydraConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        cfg = _coerce_config(config)
        cfg.validate()
        self.original = original
        self.config = cfg
        self.scaling = float(cfg.alpha) / float(max(cfg.rank, 1))
        for parameter in self.original.parameters():
            parameter.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features
        self.content_down = nn.Parameter(torch.empty(cfg.content_experts, cfg.rank, in_features))
        self.content_up = nn.Parameter(torch.empty(cfg.content_experts, out_features, cfg.rank))
        self.frequency_down = nn.Parameter(torch.empty(cfg.frequency_experts, cfg.rank, in_features))
        self.frequency_up = nn.Parameter(torch.empty(cfg.frequency_experts, out_features, cfg.rank))
        nn.init.kaiming_uniform_(self.content_down, a=5**0.5)
        nn.init.kaiming_uniform_(self.frequency_down, a=5**0.5)
        nn.init.zeros_(self.content_up)
        nn.init.zeros_(self.frequency_up)

        self.content_gate = nn.Linear(in_features, cfg.content_experts, bias=False)
        self.frequency_gate = nn.Linear(in_features, cfg.frequency_experts, bias=False)
        nn.init.normal_(self.content_gate.weight, mean=0.0, std=cfg.gate_init_std)
        nn.init.normal_(self.frequency_gate.weight, mean=0.0, std=cfg.gate_init_std)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, *, frequency_features: torch.Tensor | None = None) -> torch.Tensor:
        base = self.original(x)
        content_x = self.dropout(x)
        freq_x = self.dropout(frequency_features if frequency_features is not None else x)
        if freq_x.shape != x.shape:
            raise ValueError("frequency_features must match input shape")
        content_weights = self._pool_weights(self.content_gate(x), self.config.content_top_k)
        frequency_weights = self._pool_weights(self.frequency_gate(freq_x), self.config.frequency_top_k)
        content_delta = _mixed_delta(content_x, self.content_down, self.content_up, content_weights)
        frequency_delta = _mixed_delta(freq_x, self.frequency_down, self.frequency_up, frequency_weights)
        return base + (content_delta + frequency_delta) * self.scaling

    def _pool_weights(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if self.config.routing == "dense":
            return F.softmax(logits, dim=-1)
        topk_vals, topk_idx = logits.topk(top_k, dim=-1)
        normalized = F.softmax(topk_vals, dim=-1).to(dtype=logits.dtype)
        weights = torch.zeros_like(logits, dtype=logits.dtype)
        weights.scatter_(-1, topk_idx, normalized)
        return weights

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [
            self.content_down,
            self.content_up,
            self.frequency_down,
            self.frequency_up,
            self.content_gate.weight,
            self.frequency_gate.weight,
        ]

    def metadata(self) -> dict[str, str]:
        return build_chimera_hydra_metadata(self.config)

    def merge_decision(self) -> dict[str, Any]:
        return build_chimera_hydra_merge_decision(self.metadata())

    def routing_summary(self, x: torch.Tensor, *, frequency_features: torch.Tensor | None = None) -> dict[str, Any]:
        freq_x = frequency_features if frequency_features is not None else x
        if freq_x.shape != x.shape:
            raise ValueError("frequency_features must match input shape")
        with torch.no_grad():
            content_idx = self.content_gate(x).argmax(dim=-1).reshape(-1)
            freq_idx = self.frequency_gate(freq_x).argmax(dim=-1).reshape(-1)
        return {
            "content_experts_used": sorted(int(item) for item in content_idx.unique().tolist()),
            "frequency_experts_used": sorted(int(item) for item in freq_idx.unique().tolist()),
        }


def build_chimera_hydra_metadata(config: ChimeraHydraConfig | Mapping[str, Any] | None = None) -> dict[str, str]:
    cfg = _coerce_config(config)
    cfg.validate()
    return {
        "ss_adapter_type": "chimera_hydra",
        "ss_chimera_hydra_version": str(cfg.metadata_version),
        "ss_chimera_hydra_content_experts": str(cfg.content_experts),
        "ss_chimera_hydra_frequency_experts": str(cfg.frequency_experts),
        "ss_chimera_hydra_rank": str(cfg.rank),
        "ss_chimera_hydra_alpha": _fmt_float(cfg.alpha),
        "ss_chimera_hydra_routing": cfg.routing,
        "ss_chimera_hydra_content_top_k": str(cfg.content_top_k),
        "ss_chimera_hydra_frequency_top_k": str(cfg.frequency_top_k),
        "ss_chimera_hydra_non_mergeable": "true",
        "ss_chimera_hydra_requires_live_routing": "true",
        "ss_chimera_hydra_separate_from_hydralora": "true",
        "ss_training_path_enabled": "false",
        "ss_default_behavior_changed": "false",
    }


def build_chimera_hydra_merge_decision(metadata: Mapping[str, Any], *, requested_merge: bool = True) -> dict[str, Any]:
    adapter_type = str(metadata.get("ss_adapter_type") or "").strip()
    non_mergeable = str(metadata.get("ss_chimera_hydra_non_mergeable") or "").strip().lower() == "true"
    live_routing = str(metadata.get("ss_chimera_hydra_requires_live_routing") or "").strip().lower() == "true"
    separate = str(metadata.get("ss_chimera_hydra_separate_from_hydralora") or "").strip().lower() == "true"
    blockers: list[str] = []
    if adapter_type != "chimera_hydra":
        blockers.append("unexpected_adapter_type")
    if not separate:
        blockers.append("hydralora_separation_missing")
    if requested_merge and non_mergeable:
        blockers.append("chimera_hydra_dual_pool_is_dynamic")
    if requested_merge and live_routing:
        blockers.append("live_dual_pool_routing_required")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "chimera_hydra_merge_refusal_v0",
        "ok": ready,
        "merge_allowed": ready,
        "requested_merge": bool(requested_merge),
        "non_mergeable": bool(non_mergeable),
        "requires_live_routing": bool(live_routing),
        "separate_from_hydralora": bool(separate),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "merge may proceed"
            if ready
            else "keep ChimeraHydra as a separate live dual-pool adapter; do not mutate HydraLoRA"
        ),
    }


def build_chimera_hydra_scorecard(
    *,
    config: ChimeraHydraConfig | Mapping[str, Any] | None = None,
    disabled_parity_ok: bool = False,
    content_pool_active: bool = False,
    frequency_pool_active: bool = False,
    metadata_roundtrip_ok: bool = False,
    merge_refusal_ok: bool = False,
) -> dict[str, Any]:
    cfg = _coerce_config(config)
    blockers: list[str] = []
    try:
        cfg.validate()
    except ValueError as exc:
        blockers.append(f"invalid_config:{exc}")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_missing")
    if not content_pool_active:
        blockers.append("content_pool_activity_missing")
    if not frequency_pool_active:
        blockers.append("frequency_pool_activity_missing")
    if not metadata_roundtrip_ok:
        blockers.append("metadata_roundtrip_missing")
    if not merge_refusal_ok:
        blockers.append("merge_refusal_missing")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "chimera_hydra_adapter_primitive_v0",
        "ok": ready,
        "primitive_ready": ready,
        "adapter_type": "chimera_hydra",
        "content_experts": cfg.content_experts,
        "frequency_experts": cfg.frequency_experts,
        "rank": cfg.rank,
        "routing": cfg.routing,
        "non_mergeable": True,
        "requires_live_routing": True,
        "separate_from_hydralora": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add ChimeraHydra trainer preflight before any adapter registry wiring"
            if ready
            else "complete ChimeraHydra parity, dual-pool activity, metadata, and merge-refusal proof"
        ),
    }


def _mixed_delta(
    x: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    projected = torch.einsum("...i,eri->...er", x, down)
    deltas = torch.einsum("...er,eor->...eo", projected, up)
    return (weights.unsqueeze(-1) * deltas).sum(dim=-2)


def _coerce_config(config: ChimeraHydraConfig | Mapping[str, Any] | None) -> ChimeraHydraConfig:
    if isinstance(config, ChimeraHydraConfig):
        return config
    values = dict(config or {})
    return ChimeraHydraConfig(
        content_experts=int(values.get("content_experts", ChimeraHydraConfig.content_experts)),
        frequency_experts=int(values.get("frequency_experts", ChimeraHydraConfig.frequency_experts)),
        rank=int(values.get("rank", ChimeraHydraConfig.rank)),
        alpha=float(values.get("alpha", ChimeraHydraConfig.alpha)),
        routing=str(values.get("routing", ChimeraHydraConfig.routing)),
        content_top_k=int(values.get("content_top_k", ChimeraHydraConfig.content_top_k)),
        frequency_top_k=int(values.get("frequency_top_k", ChimeraHydraConfig.frequency_top_k)),
        gate_init_std=float(values.get("gate_init_std", ChimeraHydraConfig.gate_init_std)),
        dropout=float(values.get("dropout", ChimeraHydraConfig.dropout)),
        adapter_scope=str(values.get("adapter_scope", ChimeraHydraConfig.adapter_scope)),
        metadata_version=int(values.get("metadata_version", ChimeraHydraConfig.metadata_version)),
    )


def _fmt_float(value: float) -> str:
    return ("%0.6f" % float(value)).rstrip("0").rstrip(".")


__all__ = [
    "ChimeraHydraConfig",
    "ChimeraHydraLinear",
    "build_chimera_hydra_merge_decision",
    "build_chimera_hydra_metadata",
    "build_chimera_hydra_scorecard",
]
