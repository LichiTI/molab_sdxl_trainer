"""Default-off Step Expert routing primitive for Turbo-style adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class StepExpertConfig:
    num_experts: int = 4
    rank: int = 4
    alpha: float = 4.0
    boundaries: tuple[float, ...] = (0.25, 0.5, 0.75)
    adapter_scope: str = "turbo_only"
    metadata_version: int = 1

    def validate(self) -> None:
        if self.num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if len(self.boundaries) != self.num_experts - 1:
            raise ValueError("boundaries must contain num_experts - 1 entries")
        last = 0.0
        for value in self.boundaries:
            if not 0.0 < float(value) < 1.0:
                raise ValueError("boundaries must be inside (0, 1)")
            if float(value) <= last:
                raise ValueError("boundaries must be strictly increasing")
            last = float(value)
        if self.adapter_scope != "turbo_only":
            raise ValueError("Step Expert is currently limited to turbo_only scope")


class StepExpertLoRALinear(nn.Module):
    """LoRA layer whose expert is selected from the denoise step.

    This is intentionally not a drop-in trainer integration. The adapter is
    step-dependent, so it is non-mergeable unless a future route defines a
    separate kept-live runtime adapter.
    """

    def __init__(self, original: nn.Linear, config: StepExpertConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        cfg = _coerce_config(config)
        cfg.validate()
        self.original = original
        self.config = cfg
        self.scaling = float(cfg.alpha) / float(max(cfg.rank, 1))
        for parameter in self.original.parameters():
            parameter.requires_grad = False

        self.lora_down = nn.Parameter(torch.empty(cfg.num_experts, cfg.rank, original.in_features))
        self.lora_up = nn.Parameter(torch.empty(cfg.num_experts, original.out_features, cfg.rank))
        nn.init.kaiming_uniform_(self.lora_down, a=5**0.5)
        nn.init.zeros_(self.lora_up)

    def forward(self, x: torch.Tensor, *, timestep: Any = None, total_steps: int | None = None) -> torch.Tensor:
        base = self.original(x)
        if timestep is None:
            # Fallback: average all experts when timestep is not provided
            deltas = self._all_expert_deltas(x)  # [batch, ..., num_experts, out_features]
            averaged = deltas.mean(dim=-2)  # [batch, ..., out_features]
            return base + averaged
        indices = select_step_expert(timestep, total_steps=total_steps, boundaries=self.config.boundaries)
        deltas = self._all_expert_deltas(x)
        selected = _select_expert_delta(deltas, indices, x.shape[:-1])
        return base + selected

    def _all_expert_deltas(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.einsum("...i,eri->...er", x, self.lora_down)
        deltas = torch.einsum("...er,eor->...eo", projected, self.lora_up)
        return deltas * self.scaling

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [self.lora_down, self.lora_up]

    def metadata(self) -> dict[str, str]:
        return build_step_expert_metadata(self.config)

    def merge_decision(self) -> dict[str, Any]:
        return build_step_expert_merge_decision(self.metadata())


def select_step_expert(
    timestep: Any,
    *,
    total_steps: int | None = None,
    boundaries: Sequence[float] = (0.25, 0.5, 0.75),
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


def build_step_expert_metadata(config: StepExpertConfig | Mapping[str, Any] | None = None) -> dict[str, str]:
    cfg = _coerce_config(config)
    cfg.validate()
    return {
        "ss_adapter_type": "step_expert",
        "ss_step_expert_version": str(cfg.metadata_version),
        "ss_step_expert_scope": cfg.adapter_scope,
        "ss_step_expert_num_experts": str(cfg.num_experts),
        "ss_step_expert_rank": str(cfg.rank),
        "ss_step_expert_alpha": _fmt_float(cfg.alpha),
        "ss_step_expert_boundaries": ",".join(_fmt_float(item) for item in cfg.boundaries),
        "ss_step_expert_non_mergeable": "true",
        "ss_step_expert_requires_live_routing": "true",
        "ss_training_path_enabled": "false",
        "ss_default_behavior_changed": "false",
    }


def build_step_expert_merge_decision(metadata: Mapping[str, Any], *, requested_merge: bool = True) -> dict[str, Any]:
    adapter_type = str(metadata.get("ss_adapter_type") or "").strip()
    non_mergeable = str(metadata.get("ss_step_expert_non_mergeable") or "").strip().lower() == "true"
    live_routing = str(metadata.get("ss_step_expert_requires_live_routing") or "").strip().lower() == "true"
    blockers: list[str] = []
    if adapter_type != "step_expert":
        blockers.append("unexpected_adapter_type")
    if requested_merge and non_mergeable:
        blockers.append("step_expert_is_step_dependent")
    if requested_merge and live_routing:
        blockers.append("live_step_routing_required")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "step_expert_merge_refusal_v0",
        "ok": ready,
        "merge_allowed": ready,
        "requested_merge": bool(requested_merge),
        "non_mergeable": bool(non_mergeable),
        "requires_live_routing": bool(live_routing),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "merge may proceed"
            if ready
            else "keep Step Expert as a live Turbo-only adapter and do not bake it into base weights"
        ),
    }


def build_step_expert_routing_scorecard(
    *,
    config: StepExpertConfig | Mapping[str, Any] | None = None,
    observed_experts: Sequence[int] = (),
    disabled_parity_ok: bool = False,
    metadata_roundtrip_ok: bool = False,
    merge_refusal_ok: bool = False,
) -> dict[str, Any]:
    cfg = _coerce_config(config)
    blockers: list[str] = []
    try:
        cfg.validate()
    except ValueError as exc:
        blockers.append(f"invalid_config:{exc}")
    observed = tuple(int(item) for item in observed_experts)
    if not observed:
        blockers.append("observed_experts_missing")
    elif set(observed) != set(range(cfg.num_experts)):
        blockers.append("expert_coverage_incomplete")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_missing")
    if not metadata_roundtrip_ok:
        blockers.append("metadata_roundtrip_missing")
    if not merge_refusal_ok:
        blockers.append("merge_refusal_missing")
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "step_expert_routing_primitive_v0",
        "ok": ready,
        "primitive_ready": ready,
        "adapter_scope": cfg.adapter_scope,
        "num_experts": cfg.num_experts,
        "rank": cfg.rank,
        "boundaries": list(cfg.boundaries),
        "observed_experts": list(observed),
        "non_mergeable": True,
        "requires_live_routing": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add Step Expert trainer preflight before any Turbo route wiring"
            if ready
            else "complete Step Expert primitive coverage, metadata, and merge-refusal proof"
        ),
    }


def _coerce_config(config: StepExpertConfig | Mapping[str, Any] | None) -> StepExpertConfig:
    if isinstance(config, StepExpertConfig):
        return config
    values = dict(config or {})
    boundaries = values.get("boundaries", StepExpertConfig.boundaries)
    if isinstance(boundaries, str):
        boundaries = tuple(float(item.strip()) for item in boundaries.split(",") if item.strip())
    return StepExpertConfig(
        num_experts=int(values.get("num_experts", StepExpertConfig.num_experts)),
        rank=int(values.get("rank", StepExpertConfig.rank)),
        alpha=float(values.get("alpha", StepExpertConfig.alpha)),
        boundaries=tuple(float(item) for item in boundaries),
        adapter_scope=str(values.get("adapter_scope", StepExpertConfig.adapter_scope)),
        metadata_version=int(values.get("metadata_version", StepExpertConfig.metadata_version)),
    )


def _tensor_fraction(timestep: torch.Tensor, total_steps: int | None) -> torch.Tensor:
    value = timestep.to(dtype=torch.float32)
    if total_steps is not None:
        denom = max(int(total_steps) - 1, 1)
        value = value / float(denom)
    return value.clamp(0.0, 1.0)


def _scalar_fraction(timestep: Any, total_steps: int | None) -> float:
    value = float(timestep)
    if total_steps is not None:
        value = value / float(max(int(total_steps) - 1, 1))
    return min(max(value, 0.0), 1.0)


def _select_expert_delta(deltas: torch.Tensor, indices: int | torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if isinstance(indices, int):
        return deltas[..., indices, :]
    idx = indices.to(device=deltas.device, dtype=torch.long)
    while idx.dim() < len(target_shape):
        idx = idx.unsqueeze(-1)
    idx = idx.expand(target_shape)
    gather_index = idx.unsqueeze(-1).unsqueeze(-1).expand(*target_shape, 1, deltas.shape[-1])
    return deltas.gather(dim=-2, index=gather_index).squeeze(-2)


def _fmt_float(value: float) -> str:
    return ("%0.6f" % float(value)).rstrip("0").rstrip(".")


__all__ = [
    "StepExpertConfig",
    "StepExpertLoRALinear",
    "build_step_expert_merge_decision",
    "build_step_expert_metadata",
    "build_step_expert_routing_scorecard",
    "select_step_expert",
]
