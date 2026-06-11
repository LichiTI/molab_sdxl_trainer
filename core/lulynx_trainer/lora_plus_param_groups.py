"""LoRA+ optimizer param-group helpers.

The training routes all inject adapters slightly differently (standard LoRA,
DoRA, LoRA-FA, FeRA, HydraLoRA, T-LoRA, Flux targets, etc.).  Keep the A/B
classification here so optimizer construction can stay route-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_LORA_A_TOKENS = (
    "lora_down",
    "lora_a",
    "hada_w1",
    "lokr_w1_a",
)
_LORA_B_TOKENS = (
    "lora_up",
    "lora_b",
    "hada_w2",
    "lokr_w1_b",
)


@dataclass(frozen=True)
class LoraPlusParamGroupPlan:
    param_groups: list[dict[str, Any]]
    applied: bool
    lora_a_count: int = 0
    lora_b_count: int = 0
    other_count: int = 0
    fallback_reason: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": bool(self.applied),
            "lora_a_count": int(self.lora_a_count),
            "lora_b_count": int(self.lora_b_count),
            "other_count": int(self.other_count),
            "fallback_reason": self.fallback_reason,
            "note": self.note,
        }


def _param_requires_grad(param: Any) -> bool:
    return bool(getattr(param, "requires_grad", False))


def _classify_lora_plus_param(name: str) -> str:
    lowered = str(name or "").lower()
    if any(token in lowered for token in _LORA_A_TOKENS):
        return "a"
    if any(token in lowered for token in _LORA_B_TOKENS):
        return "b"
    return "other"


def _iter_named_trainable_params(layer: Any) -> Iterable[tuple[str, Any]]:
    named_parameters = getattr(layer, "named_parameters", None)
    if not callable(named_parameters):
        return ()
    return (
        (str(name), param)
        for name, param in named_parameters()
        if _param_requires_grad(param)
    )


def build_lora_plus_param_groups(
    *,
    injected_layers: dict[str, Any],
    trainable_params: Iterable[Any],
    base_lr: float,
    weight_decay: float,
    b_lr_ratio: float = 16.0,
) -> LoraPlusParamGroupPlan:
    """Split flat trainable params into LoRA+ A/B/other groups.

    LoRA+ only has an optimizer-side effect when trainable B/up parameters are
    present; other-only adapters are reported as profile-only instead of being
    wrapped in a no-op group split.
    """

    if not isinstance(injected_layers, dict) or not injected_layers:
        return LoraPlusParamGroupPlan(
            param_groups=[],
            applied=False,
            fallback_reason="LoRA+ param-group split requires injected adapter layers, but none were available.",
            note="LoRA+ stayed profile-only because no injected adapter layers were found.",
        )

    lora_a_params: list[Any] = []
    lora_b_params: list[Any] = []
    other_params: list[Any] = []
    seen: set[int] = set()

    for layer in injected_layers.values():
        for name, param in _iter_named_trainable_params(layer):
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            kind = _classify_lora_plus_param(name)
            if kind == "a":
                lora_a_params.append(param)
            elif kind == "b":
                lora_b_params.append(param)
            else:
                other_params.append(param)

    for param in list(trainable_params or []):
        if id(param) not in seen and _param_requires_grad(param):
            seen.add(id(param))
            other_params.append(param)

    if not lora_b_params:
        return LoraPlusParamGroupPlan(
            param_groups=[],
            applied=False,
            lora_a_count=len(lora_a_params),
            lora_b_count=0,
            other_count=len(other_params),
            fallback_reason="LoRA+ param-group split did not find trainable LoRA B/up parameters.",
            note="LoRA+ stayed profile-only because no trainable LoRA B/up params were detected.",
        )

    groups: list[dict[str, Any]] = []
    if lora_a_params:
        groups.append({
            "params": lora_a_params,
            "lr": float(base_lr),
            "weight_decay": float(weight_decay),
        })
    groups.append({
        "params": lora_b_params,
        "lr": float(base_lr) * float(b_lr_ratio or 16.0),
        "weight_decay": float(weight_decay),
    })
    if other_params:
        groups.append({
            "params": other_params,
            "lr": float(base_lr),
            "weight_decay": float(weight_decay),
        })

    return LoraPlusParamGroupPlan(
        param_groups=groups,
        applied=True,
        lora_a_count=len(lora_a_params),
        lora_b_count=len(lora_b_params),
        other_count=len(other_params),
        note="Applied LoRA+ param-group split during optimizer construction.",
    )


__all__ = ["LoraPlusParamGroupPlan", "build_lora_plus_param_groups"]
