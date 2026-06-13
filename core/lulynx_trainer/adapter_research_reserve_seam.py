# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Default-off opt-in runtime seam for the P3 adapter-research reserves.

The P3 primitives (``step_expert_routing``, ``chimera_hydra``, ``soft_tokens``)
were each proven in isolation.  This module is the *real integration* step that
the roadmap tracked as "remains a separate route decision": a seam that can
actually run those adapters inside a genuine model forward when a caller opts in,
while staying **bitwise-parity** when off.

Design red-lines (identical to every other Lulynx default-off reserve)
----------------------------------------------------------------------
* **Default-off parity.** ``install_adapter_research_reserve(model, "none")``
  performs no module swaps and returns an inert handle, so an installed-but-off
  reserve is byte-for-byte today's model.  ``apply_soft_tokens_reserve`` with no
  bank / ``enabled=False`` returns its inputs unchanged.
* **Identity at init.** The Linear-wrapper adapters (StepExpert / ChimeraHydra)
  are zero-init on their up-projection, so a freshly-swapped reserve reproduces
  the original linear output exactly until the adapter is trained.
* **Opt-in only, no auto-consumption.** Installing the seam is caller-driven; the
  production training loop does not auto-consume it.  ``training_step_consumption``
  stays ``False`` and ``promotion_ready`` stays ``False`` in the readiness report.

The step-routed adapters read the live denoise step from a generation-scoped
``ContextVar`` (``adapter_research_step_context``), mirroring how the cache /
T-GATE seams publish their per-step state, so a real sampler loop can drive
expert routing without threading the timestep through every call site.

Clean-room Lulynx module; references no external adapter source.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

try:  # package import
    from .step_expert_routing import StepExpertConfig, StepExpertLoRALinear
    from .chimera_hydra import ChimeraHydraConfig, ChimeraHydraLinear
    from .soft_tokens import SoftTokenBank, prepend_soft_tokens
except ImportError:  # pragma: no cover - direct-file smoke fallback
    from core.lulynx_trainer.step_expert_routing import StepExpertConfig, StepExpertLoRALinear
    from core.lulynx_trainer.chimera_hydra import ChimeraHydraConfig, ChimeraHydraLinear
    from core.lulynx_trainer.soft_tokens import SoftTokenBank, prepend_soft_tokens


ADAPTER_RESEARCH_LINEAR_METHODS: Tuple[str, ...] = ("step_expert", "chimera_hydra")
ADAPTER_RESEARCH_METHODS: Tuple[str, ...] = ("none",) + ADAPTER_RESEARCH_LINEAR_METHODS + ("soft_tokens",)

# Common LoRA-style target linear suffixes on Anima executable-subset blocks.
_DEFAULT_TARGET_SUFFIXES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "output_proj",
    "layer1",
    "layer2",
)


class _AdapterResearchStep:
    """Generation-scoped step state consumed by step-routed reserves."""

    __slots__ = ("timestep", "total_steps", "frequency_features")

    def __init__(self, timestep: Any, total_steps: Optional[int], frequency_features: Optional[torch.Tensor]) -> None:
        self.timestep = timestep
        self.total_steps = total_steps
        self.frequency_features = frequency_features


_ADAPTER_RESEARCH_STEP: ContextVar[Optional[_AdapterResearchStep]] = ContextVar(
    "lulynx_adapter_research_step", default=None
)


@contextlib.contextmanager
def adapter_research_step_context(
    timestep: Any = None,
    *,
    total_steps: Optional[int] = None,
    frequency_features: Optional[torch.Tensor] = None,
) -> Iterator[None]:
    """Publish the live denoise step so step-routed reserves can read it.

    Outside this context the published state is ``None`` and the wrappers fall
    back to their step-agnostic path (which is still identity at init).
    """
    token = _ADAPTER_RESEARCH_STEP.set(
        _AdapterResearchStep(timestep, total_steps, frequency_features)
    )
    try:
        yield
    finally:
        _ADAPTER_RESEARCH_STEP.reset(token)


def _published_step() -> Optional[_AdapterResearchStep]:
    return _ADAPTER_RESEARCH_STEP.get()


def _default_target_predicate(name: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in _DEFAULT_TARGET_SUFFIXES)


def _iter_linear_targets(
    model: nn.Module, predicate: Callable[[str, nn.Module], bool]
) -> Iterator[Tuple[nn.Module, str, str, nn.Linear]]:
    name_to_module = dict(model.named_modules())
    for qualified_name, module in list(name_to_module.items()):
        if not predicate(qualified_name, module):
            continue
        if "." in qualified_name:
            parent_name, attr = qualified_name.rsplit(".", 1)
            parent = name_to_module.get(parent_name)
        else:
            parent, attr = model, qualified_name
        if parent is None or not isinstance(getattr(parent, attr, None), nn.Linear):
            continue
        yield parent, attr, qualified_name, getattr(parent, attr)


def _build_linear_wrapper(method: str, original: nn.Linear, config: Any) -> nn.Module:
    if method == "step_expert":
        return StepExpertLoRALinear(original, config if isinstance(config, (StepExpertConfig, dict)) else None)
    if method == "chimera_hydra":
        return ChimeraHydraLinear(original, config if isinstance(config, (ChimeraHydraConfig, dict)) else None)
    raise ValueError(f"unsupported linear reserve method: {method}")


def _bind_step_routed_forward(primitive: nn.Module, method: str) -> None:
    """Make the wrapper consult the published step state on each call.

    The wrappers already accept a plain positional ``module(x)`` call (identity at
    init), so binding only *adds* live routing when a step context is active.
    """
    inner = primitive.forward

    if method == "step_expert":
        def routed_forward(x: torch.Tensor) -> torch.Tensor:
            step = _published_step()
            if step is None:
                return inner(x)
            return inner(x, timestep=step.timestep, total_steps=step.total_steps)

        primitive.forward = routed_forward  # type: ignore[assignment]
    elif method == "chimera_hydra":
        def freq_forward(x: torch.Tensor) -> torch.Tensor:
            step = _published_step()
            if step is None or step.frequency_features is None:
                return inner(x)
            return inner(x, frequency_features=step.frequency_features)

        primitive.forward = freq_forward  # type: ignore[assignment]


class AdapterResearchReserveHandle:
    """Restores every linear swapped by :func:`install_adapter_research_reserve`."""

    def __init__(self, method: str, swaps: List[Tuple[nn.Module, str, nn.Linear, nn.Module]]) -> None:
        self.method = method
        self._swaps = swaps
        self.active = True

    @property
    def wrapped_count(self) -> int:
        return len(self._swaps)

    @property
    def target_names(self) -> Tuple[str, ...]:
        return tuple(getattr(primitive, "_lulynx_reserve_name", "") for *_, primitive in self._swaps)

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for *_, primitive in self._swaps:
            getter = getattr(primitive, "get_trainable_params", None)
            if callable(getter):
                params.extend(getter())
        return params

    def remove(self) -> None:
        if not self.active:
            return
        for parent, attr, original, _primitive in self._swaps:
            setattr(parent, attr, original)
        self.active = False


def install_adapter_research_reserve(
    model: nn.Module,
    method: str = "none",
    *,
    target_predicate: Optional[Callable[[str, nn.Module], bool]] = None,
    config: Any = None,
) -> AdapterResearchReserveHandle:
    """Opt-in swap of target linears for a P3 adapter-research reserve.

    ``method="none"`` (default) performs **no** swaps -> bitwise parity.  For
    ``"step_expert"`` / ``"chimera_hydra"`` each matching ``nn.Linear`` is replaced
    by the corresponding zero-init wrapper (identity at init).  Returns a handle
    whose ``remove()`` restores every original linear.
    """
    if method not in ADAPTER_RESEARCH_LINEAR_METHODS and method != "none":
        raise ValueError(
            f"method must be one of {('none',) + ADAPTER_RESEARCH_LINEAR_METHODS}, got {method!r}"
        )
    if method == "none":
        return AdapterResearchReserveHandle("none", [])

    predicate = target_predicate or _default_target_predicate
    swaps: List[Tuple[nn.Module, str, nn.Linear, nn.Module]] = []
    for parent, attr, qualified_name, original in _iter_linear_targets(model, predicate):
        primitive = _build_linear_wrapper(method, original, config)
        primitive._lulynx_reserve_name = qualified_name  # type: ignore[attr-defined]
        _bind_step_routed_forward(primitive, method)
        setattr(parent, attr, primitive)
        swaps.append((parent, attr, original, primitive))
    if not swaps:
        raise ValueError("no target linear modules matched the predicate; nothing to swap")
    return AdapterResearchReserveHandle(method, swaps)


def apply_soft_tokens_reserve(
    bank: Optional[SoftTokenBank],
    text_embeds: torch.Tensor,
    *,
    layer_index: int = 0,
    timestep: Any = None,
    total_steps: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    enabled: bool = True,
):
    """Default-off soft-token prepend reserve.

    With no bank or ``enabled=False`` this returns the inputs unchanged
    (parity).  Otherwise it prepends the bank's layer/timestep-selected tokens.
    Returns the primitive's :class:`SoftTokenPrependResult`.
    """
    if bank is None or not enabled:
        return prepend_soft_tokens(text_embeds, None, attention_mask=attention_mask, layer_index=layer_index)
    return bank.prepend(
        text_embeds,
        layer_index=layer_index,
        timestep=timestep,
        total_steps=total_steps,
        attention_mask=attention_mask,
        enabled=True,
    )


def adapter_research_reserve_readiness() -> dict:
    """Read-only readiness report for the P3 adapter-research reserves."""
    return {
        "family": "anima",
        "scope": "adapter_research_reserve_only",
        "default_method": "none",
        "linear_methods": list(ADAPTER_RESEARCH_LINEAR_METHODS),
        "methods": {
            "step_expert": {"wired": True, "kind": "linear_swap", "requires_live_routing": True},
            "chimera_hydra": {"wired": True, "kind": "linear_swap", "requires_live_routing": True},
            "soft_tokens": {"wired": True, "kind": "text_prepend", "requires_live_routing": True},
        },
        "wired": True,
        # Honesty red-lines: opt-in only, not auto-consumed by the training loop.
        "training_step_consumption": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
    }


__all__ = [
    "ADAPTER_RESEARCH_METHODS",
    "ADAPTER_RESEARCH_LINEAR_METHODS",
    "adapter_research_step_context",
    "install_adapter_research_reserve",
    "apply_soft_tokens_reserve",
    "adapter_research_reserve_readiness",
    "AdapterResearchReserveHandle",
]
