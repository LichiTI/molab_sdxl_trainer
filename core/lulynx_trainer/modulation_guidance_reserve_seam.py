# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Default-off opt-in runtime seam for the **full pooled-text Modulation Guidance**
route (roadmap #11 / Frontier line 756 "AdaLN-Zero improvements").

Background -- partial vs full route
-----------------------------------
The *partial* route already shipped as :mod:`adaln_guidance`: a per-block
**constant** learnable ``shift``/``scale`` bias added to the AdaLN modulation
output.  It is text-independent -- the same bias is added regardless of the
prompt -- which is why the reconciliation contract
(:mod:`modulation_guidance_reconciliation`) marks it ``named/partial`` and blocks
``full_route_ready`` until five contracts exist:
``pooled_text_projection_contract`` / ``dedicated_distill_loop`` /
``inference_projection_contract`` / ``save_metadata_stamp`` /
``request_config_replay``.

This module is the *real integration* step that supplies those five contracts as
a single default-off reserve, mirroring every other Lulynx ``*_reserve_seam``:

* **Pooled-text projection** -- :class:`PooledTextModulationProjector` maps a
  pooled text vector to per-site ``(shift_delta, scale_delta)`` AdaLN deltas, so
  the guidance now *depends on the prompt*.  Its output projection is zero-init,
  so a freshly-installed projector emits **zero** deltas (bitwise parity).
* **Inference projection** -- :func:`modulation_guidance_pooled_text_context`
  publishes the pooled text on a generation-scoped ``ContextVar`` (the same shape
  as the cache / adapter-research seams), and
  :func:`install_modulation_guidance_reserve` wraps the AdaLN modulation modules
  so they consult it on every forward (training step *and* sampler loop).
* **Dedicated distill loop** -- :func:`compose_modulation_guidance_distill_loss`
  composes a teacher->student distillation term onto a base loss (off == base
  loss unchanged; on == real MSE + optional projector L2, gradients flow).
* **Save metadata / request replay** --
  :class:`ModulationGuidanceReplayRequest` round-trips the run config and reuses
  :func:`build_modulation_guidance_metadata` for the checkpoint stamp.

Design red-lines (identical to every other Lulynx default-off reserve)
----------------------------------------------------------------------
* **Default-off parity.** ``method="none"`` performs no wrapping; with the seam
  installed but no pooled text published, or a zero-init projector, the deltas
  are exactly zero, so the model is byte-for-byte today's model.
* **No base-model mutation.** The projector is a *separate, additive* module.
  The base ``llm_adapter.proj`` / ``timestep_embed`` / ``adaln_modulation``
  weights are never touched (Frontier line 756: do not mutate base init).
* **Opt-in only, no auto-consumption.** Installing the seam is caller-driven;
  the production training loop does not auto-consume it.  The readiness report
  keeps every operator/promotion flag ``False``.

Clean-room Lulynx module; references no external guidance source.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

try:  # package import
    from .modulation_guidance_reconciliation import (
        ModulationGuidanceObservation,
        build_modulation_guidance_metadata,
        build_modulation_guidance_reconciliation,
    )
except ImportError:  # pragma: no cover - direct-file smoke fallback
    from core.lulynx_trainer.modulation_guidance_reconciliation import (
        ModulationGuidanceObservation,
        build_modulation_guidance_metadata,
        build_modulation_guidance_reconciliation,
    )


MODULATION_GUIDANCE_METHODS: Tuple[str, ...] = ("none", "pooled_text")
_DEFAULT_BOTTLENECK = 256


# ---------------------------------------------------------------------------
# Generation-scoped pooled-text publication (training step + sampler loop)
# ---------------------------------------------------------------------------

_POOLED_TEXT: ContextVar[Optional[torch.Tensor]] = ContextVar(
    "lulynx_modulation_guidance_pooled_text", default=None
)


@contextlib.contextmanager
def modulation_guidance_pooled_text_context(pooled_text: Optional[torch.Tensor]) -> Iterator[None]:
    """Publish the batch's pooled text so the installed projector can read it.

    Outside this context (or with ``pooled_text=None``) the published value is
    ``None`` and every wrapped modulation site falls back to its original output
    (bitwise parity).
    """
    token = _POOLED_TEXT.set(pooled_text)
    try:
        yield
    finally:
        _POOLED_TEXT.reset(token)


def _published_pooled_text() -> Optional[torch.Tensor]:
    return _POOLED_TEXT.get()


# ---------------------------------------------------------------------------
# Pooled-text -> AdaLN modulation delta projector (the full-route keystone)
# ---------------------------------------------------------------------------

class PooledTextModulationProjector(nn.Module):
    """Project a pooled text vector to per-site ``(shift_delta, scale_delta)``.

    A single shared trunk plus a per-site embedding produces site-specific
    deltas, distilling the prompt into AdaLN steering.  The output projection is
    **zero-initialised**, so at init every delta is exactly zero -> the wrapped
    model reproduces its original output (parity) until the projector is trained.
    """

    def __init__(self, text_dim: int, hidden: int, num_sites: int, bottleneck: int = _DEFAULT_BOTTLENECK) -> None:
        super().__init__()
        if text_dim <= 0 or hidden <= 0 or num_sites <= 0:
            raise ValueError("text_dim, hidden and num_sites must be positive")
        self.text_dim = int(text_dim)
        self.hidden = int(hidden)
        self.num_sites = int(num_sites)
        self.norm = nn.LayerNorm(self.text_dim)
        self.in_proj = nn.Linear(self.text_dim, int(bottleneck))
        self.site_embed = nn.Embedding(self.num_sites, int(bottleneck))
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(int(bottleneck), 2 * self.hidden)
        # Zero-init the output projection and the site embedding so the initial
        # delta is identically zero (parity red-line).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.site_embed.weight)

    def forward(self, pooled_text: torch.Tensor, site_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if pooled_text.dim() == 1:
            pooled_text = pooled_text.unsqueeze(0)
        site = torch.full((pooled_text.shape[0],), int(site_index), dtype=torch.long, device=pooled_text.device)
        hidden = self.act(self.in_proj(self.norm(pooled_text)) + self.site_embed(site))
        delta = self.out_proj(hidden)
        shift_delta, scale_delta = delta[..., : self.hidden], delta[..., self.hidden :]
        return shift_delta, scale_delta


# ---------------------------------------------------------------------------
# Install / remove the reserve on a model's AdaLN modulation sites
# ---------------------------------------------------------------------------

def _default_modulation_predicate(name: str, module: nn.Module) -> bool:
    """Match AdaLN modulation *containers* whose forward returns chunked tuples.

    Selects the ``adaln_modulation`` container (e.g. the faithful ``_AdaLn``
    ``SiLU + Linear`` block) and deliberately **excludes** its child ``nn.Linear``
    projections -- those return a plain tensor, not the ``(shift, scale, gate)``
    tuple we add deltas to, and matching them would double-count sites.
    """
    if "adaln_modulation" not in name.lower():
        return False
    if isinstance(module, nn.Linear):
        return False
    # Must own at least one Linear child (the modulation projection).
    return any(isinstance(child, nn.Linear) for child in module.modules())


def _infer_site_hidden(module: nn.Module) -> Optional[int]:
    """Infer per-chunk hidden width = (max linear out_features) / chunk_count."""
    chunks = int(getattr(module, "chunks", 3) or 3)
    out_features = [child.out_features for child in module.modules() if isinstance(child, nn.Linear)]
    if not out_features:
        return None
    widest = max(out_features)
    if chunks <= 0 or widest % chunks != 0:
        return None
    return widest // chunks


def _wrap_site_forward(module: nn.Module, projector: PooledTextModulationProjector, site_index: int) -> Callable:
    original_forward = module.forward

    def patched(*args: Any, **kwargs: Any):
        chunks = original_forward(*args, **kwargs)
        pooled = _published_pooled_text()
        if pooled is None or not isinstance(chunks, tuple) or len(chunks) < 2:
            return chunks
        shift, scale = chunks[0], chunks[1]
        shift_delta, scale_delta = projector(pooled.to(dtype=shift.dtype, device=shift.device), site_index)
        return (shift + shift_delta, scale + scale_delta) + tuple(chunks[2:])

    return patched


class ModulationGuidanceReserveHandle:
    """Restores every modulation site wrapped by the reserve and owns the projector."""

    def __init__(
        self,
        method: str,
        projector: Optional[PooledTextModulationProjector],
        wraps: List[Tuple[nn.Module, Callable]],
    ) -> None:
        self.method = method
        self.projector = projector
        self._wraps = wraps
        self.active = True

    @property
    def wrapped_count(self) -> int:
        return len(self._wraps)

    def trainable_parameters(self) -> List[nn.Parameter]:
        if self.projector is None:
            return []
        return [p for p in self.projector.parameters() if p.requires_grad]

    def remove(self) -> None:
        if not self.active:
            return
        for module, _patched in self._wraps:
            # Drop the instance-level override -> reverts to the class method.
            if "forward" in module.__dict__:
                del module.__dict__["forward"]
        self.active = False


def install_modulation_guidance_reserve(
    model: nn.Module,
    method: str = "none",
    *,
    text_dim: Optional[int] = None,
    bottleneck: int = _DEFAULT_BOTTLENECK,
    target_predicate: Optional[Callable[[str, nn.Module], bool]] = None,
    projector: Optional[PooledTextModulationProjector] = None,
) -> ModulationGuidanceReserveHandle:
    """Opt-in wrap of AdaLN modulation sites with a pooled-text delta projector.

    ``method="none"`` (default) performs **no** wrapping -> bitwise parity.  For
    ``"pooled_text"`` every modulation site matched by *target_predicate* is
    wrapped so its ``(shift, scale)`` chunks get the projector's pooled-text
    deltas added (the gate chunk is left untouched).  With a zero-init projector
    or no published pooled text the deltas are zero, so an installed-but-idle
    reserve is still byte-for-byte today's model.
    """
    if method not in MODULATION_GUIDANCE_METHODS:
        raise ValueError(f"method must be one of {MODULATION_GUIDANCE_METHODS}, got {method!r}")
    if method == "none":
        return ModulationGuidanceReserveHandle("none", None, [])

    predicate = target_predicate or _default_modulation_predicate
    sites: List[Tuple[str, nn.Module]] = [
        (name, module) for name, module in model.named_modules() if predicate(name, module)
    ]
    if not sites:
        raise ValueError("no AdaLN modulation sites matched the predicate; nothing to wrap")

    if projector is None:
        if text_dim is None:
            raise ValueError("text_dim is required to build the pooled-text projector")
        hidden = next((h for h in (_infer_site_hidden(m) for _, m in sites) if h), None)
        if hidden is None:
            raise ValueError("could not infer modulation hidden width from the matched sites")
        projector = PooledTextModulationProjector(text_dim, hidden, len(sites), bottleneck=bottleneck)
        ref = next(model.parameters(), None)
        if ref is not None:
            projector.to(device=ref.device, dtype=ref.dtype)

    wraps: List[Tuple[nn.Module, Callable]] = []
    for site_index, (_name, module) in enumerate(sites):
        patched = _wrap_site_forward(module, projector, site_index)
        module.forward = patched  # type: ignore[assignment]
        wraps.append((module, patched))
    return ModulationGuidanceReserveHandle(method, projector, wraps)


# ---------------------------------------------------------------------------
# Dedicated distillation loss composition (default-off)
# ---------------------------------------------------------------------------

def compose_modulation_guidance_distill_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    *,
    enabled: bool = False,
    base_loss: Optional[torch.Tensor] = None,
    projector: Optional[PooledTextModulationProjector] = None,
    distill_weight: float = 1.0,
    reg_weight: float = 0.0,
) -> Dict[str, Any]:
    """Compose the pooled-text guidance distillation loss onto a base loss.

    With ``enabled=False`` the returned ``total`` is exactly ``base_loss`` (or a
    zero scalar) and no distillation terms are added (parity).  With
    ``enabled=True`` it adds ``distill_weight * MSE(student, teacher.detach())``
    (the teacher is the guided target the projection distills toward) plus an
    optional ``reg_weight`` L2 on the projector parameters; gradients flow to the
    projector.
    """
    zeros = student_pred.new_zeros(())
    base = base_loss if base_loss is not None else zeros
    if not enabled:
        return {"total": base, "distill": zeros, "projector_reg": zeros, "applied": False}
    distill = torch.nn.functional.mse_loss(student_pred, teacher_pred.detach())
    reg = zeros
    if projector is not None and reg_weight:
        reg = sum((p.pow(2).sum() for p in projector.parameters()), zeros)
    total = base + distill_weight * distill + reg_weight * reg
    return {"total": total, "distill": distill, "projector_reg": reg, "applied": True}


# ---------------------------------------------------------------------------
# Request / config replay contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModulationGuidanceReplayRequest:
    method: str = "none"
    text_dim: int = 0
    bottleneck: int = _DEFAULT_BOTTLENECK
    target_sites: int = 0
    distill_weight: float = 1.0
    reg_weight: float = 0.0

    def as_metadata(self) -> Dict[str, str]:
        full = self.method == "pooled_text"
        meta = build_modulation_guidance_metadata(
            ModulationGuidanceObservation(
                pooled_text_projection_contract=full,
                dedicated_distill_loop=full,
                inference_projection_contract=full,
                save_metadata_stamp=full,
                request_config_replay=full,
            )
        )
        meta.update(
            {
                "ss_modulation_guidance_method": self.method,
                "ss_modulation_guidance_text_dim": str(int(self.text_dim)),
                "ss_modulation_guidance_bottleneck": str(int(self.bottleneck)),
                "ss_modulation_guidance_target_sites": str(int(self.target_sites)),
                "ss_modulation_guidance_distill_weight": repr(float(self.distill_weight)),
                "ss_modulation_guidance_reg_weight": repr(float(self.reg_weight)),
            }
        )
        return meta


def build_modulation_guidance_replay_request(
    request: ModulationGuidanceReplayRequest | Mapping[str, Any] | None = None,
) -> ModulationGuidanceReplayRequest:
    """Round-trip a replay request from a dataclass / mapping (config replay)."""
    if isinstance(request, ModulationGuidanceReplayRequest):
        return request
    values = dict(request or {})
    method = str(values.get("method", "none"))
    if method not in MODULATION_GUIDANCE_METHODS:
        raise ValueError(f"method must be one of {MODULATION_GUIDANCE_METHODS}, got {method!r}")
    return ModulationGuidanceReplayRequest(
        method=method,
        text_dim=int(values.get("text_dim", 0) or 0),
        bottleneck=int(values.get("bottleneck", _DEFAULT_BOTTLENECK) or _DEFAULT_BOTTLENECK),
        target_sites=int(values.get("target_sites", 0) or 0),
        distill_weight=float(values.get("distill_weight", 1.0) or 0.0),
        reg_weight=float(values.get("reg_weight", 0.0) or 0.0),
    )


# ---------------------------------------------------------------------------
# Readiness -- bridge the wired seam to the governance contract
# ---------------------------------------------------------------------------

def modulation_guidance_reserve_readiness() -> Dict[str, Any]:
    """Read-only readiness report for the full pooled-text modulation guidance seam.

    The seam supplies all five reconciliation contracts, so the reconciliation
    built here reports ``full_route_ready=True`` -- but every operator/promotion
    flag stays ``False`` (lulynx does not self-sign runtime activation).  The
    *default* (no-arg) reconciliation is unchanged and still reports the current
    AdaLN bias route as partial.
    """
    reconciliation = build_modulation_guidance_reconciliation(
        ModulationGuidanceObservation(
            pooled_text_projection_contract=True,
            dedicated_distill_loop=True,
            inference_projection_contract=True,
            save_metadata_stamp=True,
            request_config_replay=True,
        )
    )
    return {
        "family": "anima",
        "scope": "modulation_guidance_reserve_only",
        "default_method": "none",
        "methods": list(MODULATION_GUIDANCE_METHODS),
        "existing_partial_route": "adaln_guidance",
        "wired": True,
        "full_route_ready": bool(reconciliation["full_route_ready"]),
        "reconciliation": reconciliation,
        # Honesty red-lines: opt-in only, not auto-consumed by sampler/trainer.
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "trainer_wiring_allowed": False,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
    }


__all__ = [
    "MODULATION_GUIDANCE_METHODS",
    "PooledTextModulationProjector",
    "ModulationGuidanceReplayRequest",
    "ModulationGuidanceReserveHandle",
    "modulation_guidance_pooled_text_context",
    "install_modulation_guidance_reserve",
    "compose_modulation_guidance_distill_loss",
    "build_modulation_guidance_replay_request",
    "modulation_guidance_reserve_readiness",
]
