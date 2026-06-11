"""Loss-stage planning for the Lulynx train-step pipeline.

The planner summarizes which loss family and auxiliary loss branches are active.
It does not compute loss values and does not inspect tensor contents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


LULYNX_TRAINING_STEP_LOSS_STAGE_PLAN = "lulynx_training_step_loss_stage_plan_v0"


@dataclass(frozen=True)
class LulynxTrainingStepLossStagePlan:
    model_arch: str
    core_loss_route: str
    loss_type: str
    batch_size: int
    uses_flow_matching: bool
    uses_sdxl_flow: bool
    uses_v_prediction_scaling: bool
    uses_masked_loss: bool
    strict_masked_loss: bool
    uses_debiased_estimation: bool
    uses_snr_weighting: bool
    uses_adaptive_loss_weighting: bool
    auxiliary_losses: tuple[str, ...]
    compile_caution_reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.batch_size > 0

    @property
    def compile_static_graph_risk(self) -> bool:
        return bool(self.compile_caution_reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan": LULYNX_TRAINING_STEP_LOSS_STAGE_PLAN,
            "ok": self.ok,
            "model_arch": self.model_arch,
            "core_loss_route": self.core_loss_route,
            "loss_type": self.loss_type,
            "batch_size": self.batch_size,
            "uses_flow_matching": self.uses_flow_matching,
            "uses_sdxl_flow": self.uses_sdxl_flow,
            "uses_v_prediction_scaling": self.uses_v_prediction_scaling,
            "uses_masked_loss": self.uses_masked_loss,
            "strict_masked_loss": self.strict_masked_loss,
            "uses_debiased_estimation": self.uses_debiased_estimation,
            "uses_snr_weighting": self.uses_snr_weighting,
            "uses_adaptive_loss_weighting": self.uses_adaptive_loss_weighting,
            "auxiliary_losses": list(self.auxiliary_losses),
            "compile_static_graph_risk": self.compile_static_graph_risk,
            "compile_caution_reasons": list(self.compile_caution_reasons),
        }


def build_lulynx_training_step_loss_stage_plan(
    *,
    batch: Mapping[str, Any],
    model_arch: str,
    batch_size: int,
    loss_type: str,
    uses_flow_matching: bool,
    uses_sdxl_flow: bool,
    sdxl_flow_sigmas_available: bool = False,
    v_parameterization: bool = False,
    masked_loss: bool = False,
    alpha_mask: bool = False,
    strict_masked_loss: bool = False,
    debiased_estimation: bool = False,
    snr_gamma: float = 0.0,
    adaptive_loss_weighter_available: bool = False,
    wavelet_loss_enabled: bool = False,
    pattern_loss_enabled: bool = False,
    prior_loss_weight: float = 0.0,
    reg_dataloader_available: bool = False,
    lulynx_wrapper_available: bool = False,
    repa_active: bool = False,
    dop_active: bool = False,
    b_tier_runtime_available: bool = False,
    do_backward: bool = True,
) -> LulynxTrainingStepLossStagePlan:
    arch = str(model_arch or "").strip().lower()
    size = max(_as_int(batch_size), 0)
    flow = bool(uses_flow_matching)
    sdxl_flow = bool(uses_sdxl_flow and sdxl_flow_sigmas_available)
    mask_enabled = bool(masked_loss or alpha_mask)
    aux = _auxiliary_losses(
        do_backward=bool(do_backward),
        wavelet_loss_enabled=wavelet_loss_enabled,
        pattern_loss_enabled=pattern_loss_enabled,
        prior_loss_weight=prior_loss_weight,
        reg_dataloader_available=reg_dataloader_available,
        lulynx_wrapper_available=lulynx_wrapper_available,
        repa_active=repa_active,
        dop_active=dop_active,
        b_tier_runtime_available=b_tier_runtime_available,
    )
    cautions: list[str] = []
    if mask_enabled and not _has_loss_masks(batch):
        cautions.append("masked_loss_requested_without_loss_masks")
    if aux:
        cautions.append("auxiliary_losses_extend_core_loss_graph")
    if bool(debiased_estimation):
        cautions.append("debiased_loss_uses_timestep_dependent_weights")
    if bool(adaptive_loss_weighter_available) or _as_float(snr_gamma) > 0.0:
        cautions.append("snr_loss_weighting_uses_timestep_dependent_weights")
    if size <= 0:
        cautions.append("batch_size_not_observable")
    return LulynxTrainingStepLossStagePlan(
        model_arch=arch,
        core_loss_route=_core_loss_route(
            model_arch=arch,
            uses_flow_matching=flow,
            uses_sdxl_flow=sdxl_flow,
            debiased_estimation=bool(debiased_estimation),
            snr_gamma=_as_float(snr_gamma),
            adaptive_loss_weighter_available=bool(adaptive_loss_weighter_available),
        ),
        loss_type=str(loss_type or "l2").strip().lower(),
        batch_size=size,
        uses_flow_matching=flow,
        uses_sdxl_flow=sdxl_flow,
        uses_v_prediction_scaling=bool(v_parameterization and not flow),
        uses_masked_loss=mask_enabled,
        strict_masked_loss=bool(strict_masked_loss),
        uses_debiased_estimation=bool(debiased_estimation and not flow),
        uses_snr_weighting=bool(_as_float(snr_gamma) > 0.0 and not flow),
        uses_adaptive_loss_weighting=bool(adaptive_loss_weighter_available and not flow),
        auxiliary_losses=aux,
        compile_caution_reasons=tuple(_dedupe(cautions)),
    )


def _core_loss_route(
    *,
    model_arch: str,
    uses_flow_matching: bool,
    uses_sdxl_flow: bool,
    debiased_estimation: bool,
    snr_gamma: float,
    adaptive_loss_weighter_available: bool,
) -> str:
    if model_arch == "anima":
        return "anima_flow_weighted_loss"
    if uses_sdxl_flow:
        return "sdxl_flow_weighted_loss"
    if (debiased_estimation or snr_gamma > 0.0 or adaptive_loss_weighter_available) and not uses_flow_matching:
        return "ddpm_weighted_loss"
    return "flow_loss" if uses_flow_matching else "ddpm_standard_loss"


def _auxiliary_losses(
    *,
    do_backward: bool,
    wavelet_loss_enabled: bool,
    pattern_loss_enabled: bool,
    prior_loss_weight: float,
    reg_dataloader_available: bool,
    lulynx_wrapper_available: bool,
    repa_active: bool,
    dop_active: bool,
    b_tier_runtime_available: bool,
) -> tuple[str, ...]:
    aux: list[str] = []
    if wavelet_loss_enabled:
        aux.append("wavelet")
    if pattern_loss_enabled:
        aux.append("pattern")
    if do_backward and _as_float(prior_loss_weight) > 0.0 and reg_dataloader_available:
        aux.append("prior_preservation")
    if do_backward and lulynx_wrapper_available:
        aux.append("lulynx_wrapper")
    if repa_active:
        aux.append("repa")
    if dop_active:
        aux.append("dop")
    if do_backward and b_tier_runtime_available:
        aux.append("b_tier")
    return tuple(aux)


def _has_loss_masks(batch: Mapping[str, Any]) -> bool:
    if not isinstance(batch, Mapping):
        return False
    return hasattr(batch.get("loss_masks"), "shape") or hasattr(batch.get("alpha_masks"), "shape")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


__all__ = [
    "LULYNX_TRAINING_STEP_LOSS_STAGE_PLAN",
    "LulynxTrainingStepLossStagePlan",
    "build_lulynx_training_step_loss_stage_plan",
]
