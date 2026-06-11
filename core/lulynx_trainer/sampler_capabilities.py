"""Sampler and scheduler capability matrix for preview/generation routes.

The matrix is intentionally static and import-light.  It lets request adapters,
smokes, and UI catalog code ask what a sampler means without importing
``diffusers`` or touching a live pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple


DIFFUSERS_FAMILIES: Tuple[str, ...] = ("sd15", "sdxl")
FLOW_FAMILIES: Tuple[str, ...] = ("anima", "newbie")


@dataclass(frozen=True)
class SamplerCapability:
    id: str
    label: str
    backend: str
    families: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()
    scheduler_class: Optional[str] = None
    scheduler_kwargs: Tuple[Tuple[str, Any], ...] = ()
    karras_supported: bool = False
    stochastic: bool = False
    flow_matching_supported: bool = False
    status: str = "available"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["families"] = list(self.families)
        payload["aliases"] = list(self.aliases)
        payload["scheduler_kwargs"] = dict(self.scheduler_kwargs)
        return payload


_CAPABILITIES: Tuple[SamplerCapability, ...] = (
    SamplerCapability(
        id="euler_a",
        label="Euler a",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        aliases=("euler_ancestral", "euler-ancestral", "eulera"),
        scheduler_class="EulerAncestralDiscreteScheduler",
        stochastic=True,
    ),
    SamplerCapability(
        id="euler",
        label="Euler",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        scheduler_class="EulerDiscreteScheduler",
    ),
    SamplerCapability(
        id="ddim",
        label="DDIM",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        scheduler_class="DDIMScheduler",
    ),
    SamplerCapability(
        id="pndm",
        label="PNDM",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        scheduler_class="PNDMScheduler",
    ),
    SamplerCapability(
        id="lms",
        label="LMS",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        scheduler_class="LMSDiscreteScheduler",
    ),
    SamplerCapability(
        id="heun",
        label="Heun",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        scheduler_class="HeunDiscreteScheduler",
    ),
    SamplerCapability(
        id="dpm++_2m",
        label="DPM++ 2M",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        aliases=("dpmsolver", "dpmsolver++", "dpmpp_2m", "dpmpp2m"),
        scheduler_class="DPMSolverMultistepScheduler",
        karras_supported=True,
    ),
    SamplerCapability(
        id="dpm++_2m_karras",
        label="DPM++ 2M Karras",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        aliases=("dpmsolver++_karras", "dpmpp_2m_karras", "dpmpp2m_karras"),
        scheduler_class="DPMSolverMultistepScheduler",
        scheduler_kwargs=(("use_karras_sigmas", True),),
        karras_supported=True,
    ),
    SamplerCapability(
        id="dpm++_sde",
        label="DPM++ SDE",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        aliases=("dpm++_2m_sde", "dpmpp_sde", "dpmpp_2m_sde"),
        scheduler_class="DPMSolverSinglestepScheduler",
        karras_supported=True,
        stochastic=True,
        notes="The dpm++_2m_sde alias maps to the existing singlestep SDE route.",
    ),
    SamplerCapability(
        id="dpm++_sde_karras",
        label="DPM++ SDE Karras",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        aliases=("dpm++_2m_sde_karras", "dpmpp_sde_karras", "dpmpp_2m_sde_karras"),
        scheduler_class="DPMSolverSinglestepScheduler",
        scheduler_kwargs=(("use_karras_sigmas", True),),
        karras_supported=True,
        stochastic=True,
    ),
    SamplerCapability(
        id="dpm_multistep",
        label="DPM2",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        aliases=("dpm_2", "k_dpm_2", "kdpm2"),
        scheduler_class="KDPM2DiscreteScheduler",
    ),
    SamplerCapability(
        id="dpm_ancestral",
        label="DPM2 a",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        aliases=("dpm_2_a", "k_dpm_2_a", "kdpm2_a"),
        scheduler_class="KDPM2AncestralDiscreteScheduler",
        stochastic=True,
    ),
    SamplerCapability(
        id="uni_pc",
        label="UniPC",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        aliases=("unipc", "uni-pc", "unipc_multistep"),
        scheduler_class="UniPCMultistepScheduler",
    ),
    SamplerCapability(
        id="deis",
        label="DEIS",
        backend="diffusers",
        families=DIFFUSERS_FAMILIES,
        aliases=("deis_multistep",),
        scheduler_class="DEISMultistepScheduler",
    ),
    SamplerCapability(
        id="flow_euler",
        label="Flow Euler",
        backend="lulynx_flow",
        families=FLOW_FAMILIES,
        aliases=("euler", "flow_euler", "flow-euler"),
        flow_matching_supported=True,
        notes="Anima and Newbie flow preview default.",
    ),
    SamplerCapability(
        id="flow_dpm_solver",
        label="Flow DPM-Solver",
        backend="lulynx_flow",
        families=("anima",),
        aliases=("dpm_solver", "dpm-solver", "flow_dpm_solver"),
        flow_matching_supported=True,
        status="experimental",
        notes="Current Anima implementation is first-order and Euler-equivalent.",
    ),
    SamplerCapability(
        id="cns",
        label="CNS Colored Noise Sampling",
        backend="sampler_feature",
        families=("anima", "newbie"),
        status="available",
        notes="CNS ER-SDE sampler variants available for Anima/Newbie; requires calibration .npz file.",
    ),
    SamplerCapability(
        id="smc_cfg",
        label="SMC-CFG",
        backend="sampler_feature",
        families=("anima", "newbie", "sd15", "sdxl"),
        aliases=("smc-cfg", "sliding_mode_cfg"),
        status="partial",
        notes="Available for Anima/Newbie flow CFG combine; SDXL/SD1.5 output-space support is not asserted.",
    ),
    SamplerCapability(
        id="tgate",
        label="T-GATE",
        backend="sampler_feature",
        families=("anima", "newbie"),
        aliases=("t-gate", "tgate_probe", "t_gate"),
        status="available",
        notes="T-GATE cross-attention skip/cache available; use run_with_tgate_skip wrapper for manual integration.",
    ),
    SamplerCapability(
        id="spectrum",
        label="Spectrum",
        backend="sampler_feature",
        families=("anima", "newbie"),
        aliases=("spectrum_probe", "spectrum_observe"),
        status="available",
        notes="Spectrum block skip with linear extrapolation available; use run_with_spectrum_skip wrapper for manual integration.",
    ),
    SamplerCapability(
        id="smoothcache",
        label="SmoothCache",
        backend="sampler_feature",
        families=("anima", "newbie"),
        aliases=("smooth_cache", "smoothcache_probe"),
        status="available",
        notes="Error-guided per-block feature caching; observe-only probe wired into the DiT/sampler, use run_with_smoothcache wrapper for manual reuse. Inference/preview accel only.",
    ),
)


def _normalize_key(name: Optional[str]) -> str:
    value = str(name or "").strip().lower()
    value = value.replace(" ", "_").replace("-", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value


def _alias_map() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for capability in _CAPABILITIES:
        aliases.setdefault(_normalize_key(capability.id), capability.id)
        for alias in capability.aliases:
            aliases.setdefault(_normalize_key(alias), capability.id)
    return aliases


_ALIASES = _alias_map()
_BY_ID = {capability.id: capability for capability in _CAPABILITIES}


def normalize_sampler_name(name: Optional[str], *, family: Optional[str] = None) -> str:
    """Return the canonical sampler id for a user-facing name.

    ``family`` disambiguates names such as ``euler``: diffusers routes keep
    ``euler`` while flow-matching routes map it to ``flow_euler``.
    """
    family_key = _normalize_key(family)
    name_key = _normalize_key(name) or ("euler" if family_key in FLOW_FAMILIES else "euler_a")

    if family_key in FLOW_FAMILIES:
        if name_key in {"euler", "flow_euler", "flow-euler"}:
            return "flow_euler"
        if name_key in {"dpm_solver", "dpm-solver", "flow_dpm_solver"}:
            return "flow_dpm_solver" if family_key == "anima" else "flow_euler"

    return _ALIASES.get(name_key, name_key)


def get_sampler_capability(name: Optional[str], *, family: Optional[str] = None) -> Optional[SamplerCapability]:
    canonical = normalize_sampler_name(name, family=family)
    capability = _BY_ID.get(canonical)
    if capability is None:
        return None
    family_key = _normalize_key(family)
    if family_key and family_key not in capability.families:
        return None
    return capability


def list_sampler_capabilities(
    *,
    family: Optional[str] = None,
    backend: Optional[str] = None,
    include_not_wired: bool = True,
) -> List[SamplerCapability]:
    family_key = _normalize_key(family)
    backend_key = _normalize_key(backend)
    result: List[SamplerCapability] = []
    for capability in _CAPABILITIES:
        if family_key and family_key not in capability.families:
            continue
        if backend_key and backend_key != _normalize_key(capability.backend):
            continue
        if not include_not_wired and capability.status == "not_wired":
            continue
        result.append(capability)
    return result


def scheduler_capability_matrix(*, family: Optional[str] = None) -> List[Dict[str, Any]]:
    return [capability.to_dict() for capability in list_sampler_capabilities(family=family)]


def create_diffusers_scheduler(name: Optional[str], scheduler: Any, *, family: Optional[str] = None) -> Any:
    """Create a diffusers scheduler from the static matrix.

    Unknown names, non-diffusers capabilities, or import/config failures return
    the original scheduler to preserve existing preview behavior.
    """
    capability = get_sampler_capability(name, family=family)
    if capability is None or capability.backend != "diffusers" or capability.scheduler_class is None:
        return scheduler

    module = import_module("diffusers")
    scheduler_cls = getattr(module, capability.scheduler_class)
    kwargs = dict(capability.scheduler_kwargs)
    return scheduler_cls.from_config(scheduler.config, **kwargs)


def runtime_sampler_name(name: Optional[str], *, family: Optional[str] = None) -> str:
    """Return the sampler string expected by the current runtime implementation."""
    canonical = normalize_sampler_name(name, family=family)
    if canonical == "flow_euler":
        return "euler"
    if canonical == "flow_dpm_solver":
        return "dpm_solver"
    return canonical


def supported_sampler_ids(*, family: Optional[str] = None) -> List[str]:
    return [capability.id for capability in list_sampler_capabilities(family=family)]


def unsupported_sampler_features(*, family: Optional[str] = None) -> List[str]:
    return [
        capability.id
        for capability in list_sampler_capabilities(family=family)
        if capability.status == "not_wired"
    ]
