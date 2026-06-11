"""Smoke test for the sampler capability matrix."""

from __future__ import annotations

import sys
import types
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.sampler_capabilities import (  # noqa: E402
    create_diffusers_scheduler,
    get_sampler_capability,
    normalize_sampler_name,
    runtime_sampler_name,
    unsupported_sampler_features,
)


class _FakeSchedulerInput:
    config = {"beta_start": 0.1}


def _install_fake_diffusers():
    module = types.ModuleType("diffusers")

    class _BaseScheduler:
        @classmethod
        def from_config(cls, config, **kwargs):
            return {
                "class": cls.__name__,
                "config": dict(config),
                "kwargs": dict(kwargs),
            }

    for name in (
        "EulerAncestralDiscreteScheduler",
        "EulerDiscreteScheduler",
        "DDIMScheduler",
        "DPMSolverMultistepScheduler",
        "DPMSolverSinglestepScheduler",
        "PNDMScheduler",
        "LMSDiscreteScheduler",
        "HeunDiscreteScheduler",
        "UniPCMultistepScheduler",
        "KDPM2DiscreteScheduler",
        "KDPM2AncestralDiscreteScheduler",
        "DEISMultistepScheduler",
    ):
        setattr(module, name, type(name, (_BaseScheduler,), {}))

    old = sys.modules.get("diffusers")
    sys.modules["diffusers"] = module
    return old


def test_aliases() -> None:
    assert normalize_sampler_name("dpmsolver++") == "dpm++_2m"
    assert normalize_sampler_name("dpm++_2m_sde") == "dpm++_sde"
    assert normalize_sampler_name("dpm_2_a") == "dpm_ancestral"
    assert normalize_sampler_name("uni-pc") == "uni_pc"
    assert normalize_sampler_name("deis_multistep") == "deis"


def test_flow_family_disambiguation() -> None:
    assert normalize_sampler_name("euler", family="sdxl") == "euler"
    assert normalize_sampler_name("euler", family="anima") == "flow_euler"
    assert runtime_sampler_name("dpm-solver", family="anima") == "dpm_solver"
    assert runtime_sampler_name("dpm-solver", family="newbie") == "euler"
    assert get_sampler_capability("dpm_solver", family="anima").status == "experimental"


def test_not_wired_sampler_features() -> None:
    unsupported = set(unsupported_sampler_features(family="anima"))
    assert "cns" in unsupported
    assert "smc_cfg" not in unsupported
    assert "tgate_probe" not in unsupported
    assert "spectrum_probe" not in unsupported
    assert get_sampler_capability("cns", family="sdxl").status == "not_wired"
    assert get_sampler_capability("smc_cfg", family="anima").status == "partial"
    assert get_sampler_capability("t-gate", family="anima").status == "probe"
    assert get_sampler_capability("spectrum", family="anima").status == "probe"
    assert get_sampler_capability("spectrum_probe", family="newbie").status == "probe"
    assert get_sampler_capability("spectrum_probe", family="sdxl") is None


def test_diffusers_scheduler_factory() -> None:
    old = _install_fake_diffusers()
    try:
        result = create_diffusers_scheduler("uni-pc", _FakeSchedulerInput(), family="sdxl")
        assert result["class"] == "UniPCMultistepScheduler"
        assert result["kwargs"] == {}

        result = create_diffusers_scheduler("dpm++_2m_karras", _FakeSchedulerInput(), family="sdxl")
        assert result["class"] == "DPMSolverMultistepScheduler"
        assert result["kwargs"] == {"use_karras_sigmas": True}

        result = create_diffusers_scheduler("dpm_2", _FakeSchedulerInput(), family="sd15")
        assert result["class"] == "KDPM2DiscreteScheduler"
    finally:
        if old is None:
            sys.modules.pop("diffusers", None)
        else:
            sys.modules["diffusers"] = old


def main() -> int:
    tests = (
        test_aliases,
        test_flow_family_disambiguation,
        test_not_wired_sampler_features,
        test_diffusers_scheduler_factory,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("Sampler capability matrix smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
