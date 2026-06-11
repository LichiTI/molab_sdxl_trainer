"""Smoke test for SMC-CFG primitive."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.smc_cfg import SMCCFGConfig, SMCCFGState, build_smc_cfg_state, standard_cfg  # noqa: E402


def test_alpha_zero_matches_standard_cfg() -> None:
    cond = torch.tensor([1.0, -2.0, 0.5])
    uncond = torch.tensor([0.25, -1.0, -0.5])
    state = SMCCFGState(lam=5.0, alpha=0.0)
    assert torch.allclose(state.combine(cond, uncond, 4.0), standard_cfg(cond, uncond, 4.0))


def test_adaptive_correction_is_bounded_and_stateful() -> None:
    cond = torch.tensor([1.0, -2.0, 0.5])
    uncond = torch.tensor([0.25, -1.0, -0.5])
    state = SMCCFGState(lam=5.0, alpha=0.2)
    first = state.combine(cond, uncond, 4.0)
    second = state.combine(cond * 0.8, uncond, 4.0)
    vanilla = standard_cfg(cond, uncond, 4.0)
    assert not torch.allclose(first, vanilla)
    assert not torch.allclose(first, second)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_builder_respects_default_off() -> None:
    assert build_smc_cfg_state(SMCCFGConfig(enabled=False)) is None
    assert build_smc_cfg_state(SMCCFGConfig(enabled=True, alpha=0.0)) is None
    assert build_smc_cfg_state(SMCCFGConfig(enabled=True, alpha=0.2)) is not None


def main() -> int:
    tests = (
        test_alpha_zero_matches_standard_cfg,
        test_adaptive_correction_is_bounded_and_stateful,
        test_builder_respects_default_off,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("SMC-CFG smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
