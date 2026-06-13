# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test FusedAdamW foreach rewrite: parity vs torch.optim.AdamW and the
per-parameter fallback path, plus state-dict/warm-start behaviour.

The foreach step must be element-wise identical math to the legacy
per-parameter ``_fused_adamw_step_`` (same op sequence), so trajectories are
compared against both torch's reference AdamW (tight tolerance) and the
internal per-param path (near-bitwise on CPU).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.core.lulynx_trainer.fused_adamw import FusedAdamW


def _make_params(seed: int, *, dtypes=(torch.float32,)) -> list[torch.nn.Parameter]:
    gen = torch.Generator().manual_seed(seed)
    params = []
    for i, shape in enumerate([(8, 16), (32,), (4, 4, 4), (128, 2)]):
        dtype = dtypes[i % len(dtypes)]
        params.append(torch.nn.Parameter(torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype)))
    return params


def _clone_params(params) -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(p.detach().clone()) for p in params]


def _run_steps(optimizer, params, n_steps: int, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_steps):
        for p in params:
            p.grad = torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.dtype)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def test_foreach_matches_torch_adamw() -> None:
    for amsgrad in (False, True):
        for weight_decay in (0.0, 1e-2):
            a = _make_params(7)
            b = _clone_params(a)
            opt_a = FusedAdamW(a, lr=2e-3, weight_decay=weight_decay, amsgrad=amsgrad)
            opt_b = torch.optim.AdamW(b, lr=2e-3, weight_decay=weight_decay, amsgrad=amsgrad)
            _run_steps(opt_a, a, 20, seed=11)
            _run_steps(opt_b, b, 20, seed=11)
            for pa, pb in zip(a, b):
                # torch's AdamW applies decay as mul_(1-lr*wd) vs our legacy
                # add_(p, alpha=-lr*wd): same math, different rounding order,
                # so reference parity is tolerance-based; exact parity vs our
                # own legacy path is asserted separately at rtol=0.
                torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-5)


def test_foreach_matches_per_param_path() -> None:
    a = _make_params(13)
    b = _clone_params(a)
    opt_foreach = FusedAdamW(a, lr=1e-3, weight_decay=1e-2)
    # capturable=True forces the legacy per-parameter path (identical math,
    # per-tensor launches; step lives on the param device).
    opt_legacy = FusedAdamW(b, lr=1e-3, weight_decay=1e-2, capturable=True)
    _run_steps(opt_foreach, a, 15, seed=23)
    _run_steps(opt_legacy, b, 15, seed=23)
    for pa, pb in zip(a, b):
        torch.testing.assert_close(pa, pb, rtol=0.0, atol=0.0)


def test_mixed_dtype_buckets() -> None:
    params = _make_params(29, dtypes=(torch.float32, torch.bfloat16))
    opt = FusedAdamW(params, lr=1e-3)
    _run_steps(opt, params, 3, seed=31)
    for p in params:
        assert torch.isfinite(p.detach().float()).all()
    # one CPU step tensor per param, all advanced in lockstep without sync
    for state in opt.state.values():
        assert state["step"].device.type == "cpu"
        assert float(state["step"].item()) == 3.0


def test_state_dict_roundtrip_and_legacy_step_device() -> None:
    params = _make_params(37)
    opt = FusedAdamW(params, lr=1e-3)
    _run_steps(opt, params, 5, seed=41)
    snapshot = copy.deepcopy(opt.state_dict())

    fresh_params = _clone_params(params)
    fresh = FusedAdamW(fresh_params, lr=1e-3)
    fresh.load_state_dict(snapshot)
    # Simulate a legacy checkpoint that kept step on the param device/dtype.
    for state in fresh.state.values():
        state["step"] = state["step"].to(dtype=torch.float64)
    _run_steps(fresh, fresh_params, 2, seed=43)
    for state in fresh.state.values():
        assert state["step"].device.type == "cpu"
        assert float(state["step"].item()) == 7.0


def main() -> int:
    test_foreach_matches_torch_adamw()
    test_foreach_matches_per_param_path()
    test_mixed_dtype_buckets()
    test_state_dict_roundtrip_and_legacy_step_device()
    print("fused_adamw_foreach_parity_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
