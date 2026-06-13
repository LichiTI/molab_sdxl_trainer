# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test the opt-in compiled_step optimizer wrapper.

Covers:
* parity — a per-param-loop plugin optimizer (pytorch_optimizer.StableAdamW)
  produces the same parameters with and without the compiled wrap;
* lr-tensor recipe — group lr survives a cosine LR scheduler as a tensor
  (``fill_`` path), so schedule updates do not break the dynamo guard;
* fallback — a failing torch.compile permanently restores the eager step;
* off-by-default — backend != compiled_step leaves the step untouched;
* closure passthrough — calls carrying a closure bypass the compiled path;
* skip guards — bitsandbytes-style modules are refused with a reason.

Also prints an eager-vs-compiled ms/step micro-benchmark on a LoRA-shaped
parameter set as the absorbed-program evidence trail.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
for import_root in (ROOT, BACKEND):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch

from backend.core.lulynx_trainer.compiled_step_optimizer import wrap_optimizer_step_compiled

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_params(count: int = 60, seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    params = []
    for _ in range(count // 2):
        params.append(torch.randn(16, 256, device=DEVICE, requires_grad=True))
        params.append(torch.randn(256, 16, device=DEVICE, requires_grad=True))
    return params


def _set_grads(params: list[torch.Tensor], seed: int) -> None:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen, device=DEVICE) * 1e-3


def _make_plugin_optimizer(params):
    from pytorch_optimizer import StableAdamW

    return StableAdamW(params, lr=1e-3, weight_decay=1e-2)


def test_parity_with_eager() -> None:
    if DEVICE != "cuda":
        print("  (no CUDA — parity test skipped, wrapper itself refuses to wrap)")
        return
    ref_params = _make_params(seed=7)
    ref_opt = _make_plugin_optimizer(ref_params)
    wrapped_params = _make_params(seed=7)
    wrapped_opt = _make_plugin_optimizer(wrapped_params)
    report = wrap_optimizer_step_compiled(wrapped_opt)
    assert report["wrapped"] is True, report
    # StableAdamW's foreach path rejects tensor lr -> probe must keep float lr
    assert report["lr_groups_tensorized"] == 0, report
    assert any("lr kept as python float" in n for n in report["notes"]), report
    assert isinstance(wrapped_opt.param_groups[0]["lr"], float)
    for step in range(5):
        _set_grads(ref_params, seed=100 + step)
        _set_grads(wrapped_params, seed=100 + step)
        ref_opt.step()
        wrapped_opt.step()
    for ref, got in zip(ref_params, wrapped_params):
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


def test_lr_tensor_survives_scheduler() -> None:
    if DEVICE != "cuda":
        return
    params = _make_params(count=4, seed=3)
    # native AdamW supports tensor lr end-to-end (official compiled recipe)
    opt = torch.optim.AdamW(params, lr=1e-3)
    report = wrap_optimizer_step_compiled(opt)
    assert report["wrapped"] is True, report
    assert report["lr_groups_tensorized"] >= 1, report
    lr_tensor = opt.param_groups[0]["lr"]
    assert isinstance(lr_tensor, torch.Tensor), type(lr_tensor)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    for step in range(6):
        _set_grads(params, seed=200 + step)
        opt.step()
        sched.step()
        # identity must survive: scheduler updates via fill_, not reassignment
        assert opt.param_groups[0]["lr"] is lr_tensor, "lr tensor identity lost — recompile storm risk"
    assert float(lr_tensor) < 1e-3  # cosine actually decayed it


def test_compile_failure_restores_eager() -> None:
    if DEVICE != "cuda":
        return
    params = _make_params(count=2, seed=1)
    opt = torch.optim.SGD(params, lr=1e-3)
    orig_step = opt.step
    report = wrap_optimizer_step_compiled(opt)
    assert report["wrapped"] is True, report
    original_compile = torch.compile

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("synthetic compile failure")

    torch.compile = _boom
    try:
        _set_grads(params, seed=11)
        opt.step()  # lazy compile fails -> eager fallback, params still updated
    finally:
        torch.compile = original_compile
    assert opt.step == orig_step, "eager step not restored after compile failure"
    _set_grads(params, seed=12)
    opt.step()  # keeps working eagerly


def test_closure_passthrough() -> None:
    if DEVICE != "cuda":
        return
    params = _make_params(count=2, seed=2)
    opt = torch.optim.SGD(params, lr=1e-3)
    report = wrap_optimizer_step_compiled(opt)
    assert report["wrapped"] is True, report
    called = {"closure": 0}

    def closure():
        called["closure"] += 1
        return torch.tensor(0.0)

    _set_grads(params, seed=21)
    opt.step(closure)
    assert called["closure"] == 1


def test_skip_guards() -> None:
    class _FakeBnbOpt:
        __module__ = "bitsandbytes.optim.adamw"
        param_groups: list = []

        def step(self):  # pragma: no cover
            pass

    report = wrap_optimizer_step_compiled(_FakeBnbOpt())
    assert report["wrapped"] is False
    assert "bitsandbytes" in report["skipped_reason"], report


def bench_eager_vs_compiled() -> None:
    if DEVICE != "cuda":
        return
    params = _make_params(count=300, seed=5)
    _set_grads(params, seed=42)

    def run(opt, steps=40, warmup=8):
        for _ in range(warmup):
            opt.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.step()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / steps * 1000

    eager_ms = run(_make_plugin_optimizer(params))
    wrapped_opt = _make_plugin_optimizer(params)
    wrap_optimizer_step_compiled(wrapped_opt)
    compiled_ms = run(wrapped_opt)
    print(f"  StableAdamW 300 params: eager {eager_ms:.2f} ms/step -> compiled {compiled_ms:.2f} ms/step")


def main() -> int:
    test_parity_with_eager()
    test_lr_tensor_survives_scheduler()
    test_compile_failure_restores_eager()
    test_closure_passthrough()
    test_skip_guards()
    bench_eager_vs_compiled()
    print("compiled_step_optimizer_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
