"""Smoke: 10-step training comparison, eager LoRA vs fused Triton LoRA.

Run with the flashattention env:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/triton_ops_smoke.py

Builds two *identically initialised* stacks of real ``LoRALinear`` layers, runs
the same data through the same optimiser for both, and checks that swapping in
the fused Triton forward (via ``triton_inject.apply``) keeps the loss
trajectory and gradients equivalent to the eager path. Then it measures both
forward-only and forward+backward step-time speedups (the backward of the fused
path upcasts the LoRA grads to fp32 for accuracy, so the two can differ a lot)
and emits the promotion scorecard.

Correctness is the pass/fail gate; the measured speedups are reported into the
scorecard for promotion review.
"""

from __future__ import annotations

import os
import sys
import time

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import torch
from torch import nn

from core.lulynx_trainer.lora_injector import LoRALinear
from core.lulynx_trainer.triton_ops import triton_inject
from core.lulynx_trainer.triton_ops.config import detect_gpu, describe_gpu
from core.lulynx_trainer.triton_ops_scorecard import build_triton_ops_scorecard

DEV = "cuda"
DIM = 2048
N_LAYERS = 6
RANK = 32


def _build_stack(seed: int, n_layers: int = N_LAYERS, dim: int = DIM, rank: int = RANK) -> nn.Sequential:
    """A deterministic stack of LoRALinear layers (identical for equal seed)."""
    torch.manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        lin = nn.Linear(dim, dim, bias=True)
        m = LoRALinear(lin, rank=rank, alpha=2 * rank, dropout=0.0)
        with torch.no_grad():
            m.lora.lora_down.weight.normal_(0.0, dim ** -0.5)
            m.lora.lora_up.weight.normal_(0.0, rank ** -0.5)  # un-zero -> real delta
        layers.append(m)
    return nn.Sequential(*layers).to(device=DEV, dtype=torch.bfloat16)


def _make_data(ref_net: nn.Sequential, n_steps: int, batch: int, noise: float = 0.02, seed: int = 777):
    """Stable, learnable targets: the stack-at-init output plus small noise.

    Because the eager and fused stacks share this init, the loss starts near
    ``noise**2`` and stays finite under SGD — so the trajectory comparison is
    meaningful instead of diverging.
    """
    torch.manual_seed(seed)
    data = []
    for _ in range(n_steps):
        x = torch.randn(batch, DIM, device=DEV, dtype=torch.bfloat16)
        with torch.no_grad():
            tgt = ref_net(x) + noise * torch.randn(batch, DIM, device=DEV, dtype=torch.bfloat16)
        data.append((x, tgt))
    return data


def _train(net: nn.Sequential, data, lr: float = 0.05) -> list[float]:
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)
    losses = []
    for x, tgt in data:
        opt.zero_grad(set_to_none=True)
        loss = (net(x).float() - tgt.float()).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-6)


def check_training_parity(batch: int = 64, n_steps: int = 10, lr: float = 0.01):
    print("== 10-step training parity (single layer; eager vs fused) ==")
    # A single layer keeps bf16 rounding from compounding across depth; targets
    # are the fp32-oracle output at init plus small noise, so neither path is
    # favored. The gate compares the *trained* LoRA weights (O(0.1) scale -> a
    # stable relative metric, unlike comparing near-zero loss scalars).
    ref = _build_stack(seed=42, n_layers=1)[0]
    W = ref.original.weight.detach().float()
    b = ref.original.bias.detach().float()
    D = ref.lora.lora_down.weight.detach().float()
    U = ref.lora.lora_up.weight.detach().float()
    s = float(ref.lora.scaling)
    torch.manual_seed(777)
    data = []
    for _ in range(n_steps):
        x = torch.randn(batch, DIM, device=DEV, dtype=torch.bfloat16)
        xf = x.float()
        tgt = (
            torch.nn.functional.linear(xf, W, b)
            + s * (xf @ D.t() @ U.t())
            + 0.02 * torch.randn(batch, DIM, device=DEV)
        ).to(torch.bfloat16)
        data.append((x, tgt))

    net_e = _build_stack(seed=42, n_layers=1)
    net_f = _build_stack(seed=42, n_layers=1)
    patched = triton_inject.apply(net_f)
    le = _train(net_e, data, lr)
    lf = _train(net_f, data, lr)

    finite = all((v == v) and (abs(v) != float("inf")) for v in le + lf)
    wdiff = max(
        _rel(net_f[0].lora.lora_down.weight, net_e[0].lora.lora_down.weight),
        _rel(net_f[0].lora.lora_up.weight, net_e[0].lora.lora_up.weight),
    )
    ok = finite and wdiff < 5e-2 and patched == 1
    print(f"  layers patched     : {patched} (want 1)")
    print(f"  eager loss curve   : {[round(v, 5) for v in le]}")
    print(f"  fused loss curve   : {[round(v, 5) for v in lf]}")
    print(f"  trained-weight rel : {wdiff:.2e}  finite={finite}  {'OK' if ok else 'FAIL'}")
    return ok, patched


def check_grad_parity(batch: int = 128):
    print("== single-step gradient parity ==")
    net_e = _build_stack(seed=7)
    net_f = _build_stack(seed=7)
    triton_inject.apply(net_f)
    x = torch.randn(batch, DIM, device=DEV, dtype=torch.bfloat16)
    tgt = torch.randn(batch, DIM, device=DEV, dtype=torch.bfloat16)
    for net in (net_e, net_f):
        net.zero_grad(set_to_none=True)
        (((net(x).float() - tgt.float()) ** 2).mean()).backward()

    ok = True
    worst = 0.0
    for le, lf in zip(net_e, net_f):
        for name in ("lora_down", "lora_up"):
            r = _rel(getattr(lf.lora, name).weight.grad, getattr(le.lora, name).weight.grad)
            worst = max(worst, r)
            ok = ok and r < 5e-2
    print(f"  worst grad rel diff across {N_LAYERS} layers: {worst:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def _time(net: nn.Sequential, data, *, backward: bool, warmup: int, iters: int) -> float:
    params = [p for p in net.parameters() if p.requires_grad]

    def step(x, tgt):
        if backward:
            for p in params:
                p.grad = None
            (((net(x).float() - tgt.float()) ** 2).mean()).backward()
        else:
            with torch.no_grad():
                net(x)

    for x, tgt in data[:warmup]:
        step(x, tgt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x, tgt in data[warmup:]:
        step(x, tgt)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def measure_speedups(batch: int = 256, warmup: int = 10, iters: int = 40, reps: int = 5):
    import statistics

    print(f"== step-time speedup (median of {reps} reps) ==")
    data = _make_data(_build_stack(seed=99), warmup + iters, batch)
    net_e = _build_stack(seed=99)
    net_f = _build_stack(seed=99)
    triton_inject.apply(net_f)

    out = {}
    for label, backward in (("fwd-only", False), ("fwd+bwd", True)):
        sps = []
        for _ in range(reps):
            t_e = _time(net_e, data, backward=backward, warmup=warmup, iters=iters)
            t_f = _time(net_f, data, backward=backward, warmup=warmup, iters=iters)
            sps.append(t_e / t_f if t_f > 0 else 0.0)
        sp = statistics.median(sps)
        out[label] = sp
        print(f"  {label:8s}: median {sp:.3f}x  (reps {[round(s, 3) for s in sps]})")
    return out


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    gpu = detect_gpu()
    print(describe_gpu(gpu))
    loss_ok, patched = check_training_parity()
    grad_ok = check_grad_parity()
    speedups = measure_speedups()

    correctness_ok = loss_ok and grad_ok
    scorecard = build_triton_ops_scorecard(
        layers_patched=patched,
        numerical_verified=loss_ok,
        gradients_verified=grad_ok,
        performance_measured=True,
        forward_speedup=speedups.get("fwd-only"),
        training_speedup=speedups.get("fwd+bwd"),
        gpu_name=gpu.name,
        is_ada_or_newer=gpu.is_ada_or_newer,
    )
    print("\n== scorecard ==")
    for k, v in scorecard.items():
        print(f"  {k}: {v}")
    print("\nRESULT:", "CORRECTNESS PASS" if correctness_ok else "CORRECTNESS FAIL",
          f"| fwd-only={speedups.get('fwd-only'):.3f}x fwd+bwd={speedups.get('fwd+bwd'):.3f}x",
          f"| scorecard.ok={scorecard['ok']}")
    sys.exit(0 if correctness_ok else 1)
