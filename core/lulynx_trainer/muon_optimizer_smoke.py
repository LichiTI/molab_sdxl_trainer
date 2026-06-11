# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for the Muon optimizer (cleanroom).

Run with the flashattention env (CPU is fine for this test):
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/muon_optimizer_smoke.py

Checks: (1) Newton-Schulz actually orthogonalizes (Gram → I) far better than the
raw gradient; (2) the hybrid step routes 2D→Muon and 1D→AdamW and never crashes;
(3) a tiny LoRA-shaped model trains (loss decreases); (4) optimizer state
round-trips through save/load.  Emits the promotion scorecard.
"""

from __future__ import annotations

import os
import sys

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import torch
from torch import nn

from core.lulynx_trainer.muon_optimizer import Muon, _zeropower_via_newtonschulz5
from core.lulynx_trainer.muon_optimizer_scorecard import build_muon_scorecard

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _gram_error(x: torch.Tensor) -> float:
    """‖GramᵀG − I‖ on the min-dim orientation, normalized — 0 == orthogonal."""
    x = x.float()
    gram = x @ x.t() if x.size(0) <= x.size(1) else x.t() @ x
    eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    return (gram - eye).norm().item() / (gram.size(0) ** 0.5)


def check_orthogonality() -> tuple[bool, float]:
    print("== Newton-Schulz orthogonalization ==")
    ok = True
    worst = 0.0
    for shape in [(64, 64), (128, 32), (32, 128), (256, 48)]:
        g = torch.randn(*shape, device=DEV)
        raw_err = _gram_error(g / (g.norm() + 1e-7))
        ortho = _zeropower_via_newtonschulz5(g, steps=5)
        ortho_err = _gram_error(ortho)
        worst = max(worst, ortho_err)
        # orthogonalized must be near-identity Gram AND much better than raw
        passed = ortho_err < 0.5 and ortho_err < raw_err * 0.5
        ok &= passed
        print(f"  {shape}: raw_gram_err={raw_err:.3f}  ortho_gram_err={ortho_err:.3f}  {'OK' if passed else 'FAIL'}")
    return ok, worst


def check_hybrid_routing() -> bool:
    print("== hybrid 2D(Muon) / 1D(AdamW) routing ==")
    w2d = nn.Parameter(torch.randn(32, 16, device=DEV))
    b1d = nn.Parameter(torch.randn(32, device=DEV))
    stray_1d = nn.Parameter(torch.randn(16, device=DEV))  # 1D inside a muon group
    opt = Muon([
        {"params": [w2d, stray_1d], "use_muon": True, "lr": 0.02, "weight_decay": 0.0},
        {"params": [b1d], "use_muon": False, "lr": 1e-2, "weight_decay": 0.0},
    ], lr=0.02)

    w0, b0, s0 = w2d.detach().clone(), b1d.detach().clone(), stray_1d.detach().clone()
    (w2d.sum() + b1d.sum() + stray_1d.sum()).backward()
    opt.step()

    moved_2d = not torch.allclose(w2d.detach(), w0)
    moved_1d = not torch.allclose(b1d.detach(), b0)
    moved_stray = not torch.allclose(stray_1d.detach(), s0)
    # 2D in muon group should have a momentum_buffer; stray 1D should NOT (it
    # was routed to the adamw path), and should instead have exp_avg state.
    has_buf_2d = "momentum_buffer" in opt.state[w2d]
    stray_is_adamw = "exp_avg" in opt.state[stray_1d] and "momentum_buffer" not in opt.state[stray_1d]
    b1d_is_adamw = "exp_avg" in opt.state[b1d]
    ok = moved_2d and moved_1d and moved_stray and has_buf_2d and stray_is_adamw and b1d_is_adamw
    print(f"  moved 2D={moved_2d} 1D={moved_1d} stray={moved_stray}; "
          f"2D has momentum={has_buf_2d}; stray→adamw={stray_is_adamw}; b1d→adamw={b1d_is_adamw}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def check_training() -> tuple[bool, float, float]:
    print("== tiny training (loss must decrease) ==")
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(32, 64, bias=True),
        nn.GELU(),
        nn.Linear(64, 32, bias=True),
    ).to(DEV)
    x = torch.randn(128, 32, device=DEV)
    target = torch.randn(128, 32, device=DEV)

    muon_p = [p for p in model.parameters() if p.dim() == 2]
    other_p = [p for p in model.parameters() if p.dim() != 2]
    opt = Muon([
        {"params": muon_p, "use_muon": True, "lr": 0.02, "weight_decay": 0.0},
        {"params": other_p, "use_muon": False, "lr": 1e-2, "weight_decay": 0.0},
    ], lr=0.02)

    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    init_l, final_l = losses[0], losses[-1]
    ok = final_l < init_l * 0.7 and all(l == l for l in losses)  # decreased, no NaN
    print(f"  loss {init_l:.4f} -> {final_l:.4f}  {'OK' if ok else 'FAIL'}")
    return ok, init_l, final_l


def check_resume() -> bool:
    print("== optimizer state resume ==")
    p = nn.Parameter(torch.randn(16, 8, device=DEV))
    opt = Muon([{"params": [p], "use_muon": True, "lr": 0.02}], lr=0.02)
    for _ in range(3):
        opt.zero_grad()
        (p ** 2).sum().backward()
        opt.step()
    sd = opt.state_dict()

    p2 = nn.Parameter(p.detach().clone())
    opt2 = Muon([{"params": [p2], "use_muon": True, "lr": 0.02}], lr=0.02)
    opt2.load_state_dict(sd)
    buf1 = opt.state[p]["momentum_buffer"]
    buf2 = opt2.state[p2]["momentum_buffer"]
    ok = torch.allclose(buf1, buf2)
    print(f"  momentum buffer match after load={ok}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"device: {DEV}")
    ortho_ok, ortho_err = check_orthogonality()
    routing_ok = check_hybrid_routing()
    train_ok, init_l, final_l = check_training()
    resume_ok = check_resume()

    scorecard = build_muon_scorecard(
        orthogonality_verified=ortho_ok,
        orthogonality_error=ortho_err,
        hybrid_routing_verified=routing_ok,
        muon_param_count=2,
        adamw_param_count=2,
        loss_decreased=train_ok,
        initial_loss=init_l,
        final_loss=final_l,
        resume_verified=resume_ok,
    )
    print("\n== scorecard ==")
    for k, v in scorecard.items():
        print(f"  {k}: {v}")

    all_ok = ortho_ok and routing_ok and train_ok and resume_ok
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT", f"| scorecard.ok={scorecard['ok']}")
    sys.exit(0 if all_ok else 1)
