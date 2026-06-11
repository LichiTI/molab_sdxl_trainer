# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Tiny quality A/B probe for SVD/GaLore-style gradient projection.

The probe trains two identical small regression models: baseline AdamW and
AdamW wrapped by SVDGradientProjectionWrapper. It is intentionally short and
trainer-outside; it reports loss deltas and final parameter drift, not model
quality guarantees.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.svd_grad_projection import SVDGradientProjectionWrapper  # noqa: E402


@dataclass
class OptimizerRun:
    name: str
    success: bool
    failed_reason: str = ""
    initial_loss: float = 0.0
    final_loss: float = 0.0
    min_loss: float = 0.0
    loss_delta: float = 0.0
    losses: List[float] | None = None
    wrapper: str = ""

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["losses"] = list(self.losses or [])
        return data


class _TinyRegressor(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([param.detach().float().reshape(-1).cpu() for param in model.parameters()])


def _run_training(
    *,
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    steps: int,
    lr: float,
    rank: int,
    update_interval: int,
    scale: float,
    warmup_steps: int,
) -> tuple[OptimizerRun, torch.Tensor | None]:
    model = model.to(device).train()
    x = x.to(device)
    y = y.to(device)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    optimizer: Any = base_optimizer
    wrapper = "AdamW"
    if name == "galore_svd":
        optimizer = SVDGradientProjectionWrapper(
            base_optimizer,
            rank=rank,
            update_interval=update_interval,
            scale=scale,
            warmup_steps=warmup_steps,
        )
        wrapper = repr(optimizer)
    losses: List[float] = []
    try:
        for _ in range(max(int(steps), 1)):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred.float(), y.float())
            loss.backward()
            optimizer.step()
            _sync(device)
            losses.append(float(loss.detach().float().cpu().item()))
        return (
            OptimizerRun(
                name=name,
                success=True,
                initial_loss=losses[0],
                final_loss=losses[-1],
                min_loss=min(losses),
                loss_delta=losses[-1] - losses[0],
                losses=[round(item, 8) for item in losses],
                wrapper=wrapper,
            ),
            _flatten_params(model),
        )
    except Exception as exc:
        return OptimizerRun(name=name, success=False, failed_reason=f"{type(exc).__name__}: {exc}", wrapper=wrapper), None


def run_galore_svd_ab_probe(
    *,
    device: str = "auto",
    steps: int = 20,
    dim: int = 32,
    hidden: int = 64,
    samples: int = 64,
    lr: float = 1e-2,
    rank: int = 8,
    update_interval: int = 1,
    scale: float = 1.0,
    warmup_steps: int = 0,
    seed: int = 1234,
) -> Dict[str, Any]:
    requested = str(device or "auto").lower()
    torch_device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)
    torch.manual_seed(seed)
    teacher = _TinyRegressor(dim, hidden)
    base = _TinyRegressor(dim, hidden)
    x = torch.randn(samples, dim)
    with torch.no_grad():
        y = teacher(x) + 0.01 * torch.randn(samples, dim)
    baseline, baseline_params = _run_training(
        name="baseline_adamw",
        model=copy.deepcopy(base),
        x=x,
        y=y,
        device=torch_device,
        steps=steps,
        lr=lr,
        rank=rank,
        update_interval=update_interval,
        scale=scale,
        warmup_steps=warmup_steps,
    )
    galore, galore_params = _run_training(
        name="galore_svd",
        model=copy.deepcopy(base),
        x=x,
        y=y,
        device=torch_device,
        steps=steps,
        lr=lr,
        rank=rank,
        update_interval=update_interval,
        scale=scale,
        warmup_steps=warmup_steps,
    )
    param_l2 = None
    param_max_abs = None
    if baseline_params is not None and galore_params is not None:
        diff = galore_params - baseline_params
        param_l2 = float(torch.linalg.vector_norm(diff).item())
        param_max_abs = float(diff.abs().max().item())
    return {
        "probe": "galore_svd_ab_probe",
        "device": str(torch_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "config": {
            "steps": int(steps),
            "dim": int(dim),
            "hidden": int(hidden),
            "samples": int(samples),
            "lr": float(lr),
            "rank": int(rank),
            "update_interval": int(update_interval),
            "scale": float(scale),
            "warmup_steps": int(warmup_steps),
            "seed": int(seed),
        },
        "runs": [baseline.as_dict(), galore.as_dict()],
        "comparison": {
            "final_loss_delta_galore_minus_baseline": None
            if not (baseline.success and galore.success)
            else float(galore.final_loss - baseline.final_loss),
            "param_l2_delta": param_l2,
            "param_max_abs_delta": param_max_abs,
        },
        "interpretation": "Short optimizer-behavior A/B only; not a training-quality guarantee.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="temp/galore_svd_ab_probe.json")
    args = parser.parse_args()
    payload = run_galore_svd_ab_probe(
        device=args.device,
        steps=max(1, min(int(args.steps), 40)),
        dim=args.dim,
        hidden=args.hidden,
        samples=args.samples,
        lr=args.lr,
        rank=args.rank,
        update_interval=args.update_interval,
        scale=args.scale,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

